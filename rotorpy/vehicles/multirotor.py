from typing import List
from collections import deque
import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.hummingbird_params import quad_params

from scipy.spatial.transform import Rotation as R

# imports for Batched Dynamics
try:
    import torch
    from torchdiffeq import odeint
    import roma
except ImportError:
    pass

import time

"""
Multirotor models
"""

def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        duat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[ q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [ q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Rely on post-step renormalization instead of a penalty term
    return quat_dot


def quat_dot_torch(quat, omega):
    """
    Parameters:
        quat, (...,[i,j,k,w])
        omega, angular velocity of body in body axes: (...,3)

    Returns
        duat_dot, (...,[i,j,k,w])

    """
    b = quat.shape[0]
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[...,0], quat[...,1], quat[...,2], quat[...,3])
    G = torch.stack([q3, q2, -q1, -q0,
                     -q2, q3, q0, -q1,
                     q1, -q0, q3, -q2], dim=1).view((b, 3, 4))

    quat_dot = 0.5 * torch.transpose(G, 1,2) @ omega.unsqueeze(-1)
    # Augment to maintain unit quaternion.
    quat_err = torch.sum(quat**2, dim=-1) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot.squeeze(-1) - quat_err.unsqueeze(-1) * quat_err_grad
    return quat_dot


class Multirotor(object):
    """
    Multirotor forward dynamics model. 

    states: [position, velocity, attitude, body rates, wind, rotor speeds]

    Parameters:
        quad_params: a dictionary containing relevant physical parameters for the multirotor. 
        initial_state: the initial state of the vehicle. 
        control_abstraction: the appropriate control abstraction that is used by the controller, options are...
                                'cmd_motor_speeds': the controller directly commands motor speeds. 
                                'cmd_motor_thrusts': the controller commands forces for each rotor.
                                'cmd_ctbr': the controller commands a collective thrsut and body rates. 
                                'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
                                'cmd_ctatt': the controller commands a collective thrust and attitude (as a quaternion).
                                'cmd_vel': the controller commands a velocity vector in the world frame. 
                                'cmd_acc': the controller commands a mass normalized thrust vector (acceleration) in the world frame.
        aero: boolean, determines whether or not aerodynamic drag forces are computed. 
        enable_ground: boolean, determines whether or not ground contact is enabled.
        integrator_kwargs: dictionary of keyword arguments passed to scipy.integrate.solve_ivp
    """
    def __init__(self, quad_params, initial_state = {'x': np.array([0,0,0]),
                                            'v': np.zeros(3,),
                                            'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                            'w': np.zeros(3,),
                                            'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                            'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])},
                       control_abstraction='cmd_motor_speeds',
                       aero = True,
                       enable_ground = False,
                       integrator_kwargs = None,
                ):
        """
        Initialize quadrotor physical parameters.
        """

        # Inertial parameters
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.Ixy             = quad_params['Ixy']  # kg*m^2
        self.Ixz             = quad_params['Ixz']  # kg*m^2
        self.Iyz             = quad_params['Iyz']  # kg*m^2

        # Frame parameters
        self.c_Dx            = quad_params.get('c_Dx', 0.0)  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params.get('c_Dy', 0.0)  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params.get('c_Dz', 0.0)  # drag coeff, N/(m/s)**2

        self.num_rotors      = quad_params['num_rotors']
        self.rotor_pos       = quad_params['rotor_pos']

        self.rotor_dir       = quad_params['rotor_directions']

        self.extract_geometry()

        # Rotor parameters    
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s

        # Thrust and yaw-moment coefficients. Accept either a scalar (shared by all rotors, the
        # common case) or a per-rotor array of length num_rotors. Real motors/props are unmatched,
        # so per-rotor coefficients are a key sim-to-real / SkyDreamer-parity axis. Stored
        # internally as arrays of shape (num_rotors,); a scalar is broadcast, preserving existing
        # parameter files exactly.
        self.k_eta           = np.broadcast_to(np.asarray(quad_params['k_eta'], dtype=float), (self.num_rotors,)).copy()  # quadratic thrust coeff c2, N/(rad/s)**2
        self.k_m             = np.broadcast_to(np.asarray(quad_params['k_m'], dtype=float), (self.num_rotors,)).copy()     # quadratic yaw moment coeff d2, Nm/(rad/s)**2

        # Optional linear + constant terms of the thrust/torque curves (Crazyflow first-principles):
        #   thrust_i = thrust_c0 + thrust_c1*Omega_i + k_eta*Omega_i^2
        #   torque_i = torque_c0 + torque_c1*Omega_i + k_m *Omega_i^2   (magnitude; sign from rotor_dir)
        # All default to 0, so the model reduces exactly to the quadratic k_eta*Omega^2 / k_m*Omega^2.
        # Each may be a scalar (shared) or a per-rotor array of length num_rotors.
        self.thrust_c0       = np.broadcast_to(np.asarray(quad_params.get('thrust_c0', 0.0), dtype=float), (self.num_rotors,)).copy()
        self.thrust_c1       = np.broadcast_to(np.asarray(quad_params.get('thrust_c1', 0.0), dtype=float), (self.num_rotors,)).copy()
        self.torque_c0       = np.broadcast_to(np.asarray(quad_params.get('torque_c0', 0.0), dtype=float), (self.num_rotors,)).copy()
        self.torque_c1       = np.broadcast_to(np.asarray(quad_params.get('torque_c1', 0.0), dtype=float), (self.num_rotors,)).copy()
        # Whether the thrust curve has non-quadratic terms (selects the exact inversion path).
        self._thrust_poly_active = bool(np.any(self.thrust_c0 != 0.0) or np.any(self.thrust_c1 != 0.0))
        self.k_d             = quad_params.get('k_d', 0.0)       # rotor drag coeff, N/(m/s)
        self.k_z             = quad_params.get('k_z', 0.0)       # induced inflow coeff N/(m/s)
        self.k_h             = quad_params.get('k_h', 0.0)       # translational lift coeff N/(m/s)^2
        self.k_flap          = quad_params.get('k_flap', 0.0)    # Flapping moment coefficient Nm/(m/s)

        # Thrust dependence on rotor angle-of-attack and advance ratio (SkyDreamer B1). All default
        # to 0 (thrust independent of airspeed beyond k_h/k_z). k_angle, k_hor multiply the rotor
        # thrust by (1 + k_angle*alpha + k_hor*mu); k_v2 is a collective vertical airspeed-squared
        # term. r_prop is the propeller radius used to form the inflow angles. NOTE: use either
        # {k_angle,k_hor} or k_h for the in-plane effect, not both, to avoid double counting.
        self.k_angle         = quad_params.get('k_angle', 0.0)   # thrust vs rotor angle of attack, dimensionless
        self.k_hor           = quad_params.get('k_hor', 0.0)     # thrust vs advance-ratio angle, dimensionless
        self.k_v2            = quad_params.get('k_v2', 0.0)      # collective vertical drag-like coeff, N/(m/s)^2
        self.r_prop          = quad_params.get('r_prop', quad_params.get('rotor_radius', 0.0))  # propeller radius, m

        # Per-rotor thrust axis (unit vectors in the body frame), shape (num_rotors, 3). Default is
        # +body-z for every rotor (the ideal, perfectly-aligned case). Small tilts model real
        # assembly/manufacturing misalignment (thrust vector not exactly body-z), which produces
        # parasitic body forces and moments and is a useful sim-to-real / domain-randomization axis.
        _thrust_dir = np.asarray(quad_params.get('thrust_dir',
                                 np.tile(np.array([0.0, 0.0, 1.0]), (self.num_rotors, 1))), dtype=float)
        _thrust_dir = _thrust_dir.reshape(self.num_rotors, 3)
        self.thrust_dir = _thrust_dir / np.linalg.norm(_thrust_dir, axis=1, keepdims=True)  # normalize each axis

        # Translational lift (k_h) and the advance-ratio thrust correction (k_hor) both model the
        # increase in thrust with in-plane airspeed; the angle-of-attack term (k_angle) is the
        # vertical companion of that same SkyDreamer thrust-correction model. Enabling k_h together
        # with either k_angle or k_hor double-counts the airspeed effect, so reject that config.
        if self.k_h != 0.0 and (self.k_angle != 0.0 or self.k_hor != 0.0):
            raise ValueError(
                "Conflicting aerodynamic thrust models: k_h (translational lift) cannot be combined "
                "with k_angle/k_hor (angle-of-attack / advance-ratio thrust correction) because they "
                "model the same airspeed-dependent thrust and would double-count. Set k_h=0 or "
                "k_angle=k_hor=0."
            )

        # Motor parameters
        self.tau_m           = quad_params['tau_m']     # motor reponse time, seconds
        self.motor_noise     = quad_params.get('motor_noise_std', 0) # noise added to the actual motor speed, rad/s / sqrt(Hz)
        self.rotor_inertia   = quad_params.get('rotor_inertia', 0.0) # moment of inertia of one rotor about its spin axis, kg*m^2 (0 => gyroscopic/reaction torques off)
        # Optional asymmetric spin-up/spin-down motor dynamics (Crazyflow first-principles):
        #   Omega_dot = ka1*(Omega_c - Omega) + ka2*(Omega_c^2 - Omega^2)   if Omega_c > Omega (spin-up)
        #   Omega_dot = kd1*(Omega_c - Omega) + kd2*(Omega_c^2 - Omega^2)   otherwise          (spin-down)
        # Given as rotor_dyn_coef = [ka1, ka2, kd1, kd2] in RotorPy's rad/s units. When absent, the
        # motor uses the first-order lag 1/tau_m (recovered exactly by [1/tau_m, 0, 1/tau_m, 0]).
        rotor_dyn_coef = quad_params.get('rotor_dyn_coef', None)
        self.rotor_dyn_coef = None if rotor_dyn_coef is None else np.asarray(rotor_dyn_coef, dtype=float)
        self._rotor_dyn_active = self.rotor_dyn_coef is not None
        # Shape of the normalized-throttle -> steady-state-speed curve for the 'cmd_motor_throttle'
        # abstraction (SkyDreamer). k in [0,1]: k=1 is a linear speed map; k=0 is a sqrt curve
        # (linear-in-thrust ESC). SkyDreamer identified k=0.5 for a 5" racer.
        self.motor_curve_k   = quad_params.get('motor_curve_k', 1.0)
        # Pure transport delay on the commanded control (RC link + ESC + loop latency), seconds.
        # 0 => no delay (default). Implemented as a per-step command delay line in step().
        self.motor_delay_time = quad_params.get('motor_delay_time', 0.0)
        self._cmd_buffer = None

        # PWM command quantization for the 'cmd_motor_throttle' abstraction. When pwm_max > pwm_min
        # the normalized throttle u is snapped to the integer PWM grid (real ESCs are quantized).
        # Defaults (0,0) => no quantization.
        self.pwm_min          = quad_params.get('pwm_min', 0.0)
        self.pwm_max          = quad_params.get('pwm_max', 0.0)
        # Battery sag hook: a multiplicative scale on the achievable maximum rotor speed, set
        # externally each step by a battery model (see rotorpy/battery.py). 1.0 => full battery.
        self.rotor_speed_max_scale = 1.0

        # Lower level controller parameters 
        self.k_w             = quad_params.get('k_w', 1)            # The body rate P gain        (for cmd_ctbr)
        self.k_v             = quad_params.get('k_v', 10)           # The *world* velocity P gain (for cmd_vel)
        self.kp_att          = quad_params.get('kp_att', 3000.0)    # The attitude P gain (for cmd_vel, cmd_acc, and cmd_ctatt)
        self.kd_att          = quad_params.get('kd_att', 360.0)     # The attitude D gain (for cmd_vel, cmd_acc, and cmd_ctatt)

        # Additional constants.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])
        self.rotor_drag_matrix = np.array([[self.k_d,   0,                 0],
                                           [0,          self.k_d,          0],
                                           [0,          0,          self.k_z]])
        self.drag_matrix = np.array([[self.c_Dx,    0,          0],
                                     [0,            self.c_Dy,  0],
                                     [0,            0,          self.c_Dz]])
        self.g = 9.81 # m/s^2
        self._enable_ground = enable_ground
        # Ground contact horizontal friction (velocity clamp). Beta in [0.1, 0.5].
        self.ground_friction_beta = float(quad_params.get('ground_friction_beta', 0.3))
        # Clamp to a stable range
        self.ground_friction_beta = max(0.1, min(0.5, self.ground_friction_beta))

        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])

        # Control allocation
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient. 

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

        # Set the initial state
        self.initial_state = initial_state

        self.control_abstraction = control_abstraction

        self.aero = aero

        # External disturbance wrench, held constant across an integration step (resample-and-hold).
        # Set externally each step by a disturbance profile (see rotorpy/disturbances/); zero by
        # default so dynamics are unchanged. Force is in the world frame (N), torque in the body
        # frame (N*m). Keeping these as attributes (rather than state) leaves the ODE deterministic
        # for a given step.
        self.external_force = np.zeros(3)
        self.external_torque = np.zeros(3)
        # Additive noise on the normalized actuator command u in [0,1] (SkyDreamer eps_u), set
        # externally each step by a disturbance profile. Only used by 'cmd_motor_throttle'. Zero
        # by default.
        self.motor_cmd_noise = np.zeros(self.num_rotors)

        # Integrator settings.
        if integrator_kwargs is None:
            self.integrator_kwargs = {'method':'RK45'}
        else:
            self.integrator_kwargs = integrator_kwargs

    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of 
        wrenches acting on the rigid body.
        The rotor_geometry is an array of length (n,3), where n is the number of rotors. 
        Each row corresponds to the position vector of the rotor relative to the CoM. 
        """

        self.rotor_geometry = np.array([]).reshape(0,3)
        for rotor in self.rotor_pos:
            r = self.rotor_pos[rotor]
            self.rotor_geometry = np.vstack([self.rotor_geometry, r])

        return

    def statedot(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max) 

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds)
        s = Multirotor._pack_state(state)
        
        s_dot = s_dot_fn(0, s)
        v_dot = s_dot[3:6]
        w_dot = s_dot[10:13]

        state_dot = {'vdot': v_dot,'wdot': w_dot}
        return state_dot 


    def _apply_command_delay(self, control, t_step):
        """
        Apply a pure transport delay to the commanded control: the control used at this step is
        the one commanded round(motor_delay_time / t_step) steps ago. Returns the control unchanged
        (and touches no state) when motor_delay_time == 0, so nominal behavior is identical.

        During the initial fill of the delay line, the earliest command is held (a standard
        zero-order startup). The delay line advances once per step() call.
        """
        n_delay = int(round(self.motor_delay_time / t_step)) if self.motor_delay_time > 0 else 0
        if n_delay <= 0:
            return control
        if self._cmd_buffer is None or self._cmd_buffer.maxlen != n_delay + 1:
            self._cmd_buffer = deque(maxlen=n_delay + 1)
        self._cmd_buffer.append(control)
        return self._cmd_buffer[0]   # oldest entry: n_delay steps back once the line is full

    def step(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        """

        control = self._apply_command_delay(control, t_step)
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min or above max. The max is scaled by the
        # current battery state (rotor_speed_max_scale, 1.0 by default) to model voltage sag.
        effective_max = self.rotor_speed_max * self.rotor_speed_max_scale
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, effective_max)

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds)
        s = Multirotor._pack_state(state)

        # Integrate
        sol = scipy.integrate.solve_ivp(
            s_dot_fn,
            (0.0, t_step),
            s,
            **self.integrator_kwargs
        )
        s = sol['y'][:, -1]

        # Unpack the state vector.
        state = Multirotor._unpack_state(s)

        # Re-normalize unit quaternion.
        state['q'] = state['q'] / norm(state['q'])
        
        # Apply ground constraints (unified across vehicles)
        if self._enable_ground and self._on_ground(state):
            state = self._handle_vehicle_on_ground(state)

        # Add noise to the motor speed measurement
        state['rotor_speeds'] += np.random.normal(scale=np.abs(self.motor_noise), size=(self.num_rotors,))
        state['rotor_speeds'] = np.clip(state['rotor_speeds'], self.rotor_speed_min, effective_max)

        return state

    def _s_dot_fn(self, t, s, cmd_rotor_speeds):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = Multirotor._unpack_state(s)

        rotor_speeds = state['rotor_speeds']
        inertial_velocity = state['v']
        wind_velocity = state['wind']

        R = Rotation.from_quat(state['q']).as_matrix()

        # Rotor speed derivative (first-order lag, or asymmetric spin-up/spin-down if configured)
        rotor_accel = self._rotor_accel(cmd_rotor_speeds, rotor_speeds)

        # Position derivative.
        x_dot = state['v']

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.T@(inertial_velocity - wind_velocity)

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'], rotor_speeds, body_airspeed_vector)

        # Add the gyroscopic + reaction moments from the spinning rotors (zero if rotor_inertia == 0).
        MtotB = MtotB + self.rotor_inertia_moment(state['w'], rotor_speeds, rotor_accel)

        # Rotate the force from the body frame to the inertial frame, then add the external
        # disturbance force (world frame, zero by default).
        Ftot = R@FtotB + self.external_force

        # Ground reaction force: apply normal force when on ground to prevent penetration
        if self._enable_ground and self._on_ground(state):
            total_force_no_ground = self.weight + Ftot
            if total_force_no_ground[2] < 0:
                ground_normal_force = np.array([0, 0, -total_force_no_ground[2]])
                Ftot += ground_normal_force

        # Velocity derivative.
        v_dot = (self.weight + Ftot) / self.mass

        # Angular velocity derivative. Add the external disturbance torque (body frame, zero by default).
        w = state['w']
        w_hat = Multirotor.hat_map(w)
        w_dot = self.inv_inertia @ (MtotB + self.external_torque - w_hat @ (self.inertia @ w))

        # NOTE: the wind dynamics are currently handled in the wind_profile object. 
        # The line below doesn't do anything, as the wind state is assigned elsewhere. 
        wind_dot = np.zeros(3,)

        # Pack into vector of derivatives.
        s_dot = np.zeros((16+self.num_rotors,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot
        s_dot[13:16] = wind_dot
        s_dot[16:]   = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed 
        for aerodynamic forces. 
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame. 
        The net moment Mtot is represented in the body frame. 
        """

        # Get the local airspeeds for each rotor
        local_airspeeds = body_airspeed_vector[:, np.newaxis] + Multirotor.hat_map(body_rates)@(self.rotor_geometry.T)

        # Thrust MAGNITUDE of each rotor: polynomial thrust_c0 + thrust_c1*Omega + k_eta*Omega^2
        # (reduces to k_eta*Omega^2 by default). Per-rotor (shape (num_rotors,)). The direction is
        # applied after the aero corrections via self.thrust_dir (default +body-z).
        T_mag = self.thrust_c0 + self.thrust_c1 * rotor_speeds + self.k_eta * rotor_speeds**2

        # Add in aero wrenches (if applicable)
        if self.aero:
            # Parasitic drag force acting at the CoM
            D = -Multirotor._norm(body_airspeed_vector)*self.drag_matrix@body_airspeed_vector
            # Rotor drag (aka H force) acting at each propeller hub.
            H = -rotor_speeds*(self.rotor_drag_matrix@local_airspeeds)

            # Pitching flapping moment acting at each propeller hub.
            M_flap = -self.k_flap*rotor_speeds*((Multirotor.hat_map(local_airspeeds.T).transpose(2, 0, 1))@np.array([0,0,1])).T

            # Thrust correction for rotor angle-of-attack (alpha) and advance ratio (mu),
            # multiplying the base rotor thrust by (1 + k_angle*alpha + k_hor*mu). Default inert
            # (k_angle = k_hor = 0). Applied before translational lift so it scales only the
            # base thrust, matching SkyDreamer's k_w*(...)*sum Omega^2 form.
            if self.k_angle != 0.0 or self.k_hor != 0.0:
                w_bar = np.mean(rotor_speeds)
                denom = self.r_prop * w_bar
                alpha = np.arctan2(body_airspeed_vector[2], denom)
                mu = np.arctan2(np.hypot(body_airspeed_vector[0], body_airspeed_vector[1]), denom)
                T_mag = T_mag * (1.0 + self.k_angle*alpha + self.k_hor*mu)
            # Collective vertical airspeed-squared term (acts at the CoM along body z).
            if self.k_v2 != 0.0:
                vaz = body_airspeed_vector[2]
                D = D + np.array([0.0, 0.0, -self.k_v2 * vaz * abs(vaz)])

            # Translational lift (adds to thrust magnitude along the rotor axis).
            T_mag = T_mag + self.k_h*(local_airspeeds[0, :]**2 + local_airspeeds[1, :]**2)

        else:
            D = np.zeros(3,)
            H = np.zeros((3,self.num_rotors))
            M_flap = np.zeros((3,self.num_rotors))

        # Project thrust magnitudes onto each rotor's thrust axis (default +body-z). With a tilted
        # axis this yields parasitic x/y thrust components and the associated moments.
        T = self.thrust_dir.T * T_mag                                  # (3, num_rotors)

        # Compute the moments due to the rotor thrusts, rotor drag (if applicable), and rotor drag torques
        M_force = -np.einsum('ijk, ik->j', Multirotor.hat_map(self.rotor_geometry), T+H)
        # Yaw/reaction moment magnitude per rotor: dir_i * (torque_c0 + torque_c1*Omega_i + k_m_i*Omega_i^2),
        # directed along the rotor spin axis (thrust_dir). Reduces to dir_i * k_m_i * Omega_i^2 * z by default.
        M_yaw_mag = self.rotor_dir * (self.torque_c0 + self.torque_c1 * rotor_speeds + self.k_m * rotor_speeds**2)
        M_yaw = self.thrust_dir.T * M_yaw_mag                          # (3, num_rotors)

        # Sum all elements to compute the total body wrench
        FtotB = np.sum(T + H, axis=1) + D
        MtotB = M_force + np.sum(M_yaw + M_flap, axis=1)

        return (FtotB, MtotB)

    def rotor_inertia_moment(self, body_rates, rotor_speeds, rotor_accel):
        """
        Body-frame moment from the angular momentum of the spinning rotors:
          - reaction torque (yaw) opposing rotor angular acceleration, and
          - gyroscopic precession (roll/pitch) as the body rotation carries the rotor spin axes.
        Both vanish for balanced counter-rotating rotors at constant speed, and are exactly zero
        when rotor_inertia == 0 (the default), so this leaves nominal dynamics unchanged.

        Convention: the physical spin direction is spin_i = -rotor_dir_i, because rotor_dir encodes
        the sign of the aerodynamic *yaw torque* a rotor produces, which is opposite to its spin
        (finding F-6). Derivation: the reaction torque on the airframe is tau = -d/dt(h)|_inertial
        with net rotor angular momentum h = I_r * (sum_i spin_i * Omega_i) * zhat, and
        d/dt(h)|_inertial = I_r * (sum_i spin_i * Omega_dot_i) * zhat + omega x h.
        With omega x zhat = (q, -p, 0), the gyroscopic part is -h_z * (q, -p, 0) = (-h_z q, h_z p, 0).
        """
        if self.rotor_inertia == 0.0:
            return np.zeros(3)
        p, q, _r = body_rates
        spin = -self.rotor_dir
        h_z = self.rotor_inertia * np.sum(spin * rotor_speeds)   # net rotor angular momentum about +z_body
        # Gyroscopic precession (roll/pitch): -omega x (h_z zhat).
        M_gyro = np.array([-h_z * q, h_z * p, 0.0])
        # Reaction torque (yaw): -I_r * sum_i spin_i * Omega_dot_i.
        M_reaction_z = -self.rotor_inertia * np.sum(spin * rotor_accel)
        return M_gyro + np.array([0.0, 0.0, M_reaction_z])

    def _rotor_accel(self, cmd_rotor_speeds, rotor_speeds):
        """
        Rotor angular acceleration Omega_dot for the motor model.

        Default: first-order lag (Omega_c - Omega)/tau_m. When rotor_dyn_coef is set, use the
        asymmetric spin-up/spin-down model with separate coefficients for the accelerating and
        decelerating branches (motors typically brake more slowly than they spin up).
        """
        if not self._rotor_dyn_active:
            return (1.0 / self.tau_m) * (cmd_rotor_speeds - rotor_speeds)
        ka1, ka2, kd1, kd2 = self.rotor_dyn_coef
        d = cmd_rotor_speeds - rotor_speeds
        dsq = cmd_rotor_speeds**2 - rotor_speeds**2
        return np.where(cmd_rotor_speeds > rotor_speeds, ka1*d + ka2*dsq, kd1*d + kd2*dsq)

    def _thrust_to_speed(self, forces):
        """
        Invert the per-rotor thrust curve: force -> rotor speed.

        Default (pure quadratic, no linear/constant terms): sign(f)*sqrt(|f|/k_eta), preserving
        the original behavior exactly, including the sign convention for the (nonphysical) negative
        per-motor forces the SE3 allocation can request transiently.

        Polynomial curve active: solve thrust_c0 + thrust_c1*w + k_eta*w^2 = f for w >= 0 via the
        quadratic formula (Crazyflow's motor_force2rotor_vel), taking the physical (increasing)
        root and clamping the discriminant at 0.
        """
        forces = np.asarray(forces, dtype=float)
        if not self._thrust_poly_active:
            s = forces / self.k_eta
            return np.sign(s) * np.sqrt(np.abs(s))
        disc = self.thrust_c1**2 - 4.0*self.k_eta*(self.thrust_c0 - forces)
        disc = np.clip(disc, 0.0, None)
        return (-self.thrust_c1 + np.sqrt(disc)) / (2.0*self.k_eta)

    def get_cmd_motor_speeds(self, state, control):
        """
        Computes the commanded motor speeds depending on the control abstraction.
        For higher level control abstractions, we have low-level controllers that will produce motor speeds based on the higher level commmand.

        """

        if self.control_abstraction == 'cmd_motor_speeds':
            # The controller directly controls motor speeds, so command that.
            return control['cmd_motor_speeds']

        elif self.control_abstraction == 'cmd_motor_thrusts':
            # The controller commands individual motor forces.
            return self._thrust_to_speed(control['cmd_motor_thrusts'])

        elif self.control_abstraction == 'cmd_motor_throttle':
            # The controller commands normalized per-motor throttle u in [0, 1]. This is the
            # command path of a real ESC+battery: throttle maps nonlinearly to steady-state speed
            # (SkyDreamer). w_c = (w_max - w_min)*sqrt(k*u^2 + (1-k)*u) + w_min.
            # Add actuator command noise (eps_u, zero by default) before clipping to [0,1].
            u = np.asarray(control['cmd_motor_throttle'], dtype=float) + self.motor_cmd_noise
            u = np.clip(u, 0.0, 1.0)
            # Optional PWM quantization: snap u to the integer PWM grid (real ESCs are quantized).
            if self.pwm_max > self.pwm_min:
                levels = self.pwm_max - self.pwm_min
                u = np.round(u * levels) / levels
            k = self.motor_curve_k
            return (self.rotor_speed_max - self.rotor_speed_min) * np.sqrt(k*u**2 + (1.0 - k)*u) + self.rotor_speed_min

        elif self.control_abstraction == 'cmd_ctbm':
            # The controller commands collective thrust and moment on each axis. 
            cmd_thrust = control['cmd_thrust']
            cmd_moment = control['cmd_moment']  

        elif self.control_abstraction == 'cmd_ctbr':
            # The controller commands collective thrust and body rates on each axis. 

            cmd_thrust = control['cmd_thrust']

            # First compute the error between the desired body rates and the actual body rates given by state. 
            w_err = state['w'] - control['cmd_w']

            # Computed commanded moment based on the attitude error and body rate error
            wdot_cmd = -self.k_w*w_err
            cmd_moment = self.inertia@wdot_cmd

            # Now proceed with the cmd_ctbm formulation.

        elif self.control_abstraction == 'cmd_vel':
            # The controller commands a velocity vector. 
            
            # Get the error in the current velocity. 
            v_err = state['v'] - control['cmd_v']

            # Get desired acceleration based on P control of velocity error. 
            a_cmd = -self.k_v*v_err

            # Get desired force from this acceleration. 
            F_des = self.mass*(a_cmd + np.array([0, 0, self.g]))

            R = Rotation.from_quat(state['q']).as_matrix()
            b3 = R @ np.array([0, 0, 1])
            cmd_thrust = np.dot(F_des, b3)

            # Follow rest of SE3 controller to compute cmd moment. 

            # Desired orientation to obtain force vector.
            b3_des = F_des/np.linalg.norm(F_des)
            c1_des = np.array([1, 0, 0])
            b2_des = np.cross(b3_des, c1_des)/np.linalg.norm(np.cross(b3_des, c1_des))
            b1_des = np.cross(b2_des, b3_des)
            R_des = np.stack([b1_des, b2_des, b3_des]).T

            # Orientation error.
            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Angular control; vector units of N*m.
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])

        elif self.control_abstraction == 'cmd_ctatt':
            # The controller commands the collective thrust and attitude.

            cmd_thrust = control['cmd_thrust']

            # Compute the shape error from the current attitude and the desired attitude. 
            R = Rotation.from_quat(state['q']).as_matrix()
            R_des = Rotation.from_quat(control['cmd_q']).as_matrix()

            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Compute command moment based on attitude error. 
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])
        
        elif self.control_abstraction == 'cmd_acc':
            # The controller commands an acceleration vector (or thrust vector). This is equivalent to F_des in the SE3 controller. 
            F_des = control['cmd_acc']*self.mass

            R = Rotation.from_quat(state['q']).as_matrix()
            b3 = R @ np.array([0, 0, 1])
            cmd_thrust = np.dot(F_des, b3)

            # Desired orientation to obtain force vector.
            b3_des = F_des/np.linalg.norm(F_des)
            c1_des = np.array([1, 0, 0])
            b2_des = np.cross(b3_des, c1_des)/np.linalg.norm(np.cross(b3_des, c1_des))
            b1_des = np.cross(b2_des, b3_des)
            R_des = np.stack([b1_des, b2_des, b3_des]).T

            # Orientation error.
            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Angular control; vector units of N*m.
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])
        else:
            raise ValueError("Invalid control abstraction selected. Options are: cmd_motor_speeds, cmd_motor_thrusts, cmd_motor_throttle, cmd_ctbm, cmd_ctbr, cmd_ctatt, cmd_vel, cmd_acc")

        # Take the commanded thrust and body moments and convert them to motor speeds
        TM = np.concatenate(([cmd_thrust], cmd_moment))               # Concatenate thrust and moment into an array
        cmd_motor_forces = self.TM_to_f @ TM                                                # Convert to cmd_motor_forces from allocation matrix
        cmd_motor_speeds = self._thrust_to_speed(cmd_motor_forces)                          # Invert the (per-rotor, possibly polynomial) thrust curve
        # NOTE: the linear allocator TM_to_f assumes a quadratic-dominant thrust/torque model; when
        # a polynomial curve (thrust_c0/c1, torque_c0/c1) is active the force->speed inversion is
        # exact but the thrust/moment split is approximate. The forward dynamics remain exact.

        return cmd_motor_speeds

    def _on_ground(self, state):
        """
        Check if the vehicle is on the ground. 
        """
        return state['x'][2] <= 0.001

    def _handle_vehicle_on_ground(self, state):
        """
        Handle vehicle state while on ground.
        - Clamp altitude to ground plane (z = 0)
        - Prevent downward motion (v_z >= 0)
        - Apply horizontal velocity damping to emulate friction
        - Zero angular velocity and flatten attitude (keep yaw)
        """
        # Clamp position to ground
        state['x'][2] = 0.0

        # Prevent downward velocity
        if state['v'][2] < 0.0:
            state['v'][2] = 0.0

        # Horizontal velocity damping (friction-like)
        beta = self.ground_friction_beta
        state['v'][0:2] = (1.0 - beta) * state['v'][0:2]

        # Zero angular velocity and flatten attitude (keep yaw)
        state['w'] = np.zeros(3,)
        state['q'] = self.flatten_attitude(state['q'])

        return state

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)    ])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
        """
        if len(s.shape) > 1:  # Vectorized implementation
            return np.array([[ np.zeros(s.shape[0]), -s[:,2],  s[:,1]],
                             [ s[:,2],     np.zeros(s.shape[0]), -s[:,0]],
                             [-s[:,1],  s[:,0],     np.zeros(s.shape[0])]])
        else:
            return np.array([[    0, -s[2],  s[1]],
                             [ s[2],     0, -s[0]],
                             [-s[1],  s[0],     0]])

    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((20,))   # FIXME: this shouldn't be hardcoded. Should vary with the number of rotors. 
        s[0:3]   = state['x']       # inertial position
        s[3:6]   = state['v']       # inertial velocity
        s[6:10]  = state['q']       # orientation
        s[10:13] = state['w']       # body rates
        s[13:16] = state['wind']    # wind vector
        s[16:]   = state['rotor_speeds']     # rotor speeds

        return s

    @classmethod
    def _norm(cls, v):
        """
        Given a vector v in R^3, return the 2 norm (length) of the vector
        """
        norm = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        return norm

    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        """
        state = {'x':s[0:3], 'v':s[3:6], 'q':s[6:10], 'w':s[10:13], 'wind':s[13:16], 'rotor_speeds':s[16:]}
        return state

    @staticmethod
    def flatten_attitude(quaternion : List[float]) -> List[float]:
        """
        Set roll and pitch to 0 while keeping yaw unchanged.
        
        Parameters:
            quaternion (array-like): Quaternion [x, y, z, w] representing the quadrotor's attitude.
        
        Returns:
            numpy.ndarray: New quaternion with roll and pitch set to 0.
        """

        # Extract Euler angles in the 'XYZ' (roll, pitch, heading) convention wrt the world frame
        _, _, heading = R.from_quat(quaternion).as_euler('XYZ', degrees=False)
        
        # Create a new rotation object with roll and pitch set to 0
        flattened_rotation = R.from_euler('Z', heading, degrees=False)
        
        # Convert the new rotation back to a quaternion
        return flattened_rotation.as_quat()


class BatchedMultirotorParams:
    """ 
    A container class for various multirotor params. 
    Parameters:
        multirotor_params_list: list of dictionaries containing the parameters for each drone (see vehicles/crazyflie_params.py for example of such a dictionary).
        num_drones: number of drones in the simulation.
        device: the device to use for the simulation (e.g. torch.device('cuda')).
    """
    def __init__(self, multirotor_params_list, num_drones, device):
        assert len(multirotor_params_list) == num_drones
        self.num_drones = num_drones
        self.device = device
        self.mass = torch.tensor([multirotor_params['mass'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device) # kg

        self.num_rotors = multirotor_params_list[0]["num_rotors"]
        for multirotor_params in multirotor_params_list:
            assert multirotor_params["num_rotors"] == self.num_rotors

        self.rotor_pos = [multirotor_params["rotor_pos"] for multirotor_params in multirotor_params_list]

        self.rotor_dir_np       = np.array([qp['rotor_directions'] for qp in multirotor_params_list])

        self.extract_geometry()

        # Rotor parameters
        self.rotor_speed_min = torch.tensor([multirotor_params['rotor_speed_min'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device) # rad/s
        self.rotor_speed_max = torch.tensor([multirotor_params['rotor_speed_max'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device) # rad/s

        self.k_eta = np.array([multirotor_params['k_eta'] for multirotor_params in multirotor_params_list])  # .unsqueeze(-1).to(device)     # thrust coeff, N/(rad/s)**2
        self.k_m = np.array([multirotor_params['k_m'] for multirotor_params in multirotor_params_list])  # .unsqueeze(-1).to(device)
        self.k_flap = torch.tensor([multirotor_params['k_flap'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)   # Flapping moment coefficient Nm/(m/s)
        self.k_h = torch.tensor([multirotor_params['k_h'] for multirotor_params in multirotor_params_list]).unsqueeze(-1).double().to(device)   # translational lift coeff N/(m/s)**2

        # Motor parameters
        self.tau_m           = torch.tensor([multirotor_params['tau_m'] for multirotor_params in multirotor_params_list], device=device).unsqueeze(-1)
        self.motor_noise     = torch.tensor([multirotor_params['motor_noise_std'] for multirotor_params in multirotor_params_list], device=device).unsqueeze(-1) # noise added to the actual motor speed, rad/s / sqrt(Hz)

        # ---- Optional physics additions (all default-inert, mirroring the single-drone Multirotor) ----
        # NOTE: build tensors with dtype=double directly. torch.tensor(pylist).double() first rounds
        # the python floats to float32 (torch's default) and only then upcasts, silently losing
        # precision (e.g. 1/0.072 -> 13.88888931), which shows up as ~1e-4 rotor-accel errors.
        def _col(key, default):
            return torch.tensor([qp.get(key, default) for qp in multirotor_params_list], device=device, dtype=torch.double).unsqueeze(-1)
        # A2 polynomial thrust/torque curve linear+constant terms (k_eta/k_m are the quadratic terms).
        self.thrust_c0 = _col('thrust_c0', 0.0)
        self.thrust_c1 = _col('thrust_c1', 0.0)
        self.torque_c0 = _col('torque_c0', 0.0)
        self.torque_c1 = _col('torque_c1', 0.0)
        self.thrust_poly_active = any(qp.get('thrust_c0', 0.0) != 0.0 or qp.get('thrust_c1', 0.0) != 0.0 for qp in multirotor_params_list)
        # A5 rotor inertia (gyroscopic + reaction), 0 -> off.
        self.rotor_inertia = _col('rotor_inertia', 0.0)
        # A4 asymmetric spin-up/spin-down coefficients [ka1,ka2,kd1,kd2]; default [1/tau,0,1/tau,0] == first-order.
        _rdc = []
        for qp in multirotor_params_list:
            if 'rotor_dyn_coef' in qp:
                _rdc.append(list(qp['rotor_dyn_coef']))
            else:
                inv_tau = 1.0 / qp['tau_m']
                _rdc.append([inv_tau, 0.0, inv_tau, 0.0])
        self.rotor_dyn_coef = torch.tensor(_rdc, device=device, dtype=torch.double)  # (num_drones, 4)
        # B1 angle-of-attack / advance-ratio thrust correction, default inert.
        self.k_angle = _col('k_angle', 0.0)
        self.k_hor = _col('k_hor', 0.0)
        self.k_v2 = _col('k_v2', 0.0)
        self.r_prop = torch.tensor([qp.get('r_prop', qp.get('rotor_radius', 0.0)) for qp in multirotor_params_list], device=device, dtype=torch.double).unsqueeze(-1)
        self.aoa_active = any(qp.get('k_angle', 0.0) != 0.0 or qp.get('k_hor', 0.0) != 0.0 or qp.get('k_v2', 0.0) != 0.0 for qp in multirotor_params_list)
        # A3 motor throttle curve shape + PWM quantization for cmd_motor_throttle.
        self.motor_curve_k = _col('motor_curve_k', 1.0)
        self.pwm_min = _col('pwm_min', 0.0)
        self.pwm_max = _col('pwm_max', 0.0)
        # Guard: k_h (translational lift) cannot combine with k_angle/k_hor (double-counts in-plane airspeed).
        for qp in multirotor_params_list:
            if qp.get('k_h', 0.0) != 0.0 and (qp.get('k_angle', 0.0) != 0.0 or qp.get('k_hor', 0.0) != 0.0):
                raise ValueError("Conflicting aerodynamic thrust models: k_h cannot be combined with "
                                 "k_angle/k_hor (they double-count airspeed-dependent thrust). Set one to zero.")

        # Additional constants.
        self.inertia = torch.from_numpy(np.array([[[qp["Ixx"], qp["Ixy"], qp["Ixz"]],
                                                   [qp["Ixy"], qp["Iyy"], qp["Iyz"]],
                                                   [qp["Ixz"], qp["Iyz"], qp["Izz"]]] for qp in multirotor_params_list])).double().to(device)
        self.rotor_drag_matrix = torch.tensor([[[qp["k_d"],   0,                 0],
                                                [0,          qp["k_d"],          0],
                                                [0,          0,          qp["k_z"]]] for qp in multirotor_params_list], device=device).double()

        self.drag_matrix = torch.tensor([[[qp["c_Dx"],   0,                 0],
                                          [0,          qp["c_Dy"],          0],
                                          [0,          0,          qp["c_Dz"]]] for qp in multirotor_params_list], device=device).double()

        # keep this the same across drones.
        self.g = 9.81 # m/s^2

        self.inv_inertia = torch.linalg.inv(self.inertia).double()
        self.weight = torch.zeros(num_drones, 3, device=device).double()
        self.weight[:,-1] = -self.mass.squeeze(-1) * self.g

        # Control allocation
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient.

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        self.f_to_TM = torch.stack([torch.from_numpy(np.vstack((np.ones((1,self.num_rotors)),
                                                                np.hstack([np.cross(self.rotor_pos[i][key],
                                                                                    np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos[i]]),
                                                                (k[i] * self.rotor_dir_np[i]).reshape(1,-1)))).to(device) for i in range(num_drones)])
        self.k_eta = torch.from_numpy(self.k_eta).unsqueeze(-1).to(device)
        self.k_m = torch.from_numpy(self.k_m).unsqueeze(-1).to(device)
        self.rotor_dir = torch.from_numpy(self.rotor_dir_np).to(device)
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        # Low-Level Control Gains
        self.k_w = torch.tensor([multirotor_params.get('k_w', 1) for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)  # The body rate P gain (for cmd_ctbr)
        self.k_v = torch.tensor([multirotor_params.get('k_v', 10) for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)  # The body rate P gain (for cmd_ctbr)
        self.kp_att = torch.tensor([multirotor_params.get('kp_att', 3000.0) for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)
        self.kd_att = torch.tensor([multirotor_params.get('kd_att', 360.0) for multirotor_params in multirotor_params_list]).unsqueeze(-1).to(device)

    # Update methods that require some additional computations.
    # These are useful for dynamically updating the parameters of this object, for example for domain randomization.
    def update_mass(self, idx, mass):
        """
        Update the mass and weight of the drone at index idx.
        """
        self.mass[idx] = mass
        self.weight[idx,-1] = -mass * self.g

    def update_thrust_and_rotor_params(self, idx, k_eta = None, k_m = None, rotor_pos = None):
        """
        Update k_eta, k_m, and rotor positions of the drone at index idx.
        Some parameters are optional, and if not provided, the current value is kept.
        """
        if k_eta is not None:
            self.k_eta[idx] = k_eta
        if k_m is not None:
            self.k_m[idx] = k_m
        k_idx = self.k_m[idx]/self.k_eta[idx]
        if rotor_pos is not None:
            assert 'r1' in rotor_pos.keys() and 'r2' in rotor_pos.keys() and 'r3' in rotor_pos.keys() and 'r4' in rotor_pos.keys()
            self.rotor_pos[idx] = dict(rotor_pos[k_idx])
            rotor_geometry = np.array([]).reshape(0, 3)
            for rotor in rotor_pos:
                r = rotor_pos[rotor]
                rotor_geometry = np.vstack([rotor_geometry, r])
            self.rotor_geometry[idx] = torch.from_numpy(np.array(rotor_geometry)).double().to(self.device)
            self.rotor_geometry_hat_maps[idx] = BatchedMultirotor.hat_map(torch.from_numpy(rotor_geometry.squeeze())).double().to(self.device)

        self.f_to_TM[idx] = torch.from_numpy(np.vstack((np.ones((1,self.num_rotors)),
                                                                np.hstack([np.cross(self.rotor_pos[idx][key],
                                                                                    np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos[idx]]),
                                                                (k_idx.cpu() * self.rotor_dir_np[idx]).reshape(1,-1)))).to(self.device)
        self.TM_to_f[idx] = torch.linalg.inv(self.f_to_TM[idx])

    def update_inertia(self, idx, Ixx=None, Iyy=None, Izz=None):
        """
        Update the inertia of the drone at index idx. Parameters with value None are not updated.
        """
        self.inertia[idx][0,0] = Ixx
        self.inertia[idx][1,1] = Iyy
        self.inertia[idx][2,2] = Izz
        self.inv_inertia[idx] = torch.linalg.inv(self.inertia[idx])

    def update_drag(self, idx, c_Dx=None, c_Dy=None, c_Dz=None, k_d=None, k_z=None):
        """
        Update various drag parameters of the drone at index idx. Parameters with value None are not updated.
        """
        if c_Dx is not None:
            self.drag_matrix[idx][0,0] = c_Dx
        if c_Dy is not None:
            self.drag_matrix[idx][1,1] = c_Dy
        if c_Dz is not None:
            self.drag_matrix[idx][2,2] = c_Dz
        if k_d is not None:
            self.rotor_drag_matrix[idx][0,0] = k_d
            self.rotor_drag_matrix[idx][1,1] = k_d
        if k_z is not None:
            self.rotor_drag_matrix[idx][2,2] = k_z

    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of
        wrenches acting on the rigid body.
        The rotor_geometry is a tensor of shape (num_drones, n,3), where n is the number of rotors.
        Each row corresponds to the position vector of the rotor relative to the CoM.
        """

        geoms = []
        geom_hat_maps = []
        for i in range(self.num_drones):
            rotor_geometry = np.array([]).reshape(0, 3)
            for rotor in self.rotor_pos[i]:
                r = self.rotor_pos[i][rotor]
                rotor_geometry = np.vstack([rotor_geometry, r])
            geoms.append(rotor_geometry)
            geom_hat_maps.append(BatchedMultirotor.hat_map(torch.from_numpy(rotor_geometry.squeeze())).numpy())
        self.rotor_geometry = torch.from_numpy(np.array(geoms)).to(self.device)
        self.rotor_geometry_hat_maps = torch.from_numpy(np.array(geom_hat_maps)).to(self.device)


class BatchedMultirotor(object):
    """
    Batched Multirotor forward dynamics model, implemented in PyTorch.
    This class follows the same API as the original Multirotor class, but is designed to work with batches of drones.
    Generally, quantities that would be numpy arrays in `Multirotor` are instead torch tensors in this class.

    states: [position, velocity, attitude, body rates, wind, rotor speeds]

    Parameters:
        batched_params: BatchedMultirotorParams object, containing relevant physical parameters for the multirotor.
        num_drones: the number of drones in the batch.
        initial_states: the initial state of the vehicle. Contains the same keys as "initial_states" of `Multirotor`, but each 
                                value is a pytorch tensor with a prepended batch dimension. e.g. initial_states['x'].shape = (num_drones, 3).
                                To maintain fidelity during the simulation, the expected datatype for the tensors is double.
        control_abstraction: the appropriate control abstraction that is used by the controller, options are...
                                'cmd_motor_speeds': the controller directly commands motor speeds.
                                'cmd_motor_thrusts': the controller commands forces for each rotor.
                                'cmd_ctbr': the controller commands a collective thrsut and body rates.
                                'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
                                'cmd_ctatt': the controller commands a collective thrust and attitude (as a quaternion).
                                'cmd_vel': the controller commands a velocity vector in the world frame.
                                'cmd_acc': the controller commands a mass normalized thrust vector (acceleration) in the world frame.
        aero: boolean, determines whether or not aerodynamic drag forces are computed.
        integrator: str, "dopri5" or "rk4", which are adaptive or fixed step size integrators. "rk4" will be faster, but potentially less accurate.
    """

    def __init__(self, batched_params,
                 num_drones,
                 initial_states,
                 device,
                 control_abstraction='cmd_motor_speeds',
                 aero=True,
                 integrator='dopri5'
                 ):
        """
        Initialize quadrotor physical parameters.
        """
        assert initial_states['x'].device == device, "Initial states must already be on the specified device."
        assert initial_states['x'].shape[0] == num_drones
        assert batched_params.device == device
        self.num_drones = num_drones
        self.device = device
        self.params = batched_params

        # Set the initial state
        self.initial_states = initial_states
        self.control_abstraction = control_abstraction

        self.aero = aero

        # External disturbance wrench (world-frame force N, body-frame torque N*m), per drone,
        # held constant across an integration step. Zero by default so dynamics are unchanged.
        # See the single-drone Multirotor for rationale.
        self.external_force = torch.zeros(num_drones, 3, device=device).double()
        self.external_torque = torch.zeros(num_drones, 3, device=device).double()

        assert integrator == 'dopri5' or integrator == "rk4"
        self.integrator = integrator

    def statedot(self, state, control, t_step, idxs):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        Returns:
            vdot: torch.tensor of shape (num_drones, 3), with zeros for drones not in idxs
            wdot: torch.tensor of shape (num_drones, 3), with zeros for drones not in idxs
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control, idxs)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds, self.params.rotor_speed_min[idxs],
                                      self.params.rotor_speed_max[idxs])

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, idxs)

        s = BatchedMultirotor._pack_state(state, self.num_drones, self.device)

        s_dot = s_dot_fn(0, s[idxs])
        v_dot = torch.zeros_like(state["v"])
        w_dot = torch.zeros_like(state["w"])
        v_dot[idxs] = s_dot[..., 3:6].double()
        w_dot[idxs] = s_dot[..., 10:13].double()

        state_dot = {'vdot': v_dot, 'wdot': w_dot}
        return state_dot

    def step(self, state, control, t_step, idxs=None):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        Params:
            - state: dictionary containing keys ['x', 'v', 'q', 'w', 'wind', 'rotor_speeds'], and values which are pytorch tensors with dtype double and which have a batch dimension
            - control: dictionary with keys depending on the chosen control mode. Values are torch tensors, again with dtype double and with a batch dimension equal to the number of drones.
            - t_step: float, the duration for which to step the simulation.
            - idxs: integer array of shape (num_running_drones, )
        """
        if idxs is None:
            idxs = [i for i in range(self.num_drones)]
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control, idxs)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = torch.clip(cmd_rotor_speeds, self.params.rotor_speed_min[idxs],
                                      self.params.rotor_speed_max[idxs])

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, idxs)

        s = BatchedMultirotor._pack_state(state, self.num_drones, self.device)

        # Option 1 - RK45 integration
        # sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        sol = odeint(s_dot_fn, s[idxs], t=torch.tensor([0.0, t_step], device=self.device), method=self.integrator)
        # s = sol['y'][:,-1]
        s = sol[-1, :]
        # Option 2 - Euler integration
        # s = s + s_dot_fn(0, s) * t_step  # first argument doesn't matter. It's time invariant model

        state = BatchedMultirotor._unpack_state(s, idxs, self.num_drones)

        # Re-normalize unit quaternion.
        state['q'][idxs] = state['q'][idxs] / torch.norm(state['q'][idxs], dim=-1).unsqueeze(-1)

        # Add noise to the motor speed measurement
        state['rotor_speeds'][idxs] += torch.normal(mean=torch.zeros(self.params.num_rotors, device=self.device),
                                                    std=torch.ones(len(idxs), self.params.num_rotors,
                                                                   device=self.device) * torch.abs(
                                                        self.params.motor_noise[idxs]))
        state['rotor_speeds'][idxs] = torch.clip(state['rotor_speeds'][idxs], self.params.rotor_speed_min[idxs],
                                                 self.params.rotor_speed_max[idxs])

        return state

    # Cmd rotor speeds should already have the appropriate drones selected.
    def _s_dot_fn(self, t, s, cmd_rotor_speeds, idxs):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        # so this will be zero for some stuff.
        state = BatchedMultirotor._unpack_state(s, idxs, self.num_drones)

        rotor_speeds = state['rotor_speeds'][idxs]
        inertial_velocity = state['v'][idxs]
        wind_velocity = state['wind'][idxs]

        # R = Rotation.from_quat(state['q']).as_matrix()
        R = roma.unitquat_to_rotmat(state['q'][idxs]).double()

        # Rotor speed derivative: asymmetric spin-up/spin-down (reduces to first-order 1/tau_m by
        # default, since rotor_dyn_coef defaults to [1/tau,0,1/tau,0]).
        rdc = self.params.rotor_dyn_coef[idxs]
        ka1, ka2, kd1, kd2 = rdc[:, 0:1], rdc[:, 1:2], rdc[:, 2:3], rdc[:, 3:4]
        _d = cmd_rotor_speeds - rotor_speeds
        _dsq = cmd_rotor_speeds ** 2 - rotor_speeds ** 2
        rotor_accel = torch.where(cmd_rotor_speeds > rotor_speeds, ka1 * _d + ka2 * _dsq, kd1 * _d + kd2 * _dsq)

        # Position derivative.
        x_dot = state['v'][idxs]

        # Orientation derivative.
        q_dot = quat_dot_torch(state['q'][idxs], state['w'][idxs])

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.transpose(1, 2) @ (inertial_velocity - wind_velocity).unsqueeze(-1).double()
        body_airspeed_vector = body_airspeed_vector.squeeze(-1)

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'][idxs], rotor_speeds, body_airspeed_vector, idxs)

        # Add gyroscopic + reaction moments from the spinning rotors (zero if rotor_inertia == 0).
        MtotB = MtotB + self._rotor_inertia_moment(state['w'][idxs], rotor_speeds, rotor_accel, idxs)

        # Rotate the force from the body frame to the inertial frame, then add the external
        # disturbance force (world frame, zero by default).
        Ftot = R @ FtotB.unsqueeze(-1)
        Ftot = Ftot.squeeze(-1) + self.external_force[idxs]

        # Velocity derivative.
        v_dot = (self.params.weight[idxs] + Ftot) / self.params.mass[idxs]

        # Angular velocity derivative. Add the external disturbance torque (body frame, zero by default).
        w = state['w'][idxs].double()
        w_hat = BatchedMultirotor.hat_map(w).permute(2, 0, 1)
        w_dot = self.params.inv_inertia[idxs] @ (
                    MtotB + self.external_torque[idxs] - (w_hat.double() @ (self.params.inertia[idxs] @ w.unsqueeze(-1))).squeeze(-1)).unsqueeze(-1)

        # NOTE: the wind dynamics are currently handled in the wind_profile object.
        # The line below doesn't do anything, as the wind state is assigned elsewhere.
        wind_dot = torch.zeros((len(idxs), 3), device=self.device)

        # Pack into vector of derivatives. Use double: a float32 buffer here would truncate every
        # derivative (e.g. rotor accelerations ~1e4 rad/s^2 lose ~1e-4 abs precision) before the
        # integrator sees them, which is the dominant batched-vs-NumPy discrepancy (finding F-8).
        s_dot = torch.zeros((len(idxs), 16 + self.params.num_rotors,), device=self.device, dtype=torch.double)
        s_dot[:, 0:3] = x_dot
        s_dot[:, 3:6] = v_dot
        s_dot[:, 6:10] = q_dot
        s_dot[:, 10:13] = w_dot.squeeze(-1)
        s_dot[:, 13:16] = wind_dot
        s_dot[:, 16:] = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector, idxs):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed
        for aerodynamic forces.
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame.
        The net moment Mtot is represented in the body frame.
        """
        assert (body_rates.shape[0] == rotor_speeds.shape[0])
        assert (body_rates.shape[0] == body_airspeed_vector.shape[0])

        num_drones = body_rates.shape[0]
        # Get the local airspeeds for each rotor
        local_airspeeds = body_airspeed_vector.unsqueeze(-1) + (
            BatchedMultirotor.hat_map(body_rates).permute(2, 0, 1)) @ (self.params.rotor_geometry[idxs].transpose(1, 2))

        # Compute the thrust of each rotor (polynomial thrust_c0 + thrust_c1*Omega + k_eta*Omega^2;
        # reduces to k_eta*Omega^2 by default). Rotors point in the body z direction.
        T = torch.zeros(num_drones, 3, 4, device=self.device).double()
        T[..., -1, :] = (self.params.thrust_c0[idxs] + self.params.thrust_c1[idxs] * rotor_speeds
                         + self.params.k_eta[idxs] * rotor_speeds ** 2)

        # Add in aero wrenches (if applicable)
        if self.aero:
            # Parasitic drag force acting at the CoM
            tmp = self.params.drag_matrix[idxs] @ (body_airspeed_vector).unsqueeze(-1)
            D = -BatchedMultirotor._norm(body_airspeed_vector).unsqueeze(-1) * tmp.squeeze()
            # Rotor drag (aka H force) acting at each propeller hub.
            tmp = self.params.rotor_drag_matrix[idxs] @ local_airspeeds.double()
            H = -rotor_speeds.unsqueeze(1) * tmp
            # Pitching flapping moment acting at each propeller hub.
            M_flap = BatchedMultirotor.hat_map(local_airspeeds.transpose(1, 2).reshape(num_drones * 4, 3))
            M_flap = M_flap.permute(2, 0, 1).reshape(num_drones, 4, 3, 3).double()
            M_flap = M_flap @ torch.tensor([0, 0, 1.0], device=self.device).double()
            M_flap = (-self.params.k_flap[idxs] * rotor_speeds).unsqueeze(1) * M_flap.transpose(-1, -2)

            # Thrust correction for rotor angle-of-attack (alpha) and advance ratio (mu); default
            # inert (per-drone coeffs 0 -> factor 1). Applied before translational lift.
            if self.params.aoa_active:
                w_bar = torch.mean(rotor_speeds, dim=1, keepdim=True)
                denom = self.params.r_prop[idxs] * w_bar
                vbz = body_airspeed_vector[:, 2:3]
                v_hor = torch.sqrt(body_airspeed_vector[:, 0:1] ** 2 + body_airspeed_vector[:, 1:2] ** 2)
                alpha = torch.atan2(vbz, denom)
                mu = torch.atan2(v_hor, denom)
                factor = 1.0 + self.params.k_angle[idxs] * alpha + self.params.k_hor[idxs] * mu   # (n,1)
                T[..., -1, :] = T[..., -1, :] * factor
                D[:, 2:3] = D[:, 2:3] - self.params.k_v2[idxs] * vbz * torch.abs(vbz)

            lift = torch.zeros(num_drones, 3, 1, device=self.device).double()
            lift[:, 2, :] = self.params.k_h[idxs]
            lift = torch.bmm(lift, (local_airspeeds[:, 0, :] ** 2 + local_airspeeds[:, 1, :] ** 2).unsqueeze(1))
            T += lift
        else:
            D = torch.zeros(num_drones, 3, device=self.device).double()
            H = torch.zeros((num_drones, 3, self.params.num_rotors), device=self.device).double()
            M_flap = torch.zeros((num_drones, 3, self.params.num_rotors), device=self.device).double()

        # Compute the moments due to the rotor thrusts, rotor drag (if applicable), and rotor drag torques
        M_force = -torch.einsum('bijk, bik->bj', self.params.rotor_geometry_hat_maps[idxs], T + H)
        # Yaw moment per rotor: dir * (torque_c0 + torque_c1*Omega + k_m*Omega^2).
        M_yaw = torch.zeros(num_drones, 3, 4, device=self.device).double()
        M_yaw[..., -1, :] = self.params.rotor_dir[idxs] * (self.params.torque_c0[idxs]
                            + self.params.torque_c1[idxs] * rotor_speeds + self.params.k_m[idxs] * rotor_speeds ** 2)

        # Sum all elements to compute the total body wrench
        FtotB = torch.sum(T + H, dim=2) + D
        MtotB = M_force + torch.sum(M_yaw + M_flap, dim=2)

        return (FtotB, MtotB)

    def _rotor_inertia_moment(self, w, rotor_speeds, rotor_accel, idxs):
        """
        Batched gyroscopic + reaction moment from the spinning rotors (see the single-drone
        Multirotor.rotor_inertia_moment). Physical spin = -rotor_dir (finding F-6); zero when
        rotor_inertia == 0. w: (n,3) body rates; rotor_speeds, rotor_accel: (n,4).
        """
        I_r = self.params.rotor_inertia[idxs]              # (n,1)
        spin = -self.params.rotor_dir[idxs]                # (n,4)
        h_z = I_r * torch.sum(spin * rotor_speeds, dim=1, keepdim=True)   # (n,1)
        p = w[:, 0:1].double()
        q = w[:, 1:2].double()
        M = torch.zeros(len(idxs), 3, device=self.device).double()
        M[:, 0:1] = -h_z * q
        M[:, 1:2] = h_z * p
        M[:, 2:3] = -I_r * torch.sum(spin * rotor_accel, dim=1, keepdim=True)
        return M

    def _thrust_to_speed(self, forces, idxs):
        """
        Invert the thrust curve force -> rotor speed (batched). Default (no poly terms):
        sign(f)*sqrt(|f|/k_eta), exactly as before. Polynomial active: quadratic-formula root.
        """
        if not self.params.thrust_poly_active:
            s = forces / self.params.k_eta[idxs]
            return torch.sign(s) * torch.sqrt(torch.abs(s))
        c0 = self.params.thrust_c0[idxs]
        c1 = self.params.thrust_c1[idxs]
        c2 = self.params.k_eta[idxs]
        disc = torch.clamp(c1 ** 2 - 4.0 * c2 * (c0 - forces), min=0.0)
        return (-c1 + torch.sqrt(disc)) / (2.0 * c2)

    # FIXME(hersh500): since so much of this code is shared with the SE3 Controller, it should really be
    # cleaned up and split into different functions that can be shared across both objects.
    def get_cmd_motor_speeds(self, state, control, idxs):
        """
        Computes the commanded motor speeds depending on the control abstraction.
        For higher level control abstractions, we have low-level controllers that will produce motor speeds based on the higher level commmand.
        """

        if self.control_abstraction == 'cmd_motor_speeds':
            # The controller directly controls motor speeds, so command that.
            return control['cmd_motor_speeds'][idxs]
        elif self.control_abstraction == "cmd_motor_thrusts":
            return self._thrust_to_speed(control["cmd_motor_thrusts"][idxs], idxs)
        elif self.control_abstraction == "cmd_motor_throttle":
            u = torch.clip(control['cmd_motor_throttle'][idxs], 0.0, 1.0).double()
            levels = self.params.pwm_max[idxs] - self.params.pwm_min[idxs]
            safe_levels = torch.where(levels > 0, levels, torch.ones_like(levels))
            u = torch.where(levels > 0, torch.round(u * safe_levels) / safe_levels, u)
            k = self.params.motor_curve_k[idxs]
            wmin = self.params.rotor_speed_min[idxs]
            wmax = self.params.rotor_speed_max[idxs]
            return (wmax - wmin) * torch.sqrt(k * u ** 2 + (1.0 - k) * u) + wmin
        elif self.control_abstraction == "cmd_ctbm":
            cmd_thrust = control['cmd_thrust'][idxs]
            cmd_moment = control['cmd_moment'][idxs]
        elif self.control_abstraction == "cmd_ctbr":
            cmd_thrust = control['cmd_thrust'][idxs]

            # First compute the error between the desired body rates and the actual body rates given by state.
            w_err = state['w'][idxs] - control['cmd_w'][idxs]

            # Computed commanded moment based on the attitude error and body rate error
            wdot_cmd = -self.params.k_w[idxs] * w_err
            cmd_moment = self.params.inertia[idxs] @ wdot_cmd.unsqueeze(-1)
        elif self.control_abstraction == "cmd_vel":
            # The controller commands a velocity vector.
            # Get the error in the current velocity.
            v_err = state['v'][idxs] - control['cmd_v'][idxs]

            # Get desired acceleration based on P control of velocity error.
            a_cmd = -self.params.k_v[idxs] * v_err

            # Get desired force from this acceleration.
            F_des = self.params.mass[idxs] * (a_cmd + np.array([0, 0, self.params.g]))

            R = roma.unitquat_to_rotmat(state['q'][idxs]).double()
            b3 = R @ torch.tensor([0.0, 0.0, 1.0], device=self.device).double()
            cmd_thrust = torch.sum(F_des * b3, dim=-1).double().unsqueeze(-1)

            # Follow rest of SE3 controller to compute cmd moment.
            # Desired orientation to obtain force vector.
            b3_des = F_des / torch.norm(F_des, dim=-1, keepdim=True)
            c1_des = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).double()
            b2_des = torch.cross(b3_des, c1_des, dim=-1) / torch.norm(torch.cross(b3_des, c1_des, dim=-1), dim=-1,
                                                                      keepdim=True)
            b1_des = torch.cross(b2_des, b3_des, dim=-1)
            R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

            # Orientation error.
            S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

            # Angular control; vector units of N*m.
            Iw = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).double()
            tmp = -self.params.kp_att[idxs] * att_err - self.params.kd_att[idxs] * state['w']
            cmd_moment = (self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1) + torch.cross(state['w'][idxs],
                                                                                                   Iw.squeeze(-1),
                                                                                                   dim=-1)
        elif self.control_abstraction == "cmd_ctatt":
            cmd_thrust = control["cmd_thrust"][idxs]
            R = roma.unitquat_to_rotmat(state['q'][idxs]).double()
            R_des = roma.unitquat_to_rotmat(control["cmd_q"][idxs]).double()
            S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)
            Iw = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).double()
            tmp = -self.params.kp_att[idxs] * att_err - self.params.kd_att[idxs] * state['w'][idxs]
            cmd_moment = (self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1) + torch.cross(state['w'][idxs],
                                                                                                   Iw.squeeze(-1),
                                                                                                   dim=-1)
        elif self.control_abstraction == "cmd_acc":
            F_des = control['cmd_acc'][idxs] * self.params.mass[idxs]
            R = roma.unitquat_to_rotmat(state['q'][idxs]).double()
            b3 = R @ torch.tensor([0.0, 0.0, 1.0], device=self.device).double()
            cmd_thrust = torch.sum(F_des * b3, dim=-1).double().unsqueeze(-1)

            # Follow rest of SE3 controller to compute cmd moment.
            # Desired orientation to obtain force vector.
            b3_des = F_des / torch.norm(F_des, dim=-1, keepdim=True)
            c1_des = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).double()
            b2_des = torch.cross(b3_des, c1_des, dim=-1) / torch.norm(torch.cross(b3_des, c1_des, dim=-1), dim=-1,
                                                                      keepdim=True)
            b1_des = torch.cross(b2_des, b3_des, dim=-1)
            R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

            # Orientation error.
            S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
            att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

            # Angular control; vector units of N*m.
            Iw = self.params.inertia[idxs] @ state['w'][idxs].unsqueeze(-1).double()
            tmp = -self.params.kp_att[idxs] * att_err - self.params.kd_att[idxs] * state['w']
            cmd_moment = (self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1) + torch.cross(state['w'][idxs],
                                                                                                   Iw.squeeze(-1),
                                                                                                   dim=-1)
        else:
            raise ValueError(
                "Invalid control abstraction selected. Options are: cmd_motor_speeds, cmd_motor_thrusts, cmd_ctbm, cmd_ctbr, cmd_ctatt, cmd_vel, cmd_acc")

        TM = torch.cat([cmd_thrust, cmd_moment.squeeze(-1)], dim=-1)
        cmd_rotor_thrusts = (self.params.TM_to_f[idxs] @ TM.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
        cmd_motor_speeds = self._thrust_to_speed(cmd_rotor_thrusts, idxs)
        return cmd_motor_speeds

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                         2 * (q[1] * q[2] - q[0] * q[3]),
                         1 - 2 * (q[0] ** 2 + q[1] ** 2)])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
        """
        device = s.device
        if len(s.shape) > 1:  # Vectorized implementation
            s = s.unsqueeze(-1)
            hat = torch.cat([torch.zeros(s.shape[0], 1, device=device), -s[:, 2], s[:, 1],
                             s[:, 2], torch.zeros(s.shape[0], 1, device=device), -s[:, 0],
                             -s[:, 1], s[:, 0], torch.zeros(s.shape[0], 1, device=device)], dim=0).view(3, 3, s.shape[
                0]).double()
            return hat
        else:
            return torch.tensor([[0, -s[2], s[1]],
                                 [s[2], 0, -s[0]],
                                 [-s[1], s[0], 0]], device=device)

    @classmethod
    def _pack_state(cls, state, num_drones, device):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = torch.zeros(num_drones, 20,
                        device=device).double()  # FIXME: this shouldn't be hardcoded. Should vary with the number of rotors.
        s[..., 0:3] = state['x']  # inertial position
        s[..., 3:6] = state['v']  # inertial velocity
        s[..., 6:10] = state['q']  # orientation
        s[..., 10:13] = state['w']  # body rates
        s[..., 13:16] = state['wind']  # wind vector
        s[..., 16:] = state['rotor_speeds']  # rotor speeds

        return s

    @classmethod
    def _norm(cls, v):
        """
        Given a vector v in R^3, return the 2 norm (length) of the vector
        """
        # norm = (v[...,0]**2 + v[...,1]**2 + v[...,2]**2)**0.5
        norm = torch.linalg.norm(v, dim=-1)
        return norm

    @classmethod
    def _unpack_state(cls, s, idxs, num_drones):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        """
        device = s.device
        state = {'x': torch.full((num_drones, 3), float("nan"), device=device).double(),
                 'v': torch.full((num_drones, 3), float("nan"), device=device).double(),
                 'q': torch.full((num_drones, 4), float("nan"), device=device).double(),
                 'w': torch.full((num_drones, 3), float("nan"), device=device).double(),
                 'wind': torch.full((num_drones, 3), float("nan"), device=device).double(),
                 'rotor_speeds': torch.full((num_drones, 4), float("nan"), device=device).double()}
        state['q'][..., -1] = 1  # make sure we're returning a valid quaternion
        state['x'][idxs] = s[:, 0:3]
        state['v'][idxs] = s[:, 3:6]
        state['q'][idxs] = s[:, 6:10]
        state['w'][idxs] = s[:, 10:13]
        state['wind'][idxs] = s[:, 13:16]
        state['rotor_speeds'][idxs] = s[:, 16:]
        return state
