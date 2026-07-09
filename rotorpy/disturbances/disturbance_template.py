"""
Disturbance profiles are implemented here, analogous to the wind profiles in rotorpy/wind/.

A disturbance profile injects an *external wrench* into the rigid-body dynamics: a force
(world frame, N) and a torque (body frame, N*m) that stand in for everything the analytical
model does not capture (unmodeled aerodynamics, motor mismatch, turbulence-like buffeting, and
so on). This is the SkyDreamer sim-to-real device (arXiv:2510.14783, Table III) and mirrors the
first-class disturbance inputs in Crazyflow's first-principles dynamics.

The disturbance is held constant across a single integration step (resample-and-hold), which is
what makes it a physically consistent piecewise-constant process rather than integration noise.

Each profile must provide:
- __init__ : any parameters/constants defining the disturbance.
- update(t, state) : return the current disturbance wrench for the step starting at time t.
  It must return a dict with keys 'force' (shape (3,), world frame, N) and 'torque'
  (shape (3,), body frame, N*m).
"""
import numpy as np


class DisturbanceTemplate(object):
    """Base disturbance profile: returns a zero wrench for all time."""

    def __init__(self):
        pass

    def update(self, t, state):
        """
        Given the present time and vehicle state dict, return the disturbance to hold over the
        upcoming integration step.

        Returns:
            dict with:
                'force',  np.ndarray shape (3,), external force in the WORLD frame [N]
                'torque', np.ndarray shape (3,), external torque in the BODY frame [N*m]
                'motor_cmd_noise', np.ndarray shape (num_rotors,), additive noise on the normalized
                    actuator command u in [0,1] (SkyDreamer eps_u). Only affects the
                    'cmd_motor_throttle' abstraction; ignored by other control modes.
        """
        return {'force': np.zeros(3), 'torque': np.zeros(3), 'motor_cmd_noise': np.zeros(4)}
