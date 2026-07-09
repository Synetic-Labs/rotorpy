"""
Cross-simulator validation against Crazyflow's ACTUAL first-principles dynamics
(learnsyslab/crazyflow, crazyflow/dynamics/first_principles/dynamics.py), run as real code.

We import their numeric dynamics() via a bare-package shim (avoids the crazyflow package __init__,
which imports mujoco.mjx). array_api_compat routes numpy inputs through numpy, so we run their
exact formula.

Validates, component by component:
  1. Polynomial thrust (A2): RotorPy body-z thrust == Crazyflow motor thrust sum.
  2. Rotor dynamics (A4): RotorPy _rotor_accel == Crazyflow rotor_vel_dot.
  3. Prop-inertia torque (A5): RotorPy rotor_inertia_moment vs Crazyflow torque_inertia,
     isolated as J @ (ang_vel_dot[I_r] - ang_vel_dot[0]).  <-- confirms/【corrects】finding F-3.

Convention mapping used: Crazyflow mixing_matrix z-row == RotorPy rotor_directions (both encode the
yaw-torque sign; physical spin = -that). Crazyflow rotor speeds are RPM; RotorPy uses rad/s
(factor 2*pi/60). Crazyflow coefficients are RPM-unit; converted where needed.

Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_xsim_crazyflow.py
"""
import os
import sys
import types
import numpy as np

# --- import Crazyflow's real dynamics() without running its package __init__ (which needs mjx) ---
# Look for the clone in the stable repo-local location first, then the old session /tmp path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "..", "refs", "crazyflow"),
    "/tmp/claude-1000/-home-james-code-RotorPy/946a5ea1-e934-4627-a38d-1b10cfb77116/scratchpad/crazyflow",
]
CF = next((os.path.abspath(c) for c in _CANDIDATES if os.path.isdir(c)), None)
if CF is None:
    print("SKIP: crazyflow clone not found (expected scratchpad/refs/crazyflow). "
          "Clone with: git clone --depth 1 https://github.com/learnsyslab/crazyflow scratchpad/refs/crazyflow")
    sys.exit(0)
sys.path.insert(0, CF)
_pkg = types.ModuleType("crazyflow")
_pkg.__path__ = [os.path.join(CF, "crazyflow")]
sys.modules["crazyflow"] = _pkg
from crazyflow.dynamics.first_principles.dynamics import dynamics as cf_dynamics  # noqa: E402

from rotorpy.vehicles.multirotor import Multirotor  # noqa: E402
from rotorpy.vehicles.crazyflie_params import quad_params  # noqa: E402
import copy  # noqa: E402

RPM2RAD = 2 * np.pi / 60.0

# cf2x_L250 identified params (crazyflow drones/params.toml).
CFP = dict(
    mass=0.0319, L=0.03253, prop_inertia=34.52e-9,
    gravity_vec=np.array([0.0, 0.0, -9.81]),
    J=np.diag([16.8e-6, 16.8e-6, 29.8e-6]),
    rpm2thrust=np.array([0.0, -5.382196214637237e-7, 2.4582929831265485e-10]),
    rpm2torque=np.array([0.0, 1.410454111996297e-9, 1.4592584373980652e-12]),
    mixing_matrix=np.array([[-1., -1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1]]),
    drag_matrix=np.diag([-0.01471782, -0.01471782, -0.01277641]),
    rotor_dyn_coef=np.array([7.355623702172756, 0.0, 0.0, 0.00024443862952110715]),
)
MIX_Z = CFP["mixing_matrix"][2, :]   # yaw-torque sign row -> RotorPy rotor_directions


def cf_call(pos, quat, vel, ang, cmd, rotor, prop_inertia):
    p = dict(CFP)
    p["prop_inertia"] = prop_inertia
    J = p["J"]
    out = cf_dynamics(pos, quat, vel, ang, cmd, rotor_vel=rotor, J_inv=np.linalg.inv(J), **p)
    return tuple(np.asarray(o) for o in out)


def rotorpy_vehicle(**over):
    p = copy.deepcopy(quad_params)
    # Match the cf2x_L250 platform where it matters for the compared component.
    p["mass"] = CFP["mass"]
    p["Ixx"], p["Iyy"], p["Izz"] = 16.8e-6, 16.8e-6, 29.8e-6
    p["rotor_directions"] = MIX_Z.astype(float)
    p.update(over)
    return Multirotor(p, control_abstraction="cmd_motor_speeds", aero=True)


def test_poly_thrust():
    print("=== Crazyflow xsim: polynomial thrust (A2) ===")
    # Isolate thrust: zero gravity and drag so Crazyflow vel_dot == R @ thrust / mass; identity quat.
    p = dict(CFP); p["gravity_vec"] = np.zeros(3); p["drag_matrix"] = np.zeros((3, 3))
    rotor_rpm = np.array([9000., 11000., 9500., 10500.])
    out = cf_dynamics(np.zeros(3), np.array([0, 0, 0, 1.]), np.zeros(3), np.zeros(3),
                      rotor_rpm, rotor_vel=rotor_rpm, J_inv=np.linalg.inv(p["J"]), **p)
    vel_dot = np.asarray(out[2])
    cf_thrust_z = vel_dot[2] * CFP["mass"]     # body/world z aligned at identity quat

    # RotorPy: poly thrust with coefficients converted RPM -> rad/s, aero off (pure thrust).
    c1 = CFP["rpm2thrust"][1] / RPM2RAD
    c2 = CFP["rpm2thrust"][2] / RPM2RAD**2
    m = rotorpy_vehicle(thrust_c1=c1, k_eta=c2, k_d=0.0, k_z=0.0, k_h=0.0,
                        c_Dx=0.0, c_Dy=0.0, c_Dz=0.0)
    m.aero = False
    F, _ = m.compute_body_wrench(np.zeros(3), rotor_rpm * RPM2RAD, np.zeros(3))
    err = abs(F[2] - cf_thrust_z)
    ok = err < 1e-9
    print(f"  [{'OK ' if ok else 'FAIL'}] RotorPy thrust {F[2]:.6e} == Crazyflow {cf_thrust_z:.6e} (err={err:.1e})")
    return ok


def test_rotor_dynamics():
    print("=== Crazyflow xsim: rotor spin-up/spin-down dynamics (A4) ===")
    rotor_rpm = np.array([9000., 11000., 9500., 10500.])
    cmd_rpm = np.array([12000., 8000., 9500., 11000.])   # mix of spin-up and spin-down
    out = cf_call(np.zeros(3), np.array([0, 0, 0, 1.]), np.zeros(3), np.zeros(3),
                  cmd_rpm, rotor_rpm, CFP["prop_inertia"])
    cf_rotor_dot = out[4]   # RPM/s

    # RotorPy _rotor_accel with the SAME coefficients, in RPM units (unit-agnostic form).
    m = rotorpy_vehicle(rotor_dyn_coef=list(CFP["rotor_dyn_coef"]))
    rp_rotor_dot = m._rotor_accel(cmd_rpm, rotor_rpm)
    err = np.max(np.abs(rp_rotor_dot - cf_rotor_dot))
    ok = err < 1e-6
    print(f"  [{'OK ' if ok else 'FAIL'}] max err = {err:.2e}")
    print(f"        cf   = {cf_rotor_dot}")
    print(f"        rpy  = {rp_rotor_dot}")
    return ok


def test_prop_inertia_F3():
    print("=== Crazyflow xsim: prop-inertia torque (A5) -- FINDING F-3 ===")
    I_r = CFP["prop_inertia"]
    rotor_rpm = np.array([9000., 11000., 9500., 10500.])
    cmd_rpm = np.array([12000., 8000., 9500., 11000.])
    ang = np.array([1.3, -0.7, 0.4])   # p, q, r (nonzero to expose gyro x/y)

    # Isolate Crazyflow's torque_inertia: J @ (ang_vel_dot[I_r] - ang_vel_dot[0]).
    out_Ir = cf_call(np.zeros(3), np.array([0, 0, 0, 1.]), np.zeros(3), ang, cmd_rpm, rotor_rpm, I_r)
    out_0 = cf_call(np.zeros(3), np.array([0, 0, 0, 1.]), np.zeros(3), ang, cmd_rpm, rotor_rpm, 0.0)
    tau_cf = CFP["J"] @ (out_Ir[3] - out_0[3])

    # RotorPy rotor_inertia_moment with matched drone: rotor_dir = mix_z, speeds/accel in rad/s.
    m = rotorpy_vehicle(rotor_inertia=I_r)
    Omega = rotor_rpm * RPM2RAD
    Omega_dot = out_Ir[4] * RPM2RAD          # Crazyflow's own rotor_vel_dot, converted
    tau_rp = m.rotor_inertia_moment(ang, Omega, Omega_dot)

    dz = abs(tau_cf[2] - tau_rp[2])
    dy = abs(tau_cf[1] - tau_rp[1])
    dx = abs(tau_cf[0] - tau_rp[0])
    print(f"        Crazyflow torque_inertia = {tau_cf}")
    print(f"        RotorPy   rotor_inertia  = {tau_rp}")
    print(f"        |diff| = [x {dx:.3e}, y {dy:.3e}, z {dz:.3e}]")
    # Expectation from analysis: reaction (z) agrees; the gyroscopic cross term disagrees on exactly
    # one axis because Crazyflow gives x and y the same leading sign while -omega x h needs opposite.
    scale = np.max(np.abs(tau_cf)) + 1e-30
    agree = lambda d: d / scale < 1e-6
    n_axes_agree = sum(agree(d) for d in (dx, dy, dz))
    z_ok = agree(dz)
    one_axis_flips = (n_axes_agree == 2)   # exactly one of x/y disagrees
    # Confirm the disagreeing axis is a pure sign flip (magnitudes equal).
    flip_x = np.isclose(tau_cf[0], -tau_rp[0]) and not agree(dx)
    flip_y = np.isclose(tau_cf[1], -tau_rp[1]) and not agree(dy)
    sign_flip = flip_x or flip_y
    ok = z_ok and one_axis_flips and sign_flip
    which = "x" if flip_x else ("y" if flip_y else "?")
    print(f"  [{'OK ' if ok else 'FAIL'}] reaction(z) agrees; gyro sign-flips on '{which}' axis "
          f"(exactly one axis) -> F-3 confirmed against real Crazyflow code")
    return ok, which


if __name__ == "__main__":
    r1 = test_poly_thrust()
    r2 = test_rotor_dynamics()
    r3, which = test_prop_inertia_F3()
    print()
    print(f"NOTE: F-3 divergence is on the '{which}' axis.")
    print("ALL PASSED" if all([r1, r2, r3]) else "SOME FAILED")
