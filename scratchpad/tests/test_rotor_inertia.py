"""
Verification for A5: rotor-inertia reaction torque + gyroscopic precession.

Checks:
1. Regression: rotor_inertia=0 (default) => zero moment, nominal dynamics unchanged.
2. Gyroscopic term equals an independent -omega x h cross product (h = I_r*(sum spin_i Omega_i)*zhat),
   with spin_i = -rotor_dir_i.
3. Reaction term: differential spin-up (accelerate each rotor along its spin direction) produces
   yaw reaction -I_r * sum spin_i Omega_dot_i.
4. Balance invariance: equal speeds on a balanced counter-rotating quad + constant speed => zero
   moment even at nonzero body rate.
5. Sign contrast vs Crazyflow (finding F-3): our gyro x,y have OPPOSITE signs; document the delta.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_rotor_inertia.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

I_R = 3.5e-8  # kg m^2, ~ crazyflie prop inertia (Crazyflow cf2x_L250)


def make(rotor_inertia):
    p = copy.deepcopy(quad_params)
    p['rotor_inertia'] = rotor_inertia
    return Multirotor(p, control_abstraction='cmd_motor_speeds', aero=False)


def test_regression_zero():
    print("=== A5 regression: rotor_inertia=0 => zero moment ===")
    m = make(0.0)
    M = m.rotor_inertia_moment(np.array([1., 2., 3.]), np.full(4, 2000.), np.full(4, 500.))
    ok = np.allclose(M, 0.0)
    print(f"  [{'OK ' if ok else 'FAIL'}] |M|={np.max(np.abs(M)):.2e}")
    return ok


def test_gyro_cross_product():
    print("=== A5 gyroscopic term == -omega x h (independent cross product) ===")
    m = make(I_R)
    spin = -m.rotor_dir
    # Imbalanced rotor speeds so h_z != 0.
    speeds = np.array([2200., 1800., 2100., 1700.])
    omega = np.array([3.0, -2.0, 1.5])
    # Isolate the gyro part by zeroing rotor acceleration.
    M = m.rotor_inertia_moment(omega, speeds, np.zeros(4))
    h_z = I_R * np.sum(spin * speeds)
    h = np.array([0.0, 0.0, h_z])
    M_ref = -np.cross(omega, h)
    err = np.max(np.abs(M - M_ref))
    ok = err < 1e-18
    print(f"  [{'OK ' if ok else 'FAIL'}] err vs -omega x h = {err:.2e}  (h_z={h_z:.3e})")
    print(f"        M_gyro = {M},  ref = {M_ref}")
    return ok


def test_reaction():
    print("=== A5 reaction torque from differential spin-up ===")
    m = make(I_R)
    spin = -m.rotor_dir
    a = 800.0  # rad/s^2 magnitude
    rotor_accel = spin * a          # accelerate each rotor along its own spin direction
    M = m.rotor_inertia_moment(np.zeros(3), np.full(4, 2000.), rotor_accel)
    expect_z = -I_R * np.sum(spin * rotor_accel)   # = -I_R * sum(spin^2 * a) = -I_R * 4a
    ok = abs(M[2] - expect_z) < 1e-18 and np.max(np.abs(M[:2])) < 1e-18
    print(f"  [{'OK ' if ok else 'FAIL'}] Mz={M[2]:.3e} expect={expect_z:.3e} (=-I_r*4a)")
    return ok


def test_balance_invariance():
    print("=== A5 balanced quad, equal constant speeds => zero moment ===")
    m = make(I_R)
    # rotor_dir = [1,-1,1,-1] balanced; equal speeds => sum spin_i Omega_i = 0.
    M = m.rotor_inertia_moment(np.array([2.0, -1.0, 0.5]), np.full(4, 2000.), np.zeros(4))
    ok = np.allclose(M, 0.0, atol=1e-18)
    print(f"  [{'OK ' if ok else 'FAIL'}] |M|={np.max(np.abs(M)):.2e} at nonzero body rate")
    return ok


def test_sign_contrast_crazyflow():
    print("=== A5 sign contrast vs Crazyflow (F-3) ===")
    m = make(I_R)
    spin = -m.rotor_dir
    speeds = np.array([2200., 1800., 2100., 1700.])
    omega = np.array([1.0, 1.0, 0.0])  # p=q=1 to expose the x/y sign difference
    M = m.rotor_inertia_moment(omega, speeds, np.zeros(4))
    h_z = I_R * np.sum(spin * speeds)
    # Ours (correct): (-h_z*q, +h_z*p, 0). Crazyflow: (-h_z*q, -h_z*p, ...) -- same sign both axes.
    ours_ok = np.isclose(M[0], -h_z * omega[1]) and np.isclose(M[1], h_z * omega[0])
    crazyflow_would = np.array([-h_z * omega[1], -h_z * omega[0]])
    differs = not np.isclose(M[1], crazyflow_would[1])
    ok = ours_ok and differs
    print(f"  [{'OK ' if ok else 'FAIL'}] ours=({M[0]:.3e},{M[1]:.3e}); "
          f"crazyflow y would be {crazyflow_would[1]:.3e} (opposite sign) -> divergence confirmed")
    return ok


if __name__ == "__main__":
    r = [test_regression_zero(), test_gyro_cross_product(), test_reaction(),
         test_balance_invariance(), test_sign_contrast_crazyflow()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
