"""
Verification for A1: per-rotor thrust/torque coefficients.

Checks:
1. Regression: scalar k_eta/k_m (broadcast to equal per-rotor arrays) reproduce the analytical
   hover balance exactly (backward compatibility with all existing parameter files).
2. Asymmetry physics: making one rotor's k_eta larger produces the correct extra thrust and the
   correct body moment Sum r_i x T_i (checked against a hand computation).
3. Allocation round-trip: with per-rotor coefficients, commanding a known collective thrust +
   body moment (cmd_ctbm), converting to motor speeds, then recomputing the wrench recovers the
   commanded thrust and moment (the allocation matrix and the wrench must share the per-rotor k).
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_per_rotor_coeffs.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params


def make(params, abstraction='cmd_motor_speeds'):
    return Multirotor(params, control_abstraction=abstraction, aero=False)


def test_regression_scalar():
    print("=== A1 regression: scalar coeffs reproduce hover balance ===")
    m = make(quad_params)
    assert m.k_eta.shape == (m.num_rotors,), "k_eta should be stored per-rotor"
    w_hover = np.sqrt(m.mass * m.g / np.sum(m.k_eta))
    speeds = np.full(m.num_rotors, w_hover)
    F, M = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    okF = abs(F[2] - m.mass * m.g) < 1e-9 and np.max(np.abs(F[:2])) < 1e-12
    okM = np.max(np.abs(M)) < 1e-12
    print(f"  [{'OK ' if okF else 'FAIL'}] net thrust = mg, no lateral force (Fz-mg={F[2]-m.mass*m.g:.2e})")
    print(f"  [{'OK ' if okM else 'FAIL'}] balanced hover has zero moment (|M|={np.max(np.abs(M)):.2e})")
    return okF and okM


def test_asymmetry():
    print("=== A1 asymmetry: per-rotor k_eta gives correct thrust + moment ===")
    params = copy.deepcopy(quad_params)
    base_keta = float(np.asarray(quad_params['k_eta']))
    # Make rotor 0 20% stronger.
    per_rotor = np.full(4, base_keta)
    per_rotor[0] *= 1.2
    params['k_eta'] = per_rotor
    m = make(params)
    speeds = np.full(m.num_rotors, 1500.0)
    F, M = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    # Hand computation.
    T_i = per_rotor * speeds**2            # per-rotor thrust magnitude (z)
    expect_Fz = np.sum(T_i)
    # Moment = sum r_i x (T_i zhat)
    geom = m.rotor_geometry                 # (num_rotors, 3)
    zhat = np.array([0, 0, 1.0])
    expect_M = np.sum([np.cross(geom[i], T_i[i] * zhat) for i in range(m.num_rotors)], axis=0)
    # Add yaw from k_m (scalar here).
    expect_M[2] += np.sum(m.rotor_dir * m.k_m * speeds**2)
    okF = abs(F[2] - expect_Fz) < 1e-6
    okM = np.max(np.abs(M - expect_M)) < 1e-9
    print(f"  [{'OK ' if okF else 'FAIL'}] Fz matches hand calc (err={abs(F[2]-expect_Fz):.2e})")
    print(f"  [{'OK ' if okM else 'FAIL'}] moment matches Sum r_i x T_i (err={np.max(np.abs(M-expect_M)):.2e})")
    print(f"        (asymmetry induced moment Mx,My = {M[0]:.3e}, {M[1]:.3e})")
    return okF and okM


def test_allocation_roundtrip():
    print("=== A1 allocation round-trip with per-rotor coeffs (cmd_ctbm) ===")
    params = copy.deepcopy(quad_params)
    base_keta = float(np.asarray(quad_params['k_eta']))
    base_km = float(np.asarray(quad_params['k_m']))
    params['k_eta'] = base_keta * np.array([1.0, 1.1, 0.9, 1.05])
    params['k_m'] = base_km * np.array([1.0, 0.95, 1.08, 1.0])
    m = make(params, abstraction='cmd_ctbm')

    cmd_thrust = m.mass * m.g * 1.1
    cmd_moment = np.array([2e-4, -1.5e-4, 3e-5])
    state = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0., 0., 0., 1.]),
             'w': np.zeros(3), 'wind': np.zeros(3), 'rotor_speeds': np.zeros(m.num_rotors)}
    control = {'cmd_thrust': cmd_thrust, 'cmd_moment': cmd_moment}
    speeds = m.get_cmd_motor_speeds(state, control)
    F, M = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    okT = abs(F[2] - cmd_thrust) < 1e-6
    okM = np.max(np.abs(M - cmd_moment)) < 1e-9
    print(f"  [{'OK ' if okT else 'FAIL'}] recovered thrust (err={abs(F[2]-cmd_thrust):.2e})")
    print(f"  [{'OK ' if okM else 'FAIL'}] recovered moment (err={np.max(np.abs(M-cmd_moment)):.2e})")
    return okT and okM


if __name__ == "__main__":
    r = [test_regression_scalar(), test_asymmetry(), test_allocation_roundtrip()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
