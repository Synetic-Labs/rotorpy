"""
Verification for A2: polynomial thrust/torque curves
  thrust = c0 + c1*Omega + k_eta*Omega^2 ,  torque = d0 + d1*Omega + k_m*Omega^2

Checks:
1. Regression: with no poly terms, forward thrust/yaw == k_eta*Omega^2 / dir*k_m*Omega^2, and the
   inversion equals the original sign*sqrt(f/k_eta). Allocation round-trip still exact.
2. Polynomial forward matches a hand computation.
3. Exact inversion: _thrust_to_speed(thrust_curve(Omega)) == Omega on the physical branch.
4. Crazyflow parity: converting rpm2thrust (RPM units) to rad/s reproduces Crazyflow's thrust.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_poly_thrust.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

RPM2RAD = 2 * np.pi / 60.0


def make(extra=None, abstraction='cmd_motor_speeds'):
    p = copy.deepcopy(quad_params)
    if extra:
        p.update(extra)
    return Multirotor(p, control_abstraction=abstraction, aero=False)


def test_regression():
    print("=== A2 regression: no poly terms reproduce quadratic model + inversion ===")
    m = make()
    speeds = np.array([1400., 1600., 1500., 1550.])
    F, M = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    okF = abs(F[2] - np.sum(m.k_eta * speeds**2)) < 1e-9
    okM = abs(M[2] - np.sum(m.rotor_dir * m.k_m * speeds**2)) < 1e-12
    # Inversion equals original sign*sqrt(f/k_eta).
    forces = np.array([0.01, 0.02, -0.005, 0.015])
    inv = m._thrust_to_speed(forces)
    ref = np.sign(forces) * np.sqrt(np.abs(forces / m.k_eta))
    okI = np.max(np.abs(inv - ref)) < 1e-12
    print(f"  [{'OK ' if okF else 'FAIL'}] thrust quadratic (err={abs(F[2]-np.sum(m.k_eta*speeds**2)):.2e})")
    print(f"  [{'OK ' if okM else 'FAIL'}] yaw quadratic")
    print(f"  [{'OK ' if okI else 'FAIL'}] inversion == sign*sqrt(f/k_eta) (err={np.max(np.abs(inv-ref)):.2e})")
    return okF and okM and okI


def test_poly_forward():
    print("=== A2 polynomial forward matches hand calc ===")
    c0, c1 = 1e-3, -5e-6
    m = make({'thrust_c0': c0, 'thrust_c1': c1})
    speeds = np.array([1400., 1600., 1500., 1550.])
    F, _ = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    expect = np.sum(c0 + c1 * speeds + m.k_eta * speeds**2)
    ok = abs(F[2] - expect) < 1e-9
    print(f"  [{'OK ' if ok else 'FAIL'}] Fz={F[2]:.5f} expect={expect:.5f} (err={abs(F[2]-expect):.2e})")
    return ok


def test_inversion_roundtrip():
    print("=== A2 exact inversion on the physical branch ===")
    # Crazyflow cf2x_L250 rpm2thrust -> rad/s.
    b_rpm, c_rpm = -5.382196214637237e-7, 2.4582929831265485e-10
    c1 = b_rpm / RPM2RAD          # convert per-rpm to per-(rad/s):  f uses rpm = Omega/RPM2RAD
    c2 = c_rpm / RPM2RAD**2
    m = make({'thrust_c1': c1, 'k_eta': c2})
    assert m._thrust_poly_active
    # Physical branch is above the vertex Omega* = -c1/(2 c2). The helper is per-rotor
    # (length num_rotors), so evaluate one test speed across all rotors at a time.
    vertex = -c1 / (2 * c2)
    err = 0.0
    for omega in np.linspace(vertex + 50, 2500, 40):
        speeds4 = np.full(m.num_rotors, omega)
        forces4 = c1 * speeds4 + c2 * speeds4**2
        recovered = m._thrust_to_speed(forces4)
        err = max(err, np.max(np.abs(recovered - speeds4)))
    ok = err < 1e-6
    print(f"  [{'OK ' if ok else 'FAIL'}] w -> f -> w round-trip err={err:.2e} (vertex={vertex:.1f} rad/s)")
    return ok


def test_crazyflow_parity():
    print("=== A2 Crazyflow unit-conversion parity ===")
    a, b, c = 0.0, -5.382196214637237e-7, 2.4582929831265485e-10  # rpm2thrust, per RPM
    c0, c1, c2 = a, b / RPM2RAD, c / RPM2RAD**2                     # per rad/s
    m = make({'thrust_c0': c0, 'thrust_c1': c1, 'k_eta': c2})
    ok = True
    for rpm in (4000., 6000., 8000., 10000.):
        omega = rpm * RPM2RAD
        F, _ = m.compute_body_wrench(np.zeros(3),
                                     np.array([omega, 0., 0., 0.]), np.zeros(3))
        # Only rotor 0 spinning: its thrust is F[2].
        f_crazyflow = a + b * rpm + c * rpm**2
        e = abs(F[2] - f_crazyflow)
        good = e < 1e-9
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] rpm={rpm:.0f}: rotorpy={F[2]:.6e} crazyflow={f_crazyflow:.6e} (err={e:.1e})")
    return ok


def test_allocation_roundtrip_default():
    print("=== A2 allocation round-trip unaffected at default ===")
    m = make(abstraction='cmd_ctbm')
    cmd_thrust = m.mass * m.g * 1.1
    cmd_moment = np.array([1e-4, -2e-4, 5e-5])
    state = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0., 0., 0., 1.]),
             'w': np.zeros(3), 'wind': np.zeros(3), 'rotor_speeds': np.zeros(m.num_rotors)}
    speeds = m.get_cmd_motor_speeds(state, {'cmd_thrust': cmd_thrust, 'cmd_moment': cmd_moment})
    F, M = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    ok = abs(F[2] - cmd_thrust) < 1e-6 and np.max(np.abs(M - cmd_moment)) < 1e-9
    print(f"  [{'OK ' if ok else 'FAIL'}] recovered thrust+moment (Terr={abs(F[2]-cmd_thrust):.2e}, Merr={np.max(np.abs(M-cmd_moment)):.2e})")
    return ok


if __name__ == "__main__":
    r = [test_regression(), test_poly_forward(), test_inversion_roundtrip(),
         test_crazyflow_parity(), test_allocation_roundtrip_default()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
