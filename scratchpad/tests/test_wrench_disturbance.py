"""
Verification for C1: two-band external-wrench disturbance
(see scratchpad/rotorpy-physics-additions.md, section C and finding tests).

Checks:
1. Regression-zero: with the default zero wrench, statedot equals the analytical
   hover derivative (external hooks add nothing).
2. Injection physics: a world-frame force F shifts v_dot by exactly F/m; a body-frame
   torque M shifts w_dot by exactly I^-1 M at zero body rate.
3. WrenchDisturbance generator: force = m*eps_a and torque = I*(eps_M_lf+eps_M_hf),
   values bounded by the configured ranges, resample-and-hold timing correct, and
   seeding is reproducible.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_wrench_disturbance.py
"""
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.disturbances.default_disturbances import NoDisturbance, WrenchDisturbance


def hover_state(m):
    kg = quad_params['k_eta']
    w_hover = np.sqrt(m.mass * m.g / (m.num_rotors * kg))
    return {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0., 0., 0., 1.]),
            'w': np.zeros(3), 'wind': np.zeros(3),
            'rotor_speeds': np.full(m.num_rotors, w_hover)}


def statedot(m, state):
    control = {'cmd_motor_speeds': state['rotor_speeds']}
    m.control_abstraction = 'cmd_motor_speeds'
    return m.statedot(state, control, 0.01)


def test_regression_zero():
    print("=== C1 regression-zero (default wrench => analytical hover derivative) ===")
    m = Multirotor(quad_params, control_abstraction='cmd_motor_speeds', aero=False)
    s = hover_state(m)
    sd = statedot(m, s)
    # At hover with balanced rotors: v_dot ~ 0 (thrust cancels gravity), w_dot ~ 0.
    da = np.max(np.abs(sd['vdot']))
    dw = np.max(np.abs(sd['wdot']))
    ok = da < 1e-6 and dw < 1e-9
    print(f"  [{'OK ' if ok else 'FAIL'}] |vdot|={da:.2e} |wdot|={dw:.2e}")
    return ok


def test_injection():
    print("=== C1 injection physics (force -> F/m, torque -> I^-1 M) ===")
    m = Multirotor(quad_params, control_abstraction='cmd_motor_speeds', aero=False)
    s = hover_state(m)
    base = statedot(m, s)
    ok = True

    F = np.array([0.05, -0.02, 0.1])
    m.external_force = F
    m.external_torque = np.zeros(3)
    sd = statedot(m, s)
    dv = sd['vdot'] - base['vdot']
    err_v = np.max(np.abs(dv - F / m.mass))
    okv = err_v < 1e-9
    ok = ok and okv
    print(f"  [{'OK ' if okv else 'FAIL'}] force injection err={err_v:.2e}")

    M = np.array([1e-4, -2e-4, 5e-5])
    m.external_force = np.zeros(3)
    m.external_torque = M
    sd = statedot(m, s)
    dw = sd['wdot'] - base['wdot']
    err_w = np.max(np.abs(dw - m.inv_inertia @ M))
    okw = err_w < 1e-9
    ok = ok and okw
    print(f"  [{'OK ' if okw else 'FAIL'}] torque injection err={err_w:.2e}")
    return ok


def test_generator():
    print("=== C1 WrenchDisturbance generator ===")
    m = Multirotor(quad_params, control_abstraction='cmd_motor_speeds', aero=False)
    ok = True

    # force = m*eps_a, torque = I*(eps_lf+eps_hf); check the mapping is exact by inverting.
    d = WrenchDisturbance(m.mass, m.inertia,
                          accel_range=3.0, angaccel_lf_range=3.0, angaccel_hf_range=125.0,
                          seed=42)
    w = d.update(0.0, None)
    eps_a = w['force'] / m.mass
    eps_M = m.inv_inertia @ w['torque']
    okmap = np.all(np.abs(eps_a) <= 3.0 + 1e-9) and np.all(np.abs(eps_M) <= (3.0 + 125.0) + 1e-9)
    ok = ok and okmap
    print(f"  [{'OK ' if okmap else 'FAIL'}] eps_a in +/-3, eps_M in +/-128: "
          f"eps_a_max={np.max(np.abs(eps_a)):.3f} eps_M_max={np.max(np.abs(eps_M)):.3f}")

    # resample-and-hold: at 100 Hz, the 1 Hz linear band holds for ~100 steps.
    d2 = WrenchDisturbance(m.mass, m.inertia, seed=7)
    dt = 0.01
    f0 = d2.update(0.0, None)['force'].copy()
    held = all(np.allclose(d2.update(k * dt, None)['force'], f0) for k in range(1, 100))
    changed = not np.allclose(d2.update(1.0, None)['force'], f0)  # at t=1.0s the 1Hz band resamples
    okhold = held and changed
    ok = ok and okhold
    print(f"  [{'OK ' if okhold else 'FAIL'}] linear band held 1s then resampled "
          f"(held={held}, changed={changed})")

    # high-frequency band resamples every step at 100 Hz (period 1/90 s < 0.01... actually >0.01).
    # 90 Hz period = 0.0111 s, so it resamples roughly every ~1-2 steps. Just check it varies.
    d3 = WrenchDisturbance(m.mass, m.inertia, seed=1)
    torques = [d3.update(k * dt, None)['torque'].copy() for k in range(20)]
    hf_varies = any(not np.allclose(torques[i], torques[i + 1]) for i in range(19))
    ok = ok and hf_varies
    print(f"  [{'OK ' if hf_varies else 'FAIL'}] high-freq torque band varies over steps")

    # determinism: same seed -> same sequence.
    da = WrenchDisturbance(m.mass, m.inertia, seed=123)
    db = WrenchDisturbance(m.mass, m.inertia, seed=123)
    seq_a = [da.update(k * dt, None)['torque'].copy() for k in range(50)]
    seq_b = [db.update(k * dt, None)['torque'].copy() for k in range(50)]
    okdet = all(np.allclose(a, b) for a, b in zip(seq_a, seq_b))
    ok = ok and okdet
    print(f"  [{'OK ' if okdet else 'FAIL'}] seeded determinism")

    # NoDisturbance returns zero.
    nz = NoDisturbance().update(3.3, None)
    oknz = np.allclose(nz['force'], 0) and np.allclose(nz['torque'], 0)
    ok = ok and oknz
    print(f"  [{'OK ' if oknz else 'FAIL'}] NoDisturbance is zero")
    return ok


if __name__ == "__main__":
    r = [test_regression_zero(), test_injection(), test_generator()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
