"""
Verification for A3: nonlinear normalized-throttle -> steady-state-speed curve
('cmd_motor_throttle' abstraction).

    w_c = (w_max - w_min) * sqrt(k*u^2 + (1-k)*u) + w_min,  u in [0,1], k in [0,1]

Checks:
1. Endpoints: u=0 -> w_min, u=1 -> w_max (any k).
2. Monotonic increasing on [0,1].
3. k=1 is the linear speed map; k=0 is the sqrt curve.
4. Reproduces SkyDreamer reference values (w_min=341.75, w_max=3100, k=0.5) via an
   independent reimplementation of skydreamer.py's formula.
5. Command clipping: u outside [0,1] is clamped.
6. Integration: constant throttle drives the motor to its commanded steady-state speed.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_throttle_curve.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params


def make(k=1.0, wmin=None, wmax=None):
    p = copy.deepcopy(quad_params)
    p['motor_curve_k'] = k
    if wmin is not None:
        p['rotor_speed_min'] = wmin
    if wmax is not None:
        p['rotor_speed_max'] = wmax
    return Multirotor(p, control_abstraction='cmd_motor_throttle', aero=False)


def curve_ref(u, k, wmin, wmax):
    """Independent reimplementation of the SkyDreamer motor curve."""
    u = np.clip(u, 0.0, 1.0)
    return (wmax - wmin) * np.sqrt(k * u**2 + (1 - k) * u) + wmin


def speeds_for(m, u):
    return m.get_cmd_motor_speeds(None, {'cmd_motor_throttle': np.full(m.num_rotors, u)})


def test_endpoints():
    print("=== A3 endpoints u=0->w_min, u=1->w_max ===")
    ok = True
    for k in (0.0, 0.5, 1.0):
        m = make(k)
        s0 = speeds_for(m, 0.0)
        s1 = speeds_for(m, 1.0)
        e0 = np.max(np.abs(s0 - m.rotor_speed_min))
        e1 = np.max(np.abs(s1 - m.rotor_speed_max))
        good = e0 < 1e-9 and e1 < 1e-9
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] k={k}: u0_err={e0:.2e} u1_err={e1:.2e}")
    return ok


def test_monotonic():
    print("=== A3 monotonic on [0,1] ===")
    ok = True
    for k in (0.0, 0.25, 0.5, 1.0):
        m = make(k)
        us = np.linspace(0, 1, 50)
        vals = np.array([speeds_for(m, u)[0] for u in us])
        good = np.all(np.diff(vals) >= -1e-9)
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] k={k}: monotonic")
    return ok


def test_linear_and_sqrt():
    print("=== A3 k=1 linear, k=0 sqrt ===")
    m1 = make(1.0)
    us = np.linspace(0, 1, 11)
    lin_expect = (m1.rotor_speed_max - m1.rotor_speed_min) * us + m1.rotor_speed_min
    lin_got = np.array([speeds_for(m1, u)[0] for u in us])
    ok_lin = np.max(np.abs(lin_got - lin_expect)) < 1e-9

    m0 = make(0.0)
    sqrt_expect = (m0.rotor_speed_max - m0.rotor_speed_min) * np.sqrt(us) + m0.rotor_speed_min
    sqrt_got = np.array([speeds_for(m0, u)[0] for u in us])
    ok_sqrt = np.max(np.abs(sqrt_got - sqrt_expect)) < 1e-9
    print(f"  [{'OK ' if ok_lin else 'FAIL'}] k=1 linear (err={np.max(np.abs(lin_got-lin_expect)):.2e})")
    print(f"  [{'OK ' if ok_sqrt else 'FAIL'}] k=0 sqrt (err={np.max(np.abs(sqrt_got-sqrt_expect)):.2e})")
    return ok_lin and ok_sqrt


def test_skydreamer_values():
    print("=== A3 matches SkyDreamer reference (w_min=341.75, w_max=3100, k=0.5) ===")
    wmin, wmax, k = 341.75, 3100.0, 0.5
    m = make(k, wmin, wmax)
    us = np.array([0.0, 0.1, 0.37, 0.5, 0.8, 1.0])
    got = np.array([speeds_for(m, u)[0] for u in us])
    ref = curve_ref(us, k, wmin, wmax)
    err = np.max(np.abs(got - ref))
    ok = err < 1e-9
    print(f"  [{'OK ' if ok else 'FAIL'}] max err vs reference = {err:.2e}")
    return ok


def test_clip_and_integration():
    print("=== A3 clipping + integration to steady state ===")
    m = make(0.5, 341.75, 3100.0)
    # Clipping: u=-0.5 -> w_min, u=1.5 -> w_max.
    lo = speeds_for(m, -0.5)
    hi = speeds_for(m, 1.5)
    ok_clip = np.allclose(lo, m.rotor_speed_min) and np.allclose(hi, m.rotor_speed_max)
    print(f"  [{'OK ' if ok_clip else 'FAIL'}] out-of-range throttle clamped")

    # Integration: hold throttle 0.6, motor should relax toward w_c(0.6).
    u = 0.6
    wc = curve_ref(u, 0.5, 341.75, 3100.0)
    state = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0., 0., 0., 1.]),
             'w': np.zeros(3), 'wind': np.zeros(3), 'rotor_speeds': np.full(m.num_rotors, m.rotor_speed_min)}
    control = {'cmd_motor_throttle': np.full(m.num_rotors, u)}
    for _ in range(200):  # 2 s at dt=0.01, tau_m ~ 0.07 s
        state = m.step(state, control, 0.01)
    err = np.max(np.abs(state['rotor_speeds'] - wc))
    ok_int = err < 1.0  # within 1 rad/s of the commanded steady state
    print(f"  [{'OK ' if ok_int else 'FAIL'}] motor reached w_c(0.6)={wc:.1f} (err={err:.2e})")
    return ok_clip and ok_int


if __name__ == "__main__":
    r = [test_endpoints(), test_monotonic(), test_linear_and_sqrt(),
         test_skydreamer_values(), test_clip_and_integration()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
