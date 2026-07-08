"""
Verification for A7/C2: PWM quantization + battery voltage-sag layer.

Checks:
1. Regression: default (no pwm range, scale=1) leaves cmd_motor_throttle and the speed clip
   unchanged.
2. PWM quantization: throttle snaps to the integer PWM grid; a tiny grid shows visible steps.
3. Battery scale: rotor_speed_max_scale caps the achievable rotor speed in step().
4. Battery models: LinearBatterySag and VoltageBatterySag produce correct monotone scale
   trajectories in (0,1].
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_pwm_battery.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.battery import NoBatterySag, LinearBatterySag, VoltageBatterySag


def make(extra=None):
    p = copy.deepcopy(quad_params)
    p['rotor_speed_min'] = 0.0
    if extra:
        p.update(extra)
    return Multirotor(p, control_abstraction='cmd_motor_throttle', aero=False)


def speeds_for(m, u):
    return m.get_cmd_motor_speeds(None, {'cmd_motor_throttle': np.full(m.num_rotors, u)})


def test_regression():
    print("=== A7 regression: no PWM range, scale=1 unchanged ===")
    m = make()  # pwm_min=pwm_max=0 -> no quantization
    # cmd_motor_throttle curve with k=1 default is linear.
    u = 0.6137
    got = speeds_for(m, u)[0]
    expect = (m.rotor_speed_max - m.rotor_speed_min) * u + m.rotor_speed_min
    ok = abs(got - expect) < 1e-9 and m.rotor_speed_max_scale == 1.0
    print(f"  [{'OK ' if ok else 'FAIL'}] unquantized throttle exact (err={abs(got-expect):.2e})")
    return ok


def test_pwm_quantization():
    print("=== A7 PWM quantization snaps to grid ===")
    # Use a coarse 10-level grid to see steps clearly.
    m = make({'pwm_min': 0.0, 'pwm_max': 10.0})
    # Two throttles within the same PWM cell should map to the same speed.
    s1 = speeds_for(m, 0.31)[0]   # rounds to 3/10
    s2 = speeds_for(m, 0.34)[0]   # rounds to 3/10
    s3 = speeds_for(m, 0.36)[0]   # rounds to 4/10
    same_cell = abs(s1 - s2) < 1e-9
    diff_cell = abs(s3 - s1) > 1e-6
    # Quantized value equals the grid throttle 0.3.
    expect_03 = (m.rotor_speed_max - m.rotor_speed_min) * 0.3 + m.rotor_speed_min
    on_grid = abs(s1 - expect_03) < 1e-9
    ok = same_cell and diff_cell and on_grid
    print(f"  [{'OK ' if ok else 'FAIL'}] 0.31&0.34->same cell, 0.36->next; snapped to 0.3 (err={abs(s1-expect_03):.2e})")
    return ok


def test_battery_scale_clip():
    print("=== C2 battery scale caps achievable rotor speed in step() ===")
    m = make()
    wmax = m.rotor_speed_max
    state = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0., 0., 0., 1.]),
             'w': np.zeros(3), 'wind': np.zeros(3), 'rotor_speeds': np.full(m.num_rotors, wmax)}
    control = {'cmd_motor_throttle': np.ones(m.num_rotors)}  # full throttle -> commands wmax
    # Sag to 70% -> speeds should be capped at 0.7*wmax.
    m.rotor_speed_max_scale = 0.7
    s = m.step(state, control, 0.05)
    cap = 0.7 * wmax
    ok = np.all(s['rotor_speeds'] <= cap + 1e-6) and abs(np.max(s['rotor_speeds']) - cap) < 1.0
    print(f"  [{'OK ' if ok else 'FAIL'}] max speed {np.max(s['rotor_speeds']):.1f} capped at 0.7*wmax={cap:.1f}")
    return ok


def test_battery_models():
    print("=== C2 battery models produce correct scale trajectories ===")
    ok = True

    nb = NoBatterySag()
    ok_nb = all(nb.update(t) == 1.0 for t in (0, 10, 100))
    ok = ok and ok_nb
    print(f"  [{'OK ' if ok_nb else 'FAIL'}] NoBatterySag == 1")

    lin = LinearBatterySag(drop_frac=0.30, duration=100.0)
    ok_lin = (abs(lin.update(0) - 1.0) < 1e-12 and abs(lin.update(50) - 0.85) < 1e-12
              and abs(lin.update(100) - 0.70) < 1e-12 and abs(lin.update(200) - 0.70) < 1e-12)
    ok = ok and ok_lin
    print(f"  [{'OK ' if ok_lin else 'FAIL'}] LinearBatterySag: 1.0 -> 0.85 @50s -> 0.70 @100s (clamped)")

    volt = VoltageBatterySag(vmotor2rpm=(2968.18, 6647.95), V0=4.2, discharge_rate=0.005)
    s0 = volt.update(0.0)
    s_late = volt.update(100.0)
    ok_v = abs(s0 - 1.0) < 1e-9 and 0.0 < s_late < 1.0
    ok = ok and ok_v
    print(f"  [{'OK ' if ok_v else 'FAIL'}] VoltageBatterySag: 1.0 -> {s_late:.4f} @100s (V=3.7)")
    return ok


if __name__ == "__main__":
    r = [test_regression(), test_pwm_quantization(), test_battery_scale_clip(), test_battery_models()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
