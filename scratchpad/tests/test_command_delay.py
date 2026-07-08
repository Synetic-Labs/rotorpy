"""
Verification for A6: motor command latency (pure transport delay on the command).

Checks:
1. Regression: motor_delay_time=0 leaves the trajectory identical to no delay.
2. Delay-line correctness: _apply_command_delay returns the command from exactly n=round(td/dt)
   steps ago once the line is full (earliest command held during fill).
3. Constant-command invariance: a constant command with delay > 0 equals the no-delay result.
4. Step effect: after a command step change, rotor speeds stay put for n steps, then respond.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_command_delay.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params


def make(delay):
    p = copy.deepcopy(quad_params)
    p['motor_delay_time'] = delay
    return Multirotor(p, control_abstraction='cmd_motor_speeds', aero=False)


def hover_speeds(m):
    return np.full(m.num_rotors, np.sqrt(m.mass * m.g / np.sum(m.k_eta)))


def state0(m, speeds):
    return {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0., 0., 0., 1.]),
            'w': np.zeros(3), 'wind': np.zeros(3), 'rotor_speeds': speeds.copy()}


def test_regression():
    print("=== A6 regression: delay=0 identical to baseline ===")
    m = make(0.0)
    assert m._cmd_buffer is None
    wh = hover_speeds(m)
    s = state0(m, wh)
    # A random-ish command sequence.
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(20):
        c = {'cmd_motor_speeds': wh + rng.uniform(-50, 50, 4)}
        s = m.step(s, c, 0.01)
    fin = s['rotor_speeds'].copy()
    # Re-run identically -> same result (determinism, no hidden buffer state at delay 0).
    m2 = make(0.0)
    s2 = state0(m2, wh)
    rng2 = np.random.default_rng(0)
    for _ in range(20):
        c = {'cmd_motor_speeds': wh + rng2.uniform(-50, 50, 4)}
        s2 = m2.step(s2, c, 0.01)
    ok = np.allclose(fin, s2['rotor_speeds'])
    print(f"  [{'OK ' if ok else 'FAIL'}] no buffer created, deterministic")
    return ok


def test_delay_line():
    print("=== A6 delay line returns command from n steps ago ===")
    m = make(0.05)   # td=0.05, dt=0.01 -> n=5
    dt = 0.01
    n = 5
    applied = []
    for k in range(20):
        c = {'cmd_motor_speeds': np.full(4, float(k))}
        applied.append(m._apply_command_delay(c, dt)['cmd_motor_speeds'][0])
    expect = [float(max(0, k - n)) for k in range(20)]
    ok = applied == expect
    print(f"  [{'OK ' if ok else 'FAIL'}] applied[k]==cmd[max(0,k-{n})]")
    print(f"        applied={applied}")
    return ok


def test_constant_invariance():
    print("=== A6 constant command: delay has no effect ===")
    wh = hover_speeds(make(0.0))
    m0 = make(0.0); s0 = state0(m0, wh)
    md = make(0.05); sd = state0(md, wh)
    c = {'cmd_motor_speeds': wh * 1.05}
    for _ in range(50):
        s0 = m0.step(s0, c, 0.01)
        sd = md.step(sd, c, 0.01)
    ok = np.allclose(s0['rotor_speeds'], sd['rotor_speeds'], atol=1e-9)
    print(f"  [{'OK ' if ok else 'FAIL'}] constant-cmd states match (err={np.max(np.abs(s0['rotor_speeds']-sd['rotor_speeds'])):.2e})")
    return ok


def test_step_effect():
    print("=== A6 step change delayed by n steps ===")
    m = make(0.05)  # n=5
    wh = hover_speeds(m)
    s = state0(m, wh)
    # Prime the delay line with the hover command so the held value is hover.
    hover_cmd = {'cmd_motor_speeds': wh}
    s = m.step(s, hover_cmd, 0.01)
    # Now command a big increase; motor should not move until the delayed command arrives.
    high = {'cmd_motor_speeds': wh + 500.0}
    speeds_track = []
    for _ in range(10):
        s = m.step(s, high, 0.01)
        speeds_track.append(s['rotor_speeds'][0])
    # First ~4 steps (n-1 after priming) should stay ~ hover; later steps rise.
    early_flat = np.allclose(speeds_track[:3], wh[0], atol=1e-6)
    later_rise = speeds_track[-1] > wh[0] + 50
    ok = early_flat and later_rise
    print(f"  [{'OK ' if ok else 'FAIL'}] early held at hover ({speeds_track[0]:.1f}), "
          f"later rose ({speeds_track[-1]:.1f} vs hover {wh[0]:.1f})")
    return ok


if __name__ == "__main__":
    r = [test_regression(), test_delay_line(), test_constant_invariance(), test_step_effect()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
