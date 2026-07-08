"""
Verification for A4: asymmetric rotor spin-up/spin-down dynamics.

    Omega_dot = ka1*(Oc-O) + ka2*(Oc^2-O^2)   if Oc > O   (spin-up)
    Omega_dot = kd1*(Oc-O) + kd2*(Oc^2-O^2)   otherwise    (spin-down)

Checks:
1. Regression: no rotor_dyn_coef -> Omega_dot = (Oc-O)/tau_m exactly.
2. Reduction: rotor_dyn_coef = [1/tau, 0, 1/tau, 0] reproduces the first-order lag.
3. Asymmetry: spin-up and spin-down give different accel for the same |delta|.
4. Crazyflow formula match (definitional check of the implementation).
5. Integration: with cf21B_500 coeffs (spin-up gain > spin-down), a step-up settles faster
   than a step-down of equal magnitude.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_motor_dynamics.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params


def make(coef=None):
    p = copy.deepcopy(quad_params)
    if coef is not None:
        p['rotor_dyn_coef'] = coef
    return Multirotor(p, control_abstraction='cmd_motor_speeds', aero=False)


def test_regression():
    print("=== A4 regression: default is first-order 1/tau_m ===")
    m = make()
    cmd = np.full(4, 2000.); spd = np.full(4, 1500.)
    a = m._rotor_accel(cmd, spd)
    ref = (cmd - spd) / m.tau_m
    ok = np.max(np.abs(a - ref)) < 1e-12
    print(f"  [{'OK ' if ok else 'FAIL'}] err={np.max(np.abs(a-ref)):.2e}")
    return ok


def test_reduction():
    print("=== A4 [1/tau,0,1/tau,0] reduces to first-order ===")
    m0 = make()
    tau = m0.tau_m
    m = make([1.0/tau, 0.0, 1.0/tau, 0.0])
    ok = True
    for cmd_v, spd_v in [(2000., 1500.), (1500., 2000.), (1800., 1800.)]:
        cmd = np.full(4, cmd_v); spd = np.full(4, spd_v)
        a = m._rotor_accel(cmd, spd)
        ref = (cmd - spd) / tau
        good = np.max(np.abs(a - ref)) < 1e-12
        ok = ok and good
    print(f"  [{'OK ' if ok else 'FAIL'}] matches first-order in up/down/equal cases")
    return ok


def test_asymmetry():
    print("=== A4 spin-up != spin-down for equal |delta| ===")
    # ka1=14, kd1=6 (linear only): up-accel and down-accel differ by the gain ratio.
    m = make([14.0, 0.0, 6.0, 0.0])
    a_up = m._rotor_accel(np.full(4, 2000.), np.full(4, 1500.))   # +500 delta
    a_dn = m._rotor_accel(np.full(4, 1000.), np.full(4, 1500.))   # -500 delta
    ok = np.allclose(a_up, 14.0*500) and np.allclose(a_dn, 6.0*(-500))
    print(f"  [{'OK ' if ok else 'FAIL'}] up={a_up[0]:.1f} (=14*500), down={a_dn[0]:.1f} (=6*-500)")
    return ok


def test_crazyflow_formula():
    print("=== A4 matches Crazyflow where() formula ===")
    coef = [13.996001897562685, 0.00011093207920685363, 5.933168530682111, 0.00031951312393561264]
    m = make(coef)
    ka1, ka2, kd1, kd2 = coef
    ok = True
    for cmd_v, spd_v in [(2200., 1800.), (1600., 2100.), (2000., 2000.)]:
        cmd = np.full(4, cmd_v); spd = np.full(4, spd_v)
        got = m._rotor_accel(cmd, spd)
        d = cmd - spd; dsq = cmd**2 - spd**2
        ref = np.where(cmd > spd, ka1*d + ka2*dsq, kd1*d + kd2*dsq)
        good = np.max(np.abs(got - ref)) < 1e-9
        ok = ok and good
    print(f"  [{'OK ' if ok else 'FAIL'}] formula reproduced")
    return ok


def test_integration_asymmetry():
    print("=== A4 integration: step-up settles faster than step-down (cf21B_500) ===")
    m = make([13.996001897562685, 0.00011093207920685363, 5.933168530682111, 0.00031951312393561264])
    lo, hi = 1500.0, 2500.0

    def settle_time(start, target):
        state = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0., 0., 0., 1.]),
                 'w': np.zeros(3), 'wind': np.zeros(3), 'rotor_speeds': np.full(m.num_rotors, start)}
        ctl = {'cmd_motor_speeds': np.full(m.num_rotors, target)}
        for step in range(2000):
            state = m.step(state, ctl, 0.001)
            if abs(state['rotor_speeds'][0] - target) < 0.1 * abs(target - start):  # within 10%
                return step * 0.001
        return None

    t_up = settle_time(lo, hi)
    t_dn = settle_time(hi, lo)
    ok = t_up is not None and t_dn is not None and t_up < t_dn
    print(f"  [{'OK ' if ok else 'FAIL'}] 90% settle: up={t_up}s, down={t_dn}s (up faster)")
    return ok


if __name__ == "__main__":
    r = [test_regression(), test_reduction(), test_asymmetry(),
         test_crazyflow_formula(), test_integration_asymmetry()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
