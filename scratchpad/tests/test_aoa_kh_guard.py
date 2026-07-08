"""
Verification for the k_h vs k_angle/k_hor conflict guard (B1 double-count).

k_h (translational lift) and k_hor (advance-ratio thrust correction) BOTH raise thrust with
in-plane airspeed. If both are active the horizontal-airspeed effect is counted twice.

This test:
1. DEMONSTRATES the double count: with both terms active (forced on post-construction, bypassing
   the guard), the horizontal-airspeed thrust increment equals the SUM of each model's individual
   increment -- i.e. it is counted twice, and both pieces are individually nonzero.
2. CONFIRMS THE FIX: constructing a vehicle whose params set both k_h and k_angle/k_hor now raises
   ValueError, so the double-count configuration cannot be built.
3. Single-model / neither configs still construct fine.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_aoa_kh_guard.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

K_H = 1.0e-3     # translational lift coeff (deliberately large to make the effect visible)
K_HOR = 7.245    # advance-ratio coeff (SkyDreamer)
R_PROP = 0.0635


def base_params(**over):
    p = copy.deepcopy(quad_params)
    p['k_h'] = 0.0
    p['k_angle'] = 0.0
    p['k_hor'] = 0.0
    p['r_prop'] = R_PROP
    p.update(over)
    return p


def thrust_z(m, v_hor):
    """Body-z force at horizontal airspeed v_hor (m/s along body x), zero body rate."""
    speeds = np.full(m.num_rotors, 1800.0)
    F, _ = m.compute_body_wrench(np.zeros(3), speeds, np.array([v_hor, 0.0, 0.0]))
    return F[2]


def test_double_count_demonstrated():
    print("=== double-count DEMONSTRATED (both terms active, guard bypassed) ===")
    v = 8.0  # m/s horizontal airspeed

    m_none = Multirotor(base_params(), control_abstraction='cmd_motor_speeds', aero=True)
    m_kh = Multirotor(base_params(k_h=K_H), control_abstraction='cmd_motor_speeds', aero=True)
    m_hor = Multirotor(base_params(k_hor=K_HOR), control_abstraction='cmd_motor_speeds', aero=True)

    T0 = thrust_z(m_none, v)
    dT_kh = thrust_z(m_kh, v) - T0            # increment from translational lift alone
    dT_hor = thrust_z(m_hor, v) - T0          # increment from advance-ratio correction alone

    # Force BOTH on by setting attributes after construction (bypasses the __init__ guard).
    m_both = Multirotor(base_params(), control_abstraction='cmd_motor_speeds', aero=True)
    m_both.k_h = K_H
    m_both.k_hor = K_HOR
    dT_both = thrust_z(m_both, v) - T0

    both_nonzero = abs(dT_kh) > 1e-6 and abs(dT_hor) > 1e-6
    stacks = abs(dT_both - (dT_kh + dT_hor)) < 1e-9   # the two effects add -> counted twice
    ok = both_nonzero and stacks
    print(f"  dT_kh={dT_kh:.6e}, dT_hor={dT_hor:.6e}, dT_both={dT_both:.6e}, sum={dT_kh+dT_hor:.6e}")
    print(f"  [{'OK ' if ok else 'FAIL'}] both nonzero and dT_both == dT_kh + dT_hor "
          f"(horizontal airspeed counted twice)")
    return ok


def test_guard_raises():
    print("=== FIX: constructing with both k_h and k_angle/k_hor raises ===")
    ok = True
    for over in (dict(k_h=K_H, k_hor=K_HOR),
                 dict(k_h=K_H, k_angle=3.145),
                 dict(k_h=K_H, k_angle=3.145, k_hor=K_HOR)):
        try:
            Multirotor(base_params(**over), control_abstraction='cmd_motor_speeds', aero=True)
            good = False
        except ValueError:
            good = True
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] raises for {list(over.keys())}")
    return ok


def test_single_model_ok():
    print("=== single-model / neither configs still construct ===")
    ok = True
    for label, over in (("neither", dict()),
                        ("k_h only", dict(k_h=K_H)),
                        ("k_angle+k_hor only", dict(k_angle=3.145, k_hor=K_HOR))):
        try:
            Multirotor(base_params(**over), control_abstraction='cmd_motor_speeds', aero=True)
            good = True
        except ValueError:
            good = False
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] {label} constructs")
    return ok


if __name__ == "__main__":
    r = [test_double_count_demonstrated(), test_guard_raises(), test_single_model_ok()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
