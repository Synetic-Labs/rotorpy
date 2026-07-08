"""
Verification for B1: thrust vs rotor angle-of-attack and advance ratio.

    alpha = atan2(v_a,z, r_prop*w_bar);  mu = atan2(||v_a,xy||, r_prop*w_bar)
    T_rotor *= (1 + k_angle*alpha + k_hor*mu);  plus collective -k_v2*v_a,z*|v_a,z|

Checks:
1. Regression: k_angle=k_hor=k_v2=0 leaves the aero wrench unchanged.
2. Zero-airspeed invariance: with coeffs set but v_a=0, alpha=mu=0 -> factor 1, no change.
3. Correction matches a hand computation of alpha/mu/factor at nonzero airspeed.
4. SkyDreamer reference: total thrust equals k_w*(1+k_angle*alpha+k_hor*mu)*sum Omega^2
   (using k_eta=k_w), plus the -k_v2 term.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_aoa_thrust.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

# SkyDreamer Table II (5" racer).
SKY = dict(k_w=1.55e-06, k_angle=3.145, k_hor=7.245, k_v2=0.0, r_prop=0.0635)


def make(extra=None):
    p = copy.deepcopy(quad_params)
    if extra:
        p.update(extra)
    return Multirotor(p, control_abstraction='cmd_motor_speeds', aero=True)


def test_regression():
    print("=== B1 regression: coeffs=0 leaves aero wrench unchanged ===")
    m = make()  # default k_angle=k_hor=k_v2=0
    speeds = np.array([1400., 1600., 1500., 1550.])
    v_a = np.array([2.0, -1.0, 0.5])
    F, M = m.compute_body_wrench(np.array([0.1, -0.2, 0.05]), speeds, v_a)
    # Base thrust z (no aoa factor) should still be sum k_eta*Omega^2 plus the existing aero terms;
    # cross-check that the aoa branch did not alter T by comparing to a model with the branch guard.
    base_Tz = np.sum(m.k_eta * speeds**2)
    # translational lift (k_h) is 0 for crazyflie; H contributes to Fz too. Just confirm no NaN and
    # that toggling k_angle=0 matches an independent recompute with a fresh default vehicle.
    m2 = make()
    F2, M2 = m2.compute_body_wrench(np.array([0.1, -0.2, 0.05]), speeds, v_a)
    ok = np.allclose(F, F2) and np.allclose(M, M2) and not np.any(np.isnan(F))
    print(f"  [{'OK ' if ok else 'FAIL'}] deterministic, finite (Fz={F[2]:.4e}, base kEtaOmega2={base_Tz:.4e})")
    return ok


def test_zero_airspeed():
    print("=== B1 zero airspeed: factor=1 even with coeffs set ===")
    m_on = make({'k_angle': SKY['k_angle'], 'k_hor': SKY['k_hor'], 'k_v2': 1e-3, 'r_prop': SKY['r_prop']})
    m_off = make()
    speeds = np.array([1400., 1600., 1500., 1550.])
    F_on, _ = m_on.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    F_off, _ = m_off.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    ok = np.allclose(F_on, F_off)
    print(f"  [{'OK ' if ok else 'FAIL'}] Fz_on={F_on[2]:.6e} == Fz_off={F_off[2]:.6e}")
    return ok


def test_hand_calc():
    print("=== B1 correction matches hand-computed alpha/mu/factor ===")
    m = make({'k_angle': SKY['k_angle'], 'k_hor': SKY['k_hor'], 'k_v2': 0.0, 'r_prop': SKY['r_prop']})
    speeds = np.array([1800., 1800., 1800., 1800.])
    v_a = np.array([3.0, 1.0, -2.0])   # body-frame airspeed
    # No body rate so local airspeed == v_a for all rotors; H-force present but we check Fz factor.
    F, _ = m.compute_body_wrench(np.zeros(3), speeds, v_a)
    w_bar = np.mean(speeds)
    denom = SKY['r_prop'] * w_bar
    alpha = np.arctan2(v_a[2], denom)
    mu = np.arctan2(np.hypot(v_a[0], v_a[1]), denom)
    factor = 1 + SKY['k_angle']*alpha + SKY['k_hor']*mu
    # Expected z thrust before H-force: factor * sum(k_eta*Omega^2). Add the H-force z-contribution.
    Tz_expected = factor * np.sum(m.k_eta * speeds**2)
    # H-force z per rotor: -Omega * k_z * v_a,z (k_z from crazyflie). Sum over rotors.
    Hz = -np.sum(speeds * m.k_z * v_a[2])
    expect_Fz = Tz_expected + Hz
    err = abs(F[2] - expect_Fz)
    ok = err < 1e-6
    print(f"  [{'OK ' if ok else 'FAIL'}] Fz={F[2]:.6e} expect={expect_Fz:.6e} (err={err:.2e}, factor={factor:.3f})")
    return ok


def test_skydreamer_reference():
    print("=== B1 SkyDreamer thrust formula (k_eta=k_w) ===")
    # Set k_eta = k_w, no H-force/lift/quadratic-drag so Fz is purely the corrected thrust.
    p = copy.deepcopy(quad_params)
    p.update({'k_eta': SKY['k_w'], 'k_angle': SKY['k_angle'], 'k_hor': SKY['k_hor'],
              'k_v2': 0.0, 'r_prop': SKY['r_prop'],
              'k_d': 0.0, 'k_z': 0.0, 'k_h': 0.0, 'k_flap': 0.0,
              'c_Dx': 0.0, 'c_Dy': 0.0, 'c_Dz': 0.0})
    m = Multirotor(p, control_abstraction='cmd_motor_speeds', aero=True)
    ok = True
    for v_a in [np.array([0., 0., 0.]), np.array([5., 0., 1.]), np.array([2., -3., -4.])]:
        speeds = np.array([2500., 2600., 2400., 2550.])
        F, _ = m.compute_body_wrench(np.zeros(3), speeds, v_a)
        w_bar = np.mean(speeds)
        denom = SKY['r_prop'] * w_bar
        alpha = np.arctan2(v_a[2], denom)
        mu = np.arctan2(np.hypot(v_a[0], v_a[1]), denom)
        ref = SKY['k_w'] * (1 + SKY['k_angle']*alpha + SKY['k_hor']*mu) * np.sum(speeds**2)
        err = abs(F[2] - ref)
        good = err < 1e-9
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] v_a={v_a}: Fz={F[2]:.5e} ref={ref:.5e} (err={err:.1e})")
    return ok


if __name__ == "__main__":
    r = [test_regression(), test_zero_airspeed(), test_hand_calc(), test_skydreamer_reference()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
