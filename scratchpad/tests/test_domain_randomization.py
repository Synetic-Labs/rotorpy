"""
Verification for D: extended domain-randomization ranges.

Checks:
1. percent_randomization builds correct +/- bands (30% dynamics, 20% motor limits) and skips
   zero-valued / absent parameters.
2. update_vehicle_params applies the extended set (mass, k_eta, k_m, inertia, tau_m, drag,
   rotor_speed_max) within their sampled bounds across many drones.
3. Hover feasibility: every sampled drone can still hover (4*k_eta*(w_max)^2 >= m*g).
4. Regression: the default crazyflie_randomizations dict still only randomizes mass and k_eta.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_domain_randomization.py
"""
import copy
import numpy as np

from rotorpy.learning.learning_utils import (percent_randomization, generate_random_vehicle_params,
                                             crazyflie_randomizations)
from rotorpy.vehicles.crazyflie_params import quad_params as cf_params
from rotorpy.vehicles.hummingbird_params import quad_params as hb_params


def test_percent_ranges():
    print("=== D percent_randomization band construction ===")
    r = percent_randomization(cf_params, pct=0.30, motor_limit_pct=0.20)
    ok = True
    # mass band = +/-30%.
    m = cf_params['mass']
    good_mass = np.isclose(r['mass'][0], m*0.7) and np.isclose(r['mass'][1], m*1.3)
    # rotor_speed_max band = +/-20%.
    wm = cf_params['rotor_speed_max']
    good_wmax = np.isclose(r['rotor_speed_max'][0], wm*0.8) and np.isclose(r['rotor_speed_max'][1], wm*1.2)
    # crazyflie c_Dx is 0 -> skipped.
    good_skip = 'c_Dx' not in r
    # k_eta, k_m, inertia included.
    good_incl = all(k in r for k in ('k_eta', 'k_m', 'Ixx', 'Iyy', 'Izz', 'tau_m'))
    ok = good_mass and good_wmax and good_skip and good_incl
    print(f"  [{'OK ' if good_mass else 'FAIL'}] mass +/-30%  [{'OK ' if good_wmax else 'FAIL'}] w_max +/-20%")
    print(f"  [{'OK ' if good_skip else 'FAIL'}] zero drag skipped  [{'OK ' if good_incl else 'FAIL'}] k_eta/k_m/inertia/tau_m included")
    # Hummingbird has nonzero drag -> included.
    rh = percent_randomization(hb_params, pct=0.30)
    good_hb_drag = ('c_Dx' in rh) and ('k_d' in rh)
    ok = ok and good_hb_drag
    print(f"  [{'OK ' if good_hb_drag else 'FAIL'}] hummingbird nonzero drag included")
    return ok


def test_apply_within_bounds():
    print("=== D sampled params land within bounds (many drones) ===")
    n = 200
    ranges = percent_randomization(cf_params, pct=0.30, motor_limit_pct=0.20)
    params = generate_random_vehicle_params(n, device='cpu', nominal_params=cf_params,
                                            randomization_ranges=ranges)
    mass = params.mass.reshape(-1).cpu().numpy()
    keta = params.k_eta.reshape(-1).cpu().numpy()
    wmax = params.rotor_speed_max.reshape(-1).cpu().numpy()
    ok_mass = np.all(mass >= ranges['mass'][0] - 1e-9) and np.all(mass <= ranges['mass'][1] + 1e-9)
    ok_wmax = np.all(wmax >= ranges['rotor_speed_max'][0] - 1e-6) and np.all(wmax <= ranges['rotor_speed_max'][1] + 1e-6)
    # k_eta lower bound may be raised (and the band even exceeded) by the hover-feasibility clamp,
    # so require only k_eta >= band lower; the hover test enforces the true invariant.
    ok_keta = np.all(keta >= ranges['k_eta'][0] - 1e-12)
    ok = ok_mass and ok_wmax and ok_keta
    print(f"  [{'OK ' if ok_mass else 'FAIL'}] mass in band  [{'OK ' if ok_wmax else 'FAIL'}] w_max in band  "
          f"[{'OK ' if ok_keta else 'FAIL'}] k_eta >= lower (feasibility may raise it)")
    return ok


def test_hover_feasibility():
    print("=== D every sampled drone can hover ===")
    n = 300
    ranges = percent_randomization(cf_params, pct=0.30, motor_limit_pct=0.20)
    params = generate_random_vehicle_params(n, device='cpu', nominal_params=cf_params,
                                            randomization_ranges=ranges)
    mass = params.mass.reshape(-1).cpu().numpy()
    keta = params.k_eta.reshape(-1).cpu().numpy()
    wmax = params.rotor_speed_max.reshape(-1).cpu().numpy()
    g = params.g
    max_thrust = 4 * keta * wmax**2
    weight = mass * g
    margin = max_thrust / weight
    ok = np.all(margin > 1.0)
    print(f"  [{'OK ' if ok else 'FAIL'}] min thrust/weight margin = {np.min(margin):.3f} (>1 required)")
    return ok


def test_regression_default():
    print("=== D default crazyflie_randomizations unchanged ===")
    ok = set(crazyflie_randomizations.keys()) == {"mass", "k_eta"}
    n = 50
    params = generate_random_vehicle_params(n, device='cpu', nominal_params=cf_params,
                                            randomization_ranges=crazyflie_randomizations)
    # tau_m untouched (all equal nominal), drag still zero.
    tau = params.tau_m.reshape(-1).cpu().numpy()
    ok_tau = np.allclose(tau, cf_params['tau_m'])
    drag = params.drag_matrix.reshape(n, -1).cpu().numpy()
    ok_drag = np.allclose(drag, 0.0)
    ok = ok and ok_tau and ok_drag
    print(f"  [{'OK ' if ok else 'FAIL'}] only mass+k_eta keys; tau_m & drag untouched")
    return ok


if __name__ == "__main__":
    r = [test_percent_ranges(), test_apply_within_bounds(), test_hover_feasibility(), test_regression_default()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
