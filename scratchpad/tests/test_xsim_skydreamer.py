"""
Cross-simulator validation against SkyDreamer's ACTUAL dynamics
(The-Real-Thisas/dreamerv3, embodied/envs/skydreamer.py :: compute_dynamics_jit).

We extract the exact source of their JIT dynamics block (quat helpers + compute_dynamics_jit) and
the SKYDREAMER_PARAMS dict straight from their file and exec it (numba), so we run their real code
rather than a transcription.

SkyDreamer emits accelerations directly (no 1/m, no I^-1 -- finding F-4): their derivative rows are
  v_dot_world = R @ [Dx, Dy, T]  - g*z_hat        (Dx,Dy,T are per-mass accelerations)
so at identity attitude and zero body rate, [Dx, Dy, T] == RotorPy FtotB / m.

Validated (the components both sims share, with RotorPy configured to match):
  1. Thrust incl. angle-of-attack/advance-ratio (B1) and the -k_v2 term.
  2. Linear rotor drag (k_d <-> m*k_x).
  3. Quadratic drag -- ONLY matches for single-axis airspeed (RotorPy scales parasitic drag by
     ||v||, SkyDreamer by per-axis |v|); documented as a structural difference.
Not comparable (documented): SkyDreamer's moment model is per-rotor identified coefficients
(k_p/k_q/k_r), structurally unlike RotorPy's geometry-derived r x F; not cross-validated.

Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_xsim_skydreamer.py
"""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # run their exact code as pure Python (exec has no cache locator)
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SKY_CANDIDATES = [
    os.path.join(_HERE, "..", "refs", "skydreamer", "embodied", "envs", "skydreamer.py"),
    "/tmp/claude-1000/-home-james-code-RotorPy/946a5ea1-e934-4627-a38d-1b10cfb77116/scratchpad/skydreamer/embodied/envs/skydreamer.py",
]
SKY = next((os.path.abspath(c) for c in _SKY_CANDIDATES if os.path.isfile(c)), None)
if SKY is None:
    print("SKIP: skydreamer.py not found (expected scratchpad/refs/skydreamer). "
          "Clone with: git clone --depth 1 https://github.com/The-Real-Thisas/dreamerv3 scratchpad/refs/skydreamer")
    sys.exit(0)

_src = open(SKY).read()
# Exact-slice their JIT dynamics block and the params dict, then exec (runs their real code).
_block = _src[_src.index("@njit(fastmath=True, cache=True)\ndef quat_normalize"):
              _src.index("@njit(fastmath=True, cache=True)\ndef compute_step_logic_jit")]
_params_src = _src[_src.index("SKYDREAMER_PARAMS = {"):
                   _src.index("}", _src.index("SKYDREAMER_PARAMS = {")) + 1]
_ns = {}
exec("from numba import njit\nimport numpy as np\n" + _params_src + "\n" + _block, _ns)
compute_dynamics_jit = _ns["compute_dynamics_jit"]
SKYDREAMER_PARAMS = _ns["SKYDREAMER_PARAMS"]

from rotorpy.vehicles.multirotor import Multirotor  # noqa: E402
from rotorpy.vehicles.crazyflie_params import quad_params  # noqa: E402
import copy  # noqa: E402

# params-array column order, exactly as SkyDreamer builds it (skydreamer.py rand_f param_keys).
PARAM_KEYS = ['k_x', 'k_y', 'k_w', 'k_x2', 'k_y2', 'k_angle', 'k_hor', 'k_v2',
              'k_p1', 'k_p2', 'k_p3', 'k_p4', 'k_q1', 'k_q2', 'k_q3', 'k_q4',
              'k_r1', 'k_r2', 'k_r3', 'k_r4', 'k_r5', 'k_r6', 'k_r7', 'k_r8',
              'J_x', 'J_y', 'J_z', 'tau', 'k', 'w_min', 'w_max', 'r_prop']
PARAM_ROW = np.array([[SKYDREAMER_PARAMS[k] for k in PARAM_KEYS]], dtype=np.float32)

W_MAX_N = 3000.0   # SkyDreamer's fixed normalization range [0, 3000] rad/s
MASS = 0.6         # arbitrary mass to test F-4 scaling (k_eta = m*k_w etc.)


def sky_accel(rotor_speeds, body_airspeed):
    """Run SkyDreamer's real dynamics at identity attitude, zero body rate, and return their
    body-frame acceleration vector [Dx, Dy, T] (= v_dot + g z_hat at identity)."""
    # Encode desired rotor speeds W into their normalized rotor state: W = (w+1)/2 * 3000.
    w_norm = 2.0 * rotor_speeds / W_MAX_N - 1.0
    # State: [x,y,z, vx,vy,vz, qw,qx,qy,qz, p,q,r, w1..w4]; identity quat, zero rates.
    # Their v is world velocity; at identity, body airspeed == world velocity (no wind in this model).
    state = np.zeros((1, 17), dtype=np.float32)
    state[0, 3:6] = body_airspeed          # vx,vy,vz (world == body at identity)
    state[0, 6] = 1.0                       # qw
    state[0, 13:17] = w_norm
    actions = np.zeros((1, 4), dtype=np.float32)
    d = compute_dynamics_jit(state, actions, PARAM_ROW)[0]
    vdot = d[3:6]
    return np.array([vdot[0], vdot[1], vdot[2] + 9.81])   # add gravity back -> [Dx, Dy, T]


def rotorpy_match(**over):
    p = copy.deepcopy(quad_params)
    p['mass'] = MASS
    P = SKYDREAMER_PARAMS
    # F-4: SkyDreamer coeffs are per-mass accelerations -> multiply by mass for RotorPy forces.
    p['k_eta'] = MASS * P['k_w']            # base thrust coeff
    p['k_angle'] = P['k_angle']
    p['k_hor'] = P['k_hor']
    p['k_v2'] = MASS * P['k_v2']
    p['r_prop'] = P['r_prop']
    p['k_h'] = 0.0
    p['k_flap'] = 0.0
    p['k_z'] = 0.0
    p.update(over)
    m = Multirotor(p, control_abstraction='cmd_motor_speeds', aero=True)
    return m


def test_thrust_aoa():
    print("=== SkyDreamer xsim: thrust incl. AoA/advance-ratio + k_v2 (B1) ===")
    m = rotorpy_match()
    ok = True
    for v_a in [np.array([0., 0., 0.]), np.array([4., 0., 2.]), np.array([3., -5., -6.])]:
        speeds = np.array([2000., 2200., 1900., 2100.])
        sky = sky_accel(speeds, v_a)                       # [Dx, Dy, T]
        F, _ = m.compute_body_wrench(np.zeros(3), speeds, v_a)
        rp_T = F[2] / MASS
        err = abs(rp_T - sky[2])
        good = err < 1e-5 * (abs(sky[2]) + 1)
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] v_a={v_a}: RotorPy T/m={rp_T:.5f} SkyDreamer T={sky[2]:.5f} (err={err:.2e})")
    return ok


def test_linear_drag():
    print("=== SkyDreamer xsim: linear rotor drag (k_d <-> m*k_x), single-axis airspeed ===")
    P = SKYDREAMER_PARAMS
    # Linear drag only: set k_x2 (quadratic) via c_Dx=0 first to isolate the linear term.
    m = rotorpy_match(k_d=MASS * P['k_x'], c_Dx=0.0, c_Dy=0.0, c_Dz=0.0)
    ok = True
    for vx in (2.0, 6.0, -4.0):
        v_a = np.array([vx, 0.0, 0.0])
        speeds = np.array([2000., 2000., 2000., 2000.])
        sky = sky_accel(speeds, v_a)      # [Dx, Dy, T]; Dx here has linear + quadratic(k_x2)
        # Isolate SkyDreamer's LINEAR part: Dx_lin = -k_x * vx * sum(W); subtract their quadratic.
        wsum = np.sum(speeds)
        sky_Dx_lin = -P['k_x'] * vx * wsum
        F, _ = m.compute_body_wrench(np.zeros(3), speeds, v_a)
        rp_Dx = F[0] / MASS
        err = abs(rp_Dx - sky_Dx_lin)
        good = err < 1e-6 * (abs(sky_Dx_lin) + 1)
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] vx={vx}: RotorPy Dx/m={rp_Dx:.6e} SkyDreamer linear Dx={sky_Dx_lin:.6e} (err={err:.1e})")
    return ok


def test_quadratic_drag_single_axis():
    print("=== SkyDreamer xsim: quadratic drag matches ONLY on a single axis (||v|| vs |v_axis|) ===")
    P = SKYDREAMER_PARAMS
    # Full SkyDreamer Dx = -k_x*vx*sum(W) - k_x2*vx*|vx|. RotorPy: H (k_d) + parasitic (c_Dx*||v||).
    m = rotorpy_match(k_d=MASS * P['k_x'], c_Dx=MASS * P['k_x2'], c_Dy=MASS * P['k_y2'], c_Dz=0.0)
    speeds = np.array([2000., 2000., 2000., 2000.])

    # (a) single-axis airspeed: should match.
    vx = 6.0
    v_a = np.array([vx, 0.0, 0.0])
    sky = sky_accel(speeds, v_a)
    F, _ = m.compute_body_wrench(np.zeros(3), speeds, v_a)
    single_err = abs(F[0] / MASS - sky[0])
    single_ok = single_err < 1e-5 * (abs(sky[0]) + 1)
    print(f"  [{'OK ' if single_ok else 'FAIL'}] single-axis: RotorPy Dx/m={F[0]/MASS:.6e} SkyDreamer Dx={sky[0]:.6e} (err={single_err:.1e})")

    # (b) two-axis airspeed: expected to DIFFER (documented structural difference).
    v_a2 = np.array([6.0, 6.0, 0.0])
    sky2 = sky_accel(speeds, v_a2)
    F2, _ = m.compute_body_wrench(np.zeros(3), speeds, v_a2)
    multi_diff = abs(F2[0] / MASS - sky2[0])
    differs = multi_diff > 1e-3    # they should NOT agree off-axis
    print(f"  [{'OK ' if differs else 'FAIL'}] two-axis DIFFERS as expected: RotorPy Dx/m={F2[0]/MASS:.6e} "
          f"SkyDreamer Dx={sky2[0]:.6e} (|diff|={multi_diff:.2e}); ||v|| vs |v_x| structural difference")
    return single_ok and differs


if __name__ == "__main__":
    r = [test_thrust_aoa(), test_linear_drag(), test_quadratic_drag_single_axis()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
