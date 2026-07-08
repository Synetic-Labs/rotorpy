"""
Batched-path parity: BatchedMultirotor must reproduce the canonical NumPy Multirotor state
derivative (_s_dot_fn: x_dot, v_dot, q_dot, w_dot, rotor_accel) with each added feature active.

Comparing the derivative (not stepped states) isolates the dynamics (which we ported) from the
integrator (RK4 batched vs RK45 NumPy, which we did not touch), so parity should hold to ~1e-9.

Covers: default, A2 (poly thrust/torque), A4 (asymmetric rotor dyn), A5 (rotor inertia),
B1 (AoA/advance-ratio + k_v2), and A3 (cmd_motor_throttle incl. PWM quantization).
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_batched_parity.py
"""
import copy
import numpy as np
import torch
from rotorpy.vehicles.multirotor import Multirotor, BatchedMultirotor, BatchedMultirotorParams
from rotorpy.vehicles.crazyflie_params import quad_params

torch.manual_seed(0)
DEV = torch.device('cpu')


def bench_state():
    # Benign, non-extreme state with airspeed and nonzero body rates (exercises aero + gyro).
    return dict(
        x=np.array([0.0, 0.0, 1.0]),
        v=np.array([3.0, -2.0, 1.5]),
        q=np.array([0.0, 0.0, 0.0, 1.0]),
        w=np.array([1.2, -0.8, 0.5]),
        wind=np.array([0.0, 0.0, 0.0]),
        rotor_speeds=np.array([1800.0, 2100.0, 1700.0, 2000.0]),
    )


def single_sdot(params, state, cmd):
    m = Multirotor(params, control_abstraction='cmd_motor_speeds', aero=True)
    s = Multirotor._pack_state(state)
    return m._s_dot_fn(0.0, s, cmd)


def batched_sdot(params, state, cmd):
    bp = BatchedMultirotorParams([params], 1, DEV)
    init = {k: torch.tensor(np.asarray(state[k])[None], dtype=torch.double, device=DEV) for k in
            ('x', 'v', 'q', 'w', 'wind', 'rotor_speeds')}
    bm = BatchedMultirotor(bp, 1, init, DEV, control_abstraction='cmd_motor_speeds', integrator='rk4', aero=True)
    st = {k: init[k].clone() for k in init}
    s = BatchedMultirotor._pack_state(st, 1, DEV)
    cmd_t = torch.tensor(cmd[None], dtype=torch.double, device=DEV)
    sd = bm._s_dot_fn(0.0, s[[0]], cmd_t, [0])
    return sd[0].cpu().numpy()


def compare(label, params, cmd=None, rtol=1e-5):
    # Relative tolerance: the batched path carries a small pre-existing float32 residual on
    # large-magnitude derivative components (finding F-8 note); a *correct* feature port adds
    # nothing beyond that baseline. rtol=1e-5 is far below any real physics effect.
    state = bench_state()
    if cmd is None:
        cmd = state['rotor_speeds'].copy()
    sd_s = single_sdot(params, state, cmd)
    sd_b = batched_sdot(params, state, cmd)
    diff = np.abs(sd_s - sd_b)
    scale = np.abs(sd_s) + 1e-9
    rel = np.max(diff / scale)
    ok = rel < rtol
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:28s} max rel err = {rel:.2e}  (max abs = {np.max(diff):.2e})")
    return ok


def main():
    print("=== Batched vs NumPy state-derivative parity ===")
    ok = True

    ok &= compare("default", quad_params)

    p = copy.deepcopy(quad_params)
    p.update(thrust_c0=1e-3, thrust_c1=-5e-6, torque_c0=1e-5, torque_c1=2e-8)
    ok &= compare("A2 polynomial thrust/torque", p)

    p = copy.deepcopy(quad_params)
    p['rotor_dyn_coef'] = [14.0, 1e-4, 6.0, 3e-4]
    # Use a command != current speed so the asymmetric branch matters (affects rotor_accel row).
    ok &= compare("A4 asymmetric rotor dyn", p, cmd=np.array([2200., 1600., 1900., 2300.]))

    p = copy.deepcopy(quad_params)
    p['rotor_inertia'] = 3.5e-8
    ok &= compare("A5 rotor inertia", p, cmd=np.array([2200., 1600., 1900., 2300.]))

    p = copy.deepcopy(quad_params)
    p.update(k_angle=3.145, k_hor=7.245, k_v2=1e-3, r_prop=0.0635, k_h=0.0)
    ok &= compare("B1 AoA/advance-ratio", p)

    # A3: compare get_cmd_motor_speeds for cmd_motor_throttle (incl. PWM quantization).
    p = copy.deepcopy(quad_params)
    p.update(motor_curve_k=0.5, pwm_min=7000, pwm_max=65535)
    m = Multirotor(p, control_abstraction='cmd_motor_throttle', aero=False)
    bp = BatchedMultirotorParams([p], 1, DEV)
    st = {k: torch.zeros(1, 3, dtype=torch.double) for k in ('x', 'v', 'w', 'wind')}
    st['q'] = torch.tensor([[0., 0., 0., 1.]], dtype=torch.double)
    st['rotor_speeds'] = torch.zeros(1, 4, dtype=torch.double)
    bm = BatchedMultirotor(bp, 1, st, DEV, control_abstraction='cmd_motor_throttle', integrator='rk4', aero=False)
    u = np.array([0.137, 0.62, 0.88, 0.41])
    s_speeds = m.get_cmd_motor_speeds(None, {'cmd_motor_throttle': u})
    b_speeds = bm.get_cmd_motor_speeds({k: st[k] for k in st},
                                       {'cmd_motor_throttle': torch.tensor(u[None], dtype=torch.double)}, [0])[0].cpu().numpy()
    err = np.max(np.abs(s_speeds - b_speeds))
    good = err < 1e-9
    ok &= good
    print(f"  [{'OK ' if good else 'FAIL'}] {'A3 throttle+PWM':28s} max diff = {err:.2e}")

    print()
    print("ALL PASSED" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
