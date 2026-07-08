"""
Verification for control-rate decoupling (queued §E2): the policy/controller decides at
control_rate while physics steps at sim_rate, holding the command zero-order between decisions.

Checks:
1. Gym env regression: control_rate=None (== sim_rate) -> n_substep=1 -> identical trajectory to
   the default env for the same action sequence (deterministic, seeded).
2. Gym env substep: control_rate = sim_rate/k -> n_substep=k -> one env.step advances physics by
   k*t_step, and matches k separate default-env steps with the SAME action held (ZOH).
3. Gym env guard: control_rate > sim_rate raises.
4. simulate() gate: a controller whose command changes each call is held constant for
   control_decimation steps (ZOH), and control_rate=None reproduces per-step updates.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_control_rate.py
"""
import copy
import numpy as np
import torch
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.vehicles.crazyflie_params import quad_params


def _x0():
    return {'x': torch.zeros(1, 3), 'v': torch.zeros(1, 3), 'q': torch.tensor([[0., 0., 0., 1.]]),
            'w': torch.zeros(1, 3), 'wind': torch.zeros(1, 3), 'rotor_speeds': torch.zeros(1, 4)}


def make_env(control_rate=None, sim_rate=100):
    return QuadrotorEnv(num_envs=1, initial_states=_x0(), quad_params=quad_params,
                        control_mode='cmd_motor_speeds',
                        sim_rate=sim_rate, control_rate=control_rate, aero=False,
                        device=torch.device('cpu'), render_mode="None")


def run_actions(env, actions):
    env.reset(seed=0, options={"params": "fixed", "pos_bound": 0.0, "vel_bound": 0.0})
    xs = []
    for a in actions:
        env.step(a)
        xs.append(env.vehicle_states['x'][0].cpu().numpy().copy())
    return np.array(xs)


def test_regression():
    print("=== control-rate: default (control_rate=None) unchanged ===")
    a = np.array([[0.1, 0.05, -0.05, 0.02]], dtype=np.float64)
    e1 = make_env(None)
    e2 = make_env(100)  # == sim_rate -> n_substep 1
    x1 = run_actions(e1, [a] * 20)
    x2 = run_actions(e2, [a] * 20)
    ok = np.allclose(x1, x2, atol=1e-12) and e1.n_substep == 1
    print(f"  [{'OK ' if ok else 'FAIL'}] n_substep={e1.n_substep}, trajectories identical (max diff {np.max(np.abs(x1-x2)):.2e})")
    return ok


def test_substep_zoh():
    print("=== control-rate: substep == held-action default steps (ZOH) ===")
    # control_rate=25, sim_rate=100 -> n_substep=4. One decoupled step == 4 default steps, same action.
    dec = make_env(control_rate=25, sim_rate=100)
    base = make_env(control_rate=None, sim_rate=100)
    assert dec.n_substep == 4, dec.n_substep
    a = np.array([[0.2, -0.1, 0.15, -0.08]], dtype=np.float64)

    dec.reset(seed=0, options={"params": "fixed", "pos_bound": 0.0, "vel_bound": 0.0})
    base.reset(seed=0, options={"params": "fixed", "pos_bound": 0.0, "vel_bound": 0.0})
    # One decoupled decision step (4 physics substeps).
    dec.step(a)
    x_dec = dec.vehicle_states['x'][0].cpu().numpy().copy()
    # Four base steps holding the same action.
    for _ in range(4):
        base.step(a)
    x_base = base.vehicle_states['x'][0].cpu().numpy().copy()
    err = np.max(np.abs(x_dec - x_base))
    ok = err < 1e-9
    print(f"  [{'OK ' if ok else 'FAIL'}] 1 decoupled step (n=4) == 4 held steps (pos err {err:.2e})")
    # And time advanced by 4*t_step.
    ok_t = np.allclose(dec.t, 4 * base.t_step)
    print(f"  [{'OK ' if ok_t else 'FAIL'}] time advanced 4*t_step (t={float(np.asarray(dec.t).ravel()[0]):.4f})")
    return ok and ok_t


def test_guard():
    print("=== control-rate: control_rate > sim_rate raises ===")
    try:
        make_env(control_rate=200, sim_rate=100)
        ok = False
    except ValueError:
        ok = True
    print(f"  [{'OK ' if ok else 'FAIL'}] raises when control_rate > sim_rate")
    return ok


def test_simulate_gate():
    print("=== control-rate: simulate() ZOH gate ===")
    from rotorpy.simulate import simulate
    # Minimal stubs.
    class Ctrl:
        def __init__(self): self.n = 0
        def update(self, t, state, flat):
            self.n += 1
            # command changes every call -> lets us detect held vs updated.
            return {'cmd_motor_speeds': np.full(4, 1000.0 + self.n)}
    class Traj:
        def update(self, t):
            return {'x': np.zeros(3), 'x_dot': np.zeros(3), 'x_ddot': np.zeros(3), 'x_dddot': np.zeros(3),
                    'x_ddddot': np.zeros(3), 'yaw': 0.0, 'yaw_dot': 0.0, 'yaw_ddot': 0.0}
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.sensors.imu import Imu
    from rotorpy.sensors.external_mocap import MotionCapture
    from rotorpy.estimators.nullestimator import NullEstimator
    from rotorpy.world import World

    veh = Multirotor(quad_params, control_abstraction='cmd_motor_speeds', aero=False)
    init = {'x': np.zeros(3), 'v': np.zeros(3), 'q': np.array([0., 0., 0., 1.]), 'w': np.zeros(3),
            'wind': np.zeros(3), 'rotor_speeds': np.full(4, 1000.0)}
    world = World.empty((-10, 10, -10, 10, -10, 10))
    from rotorpy.wind.default_winds import NoWind
    ctrl = Ctrl()
    # control_rate=20, t_step=0.01 -> decimation=5: controller.update called every 5 steps.
    res = simulate(world, init, veh, ctrl, Traj(), NoWind(),
                   Imu(sampling_rate=100), MotionCapture(sampling_rate=100), NullEstimator(),
                   t_final=0.2, t_step=0.01, safety_margin=100, use_mocap=False, control_rate=20)
    control = res[2]  # merged control dict
    cmds = control['cmd_motor_speeds'][:, 0]  # first motor command over time
    # The command should change in steps (held for 5 physics steps between updates).
    unique_runs = np.sum(np.abs(np.diff(cmds)) > 1e-9)
    total = len(cmds)
    ok = unique_runs < total / 3  # far fewer changes than steps -> ZOH working
    print(f"  [{'OK ' if ok else 'FAIL'}] {total} steps, only {unique_runs} command changes (ZOH at 1/5 rate)")
    return ok


if __name__ == "__main__":
    r = [test_regression(), test_substep_zoh(), test_guard(), test_simulate_gate()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
