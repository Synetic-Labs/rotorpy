"""
Verification for thrust-axis misalignment (per-rotor thrust_dir).

Each rotor's thrust (and reaction/yaw torque) acts along its body-frame axis thrust_dir_i
(unit vector, default +z). Small tilts model real assembly misalignment.

Checks:
1. Regression: default thrust_dir = +z reproduces the aligned wrench exactly (thrust along z,
   no parasitic x/y force).
2. Tilt physics: a rotor tilted by theta about body-y produces thrust
   [T*sin(theta), 0, T*cos(theta)] and net force/moment matching a hand computation.
3. Yaw torque follows the tilted axis.
Run: PYTHONPATH=. .venv/bin/python scratchpad/tests/test_thrust_axis.py
"""
import copy
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params


def make(thrust_dir=None):
    p = copy.deepcopy(quad_params)
    if thrust_dir is not None:
        p['thrust_dir'] = thrust_dir
    return Multirotor(p, control_abstraction='cmd_motor_speeds', aero=False)


def test_regression():
    print("=== thrust-axis regression: default +z reproduces aligned wrench ===")
    m = make()
    assert np.allclose(m.thrust_dir, np.tile([0, 0, 1.0], (m.num_rotors, 1)))
    speeds = np.array([1800., 2100., 1700., 2000.])
    F, M = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    Fz = np.sum(m.k_eta * speeds**2)
    ok = abs(F[2] - Fz) < 1e-9 and np.max(np.abs(F[:2])) < 1e-12
    print(f"  [{'OK ' if ok else 'FAIL'}] thrust along z, no x/y (Fz err {abs(F[2]-Fz):.1e}, |Fxy| {np.max(np.abs(F[:2])):.1e})")
    return ok


def test_tilt_physics():
    print("=== thrust-axis tilt: single rotor tilted about body-y ===")
    theta = np.radians(5.0)
    e = np.array([np.sin(theta), 0.0, np.cos(theta)])
    # Only rotor 0 tilted; others nominal +z.
    tdir = np.tile([0, 0, 1.0], (4, 1)).astype(float)
    tdir[0] = e
    m = make(thrust_dir=tdir)
    speeds = np.array([2000., 2000., 2000., 2000.])
    F, M = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))

    # Hand computation.
    Tmag = m.k_eta * speeds**2                      # per-rotor magnitude
    T_vecs = m.thrust_dir * Tmag[:, None]           # (4,3)
    expect_F = np.sum(T_vecs, axis=0)
    geom = m.rotor_geometry
    expect_M_thrust = np.sum([np.cross(geom[i], T_vecs[i]) for i in range(4)], axis=0)
    # yaw/reaction torque along each rotor axis
    tau_mag = m.rotor_dir * m.k_m * speeds**2
    expect_M_yaw = np.sum(m.thrust_dir * tau_mag[:, None], axis=0)
    expect_M = expect_M_thrust + expect_M_yaw

    okF = np.max(np.abs(F - expect_F)) < 1e-9
    okM = np.max(np.abs(M - expect_M)) < 1e-9
    print(f"  [{'OK ' if okF else 'FAIL'}] net force matches (err {np.max(np.abs(F-expect_F)):.1e}); "
          f"parasitic Fx = {F[0]:.4e} (= T0*sin5deg)")
    print(f"  [{'OK ' if okM else 'FAIL'}] net moment matches Sum r x T + tilted yaw (err {np.max(np.abs(M-expect_M)):.1e})")
    # Sanity: Fx equals rotor-0 thrust * sin(theta).
    okx = np.isclose(F[0], (m.k_eta[0] * speeds[0]**2) * np.sin(theta))
    print(f"  [{'OK ' if okx else 'FAIL'}] Fx == T0*sin(theta)")
    return okF and okM and okx


def test_uniform_tilt_yaw():
    print("=== thrust-axis: uniform tilt tilts net thrust; yaw follows axis ===")
    theta = np.radians(3.0)
    e = np.array([0.0, np.sin(theta), np.cos(theta)])   # tilt about body-x
    tdir = np.tile(e, (4, 1))
    m = make(thrust_dir=tdir)
    speeds = np.full(4, 1900.0)
    F, M = m.compute_body_wrench(np.zeros(3), speeds, np.zeros(3))
    Ttot = np.sum(m.k_eta * speeds**2)
    okF = np.isclose(F[1], Ttot * np.sin(theta)) and np.isclose(F[2], Ttot * np.cos(theta))
    # Net yaw (z) reaction reduced by cos(theta) since torque tilts off z; balanced quad -> Mz small
    # but each rotor's yaw contributes along e; check the z-component scaling.
    tau_z = np.sum(m.rotor_dir * m.k_m * speeds**2) * np.cos(theta)
    okM = np.isclose(M[2], tau_z, atol=1e-12)
    print(f"  [{'OK ' if okF else 'FAIL'}] net thrust tilted: Fy={F[1]:.4e}, Fz={F[2]:.4e}")
    print(f"  [{'OK ' if okM else 'FAIL'}] yaw z-component scaled by cos(theta)")
    return okF and okM


if __name__ == "__main__":
    r = [test_regression(), test_tilt_physics(), test_uniform_tilt_yaw()]
    print()
    print("ALL PASSED" if all(r) else "SOME FAILED")
