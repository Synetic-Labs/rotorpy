"""
Verification for IMU findings F-1, F-2, F-2b (see scratchpad/rotorpy-physics-additions.md).

F-1  lever-arm frame mixing: rotational contribution must be formed in the body frame
     (where w, wdot, p_BS live) and rotated to world, not w(body) x p_BS(world).
F-2  gyro must be expressed in the sensor frame: gyro = R_BS.T @ w_body.
F-2b batched R_SW must be a matrix product R_BS.T @ R_BW, not an element-wise einsum.

Strategy: compare the implementations against a fully independent closed-form reference,
and check the NumPy and batched paths agree for offset + rotated configurations.
Run: python scratchpad/tests/test_imu_fixes.py
"""
import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.sensors.imu import Imu, BatchedImu

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

G = np.array([0.0, 0.0, -9.81])


def reference_measurement(q_WB, w_B, alpha_B, a_WB_W, R_BS, p_BS, g=G):
    """Independent closed-form IMU model (no repo code)."""
    R_WB = Rotation.from_quat(q_WB).as_matrix()
    a_rot_B = np.cross(alpha_B, p_BS) + np.cross(w_B, np.cross(w_B, p_BS))
    a_WS_W = a_WB_W + R_WB @ a_rot_B
    R_SW = R_BS.T @ R_WB.T
    accel = R_SW @ (a_WS_W - g)
    gyro = R_BS.T @ w_B
    return accel, gyro


def make_cases():
    rng = np.random.default_rng(0)
    cases = []
    # (label, q, w, alpha, a_WB_W, R_BS, p_BS)
    cases.append(("default level", np.array([0., 0., 0., 1.]),
                  np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3)))
    cases.append(("default, hover accel", np.array([0., 0., 0., 1.]),
                  np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3)))
    for i in range(6):
        q = Rotation.random(random_state=rng).as_quat()
        w = rng.normal(size=3)
        alpha = rng.normal(size=3)
        a = rng.normal(size=3)
        R_BS = Rotation.random(random_state=rng).as_matrix()
        p_BS = rng.normal(size=3) * 0.05
        cases.append((f"random {i}", q, w, alpha, a, R_BS, p_BS))
    # A case with offset but identity mounting (isolates F-1)
    cases.append(("offset only", Rotation.from_euler('xyz', [30, -20, 45], degrees=True).as_quat(),
                  np.array([1., -2., 0.5]), np.array([0.3, 0.1, -0.2]),
                  np.array([0.5, 0., 1.]), np.eye(3), np.array([0.1, 0.0, -0.03])))
    # A case with mounting but no offset (isolates F-2)
    cases.append(("mount only", Rotation.from_euler('xyz', [10, 40, -15], degrees=True).as_quat(),
                  np.array([0.2, 0.4, -0.1]), np.array([0.0, 0.1, 0.0]),
                  np.array([-0.3, 0.2, 0.4]),
                  Rotation.from_euler('z', 90, degrees=True).as_matrix(), np.zeros(3)))
    return cases


def test_numpy():
    print("=== NumPy Imu vs closed-form reference ===")
    ok = True
    for label, q, w, alpha, a, R_BS, p_BS in make_cases():
        imu = Imu(R_BS=R_BS, p_BS=p_BS, sampling_rate=500, gravity_vector=G)
        meas = imu.measurement({'q': q, 'w': w, 'v': np.zeros(3), 'x': np.zeros(3)},
                               {'vdot': a, 'wdot': alpha}, with_noise=False)
        ref_a, ref_g = reference_measurement(q, w, alpha, a, R_BS, p_BS)
        da = np.max(np.abs(meas['accel'] - ref_a))
        dg = np.max(np.abs(meas['gyro'] - ref_g))
        status = "OK " if (da < 1e-9 and dg < 1e-9) else "FAIL"
        ok = ok and status == "OK "
        print(f"  [{status}] {label:16s} accel_err={da:.2e} gyro_err={dg:.2e}")
    return ok


def test_batched():
    if not HAVE_TORCH:
        print("=== Batched Imu: torch not available, skipped ===")
        return True
    print("=== Batched Imu vs closed-form reference ===")
    ok = True
    for label, q, w, alpha, a, R_BS, p_BS in make_cases():
        bimu = BatchedImu(num_drones=1,
                          R_BS=torch.tensor(R_BS), p_BS=torch.tensor(p_BS),
                          sampling_rate=500,
                          gravity_vector=torch.tensor(G))
        state = {'q': torch.tensor(q).unsqueeze(0), 'w': torch.tensor(w).unsqueeze(0),
                 'v': torch.zeros(1, 3), 'x': torch.zeros(1, 3)}
        acc = {'vdot': torch.tensor(a).unsqueeze(0), 'wdot': torch.tensor(alpha).unsqueeze(0)}
        meas = bimu.measurement(state, acc, with_noise=False)
        ref_a, ref_g = reference_measurement(q, w, alpha, a, R_BS, p_BS)
        da = float(np.max(np.abs(meas['accel'][0].numpy() - ref_a)))
        dg = float(np.max(np.abs(meas['gyro'][0].numpy() - ref_g)))
        status = "OK " if (da < 1e-9 and dg < 1e-9) else "FAIL"
        ok = ok and status == "OK "
        print(f"  [{status}] {label:16s} accel_err={da:.2e} gyro_err={dg:.2e}")
    return ok


def test_default_regression():
    """With p_BS=0 and R_BS=I the measurement must reduce to R_WB.T @ (vdot - g)."""
    print("=== Default-config regression (p_BS=0, R_BS=I) ===")
    rng = np.random.default_rng(3)
    ok = True
    for i in range(4):
        q = Rotation.random(random_state=rng).as_quat()
        w = rng.normal(size=3)
        alpha = rng.normal(size=3)
        a = rng.normal(size=3)
        imu = Imu(sampling_rate=500, gravity_vector=G)
        meas = imu.measurement({'q': q, 'w': w, 'v': np.zeros(3), 'x': np.zeros(3)},
                               {'vdot': a, 'wdot': alpha}, with_noise=False)
        R_WB = Rotation.from_quat(q).as_matrix()
        expect_a = R_WB.T @ (a - G)
        expect_g = w
        da = np.max(np.abs(meas['accel'] - expect_a))
        dg = np.max(np.abs(meas['gyro'] - expect_g))
        status = "OK " if (da < 1e-12 and dg < 1e-12) else "FAIL"
        ok = ok and status == "OK "
        print(f"  [{status}] case {i} accel_err={da:.2e} gyro_err={dg:.2e}")
    return ok


if __name__ == "__main__":
    r1 = test_default_regression()
    r2 = test_numpy()
    r3 = test_batched()
    print()
    print("ALL PASSED" if (r1 and r2 and r3) else "SOME FAILED")
