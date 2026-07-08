# RotorPy Physics — Complete Model Inventory & Additions Plan

**Purpose.** RotorPy is our single source of truth for validated multirotor physics. Part I lays
out every physics model RotorPy currently implements, with the exact math as coded and a
verification mark. Part II lists the additions collected from external sources, same format.
Part III records verification findings and the test protocol. Everything is framework-free math,
ready to port to the compiled custom-firmware repo; candidate items also go upstream to RotorPy
as PRs for author review.

## Implementation status (2026-07-07) — all items landed + verified

Every addition is opt-in / zero-by-default, so nominal RotorPy behavior is bit-identical; each has
a standalone verification script in `scratchpad/tests/`. Repo regression suite (15 tests w/ deps
available) and the NumPy↔batched equivalence test still pass.

| ID | Item | Files | Test |
|---|---|---|---|
| F-1/F-2/F-2b | IMU frame fixes (lever arm, gyro mount, batched R_SW matmul) | `sensors/imu.py` | `test_imu_fixes.py` |
| C1 | Two-band external-wrench disturbance | `disturbances/`, `vehicles/multirotor.py`, `simulate.py`, `environments.py` | `test_wrench_disturbance.py` |
| A1 | Per-rotor thrust/torque coefficients | `vehicles/multirotor.py` | `test_per_rotor_coeffs.py` |
| A5 | Rotor-inertia reaction + gyroscopic precession | `vehicles/multirotor.py` | `test_rotor_inertia.py` |
| A3 | Nonlinear throttle-curve abstraction (`cmd_motor_throttle`) | `vehicles/multirotor.py` | `test_throttle_curve.py` |
| A6 | Motor command latency (delay line) | `vehicles/multirotor.py` | `test_command_delay.py` |
| A2 | Polynomial thrust/torque curves | `vehicles/multirotor.py` | `test_poly_thrust.py` |
| A4 | Asymmetric spin-up/spin-down motor dynamics | `vehicles/multirotor.py` | `test_motor_dynamics.py` |
| B1 | Thrust vs angle-of-attack / advance ratio | `vehicles/multirotor.py` | `test_aoa_thrust.py` |
| A7/C2 | PWM quantization + battery voltage-sag | `vehicles/multirotor.py`, `battery.py` | `test_pwm_battery.py` |
| D | Extended domain-randomization ranges | `learning/learning_utils.py` | `test_domain_randomization.py` |
| guard | k_h vs k_angle/k_hor double-count guard | `vehicles/multirotor.py` (both paths) | `test_aoa_kh_guard.py` |
| xsim | Cross-validation vs Crazyflow & SkyDreamer real code | (Part IV) | `test_xsim_crazyflow.py`, `test_xsim_skydreamer.py` |
| batched | A2/A3/A4/A5/B1 parity + F-8 float32 fix | `vehicles/multirotor.py` | `test_batched_parity.py` |
| E2 | Control-rate decoupling (harness, not physics) | `learning/quadrotor_environments.py`, `simulate.py`, `environments.py` | `test_control_rate.py` |

**Cross-simulator validation done** (Part IV): every added component reproduces the reference
sims' *real running code* (Crazyflow, SkyDreamer) exactly or to a documented epsilon; F-3 and F-4
confirmed against their code (F-3's flipped axis corrected to x).

**Batched parity done** for C1 (wrench hook) + A2/A3/A4/A5/B1: ported into `BatchedMultirotor`
and verified to match the canonical NumPy state derivative (relative err < 1e-5, see
`test_batched_parity.py`). This surfaced and fixed **F-8** (batched ran in float32). Still deferred
in batched: **A1 per-rotor k_eta/k_m** (needs `(num_drones, num_rotors)` allocation restructuring)
and A6 command latency (single-drone `step` only). The batched thrust→speed inversion for
SE3-family abstractions now uses the polynomial-aware `_thrust_to_speed`, matching NumPy.

Environment note: verified under a local venv (`.venv`, numpy 2.5 / torch 2.12 CPU); the two
skipped repo tests need optional deps (`stable_baselines3`, `foundation_policy`).

**Verification legend:**
- ✅ math checked against the implementation *and* a published source / independent derivation
- ⚠️ issue found — see Part III findings
- 🔶 checked against source code only (no independent published reference located)

**Sources:**
- RotorPy paper: Folk, Paulos, Kumar, *RotorPy: A Python-based Multirotor Simulator with
  Aerodynamics for Education and Research*, [arXiv:2306.04485](https://arxiv.org/abs/2306.04485)
- SkyDreamer: [arXiv:2510.14783](https://arxiv.org/abs/2510.14783) + reference implementation
  [The-Real-Thisas/dreamerv3](https://github.com/The-Real-Thisas/dreamerv3) `embodied/envs/skydreamer.py`
- Crazyflow: [learnsyslab/crazyflow](https://github.com/learnsyslab/crazyflow)
  (`crazyflow/dynamics/first_principles/dynamics.py`, `crazyflow/drones/params.toml`)
- Mahony, Kumar, Corke, *Multirotor Aerial Vehicles: Modeling, Estimation, and Control of
  Quadrotor*, IEEE RAM 2012 (rotor drag / flapping background)
- Graf, *Quaternions and Dynamics* (quaternion kinematics, cited in RotorPy code)
- Genesis ([Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis)):
  reviewed, **nothing to port** — prop joints fixed, RPM → `KF·rpm²` force + `KM·rpm²` yaw torque
  only, prop spin visual-only, no gyroscopic effects.

**Notation.** World frame ENU, gravity `g = 9.81`. Body frame x-forward, z-up through rotors.
`x, v` world position/velocity; `q = [q_x,q_y,q_z,q_w]` body→world quaternion; `R = R(q)`;
`ω = (p,q,r)` body rates (body frame); `Ω_i ≥ 0` rotor speed (rad/s); `m` mass; `I` inertia
matrix; `r_i` rotor position in body frame; `v_a = Rᵀ(v − v_wind)` body-frame airspeed.

---

# Part I — RotorPy current physics (as implemented, canonical)

All references `rotorpy/vehicles/multirotor.py` unless noted. The batched PyTorch twin
(`BatchedMultirotor`, same file) implements identical math; NumPy is canonical (see P1.8).

## 1.1 State and rigid-body equations of motion ✅

State: `{x(3), v(3), q(4), ω(3), v_wind(3), Ω(num_rotors)}` (lines 540–575).

```
ẋ = v                                                       (:284)
v̇ = ( [0,0,−mg] + R·F_B ) / m                               (:296,306)
q̇ = ½ Gᵀ(q) ω          (quaternion kinematics)              (:24–41,287)
ω̇ = I⁻¹ ( M_B − ω × (I ω) )      (Euler's equation)         (:308–311)
v̇_wind = 0   (wind overwritten externally each step by wind profile)   (:313–315)
```

with, for `q = [q_x,q_y,q_z,q_w] = [q0,q1,q2,q3]`:

```
        ⎡  q3   q2  −q1  −q0 ⎤
G(q) =  ⎢ −q2   q3   q0  −q1 ⎥        q̇ = ½ Gᵀ ω
        ⎣  q1  −q0   q3  −q2 ⎦
```

**Verified:** expanding row 1 gives `q̇_x = ½(q_w·p − q_z·q + q_y·r)`, matching the standard
xyzw quaternion derivative (Graf). Renormalization after each integration step (:254).
Euler equation sign/order verified. Full inertia matrix incl. products of inertia supported
(:148–150).

Note (batched): `quat_dot_torch` (:44–66) adds a unit-norm penalty `−(‖q‖²−1)·2q` to `q̇` —
a Baumgarte-style constraint stabilization absent from the NumPy path. Intentional divergence;
harmless, but cross-sim tests must compare post-normalization states.

## 1.2 Motor model ✅

```
Ω̇_i = (Ω_c,i − Ω_i) / τ_m                                   (:281)
Ω_c,i clipped to [Ω_min, Ω_max] before integration          (:211,234)
Ω_i  clipped to [Ω_min, Ω_max] after each step              (:262)
measured Ω_i = Ω_i + N(0, σ_motor²)   (σ_motor default 0)   (:260–262)
```

First-order lag; rotor speeds are integrated state. `τ_m`: crazyflie 0.072 s, brushless 0.050 s,
hummingbird 0.005 s. Standard identified motor model.

## 1.3 Rotor thrust, yaw torque, control allocation ✅

```
T_i   = k_η · Ω_i² · ẑ                    (thrust, body z)   (:341)
M_yaw = Σ_i σ_i · k_m · Ω_i² · ẑ          (yaw)              (:362)
M_force = Σ_i r_i × (T_i + H_i)           (thrust/drag moments)  (:361)
```

**⚠ Convention (critical for ports):** `rotor_directions` (σ_i) is defined as **the sign of the
yaw torque the rotor produces** (crazyflie_params.py:37 "direction of the torque"), i.e. the
*negative* of the physical spin direction: `spin_i = −σ_i`. Any added spin-dependent physics
(gyroscopic terms, A5) must use `spin_i = −σ_i`, not σ_i.

**Verified:** `M_force = −einsum('ijk,ik->j', hat(r), T+H)` (:361) looks sign-flipped but is
correct — `hat_map` on an (n,3) array returns shape (3,3,n) (:531–534), so the einsum contracts
`Σ_i hat(r_k)[i,j]·F_k[i] = (hat(r_k)ᵀF_k)[j] = −(r_k×F_k)[j]`; the leading minus restores
`+Σ r_k × F_k = Σ r × F`. ✓

Allocation (control side): `f_to_TM = [1ᵀ; (r_i×ẑ)_{x,y}; (k_m/k_η)·σᵀ]`, `TM_to_f` its inverse
(:167–173); speeds from forces via `sign(f)·√|f/k_η|` (:479–480).

## 1.4 Aerodynamics ✅ (source: RotorPy paper §III; Mahony et al. 2012)

Enabled by `aero=True`. Per-rotor local airspeed includes rotation lever arm:

```
v_i = v_a + ω × r_i                                          (:338)
```

| Effect | Equation | Acts at | Line |
|---|---|---|---|
| Parasitic drag | `D = −‖v_a‖ · diag(c_Dx,c_Dy,c_Dz) · v_a` | CoM | :346 |
| Rotor drag (H-force) | `H_i = −Ω_i · diag(k_d, k_d, k_z) · v_i` | hub i | :348 |
| Blade-flapping moment | `M_flap,i = −k_flap · Ω_i · (v_i × ẑ)` | hub i | :351 |
| Translational lift | `T_i += k_h · (v_i,x² + v_i,y²) · ẑ` | hub i | :352–353 |

Totals: `F_B = Σ(T_i + H_i) + D`;  `M_B = Σ r_i×(T_i+H_i) + M_yaw + Σ M_flap,i` (:365–366).

**Verified:** `hat(v)·ẑ = v×ẑ = (v_y, −v_x, 0)`, so forward flight (`v_x>0`) gives
`M_flap,y = +k_flap·Ω·v_x` → pitch-up, the correct rotor-flapping response (Mahony et al. §III).
H-force linear in `Ω·v` matches the induced-drag model (k_z = induced inflow on the rotor axis).
Note SkyDreamer's drag `−k_x·v_x·ΣΩ − k_x2·v_x|v_x|` is the same physics as {k_d summed over
rotors} + {c_Dx}; do not double-count when porting parameter values.

**Explicitly not modeled** (README + code): ground effect, downwash/rotor-rotor interaction,
frame lift, rotor inertia (→ A5), thrust variation with airspeed beyond k_h/k_z (→ B1).

## 1.5 Ground contact (heuristic, not physics-grade) 🔶

Optional (`enable_ground`). Normal force cancels net downward force at z≤0 (:299–303); post-step:
clamp z=0, v_z≥0, horizontal velocity damping `v_xy ← (1−β)·v_xy` (β∈[0.1,0.5]), zero body rates,
flatten roll/pitch (:484–513). Adequate for takeoff/landing bookkeeping; do **not** port as
contact physics.

## 1.6 Sensor models

**IMU** (`rotorpy/sensors/imu.py`) — intended measurement model, for a sensor at body offset
`p_BS`, mounting rotation `R_BS`, world gravity `g_W = [0,0,−g]`:

```
a_S,W  = a_B,W + R_WB·( ω̇_B × p_BS + ω_B × (ω_B × p_BS) )        (rigid-body point acceleration)
accel  = R_BSᵀ · R_WBᵀ · (a_S,W − g_W)  + b_a + η_a               (specific force)
gyro   = R_BSᵀ · ω_B                    + b_g + η_g
η ~ N(0, (noise_density)²·f_s/2),   ḃ ~ random walk (bias_step)
```

⚠️ **Two frame-handling defects found in the implementation** (imu.py:100–128) — see Part III
findings F-1, F-2. The equations above are the correct target; the defects are invisible in the
default config (`p_BS = 0`, `R_BS = I`).

**Motion capture** (`sensors/external_mocap.py`) ✅: pose+twist with per-channel Gaussian noise;
attitude noise applied as SO(3) perturbation quaternion; optional artifact spikes on v/ω.

**Motor speed measurement**: Gaussian noise on Ω (P1.2).

## 1.7 Wind models (`rotorpy/wind/`) ✅

Interface: `v_wind = f(t, x)` world frame, injected into the state each step. `NoWind`,
`ConstantWind`, `SinusoidWind` (per-axis `A·sin(2πft + φ)`), `LadderWind` (steps), `WindTunnel`
(spatial cylinder), `DrydenGust` — MIL-HDBK-1797 Dryden turbulence spectra via the
`wind-dynamics` package (published standard model).

## 1.8 Integration ✅

NumPy: `scipy.solve_ivp`, default adaptive RK45, configurable (:183–186,242–247). Batched:
`torchdiffeq.odeint`, dopri5 or fixed-step rk4. Sim rate default 100 Hz. Post-step quaternion
renormalization both paths. (For firmware port: fixed-step RK4 is the reference-matching choice;
SkyDreamer used RK4 @ 2.2 ms.)

## 1.9 Existing domain-randomization machinery (data plumbing, not physics)

`learning_utils.py`: samples/applies mass, k_η, k_m, inertia, τ_m, motor noise;
`BatchedMultirotorParams.update_*` also covers drag coefficients. Default ranges only cover
mass + k_η → extended by Part II-D.

---

# Part II — Additions (external sources, math verified)

## A. Actuator / motor models

### A1. Per-rotor coefficient asymmetry ✅(source-verified)
**Source:** SkyDreamer Table II — individual `k_p1..4`, `k_q1..4`, `k_r1..8` per motor (values
differ motor-to-motor ~25%); implementation lines 282–285 confirm per-rotor use.
**Math:** promote `k_η`, `k_m` (optionally `k_d`, `k_z`) to per-rotor values:
```
T_i = k_η,i · Ω_i² · ẑ        M_yaw = Σ σ_i · k_m,i · Ω_i² · ẑ
```
**⚠ Porting note (verified against their code):** SkyDreamer's dynamics outputs *accelerations
directly* — `v̇ = R·(D_x,D_y,T) − g·ẑ` with **no mass division** (their lines 291–293) and
`ω̇ = (M_x,M_y,M_z)` with **no inertia inverse** (lines 302–304). Their coefficients are therefore
mass-normalized (k_w: m/s² per (rad/s)²) and inertia-normalized (k_p: rad/s² per (rad/s)²).
Porting into RotorPy's force/torque form requires `k_η = m·k_w`, `k_p^RotorPy = I_xx·k_p^SkyD`,
etc. Their lumped `J_x·q·r` terms are RotorPy's `−ω×Iω` (already present — sign structure
verified: `J_x=(I_yy−I_zz)/I_xx<0`, `J_y=(I_zz−I_xx)/I_yy>0` ✓); do not add twice.
**RotorPy site:** wrench (:341,362) + allocation `TM_to_f`; defaults identical per rotor.

> **STATUS (implemented 2026-07-07):** `Multirotor` (NumPy, canonical) now stores `k_eta`, `k_m`
> as per-rotor arrays of shape (num_rotors,); a scalar in the params dict is broadcast, so every
> existing parameter file is unchanged. Thrust, yaw moment, the `k = k_m/k_eta` allocation ratio,
> and the force→speed inversion are all element-wise per rotor. Verified by
> `scratchpad/tests/test_per_rotor_coeffs.py`: scalar-broadcast reproduces hover exactly;
> asymmetric k_eta matches a hand-computed `Σ r_i×T_i`; cmd_ctbm allocation round-trip recovers
> commanded thrust+moment to ~1e-18 with per-rotor k_eta and k_m. Repo tests
> (test_multirotor/sensors/env/winds/batched_sims) pass.
> - **Deferred (batched):** `BatchedMultirotorParams` keeps one k_eta/k_m per drone (broadcast
>   over rotors) — per-rotor arrays in the batched path are a follow-up; NumPy is canonical.

### A2. Polynomial thrust/torque curves ✅(source-verified)
**Source:** Crazyflow `first_principles/dynamics.py:121,129`; identified in `params.toml`.
```
T_i = c₀ + c₁·Ω_i + c₂·Ω_i²          τ_yaw,i = d₀ + d₁·Ω_i + d₂·Ω_i²
Ω_i(T) = ( −c₁ + √(c₁² − 4c₂(c₀ − T_i)) ) / (2c₂)      (allocation inverse, transform.py)
```
Identified (Crazyflie, **RPM units** — convert: `c₁^rad/s = c₁·60/2π`, `c₂^rad/s = c₂·(60/2π)²`):
cf2x_L250 thrust `[0, −5.382e-7, 2.458e-10]`, torque `[0, 1.410e-9, 1.459e-12]`; 3 more variants
in params.toml. Quadratic-formula inversion verified (positive root ✓).
**RotorPy site:** params `k_eta → [c₀,c₁,c₂]` default `[0,0,k_η]`; wrench + inversions :384/:480.

> **STATUS (implemented 2026-07-07):** Forward thrust `thrust_c0 + thrust_c1·Ω + k_eta·Ω²` and
> yaw `dir·(torque_c0 + torque_c1·Ω + k_m·Ω²)` in `compute_body_wrench` (k_eta/k_m are the
> quadratic c2/d2; new optional params `thrust_c0/c1`, `torque_c0/c1`, per-rotor or scalar,
> default 0). Inversion refactored into `_thrust_to_speed`: exact `sign·√(f/k_eta)` when no poly
> terms (regression), quadratic-formula root (Crazyflow) when active; used by `cmd_motor_thrusts`
> and the SE3 allocation. Verified by `scratchpad/tests/test_poly_thrust.py`: regression, poly
> forward vs hand calc, exact w→f→w round-trip (4e-13) on the physical branch, and **exact
> Crazyflow parity** (cf2x_L250 rpm2thrust converted RPM→rad/s, err ~1e-18). **Caveat:** the linear
> allocator TM_to_f assumes quadratic-dominant; with a polynomial active the thrust/moment split
> is approximate (forward dynamics exact) — documented at the allocation site.

### A3. Nonlinear command→steady-state speed curve ✅(source-verified)
**Source:** SkyDreamer paper Eq. (motor model) + impl lines 253–256.
```
Ω_c = (Ω_max − Ω_min) · √( k·u² + (1−k)·u ) + Ω_min ,    u ∈ [0,1],  k ∈ [0,1]
```
Verified: u=0 → Ω_min, u=1 → Ω_max, monotone on [0,1] for k∈[0,1] ✓. Identified: k = 0.50,
Ω_min = 341.75, Ω_max = 3100 rad/s (5" racer). Note their sim integrates a *normalized* rotor
state over a fixed [0, 3000] rad/s range (lines 243–246, 306–309) — an implementation detail,
not physics; port the physical form above.
**RotorPy site:** new control abstraction in `get_cmd_motor_speeds`.

> **STATUS (implemented 2026-07-07):** New `cmd_motor_throttle` control abstraction in
> `get_cmd_motor_speeds`; command `u∈[0,1]` per rotor mapped via the curve using
> `rotor_speed_min/max` and new param `motor_curve_k` (default 1.0 = linear speed map). Verified
> by `scratchpad/tests/test_throttle_curve.py`: endpoints, monotonicity, k=1 linear / k=0 sqrt,
> exact match to the SkyDreamer reference at k=0.5 (w_min=341.75, w_max=3100), out-of-range
> clipping, and integration to the commanded steady-state speed.
> - **Deferred:** command noise ε_u (from C1) is the natural companion here (perturb `u` pre-curve);
>   not yet added.

### A4. Asymmetric rotor spin-up/spin-down dynamics ✅(source-verified)
**Source:** Crazyflow `first_principles/dynamics.py:115–119` + params.toml.
```
Ω̇ = k_a1·(Ω_c − Ω) + k_a2·(Ω_c² − Ω²)     if Ω_c > Ω    (spin-up)
Ω̇ = k_d1·(Ω_c − Ω) + k_d2·(Ω_c² − Ω²)     otherwise     (spin-down)
```
Reduces to P1.2 with `k_a1 = k_d1 = 1/τ_m`, `k_2 = 0`. Consistency check (cf2x_L250, RPM units,
coefs `[7.356, 0, 0, 2.444e-4]`): near hover (~10 kRPM) the quadratic term contributes
`k₂·(Ω_c+Ω) ≈ 4.9 s⁻¹`, total effective rate ≈ 12 s⁻¹ → τ ≈ 0.08 s, agreeing with RotorPy's
identified crazyflie `τ_m = 0.072 s` ✓. cf21B_500 spin-up ≈ 2.4× faster than spin-down.
**RotorPy site:** :281 (+batched :902); 4 optional coefficients, `τ_m` fallback.

> **STATUS (implemented 2026-07-07):** `Multirotor._rotor_accel(cmd, Omega)` implements the
> asymmetric model, routed from `_s_dot_fn`. New optional param `rotor_dyn_coef = [ka1,ka2,kd1,kd2]`
> (rad/s units); absent → first-order `1/tau_m` (exact recovery via `[1/τ,0,1/τ,0]`). Verified by
> `scratchpad/tests/test_motor_dynamics.py`: regression, first-order reduction, up/down asymmetry,
> exact Crazyflow `where()` formula match, and integration (cf21B_500: step-up settles 0.158 s vs
> step-down 0.328 s). **Porting note:** Crazyflow coefficients are in RPM units — convert the
> quadratic coefs by (60/2π)² and linear by (60/2π) to rad/s.
> - **Deferred (batched):** batched motor model still first-order (`τ_m`); default-equivalent.

### A5. Rotor inertia: reaction torque + gyroscopic precession ✅(derivation) / ⚠(Crazyflow sign)
**Sources:** Crazyflow `first_principles/dynamics.py:142–149` (identified `prop_inertia`);
SkyDreamer `k_r5..8·Ω̇` yaw terms (reaction part, identified); independent derivation below.
Genesis implements **neither** (verified).
**Math.** Let `s_i = spin direction = −σ_i` under RotorPy's torque-sign convention (P1.3!).
Rotor angular momentum `h = I_r · (Σ_i s_i Ω_i) · ẑ`:
```
τ_reaction = −I_r · Σ_i s_i·Ω̇_i · ẑ  =  +I_r · Σ_i σ_i·Ω̇_i · ẑ          (yaw)
τ_gyro     = −ω × h  =  I_r·(Σ_i s_i Ω_i) · ( −q, +p, 0 )ᵀ               (roll/pitch)
```
Derivation of τ_gyro: `ω×h = (p,q,r)×(0,0,h_z) = (q·h_z, −p·h_z, 0)`; torque on body `= −ω×h
= (−q·h_z, +p·h_z, 0)`. **The x and y components must have opposite signs.**
**⚠ Crazyflow discrepancy (report upstream to them):** their implementation uses the same sign
on both components — `I_p·(−q·S, −p·S, ·)` with `S = Σ mix_z,i·Ω_i` (accessor `torque_inertia`).
Whichever spin convention their `mixing_matrix` z-row encodes, exactly one of the two components
is flipped. Their z (reaction) term is consistent with their yaw-drag convention ✓.
**Identified `I_r`:** 34.52e-9 (cf2x_L250), 26.97e-9 (P250), 38.93e-9 (T350/B500) kg·m²;
5" racer ≈ 5–8e-6 kg·m² (estimate from prop mass/geometry). Magnitude: vanishes for balanced
counter-rotating pairs; ≈5–10% of available roll/pitch moment during hard yaw + high body rate.
**RotorPy site:** new `I_rotor` param; two terms in `compute_body_wrench` (`Ω̇` available from the
motor model).

> **STATUS (implemented 2026-07-07):** `Multirotor.rotor_inertia_moment(body_rates, rotor_speeds,
> rotor_accel)` adds gyroscopic precession + reaction torque, called from `_s_dot_fn` (where Ω̇ is
> already computed). New param `rotor_inertia` (kg·m², default 0 → term off). Uses physical spin
> `spin = −rotor_dir` per F-6. Verified by `scratchpad/tests/test_rotor_inertia.py`: gyro term
> equals an independent `−ω×h` to 0.0; reaction `−I_r Σ spinΩ̇` exact; balanced-quad invariance;
> regression-zero at default. **F-3 confirmed against Crazyflow's real code** (see
> `test_xsim_crazyflow.py`): our gyro **x**-component is opposite in sign to theirs (the initial
> review mislabeled it as y); reaction(z) and gyro-y agree exactly. Flag in upstream PR + report
> to Crazyflow.
> - **Deferred (batched):** term not added to `BatchedMultirotor` yet (no param file sets
>   rotor_inertia, so batched≡NumPy at default; batched_sims equivalence test still passes).

### A6. Motor command latency ✅(paper)
**Source:** SkyDreamer paper (11 ms action delay in training; not in their repo).
```
u_applied(t) = u_cmd(t − t_d)        (ring buffer of round(t_d/dt) steps, default t_d = 0)
```
**RotorPy site:** `step()` before the motor model, both paths.

> **STATUS (implemented 2026-07-07):** `Multirotor._apply_command_delay(control, t_step)` delays
> the whole control dict by `round(motor_delay_time/t_step)` steps via a `deque(maxlen=n+1)`,
> called at the top of `step()`. New param `motor_delay_time` (s, default 0 → no delay, buffer
> never created). Verified by `scratchpad/tests/test_command_delay.py`: delay-line returns
> `cmd[k−n]` exactly, constant-command invariance, step-change held for n steps then responds,
> regression at delay 0. **Caveat:** applied in `step()` only; `statedot()` (IMU diagnostic) uses
> the undelayed command, so IMU accel is slightly inconsistent under nonzero delay — a wiring
> detail for the firmware port, noted here.

### A7. PWM quantization + supply-voltage layer ✅(source-verified)
**Source:** Crazyflow params.toml (`pwm_min=7000, pwm_max=65535`, `vmotor2*` polynomials),
`control/transform.py`.
```
u_q = round(u·(pwm_max−pwm_min)) / (pwm_max−pwm_min)             (quantization, optional)
T(V) = a₀ + a₁V + a₂V² + a₃V³        (voltage→thrust, per motor)
Ω(V) = b₀ + b₁V                      (voltage→RPM)
V(t) = V₀ − R_int·i(t) − k_sag·∫i dt   (sag; minimal variant: Ω_max(t) = Ω_max,0·(1−c·t))
```
Identified: cf2x_L250 `vmotor2thrust = [−0.01483, 0.04724, −0.01847, 0.005961]`,
`vmotor2rpm = [2968.18, 6647.95]`; all four variants in params.toml. Supporting field data:
SkyDreamer measured Ω_max 3200→2200 rad/s (−30%) over one battery.
**RotorPy site:** optional battery state modulating Ω_max / thrust curve; quantization in the
command path.

> **STATUS (implemented 2026-07-07):** (a) **PWM quantization** in `cmd_motor_throttle`: when
> `pwm_max > pwm_min`, throttle `u` snaps to the integer grid `round(u·(pwm_max−pwm_min))/…`
> (params `pwm_min`, `pwm_max`, default 0/off). (b) **Battery sag hook**: vehicle attribute
> `rotor_speed_max_scale` (default 1.0) scales the achievable max in `step()`'s clips.
> (c) **Battery models** in `rotorpy/battery.py`: `NoBatterySag`, `LinearBatterySag(drop_frac,
> duration)`, `VoltageBatterySag(vmotor2rpm, V0, discharge_rate)` (Crazyflow voltage→RPM). Verified
> by `scratchpad/tests/test_pwm_battery.py`: regression, grid snapping, speed cap under sag,
> model trajectories. **Deferred:** voltage→thrust-curve modulation and `simulate()` wiring of a
> `battery_profile` (mirrors the `disturbance_profile` pattern) — the hook + models are in place.

## B. Aerodynamics

### B1. Thrust vs. rotor angle-of-attack and advance ratio ✅(source-verified)
**Source:** SkyDreamer paper + impl lines 274–280; identified to ~racing speeds.
```
ω̄ = mean(Ω_i)
α  = atan2( v_a,z ,  r_prop·ω̄ )                    (rotor angle of attack)
μ  = atan2( ‖v_a,xy‖ ,  r_prop·ω̄ )                 (advance ratio angle)
T_total = k_w · (1 + k_angle·α + k_hor·μ) · Σ Ω_i²  −  k_v2 · v_a,z·|v_a,z|
```
Identified (**mass-normalized** — multiply by m when porting, see A1 note): `k_w = 1.55e-6`,
`k_angle = 3.145`, `k_hor = 7.245`, `k_v2 = 0`, `r_prop = 0.0635 m`.
Relation to P1.4: `k_z`/`k_h` are the small-airspeed linearizations of this multiplicative
correction; a vehicle should use either {k_angle,k_hor} or {k_h}, not both, to avoid
double-counting the in-plane effect. Gate behind `k_angle = k_hor = 0` defaults.
**RotorPy site:** `compute_body_wrench`; needs `r_prop` param.

> **STATUS (implemented 2026-07-07):** In `compute_body_wrench` (aero block): base rotor thrust
> multiplied by `(1 + k_angle·α + k_hor·μ)` with `α = atan2(v_a,z, r_prop·ω̄)`,
> `μ = atan2(‖v_a,xy‖, r_prop·ω̄)`; collective `−k_v2·v_a,z·|v_a,z|` added at CoM. New params
> `k_angle`, `k_hor`, `k_v2`, `r_prop` (falls back to `rotor_radius`), all default 0/inert.
> Applied before translational lift so it scales only the base thrust. Verified by
> `scratchpad/tests/test_aoa_thrust.py`: regression, zero-airspeed invariance, hand-calc match,
> and **exact** reproduction of SkyDreamer's `k_w·(1+k_angle·α+k_hor·μ)·ΣΩ²` (err 0). Remember
> F-4: SkyDreamer k_w is mass-normalized (`k_eta = m·k_w` when porting).
>
> **Double-count guard (added 2026-07-07):** `k_h` (translational lift) and `k_hor` both raise
> thrust with in-plane airspeed (k_angle is the vertical companion of the same model), so enabling
> `k_h` together with `k_angle` or `k_hor` double-counts. `Multirotor.__init__` now **raises
> ValueError** for that combination. Verified by `scratchpad/tests/test_aoa_kh_guard.py`: the
> double count is first demonstrated (both active → horizontal-airspeed thrust increment equals the
> exact sum of each model's increment, `dT_both = dT_kh + dT_hor`, both nonzero), then the fix is
> confirmed (construction raises for all three conflicting combos; single-model/neither still
> build). No shipped param file sets k_angle/k_hor, so no existing vehicle is affected.

## C. Stochastic disturbance models

### C1. Two-band wrench disturbances + command noise ✅(paper values; Crazyflow pattern)
**Source:** SkyDreamer paper Table III (not in their repo); integration pattern = Crazyflow's
first-class `dist_f`/`dist_t` inputs (`first_principles/dynamics.py:154–162`).
```
v̇ += ε_a(t)            (specific force, m/s² — for force-form multiply by m; fix & document frame)
ω̇ += ε_M(t)            (Table III specifies angular acceleration, rad/s² — torque-form: I·ε_M)
u_i ← clip(u_i + ε_u,i(t), 0, 1)
```
Piecewise-constant, resampled per channel:

| Channel | Rate | Training | Eval |
|---|---|---|---|
| ε_a | 1 Hz | ±3 m/s² | ±2 m/s² |
| ε_M | 1 Hz | ±3 rad/s² | ±2 rad/s² |
| ε_M | 90 Hz | ±125 rad/s² | ±100 rad/s² |
| ε_u | 90 Hz | ±0.2 | — |

**RotorPy site:** a `wrench_profile` channel parallel to `wind_profile` (keeps `Multirotor`
deterministic). Default off.

> **STATUS (implemented 2026-07-07):** Wrench disturbance (ε_a, ε_M low+high bands) implemented.
> - `rotorpy/disturbances/` package: `disturbance_template.py`, `default_disturbances.py`
>   (`NoDisturbance`, `WrenchDisturbance`). Generator is specified in Table III acceleration
>   units and converts via mass/inertia (`force = m·ε_a` world frame; `torque = I·(ε_M,lf+ε_M,hf)`
>   body frame), resample-and-hold per band, seeded RNG.
> - `Multirotor`/`BatchedMultirotor`: `external_force` (world N) + `external_torque` (body N·m)
>   attributes, added in `_s_dot_fn`; zero by default. Wired through `simulate()` and
>   `Environment` (optional `disturbance_profile`, defaults to `NoDisturbance`).
> - **Deferred:** command noise ε_u lives in normalized command space [0,1]; it is a natural
>   companion to the A3 throttle-curve abstraction and will be added there, not as a wrench.
> - Verified by `scratchpad/tests/test_wrench_disturbance.py`: regression-zero hover, exact
>   `F/m` and `I⁻¹M` injection, generator range bounds, resample-and-hold timing, seed
>   determinism; plus an integration check (0.02 N up → `vz = F/m·t`).

### C2. Slow parameter drift (battery sag)
Covered by A7; minimal variant `Ω_max(t) = Ω_max,0·(1 − c·t)`, c ≈ 30% per battery duration
(SkyDreamer field data). **Implemented — see A7 status (`rotorpy/battery.py`).**

## D. Domain-randomization ranges (data, not a feature)

Extend `learning_utils.py` defaults to all physical parameters, per-episode:

| Parameter group | Range (SkyDreamer Table III, training) |
|---|---|
| Ω_min, Ω_max | ±20% |
| mass, inertia, thrust/torque/drag coefficients, τ_m (or A4 coefs), curve shape k, r_prop, I_rotor, per-rotor k_η,i/k_m,i | ±30% |
| Eval (if distinguished) | ±20% |

Keep the existing hover-feasibility bound on k_η (learning_utils.py:40–53).

> **STATUS (implemented 2026-07-07):** `learning_utils.percent_randomization(nominal_params, pct=0.30,
> motor_limit_pct=0.20)` builds a full ranges dict (±30% dynamics, ±20% motor limits), skipping
> zero/absent params. `update_vehicle_params` extended to apply drag (`c_Dx/c_Dy/c_Dz/k_d/k_z` via
> `update_drag`) and `rotor_speed_min/max`, sampling the speed limits first so the hover-feasibility
> clamp uses the randomized max. **Robustness fix:** the k_eta feasibility clamp now raises the band
> *upper* bound too when `min_k_eta` exceeds it (previously the uniform range could invert and yield
> infeasible k_eta once rotor_speed_max is randomized low). Verified by
> `scratchpad/tests/test_domain_randomization.py`: band construction, in-bounds sampling, **every
> sampled drone hovers** (min thrust/weight margin 1.44), default dict unchanged.
> - **Deferred (batched):** ranges cover only params with batched `update_*` setters; per-rotor
>   arrays, r_prop, rotor_inertia, curve k, A4 coefs need batched param plumbing first (NumPy is
>   canonical).

## E. Out of scope for RotorPy (tracked so nothing is lost)

Camera models (extrinsics DR, rolling shutter, mask erosion, StochGAN); learning plumbing
(image delay, rewards, privileged info); firmware controller emulation (Crazyflow Mellinger
port); MJX contact/ray rendering; CasADi twins.

## E2. Control-rate decoupling (approved + implemented 2026-07-07, out of original physics scope)

Not physics (a harness/timing feature), but wanted for training realism: the policy/controller
decides at a lower rate than physics runs, holding the command zero-order between decisions
(Crazyflow's `controllable` gate). Distinct from A6 (a fixed command *transport delay*); this is a
ZOH *staircase* between control updates.

> **STATUS (implemented 2026-07-07):**
> - Gym env ([quadrotor_environments.py](../rotorpy/learning/quadrotor_environments.py)): new
>   `control_rate` arg; `n_substep = round(sim_rate/control_rate)`; `step()` loops the physics
>   `n_substep` times holding the action (ZOH), builds the observation at the decision boundary,
>   advances `t` by `n_substep·t_step`. Raises if `control_rate > sim_rate`.
> - `simulate.py` + `Environment`: `control_rate` gate — `control_decimation = round((1/t_step)/
>   control_rate)`; the controller updates only on decimation boundaries, else the previous command
>   is held.
> - Opt-in, default (`control_rate = None → = sim_rate`) reproduces the prior behavior exactly.
> Verified by `scratchpad/tests/test_control_rate.py`: default trajectory identical; one decoupled
> step (n=4) equals 4 held physics steps to 0.0; guard fires; `simulate()` holds the command at the
> 1/decimation rate. A separate observation/IMU rate remains a possible future extension.

---

# Part III — Verification findings & test protocol

## Findings from this review (2026-07-07)

**F-1 ⚠ RotorPy IMU lever-arm frame mixing** (`sensors/imu.py:110–111`): the point-acceleration
term computes `ω̇×p_BS,W` and `ω×(ω×p_BS,W)` using **body-frame** `ω`, `ω̇` (as produced by
`statedot` — the Euler equation output is body-frame) crossed with the **world-frame** lever arm
`p_BS,W = R·p_BS`. Cross products must be evaluated in one frame:
`a_S,W = a_B,W + R·(ω̇_B×p_BS + ω_B×(ω_B×p_BS))`. No effect when `p_BS = 0` (default).

**F-2 ⚠ RotorPy IMU gyro ignores mounting rotation** (`sensors/imu.py:117`): accelerometer
applies `R_BSᵀ` but the gyroscope returns body-frame `ω` directly; should be `R_BSᵀ·ω_B`.
No effect when `R_BS = I` (default).

**F-2b ⚠ Batched IMU accel rotation was element-wise, not matmul** (`sensors/imu.py`
BatchedImu): `R_SW` was built with `einsum('ij,bij->bij', R_BS.T, R_BW)`, a Hadamard product,
not the matrix product `R_BS.T @ R_BW`. Wrong for any non-level attitude even with `R_BS = I`;
the `__main__` demo only exercised the level case so it went unnoticed. Discovered while
porting the F-1/F-2 fixes to the batched path.

> **STATUS (implemented 2026-07-07):** F-1, F-2, F-2b fixed in `rotorpy/sensors/imu.py` (both
> `Imu` and `BatchedImu`). Verified by `scratchpad/tests/test_imu_fixes.py`: NumPy and batched
> paths now match an independent closed-form IMU model to < 4e-15 across rotated / offset /
> mounted cases, and the default config (p_BS=0, R_BS=I) reduces exactly to `Rᵀ(v̇ − g)`, `ω`.

**F-3 ⚠ Crazyflow gyroscopic precession sign** (`first_principles/dynamics.py:142–149`): their
`torque_inertia` writes x and y with the *same* leading sign
(`x = −I_r·q·S`, `y = −I_r·p·S`, `S = Σ mix_z·Ω`), but `τ = −ω×h` requires *opposite* signs.
**VERIFIED against their real running code (2026-07-07, `scratchpad/tests/test_xsim_crazyflow.py`):**
with an identically-configured drone, reaction(z) and gyro-y agree exactly, and gyro-**x** is a
pure sign flip (`+7.59e-6` vs RotorPy `−7.59e-6`). Hand-check confirms RotorPy equals `−ω×h`, so
Crazyflow's **x**-component has the wrong sign.
> **Correction:** the initial review guessed the flipped axis was *y*; running the actual code
> shows it is *x*. (The identity of which axis depends on the p/q ordering in their expression,
> which is why the numeric cross-check was needed.) Report to Crazyflow as an x-axis gyro sign bug.

**F-4 ⚠ SkyDreamer coefficients are mass-/inertia-normalized**: their dynamics emits
accelerations with no `1/m` or `I⁻¹` (impl lines 291–304). All Table II force coefficients are
per-unit-mass, all moment coefficients per-unit-inertia. Scale when porting (A1/B1 notes).

**F-5 ✅ RotorPy core dynamics verified correct**, including the two subtle spots: the
`M_force` einsum minus sign (compensates the `hat_map` (3,3,n) layout — net `+Σ r×F`, P1.3) and
the quaternion `G` matrix (matches Graf xyzw convention, P1.1). Flapping-moment sign gives
pitch-up in forward flight ✓. `−ω×Iω` ✓.

**F-6 ⚠ Convention trap:** RotorPy `rotor_directions` = yaw-**torque** sign = −(spin sign).
Every added spin-dependent term (A5) must use `spin = −rotor_dir`.

**F-7** Genesis reviewed and excluded: no rotor physics beyond `KF·rpm²`/`KM·rpm²` on fixed
joints; the "gyroscopic effects" attribution is unfounded.

**F-8 ⚠ Batched dynamics ran in float32 (precision)** (`BatchedMultirotor._s_dot_fn`): the state
derivative buffer was `torch.zeros(..., device=...)` (default float32), truncating every derivative
to float32 before integration — the dominant source of the batched-vs-NumPy gap (why the repo's
own test tolerates 5e-2/1.0). Large-magnitude rows (rotor accelerations ~1e4 rad/s²) lost ~1e-4
absolute precision. Also latent: `torch.tensor(pylist).double()` in param construction rounds to
float32 *before* upcasting (e.g. `1/0.072 → 13.88888931`). Both fixed during batched parity work.
> **STATUS:** `s_dot` buffer now `dtype=torch.double`; new params built with `dtype=torch.double`
> at creation. Batched↔NumPy state-derivative agreement improved from ~1e-4 to ~6e-7 (float32
> floor still present elsewhere in the pre-existing batched path, e.g. `tau_m`/`k_flap` tensors;
> not chased). Verified by `scratchpad/tests/test_batched_parity.py`.

## Test protocol (run everything in RotorPy before any port)

1. **Regression-zero**: every addition defaults off/neutral; with defaults, trajectories match
   current RotorPy bit-for-bit (fixed-seed golden trajectories, both NumPy and batched).
2. **NumPy ↔ batched equivalence**: same inputs → same derivatives to float64 tolerance
   (compare post-quaternion-normalization; see P1.1 batched note).
3. **Cross-simulator, SkyDreamer**: configure RotorPy with Table II parameters (applying F-4
   scaling and the P1.4 drag-convention mapping); match their Numba sim's *state derivatives*
   at sampled states to tolerance, then closed-loop trajectories under identical Euler @ 100 Hz.
4. **Cross-simulator, Crazyflow**: params.toml values (RPM→rad/s conversions per A2); match
   `first_principles` derivatives at sampled states — expect disagreement only in the A5
   precession component per F-3 (document the delta; our sign wins per derivation).
5. **Limit checks**: A5 terms vanish for balanced counter-rotating speeds; A3 endpoints
   Ω_min/Ω_max; A4 reduces to first-order when k₂=0; B1 reduces to k_w·ΣΩ² at zero airspeed;
   C1 off → deterministic.
6. **Energy sanity**: drag/H-force/flapping terms strictly dissipative for v_wind = 0
   (power `F_aero·v_a + M_aero·ω ≤ 0` sampled over random states).
7. **Fix F-1/F-2** (small, isolated) and add offset/rotated-IMU unit tests against the
   closed-form point-acceleration formula.
8. **Upstream PRs** split: (i) IMU fixes F-1/F-2, (ii) actuator models A1–A7, (iii) aero B1,
   (iv) disturbances C1, (v) DR ranges D — each with the derivation notes above.

## Implementation order

**F-1/F-2 fixes → C1 → A1 → A5 → A3 → A6 → A2 → A4 → B1 → A7/C2 → D** (payoff ÷ effort,
fixes first because they're bugs). **All done + unit-verified.** Then cross-sim validation (below).

---

# Part IV — Cross-simulator validation results (2026-07-07)

Beyond the per-item unit tests (which check against published *formulas*), we ran RotorPy against
the reference simulators' **actual running code**. Both reference sims were executed in the local
venv: Crazyflow's `first_principles.dynamics` imported via a bare-package shim (bypassing its
`mujoco.mjx` package init), and SkyDreamer's `compute_dynamics_jit` exec'd verbatim from their
source with `NUMBA_DISABLE_JIT=1`. Tests: `scratchpad/tests/test_xsim_crazyflow.py`,
`test_xsim_skydreamer.py`.

### Crazyflow (`test_xsim_crazyflow.py`) — cf2x_L250 params
| Component | Result |
|---|---|
| A2 polynomial thrust | **exact** (err 0) — RotorPy body-z thrust == Crazyflow motor-thrust sum (coeffs RPM→rad/s) |
| A4 rotor spin-up/down | **exact** (err 0) — `_rotor_accel` == Crazyflow `rotor_vel_dot` for mixed up/down |
| A5 prop-inertia torque | reaction(z) + gyro-y **exact**; gyro-**x** is a pure sign flip → **F-3 confirmed** |

Isolation method for A5: `τ_inertia = J·(ang_vel_dot[I_r] − ang_vel_dot[0])` from their real
function (cancels thrust/drag/`ω×Jω`). RotorPy's value hand-verified `= −ω×h`; Crazyflow's x has
the opposite sign. **This run corrected F-3's axis label (x, not y).**

### SkyDreamer (`test_xsim_skydreamer.py`) — Table II params, mass-scaled per F-4
| Component | Result |
|---|---|
| B1 thrust (AoA + advance ratio + `k_v2`) | **match** to ~1e-6 (their `+1e-6` atan2 epsilon), with `k_eta=m·k_w` etc. |
| Linear rotor drag | **exact** (err 0) with `k_d = m·k_x` |
| Quadratic drag | **exact on a single airspeed axis**; **differs off-axis** (RotorPy scales parasitic drag by `‖v‖`, SkyDreamer by per-axis `|v|`) — confirmed numerically, a real structural difference |

**F-4 confirmed**: SkyDreamer emits accelerations (no `1/m`, no `I⁻¹`); RotorPy reproduces their
forces only after multiplying coefficients by mass. **Not cross-validated (structural mismatch,
documented):** SkyDreamer's moment model is per-rotor identified `k_p/k_q/k_r` coefficients,
unlike RotorPy's geometry-derived `r×F`; the two are not expected to agree and were not compared.

### Net
Every component we added to mirror a reference reproduces that reference's **real code** exactly
(or to a documented, understood epsilon). The two intentional divergences are both explained:
Crazyflow's gyro-x sign bug (F-3, ours is correct) and the quadratic-drag norm convention
(`‖v‖` vs per-axis). This is the evidence base for the upstream PR.

## Parameter appendix

- SkyDreamer Table II: `SKYDREAMER_PARAMS`, their `skydreamer.py:37–58` (apply F-4 scaling).
- Crazyflow identified sets: `crazyflow/drones/params.toml` (cf2x_L250/P250/T350, cf21B_500) —
  mass, J, thrust/torque polys, rotor_dyn_coef, drag matrices, prop_inertia, PWM/voltage maps
  (RPM units).
- RotorPy vehicles: `rotorpy/vehicles/*_params.py` (crazyflie, crazyflie-brushless, hummingbird,
  px4) — rad/s units, torque-sign rotor_directions convention (F-6).
