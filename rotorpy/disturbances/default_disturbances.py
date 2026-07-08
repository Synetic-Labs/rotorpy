"""
Default disturbance profiles.

These implement the SkyDreamer two-band external-wrench model (arXiv:2510.14783, Table III),
which is the highest-leverage sim-to-real device in that work: rather than modeling every real
effect, inject bounded random accelerations/angular-accelerations so the controller learns to
tolerate model error. The wrench is resampled at a fixed rate and held constant between samples.

The disturbance is specified in acceleration units (m/s^2, rad/s^2) exactly as in Table III and
converted to a force/torque using the vehicle mass and inertia, so the numbers here are directly
comparable to the paper.

    Channel                       Rate     Training range     Eval range
    linear accel   eps_a          1 Hz     +/- 3   m/s^2       +/- 2   m/s^2
    angular accel  eps_M (low)    1 Hz     +/- 3   rad/s^2     +/- 2   rad/s^2
    angular accel  eps_M (high)   90 Hz    +/- 125 rad/s^2     +/- 100 rad/s^2

See rotorpy/disturbances/disturbance_template.py for the interface contract.
"""
import numpy as np

from rotorpy.disturbances.disturbance_template import DisturbanceTemplate


class NoDisturbance(DisturbanceTemplate):
    """Trivial disturbance: zero external wrench for all time (default)."""

    def __init__(self):
        self._zero_force = np.zeros(3)
        self._zero_torque = np.zeros(3)

    def update(self, t, state):
        return {'force': self._zero_force, 'torque': self._zero_torque}


class WrenchDisturbance(DisturbanceTemplate):
    """
    SkyDreamer two-band resample-and-hold external-wrench disturbance (Table III).

    The disturbance is drawn uniformly from a box in acceleration space and converted to a
    physical wrench via mass and inertia:
        force  (world frame) = mass * eps_a
        torque (body frame)  = inertia @ (eps_M_low + eps_M_high)

    Parameters:
        mass, float, vehicle mass [kg].
        inertia, np.ndarray (3,3), vehicle inertia matrix [kg m^2].
        accel_range, float, half-width of the linear-acceleration disturbance [m/s^2] (eps_a).
        accel_rate, float, resample rate of the linear-acceleration band [Hz].
        angaccel_lf_range, float, half-width of the low-frequency angular-accel band [rad/s^2].
        angaccel_lf_rate, float, resample rate of the low-frequency angular-accel band [Hz].
        angaccel_hf_range, float, half-width of the high-frequency angular-accel band [rad/s^2].
        angaccel_hf_rate, float, resample rate of the high-frequency angular-accel band [Hz].
        seed, int or None, RNG seed for reproducibility.

    Defaults reproduce the SkyDreamer *training* ranges. For the *eval* ranges use
    accel_range=2.0, angaccel_lf_range=2.0, angaccel_hf_range=100.0.

    Notes:
        - 'force' is returned in the WORLD frame (an external buffeting force, direction fixed in
          the world over the hold interval). 'torque' is returned in the BODY frame (body-axis
          moments), matching how Multirotor consumes external_torque.
        - update() is called once per simulator step; the values it returns are held constant
          across that step's internal integration, giving a piecewise-constant process.
    """

    def __init__(self, mass, inertia,
                 accel_range=3.0, accel_rate=1.0,
                 angaccel_lf_range=3.0, angaccel_lf_rate=1.0,
                 angaccel_hf_range=125.0, angaccel_hf_rate=90.0,
                 seed=None):
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=float)

        self.accel_range = float(accel_range)
        self.angaccel_lf_range = float(angaccel_lf_range)
        self.angaccel_hf_range = float(angaccel_hf_range)

        # Hold intervals (seconds) per band.
        self.accel_period = 1.0 / accel_rate
        self.angaccel_lf_period = 1.0 / angaccel_lf_rate
        self.angaccel_hf_period = 1.0 / angaccel_hf_rate

        self.rng = np.random.default_rng(seed)

        # Current held samples and the time at which each band was last resampled.
        self._accel = self._draw(self.accel_range)
        self._angaccel_lf = self._draw(self.angaccel_lf_range)
        self._angaccel_hf = self._draw(self.angaccel_hf_range)
        # Initialise last-sample times so the first update() past t=0 does not immediately
        # resample; bands resample once t advances past their period.
        self._t_accel = 0.0
        self._t_angaccel_lf = 0.0
        self._t_angaccel_hf = 0.0

    def _draw(self, half_width):
        return self.rng.uniform(-half_width, half_width, size=3)

    def update(self, t, state):
        # Resample any band whose hold interval has elapsed. Using a while loop keeps the
        # schedule correct even if a step is larger than a band's period (e.g. the 90 Hz band
        # at a 100 Hz sim resamples every step; at a slower sim it stays consistent).
        if t - self._t_accel >= self.accel_period:
            self._accel = self._draw(self.accel_range)
            self._t_accel = t
        if t - self._t_angaccel_lf >= self.angaccel_lf_period:
            self._angaccel_lf = self._draw(self.angaccel_lf_range)
            self._t_angaccel_lf = t
        if t - self._t_angaccel_hf >= self.angaccel_hf_period:
            self._angaccel_hf = self._draw(self.angaccel_hf_range)
            self._t_angaccel_hf = t

        force = self.mass * self._accel
        torque = self.inertia @ (self._angaccel_lf + self._angaccel_hf)
        return {'force': force, 'torque': torque}
