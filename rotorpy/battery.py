"""
Battery / supply-voltage models.

A battery model produces a slowly time-varying multiplicative scale on the vehicle's achievable
maximum rotor speed (rotor_speed_max_scale). As a LiPo discharges its terminal voltage sags,
reducing the maximum RPM the motors can reach and therefore the available thrust. SkyDreamer
observed this directly in real flight: the estimated max RPM dropped ~30% over a single battery
(arXiv:2510.14783), well outside the training randomization -- capturing it here lets a policy be
tested against that failure mode.

Interface: update(t) -> float scale in (0, 1], applied by the vehicle as
    effective_rotor_speed_max = rotor_speed_max * scale
and used in the motor-speed clip inside Multirotor.step().

Usage (mirrors the wind/disturbance pattern):
    battery = LinearBatterySag(drop_frac=0.30, duration=180.0)
    ...
    vehicle.rotor_speed_max_scale = battery.update(t)   # each step, before vehicle.step()
"""
import numpy as np


class NoBatterySag(object):
    """Trivial battery model: full voltage for all time (scale == 1)."""

    def update(self, t):
        return 1.0


class LinearBatterySag(object):
    """
    Linear droop of achievable max rotor speed.

        scale(t) = 1 - drop_frac * clip(t / duration, 0, 1)

    Parameters:
        drop_frac, fractional reduction of max rotor speed at end of the discharge window
                   (SkyDreamer field data ~ 0.30).
        duration, seconds over which the full drop occurs (e.g. a battery's usable flight time).
    """

    def __init__(self, drop_frac=0.30, duration=180.0):
        self.drop_frac = float(drop_frac)
        self.duration = float(duration)

    def update(self, t):
        frac = min(max(t / self.duration, 0.0), 1.0)
        return 1.0 - self.drop_frac * frac


class VoltageBatterySag(object):
    """
    Voltage-based sag using an identified voltage->max-RPM map (Crazyflow vmotor2rpm).

    The terminal voltage falls linearly with time (a simple coulomb-count proxy):
        V(t) = V0 - discharge_rate * t
    and the achievable max rotor speed follows the affine map Omega_max(V) = b0 + b1*V (rad/s).
    The returned scale is Omega_max(V(t)) / Omega_max(V0), clipped to (0, 1].

    Parameters:
        vmotor2rpm, [b0, b1] affine voltage->RPM coefficients (Crazyflow drones/params.toml).
                    Interpreted in RPM; the scale is unit-free so RPM vs rad/s does not matter.
        V0, initial (full) battery voltage.
        discharge_rate, volts per second of terminal-voltage droop.
    """

    def __init__(self, vmotor2rpm=(2968.18, 6647.95), V0=4.2, discharge_rate=0.0):
        self.b0, self.b1 = float(vmotor2rpm[0]), float(vmotor2rpm[1])
        self.V0 = float(V0)
        self.discharge_rate = float(discharge_rate)
        self._rpm_full = self.b0 + self.b1 * self.V0

    def update(self, t):
        V = self.V0 - self.discharge_rate * t
        rpm = self.b0 + self.b1 * V
        return float(np.clip(rpm / self._rpm_full, 1e-6, 1.0))
