from rotorpy.vehicles.multirotor import BatchedMultirotorParams
from rotorpy.vehicles.crazyflie_params import quad_params as cf_params
import numpy as np

crazyflie_randomizations = {
    "mass": [0.027, 0.033],
    "k_eta": [2.1e-8, 2.5e-8],
}


def percent_randomization(nominal_params, pct=0.30, motor_limit_pct=0.20):
    """
    Build a randomization_ranges dict spanning all supported physical parameters, each centered on
    its nominal value with a +/- pct band. Motor speed limits use motor_limit_pct. This reproduces
    the SkyDreamer Table III scheme (arXiv:2510.14783): +/-30% on dynamics parameters, +/-20% on
    motor speed limits. Only parameters that are present and nonzero in nominal_params are included
    (so, e.g., zero drag coefficients are left untouched).

    Inputs:
        nominal_params: the nominal parameter dict (center of each distribution).
        pct: fractional half-width for dynamics parameters (default 0.30).
        motor_limit_pct: fractional half-width for rotor_speed_min/max (default 0.20).
    Returns:
        ranges: dict mapping parameter name -> [low, high], consumable by update_vehicle_params.
    """
    ranges = {}

    def band(key, p):
        val = nominal_params.get(key, None)
        if val is None:
            return
        val = float(np.asarray(val).reshape(-1)[0])  # scalar or per-rotor: use the representative value
        if val == 0.0:
            return
        lo, hi = val * (1.0 - p), val * (1.0 + p)
        ranges[key] = [min(lo, hi), max(lo, hi)]

    for key in ("mass", "k_eta", "k_m", "Ixx", "Iyy", "Izz", "tau_m",
                "c_Dx", "c_Dy", "c_Dz", "k_d", "k_z"):
        band(key, pct)
    for key in ("rotor_speed_min", "rotor_speed_max"):
        band(key, motor_limit_pct)
    return ranges

def generate_random_vehicle_params(num_drones,
                                    device,
                                    nominal_params = cf_params,
                                    randomization_ranges = crazyflie_randomizations):
    """ 
    Generate random vehicle params. 
    Inputs:
        num_drones: the number of drones to generate params for. 
        nominal_params: the nominal parameters for the drone, i.e. the center of each sampling distribution.
        randomization_ranges: the range with which parameters are selected. 
    """

    batch_params = BatchedMultirotorParams([nominal_params for _ in range(num_drones)], num_drones, device)
    for idx in range(num_drones):
        update_vehicle_params(idx,
                              randomization_ranges,
                              batch_params)
    return batch_params

def update_vehicle_params(idx,
                          ranges,
                          params_obj):
    """
    Update vehicle parameters. 
    Inputs:
        idx: the particular idx of the drone. 
        ranges: the range of values to sample from. 
        prams_obj: the object (type BatchedMultirotorParams) to modify.

    """
    # Sample the rotor speed limits first so the k_eta hover-feasibility bound below uses the
    # randomized (not nominal) maximum.
    if "rotor_speed_min" in ranges:
        params_obj.rotor_speed_min[idx] = np.random.uniform(ranges["rotor_speed_min"][0], ranges["rotor_speed_min"][1])
    if "rotor_speed_max" in ranges:
        params_obj.rotor_speed_max[idx] = np.random.uniform(ranges["rotor_speed_max"][0], ranges["rotor_speed_max"][1])

    min_k_eta = 0
    if "mass" in ranges:
        mass_val = np.random.uniform(ranges["mass"][0], ranges["mass"][1])
        params_obj.update_mass(idx, mass_val)

        # divide rotor speed max by 1.2 to give some margin to allow flight
        min_k_eta = (mass_val * (params_obj.g / 4) / ((params_obj.rotor_speed_max[idx]/1.2) ** 2)).item()
    if "k_eta" in ranges or "k_m" in ranges or "rotor_pos" in ranges:
        # Clamp the lower bound up to min_k_eta for hover feasibility. If that exceeds the band's
        # upper bound (possible once mass is high and rotor_speed_max is randomized low), raise the
        # upper bound to match so the range never inverts and the sample stays >= min_k_eta.
        if "k_eta" in ranges:
            k_eta_lo = max(ranges["k_eta"][0], min_k_eta)
            k_eta_hi = max(ranges["k_eta"][1], k_eta_lo)
            k_eta = np.random.uniform(k_eta_lo, k_eta_hi)
        else:
            k_eta = None
        k_m = np.random.uniform(ranges["k_m"][0],
                                ranges["k_m"][1]) if "k_m" in ranges else None
        rotor_pos=None  # not implemented yet
        params_obj.update_thrust_and_rotor_params(idx, k_eta, k_m, rotor_pos)
    if "Ixx" in ranges or "Iyy" in ranges or "Izz" in ranges:
        Ixx = np.random.uniform(ranges["Ixx"][0],
                                ranges["Ixx"][1]) if "Ixx" in ranges else None
        Iyy  = np.random.uniform(ranges["Iyy"][0],
                                 ranges["Iyy"][1]) if "Iyy" in ranges else None
        Izz  = np.random.uniform(ranges["Izz"][0],
                                 ranges["Izz"][1]) if "Izz" in ranges else None
        params_obj.update_inertia(idx, Ixx, Iyy, Izz)
    if "tau_m" in ranges:
        tau_m = np.random.uniform(ranges["tau_m"][0],ranges["tau_m"][1])
        params_obj.tau_m[idx] = tau_m
    if "motor_noise" in ranges:
        motor_noise = np.random.uniform(ranges["motor_noise"][0], ranges["motor_noise"][1])
        params_obj.motor_noise[idx] = motor_noise
    if any(k in ranges for k in ("c_Dx", "c_Dy", "c_Dz", "k_d", "k_z")):
        sample = lambda k: np.random.uniform(ranges[k][0], ranges[k][1]) if k in ranges else None
        params_obj.update_drag(idx, sample("c_Dx"), sample("c_Dy"), sample("c_Dz"), sample("k_d"), sample("k_z"))

    return