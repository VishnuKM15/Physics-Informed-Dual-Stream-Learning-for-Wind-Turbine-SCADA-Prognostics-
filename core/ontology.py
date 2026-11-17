"""
Universal Physical SCADA Ontology Definitions and Normalization Routines.
Maps heterogeneous wind turbine SCADA telemetry into invariant dual-stream tensors.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch

FEATURE_NAMES: List[str] = [
    "wind_speed_norm",        # 0: Aero Stream (v / 12.0 m/s)
    "rotor_speed_norm",       # 1: Aero Stream (omega / 16.0 rpm)
    "active_power_norm",      # 2: Aero Stream (P / 3000.0 kW)
    "pitch_angle_norm",       # 3: Aero Stream (theta / 30.0 deg)
    "ambient_temp_norm",      # 4: Aero Stream (T_amb / 40.0 C)
    "power_residual",         # 5: Aero Stream (P_aero - P_meas)
    "oil_temp_rise_norm",     # 6: Therm Stream (Delta T_oil / 60.0 C)
    "bearing_temp_rise_norm", # 7: Therm Stream (Delta T_brg / 60.0 C)
    "stator_current_norm",    # 8: Aux Stream (I / 2500.0 A)
    "stator_current_mask",    # 9: Aux Stream (Binary sensor mask)
    "hydraulic_pressure_norm",# 10: Aux Stream (p_hyd / 200.0 bar)
    "hydraulic_mask"          # 11: Aux Stream (Binary sensor mask)
]

AERO_STREAM_INDICES: List[int] = [0, 1, 2, 3, 4, 5]
THERM_STREAM_INDICES: List[int] = [6, 7, 8, 9, 10, 11]
CORE_8_INDICES: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]
AUXILIARY_INDICES: List[int] = [8, 9, 10, 11]

# Physical scaling denominators
SCALING_CONSTANTS: Dict[str, float] = {
    "wind_speed": 12.0,           # Rated wind speed (Region II/III transition per IEC 61400-1)
    "rotor_speed": 16.0,          # Nominal synchronous rotor speed (rpm)
    "active_power": 3000.0,       # Fleet nameplate rated electrical power ceiling (kW)
    "pitch_angle": 30.0,          # Nominal operational pitch range (degrees)
    "ambient_temp": 40.0,         # Maximum ambient operating boundary (C)
    "oil_temp_rise": 60.0,        # Drivetrain oil thermal rise above ambient (C)
    "bearing_temp_rise": 60.0,    # Main bearing thermal rise above ambient (C)
    "stator_current": 2500.0,     # Rated full-load stator current at 690 V (A)
    "hydraulic_pressure": 200.0   # Hydraulic accumulator operating pressure (bar)
}

# Physical bounds for robust clipping
PHYSICAL_LIMITS: Dict[str, Tuple[float, float]] = {
    "wind_speed": (0.0, 35.0),
    "rotor_speed": (0.0, 25.0),
    "active_power": (0.0, 4000.0),
    "pitch_angle": (-5.0, 95.0),
    "ambient_temp": (-25.0, 50.0),
    "oil_temp": (-10.0, 120.0),
    "bearing_temp": (-10.0, 120.0),
    "stator_current": (0.0, 3500.0),
    "hydraulic_pressure": (0.0, 250.0)
}


def compute_aerodynamic_power_curve(wind_speed_ms: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes theoretical cubic aerodynamic power curve residual reference.
    Region II cubic tracking with unity saturation in Region III.
    """
    if isinstance(wind_speed_ms, torch.Tensor):
        v_norm = torch.clamp(wind_speed_ms / SCALING_CONSTANTS["wind_speed"], 0.0, 3.0)
        return torch.clamp(v_norm ** 3, 0.0, 1.0)
    else:
        v_norm = np.clip(wind_speed_ms / SCALING_CONSTANTS["wind_speed"], 0.0, 3.0)
        return np.clip(v_norm ** 3, 0.0, 1.0)


def raw_to_normalized_tensor(
    wind_speed: np.ndarray,
    rotor_speed: np.ndarray,
    active_power: np.ndarray,
    pitch_angle: np.ndarray,
    ambient_temp: np.ndarray,
    oil_temp: np.ndarray,
    bearing_temp: np.ndarray,
    stator_current: np.ndarray = None,
    hydraulic_pressure: np.ndarray = None
) -> np.ndarray:
    """
    Normalizes raw SCADA time-series into the standardized 12-channel physical ontology.
    Handles sensor absence gracefully via automated binary masking.
    """
    N = len(wind_speed)
    X = np.zeros((N, 12), dtype=np.float32)

    # 0. Wind speed
    v = np.clip(wind_speed, PHYSICAL_LIMITS["wind_speed"][0], PHYSICAL_LIMITS["wind_speed"][1])
    X[:, 0] = v / SCALING_CONSTANTS["wind_speed"]

    # 1. Rotor speed
    w = np.clip(rotor_speed, PHYSICAL_LIMITS["rotor_speed"][0], PHYSICAL_LIMITS["rotor_speed"][1])
    X[:, 1] = w / SCALING_CONSTANTS["rotor_speed"]

    # 2. Active power
    p = np.clip(active_power, PHYSICAL_LIMITS["active_power"][0], PHYSICAL_LIMITS["active_power"][1])
    X[:, 2] = p / SCALING_CONSTANTS["active_power"]

    # 3. Pitch angle
    theta = np.clip(pitch_angle, PHYSICAL_LIMITS["pitch_angle"][0], PHYSICAL_LIMITS["pitch_angle"][1])
    X[:, 3] = theta / SCALING_CONSTANTS["pitch_angle"]

    # 4. Ambient temperature
    t_amb = np.clip(ambient_temp, PHYSICAL_LIMITS["ambient_temp"][0], PHYSICAL_LIMITS["ambient_temp"][1])
    X[:, 4] = t_amb / SCALING_CONSTANTS["ambient_temp"]

    # 5. Theoretical power curve residual: P_aero(v) - P_meas
    p_aero = compute_aerodynamic_power_curve(v)
    p_meas = X[:, 2]
    X[:, 5] = np.clip(p_aero - p_meas, -2.0, 2.0)

    # 6. Gearbox oil thermal rise above ambient: Delta T_oil = T_oil - T_amb
    t_oil = np.clip(oil_temp, PHYSICAL_LIMITS["oil_temp"][0], PHYSICAL_LIMITS["oil_temp"][1])
    delta_t_oil = np.clip(t_oil - t_amb, 0.0, 80.0)
    X[:, 6] = delta_t_oil / SCALING_CONSTANTS["oil_temp_rise"]

    # 7. Main bearing thermal rise above ambient: Delta T_brg = T_brg - T_amb
    t_brg = np.clip(bearing_temp, PHYSICAL_LIMITS["bearing_temp"][0], PHYSICAL_LIMITS["bearing_temp"][1])
    delta_t_brg = np.clip(t_brg - t_amb, 0.0, 80.0)
    X[:, 7] = delta_t_brg / SCALING_CONSTANTS["bearing_temp_rise"]

    # 8-9. Stator current and availability mask
    if stator_current is not None and not np.isnan(stator_current).all():
        i_stat = np.clip(stator_current, PHYSICAL_LIMITS["stator_current"][0], PHYSICAL_LIMITS["stator_current"][1])
        X[:, 8] = i_stat / SCALING_CONSTANTS["stator_current"]
        X[:, 9] = 1.0
    else:
        X[:, 8] = 0.0
        X[:, 9] = 0.0

    # 10-11. Hydraulic pressure and availability mask
    if hydraulic_pressure is not None and not np.isnan(hydraulic_pressure).all():
        p_hyd = np.clip(hydraulic_pressure, PHYSICAL_LIMITS["hydraulic_pressure"][0], PHYSICAL_LIMITS["hydraulic_pressure"][1])
        X[:, 10] = p_hyd / SCALING_CONSTANTS["hydraulic_pressure"]
        X[:, 11] = 1.0
    else:
        X[:, 10] = 0.0
        X[:, 11] = 0.0

    return X


def apply_core8_zeropad(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies strict Core-8 projection (zeropad8) by zeroing auxiliary channels 8:12.
    Eliminates ghost-sensor error collapse when evaluating on target turbines.
    """
    if isinstance(X, torch.Tensor):
        X_proj = X.clone()
        X_proj[..., 8:12] = 0.0
        return X_proj
    else:
        X_proj = X.copy()
        X_proj[..., 8:12] = 0.0
        return X_proj
