"""
Causal Anomaly Scoring, MAD Standardization, and Temporal Persistence Gating.
Standardizes reconstruction residuals and enforces K-of-K unanimous exceedance.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class ChannelMADScorer:
    """
    Standardizes reconstruction squared errors using source-calibrated Median Absolute Deviation (MAD).
    Restricts scoring strictly to Core-8 channels to avoid ghost-sensor corruption.
    """
    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        mad: Optional[np.ndarray] = None,
        scored_channels: List[int] = None,
        clip_max: float = 20.0
    ):
        self.scored_channels = scored_channels if scored_channels is not None else list(range(8))
        self.clip_max = clip_max
        self.mu = mu
        self.mad = mad

    def fit(self, source_normal_errors: np.ndarray):
        """
        Fits robust median and MAD scale parameters from source normal reconstruction errors.
        Input source_normal_errors: [N_samples, Channels=8 or 12]
        """
        errs = source_normal_errors[:, self.scored_channels]
        self.mu = np.median(errs, axis=0)
        self.mad = 1.4826 * np.median(np.abs(errs - self.mu), axis=0) + 1e-9

    def score_windows(
        self,
        x_true: Union[np.ndarray, torch.Tensor],
        x_hat: Union[np.ndarray, torch.Tensor],
        aggregate_last_steps: int = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates last-hour Core-8 channel z-scores and composite anomaly score.
        Input tensors: [N_windows, 36, 12]
        Returns:
            composite_score: [N_windows] mean Core-8 z-score per window
            channel_zscores: [N_windows, 8] per-channel standardized error
        """
        if isinstance(x_true, torch.Tensor):
            x_true = x_true.detach().cpu().numpy()
        if isinstance(x_hat, torch.Tensor):
            x_hat = x_hat.detach().cpu().numpy()

        # Last-hour squared reconstruction errors: tau in [36-aggregate_last_steps : 36]
        diff_sq = (x_true[:, -aggregate_last_steps:, :] - x_hat[:, -aggregate_last_steps:, :]) ** 2
        last_hour_mse = np.mean(diff_sq, axis=1) # [N_windows, 12]

        # Extract Core-8
        core_mse = last_hour_mse[:, self.scored_channels] # [N_windows, 8]

        # Standardize using source calibration
        if self.mu is None or self.mad is None:
            # Fallback to internal sample statistics if unfitted
            mu = np.median(core_mse, axis=0)
            mad = 1.4826 * np.median(np.abs(core_mse - mu), axis=0) + 1e-9
        else:
            mu = self.mu
            mad = self.mad

        z_scores = np.clip((core_mse - mu) / mad, 0.0, self.clip_max)
        composite_score = np.mean(z_scores, axis=1)

        return composite_score, z_scores


class RollingMinGate:
    """
    Causal rolling-minimum persistence filter.
    Enforces that an anomaly must persist for K consecutive steps (e.g. K=24 for 4 hours)
    before an alert is triggered, eliminating transient false alarms.
    
    Mathematical Equivalence:
        S_rmin[t] > tau <=> S[t-j] > tau for all j in {0, ..., K-1}
    """
    def __init__(self, k_steps: int = 24):
        self.k_steps = k_steps

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Applies causal rolling minimum over raw chronological anomaly scores.
        Input raw_scores: [N_timesteps]
        Returns:
            filtered_scores: [N_timesteps]
        """
        n = len(raw_scores)
        filtered = np.zeros(n, dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.k_steps + 1)
            filtered[i] = np.min(raw_scores[start : i + 1])
        return filtered


def attribute_subsystem_health(channel_zscores: np.ndarray) -> Dict[str, float]:
    """
    Maps Core-8 channel z-scores into physical subsystem severity indices:
    - Aerodynamic / Pitch: Channels 0, 1, 3, 5
    - Drivetrain Mechanical: Channels 6, 7 (Oil & Bearing thermal rises)
    - Electrical / Conversion: Channels 2 (Power), and auxiliary stator if present
    """
    aero_indices = [0, 1, 3, 5]
    mech_indices = [6, 7]
    elec_indices = [2]

    aero_score = float(np.mean(channel_zscores[aero_indices]))
    mech_score = float(np.mean(channel_zscores[mech_indices]))
    elec_score = float(np.mean(channel_zscores[elec_indices]))

    return {
        "aerodynamic_pitch_severity": aero_score,
        "drivetrain_mechanical_severity": mech_score,
        "electrical_generator_severity": elec_score
    }
