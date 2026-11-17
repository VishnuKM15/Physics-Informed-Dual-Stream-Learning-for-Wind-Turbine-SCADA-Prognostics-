"""
Clean software-only inference and physics verification engine for DualStreamV5.

Replaces legacy hardware/UI demo wrappers with exact verified equations matching
build_tensors.py, train_v5.py, and final_frozen_evaluation.py.

Authoritative checkpoint: weights/epoch=17-step=2178.ckpt
Architecture: DualStreamV5 (763,572 parameters; 2.913 MiB FP32 state_dict; 20.73 MFLOPs/window)
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN: int = 36
N_FEAT: int = 12
CORE8: int = 8
DELTA_T_MIN: float = 10.0
DELTA_T_HOUR: float = DELTA_T_MIN / 60.0

AERO_IDX = (0, 1, 2, 3, 4, 8)     # wind, rotor, power_pu, pitch, ambient, power_residual
THERM_IDX = (5, 6, 7, 9, 10, 11)  # oil_rise, brg_rise, brg_spread, electrical, hydraulic, hyd_mask

FEATURE_NAMES = (
    "wind",
    "rotor",
    "power",
    "pitch",
    "ambient",
    "oil_rise",
    "brg_max_rise",
    "brg_spread",
    "power_residual",
    "electrical",
    "hydraulic",
    "hyd_mask",
)

# Source-calibrated reference threshold (Farm A+B normal 95th percentile under rmin24 in frozen eval)
CALIBRATED_SRC_P95_THRESHOLD: float = 2.3263


class StreamEnc(nn.Module):
    """Single-layer bidirectional LSTM encoder returning final step projection."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, latent_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class DualStreamV5(nn.Module):
    """Dual-stream subsystem-aligned physics-constrained Bi-LSTM autoencoder."""

    def __init__(self, seq_len: int = SEQ_LEN, n_feat: int = N_FEAT) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_feat = n_feat

        self.aero = StreamEnc(len(AERO_IDX), hidden_dim=128, latent_dim=64)
        self.therm = StreamEnc(len(THERM_IDX), hidden_dim=128, latent_dim=64)

        self.fusion = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.dec = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * n_feat),
        )

        # Learned forward-Euler step parameters (exponentially parameterized for positivity)
        self.log_alpha_oil = nn.Parameter(torch.tensor(-4.5))
        self.log_beta_oil = nn.Parameter(torch.tensor(-4.0))
        self.log_alpha_brg = nn.Parameter(torch.tensor(-4.5))
        self.log_beta_brg = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_aero = x[:, :, AERO_IDX]
        x_therm = x[:, :, THERM_IDX]
        z_raw = torch.cat([self.aero(x_aero), self.therm(x_therm)], dim=-1)
        z = self.fusion(z_raw)
        xr = self.dec(z).view(-1, self.seq_len, self.n_feat)
        return xr, z

    def phys_params(self) -> Dict[str, float]:
        """Return discrete step parameters, continuous-time rates, time constants, and steady-state gains."""
        ao = float(torch.exp(self.log_alpha_oil).item())
        bo = float(torch.exp(self.log_beta_oil).item())
        ab = float(torch.exp(self.log_alpha_brg).item())
        bb = float(torch.exp(self.log_beta_brg).item())

        return {
            "alpha_oil_step": ao,
            "beta_oil_step": bo,
            "pole_oil": 1.0 - bo,
            "alpha_oil_cont_per_h": ao / DELTA_T_HOUR,
            "beta_oil_cont_per_h": bo / DELTA_T_HOUR,
            "tau_oil_min": DELTA_T_MIN / bo,
            "tau_oil_hours": DELTA_T_HOUR / bo,
            "steady_state_oil_dT_full_load_C": (ao / bo) * 100.0,
            "alpha_brg_step": ab,
            "beta_brg_step": bb,
            "pole_brg": 1.0 - bb,
            "alpha_brg_cont_per_h": ab / DELTA_T_HOUR,
            "beta_brg_cont_per_h": bb / DELTA_T_HOUR,
            "tau_brg_min": DELTA_T_MIN / bb,
            "tau_brg_hours": DELTA_T_HOUR / bb,
            "steady_state_brg_dT_full_load_C": (ab / bb) * 100.0,
        }

    def physics_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window discrete Euler thermal conservation residual (normalized^2 units)."""
        ao = torch.exp(self.log_alpha_oil)
        bo = torch.exp(self.log_beta_oil)
        ab = torch.exp(self.log_alpha_brg)
        bb = torch.exp(self.log_beta_brg)

        P = x[:, :, 2]
        oil = x[:, :, 5]
        brg = x[:, :, 6]

        oil_hat = (1.0 - bo) * oil[:, :-1] + ao * P[:, :-1]
        brg_hat = (1.0 - bb) * brg[:, :-1] + ab * P[:, :-1]

        r_oil = ((oil[:, 1:] - oil_hat) ** 2).mean(dim=1)
        r_brg = ((brg[:, 1:] - brg_hat) ** 2).mean(dim=1)
        return r_oil + r_brg


def extract_physical_ontology(
    wind_ms: np.ndarray,
    rotor_rpm: np.ndarray,
    power_pu: np.ndarray,
    pitch_deg: np.ndarray,
    ambient_C: np.ndarray,
    oil_C: np.ndarray,
    bearing_C_matrix: np.ndarray,
    current_A_matrix: Optional[np.ndarray] = None,
    hydraulic_bar_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transform turbine SCADA signals into the exact 12-feature universal ontology
    matching src/data/build_tensors.py (note power_pu is already per-unit [-0.1, 1.1]).
    """
    n = len(wind_ms)
    wind = np.clip(np.nan_to_num(wind_ms, nan=0.0), 0.0, 35.0)
    rotor = np.clip(np.nan_to_num(rotor_rpm, nan=0.0), 0.0, 20.0)
    power = np.clip(np.nan_to_num(power_pu, nan=0.0), -0.1, 1.1)
    pitch = np.clip(np.nan_to_num(pitch_deg, nan=0.0), -5.0, 95.0)

    amb_fill = float(np.nanmedian(ambient_C)) if np.isfinite(ambient_C).any() else 15.0
    amb = np.clip(np.nan_to_num(ambient_C, nan=amb_fill), -50.0, 60.0)
    oil = np.where(np.isfinite(oil_C), np.clip(oil_C, -50.0, 150.0), amb)

    brg_2d = np.atleast_2d(bearing_C_matrix)
    if brg_2d.shape[0] != n:
        brg_2d = brg_2d.reshape(n, -1)
    brg_valid = np.where(np.isfinite(brg_2d), np.clip(brg_2d, -50.0, 150.0), amb[:, None])
    brg_max = brg_valid.max(axis=1)
    brg_min = brg_valid.min(axis=1)

    if current_A_matrix is not None and current_A_matrix.size > 0:
        cur_2d = np.nan_to_num(np.clip(np.atleast_2d(current_A_matrix).reshape(n, -1), 0.0, 2500.0), nan=0.0)
        electrical = cur_2d.max(axis=1) / 2500.0
    else:
        electrical = np.zeros(n, dtype=np.float64)

    if hydraulic_bar_matrix is not None and hydraulic_bar_matrix.size > 0:
        hyd_2d = np.nan_to_num(np.clip(np.atleast_2d(hydraulic_bar_matrix).reshape(n, -1), 0.0, 1500.0), nan=0.0)
        hydraulic = np.clip(hyd_2d.mean(axis=1) / 500.0, 0.0, 1.0)
        hyd_mask = np.ones(n, dtype=np.float64)
    else:
        hydraulic = np.zeros(n, dtype=np.float64)
        hyd_mask = np.zeros(n, dtype=np.float64)

    ideal_power = np.minimum(1.0, (wind / 12.0) ** 3)
    power_residual = np.clip(power - ideal_power, -0.5, 0.5)

    feats = np.stack(
        [
            wind / 25.0,
            rotor / 20.0,
            power,
            pitch / 90.0,
            amb / 100.0,
            np.clip(oil - amb, -20.0, 100.0) / 100.0,
            np.clip(brg_max - amb, -20.0, 120.0) / 100.0,
            np.clip(brg_max - brg_min, 0.0, 80.0) / 50.0,
            power_residual,
            electrical,
            hydraulic,
            hyd_mask,
        ],
        axis=1,
    ).astype(np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def causal_rolling_min(scores: np.ndarray, k: int = 24) -> np.ndarray:
    """
    Causal K-of-K unanimous exceedance gate:
    score_rmin[t] = min_{j=0..K-1} score[t - j] (left-padded with scores[0]).
    An alarm fires iff score exceeds threshold at all K consecutive 10-min intervals.
    """
    if k <= 1 or len(scores) == 0:
        return scores
    pad = np.concatenate([np.full(k - 1, scores[0], dtype=scores.dtype), scores])
    return np.lib.stride_tricks.sliding_window_view(pad, k).min(axis=1)


def robust_mad_zscore(
    core8_errs: np.ndarray,
    med: Optional[np.ndarray] = None,
    mad: Optional[np.ndarray] = None,
    clip_max: float = 20.0,
) -> np.ndarray:
    """Compute clipped robust MAD Z-score per Core-8 channel (consistent with final_frozen_evaluation.py)."""
    if med is None:
        med = np.median(core8_errs, axis=0)
    if mad is None:
        mad = np.median(np.abs(core8_errs - med), axis=0) * 1.4826
    z = np.clip((core8_errs - med) / (mad + 1e-9), 0.0, clip_max).astype(np.float32)
    return z.mean(axis=1)


class SoftwareInferenceEngine:
    """Verified PyTorch software inference engine backed by DualStreamV5 weights."""

    def __init__(
        self,
        ckpt_path: str = os.path.join("weights", "epoch=17-step=2178.ckpt"),
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = DualStreamV5(SEQ_LEN, N_FEAT)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score_windows(
        self,
        windows_12: np.ndarray,
        input_mode: str = "full12",
        batch_size: int = 2048,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Core-8 last-6-step reconstruction MSE and ODE conservation residuals.

        Args:
            windows_12: shape (N, 36, 12)
            input_mode: 'full12' or 'zeropad8' (zeroes auxiliary/mask channels 8..11)
        """
        if windows_12.ndim != 3 or windows_12.shape[1:] != (SEQ_LEN, N_FEAT):
            raise ValueError(f"Expected shape (N, {SEQ_LEN}, {N_FEAT}), got {windows_12.shape}")

        xt = torch.from_numpy(np.ascontiguousarray(windows_12, dtype=np.float32)).to(self.device)
        if input_mode == "zeropad8":
            xt = torch.cat([xt[:, :, :CORE8], torch.zeros_like(xt[:, :, CORE8:])], dim=-1)
        elif input_mode != "full12":
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        core8_err_list = []
        phys_res_list = []
        latent_list = []

        for i in range(0, len(xt), batch_size):
            xb = xt[i : i + batch_size]
            xr, z = self.model(xb)
            sq_core8 = (xr[:, :, :CORE8] - xb[:, :, :CORE8]) ** 2
            core8_err_list.append(sq_core8[:, -6:, :].mean(dim=1).cpu().numpy())
            phys_res_list.append(self.model.physics_residual(xb).cpu().numpy())
            latent_list.append(z.cpu().numpy())

        return {
            "core8_last6_mse": np.concatenate(core8_err_list, axis=0),
            "physics_residual": np.concatenate(phys_res_list, axis=0),
            "latent": np.concatenate(latent_list, axis=0),
        }

    def benchmark_cpu(self, batch_size: int = 1, n_warmup: int = 20, n_runs: int = 100) -> Dict[str, float]:
        """Measure actual local software inference latency without synthetic estimates."""
        x = torch.randn(batch_size, SEQ_LEN, N_FEAT, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(x)
            latencies_ms = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = self.model(x)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(latencies_ms, dtype=np.float64)
        mean_ms = float(arr.mean())
        return {
            "batch_size": float(batch_size),
            "mean_batch_ms": mean_ms,
            "p95_batch_ms": float(np.percentile(arr, 95)),
            "per_window_ms": mean_ms / batch_size,
            "throughput_win_per_s": (batch_size * 1000.0) / mean_ms,
        }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = os.path.join(base_dir, "weights", "epoch=17-step=2178.ckpt")
    engine = SoftwareInferenceEngine(ckpt_path=ckpt, device="cpu")
    n_params = sum(p.numel() for p in engine.model.parameters())
    print(f"Loaded DualStreamV5 strictly: {n_params:,} parameters")
    phys = engine.model.phys_params()
    print(
        f"Learned Euler ODE -> Oil tau={phys['tau_oil_hours']:.2f} h (dT_ss={phys['steady_state_oil_dT_full_load_C']:.2f} C) | "
        f"Brg tau={phys['tau_brg_hours']:.2f} h (dT_ss={phys['steady_state_brg_dT_full_load_C']:.2f} C)"
    )
    bench = engine.benchmark_cpu(batch_size=1, n_warmup=10, n_runs=40)
    print(f"CPU single-window smoke test: {bench['mean_batch_ms']:.3f} ms ({bench['throughput_win_per_s']:.1f} win/s)")

