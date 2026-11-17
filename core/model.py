"""
DualStreamV5 Physics-Informed Neural Network (PINN) Architecture.
Decouples high-frequency aerodynamic boundaries from high-inertia drivetrain thermodynamics.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 36
TOTAL_FEAT = 12
AERO_FEAT = 6
THERM_FEAT = 6


class StreamEnc(nn.Module):
    """
    Bidirectional LSTM physical stream encoder.
    Maps temporal window of physical measurements into a compact latent vector.
    """
    def __init__(self, in_features: int = 6, hidden_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Sequence_Length=36, Features=6]
        out, _ = self.lstm(x)
        # Extract terminal hidden state of bidirectional representation
        return self.fc(out[:, -1])


class DualStreamV5(nn.Module):
    """
    Physics-Informed Dual-Stream Sequence Autoencoder (DualStreamV5).
    Incorporates forward-Euler thermal ODE constraints and multi-kernel latent alignment.
    """
    def __init__(
        self,
        seq_len: int = 36,
        total_dim: int = 12,
        stream_dim: int = 6,
        hidden_dim: int = 128,
        proj_dim: int = 64,
        beta_max: float = 0.5
    ):
        super().__init__()
        self.seq_len = seq_len
        self.total_dim = total_dim
        self.beta_max = beta_max

        # Stream 1: Aerodynamic / Atmospheric boundary encoder
        self.aero_enc = StreamEnc(in_features=stream_dim, hidden_dim=hidden_dim, proj_dim=proj_dim)

        # Stream 2: Thermodynamic drivetrain / Auxiliary encoder
        self.therm_enc = StreamEnc(in_features=stream_dim, hidden_dim=hidden_dim, proj_dim=proj_dim)

        # Cross-stream latent fusion MLP: [128 -> 256 -> 128]
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Temporal sequence decoder MLP: [128 -> 256 -> 512 -> (36*12)=432]
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * total_dim)
        )

        # Learnable discrete forward-Euler thermodynamic ODE parameters
        # Bounded parameterization: beta_step = beta_max * sigmoid(tilde_beta)
        # Guarantees 0 < beta_step < beta_max < 1, ensuring non-oscillatory asymptotic stability
        self.raw_alpha_oil = nn.Parameter(torch.tensor(-5.2983))  # exp(-5.2983) approx 0.005
        self.raw_beta_oil  = nn.Parameter(torch.tensor(-4.5000))  # initial beta_step approx 0.0055
        self.raw_alpha_brg = nn.Parameter(torch.tensor(-5.6248))  # exp(-5.6248) approx 0.0036
        self.raw_beta_brg  = nn.Parameter(torch.tensor(-4.7700))  # initial beta_step approx 0.0042

    def get_phys_params(self) -> Dict[str, torch.Tensor]:
        """
        Returns bounded physical ODE heating rates (alpha) and dissipation rates (beta).
        """
        alpha_oil = torch.exp(self.raw_alpha_oil)
        beta_oil  = self.beta_max * torch.sigmoid(self.raw_beta_oil)

        alpha_brg = torch.exp(self.raw_alpha_brg)
        beta_brg  = self.beta_max * torch.sigmoid(self.raw_beta_brg)

        return {
            "alpha_oil": alpha_oil,
            "beta_oil": beta_oil,
            "alpha_brg": alpha_brg,
            "beta_brg": beta_brg
        }

    def compute_time_constants_hours(self) -> Tuple[float, float]:
        """
        Derives continuous-time drivetrain thermal time constants (tau = 10 min / beta_step).
        """
        p = self.get_phys_params()
        beta_oil = p["beta_oil"].item()
        beta_brg = p["beta_brg"].item()

        tau_oil_hours = 10.0 / (beta_oil * 60.0)
        tau_brg_hours = 10.0 / (beta_brg * 60.0)
        return tau_oil_hours, tau_brg_hours

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass over input SCADA window tensor.
        Input x: [Batch, Sequence_Length=36, Total_Features=12]
        Returns:
            x_hat: Reconstructed sequence tensor [Batch, 36, 12]
            z: Latent fusion embedding [Batch, 128]
            phys: Learned physical parameters dict
        """
        # Split into physical streams
        x_aero = x[:, :, 0:6]
        x_therm = x[:, :, 6:12]

        # Encode streams
        h_aero = self.aero_enc(x_aero)    # [B, 64]
        h_therm = self.therm_enc(x_therm)  # [B, 64]

        # Cross-stream latent bottleneck
        h_cat = torch.cat([h_aero, h_therm], dim=-1) # [B, 128]
        z = self.fusion(h_cat)                        # [B, 128]

        # Decode sequence
        flat_recon = self.decoder(z)                  # [B, 432]
        x_hat = flat_recon.view(-1, self.seq_len, self.total_dim)

        phys = self.get_phys_params()
        return x_hat, z, phys

    def predict_next_thermal_rise(
        self,
        delta_T_t: torch.Tensor,
        P_pu_t: torch.Tensor,
        subsystem: str = "oil"
    ) -> torch.Tensor:
        """
        Discrete forward-Euler thermal ODE step:
        Delta_T[t+1] = (1 - beta_step) * Delta_T[t] + alpha_step * P_pu[t]
        """
        phys = self.get_phys_params()
        if subsystem == "oil":
            a = phys["alpha_oil"]
            b = phys["beta_oil"]
        elif subsystem == "bearing":
            a = phys["alpha_brg"]
            b = phys["beta_brg"]
        else:
            raise ValueError(f"Unknown subsystem {subsystem}")

        return (1.0 - b) * delta_T_t + a * P_pu_t
