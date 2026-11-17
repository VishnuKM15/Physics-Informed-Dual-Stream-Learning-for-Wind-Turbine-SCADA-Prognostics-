"""
Observed-State Discrete Forward-Euler Thermal ODE Physics Engine.
Enforces thermodynamic energy balance and physical drivetrain time constants.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class DiscreteEulerLoss(nn.Module):
    """
    Computes thermodynamic ODE residual loss over 10-minute SCADA trajectories.
    
    First-principles lumped-parameter thermal ODE:
        d(Delta_T)/dt = alpha_continuous * P_pu - beta_continuous * Delta_T
    
    Forward-Euler discretization with Delta_t = 10 minutes:
        Delta_T[t+1] = (1 - beta_step) * Delta_T[t] + alpha_step * P_pu[t]
        
    where beta_step = beta_continuous * Delta_t and alpha_step = alpha_continuous * Delta_t.
    """
    def __init__(self, power_idx: int = 2, oil_idx: int = 6, brg_idx: int = 7):
        super().__init__()
        self.power_idx = power_idx
        self.oil_idx = oil_idx
        self.brg_idx = brg_idx

    def forward(
        self,
        x_true: torch.Tensor,
        phys_params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates mean squared ODE residual over the temporal window.
        Input x_true: [Batch, Sequence_Length=36, Features=12]
        Returns:
            loss_phys: Combined oil and bearing ODE residual loss
            res_oil: Oil ODE residual tensor
            res_brg: Bearing ODE residual tensor
        """
        # Observed trajectories: [Batch, 35]
        p_t = x_true[:, :-1, self.power_idx]
        oil_t = x_true[:, :-1, self.oil_idx]
        brg_t = x_true[:, :-1, self.brg_idx]

        oil_next = x_true[:, 1:, self.oil_idx]
        brg_next = x_true[:, 1:, self.brg_idx]

        # Euler step predictions
        a_oil = phys_params["alpha_oil"]
        b_oil = phys_params["beta_oil"]
        a_brg = phys_params["alpha_brg"]
        b_brg = phys_params["beta_brg"]

        pred_oil_next = (1.0 - b_oil) * oil_t + a_oil * p_t
        pred_brg_next = (1.0 - b_brg) * brg_t + a_brg * p_t

        # Residuals
        res_oil = oil_next - pred_oil_next
        res_brg = brg_next - pred_brg_next

        loss_oil = torch.mean(res_oil ** 2)
        loss_brg = torch.mean(res_brg ** 2)
        loss_phys = loss_oil + loss_brg

        return loss_phys, res_oil, res_brg
