"""
Multi-Kernel Maximum Mean Discrepancy (MK-MMD) Latent Domain Alignment.
Minimizes distribution shift between source and target wind farms in latent space.
"""

from typing import List
import torch
import torch.nn as nn


class MultiKernelMMDLoss(nn.Module):
    """
    Computes unbiased Multi-Kernel Maximum Mean Discrepancy (MMD) using RBF kernels.
    Reference: Gretton et al. (2012) "A Kernel Two-Sample Test", JMLR.
    """
    def __init__(self, bandwidths: List[float] = None):
        super().__init__()
        if bandwidths is None:
            self.bandwidths = [1.0, 5.0, 10.0]
        else:
            self.bandwidths = bandwidths

    def _rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor, sigma_sq: float) -> torch.Tensor:
        """
        Pairwise RBF kernel matrix: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2)).
        """
        # x1: [N, D], x2: [M, D]
        dist_sq = torch.cdist(x1, x2, p=2) ** 2
        return torch.exp(-dist_sq / (2.0 * sigma_sq))

    def forward(self, z_source: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Computes multi-kernel MMD between source and target latent embeddings.
        Input:
            z_source: [N, Latent_Dim=128]
            z_target: [M, Latent_Dim=128]
        Returns:
            Scalar MMD loss
        """
        n = z_source.size(0)
        m = z_target.size(0)
        if n == 0 or m == 0:
            return torch.tensor(0.0, device=z_source.device)

        mmd_total = torch.tensor(0.0, device=z_source.device)

        for bw in self.bandwidths:
            k_ss = self._rbf_kernel(z_source, z_source, bw)
            k_tt = self._rbf_kernel(z_target, z_target, bw)
            k_st = self._rbf_kernel(z_source, z_target, bw)

            # Unbiased estimate: zero out diagonals for within-domain kernels
            if n > 1:
                k_ss_sum = (k_ss.sum() - torch.trace(k_ss)) / (n * (n - 1))
            else:
                k_ss_sum = k_ss.sum() / (n * n)

            if m > 1:
                k_tt_sum = (k_tt.sum() - torch.trace(k_tt)) / (m * (m - 1))
            else:
                k_tt_sum = k_tt.sum() / (m * m)

            k_st_sum = k_st.sum() / (n * m)

            mmd_k = k_ss_sum + k_tt_sum - 2.0 * k_st_sum
            mmd_total = mmd_total + torch.clamp(mmd_k, min=0.0)

        return mmd_total / len(self.bandwidths)
