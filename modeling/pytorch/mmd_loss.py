# https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
r"""
math::
       \begin{align*}
           \text{MMD}^2 (P,Q) &= \underset{\| f \| \leq 1}{\text{sup}} | \mathbb{E}_{X\sim P}[f(X)]
           - \mathbb{E}_{Y\sim Q}[f(Y)] |^2 \\
           &\approx \frac{1}{B(B-1)} \sum_{i=1}^B \sum_{\substack{j=1 \\ j\neq i}}^B k(\mathbf{x}_i,\mathbf{x}_j)
           -\frac{2}{B^2}\sum_{i=1}^B \sum_{j=1}^B k(\mathbf{x}_i,\mathbf{y}_j)
           + \frac{1}{B(B-1)} \sum_{i=1}^B \sum_{\substack{j=1 \\ j\neq i}}^B k(\mathbf{y}_i,\mathbf{y}_j)
       \end{align*}
"""

import torch
from torch import nn


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.sum() / (n_samples**2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(X.device)
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).mean(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        Y_size = Y.shape[0]
        XX = (K[:X_size, :X_size].sum() - X_size) / (X_size * (X_size - 1))  # K(X_i, X_i) = 1となるためX_sizeを引く
        XY = K[:X_size, X_size:].sum() / (X_size * Y_size)
        YY = (K[X_size:, X_size:].sum() - Y_size) / (Y_size * (Y_size - 1))  # K(Y_i, Y_i) = 1となるためY_sizeを引く
        MMD2 = XX - 2 * XY + YY
        if torch.isnan(MMD2):
            print(K, XX.item(), XY.item(), YY.item(), X_size, Y_size)
        return MMD2
