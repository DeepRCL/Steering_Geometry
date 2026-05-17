from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn


class RFF(nn.Module):
    def __init__(
        self, 
        n_components: Optional[int] = None, 
        sigma: float | str = 'median',
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_components = n_components
        self.sigma = sigma
        self.random_state = random_state
        
    def fit(self, X: Tensor):
        d, device, dtype = X.shape[1], X.device, X.dtype
        if self.n_components is None:
            self.n_components = d
        if self.random_state is not None:
            generator = torch.Generator(device = device).manual_seed(self.random_state)
        else:
            generator = None
        self.register_buffer(
            'W', 
            torch.randn(self.n_components, d, device = device, dtype = dtype, generator = generator) / self.get_sigma(X)
        )
        self.register_buffer(
            'b', 
            torch.rand(self.n_components, device = device, dtype = dtype, generator = generator) * 2 * np.pi
        )
        return self

    def forward(self, X: Tensor) -> Tensor:
        return (2 / self.n_components)**0.5 * torch.cos(X @ self.W.T + self.b)
    
    def transform(self, X: Tensor) -> Tensor:
        return self(X)
    
    def fit_transform(self, X: Tensor) -> Tensor:
        self.fit(X)
        return self.transform(X)
    
    def jacobian(self, X: Tensor) -> Tensor:
        XW = X @ self.W.T + self.b
        term = -(2 / self.n_components)**0.5 * torch.sin(XW)
        return self.W * term.unsqueeze(-1)
    
    def jvp(self, X: Tensor, v: Tensor) -> Tensor:
        XW = torch.matmul(X, self.W.T, out=None)
        torch.add(XW, self.b, out=XW)
        torch.sin(XW, out=XW)
        XW.mul_(-(2 / self.n_components)**0.5)
        
        if v.ndim == 1:
            Wv = torch.matmul(self.W, v, out=None)
            return XW * Wv
        else:
            assert X.shape[0] == v.shape[0]
            Wv = torch.einsum('nd, bd -> bn', self.W, v)
            return XW * Wv
        
    def vjp(self, X: Tensor, v: Tensor) -> Tensor:
        XW = torch.matmul(X, self.W.T, out=None)
        torch.add(XW, self.b, out=XW)
        torch.sin(XW, out=XW)
        XW.mul_(-(2 / self.n_components)**0.5)
        
        if v.ndim == 1:
            # v is [n_components], XW is [batch, n_components]
            # Result should be [batch, input_dim]
            v_scaled = XW * v  # [batch, n_components]
            return torch.einsum('bn, nd -> bd', v_scaled, self.W)
        else:
            # v is [batch, n_components]
            v_scaled = XW * v
            return torch.einsum('nd, bn -> bd', self.W, v_scaled)
        
    def laplacian(self, X: Tensor) -> Tensor:
        w_norm = torch.norm(self.W, dim = 1)**2
        term = -(2 / self.n_components)**0.5 * torch.cos(X @ self.W.T + self.b)
        return term * w_norm.unsqueeze(0)
    
    def get_sigma(self, X: Tensor) -> float:
        if self.sigma == 'median':
            n_samples = min(1000, X.shape[0])
            X_sampled = X[torch.randperm(X.shape[0])[:n_samples]]
            cdist = torch.tril(torch.cdist(X_sampled, X_sampled))
            median = torch.median(cdist[cdist != 0].flatten())
            return median / 2**0.5
        elif self.sigma == 'scale':
            return (X.shape[1] / 2)**0.5 * X.std()
        else:
            return self.sigma