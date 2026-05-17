'''
Paper: Controlling Language and Diffusion Models by Transporting Activations
Reference: 
- Paper: https://openreview.net/forum?id=l2zFn6TIQi
- Code: https://github.com/apple/ml-act
'''

import torch
from torch import Tensor, nn

from ._base_steer import Steer


class LinOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fitted = False
        
    def fit(self, X_target: Tensor, X_source: Tensor):
        n = min(len(X_target), len(X_source))
        X_target, X_source = X_target[:n], X_source[:n]
        m_target, m_source = X_target.mean(dim = 0), X_source.mean(dim = 0)
        X_target_centered = (X_target - m_target).sort(dim = 0).values
        X_source_centered = (X_source - m_source).sort(dim = 0).values
        w_num = torch.sum(X_target_centered * X_source_centered, dim = 0)
        w_den = torch.sum(X_source_centered ** 2, dim = 0)
        self.register_buffer('w', w_num / (w_den + 1e-10))
        self.register_buffer('b', m_target - self.w * m_source)
        self.fitted = True
        return self
    
    def forward(self, X: Tensor) -> Tensor:
        return X * self.w + self.b
    
    
class LinAcT(Steer):
    def __init__(self):
        super().__init__()
        self.lin_ot = LinOT()
        
    def fit(self, pos_X: Tensor, neg_X: Tensor):
        self.lin_ot.fit(pos_X, neg_X)
        return self
    
    def steer(self, X: Tensor, T: float = 1.0):
        self.lin_ot.to(X.device)
        return self.lin_ot(X)
    
    def vector_field(self, X: Tensor):
        self.lin_ot.to(X.device)
        return self.lin_ot(X) - X