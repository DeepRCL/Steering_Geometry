'''
Paper: Representation Engineering: A Top-Down Approach to AI Transparency
Reference: 
- Paper: http://arxiv.org/abs/2310.01405
- Code: https://github.com/andyzoujm/representation-engineering
'''

from sklearn.decomposition import PCA

import torch
from torch import Tensor

from ._base_steer import VecSteer


class RepE(VecSteer):
    def __init__(self):
        super().__init__()
        self.pca = PCA(n_components = 1)
    
    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor) -> 'RepE':
        if len(pos_X) != len(neg_X):
            n_Xs = min(len(pos_X), len(neg_X))
            pos_X, neg_X = pos_X[:n_Xs], neg_X[:n_Xs]
        diff = (pos_X - neg_X).detach().cpu()
        self.pca.fit(diff.numpy())
        self.steer_vec = torch.as_tensor(
            self.pca.components_[0],
            device = pos_X.device,
        )
        return self