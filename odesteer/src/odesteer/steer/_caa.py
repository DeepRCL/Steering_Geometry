'''
Paper: Steering Llama 2 via Contrastive Activation Addition (ACL 2024)
Reference: 
- Paper: https://aclanthology.org/2024.acl-long.828/
- Code: https://github.com/nrimsky/CAA
'''

import torch
from torch import Tensor

from ._base_steer import VecSteer


class CAA(VecSteer):
    @torch.no_grad()
    def fit(
        self,
        pos_X: Tensor,
        neg_X: Tensor,
    ) -> 'CAA':
        pos_mean = pos_X.mean(dim = 0)
        neg_mean = neg_X.mean(dim = 0)
        self.steer_vec = pos_mean - neg_mean
        return self