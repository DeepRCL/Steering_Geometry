'''
Paper: Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (NeurIPS 2023)
Reference: 
- Paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html
- Code: https://github.com/likenneth/honest_llama
'''
    

from sklearn.linear_model import LogisticRegression

import torch
from torch import Tensor

from ._base_steer import VecSteer
        

class ITI(VecSteer):
    def __init__(self):
        super().__init__()
        self.clf = LogisticRegression(max_iter = 1000)
    
    @torch.no_grad()
    def fit(
        self,
        pos_X: Tensor,
        neg_X_or_labels: Tensor,
    ) -> 'ITI':
        if len(neg_X_or_labels.shape) == 1:
            self.fit_labels(pos_X, neg_X_or_labels)
        elif neg_X_or_labels.shape[1] == pos_X.shape[1]:
            self.fit_pref(pos_X, neg_X_or_labels)
        else:
            raise ValueError(
                'The shape of unpref Xs or labels must match pref Xs.'
            )
        return self
        
    @torch.no_grad()
    def fit_pref(
        self,
        pos_X: Tensor,
        neg_X: Tensor,
    ) -> None:
        Xs = torch.cat([pos_X, neg_X], dim = 0)
        labels = torch.cat([
            torch.ones(pos_X.shape[0]), 
            torch.zeros(neg_X.shape[0]),
        ], dim = 0)
        self.fit_labels(Xs, labels)
    
    @torch.no_grad()
    def fit_labels(
        self,
        Xs: Tensor,
        labels: Tensor,
    ) -> None:
        Xs_np = Xs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.clf.fit(Xs_np, labels_np)
        self.steer_vec = torch.as_tensor(
            self.clf.coef_.ravel(),
            device = Xs.device,
            dtype = Xs.dtype,
        )
        raw_steer_vec_norm = self.steer_vec.norm().item()
        self.steer_vec = self.steer_vec / raw_steer_vec_norm