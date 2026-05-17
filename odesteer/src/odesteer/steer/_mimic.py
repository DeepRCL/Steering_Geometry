'''
Paper: Representation Surgery: Theory and Practice of Affine Steering (ICML 2024)
Reference: 
- Paper: https://openreview.net/forum?id=GwA4go0Mw4
- Code: https://github.com/shauli-ravfogel/affine-steering
'''

import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
import ot

from ._base_steer import Steer


class MiMiC(Steer):
    def __init__(self):
        super().__init__()
        self.clf = LogisticRegression(max_iter = 1000)
        self.ot_linear = ot.da.LinearTransport(reg = 1e-2)
    
    def fit(self, pos_X: Tensor, neg_X: Tensor):
        train_X = torch.cat([pos_X, neg_X], dim = 0).cpu().numpy()
        train_Y = torch.cat([torch.ones(len(pos_X)), torch.zeros(len(neg_X))], dim = 0).numpy()
        self.clf.fit(train_X, train_Y)
        self.ot_linear.fit(Xs = neg_X.cpu().numpy(), Xt = pos_X.cpu().numpy())
        return self
    
    def steer(self, X: Tensor, T: float = 1.0):
        y_pred = self.clf.predict(X.detach().cpu().numpy())
        steered_X = X.detach().cpu().numpy().copy()
        steered_X[y_pred == 0] = self.ot_linear.transform(steered_X[y_pred == 0])
        return torch.as_tensor(steered_X, device = X.device, dtype = X.dtype)
    
    def vector_field(self, X: Tensor):
        return