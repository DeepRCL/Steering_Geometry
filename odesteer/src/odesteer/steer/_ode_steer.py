'''
Proposed method by our paper
'''

from abc import abstractmethod
from typing import Literal

import torch
from torch import Tensor
from torchdiffeq import odeint

from ._base_steer import Steer
from ..utils.kernels import KernelClassifier, RFFClassifier
from ..utils.kernels import NormedPolyClassifier


class BaseODESteer(Steer):
    def __init__(
        self, 
        solver: Literal['euler', 'midpoint', 'rk4'] = 'euler',
        steps: int = 10,
        **kwargs
    ):
        super().__init__()
        self.solver = solver
        self.steps = steps
        self.clf = self._init_clf(**kwargs)
                
    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> 'BaseODESteer':
        self.clf.fit(pos_X, neg_X_or_labels)
        return self
    
    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if T == 0.: 
            return X
        return odeint(
            func = lambda t, state: self.vector_field(state), 
            y0 = X, 
            t = torch.tensor([0., T], device = X.device), 
            method = self.solver, options = {'step_size': T / self.steps}
        )[1]
    
    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        raw_grad = self.clf.grad(X)
        return raw_grad / (raw_grad.norm(dim = -1, keepdim = True) + 1e-10)
    
    @abstractmethod
    def _init_clf(self, **kwargs) -> KernelClassifier:
        raise NotImplementedError
        


class ODESteer(BaseODESteer):
    '''
    ODESteer used in the paper with NormedPolyCntSketch classifier
    '''
    def _init_clf(self, **kwargs) -> NormedPolyClassifier:
        return NormedPolyClassifier(**kwargs)
    
    
class RFFODESteer(BaseODESteer):
    '''
    Ablation study for ODESteer with RFF classifier
    '''
    def _init_clf(self, **kwargs) -> RFFClassifier:
        return RFFClassifier(**kwargs)
    