"""
Abstract base class for steering methods in the Cross-Value Transfer experiment.

Each concrete subclass encapsulates:
  - How pre-computed steering vectors are loaded from disk.
  - How a forward hook is installed / removed for a given steering vector.

This interface is intentionally minimal so that methods such as plain CAA,
SAE-based steering (QwenScopeCAA), spherical steering, etc. can all be
plugged in by implementing three abstract methods.

Note: This ABC is experiment-scoped and is separate from the internal
      CAA/Geometry/steering/base.py which is tightly coupled to the
      activation-extraction pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class SteeringMethod(ABC):
    """Abstract steering method for the cross-value transfer experiment.

    Subclass contract
    -----------------
    1. ``name`` — unique identifier used for output directory naming
       (e.g. ``"caa"``, ``"qwenscope_caa"``).

    2. ``layer`` — the model layer index at which this method applies steering.
       Implementations may resolve this lazily (e.g. by reading
       ``selected_layer.json`` from the run directory).

    3. ``load_vectors()`` — load and return {value_name: steering_tensor} for
       all 20 Schwartz values.  Vectors should be returned on CPU; the runner
       will move them to the appropriate device before calling ``apply_hook``.
       Implementations should normalise vectors to unit length here.

    4. ``apply_hook(model_info, vector, alpha)`` — register a forward hook that
       injects ``alpha * vector`` (or the method's equivalent) into the
       residual stream at the layer returned by ``self.layer``.  Returns an
       opaque handle that will be passed to ``remove_hook``.

    5. ``remove_hook(handle)`` — remove the hook registered by ``apply_hook``.
    """

    # ── identity ─────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this method (used in output paths)."""

    @property
    @abstractmethod
    def layer(self) -> int:
        """Model layer index at which steering is applied."""

    # ── vectors ──────────────────────────────────────────────────────────────

    @abstractmethod
    def load_vectors(self) -> Dict[str, torch.Tensor]:
        """Load per-value steering vectors from disk.

        Returns
        -------
        dict mapping each of the 20 Schwartz value names to a 1-D CPU tensor
        of shape ``(hidden_dim,)``.  Vectors should be unit-normalised.
        """

    # ── hooks ────────────────────────────────────────────────────────────────

    @abstractmethod
    def apply_hook(
        self,
        model_info: Any,
        vector: torch.Tensor,
        alpha: float,
    ) -> Any:
        """Register a steering hook on the model.

        Parameters
        ----------
        model_info:
            ``ModelInfo`` object from ``CAA/Geometry/model_loader.py``.
        vector:
            Unit-normalised steering vector (CPU tensor; implementations
            should move to ``model_info.device`` internally).
        alpha:
            Steering strength multiplier.

        Returns
        -------
        An opaque handle (or list of handles) to be passed to ``remove_hook``.
        """

    @abstractmethod
    def remove_hook(self, handle: Any) -> None:
        """Remove the hook(s) returned by ``apply_hook``."""
