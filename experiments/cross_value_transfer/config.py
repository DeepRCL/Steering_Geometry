"""
Configuration dataclass for the Cross-Value Steering Transfer experiment.

All path fields are relative to the project root unless they are absolute.
Use ``TransferExperimentConfig.resolve_paths(project_root)`` to make them
absolute before passing to the runner.

Serialisation:
    config.save("path/to/config.json")      # write JSON
    config = TransferExperimentConfig.load("path/to/config.json")
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TransferExperimentConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name: Optional[str] = None
    """HuggingFace model identifier (e.g. ``"Qwen/Qwen3.5-9B"``).
    If None, the runner reads ``model_name`` from the first method run dir's
    ``config.json``."""

    device: str = "auto"
    """Torch device string or ``"auto"`` to use CUDA when available."""

    # ── Steering ───────────────────────────────────────────────────────────
    alpha: float = 20.0
    """Steering strength multiplier α."""

    # ── CAA-specific ───────────────────────────────────────────────────────
    caa_run_dir: str = ""
    """Path to the model-specific CAA output directory that directly
    contains ``vectors/`` and ``config.json``
    (e.g. ``CAA/Geometry/outputs/qwen3_5_9b/Qwen__Qwen3.5-9B``)."""

    caa_layer: Optional[int] = None
    """Layer index to load for CAA vectors.  If None, auto-discovered from
    ``{caa_run_dir}/layer_selection/selected_layer.json``."""

    caa_vector_source: str = "vectors"
    """Which vectors to evaluate: ``"vectors"`` for ordinary layer_N.pt CAA
    vectors, or ``"geometry_vectors"`` for transformed geometry vectors."""

    # ── SphericalSteer-specific ────────────────────────────────────────────
    spherical_run_dir: str = ""
    """Path to the model-specific SphericalSteer output directory that directly
    contains ``vectors/`` and ``config.json``."""

    spherical_layer: Optional[int] = None
    """Layer index to load for SphericalSteer vectors.  If None, read from
    ``config.json`` layer_override or ``geometry_vectors/manifest.json``."""

    spherical_kappa: Optional[float] = None
    """Optional override for SphericalSteer vMF concentration.  If None, read
    from ``{spherical_run_dir}/config.json``."""

    spherical_beta: Optional[float] = None
    """Optional override for the SphericalSteer trigger threshold.  If None,
    read from ``{spherical_run_dir}/config.json``."""

    spherical_steer_position: Optional[str] = None
    """Optional override for SphericalSteer hook position: ``"last"`` or
    ``"all"``.  If None, read from ``{spherical_run_dir}/config.json``."""

    # ── Evaluation dataset ─────────────────────────────────────────────────
    eval_dataset_path: str = (
        "experiments/cross_value_transfer/data/"
        "touche_gemma4-v2_remaining-validated-final.csv"
    )
    """Path to the held-out MCQ evaluation CSV (Touché/validated)."""

    n_eval_samples: int = 100
    """Number of evaluation instances to sample per Schwartz value.
    The actual count may be lower for values with fewer ``caa_suitable`` rows."""

    eval_splits: Optional[List[str]] = field(default_factory=lambda: ["validation", "test"])
    """CSV split labels to evaluate on.  Set to None or [] to use all rows."""

    eval_split_fraction: float = 0.1
    """Fallback held-out fraction for datasets without a ``split`` column.
    Mirrors ``CAA.Geometry.data_loader.DataLoader(eval_split=...)``."""

    seed: int = 42
    """Random seed for reproducible eval-instance sampling and pos_is_a assignment."""

    # ── Relations ──────────────────────────────────────────────────────────
    relations_path: str = "CAA/value_data/schwartz_relations-new.json"
    """Path to the Schwartz theoretical relationship JSON."""

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "experiments/cross_value_transfer/outputs"
    """Root directory for all experiment outputs."""

    # ── Methods ────────────────────────────────────────────────────────────
    methods: List[str] = field(default_factory=lambda: ["caa"])
    """Ordered list of method names to evaluate (e.g. ``["caa"]``)."""

    force_recompute: bool = False
    """If True, recompute T matrix and metrics even if outputs already exist."""

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TransferExperimentConfig":
        """Load config from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop unknown keys for forward-compatibility
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    def resolve_paths(self, project_root: str | Path) -> "TransferExperimentConfig":
        """Return a copy of this config with all relative paths made absolute."""
        root = Path(project_root)

        def abs_if_relative(p: str) -> str:
            if not p:
                return p
            pp = Path(p)
            return str(root / pp) if not pp.is_absolute() else p

        return TransferExperimentConfig(
            model_name=self.model_name,
            device=self.device,
            alpha=self.alpha,
            caa_run_dir=abs_if_relative(self.caa_run_dir),
            caa_layer=self.caa_layer,
            caa_vector_source=self.caa_vector_source,
            spherical_run_dir=abs_if_relative(self.spherical_run_dir),
            spherical_layer=self.spherical_layer,
            spherical_kappa=self.spherical_kappa,
            spherical_beta=self.spherical_beta,
            spherical_steer_position=self.spherical_steer_position,
            eval_dataset_path=abs_if_relative(self.eval_dataset_path),
            n_eval_samples=self.n_eval_samples,
            eval_splits=None if self.eval_splits is None else list(self.eval_splits),
            eval_split_fraction=self.eval_split_fraction,
            seed=self.seed,
            relations_path=abs_if_relative(self.relations_path),
            output_dir=abs_if_relative(self.output_dir),
            methods=list(self.methods),
            force_recompute=self.force_recompute,
        )
