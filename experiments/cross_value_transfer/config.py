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
    """Fallback steering strength multiplier α."""

    use_method_default_alphas: bool = True
    """If True, the CLI may attach method-specific best-known α values.
    Passing ``--alpha`` disables this and forces the same α for every method."""

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

    # ── BiPO / optimized-vector-specific ──────────────────────────────────
    bipo_run_dir: str = ""
    """Path to the model-specific BiPO/optimized-vector output directory that
    directly contains ``vectors/`` and ``config.json``."""

    bipo_layer: Optional[int] = None
    """Layer index to load for BiPO vectors.  If None, read from
    ``config.json`` layer_override or ``geometry_vectors/manifest.json``."""

    bipo_steer_position: Optional[str] = None
    """Optional override for BiPO hook position: ``"all"`` or ``"last"``.
    If None, read ``opt_steer_position`` from ``{bipo_run_dir}/config.json``."""

    bipo_vector_source: str = "vectors"
    """Which BiPO vectors to evaluate: ``"vectors"`` for ordinary learned
    vectors, or ``"geometry_vectors"`` for transformed geometry vectors."""

    bipo_normalize_vectors: bool = False
    """If True, unit-normalise BiPO vectors before steering.  Defaults to False
    to match the original optimized-vector evaluator."""

    # ── SparseCAA-specific ─────────────────────────────────────────────────
    sparsecaa_run_dir: str = ""
    """Path to the SparseCAA output directory that directly contains
    ``sparse_vectors/``, ``pipeline_config.json``, and ``sae_finetuned.pt``."""

    sparsecaa_layer: Optional[int] = None
    """Layer index whose MLP output SparseCAA hooks. If None, read
    ``mlp_layer`` from ``{sparsecaa_run_dir}/pipeline_config.json``."""

    sparsecaa_sae_path: Optional[str] = None
    """Optional override for the SAE checkpoint. If None, use
    ``{sparsecaa_run_dir}/sae_finetuned.pt`` when present."""

    sparsecaa_normalize_vectors: bool = False
    """If True, unit-normalise SparseCAA sparse vectors before steering.
    Defaults to False to match the original SparseCAA evaluator."""

    # ── QwenScopeCAA-specific ───────────────────────────────────────────────
    qwenscope_run_dir: str = ""
    """Path to the QwenScopeCAA output directory that directly contains
    ``sparse_vectors_caa_base/``, ``pipeline_config.json``, and
    ``sae_finetuned_layer{layer}.pt``."""

    qwenscope_layer: Optional[int] = None
    """Transformer layer whose post-layer residual stream QwenScopeCAA hooks.
    If None, read ``layer`` from ``{qwenscope_run_dir}/pipeline_config.json``."""

    qwenscope_sae_path: Optional[str] = None
    """Optional override for the Qwen-Scope SAE checkpoint. If None, use the
    run's fine-tuned SAE checkpoint."""

    qwenscope_vector_source: str = "auto"
    """Which QwenScope persona vectors to load: ``"auto"`` prefers
    ``sparse_vectors_caa_base/`` and falls back to ``sparse_vectors/``."""

    qwenscope_normalize_vectors: bool = False
    """If True, unit-normalise QwenScope persona vectors before steering.
    Defaults to False to match the original evaluator."""

    # ── ODESteer-specific ──────────────────────────────────────────────────
    odesteer_run_dir: str = ""
    """Optional ODESteer Schwartz output directory.  Required for
    ``odesteer_vectors``, which loads saved displacement vectors."""

    odesteer_layer: Optional[int] = 18
    """Layer index for ODESteer.  Defaults to 18; if set to None, falls back
    to ``caa_layer``."""

    odesteer_type: str = "ODESteer"
    """ODESteer class name: ``ODESteer`` or ``StepODESteer``."""

    odesteer_solver: str = "euler"
    odesteer_steps: int = 10
    odesteer_n_components: int = 8000
    odesteer_degree: int = 2
    odesteer_gamma: float = 0.1
    odesteer_coef0: float = 1.0
    odesteer_lin_clf_type: str = "lr"
    # ── llm-steering-opt-specific ──────────────────────────────────────────
    llm_steering_opt_run_dir: str = ""
    """Path to the llm-steering-opt run directory that directly contains
    ``vectors/manifest.json`` and usually ``config.json``."""

    llm_steering_opt_layer: Optional[int] = None
    """Layer index for llm-steering-opt vectors. If None, read from
    ``vectors/manifest.json``."""

    llm_steering_opt_normalize_vectors: bool = False
    """If True, L2-normalise llm-steering-opt vectors before applying alpha.
    False mirrors llm-steering-opt's native evaluation, where vector norm is
    part of the learned steering vector."""

    # ── COLD-Steer-specific ────────────────────────────────────────────────
    cold_steer_run_dir: str = ""
    """Path to the COLD-Steer Schwartz run directory containing
    ``vectors/manifest.json`` and usually ``config.json``."""

    cold_steer_layer: Optional[int] = None
    """Layer index for COLD-Steer vectors. If None, infer from manifest/config."""

    cold_steer_position: str = "all"
    """COLD-Steer hook position: ``"all"`` or ``"last"``."""

    # ── Evaluation dataset ─────────────────────────────────────────────────
    eval_dataset_path: str = (
        "experiments/cross_value_transfer/data/"
        "final_dataset_200.csv"
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
            use_method_default_alphas=self.use_method_default_alphas,
            caa_run_dir=abs_if_relative(self.caa_run_dir),
            caa_layer=self.caa_layer,
            caa_vector_source=self.caa_vector_source,
            spherical_run_dir=abs_if_relative(self.spherical_run_dir),
            spherical_layer=self.spherical_layer,
            spherical_kappa=self.spherical_kappa,
            spherical_beta=self.spherical_beta,
            spherical_steer_position=self.spherical_steer_position,
            bipo_run_dir=abs_if_relative(self.bipo_run_dir),
            bipo_layer=self.bipo_layer,
            bipo_steer_position=self.bipo_steer_position,
            bipo_vector_source=self.bipo_vector_source,
            bipo_normalize_vectors=self.bipo_normalize_vectors,
            sparsecaa_run_dir=abs_if_relative(self.sparsecaa_run_dir),
            sparsecaa_layer=self.sparsecaa_layer,
            sparsecaa_sae_path=(
                None if self.sparsecaa_sae_path is None else abs_if_relative(self.sparsecaa_sae_path)
            ),
            sparsecaa_normalize_vectors=self.sparsecaa_normalize_vectors,
            qwenscope_run_dir=abs_if_relative(self.qwenscope_run_dir),
            qwenscope_layer=self.qwenscope_layer,
            qwenscope_sae_path=(
                None if self.qwenscope_sae_path is None else abs_if_relative(self.qwenscope_sae_path)
            ),
            qwenscope_vector_source=self.qwenscope_vector_source,
            qwenscope_normalize_vectors=self.qwenscope_normalize_vectors,
            odesteer_run_dir=abs_if_relative(self.odesteer_run_dir),
            odesteer_layer=self.odesteer_layer,
            odesteer_type=self.odesteer_type,
            odesteer_solver=self.odesteer_solver,
            odesteer_steps=self.odesteer_steps,
            odesteer_n_components=self.odesteer_n_components,
            odesteer_degree=self.odesteer_degree,
            odesteer_gamma=self.odesteer_gamma,
            odesteer_coef0=self.odesteer_coef0,
            odesteer_lin_clf_type=self.odesteer_lin_clf_type,
            llm_steering_opt_run_dir=abs_if_relative(self.llm_steering_opt_run_dir),
            llm_steering_opt_layer=self.llm_steering_opt_layer,
            llm_steering_opt_normalize_vectors=self.llm_steering_opt_normalize_vectors,
            cold_steer_run_dir=abs_if_relative(self.cold_steer_run_dir),
            cold_steer_layer=self.cold_steer_layer,
            cold_steer_position=self.cold_steer_position,
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
