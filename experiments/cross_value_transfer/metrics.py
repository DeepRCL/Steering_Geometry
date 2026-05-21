"""
Metric functions for the Cross-Value Steering Transfer experiment.

All functions operate on plain numpy arrays so they are reusable regardless
of the concrete steering method.  No torch dependencies here.

Metrics
-------
CTC-ρ  Circumplex Transfer Correlation — Spearman ρ between the 380
        off-diagonal T entries and the 380 corresponding theoretical R values.

BMD-ρ  Bin-wise Monotonic Decay — Spearman ρ between angular-step bins
        k ∈ {1..10} and the bin mean transfer Δacc, negated so higher = better.

AOG    Adjacent–Opposite Gap — mean T for k ≤ 2 minus mean T for k ≥ 9.

CFS    Circumplex Fidelity Score — equal-weight normalised composite in [0,1].
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.stats

from .circumplex_utils import (
    CIRCUMPLEX_ORDER,
    circular_distance,
    flatten_off_diagonal,
    get_off_diagonal_pairs,
)

_N = len(CIRCUMPLEX_ORDER)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1: Circumplex Transfer Correlation (CTC-ρ)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ctc_rho(
    flat_R: np.ndarray,
    flat_T: np.ndarray,
) -> Tuple[float, float]:
    """Spearman ρ between flat_R and flat_T (both length-380 arrays).

    Returns
    -------
    (CTC_rho, CTC_pvalue)
    """
    rho, pvalue = scipy.stats.spearmanr(flat_R, flat_T)
    return float(rho), float(pvalue)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2: Bin-wise Monotonic Decay (BMD-ρ)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bmd_rho(
    T_matrix: np.ndarray,
) -> Tuple[float, float, Dict[int, float]]:
    """Compute Bin-wise Monotonic Decay metric.

    For each angular step k ∈ {1..10} collect all (A, B) pairs with
    circular_distance(A, B) == k, take the mean T[A,B], then compute
    Spearman ρ between [1..10] and [mu_1..mu_10].  The raw ρ is negated
    before returning so that +1.0 means perfect monotonic decay (desirable).

    Parameters
    ----------
    T_matrix:
        (20, 20) float array where T[i, j] = steered_acc_on_B − baseline_acc_on_B
        with steering value A = CIRCUMPLEX_ORDER[i], eval value B = CIRCUMPLEX_ORDER[j].

    Returns
    -------
    (BMD_rho_reported, BMD_pvalue, bin_means)
        BMD_rho_reported is negated raw Spearman ρ; higher = better.
        bin_means is a dict {k: mu_k} for k in 1..10.
    """
    # Collect T values per bin
    bins: Dict[int, list] = {k: [] for k in range(1, 11)}
    for i, va in enumerate(CIRCUMPLEX_ORDER):
        for j, vb in enumerate(CIRCUMPLEX_ORDER):
            if i == j:
                continue
            k = circular_distance(va, vb)
            bins[k].append(T_matrix[i, j])

    bin_means: Dict[int, float] = {k: float(np.mean(vals)) for k, vals in bins.items()}

    ks = list(range(1, 11))
    mu_values = [bin_means[k] for k in ks]

    raw_rho, pvalue = scipy.stats.spearmanr(ks, mu_values)
    # Negate: perfect monotonic decay (mu decreasing with k) gives raw_rho = -1
    # → reported = +1 (better)
    bmd_rho_reported = -float(raw_rho)

    return bmd_rho_reported, float(pvalue), bin_means


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3: Adjacent–Opposite Gap (AOG)
# ─────────────────────────────────────────────────────────────────────────────

def compute_aog(
    T_matrix: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute the Adjacent–Opposite Gap.

    T_adjacent = mean T[A,B] for pairs with circular_distance ≤ 2 (k=1,2 → 80 pairs)
    T_opposite = mean T[A,B] for pairs with circular_distance ≥ 9 (k=9,10 → 60 pairs)
    AOG = T_adjacent − T_opposite

    Accuracies are assumed to be stored as fractions in [0, 1].

    Returns
    -------
    (AOG, T_adjacent, T_opposite)
    """
    adj_vals: list = []
    opp_vals: list = []
    for i, va in enumerate(CIRCUMPLEX_ORDER):
        for j, vb in enumerate(CIRCUMPLEX_ORDER):
            if i == j:
                continue
            k = circular_distance(va, vb)
            if k <= 2:
                adj_vals.append(T_matrix[i, j])
            elif k >= 9:
                opp_vals.append(T_matrix[i, j])

    t_adj = float(np.mean(adj_vals)) if adj_vals else 0.0
    t_opp = float(np.mean(opp_vals)) if opp_vals else 0.0
    aog = t_adj - t_opp
    return aog, t_adj, t_opp


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE: Circumplex Fidelity Score (CFS)
# ─────────────────────────────────────────────────────────────────────────────

_AOG_SCALE = 0.30  # practical normalisation constant


def compute_cfs(
    ctc_rho: float,
    bmd_rho_reported: float,
    aog: float,
) -> Tuple[float, float, float, float]:
    """Compute the Circumplex Fidelity Score and its normalised components.

    Normalisation:
        CTC_norm = (CTC_rho + 1) / 2          # [-1,1] → [0,1]
        BMD_norm = (BMD_rho_reported + 1) / 2  # [-1,1] → [0,1]
        AOG_norm = clip((AOG / 0.30 + 1) / 2, 0, 1)
            → AOG = +0.30 maps to 1.0, AOG = 0.0 to 0.5, AOG = -0.30 to 0.0

    CFS = (CTC_norm + BMD_norm + AOG_norm) / 3  (equal weights)

    Returns
    -------
    (CFS, CTC_norm, BMD_norm, AOG_norm)
    """
    ctc_norm = (ctc_rho + 1.0) / 2.0
    bmd_norm = (bmd_rho_reported + 1.0) / 2.0
    aog_norm = float(np.clip((aog / _AOG_SCALE + 1.0) / 2.0, 0.0, 1.0))
    cfs = (ctc_norm + bmd_norm + aog_norm) / 3.0
    return float(cfs), float(ctc_norm), float(bmd_norm), float(aog_norm)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute everything from T and R matrices
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    T_matrix: np.ndarray,
    R_matrix: np.ndarray,
    method_name: str = "",
    alpha: float = 0.0,
) -> dict:
    """Run all three metrics and CFS, returning a metrics.json-ready dict.

    Parameters
    ----------
    T_matrix:
        (20, 20) float array of Δacc values (fractions, not percentages).
    R_matrix:
        (20, 20) float array of theoretical relationship scores.
    method_name:
        Human-readable method identifier stored in the output dict.
    alpha:
        Steering strength used for this run.

    Returns
    -------
    dict matching the metrics.json schema from Section 4 of the spec.
    """
    flat_R = flatten_off_diagonal(R_matrix)
    flat_T = flatten_off_diagonal(T_matrix)

    ctc_rho, ctc_pvalue = compute_ctc_rho(flat_R, flat_T)
    bmd_rho, bmd_pvalue, bin_means = compute_bmd_rho(T_matrix)
    aog, t_adj, t_opp = compute_aog(T_matrix)
    cfs, ctc_norm, bmd_norm, aog_norm = compute_cfs(ctc_rho, bmd_rho, aog)

    return {
        "method": method_name,
        "alpha": alpha,
        "CTC_rho": ctc_rho,
        "CTC_pvalue": ctc_pvalue,
        "BMD_rho": bmd_rho,
        "BMD_pvalue": bmd_pvalue,
        "bin_means": {str(k): v for k, v in bin_means.items()},
        "AOG": aog,
        "T_adjacent": t_adj,
        "T_opposite": t_opp,
        "CFS": cfs,
        "CTC_norm": ctc_norm,
        "BMD_norm": bmd_norm,
        "AOG_norm": aog_norm,
    }
