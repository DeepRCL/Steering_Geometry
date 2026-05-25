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

TWTM   Theory-Weighted Transfer Magnitude — mean T[A,B] × R[A,B] over
        off-diagonal entries.  This is magnitude-sensitive.

β      Theory-transfer slope from an OLS regression T[A,B] = intercept + βR[A,B].
        This is the accuracy-gain amplitude associated with theory alignment.

CFS    Circumplex Fidelity Score — equal-weight normalised composite in [0,1].

STS    Self-scaled Transfer Stability — compares each observed transfer T[A,B]
       with the self-gain-scaled theoretical expectation R[A,B] × T[A,A].
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.stats

from .circumplex_utils import (
    CIRCUMPLEX_ORDER,
    circular_distance,
    flatten_off_diagonal,
)

_N = len(CIRCUMPLEX_ORDER)


def residualize_transfer_matrix(T_matrix: np.ndarray) -> np.ndarray:
    """Remove method/value main effects from off-diagonal transfer entries.

    The raw transfer matrix often contains a large generic lift:
    some steering values improve almost every evaluation value, and some
    evaluation values improve under almost every steering direction.  To ask
    whether there is remaining circumplex-specific structure, fit the additive
    model

        T[A,B] = intercept + steering_main[A] + eval_main[B] + residual[A,B]

    on the 380 off-diagonal entries and return the residual matrix.  The
    diagonal is set to 0 because all current structure metrics ignore it.
    """
    T = np.asarray(T_matrix, dtype=np.float64)
    n = T.shape[0]
    if T.shape != (n, n):
        raise ValueError(f"Expected a square matrix, got shape {T.shape}")

    rows = []
    y = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = np.zeros(1 + n + n, dtype=np.float64)
            row[0] = 1.0
            row[1 + i] = 1.0
            row[1 + n + j] = 1.0
            rows.append(row)
            y.append(T[i, j])

    X = np.vstack(rows)
    y_arr = np.array(y, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    fitted = X @ coef
    residuals = y_arr - fitted

    out = np.zeros_like(T, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out[i, j] = residuals[k]
            k += 1
    return out


def compute_transfer_summary(T_matrix: np.ndarray) -> dict:
    """Summarise generic transfer effects that can masquerade as structure."""
    T = np.asarray(T_matrix, dtype=np.float64)
    off_diag = flatten_off_diagonal(T)
    diag = np.diag(T)
    return {
        "mean_transfer": float(np.mean(T)),
        "off_diagonal_mean_transfer": float(np.mean(off_diag)),
        "diagonal_mean_transfer": float(np.mean(diag)),
        "off_diagonal_positive_fraction": float(np.mean(off_diag > 0.0)),
        "off_diagonal_negative_fraction": float(np.mean(off_diag < 0.0)),
        "min_transfer": float(np.min(T)),
        "max_transfer": float(np.max(T)),
    }


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


def compute_binned_transfer_profile(T_matrix: np.ndarray) -> Dict[int, dict]:
    """Return magnitude profile statistics for each angular distance bin."""
    bins: Dict[int, list] = {k: [] for k in range(1, 11)}
    for i, va in enumerate(CIRCUMPLEX_ORDER):
        for j, vb in enumerate(CIRCUMPLEX_ORDER):
            if i == j:
                continue
            bins[circular_distance(va, vb)].append(float(T_matrix[i, j]))

    profile: Dict[int, dict] = {}
    for k, vals in bins.items():
        arr = np.asarray(vals, dtype=np.float64)
        profile[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        }
    return profile


def compute_theory_weighted_transfer_magnitude(
    flat_R: np.ndarray,
    flat_T: np.ndarray,
) -> float:
    """Compute mean T[A,B] × R[A,B] over off-diagonal entries."""
    return float(np.mean(np.asarray(flat_T, dtype=np.float64) * np.asarray(flat_R, dtype=np.float64)))


def compute_centered_theory_weighted_transfer_magnitude(
    flat_R: np.ndarray,
    flat_T: np.ndarray,
) -> float:
    """Compute mean centered(T[A,B]) × centered(R[A,B]).

    The 20-value off-diagonal circumplex R values are not exactly mean-zero.
    This covariance-style variant removes broad generic transfer before
    measuring theory-aligned magnitude.
    """
    r = np.asarray(flat_R, dtype=np.float64)
    t = np.asarray(flat_T, dtype=np.float64)
    return float(np.mean((t - np.mean(t)) * (r - np.mean(r))))


def compute_theory_transfer_regression(
    flat_R: np.ndarray,
    flat_T: np.ndarray,
) -> dict:
    """Fit T = intercept + beta * R on off-diagonal entries.

    beta is magnitude-sensitive: it estimates how much empirical transfer
    changes per unit increase in theoretical relatedness.
    """
    x = np.asarray(flat_R, dtype=np.float64)
    y = np.asarray(flat_T, dtype=np.float64)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    resid = y - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "intercept": float(coef[0]),
        "beta": float(coef[1]),
        "r_squared": float(r2),
    }


def compute_self_scaled_transfer_stability(
    observed_matrix: np.ndarray,
    R_matrix: np.ndarray,
    self_gain_matrix: np.ndarray | None = None,
    eps: float = 1e-8,
) -> dict:
    """Measure how closely observed transfer follows self-scaled theory.

    For each ordered off-diagonal pair (A, B), let the self gain be the raw
    diagonal transfer T[A,A].  The expected transfer is

        Expected[A,B] = R[A,B] * T[A,A]

    and the relative error is

        |Observed[A,B] - Expected[A,B]| / |T[A,A]|.

    The absolute denominator keeps the error non-negative even if a method has
    a negative self-gain for some value.  Values with near-zero self gain use
    ``eps`` as the denominator, which intentionally penalises unstable or
    non-steering rows instead of silently dropping pairs.

    For residualized stability, pass the residualized matrix as
    ``observed_matrix`` and the raw matrix as ``self_gain_matrix``.  The current
    residualized matrix has an undefined/zero diagonal by construction, so the
    raw diagonal remains the self-steering scale.
    """
    observed = np.asarray(observed_matrix, dtype=np.float64)
    R = np.asarray(R_matrix, dtype=np.float64)
    self_source = observed if self_gain_matrix is None else np.asarray(self_gain_matrix, dtype=np.float64)
    if observed.shape != R.shape or observed.shape != self_source.shape:
        raise ValueError(
            "observed_matrix, R_matrix, and self_gain_matrix must have the same shape; "
            f"got {observed.shape}, {R.shape}, and {self_source.shape}."
        )

    errors = []
    expected_values = []
    observed_values = []
    for i in range(observed.shape[0]):
        self_gain = float(self_source[i, i])
        denom = max(abs(self_gain), eps)
        for j in range(observed.shape[1]):
            if i == j:
                continue
            expected = float(R[i, j] * self_gain)
            obs = float(observed[i, j])
            errors.append(abs(obs - expected) / denom)
            expected_values.append(expected)
            observed_values.append(obs)

    error_arr = np.asarray(errors, dtype=np.float64)
    exp_stability = np.exp(-error_arr)
    inverse_stability = 1.0 / (1.0 + error_arr)
    return {
        "description": (
            "Mean stability against Expected[A,B] = R[A,B] * T[A,A], with "
            "relative error |Observed[A,B] - Expected[A,B]| / |T[A,A]|."
        ),
        "n_pairs": int(error_arr.size),
        "mean_relative_error": float(np.mean(error_arr)),
        "median_relative_error": float(np.median(error_arr)),
        "stability_exp_neg_error": float(np.mean(exp_stability)),
        "stability_inverse_error": float(np.mean(inverse_stability)),
        "mean_expected_transfer": float(np.mean(expected_values)),
        "mean_observed_transfer": float(np.mean(observed_values)),
    }


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
    include_residualized: bool = True,
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
    binned_profile = compute_binned_transfer_profile(T_matrix)
    aog, t_adj, t_opp = compute_aog(T_matrix)
    twtm = compute_theory_weighted_transfer_magnitude(flat_R, flat_T)
    centered_twtm = compute_centered_theory_weighted_transfer_magnitude(flat_R, flat_T)
    theory_regression = compute_theory_transfer_regression(flat_R, flat_T)
    stability = compute_self_scaled_transfer_stability(T_matrix, R_matrix)
    cfs, ctc_norm, bmd_norm, aog_norm = compute_cfs(ctc_rho, bmd_rho, aog)

    out = {
        "method": method_name,
        "alpha": alpha,
        "transfer_summary": compute_transfer_summary(T_matrix),
        "CTC_rho": ctc_rho,
        "CTC_pvalue": ctc_pvalue,
        "BMD_rho": bmd_rho,
        "BMD_pvalue": bmd_pvalue,
        "bin_means": {str(k): v for k, v in bin_means.items()},
        "binned_profile": {str(k): v for k, v in binned_profile.items()},
        "AOG": aog,
        "T_adjacent": t_adj,
        "T_opposite": t_opp,
        "TWTM": twtm,
        "centered_TWTM": centered_twtm,
        "theory_regression": theory_regression,
        "self_scaled_transfer_stability": stability,
        "CFS": cfs,
        "CTC_norm": ctc_norm,
        "BMD_norm": bmd_norm,
        "AOG_norm": aog_norm,
    }

    if include_residualized:
        T_resid = residualize_transfer_matrix(T_matrix)
        flat_T_resid = flatten_off_diagonal(T_resid)
        resid_ctc_rho, resid_ctc_pvalue = compute_ctc_rho(flat_R, flat_T_resid)
        resid_bmd_rho, resid_bmd_pvalue, resid_bin_means = compute_bmd_rho(T_resid)
        resid_binned_profile = compute_binned_transfer_profile(T_resid)
        resid_aog, resid_t_adj, resid_t_opp = compute_aog(T_resid)
        resid_twtm = compute_theory_weighted_transfer_magnitude(flat_R, flat_T_resid)
        resid_centered_twtm = compute_centered_theory_weighted_transfer_magnitude(flat_R, flat_T_resid)
        resid_theory_regression = compute_theory_transfer_regression(flat_R, flat_T_resid)
        resid_stability = compute_self_scaled_transfer_stability(
            T_resid,
            R_matrix,
            self_gain_matrix=T_matrix,
        )
        resid_cfs, resid_ctc_norm, resid_bmd_norm, resid_aog_norm = compute_cfs(
            resid_ctc_rho,
            resid_bmd_rho,
            resid_aog,
        )
        out["residualized"] = {
            "description": (
                "Metrics after regressing off-diagonal T[A,B] on steering-value "
                "main effects, eval-value main effects, and a global intercept."
            ),
            "transfer_summary": compute_transfer_summary(T_resid),
            "CTC_rho": resid_ctc_rho,
            "CTC_pvalue": resid_ctc_pvalue,
            "BMD_rho": resid_bmd_rho,
            "BMD_pvalue": resid_bmd_pvalue,
            "bin_means": {str(k): v for k, v in resid_bin_means.items()},
            "binned_profile": {str(k): v for k, v in resid_binned_profile.items()},
            "AOG": resid_aog,
            "T_adjacent": resid_t_adj,
            "T_opposite": resid_t_opp,
            "TWTM": resid_twtm,
            "centered_TWTM": resid_centered_twtm,
            "theory_regression": resid_theory_regression,
            "self_scaled_transfer_stability": resid_stability,
            "CFS": resid_cfs,
            "CTC_norm": resid_ctc_norm,
            "BMD_norm": resid_bmd_norm,
            "AOG_norm": resid_aog_norm,
        }

    return out
