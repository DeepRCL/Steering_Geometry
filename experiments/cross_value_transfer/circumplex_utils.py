"""
Shared utilities for the Schwartz circumplex structure used across the
cross-value transfer experiment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Circumplex order (counter-clockwise, 20 refined Schwartz values)
# ─────────────────────────────────────────────────────────────────────────────

CIRCUMPLEX_ORDER: List[str] = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: dependability",
    "Benevolence: caring",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity",
]

CIRCUMPLEX_IDX: Dict[str, int] = {v: i for i, v in enumerate(CIRCUMPLEX_ORDER)}

# ─────────────────────────────────────────────────────────────────────────────
# Higher-order group membership and block boundaries on the ordered axis
# ─────────────────────────────────────────────────────────────────────────────

HO_GROUPS: Dict[str, List[str]] = {
    "Openness to Change": [
        "Self-direction: thought",
        "Self-direction: action",
        "Stimulation",
        "Hedonism",
    ],
    "Self-Enhancement": [
        "Achievement",
        "Power: dominance",
        "Power: resources",
        "Face",
    ],
    "Conservation": [
        "Security: personal",
        "Security: societal",
        "Tradition",
        "Conformity: rules",
        "Conformity: interpersonal",
        "Humility",
    ],
    "Self-Transcendence": [
        "Benevolence: dependability",
        "Benevolence: caring",
        "Universalism: concern",
        "Universalism: nature",
        "Universalism: tolerance",
        "Universalism: objectivity",
    ],
}

# (start_idx_inclusive, end_idx_inclusive) for each HO group in CIRCUMPLEX_ORDER
HO_BLOCK_BOUNDARIES: List[Tuple[int, int]] = [
    (0, 3),   # Openness to Change
    (4, 7),   # Self-Enhancement
    (8, 13),  # Conservation
    (14, 19), # Self-Transcendence
]

_N = len(CIRCUMPLEX_ORDER)  # 20


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def circular_distance(a: str, b: str) -> int:
    """Minimum number of steps between values a and b on the 20-point circumplex.

    Returns an integer in [0, 10]. k=0 iff a==b; k=10 is the antipodal pair.
    """
    i, j = CIRCUMPLEX_IDX[a], CIRCUMPLEX_IDX[b]
    diff = abs(i - j)
    return min(diff, _N - diff)


def load_R_matrix(relations_path: Path) -> np.ndarray:
    """Load the basic_value_relationship_matrix from schwartz_relations-new.json.

    Returns a (20, 20) float64 numpy array where R[i, j] = cos(k × 18°) for
    the circular distance k between CIRCUMPLEX_ORDER[i] and CIRCUMPLEX_ORDER[j].
    Rows and columns are ordered by CIRCUMPLEX_ORDER.
    """
    with open(relations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw: Dict[str, Dict[str, float]] = data["basic_value_relationship_matrix"]
    R = np.zeros((_N, _N), dtype=np.float64)
    for i, va in enumerate(CIRCUMPLEX_ORDER):
        for j, vb in enumerate(CIRCUMPLEX_ORDER):
            R[i, j] = raw[va][vb]
    return R


def get_off_diagonal_pairs() -> List[Tuple[str, str]]:
    """Return all 380 ordered (A, B) pairs with A ≠ B in consistent order.

    Iteration order: outer loop over CIRCUMPLEX_ORDER (A), inner loop over
    CIRCUMPLEX_ORDER (B), skipping diagonal entries.
    """
    pairs: List[Tuple[str, str]] = []
    for va in CIRCUMPLEX_ORDER:
        for vb in CIRCUMPLEX_ORDER:
            if va != vb:
                pairs.append((va, vb))
    return pairs


def flatten_off_diagonal(matrix_20x20: np.ndarray) -> np.ndarray:
    """Extract the 380 off-diagonal entries in get_off_diagonal_pairs() order.

    Parameters
    ----------
    matrix_20x20:
        A (20, 20) numpy array indexed by CIRCUMPLEX_ORDER.

    Returns
    -------
    np.ndarray of shape (380,).
    """
    out = []
    for i in range(_N):
        for j in range(_N):
            if i != j:
                out.append(matrix_20x20[i, j])
    return np.array(out, dtype=np.float64)
