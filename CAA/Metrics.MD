# Steering Geometry Metrics

This note explains the geometry metrics used in the `CAA`, `SparseCAA`, and `llm-steering-opt` pipelines for analyzing whether steering vectors reflect Schwartz value structure.

The intended audience is a paper co-author who needs to understand what each metric means, how it is computed, and how it should be interpreted in the paper.

## Overview

The geometry analysis starts from one steering vector per Schwartz value:

- `v_1, v_2, ..., v_20`

These vectors may come from:

- raw dense CAA vectors
- mean-centered dense vectors
- centered-renormalized dense vectors
- sparse SAE vectors
- optimization-based steering vectors

For geometry analysis, the code first computes a unit-length version of each vector:

- `u_i = v_i / ||v_i||`

Then it builds the empirical similarity matrix:

- `E[i, j] = cosine(u_i, u_j)`

This matrix is the basis for almost all reported geometry metrics.

Important:

- The core metrics are computed in the original high-dimensional steering space.
- They are **not** computed from the 2D UMAP/PCA/t-SNE/MDS plots.
- The 2D plots are only visualizations of that high-dimensional structure.

## Theoretical Targets

There are three different kinds of theory target used by the metrics.

### 1. Theoretical Similarity Matrix

This comes from `relations_path`, typically one of:

- `CAA/value_data/schwartz_relations.json`
- `CAA/value_data/schwartz_relations-new.json`

The matrix is:

- `T[i, j]`

Depending on the chosen JSON:

- the old file uses a coarse discrete target in `{ -1, 0, 1 }`
- the new file uses a continuous circumplex cosine target derived from angular distance on the Schwartz circle

### 2. Circular Step Distance

This uses only the fixed `SCHWARTZ_CIRCUMPLEX_ORDER`.

For two values at positions `i` and `j` on the 20-item ring:

- `step(i, j) = min(|i - j|, 20 - |i - j|)`

So the possible step distances are:

- `1, 2, 3, ..., 10`

Interpretation:

- `1` means immediate neighbors on the circumplex
- `10` means exact opposites on the circle

### 3. Hierarchical Theory Distance

This is a theory-aware categorical distance that does not come from the raw JSON matrix.

It is defined as:

- `1` if two values are in the same lower-order family
- `2` if they are in the same higher-order group but not the same lower-order family
- `5` if they are neither clearly aligned nor opposite
- `10` if they belong to opposite higher-order groups

Boundary values are treated as members of two higher-order groups:

- `Hedonism`: `Openness to Change` and `Self-Enhancement`
- `Face`: `Self-Enhancement` and `Conservation`
- `Humility`: `Conservation` and `Self-Transcendence`

This hierarchical metric was introduced to be more faithful to Schwartz theory than a simple four-group collapse.

## Main Metrics

## 1. `spearman_rho`

### Definition

Take the upper triangle of the empirical and theoretical similarity matrices:

- `E_flat`
- `T_flat`

Then compute:

- `Spearman(E_flat, T_flat)`

### What it measures

This is a global rank-based alignment metric.

It asks:

- if theory says pair A should be more similar than pair B, does the model rank them that way too?

### Why it matters

This is the most general theory-alignment metric.

It does not assume the relationship must be linear. It only cares about whether the ordering of pairwise relationships matches theory.

### Interpretation

- higher positive values mean better theory-consistent ordering
- values near `0` mean little alignment
- negative values mean anti-alignment

### P-value

Yes. The pipelines report:

- `spearman_p_value`

This comes from the standard Spearman test over the 190 unique value pairs.

### Caveat

The 190 pairwise observations are not fully independent, because many pairs share values. So the p-value is useful, but effect size should be emphasized more than significance.

## 2. `pearson_r`

### Definition

Again use:

- `E_flat`
- `T_flat`

Then compute:

- `Pearson(E_flat, T_flat)`

### What it measures

This is a global linear alignment metric.

It asks:

- not only whether the ordering matches theory
- but whether the actual similarity values scale proportionally with theory

### Why it matters

This metric becomes especially meaningful when using the continuous cosine-based theory file:

- `schwartz_relations-new.json`

With the old discrete `{ -1, 0, 1 }` theory file, Pearson is less informative.

### Interpretation

- higher positive values mean stronger linear agreement with theory
- near `0` means no linear relationship

### P-value

Yes. The pipelines report:

- `pearson_p_value`

## 3. `circular_distance_spearman`

### Definition

For every pair of values `(i, j)`:

- compute empirical cosine similarity `E[i, j]`
- compute circular step distance `step(i, j)` from the Schwartz order

Then correlate:

- `Spearman(E_flat, -step_flat)`

The minus sign is used because smaller step distance should correspond to larger similarity.

### What it measures

This is a circumplex-continuity metric.

It asks:

- do values that are close on the Schwartz circle tend to be similar in steering space?
- do values that are far apart on the circle tend to be dissimilar?

### Important difference from `spearman_rho`

`spearman_rho` compares empirical similarities to the explicit theoretical similarity matrix from the JSON file.

`circular_distance_spearman` ignores the exact numeric theory matrix and uses only:

- the circular order
- the ring distance along that order

So even though both metrics use the same high-dimensional empirical similarities, they test different things:

- `spearman_rho`: full matrix agreement
- `circular_distance_spearman`: circumplex continuity only

### Interpretation

- positive values mean the empirical space respects the circular progression of values
- higher values mean a stronger circumplex-like organization

### P-value

Yes. The pipelines report:

- `circular_distance_p_value`

## 4. `hierarchical_distance_spearman`

### Definition

For every value pair, assign a theory distance:

- `1` = same lower-order family
- `2` = same higher-order group
- `5` = no clear relation
- `10` = opposite higher-order groups

Then correlate:

- `Spearman(E_flat, -hierarchical_distance_flat)`

### What it measures

This is a hierarchical theory-structure metric.

It asks:

- are values in the same fine-grained family most similar?
- are values in the same broad higher-order group next most similar?
- are opposite higher-order values least similar?

### Why it matters

This metric is more theory-aware than a pure circle-distance metric, because it encodes:

- lower-order family structure
- higher-order group structure
- opposition
- neutral/no-relation cases

### Interpretation

- higher positive values mean stronger agreement with the hierarchical organization of Schwartz theory

### P-value

Yes. The pipelines report:

- `hierarchical_distance_p_value`

## 5. `within_minus_across_cosine`

### Definition

Compute:

- mean empirical cosine over pairs assigned to the same higher-order group
- minus mean empirical cosine over all other pairs

So:

- `within_minus_across = mean_same_group - mean_different_group`

### What it measures

This is a simple group-coherence metric.

It asks:

- are same-group values more similar than non-same-group values?

### Important limitation

This is a coarse metric.

It does **not** isolate only opposite groups. The "across" set includes:

- opposite groups
- neutral groups
- unrelated groups

It also uses a simplified group assignment, so it is less theory-faithful than `hierarchical_distance_spearman`.

### Interpretation

- positive values mean higher-order groups are internally more coherent than cross-group pairs

### P-value

No. The current pipelines do **not** report a p-value for this metric.

## Supporting Metrics

## `lower_minus_opposite_cosine`

### Definition

This is:

- `same_lower_order_mean_cosine - opposite_higher_order_mean_cosine`

where:

- `same_lower_order_mean_cosine` is the mean empirical cosine for pairs in the same lower-order family
- `opposite_higher_order_mean_cosine` is the mean empirical cosine for pairs in opposite higher-order groups

### What it measures

This is a direct separation metric between:

- the most closely related value pairs
- and the most theoretically opposed value pairs

### Why it matters

It is easy to interpret and complements the correlation-style metrics well.

If this value is large and positive, then:

- same-family values are clustered together
- opposite values are strongly separated

### P-value

No p-value is currently reported.

## `neighbor_minus_opposite_cosine`

### Definition

Take:

- mean empirical cosine for value pairs at circular step distance `1`
- minus mean empirical cosine for value pairs at circular step distance `10`

### What it measures

This tests an extreme local-vs-opposite contrast:

- are immediate neighbors much more similar than diametrically opposite values?

### Caveat

This is intuitive, but it uses only the two extreme ends of the circular structure. It is less complete than `circular_distance_spearman`.

### P-value

No p-value is currently reported.

## `silhouette_by_higher_order_group`

### Definition

Compute silhouette score using:

- pairwise cosine-derived distances
- higher-order group labels

### What it measures

This asks whether the 4 broad higher-order groups form clean clusters.

### Caveat

Because Schwartz theory is a circumplex and not a pure clustering theory, silhouette can be low or negative even when the geometry is meaningful.

### P-value

No p-value is currently reported.

## `procrustes_disparity` and `procrustes_rmse_after_alignment`

### Definition

These compare the 2D MDS embedding of the empirical geometry to an ideal Schwartz circle after alignment.

### What they measure

They are shape-match metrics for the 2D projection.

### Caveat

These depend on the MDS projection and are therefore more about 2D shape similarity than about the original high-dimensional geometry directly.

### P-value

No p-value is currently reported.

## `mds_stress`

### Definition

MDS first converts cosine similarity to distance:

- `D[i, j] = 1 - E[i, j]`

Then MDS embeds those distances into 2D. `mds_stress` measures how much distortion is introduced by that projection.

### What it measures

Projection quality only.

### Caveat

This is useful for diagnosing the 2D figure, but it is not itself a theory-alignment metric.

### P-value

No p-value is currently reported.

## Which Metrics Are Computed In High-Dimensional Space?

Computed directly from the original high-dimensional steering-vector similarities:

- `spearman_rho`
- `pearson_r`
- `circular_distance_spearman`
- `hierarchical_distance_spearman`
- `within_minus_across_cosine`
- `neighbor_minus_opposite_cosine`
- `silhouette_by_higher_order_group`

Computed from the 2D MDS embedding:

- `procrustes_disparity`
- `procrustes_rmse_after_alignment`
- `mds_stress`

Visualizations only:

- `UMAP`
- `PCA`
- `t-SNE`
- `MDS circumplex plot`

## Recommended Main Metrics For The Paper

For a compact main table, the recommended set is:

- `spearman_rho`
- `pearson_r`
- `circular_distance_spearman`
- `hierarchical_distance_spearman`
- `lower_minus_opposite_cosine`

Optionally include:

- `within_minus_across_cosine`

if space allows or if a simpler, more intuitive group-coherence statistic is useful for readers.

## Suggested Paper Framing

One possible summary sentence:

> We evaluate whether steering-vector geometry aligns with Schwartz value theory using four complementary metrics: global theory alignment (`Spearman ρ`, `Pearson r`), circumplex continuity (`circular-distance Spearman`), and hierarchical value organization (`hierarchical-distance Spearman`).

Another useful clarification:

> All reported theory-alignment metrics are computed in the original high-dimensional steering space; the 2D plots are visualization tools rather than the basis of the quantitative results.

## Files That Contain These Metrics

In the current codebase, these metrics are written to:

- `CAA/Geometry/.../geometry/geometry_metrics.json`
- `SAE/SparseCAA/.../geometry_raw/geometry_metrics.json`
- `SAE/SparseCAA/.../geometry_centered/geometry_metrics.json`
- `llm-steering-opt/.../geometry/geometry_metrics.json`

Legacy compatibility:

- `spearman_report.json` is still written for backward compatibility, but the richer metrics are in `geometry_metrics.json`.
