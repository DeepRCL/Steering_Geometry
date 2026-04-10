# Steering Geometry Metrics — One-Page Summary

This note gives the short paper-facing definitions of the main geometry metrics used in the project.

## Core Setup

For each of the 20 Schwartz values, the pipeline computes one steering vector.

To analyze geometry:

- each vector is normalized to unit length
- empirical similarity between two values is computed by cosine similarity
- all quantitative geometry metrics are computed in the original high-dimensional steering space

The 2D plots (UMAP, PCA, t-SNE, MDS) are only visualizations of that high-dimensional structure.

## Recommended Main Metrics

The most useful four metrics for the paper are:

1. `spearman_rho`
2. `pearson_r`
3. `circular_distance_spearman`
4. `hierarchical_distance_spearman`

Optional supporting metric:

5. `within_minus_across_cosine`
6. `lower_minus_opposite_cosine`

## Metric Definitions

## 1. `spearman_rho`

This is the Spearman correlation between:

- the empirical pairwise cosine similarities of the steering vectors
- the theoretical Schwartz similarity matrix from `relations_path`

Interpretation:

- asks whether value pairs that theory considers more aligned are also ranked as more aligned in the model
- higher positive values mean better global theory alignment

Why report it:

- it is the most general theory-alignment metric
- it is robust because it depends on rank order rather than exact scaling

Reported with p-value:

- yes

## 2. `pearson_r`

This is the Pearson correlation between:

- the empirical pairwise cosine similarities
- the theoretical Schwartz similarity matrix

Interpretation:

- asks whether empirical similarities scale linearly with the theoretical similarities
- more meaningful when using the continuous cosine-theory file `schwartz_relations-new.json`

Why report it:

- complements Spearman by testing calibrated continuous alignment, not only ordering

Reported with p-value:

- yes

## 3. `circular_distance_spearman`

This metric uses the fixed Schwartz circumplex order.

For every pair of values, compute the circular step distance on the 20-item ring:

- step `1` = adjacent values
- step `10` = opposite values

Then compute the Spearman correlation between:

- empirical cosine similarity
- negative circular distance

Interpretation:

- asks whether nearby values on the Schwartz circle are more similar than distant values
- captures circumplex continuity directly

Why report it:

- it tests the central geometric claim of the project: whether steering space is organized like a circumplex

Reported with p-value:

- yes

## 4. `hierarchical_distance_spearman`

This metric uses a theory-aware hierarchical distance:

- `1` = same lower-order family
- `2` = same higher-order group
- `5` = no strong relation
- `10` = opposite higher-order groups

Boundary values are treated as belonging to two higher-order groups:

- `Hedonism`
- `Face`
- `Humility`

Then compute the Spearman correlation between:

- empirical cosine similarity
- negative hierarchical distance

Interpretation:

- asks whether the model respects Schwartz structure at both fine-grained and broad levels

Why report it:

- it is more theory-faithful than a simple same-group vs different-group comparison

Reported with p-value:

- yes

## 5. `within_minus_across_cosine` (optional support metric)

This is:

- mean cosine similarity for same higher-order group pairs
- minus mean cosine similarity for all other pairs

Interpretation:

- asks whether higher-order groups are internally more coherent than cross-group pairs

Caveat:

- this is a coarse metric
- it does not separate opposite groups from neutral groups

Reported with p-value:

- no

## 6. `lower_minus_opposite_cosine` (recommended support metric)

This is:

- `same_lower_order_mean_cosine - opposite_higher_order_mean_cosine`

Interpretation:

- asks whether values from the same lower-order family are substantially closer than theoretically opposite values

Why report it:

- it is very interpretable
- it complements the correlation metrics with a direct separation score

Reported with p-value:

- no

## Suggested Paper Framing

Recommended short description:

> We quantify the alignment between steering-vector geometry and Schwartz value theory using four complementary metrics: global theory alignment (`Spearman ρ`, `Pearson r`), circumplex continuity (`circular-distance Spearman`), and hierarchical theory alignment (`hierarchical-distance Spearman`).

Recommended methodological clarification:

> All reported geometry metrics are computed in the original high-dimensional steering space. Two-dimensional projections such as MDS, UMAP, PCA, and t-SNE are used only for visualization.

Recommended interpretation:

> Positive values indicate that the steering space is organized in a way that is increasingly consistent with human value structure as described by Schwartz theory.

## Files

Full details:

- [Metrics.MD](/home/mahdi.abootorabi/Steering_Geometry/CAA/Metrics.MD)

Main output JSON:

- `geometry_metrics.json`
