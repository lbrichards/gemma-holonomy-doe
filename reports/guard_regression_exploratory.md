# Exploratory Guard Regression Check

Status: `EXPLORATORY_POST_HOC_NOT_CONFIRMATORY`.

This report checks whether the k-nearest-neighbour guard-distance difference predicts the paired active-feature minus mixed-feature holonomy gap. It reads only frozen experiment artifacts and does not enter any confirmatory verdict rule.

## Sources

- Results: `run/holonomy_results_n390.parquet`
- Results SHA-256: `4c1a71bcfbf0331b2b5c15e51d33fdcfb21b54304cb0c1192bf61ed77ab8427a`
- Manifest: `run/manifest_n390.json`
- Manifest SHA-256: `f8f36b258420e2571c67a2229024402051aeb1d5f4156b853b63a7b33073c6d5`

## Definitions

- Outcome: `d_b = 0.5 * [(log_H_real_low - log_H_shuffled_low) + (log_H_real_high - log_H_shuffled_high)]`
- Guard difference: `manifest.manifold_distance_mahalanobis(real) - manifest.manifold_distance_mahalanobis(shuffled); in this run the field stores the preregistered k-NN fallback distance.`
- Reconstruction difference: `manifest.manifold_distance_recon(real) - manifest.manifold_distance_recon(shuffled)`
- Centering: `All covariates are centered at their sample means before OLS, so the intercept is the adjusted mean paired gap at the sample covariate centroid.`

## Results

- Base points: `390`
- Paired gap mean: `-0.294391`

| Model | Covariates | Intercept | R^2 | Guard slope [95% CI] | Reconstruction slope [95% CI] |
|---|---|---:|---:|---:|---:|
| `guard_only` | `guard_distance_diff` | -0.294391 | 0.00888315 | 0.0357663 [-0.00194244, 0.0734749] | -- |
| `reconstruction_only` | `reconstruction_distance_diff` | -0.294391 | 0.000805302 | -- | -0.770092 [-3.47765, 1.93746] |
| `primary_reference_reconstruction_plus_phi` | `reconstruction_distance_diff`, `phi_diff` | -0.294391 | 0.0153298 | -- | -0.957872 [-3.6536, 1.73786] |
| `guard_plus_phi` | `guard_distance_diff`, `phi_diff` | -0.294391 | 0.0225949 | 0.0350058 [-0.00249533, 0.0725068] | -- |
| `reconstruction_plus_guard_plus_phi` | `reconstruction_distance_diff`, `guard_distance_diff`, `phi_diff` | -0.294391 | 0.0284378 | 0.0470477 [0.0065132, 0.0875822] | -2.24961 [-5.15263, 0.653399] |

The guard-only model has R^2 about 0.009, and its slope interval includes zero.
The reconstruction-only model has R^2 about 0.001.
Because the regressions center covariates, every adjusted intercept equals the paired gap mean to displayed precision.
