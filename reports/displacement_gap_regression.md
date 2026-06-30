# Displacement Gap Regression

Status: `POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY`.

This report checks whether active-feature minus mixed-feature matched-center displacement, measured in loop-radius units, predicts the paired active-feature minus mixed-feature holonomy gap. It reads only frozen experiment artifacts and the derived center-displacement diagnostic; it does not enter any confirmatory verdict rule.

## Sources

- Results: `run/holonomy_results_n390.parquet`
- Results SHA-256: `4c1a71bcfbf0331b2b5c15e51d33fdcfb21b54304cb0c1192bf61ed77ab8427a`
- Displacement diagnostic: `reports/center_displacement_diagnostic.json`
- Displacement SHA-256: `89422479d5cb6371f9574d9658f1ce824836aac540578510735a40061a602c35`

## Definitions

- Outcome: `d_b = 0.5 * [(log_H_real_low - log_H_shuffled_low) + (log_H_real_high - log_H_shuffled_high)]`
- Average displacement difference: `0.5 * [((||c_real_low-h||/rho) - (||c_shuffled_low-h||/rho)) + ((||c_real_high-h||/rho) - (||c_shuffled_high-h||/rho))]`
- Centering: `All covariates are centered at their sample means before OLS, so the intercept is the adjusted mean paired gap at the sample covariate centroid.`

## Results

- Base points: `390`
- Paired gap mean: `-0.294391`
- Mean average displacement difference: `-5.06501`

| Model | Covariates | Intercept | R^2 | Slope(s) [95% CI] |
|---|---|---:|---:|---:|
| `average_displacement` | `displacement_over_rho_diff` | -0.294391 | 0.00644881 | `displacement_over_rho_diff`: 0.00726881 [-0.00173668, 0.0162743], p=0.113341 |
| `low_displacement` | `low_displacement_over_rho_diff` | -0.294391 | 0.00631469 | `low_displacement_over_rho_diff`: 0.0075113 [-0.00189356, 0.0169162], p=0.117173 |
| `high_displacement` | `high_displacement_over_rho_diff` | -0.294391 | 0.00490926 | `high_displacement_over_rho_diff`: 0.00525793 [-0.0022139, 0.0127298], p=0.167294 |
| `low_plus_high_displacement` | `low_displacement_over_rho_diff`, `high_displacement_over_rho_diff` | -0.294391 | 0.00670754 | `low_displacement_over_rho_diff`: 0.00565011 [-0.00762139, 0.0189216], p=0.403088<br>`high_displacement_over_rho_diff`: 0.00209656 [-0.00843973, 0.0126329], p=0.695845 |

The average-displacement model has low explanatory power and its slope interval includes zero. Thus the displacement asymmetry is present and aligned with plane type, but this diagnostic does not support matched-center displacement as a linear predictor of the paired holonomy gap.
