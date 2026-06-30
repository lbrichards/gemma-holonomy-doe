# Transport shear / non-orthogonality diagnostic

Status: POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY

Selection rule: first 4 frozen manifest base points; arms `['real', 'shuffled']`; levels `['low', 'high']`; frozen radius.

Definitions:

- `symmetric_residual_norm`: Frobenius norm of `0.5 * ((T - I) + (T - I)^T)`.
- `||T.T @ T - I||`: reported as `non_orthogonality_norm`.
- `||T - I||`: reported as `transport_residual_norm`.

- Device: `mps:0`
- Total loop runtime seconds: 598.07

## Arm/level summary

| plane_type | magnitude_level | n | symmetric_residual_norm_mean | symmetric_residual_norm_median | non_orthogonality_norm_mean | non_orthogonality_norm_median |
| --- | --- | --- | --- | --- | --- | --- |
| active-feature | low | 4 | 5.21933e-06 | 4.29819e-06 | 1.04383e-05 | 8.59577e-06 |
| active-feature | high | 4 | 5.9608e-06 | 3.87746e-06 | 1.19214e-05 | 7.75471e-06 |
| mixed-feature | low | 4 | 1.1817e-05 | 7.09647e-06 | 2.36336e-05 | 1.41925e-05 |
| mixed-feature | high | 4 | 8.72071e-06 | 8.97488e-06 | 1.74411e-05 | 1.79494e-05 |

## Correlations with log_H

| symmetric_residual_norm | non_orthogonality_norm | transport_residual_norm |
| --- | --- | --- |
| 0.429742 | 0.429722 | 0.953561 |

## Paired active-feature minus mixed-feature diagnostic differences

| base_point_id | magnitude_level | active_feature_minus_mixed_feature_symmetric_residual_norm | active_feature_minus_mixed_feature_non_orthogonality_norm |
| --- | --- | --- | --- |
| bp-000 | low | -1.34431e-06 | -2.68849e-06 |
| bp-000 | high | -4.28657e-06 | -8.57289e-06 |
| bp-001 | low | 2.47692e-06 | 4.95396e-06 |
| bp-001 | high | 3.66921e-06 | 7.33859e-06 |
| bp-002 | low | -4.38893e-06 | -8.7773e-06 |
| bp-002 | high | -3.67731e-06 | -7.35424e-06 |
| bp-003 | low | -2.31344e-05 | -4.62692e-05 |
| bp-003 | high | -6.74495e-06 | -1.34902e-05 |

## Measurement rows

| base_point_id | plane_type | magnitude_level | H | theta | area | symmetric_residual_norm | non_orthogonality_norm | transport_residual_norm | max_condition | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bp-000 | active-feature | low | 6.03226e-06 | -3.89825e-06 | 0.646234 | 5.11946e-06 | 1.02389e-05 | 7.5234e-06 | 1.14599 | 36.6127 |
| bp-000 | active-feature | high | 5.42388e-06 | 3.56662e-06 | 0.657577 | 4.48574e-06 | 8.97144e-06 | 6.75006e-06 | 1.1832 | 35.0001 |
| bp-000 | mixed-feature | low | 1.98047e-05 | 1.14843e-05 | 0.579878 | 6.46377e-06 | 1.29274e-05 | 1.74802e-05 | 1.30224 | 34.6826 |
| bp-000 | mixed-feature | high | 2.99567e-05 | 1.8131e-05 | 0.605241 | 8.77231e-06 | 1.75443e-05 | 2.71002e-05 | 1.36089 | 35.1421 |
| bp-001 | active-feature | low | 9.13324e-06 | -6.24028e-06 | 0.683249 | 9.22812e-06 | 1.84561e-05 | 1.27687e-05 | 1.19169 | 36.0579 |
| bp-001 | active-feature | high | 8.97156e-06 | 6.42579e-06 | 0.71624 | 1.28467e-05 | 2.56931e-05 | 1.57359e-05 | 1.24187 | 36.7488 |
| bp-001 | mixed-feature | low | 2.21274e-05 | 1.25145e-05 | 0.565568 | 6.75119e-06 | 1.35022e-05 | 1.89422e-05 | 1.11096 | 37.5545 |
| bp-001 | mixed-feature | high | 2.47874e-05 | -1.53772e-05 | 0.620364 | 9.17744e-06 | 1.83545e-05 | 2.36039e-05 | 1.11794 | 37.982 |
| bp-002 | active-feature | low | 5.02604e-06 | 3.5186e-06 | 0.700074 | 3.05283e-06 | 6.10563e-06 | 5.83788e-06 | 1.02241 | 38.9041 |
| bp-002 | active-feature | high | 7.4935e-06 | 5.17826e-06 | 0.691033 | 3.24162e-06 | 6.48319e-06 | 8.00854e-06 | 1.03464 | 38.8038 |
| bp-002 | mixed-feature | low | 3.76662e-05 | 2.30778e-05 | 0.612693 | 7.44175e-06 | 1.48829e-05 | 3.34746e-05 | 1.16835 | 38.5456 |
| bp-002 | mixed-feature | high | 3.05965e-05 | 1.88283e-05 | 0.615375 | 6.91893e-06 | 1.38374e-05 | 2.75115e-05 | 1.16547 | 38.753 |
| bp-003 | active-feature | low | 4.86301e-05 | 3.14056e-05 | 0.645806 | 3.47692e-06 | 6.95267e-06 | 4.45501e-05 | 1.02766 | 38.6029 |
| bp-003 | active-feature | high | 2.88507e-05 | 1.87965e-05 | 0.651509 | 3.26919e-06 | 6.53798e-06 | 2.67825e-05 | 1.03481 | 38.2769 |
| bp-003 | mixed-feature | low | 7.21327e-05 | 2.79043e-05 | 0.386846 | 2.66113e-05 | 5.32219e-05 | 4.75968e-05 | 2.17403 | 38.4365 |
| bp-003 | mixed-feature | high | 1.07018e-05 | -3.79388e-06 | 0.354508 | 1.00141e-05 | 2.00282e-05 | 1.13609e-05 | 2.36227 | 37.9706 |

The confirmatory verdict remains the preregistered frozen result. These analyses are post-freeze diagnostics and validation checks added to characterize the instrument and residual threats to validity; they do not enter or modify the preregistered decision rule.
