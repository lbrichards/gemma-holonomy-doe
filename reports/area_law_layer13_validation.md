# Area-law validation at layer-13 readout

Status: POST_FREEZE_INSTRUMENT_VALIDATION_NOT_PART_OF_FROZEN_VERDICT

Selection rule: first 4 frozen manifest base points; arms `['real', 'shuffled']`; levels `['low']`; radius scales `[0.5, 1.0, 2.0]`.

Target: signed rotation should scale approximately as `rho^2`, so the pooled/per-loop log-log slope of `abs(theta)` versus `rho` should be near 2.

- Device: `mps:0`
- Pooled log(abs(theta)) vs log(rho) slope: 2.0306926100799747
- Total loop runtime seconds: 967.62

## Per-loop slope summary

| base_point_id | plane_type | magnitude_level | log_abs_theta_vs_log_rho_slope | H_min | H_max |
| --- | --- | --- | --- | --- | --- |
| bp-000 | active-feature | low | 2.00233 | 6.03226e-06 | 6.03639e-06 |
| bp-000 | mixed-feature | low | 2.00503 | 1.96923e-05 | 1.98047e-05 |
| bp-001 | active-feature | low | 1.98921 | 9.07776e-06 | 9.22749e-06 |
| bp-001 | mixed-feature | low | 2.00464 | 2.20077e-05 | 2.21274e-05 |
| bp-002 | active-feature | low | 1.99707 | 5.01707e-06 | 5.03837e-06 |
| bp-002 | mixed-feature | low | 2.00004 | 3.76127e-05 | 3.76662e-05 |
| bp-003 | active-feature | low | 1.99963 | 4.86015e-05 | 4.86483e-05 |
| bp-003 | mixed-feature | low | 1.99404 | 7.13639e-05 | 7.23845e-05 |

## Measurement rows

| base_point_id | plane_type | magnitude_level | radius_scale | rho | theta | abs_theta | area | H | theta_over_rho2 | abs_theta_over_rho2 | symmetric_residual_norm | max_condition | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bp-000 | active-feature | low | 0.5 | 0.361445 | -9.74013e-07 | 9.74013e-07 | 0.161357 | 6.03639e-06 | -7.45557e-06 | 7.45557e-06 | 1.33556e-06 | 1.14466 | 34.7506 |
| bp-000 | active-feature | low | 1 | 0.722889 | -3.89825e-06 | 3.89825e-06 | 0.646234 | 6.03226e-06 | -7.45978e-06 | 7.45978e-06 | 5.11946e-06 | 1.14599 | 40.386 |
| bp-000 | active-feature | low | 2 | 1.44578 | -1.56347e-05 | 1.56347e-05 | 2.59146 | 6.03316e-06 | -7.47973e-06 | 7.47973e-06 | 2.06203e-05 | 1.14868 | 42.3273 |
| bp-000 | mixed-feature | low | 0.5 | 0.361445 | 2.85078e-06 | 2.85078e-06 | 0.144766 | 1.96923e-05 | 2.18212e-05 | 2.18212e-05 | 1.57632e-06 | 1.30034 | 44.8218 |
| bp-000 | mixed-feature | low | 1 | 0.722889 | 1.14843e-05 | 1.14843e-05 | 0.579878 | 1.98047e-05 | 2.19766e-05 | 2.19766e-05 | 6.46377e-06 | 1.30224 | 46.6302 |
| bp-000 | mixed-feature | low | 2 | 1.44578 | 4.59318e-05 | 4.59318e-05 | 2.32607 | 1.97465e-05 | 2.1974e-05 | 2.1974e-05 | 2.59659e-05 | 1.30606 | 45.7798 |
| bp-001 | active-feature | low | 0.5 | 0.379048 | -1.57545e-06 | 1.57545e-06 | 0.170735 | 9.22749e-06 | -1.09652e-05 | 1.09652e-05 | 2.23323e-06 | 1.19007 | 44.5222 |
| bp-001 | active-feature | low | 1 | 0.758096 | -6.24028e-06 | 6.24028e-06 | 0.683249 | 9.13324e-06 | -1.08582e-05 | 1.08582e-05 | 9.22812e-06 | 1.19169 | 43.3034 |
| bp-001 | active-feature | low | 2 | 1.51619 | -2.48329e-05 | 2.48329e-05 | 2.73557 | 9.07776e-06 | -1.08024e-05 | 1.08024e-05 | 3.68909e-05 | 1.19498 | 40.8337 |
| bp-001 | mixed-feature | low | 0.5 | 0.379048 | 3.10564e-06 | 3.10564e-06 | 0.141116 | 2.20077e-05 | 2.16154e-05 | 2.16154e-05 | 1.69529e-06 | 1.11071 | 39.9735 |
| bp-001 | mixed-feature | low | 1 | 0.758096 | 1.25145e-05 | 1.25145e-05 | 0.565568 | 2.21274e-05 | 2.17754e-05 | 2.17754e-05 | 6.75119e-06 | 1.11096 | 39.7321 |
| bp-001 | mixed-feature | low | 2 | 1.51619 | 5.0011e-05 | 5.0011e-05 | 2.27142 | 2.20175e-05 | 2.17549e-05 | 2.17549e-05 | 2.70327e-05 | 1.11146 | 39.9943 |
| bp-002 | active-feature | low | 0.5 | 0.374559 | 8.81746e-07 | 8.81746e-07 | 0.175006 | 5.03837e-06 | 6.28498e-06 | 6.28498e-06 | 7.62917e-07 | 1.02159 | 39.8623 |
| bp-002 | active-feature | low | 1 | 0.749117 | 3.5186e-06 | 3.5186e-06 | 0.700074 | 5.02604e-06 | 6.27004e-06 | 6.27004e-06 | 3.05283e-06 | 1.02241 | 40.1109 |
| bp-002 | active-feature | low | 2 | 1.49823 | 1.40508e-05 | 1.40508e-05 | 2.80059 | 5.01707e-06 | 6.25951e-06 | 6.25951e-06 | 1.18323e-05 | 1.02407 | 40.0835 |
| bp-002 | mixed-feature | low | 0.5 | 0.374559 | 5.76221e-06 | 5.76221e-06 | 0.153154 | 3.76235e-05 | 4.10723e-05 | 4.10723e-05 | 1.79885e-06 | 1.16763 | 39.5904 |
| bp-002 | mixed-feature | low | 1 | 0.749117 | 2.30778e-05 | 2.30778e-05 | 0.612693 | 3.76662e-05 | 4.1124e-05 | 4.1124e-05 | 7.44175e-06 | 1.16835 | 39.2859 |
| bp-002 | mixed-feature | low | 2 | 1.49823 | 9.22009e-05 | 9.22009e-05 | 2.45132 | 3.76127e-05 | 4.10748e-05 | 4.10748e-05 | 2.98055e-05 | 1.16973 | 39.0116 |
| bp-003 | active-feature | low | 0.5 | 0.386231 | 7.85316e-06 | 7.85316e-06 | 0.161427 | 4.86483e-05 | 5.26443e-05 | 5.26443e-05 | 8.58184e-07 | 1.02719 | 38.2885 |
| bp-003 | active-feature | low | 1 | 0.772461 | 3.14056e-05 | 3.14056e-05 | 0.645806 | 4.86301e-05 | 5.26325e-05 | 5.26325e-05 | 3.47692e-06 | 1.02766 | 37.5093 |
| bp-003 | active-feature | low | 2 | 1.54492 | 0.000125587 | 0.000125587 | 2.58401 | 4.86015e-05 | 5.26175e-05 | 5.26175e-05 | 1.396e-05 | 1.02858 | 37.4688 |
| bp-003 | mixed-feature | low | 0.5 | 0.386231 | 6.98671e-06 | 6.98671e-06 | 0.0965221 | 7.23845e-05 | 4.68359e-05 | 4.68359e-05 | 6.66918e-06 | 2.16693 | 37.5322 |
| bp-003 | mixed-feature | low | 1 | 0.772461 | 2.79043e-05 | 2.79043e-05 | 0.386846 | 7.21327e-05 | 4.67646e-05 | 4.67646e-05 | 2.66113e-05 | 2.17403 | 37.8668 |
| bp-003 | mixed-feature | low | 2 | 1.54492 | 0.000110868 | 0.000110868 | 1.55356 | 7.13639e-05 | 4.64507e-05 | 4.64507e-05 | 0.000105292 | 2.18841 | 37.9569 |

The confirmatory verdict remains the preregistered frozen result. These analyses are post-freeze diagnostics and validation checks added to characterize the instrument and residual threats to validity; they do not enter or modify the preregistered decision rule.
