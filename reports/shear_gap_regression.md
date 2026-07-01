# Full Shear Gap Regression

Status: POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY

This post-freeze diagnostic recomputes transport distortion quantities on the frozen verdict-bearing real/shuffled planes. It does not enter or modify the preregistered decision rule.

## Source

- Checkpoint: `reports/full_transport_shear_diagnostic.checkpoint.jsonl`
- Checkpoint rows: `1560`

## Definitions

- Outcome: `0.5 * [(log_H_real_low - log_H_shuffled_low) + (log_H_real_high - log_H_shuffled_high)]`
- Metric difference: `0.5 * [(metric_real_low - metric_shuffled_low) + (metric_real_high - metric_shuffled_high)]`
- Primary metric: `symmetric_residual_norm`
- Centering: `Each metric difference is centered at its sample mean before OLS; the intercept equals the mean paired gap.`

## Primary Result

- Base points: `390`
- Mean paired gap: `-0.294391`
- Mean shear difference: `-2.48754e-06`
- Centered-OLS intercept: `-0.294391 (SE 0.068882, 95% CI [-0.42982, -0.158963], p=2.42056e-05)`
- Slope: `14455.5 (SE 1784.43, 95% CI [10947.2, 17963.9], p=7.1587e-15)`
- R^2: `0.144668`
- HC3 slope: `14455.5 (SE 12533.6, 95% CI [-10186.8, 39097.9], p=0.249481)`

Transport shear/non-orthogonality is a live mechanism or confound for the observed reversal. The frozen verdict remains unchanged, but this should be reported as a serious post-freeze diagnostic finding and the paper should avoid claiming that shear has been bounded away.

Nonlinear, interaction, and readout-dependent explanations remain possible.

## Regression Table

| metric | n_base_points | mean_paired_gap | mean_metric_difference | slope | slope_se | slope_ci | slope_p | r_squared | hc3_slope_se | hc3_slope_ci | hc3_slope_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symmetric_residual_norm | 390 | -0.294391 | -2.48754e-06 | 14455.5 | 1784.43 | [10947.2, 17963.9] | 7.1587e-15 | 0.144668 | 12533.6 | [-10186.8, 39097.9] | 0.249481 |
| non_orthogonality_norm | 390 | -0.294391 | -4.97441e-06 | 7229.9 | 892.38 | [5475.39, 8984.4] | 7.11456e-15 | 0.144695 | 6265.29 | [-5088.27, 19548.1] | 0.249227 |
| transport_residual_norm | 390 | -0.294391 | 8.12426e-07 | 11711.6 | 735.365 | [10265.8, 13157.4] | 2.6618e-44 | 0.395305 | 2089.16 | [7604.12, 15819.1] | 3.93846e-08 |
| max_condition | 390 | -0.294391 | -0.01012 | 0.919003 | 0.156528 | [0.611254, 1.22675] | 9.29965e-09 | 0.0815934 | 0.217938 | [0.490516, 1.34749] | 3.08657e-05 |
