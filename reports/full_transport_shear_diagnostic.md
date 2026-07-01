# Full Transport Shear Diagnostic

Status: POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY

This post-freeze diagnostic recomputes transport distortion quantities on the frozen verdict-bearing real/shuffled planes. It does not enter or modify the preregistered decision rule.

## Scope

- Arms: `real` (active-feature) and `shuffled` (mixed-feature)
- Magnitude levels: `low`, `high`
- Checkpoint: `reports/full_transport_shear_diagnostic.checkpoint.jsonl`
- Rows: `1560`
- Error rows: `0`
- Frozen-row validation failures: `0`
- Total runtime seconds: `56511.11`

## Arm/level summary

| plane_type | magnitude_level | n | symmetric_residual_norm_mean | non_orthogonality_norm_mean | transport_residual_norm_mean | max_condition_mean |
| --- | --- | --- | --- | --- | --- | --- |
| active-feature | low | 390 | 1.18275e-05 | 2.36533e-05 | 4.36304e-05 | 1.3105 |
| active-feature | high | 390 | 1.15438e-05 | 2.30864e-05 | 3.9676e-05 | 1.34653 |
| mixed-feature | low | 390 | 1.48383e-05 | 2.96745e-05 | 4.29936e-05 | 1.3275 |
| mixed-feature | high | 390 | 1.3508e-05 | 2.7014e-05 | 3.86879e-05 | 1.34976 |

The confirmatory verdict remains the preregistered frozen result. These analyses are post-freeze diagnostics and validation checks added to characterize the instrument and residual threats to validity; they do not enter or modify the preregistered decision rule.
