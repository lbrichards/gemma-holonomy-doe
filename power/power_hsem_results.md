# H-sem Power Calculation

Scale: log-scale across-base SD of the feature-vs-control semantic contrast.

- `ALPHA = 0.05`
- `POWER = 0.9`
- `DELTA_MATERIAL = log(1.25) = 0.223143551314`
- `DELTA_OBSERVED_SUBMATERIAL = log(1.103) = 0.098033740271`

| Target | Tau source | Tau | Sidedness | Delta | Normal N | t-corrected N |
|---|---:|---:|---|---:|---:|---:|
| Material target: log(1.25) | Iter 8 tau | 0.586 | one-sided | 0.223144 | 60 | 61 |
| Material target: log(1.25) | Iter 8 tau | 0.586 | two-sided | 0.223144 | 73 | 75 |
| Material target: log(1.25) | Iter 7 tau | 0.649 | one-sided | 0.223144 | 73 | 75 |
| Material target: log(1.25) | Iter 7 tau | 0.649 | two-sided | 0.223144 | 89 | 91 |
| Not our target: observed sub-material log(1.103) | Iter 8 tau | 0.586 | one-sided | 0.098034 | 306 | 308 |
| Not our target: observed sub-material log(1.103) | Iter 8 tau | 0.586 | two-sided | 0.098034 | 376 | 378 |
| Not our target: observed sub-material log(1.103) | Iter 7 tau | 0.649 | one-sided | 0.098034 | 376 | 377 |
| Not our target: observed sub-material log(1.103) | Iter 7 tau | 0.649 | two-sided | 0.098034 | 461 | 463 |

The observed sub-material rows are included only to document the cost of chasing a
smaller-than-material effect; they are not the pre-registered target.
