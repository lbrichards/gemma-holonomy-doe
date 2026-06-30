# Center-displacement diagnostic

Status: POST_FREEZE_DIAGNOSTIC_NOT_CONFIRMATORY

This post-freeze diagnostic reads the frozen manifest and reports matched-center displacement
from the raw activation in units of the loop radius. It does not enter or modify the
preregistered decision rule.

Definition: `||c - h|| / (radius_relative * ||h||)`.

## Combined by plane type

| plane_type | n | mean | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active-feature | 780 | 13.1 | 10.6 | 26.4 | 29.7 | 87.8 |
| mixed-feature | 780 | 18.2 | 13.9 | 37.7 | 52.3 | 145 |
| random | 780 | 44 | 43.8 | 59 | 63.7 | 79.5 |

## By plane type and magnitude level

| plane_type | magnitude_level | n | mean | median | p90 | p95 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active-feature | low | 390 | 15.5 | 14.8 | 27.7 | 33.2 | 68.6 |
| active-feature | high | 390 | 10.7 | 8.27 | 23 | 27.4 | 87.8 |
| mixed-feature | low | 390 | 14.3 | 12.2 | 25.9 | 35 | 96.6 |
| mixed-feature | high | 390 | 22 | 17.6 | 44.7 | 60.4 | 145 |
| random | low | 390 | 35.3 | 35 | 44.3 | 48 | 55.7 |
| random | high | 390 | 52.8 | 52 | 63.7 | 67.5 | 79.5 |

The confirmatory verdict remains the preregistered frozen result. This diagnostic
characterizes a residual threat to validity and is not confirmatory.
