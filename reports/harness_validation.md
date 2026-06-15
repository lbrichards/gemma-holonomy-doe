# H-sem Harness Validation

Prepared before freezing the H-sem analysis addendum. No real H-sem estimate was run.

## Scope

The harness lives in `analysis/hsem/harness.py`. It implements the addendum's execution order for
H-sem, but its CLI defaults to a dry run: it loads the real parquet only for structural assertions
and then halts before computing any real `d_b`.

Dry-run command:

```bash
uv run python -m analysis.hsem --results run/holonomy_results_n390.parquet
```

Observed halt message:

```text
Plan not frozen — halting before estimation.
```

## Toy-Oracle Tests

Command:

```bash
uv run pytest tests/test_hsem_harness.py -v
```

Result: `6 passed`.

Validated checks:

1. R1 estimand identity: averaging low/high log-H within arm before differencing equals averaging
   the low-only and high-only paired contrasts with equal weight.
2. R2 centered OLS recovery: the centered-covariate intercept recovers a planted adjusted mean on
   simulated data; repeated simulations show the classical t interval has the expected coverage
   behavior. HC3 is reported alongside and agrees on the verdict in the homoskedastic oracle.
3. R2 centering demonstration: with deliberate covariate imbalance, the centered intercept equals
   the adjusted mean, while the uncentered intercept equals the effect-at-parity and differs by
   slope times mean covariate difference as predicted.
4. Verdict logic: CORROBORATED, FALSIFIED, and INCONCLUSIVE fire at the correct positions relative
   to the materiality threshold.
5. R4 NULL-ATTRIBUTED logic: raw-Corroborated/adjusted-Falsified flags NULL-ATTRIBUTED;
   raw-Corroborated/adjusted-Inconclusive flags attenuated-inconclusive; NULL-ATTRIBUTED can only
   co-occur with a primary verdict of Falsified.
6. End-to-end toy harness: simulated real/shuffled/random cells and simulated covariate
   differences produce the expected toy verdict without touching the real result values.

## Real Artifact Staging

The harness is staged against `run/holonomy_results_n390.parquet`, but real execution is disabled
unless `--execute` is passed. In this preparation task, `--execute` was not passed.

Readiness: harness validated on toy oracle, staged against real parquet, awaiting
"plan frozen, proceed."

