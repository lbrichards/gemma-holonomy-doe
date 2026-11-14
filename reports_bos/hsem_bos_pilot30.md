# H-sem with <bos> prepended: 30-base pilot

**Status: POST_FREEZE_NOT_CONFIRMATORY.** Post-freeze robustness check prompted by external review. No verdict label is applied; the preregistered result in `reports/hsem_results.md` is unchanged.

Estimator: the frozen H-sem primary (covariate-adjusted OLS-t intercept of d_b, covariates centered), 95% CI.

| estimate | n bases | gap [95% CI] |
|---|---|---|
| Frozen, no <bos> (preregistered) | 390 | -0.2944 [-0.4399, -0.1489] |
| Frozen, no <bos>, same 30 bases | 30 | +0.0192 [-0.5143, +0.5527] |
| **This pilot, with <bos>** | 30 | **+0.4957 [+0.1247, +0.8668]** |
| External reviewer (Tebow), with <bos> | n/a | -0.0010 [-0.1160, +0.1140] |

Pilot detail:

- HC3: +0.4957 [+0.1188, +0.8726]; unadjusted: +0.4957 [+0.0736, +0.9178]
- Within-magnitude, low: +0.5921 [+0.0091, +1.1752]; high: +0.3993 [-0.0071, +0.8057]
- Paired per-base change in d_b (with <bos> minus frozen): +0.4765 [-0.2055, +1.1585]
- Rows: 180; manifest kind `smoke`; band `TWO_ARM_FALLBACK` (m_low=49.11232273743059, m_high=65.43142059402942)
- Smoke magnitude source: `None`
- Stage A balance SMDs (real vs shuffled): recon -0.179, guard -0.246, phi -0.947

Caveats: the band (m_low/m_high) and covariate reference distribution are fit on these 30 bases only, so centers differ from what a full 390-base <bos> run would construct. The CI is wide at this n; this pilot sizes the effect, it does not settle it.
