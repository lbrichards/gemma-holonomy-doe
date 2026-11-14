# H-sem with <bos> prepended: 390-base full run

**Status: POST_FREEZE_NOT_CONFIRMATORY.** Post-freeze robustness check prompted by external review. No verdict label is applied; the preregistered result in `reports/hsem_results.md` is unchanged.

Estimator: the frozen H-sem primary (covariate-adjusted OLS-t intercept of d_b, covariates centered), 95% CI.

| estimate | n bases | gap [95% CI] |
|---|---|---|
| Frozen, no <bos> (preregistered) | 390 | -0.2944 [-0.4399, -0.1489] |
| Frozen, no <bos>, same 390 bases | 390 | -0.2944 [-0.4399, -0.1489] |
| **This full run, with <bos>** | 390 | **+0.0783 [-0.0424, +0.1991]** |
| External reviewer (Tebow), with <bos> | n/a | -0.0010 [-0.1160, +0.1140] |

Full run detail:

- HC3: +0.0783 [-0.0440, +0.2007]; unadjusted: +0.0783 [-0.0430, +0.1997]
- Within-magnitude, low: +0.0832 [-0.0502, +0.2166]; high: +0.0735 [-0.0603, +0.2073]
- Paired per-base change in d_b (with <bos> minus frozen): +0.3727 [+0.1880, +0.5574]
- Rows: 2340; manifest kind `post_freeze`; band `TWO_ARM_FALLBACK` (m_low=48.328951224036594, m_high=63.33329149095895)
- Smoke magnitude source: `None`
- Stage A balance SMDs (real vs shuffled): recon -0.053, guard -0.174, phi -0.792
