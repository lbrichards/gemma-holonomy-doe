# Feature 10762 at the final token, without vs with <bos>

**Status: POST_FREEZE_NOT_CONFIRMATORY.** Post-freeze robustness check prompted by external review; it does not enter or modify the preregistered verdict.

First 20 manifest base points. Without <bos>: frozen manifest activations and planes. With <bos>: Stage A pass-1 rerun with prepend_bos (same seeds, same final content token).

| metric | no <bos> | with <bos> |
|---|---|---|
| n_active | 20 | 0 |
| n_top_feature | 3 | 0 |
| median_activation | 26.27 | 0 |
| median_l1_share | 0.04436 | 0 |
| real_contains_feature | 10 | 0 |
| real_median_projection_sq | 0.5045 | 0.001065 |
| shuffled_contains_feature | 5 | 0 |
| shuffled_median_projection_sq | 0.001105 | 0.0002493 |
| random_contains_feature | 0 | 0 |
| random_median_projection_sq | 0.0008538 | 0.0008538 |

`projection_sq` is the squared norm of the unit feature atom projected onto the plane span (1.0 when the feature is one of the two plane atoms).

## Per base point

| base | f act (no) | rank (no) | L1 share (no) | real plane (no) | f act (bos) | rank (bos) | L1 share (bos) | real plane (bos) |
|---|---|---|---|---|---|---|---|---|
| bp-000 | 20 | 5 | 0.045 | [2230, 12342] | 0 | None | 0.000 | [6810, 12416] |
| bp-001 | 22.6 | 1 | 0.045 | [10762, 12358] | 0 | None | 0.000 | [6314, 12358] |
| bp-002 | 22.1 | 3 | 0.046 | [785, 12342] | 0 | None | 0.000 | [785, 15596] |
| bp-003 | 26.8 | 3 | 0.046 | [3750, 10762] | 0 | None | 0.000 | [3750, 7507] |
| bp-004 | 23.6 | 4 | 0.047 | [6810, 5327] | 0 | None | 0.000 | [5327, 14356] |
| bp-005 | 26.5 | 3 | 0.051 | [2564, 6810] | 0 | None | 0.000 | [2564, 14493] |
| bp-006 | 20.1 | 5 | 0.039 | [5511, 12342] | 0 | None | 0.000 | [5511, 6810] |
| bp-007 | 27.6 | 2 | 0.038 | [10762, 2009] | 0 | None | 0.000 | [6810, 8716] |
| bp-008 | 26.5 | 2 | 0.040 | [3240, 10762] | 0 | None | 0.000 | [12555, 3240] |
| bp-009 | 27.5 | 2 | 0.040 | [10762, 7541] | 0 | None | 0.000 | [7507, 2620] |
| bp-010 | 26.1 | 3 | 0.046 | [6810, 1322] | 0 | None | 0.000 | [1041, 1322] |
| bp-011 | 29.1 | 2 | 0.044 | [12395, 10762] | 0 | None | 0.000 | [12395, 12492] |
| bp-012 | 27.1 | 2 | 0.045 | [6810, 8287] | 0 | None | 0.000 | [15869, 5463] |
| bp-013 | 27.5 | 2 | 0.033 | [6810, 10762] | 0 | None | 0.000 | [5055, 2291] |
| bp-014 | 26.6 | 3 | 0.046 | [6810, 13430] | 0 | None | 0.000 | [6810, 13430] |
| bp-015 | 18.5 | 4 | 0.065 | [6810, 8641] | 0 | None | 0.000 | [6468, 8517] |
| bp-016 | 32.6 | 1 | 0.033 | [10762, 13295] | 0 | None | 0.000 | [1041, 1322] |
| bp-017 | 22.8 | 1 | 0.030 | [10762, 2167] | 0 | None | 0.000 | [2167, 2620] |
| bp-018 | 22.2 | 3 | 0.041 | [12342, 10762] | 0 | None | 0.000 | [15663, 12358] |
| bp-019 | 22.6 | 4 | 0.035 | [8081, 15030] | 0 | None | 0.000 | [6810, 5074] |
