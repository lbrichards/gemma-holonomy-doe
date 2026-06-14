# Tau Re-Pilot at Corrected resid_post Site

Reserve-only pilot; no stage-1 base points touched.

- Extraction site: `HF model.model.layers[12] forward-hook output; TransformerLens blocks.12.hook_resid_post`
- Downstream readout: `HF model.model.layers[13] forward-hook output after patching layer-12 resid_post`
- SAE: `google/gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_82/params.npz`
- Reserve draw orders used: [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
- n bases: 16
- n steps: 200
- tau_pilot: 1.11194
- tau 95% CI: [0.821392, 1.72093]
- frozen tau comparison: Iter 7 = 0.649; Iter 8 = 0.586
- SAE L0 sanity: mean 103.38, SD 23.01, range [72, 148]
- det M floor: 0.413
- det M rejected/tested, real: 13/29
- det M rejected/tested, shuffled: 1616/1632
- elapsed: 30.24 min

| reserve idx | draw order | title | L0 | d_b | real H | shuffled H | real detM | shuffled detM |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 96 | 96 | California State Route 188 | 72 | -0.248195 | 2.3121e-05 | 2.96344e-05 | 2.15284 | 1.86779 |
| 97 | 97 | The Voice Within | 127 | 2.02906 | 1.6556e-05 | 2.17644e-06 | 1.28253 | 0.89217 |
| 98 | 98 | Trial of Lord George Gordon | 79 | 0.246942 | 4.7019e-06 | 3.67306e-06 | 1.32794 | 0.998954 |
| 99 | 99 | Smoothtooth blacktip shark | 115 | 1.64039 | 8.17622e-05 | 1.5854e-05 | 1.12844 | 1.07456 |
| 100 | 100 | R. K. Narayan | 103 | 0.585046 | 3.43041e-05 | 1.91101e-05 | 0.875312 | 1.10785 |
| 101 | 101 | Battle of Albuera | 81 | -0.629983 | 2.34734e-05 | 4.40731e-05 | 1.73916 | 1.52353 |
| 102 | 102 | Wish You Were Here ( Pink Floyd album ) | 111 | -1.41045 | 2.09465e-05 | 8.58346e-05 | 0.422911 | 0.42664 |
| 103 | 103 | Battle of the Îles Saint @-@ Marcouf | 81 | -1.04199 | 1.70137e-05 | 4.82315e-05 | 0.872787 | 0.596435 |
| 104 | 104 | 1907 Tour de France | 90 | -1.68136 | 1.07862e-05 | 5.79528e-05 | 1.42769 | 1.04242 |
| 105 | 105 | Saving Private Ryan | 99 | 0.361111 | 3.60752e-05 | 2.51409e-05 | 0.438475 | 1.20352 |
| 106 | 106 | Boyce McDaniel | 148 | 1.74867 | 8.13786e-05 | 1.41603e-05 | 1.16185 | 0.955551 |
| 107 | 107 | Pyjama shark | 139 | -0.723651 | 1.05187e-05 | 2.16891e-05 | 1.17641 | 0.77499 |
| 108 | 108 | Albatross | 91 | 0.5858 | 1.20157e-05 | 6.68868e-06 | 1.35761 | 1.14361 |
| 109 | 109 | 2011 – 12 Sheffield United F.C. season | 98 | -0.322736 | 5.5745e-05 | 7.69784e-05 | 1.36044 | 1.31164 |
| 110 | 110 | Shadowrun ( 1993 video game ) | 131 | -0.00386664 | 2.03132e-05 | 2.03919e-05 | 1.22716 | 0.915099 |
| 111 | 111 | Selena | 89 | 1.05588 | 3.13874e-05 | 1.09193e-05 | 1.58556 | 1.08292 |
