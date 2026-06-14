# Glossary

Cross-linked terminology for the Gemma Holonomy DOE project.

---

## <span style="background-color: #4051b5; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">COINED</span> Coined Terms

*Terms introduced by this project.*

- [active feature](active-feature.md) — An SAE feature with positive activation (code > 0) at a given base point, indicating the feature fires on that input.
- [article reconstruction](article-reconstruction.md) — The process of joining WikiText-103 raw rows between consecutive top-level headers into coherent articles, prior to passage sampling.
- [balance diagnostic](balance-diagnostic.md) — Pre-unblinding checks (SMD, overlap, log sin phi) comparing real-vs-shuffled covariate distributions to characterize imbalance.
- [band-collapse contingency](band-collapse-contingency.md) — The pre-registered fallback procedure when the three-arm common-support band is empty or too narrow to support matched magnitude comparisons.
- [base point](base-point.md) — A single layer-12 resid_post activation vector extracted from a WikiText passage, serving as the blocking unit and loop center origin.
- [blind handoff](blind-handoff.md) — The design principle that Stage A outputs are frozen and complete before Stage B begins, with no feedback loop from responses to design.
- [blind manifest boundary](blind-manifest-boundary.md) — The partition point separating the 390 experiment-sample base points from the reserve, determined by draw order and recorded before any holonomy is observed.
- [burned pilot points](burned-pilot-points.md) — The 16 reserve points (draw orders 96-111) used in the resid_post variance re-pilot, permanently excluded from both experiment sample and reserve.
- [center placement](center-placement.md) — The procedure for constructing a loop center at a target in-plane magnitude while preserving the out-of-plane component of the base activation.
- [common-support band](common-support-band.md) — The magnitude range where all three plane-type arms (random, shuffled-feature, real-feature) have overlapping in-plane magnitude distributions, enabling matched comparisons.
- [corpus draw](corpus-draw.md) — The deterministic procedure for selecting WikiText passages: article reconstruction, seed-42 shuffle, raw word filtering, token truncation to 64 Gemma tokens.
- [CORROBORATED verdict](corroborated-verdict.md) — The verdict when the CI lower bound lies above the materiality threshold, indicating the hypothesis is supported by the data.
- [covariate adjustment](covariate-adjustment.md) — Regression adjustment for manifold distance and phi in the real-vs-shuffled contrast, applied by default regardless of balance.
- [det M degeneracy floor](det-m-degeneracy-floor.md) — A threshold (tau_detM = 0.413) below which planes are excluded because their pullback-metric area approaches zero, making holonomy ill-defined.
- [draw order](draw-order.md) — A sequential integer assigned to each surviving WikiText article after seed-42 shuffling, determining the experiment-vs-reserve partition.
- [enclosed area (wedge)](enclosed-area-wedge.md) — The pullback-metric area of the loop, computed as A_enclosed = rho^2 * sqrt(det M), which normalizes the rotation angle to yield holonomy.
- [eps_mag fallback](eps-mag-fallback.md) — When in-plane magnitude is below eps_mag = 2.66, the offset direction switches to G-normalized d1 instead of the projection direction.
- [experiment sample](experiment-sample.md) — The first 390 surviving base points (in seed order, excluding burned pilot points), constituting the fixed-N dataset for confirmatory analysis.
- [FALSIFIED verdict](falsified-verdict.md) — The verdict when the CI upper bound lies below the materiality threshold, indicating the effect is reliably too small to matter.
- [forward hook](forward-hook.md) — A PyTorch hook registered on a module to capture intermediate activations during the forward pass, used for extraction and JVP computation.
- [G-projection coefficients](g-projection-coefficients.md) — The coefficients a = M^{-1}(JD)^T(Jh) expressing the G-orthogonal projection of h onto the plane, used for center placement.
- [H-grad](h-grad.md) — The gradient hypothesis predicting a monotone ordering of holonomy: random < shuffled < real, indicating a semantic gradient.
- [H-mag](h-mag.md) — The magnitude hypothesis predicting that holonomy increases with in-plane pullback-metric magnitude, pooled across plane types.
- [H-sem](h-sem.md) — The powered semantic hypothesis predicting that real-feature planes show materially greater holonomy than magnitude-matched shuffled-feature planes after covariate adjustment.
- [holonomy-magnitude response](holonomy-magnitude-response.md) — The measured relationship between in-plane magnitude and holonomy (rotation per enclosed area), the primary response variable testing H-mag.
- [in-plane magnitude (pullback)](in-plane-magnitude-pullback.md) — The G-norm of a vector's G-orthogonal projection onto a plane, computed as mag(h) = sqrt(a^T M a) where M is the plane Gram matrix under the pullback metric.
- [inactive feature](inactive-feature.md) — An SAE feature with zero or negative activation (code <= 0) at a given base point, used as the second direction in shuffled-feature planes.
- [INCONCLUSIVE verdict](inconclusive-verdict.md) — The verdict when the CI spans the materiality threshold, indicating insufficient precision to determine materiality.
- [kNN distance fallback](knn-distance-fallback.md) — A fallback guard-distance method using mean k-nearest-neighbor distance when the covariance matrix is ill-conditioned for Mahalanobis.
- [log sin phi check](log-sin-phi-check.md) — A belt-and-suspenders balance diagnostic requiring |mean(log sin phi_real) - mean(log sin phi_shuffled)| < 0.045.
- [log-scale paired contrast](log-scale-paired-contrast.md) — The within-base-point difference d_b = log H_real - log H_shuffled, the primary test statistic for H-sem.
- [loop center](loop-center.md) — The center point c around which the holonomy loop is traced, positioned at the target in-plane magnitude via center placement.
- [loop radius](loop-radius.md) — The radius rho = radius_relative × ||base activation|| used for the holonomy loop, scaled to the local activation norm.
- [loop sweep](loop-sweep.md) — The n_steps + 1 points traced along the gamma(t) = c + rho(cos(2πt) d1 + sin(2πt) d2) circle for transport and covariate measurement.
- [magnitude level](magnitude-level.md) — One of two fixed in-plane magnitudes (m_low at 25th percentile, m_high at 75th percentile) cut from the common-support band and applied identically to all three arms.
- [magnitude-matched shuffled-feature plane](magnitude-matched-shuffled-feature-plane.md) — A control plane constructed by pairing one active SAE feature with one inactive real dictionary feature, matched to real-feature planes in pullback-metric in-plane magnitude.
- [manifold distance](manifold-distance.md) — A nuisance covariate measuring how far loop points lie from the learned manifold, via SAE reconstruction error and/or reference distribution distance.
- [materiality threshold](materiality-threshold.md) — The minimum effect size log(1.25) = 0.2231 on the log scale (a 25% multiplicative change) below which effects are not considered material.
- [minimum band width threshold](minimum-band-width-threshold.md) — The requirement that a common-support band have width >= 0.5 × pooled IQR to be considered valid for matched comparisons.
- [NULL-ATTRIBUTED verdict](null-attributed-verdict.md) — A special FALSIFIED branch for H-sem when an unadjusted effect vanishes under covariate adjustment, attributing the apparent effect to confounds.
- [one-metric principle](one-metric-principle.md) — The design choice to compute all geometric quantities (magnitude, area, transport) under a single consistent pullback metric G = J^T J.
- [out-of-plane component](out-of-plane-component.md) — The portion of the base activation orthogonal to the plane under the pullback metric, preserved during center placement.
- [pairing rule](pairing-rule.md) — The fixed procedure for constructing plane arms before holonomy computation, specifying how SAE features are selected and combined for each plane type.
- [phi (plane angle)](phi-plane-angle.md) — The mutual Euclidean angle between the two raw normalized plane directions, computed as arccos(d1 · d2).
- [plane arm](plane-arm.md) — One of three treatment levels in Factor A—real-feature, shuffled-feature, or random—each constructed according to the pairing rule.
- [planning tau](planning-tau.md) — The assumed standard deviation (tau = 1.30) of the log-scale paired contrast used to power the study at N = 390.
- [pooled IQR](pooled-iqr.md) — The interquartile range computed over the concatenated magnitude values from all arms in a band computation, used to set the minimum width threshold.
- [random plane](random-plane.md) — A noise-floor plane spanned by two random unit directions, providing the lower anchor of the H-grad semantic gradient.
- [readout map](readout-map.md) — The function F: resid_post_12 → resid_post_13 computed by patching layer-12 output and capturing layer-13 output, whose Jacobian drives transport.
- [real-feature plane](real-feature-plane.md) — A plane spanned by two jointly active SAE features at the base point, representing intact semantic structure.
- [reference distribution](reference-distribution.md) — The Mahalanobis (or kNN fallback) distribution fitted to experiment-sample activations, used as a guard proxy for manifold distance.
- [reserve pool](reserve-pool.md) — Base points beyond the 390 experiment sample, held back for dropout replacement or future analysis but never analyzed as part of the confirmatory study.
- [resid_post](resid-post.md) — The residual-stream activation at the output of a transformer block (post-layernorm, pre-next-block), the extraction site for this study.
- [restricted Jacobian](restricted-jacobian.md) — The 2-column matrix [J d1, J d2] giving the Jacobian of the readout map restricted to the plane directions, computed via JVP.
- [rotation angle (theta)](rotation-angle-theta.md) — The signed angle extracted from the antisymmetric part of the transport operator, representing the accumulated rotation around the loop.
- [run manifest](run-manifest.md) — The JSON artifact serializing all Stage A decisions (planes, centers, covariates, band) but structurally forbidding response fields.
- [shuffled-feature plane](shuffled-feature-plane.md) — A control plane pairing one active SAE feature with one inactive dictionary feature, magnitude-matched to the real-feature plane.
- [Stage A](stage-a.md) — The blind setup phase that extracts activations, selects planes, computes the magnitude band, constructs centers, measures covariates, and produces the manifest.
- [Stage B](stage-b.md) — The measurement phase that reads a Stage A manifest and writes holonomy response values to a separate artifact, never mutating the manifest.
- [symmetric residual](symmetric-residual.md) — The Frobenius norm of the symmetric part of (H - I), a diagnostic for non-rotational distortion in the transport operator.
- [three-arm band](three-arm-band.md) — The overlap of the [p5, p95] magnitude ranges across all three plane arms (real, shuffled, random), required to have width >= 0.5 × pooled IQR.
- [transport operator](transport-operator.md) — The 2×2 matrix H accumulating the effect of parallel-transporting a probe frame around the loop via iterated restricted Jacobian steps.
- [two-arm band](two-arm-band.md) — The overlap of the [p5, p95] magnitude ranges for real and shuffled arms only, used as a fallback when the three-arm band collapses.
- [two-arm terminal collapse](two-arm-terminal-collapse.md) — The fallback case where even the two-arm real+shuffled common-support band fails the minimum width requirement, making matched semantic comparison impossible.

---

## <span style="background-color: #448aff; color: white; padding: 0.2em 0.6em; border-radius: 0.25em; font-size: 0.85em; font-weight: 600;">INHERITED</span> Inherited Terms

*Standard terms from prior literature.*

- [alpha (significance level)](alpha-significance-level.md) — The pre-specified probability of falsely rejecting a true null hypothesis (Type I error rate), here 0.05.
- [attention mask](attention-mask.md) — A tensor indicating which tokens should attend to which, used to prevent attending to padding or future tokens.
- [autograd](autograd.md) — PyTorch's automatic differentiation engine that records operations on tensors and computes gradients via backpropagation.
- [bitwise reproducibility](bitwise-reproducibility.md) — The property that repeated runs produce identical outputs down to the bit level, given the same inputs, code, and hardware.
- [blocking unit](blocking-unit.md) — The unit (here base point) within which all treatment combinations are observed, reducing between-unit variance.
- [chi-squared distribution](chi-squared-distribution.md) — The distribution of the sum of squared standard normal random variables, used for variance confidence intervals.
- [closed loop](closed-loop.md) — A curve that starts and ends at the same point, around which parallel transport may accumulate a non-trivial rotation (holonomy).
- [common support / positivity](common-support-positivity.md) — The requirement that all treatment groups have overlapping covariate distributions, ensuring covariate adjustment does not extrapolate beyond observed data.
- [confidence interval](confidence-interval.md) — An interval estimate of a parameter such that repeated sampling would capture the true value a specified proportion (e.g., 95%) of the time.
- [confounding](confounding.md) — A situation where a third variable influences both the treatment and outcome, creating a spurious association.
- [connection](connection.md) — A rule for parallel-transporting tangent vectors along curves, defining how to compare vectors at different points.
- [covariance matrix](covariance-matrix.md) — A matrix whose (i,j) entry is the covariance between variables i and j, capturing the full correlation structure.
- [curvature](curvature.md) — A measure of how a manifold deviates from being flat, detected by the failure of parallel transport around closed loops to return vectors unchanged.
- [determinant](determinant.md) — A scalar value encoding the signed volume scaling factor of a linear transformation, with det = 0 indicating singularity.
- [effect size](effect-size.md) — A standardized measure of the magnitude of a phenomenon, independent of sample size, used for power analysis and scientific interpretation.
- [factorial design](factorial-design.md) — An experimental design that crosses multiple factors (here plane type × magnitude level), enabling estimation of main effects and interactions.
- [forward pass](forward-pass.md) — A single evaluation of the neural network from inputs to outputs, during which intermediate activations can be captured via hooks.
- [Frobenius norm](frobenius-norm.md) — The square root of the sum of squared matrix elements, a natural matrix norm corresponding to treating the matrix as a vector.
- [Gemma 2 2B](gemma-2-2b.md) — A 2-billion parameter decoder-only language model from Google DeepMind, the model under study.
- [Gemma Scope](gemma-scope.md) — A suite of sparse autoencoders trained on Gemma model activations, providing interpretable feature dictionaries.
- [Gram matrix](gram-matrix.md) — A matrix M = D^T G D encoding inner products of a set of vectors D under a metric G, whose determinant gives the squared volume of the parallelepiped they span.
- [holonomy](holonomy.md) — The rotation angle accumulated by parallel-transporting a vector around a closed loop, divided by the enclosed area, measuring local curvature.
- [inner product](inner-product.md) — A generalization of the dot product defining angles and lengths in a vector space, here given by the pullback metric.
- [interquartile range (IQR)](interquartile-range-iqr.md) — The difference between the 75th and 25th percentiles, a robust measure of spread.
- [Jacobian-vector product (JVP)](jacobian-vector-product-jvp.md) — The product J*v of the Jacobian matrix J with a vector v, computed efficiently via forward-mode automatic differentiation without materializing J.
- [JumpReLU](jumprelu.md) — An activation function that outputs zero below a learned threshold and the pre-activation above it, used in Gemma Scope SAEs.
- [L0 sparsity](l0-sparsity.md) — The number of non-zero elements in the SAE code vector, measuring how many features are active for a given activation.
- [Mahalanobis distance](mahalanobis-distance.md) — A distance metric that accounts for correlations in the data by scaling by the inverse covariance matrix, measuring how many standard deviations a point lies from a distribution.
- [matrix condition number](matrix-condition-number.md) — The ratio of the largest to smallest singular value, measuring sensitivity of matrix operations to numerical error.
- [mixed-effects model](mixed-effects-model.md) — A regression model containing both fixed effects (population-level coefficients) and random effects (group-level variation), used here to model base-point blocking.
- [MPS backend](mps-backend.md) — Apple's Metal Performance Shaders backend for PyTorch, enabling GPU-accelerated computation on Apple Silicon.
- [orthogonal projection](orthogonal-projection.md) — The closest point in a subspace to a given vector, obtained by minimizing the distance under the chosen metric.
- [parallel transport](parallel-transport.md) — The procedure for moving a tangent vector along a curve while keeping it as parallel as possible according to the connection, preserving its length under the metric.
- [percentile](percentile.md) — The value below which a given percentage of observations fall, used to define magnitude level cutpoints.
- [power](power.md) — The probability of correctly rejecting a false null hypothesis, here 0.90 for detecting the material effect size at the corrected-site variance.
- [pre-registration](pre-registration.md) — The practice of publicly committing to hypotheses, methods, and analysis plans before observing data, preventing post-hoc hypothesizing.
- [pseudoinverse](pseudoinverse.md) — A generalization of the matrix inverse that exists for non-square or singular matrices, minimizing the least-squares residual.
- [pullback metric](pullback-metric.md) — The induced metric G = J^T J on the input space, pulled back from the output space via the Jacobian J, measuring distances as the model perceives them.
- [PyTorch tensor](pytorch-tensor.md) — A multi-dimensional array supporting automatic differentiation, the fundamental data structure for neural network computation.
- [random seed](random-seed.md) — An integer initializing a pseudorandom number generator, ensuring reproducible random sequences.
- [randomization](randomization.md) — The use of random assignment to treatment conditions, ensuring unbiased estimation and validity of statistical inference.
- [residual stream](residual-stream.md) — The additive information flow through a transformer where each layer adds its contribution to the accumulated hidden state.
- [SAE decoder](sae-decoder.md) — The W_dec matrix of the SAE whose rows are the learned dictionary directions (features), used to span planes in feature-based arms.
- [SAE encoder](sae-encoder.md) — The W_enc matrix and bias b_enc of the SAE that maps activations to sparse codes via x @ W_enc + b_enc followed by thresholding.
- [SAE reconstruction error](sae-reconstruction-error.md) — The L2 distance between an activation and its reconstruction through a sparse autoencoder, serving as a proxy for how well the SAE captures the activation.
- [sparse autoencoder (SAE)](sparse-autoencoder-sae.md) — A neural network that learns a sparse overcomplete dictionary for reconstructing activations, decomposing them into interpretable feature directions.
- [standard deviation](standard-deviation.md) — The square root of the variance, a measure of spread in the same units as the data.
- [standardized mean difference (SMD)](standardized-mean-difference-smd.md) — The difference in group means divided by the pooled standard deviation, a scale-free measure of effect size used in balance diagnostics.
- [tangent vector](tangent-vector.md) — A vector in the tangent space at a point, representing an infinitesimal direction of motion on the manifold.
- [token](token.md) — A discrete unit of text (word, subword, or character) produced by the tokenizer and consumed by the model.
- [tokenizer](tokenizer.md) — The algorithm that converts raw text to token IDs and vice versa, determining the vocabulary and segmentation.
- [treatment effect](treatment-effect.md) — The causal difference in outcomes attributable to a treatment, here the holonomy difference between plane types.
- [unit vector](unit-vector.md) — A vector normalized to have length 1 under the relevant norm, used to represent directions.
- [variance](variance.md) — The expected squared deviation from the mean, measuring the spread of a distribution.
- [vector norm](vector-norm.md) — A function assigning a non-negative length to a vector, typically the Euclidean (L2) norm.
- [WikiText-103](wikitext-103.md) — A large-scale language modeling dataset of Wikipedia articles, used as the corpus source for base-point passages.
