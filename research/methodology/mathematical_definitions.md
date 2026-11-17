# Mathematical Definitions

## 1. Models and heads

Let the encoder be (M_E) and decoder be (M_D). A head is indexed by (h=(ell,r)), where (ell) is layer and (r) is head index within that layer. For the initial checkpoints, the expected head count is (H_E=H_D=12\times12=144); verify from model configs.

For input example (x), let (Z_h^M(x)\in\mathbb{R}^{T\times d_h}) denote the head's context-vector output before concatenation/output projection, with padding positions excluded from pooling.

## 2. Comparable per-example representations

Define a documented pooling operator (P_M) over valid token positions. The pooled vector is:

[
u_h^M(x)=P_M(Z_h^M(x))\in\mathbb{R}^{d_h}.
]

Because head dimensions may differ across models, the similarity method must support different feature dimensions. The same raw text examples and sample IDs must be used for both models. Tokenization and masks remain model-specific and are recorded. Pooling variants (masked mean, designated readout position, task-readout contribution) are separate conditions, not silently mixed.

## 3. Linear CKA

Given (n) paired examples, stack pooled vectors into (X\in\mathbb{R}^{n\times p}) and (Y\in\mathbb{R}^{n\times q}). Center columns to obtain (X_c,Y_c). Linear centered kernel alignment is:

[
\operatorname{CKA}(X,Y)=
\frac{\|X_c^\top Y_c\|_F^2}
{\|X_c^\top X_c\|_F\,\|Y_c^\top Y_c\|_F}.
]

Equivalent centered Gram-matrix form may be used. Guard against zero denominators and report numerical precision. CKA measures similarity of sample-wise geometry; it does not imply causal equivalence.

For every encoder head (i) and decoder head (j), compute (S^{repr}_{ij}=\operatorname{CKA}(X_i,Y_j)), producing a (144\times144) matrix.

## 4. Partial matching

Let (S\in\mathbb{R}^{H_E\times H_D}) be a similarity matrix. Find a partial injective matching (\mathcal{M}\subseteq\{1,\dots,H_E\}\times\{1,\dots,H_D\}) maximizing:

[
\max_{\mathcal{M}}\sum_{(i,j)\in\mathcal{M}}(S_{ij}-\tau)
]

subject to each source and target head appearing at most once. Here (\tau) is a predeclared match threshold or unmatched penalty. Implement using an augmented assignment matrix with dummy unmatched nodes. Do not force every head to match. Record unmatched source and target heads and sensitivity to (\tau).

Baselines:
- same layer/head index;
- layer-constrained assignment;
- unconstrained global partial assignment;
- random assignment;
- shuffled/permuted sample-label null.

## 5. Bootstrap stability

For bootstrap replicate (b\), sample paired example indices with replacement, recompute (S^{(b)}) and the partial matching (\mathcal{M}^{(b)}). For candidate pair ((i,j)):

[
C_{ij}=\frac{1}{B}\sum_{b=1}^{B}\mathbb{1}[(i,j)\in\mathcal{M}^{(b)}].
]

Report pair frequency, per-head matched frequency, assignment Jaccard similarity across replicates and score intervals. Five replicates are for debugging only; pilot inference should use at least 50 and final analyses target 200 where feasible.

## 6. Task-conditioned causal response

For a shared classification task, let logits be (z^M(x)\in\mathbb{R}^{K}) and true label (y). Define the true-class margin:

[
m^M(x)=z_y^M(x)-\max_{c\ne y}z_c^M(x).
]

Introduce an intervention gate (g_h\in\{0,1\}) multiplying the contribution of head (h) at its output interface. Exact ablation effect:

[
\delta_h^M(x)=m^M(x;g_h=1)-m^M(x;g_h=0).
]

A positive value means the head supports the correct-class margin under this sign convention. Also report loss difference and prediction flips. Gradient proxy:

[
r_h^M(x)=\left.\frac{\partial m^M(x;g_h)}{\partial g_h}\right|_{g_h=1}.
]

Gradient proxy is not interchangeable with exact ablation; validate it empirically.

For each head, the causal signature is the vector (c_h^M=[\delta_h^M(x_1),\dots,\delta_h^M(x_n)]\), optionally augmented with class-conditional effects and loss deltas. Standardize only using calibration data and preserve raw values.

## 7. Causal similarity and combined score

Candidate causal similarity may use centered cosine similarity or correlation between causal signatures, with a documented handling of near-zero signatures:

[
S^{causal}_{ij}=\operatorname{corr}(c_i^{E},c_j^{D}).
]

A combined score may be explored:

[
S_{ij}=\alpha S^{repr}_{ij}+\beta S^{causal}_{ij}+\gamma S^{aux}_{ij},
]

where (S^{aux}) is optional and must be defined before use. The primary method should not tune weights on final test outcomes. Prefer causal similarity as primary and representation similarity as a baseline/auxiliary signal. Report each component separately.

## 8. Importance and transfer

Source head importance may be estimated on source calibration data by mean absolute gradient of loss with respect to gate:

[
I_i^E=\mathbb{E}_{x\sim D_{cal}}\left[\left|\frac{\partial\mathcal{L}(x)}{\partial g_i}\right|\right].
]

Alternative importance measures (HIES, exact ablation) are separate baselines. For target head (j), confidence-weighted transfer is:

[
\widehat I_j^D=
\frac{\sum_i C_{ij}w_{ij}I_i^E}
{\sum_i C_{ij}w_{ij}+\epsilon},
]

where (w_{ij}\) is a nonnegative normalized match weight and low-confidence/unmatched heads are explicitly abstained from. Specify whether larger importance means "retain" or "prune"; never invert this convention implicitly.

## 9. Target pruning utility and interactions

For target head set (P), define:

[
U_D(P)=\mathcal{L}_D(M_D\setminus P)-\mathcal{L}_D(M_D).
]

This is the loss increase caused by removing (P); lower is better for a fixed pruning budget. Per-head utility is (U_D(\{j\})). Importance scores are predictors, not utility itself.

Pairwise interaction:

[
\kappa_{ij}=U_D(\{i,j\})-U_D(\{i\})-U_D(\{j\}).
]

Nonzero interaction means joint effects are not additive. Use group ablations for selected pairs/sets and avoid claiming independent head scores fully describe joint pruning.

## 10. Statistical reporting

Report paired bootstrap confidence intervals for performance differences, rank correlations (Spearman and Kendall where appropriate), top-k overlap, effect sizes, and multiple-comparison correction for families of confirmatory tests. Seeds, split IDs and all thresholds must be logged. Do not select thresholds based on final test results.
