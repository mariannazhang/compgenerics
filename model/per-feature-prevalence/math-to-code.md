# Pipeline: math ↔ code

Maps the scientist-inference pipeline (one `log_joint(θ)` evaluation) to its
implementation in `model_jax.py` and `model-study9-vbmc.ipynb`. Notation follows
`README.md` and the handwritten 6-step diagram.

Symbols:

| symbol | meaning | shape |
|---|---|---|
| $\theta = (\ell, \mu_0)$ | GP params being inferred: length scale, prior mean | — |
| $\sigma, \beta$ | output scale, speaker rationality (currently **fixed**: 1.5, 3.0) | — |
| $\vec{u}$ | training utterance types (all generic: $u_i = 0$) | $(n_\text{tr}=45,)$ |
| $\vec{y}$ | training pseudocoherences (latent) | $(45,)$ |
| $\vec{y}'$ | test pseudocoherences (latent) | $(J=15,)$ |
| $z$ | kind-linkedness indicator, $z\in\{0,1\}$ | — |
| $\mathbf{r}$ | participant prevalence ratings, $r_{ij}\in[0,1]$ | $(N=402, J)$ |
| $\phi$ | linking-function (Beta-mixture) shape params per feature | $(J, 4)$ |

---

## Outer loop — scientist inference $P(\theta \mid \text{data})$

$$
P(\theta \mid \mathbf{r}, \vec{u}) \;\propto\; P(\mathbf{r} \mid \vec{u}, \theta), P(\theta)
$$

pyVBMC proposes $\theta$ in unconstrained space $\varphi = (\log\ell, \mu_0)$ and
calls `log_joint(phi)` for each. **Code:** `cell-vbmc-run` builds
`VBMC(log_joint, x0, lb, ub, plb, pub, ...)`.

**Log prior** $\log P(\theta)$ — log-normal on $\ell$, Gaussian on $\mu_0$:

$$
\log P(\theta) = -\tfrac12\!\left(\frac{\log\ell - \log 0.5}{1.5}\right)^2 \;-\; \tfrac12\,\mu_0^2
$$

**Code** (`cell-log-joint`):
```python
log_prior_ls  = -0.5 * ((log_ls - np.log(0.5)) / 1.5) ** 2
log_prior_mu0 = -0.5 * mu_0 ** 2
```

The rest of this file is **one evaluation of** $\log P(\mathbf{r}\mid\vec{u},\theta)$,
the expensive log-likelihood, which is what we Monte-Carlo estimate.

---

## The quantity being estimated

The pseudocoherences are latent and integrated out:

$$
P(\mathbf{r}\mid\vec{u},\theta)
= \int P(\mathbf{r}\mid \vec{y}')\;P(\vec{y}',\vec{y}\mid\vec{u},\theta)\;d\vec{y}'d\vec{y}
$$

Estimated by sampling $\vec{y}' \sim P(\vec{y}'\mid\vec{u},\theta)$ (steps 1–3) and
averaging the rating-likelihood $P(\mathbf{r}\mid\vec{y}')$ over draws (steps 4–5).

> **Key decoupling:** the sampling target (step 1) contains **only** utterances and
> the GP prior — *not* the ratings $\mathbf{r}$ and *not* the linking params $\phi$.
> The linking function is the observation model, applied only at scoring (steps 4–5).
> So the $\vec{y}'$ draws are fully determined before any Beta fitting happens.

---

## Step 1 — build the MCMC target density

$$
\log P(\vec{y}',\vec{y}\mid\vec{u},\theta)
\;\propto\;
\underbrace{\sum_i \log P(u_i\mid y_i)}_{\text{speaker / utterance model}}
\;+\;
\underbrace{\log \mathrm{MVN}\!\big(\vec{y}',\vec{y};\mu_0,K_\theta\big)}_{\text{GP prior}}
$$

**Code:** `make_log_density_fn_joint(u_train, x_train, x_test, params)` →
`log_density(position)`, with `position = {training_coherences, test_coherences}`.

### Speaker / utterance model $P(u_i \mid y_i)$

RSA stack, all in `model_jax.py`:

$$
P(z{=}1\mid y) = \mathrm{sigmoid}(y)
\qquad
P(u_i\mid y_i) = \sum_{z\in\{0,1\}} S_1(u_i\mid z)\,P(z\mid y_i)
$$

| math | function |
|---|---|
| $L_0(z\mid u)$ literal listener | `literal_listener(z_i, u_i, prior_z1)` |
| $S_1(u\mid z)$ ε-noisy speaker, softmax over $\beta\cdot$utility | `speaker(u_i, z_i, beta, prior_z1, epsilon=0.05)` |
| $P(u_i\mid y_i)$ | `p_u_given_y(u_i, y_i, beta)` |

$\beta$ (rationality) enters **only here**, via `jax.nn.softmax(beta * utilities)`.

### GP prior $\mathrm{MVN}(\,\cdot\,;\mu_0, K_\theta)$

$$
K_\theta[a,b] = \sigma^2 \exp\!\left(-\frac{\lVert x_a - x_b\rVert^2}{2\ell^2}\right) + 10^{-4}\delta_{ab}
$$

| math | function |
|---|---|
| RBF kernel $K_\theta$ | `rbf_kernel(X_all, length_scale, output_scale)` |
| $\log\mathrm{MVN}$ via Cholesky | `_mvn_logpdf_chol(full_y, mu, cov)` |

$\ell$ and $\mu_0$ (the inferred params) enter **only here**. Features are stacked
test-first: `X_all = vstack([test_features, x_vec])`.

---

## Step 2 — NUTS sampling (one chain)

Sample the joint $(n_\text{tr}+J = 60)$-dim vector with BlackJAX NUTS:
300 warmup → `N_SAMPLES` steps; each step yields one full proposal vector.
Keep test coherences only → `test_coherences_all`, shape `(N_SAMPLES-1, J)`.

**Code** (`cell-log-joint`): `window_adaptation` warmup, then the `one_step` loop
appending `state.position['test_coherences']`.

$$
\{\vec{y}'^{(t)}\}_{t=1}^{\sim 1500} \sim P(\vec{y}'\mid\vec{u},\theta)
$$

---

## Step 3 — thinning

Keep every `STEP=5`-th draw → $S \approx 300$ samples, reducing autocorrelation
(single chain, so draws are still correlated — $S$ overstates effective sample size).

$$
\{\vec{y}'_s\}_{s=1}^{S},\qquad \mathrm{pz1}_s = \mathrm{sigmoid}(\vec{y}'_s)\in[0,1]^J
$$

---

## Step 4 — score each draw under the linking function

For each thinned draw, the rating-likelihood is a per-feature Beta **mixture**, with
the mixture weight given by that draw's pz1:

$$
P(\mathbf{r}\mid \vec{y}'_s)
= \prod_{j=1}^{J}\prod_{i=1}^{N}
\Big[\;\mathrm{pz1}_{s,j}\,\mathcal{B}(r_{ij};\alpha^{kl}_j,\beta^{kl}_j)
\;+\;(1-\mathrm{pz1}_{s,j})\,\mathcal{B}(r_{ij};\alpha^{nkl}_j,\beta^{nkl}_j)\Big]
$$

### Interval (binned) likelihood

Ratings are discrete 0–100 slider values, so a rating $r$ is the event
$r\in[r-h, r+h]$ with $h = 1/200$, scored by the mixture CDF — **bounded by
construction** (each term $\le 1$), unlike the density, which is unbounded when a
component collapses to a spike on a tie pile:

$$
P(r) = F(r+h) - F(r-h),\qquad F = \text{Beta mixture CDF}
$$

| math | function | notes |
|---|---|---|
| per-feature negative interval log-lik | `_neg_ll_one_feature(log_params, r, th, counts)` | float64; `scipy.special.betainc` for $F$ |
| fit $\phi$ for all features | `fit_beta_mixtures_all_features(responses, pz1)` | best-of-4-inits L-BFGS-B; compacts to unique $(r,\theta)$ pairs |
| total log-lik given $\phi$ | `beta_mixture_log_likelihood(responses, pz1, beta_params)` | sums over $N\times J$ |

Shape params capped at $\log$-param $\le 15$ (`_LOG_PARAM_LIMIT`): once a component
is narrower than a response bin, $\alpha,\beta$ are unidentifiable and the likelihood
plateaus. `alpha_kl` is forced to be the higher-mean component.

### ⚠️ Where the linking fit happens — current vs. proposed

This is the only structural change under discussion. **The mixture weight
$\mathrm{pz1}_s$ varies per draw in both versions** (that *is* the $\vec{y}'$
dependence / the marginalization). Only the *shape params* $\phi$ move.

**CURRENT — refit $\phi$ every draw** (`cell-log-joint`, the `for s` loop):

$$
\hat\phi_s = \arg\max_\phi P(\mathbf{r}\mid\mathrm{pz1}_s,\phi),
\qquad
\ell\ell_s = \log P(\mathbf{r}\mid\mathrm{pz1}_s,\hat\phi_s)
$$

```python
for s in range(0, test_coherences_all.shape[0], STEP):
    pz1_s    = jax.nn.sigmoid(test_coherences_all[s])     # weight, varies
    beta_params_s = fit_beta_mixtures_all_features(responses, tile(pz1_s, (N,1)))  # REFIT
    ll_s = beta_mixture_log_likelihood(responses, pz1_s, beta_params_s)
```

Problem: this computes $\mathbb{E}_s[\max_\phi \ell\ell]$, not $\max_\phi \mathbb{E}_s[\ell\ell]$.
Each draw re-optimizes 60 shape params to itself, so extreme-pz1 draws spike on the
50-piles and inflate $\ell\ell_s$. Inflation is largest where pz1 is most variable
(small $\ell$), biasing $\theta$ toward the boundary and giving `noise_std` in the
hundreds–thousands.

**PROPOSED — fit $\phi$ once per evaluation** on the posterior-mean pz1:

$$
\bar{\mathrm{pz1}}_j = \frac1S\sum_{s} \mathrm{pz1}_{s,j},
\qquad
\hat\phi = \arg\max_\phi P(\mathbf{r}\mid\bar{\mathrm{pz1}},\phi),
\qquad
\ell\ell_s = \log P(\mathbf{r}\mid\mathrm{pz1}_s,\hat\phi)
$$

```python
thinned   = test_coherences_all[::STEP]            # (S, J)
pz1_draws = jax.nn.sigmoid(thinned)                # (S, J), weight per draw
pz1_bar   = jnp.mean(pz1_draws, axis=0)            # (J,) posterior mean P(z=1)
beta_params = fit_beta_mixtures_all_features(responses, jnp.tile(pz1_bar, (N,1)))  # ONE fit
log_liks_s = [beta_mixture_log_likelihood(responses, pz1_draws[s], beta_params)
              for s in range(pz1_draws.shape[0])]
```

Note `mean(sigmoid(y))`, **not** `sigmoid(mean(y))`. Removes per-draw shape variance
(`noise_std` → tens) and the small-$\ell$ inflation; also ~$S\times$ faster (one fit
instead of ~300). Alternative ("fit across all draws"): choose $\hat\phi$ to maximize
the pooled step-5 objective — more self-consistent when the pz1 posterior is wide.

---

## Step 5 — aggregate over draws

Monte-Carlo estimate of the marginal log-likelihood:

$$
\log P(\mathbf{r}\mid\vec{u},\theta)
\approx \operatorname{logsumexp}_s(\ell\ell_s) - \log S
= \log\frac1S\sum_s P(\mathbf{r}\mid \vec{y}'_s)
$$

$$
\texttt{noise\_std} = \mathrm{std}_s(\ell\ell_s)\ \longrightarrow\ \text{reported to pyVBMC}
$$

**Code:** `scipy_logsumexp(log_liks_s) - np.log(len(log_liks_s))`;
`noise_std` returned alongside (pyVBMC `specify_target_noise=True`).

---

## Step 6 — return

$$
\texttt{log\_joint}(\theta) = \log P(\mathbf{r}\mid\vec{u},\theta) + \log P(\theta)
$$

`return log_joint_val, noise_std`.

---

## Fixed vs. inferred parameters (entry points)

| param | role | enters at | status |
|---|---|---|---|
| $\ell$ length scale | GP correlation range | step 1, `rbf_kernel` | **inferred** |
| $\mu_0$ prior mean | GP mean coherence | step 1, `_mvn_logpdf_chol` | **inferred** |
| $\sigma$ output scale | GP marginal variance | step 1, `rbf_kernel` | fixed 1.5 (candidate to free) |
| $\beta$ rationality | speaker softmax temp | step 1, `speaker` | fixed 3.0 (candidate to free; weak ID — all utterances generic) |
| $\phi$ Beta shapes | linking function | step 4, `fit_beta_mixtures_all_features` | profiled per eval |
