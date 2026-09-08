# Parameter Estimation for Participant Data: math ↔ code (Laplace inner marginalization)

This is the math to code writeup for the Laplace approximation implementation of the model (as opposed to the earlier implementation, which used NUTS + Monte-Carlo averaging).
The code for this parameter estimation is all in `inference.py`, and the code for the model of the learner is in `model_jax.py`.

Notation:  
$\theta = (\ell, \mu_0, \sigma, \beta)$ = the parameters of the coherence function to fit: GP length scale, GP mean, GP output scale, RSA speaker rationality  
$\vec{u}$ = training utterances  
$\mathbf{r}$ = prevalence ratings $(N,J)$ (# participants, # test features)  
$\vec{y},\vec{y}'$ = latent train/test pseudocoherences;  
$y = [\vec{y}';\vec{y}]$ is the flat vector the code optimizes,   
$D = \dim(y)$  
$\phi$ = Beta-mixture linking function shapes
(naming collision: in code, `phi` is the vector of **VBMC coordinates** $\psi$ below, *not* the shapes — shapes are `link_shapes`.)

## Inference
As "the scientists" we want to estimate which parameters are most likely to give rise to the participant prevalence response data $\mathbf{r}$ given the utterances $\vec{u}$.

$$
\underbrace{P(\theta \mid \mathbf{r},\vec{u})}_{\texttt{VBMC.optimize()}}
\;\propto\;
\underbrace{P(\mathbf{r}\mid\vec{u},\theta)}_{\texttt{total\_log\_lik}}
\;\cdot\;
\underbrace{P(\theta)}_{\texttt{log\_prior\_theta}}
$$

pyVBMC returns the following:
- **posterior samples** over $\theta$ (`ls_samples`, `mu0_samples`, `sigma_samples`, `beta_samples`)
- the **ELBO** $\approx \log \int P(\mathbf{r}\mid\vec{u},\theta)\,P(\theta)\,d\theta$,
  the per-theta / per-"model" log evidence used for model comparison.

Everything below is one evaluation of $\log P(\mathbf{r}\mid\vec{u},\theta)$ —
the body of `log_joint(phi)` (`make_log_joint*`), which sums the per-condition
Laplace marginal $\log Z_c(\theta)$ over conditions (`total_log_lik`).

## Likelihood, marginalizing out latent coherences

For a fixed set of parameters and fixed utterance set $\vec{u}, we calculate the likelihood of the human ratings $mathbf{r}$, marginalizing out the possible latent coherences.
$$
\log Z(\theta) = \log P(\mathbf{r}\mid\vec{u},\theta)
= \log \int P(\mathbf{r}\mid\vec{y}')\,P(\vec{y}',\vec{y}\mid\vec{u},\theta)\,dy
$$

Define the **log joint over the latents** (note: unlike the MCMC sampling
target, this *includes* the ratings term)

$$
g(y) \;=\;
\underbrace{\log\mathrm{MVN}\big(y;\,\mu_0\mathbf{1},\,K_\theta\big)}_{\substack{\texttt{\_mvn\_logpdf\_chol} \\ K_\theta = \texttt{rbf\_kernel}}}
\;+\;
\underbrace{\sum_i \log P(u_i\mid y_i)}_{\texttt{p\_u\_given\_y}}
\;+\;
\underbrace{\sum_{ij} \log\!\Big[\mathrm{pz1}_j\,\mathcal{B}(r_{ij};\phi^{kl}_j) + (1-\mathrm{pz1}_j)\,\mathcal{B}(r_{ij};\phi^{nkl}_j)\Big]}_{\substack{\texttt{logaddexp(log(pz1)+logpdf\_kl,}\\\texttt{\ \ log1p(-pz1)+logpdf\_nkl)}}}
$$

with $\mathrm{pz1}_j = \operatorname{sigmoid}(y'_j)$. The code works with
$-g$ = `_neg_log_post`, jitted once with val/grad/hess
(`_val_fn` / `_grad_fn` / `_hess_fn`; $\theta$ and the data are traced
arguments, so one compilation serves every condition and every $\theta$).
The shapes $\phi$ don't depend on $y$, so the per-$(i,j)$ component log-densities
are precomputed in `prepare_condition` (`logpdf_kl`, `logpdf_nkl`).

**Mode:** damped Newton with backtracking (Armijo) line search minimizes $-g$
from the init $y = \mu_0\mathbf{1}$:

$$
\hat{y} = \arg\max_y g(y)
\qquad \leadsto \qquad \texttt{laplace\_log\_Z} \text{ (Newton loop)}
$$

**Laplace approximation:** with $H = -\nabla^2 g(\hat{y})$ (the posterior
precision, PD-guarded with $10^{-6}I$),

$$
\log Z(\theta) \;\approx\;
\underbrace{g(\hat{y}) + \tfrac{D}{2}\log 2\pi - \tfrac12 \log\det H}_{\texttt{laplace\_log\_Z} \ \to\ (\texttt{log\_Z},\ \hat{y},\ \texttt{ok})}
$$

The `ok` flag checks $H \succ 0$, finiteness, and $\|\nabla g(\hat{y})\|_\infty < 10^{-2}$.

$$
\log P(\mathbf{r}\mid\vec{u},\theta) = \sum_c \log Z_c(\theta)
\qquad \leadsto \qquad \texttt{total\_log\_lik}
$$

### Density, not interval mass

The MCMC doc's rating likelihood uses the **interval mass**
$F(r+\tfrac1{200})-F(r-\tfrac1{200})$. The Laplace objective needs a smooth,
twice-differentiable $g$, so here each $\mathcal{B}$ is the Beta **density**,
with exact 0/1 ratings nudged half a slider bin inward (`H = 1/200`,
`clip_interior` in `load_responses`). The interval-mass likelihood survives
only inside the shape fit (next section). Consequence: near-delta shape
components that are harmless under interval mass are pathological under the
density, hence `SHAPE_CLIP`.

## Linking shapes $\phi$: fixed or profiled out
<!-- TODO: fix what should be named phi, do we want it to be the linking function or the coherence function? according to overleaf, coherence function. according to old code, ?? -->

**Fixed** (`make_log_joint`, `run_vbmc`): $\phi$ = `DEFAULT_LINK_SHAPES`,
baked into `cond_data` by `prepare_condition`.

**Profiled** (`make_log_joint_free_shapes`, `run_vbmc_free_shapes`): at every
$\theta$, re-fit shared per-feature shapes empirical-Bayes style — a **profile
likelihood**, not a marginalization. With the default `n_alt=1` (the
MCMC-notebook analog: the coherence step never sees the test ratings):

$$
\mathrm{pz1}(\theta) = \operatorname{sigmoid}(\hat{y}'_{\text{ratings-free}})
\;\leadsto\; \texttt{prior\_pz1\_modes}
\qquad
\hat\phi(\theta) = \arg\max_\phi \log P_{\text{interval}}(\mathbf{r}\mid\mathrm{pz1},\phi)
\;\leadsto\; \texttt{fit\_link\_shapes}
$$

$$
\ell\ell(\theta) = \textstyle\sum_c \log Z_c\big(\theta;\hat\phi(\theta)\big)
\;\leadsto\; \texttt{total\_log\_lik\_free\_shapes}
$$

The ratings-free mode maximizes GP prior + speaker only
(`_ratings_free_cond_data`: zero ratings rows). `n_alt > 1` alternates
mode ↔ shape-fit with ratings included, but the two objectives differ
(interval fit vs. density log Z), so extra rounds are **not monotone** in
$\ell\ell$ — see the caveat comment in `inference.py`.

## VBMC layer: coordinates, priors, variants

VBMC runs over unconstrained-ish coordinates
$\psi = [\log\ell,\ \mu_0,\ \log\sigma\,(,\ \log\beta)]$ (code name: `phi`)
inside a plausible box (`X0/LB/UB/PLB/PUB` and variants; the $\log\ell$
coordinate follows the active embedding profile — `set_embed('2d'|'384d')`).
Independent Gaussian priors in $\psi$:

$$
\log\ell \sim \mathcal{N}(\log \ell_0, 1.5^2),\quad
\mu_0 \sim \mathcal{N}(0,1),\quad
\log\sigma \sim \mathcal{N}(\log 1.5, 1),\quad
\log\beta \sim \mathcal{N}(\log 3, 1)
$$

($\ell_0$ = `ls_prior_center` of the embedding profile.)

The objective is **deterministic**, but is declared to pyVBMC with a small
constant observation noise (`TARGET_NOISE`, `specify_target_noise=True`) to
(1) avoid a gpyreg exact-observation gradient bug and (2) regularize the
surrogate. It is tiny relative to the spread of `log_joint`, so it doesn't
smear the posterior.

| variant | $\psi$ | shapes | driver |
|---|---|---|---|
| GP model | $[\log\ell, \mu_0, \log\sigma]$ | fixed / profiled | `run_vbmc` / `run_vbmc_free_shapes` |
| null (distance-blind): $\ell$ pinned at `LS_FLAT` | $[\mu_0, \log\sigma]$ | fixed / profiled | `run_vbmc_fixed_ls` / `run_vbmc_free_shapes_fixed_ls` |
| GP + free rationality $\beta$ | $[\log\ell, \mu_0, \log\sigma, \log\beta]$ | fixed / profiled | `run_vbmc_beta` / `run_vbmc_free_shapes_beta` |
| null + free $\beta$ | $[\mu_0, \log\sigma, \log\beta]$ | fixed / profiled | `run_vbmc_fixed_ls_beta` / `run_vbmc_free_shapes_fixed_ls_beta` |

Pinned-$\ell$ nulls omit the (constant) $\ell$-prior term, so their raw
log-joint values are not comparable to the GP fits' — compare via ELBO /
summed $\log Z$.

## MCMC vs. Laplace

| | MCMC (`math-to-code-mcmc.md`) | Laplace (here) |
|---|---|---|
| inner integral over $y$ | NUTS samples of $P(\vec{y}',\vec{y}\mid\vec{u},\theta)$ (no ratings), then $\tfrac1S\sum_s P(\mathbf{r}\mid\vec{y}'_s)$ | Gaussian around the mode of the **full** joint (ratings included) |
| rating likelihood | interval mass | density (half-bin nudge), interval mass only in the shape fit |
| stochastic? | yes (per-eval MC noise) | no — deterministic given $(\theta, \texttt{n\_alt})$ |
| shape fit | per-draw refit (or mean-pz1 proposal) | once per $\theta$ at the ratings-free mode pz1 (`n_alt=1`) |

Both target the same integral $\int P(\mathbf{r}\mid\vec{y}')P(\vec{y}',\vec{y}\mid\vec{u},\theta)\,dy$;
the estimators differ in where the ratings enter and in bias/variance
trade-off (MC: unbiased but noisy; Laplace: deterministic but biased where the
latent posterior is non-Gaussian).
