# Parameter Estimation for Participant Data: math ↔ code

Notation: $\theta$ = GP params; $\vec{u}$ = utterances; $\mathbf{r}$ = ratings $(N,J)$;
$\vec{y},\vec{y}'$ = latent train/test pseudocoherences; $\phi$ = Beta-mixture shapes.

## The inference problem

$$
\underbrace{P(\theta \mid \mathbf{r},\vec{u})}_{\texttt{VBMC.optimize()}}
\;\propto\;
\underbrace{P(\mathbf{r}\mid\vec{u},\theta)}_{\texttt{log\_lik}}
\;\cdot\;
\underbrace{P(\theta)}_{\texttt{log\_prior}}
$$

Everything below is one evaluation of $P(\mathbf{r}\mid\vec{u},\theta)$ — the body of
`log_joint(phi)`.

## Likelihood = marginalize out the latent coherences

$$
P(\mathbf{r}\mid\vec{u},\theta)
= \int P(\mathbf{r}\mid\vec{y}')P(\vec{y}',\vec{y}\mid\vec{u},\theta)d\vec{y}d\vec{y}'
\;\approx\;
\underbrace{\frac1S\sum_{s=1}^{S} P(\mathbf{r}\mid\vec{y}'_s)}_{\texttt{logsumexp(log\_liks\_s) - log S}},
\qquad
\underbrace{\vec{y}'_s \sim P(\vec{y}',\vec{y}\mid\vec{u},\theta)}_{\substack{\texttt{blackjax NUTS on}\\\texttt{make\_log\_density\_fn\_joint}}}
$$

The sampling target involves only $\vec{u}$ and the GP — **not** $\mathbf{r}$ or $\phi$ —
so the draws $\vec{y}'_s$ exist before any Beta fitting.

## The sampling target (what NUTS explores)

$$
\log P(\vec{y}',\vec{y}\mid\vec{u},\theta)
\;\propto\;
\underbrace{\sum_i \log P(u_i\mid y_i)}_{\substack{\texttt{p\_u\_given\_y} \to \texttt{speaker} \\ \to\ \texttt{literal\_listener}}}
\;+\;
\underbrace{\log\mathrm{MVN}\big(\vec{y}',\vec{y};\,\mu_0,\,K_\theta\big)}_{\substack{\texttt{\_mvn\_logpdf\_chol} \\ K_\theta = \texttt{rbf\_kernel}}}
$$

with $P(z{=}1\mid y)=\operatorname{sigmoid}(y)$ and
$P(u_i\mid y_i)=\sum_z S_1(u_i\mid z)\,P(z\mid y_i)$.

## Per-draw rating likelihood (the linking function)

$$
P(\mathbf{r}\mid\vec{y}'_s)
= \prod_{j}\prod_{i}
\Big[\,\underbrace{\operatorname{sigmoid}(y'_{s,j})}_{\mathrm{pz1}_{s,j}}\,\mathcal{B}(r_{ij};\phi^{kl}_j)
+ (1-\mathrm{pz1}_{s,j})\,\mathcal{B}(r_{ij};\phi^{nkl}_j)\Big]
$$

Ratings are discrete (0–100 slider), so each $\mathcal{B}$ is an **interval** mass
$F(r+\tfrac{1}{200})-F(r-\tfrac{1}{200})$, not a density — bounded, so no spike blow-up.

$$
\underbrace{\log P(\mathbf{r}\mid\mathrm{pz1},\phi)}_{\texttt{beta\_mixture\_log\_likelihood}},
\qquad
\underbrace{\hat\phi = \arg\max_\phi \log P(\mathbf{r}\mid\mathrm{pz1},\phi)}_{\texttt{fit\_beta\_mixtures\_all\_features}}
$$

## Where $\hat\phi$ is fit — current vs. proposed

The weight $\mathrm{pz1}_s$ varies per draw either way (that *is* the $\vec{y}'$
dependence). Only the shapes $\hat\phi$ move.

**Current** — refit per draw, so $\ell\ell_s = \log P(\mathbf{r}\mid\mathrm{pz1}_s,\hat\phi_s)$.
This is $\mathbb{E}_s[\max_\phi]$: each draw re-optimizes $\phi$ to itself, extreme-pz1
draws spike, inflating $\ell\ell_s$ where pz1 is most variable (small $\ell$).

**Proposed** — fit once on the mean weight, then reuse:

$$
\bar{\mathrm{pz1}} = \tfrac1S\textstyle\sum_s \mathrm{pz1}_s,
\qquad
\hat\phi = \arg\max_\phi \log P(\mathbf{r}\mid\bar{\mathrm{pz1}},\phi),
\qquad
\ell\ell_s = \log P(\mathbf{r}\mid\mathrm{pz1}_s,\hat\phi)
$$

i.e. $\max_\phi\mathbb{E}_s$. Use `mean(sigmoid(y))`, not `sigmoid(mean(y))`.
