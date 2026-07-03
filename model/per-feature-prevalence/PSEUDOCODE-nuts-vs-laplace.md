# full inference pseudocode

there are multiple ways to compute the "per-evaluation objective function" (`log_joint(theta)`) that VBMC optimizes `theta` over. For now,
`theta = [log_ls, mu_0]` (eventually will extend to `sigma` and perhaps `beta`). 

we're trying to calculate the marginal likelihood of the ratings,
`p(ratings | theta)`, summed over conditions. this can be estimated in different ways.

- **Version 1 (NUTS + logsumexp):** sample coherences from the GP+RSA *prior*, reweight
  by the ratings. This is importance sampling from the prior. 
  - after testing with parameter recovery, we see that this sampling procedure leads  to too high of variance, so VBMC fails to converge.
  - why too high variance? basically our proposal distribution for $y$ pseudocoherence values doesn't overlap with the true underlying distribution.
- **Version 2 (Laplace):** condition the coherence field on the ratings and approximate that posterior with a Gaussian at its mode. Deterministic; VBMC converges.

---

## Version 1 (NUTS + logsumexp importance sampling)

Transcribed by Claude from my handwritten pseudocode w/ some comments added after.   
`theta = [exp(log_ls), mu_0]`; `output_scale`, `beta`
fixed at their true values `m`.

```
for vbmc_i in VBMC_MAX_EVALS:
    theta_i = vbmc.next_proposal()
    ls_i, mu_i = exp(theta[0]), theta[1]
    params = {ls_i, mu_i, sigma_i, beta_i}        # sigma, beta fixed at m

    log_prior = log N(log_ls | ...) + log N(mu_0 | ...)

    # ---- 1. per-condition coherence draws (from the PRIOR) ----
    for cond_i in CONDITIONS:
        # No U-Turn Sampling params: num_samples=800, warmup=300, thin=5
        y_train_ci, y_test_ci = NUTS_sample(
            log_density = log_GP_prior(y_train_ci, y_test_ci | x_train_ci, x_test, theta_i)
                        + log_RSA_speaker(u_train_ci | y_train_ci, beta_i)
        )                                          
        #  note: because ratings don't appear in this density, draws come from the prior

        pz1_draws[cond_i] = sigmoid(y_test_ci)  # size: (n_samples, n_features)
        pz1_bar[cond_i]   = mean over draws     # size: (n_features,) estimated coherence per f

    # ---- 2. fit the shared nuisance params once per theta ----
    #  (no-linking: a single scalar kappa acting like coherence would;  linking: shared Beta-mixture shapes)
    pz1_weights = stack([ tile(pz1_bar[cond_i], N_cond_i) for cond_i in CONDITIONS ])
    psi_i = argmax_{shapes}  sum_n sum_f  log P_interval( r_nf | pz1_weights_nf, shapes )

    # ---- 3. score each draw against the fitted nuisance; logsumexp over draws ----
    L = 0
    for j in CONDITIONS:
        for k in NUM_THINNED_MCMC_DRAWS:
            ll[k] = sum_n sum_f  log P_interval( r_nf | pz1_draws[j][k, f], psi_i )
        l_j          = logsumexp(ll) - log(NUM_THINNED_MCMC_DRAWS)     # marginal-lik estimate
        L           += l_j
        noise_var[j] = var(ll)                                          # per-condition MC variance

    log_joint = L + log_prior
    noise_std = sqrt( sum(noise_var) )
    vbmc.tell( theta_i, log_joint, noise_std )

return vbmc.posterior_over(theta)
```



---

## Version 2 — Laplace approximation

Instead of drawing from the prior and doing importance sampling, we could incorporate the information from the participant ratings 
The fix is one conceptual move with three code consequences: **put the ratings term inside the
sampled density so we work with the POSTERIOR over `y`, then approximate that posterior's normalizing
constant (= the marginal likelihood) with a Laplace approximation instead of importance sampling.**



### Version 2 pseudocode

```
for vbmc_i in VBMC_MAX_EVALS:
    theta_i = vbmc.next_proposal()
    ls_i, mu_i = exp(theta[0]), theta[1]
    params = {ls_i, mu_i, sigma_i, beta_i}        # sigma, beta fixed at m

    log_prior = log N(log_ls | ...) + log N(mu_0 | ...)

    L = 0
    for cond_i in CONDITIONS:

        # ---- define the RATINGS-CONDITIONED log-posterior over y = [y_test, y_train] ----
        #  identical to Version 1's density PLUS the ratings term (the one line that was missing)
        def g(y):
            return log_GP_prior(y | x_train_ci, x_test, theta_i)
                 + log_RSA_speaker(u_train_ci | y_train, beta_i)
                 + sum_n sum_f log P_ratings( r_nf | sigmoid(y_test)_f, psi_TRUE )   # <-- NEW
            #  no-linking:  P_ratings = Beta( r | kappa*pz1, kappa*(1-pz1) )
            #  linking:     P_ratings = pz1*Beta(r|kl) + (1-pz1)*Beta(r|nkl)   (log via logaddexp)
            #  psi_TRUE = the nuisance FIXED at its true value (not fit per eval)

        # ---- Laplace: mode-find, then closed-form marginal likelihood ----
        y_hat = argmax_y g(y)              # damped Newton (Armijo line search) on -g; D = J + n_train
        H     = -grad^2 g(y_hat)           # posterior precision (Hessian of -g at the mode)
        log_Z = g(y_hat) + (D/2) log(2*pi) - (1/2) log det H     # Laplace marginal likelihood

        L += log_Z

    log_joint = L + log_prior
    vbmc.tell( theta_i, log_joint, TARGET_NOISE )   # objective deterministic; TARGET_NOISE tiny & nominal

return vbmc.posterior_over(theta)
```

### The three concrete edits vs Version 1

1. **Add the ratings term to the sampled density.** The single line
   `+ sum log P_ratings(r | sigmoid(y_test), psi)` moves `y` from the prior to the ratings-conditioned
   posterior. Everything else follows from this.

2. **Replace NUTS-draws + `logsumexp` with mode + Hessian.** Because `y` is now posterior-distributed,
   importance-sampling reweighting is invalid *and* unnecessary — we approximate the posterior as
   Gaussian at its mode and read off the marginal likelihood in closed form. No draws, no `logsumexp`,
   no MC variance. (Uses the Beta *density* `jax.scipy.stats.beta.logpdf`, not the binned `betainc`
   CDF, because Laplace needs the likelihood differentiable in `y`.)

3. **Fix the nuisance; drop the per-eval fit and the noise bookkeeping.** `kappa` (no-linking) /
   Beta-mixture shapes (linking) can no longer be fit from `pz1_bar`: they now live *inside* the
   density we differentiate, so they'd be circular. For a recovery test we know them, so we FIX them at
   their true values — isolating `(length_scale, mu_0)`. The `pz1_weights` stacking, the
   `argmax_{shapes}` step, and `noise_var[j] = var(ll)` all disappear.

### Why this converges

`log_Z(theta)` is now a **smooth, deterministic** function of `theta` (no Monte Carlo), so VBMC's
surrogate GP fits it cleanly. On synthetic data the objective peaks at the truth when the true
length scale is in the identifiable regime (e.g. `ls=0.4-0.5`, below the ~0.78 feature-cloud
diameter); at `ls=2.1` the `ls` direction is a genuine plateau (spatial saturation), which is a
property of the design, not of the estimator.

> Caveat: Laplace assumes the coherence posterior is ~Gaussian in `y` (the logit scale). With 100
> participants/feature it is sharply concentrated and near-Gaussian, and the recovery check validates
> that. If the posterior over `y` were badly skewed, the closed-form `log_Z` would carry a
> theta-dependent bias — in that case escalate to bridge sampling or thermodynamic integration.
