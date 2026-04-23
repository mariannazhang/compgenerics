"""
gp_rsa.py — GP-RSA Integration Model

RSA is run independently per training feature to produce soft labels,
which are then used as GP training targets to predict coherence at test features.

    z_i = P(kind-linked | u_i, x_i)   [via per-feature RSA]
    P(z' | x', X_train, z_soft) = BGP(z' | x', X_train, z_soft)  [via GP]

Flow:
  1. For each training feature i: run RSA with (u_i, inst_i) -> soft label z_i
  2. Stack soft labels z_soft = [z_1, ..., z_N]
  3. Feed (X_train, z_soft) into GP to predict at test features x_test
"""

import jax.numpy as jnp
from jax.scipy.special import gammaln

from rsa import (
    Utterance, Instance,
    encode, decode,
    DEFAULT_ALPHA, DEFAULT_BETA,
    literal_listener, pragmatic_listener,
)
from gp import (
    laplace_predict,
    laplace_log_marginal_likelihood,
    laplace_log_marginal_likelihood_vec,
    bernoulli_gp_classify,
    get_vectorized_kernel,
    create_grid_hparams_2d_ard,
    RBF_2D_ARD_PARAM_NAMES,
)


#####################
# PER-FEATURE SOFT LABELS
#####################

def rsa_soft_labels(
    data: list,
    vocab: list,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    listener: str = "L0",
    lesioned: bool = False,
) -> jnp.ndarray:
    """
    Run RSA independently per training feature to get soft kind-linked labels.

    For each (utterance, instance) pair, runs RSA with V=1 (just that feature)
    and returns P(z_i = 1 | u_i, x_i) as a scalar.

    Args:
        data:     list of (Utterance, Instance) pairs, one per training feature
                  Instance.features must be a binary jnp array of shape (1,)
                  (i.e. scoped to just that feature)
        vocab:    list of length 1 per call — the single feature name
                  OR a list of V feature names if instances are full V-dim vectors
                  (in that case, marginal is extracted for the uttered feature)
        alpha:    Beta prior alpha (pseudocount for kind-linked)
        beta:     Beta prior beta  (pseudocount for not-kind-linked)
        listener: "L0" for literal listener, "L1" for pragmatic listener
        lesioned: if True, generic utterances treated as specific

    Returns:
        z_soft: (N,) soft labels in [0, 1], one per training feature
    """
    listener_fn = literal_listener if listener == "L0" else pragmatic_listener

    z_soft = []
    for utt, inst in data:
        # Run RSA with just this one (utterance, instance) pair
        # vocab here is the full vocab; inst.features is full V-dim
        # We extract the marginal P(z_i = 1) for the uttered feature
        V = inst.features.shape[0]
        post = listener_fn([(utt, inst)], list(range(V)), alpha, beta, lesioned)

        weights = post["weights"]      # (2^V,)
        masks   = post["hyp_masks"]    # (2^V, V)

        # Marginal P(z_{feat_idx} = 1)
        feat_idx = utt.feature_idx
        p_kind = jnp.sum(weights * masks[:, feat_idx])
        z_soft.append(p_kind)

    return jnp.array(z_soft)  # (N,)


#####################
# TEST COHERENCE PREDICTION (GP v2)
#####################

def predict_test_coherence(
    x_test: jnp.ndarray,
    X_train: jnp.ndarray,
    data_train: list,
    vocab: list,
    kernel_params,
    m: float = 0.0,
    kernel_name: str = "rbf_2d_ard",
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    listener: str = "L0",
    lesioned: bool = False,
):
    """
    Predict coherence P(kind-linked) at test features using GP v2.

    RSA is run per training feature to get soft labels, which are then used
    as GP training targets to predict at test feature embeddings.

    Args:
        x_test:        (M, D) or (D,) — test feature embedding(s)
        X_train:       (N, D) — training feature embeddings
        data_train:    list of N (Utterance, Instance) pairs, one per training feature
        vocab:         list of V feature name strings
        kernel_params: GP kernel hyperparameters
        m:             GP prior mean
        kernel_name:   GP kernel to use
        alpha:         RSA Beta prior alpha
        beta:          RSA Beta prior beta
        listener:      "L0" or "L1"
        lesioned:      if True, generic utterances treated as specific

    Returns:
        theta_test: (M,) — P(kind-linked) at test features
        f_mean:     (M,) — GP latent mean at test features
        f_var:      (M,) — GP latent variance at test features
    """
    x_test = jnp.atleast_2d(x_test)

    z_soft = rsa_soft_labels(data_train, vocab, alpha, beta, listener, lesioned)  # (N,)

    return laplace_predict(
        X_train, z_soft, x_test, kernel_params,
        kernel_name=kernel_name, prior_mean=m,
    )


#####################
# LOG-LIKELIHOODS FOR FITTING
#####################

def prevalence_log_likelihood(
    x_tests: jnp.ndarray,
    p_tests: jnp.ndarray,
    X_train: jnp.ndarray,
    data_train: list,
    vocab: list,
    kernel_params,
    m: float = 0.0,
    kernel_name: str = "rbf_2d_ard",
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    listener: str = "L0",
    lesioned: bool = False,
    beta_concentration: float = 5.0,
) -> jnp.ndarray:
    """
    Log P(prevalence judgments | GP params) using a Beta linking function.

    Models participant prevalence judgments p_tests as Beta-distributed around
    GP-predicted coherence theta_tests:

        p_test ~ Beta(theta * c, (1-theta) * c)

    Args:
        x_tests:            (M, D) — test feature embeddings
        p_tests:            (M,)   — observed participant prevalence judgments in [0, 1]
        X_train:            (N, D) — training feature embeddings
        data_train:         list of N (Utterance, Instance) pairs
        vocab:              list of V feature name strings
        kernel_params:      GP kernel hyperparameters
        m:                  GP prior mean
        kernel_name:        GP kernel to use
        alpha:              RSA Beta prior alpha
        beta:               RSA Beta prior beta
        listener:           "L0" or "L1"
        lesioned:           if True, generic utterances treated as specific
        beta_concentration: sharpness of Beta linking function

    Returns:
        scalar log-likelihood
    """
    theta_tests, _, _ = predict_test_coherence(
        x_tests, X_train, data_train, vocab,
        kernel_params, m, kernel_name, alpha, beta, listener, lesioned,
    )
    theta_tests = jnp.clip(theta_tests, 1e-6, 1 - 1e-6)

    alpha_beta = theta_tests * beta_concentration
    beta_beta  = (1 - theta_tests) * beta_concentration
    p_tests    = jnp.clip(p_tests, 1e-6, 1 - 1e-6)

    log_beta_fn = lambda a, b: gammaln(a) + gammaln(b) - gammaln(a + b)
    log_liks = (
        (alpha_beta - 1) * jnp.log(p_tests)
        + (beta_beta - 1) * jnp.log(1 - p_tests)
        - log_beta_fn(alpha_beta, beta_beta)
    )
    return jnp.sum(log_liks)


#####################
# JOINT FITTING
#####################

def fit_gp_rsa(
    data_train: list,
    vocab: list,
    X_train: jnp.ndarray,
    x_tests: jnp.ndarray,
    p_tests: jnp.ndarray,
    kernel_param_grid=None,
    m_grid=None,
    kernel_name: str = "rbf_2d_ard",
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    listener: str = "L0",
    lesioned: bool = False,
    beta_concentration: float = 5.0,
) -> dict:
    """
    Fit GP hyperparameters (kernel_params, m) from participant prevalence judgments,
    using RSA-derived soft labels as GP training targets.

    Uses grid search over kernel_params x m.

    Args:
        data_train:          list of N (Utterance, Instance) pairs — one per training feature
        vocab:               list of V feature name strings
        X_train:             (N, D) — training feature embeddings
        x_tests:             (M, D) — test feature embeddings
        p_tests:             (M,)   — observed prevalence judgments at test features
        kernel_param_grid:   (K, n_params) array of kernel params; if None uses default 2D ARD grid
        m_grid:              1D array of m values; if None uses [-4, -3, -2, -1, 0]
        kernel_name:         GP kernel to use
        alpha:               RSA Beta prior alpha
        beta:                RSA Beta prior beta
        listener:            "L0" or "L1"
        lesioned:            if True, use lesioned meaning function
        beta_concentration:  Beta linking function concentration

    Returns dict:
        "best_kernel_params":  best kernel hyperparameters
        "best_m":              best GP prior mean
        "best_ll":             best log-likelihood value
        "log_likelihoods":     (K * P,) array of log-likelihoods
        "kernel_param_grid":   the grid of kernel params evaluated
        "m_grid":              the grid of m values evaluated
    """
    if kernel_param_grid is None:
        kernel_param_grid = create_grid_hparams_2d_ard()
    if m_grid is None:
        m_grid = jnp.array([-4.0, -3.0, -2.0, -1.0, 0.0])

    # Compute soft labels once — they don't depend on kernel params or m
    z_soft = rsa_soft_labels(data_train, vocab, alpha, beta, listener, lesioned)

    best_ll = -jnp.inf
    best_kernel_params = kernel_param_grid[0]
    best_m = m_grid[0]
    log_likelihoods = []

    for kp in kernel_param_grid:
        for m in m_grid:
            theta_tests, _, _ = laplace_predict(
                X_train, z_soft, x_tests, kp,
                kernel_name=kernel_name, prior_mean=float(m),
            )
            theta_tests = jnp.clip(theta_tests, 1e-6, 1 - 1e-6)

            alpha_beta = theta_tests * beta_concentration
            beta_beta  = (1 - theta_tests) * beta_concentration
            p_clipped  = jnp.clip(p_tests, 1e-6, 1 - 1e-6)

            log_beta_fn = lambda a, b: gammaln(a) + gammaln(b) - gammaln(a + b)
            ll = jnp.sum(
                (alpha_beta - 1) * jnp.log(p_clipped)
                + (beta_beta - 1) * jnp.log(1 - p_clipped)
                - log_beta_fn(alpha_beta, beta_beta)
            )
            log_likelihoods.append(float(ll))

            if ll > best_ll:
                best_ll = ll
                best_kernel_params = kp
                best_m = m

    return {
        "best_kernel_params": best_kernel_params,
        "best_m":             best_m,
        "best_ll":            best_ll,
        "log_likelihoods":    jnp.array(log_likelihoods),
        "kernel_param_grid":  kernel_param_grid,
        "m_grid":             m_grid,
    }
