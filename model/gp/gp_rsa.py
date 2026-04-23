"""
gp_rsa.py — GP-RSA Integration Model

The listener is uncertain about the true kind-feature set. RSA produces a joint
posterior over all 2^V possible feature-set masks after getting data (utterance, feature)
from the speaker. Each mask implies a different GP over the embedding space. The final 
prediction marginalizes over all masks:

    P(z' | x', X_train, u) = sum_z  BGP(z' | X_train, z, x') * P(z | u, X_train)

Flow:
  1. Run RSA (L0 or L1) on all training data jointly
     -> posterior weights over 2^V masks, shape (2^V,)
  2. vmap laplace_predict over all masks (or top-K by weight)
     -> theta per mask, shape (2^V, M) or (K, M)
  3. Marginalize: theta_test = weights @ thetas, shape (M,)
"""

import jax
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
# JOINT RSA POSTERIOR
#####################

def rsa_joint_posterior(
    data: list,
    vocab: list,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    listener: str = "L0",
    lesioned: bool = False,
) -> dict:
    """
    Run RSA on all training data jointly to get a posterior over feature-set masks.

    Args:
        data:     list of (Utterance, Instance) pairs, all training observations
        vocab:    list of V feature name strings
        alpha:    Beta prior alpha
        beta:     Beta prior beta
        listener: "L0" for literal listener, "L1" for pragmatic listener
        lesioned: if True, generic utterances treated as specific

    Returns dict:
        "hyp_masks":   (2^V, V) — all binary masks
        "weights":     (2^V,)   — normalized posterior P(z | u, x)
        "log_weights": (2^V,)   — unnormalized log posterior
    """
    listener_fn = literal_listener if listener == "L0" else pragmatic_listener
    return listener_fn(data, vocab, alpha, beta, lesioned)


#####################
# TEST COHERENCE PREDICTION (GP v2, joint)
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
    top_k: int = None,
):
    """
    Predict coherence P(kind-linked) at test features, marginalizing over
    the listener's uncertainty about which features are kind-linked.

    For each hypothesis mask z in the RSA posterior:
        BGP(z' | X_train, z, x_test)  [GP conditioned on z as training labels]
    Then marginalizes:
        theta_test = sum_z  P(z | u, x) * BGP(z' | X_train, z, x_test)

    Args:
        x_test:        (M, D) or (D,) — test feature embedding(s)
        X_train:       (N, D) — training feature embeddings
        data_train:    list of (Utterance, Instance) pairs
        vocab:         list of V feature name strings
        kernel_params: GP kernel hyperparameters
        m:             GP prior mean
        kernel_name:   GP kernel to use
        alpha:         RSA Beta prior alpha
        beta:          RSA Beta prior beta
        listener:      "L0" or "L1"
        lesioned:      if True, generic utterances treated as specific
        top_k:         if set, only use the top-K highest-weight masks;
                       if None, use all 2^V masks

    Returns:
        theta_test: (M,) — P(kind-linked) at test features
        f_mean:     (M,) — GP latent mean (weighted average across masks)
        f_var:      (M,) — GP latent variance (weighted average across masks)
    """
    x_test = jnp.atleast_2d(x_test)

    posterior = rsa_joint_posterior(data_train, vocab, alpha, beta, listener, lesioned)
    hyp_masks = posterior["hyp_masks"]  # (2^V, V)
    weights   = posterior["weights"]    # (2^V,)

    if top_k is not None:
        top_k_idx = jnp.argsort(weights)[-top_k:]
        hyp_masks = hyp_masks[top_k_idx]           # (K, V)
        weights   = weights[top_k_idx]
        weights   = weights / jnp.sum(weights)     # renormalize

    # vmap laplace_predict over masks — each mask is a different set of training labels
    def predict_single_mask(mask):
        theta, f_mean, f_var = laplace_predict(
            X_train, mask, x_test, kernel_params,
            kernel_name=kernel_name, prior_mean=m,
        )
        return theta, f_mean, f_var

    thetas, f_means, f_vars = jax.vmap(predict_single_mask)(hyp_masks)
    # thetas, f_means, f_vars each shape (2^V or K, M)

    theta_test = jnp.dot(weights, thetas)   # (M,)
    f_mean_out = jnp.dot(weights, f_means)  # (M,)
    f_var_out  = jnp.dot(weights, f_vars)   # (M,)

    return theta_test, f_mean_out, f_var_out


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
    top_k: int = None,
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
        data_train:         list of (Utterance, Instance) pairs
        vocab:              list of V feature name strings
        kernel_params:      GP kernel hyperparameters
        m:                  GP prior mean
        kernel_name:        GP kernel to use
        alpha:              RSA Beta prior alpha
        beta:               RSA Beta prior beta
        listener:           "L0" or "L1"
        lesioned:           if True, generic utterances treated as specific
        beta_concentration: sharpness of Beta linking function
        top_k:              if set, only use top-K highest-weight masks

    Returns:
        scalar log-likelihood
    """
    theta_tests, _, _ = predict_test_coherence(
        x_tests, X_train, data_train, vocab,
        kernel_params, m, kernel_name, alpha, beta, listener, lesioned, top_k,
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
    top_k: int = None,
) -> dict:
    """
    Fit GP hyperparameters (kernel_params, m) from participant prevalence judgments,
    marginalizing over the RSA joint posterior over feature-set masks.

    Uses grid search over kernel_params x m.

    Args:
        data_train:          list of (Utterance, Instance) pairs
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
        top_k:               if set, only use top-K highest-weight RSA masks

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

    # Compute RSA posterior once — it doesn't depend on kernel params or m
    posterior  = rsa_joint_posterior(data_train, vocab, alpha, beta, listener, lesioned)
    hyp_masks  = posterior["hyp_masks"]  # (2^V, V)
    weights    = posterior["weights"]    # (2^V,)

    if top_k is not None:
        top_k_idx = jnp.argsort(weights)[-top_k:]
        hyp_masks = hyp_masks[top_k_idx]
        weights   = weights[top_k_idx]
        weights   = weights / jnp.sum(weights)

    log_beta_fn = lambda a, b: gammaln(a) + gammaln(b) - gammaln(a + b)
    p_clipped   = jnp.clip(p_tests, 1e-6, 1 - 1e-6)

    best_ll = -jnp.inf
    best_kernel_params = kernel_param_grid[0]
    best_m = m_grid[0]
    log_likelihoods = []

    for kp in kernel_param_grid:
        for m in m_grid:
            def predict_single_mask(mask):
                theta, _, _ = laplace_predict(
                    X_train, mask, x_tests, kp,
                    kernel_name=kernel_name, prior_mean=float(m),
                )
                return theta

            thetas      = jax.vmap(predict_single_mask)(hyp_masks)  # (K or 2^V, M)
            theta_tests = jnp.dot(weights, thetas)                   # (M,)
            theta_tests = jnp.clip(theta_tests, 1e-6, 1 - 1e-6)

            alpha_beta = theta_tests * beta_concentration
            beta_beta  = (1 - theta_tests) * beta_concentration
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
