"""
gp_rsa.py — GP-RSA Integration Model

Given a set of utterances and labeled instances, the model infers the coherence and resulting prevalence of each feature.

Merges the Bernoulli GP (gp.py) with the RSA model (rsa.py):
instead of a flat Beta-Binomial prior over coherence, uses a GP over
continuous feature embedding space to define per-feature coherence:

    theta_i = sigmoid(f(x_i))   where f ~ GP(m, k)

This ties together:
  - RSA training: (x_i, u_i) — feature embeddings + teacher utterances
  - Test: (x', p) — new feature embedding + prevalence judgment

m = prior mean of the GP (shared across both RSA and prevalence likelihood).
"""

import jax
import jax.numpy as jnp
from jax import vmap
from jax.nn import softmax, log_softmax
from itertools import product as iterproduct
from collections import namedtuple

from rsa import (
    Utterance, Instance, Kind,
    encode, decode,
    meaning, jaccard_similarity,
    DEFAULT_ALPHA, DEFAULT_BETA,
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
# FEATURE SET HYPOTHESIS ENUMERATION
#####################

def gp_enumerate_feature_set_hyps(
    X_vocab: jnp.ndarray, # embedding coords for all vocab features
    X_train: jnp.ndarray, # embedding coords of features with known kind-linked labels
    y_train: jnp.ndarray, # binary kind-linked labels for training features
    kernel_params, # GP kernel hyperparameters
    m: float = 0, # GP prior mean ; used to default to -2.0
    kernel_name: str = "rbf_2d_ard",
):
    """
    Enumerate all 2^V kind-feature masks with GP-derived per-feature coherence prior.

    Replaces rsa.enumerate_feature_set_hyps(V, alpha, beta): instead of a flat Beta-Binomial,
    the prior over masks is defined by per-feature coherence from the GP:

        theta_i = P(kind-linked | x_i, training data)   [via GP]
        log P(mask | f) = sum_i [ mask_i * log(theta_i) + (1-mask_i) * log(1-theta_i) ]

    Args:
        X_vocab:       (V, D)    — embedding coords for all vocab features
        X_train:       (N, D)    — embedding coords of features with known kind-linked labels
        y_train:       (N,)      — binary kind-linked labels for training features
        kernel_params:           — GP kernel hyperparameters
        m:    float     — GP prior mean 
        kernel_name:   str       — kernel to use (default "rbf_2d_ard")

    Returns:
        hyp_masks:       (2^V, V)  — binary feature mask per hypothesis
        prior_log_weights:  (2^V,)    — log P(mask | GP)
    """
    V = X_vocab.shape[0]
    # Get per-feature coherence probabilities from the GP
    # laplace_predict returns (probs, f_mean, f_var); probs shape (V,)
    theta, _, _ = laplace_predict(X_train, y_train, X_vocab, kernel_params,
                                   kernel_name=kernel_name, prior_mean=m)
    theta = jnp.clip(theta, 1e-6, 1 - 1e-6)  # numerical stability

    # Enumerate all 2^V masks
    hyp_masks = jnp.array(list(iterproduct([0.0, 1.0], repeat=V)))  # (2^V, V)

    # log P(mask | theta) = sum_i mask_i*log(theta_i) + (1-mask_i)*log(1-theta_i)
    log_theta     = jnp.log(theta)        # (V,)
    log_one_minus = jnp.log(1 - theta)   # (V,)

    def mask_log_prior(mask):
        return jnp.dot(mask, log_theta) + jnp.dot(1 - mask, log_one_minus)

    prior_log_weights = vmap(mask_log_prior)(hyp_masks)  # (2^V,)

    return hyp_masks, prior_log_weights


#####################
# LITERAL LISTENER (L0)
#####################

def gp_literal_listener(
    data: list,
    vocab: list,
    X_vocab: jnp.ndarray,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    kernel_params,
    m: float = -2.0,
    kernel_name: str = "rbf_2d_ard",
    lesioned: bool = False,
) -> dict:
    """
    GP-RSA Literal Listener (L0): infers kind-linked features from (utterance, instance) data.

    Identical to rsa.literal_listener but uses gp_enumerate_feature_set_hyps for the prior
    instead of the flat Beta-Binomial prior.

    Args:
        data:          list of (Utterance, Instance) pairs
                       Instance.features must be a binary jnp array of shape (V,)
        vocab:         list of V feature name strings
        X_vocab:       (V, D) — embedding coords for vocab features (same order as vocab)
        X_train:       (N, D) — embedding coords of GP training features
        y_train:       (N,)   — binary kind-linked labels for GP training features
        kernel_params:         — GP kernel hyperparameters
        m:    float   — GP prior mean
        kernel_name:   str     — GP kernel to use
        lesioned:      bool    — if True, generic utterances are treated as specific

    Returns dict:
        "hyp_masks":  (2^V, V)  kind-feature mask per hypothesis
        "log_weights":   (2^V,)    unnormalized log posterior
        "weights":       (2^V,)    normalized posterior (sums to 1)
        "theta":         (V,)      GP-predicted per-feature coherence probabilities
    """
    V = len(vocab)
    hyp_masks, prior_log_weights = gp_enumerate_feature_set_hyps(
        X_vocab, X_train, y_train, kernel_params, m, kernel_name
    )

    log_lik = jnp.zeros(hyp_masks.shape[0])

    for utt, inst in data:
        inst_features = inst.features  # (V,)

        def state_loglik(kind_features):
            m = meaning(kind_features, inst_features,
                        utt.subj, utt.feature_idx, lesioned)
            likelihood = jnp.where(m > 0.5, 0.95, 0.05)
            return jnp.log(likelihood)

        log_lik = log_lik + vmap(state_loglik)(hyp_masks)  # (2^V,)

    log_weights = prior_log_weights + log_lik
    weights = softmax(log_weights)

    # Also return theta for inspection
    theta, _, _ = laplace_predict(X_train, y_train, X_vocab, kernel_params,
                                   kernel_name=kernel_name, prior_mean=m)

    return {
        "hyp_masks": hyp_masks,
        "log_weights":  log_weights,
        "weights":      weights,
        "theta":        theta,
    }


#####################
# SPEAKER (S1)
#####################

def _expected_jaccard_gp_l0(
    kind_features: jnp.ndarray,
    inst_features: jnp.ndarray,
    utt_subj: int,
    utt_feat_idx: int,
    vocab: list,
    X_vocab: jnp.ndarray,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    kernel_params,
    m: float = -2.0,
    kernel_name: str = "rbf_2d_ard",
    lesioned: bool = False,
) -> jnp.ndarray:
    """
    E[Jaccard(true_kind & inst, inferred_kind & inst)] under GP-L0 posterior.
    """
    utt  = Utterance(subj=utt_subj, feature_idx=utt_feat_idx)
    inst = Instance(kind="", features=inst_features)

    posterior = gp_literal_listener(
        [(utt, inst)], vocab, X_vocab, X_train, y_train,
        kernel_params, m, kernel_name, lesioned
    )
    weights = posterior["weights"]       # (2^V,)
    masks   = posterior["hyp_masks"]  # (2^V, V)

    true_restricted = kind_features * inst_features  # (V,)

    def jaccard_for_state(inferred_mask):
        inferred_restricted = inferred_mask * inst_features
        return jaccard_similarity(true_restricted, inferred_restricted)

    jaccards = vmap(jaccard_for_state)(masks)
    return jnp.dot(weights, jaccards)


def gp_speaker(
    kind_features: jnp.ndarray,
    observed_instance: Instance,
    vocab: list,
    X_vocab: jnp.ndarray,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    kernel_params,
    m: float = -2.0,
    kernel_name: str = "rbf_2d_ard",
    inv_temp: float = 20.0,
    lesioned: bool = False,
) -> jnp.ndarray:
    """
    GP-RSA Speaker (S1): chooses utterance to maximize GP-L0's ability to infer kind features.

    Returns utterance probability distribution of shape (2*V,).
    Utterance indices: 0..(V-1) = subj=0 ("This Zarpie"), V..(2V-1) = subj=1 ("Zarpies").
    """
    V = len(vocab)
    inst_features = observed_instance.features  # (V,)

    utilities = []
    for subj in [0, 1]:
        for feat_idx in range(V):
            u = _expected_jaccard_gp_l0(
                kind_features, inst_features,
                subj, feat_idx,
                vocab, X_vocab, X_train, y_train,
                kernel_params, m, kernel_name, lesioned
            )
            utilities.append(u)

    utilities = jnp.stack(utilities)   # (2V,)
    return softmax(inv_temp * utilities)


#####################
# PRAGMATIC LISTENER (L1)
#####################

def gp_pragmatic_listener(
    data: list,
    vocab: list,
    X_vocab: jnp.ndarray,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    kernel_params,
    m: float = -2.0,
    kernel_name: str = "rbf_2d_ard",
    inv_temp: float = 20.0,
    lesioned: bool = False,
) -> dict:
    """
    GP-RSA Pragmatic Listener (L1): like GP-L0 but conditions on GP-speaker likelihood.

    Args: same as gp_literal_listener, plus inv_temp for the speaker.

    Returns same dict format as gp_literal_listener.
    """
    V = len(vocab)
    hyp_masks, prior_log_weights = gp_enumerate_feature_set_hyps(
        X_vocab, X_train, y_train, kernel_params, m, kernel_name
    )
    N_hyps = hyp_masks.shape[0]

    log_lik = jnp.zeros(N_hyps)

    for utt, inst in data:
        inst_features = inst.features  # (V,)
        utt_idx = utt.subj * V + utt.feature_idx

        def state_log_speaker_lik(kind_features):
            utt_probs = gp_speaker(
                kind_features, inst, vocab,
                X_vocab, X_train, y_train,
                kernel_params, m, kernel_name, inv_temp, lesioned
            )
            return jnp.log(jnp.clip(utt_probs[utt_idx], 1e-9, 1.0))

        log_lik = log_lik + vmap(state_log_speaker_lik)(hyp_masks)

    log_weights = prior_log_weights + log_lik
    weights = softmax(log_weights)

    theta, _, _ = laplace_predict(X_train, y_train, X_vocab, kernel_params,
                                   kernel_name=kernel_name, prior_mean=m)

    return {
        "hyp_masks": hyp_masks,
        "log_weights":  log_weights,
        "weights":      weights,
        "theta":        theta,
    }


#####################
# TEST PREVALENCE PREDICTION
#####################

def predict_test_coherence(
    x_test: jnp.ndarray,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    kernel_params,
    m: float = 0, # used to default to -2.0
    kernel_name: str = "rbf_2d_ard",
):
    """
    Predict coherence (P(kind-linked)) at a new test feature embedding.

    This is the GP's generalization from training features to test features.
    The GP is conditioned on (X_train, y_train) and predicts at x_test.

    Args:
        x_test:        (D,) or (M, D) — test feature embedding(s)
        X_train:       (N, D) — training feature embeddings
        y_train:       (N,)   — binary kind-linked labels for training features
        kernel_params:         — GP kernel hyperparameters
        m:    float   — GP prior mean
        kernel_name:   str     — GP kernel to use

    Returns:
        theta_test: (M,) — P(kind-linked) at test feature(s)
        f_mean:     (M,) — GP latent mean at test feature(s)
        f_var:      (M,) — GP latent variance at test feature(s)
    """
    x_test = jnp.atleast_2d(x_test)  # ensure (M, D)
    return laplace_predict(X_train, y_train, x_test, kernel_params,
                           kernel_name=kernel_name, prior_mean=m)


#####################
# LOG-LIKELIHOODS FOR FITTING
#####################

def rsa_log_likelihood(
    data: list,
    vocab: list,
    X_vocab: jnp.ndarray,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    kernel_params,
    m: float = -2.0,
    kernel_name: str = "rbf_2d_ard",
    inv_temp: float = 20.0,
    lesioned: bool = False,
) -> jnp.ndarray:
    """
    Log P(utterances | GP params) under the GP-RSA pragmatic listener.

    For each (utterance, instance) pair in data, computes the log probability
    that a GP-speaker would produce that utterance, then sums across pairs.

    This is one term in the joint log-likelihood for fitting.

    Args:
        data:  list of (Utterance, Instance) pairs — RSA training observations
        (other args same as gp_pragmatic_listener)

    Returns:
        scalar log-likelihood
    """
    V = len(vocab)
    total_log_lik = 0.0

    for utt, inst in data:
        # Run speaker model to get P(utterance | features) for each hypothesized mask
        # Then marginalize over masks using GP prior
        hyp_masks, prior_log_weights = gp_enumerate_feature_set_hyps(
            X_vocab, X_train, y_train, kernel_params, m, kernel_name
        )
        inst_features = inst.features
        utt_idx = utt.subj * V + utt.feature_idx

        def state_speaker_log_lik(kind_features):
            utt_probs = gp_speaker(
                kind_features, inst, vocab,
                X_vocab, X_train, y_train,
                kernel_params, m, kernel_name, inv_temp, lesioned
            )
            return jnp.log(jnp.clip(utt_probs[utt_idx], 1e-9, 1.0))

        log_speaker_liks = vmap(state_speaker_log_lik)(hyp_masks)  # (2^V,)

        # log P(utt) = log sum_mask P(utt | mask) * P(mask)
        #            = logsumexp(log P(utt | mask) + log P(mask))
        log_weights_norm = jax.nn.log_softmax(prior_log_weights)
        total_log_lik += jax.scipy.special.logsumexp(log_speaker_liks + log_weights_norm)

    return total_log_lik


def prevalence_log_likelihood(
    x_tests: jnp.ndarray,
    p_tests: jnp.ndarray,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    kernel_params,
    m: float = -2.0,
    kernel_name: str = "rbf_2d_ard",
    beta_concentration: float = 5.0,
) -> jnp.ndarray:
    """
    Log P(prevalence judgments | GP params) using a Beta linking function.

    Models participant prevalence judgments p_tests as Beta-distributed around
    GP-predicted coherence theta_tests:

        p_test ~ Beta(theta * c, (1-theta) * c)

    where c = beta_concentration controls sharpness of the linking function.

    Args:
        x_tests:            (M, D) — test feature embeddings
        p_tests:            (M,)   — observed participant prevalence judgments in [0, 1]
        X_train:            (N, D) — GP training feature embeddings
        y_train:            (N,)   — binary kind-linked labels for GP training features
        kernel_params:               — GP kernel hyperparameters
        m:         float   — GP prior mean
        kernel_name:        str     — GP kernel to use
        beta_concentration: float   — sharpness of Beta linking function (higher = more peaked)

    Returns:
        scalar log-likelihood
    """
    theta_tests, _, _ = laplace_predict(
        X_train, y_train, x_tests, kernel_params,
        kernel_name=kernel_name, prior_mean=m
    )
    theta_tests = jnp.clip(theta_tests, 1e-6, 1 - 1e-6)

    # Beta distribution: p ~ Beta(alpha, beta) where alpha = theta*c, beta = (1-theta)*c
    alpha_beta = theta_tests * beta_concentration          # (M,)
    beta_beta  = (1 - theta_tests) * beta_concentration   # (M,)
    p_tests    = jnp.clip(p_tests, 1e-6, 1 - 1e-6)

    # log Beta(p | a, b) = (a-1)*log(p) + (b-1)*log(1-p) - log_beta(a, b)
    from jax.scipy.special import gammaln
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
    X_vocab: jnp.ndarray,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_tests: jnp.ndarray,
    p_tests: jnp.ndarray,
    kernel_param_grid=None,
    m_grid=None,
    kernel_name: str = "rbf_2d_ard",
    inv_temp: float = 20.0,
    beta_concentration: float = 5.0,
    lesioned: bool = False,
    rsa_weight: float = 1.0,
    prevalence_weight: float = 1.0,
) -> dict:
    """
    Fit GP hyperparameters (kernel_params, m) jointly from:
      - RSA training data: (utterance, instance) pairs with feature embeddings
      - Test prevalence judgments: feature embeddings + participant responses

    Uses grid search over kernel_params x m.

    Args:
        data_train:          list of (Utterance, Instance) — RSA training observations
        vocab:               list of V feature name strings
        X_vocab:             (V, D) — embedding coords for all vocab features
        X_train:             (N, D) — GP training feature embeddings
        y_train:             (N,)   — initial binary kind-linked labels
        x_tests:             (M, D) — test feature embeddings
        p_tests:             (M,)   — observed prevalence judgments at test features
        kernel_param_grid:   (K, n_params) array of kernel params to try; if None, uses default 2D ARD grid
        m_grid:              1D array of m values to try; if None, uses [-4, -3, -2, -1, 0]
        kernel_name:         GP kernel to use
        inv_temp:            RSA speaker inverse temperature
        beta_concentration:  Beta linking function concentration
        lesioned:            if True, use lesioned meaning function
        rsa_weight:          weight on RSA log-likelihood term
        prevalence_weight:   weight on prevalence log-likelihood term

    Returns dict:
        "best_kernel_params":  best kernel hyperparameters
        "best_m":              best GP prior mean
        "log_likelihoods":     (K * P,) array of joint log-likelihoods
        "kernel_param_grid":   the grid of kernel params evaluated
        "m_grid":              the grid of m values evaluated
    """
    if kernel_param_grid is None:
        kernel_param_grid = create_grid_hparams_2d_ard()  # (K, 3)
    if m_grid is None:
        m_grid = jnp.array([-4.0, -3.0, -2.0, -1.0, 0.0])

    best_ll = -jnp.inf
    best_kernel_params = kernel_param_grid[0]
    best_m = m_grid[0]
    log_likelihoods = []

    for kp in kernel_param_grid:
        for m in m_grid:
            ll_rsa = rsa_log_likelihood(
                data_train, vocab, X_vocab, X_train, y_train,
                kp, float(m), kernel_name, inv_temp, lesioned
            )
            ll_prev = prevalence_log_likelihood(
                x_tests, p_tests, X_train, y_train,
                kp, float(m), kernel_name, beta_concentration
            )
            ll = rsa_weight * ll_rsa + prevalence_weight * ll_prev
            log_likelihoods.append(float(ll))

            if ll > best_ll:
                best_ll = ll
                best_kernel_params = kp
                best_m = m

    return {
        "best_kernel_params": best_kernel_params,
        "best_m":             best_m,
        "log_likelihoods":    jnp.array(log_likelihoods),
        "kernel_param_grid":  kernel_param_grid,
        "m_grid":             m_grid,
    }
