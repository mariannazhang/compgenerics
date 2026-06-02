import jax
import jax.numpy as jnp


# MEANING_MATRIX[utt_idx, z]
# is a statement about kindlinked feature true (0) or false (1)?
# depends on the utterance and the actual kindlinkedness of feature
MEANING_MATRIX = jnp.array([
    [0, 1],   # generic stmt:  false if z=0, true if z=1
    [1, 1],   # specific stmt: true for both z
], dtype=jnp.float32)

N_UTTERANCES = MEANING_MATRIX.shape[0]

def literal_listener(z_i: int, u_i: int, prior_z1: jnp.ndarray) -> jnp.ndarray:
    """
    L0(z_i | u_i): literal listener posterior over z_i given utterance u_i.
    z_i: 0 or 1
    u_i: 0=generic, 1=specific
    prior_z1: scalar P(z=1)
    """
    prior = jnp.array([1.0 - prior_z1, prior_z1])   # [P(z=0), P(z=1)]
    meanings = MEANING_MATRIX[u_i]                  # posterior over z depends on meaning
    unnorm = prior * meanings
    total = jnp.sum(unnorm)
    return jnp.where(total > 0, unnorm[z_i] / total, 0.0)


def speaker(u_i: int, z_i: int, beta: jnp.ndarray, prior_z1: jnp.ndarray,
            epsilon: float = 0.05) -> jnp.ndarray:
    """
    S1(u_i | z_i): epsilon-noisy speaker.
    With prob (1-epsilon): softmax over L0 utilities.
    With prob epsilon: uniform over utterances.
    u_i: 0=generic, 1=specific
    z_i: 0 or 1
    """
    utt_indices = jnp.arange(N_UTTERANCES)
    # posterior over z depends on literal listener posterior
    # TODO: is it worth it to vmap if it's just 2 elements?
    utilities = jax.vmap(lambda u: literal_listener(z_i, u, prior_z1))(utt_indices)
    p_rational = jax.nn.softmax(beta * utilities)    # (2,)
    p_uniform = jnp.ones(N_UTTERANCES) / N_UTTERANCES
    probs = (1 - epsilon) * p_rational + epsilon * p_uniform
    return probs[u_i]

# P_S(u|y)
# probabiliy of the speaker
# TODO: rename to make the speaker nature clear, not a generic p_u_given_y
def p_u_given_y(u_i: int, y_i: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """
    P(u_i | y_i) = S1(u_i | z=0) * P(z=0 | y_i) + S1(u_i | z=1) * P(z=1 | y_i)
    """
    p_z1 = jax.nn.sigmoid(y_i)
    p_z0 = 1.0 - p_z1
    return speaker(u_i, 0, beta, p_z1) * p_z0 + speaker(u_i, 1, beta, p_z1) * p_z1

# TODO: rename output_scale to sigma / gp param
def rbf_kernel(X: jnp.ndarray, length_scale: jnp.ndarray,
               output_scale: jnp.ndarray) -> jnp.ndarray:
    """
    X: shape (n, 2) — 2D feature embeddings
    Returns covariance matrix shape (n, n) with diag jitter for numerical stability.
    """
    diff = X[:, None, :] - X[None, :, :]             # (n, n, 2)
    sq_dist = jnp.sum(diff ** 2, axis=-1)             # (n, n)
    K = output_scale**2 * jnp.exp(-sq_dist / (2 * length_scale**2))
    return K + 1e-4 * jnp.eye(K.shape[0])


def _mvn_logpdf_chol(x: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """MVN log-pdf via Cholesky — numerically stable for near-singular covariances."""
    n = x.shape[0]
    L = jnp.linalg.cholesky(cov)                          # lower triangular
    diff = x - mean
    z = jnp.linalg.solve(L, diff)                         # L^{-1}(x - mu)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (n * jnp.log(2 * jnp.pi) + log_det + jnp.dot(z, z))


# def rbf_kernel(x1, x2, lengthscale, sigma):
#     """Radial basis function kernel."""
#     return sigma**2 * jnp.exp(-jnp.sum((x1 - x2)**2) / (2 * lengthscale**2))

# rbf_kernel_vec = jax.vmap(
#         jax.vmap(rbf_kernel, in_axes=(0, None, None, None)),
#         in_axes=(None, 0, None, None)
#     )

def log_likelihood(training: dict, test: dict, params: dict) -> jnp.ndarray:
    """
    Log joint density of training & testing: 
    log P(u_vec | y_vec) + log P(y_prime, y_vec | X).

    training: {
        'utt_types':  jnp.array shape (n,), int32, 0=generic 1=specific
        'features':   jnp.array shape (n, 2)
        'coherences': jnp.array shape (n,)   — latent pseudocoherences
    }
    test: {
        'feature':   jnp.array shape (2,)
        'coherence': jnp.array scalar        — latent pseudocoherence
    }
    params: {'mu_0', 'length_scale', 'output_scale', 'beta'}
    """
    y_vec    = training['coherences']
    u_vec    = training['utt_types']
    x_vec    = training['features']
    y_prime  = test['coherence']
    x_prime  = test['feature']

    mu_0         = params['mu_0']
    length_scale = params['length_scale']
    output_scale = params['output_scale']
    beta         = params['beta']

    X_all  = jnp.vstack([x_prime[None, :], x_vec])           # (n+1, 2)
    full_y = jnp.concatenate([y_prime[None], y_vec])          # (n+1,)
    mu     = jnp.full(full_y.shape[0], mu_0)
    sigma  = rbf_kernel(X_all, length_scale, output_scale)    # (n+1, n+1)

    log_gp_prior = _mvn_logpdf_chol(full_y, mu, sigma)

    def log_utt_fn(u_i, y_i):
        return jnp.log(p_u_given_y(u_i, y_i, beta) + 1e-300)

    log_utt_per_feat = jax.vmap(log_utt_fn)(u_vec, y_vec)    # (n,)
    log_utterance = jnp.sum(log_utt_per_feat)

    return log_utterance + log_gp_prior

# P(z|y)
def coherence_from_pseudocoherence(y: jnp.ndarray) -> jnp.ndarray:
    """
    P(z'=1) = sigmoid(y)
    """
    return jax.nn.sigmoid(y)

############################
##  Beta Mixture Fitting  ##
############################

# For each feature j, fits the prevalence of j based on participant judgments:
#   p'_j ~ P(z'=1)_j · Beta(α_kl, β_kl)  +  (1 - P(z'=1)_j) · Beta(α_nkl, β_nkl)
#
# Params optimised in log-space (log_a1, log_b1, log_a2, log_b2) to enforce positivity.
# Uses scipy.optimize.minimize with JAX-computed gradients (jax.scipy.optimize unavailable
# in JAX < 0.4.1).

from scipy.optimize import minimize as scipy_minimize

def _neg_ll_one_feature(log_params: jnp.ndarray, r: jnp.ndarray, th: jnp.ndarray) -> jnp.ndarray:
    """Negative log-likelihood for one feature's Beta mixture, in log-param space."""
    a1, b1, a2, b2 = jnp.exp(log_params)
    log_pdf1 = jax.scipy.stats.beta.logpdf(r, a1, b1)
    log_pdf2 = jax.scipy.stats.beta.logpdf(r, a2, b2)
    log_mix = jnp.logaddexp(jnp.log(th) + log_pdf1, jnp.log1p(-th) + log_pdf2)
    return -jnp.sum(log_mix)

_neg_ll_and_grad = jax.jit(jax.value_and_grad(_neg_ll_one_feature))

# (a1,b1,a2,b2) starting points in log-space; values chosen to span skewed/symmetric Beta shapes
_LOG_INITS = jnp.log(jnp.array([
    [5., 1., 1., 5.],
    [3., 1., 1., 3.],
    [8., 2., 2., 5.],
    [2., 2., 2., 2.],
]))


def _fit_one_feature(r: jnp.ndarray, th: jnp.ndarray) -> jnp.ndarray:
    """Best-of-inits BFGS fit for one feature. Returns log_params shape (4,)."""
    best_x, best_f = None, jnp.inf
    for log_x0 in _LOG_INITS:
        def f_and_g(lp):
            val, grad = _neg_ll_and_grad(jnp.array(lp), r, th)
            return float(val), jnp.array(grad)
        result = scipy_minimize(f_and_g, jnp.array(log_x0), jac=True, method='BFGS')
        if result.fun < best_f:
            best_f = result.fun
            best_x = result.x
    return jnp.array(best_x)


def fit_beta_mixtures_all_features(
    responses: jnp.ndarray,
    pz1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Fit a Beta mixture for each feature.

    N = num participants, J = num features
    responses: (N, J) prevalence ratings in [0, 1]
    pz1:       (N, J) P(z'=1) per participant per feature
    Returns:   (J, 4) array — [alpha_kl, beta_kl, alpha_nkl, beta_nkl],
               alpha_kl is always the higher-mean (kind-linked) component.
    """
    eps = 1e-6
    r  = jnp.clip(responses, eps, 1 - eps)   # (N, J)
    th = jnp.clip(pz1,       eps, 1 - eps)   # (N, J)
    J  = r.shape[1]

    log_params = jnp.stack([_fit_one_feature(r[:, j], th[:, j]) for j in range(J)])  # (J, 4)

    params = jnp.exp(log_params)
    a1, b1, a2, b2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

    mean1 = a1 / (a1 + b1)
    mean2 = a2 / (a2 + b2)
    swap = mean1 < mean2
    alpha_kl  = jnp.where(swap, a2, a1)
    beta_kl   = jnp.where(swap, b2, b1)
    alpha_nkl = jnp.where(swap, a1, a2)
    beta_nkl  = jnp.where(swap, b1, b2)

    return jnp.stack([alpha_kl, beta_kl, alpha_nkl, beta_nkl], axis=1)  # (J, 4)


def make_log_density_fn(
    training_utt_types: jnp.ndarray,
    training_features: jnp.ndarray,
    test_feature: jnp.ndarray,
    params: dict,
):
    """
    Returns a BlackJAX-compatible log_density function for a single test feature.

    The returned function takes a position PyTree:
        position = {
            'training_coherences': jnp.array shape (n,),
            'test_coherence':      jnp.array scalar shape ()
        }
    and returns a scalar log density.
    """
    training_static = {'utt_types': training_utt_types, 'features': training_features}
    test_static = {'feature': test_feature}

    @jax.jit
    def log_density(position):
        training = {**training_static, 'coherences': position['training_coherences']}
        test = {**test_static, 'coherence': position['test_coherence']}
        return log_likelihood(training, test, params)

    return log_density


def make_log_density_fn_joint(
    training_utt_types: jnp.ndarray,
    training_features: jnp.ndarray,
    test_features: jnp.ndarray,
    params: dict,
):
    """
    Returns a BlackJAX-compatible log_density function for all test features jointly.

    Samples training and all test pseudocoherences in one (n_train + J)-dimensional chain,
    respecting GP correlations across all features simultaneously.

    The returned function takes a position PyTree:
        position = {
            'training_coherences': jnp.array shape (n_train,),
            'test_coherences':     jnp.array shape (J,)
        }
    and returns a scalar log density.
    """
    training_static = {'utt_types': training_utt_types, 'features': training_features}

    @jax.jit
    def log_density(position):
        y_train  = position['training_coherences']   # (n_train,)
        y_test   = position['test_coherences']        # (J,)
        u_vec    = training_static['utt_types']
        x_vec    = training_static['features']

        mu_0         = params['mu_0']
        length_scale = params['length_scale']
        output_scale = params['output_scale']
        beta         = params['beta']

        # Stack all features: test first, then training (matches log_likelihood convention)
        X_all  = jnp.vstack([test_features, x_vec])          # (J + n_train, 2)
        full_y = jnp.concatenate([y_test, y_train])           # (J + n_train,)
        mu     = jnp.full(full_y.shape[0], mu_0)
        sigma  = rbf_kernel(X_all, length_scale, output_scale)

        log_gp_prior = _mvn_logpdf_chol(full_y, mu, sigma)

        def log_utt_fn(u_i, y_i):
            return jnp.log(p_u_given_y(u_i, y_i, beta) + 1e-300)

        log_utterance = jnp.sum(jax.vmap(log_utt_fn)(u_vec, y_train))

        return log_utterance + log_gp_prior

    return log_density
