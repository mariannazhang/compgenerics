import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import expit  # sigmoid


UTTERANCES = ["generic", "specific"]
# soft truth values: how likely an utterance is true given z_i
LITERAL_TRUTH_VALUES = {
    ("generic",  1): 0.95,
    ("generic",  0): 0.05,
    ("specific", 1): 0.95,
    ("specific", 0): 0.95,
}


def literal_listener(z_i: int, u_i: str, prior_z1: float) -> float:
    """
    L0(z_i | u_i): literal listener posterior over z_i given utterance u_i.
    V=1 case: only two hypotheses (z=0, z=1), prior is prior_z1 = P(z=1).
    """
    prior = {1: prior_z1, 0: 1.0 - prior_z1}
    unnorm = {z: prior[z] * LITERAL_TRUTH_VALUES[(u_i, z)] for z in [0, 1]}
    total = unnorm[0] + unnorm[1]
    return unnorm[z_i] / total


def speaker(u_i: str, z_i: int, beta: float, prior_z1: float) -> float:
    """
    S1(u_i | z_i): speaker probability via softmax over L0 utility.
    Utility = L0(z_i | u_i), i.e. how well u_i communicates z_i to the literal listener.
    """
    utilities = {u: literal_listener(z_i, u, prior_z1) for u in UTTERANCES}
    norm = sum(np.exp(beta * utilities[u]) for u in UTTERANCES)
    return np.exp(beta * utilities[u_i]) / norm


def p_u_given_y(u_i: str, y_i: float, beta: float) -> float:
    """
    P(u_i | y_i) = sum_{z_i} S1(u_i | z_i) * P(z_i | y_i)

    P(z_i=1 | y_i) = sigmoid(y_i), used as both the prior for L0 and the
    marginalizing weight over z_i.
    """
    p_z1 = expit(y_i)
    p_z0 = 1.0 - p_z1
    return speaker(u_i, 1, beta, p_z1) * p_z1 + speaker(u_i, 0, beta, p_z1) * p_z0


def rbf_kernel(X: np.ndarray, length_scale: float, output_scale: float) -> np.ndarray:
    """
    X: shape (n, 2) — 2D feature embeddings
    Returns covariance matrix of shape (n, n).
    """
    diff = X[:, None, :] - X[None, :, :]        # (n, n, 2)
    sq_dist = np.sum(diff ** 2, axis=-1)         # (n, n)
    return output_scale**2 * np.exp(-sq_dist / (2 * length_scale**2))

def log_likelihood(
    y_prime: float,
    y_vec: np.ndarray,
    u_vec: list[str],
    x_prime: np.ndarray,
    x_vec: np.ndarray,
    mu_0: float,
    length_scale: float,
    output_scale: float,
    beta: float,
) -> float:
    """
    Log-likelihood of (y', y_vec) given utterances u_vec.

    y_prime: scalar pseudocoherence of the test feature
    y_vec:   shape (n,) pseudocoherences of training features
    u_vec:   length-n list of utterances ("generic" or "specific") for training features
    x_prime: shape (2,) 2D embedding of the test feature
    x_vec:   shape (n, 2) 2D embeddings of training features
    mu_0:    scalar prior mean applied uniformly to all features
    """
    # Stack test feature first, then training features
    X_all = np.vstack([x_prime[None, :], x_vec])          # (n+1, 2)
    full_y = np.concatenate([[y_prime], y_vec])            # (n+1,)

    mu = np.full(len(full_y), mu_0)
    sigma = rbf_kernel(X_all, length_scale, output_scale)

    log_utterance = sum(
        np.log(p_u_given_y(u_i, y_i, beta) + 1e-300)
        for u_i, y_i in zip(u_vec, y_vec)
    )

    log_prior = multivariate_normal.logpdf(full_y, mean=mu, cov=sigma)

    return log_utterance + log_prior
