"""Per-feature zero/one-inflated Beta (ZOIB) linking function.

Takes in a test feature's coherence pz1 = P(z=1) and returns a distribution over 
prevalence judgements r in [0, 1].
Every feature j has its own components:
- kl component (alpha, beta)
- nkl component (alpha, beta)
- P(r=0)
- P(r=1)
- (kl and nkl weighted by coherence)
- (ends of the scales fit separately w/ their own probability mass)
- this means each feature now has 8 parameters instead of 4

p(r | feature j) = pz1_j * ZOIB_kl,j(r) + (1 - pz1_j) * ZOIB_nkl,j(r)

Each component is a zero/one-inflated Beta (Ospina & Ferrari 2010, 2012):

    P(r = 0 | c, j) = pi0_cj
    P(r = 1 | c, j) = pi1_cj
    p(r     | c, j) = (1 - pi0_cj - pi1_cj) * Beta(r | a_cj, b_cj)   for 0 < r < 1

Nothing is shared across features: 8 parameters per feature
(m, kappa, pi0, pi1 for each class). Midpoint (50) answers are handled by the
Beta part.

Constraints, so the two classes keep their intended meaning:
  - single-peaked Betas: a, b > 1 (spikes at 0/1 are the point masses' job),
  - concentration kappa = a + b <= KAPPA_MAX,
  - ordered Beta means: m_kl >= m_nkl (no margin),
  - ordered point masses: pi1_kl >= pi1_nkl and pi0_kl <= pi0_nkl
    ("kind-linked makes 100 more likely and 0 less likely"; this also stops one
    class from taking BOTH extremes, the old U-shape in another form).

The fit is penalized ML: a weak Beta(1.5, 1.5) penalty on each pi
(PI_PRIOR_A/B) keeps point masses finite when a feature has few or no 0s/100s,
and breaks ties when coherence can't separate the two classes' pis.

Point-mass membership is decided on the INTEGER slider value k = round(100 r),
so ratings must be RAW (0-100 / 100). The half-bin nudge in
inference.load_responses(clip_interior=True) must be off for this path, or
100 arrives as 0.995 and never reaches pi1.

Because nothing is shared, the fit separates into J independent 8-parameter
problems. They are solved together (all features x all starts) by one vmapped,
jitted damped Newton loop with exact Hessians.

Unconstrained parameters, per feature (row of a (J, 8) array):
    [s, t, w_nkl, w_kl, u0, u1, p, q]
    m_nkl   = M_LO + (M_HI - M_LO) * sigmoid(s)
    m_kl    = m_nkl + (M_HI - m_nkl) * sigmoid(t)                 (>= m_nkl)
    kappa_c = kmin(m_c) + (KAPPA_MAX - kmin(m_c)) * sigmoid(w_c),
              kmin(m) = max(1/m, 1/(1-m))                          (=> a, b > 1)
    (pi0_nkl, pi1_nkl, rest) = softmax(u0, u1, 0)
    pi0_kl  = pi0_nkl * sigmoid(p)                                 (<= pi0_nkl)
    pi1_kl  = pi1_nkl + (1 - pi0_kl - pi1_nkl) * sigmoid(q)        (>= pi1_nkl)
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.stats import beta as jbeta

KAPPA_MAX = 100.0
M_LO, M_HI = 0.02, 0.98       # keeps kmin(m) <= 50 < KAPPA_MAX
PI_PRIOR_A, PI_PRIOR_B = 1.5, 1.5
N_PAR = 8

# response classes (by integer slider value)
INTERIOR, ZERO, ONE = 0, 1, 2


# ---------------------------------------------------------------------------
# data encoding
# ---------------------------------------------------------------------------
def encode(responses):
    """(N, J) raw ratings in [0, 1] -> (r, cls): r with point-mass entries set to
    0.5 (a harmless placeholder for the Beta logpdf, masked out later) and the
    (N, J) int class array. Raises if ratings aren't on the 0-100 slider grid
    (e.g. half-bin-nudged ratings)."""
    r = np.asarray(responses, dtype=float)
    k = np.round(r * 100.0)
    if np.max(np.abs(r * 100.0 - k)) > 1e-6:
        raise ValueError("linking.encode: ratings are not on the 0-100 slider grid; "
                         "load them with clip_interior=False (raw, un-nudged).")
    cls = np.full(r.shape, INTERIOR, dtype=np.int32)
    cls[k == 0], cls[k == 100] = ZERO, ONE
    return jnp.asarray(np.where(cls == INTERIOR, r, 0.5)), jnp.asarray(cls)


def stack(responses_cond, pz1_cond):
    """Stack groups -> (r, cls, pz1, R) with each participant weighted by their group's pz1.
    r, cls, pz1: (N, J) jnp arrays; R: the raw (N, J) numpy ratings."""
    R = np.concatenate([np.asarray(responses_cond[c]) for c in responses_cond], axis=0)
    TH = np.concatenate([np.tile(np.asarray(pz1_cond[c])[None, :], (int(np.asarray(responses_cond[c]).shape[0]), 1))
                         for c in responses_cond], axis=0)
    r, cls = encode(R)
    return r, cls, jnp.asarray(TH), R


# ---------------------------------------------------------------------------
# parameter transforms (all work on a trailing axis of length 8)
# ---------------------------------------------------------------------------
def _kmin(m):
    return jnp.maximum(1.0 / m, 1.0 / (1.0 - m))


def unpack(theta):
    """(..., 8) unconstrained -> dict of constrained arrays with shape (...)."""
    s, t, w_nkl, w_kl, u0, u1, pp, qq = [theta[..., i] for i in range(N_PAR)]
    m_nkl = M_LO + (M_HI - M_LO) * jax.nn.sigmoid(s)
    m_kl = m_nkl + (M_HI - m_nkl) * jax.nn.sigmoid(t)
    k_nkl = _kmin(m_nkl) + (KAPPA_MAX - _kmin(m_nkl)) * jax.nn.sigmoid(w_nkl)
    k_kl = _kmin(m_kl) + (KAPPA_MAX - _kmin(m_kl)) * jax.nn.sigmoid(w_kl)
    sm = jax.nn.softmax(jnp.stack([u0, u1, jnp.zeros_like(u0)], axis=-1), axis=-1)
    pi0_nkl, pi1_nkl = sm[..., 0], sm[..., 1]
    pi0_kl = pi0_nkl * jax.nn.sigmoid(pp)
    pi1_kl = pi1_nkl + (1.0 - pi0_kl - pi1_nkl) * jax.nn.sigmoid(qq)
    return {'m_kl': m_kl, 'm_nkl': m_nkl, 'kappa_kl': k_kl, 'kappa_nkl': k_nkl,
            'a_kl': m_kl * k_kl, 'b_kl': (1 - m_kl) * k_kl,
            'a_nkl': m_nkl * k_nkl, 'b_nkl': (1 - m_nkl) * k_nkl,
            'pi0_kl': pi0_kl, 'pi1_kl': pi1_kl, 'pi0_nkl': pi0_nkl, 'pi1_nkl': pi1_nkl}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def pack(m_nkl, m_kl, kappa_nkl, kappa_kl, pi0_nkl, pi1_nkl, pi0_kl, pi1_kl):
    """Constrained per-feature values (each (J,)) -> (J, 8) unconstrained (clipped into the box)."""
    f = lambda v: np.asarray(v, dtype=float)
    m_nkl = np.clip(f(m_nkl), M_LO + 1e-4, M_HI - 1e-3)
    m_kl = np.clip(np.maximum(f(m_kl), m_nkl + 1e-4), None, M_HI - 1e-4)
    kmin = lambda m: np.maximum(1 / m, 1 / (1 - m))
    kap = lambda k, m: np.clip(f(k), kmin(m) + 1e-3, KAPPA_MAX - 1e-3)
    s = _logit((m_nkl - M_LO) / (M_HI - M_LO))
    t = _logit((m_kl - m_nkl) / (M_HI - m_nkl))
    w_nkl = _logit((kap(kappa_nkl, m_nkl) - kmin(m_nkl)) / (KAPPA_MAX - kmin(m_nkl)))
    w_kl = _logit((kap(kappa_kl, m_kl) - kmin(m_kl)) / (KAPPA_MAX - kmin(m_kl)))
    pi0_nkl, pi1_nkl = np.clip(f(pi0_nkl), 1e-4, 0.4), np.clip(f(pi1_nkl), 1e-4, 0.4)
    rest = 1 - pi0_nkl - pi1_nkl
    u0, u1 = np.log(pi0_nkl / rest), np.log(pi1_nkl / rest)
    pi0_kl = np.clip(f(pi0_kl), 1e-6, pi0_nkl * (1 - 1e-4))
    pi1_kl = np.clip(f(pi1_kl), pi1_nkl + 1e-6, 1 - pi0_kl - 1e-4)
    pp = _logit(pi0_kl / pi0_nkl)
    qq = _logit((pi1_kl - pi1_nkl) / (1 - pi0_kl - pi1_nkl))
    return np.stack([s, t, w_nkl, w_kl, u0, u1, pp, qq], axis=-1)


def params_to_numpy(p):
    return {k: np.asarray(v) for k, v in p.items()}


# ---------------------------------------------------------------------------
# class-conditional log-likelihoods and the mixture
# ---------------------------------------------------------------------------
def component_logliks(p, r, cls):
    """log p(r | kl), log p(r | nkl), each shaped like r. p: unpack() output whose
    arrays broadcast against r (e.g. (J,) against (N, J)). Independent of pz1, so
    the Laplace objective can precompute them."""
    def one(c):
        interior = jnp.log1p(-(p['pi0_' + c] + p['pi1_' + c])) + jbeta.logpdf(r, p['a_' + c], p['b_' + c])
        return jnp.where(cls == ZERO, jnp.log(p['pi0_' + c]),
                         jnp.where(cls == ONE, jnp.log(p['pi1_' + c]), interior))
    return one('kl'), one('nkl')


def mixture_loglik(lp_kl, lp_nkl, pz1):
    """Elementwise log[pz1 p(r|kl) + (1-pz1) p(r|nkl)]."""
    pz1 = jnp.clip(pz1, 1e-6, 1 - 1e-6)
    return jnp.logaddexp(jnp.log(pz1) + lp_kl, jnp.log1p(-pz1) + lp_nkl)


def _pi_penalty(p):
    """Negative log of the weak Beta(A, B) prior on each of the 4 point masses."""
    lp = 0.0
    for k in ('pi0_kl', 'pi1_kl', 'pi0_nkl', 'pi1_nkl'):
        lp = lp + (PI_PRIOR_A - 1) * jnp.log(p[k]) + (PI_PRIOR_B - 1) * jnp.log1p(-p[k])
    return -lp


def _feature_objective(theta_j, r_j, cls_j, pz1_j):
    """Penalized NLL for ONE feature: theta_j (8,), data (N,)."""
    p = unpack(theta_j)
    lp_kl, lp_nkl = component_logliks(p, r_j, cls_j)
    return -jnp.sum(mixture_loglik(lp_kl, lp_nkl, pz1_j)) + _pi_penalty(p)


def _feature_nll(theta_j, r_j, cls_j, pz1_j):
    """Unpenalized NLL for ONE feature."""
    p = unpack(theta_j)
    lp_kl, lp_nkl = component_logliks(p, r_j, cls_j)
    return -jnp.sum(mixture_loglik(lp_kl, lp_nkl, pz1_j))


# ---------------------------------------------------------------------------
# batched damped Newton (saddle-free: |eigenvalues| + Levenberg damping)
# ---------------------------------------------------------------------------
N_NEWTON = 80


def _newton_one(theta0, r_j, cls_j, pz1_j):
    f = lambda th: _feature_objective(th, r_j, cls_j, pz1_j)
    g_fn, h_fn = jax.grad(f), jax.hessian(f)

    def body(_, state):
        th, fx, lam = state
        g, H = g_fn(th), h_fn(th)
        w, V = jnp.linalg.eigh(0.5 * (H + H.T))
        step = V @ ((V.T @ g) / (jnp.abs(w) + lam))
        th_new = th - step
        f_new = f(th_new)
        ok = jnp.isfinite(f_new) & (f_new < fx)
        return (jnp.where(ok, th_new, th), jnp.where(ok, f_new, fx),
                jnp.where(ok, jnp.maximum(lam * 0.3, 1e-8), lam * 10.0))

    th, fx, _ = jax.lax.fori_loop(0, N_NEWTON, body, (theta0, f(theta0), 1e-2))
    return th, fx, jnp.max(jnp.abs(g_fn(th)))


# vmap over features (theta, data columns), then over starts (theta only)
_newton_features = jax.vmap(_newton_one, in_axes=(0, 1, 1, 1))
_newton_batch = jax.jit(jax.vmap(_newton_features, in_axes=(0, None, None, None)))
_nll_features = jax.jit(jax.vmap(_feature_nll, in_axes=(0, 1, 1, 1)))


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def data_inits(R, cls):
    """Data-driven starting points: list of (J, 8) unconstrained arrays."""
    cls = np.asarray(cls)
    J = R.shape[1]
    rate0 = np.maximum((cls == ZERO).mean(0), 2e-3)
    rate1 = np.maximum((cls == ONE).mean(0), 2e-3)
    inits = []
    for q_lo, q_hi, kap in [(25, 75, 6.0), (15, 85, 4.0), (35, 65, 10.0), (50, 50, 5.0)]:
        m_nkl, m_kl = [], []
        for j in range(J):
            x = R[:, j][cls[:, j] == INTERIOR]
            m_nkl.append(np.percentile(x, q_lo)); m_kl.append(np.percentile(x, q_hi))
        inits.append(pack(m_nkl, m_kl, np.full(J, kap), np.full(J, kap),
                          1.3 * rate0, 0.7 * rate1, 0.7 * rate0, 1.3 * rate1))
    return inits


def fit(responses_cond, pz1_cond, init=None, n_random=12, seed=0, return_details=False):
    """Penalized-ML fit of the per-feature ZOIB linking at fixed per-group pz1.

    Every feature is fit independently from several starts (the data-driven
    inits, n_random perturbations of them, and `init` if given) in one batched
    Newton call; the best start is kept separately for each feature.

    responses_cond: {group: (N_g, J) RAW ratings}; pz1_cond: {group: (J,)}.
    init: optional (J, 8) unconstrained warm start. seed fixes the random
    perturbations (the fit is deterministic).
    Returns (theta_hat (J, 8), nll) where nll is the UNPENALIZED negative
    log-likelihood; with return_details=True also a dict with the per-start
    penalized objectives (S, J) and the max |gradient| of the chosen solutions.
    """
    r, cls, th, R = stack(responses_cond, pz1_cond)
    J = int(R.shape[1])
    rng = np.random.default_rng(seed)
    base = data_inits(R, cls)
    starts = list(base) + [base[i % len(base)] + rng.normal(0.0, 1.5, (J, N_PAR)) for i in range(n_random)]
    # the warm-start slot is always present (a duplicate data init when init is None)
    # so the batch shape, and hence the compiled Newton kernel, doesn't change
    # between cold and warm calls
    first = base[0] if init is None else np.asarray(init, dtype=float).reshape(J, N_PAR)
    starts = [first] + starts
    th_all, f_all, gmax = _newton_batch(jnp.asarray(np.stack(starts)), r, cls, th)
    f_all = np.where(np.isfinite(np.asarray(f_all)), np.asarray(f_all), np.inf)   # (S, J)
    pick = np.argmin(f_all, axis=0)
    theta_hat = np.asarray(th_all)[pick, np.arange(J)]
    nll = float(np.sum(np.asarray(_nll_features(jnp.asarray(theta_hat), r, cls, th))))
    if return_details:
        return theta_hat, nll, {'objective_per_start': f_all,
                                'max_abs_grad': np.asarray(gmax)[pick, np.arange(J)]}
    return theta_hat, nll


def nll(theta, responses_cond, pz1_cond):
    """Unpenalized negative log-likelihood of theta (J, 8) at the given pz1."""
    r, cls, th, _ = stack(responses_cond, pz1_cond)
    return float(np.sum(np.asarray(_nll_features(jnp.asarray(theta), r, cls, th))))


def expected_rating(p, pz1):
    """Mean rating under the mixture: pz1*E[r|kl] + (1-pz1)*E[r|nkl] (per feature)."""
    p = params_to_numpy(p)
    e = {c: p['pi1_' + c] + (1 - p['pi0_' + c] - p['pi1_' + c]) * p['m_' + c] for c in ('kl', 'nkl')}
    pz1 = np.asarray(pz1)
    return pz1 * e['kl'] + (1 - pz1) * e['nkl']
