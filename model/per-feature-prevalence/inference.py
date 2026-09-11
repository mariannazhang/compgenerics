"""Laplace + VBMC inference machinery for the per-feature-prevalence GP model.

Extracted from `model-param-recovery-vbmc-laplace.ipynb` and extended so that
`output_scale` (sigma) is inferred alongside `(length_scale, mu_0)`. The model:

  - one GP coherence field per condition over [x_test, x_train_cond] with an RBF
    kernel (length_scale, output_scale) and constant mean mu_0,
  - an epsilon-noisy RSA speaker for the trained generics (beta fixed by design),
  - a 2-component Beta-mixture linking function on slider ratings where the
    coherence pz1 = sigmoid(y_test) is the mixture weight (shapes shared & fixed).

The per-condition marginal likelihood log p(ratings | theta) is a Laplace
approximation around the mode of the coherence field (damped Newton), summed
across conditions, and pyVBMC infers the posterior over
phi = [log_ls, mu_0, log_sigma].

Compile-once design: theta and the per-condition data are TRACED ARGUMENTS to a
single set of jitted val/grad/hess functions, so JAX compiles once and reuses
the kernels for every evaluation and condition.

Import this module BEFORE importing jax elsewhere so the platform/x64 env vars
take effect (or set them yourself first).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")   # float64 for stable log-det / Cholesky

import csv
import pickle as pkl

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.stats import beta as jbeta

from model_jax import (rbf_kernel, p_u_given_y, p_u_given_y_literal, _mvn_logpdf_chol,
                       fit_beta_mixtures_all_features)


# ---------------------------------------------------------------------------
# Fixed-by-design constants
# ---------------------------------------------------------------------------
BETA_SPEAKER = 3.0          # RSA speaker rationality, fixed by design
EPS = 1e-6                  # keep sigmoid(y_test) / mixture weights off 0 and 1
H = 1.0 / 200.0             # half a slider bin -- nudges exact 0/1 responses inward for Beta.logpdf

# Default Beta-mixture linking shapes [a_kl, b_kl, a_nkl, b_nkl], shared across
# conditions/features and FIXED during theta inference. kl = high-mean
# "kind-linked" component (mean 0.8), nkl = low-mean (mean 0.2).
DEFAULT_LINK_SHAPES = np.array([8.0, 2.0, 2.0, 8.0])

CONDITIONS  = ['physical', 'diet', 'personality', 'heterogeneous']
CAT_OF_COND = {'diet': 'diet_preferences', 'personality': 'personality_behaviors', 'physical': 'physical'}
CAT_SHORT   = {'diet_preferences': 'diet', 'personality_behaviors': 'personality', 'physical': 'physical'}

CSV_TO_FEATURE = {
    'diet_can_eat_spicy_1': 'can eat spicy food', 'diet_breakfast_late_1': 'eat breakfast very late',
    'diet_five_meals_day_1': 'eat five meals a day', 'diet_like_juice_pulp_1': 'like juice with pulp',
    'diet_pepper_on_all_1': 'put pepper on all their foods', 'pers_cry_easily_1': 'cry easily',
    'pers_collect_rocks_1': 'like to collect rocks', 'pers_like_to_dance_1': 'like to dance',
    'pers_like_highfive_1': 'like to give high-fives', 'pers_read_books_1': 'like to read books',
    'phys_can_roll_tongue_1': 'can roll their tongue', 'phys_can_snap_toes_1': 'can snap with their toes',
    'phys_can_wiggle_ears_1': 'can wiggle their ears', 'phys_cold_hands_feet_1': 'have cold hands and feet',
    'phys_snore_sleep_1': 'snore when they sleep'}


# ---------------------------------------------------------------------------
# Geometry + data loading
# ---------------------------------------------------------------------------
def load_geometry(features_pkl_path='../features/set2_features_dataframe.pkl'):
    """Feature embeddings: per-condition trained regions + the shared test set.

    Uses the 2d UMAP-style projection (cols x_2d/y_2d). The GP machinery is
    dimension-agnostic, but we fix the 2d embedding here (384d was removed).

    Returns a dict with x_train_cond / u_train_cond (per condition), x_test (J,D),
    test_feature_names, test_trait, J, and embed.
    """
    with open(features_pkl_path, 'rb') as f:
        df = pkl.load(f)
    feat_idx = df.set_index('feature')
    train_df = df[df.split == 'train']

    def coords(sub_df):
        return jnp.array(sub_df[['x_2d', 'y_2d']].values)

    x_train_cond, u_train_cond, train_names_cond = {}, {}, {}
    for c in CONDITIONS:
        sub = train_df[train_df.in_heterogenous] if c == 'heterogeneous' else train_df[train_df.category == CAT_OF_COND[c]]
        x_train_cond[c] = coords(sub)
        u_train_cond[c] = jnp.zeros(len(sub), dtype=jnp.int32)   # all generic, localized to the region -- fixed by design
        train_names_cond[c] = list(sub['feature'].values)

    test_feature_names = list(CSV_TO_FEATURE.values())
    x_test = coords(feat_idx.loc[test_feature_names])   # (J, D)
    test_trait = [CAT_SHORT[feat_idx.loc[n, 'category']] for n in test_feature_names]

    return {'x_train_cond': x_train_cond, 'u_train_cond': u_train_cond,
            'train_names_cond': train_names_cond,
            'x_test': x_test, 'test_feature_names': test_feature_names,
            'test_trait': test_trait, 'J': int(x_test.shape[0]), 'embed': '2d'}


def load_responses(csv_path, clip_interior=True):
    """Participant slider ratings from a Qualtrics export, stratified by condition.

    Ratings are 0-100 integers -> divided by 100; rows whose test columns don't
    parse as ints (the two Qualtrics header rows, incompletes) are skipped.
    With clip_interior, exact 0/1 responses are nudged half a slider bin inward
    so the Beta-density likelihood stays finite.

    Returns (responses_cond, condition_labels_seen).
    """
    test_cols = list(CSV_TO_FEATURE.keys())
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    col_idx = [hdr.index(c) for c in test_cols]
    cond_i = hdr.index('condition')

    R, cond = [], []
    for r in rows[1:]:
        try:
            vals = [int(r[i]) / 100.0 for i in col_idx]
        except (ValueError, IndexError):
            continue
        R.append(vals)
        cond.append(r[cond_i])
    R = np.array(R)
    if clip_interior:
        R = np.clip(R, H, 1.0 - H)
    cond = np.array(cond)

    responses_cond = {c: jnp.array(R[cond == c]) for c in CONDITIONS}
    return responses_cond, sorted(set(cond))


# ---------------------------------------------------------------------------
# Laplace marginal likelihood log Z(theta) per condition
# ---------------------------------------------------------------------------
def _neg_log_post(y, length_scale, mu_0, output_scale, beta_speaker,
                  X_all, u_train, logpdf_kl, logpdf_nkl, J):
    """Neg log-posterior over the flat coherence vector y = concat([y_test, y_train]).

    g(y) = GP prior + speaker(u_train|y_train) + Beta-MIXTURE ratings(r | pz1=sigmoid(y_test)).
    Ratings likelihood per (i,j): pz1_j*Beta(r|a_kl,b_kl) + (1-pz1_j)*Beta(r|a_nkl,b_nkl),
    in log space via logaddexp -> differentiable in y. The per-(i,j) component logpdfs
    don't depend on y (shapes shared & fixed), so they're precomputed and passed in.
    theta = (length_scale, mu_0, output_scale) are traced args -> compile once.
    """
    y_test, y_train = y[:J], y[J:]
    mu_vec = jnp.full(y.shape[0], mu_0)
    K = rbf_kernel(X_all, length_scale, output_scale)

    log_gp = _mvn_logpdf_chol(y, mu_vec, K)

    def log_utt_fn(u_i, y_i):
        return jnp.log(p_u_given_y(u_i, y_i, beta_speaker) + 1e-300)
    log_speaker = jnp.sum(jax.vmap(log_utt_fn)(u_train, y_train))

    pz1 = jnp.clip(jax.nn.sigmoid(y_test), EPS, 1.0 - EPS)   # (J,) mixture weight
    log_mix = jnp.logaddexp(jnp.log(pz1)[None, :] + logpdf_kl,
                            jnp.log1p(-pz1)[None, :] + logpdf_nkl)   # (N, J)
    return -(log_gp + log_speaker + jnp.sum(log_mix))


# jit ONCE -- all conditions share array shapes, so one compilation serves everything
_val_fn  = jax.jit(_neg_log_post, static_argnames='J')
_grad_fn = jax.jit(jax.grad(_neg_log_post), static_argnames='J')
_hess_fn = jax.jit(jax.hessian(_neg_log_post), static_argnames='J')


# LITERAL-speaker variant of the objective: identical except the utterance
# likelihood is truth-conditional (p_u_given_y_literal) -- beta_speaker is
# accepted-and-ignored so the two objectives are call-compatible. Selected per
# group via cd['speaker'] ('rsa' default | 'literal'), stamped from
# geom['speaker'] by prepare_condition / _ratings_free_cond_data.
def _neg_log_post_literal(y, length_scale, mu_0, output_scale, beta_speaker,
                          X_all, u_train, logpdf_kl, logpdf_nkl, J):
    y_test, y_train = y[:J], y[J:]
    mu_vec = jnp.full(y.shape[0], mu_0)
    K = rbf_kernel(X_all, length_scale, output_scale)
    log_gp = _mvn_logpdf_chol(y, mu_vec, K)

    def log_utt_fn(u_i, y_i):
        return jnp.log(p_u_given_y_literal(u_i, y_i) + 1e-300)
    log_speaker = jnp.sum(jax.vmap(log_utt_fn)(u_train, y_train))

    pz1 = jnp.clip(jax.nn.sigmoid(y_test), EPS, 1.0 - EPS)
    log_mix = jnp.logaddexp(jnp.log(pz1)[None, :] + logpdf_kl,
                            jnp.log1p(-pz1)[None, :] + logpdf_nkl)
    return -(log_gp + log_speaker + jnp.sum(log_mix))


_val_fn_lit  = jax.jit(_neg_log_post_literal, static_argnames='J')
_grad_fn_lit = jax.jit(jax.grad(_neg_log_post_literal), static_argnames='J')
_hess_fn_lit = jax.jit(jax.hessian(_neg_log_post_literal), static_argnames='J')

_OBJ_FNS = {'rsa': (_val_fn, _grad_fn, _hess_fn),
            'literal': (_val_fn_lit, _grad_fn_lit, _hess_fn_lit)}


def prepare_condition(geom, c, responses_c, link_shapes=DEFAULT_LINK_SHAPES):
    """Per-condition constants for the compiled objective (data-dependent, theta-independent).

    link_shapes: (4,) shared shapes [a_kl, b_kl, a_nkl, b_nkl], or (J, 4) per-feature
    shapes (still shared across conditions) as returned by fit_link_shapes().
    """
    shapes = np.asarray(link_shapes, dtype=float)
    if shapes.ndim == 1:
        a_kl, b_kl, a_nkl, b_nkl = [float(v) for v in shapes]
    else:
        # (J,) per-feature shapes broadcast against the (N, J) response matrix
        a_kl, b_kl, a_nkl, b_nkl = [jnp.asarray(shapes[:, i]) for i in range(4)]
    r = jnp.asarray(responses_c)
    return {'X_all': jnp.vstack([geom['x_test'], geom['x_train_cond'][c]]),
            'u_train': geom['u_train_cond'][c],
            'logpdf_kl':  jbeta.logpdf(r, a_kl, b_kl),
            'logpdf_nkl': jbeta.logpdf(r, a_nkl, b_nkl),
            'J': geom['J'],
            'D': geom['J'] + int(geom['x_train_cond'][c].shape[0]),
            'speaker': geom.get('speaker', 'rsa')}


def prepare_all_conditions(geom, responses_cond, link_shapes=DEFAULT_LINK_SHAPES):
    """prepare_condition() for every condition/group in responses_cond -> {condition: cd} dict.

    Group keys are taken from responses_cond (not the study-9 CONDITIONS constant),
    so the same machinery serves any study whose geom carries matching
    x_train_cond / u_train_cond entries."""
    return {c: prepare_condition(geom, c, responses_cond[c], link_shapes) for c in responses_cond}


def laplace_log_Z(cd, length_scale, mu_0, output_scale,
                  beta_speaker=BETA_SPEAKER, n_newton=100, tol=1e-8):
    """Laplace approx of log p(ratings_c | theta) for one condition.

    DAMPED Newton (backtracking line search) minimizes -g(y) to the mode, then
    log Z = g(y_hat) + D/2 log(2pi) - 1/2 log det H, with H = grad^2 (-g)(y_hat)
    = posterior precision. `cd` is a prepare_condition() dict.
    Returns (log_Z, y_hat, ok).
    """
    theta = (float(length_scale), float(mu_0), float(output_scale), float(beta_speaker))
    args = (cd['X_all'], cd['u_train'], cd['logpdf_kl'], cd['logpdf_nkl'])
    D, J = cd['D'], cd['J']
    val_fn, grad_fn, hess_fn = _OBJ_FNS[cd.get('speaker', 'rsa')]
    y = jnp.full(D, mu_0)                    # init at the GP mean (mu_0)
    f = float(val_fn(y, *theta, *args, J=J))
    for _ in range(n_newton):
        g  = grad_fn(y, *theta, *args, J=J)
        Hm = hess_fn(y, *theta, *args, J=J) + 1e-6 * jnp.eye(D)   # numerical PD guard
        step = jnp.linalg.solve(Hm, g)
        # backtracking (Armijo) line search along the Newton direction
        t, accepted = 1.0, False
        for _ls in range(30):
            y_try = y - t * step
            f_try = float(val_fn(y_try, *theta, *args, J=J))
            if np.isfinite(f_try) and f_try <= f - 1e-4 * t * float(jnp.dot(g, step)):
                accepted = True; break
            t *= 0.5
        if not accepted:
            break                            # can't decrease further -> at (or stuck near) the mode
        converged = float(jnp.max(jnp.abs(y_try - y))) < tol
        y, f = y_try, f_try
        if converged:
            break
    Hm = hess_fn(y, *theta, *args, J=J) + 1e-6 * jnp.eye(D)
    sign, logdet = jnp.linalg.slogdet(Hm)
    gnorm = float(jnp.max(jnp.abs(grad_fn(y, *theta, *args, J=J))))   # ~0 at the mode
    log_Z = -f + 0.5 * D * jnp.log(2.0 * jnp.pi) - 0.5 * logdet
    ok = bool(sign > 0) and bool(np.isfinite(log_Z)) and gnorm < 1e-2
    return float(log_Z), np.asarray(y), ok


def total_log_lik(cond_data, length_scale, mu_0, output_scale, beta_speaker=BETA_SPEAKER):
    """Sum of Laplace log Z across the groups in cond_data at the given theta (no theta-prior)."""
    return sum(laplace_log_Z(cond_data[c], length_scale, mu_0, output_scale, beta_speaker)[0]
               for c in cond_data)


# ---------------------------------------------------------------------------
# Fitted (profiled) linking shapes
# ---------------------------------------------------------------------------
# Instead of fixing the Beta-mixture shapes, re-fit them at every theta via
# fit_beta_mixtures_all_features (the once-per-eval machinery from the MCMC
# notebooks): per-feature shapes, SHARED across conditions, fit to all
# participants with each participant weighted by their own condition's pz1.
# This is a PROFILE likelihood over the shapes (empirical-Bayes style), not a
# marginalization.
#
# Default (n_alt=1) mirrors the MCMC notebooks exactly: pz1 comes from the
# RATINGS-FREE coherence mode (GP prior + training utterances only — in the
# MCMC pipeline the test ratings never entered the coherence sampler), the
# shapes are fit once to that pz1, and log Z is scored under them. No arbitrary
# shape init enters anywhere. n_alt>1 adds refinement rounds where the
# coherence mode conditions on the ratings under the current shapes — but note
# the shape fit (interval likelihood, pz1 fixed) and log Z (Laplace marginal,
# density likelihood) are different objectives, so extra rounds are NOT
# monotone in log Z; empirically ll drifts down with more rounds.

# The interval-likelihood fit caps log-shape-params at 15 (near-delta components
# on tied slider values). The Laplace objective uses the Beta DENSITY, where such
# spikes are pathological, so fitted shapes are clipped to a sane range.
SHAPE_CLIP = (0.1, 100.0)


def laplace_pz1_modes(cond_data, length_scale, mu_0, output_scale, beta_speaker=BETA_SPEAKER):
    """Posterior-mode coherence pz1 = sigmoid(y_hat_test) per condition. Returns {c: (J,)}."""
    out = {}
    for c in cond_data:
        _, y_hat, _ = laplace_log_Z(cond_data[c], length_scale, mu_0, output_scale, beta_speaker)
        out[c] = 1.0 / (1.0 + np.exp(-y_hat[:cond_data[c]['J']]))
    return out


def _ratings_free_cond_data(geom):
    """Per-condition cd dicts with ZERO ratings rows: the objective reduces to
    GP prior + speaker, so the Laplace mode is the model's prediction of test
    coherence from training alone (the analog of the MCMC coherence sampler,
    which never saw the test ratings)."""
    cd = {}
    for c in geom['x_train_cond']:
        J = geom['J']
        cd[c] = {'X_all': jnp.vstack([geom['x_test'], geom['x_train_cond'][c]]),
                 'u_train': geom['u_train_cond'][c],
                 'logpdf_kl':  jnp.zeros((0, J)),
                 'logpdf_nkl': jnp.zeros((0, J)),
                 'J': J,
                 'D': J + int(geom['x_train_cond'][c].shape[0]),
                 'speaker': geom.get('speaker', 'rsa')}
    return cd


def prior_pz1_modes(geom, length_scale, mu_0, output_scale, beta_speaker=BETA_SPEAKER):
    """Ratings-free coherence-mode pz1 per condition (GP + training utterances only)."""
    return laplace_pz1_modes(_ratings_free_cond_data(geom),
                             length_scale, mu_0, output_scale, beta_speaker)


def rbf_cross(A, B, length_scale, output_scale):
    """RBF cross-covariance between two point sets, (len(A), len(B)); no jitter."""
    d2 = jnp.sum((jnp.asarray(A)[:, None, :] - jnp.asarray(B)[None, :, :]) ** 2, axis=-1)
    return output_scale ** 2 * jnp.exp(-d2 / (2 * length_scale ** 2))


def gp_coherence_field(geom, c, length_scale, mu_0, output_scale, X_query,
                       beta_speaker=BETA_SPEAKER):
    """Coherence field pz1 for one condition, evaluated at arbitrary query points.

    Ratings-free Laplace mode of GP + speaker over the condition's training
    utterances, then the GP conditional mean at X_query given the training-block
    mode, squashed through the sigmoid. Returns (len(X_query),) numpy array.
    """
    if int(geom['x_train_cond'][c].shape[0]) == 0:
        # no training utterances (e.g. a baseline group): the ratings-free mode
        # is the GP prior mean everywhere
        return np.full(int(np.asarray(X_query).shape[0]), 1.0 / (1.0 + np.exp(-mu_0)))
    cd = _ratings_free_cond_data(geom)[c]
    _, y_hat, _ = laplace_log_Z(cd, length_scale, mu_0, output_scale, beta_speaker)
    y_tr = jnp.asarray(y_hat[geom['J']:])
    x_tr = geom['x_train_cond'][c]
    K_tr = rbf_kernel(x_tr, length_scale, output_scale)
    K_q  = rbf_cross(X_query, x_tr, length_scale, output_scale)
    y_q  = mu_0 + K_q @ jnp.linalg.solve(K_tr, y_tr - mu_0)
    return np.asarray(jax.nn.sigmoid(y_q))


def fit_link_shapes(responses_cond, pz1_cond):
    """Per-feature Beta-mixture shapes shared across conditions, given per-condition pz1.

    Stacks all participants (condition order) with each weighted by their own
    condition's pz1, then runs the interval-likelihood fit from model_jax.
    Returns (J, 4) [a_kl, b_kl, a_nkl, b_nkl], clipped to SHAPE_CLIP.
    """
    responses_all = jnp.concatenate([responses_cond[c] for c in responses_cond], axis=0)
    pz1_weights = jnp.concatenate(
        [jnp.tile(jnp.asarray(pz1_cond[c])[None, :], (int(responses_cond[c].shape[0]), 1))
         for c in responses_cond], axis=0)                               # (N_total, J)
    shapes = np.asarray(fit_beta_mixtures_all_features(responses_all, pz1_weights))   # (J, 4)
    return np.clip(shapes, *SHAPE_CLIP)


def total_log_lik_free_shapes(geom, responses_cond, length_scale, mu_0, output_scale,
                              n_alt=1, beta_speaker=BETA_SPEAKER, return_shapes=False):
    """Summed Laplace log Z at theta with the linking shapes PROFILED OUT.

    n_alt=1 (default, MCMC-notebook analog): fit the shapes once to the
    ratings-free prior-mode pz1, then score log Z under them. n_alt>1 adds
    rounds where the coherence mode conditions on ratings under the current
    shapes (see the non-monotonicity caveat above). Deterministic given
    (theta, n_alt). Returns ll, or (ll, shapes, cond_data) with
    return_shapes=True.
    """
    pz1 = prior_pz1_modes(geom, length_scale, mu_0, output_scale, beta_speaker)
    shapes = fit_link_shapes(responses_cond, pz1)
    for _ in range(n_alt - 1):
        cond_data = prepare_all_conditions(geom, responses_cond, link_shapes=shapes)
        pz1 = laplace_pz1_modes(cond_data, length_scale, mu_0, output_scale, beta_speaker)
        shapes = fit_link_shapes(responses_cond, pz1)
    cond_data = prepare_all_conditions(geom, responses_cond, link_shapes=shapes)
    ll = total_log_lik(cond_data, length_scale, mu_0, output_scale, beta_speaker)
    if return_shapes:
        return ll, shapes, cond_data
    return ll


def conditional_ratings_log_lik(geom, responses_cond, length_scale, mu_0, output_scale,
                                beta_speaker=BETA_SPEAKER):
    """Summed log p(ratings | utterances, theta): Laplace log Z of the full posterior
    minus the ratings-free log Z (GP prior + speaker only), per group, with the
    linking shapes profiled at this theta.

    Use this (not total_log_lik) to COMPARE models that condition on different
    utterance sets (e.g. full vs literal-listener lesions): total_log_lik scores the
    utterances as data too, so a model that drops utterances stops paying their
    log p(u) terms and its log Z shifts by bookkeeping, not fit. Conditioning on
    the utterances scores every model on the same ratings data.
    """
    _, _, cond_data = total_log_lik_free_shapes(geom, responses_cond, length_scale, mu_0,
                                                output_scale, beta_speaker=beta_speaker,
                                                return_shapes=True)
    free = _ratings_free_cond_data(geom)
    ll = 0.0
    for c in cond_data:
        ll += (laplace_log_Z(cond_data[c], length_scale, mu_0, output_scale, beta_speaker)[0]
               - laplace_log_Z(free[c], length_scale, mu_0, output_scale, beta_speaker)[0])
    return ll


# ---------------------------------------------------------------------------
# VBMC over phi = [log_ls, mu_0, log_sigma]
# ---------------------------------------------------------------------------
# The Laplace objective is DETERMINISTIC, but we still declare a tiny observation
# noise to VBMC and run with specify_target_noise=True. Two reasons:
#  (1) pyVBMC/gpyreg's EXACT-observation GP path has a noise-hyperparameter gradient
#      bug that crashes train_gp on this problem; the noisy path avoids that code.
#  (2) a small jitter regularizes the surrogate GP over the objective.
# TARGET_NOISE is tiny relative to the spread of log_joint, so it doesn't smear
# the posterior.
TARGET_NOISE = 1.0

# ---------------------------------------------------------------------------
# Length-scale prior/box and the flat-ls convention, calibrated to the 2d
# embedding's pairwise-distance scale: UMAP-style projection, dists 0.003-0.78
# (max = cloud diameter 0.78); LS_FLAT=10 ~ 12.8*dmax -> kernel constant to
# ~0.997 (the null / distance-blind convention). mu_0 / sigma / beta priors are
# geometry-independent. (The model works in any dimension, but we fix the 2d
# embedding here; 384d support was removed.)
LS_FLAT = 10.0
_LS_PRIOR_CENTER = 0.5

def log_prior_ls(log_ls):       return float(-0.5 * ((log_ls - np.log(_LS_PRIOR_CENTER)) / 1.5) ** 2)
def log_prior_mu(mu_0):         return float(-0.5 * mu_0 ** 2)
def log_prior_sigma(log_sigma): return float(-0.5 * ((log_sigma - np.log(1.5)) / 1.0) ** 2)

# VBMC box for phi = [log_ls, mu_0, log_sigma]; sigma is lognormal-ish around the
# previously-fixed value 1.5.
X0  = np.array([np.log(0.3),   0.0, np.log(1.5)])
LB  = np.array([np.log(0.03), -4.0, np.log(0.05)])
UB  = np.array([np.log(4.0),   4.0, np.log(8.0)])
PLB = np.array([np.log(0.08), -2.0, np.log(0.3)])
PUB = np.array([np.log(2.5),   2.0, np.log(4.0)])


# ---------------------------------------------------------------------------
# Speaker-rationality (beta) prior + the 4-D VBMC box
# ---------------------------------------------------------------------------
# beta (RSA speaker rationality) is fit by default as a 4th coordinate,
# phi = [log_ls, mu_0, log_sigma, log_beta], on the log scale with a
# lognormal-ish prior around the design value BETA_SPEAKER=3. The base box X0..PUB
# holds [log_ls, mu_0, log_sigma]; X0_B..PUB_B extends it with the beta coordinate.
# Two independent switches collapse the box at fit time (see build_log_joint):
#   fit_beta=False  -> drop the log_beta coordinate (beta held at BETA_SPEAKER, or
#                      structurally absent, e.g. a literal listener with no speaker
#                      or a model with no utterances); beta_samples records the
#                      value it was pinned to (NaN when the model has no speaker).
#   ls_fixed=<val>  -> drop the log_ls coordinate (null / distance-blind model);
#                      ls_samples is filled with the pinned constant.
def log_prior_beta(log_beta):   return float(-0.5 * ((log_beta - np.log(3.0)) / 1.0) ** 2)

X0_B  = np.append(X0,  np.log(3.0))
LB_B  = np.append(LB,  np.log(0.2))
UB_B  = np.append(UB,  np.log(30.0))
PLB_B = np.append(PLB, np.log(1.0))
PUB_B = np.append(PUB, np.log(10.0))


def _patch_pyvbmc_varg_squeeze():
    """pyVBMC numpy-compat fix: _gp_log_joint can return varG as a (1,1) array,
    which propagates into varF and crashes _eval_full_elcbo with 'setting an array
    element with a sequence' on longer runs. Squeeze it to a scalar. Idempotent."""
    import pyvbmc.vbmc.variational_optimization as _vopt
    if getattr(_vopt._gp_log_joint, "_varg_squeeze_patch", False):
        return
    _orig = _vopt._gp_log_joint
    def _patched(*a, **k):
        out = list(_orig(*a, **k))
        out[2] = None if out[2] is None else float(np.squeeze(out[2]))
        return tuple(out)
    _patched._varg_squeeze_patch = True
    _vopt._gp_log_joint = _patched


# ---------------------------------------------------------------------------
# One parametrized VBMC objective + driver.
#
# The phi vector is assembled from up to four coordinates in this fixed order:
#     [log_ls (if ls_fixed is None), mu_0, log_sigma, log_beta (if fit_beta)]
# so mu_0/log_sigma are always present, while the ls and beta coordinates are
# switched in or out. build_log_joint returns (log_joint, coord_spec); the driver
# uses coord_spec to slice the posterior samples back into natural-unit arrays.
# ---------------------------------------------------------------------------
def build_log_joint(geom_or_cond_data, responses_cond=None, *, ls_fixed=None,
                    fit_beta=True, free_shapes=True, n_alt=1, counter=None, verbose=True):
    """Assemble the VBMC log-joint objective and its coordinate spec.

    free_shapes=True  -> geom_or_cond_data is `geom`, `responses_cond` is required,
                         and the per-feature Beta-mixture linking shapes are PROFILED
                         OUT (re-fit) at every theta (via total_log_lik_free_shapes).
    free_shapes=False -> geom_or_cond_data is `cond_data` with the linking shapes
                         baked in; `responses_cond` is ignored (via total_log_lik).

    ls_fixed None -> length scale is a fitted coordinate; a float pins it (null model).
    fit_beta      -> whether the RSA speaker rationality beta is a fitted coordinate;
                     when False it is held at BETA_SPEAKER (its design value).

    Returns (log_joint, coord_spec) where coord_spec is a dict with the box arrays
    (x0, lb, ub, plb, pub) and the flags, used by _run_vbmc to slice samples.
    """
    counter = counter if counter is not None else [0]
    fit_ls = ls_fixed is None

    # ---- assemble the box for the active coordinates --------------------
    # order: [log_ls?, mu_0, log_sigma, log_beta?]
    base_x0, base_lb, base_ub, base_plb, base_pub = X0_B, LB_B, UB_B, PLB_B, PUB_B
    #                idx: [0=log_ls, 1=mu_0, 2=log_sigma, 3=log_beta]
    keep = ([0] if fit_ls else []) + [1, 2] + ([3] if fit_beta else [])
    x0  = base_x0[keep].astype(float)
    lb, ub, plb, pub = base_lb[keep], base_ub[keep], base_plb[keep], base_pub[keep]
    coord_spec = dict(fit_ls=fit_ls, fit_beta=fit_beta, ls_fixed=ls_fixed,
                      x0=x0, lb=lb, ub=ub, plb=plb, pub=pub)

    def _unpack(phi):
        """phi (active coords) -> (ls, mu_0, sigma, beta) in natural units."""
        phi = np.asarray(phi).ravel()
        i = 0
        if fit_ls:
            ls = float(np.exp(phi[i])); i += 1
        else:
            ls = float(ls_fixed)
        mu_0 = float(phi[i]); i += 1
        sigma = float(np.exp(phi[i])); i += 1
        beta = float(np.exp(phi[i])) if fit_beta else float(BETA_SPEAKER)
        return ls, mu_0, sigma, beta

    def log_joint(phi):
        ls, mu_0, sigma, beta = _unpack(phi)
        counter[0] += 1
        if free_shapes:
            ll = total_log_lik_free_shapes(geom_or_cond_data, responses_cond, ls, mu_0,
                                           sigma, n_alt=n_alt, beta_speaker=beta)
        else:
            ll = total_log_lik(geom_or_cond_data, ls, mu_0, sigma, beta_speaker=beta)
        lp = log_prior_mu(mu_0) + log_prior_sigma(np.log(sigma))
        if fit_ls:
            lp += log_prior_ls(np.log(ls))
        if fit_beta:
            lp += log_prior_beta(np.log(beta))
        val = ll + lp
        if verbose:
            ls_tag = f"{ls:.3f}" + ("" if fit_ls else " (pinned)")
            beta_tag = f"  beta={beta:.3f}" + ("" if fit_beta else " (fixed)")
            print(f"  eval {counter[0]:3d}: ls={ls_tag}  mu_0={mu_0:+.3f}  "
                  f"sigma={sigma:.3f}{beta_tag}  log_lik={ll:.1f}  log_joint={val:.1f}")
        return val, TARGET_NOISE

    return log_joint, coord_spec


def _run_vbmc(log_joint, coord_spec, max_evals, x0, n_posterior_samples):
    """pyVBMC driver over the active-coordinate box in coord_spec. Slices the
    posterior samples back into natural-unit arrays, filling pinned/fixed
    coordinates with their constants so the return dict (and saved .npz) has the
    same keys regardless of which coordinates were fitted."""
    from pyvbmc import VBMC
    import time
    _patch_pyvbmc_varg_squeeze()
    t0 = time.time()
    x0 = coord_spec['x0'] if x0 is None else np.asarray(x0, dtype=float)
    vbmc = VBMC(log_joint, x0,
                coord_spec['lb'], coord_spec['ub'], coord_spec['plb'], coord_spec['pub'],
                options={'specify_target_noise': True, 'max_fun_evals': max_evals})
    result, stats = vbmc.optimize()
    phi_samples, _ = result.sample(n_posterior_samples)
    n = phi_samples.shape[0]

    i = 0
    if coord_spec['fit_ls']:
        ls_samples = np.exp(phi_samples[:, i]); i += 1
    else:
        ls_samples = np.full(n, float(coord_spec['ls_fixed']))
    mu0_samples = phi_samples[:, i]; i += 1
    sigma_samples = np.exp(phi_samples[:, i]); i += 1
    if coord_spec['fit_beta']:
        beta_samples = np.exp(phi_samples[:, i])
    else:
        # beta held at its design value; callers that model NO speaker (e.g. the
        # aligned literal listener) overwrite this with NaN after the fit.
        beta_samples = np.full(n, float(BETA_SPEAKER))

    return {
        'ls_samples': ls_samples, 'mu0_samples': mu0_samples,
        'sigma_samples': sigma_samples, 'beta_samples': beta_samples,
        'elbo': float(stats['elbo']), 'func_count': int(stats['func_count']),
        'convergence_status': str(stats['convergence_status']),
        'runtime_s': time.time() - t0,
        'result': result, 'stats': stats,
    }


def run_vbmc(geom, responses_cond=None, *, ls_fixed=None, fit_beta=True,
             free_shapes=True, max_evals=None, n_alt=1, x0=None,
             n_posterior_samples=int(1e4), verbose=True):
    """Fit the VBMC posterior over the GP hyperparameters, with beta fitted by default.

    Coordinates fitted (in order): length_scale (unless ls_fixed pins it), mu_0,
    output_scale, and speaker rationality beta (unless fit_beta=False). The
    per-feature Beta-mixture linking shapes are profiled out per evaluation when
    free_shapes=True (pass `geom` + `responses_cond`); with free_shapes=False pass
    a `cond_data` dict with the shapes baked in as the first argument.

    Returns a dict with posterior samples in NATURAL units — ls_samples,
    mu0_samples, sigma_samples, beta_samples (always present: fitted, or filled
    with the pinned/design constant) — plus elbo / func_count /
    convergence_status / runtime_s and the raw (result, stats) pyVBMC objects.
    Pinned length scale -> ls_samples is that constant; unfitted beta ->
    beta_samples is BETA_SPEAKER (overwrite with NaN if the model has no speaker).
    """
    if max_evals is None:
        max_evals = 200 if fit_beta else 150
    log_joint, coord_spec = build_log_joint(
        geom, responses_cond, ls_fixed=ls_fixed, fit_beta=fit_beta,
        free_shapes=free_shapes, n_alt=n_alt, verbose=verbose)
    return _run_vbmc(log_joint, coord_spec, max_evals, x0, n_posterior_samples)
