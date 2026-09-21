"""Fit every per-feature-prevalence model for all studies and cache the results.

One entry point for the whole fitting pipeline. For each study it fits, on the
PREREGISTERED data, the full model (fitted length scale) and its distance-blind
null (length scale pinned at LS_FLAT), plus the study-6/8 ablations. beta (RSA
speaker rationality) is inferred wherever the model has a speaker; the per-feature
Beta-mixture linking shapes are profiled out at every VBMC evaluation.

Produces, per fit, the same artifacts the notebooks consume (NO plots -- the
notebooks are the visualizers/consumers):
  <results>/<name>.npz            posterior samples (ls/mu0/sigma/beta) + meta
  <fit-dir>/fitted-link-shapes.{npz,csv}   per-feature shapes at the posterior median
  <fit-dir>/predictions.csv       per (condition|group, feature) pred_prior/pred_posterior

Caching is delete-to-refit: an existing .npz is loaded, not refit. So re-running
is cheap and only fills what's missing; `rm` a file (or a whole fit dir) to refit.

Usage:
    python run_parameter_estimation.py              # all studies
    python run_parameter_estimation.py study9       # one study (study6|study8|study9|study9-percond)
    MAX_EVALS=200 python run_parameter_estimation.py study9   # override eval budget (default 400)

Runtime is dominated by study 8 (~400 config groups, ~45 min at 400 evals).
Run headless / in the background for the full set.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")   # float64 for stable log-det / Cholesky

import sys
import csv
import json

import numpy as np
import pandas as pd
import jax.numpy as jnp

import inference as inf   # sets jax config; import before other jax users
import studies68 as s68

MAX_EVALS = int(os.environ.get("MAX_EVALS", "400"))
SAMPLE_KEYS = ('ls_samples', 'mu0_samples', 'sigma_samples', 'beta_samples')
META_KEYS   = ('elbo', 'func_count', 'convergence_status', 'runtime_s')


# ---------------------------------------------------------------------------
# shared helpers (mirror the notebooks' load_or_fit + artifact exports)
# ---------------------------------------------------------------------------
def load_or_fit(path, fit_fn):
    """Load a cached posterior .npz, or run fit_fn() and cache it. Returns the
    fit dict with SAMPLE_KEYS + META_KEYS."""
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        fit = {k: z[k] for k in SAMPLE_KEYS}
        fit.update(json.loads(str(z['meta'])))
        print(f"  loaded cached posterior: {path}  ({fit['convergence_status']})")
        return fit
    fit = fit_fn()
    np.savez_compressed(path, **{k: fit[k] for k in SAMPLE_KEYS},
                        meta=json.dumps({k: fit[k] for k in META_KEYS}))
    print(f"  saved posterior: {path}  ({fit['convergence_status']}, "
          f"{fit['func_count']} evals, {fit['runtime_s']:.0f}s)")
    return fit


def _median_theta(fit):
    return (float(np.median(fit['ls_samples'])), float(np.median(fit['mu0_samples'])),
            float(np.median(fit['sigma_samples'])), float(np.median(fit['beta_samples'])))


def export_link_shapes(results_dir, geom, responses_cond, theta, test_feature_names, test_trait):
    """Fit per-feature linking shapes at the posterior-median theta and write
    fitted-link-shapes.{npz,csv}. Returns (shapes, cond_data, mean_kl, mean_nkl)."""
    ls, mu0, sig, beta = theta
    _, shapes, cond_data = inf.total_log_lik_free_shapes(
        geom, responses_cond, ls, mu0, sig, beta_speaker=beta, return_shapes=True)
    mean_kl  = shapes[:, 0] / (shapes[:, 0] + shapes[:, 1])
    mean_nkl = shapes[:, 2] / (shapes[:, 2] + shapes[:, 3])
    np.savez_compressed(os.path.join(results_dir, 'fitted-link-shapes.npz'),
                        shapes=shapes, feature_names=np.array(test_feature_names),
                        theta=np.array(theta))
    with open(os.path.join(results_dir, 'fitted-link-shapes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature', 'trait', 'a_kl', 'b_kl', 'a_nkl', 'b_nkl', 'mean_kl', 'mean_nkl'])
        for j, name in enumerate(test_feature_names):
            w.writerow([name, test_trait[j], *[f'{v:.4f}' for v in shapes[j]],
                        f'{mean_kl[j]:.4f}', f'{mean_nkl[j]:.4f}'])
    print(f"  saved fitted-link-shapes.{{npz,csv}}")
    return shapes, cond_data, mean_kl, mean_nkl


def export_predictions(results_dir, geom, responses_cond, theta, cond_data,
                       mean_kl, mean_nkl, test_feature_names, keys, extra_cols=None):
    """Per (key, feature) ratings-free (pred_prior) + ratings-conditioned
    (pred_posterior) forward predictions, written to predictions.csv.

    keys: the condition/group keys to iterate (geom's train-cond keys).
    extra_cols: optional dict key -> dict of extra columns (e.g. study-8 group meta)."""
    ls, mu0, sig, beta = theta
    J = geom['J']
    pz1_prior = inf.prior_pz1_modes(geom, ls, mu0, sig, beta_speaker=beta)
    rows, n_bad = [], 0
    for k in keys:
        _, y_hat, ok = inf.laplace_log_Z(cond_data[k], ls, mu0, sig, beta_speaker=beta)
        n_bad += (not ok)
        pz1_post = 1.0 / (1.0 + np.exp(-y_hat[:J]))
        pred_post  = pz1_post * mean_kl + (1 - pz1_post) * mean_nkl
        pred_prior = np.asarray(pz1_prior[k]) * mean_kl + (1 - np.asarray(pz1_prior[k])) * mean_nkl
        emp = np.array(responses_cond[k]).mean(0)
        for j, f in enumerate(test_feature_names):
            row = {'key': k, 'feature': f, 'empirical_mean': float(emp[j]),
                   'pz1_prior': float(pz1_prior[k][j]),
                   'pred_prior': float(pred_prior[j]), 'pred_posterior': float(pred_post[j])}
            if extra_cols and k in extra_cols:
                row.update(extra_cols[k])
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    msg = f"  saved predictions.csv ({len(rows)} rows"
    if n_bad:
        msg += f", {n_bad} groups' Laplace mode not cleanly converged"
    print(msg + ")")


def fit_full_and_artifacts(results_dir, geom, responses_cond, keys,
                           test_feature_names, test_trait, extra_cols=None):
    """The standard per-study pipeline: fit the full model (fitted ls, beta
    inferred, profiled shapes) + the distance-blind null (ls pinned), then export
    linking shapes + predictions at the full model's posterior median."""
    os.makedirs(results_dir, exist_ok=True)
    # full model: fitted length scale
    fit_f = load_or_fit(
        os.path.join(results_dir, 'fitted-feature-shapes.npz'),
        lambda: inf.run_vbmc(geom, responses_cond, max_evals=MAX_EVALS))
    # distance-blind null: length scale pinned at LS_FLAT
    load_or_fit(
        os.path.join(results_dir, 'fitted-feature-shapes-null.npz'),
        lambda: inf.run_vbmc(geom, responses_cond, ls_fixed=inf.LS_FLAT, max_evals=MAX_EVALS))

    theta = _median_theta(fit_f)
    print(f"  posterior median: ls={theta[0]:.3f} mu_0={theta[1]:+.3f} "
          f"sigma={theta[2]:.3f} beta={theta[3]:.3f}")
    shapes, cond_data, mean_kl, mean_nkl = export_link_shapes(
        results_dir, geom, responses_cond, theta, test_feature_names, test_trait)
    export_predictions(results_dir, geom, responses_cond, theta, cond_data,
                       mean_kl, mean_nkl, test_feature_names, keys, extra_cols)


# ---------------------------------------------------------------------------
# study 6 (prereg): generic / specific / baseline; + ablations
# ---------------------------------------------------------------------------
def run_study6():
    print("\n===== STUDY 6 (prereg) =====")
    geom, responses_cond = s68.load_study6_prereg()
    keys = list(geom['x_train_cond'])
    fit_full_and_artifacts('results/study6-fit', geom, responses_cond, keys,
                           geom['test_feature_names'], geom['test_trait'])
    run_study6_ablations()


def run_study6_ablations():
    """literalsem / null / notransfer ablations (posterior only; the ablations
    notebook builds their predictions/plots)."""
    print("  --- study 6 ablations ---")
    rdir = 'results/study6-ablations'
    os.makedirs(rdir, exist_ok=True)
    # widen the beta bound as the main fits do (only affects beta-fitting ablations)
    inf.UB_B[3], inf.PUB_B[3] = np.log(1000.0), np.log(300.0)

    # literalsem: aligned literal listener (literal semantics, no speaker -> no beta)
    def fit_litsem():
        g, r = s68.load_study6_prereg()
        g['speaker'] = 'literal'
        fit = inf.run_vbmc(g, r, fit_beta=False, max_evals=MAX_EVALS)
        fit['beta_samples'] = np.full(fit['ls_samples'].shape[0], np.nan)   # no beta in this model
        return fit
    load_or_fit(os.path.join(rdir, 'posterior-litsem.npz'), fit_litsem)

    # null: distance-blind (ls pinned), beta fitted
    def fit_null():
        g, r = s68.load_study6_prereg()
        return inf.run_vbmc(g, r, ls_fixed=inf.LS_FLAT, max_evals=MAX_EVALS)
    load_or_fit(os.path.join(rdir, 'posterior-null.npz'), fit_null)

    # notransfer: every condition loses its utterances -> no channel, beta absent
    def fit_notransfer():
        g, r = s68.load_study6_prereg()
        for c in list(g['x_train_cond']):
            g['x_train_cond'][c] = jnp.zeros((0, 2))
            g['u_train_cond'][c] = jnp.zeros(0, dtype=jnp.int32)
            g['train_names_cond'][c] = []
        return inf.run_vbmc(g, r, ls_fixed=inf.LS_FLAT, fit_beta=False, max_evals=MAX_EVALS)
    load_or_fit(os.path.join(rdir, 'posterior-notransfer.npz'), fit_notransfer)


# ---------------------------------------------------------------------------
# study 8 (prereg): ~400 config groups; + literal-listener lesion
# ---------------------------------------------------------------------------
def run_study8():
    print("\n===== STUDY 8 (prereg) =====")
    geom, responses_cond, groups_df = s68.load_study8()
    keys = list(geom['x_train_cond'])
    meta = groups_df.set_index('group')
    extra = {g: {'condition': meta.loc[g, 'condition'],
                 'num_generics': int(meta.loc[g, 'num_generics']),
                 'total_utt': int(meta.loc[g, 'total_utt']),
                 'n_participants': int(meta.loc[g, 'n_participants'])}
             for g in keys}
    fit_full_and_artifacts('results/study8-fit', geom, responses_cond, keys,
                           geom['test_feature_names'], geom['test_trait'], extra_cols=extra)
    run_study8_ablations(responses_cond, groups_df)


def run_study8_ablations(responses_cond, groups_df):
    """Literal-listener lesion: specifics carry zero evidence (generic-only sets).
    Structure-matched (comparable log Z) + merged (shared-field) variants."""
    print("  --- study 8 ablations ---")
    rdir = 'results/study8-ablations'
    os.makedirs(rdir, exist_ok=True)
    inf.UB_B[3], inf.PUB_B[3] = np.log(1000.0), np.log(300.0)
    for merge, fname in [(False, 'posterior-literal-matched.npz'),
                         (True,  'posterior-literal.npz')]:
        def fit_lit(merge=merge):
            geom_lit, resp_lit, _ = s68.literal_study8(responses_cond, groups_df, merge=merge)
            return inf.run_vbmc(geom_lit, resp_lit, max_evals=MAX_EVALS)
        load_or_fit(os.path.join(rdir, fname), fit_lit)


# ---------------------------------------------------------------------------
# study 9 (prereg): 4 conditions, joint fit + per-condition fits
# ---------------------------------------------------------------------------
def run_study9():
    print("\n===== STUDY 9 (prereg, joint) =====")
    geom = inf.load_geometry()
    responses_cond, _ = inf.load_responses('../../data/study9_prereg.csv')
    keys = list(inf.CONDITIONS)
    fit_full_and_artifacts('results/study9-fit', geom, responses_cond, keys,
                           geom['test_feature_names'], geom['test_trait'])


def run_study9_percond():
    print("\n===== STUDY 9 (prereg, per-condition) =====")
    geom = inf.load_geometry()
    responses_cond, _ = inf.load_responses('../../data/study9_prereg.csv')
    rdir = 'results/study9-fit-percond'
    os.makedirs(rdir, exist_ok=True)
    for c in inf.CONDITIONS:
        print(f"  --- condition {c} ---")
        load_or_fit(
            os.path.join(rdir, f'posterior-{c}.npz'),
            lambda c=c: inf.run_vbmc(geom, {c: responses_cond[c]}, max_evals=MAX_EVALS))


# ---------------------------------------------------------------------------
STUDIES = {
    'study6':          run_study6,
    'study8':          run_study8,
    'study9':          run_study9,
    'study9-percond':  run_study9_percond,
}

if __name__ == '__main__':
    which = sys.argv[1:] or list(STUDIES)
    unknown = [w for w in which if w not in STUDIES]
    if unknown:
        sys.exit(f"unknown study/studies {unknown}; choose from {list(STUDIES)}")
    print(f"MAX_EVALS={MAX_EVALS}  running: {which}")
    for w in which:
        STUDIES[w]()
    print("\ndone.")
