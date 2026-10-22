"""Fit every per-feature-prevalence model for all studies and cache the results.

One entry point for the whole fitting pipeline. For each study it fits, on the
PREREGISTERED data, the full model (fitted length scale) and its distance-blind
null (length scale pinned at LS_FLAT). Studies 6 and 8 additionally get a
consistent ablation set: litsem (literal semantics, no speaker/beta), null
(distance-blind), and notransfer (no utterance channel). beta (RSA speaker
rationality) is inferred wherever the model has a speaker; the per-feature
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
    python run_parameter_estimation.py study9-zoib  # per-feature ZOIB linking -> results/study9-fit-zoib/
    python run_parameter_estimation.py study9-zoib-amp  # + non-stationary amplitude (amp_w fitted)
                                                    #   -> results/study9-fit-zoib-amp/
                                                    # (also study6-zoib -> results/study6-fit-zoib/
                                                    #  + ablations -> results/study6-ablations-zoib/)
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
# Fixed RNG seed for reproducibility: pyVBMC uses legacy np.random and the Laplace
# objective is deterministic, so a fixed seed makes every fit bit-for-bit
# reproducible. Override with SEED=... in the environment.
SEED = int(os.environ.get("SEED", "37"))
SAMPLE_KEYS = ('ls_samples', 'mu0_samples', 'sigma_samples', 'beta_samples')
OPTIONAL_SAMPLE_KEYS = ('amp_w_samples',)   # absent from caches written before the amplitude model
META_KEYS   = ('elbo', 'elbo_sd', 'func_count', 'convergence_status', 'runtime_s')


# ---------------------------------------------------------------------------
# shared helpers (mirror the notebooks' load_or_fit + artifact exports)
# ---------------------------------------------------------------------------
def load_or_fit(path, fit_fn):
    """Load a cached posterior .npz, or run fit_fn() and cache it. Returns the
    fit dict with SAMPLE_KEYS + META_KEYS."""
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        fit = {k: z[k] for k in SAMPLE_KEYS + OPTIONAL_SAMPLE_KEYS if k in z.files}
        fit.update(json.loads(str(z['meta'])))
        print(f"  loaded cached posterior: {path}  ({fit['convergence_status']})")
        return fit
    fit = fit_fn()
    np.savez_compressed(path, **{k: fit[k] for k in SAMPLE_KEYS + OPTIONAL_SAMPLE_KEYS if k in fit},
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
        lambda: inf.run_vbmc(geom, responses_cond, max_evals=MAX_EVALS, seed=SEED))
    # distance-blind null: length scale pinned at LS_FLAT
    load_or_fit(
        os.path.join(results_dir, 'fitted-feature-shapes-null.npz'),
        lambda: inf.run_vbmc(geom, responses_cond, ls_fixed=inf.LS_FLAT, max_evals=MAX_EVALS, seed=SEED))

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
def run_ablations(rdir, loader):
    """The shared ablation set for studies 6 and 8 (posterior only; the ablations
    notebooks build their predictions/plots):
      - litsem     : literal semantics (speaker='literal'), all utterances kept,
                     no speaker -> beta absent. The paper's null for the whole
                     RSA+beta apparatus.
      - null       : distance-blind (length scale pinned at LS_FLAT), beta fitted.
      - notransfer : every condition loses its utterances -> no channel from
                     statements to test features; ls pinned, beta absent.

    loader() must return (geom, responses_cond) freshly each call (the litsem /
    notransfer fits mutate geom in place, so each needs its own copy)."""
    print(f"  --- ablations -> {rdir} ---")
    os.makedirs(rdir, exist_ok=True)
    # widen the beta bound as the main fits do (only affects beta-fitting ablations)
    inf.UB_B[3], inf.PUB_B[3] = np.log(1000.0), np.log(300.0)

    def fit_litsem():
        g, r = loader()
        g['speaker'] = 'literal'
        fit = inf.run_vbmc(g, r, fit_beta=False, max_evals=MAX_EVALS, seed=SEED)
        fit['beta_samples'] = np.full(fit['ls_samples'].shape[0], np.nan)   # no beta in this model
        return fit
    load_or_fit(os.path.join(rdir, 'posterior-litsem.npz'), fit_litsem)

    def fit_null():
        g, r = loader()
        return inf.run_vbmc(g, r, ls_fixed=inf.LS_FLAT, max_evals=MAX_EVALS, seed=SEED)
    load_or_fit(os.path.join(rdir, 'posterior-null.npz'), fit_null)

    def fit_notransfer():
        g, r = loader()
        for c in list(g['x_train_cond']):
            g['x_train_cond'][c] = jnp.zeros((0, 2))
            g['u_train_cond'][c] = jnp.zeros(0, dtype=jnp.int32)
            g['train_names_cond'][c] = []
        return inf.run_vbmc(g, r, ls_fixed=inf.LS_FLAT, fit_beta=False, max_evals=MAX_EVALS, seed=SEED)
    load_or_fit(os.path.join(rdir, 'posterior-notransfer.npz'), fit_notransfer)


def run_study6():
    print("\n===== STUDY 6 (prereg) =====")
    geom, responses_cond = s68.load_study6_prereg()
    keys = list(geom['x_train_cond'])
    fit_full_and_artifacts('results/study6-fit', geom, responses_cond, keys,
                           geom['test_feature_names'], geom['test_trait'])
    run_ablations('results/study6-ablations', s68.load_study6_prereg)


# ---------------------------------------------------------------------------
# study 8 (prereg): ~400 config groups; same ablation set as study 6
# ---------------------------------------------------------------------------
def _load_study8_geom_resp():
    """study-8 loader adapter: (geom, responses_cond) only (drops groups_df) so it
    matches the run_ablations loader contract."""
    geom, responses_cond, _ = s68.load_study8()
    return geom, responses_cond


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
    run_ablations('results/study8-ablations', _load_study8_geom_resp)


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
            lambda c=c: inf.run_vbmc(geom, {c: responses_cond[c]}, max_evals=MAX_EVALS, seed=SEED))


# ---------------------------------------------------------------------------
# per-feature ZOIB linking (linking.py): full + null fits, own results dirs
# ---------------------------------------------------------------------------
# Same pipeline as fit_full_and_artifacts, but with link='zoib' and RAW ratings
# (no half-bin nudge). Writes ONLY to results/<study>-fit-zoib/, so the original
# free-shape fits in results/<study>-fit/ are never touched.
def export_zoib_link(results_dir, geom, responses_raw, theta, test_feature_names, test_trait,
                     amp_w=None):
    """Fit the ZOIB linking at the posterior-median theta; write
    fitted-link-zoib.{npz,csv}. Returns (link_theta, cond_data).
    amp_w: posterior-median amplitude weights (None = stationary kernel)."""
    import linking as lk
    ls, mu0, sig, beta = theta
    _, link_theta, cond_data = inf.total_log_lik_zoib(
        geom, responses_raw, ls, mu0, sig, beta_speaker=beta, return_link=True, amp_w=amp_w)
    p = lk.params_to_numpy(lk.unpack(jnp.asarray(link_theta)))
    cols = ['m_kl', 'kappa_kl', 'a_kl', 'b_kl', 'pi0_kl', 'pi1_kl',
            'm_nkl', 'kappa_nkl', 'a_nkl', 'b_nkl', 'pi0_nkl', 'pi1_nkl']
    np.savez_compressed(os.path.join(results_dir, 'fitted-link-zoib.npz'),
                        link_theta=link_theta, feature_names=np.array(test_feature_names),
                        theta=np.array(theta), **{k: p[k] for k in cols},
                        **({} if amp_w is None else {'amp_w': np.asarray(amp_w)}))
    pd.DataFrame({'feature': test_feature_names, 'trait': test_trait, **{k: p[k] for k in cols}}
                 ).round(5).to_csv(os.path.join(results_dir, 'fitted-link-zoib.csv'), index=False)
    print("  saved fitted-link-zoib.{npz,csv}")
    return link_theta, cond_data


def export_zoib_predictions(results_dir, geom, responses_raw, theta, link_theta, cond_data,
                            test_feature_names, keys, extra_cols=None, amp_w=None):
    """predictions.csv for the ZOIB fit: ratings-free (pred_prior) and
    ratings-conditioned (pred_posterior) mean ratings under the ZOIB mixture."""
    import linking as lk
    ls, mu0, sig, beta = theta
    J = geom['J']
    p = lk.unpack(jnp.asarray(link_theta))
    pz1_prior = inf.prior_pz1_modes(geom, ls, mu0, sig, beta_speaker=beta, amp_w=amp_w)
    rows, n_bad = [], 0
    for k in keys:
        _, y_hat, ok = inf.laplace_log_Z(cond_data[k], ls, mu0, sig, beta_speaker=beta, amp_w=amp_w)
        n_bad += (not ok)
        pz1_post = 1.0 / (1.0 + np.exp(-y_hat[:J]))
        pred_prior = lk.expected_rating(p, pz1_prior[k])
        pred_post = lk.expected_rating(p, pz1_post)
        emp = np.array(responses_raw[k]).mean(0)
        for j, f in enumerate(test_feature_names):
            row = {'key': k, 'feature': f, 'empirical_mean': float(emp[j]),
                   'pz1_prior': float(pz1_prior[k][j]), 'pz1_posterior': float(pz1_post[j]),
                   'pred_prior': float(pred_prior[j]), 'pred_posterior': float(pred_post[j])}
            if extra_cols and k in extra_cols:
                row.update(extra_cols[k])
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    msg = f"  saved predictions.csv ({len(rows)} rows"
    if n_bad:
        msg += f", {n_bad} groups' Laplace mode not cleanly converged"
    print(msg + ")")


def fit_zoib_and_artifacts(results_dir, geom, responses_raw, keys,
                           test_feature_names, test_trait, extra_cols=None):
    """ZOIB counterpart of fit_full_and_artifacts: full model (fitted ls) + the
    distance-blind null (ls pinned), then linking + predictions at the full
    model's posterior median. Cached like the rest (delete-to-refit)."""
    os.makedirs(results_dir, exist_ok=True)
    fit_f = load_or_fit(
        os.path.join(results_dir, 'posterior.npz'),
        lambda: inf.run_vbmc(geom, responses_raw, link='zoib', max_evals=MAX_EVALS, seed=SEED))
    load_or_fit(
        os.path.join(results_dir, 'posterior-null.npz'),
        lambda: inf.run_vbmc(geom, responses_raw, link='zoib', ls_fixed=inf.LS_FLAT,
                             max_evals=MAX_EVALS, seed=SEED))
    theta = _median_theta(fit_f)
    print(f"  posterior median: ls={theta[0]:.3f} mu_0={theta[1]:+.3f} "
          f"sigma={theta[2]:.3f} beta={theta[3]:.3f}")
    link_theta, cond_data = export_zoib_link(results_dir, geom, responses_raw, theta,
                                             test_feature_names, test_trait)
    export_zoib_predictions(results_dir, geom, responses_raw, theta, link_theta, cond_data,
                            test_feature_names, keys, extra_cols)


def run_study9_zoib_amp():
    """Study 9 ZOIB fit with the non-stationary amplitude: log sigma(x) = log sigma_0 +
    amp_w . x / coord_sd, amp_w ~ N(0, 0.5^2) fitted alongside (ls, mu_0, sigma, beta).
    Compare its ELBO with the stationary fit in results/study9-fit-zoib/posterior.npz."""
    print("\n===== STUDY 9 (prereg, joint) -- ZOIB linking + non-stationary amplitude =====")
    rdir = 'results/study9-fit-zoib-amp'
    os.makedirs(rdir, exist_ok=True)
    geom = inf.load_geometry()
    responses_raw, _ = inf.load_responses('../../data/study9_prereg.csv', clip_interior=False)
    fit = load_or_fit(
        os.path.join(rdir, 'posterior.npz'),
        lambda: inf.run_vbmc(geom, responses_raw, link='zoib', fit_amp=True,
                             max_evals=MAX_EVALS, seed=SEED))
    theta = _median_theta(fit)
    amp_w = np.median(fit['amp_w_samples'], axis=0)
    lo, hi = np.percentile(fit['amp_w_samples'], [2.5, 97.5], axis=0)
    print(f"  posterior median: ls={theta[0]:.3f} mu_0={theta[1]:+.3f} "
          f"sigma={theta[2]:.3f} beta={theta[3]:.3f}")
    for d in range(len(amp_w)):
        print(f"    amp_w[{d}] = {amp_w[d]:+.3f}  (95% [{lo[d]:+.2f}, {hi[d]:+.2f}])")
    stat_path = 'results/study9-fit-zoib/posterior.npz'
    if os.path.exists(stat_path):
        elbo_stat = json.loads(str(np.load(stat_path, allow_pickle=True)['meta']))['elbo']
        print(f"  ELBO amplitude = {fit['elbo']:.1f}  vs stationary = {elbo_stat:.1f}  "
              f"(diff {fit['elbo'] - elbo_stat:+.1f})")
    link_theta, cond_data = export_zoib_link(rdir, geom, responses_raw, theta,
                                             geom['test_feature_names'], geom['test_trait'], amp_w=amp_w)
    export_zoib_predictions(rdir, geom, responses_raw, theta, link_theta, cond_data,
                            geom['test_feature_names'], list(inf.CONDITIONS), amp_w=amp_w)


def run_study9_zoib():
    print("\n===== STUDY 9 (prereg, joint) -- per-feature ZOIB linking =====")
    geom = inf.load_geometry()
    responses_raw, _ = inf.load_responses('../../data/study9_prereg.csv', clip_interior=False)
    fit_zoib_and_artifacts('results/study9-fit-zoib', geom, responses_raw, list(inf.CONDITIONS),
                           geom['test_feature_names'], geom['test_trait'])


def run_ablations_zoib(rdir, loader):
    """ZOIB counterpart of run_ablations, for BOTH length-scale families (the full
    fitted-ls / flat-ls models are posterior{,-null}.npz from fit_zoib_and_artifacts):
      - litsem      : literal semantics, fitted ls (beta absent)
      - litsem-null : literal semantics, ls pinned at LS_FLAT (beta absent)
      - notransfer  : no utterances; ls is irrelevant (pinned), beta absent --
                      the same model in both families.
    loader() must return (geom, RAW responses_cond) freshly each call."""
    print(f"  --- ZOIB ablations -> {rdir} ---")
    os.makedirs(rdir, exist_ok=True)

    def fit_litsem(ls_fixed):
        g, r = loader()
        g['speaker'] = 'literal'
        fit = inf.run_vbmc(g, r, link='zoib', ls_fixed=ls_fixed, fit_beta=False,
                           max_evals=MAX_EVALS, seed=SEED)
        fit['beta_samples'] = np.full(fit['ls_samples'].shape[0], np.nan)   # no beta in this model
        return fit
    load_or_fit(os.path.join(rdir, 'posterior-litsem.npz'), lambda: fit_litsem(None))
    load_or_fit(os.path.join(rdir, 'posterior-litsem-null.npz'), lambda: fit_litsem(inf.LS_FLAT))

    def fit_notransfer():
        g, r = loader()
        for c in list(g['x_train_cond']):
            g['x_train_cond'][c] = jnp.zeros((0, 2))
            g['u_train_cond'][c] = jnp.zeros(0, dtype=jnp.int32)
            g['train_names_cond'][c] = []
        fit = inf.run_vbmc(g, r, link='zoib', ls_fixed=inf.LS_FLAT, fit_beta=False,
                           max_evals=MAX_EVALS, seed=SEED)
        fit['beta_samples'] = np.full(fit['ls_samples'].shape[0], np.nan)
        return fit
    load_or_fit(os.path.join(rdir, 'posterior-notransfer.npz'), fit_notransfer)


def run_study6_zoib():
    print("\n===== STUDY 6 (prereg) -- per-feature ZOIB linking =====")
    geom, responses_raw = s68.load_study6_prereg(clip_interior=False)
    fit_zoib_and_artifacts('results/study6-fit-zoib', geom, responses_raw, list(geom['x_train_cond']),
                           geom['test_feature_names'], geom['test_trait'])
    run_ablations_zoib('results/study6-ablations-zoib',
                       lambda: s68.load_study6_prereg(clip_interior=False))


# ---------------------------------------------------------------------------
STUDIES = {
    'study6':          run_study6,
    'study8':          run_study8,
    'study9':          run_study9,
    'study9-percond':  run_study9_percond,
    'study9-zoib':     run_study9_zoib,
    'study9-zoib-amp': run_study9_zoib_amp,
    'study6-zoib':     run_study6_zoib,
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
