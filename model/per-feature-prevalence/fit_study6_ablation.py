"""Fit one study-6 ablation model and cache its posterior. Usage:

    python fit_study6_ablation.py literal     # specifics carry zero evidence (dropped;
                                              # RSA speaker kept for generics, beta fitted)
    python fit_study6_ablation.py literalsem  # ALIGNED literal listener: ALL utterances
                                              # kept, likelihood = literal truth-conditioning
                                              # with 5% misspeak (speaker='literal', the
                                              # gorgo model's semantics; specifics vacuous
                                              # by semantics, beta absent -> 3-D fit)
    python fit_study6_ablation.py null        # distance-blind: ls pinned at LS_FLAT
    python fit_study6_ablation.py notransfer  # no category transfer: ALL conditions
                                              # lose their utterances (the analog of the
                                              # collaborator's 'individuals and features
                                              # only' base model; predictions are
                                              # condition-invariant). ls pinned, beta
                                              # dropped (no likelihood contribution) ->
                                              # k=2 (mu_0, sigma), shapes still profiled.

Append 'prereg' as a second argument to fit the preregistered replication data
(load_study6_prereg) instead; caches then go to results/study6prereg-ablations/.

literal/null are 'fitted shapes + fitted beta' fits with the same widened beta bound as
the main study-6/8 runs. Caches to results/study6-ablations/posterior-<which>.npz in the
same npz format as the main notebooks' load_or_fit. Consumed by
model-study6-ablations.ipynb (or its prereg clone).
"""
import os
import sys
import json

import numpy as np

import inference as inf
import studies68 as s68
import jax.numpy as jnp

which = sys.argv[1]
assert which in ('literal', 'literalsem', 'null', 'notransfer')
prereg = len(sys.argv) > 2 and sys.argv[2] == 'prereg'

MAX_EVALS = 400
RESULTS_DIR = os.path.join('results', 'study6prereg-ablations' if prereg else 'study6-ablations')
os.makedirs(RESULTS_DIR, exist_ok=True)
inf.set_embed('2d')
# widened beta coordinate (same spec as the main study-6/8 fits); the 3-D
# fixed-ls box is derived from the 4-D one at import time, so widen both
inf.UB_4[3], inf.PUB_4[3] = np.log(1000.0), np.log(300.0)
inf.UB_3B[2], inf.PUB_3B[2] = np.log(1000.0), np.log(300.0)

path = os.path.join(RESULTS_DIR, f'posterior-{which}.npz')
if os.path.exists(path):
    print(f'cache exists: {path} -- delete to refit')
    sys.exit(0)

geom, responses_cond = s68.load_study6_prereg() if prereg else s68.load_study6()
if which == 'literal':
    # specifics carry zero evidence: the specific group keeps its ratings but
    # loses its utterances (geometry identical to baseline)
    geom['x_train_cond']['specific'] = jnp.zeros((0, 2))
    geom['u_train_cond']['specific'] = jnp.zeros(0, dtype=jnp.int32)
    geom['train_names_cond']['specific'] = []
    fit = inf.run_vbmc_free_shapes_beta(geom, responses_cond, max_evals=MAX_EVALS)
elif which == 'literalsem':
    # aligned literal listener: all utterances kept, truth-conditional likelihood
    # (specifics are vacuously true -> zero evidence, by semantics not deletion);
    # no speaker -> no beta -> 3-D fit over (ls, mu_0, sigma)
    geom['speaker'] = 'literal'
    fit = inf.run_vbmc_free_shapes(geom, responses_cond, max_evals=MAX_EVALS)
    fit['beta_samples'] = np.full(fit['ls_samples'].shape[0], np.nan)   # no beta in this model
elif which == 'notransfer':
    # no channel from utterances to test features: every condition's utterance
    # set is emptied, so predictions cannot differ by condition
    for c in list(geom['x_train_cond']):
        geom['x_train_cond'][c] = jnp.zeros((0, 2))
        geom['u_train_cond'][c] = jnp.zeros(0, dtype=jnp.int32)
        geom['train_names_cond'][c] = []
    fit = inf.run_vbmc_free_shapes_fixed_ls(geom, responses_cond,
                                            ls_fixed=inf.LS_FLAT, max_evals=MAX_EVALS)
    # no utterances -> beta never enters the likelihood; record the design value
    fit['beta_samples'] = np.full(fit['ls_samples'].shape[0], inf.BETA_SPEAKER)
else:
    fit = inf.run_vbmc_free_shapes_fixed_ls_beta(geom, responses_cond,
                                                 ls_fixed=inf.LS_FLAT, max_evals=MAX_EVALS)

np.savez_compressed(
    path,
    **{k: fit[k] for k in ('ls_samples', 'mu0_samples', 'sigma_samples', 'beta_samples')},
    meta=json.dumps({k: fit[k] for k in ('elbo', 'func_count', 'convergence_status', 'runtime_s')}))
print(f'saved {path}')
