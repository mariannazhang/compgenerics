"""Fit the literal-listener lesion of the study 8 model and cache its posterior.

Specifics carry zero evidence: every group's utterances are reduced to its
generic-only set. STRUCTURE-MATCHED by default (the original 403-group structure
is kept, so summed log Z is comparable to the full fit) -> posterior-literal-matched.npz.
Pass 'merged' as an argument for the shared-field variant (identical generic sets
pooled; log Z NOT comparable) -> posterior-literal.npz. Same spec as the full
study-8 fit otherwise (Laplace + 4-D VBMC, fitted linking shapes, fitted beta,
widened beta bound, 2d). Consumed by model-study8-collapsed.ipynb.
"""
import os
import sys
import json

import numpy as np

import inference as inf
import studies68 as s68

MAX_EVALS = 400
RESULTS_DIR = os.path.join('results', 'study8-ablations')
os.makedirs(RESULTS_DIR, exist_ok=True)
inf.set_embed('2d')
inf.UB_4[3], inf.PUB_4[3] = np.log(1000.0), np.log(300.0)

merge = len(sys.argv) > 1 and sys.argv[1] == 'merged'
path = os.path.join(RESULTS_DIR, 'posterior-literal.npz' if merge
                    else 'posterior-literal-matched.npz')
if os.path.exists(path):
    print(f'cache exists: {path} -- delete to refit')
    sys.exit(0)

geom, responses_cond, groups_df = s68.load_study8()
geom_lit, responses_lit, _ = s68.literal_study8(responses_cond, groups_df, merge=merge)

fit = inf.run_vbmc_free_shapes_beta(geom_lit, responses_lit, max_evals=MAX_EVALS)
np.savez_compressed(
    path,
    **{k: fit[k] for k in ('ls_samples', 'mu0_samples', 'sigma_samples', 'beta_samples')},
    meta=json.dumps({k: fit[k] for k in ('elbo', 'func_count', 'convergence_status', 'runtime_s')}))
print(f'saved {path}')
