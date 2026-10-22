"""Profile-likelihood map over (length_scale, beta) for the study-9 prereg joint fit.

Checks that the VBMC fit (high ls, low beta) is the global optimum and that no
better solution hides in the low-ls / high-beta corner. Two stages:

  1. GRID: on a log-spaced (ls, beta) grid spanning the VBMC box, maximize the
     summed log Z (linking shapes profiled, as in VBMC) over (mu_0, log sigma)
     with Nelder-Mead from two starts (the old and new posterior medians' values).
  2. REFINE: full 4-D Nelder-Mead over (log ls, mu_0, log sigma, log beta) from
     (a) the best grid cell overall, (b) the best grid cell with ls < 0.4 and
     beta > 5, (c) the old pilot-fit posterior median.

The objective is jagged (the per-feature shape re-fit jumps between solutions),
hence derivative-free search. Maximizes the likelihood (not log joint); the log
prior is recorded alongside so either can be read off.

Saves results/study9-fit/profile-ls-beta.npz (read by study9-old-vs-new-fit.ipynb).

Run: /opt/miniconda3/envs/compgenerics/bin/python profile_study9_ls_beta.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
# one thread per worker; parallelism comes from the process pool. ALL of these are
# needed (set before numpy/jax import): with only XLA limited, the BLAS pools
# oversubscribe the cores and each eval slows ~40x (0.7s -> 33s).
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import time
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize

DATA_CSV = '../../data/study9_prereg.csv'
OUT = os.path.join('results', 'study9-fit', 'profile-ls-beta.npz')
N_WORKERS = int(os.environ.get("N_WORKERS", "10"))

LS_GRID   = np.geomspace(0.05, 4.0, 13)
BETA_GRID = np.geomspace(0.25, 30.0, 13)
# (mu_0, log sigma) starts: old and new posterior medians
STARTS_2D = [(-0.397, np.log(0.782)), (0.006, np.log(0.526))]
MU_BOUNDS, LOGSIG_BOUNDS = (-4.0, 4.0), (np.log(0.05), np.log(8.0))
LOGLS_BOUNDS, LOGBETA_BOUNDS = (np.log(0.03), np.log(4.0)), (np.log(0.2), np.log(30.0))
THETA_OLD = (0.20526066465922216, -0.39730651435898356, 0.7815199274312206, 16.035191727776876)

_G = {}


def _init():
    import inference as inf
    _G['inf'] = inf
    _G['geom'] = inf.load_geometry()
    _G['resp'], _ = inf.load_responses(DATA_CSV)


def loglik(ls, mu0, sig, beta):
    try:
        ll = _G['inf'].total_log_lik_free_shapes(_G['geom'], _G['resp'], float(ls), float(mu0),
                                                 float(sig), beta_speaker=float(beta))
    except Exception:
        return -np.inf
    return ll if np.isfinite(ll) else -np.inf


def log_prior(ls, mu0, sig, beta):
    inf = _G['inf']
    return (inf.log_prior_ls(np.log(ls)) + inf.log_prior_mu(mu0)
            + inf.log_prior_sigma(np.log(sig)) + inf.log_prior_beta(np.log(beta)))


def profile_cell(args):
    """Max over (mu_0, log sigma) at fixed (ls, beta)."""
    i, k, ls, beta = args
    best = (-np.inf, None)
    n_eval = 0
    for x0 in STARTS_2D:
        res = minimize(lambda x: -loglik(ls, x[0], np.exp(x[1]), beta), np.array(x0),
                       method='Nelder-Mead', bounds=[MU_BOUNDS, LOGSIG_BOUNDS],
                       options={'maxfev': 45, 'xatol': 1e-2, 'fatol': 0.05,
                                'initial_simplex': np.array(x0) + np.array([[0, 0], [0.4, 0], [0, 0.4]])})
        n_eval += res.nfev
        if -res.fun > best[0]:
            best = (-res.fun, res.x)
    mu0, sig = best[1][0], float(np.exp(best[1][1]))
    return i, k, best[0], mu0, sig, log_prior(ls, mu0, sig, beta), n_eval


def refine(args):
    """Full 4-D Nelder-Mead over (log ls, mu_0, log sigma, log beta)."""
    tag, theta0 = args
    ls, mu0, sig, beta = theta0
    x0 = np.array([np.log(ls), mu0, np.log(sig), np.log(beta)])
    f = lambda x: -loglik(np.exp(x[0]), x[1], np.exp(x[2]), np.exp(x[3]))
    res = minimize(f, x0, method='Nelder-Mead',
                   bounds=[LOGLS_BOUNDS, MU_BOUNDS, LOGSIG_BOUNDS, LOGBETA_BOUNDS],
                   options={'maxfev': 400, 'xatol': 1e-3, 'fatol': 0.02,
                            'initial_simplex': x0 + np.vstack([np.zeros(4), 0.3 * np.eye(4)])})
    th = (float(np.exp(res.x[0])), float(res.x[1]), float(np.exp(res.x[2])), float(np.exp(res.x[3])))
    return tag, theta0, th, -res.fun, log_prior(*th), res.nfev


if __name__ == '__main__':
    t0 = time.time()
    ctx = mp.get_context('spawn')
    jobs = [(i, k, ls, b) for i, ls in enumerate(LS_GRID) for k, b in enumerate(BETA_GRID)]
    LL = np.full((len(LS_GRID), len(BETA_GRID)), np.nan)
    MU, SIG, LP = LL.copy(), LL.copy(), LL.copy()
    with ctx.Pool(N_WORKERS, initializer=_init) as pool:
        for n, (i, k, ll, mu0, sig, lp, ne) in enumerate(pool.imap_unordered(profile_cell, jobs), 1):
            LL[i, k], MU[i, k], SIG[i, k], LP[i, k] = ll, mu0, sig, lp
            print(f"[{n:3d}/{len(jobs)}] ls={LS_GRID[i]:.3f} beta={BETA_GRID[k]:6.2f}  "
                  f"ll={ll:8.2f}  mu0={mu0:+.2f} sig={sig:.2f}  ({ne} evals, {time.time()-t0:.0f}s)", flush=True)

        i_best, k_best = np.unravel_index(np.nanargmax(LL), LL.shape)
        corner = (LS_GRID[:, None] < 0.4) & (BETA_GRID[None, :] > 5)
        i_c, k_c = np.unravel_index(np.nanargmax(np.where(corner, LL, -np.inf)), LL.shape)
        starts = [('grid best', (LS_GRID[i_best], MU[i_best, k_best], SIG[i_best, k_best], BETA_GRID[k_best])),
                  ('low-ls/high-beta best', (LS_GRID[i_c], MU[i_c, k_c], SIG[i_c, k_c], BETA_GRID[k_c])),
                  ('old pilot fit', THETA_OLD)]
        refined = []
        for tag, th0, th, ll, lp, ne in pool.imap(refine, starts):
            refined.append((tag, th0, th, ll, lp))
            print(f"refine [{tag}] from ls={th0[0]:.3f} beta={th0[3]:.2f} -> ls={th[0]:.3f} mu0={th[1]:+.3f} "
                  f"sig={th[2]:.3f} beta={th[3]:.3f}  ll={ll:.2f}  ({ne} evals)", flush=True)

    np.savez_compressed(OUT, ls_grid=LS_GRID, beta_grid=BETA_GRID, ll=LL, mu0=MU, sigma=SIG, log_prior=LP,
                        refine_tags=np.array([r[0] for r in refined]),
                        refine_start=np.array([r[1] for r in refined]),
                        refine_theta=np.array([r[2] for r in refined]),
                        refine_ll=np.array([r[3] for r in refined]),
                        refine_log_prior=np.array([r[4] for r in refined]))
    print(f"saved {OUT}  ({time.time()-t0:.0f}s total)")
