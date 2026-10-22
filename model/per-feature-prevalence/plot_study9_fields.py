"""Generate GP coherence-field plots for study 9's fitted-ls and null models.

Loads each model's posterior median (ls, mu_0, sigma, beta) from results/study9-fit/
and renders the 2x2 per-condition coherence field over the 2d embedding, saving:
  results/study9-fit/gp-coherence-field.png         (fitted length scale)
  results/study9-fit/gp-coherence-field-null.png    (flat/pinned length scale)
and, when present, the same two plots for the per-feature ZOIB linking fit in
results/study9-fit-zoib/.

Run: /opt/miniconda3/envs/compgenerics/bin/python plot_study9_fields.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import json
import numpy as np
import matplotlib.pyplot as plt

import plot_style
plot_style.use_lexend()

import inference as inf

RESULTS_DIR = os.path.join('results', 'study9-fit')
DATA_CSV = '../../data/study9_prereg.csv'

geom = inf.load_geometry()
J = geom['J']
responses_cond, _ = inf.load_responses(DATA_CSV)
emp_cond = {c: np.array(responses_cond[c]).mean(0) for c in inf.CONDITIONS}
N_cond   = {c: int(responses_cond[c].shape[0]) for c in inf.CONDITIONS}

# shared 2d grid over all feature points
all_x = np.vstack([np.array(geom['x_test'])] + [np.array(geom['x_train_cond'][c]) for c in inf.CONDITIONS])
pad = 0.08
(x_lo, y_lo), (x_hi, y_hi) = all_x.min(0) - pad, all_x.max(0) + pad
gx, gy = np.meshgrid(np.linspace(x_lo, x_hi, 80), np.linspace(y_lo, y_hi, 80))
X_grid = np.column_stack([gx.ravel(), gy.ravel()])
x_test_np = np.array(geom['x_test'])


def median_theta(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return (float(np.median(z['ls_samples'])), float(np.median(z['mu0_samples'])),
            float(np.median(z['sigma_samples'])), float(np.median(z['beta_samples'])))


def plot_field(theta, title_tag, out_png, fit_label='study9-fit'):
    LS, MU, SIG, BETA = theta
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12))
    for ax, c in zip(axes.ravel(), inf.CONDITIONS):
        field    = inf.gp_coherence_field(geom, c, LS, MU, SIG, X_grid,
                                          beta_speaker=BETA).reshape(gx.shape)
        pz1_test = inf.gp_coherence_field(geom, c, LS, MU, SIG, x_test_np, beta_speaker=BETA)
        cs = ax.contourf(gx, gy, field, levels=20, cmap='Blues')
        cb = fig.colorbar(cs, ax=ax); cb.set_label('GP P(kind-linked)')

        x_tr = np.array(geom['x_train_cond'][c])
        ax.scatter(x_tr[:, 0], x_tr[:, 1], s=60, color='#16305e', edgecolor='k',
                   zorder=3, label='train (kind-linked)')
        for (xx, yy), nm in zip(x_tr, geom['train_names_cond'][c]):
            ax.annotate(nm[:12], (xx, yy), fontsize=4, xytext=(2, 3), textcoords='offset points')
        ax.scatter(x_test_np[:, 0], x_test_np[:, 1], s=80, c=emp_cond[c], cmap='Blues', norm=cs.norm,
                   edgecolor='k', linewidth=1.2, zorder=3, label='test (mean response)')

        # a flat (null) field is ~constant across space, so its field@test values
        # are numerically-tiny variations; correlating them against ratings is
        # meaningless -> flag it rather than report a spurious r.
        if np.ptp(pz1_test) > 1e-3:
            r_txt = f"r(field @ test, mean rating) = {np.corrcoef(pz1_test, emp_cond[c])[0, 1]:+.2f}"
        else:
            r_txt = "field ~ constant (r undefined)"
        ax.set_title(f"{c.capitalize()}  (n={N_cond[c]})\n{r_txt}")
        ax.set_xlabel('Embedding Dim 1'); ax.set_ylabel('Embedding Dim 2')
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f"Posterior over Coherence in Embedding Space -- {fit_label} ({title_tag})\n"
                 f"ls={LS:.3f}, mu_0={MU:+.3f}, sigma={SIG:.3f}, beta={BETA:.3f}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_png}  (ls={LS:.3f}, beta={BETA:.3f})")


if __name__ == '__main__':
    fitted = median_theta(os.path.join(RESULTS_DIR, 'fitted-feature-shapes.npz'))
    plot_field(fitted, 'fitted length scale', os.path.join(RESULTS_DIR, 'gp-coherence-field.png'))

    null = median_theta(os.path.join(RESULTS_DIR, 'fitted-feature-shapes-null.npz'))
    plot_field(null, 'flat length scale (null)', os.path.join(RESULTS_DIR, 'gp-coherence-field-null.png'))

    # per-feature ZOIB linking fit (run_parameter_estimation.py study9-zoib), if present
    ZOIB_DIR = os.path.join('results', 'study9-fit-zoib')
    for npz, tag, png in [('posterior.npz', 'ZOIB linking, fitted length scale', 'gp-coherence-field.png'),
                          ('posterior-null.npz', 'ZOIB linking, flat length scale (null)', 'gp-coherence-field-null.png')]:
        if os.path.exists(os.path.join(ZOIB_DIR, npz)):
            plot_field(median_theta(os.path.join(ZOIB_DIR, npz)), tag, os.path.join(ZOIB_DIR, png),
                       fit_label='study9-fit-zoib')
    print("done.")
