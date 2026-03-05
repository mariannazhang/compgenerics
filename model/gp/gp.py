import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm_notebook
from typing import Tuple, Dict, Optional

#############################
##### RANDOM SEED SETUP #####
#############################

random_seed = 100
key = jax.random.PRNGKey(random_seed)
keys = jax.random.split(key, 15)

###############################
##### MULTIVARIATE NORMAL #####
###############################

def manual_mvn_logpdf(y, mu, C):
    yp = y - mu
    L = jsp.linalg.cholesky(C, lower=True)
    precision_weighted_diff = jsp.linalg.cho_solve((L, True), yp)
    alpha = jnp.dot(yp, precision_weighted_diff, precision='highest')
    beta = jnp.sum(jnp.log(jnp.diag(L)))
    n = len(mu)
    constant_term = n / 2. * jnp.log(2 * jnp.pi)
    return -0.5 * alpha - beta - constant_term

##############################
##### KERNEL DEFINITIONS #####
##############################

# 1D RBF — used in learner/teacher GP regression
def _rbf_kernel(x1, x2, params):
    """params = (sigma, lengthscale, noise)"""
    sigma, lengthscale = params[0], params[1]
    return sigma**2 * jnp.exp(-jnp.sum((x1 - x2)**2) / (2 * lengthscale**2))

RBF_PARAM_NAMES = ("sigma", "lengthscale", "noise")
RBF_PARAM_RANGES = {
    "sigma": (0.5, 3.5),
    "lengthscale": (1.0, 3.0),
    "noise": (-4, -3),  # log10 scale
}
RBF_GRID_SIZES = {"sigma": 6, "lengthscale": 4, "noise": 4}
_RBF_NOISE_IDX = 2

# 2D RBF with ARD — used in Bernoulli GP classification
def _rbf_kernel_2d_ard(x1, x2, params):
    """params = (sigma, lengthscale_x, lengthscale_y)"""
    sigma = params[0]
    lengthscales = jnp.array([params[1], params[2]])
    diff = (x1 - x2) / lengthscales
    return sigma**2 * jnp.exp(-0.5 * jnp.sum(diff**2))

RBF_2D_ARD_PARAM_NAMES = ("sigma", "lengthscale_x", "lengthscale_y")
RBF_2D_ARD_PARAM_RANGES = {
    "sigma": (0.5, 2.0),
    "lengthscale_x": (0.05, 0.3),
    "lengthscale_y": (0.05, 0.3),
}
RBF_2D_ARD_GRID_SIZES = {"sigma": 4, "lengthscale_x": 4, "lengthscale_y": 4}


class _KernelInfo:
    """Thin struct for notebooks that call gp.get_kernel(name).param_names."""
    def __init__(self, name, param_names, param_ranges, grid_sizes):
        self.name = name
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.grid_sizes = grid_sizes

def get_kernel(kernel_name: str) -> _KernelInfo:
    if kernel_name == "rbf":
        return _KernelInfo("rbf", RBF_PARAM_NAMES, RBF_PARAM_RANGES, RBF_GRID_SIZES)
    elif kernel_name == "rbf_2d_ard":
        return _KernelInfo("rbf_2d_ard", RBF_2D_ARD_PARAM_NAMES, RBF_2D_ARD_PARAM_RANGES, RBF_2D_ARD_GRID_SIZES)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name!r}. Supported: 'rbf', 'rbf_2d_ard'.")

######################################
##### KERNEL VECTORIZATION ###########
######################################

def _create_vectorized_kernel(kernel_fn):
    return jax.vmap(
        jax.vmap(kernel_fn, in_axes=(0, None, None)),
        in_axes=(None, 0, None)
    )

def _create_vectorized_kernel_hp(kernel_fn):
    return jax.vmap(_create_vectorized_kernel(kernel_fn), in_axes=(None, None, 0))

_rbf_kernel_vec    = _create_vectorized_kernel(_rbf_kernel)
_rbf_kernel_vec_hp = _create_vectorized_kernel_hp(_rbf_kernel)
_rbf_2d_ard_kernel_vec = _create_vectorized_kernel(_rbf_kernel_2d_ard)

def get_vectorized_kernel(kernel_name: str):
    if kernel_name == "rbf_2d_ard":
        return _rbf_2d_ard_kernel_vec
    return _rbf_kernel_vec

##################################
##### HYPERPARAMETER GENERATION ##
##################################

def create_grid_hparams(grid_sizes=None, ranges=None):
    """Grid of (sigma, lengthscale, noise) for 1D RBF."""
    sizes = {**RBF_GRID_SIZES, **(grid_sizes or {})}
    param_ranges = {**RBF_PARAM_RANGES, **(ranges or {})}

    sigma_arr = jnp.linspace(*param_ranges["sigma"], sizes["sigma"])
    ls_arr    = jnp.exp(jnp.linspace(*param_ranges["lengthscale"], sizes["lengthscale"]))
    noise_arr = 10 ** jnp.linspace(*param_ranges["noise"], sizes["noise"])

    grids = jnp.meshgrid(sigma_arr, ls_arr, noise_arr, indexing='ij')
    return jnp.stack([g.flatten() for g in grids], axis=1)


def create_rand_hparams(num_particles: int, random_seed: int, ranges=None):
    """Random (sigma, lengthscale, noise) for 1D RBF."""
    param_ranges = {**RBF_PARAM_RANGES, **(ranges or {})}
    rng = jax.random.PRNGKey(random_seed)
    k0, k1, k2 = jax.random.split(rng, 3)

    sigma_arr = jax.random.uniform(k0, (num_particles,), minval=param_ranges["sigma"][0],      maxval=param_ranges["sigma"][1])
    ls_arr    = jax.random.uniform(k1, (num_particles,), minval=param_ranges["lengthscale"][0], maxval=param_ranges["lengthscale"][1])
    noise_arr = 10 ** jax.random.uniform(k2, (num_particles,), minval=param_ranges["noise"][0], maxval=param_ranges["noise"][1])

    return jnp.stack([sigma_arr, ls_arr, noise_arr], axis=1)


def create_grid_hparams_2d_ard(grid_sizes=None, ranges=None):
    """Grid of (sigma, lengthscale_x, lengthscale_y) for 2D ARD RBF."""
    sizes = {**RBF_2D_ARD_GRID_SIZES, **(grid_sizes or {})}
    param_ranges = {**RBF_2D_ARD_PARAM_RANGES, **(ranges or {})}

    sigma_arr = jnp.linspace(*param_ranges["sigma"],        sizes["sigma"])
    lx_arr    = jnp.linspace(*param_ranges["lengthscale_x"], sizes["lengthscale_x"])
    ly_arr    = jnp.linspace(*param_ranges["lengthscale_y"], sizes["lengthscale_y"])

    grids = jnp.meshgrid(sigma_arr, lx_arr, ly_arr, indexing='ij')
    return jnp.stack([g.flatten() for g in grids], axis=1)

######################
##### LEARNER GP #####
######################

def learner_gp(teach_inputs, teach_outputs, target_in, learn_hparams, just_var=True):
    """GP posterior for learner given teaching points (1D RBF kernel)."""
    teach_in_K = _rbf_kernel_vec_hp(teach_inputs, teach_inputs, learn_hparams)
    noise = learn_hparams[:, _RBF_NOISE_IDX]
    noise_mat = noise[:, None, None] * jnp.eye(len(teach_inputs))[None, :, :]
    teach_in_K = teach_in_K + noise_mat

    inv_teach_in_K  = jnp.linalg.inv(teach_in_K)
    teach_target_K  = _rbf_kernel_vec_hp(teach_inputs, target_in, learn_hparams)
    target_target_K = _rbf_kernel_vec_hp(target_in, target_in, learn_hparams)

    target_mean = jnp.einsum('kdn,knm,m->kd', teach_target_K, inv_teach_in_K, teach_outputs)

    if just_var:
        target_vars = (
            jnp.diagonal(target_target_K, axis1=1, axis2=2) -
            jnp.einsum('kdn,knm,kdm->kd', teach_target_K, inv_teach_in_K, teach_target_K)
        )
        return target_mean, target_vars
    else:
        target_cov = (
            target_target_K -
            jnp.einsum('kdn,knm,kem->kde', teach_target_K, inv_teach_in_K, teach_target_K)
        )
        return target_mean, target_cov


def learner_gp_mix(teach_inputs, teach_outputs, target_in, learn_hparams, just_var=True):
    """Mixture-weighted GP posterior."""
    lhp_logprob = jax.nn.softmax(learner_logpost_fn_vec(learn_hparams, teach_inputs, teach_outputs))

    if just_var:
        means, vars = learner_gp(teach_inputs, teach_outputs, target_in, learn_hparams, just_var=True)
        weighted_vars = jnp.average(vars, weights=lhp_logprob, axis=0)
    else:
        means, covs = learner_gp(teach_inputs, teach_outputs, target_in, learn_hparams, just_var=False)
        weighted_vars = jnp.diag(jnp.average(covs, weights=lhp_logprob, axis=0))

    weighted_means = jnp.average(means, weights=lhp_logprob, axis=0)

    assert weighted_means.ndim == 1
    assert weighted_vars.ndim == 1
    return weighted_means, weighted_vars


def learner_gp_mix_bounded(teach_inputs, teach_outputs, target_in, learn_hparams,
                            just_var=True, y_min=-1, y_max=5, oob_penalty=100.0):
    """Mixture-weighted GP posterior with out-of-bounds penalty."""
    means, vars = learner_gp(teach_inputs, teach_outputs, target_in, learn_hparams, just_var=True)
    lhplogposts = learner_logpost_fn_vec(learn_hparams, teach_inputs, teach_outputs)

    mean_oob = (
        jnp.maximum(0, means - y_max)**2 +
        jnp.maximum(0, y_min - means)**2
    ).sum(axis=1)

    lhp_logprob = jax.nn.softmax(lhplogposts - oob_penalty * mean_oob)
    return jnp.average(means, weights=lhp_logprob, axis=0), jnp.average(vars, weights=lhp_logprob, axis=0)

###########################
##### LEARNER UTILITY #####
###########################

def hparam_log_prior(hyperparams):
    """Log prior for RBF hyperparameters."""
    sigma_prior  = jsp.stats.expon.logpdf(hyperparams[0], scale=1)
    length_prior = jsp.stats.gamma.logpdf(hyperparams[1], a=2.5, scale=0.7)
    return sigma_prior + length_prior


def learner_logpost_fn(learn_hparams, teach_in, teach_out):
    """Log posterior for learner hyperparameters (1D RBF)."""
    prior_cov  = _rbf_kernel_vec(teach_in, teach_in, learn_hparams)
    prior_cov  = prior_cov + (learn_hparams[_RBF_NOISE_IDX] * jnp.eye(prior_cov.shape[0]))
    prior_mean = jnp.mean(teach_out).repeat(len(teach_in))
    return hparam_log_prior(learn_hparams) + manual_mvn_logpdf(teach_out, prior_mean, prior_cov)


def learner_logpost_fn_vec(learn_hparams, teach_in, teach_out):
    """Vectorized log posterior over hyperparameters."""
    return jax.vmap(lambda hp: learner_logpost_fn(hp, teach_in, teach_out))(learn_hparams)

########################
##### LEARNER SVGD #####
########################

def svgd_kernel(x1, x2, hyperparams):
    sigma, lengthscale = hyperparams[0], hyperparams[1]
    return sigma**2 * jnp.exp(-jnp.sum((x1 - x2)**2) / (2 * lengthscale**2))


def svgd_kernel_grad_closed(x1, x2, hyperparams):
    sigma, lengthscale = hyperparams[0], hyperparams[1]
    return 1 / lengthscale**2 * (x2 - x1) * svgd_kernel(x1, x2, hyperparams)


svgd_kernel_vec = jax.vmap(jax.vmap(svgd_kernel, in_axes=(0, None, None)), in_axes=(None, 0, None))
svgd_kernel_grad_closed_vec = jax.vmap(svgd_kernel_grad_closed, in_axes=(0, None, None))


def median_heuristic_bandwidth(particles):
    gram_matrix = jnp.matmul(particles, particles.T)
    norm_x = jnp.sum(jnp.square(particles), axis=1, keepdims=True)
    pdist = norm_x + norm_x.T - 2 * gram_matrix

    mask = jnp.triu(jnp.ones_like(pdist), k=1)
    distances = pdist[mask == 1]
    median_dist = jnp.median(distances) if distances.size > 0 else 1.0
    h = median_dist / (2 * jnp.log(particles.shape[0] + 1))
    return jnp.array([1.0, jnp.sqrt(h)])


def svgd_step(learner_hparams, teach_in, teach_out, svgd_hparams):
    svgd_kernel_vec_xj         = jax.vmap(svgd_kernel, in_axes=(0, None, None))
    svgd_kernel_vec_xj_vec_xi  = jax.vmap(svgd_kernel_vec_xj, in_axes=(None, 0, None))
    svgd_kernel_grad_xj        = jax.grad(svgd_kernel, argnums=0)
    svgd_kernel_grad_xj_vec_xj = jax.vmap(svgd_kernel_grad_xj, in_axes=(None, 0, None))
    svgd_kernel_grad_xj_vec_xj_vec_xi = jax.vmap(svgd_kernel_grad_xj_vec_xj, in_axes=(0, None, None))

    learner_logpost_fn_grad_vec = jax.vmap(
        jax.grad(lambda hp, ti, to: learner_logpost_fn(hp, ti, to), argnums=0),
        in_axes=(0, None, None)
    )

    xj_xi_sim        = svgd_kernel_vec_xj_vec_xi(learner_hparams, learner_hparams, svgd_hparams)
    xj_logpost_grad  = learner_logpost_fn_grad_vec(learner_hparams, teach_in, teach_out)
    svgd_update_logpost = jnp.einsum("ji,jf->if", xj_xi_sim, xj_logpost_grad) / len(learner_hparams)
    svgd_update_logpost = jnp.clip(svgd_update_logpost, -10, 10)

    svgd_update_kernel = svgd_kernel_grad_xj_vec_xj_vec_xi(learner_hparams, learner_hparams, svgd_hparams).sum(0)
    svgd_update_kernel /= len(learner_hparams)

    return svgd_update_logpost + svgd_update_kernel


def learner_svgd(params, iters, teach_in, teach_out, print_increment=jnp.inf):
    learning_rate = jnp.array([0.001])
    for i in jnp.arange(iters):
        svgd_hparams = median_heuristic_bandwidth(params)
        update = svgd_step(params, teach_in, teach_out, svgd_hparams)
        params = params + learning_rate * update
        if i % print_increment == 0 and i != 0:
            print(f'iter {i} params\n', params)
    return params

###########################
##### TEACHER UTILITY #####
###########################

def teacher_loss_fn(teach_in, teach_out, target_in, target_out, learn_hparams, lamb=0.5, bounded=False):
    """Teacher loss based on learner's posterior. High lambda → more weight on mean, low lambda → more weight on variance."""
    if bounded:
        target_mean, target_vars = learner_gp_mix_bounded(teach_in, teach_out, target_in, learn_hparams, just_var=True)
    else:
        target_mean, target_vars = learner_gp_mix(teach_in, teach_out, target_in, learn_hparams, just_var=True)

    squared_error = (target_mean - target_out)**2
    return jnp.mean(lamb * squared_error + (1 - lamb) * target_vars)


def teacher_loss_fn_vec(teach_in_mat, teach_out_mat, target_in, target_out, learner_enum_hparams):
    return jax.vmap(
        lambda ti, to: teacher_loss_fn(ti, to, target_in, target_out, learner_enum_hparams)
    )(teach_in_mat, teach_out_mat)


def update_teaching_points(teach_points, target_x, target_y, learner_hparams,
                            learning_rate=0.01, num_steps=50, print_intermediate=False):
    def loss_fn(points):
        return teacher_loss_fn(points[:, 0], points[:, 1], target_x, target_y, learner_hparams)

    grad_loss_fn = jax.grad(loss_fn)
    current_points = teach_points

    for i in range(num_steps):
        grads = jnp.clip(grad_loss_fn(current_points), -10, 10)
        current_points = current_points - learning_rate * grads
        current_points = current_points.at[:, 0].set(
            jnp.clip(current_points[:, 0], jnp.min(target_x), jnp.max(target_x))
        )
        if i % 20 == 0 and print_intermediate:
            print(f"Step {i}, Loss: {loss_fn(current_points):.6f}")

    return current_points

################################
##### TEACHER OPTIMIZATION #####
################################

def optimize_teaching_points_for_target_function(
    target_x,
    target_y,
    num_teach_pts=5,
    num_iterations=150,
    num_particles=1,
    random_particles=False,
    learner_svgd_steps=0,
    teacher_initial_points="random",
    teacher_gd_steps=1,
    teacher_gd_learning_rate=0.1,
    random_seed=101,
    verbose=False
):
    key = jax.random.PRNGKey(random_seed)
    keys = jax.random.split(key, 10)

    d_min = jnp.min(target_x)
    d_max = jnp.max(target_x)
    section_width = (d_max - d_min) / num_teach_pts

    if isinstance(teacher_initial_points, str) and teacher_initial_points == "random":
        teach_x = jnp.array([
            d_min + i * section_width + jax.random.uniform(keys[i], (1,)).item() * section_width
            for i in range(num_teach_pts)
        ])
        interp_indices = jnp.clip(jnp.searchsorted(target_x, teach_x), 0, len(target_y) - 1)
        teach_y = target_y[interp_indices]
        teach_points = jnp.stack([teach_x, teach_y], axis=1)
    else:
        teach_points = teacher_initial_points

    initial_teach_points = teach_points.copy()

    if random_particles:
        learner_init_hparams = create_rand_hparams(num_particles, random_seed)
    else:
        learner_init_hparams = create_grid_hparams()

    learner_init_hparams = learner_svgd(
        learner_init_hparams, 10,
        teach_points[:, 0], teach_points[:, 1],
        print_increment=learner_svgd_steps if verbose else jnp.inf
    )

    initial_teacher_loss = teacher_loss_fn(
        initial_teach_points[:, 0], initial_teach_points[:, 1],
        target_x, target_y, learner_init_hparams
    )
    teacher_loss_history = jnp.zeros(num_iterations)
    learner_particles = learner_init_hparams

    pbar = tqdm_notebook(range(num_iterations), desc="Optimization Progress")
    for iteration in pbar:
        if verbose:
            print(f"\n=== Iteration {iteration+1}/{num_iterations} ===")

        if learner_svgd_steps > 0 and iteration > 0:
            if verbose:
                print("Running SVGD to optimize particles...")
            learner_particles = learner_svgd(
                learner_particles, learner_svgd_steps,
                teach_points[:, 0], teach_points[:, 1],
                print_increment=learner_svgd_steps if verbose else jnp.inf
            )

        current_teacher_loss = teacher_loss_fn(
            teach_points[:, 0], teach_points[:, 1],
            target_x, target_y, learner_particles
        )
        teacher_loss_history = teacher_loss_history.at[iteration].set(current_teacher_loss)
        pbar.set_postfix({"loss": f"{current_teacher_loss:.6f}"})

        if verbose:
            print("Updating teaching points using gradient descent...")

        teach_points = update_teaching_points(
            teach_points, target_x, target_y, learner_particles,
            learning_rate=teacher_gd_learning_rate,
            num_steps=teacher_gd_steps,
            print_intermediate=verbose
        )
        teach_points = teach_points[jnp.argsort(teach_points[:, 0])]

        if verbose:
            print(f"Teaching points after iteration {iteration+1}:")
            for i, (x, y) in enumerate(teach_points):
                print(f"  Point {i+1}: x={x:.4f}, y={y:.4f}")

    return {
        "optimized_teach_points": teach_points,
        "optimized_learner_hparams": learner_particles,
        "target_x": target_x,
        "target_y": target_y,
        "teacher_loss_history": teacher_loss_history,
        "initial_teach_points": initial_teach_points,
        "initial_learner_hparams": learner_init_hparams,
        "initial_teacher_loss": initial_teacher_loss,
    }

############################################
##### VISUALIZING TEACHER OPTIMIZATION #####
############################################

def visualize_teaching_optimization(results):
    optimized_teach_points = results["optimized_teach_points"]
    initial_teach_points   = results["initial_teach_points"]
    target_x               = results["target_x"]
    target_y               = results["target_y"]
    teacher_loss_history   = results["teacher_loss_history"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(teacher_loss_history, 'o-', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Teaching Loss', fontsize=12)
    ax1.set_title('Teaching Loss over Iterations', fontsize=14)
    ax1.grid(True)

    ax2.plot(target_x, target_y, 'b-', linewidth=2, label='Target Function')
    ax2.scatter(initial_teach_points[:, 0], initial_teach_points[:, 1], c='r', s=100, alpha=0.6, label='Initial Teaching Points')
    ax2.scatter(optimized_teach_points[:, 0], optimized_teach_points[:, 1], c='g', s=100, label='Optimized Teaching Points')
    for x in optimized_teach_points[:, 0]:
        ax2.axvline(x=x, color='g', linestyle='--', alpha=0.5)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_xlim(min(target_x), max(target_x))
    ax2.set_ylim(0, 3.5)
    ax2.set_title('Teaching Points Optimization Results', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\nInitial teaching loss: {results.get('initial_teacher_loss', 'N/A')}")
    print(f"Final teaching loss: {teacher_loss_history[-1]}")
    print(f"Improvement: {(1 - teacher_loss_history[-1]/results.get('initial_teacher_loss', 1)) * 100:.2f}% (if positive)")

###############################################
##### VISUALIZING LEARNER REPRESENTATIONS #####
###############################################

def visualize_learner_samples(learner_hparams, teaching_points, target_x, target_y,
                               num_fn_samples=10, random_seed=37):
    key = jax.random.PRNGKey(random_seed)
    cmap = plt.cm.viridis_r

    teach_x = teaching_points[:, 0]
    teach_y = teaching_points[:, 1]

    log_posts = learner_logpost_fn_vec(learner_hparams, teach_x, teach_y)
    learner_hparam_probs = jax.nn.softmax(log_posts)
    sampled_prob_values = []

    fig, ax = plt.subplots(figsize=(8, 5))

    for _ in range(num_fn_samples):
        key, subkey = jax.random.split(key)
        sampled_hparam_idx = jax.random.categorical(subkey, log_posts)
        sampled_hparam = jnp.array([learner_hparams[sampled_hparam_idx]])
        sampled_prob_values.append(float(learner_hparam_probs[sampled_hparam_idx]))

        gp_mu, gp_k = learner_gp(teach_x, teach_y, target_x, sampled_hparam, just_var=False)
        key, subkey = jax.random.split(key)
        sampled_func = random.multivariate_normal(subkey, gp_mu[0], gp_k[0], method="svd")
        ax.plot(target_x, sampled_func, alpha=0.5)

    ax.plot(target_x, target_y, zorder=50, color='black', label='Target Fn', linestyle='--', linewidth=2)
    ax.scatter(teach_x, teach_y, c='r', s=50, zorder=100, label='Teaching Points')

    if sampled_prob_values:
        min_prob = min(sampled_prob_values)
        max_prob = max(sampled_prob_values)
        if max_prob - min_prob < 1e-5:
            mean_prob = (max_prob + min_prob) / 2
            min_prob, max_prob = mean_prob * 0.95, mean_prob * 1.05

        from matplotlib.colors import Normalize
        norm = Normalize(min_prob, max_prob)
        for line, prob in zip(ax.lines[:-1], sampled_prob_values):
            line.set_color(cmap(norm(prob)))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax).set_label('Hyperparameter Probability')

    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_xlim(jnp.min(target_x), jnp.max(target_x))
    ax.set_ylim(min(jnp.min(teach_y) * 0.9, 0), max(jnp.max(teach_y) * 1.1, 3.5))
    ax.set_title('Sampled Functions (RBF Kernel)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_teacher_results_gp(x, y, teach_x, teach_y, kernel_type=None, learn_hparams=None,
                                  convert_coords=True, show_plots=True, bounded=False, lamb=0.5, ax=None):
    """Visualize GP posterior for given teaching points (1D RBF kernel).
    kernel_type is accepted for backward compatibility but ignored."""
    if learn_hparams is None:
        lhp = create_grid_hparams()
        lhp = learner_svgd(lhp, 0, teach_x, teach_y)
    else:
        lhp = learn_hparams

    if bounded:
        weighted_mean, weighted_var = learner_gp_mix_bounded(teach_x, teach_y, x, lhp, just_var=True)
    else:
        weighted_mean, weighted_var = learner_gp_mix(teach_x, teach_y, x, lhp, just_var=True)

    weighted_std = jnp.sqrt(jnp.maximum(weighted_var, 0))
    teacher_loss = teacher_loss_fn(teach_x, teach_y, x, y, lhp, lamb=lamb, bounded=bounded)

    z = 1.96
    upper_bound = weighted_mean + z * weighted_std
    lower_bound = weighted_mean - z * weighted_std

    lhp_logprob = jax.nn.softmax(learner_logpost_fn_vec(lhp, teach_x, teach_y))

    if show_plots or ax is not None:
        if ax is None:
            ax = plt.gca()
            created_ax = False
        else:
            created_ax = True

        ax.plot(x, y, 'k-', linewidth=2.5, label='True Function')
        ax.plot(x, weighted_mean, 'b-', linewidth=2, label='Weighted Mean')
        ax.fill_between(x, lower_bound, upper_bound, color='lightblue', alpha=0.3, label='95% CI')
        ax.scatter(teach_x, teach_y, color='red', s=100, zorder=5, label='Teaching Points')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 3.5)
        ax.set_title(f'RBF GP | Loss: {teacher_loss:.4f} (λ={lamb})', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if show_plots and not created_ax:
            plt.show()

    return None, lhp, lhp_logprob, weighted_mean, weighted_std, teacher_loss

#######################################
##### BERNOULLI GP CLASSIFICATION #####
#######################################

def bernoulli_log_likelihood(y, f):
    """Log likelihood for Bernoulli observations via numerically stable form."""
    return jnp.sum(y * f - jnp.logaddexp(0, f))


def laplace_mode_finding(K, y, prior_mean=0.0, max_iter=10):
    """Find posterior mode via Newton-Raphson (fixed iterations for JAX vmap/jit compat)."""
    N = K.shape[0]
    K_jitter = K + 1e-6 * jnp.eye(N)

    def newton_step(f, _):
        pi = jax.nn.sigmoid(f)
        W = jnp.maximum(pi * (1 - pi), 1e-10)
        sqrt_W = jnp.sqrt(W)
        B = jnp.eye(N) + (sqrt_W[:, None] * K_jitter) * sqrt_W[None, :]
        L = jsp.linalg.cholesky(B, lower=True)
        b = W * (f - prior_mean) + (y - pi)
        sqrt_W_Kb = sqrt_W * (K_jitter @ b)
        v = jsp.linalg.solve_triangular(L, sqrt_W_Kb, lower=True)
        a = b - sqrt_W * jsp.linalg.solve_triangular(L.T, v, lower=False)
        return prior_mean + K_jitter @ a, None

    f, _ = jax.lax.scan(newton_step, jnp.ones(N) * prior_mean, None, length=max_iter)

    pi = jax.nn.sigmoid(f)
    W = jnp.maximum(pi * (1 - pi), 1e-10)
    sqrt_W = jnp.sqrt(W)
    B = jnp.eye(N) + (sqrt_W[:, None] * K_jitter) * sqrt_W[None, :]
    L = jsp.linalg.cholesky(B, lower=True)
    return f, W, L


def laplace_predict(X_train, y_train, X_test, kernel_params, kernel_name="rbf_2d_ard", prior_mean=-2.0):
    """Predict probabilities at test points using Laplace approximation."""
    N = y_train.shape[0]
    kernel_vec = get_vectorized_kernel(kernel_name)

    K         = kernel_vec(X_train, X_train, kernel_params) + 1e-6 * jnp.eye(N)
    K_star    = kernel_vec(X_train, X_test, kernel_params).T  # (N, M)
    K_starstar = kernel_vec(X_test,  X_test, kernel_params)   # (M, M)

    f_hat, W, L = laplace_mode_finding(K, y_train, prior_mean)

    f_mean = prior_mean + K_star.T @ (y_train - jax.nn.sigmoid(f_hat))

    sqrt_W = jnp.sqrt(W)
    v = jsp.linalg.solve_triangular(L, sqrt_W[:, None] * K_star, lower=True)
    f_var = jnp.maximum(jnp.diag(K_starstar) - jnp.sum(v**2, axis=0), 1e-10)

    # Probit approximation: P(y=1) ≈ sigmoid(f_mean / sqrt(1 + π/8 * f_var))
    kappa = 1.0 / jnp.sqrt(1 + jnp.pi / 8 * f_var)
    return jax.nn.sigmoid(kappa * f_mean), f_mean, f_var


def laplace_log_marginal_likelihood(X_train, y_train, kernel_params, kernel_name="rbf_2d_ard", prior_mean=-2.0):
    """Laplace approximation to log p(y | X, theta).

    log p(y|X,θ) ≈ log p(y|f̂) + log p(f̂) - 0.5 * log|B|  where B = I + W^(1/2) K W^(1/2)
    """
    N = y_train.shape[0]
    kernel_vec = get_vectorized_kernel(kernel_name)
    K = kernel_vec(X_train, X_train, kernel_params) + 1e-6 * jnp.eye(N)

    f_hat, W, L = laplace_mode_finding(K, y_train, prior_mean)
    log_lik = bernoulli_log_likelihood(y_train, f_hat)

    f_centered = f_hat - prior_mean
    L_K = jsp.linalg.cholesky(K, lower=True)
    alpha = jsp.linalg.cho_solve((L_K, True), f_centered)
    log_det_K = 2 * jnp.sum(jnp.log(jnp.diag(L_K)))
    log_prior = -0.5 * jnp.dot(f_centered, alpha) - 0.5 * log_det_K - 0.5 * N * jnp.log(2 * jnp.pi)
    log_det_B = 2 * jnp.sum(jnp.log(jnp.diag(L)))

    return log_lik + log_prior - 0.5 * log_det_B


def laplace_log_marginal_likelihood_vec(X_train, y_train, kernel_params_batch,
                                         kernel_name="rbf_2d_ard", prior_mean=-2.0):
    return jax.vmap(
        lambda params: laplace_log_marginal_likelihood(X_train, y_train, params, kernel_name, prior_mean)
    )(kernel_params_batch)


def laplace_predict_batch(X_train, y_train, X_test, kernel_params_batch,
                           kernel_name="rbf_2d_ard", prior_mean=-2.0):
    return jax.vmap(
        lambda params: laplace_predict(X_train, y_train, X_test, params, kernel_name, prior_mean)
    )(kernel_params_batch)


def bernoulli_gp_predict_mixture(X_train, y_train, X_test, kernel_params_batch,
                                  kernel_name="rbf_2d_ard", prior_mean=-2.0):
    """Marginal-likelihood-weighted mixture prediction."""
    log_ml = laplace_log_marginal_likelihood_vec(X_train, y_train, kernel_params_batch, kernel_name, prior_mean)
    weights = jax.nn.softmax(log_ml)
    prob_means, f_means, f_vars = laplace_predict_batch(X_train, y_train, X_test, kernel_params_batch, kernel_name, prior_mean)
    return jnp.average(prob_means, weights=weights, axis=0), jnp.average(f_vars, weights=weights, axis=0)


def bernoulli_gp_classify(
    train_coords,
    train_labels,
    test_coords,
    kernel_name: str = "rbf_2d_ard",
    prior_mean: float = -2.0,
    grid_sizes: Optional[Dict[str, int]] = None,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    return_latent: bool = False
) -> Dict:
    """
    High-level interface for Bernoulli GP classification.

    Returns dict with 'probabilities', 'best_params', 'hyperparameters',
    'log_marginal_likelihoods', and optionally 'latent_mean'/'latent_var'.
    """
    hparams = create_grid_hparams_2d_ard(grid_sizes=grid_sizes, ranges=ranges)

    log_mls   = laplace_log_marginal_likelihood_vec(train_coords, train_labels, hparams, kernel_name, prior_mean)
    best_params = hparams[jnp.argmax(log_mls)]

    probs, f_vars = bernoulli_gp_predict_mixture(train_coords, train_labels, test_coords, hparams, kernel_name, prior_mean)

    result = {
        'probabilities': probs,
        'best_params': best_params,
        'log_marginal_likelihoods': log_mls,
        'hyperparameters': hparams,
    }

    if return_latent:
        _, f_mean, f_var = laplace_predict(train_coords, train_labels, test_coords, best_params, kernel_name, prior_mean)
        result['latent_mean'] = f_mean
        result['latent_var']  = f_var

    return result


def visualize_bernoulli_gp(
    train_coords,
    train_labels,
    test_coords,
    predictions: Dict,
    feature_names_train=None,
    feature_names_test=None,
    grid_resolution: int = 50,
    kernel_name: str = "rbf_2d_ard",
    prior_mean: float = -2.0,
    ax=None
):
    """Visualize Bernoulli GP classification: probability heatmap + train/test points."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    x_min = float(jnp.minimum(train_coords[:, 0].min(), test_coords[:, 0].min())) - 0.1
    x_max = float(jnp.maximum(train_coords[:, 0].max(), test_coords[:, 0].max())) + 0.1
    y_min = float(jnp.minimum(train_coords[:, 1].min(), test_coords[:, 1].min())) - 0.1
    y_max = float(jnp.maximum(train_coords[:, 1].max(), test_coords[:, 1].max())) + 0.1

    xx, yy = jnp.meshgrid(
        jnp.linspace(x_min, x_max, grid_resolution),
        jnp.linspace(y_min, y_max, grid_resolution)
    )
    grid_coords = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    probs_grid, _, _ = laplace_predict(
        train_coords, train_labels, grid_coords,
        predictions['best_params'], kernel_name=kernel_name, prior_mean=prior_mean
    )
    probs_grid = probs_grid.reshape(xx.shape)

    im = ax.contourf(xx, yy, probs_grid, levels=20, cmap='RdYlGn', alpha=0.7)
    plt.colorbar(im, ax=ax, label='P(kind-linked)')

    ax.scatter(train_coords[:, 0], train_coords[:, 1],
               c='blue', s=100, edgecolors='black', linewidths=2,
               label='Training (positive)', zorder=5)
    ax.scatter(test_coords[:, 0], test_coords[:, 1],
               c=predictions['probabilities'], cmap='RdYlGn',
               s=150, marker='s', edgecolors='black', linewidths=2,
               vmin=0, vmax=1, label='Test predictions', zorder=6)

    if feature_names_train is not None:
        for i, name in enumerate(feature_names_train):
            ax.annotate(name, (float(train_coords[i, 0]), float(train_coords[i, 1])), fontsize=6, alpha=0.7)

    if feature_names_test is not None:
        for i, name in enumerate(feature_names_test):
            prob = float(predictions['probabilities'][i])
            ax.annotate(f"{name}\n({prob:.2f})", (float(test_coords[i, 0]), float(test_coords[i, 1])), fontsize=8)

    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_title('Bernoulli GP Classification: P(kind-linked)')
    ax.legend()
    plt.tight_layout()
    return ax
