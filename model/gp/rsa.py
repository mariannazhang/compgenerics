import jax
import jax.numpy as jnp
from jax import vmap
from jax.nn import softmax, log_softmax
from itertools import product as iterproduct
from collections import namedtuple
from jax.scipy.special import gammaln

#####################
# SETUP
#####################

# Utterance: subj=0 means "This Zarpie" (specific), subj=1 means "Zarpies" (generic)
# feature_idx: integer index into vocab
Utterance = namedtuple("Utterance", ("subj", "feature_idx"))
Instance  = namedtuple("Instance",  ("kind", "features"))   # features: jnp array shape (V,)
Kind      = namedtuple("Kind",       ("name", "features"))   # features: jnp array shape (V,)

# Default Beta prior hyperparameters (alpha=beta=1 corresponds to uniform over coherence)
DEFAULT_ALPHA = 1.0
DEFAULT_BETA  = 1.0

#####################
# VOCAB HELPERS
#####################

def encode(feature_names: list, vocab: list) -> jnp.ndarray:
    """Encode a list of feature names into a binary vector of shape (V,)."""
    return jnp.array([1.0 if f in feature_names else 0.0 for f in vocab])

def decode(feature_vec: jnp.ndarray, vocab: list) -> list:
    """Decode binary vector back to list of feature names."""
    return [vocab[i] for i, v in enumerate(feature_vec) if v > 0.5]

#####################
# STATE ENUMERATION
#####################

def _log_beta(a, b):
    """log B(a, b) = log Gamma(a) + log Gamma(b) - log Gamma(a+b)"""
    return gammaln(a) + gammaln(b) - gammaln(a + b)

def enumerate_states(V: int, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA):
    """
    Enumerate all feature_mask states with Beta-Binomial prior.

    Marginalizes over coherence analytically:
      log P(mask | alpha, beta) = log B(alpha+k, beta+V-k) - log B(alpha, beta)

    where k = number of ones in the mask. alpha and beta are pseudocounts:
      alpha = prior count of kind-linked features
      beta  = prior count of non-kind-linked features
    alpha=beta=1 is uniform (equivalent to the old discrete uniform prior).

    Returns:
      states_masks:       (2^V, V)  binary feature mask per state
      prior_log_weights:  (2^V,)    log P(mask | alpha, beta)
    """
    states_masks = jnp.array(list(iterproduct([0.0, 1.0], repeat=V)))  # (2^V, V)
    k = jnp.sum(states_masks, axis=1)                                   # (2^V,)

    # Each mask has probability B(alpha+k, beta+V-k) / B(alpha, beta)
    # (no binomial coefficient — we enumerate individual masks, not counts)
    log_bb = _log_beta(alpha + k, beta + V - k) - _log_beta(alpha, beta)

    return states_masks, log_bb

#####################
# MEANING FUNCTION
#####################

def meaning(kind_features: jnp.ndarray, inst_features: jnp.ndarray,
                utt_subj: int, utt_feat_idx: int,
                lesioned: bool = False) -> jnp.ndarray:
    """
    Semantic meaning of an utterance.

    utt_subj=0 ("This Zarpie"): true iff feature is in instance
    utt_subj=1 ("Zarpies"):     true iff feature is in kind  (or instance if lesioned)

    Returns scalar float (0.0 or 1.0).
    """
    specific_meaning = inst_features[utt_feat_idx]
    generic_meaning  = inst_features[utt_feat_idx] if lesioned else kind_features[utt_feat_idx]
    return jnp.where(utt_subj == 0, specific_meaning, generic_meaning)

#####################
# JACCARD SIMILARITY
#####################

def jaccard_similarity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Jaccard similarity between two binary vectors.
    Returns 1.0 when both are all-zero.
    """
    intersection = jnp.sum(a * b)
    union = jnp.sum(jnp.clip(a + b, 0.0, 1.0))
    return jnp.where(union == 0, 1.0, intersection / union)

#####################
# LITERAL LISTENER (L0)
#####################

def literal_listener(data: list,
                     vocab: list,
                     alpha: float = DEFAULT_ALPHA,
                     beta: float = DEFAULT_BETA,
                     lesioned: bool = False) -> dict:
    """
    Literal Listener (L0): infers kind-linked features from (utterance, instance) data.

    data: list of (Utterance, Instance) pairs
          Instance.features must be a binary jnp array of shape (V,)

    alpha, beta: Beta prior hyperparameters over coherence (pseudocounts).
                 alpha=beta=1 is uniform.

    Returns dict:
      "states_masks":     (2^V, V)  kind-feature mask per state
      "log_weights":      (2^V,)    unnormalized log posterior
      "weights":          (2^V,)    normalized posterior (sums to 1)
    """
    V = len(vocab)
    states_masks, prior_log_weights = enumerate_states(V, alpha, beta)

    log_lik = jnp.zeros(states_masks.shape[0])

    for utt, inst in data:
        inst_features = inst.features  # (V,)

        def state_loglik(kind_features):
            m = meaning(kind_features, inst_features,
                            utt.subj, utt.feature_idx, lesioned)
            likelihood = jnp.where(m > 0.5, 0.95, 0.05)
            return jnp.log(likelihood)

        log_lik = log_lik + vmap(state_loglik)(states_masks)  # (2^V,)

    log_weights = prior_log_weights + log_lik
    weights = softmax(log_weights)

    return {
        "states_masks":  states_masks,
        "log_weights":   log_weights,
        "weights":       weights,
    }

#####################
# SPEAKER (S1)
#####################

def _expected_jaccard_l0(kind_features: jnp.ndarray,
                          inst_features: jnp.ndarray,
                          utt_subj: int,
                          utt_feat_idx: int,
                          vocab: list,
                          alpha: float = DEFAULT_ALPHA,
                          beta: float = DEFAULT_BETA,
                          lesioned: bool = False) -> jnp.ndarray:
    """
    E[Jaccard(true_kind & inst, inferred_kind & inst)] under L0 posterior,
    given a single (utterance, instance) observation.
    """
    utt  = Utterance(subj=utt_subj, feature_idx=utt_feat_idx)
    inst = Instance(kind="", features=inst_features)

    posterior = literal_listener([(utt, inst)], vocab, alpha, beta, lesioned)
    weights   = posterior["weights"]       # (2^V,)
    masks     = posterior["states_masks"]  # (2^V, V)

    true_restricted = kind_features * inst_features   # (V,)

    def jaccard_for_state(inferred_mask):
        inferred_restricted = inferred_mask * inst_features
        return jaccard_similarity(true_restricted, inferred_restricted)

    jaccards = vmap(jaccard_for_state)(masks)
    return jnp.dot(weights, jaccards)


def speaker(kind_features: jnp.ndarray,
            observed_instance: Instance,
            vocab: list,
            inv_temp: float = 20.0,
            alpha: float = DEFAULT_ALPHA,
            beta: float = DEFAULT_BETA,
            lesioned: bool = False) -> jnp.ndarray:
    """
    Speaker (S1): chooses utterance to maximize L0's ability to infer kind features.

    Utterance space: 2*V utterances indexed as [subj*V + feat_idx]
      indices 0..(V-1):   subj=0 ("This Zarpie"), feature 0..V-1
      indices V..(2V-1):  subj=1 ("Zarpies"),     feature 0..V-1

    Returns utterance probability distribution of shape (2*V,).
    """
    V = len(vocab)
    inst_features = observed_instance.features  # (V,)

    utilities = []
    for subj in [0, 1]:
        for feat_idx in range(V):
            u = _expected_jaccard_l0(kind_features, inst_features,
                                     subj, feat_idx,
                                     vocab, alpha, beta, lesioned)
            utilities.append(u)

    utilities = jnp.stack(utilities)
    log_weights = inv_temp * utilities
    return softmax(log_weights)

#####################
# PRAGMATIC LISTENER (L1)
#####################

def pragmatic_listener(data: list,
                       vocab: list,
                       alpha: float = DEFAULT_ALPHA,
                       beta: float = DEFAULT_BETA,
                       lesioned: bool = False) -> dict:
    """
    Pragmatic Listener (L1): like L0 but conditions on speaker (S1) likelihood.

    data: list of (Utterance, Instance) pairs

    Returns same dict format as literal_listener.
    """
    V = len(vocab)
    states_masks, prior_log_weights = enumerate_states(V, alpha, beta)
    N_states = states_masks.shape[0]

    log_lik = jnp.zeros(N_states)

    for utt, inst in data:
        inst_features = inst.features  # (V,)
        utt_idx = utt.subj * V + utt.feature_idx

        def state_log_speaker_lik(kind_features):
            utt_probs = speaker(kind_features, inst, vocab, 20.0,
                                alpha, beta, lesioned)
            return jnp.log(jnp.clip(utt_probs[utt_idx], 1e-9, 1.0))

        log_lik = log_lik + vmap(state_log_speaker_lik)(states_masks)

    log_weights = prior_log_weights + log_lik
    weights = softmax(log_weights)

    return {
        "states_masks":  states_masks,
        "log_weights":   log_weights,
        "weights":       weights,
    }

#####################
# SPEAKER 2 (S2)
#####################

def _expected_jaccard_l1(kind_features: jnp.ndarray,
                          inst_features: jnp.ndarray,
                          utt_subj: int,
                          utt_feat_idx: int,
                          vocab: list,
                          alpha: float = DEFAULT_ALPHA,
                          beta: float = DEFAULT_BETA,
                          lesioned: bool = False) -> jnp.ndarray:
    """
    E[Jaccard(true_kind & inst, inferred_kind & inst)] under L1 posterior,
    given a single (utterance, instance) observation.
    """
    utt  = Utterance(subj=utt_subj, feature_idx=utt_feat_idx)
    inst = Instance(kind="", features=inst_features)

    posterior = pragmatic_listener([(utt, inst)], vocab, alpha, beta, lesioned)
    weights   = posterior["weights"]       # (2^V,)
    masks     = posterior["states_masks"]  # (2^V, V)

    true_restricted = kind_features * inst_features

    def jaccard_for_state(inferred_mask):
        inferred_restricted = inferred_mask * inst_features
        return jaccard_similarity(true_restricted, inferred_restricted)

    jaccards = vmap(jaccard_for_state)(masks)
    return jnp.dot(weights, jaccards)


def speaker2(kind_features: jnp.ndarray,
             observed_instance: Instance,
             vocab: list,
             inv_temp: float = 5.0,
             alpha: float = DEFAULT_ALPHA,
             beta: float = DEFAULT_BETA,
             lesioned: bool = False) -> jnp.ndarray:
    """
    Speaker 2 (S2): like S1 but reasons about L1 (pragmatic listener).

    Returns utterance probability distribution of shape (2*V,).
    """
    V = len(vocab)
    inst_features = observed_instance.features

    utilities = []
    for subj in [0, 1]:
        for feat_idx in range(V):
            u = _expected_jaccard_l1(kind_features, inst_features,
                                     subj, feat_idx,
                                     vocab, alpha, beta, lesioned)
            utilities.append(u)

    utilities = jnp.stack(utilities)
    log_weights = inv_temp * utilities
    return softmax(log_weights)
