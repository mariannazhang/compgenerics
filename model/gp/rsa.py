import jax
import jax.numpy as jnp
from jax import vmap
from jax.nn import softmax, log_softmax
from itertools import product as iterproduct
from collections import namedtuple
from typing import Optional

#####################
# SETUP
#####################

# Utterance: subj=0 means "This Zarpie" (specific), subj=1 means "Zarpies" (generic)
# feature_idx: integer index into vocab
Utterance = namedtuple("Utterance", ("subj", "feature_idx"))
Instance  = namedtuple("Instance",  ("kind", "features"))   # features: jnp array shape (V,)
Kind      = namedtuple("Kind",       ("name", "features"))   # features: jnp array shape (V,)

COHERENCE_PRIOR = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

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

def enumerate_states(V: int, coherence_prior: jnp.ndarray = None):
    """
    Enumerate all (coherence, feature_mask) states.

    For coherence c and mask of k ones out of V features:
      log P(mask, c) = log(1/C) + k*log(c) + (V-k)*log(1-c)

    Returns:
      states_coherence:   (N_states,)    coherence value per state
      states_masks:       (N_states, V)  binary feature mask per state
      prior_log_weights:  (N_states,)    log P(mask, coherence)
    where N_states = C * 2^V, C = len(coherence_prior)
    """
    if coherence_prior is None:
        coherence_prior = COHERENCE_PRIOR

    C = len(coherence_prior)
    all_masks = jnp.array(list(iterproduct([0.0, 1.0], repeat=V)))  # (2^V, V)
    n_masks = all_masks.shape[0]

    # Tile masks for each coherence value and repeat coherence for each mask
    states_masks = jnp.tile(all_masks, (C, 1))                         # (C*2^V, V)
    states_coherence = jnp.repeat(coherence_prior, n_masks)            # (C*2^V,)

    k = jnp.sum(states_masks, axis=1)                                  # (N_states,)

    log_c     = jnp.log(jnp.clip(states_coherence, 1e-9, 1 - 1e-9))
    log_1mc   = jnp.log(jnp.clip(1 - states_coherence, 1e-9, 1 - 1e-9))
    log_prior = -jnp.log(C)  # uniform over coherence values

    prior_log_weights = log_prior + k * log_c + (V - k) * log_1mc     # (N_states,)

    return states_coherence, states_masks, prior_log_weights

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
                     coherence_prior: Optional[jnp.ndarray] = None,
                     lesioned: bool = False) -> dict:
    """
    Literal Listener (L0): infers kind-linked features from (utterance, instance) data.

    data: list of (Utterance, Instance) pairs
          Instance.features must be a binary jnp array of shape (V,)

    Returns dict:
      "states_coherence": (N_states,)   coherence per state
      "states_masks":     (N_states, V) kind-feature mask per state
      "log_weights":      (N_states,)   unnormalized log posterior
      "weights":          (N_states,)   normalized posterior (sums to 1)
    """
    if coherence_prior is None:
        coherence_prior = COHERENCE_PRIOR

    V = len(vocab)
    states_coherence, states_masks, prior_log_weights = enumerate_states(V, coherence_prior)

    # For each state, accumulate log-likelihood over all (utt, inst) pairs
    # per_state_loglik: (N_states,)
    log_lik = jnp.zeros(states_masks.shape[0])

    for utt, inst in data:
        inst_features = inst.features  # (V,)

        # Vectorize meaning over all states (each state has a different kind_features mask)
        def state_loglik(kind_features):
            m = meaning(kind_features, inst_features,
                            utt.subj, utt.feature_idx, lesioned)
            likelihood = jnp.where(m > 0.5, 0.95, 0.05)
            return jnp.log(likelihood)

        log_lik = log_lik + vmap(state_loglik)(states_masks)  # (N_states,)

    log_weights = prior_log_weights + log_lik
    weights = softmax(log_weights)

    return {
        "states_coherence": states_coherence,
        "states_masks":     states_masks,
        "log_weights":      log_weights,
        "weights":          weights,
    }

#####################
# SPEAKER (S1)
#####################

def _expected_jaccard_l0(kind_features: jnp.ndarray,
                          inst_features: jnp.ndarray,
                          utt_subj: int,
                          utt_feat_idx: int,
                          vocab: list,
                          coherence_prior: Optional[jnp.ndarray] = None,
                          lesioned: bool = False) -> jnp.ndarray:
    """
    E[Jaccard(true_kind & inst, inferred_kind & inst)] under L0 posterior,
    given a single (utterance, instance) observation.
    """
    utt  = Utterance(subj=utt_subj, feature_idx=utt_feat_idx)
    inst = Instance(kind="", features=inst_features)

    posterior = literal_listener([(utt, inst)], vocab, coherence_prior, lesioned)
    weights   = posterior["weights"]          # (N_states,)
    masks     = posterior["states_masks"]     # (N_states, V)

    # Restrict to features present in the observed instance
    true_restricted = kind_features * inst_features   # (V,)

    def jaccard_for_state(inferred_mask):
        inferred_restricted = inferred_mask * inst_features
        return jaccard_similarity(true_restricted, inferred_restricted)

    jaccards = vmap(jaccard_for_state)(masks)          # (N_states,)
    return jnp.dot(weights, jaccards)                  # scalar


def speaker(kind_features: jnp.ndarray,
            observed_instance: Instance,
            vocab: list,
            inv_temp: float = 20.0,
            coherence_prior: Optional[jnp.ndarray] = None,
            lesioned: bool = False) -> jnp.ndarray:
    """
    Speaker (S1): chooses utterance to maximize L0's ability to infer kind features.

    Utterance space: 2*V utterances indexed as [subj*V + feat_idx]
      indices 0..(V-1):   subj=0 ("This Zarpie"), feature 0..V-1
      indices V..(2V-1):  subj=1 ("Zarpies"),     feature 0..V-1

    Returns utterance probability distribution of shape (2*V,).
    """
    if coherence_prior is None:
        coherence_prior = COHERENCE_PRIOR

    V = len(vocab)
    inst_features = observed_instance.features  # (V,)

    utilities = []
    for subj in [0, 1]:
        for feat_idx in range(V):
            u = _expected_jaccard_l0(kind_features, inst_features,
                                     subj, feat_idx,
                                     vocab, coherence_prior, lesioned)
            utilities.append(u)

    utilities = jnp.stack(utilities)          # (2*V,)
    log_weights = inv_temp * utilities
    return softmax(log_weights)               # (2*V,)

#####################
# PRAGMATIC LISTENER (L1)
#####################

def pragmatic_listener(data: list,
                       vocab: list,
                       coherence_prior: Optional[jnp.ndarray] = None,
                       lesioned: bool = False) -> dict:
    """
    Pragmatic Listener (L1): like L0 but conditions on speaker (S1) likelihood.

    data: list of (Utterance, Instance) pairs

    Returns same dict format as literal_listener.
    """
    if coherence_prior is None:
        coherence_prior = COHERENCE_PRIOR

    V = len(vocab)
    states_coherence, states_masks, prior_log_weights = enumerate_states(V, coherence_prior)
    N_states = states_masks.shape[0]

    log_lik = jnp.zeros(N_states)

    for utt, inst in data:
        inst_features = inst.features  # (V,)
        utt_idx = utt.subj * V + utt.feature_idx

        # For each state, get speaker distribution and extract P(utt)
        def state_log_speaker_lik(kind_features):
            utt_probs = speaker(kind_features, inst, vocab, 20.0,
                                coherence_prior, lesioned)
            return jnp.log(jnp.clip(utt_probs[utt_idx], 1e-9, 1.0))

        log_lik = log_lik + vmap(state_log_speaker_lik)(states_masks)

    log_weights = prior_log_weights + log_lik
    weights = softmax(log_weights)

    return {
        "states_coherence": states_coherence,
        "states_masks":     states_masks,
        "log_weights":      log_weights,
        "weights":          weights,
    }

#####################
# SPEAKER 2 (S2)
#####################

def _expected_jaccard_l1(kind_features: jnp.ndarray,
                          inst_features: jnp.ndarray,
                          utt_subj: int,
                          utt_feat_idx: int,
                          vocab: list,
                          coherence_prior: Optional[jnp.ndarray] = None,
                          lesioned: bool = False) -> jnp.ndarray:
    """
    E[Jaccard(true_kind & inst, inferred_kind & inst)] under L1 posterior,
    given a single (utterance, instance) observation.
    """
    utt  = Utterance(subj=utt_subj, feature_idx=utt_feat_idx)
    inst = Instance(kind="", features=inst_features)

    posterior = pragmatic_listener([(utt, inst)], vocab, coherence_prior, lesioned)
    weights   = posterior["weights"]       # (N_states,)
    masks     = posterior["states_masks"]  # (N_states, V)

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
             coherence_prior: Optional[jnp.ndarray] = None,
             lesioned: bool = False) -> jnp.ndarray:
    """
    Speaker 2 (S2): like S1 but reasons about L1 (pragmatic listener).

    Returns utterance probability distribution of shape (2*V,).
    """
    if coherence_prior is None:
        coherence_prior = COHERENCE_PRIOR

    V = len(vocab)
    inst_features = observed_instance.features

    utilities = []
    for subj in [0, 1]:
        for feat_idx in range(V):
            u = _expected_jaccard_l1(kind_features, inst_features,
                                     subj, feat_idx,
                                     vocab, coherence_prior, lesioned)
            utilities.append(u)

    utilities = jnp.stack(utilities)
    log_weights = inv_temp * utilities
    return softmax(log_weights)
