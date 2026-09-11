"""Data loaders for studies 6 and 8 (feature set 1) for the per-feature-prevalence model.

Both studies rate the 16 TEST-split features of set1 and hear statements about
(subsets of) the 16 TRAIN-split features. Unlike study 9, statements can be
SPECIFIC (u=1) as well as generic (u=0); the RSA speaker in model_jax already
distinguishes them (MEANING_MATRIX), so specifics enter as pragmatic soft
evidence against kind-linkedness.

  - Study 6 (study6_prereg.csv): 3 groups. 'generic' = one generic per train
    feature (16 utterances, u=0), 'specific' = one specific per train feature
    (u=1), 'baseline' = no statements.
  - Study 8 (study8_prereg.csv): each participant gets their own random assignment of
    G/S over a subset of the train features (training_features_order aligned
    with training_structure). Participants are grouped by their exact
    feature->utterance configuration; each unique configuration is one
    likelihood group (its own GP + Laplace term). Exclusions mirror
    analyses/study 8/study8_analysis.Rmd: attention check == 100, AI == 'No',
    and the task-comprehension check.

Loaders return (geom, responses_cond[, groups_df]) in the same format
inference.py uses for study 9, with geom['x_train_cond'] / geom['u_train_cond']
keyed by group. inference.prepare_all_conditions / total_log_lik(_free_shapes)
iterate the data dict keys, so everything downstream just works.
"""
import pickle as pkl

import numpy as np
import pandas as pd

from inference import H   # BEFORE any jax import: inference sets JAX_ENABLE_X64=1
import jax.numpy as jnp

SET1_PKL = '../features/set1_features_dataframe.pkl'

# --- study 6 rating columns -> set1 test-split feature names ---------------
STUDY6_COL_TO_FEATURE = {
    'live_caves': 'live in caves',
    'ride_lions': 'ride lions',
    'farm_potatoes': 'farm potatoes',
    'play_banjo': 'play banjos',
    'look_left': 'look to their left when spoken to',
    'clap_three': 'clap three times before entering a room',
    'smile_sad': 'smile when they are sad',
    'chug_syrup': 'chug maple syrup',
    'yell_cats': 'yell at stray cats',
    'go_opera': 'go to the opera',
    'dance_fire': 'dance around a fire on their 10th birthday',
    'sing_songs': 'sing beautiful songs',
    'scream_windows': 'scream out windows',
    'smell_garbage': 'smell garbage for fun',
    'wash_ponds': 'wash their clothes in ponds',
    'paint_hands': 'paint their hands yellow',
}

# --- study 8 rating columns (Qualtrics) -> set1 test-split feature names ---
STUDY8_COL_TO_FEATURE = {
    'cave_1': 'live in caves',
    'lion_1': 'ride lions',
    'potatoes_1': 'farm potatoes',
    'banjo_1': 'play banjos',
    'look_left_1': 'look to their left when spoken to',
    'clap_1': 'clap three times before entering a room',
    'sad_1': 'smile when they are sad',
    'maple_syrup_1': 'chug maple syrup',
    'cats_1': 'yell at stray cats',
    'opera_1': 'go to the opera',
    'dance_1': 'dance around a fire on their 10th birthday',
    'song_1': 'sing beautiful songs',
    'window_1': 'scream out windows',
    'garbage_1': 'smell garbage for fun',
    'pond_1': 'wash their clothes in ponds',
    'yellow_1': 'paint their hands yellow',
}

# --- study 8 training_features_order tokens -> set1 train-split features ---
STUDY8_TOKEN_TO_FEATURE = {
    'scared_ladybugs': 'are scared of ladybugs',
    'babies_blankets': 'babies are wrapped in orange blankets',
    'bounce_ball_head': 'can bounce a ball on their heads',
    'can_flip_air': 'can flip in the air',
    'chase_shadows': 'chase shadows',
    'climb_fences': 'climb tall fences',
    'draw_stars_knees': 'draw stars on their knees',
    'freckles_feet': 'have freckles on their feet',
    'stripes_hair': 'have stripes in their hair',
    'hop_puddles': 'hop over puddles',
    'like_sing': 'like to sing',
    'eat_flowers': 'love to eat flowers',
    'dont_like_icecream': "really don't like ice cream",
    'dont_like_mud': "really don't like walking in the mud",
    'sleep_trees': 'sleep in tall trees',
    'flap_arms_happy': 'flap their arms when they are happy',
}

STUDY8_CHECK_TASK_ANSWER = 'Answering questions about a fictional group of people'


def load_set1(features_pkl_path=SET1_PKL):
    """set1 dataframe -> (df, coords) with coords = {feature: (2,) x_2d/y_2d array}.

    Only the 2d embedding is available for set1: the 384d pkl's row->feature
    alignment is unresolved (see feature-spaces.ipynb), so embedding_384d is
    None by design.
    """
    with open(features_pkl_path, 'rb') as f:
        df = pkl.load(f)
    coords = {r.feature: np.array([r.x_2d, r.y_2d]) for r in df.itertuples()}
    return df, coords


def _base_geom(df, coords):
    """Shared test-side geometry: the 16 test-split features in sorted order."""
    test_df = df[df.split == 'test'].sort_values('feature')
    test_feature_names = list(test_df['feature'].values)
    x_test = jnp.array(np.stack([coords[f] for f in test_feature_names]))
    test_trait = list(test_df['category'].values)
    return {'x_test': x_test, 'test_feature_names': test_feature_names,
            'test_trait': test_trait, 'J': int(x_test.shape[0]), 'embed': '2d'}


def _clip_ratings(R):
    return np.clip(np.asarray(R, dtype=float) / 100.0, H, 1.0 - H)


# ---------------------------------------------------------------------------
# Study 6
# ---------------------------------------------------------------------------
STUDY6_U_OF_COND = {'generic': 0, 'specific': 1, 'baseline': None}


def _study6_geom(features_pkl_path):
    """Shared study-6 geometry: generic/specific groups hear one statement per
    train-split feature (u=0 / u=1, sorted feature order); baseline hears none."""
    df, coords = load_set1(features_pkl_path)
    geom = _base_geom(df, coords)

    train_names = sorted(df[df.split == 'train']['feature'].values)
    x_train = jnp.array(np.stack([coords[f] for f in train_names]))
    n_tr = len(train_names)

    geom['x_train_cond'], geom['u_train_cond'], geom['train_names_cond'] = {}, {}, {}
    for c, u in STUDY6_U_OF_COND.items():
        if u is None:
            geom['x_train_cond'][c] = jnp.zeros((0, 2))
            geom['u_train_cond'][c] = jnp.zeros(0, dtype=jnp.int32)
            geom['train_names_cond'][c] = []
        else:
            geom['x_train_cond'][c] = x_train
            geom['u_train_cond'][c] = jnp.full(n_tr, u, dtype=jnp.int32)
            geom['train_names_cond'][c] = list(train_names)
    return geom


def load_study6_prereg(csv_path='../../data/study6_prereg.csv', features_pkl_path=SET1_PKL):
    """(geom, responses_cond) for the PREREGISTERED study 6 replication.

    Design/geometry: 3 groups (generic/specific/baseline) over the set1 train/test
    split (see _study6_geom). The CSV is a raw Qualtrics export in the study-8
    format (cave_1..yellow_1 rating columns, 2 header rows) with the preregistered
    exclusions: attention check == 100, AI == 'No', task check.
    """
    geom = _study6_geom(features_pkl_path)

    data = pd.read_csv(csv_path, low_memory=False).iloc[2:].copy()   # 2 Qualtrics header rows
    n0 = len(data)
    attn = pd.to_numeric(data['attn_check_1'], errors='coerce')
    keep = ((attn == 100) & (data['AI'] == 'No')
            & (data['check_task'] == STUDY8_CHECK_TASK_ANSWER)
            & data['condition'].isin(STUDY6_U_OF_COND))
    data = data[keep]
    print(f"load_study6_prereg: {len(data)} of {n0} rows kept after exclusions")

    col_of_feature = {v: k for k, v in STUDY8_COL_TO_FEATURE.items()}
    ordered_cols = [col_of_feature[f] for f in geom['test_feature_names']]
    R = data[ordered_cols].apply(pd.to_numeric, errors='coerce')
    ok = R.notna().all(axis=1)
    if (~ok).any():
        print(f"load_study6_prereg: dropped {(~ok).sum()} further rows (unparseable ratings)")
    data, R = data[ok], R[ok]

    responses_cond = {c: jnp.array(_clip_ratings(R[data['condition'] == c].values))
                      for c in STUDY6_U_OF_COND}
    return geom, responses_cond


# ---------------------------------------------------------------------------
# Study 8
# ---------------------------------------------------------------------------
def load_study8(csv_path='../../data/study8_prereg.csv', features_pkl_path=SET1_PKL):
    """(geom, responses_cond, groups_df) for study 8 (preregistered data).

    One likelihood group per unique feature->utterance configuration (all
    participants who saw exactly the same statements about the same features,
    order-insensitive). Baseline participants (no statements) form one group.
    Group keys are deterministic: 'baseline' or '<num_generics>of<total_utt>_g<idx>'
    with idx assigned in sorted-configuration order.

    groups_df: one row per group with condition / num_generics / total_utt /
    n_participants / the config itself, for aggregating predictions later.
    """
    df, coords = load_set1(features_pkl_path)
    geom = _base_geom(df, coords)

    data = pd.read_csv(csv_path, low_memory=False).iloc[2:].copy()   # 2 Qualtrics header rows

    # exclusions as in study8_analysis.Rmd
    n0 = len(data)
    attn = pd.to_numeric(data['attn_check_1'], errors='coerce')
    keep = (attn == 100) & (data['AI'] == 'No') & (data['check_task'] == STUDY8_CHECK_TASK_ANSWER)
    data = data[keep]
    print(f"load_study8: {len(data)} of {n0} rows kept after exclusions")

    rating_cols = list(STUDY8_COL_TO_FEATURE.keys())
    col_of_feature = {v: k for k, v in STUDY8_COL_TO_FEATURE.items()}
    ordered_cols = [col_of_feature[f] for f in geom['test_feature_names']]
    R_all = data[ordered_cols].apply(pd.to_numeric, errors='coerce')
    ok = R_all.notna().all(axis=1)
    if (~ok).any():
        print(f"load_study8: dropped {(~ok).sum()} further rows (unparseable ratings)")
    data, R_all = data[ok], R_all[ok]

    # per-participant configuration: sorted tuple of (train feature, u)
    def config_of(row):
        if row['condition'] == 'baseline':
            return ()
        feats = str(row['training_features_order']).split('|')
        us = str(row['training_structure'])
        if len(feats) != len(us):
            # one known Qualtrics glitch: a feature token repeated in the order
            # string; dedupe (order-preserving) and only accept a clean resolve
            feats = list(dict.fromkeys(feats))
            assert len(feats) == len(us), f"structure/order length mismatch: {row.name}"
            print(f"load_study8: deduped repeated training tokens for row {row.name}")
        return tuple(sorted((STUDY8_TOKEN_TO_FEATURE[f], 0 if u == 'G' else 1)
                            for f, u in zip(feats, us)))

    configs = data.apply(config_of, axis=1)

    # deterministic group keys
    unique_cfgs = sorted(set(configs), key=lambda cfg: (len(cfg), cfg))
    key_of_cfg, meta = {}, []
    for i, cfg in enumerate(unique_cfgs):
        if len(cfg) == 0:
            key = 'baseline'
        else:
            n_gen = sum(1 for _, u in cfg if u == 0)
            key = f"{n_gen}of{len(cfg)}_g{i:03d}"
        key_of_cfg[cfg] = key

    geom['x_train_cond'], geom['u_train_cond'], geom['train_names_cond'] = {}, {}, {}
    responses_cond = {}
    for cfg in unique_cfgs:
        key = key_of_cfg[cfg]
        names = [f for f, _ in cfg]
        geom['x_train_cond'][key] = (jnp.array(np.stack([coords[f] for f in names]))
                                     if names else jnp.zeros((0, 2)))
        geom['u_train_cond'][key] = jnp.array([u for _, u in cfg], dtype=jnp.int32)
        geom['train_names_cond'][key] = names
        mask = (configs == cfg).values
        responses_cond[key] = jnp.array(_clip_ratings(R_all[mask].values))
        meta.append({'group': key,
                     'condition': 'baseline' if len(cfg) == 0
                                  else f"{sum(1 for _, u in cfg if u == 0)}/{len(cfg)}",
                     'num_generics': sum(1 for _, u in cfg if u == 0),
                     'total_utt': len(cfg),
                     'n_participants': int(mask.sum()),
                     'config': cfg})
    groups_df = pd.DataFrame(meta)
    print(f"load_study8: {len(unique_cfgs)} groups "
          f"({groups_df['n_participants'].sum()} participants)")
    return geom, responses_cond, groups_df


def literal_study8(responses_cond, groups_df, merge=False, features_pkl_path=SET1_PKL):
    """Literal-listener lesion of study 8: specifics carry ZERO evidence.

    Each group's config is reduced to its generic-only feature set. With
    merge=False (default, STRUCTURE-MATCHED): the original 403-group structure is
    kept — every group keeps its own coherence field, just with the specific
    utterances stripped — so summed log Z is directly comparable to the full
    model's. With merge=True: groups whose generic sets coincide are pooled under
    one shared field (identical evidence -> shared draw; all 0-generic conditions
    collapse into 'baseline'), which is the purer generative story but changes
    the likelihood's grouping structure, so its log Z is NOT comparable to the
    full fit's. Predictions at a given theta are identical either way.

    Returns (geom_lit, responses_lit, key_map) with key_map mapping original
    group key -> literal group key (identity when merge=False).
    """
    df, coords = load_set1(features_pkl_path)
    geom_lit = _base_geom(df, coords)

    lit_cfg_of = {row['group']: tuple(sorted(f for f, u in row['config'] if u == 0))
                  for _, row in groups_df.iterrows()}

    geom_lit['x_train_cond'], geom_lit['u_train_cond'], geom_lit['train_names_cond'] = {}, {}, {}
    responses_lit = {}

    def add_group(key, names, resp):
        geom_lit['x_train_cond'][key] = (jnp.array(np.stack([coords[f] for f in names]))
                                         if names else jnp.zeros((0, 2)))
        geom_lit['u_train_cond'][key] = jnp.zeros(len(names), dtype=jnp.int32)   # all generic
        geom_lit['train_names_cond'][key] = names
        responses_lit[key] = resp

    if merge:
        unique = sorted(set(lit_cfg_of.values()), key=lambda c: (len(c), c))
        key_of = {cfg: ('baseline' if not cfg else f"{len(cfg)}g_{i:03d}")
                  for i, cfg in enumerate(unique)}
        for cfg in unique:
            gs = [g for g, c in lit_cfg_of.items() if c == cfg]
            add_group(key_of[cfg], list(cfg),
                      jnp.concatenate([responses_cond[g] for g in gs], axis=0))
        key_map = {g: key_of[c] for g, c in lit_cfg_of.items()}
    else:
        for g, cfg in lit_cfg_of.items():
            add_group(g, list(cfg), responses_cond[g])
        key_map = {g: g for g in lit_cfg_of}
    print(f"literal_study8(merge={merge}): {len(responses_lit)} literal groups "
          f"(from {len(lit_cfg_of)})")
    return geom_lit, responses_lit, key_map
