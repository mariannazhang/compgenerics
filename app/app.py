# import packages for app
import streamlit as st

# import packages for model
from gorgo import infer, condition, draw_from, flip, keep_deterministic, mem
from gorgo.hashable import hashabledict

from collections import namedtuple
import math

# import packages for manipulating data
import pandas as pd
import ast

# import packages for plotting
import seaborn as sns
from matplotlib import pyplot as plt


############
# PAGE TITLE
############

st.title("Computational generics model")



############
# SETUP
############

# define named tuples
Utterance = namedtuple("Utterance", ("subj", "feature"))
Instance = namedtuple("Instance", ("kind", "features"))
Kind = namedtuple("Kind", ("name", "features"))

# initialize all features
all_features_initial = [
    "eats flowers",
    "scared of shadows", 
    "jumps over puddles", 
    "sings lovely songs",
    "have stripes in their hair",
    "bounce a ball on their heads",
    "climb tall fences"    
]

# helper functions
@keep_deterministic
def dist_to_df(dist):
    df_dist = pd.DataFrame(dist.support)
    df_dist['Probability'] = dist.probabilities
    return df_dist

@keep_deterministic
def get_observed_features(data: tuple[tuple[Utterance, Instance]]):
    every_observed_feature = []
    for pair in data:
        inst = pair[1]
        for feature in list(inst.features):             # list in case of tuple element 1
            every_observed_feature.append(feature)
    observed_features = set(every_observed_feature)     # remove duplicate features
    return observed_features

# memo-ized, so that feeding in same feature returns same kind-linked status each time
@mem
def is_zarpie_feature(feature, coherence):
    return flip(coherence)

############
# SIDEBAR
############

with st.sidebar:
    
    st.write("# Listener")
    
    listener_features_under_consideration = st.selectbox('''**Listener features under consideration**   
                                                         listener is considering whether these are kind-linked or not''', 
                                                         options = ["observed features so far", "all observed features", "all observed features and more"]) 
    
    if listener_features_under_consideration == "all observed features and more":
        all_features = ast.literal_eval(st.text_input('''**Enter all observed and more features**    
                                                        must contain all observed features''', 
                                                        value = all_features_initial))
    else:
        all_features = all_features_initial
    
    st.write('''**Observed data**  
                each trial is independent speaker and independent example''')
    
    data_initial = pd.DataFrame(
        [
            {"utterance_subject": "Zarpies", "utterance_feature": "eats flowers", 
             "observed_features": "eats flowers, sings lovely songs"},
            {"utterance_subject": "This Zarpie", "utterance_feature": "scared of shadows", 
             "observed_features": "scared of shadows"}
        ]
    )
    
    data_df = st.data_editor(
        data_initial,
        column_config = {
            "utterance_subject": st.column_config.SelectboxColumn(
                "utterance subject",
                help = "specific or generic",
                options = ["This Zarpie", "Zarpies"],
                width = "small",
                required = True
            ),
            "utterance_feature": st.column_config.SelectboxColumn(
                "utterance feature",
                help = "must be one of the example features",
                options = all_features,
                required = True
            ),
            "observed_features": st.column_config.TextColumn(
                "observed features",
                help = "features of the observed group member; must include the utterance feature",
                width = "long"
            ),
        },
        hide_index = True,
        num_rows = "dynamic"
    )
    
    st.write("# Speaker")
    st.write("independent on each trial")
    
    speaker_features_under_consideration = st.selectbox('''**Speaker features under consideration**   
                                                        speaker is trying to align listener's beliefs about whether these are kind-linked or not''', 
                                                         options = ["observed kind-linked features only", "all kind-linked features"]) 
    
    # note: streamlit doesn't support latex in options, might be a way to use format_func to alter display
    speaker_weight = st.selectbox('''**Speaker weight**  
                                        note: utility^beta may have issues when utility returns 0''', 
                                  options = ["e^(utility*beta)", "utility^beta"]) 
    
    inv_temp = st.slider('''**Speaker rationality/inverse temperature** ($\\beta$)  
                              how much speaker cares about maximizing listener belief''', 
                              0.0, 100.0, 5.0)
    
    
#####################
# DATA VALIDATION
#####################

    # FIXME: check that the utterance feature is in the observed features, for each trial
    # FIXME: check the observed features is formatted as a valid list, for each trial
    # FIXME: check observed and more features is formatted as a valid list
    # FIXME: check observed and more features contains the observed features across all trials
    
#####################
# CLEANING DATA
#####################

# convert to tuple of namedtuples
data = []

for i, row in data_df.iterrows():
    utt = Utterance(subj = row['utterance_subject'], feature = row['utterance_feature'])
    inst = Instance(kind = "Zarpie", features = tuple(row['observed_features'].split(', ')))
    data.append((utt, inst))

data = tuple(data)


#####################
# MODELS
#####################
tab1, tab2 = st.tabs(["Listener", "Speaker"])

with tab1:
    
    col1, col2 = st.columns(2)
    
    #####################
    # LITERAL LISTENER
    #####################

    with col1:

        st.header("Literal listener")
        
        # meaning function
        def meaning(kind: Kind, inst: Instance, utt: Utterance):
            if "This" in utt.subj: # specific
                return utt.feature in inst.features  # specific is true = mentioned feature is in list of example instance's features
            else: # generic
                return utt.feature in kind.features   # generic is true = mentioned feature is in list of kind-linked features
        
        # memo-ized, so that feeding in same feature returns same kind-linked status each time
        @mem
        def is_zarpie_feature(feature, coherence):
            return flip(coherence)
        
        # literal listener
        @infer
        def literal_listener(data: tuple[tuple[Utterance, Instance]], 
                             listener_features_under_consideration = listener_features_under_consideration,
                             **kwargs):
            
            # sample a coherence
            coherence = draw_from([.1, .2, .3, .4, .5, .6, .7, .8, .9])
            
            some_features = ()
            
            if listener_features_under_consideration == "observed features so far":
                # incrementally build out feature list from data
                observed_features_so_far = ()
                zarpie_features_so_far = ()
                
                # for each utterance-instance pair in data:
                for pair in data:
                    utt = pair[0]
                    inst = pair[1]
                    
                    # flip observed features & add to feature list
                    for feature in inst.features:
                        observed_features_so_far = observed_features_so_far + (feature, )
                        if(is_zarpie_feature(feature, coherence)):
                            zarpie_features_so_far = zarpie_features_so_far + (feature, )
                    
                    # convert to tuple; set discards duplicate features
                    observed_features_so_far = tuple(set(observed_features_so_far))
                    zarpie_features_so_far = tuple(set(zarpie_features_so_far))
                    
                    # define zarpie concept to have those features
                    zarpies = Kind("Zarpies", zarpie_features_so_far)
                
                    semantic_likelihood = meaning(kind = zarpies, inst = inst, utt = utt)   # get truth-value of utterance
                    condition(.95 if semantic_likelihood else .05)                          # condition softly on truth-value (sometimes speaker misspeaks and feature is not true)
                
                zarpie_features = zarpie_features_so_far
                
                return hashabledict(zarpie_features = zarpie_features, coherence = coherence)
                
                
            else:
            # if listener_features_under_consideration == "all observed features" or \
            # listener_features_under_consideration == "all observed features and more":
                
                if listener_features_under_consideration == "all observed features":
                    observed_features = get_observed_features(data)
                    features_considered = tuple(observed_features)
                    
                else:
                # if listener_features_under_consideration == "all observed features and more":
                    features_considered = tuple(all_features)
                    
                
                
                # draw a subset of all features, based on coherence
                # coherence = likelihood of sampling a given feature, higher coherence ~ more kind-linked features
                some_features = tuple([f for f in features_considered if flip(coherence)])
                
                # define zarpie concept to have those features
                zarpies = Kind("Zarpies", some_features)
                
                # for each utterance-instance pair in data:
                for pair in data:
                    utt = pair[0]
                    inst = pair[1]
                    semantic_likelihood = meaning(kind = zarpies, inst = inst, utt = utt)   # get truth-value of utterance
                    condition(.95 if semantic_likelihood else .05)                          # condition softly on truth-value (sometimes speaker misspeaks and feature is not true)
                
                zarpie_features = some_features
                
                return hashabledict(zarpie_features = zarpie_features, coherence = coherence)

        
        st.subheader("Posterior")

        # run literal listener
        dist = literal_listener(data,
                                listener_features_under_consideration = listener_features_under_consideration)

        # convert inferred set of zarpie features & coherence to df
        df = dist_to_df(dist)

        # separate inferred zarpie features vs inferred coherence
        df_coherence = df.groupby('coherence').agg({'Probability': 'sum'})
        df_zarpie_features = df.groupby('zarpie_features').agg({'Probability': 'sum'}).reset_index()
        
        print("yes")

        # convert tuples of features to string, sort by number of features
        df_zarpie_features['zarpie_features_string'] = [', '.join(el) for el in df_zarpie_features['zarpie_features']] 
        df_zarpie_features['zarpie_features_count'] = df_zarpie_features['zarpie_features'].apply(len)
        df_zarpie_features = df_zarpie_features.sort_values(by='zarpie_features_count')

        plt.clf()
        
        # plot coherence
        g = sns.lineplot(data = df_coherence,
                        x = "coherence",
                        y = "Probability")
        plt.xlim(.1, .9)
        plt.ylim(0, .2)
        plt.suptitle("inferred coherence")

        st.pyplot(g.figure)
        plt.clf()
        
        # plot number of inferred features
        df_zarpie_features['zarpie_features_count'] = df_zarpie_features['zarpie_features_count'].astype(str)
        g = sns.barplot(data = df_zarpie_features.groupby("zarpie_features_count").aggregate("sum").reset_index(),
                        x = "Probability",
                        y = "zarpie_features_count")
        plt.ylabel("number of kind-linked features")
        plt.suptitle("inferred number of Zarpie features")
        
        st.pyplot(g.figure)
        plt.clf()

        # plot inferred features
        g = sns.barplot(data = df_zarpie_features,
                        x = "Probability",
                        y = "zarpie_features_string")
        plt.ylabel("Zarpie features")
        plt.suptitle("inferred Zarpie features")

        st.pyplot(g.figure)
        plt.clf()
        
        
        st.subheader("Likelihood")
        
        # get semantic likelihood of various utterances
        data_example = (
            (Utterance(subj = "Zarpies", feature = "kind-linked feature"), Instance(kind = "Zarpie", features = ("kind-linked feature", ))),
            (Utterance(subj = "Zarpies", feature = "not kind-linked feature"), Instance(kind = "Zarpie", features = ("not kind-linked feature", ))),
            (Utterance(subj = "This Zarpie", feature = "example feature"), Instance(kind = "Zarpie", features = ("example feature", ))),
            (Utterance(subj = "This Zarpie", feature = "not an example feature"), Instance(kind = "Zarpie", features = ("example feature", )))
            )
        
        zarpies = Kind("Zarpies", "kind-linked feature")
        
        semantic_likelihood = []
        for pair in data_example:
            utt = pair[0]
            inst = pair[1]
            new_row = {'Utterance subject': utt.subj, 'Utterance feature': utt.feature, 
                       'Likelihood': meaning(zarpies, inst, utt)}
            semantic_likelihood.append(new_row)
        semantic_likelihood = pd.DataFrame(semantic_likelihood)
        semantic_likelihood['Utterance'] = "\"" + semantic_likelihood['Utterance subject'] + ": " + semantic_likelihood['Utterance feature'] + "\""
        
        # plot semantic likelihood
        g = sns.barplot(data = semantic_likelihood,
                        x = "Likelihood",
                        y = "Utterance")
        plt.suptitle("Semantic likelihood")
        plt.xlim(0, 1)
        st.pyplot(g.figure)
        plt.clf()
        
        
        
        
        st.subheader("Prior")
        
        
        
        
        
with tab2:
    st.header("Speaker 1")

    #####################
    # SPEAKER
    #####################

    def jaccard_similarity(set_a, set_b):
        if len(set_a) == 0 and len(set_b) == 0:
            return 1 # if both lists are empty, return 1
        else:
            return len(set.intersection(set_a, set_b)) \
                / len(set.union(set_a, set_b))

    @keep_deterministic
    def utility_func(pair: tuple[Utterance, Instance],
                     kind_features: tuple[str, ...],
                     **kwargs):
        # run literal listener
        data = (pair, )
        dist = literal_listener(data,
                                **kwargs)
        
        # compare similarity of inferred kind-linked features to true kind features
        def f(x):
            
            if speaker_features_under_consideration == "observed kind-linked features only":
                observed_features_so_far = []
            
                utt = pair[0]
                inst = pair[1]
                
                # only seen one trial
                observed_features_so_far = inst.features
                observed_features = observed_features_so_far
                
                # NOTE: restricted to observed features from data
                return jaccard_similarity(set(kind_features) & set(observed_features), 
                                          set(x["zarpie_features"]) & set(observed_features))
            
            if speaker_features_under_consideration == "all kind-linked features":
                # NOTE: across all kind-linked features known to the speaker
                return jaccard_similarity(set(kind_features), 
                                          set(x["zarpie_features"]))
        
        # calculate expected value over all sets of features, collapsing across coherence
        exp_utility = dist.expected_value(f)
        
        return exp_utility

    @infer
    def speaker(kind_features: Kind.features, observed_instance: Instance, 
                **kwargs):
        
        utt = Utterance(subj = draw_from(["Zarpies", "This Zarpie"]),       # consider saying generic or specific..
                        feature = draw_from(observed_instance.features))    # ..about some feature of observed instance
        utility = utility_func((utt, observed_instance), 
                               kind_features,
                               **kwargs)  # consider what literal listener will infer (about zarpie features, coherence) from utterance                                                                    # how close is literal listener's inferences to true zarpie features
                                    
        if speaker_weight == "e^(utility*beta)":
            weight = math.exp(inv_temp*utility)                             # avoids issues with 0 utility
        else:
            if speaker_weight == "utility^beta":
                weight = utility**inv_temp                                  # maximize utility, with inv_temp rationality
        condition(weight)
        
        return utt

with tab1:
    #####################
    # PRAGMATIC LISTENER
    #####################

    with col2:

        st.header("Pragmatic listener")

        # this needs to be kept deterministic since 
        # marginalize is being transformed, which we do not want
        @keep_deterministic
        def calc_utt_likelihood(features, inst, utt, 
                                **kwargs):
            # run speaker model
            dist = speaker(features, inst, 
                           **kwargs)
            # what is likelihood of saying the utterance heard (over all coherences)
            utt_likelihood = dist.marginalize(lambda x: utt == x).prob(True)
            return utt_likelihood
        
        # pragmatic listener
        @infer
        def pragmatic_listener(data: tuple[tuple[Utterance, Instance]], 
                               listener_features_under_consideration,
                               speaker_features_under_consideration,
                               speaker_weight, inv_temp):
            # sample a coherence
            coherence = draw_from([.1, .2, .3, .4, .5, .6, .7, .8, .9])
            
            
            
            if listener_features_under_consideration == "observed features so far":
                # incrementally build out feature list from data
                observed_features_so_far = ()
                zarpie_features_so_far = ()
                
                # for each utterance-instance pair in data:
                for pair in data:
                    utt = pair[0]
                    inst = pair[1]
                    
                    # flip observed features & add to feature list
                    for feature in inst.features:
                        observed_features_so_far = observed_features_so_far + (feature, )
                        if(is_zarpie_feature(feature, coherence)):
                            zarpie_features_so_far = zarpie_features_so_far + (feature, )
                    
                    # convert to tuple; set discards duplicate features
                    observed_features_so_far = tuple(set(observed_features_so_far))
                    zarpie_features_so_far = tuple(set(zarpie_features_so_far))
                    
                    # define zarpie concept to have those features
                    zarpies = Kind("Zarpies", zarpie_features_so_far)
                
                    # get speaker likelihood
                    utt_likelihood = calc_utt_likelihood(zarpies.features, inst, utt, 
                                                         listener_features_under_consideration = listener_features_under_consideration,
                                                         speaker_features_under_consideration = speaker_features_under_consideration,
                                                         speaker_weight = speaker_weight, inv_temp = inv_temp)
                    
                    # condition on speaker likelihood
                    condition(utt_likelihood)
                
                # define zarpie concept to have those features
                zarpie_features = zarpie_features_so_far
                
                return hashabledict(zarpie_features = zarpie_features, coherence = coherence)
                
            else:
            # if listener_features_under_consideration == "all observed features" or \
            # listener_features_under_consideration == "all observed features and more":
                
                if listener_features_under_consideration == "all observed features":
                    observed_features = get_observed_features(data)
                    features_considered = tuple(observed_features)
                
                else:
                # if listener_features_under_consideration == "all observed features and more":
                    features_considered = tuple(all_features)
                
                # draw a subset of all features, based on coherence
                # coherence = likelihood of sampling a given feature, higher coherence ~ more kind-linked features
                some_features = tuple([f for f in features_considered if flip(coherence)])
                
                # define zarpie concept to have those features
                zarpies = Kind("Zarpies", some_features)
                
                # for each utterance-instance pair in data:
                for pair in data:
                    utt = pair[0]
                    inst = pair[1]
                    
                    # get speaker likelihood
                    utt_likelihood = calc_utt_likelihood(zarpies.features, inst, utt, 
                                                         listener_features_under_consideration = listener_features_under_consideration,
                                                         speaker_features_under_consideration = speaker_features_under_consideration,
                                                         speaker_weight = speaker_weight, inv_temp = inv_temp)
                    
                    # condition on speaker likelihood
                    condition(utt_likelihood)
                    
                # define zarpie concept to have those features
                zarpie_features = some_features
                
                return hashabledict(zarpie_features = zarpie_features, coherence = coherence)
                
                




        st.subheader("Posterior")
        
        # run pragmatic_listener
        dist = pragmatic_listener(data, 
                                  listener_features_under_consideration = listener_features_under_consideration,
                                  speaker_features_under_consideration = speaker_features_under_consideration,
                                  speaker_weight = speaker_weight, inv_temp = inv_temp)

        # convert inferred set of zarpie features & coherence to df
        df = dist_to_df(dist)

        # separate inferred zarpie features vs inferred coherence
        df_coherence = df.groupby('coherence').agg({'Probability': 'sum'})
        df_zarpie_features = df.groupby('zarpie_features').agg({'Probability': 'sum'}).reset_index()

        # convert tuples of features to string, sort by number of features
        df_zarpie_features['zarpie_features_string'] = [', '.join(el) for el in df_zarpie_features['zarpie_features']] 
        df_zarpie_features['zarpie_features_count'] = df_zarpie_features['zarpie_features'].apply(len)
        df_zarpie_features = df_zarpie_features.sort_values(by='zarpie_features_count')

        # plot coherence
        g = sns.lineplot(data = df_coherence,
                        x = "coherence",
                        y = "Probability")
        plt.xlim(.1, .9)
        plt.ylim(0, .2)
        plt.suptitle("inferred coherence")

        st.pyplot(g.figure)
        plt.clf()
        
        # plot number of inferred features
        df_zarpie_features['zarpie_features_count'] = df_zarpie_features['zarpie_features_count'].astype(str)
        g = sns.barplot(data = df_zarpie_features.groupby("zarpie_features_count").aggregate("sum").reset_index(),
                        x = "Probability",
                        y = "zarpie_features_count")
        plt.ylabel("number of kind-linked features")
        plt.suptitle("inferred number of Zarpie features")
        
        st.pyplot(g.figure)
        plt.clf()
        
        # plot inferred features
        g = sns.barplot(data = df_zarpie_features,
                        x = "Probability",
                        y = "zarpie_features_string")
        plt.ylabel("Zarpie features")
        plt.suptitle("inferred Zarpie features")

        st.pyplot(g.figure)
        plt.clf()
        
        
        
        
        
        st.subheader("Likelihood")
        
        # get speaker likelihood of various utterances
        data_example = (
            (Utterance(subj = "Zarpies", feature = "kind-linked feature"), Instance(kind = "Zarpie", features = ("kind-linked feature", "non-kind linked example feature"))),
            (Utterance(subj = "Zarpies", feature = "non-kind linked feature"), Instance(kind = "Zarpie", features = ("kind-linked feature", "non-kind linked example feature"))),
            (Utterance(subj = "This Zarpie", feature = "example feature"), Instance(kind = "Zarpie", features = ("kind-linked feature", "non-kind linked example feature", "example feature"))),
            (Utterance(subj = "This Zarpie", feature = "not an example feature"), Instance(kind = "Zarpie", features = ("kind-linked feature", "non-kind linked example feature", "example feature")))
            )
        
        zarpies = Kind("Zarpies", "kind-linked feature")
        
        speaker_likelihood = []
        for pair in data_example:
            utt = pair[0]
            inst = pair[1]
            
            dist = speaker(kind_features = zarpies.features, observed_instance = inst, 
                           listener_features_under_consideration = listener_features_under_consideration,
                           speaker_features_under_consideration = speaker_features_under_consideration,
                           inv_temp = inv_temp, speaker_weight = speaker_weight)
            
            utt_likelihood = dist.prob(Utterance(subj = utt.subj, feature = utt.feature))
            new_row = {'Utterance subject': utt.subj, 'Utterance feature': utt.feature, 
                       'Likelihood': utt_likelihood}
            speaker_likelihood.append(new_row)
        speaker_likelihood = pd.DataFrame(speaker_likelihood)
        speaker_likelihood['Utterance'] = "\"" + speaker_likelihood['Utterance subject'] + ": " + speaker_likelihood['Utterance feature'] + "\""
        
        # plot speaker likelihood
        g = sns.barplot(data = speaker_likelihood,
                        x = "Likelihood",
                        y = "Utterance")
        plt.suptitle("Speaker likelihood")
        plt.xlim(0, 1)
        st.pyplot(g.figure)
        plt.clf()
        
        
        
        
        st.subheader("Prior")