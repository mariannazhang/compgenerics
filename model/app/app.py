# import packages for app
import streamlit as st

# import packages for model
from gorgo import infer, condition, draw_from, flip, keep_deterministic, mem
from gorgo.hashable import hashabledict

from collections import namedtuple
import math

# import packages for manipulating data
import pandas as pd

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
    "love to eat flowers",
    "have stripes in their hair",
    "can bounce a ball on their heads",
    "like to sing",
    "climb tall fences",
    "flap their arms when they are happy",
    "have freckles on their feet",
    "hop over puddles", 
    "hate walking in the mud", 
    "draw stars on their knees",
    "can flip in the air",
    "are scared of ladybugs",
    "hate ice cream",
    "chase shadows",
    "babies are wrapped in orange blankets",
    "sleep in tall trees"
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
    
    st.write("# Speaker")
    st.write("independent on each trial")
    
    speaker_features_under_consideration = st.selectbox('''**Speaker features under consideration**   
                                                        speaker is trying to align listener's beliefs about whether these are kind-linked or not''', 
                                                         options = ["observed kind-linked features only", "all kind-linked features"]) 
    
    possible_utterances = st.multiselect('''**Possible utterances**   
                                         possible utterances the speaker can say''', 
                                        options = ["Zarpies", "This Zarpie", "silence"],
                                        default = ["Zarpies", "This Zarpie"]) 
    
    # note: streamlit doesn't support latex in options, might be a way to use format_func to alter display
    speaker_weight = st.selectbox('''**Speaker weight**  
                                        note: utility^beta may have issues when utility returns 0''', 
                                  options = ["e^(utility*beta)", "utility^beta"]) 
    # warn users about utility^beta speaker weight
    if speaker_weight == "utility^beta":
        st.warning("This speaker weight will throw an error if the utility function returns 0.",
                   icon="⚠️")
    
    inv_temp = st.slider('''**Speaker rationality/inverse temperature** ($\\beta$)  
                              how much speaker cares about maximizing listener belief''', 
                              0.0, 100.0, 5.0)
    
    
    st.write("# Listener")
    
    listener_features_under_consideration = st.selectbox('''**Listener features under consideration**   
                                                         listener is considering whether these are kind-linked or not''', 
                                                         options = ["observed features so far", "all observed features", "all observed features and more"]) 
    
    if listener_features_under_consideration == "all observed features and more":
        all_features_raw = st.text_input('''**Enter all observed and more features**    
                                            a comma-separated list; must contain all observed features''', 
                                            value = ", ".join(all_features_initial))
        # parse raw input as list
        all_features = all_features_raw.split(', ')
        all_features = [str(n) for n in all_features]
        # give users a check that "observed and more features" was parsed as intended
        st.info(f"You have inputted {len(all_features)} features.",
                icon="ℹ️")
        
    else:
        all_features = all_features_initial
    
    st.write('''**Observed data**  
                each trial is independent speaker and independent example''')
    
    # small test
    data_initial = pd.DataFrame(
        [
            {"utterance_subject": "Zarpies", "utterance_feature": "love to eat flowers", 
             "observed_features": "love to eat flowers"},
            {"utterance_subject": "This Zarpie", "utterance_feature": "have stripes in their hair", 
             "observed_features": "have stripes in their hair"}
        ]
    )
    
    # # generic to induction study 6 (CogSci draft) - generic condition
    # data_initial = pd.DataFrame(
    #     [
    #         {"utterance_subject": "Zarpies", "utterance_feature": "love to eat flowers", 
    #          "observed_features": "love to eat flowers"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "have stripes in their hair", 
    #          "observed_features": "have stripes in their hair"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "can bounce a ball on their heads", 
    #          "observed_features": "can bounce a ball on their heads"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "like to sing", 
    #          "observed_features": "like to sing"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "climb tall fences", 
    #          "observed_features": "climb tall fences"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "flap their arms when they are happy", 
    #          "observed_features": "flap their arms when they are happy"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "have freckles on their feet", 
    #          "observed_features": "have freckles on their feet"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "hop over puddles", 
    #          "observed_features": "hop over puddles"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "hate walking in the mud", 
    #          "observed_features": "hate walking in the mud"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "draw stars on their knees", 
    #          "observed_features": "draw stars on their knees"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "can flip in the air", 
    #          "observed_features": "can flip in the air"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "are scared of ladybugs", 
    #          "observed_features": "are scared of ladybugs"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "hate ice cream", 
    #          "observed_features": "hate ice cream"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "chase shadows", 
    #          "observed_features": "chase shadows"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "babies are wrapped in orange blankets", 
    #          "observed_features": "babies are wrapped in orange blankets"},
    #         {"utterance_subject": "Zarpies", "utterance_feature": "sleep in tall trees", 
    #          "observed_features": "sleep in tall trees"}
    #     ]
    # )
    
    # # generic to induction study 6 (CogSci draft) - specific condition
    # data_initial = pd.DataFrame(
    #     [
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "love to eat flowers", 
    #          "observed_features": "love to eat flowers"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "have stripes in their hair", 
    #          "observed_features": "have stripes in their hair"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "can bounce a ball on their heads", 
    #          "observed_features": "can bounce a ball on their heads"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "like to sing", 
    #          "observed_features": "like to sing"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "climb tall fences", 
    #          "observed_features": "climb tall fences"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "flap their arms when they are happy", 
    #          "observed_features": "flap their arms when they are happy"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "have freckles on their feet", 
    #          "observed_features": "have freckles on their feet"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "hop over puddles", 
    #          "observed_features": "hop over puddles"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "hate walking in the mud", 
    #          "observed_features": "hate walking in the mud"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "draw stars on their knees", 
    #          "observed_features": "draw stars on their knees"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "can flip in the air", 
    #          "observed_features": "can flip in the air"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "are scared of ladybugs", 
    #          "observed_features": "are scared of ladybugs"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "hate ice cream", 
    #          "observed_features": "hate ice cream"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "chase shadows", 
    #          "observed_features": "chase shadows"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "babies are wrapped in orange blankets", 
    #          "observed_features": "babies are wrapped in orange blankets"},
    #         {"utterance_subject": "This Zarpie", "utterance_feature": "sleep in tall trees", 
    #          "observed_features": "sleep in tall trees"}
    #     ]
    # )
    
    st.session_state['save_data'] = data_initial
    data_df = st.data_editor(
        data_initial,
        column_config = {
            "utterance_subject": st.column_config.TextColumn(
                "utterance subject",
                help = "must be one of the possible utterances",
                width = "small",
                required = True
            ),
            # FIXME: whole df resets when possible utterances is changed. 
            # likely requires a session state issue
            # "utterance_subject": st.column_config.SelectboxColumn(
            #     "utterance subject",
            #     help = "must be one of the possible utterances",
            #     options = possible_utterances,
            #     width = "small",
            #     required = True
            # ),
            "utterance_feature": st.column_config.SelectboxColumn(
                "utterance feature",
                help = "must be one of the example features",
                options = all_features,
                required = True
            ),
            "observed_features": st.column_config.TextColumn(
                "observed features",
                help = "features of the observed group member; must include the utterance feature",
                width = "long",
                required = True
            ),
        },
        hide_index = True,
        num_rows = "dynamic"
    )
    
    ###################
    # DATA VALIDATION
    ###################
    
    # check entire df for empty strings; throw error if any are detected
    if data_df.map(lambda x: x == '').any().any():
        st.error("Please fill out every row.",
                 icon="🚨")
    
    # checking trial structure
    for i, row in data_df.iterrows():
        row_utterance_subject = row['utterance_subject']
        row_observed_features = row['observed_features'].split(', ')
        row_utterance_feature = row['utterance_feature']
        
        if row_utterance_subject not in possible_utterances:
            st.error("The utterance subject must be one of the possible utterances set in the speaker model.",
                        icon="🚨")
        
        # if user selects "observed and more features",
        # check that observed features for every trial are in "observed and more features"
        for feature in row_observed_features:
            if listener_features_under_consideration == "all observed features and more" and feature not in all_features:
                st.error("The observed and more features must include all observed features.",
                            icon="🚨")
                
        # check that utterance feature is in observed features for every trial; throw error if not
        if row_utterance_feature not in row_observed_features:
            st.error("The observed features must include the utterance feature for a given trial.",
                     icon="🚨")

    
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

# convert to tuple
possible_utterances = tuple(possible_utterances)

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
            elif "Zarpies" in utt.subj: # generic
                return utt.feature in kind.features   # generic is true = mentioned feature is in list of kind-linked features
            elif "silence" in utt.subj: # silence
                return True # silence is always true
            else: # other
                return True
        
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
        ## set up toy data
        example_zarpie = Instance(kind = "Zarpie", features = ("kind-linked feature", "idiosyncratic feature"))
        zarpies = Kind("Zarpies", "kind-linked feature")
        
        ## construct space of possible utterances
        all_utts_list = []
        for subj in possible_utterances:
            for feat in example_zarpie.features + ("not an example feature",):
                all_utts_list.append((Utterance(subj = subj, feature = feat), example_zarpie))
        
        ## get meaning for each utterance
        semantic_likelihood = []
        for pair in all_utts_list:
            utt = pair[0]
            inst = pair[1]
            new_row = {'Utterance subject': utt.subj, 'Utterance feature': utt.feature, 
                       'Likelihood': meaning(zarpies, inst, utt)}
            semantic_likelihood.append(new_row)
        semantic_likelihood = pd.DataFrame(semantic_likelihood)
        semantic_likelihood['Utterance'] = "\"" + semantic_likelihood['Utterance subject'] + ": " + semantic_likelihood['Utterance feature'] + "\""
        
        ## plot semantic likelihood
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
                inst = pair[1]
                
                # only seen one trial
                observed_features_so_far = inst.features
                observed_features = observed_features_so_far
                
                # NOTE: restricted to observed features from data                               # e.g., example instance has (kind-linked, idiosyncratic) features --> similarity = 1/3
                return jaccard_similarity(set(kind_features) & set(observed_features),          # (kind) & (kind, idio) = (kind)
                                          set(x["zarpie_features"]) & set(observed_features))   # (lit list infer from generic: kind, idio) & (kind, idio) = (kind, idio)
            
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
        
        utt = Utterance(subj = draw_from(possible_utterances),              # consider saying generic or specific..
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
                               **kwargs):
                            #    listener_features_under_consideration = listener_features_under_consideration,
                            #    possible_utterances = possible_utterances,
                            #    speaker_features_under_consideration = speaker_features_under_consideration,
                            #    speaker_weight = speaker_weight, inv_temp = inv_temp):
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
                                                         **kwargs)
                    
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
                                                         **kwargs)
                    
                    # condition on speaker likelihood
                    condition(utt_likelihood)
                    
                # define zarpie concept to have those features
                zarpie_features = some_features
                
                return hashabledict(zarpie_features = zarpie_features, coherence = coherence)
                
                




        st.subheader("Posterior")
        
        # run pragmatic_listener with sidebar parameters
        dist = pragmatic_listener(data, 
                                  listener_features_under_consideration = listener_features_under_consideration,
                                  possible_utterances = possible_utterances,
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
        ## set up toy data
        example_zarpie = Instance(kind = "Zarpie", features = ("kind-linked feature", "idiosyncratic feature"))
        zarpies = Kind("Zarpies", "kind-linked feature")
        
        ## construct space of possible utterances
        all_utts_list = []
        for subj in possible_utterances:
            for feat in example_zarpie.features + ("not an example feature",):
                all_utts_list.append((Utterance(subj = subj, feature = feat), example_zarpie))
        
        ## get speaker likelihood for each utterance
        speaker_likelihood = []
        for pair in all_utts_list:
            utt = pair[0]
            inst = pair[1]
            dist = speaker(kind_features = zarpies.features, observed_instance = inst, 
                           listener_features_under_consideration = listener_features_under_consideration,
                           speaker_features_under_consideration = speaker_features_under_consideration,
                           possible_utterances = possible_utterances,
                           inv_temp = inv_temp, speaker_weight = speaker_weight)
            
            utt_likelihood = dist.prob(Utterance(subj = utt.subj, feature = utt.feature))
            new_row = {'Utterance subject': utt.subj, 'Utterance feature': utt.feature, 
                       'Likelihood': utt_likelihood}
            speaker_likelihood.append(new_row)
            
        # convert to df, assemble full utterance for plotting
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