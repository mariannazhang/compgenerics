from gorgo import infer, condition, draw_from, flip, keep_deterministic, mem
from gorgo.hashable import hashabledict
from gorgo.distributions.builtin_dists import Categorical, Beta

from collections import namedtuple
import numpy as np
import math

#####################
# SETUP
#####################

# define named tuples
Utterance = namedtuple("Utterance", ("subj", "feature"))
Instance = namedtuple("Instance", ("kind", "features"))
Kind = namedtuple("Kind", ("name", "features"))

# equivalent to:
# class Utterance:
#     def __init__(self, subj, feature):
#         self.subj = subj`
#         self.feature = feature`

## memo-ized, so that feeding in same feature returns same kind-linked status each time
@mem
def is_zarpie_feature(feature, coherence):
    return flip(coherence)



#####################
# LITERAL LISTENER
#####################

# meaning function
def meaning(kind: Kind, inst: Instance, utt: Utterance,
            lesioned_meaning = False):
    if "This" in utt.subj: # specific
        return utt.feature in inst.features  # specific is true = mentioned feature is in list of example instance's features
    elif "Zarpies" in utt.subj: # generic
        if lesioned_meaning:
            return utt.feature in inst.features # treat generic as specific in lesioned model 
        else:
            return utt.feature in kind.features   # generic is true = mentioned feature is in list of kind-linked features
    # elif "silence" in utt.subj: # silence
    #     return True
    else:
        return True

@infer
def literal_listener(data: tuple[tuple[Utterance, Instance]],
                     only_return_coherence = False,
                     prior = [.1, .2, .3, .4, .5, .6, .7, .8, .9],
                     **kwargs):
    # sample a coherence
    if isinstance(prior, list):
        coherence = draw_from(prior)
    else:
        coherence = prior.sample()

    # incrementally build out feature list from data
    # observed_features_so_far = ()
    zarpie_features_so_far = ()
        
    # for each utterance-instance pair in data:
    for pair in data:
        # get utterance and instance
        utt = pair[0]
        inst = pair[1]
        
        # flip observed features & add to feature list based on memo-ized coherence flip
        for feature in inst.features:
            # observed_features_so_far = observed_features_so_far + (feature, )
            if(is_zarpie_feature(feature, coherence)):
                zarpie_features_so_far = zarpie_features_so_far + (feature, )

        # convert to tuple; set discards duplicate features
        # observed_features_so_far = tuple(set(observed_features_so_far))
        zarpie_features_so_far = tuple(set(zarpie_features_so_far))
        
        # define zarpie concept to have those features
        zarpies = Kind("Zarpies", zarpie_features_so_far)
        
        # get truth-value of utterance
        semantic_likelihood = meaning(kind = zarpies, inst = inst, utt = utt,
                                      **kwargs)
        
        # condition softly on truth-value (sometimes speaker misspeaks and feature is not true)
        condition(.95 if semantic_likelihood else .05)
    
    # define zarpie concept to have those features
    zarpies = Kind("Zarpies", zarpie_features_so_far)
    
    if only_return_coherence:
        return coherence
    else:
        return hashabledict(zarpie_features = zarpies.features, coherence = coherence)
    # return "jumps over puddles" in zarpies.features


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
    dist = literal_listener((pair, ), 
                            **kwargs)
    
    inst = pair[1]
    
    # compare similarity of inferred kind-linked features to true kind features
    def f(x):
        
        # NOTE: restricted to features in the example instance
        return jaccard_similarity(set(kind_features) & set(inst.features), 
                                  set(x["zarpie_features"]) & set(inst.features))
        
        ## NOTE: across all kind-linked features known to the speaker
        # return jaccard_similarity(set(kind_features), 
        #                           set(x["zarpie_features"]))
    
    # calculate expected value over all sets of features, collapsing across coherence
    exp_utility = dist.expected_value(f)
    
    return exp_utility

@infer
def speaker(kind_features: Kind.features, observed_instance: Instance, 
            inv_temp = 20,
            **kwargs):
    
    # pick an utterance
    utt = Utterance(subj = draw_from(["Zarpies", "This Zarpie"]),       # consider saying generic or specific..
                    feature = draw_from(observed_instance.features))    # ..about some feature of observed instance
    # utt = Utterance(subj = draw_from(["Zarpies", "This Zarpie", "silence"]),  # consider saying generic or specific or silence..
    #                 feature = draw_from(observed_instance.features))    # ..about some feature of observed instance
    
    # calculate utility of utterance
    utility = utility_func((utt, observed_instance), 
                           kind_features,
                           **kwargs)                               # consider what literal listener will infer (about zarpie features, coherence) from utterance
                                                                        # how close is literal listener's inferences to true zarpie features
    # weight by utility, inv temp
    # weight = utility**inv_temp                                        # maximize utility, with inv_temp rationality
    weight = math.exp(inv_temp*utility)                                 # version to avoid issues with 0 utility
    condition(weight)
    
    return utt


#####################
# PRAGMATIC LISTENER
#####################

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

@infer
def pragmatic_listener(data: tuple[tuple[Utterance, Instance]],
                       only_return_coherence=False,
                       prior = [.1, .2, .3, .4, .5, .6, .7, .8, .9],
                       **kwargs):
    # sample a coherence
    if isinstance(prior, list):
        coherence = draw_from(prior)
    else:
        coherence = prior.sample()
    
    # incrementally build out feature list from data
    # observed_features_so_far = ()
    zarpie_features_so_far = ()
    
    # for each utterance-instance pair in data:
    for pair in data:
        # get utterance and instance
        utt = pair[0]
        inst = pair[1]
        
        # flip observed features & add to feature list
        for feature in inst.features:
            # observed_features_so_far = observed_features_so_far + (feature, )
            if(is_zarpie_feature(feature, coherence)):
                zarpie_features_so_far = zarpie_features_so_far + (feature, )
        
        # convert to tuple; set discards duplicate features
        # observed_features_so_far = tuple(set(observed_features_so_far))
        zarpie_features_so_far = tuple(set(zarpie_features_so_far))
              
        # define zarpie concept to have those features
        zarpies = Kind("Zarpies", zarpie_features_so_far)
        
        # run speaker on hypothesized zarpie features, instance
        # likelihood = speaker's probability of saying the utterance
        # cf. literal listener, which uses literal meaning as likelihood
        utt_likelihood = calc_utt_likelihood(zarpies.features, inst, utt,
                                            **kwargs)
        
        # condition on speaker likelihood
        condition(utt_likelihood)
    
    # define zarpie concept to have those features
    zarpies = Kind("Zarpies", zarpie_features_so_far)
    
    # for parameter estimation, just return coherence to speed up parameter estimates
    if only_return_coherence:
        return coherence
    else:
        return hashabledict(zarpie_features = zarpies.features, coherence = coherence)


#####################
# SPEAKER 2
#####################

@keep_deterministic
def utility_func2(pair: tuple[Utterance, Instance],
                  kind_features: tuple[str, ...]):
    # run *pragmatic* listener
    data = (pair, )
    dist = pragmatic_listener(data)
    
    utt = pair[0]
    inst = pair[1]
    
    # only seen one trial
    observed_features_so_far = inst.features
    
    # compare similarity of inferred kind-linked features to true kind features
    def f(x):
        
        # NOTE: restricted to observed features from data
        return jaccard_similarity(set(kind_features) & set(observed_features_so_far), 
                                  set(x["zarpie_features"]) & set(observed_features_so_far))
        
        ## NOTE: across all kind-linked features known to the speaker
        # return jaccard_similarity(set(kind_features), 
        #                           set(x["zarpie_features"]))
    
    # calculate expected value over all sets of features, collapsing across coherence
    exp_utility = dist.expected_value(f)
    
    return exp_utility

@infer
def speaker2(kind_features: Kind.features, observed_instance: Instance, 
             inv_temp = 5):
    utt = Utterance(subj = draw_from(["Zarpies", "This Zarpie"]),       # consider saying generic or specific..
                    # subj = draw_from(["Zarpies", "This Zarpie", "silence"]),       # consider saying generic or specific..
                    feature = draw_from(observed_instance.features))    # ..about some feature of observed instance
    utility = utility_func2((utt, observed_instance), kind_features)   # consider what *pragmatic* listener will infer (about zarpie features, coherence) from utterance
                                                                        # how close is *pragmatic* listener's inferences to true zarpie features
    # weight = utility**inv_temp                                        # maximize utility, with inv_temp rationality
    weight = math.exp(inv_temp*utility)                                 # version to avoid issues with 0 utility
    condition(weight)
    return utt