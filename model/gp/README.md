# Bernoulli GPs for P(kind-linked)

A **Gaussian Process (GP)** defines a prior over functions. Then, given training points, it infers a posterior distribution over the function value at any new point.

A **Bernoulli GP (GP classifier)**  models binary outcomes with binary training points. The latent GP function `f(x)` is sent through a sigmoid to produce probabilities:

```
P(y=1 | x) = sigmoid(f(x))
```

Training points labeled `y=1` (kind-linked) increase values of `f` nearby (to the extent of the lengthscale). Unlabeled regions trend toward the prior mean (`sigmoid(prior_mean)`).

## Inference

Because the likelihoods are no longer Gaussian (they are now Bernoulli), we have to approximate `f` given the data instead of having a closed-form solution. To my knowledge, we should use **Laplace approximation** to find the MAP estimate of `f` given the data, then approximate the posterior as a Gaussian centered there. Helpful blog post: https://krasserm.github.io/2020/11/04/gaussian-processes-classification/

## Hyperparameters

| Parameter | Effect |
|---|---|
| `lengthscale` | How far known training points influence neighbors (small = only close neighbors are influenced, large = even far neighbors are influenced) |
| `sigma` | average distance of `f` away from its mean |
| `prior_mean` | Baseline `f` important for baseline rate of P(kind-linked) especially in areas of the embedding space where we see no training points; `sigmoid(prior_mean)` = default P(kind-linked) |

## Study 9
- Participants only see generic statements, so training features are labeled "1" (kind-linked).
- Training features depended on condition (physical / diet / personality / heterogeneous)
- The GP predicts P(kind-linked) for 15 test features in the 2D sentence-embedding space
    - TODO: extend to 384d or middle ground d embedding space (full analysis)
    - TODO: for viz, plot in 2d space but use numbers/ similiarities from 384d (though check if we can somehow plot the bg)
    - TODO: choose to fit the beta linking function or just use the same generic beta weights for all features (based on those from feature set 1)
- Lengthscale & prior mean is fit per participant (maximizing likelihood of their slider responses)
