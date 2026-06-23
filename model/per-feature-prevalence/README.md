Overleaf notation:
$$

\begin{align*}

P(p' \mid \vec{u})

&= \int_{\mathbb{R}} \int_{\mathbb{R}^n} P(p' \mid y') P(y' \mid \vec{y}) P(\vec{y} \mid \vec{u}) d\vec{y} d y' \\

&= \int_{\mathbb{R}} P(p' \mid y') \left[\int_{\mathbb{R}^n} P(y' \mid \vec{y}) P(\vec{y} \mid \vec{u}) d\vec{y} \right] d y'

\end{align*}

$$

Recent notation that helps define the MCMC question
$$

\begin{align*}

P(p' \mid \vec{u})

&= \int_{\mathbb{R}} \int_{\mathbb{R}^n} P(p' \mid y') P(y', \vec{y}\mid \vec{u}) d\vec{y} d y' \\
\end{align*}

$$
focus on getting pseudocoherences from utterances $P(y', \vec{y}\mid \vec{u})$
$$

\begin{align*}

P(y', \vec{y}\mid \vec{u})

&\propto P(\vec{u}|\vec{y},y')P(y',\vec{y}) \\

&\propto \left[ \prod_i P(u_i|y_i) \right] P(y',\vec{y}) \\
&\propto \left[ \prod_i \sum_i P(u_i|z_i)P(z_i|y_i) \right] P(y',\vec{y})

\end{align*}

$$


$$\mathcal{L}(y',\vec{y};\vec{u}) = \log \left[\left[ \prod_i \sum_i P_S(u_i|z_i)P(z_i|y_i) \right] MVN(y',\vec{y};\mu,\Sigma)\right] $$


the above is mostly covered in `model_jax.py`, except the actual fitting of linking functions currently happens in `model-study9-fwd.ipynb`.

the above is the "participant response model" that gives the probability of prevalence judgments given utterances and parameters (gp mu, sigma, lengthscale, and rationality parameter beta).

now we are considering the "scientist model" that is trying to infer the underlying parameters of participant behavior.

$$ P(\theta|\vec{u},p') \propto P(p'|\vec{u},\theta)P(\theta)$$






## versions of model

### bayesian approx of linking fn
instead of taking MLE y' and sending it to linking function, we can instead normalize over y', then send all normalized samples of y' to linking function (because w/ all the samples we can calculate the normalizing constant), then divide all unnorm by normconst -> set of samples has normalied prob
at the moment, we're just sending the mean / expected y' but we could really do the full distribution (more bayesian)

### joint sampling of features
see if it makes sense to sample all the different test features

### results
per feature mcmc, MLE of y’ to linking function: r^2 = 0.908
per feature mcmc, samples of y’ to linking function: r^2 = 0.921
joint feature mcmc, samples of y’ to linking function: r^2 = 0.920

## eventual goal for modeling
- compare null hypothesis (large lengthscale) vs. fitting lengthscales (to see if fitting the lengthscale does indeed help explain prevalence judgments, that feature space does matter)
- plot where lengthscale is fixed vs one where it's fit
- for comparing the distributions of possible prevalences, can think about how to compare beyond the means


## for eval model fit

### in grid search
use approx log likelihood of data given parameters using log density fn (joint)

start with small grid search, then

### using an optimizer instead 
(not gradient-based)
could use black box optimizer that doesnt take gradients



https://docs.scipy.org/doc/scipy/reference/optimize.html  
for noisy fn, could use a gp to optimize it  
pybads  
https://acerbilab.github.io/pybads/index.html


## linking fn
- to what extent would things change if we fit the prevalence response params (beta fn params) outside of the inner loop
- what's the prev distribution for this specific feature if it is indeed generic vs. if it's not a generic feature
- ~if it's a prior belief about what's kl and nkl, then we can say it's independent from the predicted coherence~

- on the other hand, it could be that influence of generalization is influenced by the beta distributions (e.g., some features have more bimodal distribs, so changes in coherence have a strong effect on such features, or some features have little diff between kl vs nkl)



double check how many samples of the pseudocoherence we're taking in each iteration, since the beta params could be overfitting to the stochasticity here


also start by just using the coherence as the prevalence and optimize based on that, bypassing linking fn param fits.

check w/ claude for underflow / overflow / where it's coming from