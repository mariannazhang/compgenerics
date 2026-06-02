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


instead of taking max y' and sending it to linking function, we can instead normalize over y', then send all normalized samples of y' to linking function (because w/ all the samples we can calculate the normalizing constant), then divide all unnorm by normconst -> set of samples has normalied prob
at the moment, we're just sending the mean / expected y' but we could really do the full distribution (more bayesian)

see if it makes sense to sample all the different test features

compare null hypothesis (large lengthscale) vs. fitting lengthscales (to see if fitting the lengthscale does indeed help explain prevalence judgments)


plot where lengthscale is fixed vs one where it's fit


for comparing the distributions, can think about how to compare
