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


