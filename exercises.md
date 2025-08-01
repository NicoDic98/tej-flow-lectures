# Overview
These exercises go alongside the code developed during the lectures. They will
for the most part involve exploring or extending that work, as well as answering
some related questions.

For help with specific numpy or Pytorch syntax and details, please feel free to ask,
or you may find ChatGPT helpful!

I have intentionally included **too many** exercises for the available tutorial
time. The idea is that you can choose what you find the most interesting or
useful during tutorials. The additional questions can provide a starting point
for your own future exploration if you like.

# Exercise set 1
## Warm-up
  * Show that the ESS is related to the variance $\sigma^2 = \mathrm{Var}[w]$ and mean $M = \langle w \rangle$ according to
    $$\mathrm{ESS} = \frac{M^2}{\sigma^2 + M^2}$$
  * Use this form to show that the useless model limit corresponds to $\mathrm{ESS} = 0$ and the perfect model limit corresponds to $\mathrm{ESS} = 1$.
  * Extra: Assuming $p(U)$ is normalized in the definition $w(U) = p(U)/q(U)$, show that $\langle w \rangle = 1$.

## Critical slowing down
Using the HMC code we developed for the $\phi^4$ theory, explore critical
slowing down and topological freezing quantitatively:

  1. The _autocorrelation function_ for observable $O$ is defined by
  $$\rho(\tau) = \frac{1}{N-\tau}\sum_{i-j = \tau} \frac{O^{(i)} O^{(j)}}{\langle O^2 \rangle}$$
   which measures the correlation between the observable at points in the Markov chain separated by MC time $\tau$, assuming an ensemble of size $N$.
	 * Plot the autocorrelation function for the observable $\bar{\phi} = \frac{1}{V} \sum_x \phi_x$ at the broken and symmetric phase parameters we looked at. Confirm that for the broken phase the autocorrelation function decays to zero on a much longer timescale.
	 * In the broken phase, suppose the Markov chain tunnels between the two VEVs, $\bar{\phi} = \pm v$, according to a Poisson process with rate $r$. How should $\rho(\tau)$ scale for small $\tau$ under this assumption?
	 * Extra: solve for $\rho(\tau)$ for all $\tau$ under this assumption.
	 * Extra: use this to estimate the tunneling rate of our HMC sampler at the broken phase parameters we looked at.

 2. The _integrated autocorrelation time_ is defined by
 $$\tau_{\mathrm{int}} = \frac{1}{2} + \sum_{\tau = 1}^{\infty} \rho(\tau)$$
 and estimates the decorrelation time of the observable under MCMC. Probably Jacob will talk about this more!
	 * Estimate $\tau_{\mathrm{int}}$ for $\bar{\phi}$ using some sensible upper limit of summation in both phases.
	 * Extra: Measure $\tau_{\mathrm{int}}$ for $\bar{\phi}$ for a range of quartic couplings between the two phases. What features do you expect to see, and do you observe them?

3. More extra stuff: Study the questions above for the free theory at various $m^2$ values, scaling the number of lattice points to hold $m L$ fixed. This is the usual form of critical slowing down. Estimate the dynamic critical exponential $z$ according to which the autocorrelation time scales as $$\tau_{int} \sim m^{-z}$$

## Toy flow-based sampling
The toy flow we built has a double-peaked structure. Suppose we _actually_ wanted to sample the mixture of Gaussians:
 $$p(x) = \tfrac{1}{2} \exp(-\lVert x - \mu \rVert^2/2\sigma^2) + \tfrac{1}{2} \exp(-\lVert x + \mu \rVert^2/2\sigma^2)$$
where $x$ and $\mu$ are both 2D vectors.
  * Implement $\log p(x)$ in terms of some fixed values $\mu$ and $\sigma$.
  * Plot $\log p(x)$ alongside the plot of the flow model $\log q(x)$ we produced in the lecture.
  * Use the plot to tune $\mu$ and $\sigma$ to approximately match the output of the flow model.
  * Compute the reweighting factors needed to correct samples from the flow model to this target distribution. Estimate the corresponding ESS and confirm that your choice of $\mu$ and $\sigma$ give a reasonable non-zero value.
  * Make an unbiased estimate of $\langle \lVert x \rVert^2 \rangle_p$ using the flow samples. Check that it approximately agrees with the true value $2 \sigma^2 + \lVert \mu \rVert^2$.
  * Extra: estimate the uncertainties on your estimate and check whether you find agreement within error bars. If not, what might have gone wrong? (This may be the case even if you have done everything correctly!)


# Exercise set 2
## Warm-up
Recall that the forward and reverse KL divergence are defined respectively by
 $$D_{KL}(p || q) = \int dU p(U) [\log{p}(U) - \log{q}(U)] \geq 0$$
 $$D_{KL}(q || p) = \int dU q(U) [\log{q}(U) - \log{p}(U)] \geq 0$$.
  * If $p(U) = e^{-S(U)}/Z$ is the target distribution, expand the $\log{p}$
    term in each case and extract an additive constant $\pm \log{Z}$, which
    does not affect the gradient (for example, when using these
    divergences as loss functions).
  * Show that the expectation value of the _unnormalized_ reweighting factors
    $w(U) = e^{-S(U)}/q(U)$ is the partition function, $$\mathbb{E}_{q}[w] = Z$$.

## Neural network fits
We fitted the $\mathrm{sinc}(x)$ function using a simple neural network (NN).
  * Improve the quality of the fit by varying the network size, training time,
    and choice of sampled distribution for training data.
  * Extra: Use a NN to fit an interesting function in 4D. For example, you might
    choose the 1-loop Euclidean integrand
    $$f(q) = (q^2 + m^2)^{-1} ((q-p)^2 + m^2)^{-1}$$
    using an arbitrary fixed external momentum $p$.

## Toy learned flow
We used the reverse KL divergence to learn the Gaussian mixture distribution,
which suffers from "mode-collapse" (overfitting to specific peaks).
  * Train the same flow using the forward KL divergence. You should observe the
    "mode-covering" behavior of the flow in this case.
  * Extra: Improve the quality of the fitted model by varying the model size,
    training time, and/or training strategy.


# Exercise set 3
## Warm-up
  * Prove that an invariant prior density $r$ combined with an equivariant flow
    results in an invariant model density $q$ (hint: use the change of measure
    formula).
  * A conditional flow model for pseudofermions gives a conditional model
    density $q(\phi | U)$. Argue that a good conditional flow
    ($\mathrm{ESS} \sim 1$) gives a good estimate of $\det(D^\dagger D(U))$.

## Learned flow for phi4 theory
Our flow model gave a reasonable sampler for the $L=4$ theory, but was not fully
effective at capturing the two broken symmetry vacua (for example the ESS was only
about 50%).
  * Try to improve these $L=4$ results by vary model size, training time, and/or
    training approach (e.g. try implementing forward KL training).
  * Insert HMC updates between each flow transformation. Careful: make sure to
    compute the reweighting factors using the samples _before_ this update, and
    use the correct target theory for the HMC steps.
  * Run the flow sampler for the $L=8$ theory.
  * Compare the ESS in the $L=8$ theory vs $L=4$. Crudely, one expects this to
    scale as $$\mathrm{ESS}(L=8) = \mathrm{ESS}(L=4)^{2 \times 2}$$. Does this
    hold in this case?
  * Using that scaling, what ESS for $L=4$ would be needed to ensure
    $$\mathrm{ESS}(L=64) \gtrsim 0.5$$?
