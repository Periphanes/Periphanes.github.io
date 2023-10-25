---
title: "Road to NTK (4) : Infinite-Width Nerual Networks - 1"
date: 2023-10-24T16:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["NTK", "ML", "Math"]
author: "John Won"
# author: ["Me", "You"] # multiple authors
showToc: false
TocOpen: false
draft: false
hidemeta: false
comments: false
canonicalURL: "https://periphanes.github.org/gntk_4"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: false
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: false
---

Recalling the notion of conventional neural networks, the concept of Maximum Likelihood Estimation (MLE) is used to determine a set of parameters for the given model in order to maximize the likelihood of such parameters under given training set. On the other hand, Bayesian inference is an alternative approach where we learn the distribution of neural networks, allowing us to marginalize over specific parameters to obtain their distribution, uncertainty, and mean values.

The starting point for any type of Bayesian modeling is the use of prior distributions : i.e, our prior beliefs about the relationship we are modeling with the network, in the form of distributions for weight and biases for our neural networks. However, in MLP structures, the actual meaning of the weight values in our model is obscure and hard to quantify into a concrete distribution. In addition, models with small hidden unit counts represent a small subset of the possible function space, which means our prior belief most likely will be that the model is simply wrong.

However, in the limit of infinite hidden units, many papers have demonstrated neural networks as universal function approximaters, which gives infinite-width networks great justification as a reeasonable model for many problems. In the seminal paper, (Neal, 1994) [7] shows that infinite networks allow for easy analysis of the priors over functions, while showing that Gaussian priors result in convergence to Gaussian processes.

**********Proof********** : Consider the following single-hidden layer network taking $I$ real-valued inputs $x_i$, producing $O$  real-valued outputs $f_k(x)$, using the layer of $H$ hidden units with value $h_j(x)$:

$$
\begin{equation} f_k(x) = b_k + \sum^H_{j=1}v_{jk}h_j(x) \end{equation}
$$

$$
\begin{equation} h_j(x) = tanh(a_j + \sum_{i=1}^Iu_{ij}x_i) \end{equation}
$$

The weight and bias variables $b, v, a, u$ all are considered to have Gaussian prior distributions, with zero-mean and standard deviations of $\sigma^b,\sigma^v,\sigma^a,\sigma^u$.

In order to see the Gaussian distributions of the function outputs, consider the prior distribution $f_k(x)$ under a set value for input $x$, which is the prior distribution of the value of output unit $k$, implied by the prior distributions for the weights and biases. Looking at equation (3), $f_k(x)$ is the sum of a bias term and the weighted contributions from $H$ hidden units. Under the model hypothesis, each term in the summation is independent from each other.

From the contributions from the hidden units, all of them have identical distributions while the expected value of each contribution is zero. ($\mathbb{E}[v_{jk}h_j(x)] = \mathbb{E}[v_{jk}]\mathbb{E}[h_j(k)] = 0, \because \mathbb{E}[v_{jk}]=0$)

Given zero mean, the variance of the contributions is found as follows:

$$
\begin{split} \mathbb{V}[v_{jk}h_j(x)] &= \mathbb{E}[(v_{jk}h_j(x))^2] - \mathbb{E}[v_{jk}h_j(x)]\mathbb{E}[v_{jk}h_j(x)] \\\ &= \mathbb{E}[v_{jk}h_j(x)] \quad (\because \mathbb{E}[v_{jk}h_j(x)] = 0) \\\ &= \mathbb{E}[v_{jk}^2]\mathbb{E}[h_j(x)^2] \\\ &= (\mathbb{V}[v_{jk}] + \mathbb{E}[v_{jk}]^2)\mathbb{E[h_j(x)]} \quad (\because \mathbb{E}[v_{jk}] = 0) \\\ &=\sigma_v^2\mathbb{E}[h_j(x)^2]\end{split}
$$

Since $h_j(x)$ is bounded by the tanh function, the calculated variance $\sigma_v^2\mathbb{E}[h_j(x)^2]$ must be finite. Then, when we define $V(x) = \mathbb{E}[h_j(x)^2]$, by the central limit theorem, for large $H$, we can conclude that the total contributions of the hidden units to the value of $f_k(x)$ becomes Gaussian with zero mean and variance of $H\sigma^2_vV(x)$. Taking into account the bias term, the total prior distribution of $f_k(x)$ is normally distributed with variance $\sigma^2_b + H\sigma_v^2V(x)$.


> 🔑 ******************************Central Limit Theorem****************************** : The mean and sum of a random sample of a large enough size from an arbitrary probability distribution have approximately normal distributions. Given random sample $X_1, ...,X_n$ with mean $\mu$ and finite variance $\sigma^2$,
> - *Sample Sum* $S = \sum^n_{i=1} X_i$ is approximately normal $\mathcal{N}(n\mu, n\sigma^2)$
> - *Sample Mean* $\bar{X} = \frac{1}{n}\sum^n_{i=1}X_i$ is approximately normal $\mathcal{N}(\mu, \sigma^2/n)$

> *Please make sure to discriminate between sum of normally distributed random variables, which are normally distributed if components are independent or jointly normal, and sum of normal distributions, which creates Gaussian mixture model for camel-shaped graphs.


In order to obtain a well-defined limit on the variance of $f_k(x)$ as $H$ approches infinity, we simply scale the prior variance of $\sigma_v$ to be $\omega_v\sqrt{H}$ for some fixed value $\omega_v$. Then, the prior for $f_k(x)$ converges to a Gaussian of zero mean and variance $\sigma_b^2 + \omega_v^2V(x)$ as $H$ approaches infinity.

Now that the distribution for a single given input $x$ is determined to be Gaussian, let us take a look at a set of inputs $x_1, x_2, ..., x_n$, and the joint distribution of $f_k(x_1), f_k(x_2), ..., f_k(x_n)$.

Using arguments similar to above, as $H$ approaches infinity, this prior joint distribution converges to a multivariate Gaussian distribution with zero mean and covariances of

$$
\begin{split} \mathbb{E}[f_k(x_p)f_k(x_q)] &= \sigma_b^2 + \sum_j\sigma^2_v\mathbb{E}[h_j(x_p)h_j(x_q)] \\\ &= \sigma_b^2 + \omega_v^2 C(x_p, x_q) \end{split}
$$

where  $C(x_p, x_q) = \mathbb{E}[h_j(x_p), h_j(x_q)]$, which is the same for all $j.$ Since the resulting joint distribution of the given input points set is Gaussian, this a case of a Gaussian process! Here the kernel function is the above covariance calculation function.

Here, the prior covariance between the values of output $f_k(x_i)$ for different values of inputs are in general not zero, which is what allows for learning to occur when training data points are conditioned on. Given values for $f_k(x_1), ..., f_k(x_{n-1})$, we could condition on these values and explicitly find the predictive Gaussian distribution for $f_k(x_n)$.

<br> <br>

### REFERENCES

[1] Neal, R. M. *Priors for infinite networks* (tech. rep. no. crg-tr-94-1) University of Toronto, (1994)