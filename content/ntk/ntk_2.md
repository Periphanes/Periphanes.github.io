---
title: "Road to NTK (2) : Multivariate Gaussians and Kernel Methods"
date: 2023-10-22T11:30:03+00:00
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
canonicalURL: "https://periphanes.github.org/gntk_2"
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

Before delving into the intricacies of infinite-width neural networks, we must first go over the mathematical preliminaries, mainly the multivariate Gaussian distribution and kernel functions. While both topics have huge academic literature surrounding them, we only need a slight introduction to these concepts to go forward.

## 1. Multivariate Gaussian Distributions

### 1-1. Definition of Multivariate Gaussians

Recall the univariate Gaussian distribution from high school:

$$
X \sim \mathcal{N}(\mu, \sigma^2), \quad p(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp \left( -\frac{(x-\mu)^2}{2\sigma^2} \right) 
$$

It is a probability distribution over the real line, parameterized by its mean and the variance.

The **********************multivariate Gaussian distribution********************** is a multi-dimensional generalization of this formulation, representing the distribution of a multivariate random variable that is made up of multiple normally distributed random variables that can be correlated to each other such that their joint distribution is also Gaussian.

It is parameterized by the **********************mean vector********************** $\mu$ and the **covariance matrix $\Sigma$,** with the mean vector selecting the center of the distribution, and the covariance matrix determining the shape.

Recalling the definition of the covariance matrix, $\Sigma_{ij} = \mathbb{E}[(X_i-\mu_i)(X_j - \mu_j)]$, the diagonal entries of the matrix denote the variance of the individual variables, while other entries denote the covariance, and in extent, the correlation among variables.

To derive the probability density function of the multivariate function, first the term

$$
(x - \mu)^2 / \sigma^2
$$

can be generalized for a $p$ x 1 vector $x$ of observations as

$$
(x - \mu)^T\Sigma^{-1}(x-\mu)
$$

Hence the pdf of the multivariate Gaussian distributed vector $X = [X_1, X_2, ..., X_p]$ is

$$
\begin{equation} p(x|\mu, \Sigma) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}\exp \left( -\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x-\mu) \right) \end{equation}
$$

Where $|\Sigma|$ is the determinant of the covariance matrix

While real data is rarely exactly multivariate normally distributed, the central limit theorem, which extends to higher dimensions, causes the mulitvariate Gaussian to be a useful approximation to many multi-dimensionally sampled data.

### 1-2. Useful Properties of Gaussians

One useful aspect of Gaussian distributions is that they are closed under marginalization and conditioning. To understand these properties, consider the following bivariate distribution.

$$
P_{X,Y} = \begin{bmatrix} X \\ Y \end{bmatrix} \sim \mathcal{N}(\mu, \Sigma) = \mathcal{N}\left( \begin{bmatrix} \mu_X \\ \mu_Y \end{bmatrix}, \begin{bmatrix} \Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_{YY} \end{bmatrix} \right)
$$

******************************Marginalization****************************** refers to integrating over all other random variables to obtain a probability distribution over just one random variable.

For example, for the bivariate Gaussian case given above, we can calculate the marginal distribution of $X$ by integrating over all possible outcomes of $Y$.

$$
p_X(x) = \int_yp_{X,Y}(x,y)dy = \int_yp_{X|Y}(x|y)p_Y(y)dy
$$

************************Conditioning************************ refers to determining the probability distribution of a variable given some observation of a different variable. While the math itself is more complicated, conditioning on Gaussian distributions is also closed and yields a modified Gaussian distribution.


## 2. Kernel Methods

### 2.1 Introduction to Kernels

Prior study on Support Vector Machines (SVMs) or other traditional machine learning methodologies most likely have shown reference to the ************kernel trick************. Kernels represent similarity measures that correspond to dot products in higher-dimensional dot product spaces.

********************Definition******************** : Given empirical data $\{(x_i, y_i)\}_{i \in [1, n]}$, the kernel $k$ is defined as

$$
k:\mathcal{X} \times \mathcal{X} \to \mathbb{R}, \quad (x,x') \mapsto k(x,x')
$$

satisfying for all $x, x' \in \mathcal{X}$

$$
\begin{equation} k(x,x') = \langle \Phi(x), \Phi(x') \rangle \end{equation}
$$

where feature map $\Phi$ maps into some dot product space $\mathcal{H}$, also called the *************feature space*************. Please keep in mind that the feature map and corresponding feature space computations are mostly not directly calculated, but instead only indirectly accessed through the dot product using kernel methods on the original lower-dimensional data inputs.

While there are a plethora of types of kernels, those that satisfy equation (2), that is, corresponding to a dot product in some other dot product space, are those of the class of ******************positive definite****************** kernels. This is a very useful here are examples of positive definite kernels which can be evaluated efficiently even though they correspond to infinite dimensional dot product spaces. In such cases, substituting the high-dimensional dot product with the kernel computation is crucial.

So *what is* a positive-definite kernel?

****************************Definition (Gram Matrix)**************************** : Given a kernel $k$,

$$
K \coloneqq (k(x_i, x_j))_{ij}
$$

is called the Gram matrix (or kernel matrix) of $k$ with respect to $x_1, x_2, ..., x_n$.

********************Definition (Positive definite matrix)******************** : A real $n \times n$ symmetric matrix $K_{ij}$ satisfying

$$
\sum_{i,j}c_ic_jK_{ij} \geq 0
$$

for all $c_i \in \mathbb{R}$ is called positive definite. If the equality only occurs when all coefficients are zero, we call the matrix ***************************strictly positive definite***************************.

**************************Definition (Positive definite kernel)************************** : A kernel function $k$ which gives rise to a positive definite Gram matrix is called a positive definite kernel. If the function gives rise to a strictly positive definite Gram matrix, the kernel is called a strictly positive definite kernel.

<br> <br>

### REFERENCES

[1] [https://distill.pub/2019/visual-exploration-gaussian-processes](https://distill.pub/2019/visual-exploration-gaussian-processes/#Multivariate)

[2] [https://www.cs.toronto.edu/~hinton/csc2515/notes/gp_slides_fall08.pdf](https://www.cs.toronto.edu/~hinton/csc2515/notes/gp_slides_fall08.pdf)

[3] [https://arxiv.org/pdf/math/0701907.pdf](https://arxiv.org/pdf/math/0701907.pdf)

[4] [https://peterroelants.github.io/posts/gaussian-process-tutorial/](https://peterroelants.github.io/posts/gaussian-process-tutorial/)

[5] [https://bookdown.org/rbg/surrogates/chap5.html](https://bookdown.org/rbg/surrogates/chap5.html)

[6] [https://greeksharifa.github.io/bayesian_statistics/2020/07/12/Gaussian-Process/](https://greeksharifa.github.io/bayesian_statistics/2020/07/12/Gaussian-Process/)