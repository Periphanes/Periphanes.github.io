---
title: "Road to NTK (3) : Gaussian Processes"
date: 2023-10-22T16:30:03+00:00
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
canonicalURL: "https://periphanes.github.org/gntk_3"
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

**************Gaussian Processes**************, as a generic term in many academic fields, refers to any finite collection of empirically observed data being modeled as a multivariate normal distribution. 

In the context of machine learning, regression problems are formulated in GPs as followed: let test data be $X$ and training data be $Y$. Then the joint distribution $P_{X,Y}$ with $|X| + |Y|$ dimensions, modeled as a multivariate Gaussian distribution, spans the space of possible functions for the function that we want to regress on.

We want to predict function values on $X$ using *******************Bayesian Inference*******************, which updates hypotheses based on novel data inputs, which in our case is the training data $Y$. Hence, we wish to calculate the conditional probability $P_{X|Y}$, which is also Gaussian given our conditionalization closure property we discussed above.

### Gaussian Process Prior Distribution

As we discussed above, the GP model assigns a probability distribution over possible functions. To start out with a multivariate normal distribution prior, we need to assign values for the two determining parameters, the mean vector and the covariance matrix. In the case of the mean vector, we can without loss of generality, fix the mean vector to be the zero vector. Even if it is not the case, later addition of the original mean vector is sufficient.

On the other hand, the covariance matrix is not so trivially initialized. The covariance matrix requires positive definite in order to utilize multivariate normal distribution analysis techniques. This requirement is analagous to the one-dimensional Gaussian distribution needing a positive variance value.

As we discussed above in the kernel methods section, positive definite kernels generate a Gram matrix that is positive definite. Hence, we can utilize kernel functions in order to generate an appropriate covariance matrix. I.e. :

$$
\Sigma_{i,j} = k(X_i, X_j)
$$

One popular choice for a kernel function, also in support vector machines, is the radial basis function kernel, better known as the RBF kernel.

$$
k_{RBF}(x, x') = \exp \left( -\frac{||x - x'||^2}{2\sigma^2} \right)
$$

*The RBF kernel can be interpreted as a similarity measure in an infinite-dimensional feature space.

The specific choice of the kernel function is determined through analysis of the data and trial-and-error, similarly to hyperparameter tuning in other machine learning models. Experts can utilize their domain knowledge to capture trends in the training data to choose a great kernel fit. A more comprehensive review of kernels and when to use them can be found at:

[Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/)

Returning to the task at hand, let’s take a look at the prior distribution, the distribution considering only the testing data. The prior distribution $P_X$ is of the same dimensionality as that of $X$. Since the training data is not yet part of the model, the function distribution is determined entirely by the kernel and test data choice. Since generating random functions from a specific kernel distribution is not our task, we need further training data to truly utilize this methodology.

### Gaussian Process Posterior Distribution

What changes when empirical observations are added? In Bayesian inference terminology, we incorporate novel information by updating the distribution into the *********posterior********* distribution $P_{X|Y}$. This can be calculated by first forming the joint distribution $P_{X,Y}$, then conditioning over the training data to obtain the posterior. Since the multivariate normal distribution is closed over conditioning, our new posterior distribution is also Gaussian.

An intuitive description of the conditioning process is that it forces the functions to exactly pass through the training data points. In the constrained covariance matrix, if a predicted point lies on the training data, there is no correlation with any other points, meaning the function must pass directly through it. Such constraints mean that uncertainty of predictions of data points close to training data is smaller, while data points further away have higher uncertainty.

However, such strong constraints have a few problems. First, slackless bounding of the functions to each of the training points means that fitted functions may have unnecessarily complex structures. Also, the measurements in the training dataset themselves are most likely noisy measurements themselves, meaning there is a uncertainty bound to each training data point.

Hence, we can incorporate the error of measurements into the Gaussian process model by modeling the error themselves.

We can add an error term $\epsilon \sim \mathcal{N}(0, \psi^2)$ to each of out data points in $X$:

$$
Y = f(X) + \epsilon
$$

Then modify the joint distribution $P_{X,Y}$ as follows:

$$
P_{X,Y} = \begin{bmatrix} X \\ Y \end{bmatrix} \sim \mathcal{N}(0, \Sigma) = \mathcal{N}\left( \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix} \Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_{YY} + \psi^2I \end{bmatrix} \right)
$$

Now that we have a Gaussian process model that incorporates training and test data, alongisde support for noisy inputs, how do we make a prediction based on this model?

The most simple approach would be to simply sample from the model distribution, but this would mean that we could potentially sample an outlier from the distribution, and widely misrepresent the function distribution. Therefore, a more sophisticated and accurate approach would be to utilize marginalization of the total distribution over the individual testing data points.

This means that we calculate the marginal univariate Gaussian distribution for each of the testing data points, and obtain their mean and variance values. Extracting these values not only gives a more meaningful prediction of the function values, but also gives us values to make a statement about the confidence of the prediction, given the variance value.


<br> <br>

### REFERENCES

[1] [https://distill.pub/2019/visual-exploration-gaussian-processes](https://distill.pub/2019/visual-exploration-gaussian-processes/#Multivariate)

[2] [https://www.cs.toronto.edu/~hinton/csc2515/notes/gp_slides_fall08.pdf](https://www.cs.toronto.edu/~hinton/csc2515/notes/gp_slides_fall08.pdf)

[3] [https://arxiv.org/pdf/math/0701907.pdf](https://arxiv.org/pdf/math/0701907.pdf)

[4] [https://peterroelants.github.io/posts/gaussian-process-tutorial/](https://peterroelants.github.io/posts/gaussian-process-tutorial/)

[5] [https://bookdown.org/rbg/surrogates/chap5.html](https://bookdown.org/rbg/surrogates/chap5.html)

[6] [https://greeksharifa.github.io/bayesian_statistics/2020/07/12/Gaussian-Process/](https://greeksharifa.github.io/bayesian_statistics/2020/07/12/Gaussian-Process/)