---
title: "Road to NTK (1) : Introduction"
date: 2023-10-20T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["NTK", "ML"]
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

In classical theories of machine learning and statistical learning, there was the traditional consensus that there is a careful balance between the training error and the generalization gap. Also widely known as the *bias-variance trade-off*, models with higher complexity should have lower bias but higher variance. 

Hence, we are all too familiar with the U-shaped graph of the test risk when the complexity of a model is increased. The expected geometry of the graph is an initial decrease of both training and test risk, until models start overfitting and test risk starts to increase instead. This is why classic ML methods aim to find the 'sweet-spot' of test risk at its minimum, preventing both overfitting and underfitting

However, modern deep neural networks don't seem to follow this trend, with millions and even billions of parameters trained, that theoretically should have been critically overfitted to the training data, yet they outperform less complex models in empirical studies. Accordingly, in recent years, it has become commonplace to train highly over-parameterized models, with parameter counts orders of magnitude higher than the actual training data point numbers. These models aim to achieve near-zero training error, yet still achieve high performances on testing data. This kind of contradiction is resolved by (Belkin et al., 2018) [1] by the new updated risk curve, aptly named the ‘double descent’ risk curve. In this formultation, the traditional U-shaped curve is still shown for lower parameter counts, but as the model complexity increases past the point where the model can perfectly fit the training data (passing the *interpolation* threshold), testing error starts to drop again.

Given that larger and larger models seem to increase the effectiveness, we are naturally led to the question of what happens with neural networks with infinite complexity? And what would the theoretical properties of such constructs be? How would those networks' training dynamics look like? We aim to delve into these questions with a series of posts on infinite-width neural networks and the neural tangent kernel (NTK).

### REFERENCES

[1] Belkin, M., Hsu, D., Ma, S., & Mandal, S. *Reconciling modern machine learning practice and the bias-variance tradeoff,* Proceedings of the National Academy of Sciences, (2019)

[2] Nakkiran, P., Kaplun G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. *Deep Double Descent: Where Bigger Models and More Data Hurt*, International Conference on Learning Representations, (2020)