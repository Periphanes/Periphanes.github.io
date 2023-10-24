---
title: "Road to NTK (1) : Introduction"
date: 2023-10-20T11:30:03+00:00
# weight: 1
# aliases: ["/first"]
tags: ["NTK", "ML", "Math"]
author: "John Won"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
canonicalURL: "https://periphanes.github.org/gntk_test_1"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
---

## 1. Double Descent and Overparameterization

In classical theories of machine learning and statistical learning, there was the traditional consensus that there was a careful balance between the training error, and generalization gap. We are familiar with the U-shaped graph of the test risk when the complexity of a model is increased. The expected formulation is an initial decrease of both training and test risk, 


The (a) diagram in the figure above showcases this old wisdom, with training risk being reduced as model parameter counts increase, while the test risk rebounds upwards again as the model overfits to the training data. However, in recent years, it has become commonplace to train highly over-parameterized models, with parameter counts orders of magnitude higher than the actual training data point numbers. These models aim to achieve near-zero training error, yet still achieve high performances on testing data. This kind of contradiction is resolved by (Belkin et al., 2018) [1] by the new updated risk curve, aptly named the ‘double descent’ risk curve. In this formultation, the traditional U-shaped curve is still shown for lower parameter counts, but as the model complexity increases past the point where the model can perfectly fit the training data (passing the *interpolation* threshold), testing error starts to drop again.

While there are some theories regarding what makes such phenomena happen, there has yet to be a satisfactory conclusion. Some argue that models in the interpolating regime are indeed overfitted, but 

## 2. Infinite Width-Networks and the Neural Tangent Kernel


*REFERENCES*

[1] Belkin, M., Hsu, D., Ma, S., & Mandal, S. *Reconciling modern machine learning practice and the bias-variance tradeoff,* Proceedings of the National Academy of Sciences, (2019)
