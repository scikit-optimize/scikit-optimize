---
title: 'scikit-optimize: Sequential model-based optimization toolkit.'
tags:
  - Python
  - Machine-learning
  - Global optimization
  - Scientific Computing
authors:
 - name: Manoj Kumar
   orcid: 0000-0002-3802-8346
   affiliation: 1
 - name: Tim Head
   orcid:
   affiliation: 2
 - name: Gilles Louppe
   orcid:
   affiliation: 3
 - name: Katie Malone
   orcid:
   affiliation: 4
 - name: nel215
   orcid:
   affliation: 5
 - name: Nuno Campos
   orcid:
   affliation: 6
 - name: Zé Vinícius
   orcid:
   affliation: 7
 - name: Thomas Fan
   orcid:
   affliation: 8
 - name: Mikhail Pak
   orcid:
   affliation: 9
 - name: Iaroslav Shcherbatyi
   orcid:
   affliation: 10
 - name: Lilian Besson
   orcid:
   affliation: 11
 - name: Fabian Linzberger
   orcid:
   affliation: 12
 - name: Taylor Smith
   orcid:
   affliation: 13
 - name: andreh7
   orcid:
   affliation: 14
 - name: Mehdi Cherti
   orcid:
   affliation: 15
 - name: Alexander Fabisch
   orcid:
   affliation: 16

affiliations:
 - name: New York University
   index: 1
 - name:
   index: 2
 - name:
   index: 3
 - name:
   index: 4
 - name:
   index: 5
 - name:
   index: 6
 - name:
   index: 7
 - name:
   index: 8
 - name:
   index: 9
 - name:
   index: 10
 - name:
   index: 11
 - name:
   index: 12
 - name:
   index: 13
 - name:
   index: 14
 - name:
   index: 15
 - name:
   index: 16
date: June 2, 2017
bibliography: paper.bib
---

# Summary
scikit-optimize is a sequential model-based optimization toolkit.

scikit-optimize aims at the global minimization of functions that are noisy,
expensive or for which gradients are not available. Examples of such optimization
includes optimizing the loss of a deep neural network on validation-data or
the settings of a robot controller. The technique used by
scikit-optimize is popularly known as sequential optimization or
bayesian optimization. The key idea behind this technique is
that a predictive model is used to suggest a new candidate point given the
function evaluations at the previous points. This new point is suggested
such that there is a balance between exploitation and exploration.

The aim of scikit-optimize is to provide an easy to use interface for such
optimization techniques. Currently, we support Gaussian Process, Random Forest
and Gradient Boosting based search and also allow the power-user to plug in a custom predictive model. In terms of
interface, we support two interfaces, an interface that
mimics scipy.optimize and an ask-and-tell interface using
an Optimizer class that allows more control to the user.

# References
