---
author:
- 'Manuel Del Verme, Prof. Pierre-Luc Bacon and Prof. Doina Precup'
date: '24, November 2020'
title: 'Constrained Optimization, an Alternative to Back-Propagation'
---

I will survey two approaches of training neural networks with divide et
impera methods. The first view from [@LeCun1988] reformulates the
classical unconstrained objective with an equivalent Lagrangian
formulation, [@Lee2015] on the other side aims to formalize a
biologically inspired learning process based on the propagation of
target values.

A Lagrangian view of Back-Propagation was first introduced by
[@LeCun1988], derivative works focused on the biological plausibility of
such approaches are introduced later. Almost 30 years later [@Betti2018]
rediscovered the concept while trying to derive a local alternative to
Back-Propagation.

A first attempt of training neural networks by dividing it into
sub-problems is by [@Carreira-PerpinanWeiranWangEECS2012] who trains
networks layer-wise, later [@Gotmare] introduced the concept of
block-wise constraints.

[@Kalman2009] shows that solutions to Lagrangian problems are always
saddle point of the Lagrangian equation, from this conclusion I will
then review methods used to solve non-linear problems and to find
approximate solutions to Variational Inequalities (VIs) , such as the
projection method due to [@Stampacchia1970], the extra-gradient method
[@KORPELEVICH1976] which harmonizes optimization achieving convergence
for a larger class of VIs.

In the field of Discrete Time Optimal Control constrained problems such
as the [@LeCun1988] are well studied as multistage systems where the
network weights are the controls and the network activations are states,
the two most important methods to solve these NLP are direct collocation
[@Biegler1984] and direct multiple shooting [@Bock1985].

Moving to large scale problems I will then review [@Nemirovski2005] adds
a Bregman proximal to extra-gradient later [@Gidel] introduces a
gradient reuse step to avoid recalculating the extra gradient step
without loss of convergence guarantees.

The idea if learning target representations along the network weights
are found in [@Krogh1989] and [@rohwer1990moving], more recently
[@lee2015deeply] introduced companion objective to the original problem
to have a per-layer superivsion. [@Lee2015] formalizes target
propagation and computes targets rather than gradients, at each layer
using auto-encoders, recently [@Lillicrap2020] adds a feedback network
as an approximation for the learning signal.
