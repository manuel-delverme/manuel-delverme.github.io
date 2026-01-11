---
title: "Divide et Impera and Neural Networks"
date: 2021-01-14
draft: false
tags: ["deep-learning", "optimization", "research"]
description: "A survey of approaches to divide the parameter estimation problem for neural networks into simpler sub-problems using constrained optimization."
math: true
ShowToc: true
TocOpen: true
---

Decomposing a problem into smaller and more manageable pieces to be solved independently and then merging the solutions has been a common motif inspiring many fundamental advancements (Search, ordinary differential equations, Kepler etc.).

In this survey, I will present several approaches to divide the parameter estimation problem for neural networks by back-propagation into simpler sub-problems.

Neural networks have recently had a strong impact in both our day to day life and in other fields, partially also due to this reason, the number of parameters of the modern models keeps increasing with no yet clear limits on performance, yet the hardware to train such networks won't be able to keep up the pace of the software, the last 10 years have been supported by the move from CPUs to more specialized GPU hardware, recently dedicated hardware such as TPU or similar have managed to keep up with the demand but alas, there are limits to how many processing units we can fit in a small space, distributed algorithm have notoriously been the solution to such problem but we have yet to develop efficient optimization for dense problems like neural networks.

The problem not only lies in the real world limitations but also in the precision required to understand how to update a single parameter in a network of 175 billion other parameters.

Today, we pay great attention to making the job of credit assignment to the network parameters as easy as possible by keeping the network gradients in check with techniques like residual networks, batch normalization, orthogonal initialization and others, this, like all heuristics adds some bias to the results.

---

## Overview

In section 2 we will first see how we can reformulate the training of a neural network as a constrained optimization problem allowing us to turn global gradients into local ones, hereby allowing us to solve the global problem by solving many small problems. Section 3 will show how past methods tried to solve the constrained optimization problem, Section 4 will instead show how the field of optimal control encountered the same class of problems with a different name and we are going to see some of the most fundamental approaches.

Section 5 will show how to efficiently solve large scale saddle point and finally, Section 6 will show how many current bio-inspired methods relate to the above framework.

---

## Feed-forward Neural Networks

With the quest of subdivision we now look at a feed-forward neural network as a nested function in the form:

$$\hat{y}=\phi(x_0; w)=f_{L-1}(\dots f_i(\dots f_0(x_0; w_0)\dots;w_i)\dots;w_L)$$

And it's associated loss function and optimization problem:

$$\underset{w}{\operatorname{argmin}} \ell(f_{L-1}(\dots f_i(\dots f_0(x_0; w_0)\dots;w_i)\dots;w_L), y)$$

Where $x_0$ is the given data-set and $y^{[j]}$ are the target values for each $x_0^{[j]}$ element, $\ell$ is any kind of "loss function" that maps a set of network predictions $\hat{y}^{[j]}$ to a scalar value.

We can specify this as a layer-wise sequence of functions $f_i$ applied to the output of the previous layer $h_i$:

$$\begin{cases}
h_{1} = f_i(x_0; w_0)\\\\
\vdots\\\\
h_{i+1} = f_i(h_i; w_i)\\\\
\vdots\\\\
\hat{y}=h_{L} = f_{L-1}(h_{L-1}, w_{L-1})
\end{cases}$$

$f_i(\cdot; w_i)$ represents the input-output mapping (layer transfer function) of any type of feed-forward neural network layer such as fully connected or convolution layer including related non-linear activations.

### Constrained Formulation

LeCun (1988) shows how to cleanly cast the back-propagation algorithm into an equivalent Lagrangian formulation.

We can now start decoupling the network layers by adding extra decision variables $h_i$ as targets for each layer. Each layer now has a target $h_i$ independent on the rest of the network, but as long as the constraints $h_{i+1} - f_i(h_i, w_i)$ are satisfied the two network dynamics are equivalent.

The new optimization problem is now:

$$\underset{(w, h)}{\operatorname{argmin}} \ell(h_L, y) \quad \text{s.t.} \quad h_{i+1} = f_i(h_i; w_i)$$

Problems of constrained optimization can be solved by the method of the Lagrangian multipliers where we define a Lagrangian function:

$$L(x_0;w,h,\lambda) = \ell(h_L, y)+ \lambda^T H(x_0; w, h)$$

And seek to find $(w, h, \lambda)$ s.t. $\nabla L(x_0; w, h, \lambda)=\boldsymbol{0}$

---

## Unconstrained Optimization Algorithms

While LeCun (1988) defines the problem statement and identifies sufficient conditions we are not provided with a way to solve the optimization problem. I will now show some proposed optimization methods to solve the constrained problem.

### Method of the Auxiliary Coordinates

Carreira-Perpinan & Wang (2012) introduces a mathematical formulation (meta-algorithm) named "MAC" to divide the training of a general neural network into sub-problems.

The search of parameters is done by using a penalty based method over $(w, h)$, solving a sequence of problems for $\lambda_0 \to \infty$ by alternating the minimization over $w_i$ and $h_i$.

### Gradient Descent Ascent

Betti (2018) takes LeCun's stationary conditions and defines a search approach in the $(w, x, \lambda)$ space by gradient descent ascent (GDA).

### Block-wise Training

Gotmare introduces "BlockProp", an algorithm that splits the unconstrained optimization problem into a constrained problem over blocks of $\geq 1$ layers.

---

## Discrete Time Optimal Control

To better understand why this constrained optimization approaches, it might be useful to look back at other fields using the same tools to solve a different problem of long-term dependencies in control systems.

The problem of minimizing a final cost function (and optionally a layer-wise one too) is also found in optimal control where we look for a sequence of states $h_i$ and controls $w_i$ to minimize the costs $g_i$ and $\ell(h_L, y)$ within the realizable dynamics $h_{i+1} = f_i(h_i; w_i)$.

### Main Approaches

All trajectory methods can be described as either **shooting** methods or simultaneous **collocation** methods. Shooting methods use the system dynamics to explicitly forward the states from inputs $h_0$ to outputs $\hat{y}$, while simultaneous collocation methods define a sequence of intermediate values $h_i$ and solve for the weights $w_i$.

### Single Shooting

These methods are analogous to classical back-propagation. Shooting methods and back-propagation have similar known issues:

1. **Uninformative gradients:** Shooting methods have local minima caused by the presence of narrow regions where the local information provided by the gradient is too little.

2. **Dependency on parameter initialization:** Both fields rely on good initial guesses to guarantee well-behaved optimization.

### Multiple Shooting

Multiple shooting defines intermediate decision variables $h_i$ and imposes $h_{i+1} = f_i(h_i; w_i)$ constraints. Such methods lend themselves to parallelization.

With optimization using soft constraints, the iterates $(w, h)_k$ are allowed to exist outside the valid optimization space, allowing the optimization path to converge more robustly and with fewer iterations to the optimal solution.

### Collocation

In collocation methods, we forget about the weights $w_i$ and look for a sequence of network activations $h_i$ that minimize the loss function.

---

## Variational Inequalities

Up to now, we have seen different perspectives of the same problem, solving these problems at the scale of modern datasets would prove challenging. We would like to have a first-order algorithm with good enough convergence speed. The field of Variational Inequalities can provide such answers.

### Extragradient (EG)

If we look inside the function we want to optimize we see that it has the form of a two-player $(\theta, \lambda)$ game. Gidel shows that in even the simplest bi-linear games Gradient Descent Ascent (GDA) fails to converge, this is caused by rotations around the fixpoint.

Korpelevich (1976) dampens circling behaviour by performing an extrapolation step (look ahead) for both players and applying the extrapolated gradients instead of the original ones.

### Extrapolation from the Past

Gidel introduces an "extrapolation from the past", allowing us to avoid the computation cost of evaluating two gradients at every iterate.

---

## Brain-inspired Back-Propagation

This section reviews alternative optimization methods. These works depart from the optimization view and often also from accurate optimization objectives (which is a proxy for the real generalization objective anyway) in favour of other heuristic objectives.

None of the methods try to exactly mimic the brain but rather draw inspiration from nature. One of the major points is that to reproduce accurately back-propagation, each neuron should store both forward (primal) and backward (tangent) gradient and their values which is not empirically observed in the brain.

### Target Propagation

Krogh (1989) solves the Lagrangian with L1 distance and a fixed constant by gradient descent. They also show how to impose per hidden constraints to control the learned hidden representations.

### Difference Target Propagation

Global gradients require knowledge of the full network and this is not biologically plausible. Lee (2015) side steps the issue relying on target values and approximate inverse dynamics. This method is interesting because it encodes internal models for forward and backward dynamics by auto-encoders.

### Neural Gradient Representation by Activity Differences

Lillicrap (2020) asks whether the brain is in fact approximating backpropagation. They introduce an NGRAD Hypothesis which suggests the human cortex uses an NGRAD based method to approximate gradient descent because of the impossibility of estimating global gradients.

---

## Conclusion

One of the controversial points of backpropagation in neural networks is the lack of local computation, yet it works so well there is no reason to move away from it.

As we scale the size of neural networks we are starting to question the current optimization methods. Problems like long term credit assignment has been part of the field of discrete-time optimal control and solved with some degree of success, so it seems worth a try to stand on the shoulders of those giants.

There are still open questions such as how to use stochastic optimization methods, which Gotmare shows some degree of success by ignoring the problem and sampling the constraints as well but this approach has not been shown to scale to modern datasets.
