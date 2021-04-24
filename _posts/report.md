---
author:
- Manuel Del Verme
bibliography:
- 'cit.bib'
title: Divide et Impera and Neural Networks
---

Decomposing a problem into smaller and more manageable pieces to be
solved independently and then merging the solutions has been a common
motif inspiring many fundamental advancements (Search, ordinary
differential equations, Kepler etc.).

In this survey, I will present several approaches to divide the
parameter estimation problem for neural networks by back-propagation
into simpler sub-problems.

Neural networks have recently had a strong impact in both our day to day
life and in other fields, partially also due to this reason, the number
of parameters of the modern models keeps increasing with no yet clear
limits on performance, yet the hardware to train such networks won't be
able to keep up the pace of the software, the last 10 years have been
supported by the move from CPUs to more specialized GPU hardware,
recently dedicated hardware such as TPU or similar have managed to keep
up with the demand but alas, there are limits to how many processing
units we can fit in a small space, distributed algorithm have
notoriously been the solution to such problem but we have yet to develop
efficient optimization for dense problems like neural networks.

The problem not only lies in the real world limitations but also in the
precision required to understand how to update a single parameter in a
network of 175 billion other parameters.

Today, we pay great attention to making the job of credit assignment to
the network parameters as easy as possible by keeping the network
gradients in check with techniques like residual networks, batch
normalization, orthogonal initialization and others, this, like all
heuristics adds some bias to the results.

Overview
========

In section [2](#sec:nn){reference-type="ref" reference="sec:nn"} will
first see how we can reformulate the training of a neural network as a
constrained optimization problem allowing us to turn global gradients
into local ones, hereby allowing us to solve the global problem by
solving many small problems. Section
[3](#sec:unconstrained){reference-type="ref"
reference="sec:unconstrained"} will show how past methods tried to solve
the constrained optimization problem, Section
[4](#sec:oc){reference-type="ref" reference="sec:oc"} will instead show
how the field of optimal control encountered the same class of problems
with a different name and we are going to see some of the most
fundamental approaches.

Section [5](#sec:vi){reference-type="ref" reference="sec:vi"} will show
how to efficiently solve large scale saddle point and finally,
[6](#sec:brain){reference-type="ref" reference="sec:brain"} will show
how many current bio-inspired methods relate to the above framework.

Feed-forward neural networks {#sec:nn}
============================

With the quest of subdivision we now look at a feed-forward neural
network as a nested function in the form:\
$$\begin{aligned}
    \hat{y}=\phi(x_0; w)=f_{L-1}(\dotsf_i(\dotsf_0(x_0; w_0)\dots;w_i)\dots;w_L)
    \label{fullforwardprop}\\
    \tag*{full forward-propagation}\end{aligned}$$

And it's associated loss function and optimization problem:

$$\begin{aligned}
& \underset{w}{\operatorname{argmin}}& \ell(f_{L-1}(\dotsf_i(\dotsf_0(x_0; w_0)\dots;w_i)\dots;w_L), y)\\ 
& \operatorname{given} & x_0, y \\
& \end{aligned}$$ Where $x_0$ is the given data-set and $y^{[j]}$ are
the target values for each $x_0^{[j]}$ element, $\ell$ is any kind of
\"loss function\" that maps a set of network predictions $\hat{y}^{[j]}$
to a scalar value.

We can specify
[\[fullforwardprop\]](#fullforwardprop){reference-type="eqref"
reference="fullforwardprop"} as a layer-wise sequence of functions $f_i$
applied to the output of the previous layer $h_i$:

$$\begin{aligned}
\begin{cases}
h_{1} = f_i(x_0; w_0)\\
...\\
h_{i+1} = f_i(h_i; w_i)\\
...\\
\hat{y}=h_{L} = f_{L-1}(h_{L-1}, w_{L-1})\\
\end{cases}
\label{fullforwardpropdynamics}
\\
\tag {full propagation dynamics}\end{aligned}$$

$f_i(\cdot; w_i)$ represents the input-output mapping (layer transfer
function) of any type of feed-forward neural network layer such as fully
connected or convolution layer including related non-linear activations.

The classical way to train neural networks is to use gradient descent
and back-propagation, to do that we need to define the Jacobian of the
loss function with the form:

$$\begin{aligned}
J(x_0; w) =
\begin{bmatrix}
\frac{\partial f_{0}(x_0; w)}{\partial w_{0}} & 0 & 0 & \dots\\
\frac{\partial f_{1}(x_0; w)}{\partial w_{0}} & \frac{\partial f_{1}(x_0; w)}{\partial w_{1}} & 0 & \dots\\
\frac{\partial f_{2}(x_0; w)}{\partial w_{0}} & \frac{\partial f_{2}(x_0; w)}{\partial w_{1}} & \frac{\partial f_{2}(x_0; w)}{\partial w_{2}} & \dots &\\
\vdots &&& \ddots \\
\frac{\partial f_{L-1}(x_0; w)}{\partial w_{0}} & \frac{\partial f_{L-1}(x_0; w)}{\partial w_{1}} & \frac{\partial f_{L-1}(x_0; w)}{\partial w_{2}} & \dots & \frac{\partial f_{L-1}(x_0; w)}{\partial w_{L-1}}
\end{bmatrix}\end{aligned}$$

By looking at this matrix we can immediately notice that while the
left-most columns related to the earlier layers are more complex in
terms of partial gradients than the last layers, this is connected to
the issues of vanishing and exploding gradients.

Constrained Formulation
-----------------------

[@LeCun1988] shows how to cleanly cast the back-propagation algorithm
into an equivalent Lagrangian formulation .

We can now start decoupling the network layers by adding extra decision
variables $h_i \enspace \forall i \in (1..L-1)$ as targets for each
layer:

$$\begin{aligned}
\left\{
    \begin{array}{ll}
      \hath_{1} = f_i(x_0; w_0)\\
      ... \\
      \hath_{i+1} = f_i(h_i; w_i)\\
      ... \\
      \hath_{L} = f_{L-1}(h_{L-1}, w_{L-1})\\
    \end{array}
\right.
\text{subject to}
\left\{
    \begin{array}{ll}
      h_{1} - f_0(x_0, w_0)=0 \\
      ... \\
      h_{i+1} - f_i(h_i, w_i)=0\\
      ... \\
      h_{L-1} - f_{L-2}(h_{L-2}, w_{L-2})=0\\
    \end{array}
\right.
\tag*{constrained dynamics}
\\
\label{constrfullforwardpropdynamics}\end{aligned}$$

Each layer now has a target $h_i$ independent on the rest of the
network, but as long as the constraints $h_{i+1} - f_i(h_i, w_i)$ are
satisfied the two network dynamics are equivalent.

The new optimization problem is now: $$\begin{aligned}
& \underset{(w, h)}{\operatorname{argmin}}\enspace & \ell(h_L, y)\nonumber \\
& \text{s.t.}  &  h_{i+1} = f_i(h_i; w_i)\label{prob:lecun88}\\
& \operatorname{given} & x_0=h_0 \nonumber \\
& \forall i \in {0,...,(L-1)} \nonumber\end{aligned}$$

We can now define a \"one step\" forward-propagation function which
takes the $h_i$ as input and generates $\hat{h}_{i+1}$:

$$\begin{aligned}
\hat h= F(x_0; w, h) =
\begin{bmatrix}
f_0(x_0; w_0) \\
\vdots \\
f_{L-1}(w_{L-1}, h_{L-1})\\
\end{bmatrix}\end{aligned}$$

Unlike [\[fullforwardprop\]](#fullforwardprop){reference-type="eqref"
reference="fullforwardprop"} this kind of process moves one layer at a
time because each input is defined by the decision variable $h_i$ not by
the sequential application of the whole neural network.

And the constraints: $$\begin{aligned}
H(x_0; w, h) =
\begin{bmatrix}
h_{1} - f_0(x_0, w_0)\\
h_2 - f_1(h_1, w_1)
\vdots\\
h_{L-1} - f_{L-2}(h_{L-2}, w_{L-2})\\
0\\
\end{bmatrix}\end{aligned}$$ The null constraint comes from the fact we
have no defined target value, just a loss function to constrain the
network output.

We now have decoupled the (gradients of) the layers this can be seen by
the gradient, which has a semi-diagonal form.

$$\begin{aligned}
\nabla_hH(x_0; w, h) =
\begin{bmatrix}
1 & 0 & ... & &\\
-\nabla_{h_1} f_0(h_1, w_0)& 1 & ... & &\\
& \ddots &\\
&& -\nabla_{h_{L-2}} f_{L-2}(h_{L-2}, w_{L-2})& 1 \\
% &&&& -\D_{\x_{L-1}} f_{L-1}(\x_{L-1}, \w_{L-1})\\
\dots & 0 & \dots \\
\end{bmatrix}\end{aligned}$$

Problems of constrained optimization can be solved by the method of the
Lagrangian multipliers where we define a Lagrangian function:
$$L(x_0;w,h,\lambda) = \ell(h_L, y)+ \lambda^T H(x_0; w, h)$$

And seek to find ($w, h, \lambda$) s.t.
$$\nabla L(x_0; w, h, \lambda)=\boldsymbol{0}$$ $$\begin{aligned}
\nabla_\lambda L = 0 \label{eq:sc_lambda}\\
\nabla_hL = 0 \label{eq:sc_x}\\
\nabla_wL = 0 \label{eq:sc_w}\end{aligned}$$

When those three conditions are satisfied we can recover the classical
full forward propagation equation from eq.
[\[eq:sc\_lambda\]](#eq:sc_lambda){reference-type="ref"
reference="eq:sc_lambda"}, the classical unconstrained gradients (eq.
[\[eq:sc\_x\]](#eq:sc_x){reference-type="ref" reference="eq:sc_x"}) and
optimality conditions for the unconstrained problem of minimizing $w$
(eq. [\[eq:sc\_w\]](#eq:sc_w){reference-type="ref"
reference="eq:sc_w"}).

#### Forward Propagation

It is trivial to recover forward propagation from (eq.
[\[eq:sc\_lambda\]](#eq:sc_lambda){reference-type="ref"
reference="eq:sc_lambda"}), in fact we have:
$$\nabla_\lambda L = H = 0 \implies h_{i+1} = f_i(h_i; w_i)$$ which is
the definition of the network dynamics
[\[fullforwardpropdynamics\]](#fullforwardpropdynamics){reference-type="eqref"
reference="fullforwardpropdynamics"}.

#### Unconstrained gradient

Eq. [\[eq:sc\_x\]](#eq:sc_x){reference-type="ref" reference="eq:sc_x"}
is more interesting:

For the last layer we get:
$$\nabla_hL = \nabla_h\ell + \lambda^T \nabla_hH = \nabla_h\ell(h_L, y)+ \nabla_h\lambda^T (h_{L-1} - f_{L-2}(h_{L-2}, w_{L-2}))$$

For $h_{L-1}$ we have:
$$\nabla_hL =\nabla_h\enspace \ell(h_L, y)+ \lambda_{L-1} = 0$$
$$\lambda_{L-1} = -\nabla_h\ell(h_L, y)$$

Which is the same gradient w.r.t the last layer in classical
back-propagation.

This tells us that we are looking for an optimal point $(w, h, \lambda)$
where the classical forward propagation
[\[fullforwardpropdynamics\]](#fullforwardpropdynamics){reference-type="eqref"
reference="fullforwardpropdynamics"} is respected and its gradients are
encoded in $\lambda$.

Unconstrained Optimization Algorithms {#sec:unconstrained}
=====================================

While [@LeCun1988] defines the problem statement and identifies
sufficient conditions we are not provided with a way to solve the
optimization problem. I will now show some proposed optimization methods
to solve the constrained problem.

Method of the auxiliary coordinates
-----------------------------------

[@Carreira-PerpinanWeiranWangEECS2012] introduces a mathematical
formulation (meta-algorithm) named \"MAC\" to divide the training of a
general neural network into sub-problems.

The search of parameters is done by using a penalty based method over
$(w, h)$.

First we turn the original problem:

$$\begin{aligned}
& \underset{w}{\operatorname{argmin}}& \ell(f_{L-1}(\dotsf_i(\dotsf_0(x_0; w_0)\dots;w_i)\dots;w_L), y)\\ 
& \operatorname{given} & x_0, y \\
& \end{aligned}$$

tnto a constrained equivalent one with auxiliary variables $h_i$ as
usual,

$$\begin{aligned}
& \underset{(w, h)}{\operatorname{argmin}}\enspace & \ell(h_L, y)\\
& \text{s.t.}  &  h_{i+1} = f_i(h_i; w_i)\\
& \operatorname{given} & x_0=h_0 \\
& \forall i \in {0,...,(L-1)}\end{aligned}$$

Then we solve the constrained optimization problem solved via quadratic
penalty method where the formulation becomes again unconstrained as:

$$\begin{aligned}
& L(h, w, \lambda) = \ell(h_L, y)+ \lambda_0 ||\sum_{i=0}^{L-2} h_{i+1} - f_i(h_i, w_i)||^2\\
& \operatorname{given} x=h_0 \\\end{aligned}$$ For a fixed $\lambda_0$.

And then solve a sequence of problems for $\lambda_0 \to \infty$ by
alternating the minimization over $w_i$ and $h_i$.

#### $w$-step

The minimization over $w_i$ is now equivalent to minimizing $L$ single
layer problems in the form: $$\min_{w_{L-1}} \enspace \ell(h_L, y)$$ and
$$\min_{w_i} \enspace \lambda || h_{i+1} - f_i(h_i, w_i)||^2$$

#### $h$-step

The minimization over $h_i$ instead is:
$$\min_z \enspace \ell(h_L, y)+ \lambda ||\sum_{i=0}^{L-2} h_{i+1} - f_i(h_i, w_i)||^2$$

The problem is now optimized sequentially $h_i$ then $w_i$ by
Gauss-Newton for both $w_i$ and $h_i$.

Once the augmented problem is considered solved the auxiliary variables
$h_i$ are set to $h_{i+1} = f_i(h_i; w_i)$ to guarantee feasibility.

Gradient descent ascent
-----------------------

[@Betti2018] takes [@LeCun1988] stationary conditions and defines a
search approach in the $(w, x, \lambda)$ space by gradient descent
ascent (GDA):

$$\begin{aligned}
& \underset{(w, x)}{\operatorname{argmin}} & \ell(h_L, y)+ \sum_{i=0}^{N-1}c(h_i, w_i) \\
& \text{subject to}  &  h_{i+1}=f_i(h_i, w_i) \\
&\operatorname{given} & h_0=x_0 \\\end{aligned}$$ where
$c(\cdot, \cdot)$ is a per layer regularization factor.

$x=(x_1, x_2, .. x_N), u=(u_0, u_1, .. , u_{N-1})$

and it's Lagrangian
$$L(w, x, \lambda)=\ell(w, x)+\lambda \sum_{i=0}^{N-1}(x_{i+1} - f_i(x_i, u_i))$$

They now suggest to:

1.  GDA over $x, \lambda$ ($l \times n$ times)

2.  Gradient Descent over $w$ ($n \times pa(i)$ times)

3.  repeat

### Block-wise training

[@Gotmare] introduces \"BlockProp\", an algorithm that splits the
unconstrained optimization problem in to a constrained problem over
blocks of $\geq1$ layers

The paper then recognizes that we can consider groups of layers as a
\"block\" $b_i$: $b_i=f_i(w_i, f(\dots(f_j(w_j,h_j)))))$ and $w_{i:j}$
the weights connected to those layers $i..j$

Then backpropagation

$$\begin{aligned}
& \underset{w}{\operatorname{argmin}}& \ell(f_{L-1}(\dotsf_i(\dotsf_0(x_0; w_0)\dots;w_i)\dots;w_L), y)\\ 
& \operatorname{given} & x_0, y \\
& \end{aligned}$$

can be reformulated as:

$$\begin{aligned}
& \underset{(w, h)}{\operatorname{argmin}}\enspace & \ell(h_L, y)\\
& \text{s.t.}  &  h_{i+1} = b_i(h_i; w_i)\\
& \operatorname{given} & x_0=h_0 \\
& \forall i \in {0,...,(L-1)}\end{aligned}$$

They also purpose a stochastic view for the constrained problem, the
loss is now:
$$L(a^{[n]}, w, \lambda)= \ell(a^{[n]}_L, y)+ \sum_{i=0}^{L-1} \lambda_i ||a^{[n]}_{i+1} - b_i(a^{[n]}_i, w_i)||^2$$

Which is minimized by alternating optimization over $a^{[n]}_i$ then
$w_i$.

The optimization w.r.t $a^{[n]}_i$ is done using SGD then the
optimization w.r.t. $w_i$ using SGD again (albeit a different number of
steps are taken).

The literature reviewed so far has been focused on neural networks but
similar fields have posed the same problem and given alternative
solutions.

We are going to see how the trajectory optimization problem encountered
in discrete-time optimal control is equivalent to the problem of the
constrained neural networks
[\[prob:lecun88\]](#prob:lecun88){reference-type="eqref"
reference="prob:lecun88"}.

Discrete Time Optimal Control {#sec:oc}
=============================

To better understand why this constrained optimization approaches, it
might be useful to look back at other fields using the same tools to
solve a different problem of long-term dependencies in control systems.

The problem of minimizing a final cost function (and optionally a
layer-wise one too) is also found in optimal control where we look for a
sequence of states $h_i$ and controls $w_i$ to minimize the costs $g_i$
and $\ell(h_L, y)$ within the realizable dynamics
$h_{i+1} = f_i(h_i; w_i)$.

The most general formulation (in discrete time) is: $$\begin{aligned}
& \underset{(w, h)}{\operatorname{argmin}} & \ell(h_L, y)+ \sum_{i=0}^{L-1}g_i(h_i, w_i) \\
& \text{subject to}  &  h_{i+1}=f_i(h_i, w_i) \\
&\operatorname{given} & x_0=h_0 \\
\tag*{Learning Neural Nets as DTOC}\end{aligned}$$

We can safely set $g_{i<N}=0$ to ignore the intermediate ($g_{i<N}$)
costs, since they can be propagated forward by an identity transfer
function and added to the final cost. We find the original formulation
[\[prob:lecun88\]](#prob:lecun88){reference-type="eqref"
reference="prob:lecun88"}.

#### Main approaches

All trajectory methods can be described as either **shooting** methods
or simultaneous **collocation** methods. Shooting methods use the system
dynamics to explicitly forward the states from inputs $h_0$ to outputs
$\hat{y}$, while simultaneous collocation methods define a sequence of
intermediate values $h_i$ and solve for the weights $w_i$.

Single Shooting
---------------

These methods are analogous to classical back-propagation:

$$\begin{aligned}
& \underset{w}{\operatorname{argmin}}& \ell(f_{L-1}(\dotsf_i(\dotsf_0(x_0; w_0)\dots;w_i)\dots;w_L), y)\\ 
& \operatorname{given} & x_0, y \\
& \end{aligned}$$ here we have $w_i$ as decision variables, the
intermediate outputs $h_i$ defined implicitly by
$h_{i+1} = f_i(h_i; w_i)$.

The gradients estimated by back-propagation will guarantee optimization
paths included in the feasible region, meaning we have to respect the
$w$-space geometry.

To improve a solution we integrate across all the layers resulting in
the same issue backpropagation has. Shooting methods and
back-propagation have similar known issues:

1.  Uninformative gradients:\
    Shooting methods have local minima caused by the presence of narrow
    regions where the local information provided by the gradient is too
    little, for shallow neural network we have the same issue and with
    deep networks, while the optimization path is clear we have
    uninformative gradients (vanishing gradients) for the earliest
    weights of the network which make the local gradient in that space
    either explode or vanish.

2.  Dependency on parameter initialization:\
    Both fields rely on good initial guesses to guaranteed a
    well-behaved optimization, in neural networks we initialize layers
    with many general heuristics like variance scaling to allow the
    gradients to be of similar magnitude across all layers. Other forms
    of initializations are pretraining, using pre-trained weights such
    as word embeddings or auxiliary tasks.

Multiple Shooting
-----------------

Multiple shooting on the other hand defines intermediate decision
variables $h_i$ and imposes $h_{i+1} = f_i(h_i; w_i)$ constraints.

$$\begin{aligned}
& \underset{(w, h)}{\operatorname{argmin}}\enspace & \ell(h_L, y)\\
& \text{s.t.}  &  h_{i+1} = f_i(h_i; w_i)\\
& \operatorname{given} & x_0=h_0 \\
& \forall i \in {0,...,(L-1)}\end{aligned}$$

Such methods lend themselves to parallelization, methods such as
[@Gotmare] and [@Lee2015] are in this category.

With optimization using soft constraints, the iterates $(w, h)_k$ are
allowed to exist outside the valid optimization space
$$(w, \phi(x_0; w)) \enspace \forall \enspace w \in W$$ $\phi$ is
defined in eq.
[\[fullforwardprop\]](#fullforwardprop){reference-type="eqref"
reference="fullforwardprop"}

This space, normally available to back-propagation allowing the
optimization path to converge more robustly and with fewer iterations to
the optimal solution, the Lagrangian multipliers theorem guarantees the
optimum to be in $(w, \phi(x_0; w))$ for the convex case, in the
non-linear we can force it by running forward propagation one last time
with the final $w_i$ parameters.

[@Bock1985] splits the problem into sub-problems by defining time
intervals (similar to direct collocation) and solving each problem
independently.

In our case, each subproblem has the form of the constraint
$h_{i+1} - f_i(h_i, w_i)$.

we now have a system of non-linear equations $$H(h) = 
\begin{cases}
h_{1} - f_0(x_0, w_0)& = 0\\
...\\
h_{i+1} - f_i(h_i, w_i)& = 0 \\
...\\
h_{L-1} - f_{L-2}(h_{L-2}, w_{L-2})& = 0 \\
\end{cases}$$

Here the dynamics of the system are fixed hence the $w_i$ are not
variables. The above system $H(h)=0$ can be solved by Newton's Algorithm
finding iterates $h^k$ and Jacobian:

$$\begin{aligned}
H^\prime(h) =
\begin{bmatrix}
-I & 0 & ... &\\
\frac{\partial f_{0}(h_{i}, w_i)}{\partial h_0} & -I & ... &\\
0 & \frac{\partial f_{1}(h_{i}, w_i)}{\partial h_1} & -I & ... &\\
\vdots && \ddots \\
0 & ... & 0 & \frac{\partial f_{T-1}(h_{i}, w_i)}{\partial h_{T-1}} & -I
\end{bmatrix}\end{aligned}$$

Since both $H$ and $H^\prime$ are block-diagonal we can solve each block
in parallel allowing for parallel computation.

Collocation
-----------

In collocation methods, we forget about the weights $w_i$ and look for a
sequence of network activations $h_i$ that minimize the loss function.

We now have:

$$\begin{aligned}
& \underset{h}{\operatorname{argmin}} & \ell(h_L, y)\\
& \text{s.t.}  &  \varphi(h_i, h_{i+1}) = w_i\\
& \operatorname{given} & x=h_0 \\
& \forall i \in {0,...,(L-1)}\end{aligned}$$

Here we have an inverse layer dynamics function $\varphi$ that
implicitly defines the appropriate weights, this is similar to what
methods like [@Taylor2016] are trying to achieve.

#### Direct collocation by sequential, polynomial interpolation

[@Biegler1984] uses orthogonal collocation, it defines an appropriate
class of interpolating polynomial functions (Lagrange polynomial) and
fits them at splitting points similar to those identified by [@Gotmare].

1.  Defines a set of collocation points $h_i$, this is equivalent to
    choosing $b_i$ in [@Gotmare]\

2.  And defines an interpolating polynomial:
    $$\hat h_i(t)=\sum_{k=0}^N h_k^{(i)}L_k(t), i \in [1..n]$$
    $$\hat w_j(t)=\sum_{k=0}^N h_k^{(i)}L_k(t), j \in [1..m]$$

with $L_k$ the N-th degree Lagrange polynomial:
$$L_k(t)=\prod^N_{l=0,l\neq k} \frac{t - t_l}{t_k-t_l}$$ Which
guarantees that $L_k(t_l)=1 \enspace \text{if} \enspace l=k, \enspace 0$
otherwise.

Conclusion
----------

There is an almost perfect match between optimization for neural
networks and solving discrete control problems. the DCOPT perspective
lets us neatly characterize a continuum of methods, backpropagation
(single shooting) looks for the optimal network parameters and defines
the intermediate values implicitly, collocation methods on the other
hand look for the optimal states letting the network parameters be
defined implicitly. An important distinction in the case of single
shooting methods for neural network is that they scale with the network
size while collocation methods scales with the dataset size so
collocation methods could require a lot of resources.

Variational Inequalities {#sec:vi}
========================

Up to now, we have seen different perspectives of the same problem,
solving these problems a at the scale of modern datasets would prove
challenging.

We would like to have a first-order algorithm with good enough
convergence speed.

The field of Variational Inequalities can provide such answers, up to
now the approaches shown either relied on strong assumptions about the
network layer structure, for example [@Biegler1984] implies the layers
are of the Lagrange polynomial class, [@Betti2018] allows for a general
class of layers but uses a gradient descent ascent, which is notoriously
unstable [^1] for the general case we need to look at saddle point
optimization such the following.

Yet to be able to use the same methods we must first translate the
constrained optimization problem
[\[constrfullforwardpropdynamics\]](#constrfullforwardpropdynamics){reference-type="ref"
reference="constrfullforwardpropdynamics"} to the VI framework.

 VI problem formulation.
-----------------------

Because from the constrained optimization point of view both the weights
$w_i$ and the activations $h_i$ are equally considered parameters let
$\theta:=(w, h)$ the concatenation of both vectors be the variable of
interest for this section.

To recap, in the unconstrained case of section
[2](#sec:nn){reference-type="ref" reference="sec:nn"} we wanted the
gradient at $(\theta^\star, \lambda_0)$ to be zero where $\lambda_0$ is
a tradeoff constant, for the penalty based methods we look for something
in the form $||\nabla L(\theta^\star, \lambda)|| = 0$ with
$\lambda\rightarrow\infty$.

We are now going to look for a stationary point for which the cost is
non-negative in any feasible direction, this can be expressed as:

$$\begin{aligned}
\begin{cases}
\nabla_\theta \enspace \enspace L(\theta^\star, \lambda^\star)^T(\theta-\theta^\star) & \geq 0 \\
\nabla_\lambda -L(\theta^\star, \lambda^\star)^T(\lambda-\lambda^\star) & \geq 0 \\
\end{cases}
\\
\label{VIP_cond}\end{aligned}$$

$\forall (\theta,\lambda) \in (\Theta\times\Lambda)$ in the most general
case $\Theta = W \times \mathbb{R}^m$ and $\Lambda=\mathbb{R}^m$ with
dim($\mathcal{X}$) = m.

We can simplify the formulation
[\[VIP\_cond\]](#VIP_cond){reference-type="ref" reference="VIP_cond"}:
$$\begin{aligned}
F:=(\nabla_\theta L, -\nabla_\lambda L)\\
x:=(\theta, \lambda)\end{aligned}$$

And have: $$\begin{aligned}
F(x^\star)^T(x-x^\star)\geq0, \forall x \in \mathcal{X}
\label{eq:vip}\end{aligned}$$\
The problem $\eqref{eq:vip}$ is also called the Variational inequality
problem, $VI(\mathcal{X}, F)$.

Projection method
-----------------

One of the first methods to solve VIs, in general, was proposed by
$\cite{stampacchia1964formes}$, where they establish the equivalence
between $VI(\mathcal{X}, F)$ and a fixed point problem using a
projection operator, VIs to be solved by iterative methods.

A fixpoint in our case would be an $x^\star \in \mathcal{X}$ s.t.
$$T(x^\star)=x^\star$$

Where $T: \mathcal{X}\rightarrow \mathcal{X}$ is the mapping defining an
iteration defined by [@Stampacchia1970] as:

$$x_{k+1}=P_C(x_k−\alpha F(x_k))$$

where $P_C$ is an orthogonal projection onto C and $\alpha>0$ is a
positive step size.

Note that $P_C$, in the case of constrained neural networks, is not the
projection on the constraint set $P_h(\theta)$ defined before, since our
$VI(\mathcal{X}, F)$ encodes the unconstrained Lagrangian optimization.

Extragradient (EG)
------------------

If we look inside the function we want to optimize we see that it has
the form of a two-player ($\theta, \lambda)$ game
$F:=(\nabla_\theta L, -\nabla_\lambda L)$, [@Gidel] shows that in even
the simplest bi-linear games Gradient Descent Ascent (GDA) fails to
converge, this is caused by rotations around the fixpoint.

[@KORPELEVICH1976] dampens circling behaviour by performing an
extrapolation step (look ahead) for both players and applying the
extrapolated gradients instead of the original ones, this method does
not require strong monotonicity, which makes it a better candidate for
neural networks.\
We are interested in finding a saddle point $VI(\mathcal{X}, F)$ of the
Lagrangian $L(x_0;w,h,\lambda) = \ell(h_L, y)+ \lambda^T H(x_0; w, h)$

where: $F = [1, -1]^T\nabla L$

EG follows the simple two step update rule:\
$$\begin{aligned}
\overline{x}=P_C(x^k−\alpha F(x^k))\\
x^{k+1}=P_C(x^k−\alpha F(\overline{x}))\end{aligned}$$

In our case, the Euclidean projection $P_C$ is not necessary.
$$\begin{aligned}
\overline{x}=x^k−\alpha F(x^k)\\
x^{k+1}=x^k−\alpha F(\overline{x})\end{aligned}$$

And to keep in mind the function of interest: $$\begin{aligned}
\overline{x}=x^k−\alpha \begin{bmatrix} \nabla_\theta L \\ -H(\theta) \end{bmatrix}(x^k)\\
x^{k+1}=x^k−\alpha \begin{bmatrix} \nabla_\theta L \\ -H(\theta) \end{bmatrix}(\overline x)\end{aligned}$$

Bregman proximal, MirrorProx
----------------------------

[@Nemirovski2005] defines a Bergman Divergence function
$D_\varphi(x, x^k)$, that allows us to generalize ExtraGradient to
different divergence functions.

$$\begin{aligned}
\overline{x}=\operatorname{argmin} \alpha \langle F(x^k), x \rangle + D_\varphi(x, x^k)\\
x^{k+1}=\operatorname{argmin} \alpha \langle F(\overline{x}), x \rangle + D_\varphi(x, x^k)\\\end{aligned}$$
At every iteration MP computes $F(x^k)$ at $x^k$ and extrapolates a
point $\overline{x}$ over which re-evaluate $x^k$

To recover ExtraGradient is enough to let
$D_\varphi(x, y) = \frac{1}{2}\langle x, y \rangle$, the Euclidean case.

$D_\varphi(x^k, x)$ allows us to adjust gradient updates to fit problem
geometry via a non-euclidean penalty between iterates.

Extrapolation from the past
---------------------------

[@Gidel] Takes a problem in the form of $VI(\mathcal{X}, F)$
[\[eq:vip\]](#eq:vip){reference-type="ref" reference="eq:vip"} and then
introduces an \"extrapolation from the past\", allowing us to avoid the
computation cost of evaluating two gradient at every iterate.

\"ExtraGradient from the past\" iterates are in the form:
$$\begin{aligned}
    \bar{x}^{k+1}=x^k - \alpha F(\bar {x}^k) \\
    x^{k+1}=x^k - \alpha F(\bar{x}^{k+1})\end{aligned}$$ At each
iteration $F(\bar x^{k+1})$ will become the next $F(\bar x^{k})$ thus
saving the old extrapolated gradient to be the next extrapolation
gradient.

The paper also introduces a stochastic VIPs formulation instead where
where instead of the gradient $F=\nabla [-1, 1]^TL$ we have access to an
unbiased estimate of it $F(x, \xi), \xi \sim P$ and
$F = \mathbb{E}_{\xi \sim P} [F(x,\xi )]$

Alas this can not apply to constrained neural networks because we do not
have access to such estimate as our Lagrangian:
$$L(x_0;w,h,\lambda) = \ell(h_L, y)+ \lambda^T H(x_0; w, h)$$

Depends on $h_i$ which if sampled would mean to have $x_0 \sim P_X$ as
it's the case here but we also have $h_i \sim P_h$ which means we are
only aiming to satisfy the constraints in expected value and not almost
surely, ultimately more work is required to fully understand this last
difference.

Brain-inspired Back-Propagation {#sec:brain}
===============================

This section reviews alternative optimization methos, these works depart
from the optimization view and often also from accurate optimization
objectives (which is a proxy for the real generalization objective
anyway) in favour of other heuristically objectives.

None of the methods try to exactly mimic the brain but rather draw
inspiration from nature, one of the major points is that to reproduce
accurately back-propagation, each neuron should store both forward
(primal) and backward (tangent) gradient and their values which is not
empirically observed in the brain.

Target Propagation
------------------

[@Krogh1989] solves:
$$\underset{(w, h)}{\operatorname{argmin}}\enspace L(x_0;w,h,\lambda) = \ell(h_L, y)+ \lambda^T H(x_0; w, h)$$
With L1 distance $\ell$ and $\lambda$ a fixed constant by gradient
descent.

They also shows how to impose per hidden constraints to control the
learned hidden representations by adding additional cost terms dependent
on the activation values $h$, this can be useful to control the learned
features.

Moving Targets Algorithm
------------------------

[@rohwer1990moving] aims to increase learning speed by a per-layer cost
function to allow the hidden variables to solve non-stationary local
problems. It differs from [@Krogh1989] by the parametrization of the
$f_i(h, w)$ transfer functions.

It has two phases $a_i$-step reducing the loss by gradient descent and a
second $w_i$-step minimizing the constraint cost.

Deeply Supervised Networks
--------------------------

[@lee2015deeply] considers multiple joint losses and calculates
gradients of the joint loss.

$$\underset{w}{\operatorname{argmin}}\sum_{i=1}^L \lambda_i \ell(f_i(..f_0(x_0; w_0)..,w_i), y)$$
Where lambda is now an annealed constant.

The various losses are not calculated at the last layer but at every
layer $i$.

Similarly to [@rohwer1990moving] and [@Krogh1989] this approach aims to
provide proxy targets for the hidden layers.

Difference Target Propagation
-----------------------------

Global gradients require knowlege of the full network and this is not
biologically plausible. [@Lee2015] side steps the issue relying on
target values $a_i$ and approximate inverse dynamics $g_i$.

This approach is similar to [@Biegler1984] Collocation methods since
both methods have access to approximate dynamics.

We now have local loss functions
$\ell_i = ||h_{i+1} - f_i(h_i, w_i)||^2$ for the intermediate layers and
a standard last layer function $\ell(h_L, y)$

The last layer is trained by standard gradient descent, the interesting
part of this approach is the optimization algorithm for intermediate
layers.

For them we use an approximate inverse $g_i(\cdot)$ that satisfies the
following rules: $$\begin{aligned}
f_i(g_i(h_{i+1}))\approx h_i\\
g_i(f_i(h_i))\approx h_i\end{aligned}$$

At every iteration we first update the target values by:
$$\hat{h_i} = h_i - g_i(h_i) + g_i(\hath_i)$$

And then update the weights of $g_i$ by minimizing:
$$\underset{W_g}{\operatorname{argmin}}||g_{i+1}(f_{i+1}(h_i + \epsilon); W_g)- (h_i + \epsilon)||^2$$
The $\epsilon$ term ensures the backward map is also estimated in a
neighbourhood $a_i$.

Now we update $f_i$ by minimizing:
$$\underset{W_f}{\operatorname{argmin}}||f_{i+1}(h_i)- \hath_i||^2$$

This method is interesting because it encodes internal models for
forward and backward dynamics, by auto-encoders, something novel in this
line of work.

Neural Gradient Representation by Activity Differences
------------------------------------------------------

[@Lillicrap2020] asks wheter the brain is infact approximating
backpropagation.

It investigates bio-plausibility of target propagation approaches and
tries to establish a framework called \"Neural Gradient Representation
by Activity Differences\" (NGRAD), they also introduce an NGRAD
Hypothesis which suggests the human cortex uses an NGRAD based method to
approximate gradient descent because of the impossibility of estimating
global gradients.

In particular, the NGRAD hypothesis suggests that many of the
bio-inspired algorithms approximate back-prop by implicitly representing
gradients in terms of neural activity (spatial or temporal) differences.

Conclusion
==========

One of the controversial points of backpropagation in neural networks is
the lack of local computation, yet it works so well there is no reason
to move away from it.

As we scale the size of neural networks we are starting to question the
current optimization methods, problems like long term credit assignment
has been part of the field of discrete-time optimal control and solved
with some degree of success, so it seems worth a try to stand on the
shoulders of those giants.

In the past and how similar fields dealt with the same issues arising
when trying to answer the question of divide et impera for complex
networks.

There are still open questions such as how to use stochastic
optimization methods, which [@Gotmare] shows some degree of success by
ignoring the problem and sampling the constraints as well but this
approach has not been shown to scale to modern datasets.

[^1]: empirical hints of instability comes from personal experience and
    the lack of follow up works to such fundamental works.
