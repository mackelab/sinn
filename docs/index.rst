.. Sinn documentation master file, created by
   sphinx-quickstart on Mon Apr  9 15:47:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sinn – User Manual
==================

.. Note::
   The best way to get started with *Sinn* is to use the `*Sinn-full* template <github.com/alcrene/sinnvoll>`_, which provides a fully functioning inference workflow. The documentation below is intended for those who need to extend that template, or otherwise want to dig deeper into the library underlying it.

Purpose
-------

At its core, the purpose of *Sinn* is to symbolically integrate update equations. The target application is for problems which can be cast into this form: Given some *code* implementing a closed set of update equations for variables :math:`x_1, \dotsc, \x_n` and parameters :math:`\Theta`,

.. math::

   x_{1,k+1} &= f_1(x_{1,k}, \dotsc, x_{n,k}; \Theta) \,, \\
   \vdots \\
   x_{n,k+1} &= f_n(x_{1,k}, \dotsc, x_{n,k}; \Theta) \,,
   
and a function :math:`\hat{g}` on the set of variables *at all time points* :math:`k`,

.. math::

   g\bigl(\{x_{1,k}, \dotsc, x_{n,k}\}_{k=1}^K; \Theta \bigr) \,,
     
*construct* a *symbolic* function :math:`g` equal to :math:`\hat{g}`, but requiring (at most) only the initial condition for each variable :math:`x_i`. The function :math:`g` must thus *integrate* the update equations :math:`f_1, \dotsc f_n` to generate the variables :math:`\{x_{1,k}, \dotsc, x_{n,k}\}_{k=1}^K`, which it can then pass to :math:`\hat{g}`.

Crucially, because :math:`g` is symbolic, it can both be *compiled* and *differentiated*. This makes it possible both to efficiently simulate (i.e. integration) models, and to construct inference algorithm to fit them using *exact analytical gradients*. Hence why we named this a package for *s*imulation and *in*ference.

Note that *Sinn* is agnostic to the manner in which the update equations were obtained: they can originate from both a truly discrete process or the update step of an ODE integration scheme.

.. _key-concepts:

Key concepts
------------

We define the following:

history (:math:`\mathcal{H}`)
   A combination of:
   
   - The value of a variable at all time points: :math:`\mathcal{H}_i := \{x_{i,k}\}_{k=1}^K`.
   - A time value associated to each index: :math:`\{t_k\}_{k=1}^K`.
   
parameters (:math:`\Theta`)
   The parameters on which the update functions depend.

model (:math:`\mathcal{M}`)
   A set *histories*, *parameters*, *update equations* and anything required to integrate the equations. This may include a *random number generator*, if some update equations are stochastic.

*Sinn* proposes a stateful API, in which the compilation of :math:`g` depends on a few state variables attached to the histories:

The *current time index*
   of a history is the latest time point for which its value has been computed.
   
The *lock state*
   of a history determines whether it can be updated. This indicates whether values for that history need to be computed (via its update function) or whether they are provided as initial conditions – analogous to setting a variable as “observed” in other packages, with the difference that the lock state is mutable. Locked and unlocked histories are treated differently during compilation. (TODO: Link to fig/explanation in models.rst explaining difference in compilation.)
   
This deviation from Theano's purely functional ideals allows to specify initial conditions in a natural way, and to compile multiple functions which lock different subsets of histories. For example, when constructing an :ref:`expectation-maximization scheme <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_, one would typically unlock the latent histories for the E-step, but lock them for the M-step.

:ref:`The graph below` provides an overview of the compilation steps.
   
.. important::
   A fundamental assumption is *Sinn* is that histories are *causal*. In other words:
   
   - That :math:`x_{i,k}` is computed *always implies* that :math:`x_{i,k-l}` (:math:`l>0`)  has also been computed.
   - To compute :math:`x_{i,k}`, it always suffices to have computed :math:`x_i` up to :math:`k-1`, and all other histories up to :math:`k`.
   
.. _compilation-overview-graph:

.. mermaid::
   :caption: Typical sequence of steps to compiling a function depending on model variables. Connection and compilation steps are performed automatically.
  
   flowchart LR
     subgraph create_model["Create model"]
       direction TB
       params[List parameters]
       hists[List histories]
       upds[List update functions]
       state[List state histories]
       hists --> ConnHU["Attach update function to histories"]
       upds --> ConnHU
     end
     
     subgraph init_model["Set model state"]
       direction TB
       init_hists["Set initial value(s) of state histories"]
       lock["Lock observed histories"]
     end
   
     subgraph compile_model["Compile g"]
       direction TB
       onestep["Construct computational graph for one time<br>step update of unlocked histories"]
       scan["Iterate ('scan') graph K steps forward"]
       compile["Compile iterated graph"]
       onestep --> scan --> compile
     end
   
     create_model --> init_model
     init_model --> onestep
     compile --> g


Core components
---------------

The concepts :ref:`described above <key-concepts>` are implemented via four core classes, which are described in the sections listed below. For a first reading, we recommend skimming through the :ref:`axis` section, and then focusing on the :ref:`histories` and :ref:`models` sections, since the latter two objects are the ones with which one interacts the most. Kernels are only needed for models involving convolutions, and thus can be read later.

.. toctree::
   :maxdepth: 2
   :caption: Core model components:

   axis
   histories
   kernels
   models

.. caution:: Negative indices are *not* special

   :py:mod:`Sinn` does not recognize negative indices (or at least not the way you may expect them to). This is primarily a protection for the user against hard-to-track bugs, but also helps make the interface more intuitive. (See :ref:`sinn-indexing`.)

Sinn – User API
===============

.. toctree::
  :maxdepth: 2

  user-api/histories-api
  user-api/axis-api

Sinn – Technical references
===========================

.. toctree::
   :maxdepth: 1

   internal/index
   developer-api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
