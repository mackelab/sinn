# Sinn

*sinn* is a library for both *S*imulation and *IN*ference of dynamical systems. It provides a flexible framework for building complex mathematical models which are fully compatible with machine learning libraries, allowing almost arbitrary cost functions to be differentiated all the way back to the model parameters. Both the simulation model and cost function are compiled and run as C code.

## Motivation

Optimization frameworks like TensorFlow and PyTorch provide powerful capabilities for fitting models to data. However, they are most tailored to fitting neural networks, and implementing the type of dynamic mechanistic models often found in physics or applied mathematics within these frameworks remains error-prone and labour-intensive. *sinn* provides a set of high-level constructs designed to bridge the gap between the mathematical language of dynamical systems and the interface of a machine-learning library.

Sinn was originally developed in order to infer a mesoscopic neuron model (René et al., in press, Neural Computation; [arXiv](https://arxiv.org/abs/1910.01618)).

## Documentation

Partial documentation can be found on [Read the Docs](https://sinn.readthedocs.io/en/latest/). This will improve as development continues.

## Features

- Automatic differentiation and C-compilation provided by Theano, although one [could](#dependence-on-a-machine-learning-library) in theory use other use other machine-learning frameworks.

- Compatible with PyMC3

  Make your model probabilistic with a few extra lines of code, for easy implementation of Bayesian inference and Monte Carlo sampling.

- Use the optimization library only when desired.

  No code change is required to run models with either Numpy or Theano – the single line `shim.load('theano')` suffices to load the optimization library.
  Since a pure Numpy model does not require compilation every time it is run, this allows you to first develop your model faster with more easily traceable errors, and then benefit from the C-acceleration and automatic differentiation by loading the optimization library.

- Data structures which map naturally to the mathematical models

  + `Axis`: Unit-aware structure for continuous quantities such as time, space, temperature…
  + `DynArray`: combining *n*-dimensional data with *n* axes.
    A development goal is to allow easier translation to PyData's analogous `DataArray` (the main difference being that a `DynArray` is intended for data generation, and is associated to a function which can fill the entries *dynamically*).
  + `History`: A `DynArray` instance where one axis is time.

  **Note** This organization of `Axis`, `DynArray` and `History` is still work in progress and subject to change.

- Dynamic programming, aka lazy evaluation.

    Data is computed only as needed. This allows you to specify functions as

    x<sub>k</sub> = f<sub>x</sub>(x<sub>k-1</sub>, y<sub>k-1</sub>) \
    y<sub>k</sub> = f<sub>y</sub>(x<sub>k</sub>, y<sub>k-1</sub>)

    and then compute either *x* or *y* at any point *k<sup>\*</sup>*, without worrying¹ about the fact that f<sub>x</sub> and f<sub>y</sub> both depend on the arrays *x* and *y*, and without unnecessary calculations for points beyond *k<sup>\*</sup>*.

- Fully serializable models
  Models are implemented as [Pydantic](https://pydantic-docs.helpmanual.io) models, and can be easily exported as dictionaries or serialized JSON:

      mymodel.dict()
      mymodel.json()

  This is especially useful for repeating part of a workflow with different models or parameters, or deploying a model to a remote machine. Archiving the exact parameterization of a model is also a key component in a reproducible science workflow.

¹ Unless of course there is a circular dependency between f<sub>x</sub> and f<sub>y</sub>.

## Dependence on a machine-learning library

The choice of Theano as an underlying ML library is largely historical, and while I would likely make a different choice today, I currently have no plans to change this because:

  - It works ;-)
  - I like the functional style of Theano. It is more “math-like” then the declarative and imperative approaches used by TensorFlow and PyTorch respectively. My (absolutely untested) assumption is that in theory this makes fewer corner case with regard to automatic differentation, and I care more about robust differentiation than easy specification of for loops. I suspect this is an observation shared by the [JaX](https://jax.readthedocs.io) development team.
  - Static graph optimization at least in theory makes for faster executation.
  - Everything Theano related is routed through the [theano_shim](https://github.com/mackelab/theano_shim) backend, which removes much of the pain of debugging Theano code.

This last point also means that to use a different ML framework one would only need to port `theano_shim`. Most of the code in the backend is of the form

    import theano.tensor as tt
    def exp(x):
        if symbolic(x):
            return tt.exp
        else:
            return np.exp

(The required changes for a TensorFlow-compatible function are left as an exercise to the reader ;-). ) While not a negligible undertaking, porting `theano_shim` is thus certainly feasible.

## Development status

At present *sinn* is at a pre-alpha stage of development. Version 0.2 should settle the core API (everything related to the `History` and `Model` classes), but less mature elements may still see some changes.

### v0.2dev release

The current version is a near-complete rewrite of the library, with focus on eliminating stale and duplicated code, more natural model definitions, proper unit testing, and simpler integration into larger workflows. In particular, the use of *Pydantic* throughout means that model objects and parameters can be trivially saved and loaded from disk. Although some [planned changes](https://github.com/mackelab/sinn/issues/1) are still work in progress, the update of the core functionality is complete and already better tested than in the previous version.

These changes were motivated by my own frustrations with v0.1, with regard to managing large numbers of simulations and workflows with multiple steps.

### Disclaimer

Although *sinn* tries hard to protect users from their own mistakes, users should still treat it as any fallible tool and check that it performs as expected in their situation.

## Installation

- Clone this repository to e.g. `~/code/sinn`.

- Create the virtual environment if required, e.g.

  `conda create --prefix ~/envs/myproject` (Anaconda) \
  `python3 -m venv --system-site-packages ~/envs/sinn` (Python venv)

  You can omit `--system-site-packages` if you install all dependencies within the virtual environment.


- Activate the virtual environment

  `conda activate ~/envs/myproject` (Anaconda) \
  `source $HOME/usr/venv/sinn/bin/activate` (Python venv)

- `cd` to the directory containing this file.

- Install with

  `pip install .`

  As usual, if you want to be able to modify the code, add the `-e` flag to obtain a development installation.

## Running tests

  Install the development packages

      pip install sinn[dev]

  This will install `pytest` and `pytest-xdist`. Now run the test suite as

      pytest --forked

  The `--forked` option ensures that each test is run in a separate process. This is required to fully test *sinn* both with and without the auto-diff library loaded.


Copyright (c) 2017-2020 Alexandre René
