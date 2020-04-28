# Sinn

*sinn* is a library for both *S*imulation and *IN*ference of dynamical systems. It seeks to provide a flexible framework for building complex mathematical models which are fully compatible with machine learning libraries, allowing them to be differentiated through and compiled to C code.

## Motivation

Optimization frameworks like Theano and TensorFlow provide powerful capabilities for fitting models to data. However, they are most tailored to fitting neural networks – implementing the type of dynamic mechanistic models often found in physics or applied mathematics within these frameworks remains error-prone and labour-intensive. Sinn provides a set of high-level constructs designed to integrate well with lower-level optimization libraries.

Sinn was originally developed in order to infer a mesoscopic neuron model (René et al., in press, Neural Computation; [arXiv](https://arxiv.org/abs/1910.01618)).

## Features

- Automatic differentiation and C-compilation provided by Theano.

- Compatible with PyMC3

  Make your model probabilistic with a few extra lines of code, for easy implementation of Bayesian inference and Monte Carlo sampling.

- Use optimization library only when desired.

  No code change is required to run models with either Numpy or Theano – the single line `shim.load('theano')` suffices to load the optimization library.
  Since a pure Numpy model does not require compilation every time it is run, allowing you to first develop your model faster with more easily traceable errors, and then benefit from the C-acceleration and automatic differentiation by loading the optimization library.

- Data structures which map naturally to the mathematical models

  + `Axis`: Unit-aware structure for continuous quantities such as time, space, temperature…
  + `DataAxes`: combining *n*-dimensional data with *n* axes.
    A development goal is to allow easier translation to Pandas' analogous `DataFrame` (the main difference between a frame and an axis being that the latter is continuous by design)
  + `History`: A `DataAxes` instance where one axis is time.

  **Note** This organization of `Axis`, `DataAxes` and `History` is still incomplete work in progress and subject to change.


- Dynamic programming, aka lazy evaluation.

    Data is computed only as needed. This allows you to specify functions as

    x<sub>k</sub> = f<sub>x</sub>(x<sub>k-1</sub>, y<sub>k-1</sub>) \
    y<sub>k</sub> = f<sub>y</sub>(x<sub>k</sub>, y<sub>k-1</sub>)

    and then compute either *x* or *y* at any point *k<sup>\*</sup>*, without worrying¹ about the fact that f<sub>x</sub> and f<sub>y</sub> both depend on the arrays *x* and *y*, and without unnecessary calculations for points beyond *k<sup>\*</sup>*.

¹ Unless of course there is a circular dependency between f<sub>x</sub> and f<sub>y</sub>.

## Development status

At present Sinn is at a pre-alpha stage of development. Some of the core functionality is being reorganized to reduce code duplication and make model specification more intuitive. These changes should stabilize the API, but for now one should expect to run into the occasional bug.
In any case users should treat this library as any fallible tool and check that it performs as expected in their situation.

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


Copyright (c) 2017-2020 Alexandre René
