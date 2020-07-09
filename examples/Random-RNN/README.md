## Summary

This example illustrates *sinn* usage on a simple random RNN model, defined by

$$\begin{aligned}
   \dot{h}_i &= -h_i + \sum_{j=1}^N J_{ij} \phi(h_j) \\
   \phi(h_j) &= \tanh(g h_j)
   J &= \sim \mathcal{N}(0,1)
\end{aligned}$$

We call this the SCS model, for Sompolinsky, Crisanti and Sommers (PRL 1988).

The [notebook](./scs_match.ipynb) goes through the process of

- Initializing a model;
- Integrating it forward in time;
- Defining a cost function;
- Converting said cost function into updates for $J$ and $g$ (using automatic differentiation and [Adam](http://arxiv.org/abs/1412.6980));
- Iterating these updates to train the model to match the target.

The model itself is defined in the module [scs.py](./scs.py).

## Installation

If you are within a conda environment, you can install all requirements by executing (replace `ENVNAME` with the name of the environment):

    conda env update --file random-rnn.yaml --name ENVNAME
