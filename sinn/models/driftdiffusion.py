# -*- coding: utf-8 -*-
"""
Drift-diffusion model (aka Brownian motion)

Created Wed Aug 7 2018

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict, Iterable, Callable
import logging

import theano_shim as shim
import sinn
from sinn.models.common import Model, register_model

# TODO: Package simpleeval boilerplate in mackelab.utils
import simpleeval
import ast
import operator

# TODO: Find a way to pre-evaluate strings to some more efficient expresison, so
#       we don't need to parse the string every time.
class StringFunction:
    # Replace the "safe" operators with their standard forms
    # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
    #  input but this does not work with non-numerical types.)
    _operators = simpleeval.DEFAULT_OPERATORS
    _operators.update(
        {ast.Add: operator.add,
         ast.Mult: operator.mul,
         ast.Pow: operator.pow})
    # Allow evaluation to find operations in standard namespaces
    namespaces = {'np': np,
                  'sp': sp}

    def __init__(self, expr, args):
        """
        Parameters
        ----------
        expr: str
            String to evaluate.
        args: iterable of strings
            The function argument names.
        """
        self.expr = expr
        self.args = args
    def __call__(self, *args, **kwargs):
        names = {nm: arg for nm, arg in zip(self.args, args)}
        names.update(kwargs)  # FIXME: Unrecognized args ?
        names.update(self.namespaces)  # FIXME: Overwriting of arguments ?
        try:
            res = simpleeval.simple_eval(
                self.expr,
                operators=self._operators,
                names=names)
        except simpleeval.NameNotDefined as e:
            e.args = ((e.args[0] +
                       "\n\nThis may be due to a module function in the transform "
                       "expression (only numpy and scipy, as 'np' and 'sp', are "
                       "available by default).\nIf '{}' is a module or class, you can "
                       "make it available by adding it to the function namespace: "
                       "`StringFunction.namespaces.update({{'{}': {}}})`.\nSuch a line would "
                       "typically be included at the beginning of the execution script "
                       "(it does not need to be in the same module as the one where "
                       "the string function is defined, as long as it is executed before)."
                       .format(e.name, e.name, e.name),)
                      + e.args[1:])
            raise
        return res

class DriftDiffusion(Model):
    """
    Implements a drift-diffusion model using Euler-Maruyama.
    """

    requires_rng = True
    Parameter_info = OrderedDict([])
        # Parameters will be within free-form function arguments to __init__
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['x'])

    def __init__(self, params, x_history,
                 drift=None, diffusion=None, random_stream=None):
        """
        Default `drift` and `diffusion` implement
            `dx = -x dt + dW`

        Parameters
        ----------
        drift: function (t, x) -> floatX (shape: x_history.shape)
           Function which takes a time and history object, and returns
           the drift. Should output the same shape as the history.
        diffusion: function (t, x) -> floatX (shape: x_history.shape)
           Function which takes a time and history object, and returns
           the diffusion. Should output the same shape as the history.
        """

        if drift is not None: self.drift = drift
        if diffusion is not None: self.diffusion = diffusion
        self.rndstream = random_stream

        super().__init__(params, public_histories=(x_history,))
        # NOTE: Do not use `params` beyond here. Always use self.params.

        self.x = x_history

        # Parameter tests
        Model.output_rng(self.x, self.rndstream)

        self.add_history(self.x)
        self.x.set_update_function(self.x_fn)
        self.x.add_input(self.x)

        if isinstance(drift, str):
            # Evaluate string to get function
            self.drift = StringFunction(drift, ('t', 'x'))
        elif drift is not None:
            # Just assign whatever was pased.
            assert(isinstance(drift, Callable))
            self.drift = drift
        if isinstance(diffusion, str):
            # Evaluate string to get function
            self.diffusion = StringFunction(diffusion, ('t', 'x'))
        elif diffusion is not None:
            # Just assign whatever was pased.
            assert(isinstance(diffusion, Callable))
            self.diffusion = diffusion

    def x_fn(self, t):
        t = self.x.get_tidx(t)
        return (self.drift(t, self.x) * self.x.dt
                + self.diffusion(t, self.x) * np.sqrt(self.x.dt))

    def drift(self, t):
        tidx = x.get_t_idx(t)
        return -x[tidx-1]

    def diffusion(self, t, x):
        tidx = x.get_t_idx(t)
        return 1

register_model(DriftDiffusion)
