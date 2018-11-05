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
from sinn.histories import Series
from sinn.models.common import Model, register_model

from mackelab.utils import StringFunction

class DriftDiffusion(Model):
    """
    Implements a drift-diffusion model using Euler-Maruyama.
    """

    requires_rng = True
    Parameter_info = OrderedDict([])
        # Parameters will be within free-form function arguments to __init__
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['x'])

    def __init__(self, params, x_history, random_stream=None,
                 drift=None, diffusion=None, namespace=None):
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
        namespace: dict
            If either `drift` or `diffusion` are string expressions,
            `namespace` is added to their SimpleEval namespaces. Allows to
            define parameters which can be used in the string expressions.
        """

        #if drift is not None: self.drift = drift
        #if diffusion is not None: self.diffusion = diffusion
        self.rndstream = random_stream

        # Argument tests
        if not isinstance(x_history, Series):
            raise TypeError("`x_history` argument must be a sinn `Series`.")
        Model.output_rng(x_history, self.rndstream)

        super().__init__(params, public_histories=(x_history,))
        # NOTE: Do not use `params` beyond here. Always use self.params.

        self.x = x_history

        self.add_history(self.x)
        self.x.set_update_function(self.x_fn)
        self.x.add_input(self.x)

        if isinstance(drift, str):
            # Evaluate string to get function
            self.drift = StringFunction(drift, ('t', 'x'))
            if namespace is not None:
                self.drift.namespaces.update(namespace)
        elif drift is not None:
            # Just assign whatever was pased.
            assert(isinstance(drift, Callable))
            self.drift = drift
        if isinstance(diffusion, str):
            # Evaluate string to get function
            self.diffusion = StringFunction(diffusion, ('t', 'x'))
            if namespace is not None:
                self.diffusion.namespaces.update(namespace)
        elif diffusion is not None:
            # Just assign whatever was pased.
            assert(isinstance(diffusion, Callable))
            self.diffusion = diffusion

    def initialize(self, init_cond=None):
        if init_cond is not None:
            # Initialize the history
            init = init_cond['x']
            if init.shape == self.x.shape:
                # Add time dimension
                init = init.reshape((1,) + self.x.shape)
            elif not init.shape[1:] == self.x.shape:
                raise ValueError("`init_cond` (shape: {}) doesn't match "
                                 "the expected shape ({})."
                                 .format(init.shape, self.x.shape))
            self.x.pad(len(init))
            self.x[:len(init)] = init


    def x_fn(self, t):
        t = self.x.get_tidx(t)
        return self.x[t-1] + (self.drift(t, self.x) * self.x.dt
                              + self.diffusion(t, self.x) * self.dW())

    def dW(self):
        """
        Returns a step from a Wiener process. Specifically, samples from
        .. math::
          dW \sim \mathcal{N}(0, dt^2)
        """
        # TODO: Allow noise to have different shape from X
        # TODO: Preevaluate arguments ? Something like
        #      __init__: `sampler = rndstream(*args)`
        #      here:     `sampler()`
        return self.rndstream.normal(size=self.x.shape, avg=0, std=np.sqrt(self.x.dt))

    def drift(self, t):
        tidx = x.get_t_idx(t)
        return -x[tidx-1]

    def diffusion(self, t, x):
        tidx = x.get_t_idx(t)
        return 1

register_model(DriftDiffusion)
