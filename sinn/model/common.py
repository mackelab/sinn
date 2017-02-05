# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
#from scipy.integrate import quad
#from collections import namedtuple

import sinn.config as config
import sinn.theano_shim as shim
import sinn.common as com
floatX = config.floatX
lib = shim.lib


class Model(com.ParameterMixin):
    """Abstract model class.

    A model implementations should derive from this class.
    It must minimally provide:
    - A `Parameter_info` dictionary of the form:
        (See sinn.common.Parameterize)
        ```
        Parameter_info = OrderedDict{ 'param_name': Parameter([cast function], [default value]),
                                      ... }
        ```
    - A class-level (outside any method) call
        `Parameters = com.define_parameters(Parameter_info)`

    If an `eval` method also provided, the default initializer can also attach it to
    a history object. It should have the signature
    `def eval(self, t)`
    where `t` is a time.

    Models are typically initialized with a reference to a history object,
    which is appropriate for storing the output of `eval`.

    Implementations may also provide class methods to aid inference:
    - likelihood: (params) -> float
    - likelihood_gradient: (params) -> vector
    If not provided, `likelihood_gradient` will be calculated by appyling theano's
    grad method to `likelihood`. (TODO)
    As class methods, these don't require an instance – they can be called on the class directly.
    """

    def __init__(self, params, history=None):
        """
        Parameters
        ----------
        params: self.Parameters instance

        history: History instance
            If provided, and if this model has an `eval` method, then that method is
            attached to `history` as its update function.
        """
        # Try to attach default updater
        if history is not None and hasattr(self, 'eval'):
            history.set_update_function(self.eval)

    #TODO: Provide default gradient (through theano.grad) if likelihood is provided

