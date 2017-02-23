# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
#from scipy.integrate import quad
#from collections import namedtuple

import theano_shim as shim
import sinn.config as config
import sinn.common as com
import sinn.histories
import sinn.kernels
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
        # History is optional because more complex models have multiple histories.
        # They should keep track of them themselves.
        # ParameterMixin requires params as a keyword parameter
        """
        Parameters
        ----------
        params: self.Parameters instance

        history: History instance
            If provided, a reference is kept to this history: Model evaluation may require
            querying some of its attributes, such as time step (dt). These may not be
            constant in time.
            If this model has an `eval` method, then that method is also
            attached to `history` as its update function.
        """
        self.kernel_list = []
        self.history_list = []

        if history is not None:
            self.history = history
            self.cache(history)
            if hasattr(self, 'eval'):
                history.set_update_function(self.eval)

        super().__init__(params=params)

    #TODO: Provide default gradient (through theano.grad) if likelihood is provided

    def cache(self, obj):
        """
        Call this function on all Kernel and History objects that should be
        saved to the disk cache.
        This function is cheap to call: the object is only written out when
        its removed from program memory.
        """

        if isinstance(obj, sinn.kernels.Kernel):
            self.kernel_list.append(obj)
        else:
            assert(isinstance(obj, sinn.histories.History))
            self.history_list.append(obj)

    def update_params(self, new_params):
        kernels_to_update = []
        for kernel in self.kernel_list:
            if sinn.get_parameter_subset(kernel, new_params) != kernel.params:
                kernels_to_update.append(kernel)

        for obj in self.history_list + self.kernel_list:
            if obj not in kernels_to_update:
                for op in obj.cached_ops:
                    for kernel in kernels_to_update:
                        if hash(kernel) in op.cache:
                            diskcache.save(op[hash(kernel)])
                            # TODO subclass op[other] and define __hash__
                            logger.info("Removing cache for binary op {} ({},{}) from heap."
                                        .format(str(op), obj.name, kernel.name))
                            del op[hash(kernel)]

                        diskcache.save(kernel)
                        logger.info("Updating kernel {}.".format(kernel.name))
                        kernel.update_params(new_params)

