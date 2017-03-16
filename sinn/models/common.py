# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
#from scipy.integrate import quad
#from collections import namedtuple
import logging
logger = logging.getLogger("sinn.models.common")

import theano_shim as shim
import sinn.config as config
import sinn.common as com
import sinn.histories
import sinn.kernels
import sinn.diskcache as diskcache

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
            self.add_history(history)
            if hasattr(self, 'eval'):
                history.set_update_function(self.eval)

        super().__init__(params=params)

    # Simple consistency check functions
    @staticmethod
    def same_shape(*args):
        assert(all(arg1.shape == arg2.shape for arg1, arg2 in zip(args[:-1], args[1:])))
    @staticmethod
    def same_dt(*args):
        assert(all(arg1.dt == arg2.dt for arg1, arg2 in zip(args[:-1], args[1:])))
    @staticmethod
    def output_rng(outputs, rngs):
        """
        Parameters
        ----------
        outputs: History
            Can also be a list of Histories
        rngs: random stream, or list of random streams
            The random stream(s) required to generate the histories in
            `outputs`
        """
        if isinstance(outputs, sinn.histories.History):
            outputs = [outputs]
        else:
            assert(all(isinstance(output, sinn.histories.History) for output in outputs))
        try:
            len(rngs)
        except TypeError:
            rngs = [rngs]
        if ( any( outhist._cur_tidx < len(outhist) - 1 for outhist in outputs )
             and any( rng is None for rng in rngs) ) :
            raise ValueError("Cannot generate {} without the required random number generator(s).".format(str([outhist.name for outhist in outputs])))
        elif ( all( outhist._cur_tidx < len(outhist) - 1 for outhist in outputs )
             and all( rng is None for rng in rngs) ) :
            logger.warning("Your random number generator(s) will be unused, "
                           "since your data is already generated.")

    def cache(self, obj):
        """
        Call this function on all Kernel and History objects that should be
        saved to the disk cache.
        This function is cheap to call: the object is only written out when
        its removed from program memory.
        """

        if isinstance(obj, sinn.kernels.Kernel):
            logger.warning("Deprecated. Use add_kernel instead.")
            self.kernel_list.append(obj)
        else:
            assert(isinstance(obj, sinn.histories.History))
            logger.warning("Histories aren't written to disk. Use add_history instead")
            self.history_list.append(obj)

    def add_history(self, hist):
        assert(isinstance(hist, sinn.histories.History))
        if hist not in self.history_list:
            self.history_list.append(hist)
    def add_kernel(self, kernel):
        assert(isinstance(kernel, sinn.kernels.Kernel))
        if kernel not in self.kernel_list:
            self.kernel_list.append(kernel)

    def update_params(self, new_params):
        """
        The `lock` attribute of histories is used to determine whether
        they need to be recomputed.
        """
        logger.info("Updating model with new parameters {}".format(new_params))
        assert(type(self.params) == type(new_params))
        self.params = new_params
        logger.info("Model params are now {}. Updating kernels...".format(self.params))
        kernels_to_update = []
        for kernel in self.kernel_list:
            if not sinn.params_are_equal(
                    kernel.get_parameter_subset(new_params), kernel.params):
                # Grab the subset of the new parameters relevant to this kernel,
                # and compare to the kernel's current parameters. If any of
                # them differ, add the kernel to the list of kernels to update.
                kernels_to_update.append(kernel)

        # Loop over the list of kernels whose parameters have changed to do
        # two things:
        # - Remove any cached binary op that involves this kernel.
        #   (And write it to disk for later retrievel if these parameters
        #    are reused.)
        # - Update the kernel itself to the new parameters.
        for obj in self.history_list + self.kernel_list:
            if obj not in kernels_to_update:
                for op in obj.cached_ops:
                    for kernel in kernels_to_update:
                        if hash(kernel) in op.cache:
                            diskcache.save(op.cache[hash(kernel)])
                            # TODO subclass op[other] and define __hash__
                            logger.info("Removing cache for binary op {} ({},{}) from heap."
                                        .format(str(op), obj.name, kernel.name))
                            del op.cache[hash(kernel)]

                        diskcache.save(kernel)
                        logger.info("Updating kernel {}.".format(kernel.name))
                        kernel.update_params(new_params)

        for hist in self.history_list:
            if not hist.lock:
                self.clear_history(hist)

    def clear_history(self, history):
        # Clear the history, and remove any cached operations related to it
        # In contrast to `update_params`, we don't write these operations to
        # disk, because histories are data structures: there's no way of knowing
        # if they're equivalent to some already computed case other than comparing
        # the entire data.
        logger.info("Clearing history " + history.name)
        history.clear()
        if history in self.history_list:
            for obj in self.history_list + self.kernel_list:
                for op in obj.cached_ops:
                    if hash(history) in op.cache:
                        del op.cache[hash(history)]
        else:
            for obj in self.history_list + self.kernel_list:
                for op in obj.cached_ops:
                    if hash(history) in op.cache:
                        logger.error("Uncached history {} is member of cached "
                                     "op {}. This may indicate a memory leak."
                                     .format(history.name, str(op)))


class ModelKernelMixin:
    """
    Kernels within models should include this mixin.
    Adds interoperability with model parameters
    """
    def __init__(self, name, params, shape=None, f=None, memory_time=None, t0=0, **kwargs):
        super().__init__(name,
                         params = self.get_kernel_params(params),
                         shape  = shape,
                         f      = f,
                         memory_time=memory_time,
                         t0     = t0,
                         **kwargs)

    def update_params(self, params):
        super().update_params(self.get_parameter_subset(params))

    def get_parameter_subset(self, params):
        """Given a set of model parameters, return the set which applies
        to this kernel. These will in general not be a strict subset of
        `model_params`, but derived from them using `get_kernel_params`.
        As a special case, if each of the kernel's parameters can
        be found in `params`, then it is assumed that they have already
        been converted, and `get_kernel_params` is not called again.
        """
        if all( field in params._fields for field in self.params._fields ):
            # params already converted for kernel
            return sinn.get_parameter_subset(self, params)
        else:
            # These are model parameters. Convert them for the kernel
            return self.cast_parameters(self.get_kernel_params(params))

    @staticmethod
    def get_kernel_params(model_params):
        raise NotImplementedError("Each of your model's kernels must "
                                  "implement the method `get_kernel_params`.")
