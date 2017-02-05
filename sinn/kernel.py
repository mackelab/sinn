
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 2 2017

Author: Alexandre René
"""
import numpy as np
from collections import OrderedDict

import sinn.common as com
import sinn.model as model
import sinn.config as config
import sinn.theano_shim as shim
import sinn.mixins as mixins
floatX = config.floatX
lib = shim.lib
ConvolveMixin = mixins.ConvolveMixin
ParameterMixin = com.ParameterMixin


#TODO: Discretized kernels as a mixin class

class Kernel(ConvolveMixin, ParameterMixin):
    """Generic Kernel class. All kernels should derive from this."""

    def __init__(self, name, shape, f=None, memory_time=None, t0=0, **kwargs):
        """
        Parameters
        ----------
        name: str
            A unique identifier. May be printed to identify this kernel in output.
        f: callable
            Function defining the kernel. Signature: f(t) -> float
        memory_time: float
            Time after which we can truncate the kernel.
        t0: float
            The time corresponding to f(0). Kernel is zero before this time.
        """
        super().__init__(**kwargs)

        self.name = name
        self.shape = shape
        self.t0 = t0

        if f is not None:
            self.eval = f
        if hasattr(self, 'eval'):
            if not self.shape == self.eval(0).shape:
                raise ValueError("The parameters to the kernel's evaluation "
                                 "function seem to have incompatible shapes.")
        # TODO: add set_eval method, and make memory_time optional here
        assert(memory_time is not None)
        self.memory_time = memory_time

    def _convolve_op_single_t(self, hist, t, kernel_slice):
        return hist._convolve_op_single_t(self, t, kernel_slice)


class ExpKernel(Kernel):
    """An exponential kernel, of the form κ(s) = c exp(-(s-t0)/τ).
    """

    Parameter_info = OrderedDict( ( ( 'height'     , config.cast_floatX ),
                                    ( 'decay_const', config.cast_floatX ) ) )
    Parameters = com.define_parameters(Parameter_info)

    def __init__(self, name, shape, memory_time=None, t0=0, **kwargs):
        """
        Parameters
        ----------
        name: str
            A unique identifier. May be printed to identify this kernel in output.
        params: ExpKernel.Parameters  (in **kwargs)
            - height: float, ndarray, Theano var
              Constant multiplying the exponential. c, in the expression above.
            - decay_const: float, ndarray, Theano var
              Characteristic time of the exponential. τ, in the expression above.
        memory_time: float
            (Optional) Time after which we can truncate the kernel. If left
            unspecified, calculated automatically.
            Must *not* be a Theano variable.
        t0: float or ndarray
            Time at which the kernel 'starts', i.e. κ(t0) = c,
            and κ(t) = 0 for t< t0.
            Must *not* be a Theano variable.
        """

#        self.height = height
#        self.decay_const = decay_const

        # Truncating after memory_time should not discard more than a fraction
        # config.truncation_ratio of the total area under the kernel.
        # (Divide ∫_t^∞ by ∫_0^∞ to get this formula.)
        params = kwargs['params']
        if memory_time is None:
            # We want a numerical value, so we use the test value associated to the variables
            decay_const_val = np.max(shim.get_test_value(params.decay_const))
            memory_time = -decay_const_val * np.log(config.truncation_ratio)

        ########
        # Initialize base class
        super().__init__(name, shape, memory_time=memory_time, t0=t0, **kwargs)

        self.last_t = None     # Keep track of the last convolution time
        self.last_conv = None  # Keep track of the last convolution result
        self.last_hist = None  # Keep track of the history object used for the last convolution

    def eval(self, s, from_idx=slice(None,None)):
        return ( self.params.height[from_idx,:]
                 * lib.exp(-(s-self.t0) / self.params.decay_const[from_idx,:]) )

    def _convolve_op_single_t(self, hist, t, kernel_slice):

        #TODO: store multiple caches, one per history
        #TODO: do something with kernel_slice
        if kernel_slice != slice(None, None):
            raise NotImplementedError
        if (self.last_conv is None
            or hist is not self.last_hist
            or t < self.last_t):
            result = hist.convolve(self, t)
        else:
            Δt = t - self.last_t
            result = ( lib.exp(-Δt/self.decay_const) * self.last_conv
                       + hist.convolve(self, t, slice(0, Δt)) )

        self.last_t = t
        self.last_conv = result
        self.last_hist = hist

        return result

    #TODO: _convolve_op_batch ?

# TODO? : Indicator kernel ? Optimizations possible ?
