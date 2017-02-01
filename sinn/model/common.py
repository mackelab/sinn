# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
from scipy.integrate import quad
from collections import namedtuple

import sinn.config as config
import sinn.theano_shim as shim
floatX = config.floatX
lib = shim.lib

def make_shared_tensor_params(params):
    TParameters = namedtuple('TParameters', params._fields)
    param_lst = []
    for val, name in zip(params, params._fields):
        # TODO: Check if val is already a theano tensor and adjust accordingly
        try:
            if val.dtype.kind in sp.typecodes['Float']:
                param_lst.append(theano.shared(sp.array(val, dtype=floatX)))
            else:
                param_lst.append(theano.shared(val))
        except ValueError:
            # Can't convert val to numpy array – it's probably a theano tensor
            # FIXME: if a scalar is not of type theano.config.floatX, this will actually
            #        create a ElemWise.cast{} code, wrt which we can't differentiate
            # FIXME: does it even make sense to produce a shared variable from another Theano variable ?
            if val.dtype.kind in sp.typecodes['Float']:
                param_lst.append(T.cast(theano.shared(val), dtype=floatX))
            else:
                param_lst.append(theano.shared(val))
        param_lst[-1].name = name

    return TParameters(*param_lst)

def make_cst_tensor_params(param_names, params):
    """
    Construct a Parameters set of Theano constants from a
    Parameters set of NumPy/Python objects.
    Code seems obsolete, or at least in dire need of updating.
    """
    TParameters = namedtuple('TParameters', param_names)
    global name_counter
    id_nums = range(name_counter, name_counter + len(param_names))
    name_counter += len(param_names)
    return TParameters(*(T.constant(getattr(params,name), str(id_num) + '_' + name, dtype=theano.config.floatX)
                         for name, id_num in zip(param_names, id_nums)))

def get_parameter_subset(model, src_params):
    """
    Create a Parameters object with the same instances as src_params
    Use case: we need a handle on a kernel's parameters, e.g. because
    the parameters are shared with another kernel or some higher level
    function.

    Parameters
    ----------
    model: class instance derived from Model
        The model class for which we want a Parameter collection.
    src_params: namedtuple
        The pre-existing Parameter collection we want to reuse.
    """
    # TODO: use src_params._asdict() ?
    paramdict = {}
    for name in src_params._fields:
        if name in model.Parameters._fields:
            paramdict[name] = getattr(src_params, name)
    return class_instance.Parameters(**paramdict)


class Model:
    """Abstract model class.

    A model implementations should derive from this class.
    It must minimally provide:
    - A `Parameters` instance of namedtuple, listing the parameters it requires.
    - A `eval` function, taking as single argument a time and returning
      the value of the model at that time.

    Models are typically initialized with a reference to a history object,
    which is appropriate for storing the output of `eval`.

    Implementations may also provide class methods to aid inference:
    - likelihood: (params) -> float
    - likelihood_gradient: (params) -> vector
    If not provided, `likelihood_gradient` will be calculated by appyling theano's
    grad method to `likelihood`. (TODO)
    As class methods, these don't require an instance – they can be called on the class directly.
    """

    Parameters = namedtuple('Parameter', [])   # Overload this in derived classes

    def parameters_are_valid(self, params):
        """Returns `true` if all of the model's parameters can be set from `params`"""
        return set(self.Parameters._fields).issubset(set(params._fields))

    def get_parameter_subset(self, params):
        """
        Return the subset of parameters from params that relate to this model.

        Returns
        -------
        A Parameter namedtuple
        """
        return get_parameter_subset(self, params)

    #TODO: Provide default gradient (through theano.grad) if likelihood is provided

#TODO: Discretized kernels as a mixin class

class Kernel:
    """Generic Kernel class. All kernels should derive from this."""

    def __init__(self, name, f, memory_time, t0=0):
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
        self.name = name
        self.t0 = t0

        self.eval = f
        self.shape = f(0).shape
        self.memory_time = memory_time

    def convolve(self, hist, t):
        return hist.convolve(self, t)


class ExpKernel(Kernel):
    """An exponential kernel, of the form κ(s) = c exp(-(s-t0)/τ).
    """

    def __init__(self, name, multiplier, decay_const, memory_time=None, t0=0):
        """
        Parameters
        ----------
        name: str
            A unique identifier. May be printed to identify this kernel in output.
        multiplier: float, ndarray, Theano var
            Constant multiplying the exponential. c, in the expression above.
        decay_const: float, ndarray, Theano var
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
        self.name = name
        self.multiplier = multiplier
        self.decay_const = decay_const
        self.t0 = t0

        def f(s):
            return self.multiplier * lib.exp(-(s-self.t0) / self.decay_const)
        self.eval = f

        try:
            self.shape = f(0).shape
        except ValueError:
            raise ValueError("The shapes of the parameters 'multiplier', 'decay_const' and 't0' don't seem to match.")

        # Truncating after memory_time should not discard more than a fraction
        # config.truncation_ratio of the total area under the kernel.
        # (Divide ∫_t^∞ by ∫_0^∞ to get this formula.)
        if memory_time is None:
            # We want a numerical value, so we use the test value associated to the variables
            decay_const_val = shim.get_test_value(decay_const)
            self.memory_time = -decay_const_val * np.log(config.truncation_ratio)
        else:
            self.memory_time = memory_time

        self.last_t = None     # Keep track of the last convolution time
        self.last_conv = None  # Keep track of the last convolution result
        self.last_hist = None  # Keep track of the history object used for the last convolution

    def convolve(self, hist, t):

        #TODO: allow t to be a slice
        #TODO: store multiple caches, one per history
        if (self.last_conv is None
            or hist is not self.last_hist
            or t < self.last_t):
            result = hist.convolve(self, t)
        else:
            Δt = t - self.last_t
            result = ( lib.exp(-Δt/self.decay_const) * self.last_conv
                       + hist.convolve(self, t, 0, Δt) )

        self.last_t = t
        self.last_conv = result
        self.last_hist = hist

        return result

# TODO? : Indicator kernel ? Optimizations possible ?
