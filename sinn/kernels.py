
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 2 2017

Author: Alexandre René
"""
import numpy as np
from collections import OrderedDict
import logging
import functools
logger = logging.getLogger("sinn.kernels")

import theano_shim as shim
import sinn.common as com
import sinn.config as config
import sinn.mixins as mixins
import sinn.diskcache as diskcache

ConvolveMixin = mixins.ConvolveMixin
ParameterMixin = com.ParameterMixin


#TODO: Discretized kernels as a mixin class

# Don't hide CachedKernel within the decorator's scope, otherwise it can't be pickled
def checkcache(init):
    @functools.wraps(init)
    def cachedinit(self, name, *args, **kwargs):
        try:
            retrieved_kernel = diskcache.load(hash(self._serialize(*args, **kwargs)))
        except KeyError:
            init(self, name, *args, **kwargs)
        else:
            self.__dict__update(retrieved_kernel.__dict__)
            self.name = name
    return cachedinit
# class CachedKernel(cls):
#     def __init__(self, name, params, shape, memory_time=None, t0=0, *args, **kwargs):
#         try:
#             retrieved_kernel = diskcache.load(hash(cls._serialize(params, shape, memory_time, t0)))
#         except KeyError:
#             super().__init__(name, params, shape, memory_time, t0, *args, **kwargs)
#         else:
#             self.__dict__.update(retrieved_kernel.__dict__)
#             self.name = name # Name is not necessarily the same
# def cachekernel(cls):
#     return CachedKernel(cls)

# No cache decorator for generic kernels: f might be changed after the initialization,
# and any kernel with f=None might hit the same cache.
class Kernel(ConvolveMixin, ParameterMixin, com.KernelBase):
    """
    WARNING: Currently the `shape` is not used as below, but should simply be
    the shape of the result after application of the kernel.

    Generic Kernel class. All kernels should derive from this.
    Kernels associated to histories with shape (M,) should have shape (M,M),
    with indexing following kernel[to idx][from idx]. (See below for caveats.)

    Derived kernels must implement `_eval_f`. They should NOT implement `eval`.
    They should probably also implement `__init__` and `_convolve_single_t`.

    Optimization note:
    If you only need the diagonal of the kernel (because it only depends
    on the 'from' population), then it doesn't need to be MxM. Typical
    options then for the 'shape':
    MxM:
        Usual shape. If the output is size M, it will be repeated (with tile)
        to produce an MxM matrix. Avoid this if you can use broadcasting instead.
    1xM or Mx1:
        Leave the number of dimensions unchanged, but flatten into one column / rom.
        This will typically work best with broadcasting.
        1xM : independent of 'to' pop.
        Mx1 : independent of 'from' pop.
    M:
        Throw away the row dimension and treat the result as a 1D array. Should be
        equivalent to defining a diagonal array.
    Refer to the `shape_output` method to see exactly how the output is reshaped.
    """

    def __init__(self, name, params=None, shape=None, memory_time=None, t0=0, **kwargs):
        """
        Parameters
        ----------
        name: str
            A unique identifier. May be printed to identify this kernel in output.
        f: callable
            Function defining the kernel. Signature: f(t)/usr/lib/python3.4/site-packages/nbconvert/exporters/pdf.py -> float
        memory_time: float
            Time after which we can truncate the kernel.
        t0: float
            The time corresponding to f(0). Kernel is zero before this time.
        """

        if shim.is_theano_variable(shape):
            raise ValueError("You are trying to set the shape of kernel {} "
                             "with a Theano variable. If it depends on a "
                             "parameter, make sure you use its `get_value() "
                             "method.".format(name))
        self.initialize(name, params=params, shape=shape,
                        memory_time=memory_time, t0=t0, **kwargs)

    def initialize(self, name, params=None, shape=None, f=None, memory_time=None, t0=0, **kwargs):
        if params is None:
            # ParameterMixin requires 'params'
            params = com.define_parameters({})

        super().__init__(params=params, **kwargs)

        self.name = name
        assert(shape is not None)
        self.shape = tuple(np.int16(s) for s in shape)
            # It can be useful (e.g. in conditionals) to have a known type
        self.ndim = len(shape)
        self.t0 = t0

        if f is not None:
            self._eval_f = f
            #try:
            #    shim.check(f(0).ndim >= 2)
            #except AssertionError:
            #    raise ValueError("The result of kernel functions should be "
            #                     "2-dimensional (1 or both of which may be flat.")

        assert(memory_time is not None)
        self.memory_time = memory_time

        self.evalndim = self.eval(0).ndim
            # even with a Theano function, this returns a Python scalar

        # Sanity test on the eval method's shape
        try:
            eval_at_0 = shim.get_test_value(self.eval(0), nofail=True)
                # get_test_value returns None if eval(0) is a Theano var with no test value
            if eval_at_0 is not None:
                self.shape_output(eval_at_0, ())
        except (AssertionError, ValueError):
            raise ValueError("The parameters to the kernel's evaluation "
                                "function seem to have incompatible shapes. "
                                "The kernel's output has shape {}, but "
                                "you've set it to be reshaped to {}."
                                .format(self.eval(0).shape, self.shape))
        # TODO: add set_eval method

    def get_parameter_subset(self, params):
        """Given a set of parameters, return the subset which applies
        to this kernel.
        """
        return sinn.get_parameter_subset(self, params)

    def eval(self, t, from_idx=slice(None,None)):
        """
        Returns 0 for t < t0 or t >= t0 + memory_time.
        The asymmetric bounds ensure that if e.g. memory_time = 4Δt,
        than exactly 4 time bins will have a non-zero value.
        """
        if not shim.isscalar(t):
            tshape = t.shape
            #t = shim.add_axes(t, self.params[0].ndim-1, 'right')
                # FIXME: This way of fixing t dimensions is not robust
            if shim.isscalar(from_idx):
                t = shim.add_axes(t, self.evalndim - 1, 'right')
            else:
                t = shim.add_axes(t, self.evalndim, 'right')
            final_shape, ndim = self.get_final_shape(tshape)
            res = self.shape_output(self._eval_f(t, from_idx), tshape)
            return shim.switch(shim.and_(shim.ge(t, self.t0), shim.lt(t, self.t0+self.memory_time)),
                               res,
                               shim.zeros_like(res))
                               #shim.zeros(final_shape, ndim=ndim))
        else:
            tshape = ()
            final_shape, ndim = self.get_final_shape(tshape)
            res = self.shape_output(self._eval_f(t, from_idx), tshape)
            return shim.ifelse(shim.and_(shim.ge(t, self.t0), shim.lt(t, self.t0+self.memory_time)),
                               res,
                               shim.zeros_like(res))
                               #shim.zeros(final_shape, ndim=ndim))

    def theano_reset(self):
        """Make state clean for building a new Theano graph.
        This clears any discretizations of this kernel, since those
        may depend on different parameters."""
        logger.info("Resetting kernel {} for Theano".format(self.name))
        #attr_to_del = []
        for attr in dir(self):
            if attr[:9] == "discrete_":
                logger.info("Clearing and resetting stale kernel " + attr)
                getattr(self, attr).clear()
                getattr(self, attr).theano_reset()
        #        attr_to_del.append(attr)
        #for attr in attr_to_del:
        #    logger.info("Removing stale kernel " + attr)
        #    delattr(self, attr)

    def is_theano(self):
        return any(shim.is_theano_object(p) for p in self.params)

    def _convolve_op_single_t(self, hist, t, kernel_slice):
        return hist._convolve_op_single_t(self, t, kernel_slice)

    def get_final_shape(self, tshape):
        if not shim.is_theano_object(tshape):
            assert(isinstance(tshape, tuple))
        # tshape is expected to be a tuple, but could be a Theano object
        # in that case it always has at least dimension 1 (but its shape may be 0)

        #tshapearr = shim.asarray(tshape, dtype='int8')
        if hasattr(tshape, 'ndim'):
            shapedims = tshape.ndim
        else:
            shapedims = len(tshape)
        if shapedims == 0:
            # tshape is in fact a scalar - no time dimension
            final_shape = self.shape
            ndim = len(self.shape)
        else:
            # add a time dimension
            assert(shapedims == 1)
            ndim = 1 + len(self.shape)
            final_shape = shim.concatenate( (tshape, self.shape) )
            # final_shape = shim.ifelse( shim.eq(tshapearr.shape[0], 0),
            #                            shim.asarray( (1,) + self.shape,
            #                                            dtype = self.shape.dtype,
            #                                            broadcastable=(False,)*(ndim) ),
            #                            shim.concatenate( (tshapearr, self.shape) ) )
            # ndim = shim.ifelse( shim.eq(tshapearr.shape[0], 0),
            #                     len(self.shape),
            #                     1 + len(self.shape) )
        return final_shape, ndim

    def shape_output(self, output, tshape):
        final_shape, ndim = self.get_final_shape(tshape)
        return shim.reshape(output, final_shape, ndim=ndim)

    # TODO Deprecate the following
    def old_shape_output(self, output, tshape):
        """It may be that the output is only the diagonal of the kernel
        (i.e. the kernel depends only on the 'from' population, not the
        'to' population). Check, and if so, reshape as needed by
        duplicating the columns.

        The time array over which the kernel is evaluated is also a required
        argument, since that affects the output shape.

        If the output shape matches the kernel's, return it as is.
        If the output shape is "half" the kernel's (e.g. (2,) and (2,2)),
          treat it as a column and repeat it horizontally with tile.
          This will allocate memory. (This option is still experimental.)
        Otherwise, try to reshape it with the kernel's shape.
          This will fail if the number of elements don't match.
        """
        # There will be an extra dimension in the output if t is an array
        assert(isinstance(tshape, tuple))
        if len(tshape) == 0:
            timeslice_shape = output.shape
            final_shape = self.shape
            output_ndim = output.ndim
            new_axis_pos = 0
        else:
            assert(len(tshape)==1)
            timeslice_shape = output.shape[1:]
            final_shape = tshape + self.shape
            output_ndim = output.ndim - 1
            new_axis_pos = 1
        shim.check(shim.eq(self.shape, timeslice_shape)
                   or shim.eq(self.shape, timeslice_shape*2)
                   or shim.eq(np.prod(self.shape), shim.prod(timeslice_shape)))

        # The second ifelse condition below uses some funky syntax, because
        # ifelse expects an integer (0/1).
        # FIXME Haven't really tested the shim.tile branch
        return shim.ifelse(shim.all(shim.eq(output.shape, self.shape)),
                           output.reshape(final_shape),
                           shim.ifelse(shim.and_(shim.bool(shim.eq(len(self.shape), 2*output_ndim)),
                                                 shim.all(shim.eq(self.shape,
                                                                  shim.concatenate((timeslice_shape, timeslice_shape))))),
                                       shim.tile(shim.add_axes(output, output_ndim, pos=new_axis_pos).T,
                                                 tshape + (1,)*output_ndim + output.shape,
                                                 ndim=1 + 2*output_ndim),
                                       output.reshape(final_shape),
                                       outshape=final_shape) )

    # ====================================
    # Caching interface

    @classmethod
    def _serialize(cls, params, shape, memory_time, t0, **kwargs):
        """Quick 'n dirty serializer"""
        # **kwargs are not treated, so they should be left to their
        # default value of None
        for key, val in kwargs.items():
            assert(val is None)
        return str(cls) + str(params) + str(shape) + str(memory_time) + str(t0)

    def __hash__(self):
        return hash(self._serialize(self.params, self.shape, self.memory_time, self.t0))

    # def __new__(cls, name, params=None, shape=None, f=None, memory_time=None, t0=0, **kwargs):
    #     if cls != Kernel:
    #         # Don't cache generic kernels, because f might be changed after the initialization,
    #         # and any kernel with f=None might hit the same cache.
    #         try:
    #             retrieved_kernel = diskcache.load(hash(cls._serialize(params, shape, memory_time, t0)))
    #         except KeyError:
    #             return super().__new__(cls, name, shape, f, memory_time, t0, **kwargs)
    #         else:
    #             retrieved_kernel.name = name # Name is not necessarily the same
    #             return retrieved_kernel

    def update_params(self, new_params):
        # Reset all attached discretized kernels, since those are no longer valid
        logger.info("Updating kernel " + self.name + ":")
        attr_to_del = []
        for attr in dir(self):
            if attr[:9] == "discrete_":
                if not getattr(self, attr).use_theano:
                    # Theano kernels are not precomputed, so they also
                    # don't need to be deleted
                    getattr(self, attr).clear()

        #               attr_to_del.append(attr)
        # for attr in attr_to_del:
        #     logger.info("Removing stale kernel " + attr)
        #     delattr(self, attr)

        # Unless a Kernel subclass does something special in `initialize`, the following
        # ultimately just calls set_parameters on the kernel. For Theano parameters, and
        # when updating as part of a larger model, this won't do anything: they all derive
        # from model parameters, so the changes immediately propagate down. Still we leave
        # this here a) in case something else is done in `initialize` and b) in case this
        # method is called on its own.
        logger.info("Reinitializing kernel {} with new parameters {}."
                    .format(self.name, str(new_params)))
        self.initialize(self.name, new_params, self.shape, self.memory_time, self.t0)


class ExpKernel(Kernel):
    """
    An exponential kernel, of the form κ(s) = c exp(-(s-t0)/τ).
    NOTE: The way things are coded now, t_offset is considered fixed. I.e.,
          one should not try to use this in a routine seeking to optimize t_offset.
    """

    Parameter_info = OrderedDict( ( ( 'height'     , (shim.config.floatX, None, True) ),
                                    ( 'decay_const', (shim.config.floatX, None, True) ),
                                    ( 't_offset'   , (shim.config.floatX, None, True) ) ) )
    Parameters = com.define_parameters(Parameter_info)

    @checkcache
    def initialize(self, name, params, shape, memory_time=None, t0=0, **kwargs):
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

        # Truncating after memory_time should not discard more than a fraction
        # config.truncation_ratio of the total area under the kernel.
        # (Divide ∫_t^∞ by ∫_0^∞ to get this formula.)
        if memory_time is None:
            assert(0 < config.truncation_ratio < 1)
            # We want a numerical value, so we use the test value associated to the variables
            decay_const_val = np.max(shim.get_test_value(params.decay_const))
            memory_time = t0 - decay_const_val * np.log(config.truncation_ratio)

        ########
        # Initialize base class
        super().initialize(name, params, shape, memory_time=memory_time, t0=t0, **kwargs)
            # WARNING: Don't use params below here, only self.params

        if shim.isshared(self.params.t_offset):
            # If t_offset is a shared variable, grab its value.
            t_offset = self.params.t_offset.get_value()
        elif shim.graph.is_computable([self.params.t_offset]):
            # We can evaluate the parameter (it's likely a symbolic manipulation of a
            # shared variable). This takes a few seconds, but returns a pure Python value
            t_offset = self.params.t_offset.eval()
        else:
            # There's nothing we can do: t_offset must remain symbolic
            t_offset = self.params.t_offset
        self.memory_blind_time = shim.max(t_offset) - t0
            # When we convolve, the time window of length self.memory_blind_time before t0
            # is ignored (because the kernel is zero there), and therefore not included
            # in self.last_conv. So we need to extend the kernel slice by this much when
            # we reuse the cache data

        self.last_t = None     # Keep track of the last convolution time
        self.last_conv = None  # Keep track of the last convolution result
        self.last_hist = None  # Keep track of the history object used for the last convolution

    def _eval_f(self, t, from_idx=slice(None,None)):
        return shim.switch(shim.lt(t, self.params.t_offset[:,from_idx]),
                           0,
                           self.params.height[:,from_idx]
                             * shim.exp(-(t-self.params.t_offset[:,from_idx])
                                       / self.params.decay_const[:,from_idx]) )
            # We can use indexing because ParameterMixin ensures parameters are at least 2D

    def _convolve_op_single_t(self, hist, t, kernel_slice):

        #TODO: store multiple caches, one per history
        #TODO: do something with kernel_slice
        #TODO: Allow Theano to make use of the exp kernel

        # We are careful here to avoid converting t to time if not required,
        # so that kernel slicing can work on indices

        if (kernel_slice != slice(None, None)
            or shim.is_theano_object(t)):
            # HACK Our caching does not deal with Theano times, so in that
            # case we bypass that as well.
            # FIXME Ideally we would allow Theano to use the optimized exp kernel as well,
            # when we need to do an iterative computation
            return hist.convolve(self, t, kernel_slice)
                # Exit before updating last_t and last_conv
        elif shim.asarray(t).dtype != shim.asarray(self.last_t).dtype:
            # This condition catches the case where e.g. last_t is
            # an index but t is a time (then t > last_t is a bad test).
            result = hist.convolve(self, t, kernel_slice)
            self.last_conv = result
        elif self.last_conv is not None and self.last_hist is hist:
            if t > self.last_t:
                Δt = t - self.last_t
                # Compute the amount left from the cache
                reduction_factor =  self.shape_output(shim.exp(-hist.time_interval(Δt)/self.params.decay_const), ())
                if hasattr(hist, 'pop_rmul'):
                    # FIXME The convolution needs to keep separate contributions from the different pops
                    reduced_cache = hist.pop_rmul(reduction_factor, self.last_conv)
                else:
                    reduced_cache = reduction_factor * self.last_conv

                # Add the convolution over the new time interval which is not cached
                result = ( reduced_cache
                           + hist.convolve(self, t,
                                           slice(hist.index_interval(self.memory_blind_time),
                                                 hist.index_interval(self.memory_blind_time + Δt))) )
                self.last_conv = result
                    # We only cache the convolution up to the point at which every
                    # population "remembers" it.
                result += hist.convolve(self, t,
                                        slice(0, hist.index_interval(self.memory_blind_time)))
                                              # 0 idx corresponds to self.t0

            elif t == self.last_t:
                result = self.last_conv
            else:
                result = hist.convolve(self, t)
                self.last_conv = result
        else:
            result = hist.convolve(self, t)
            self.last_conv = result

        self.last_t = t
        #self.last_conv = result
        self.last_hist = hist

        return result

    def _convolve_op_batch(self, hist, kernel_slice):
        # For batch convolutions, we punt to the history
        return hist.convolve(self, slice(None, None), kernel_slice)

# TODO? : Indicator kernel ? Optimizations possible ?
