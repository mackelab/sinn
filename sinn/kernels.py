
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 2 2017

Author: Alexandre René
"""
import numpy as np
from collections import OrderedDict

import theano_shim as shim
import sinn.common as com
import sinn.config as config
import sinn.mixins as mixins
floatX = config.floatX
lib = shim.lib
ConvolveMixin = mixins.ConvolveMixin
ParameterMixin = com.ParameterMixin


#TODO: Discretized kernels as a mixin class

class Kernel(ConvolveMixin, ParameterMixin):
    """Generic Kernel class. All kernels should derive from this.
    Kernels associated to histories with shape (M,) should have shape (M,M),
    with indexing following kernel[to idx][from idx].

    IMPORTANT OPTIMIZATION NOTE:
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
    M: DEPRECATED
        Throw away the row dimension and treat the result as a 1D array.
    Refer to the `shape_output` method to see exactly how the output is reshaped.
    """

    def __init__(self, name, shape, f=None, memory_time=None, t0=0, **kwargs):
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
        # if 'convolve_shape' in kwargs:
        #     assert( kwargs['convolve_shape'] == shape )
        # else:
        #     kwargs['convolve_shape'] = shape

        if 'params' not in kwargs:
            # ParameterMixin requires 'params'
            kwargs['params'] = com.define_parameters({})
        super().__init__(**kwargs)

        self.name = name
        self.shape = shape
        self.ndim = len(shape)
        self.t0 = t0

        if f is not None:
            self._eval_f = f
            #try:
            #    shim.check(f(0).ndim >= 2)
            #except AssertionError:
            #    raise ValueError("The result of kernel functions should be "
            #                     "2-dimensional (1 or both of which may be flat.")

        if hasattr(self, 'eval'):
            # Sanity test on the eval method
            try:
                self.shape_output(self.eval(0))
            except ValueError:
                raise ValueError("The parameters to the kernel's evaluation "
                                 "function seem to have incompatible shapes.")
        # TODO: add set_eval method, and make memory_time optional here
        assert(memory_time is not None)
        self.memory_time = memory_time

    def eval(self, t, from_idx=slice(None,None)):
        if not shim.isscalar(t):
            t = shim.add_axes(t, self.params[0].ndim-1, 'right')
                # FIXME: This way of fixing t dimensions is not robust
        return self._eval_f(t, from_idx)

    def _convolve_op_single_t(self, hist, t, kernel_slice):
        return hist._convolve_op_single_t(self, t, kernel_slice)

    def shape_output(self, output):
        """It may be that the output is only the diagonal of the kernel
        (i.e. the kernel depends only on the 'from' population, not the
        'to' population). Check, and if so, reshape as needed by
        duplicating the columns.

        If the output shape matches the kernel's, return it as is.
        If the output shape is "half" the kernel's (e.g. (2,) and (2,2)),
          treat it as a column and repeat it horizontally with tile.
          This will allocate memory.
        Otherwise, try to reshape it with the kernel's shape.
          This will fail if the number of elements don't match.
        """
        shim.check(self.shape == output.shape
                   or self.shape == output.shape*2
                   or np.prod(self.shape) == np.prod(output.shape))
        shim.ifelse(output.shape == self.shape,
                    output,
                    shim.ifelse(self.shape == output.shape*2,
                                lib.tile(shim.add_axes(output, output.ndim).T,
                                         (1,)*output.ndim + output.shape),
                                output.reshape(self.shape) ) )


class ExpKernel(Kernel):
    """An exponential kernel, of the form κ(s) = c exp(-(s-t0)/τ).
    """

    Parameter_info = OrderedDict( ( ( 'height'     , (config.cast_floatX, None, True) ),
                                    ( 'decay_const', (config.cast_floatX, None, True) ),
                                    ( 't_offset'   , (config.cast_floatX, None, True) ) ) )
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
        params = kwargs['params']

        # Truncating after memory_time should not discard more than a fraction
        # config.truncation_ratio of the total area under the kernel.
        # (Divide ∫_t^∞ by ∫_0^∞ to get this formula.)
        if memory_time is None:
            assert(0 < config.truncation_ratio < 1)
            # We want a numerical value, so we use the test value associated to the variables
            decay_const_val = np.max(shim.get_test_value(params.decay_const))
            memory_time = t0 -decay_const_val * np.log(config.truncation_ratio)

        self.memory_blind_time = np.max(params.t_offset) - t0
        # self.memory_blind_time = ifelse( (params.t_offset == params.t_offset.flatten()[0]).all(),
        #                                  (shim.asarray(params.t_offset) - t0).flatten()[0],
        #                                  params.t_offset - t0 )
            # When we convolve, the time window of length self.memory_blind_time before t0
            # is ignored (because the kernel is zero there), and therefore not included
            # in self.last_conv. So we need to extend the kernel slice by this much when
            # we reuse the cache data
            # The ifelse is to treat the case where t_offset is a matrix (i.e. multiple populations).
            # In this case, we want the blind time to be a scalar if they are all the same (more efficient),
            # but we leave the result as a matrix if they differ. In the latter case, convolutions
            # need to be calculated for each population separately and recombined.

        ########
        # Initialize base class
        super().__init__(name, shape, memory_time=memory_time, t0=t0, **kwargs)

        self.last_t = None     # Keep track of the last convolution time
        self.last_conv = None  # Keep track of the last convolution result
        self.last_hist = None  # Keep track of the history object used for the last convolution

    def eval(self, t, from_idx=slice(None,None)):
        if not shim.isscalar(t):
            t = shim.add_axes(t, self.params.t_offset.ndim-1, 'right')
            # FIXME: This way of fixing t dimension isn't robust
        return shim.switch(shim.lt(t, self.params.t_offset[:,from_idx]),
                           0,
                           self.params.height[:,from_idx]
                             * lib.exp(-(t-self.params.t_offset[:,from_idx])
                                       / self.params.decay_const[:,from_idx]) )
            # We can use indexing because ParameterMixin ensures parameters are at least 2D

    def _convolve_op_single_t(self, hist, t, kernel_slice):

        #TODO: store multiple caches, one per history
        #TODO: do something with kernel_slice

        # We are careful here to avoid converting t to time if not required,
        # so that kernel slicing can work on indices

        if (kernel_slice != slice(None, None)
            or shim.asarray(t).dtype != shim.asarray(self.last_t).dtype):
                # The second condition catches the case where e.g. last_t is
                # an index but t is a time (then t > last_t is a bad test).
            return hist.convolve(self, t, kernel_slice)
                # Exit before updating last_t and last_conv
        elif self.last_conv is not None and self.last_hist is hist:
            if t > self.last_t:
                Δt = t - self.last_t
                result = ( lib.exp(-hist.time_interval(Δt)/self.params.decay_const)
                           * self.last_conv
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

    #TODO: _convolve_op_batch ?

# TODO? : Indicator kernel ? Optimizations possible ?
