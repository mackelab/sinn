# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Copyright 2017-2020 Alexandre René
"""
import numpy as np
from typing import Any
# from pydantic import PrivateAttr  # Not used, but we would like to

import theano_shim as shim
import mackelab_toolbox.utils as utils
import sinn
import sinn.common as com
import sinn.config as config
from sinn.mixins import CachedOperation
from sinn.utils.pydantic import add_exclude_mask

def deduce_convolve_result_shape(hist, hist_or_kernel, t):
    """
    Deduce the shape of the result of a convolution from its arguments.
    The first argument (`hist`) serves as reference for the time axis.

    Parameters
    ----------
    hist: History
        Shape: S
    hist_or_kernel: History | Kernel
        Shape: S
        May also be 2D if `hist` is 1D.
    t: int | array (len T)
        Time point index, as would be passed to `convolve`.
        The resulting time index shape is based on `hist`.
    Returns
    -------
    tuple:
        S, if `t` is scalar.
        (T,) + S if `t` is an array.
    """

    if not isinstance(hist, com.HistoryBase):
        raise TypeError("`hist` must be a History.")
    f = hist
    g = hist_or_kernel
    if f.shape == g.shape:
        shape = f.shape
    elif f.ndim == 1 and g.ndim == 2:
        shape = f.shape
    # elif f.ndim == 2 and g.ndim == 1:
    #     shape = g.shape
    else:
        raise ValueError("Convolution arguments must either be the same shape, "
                         "or the hist 1D and the other 2D.")
    if not shim.isscalar(t):
        if isinstance(t, np.ndarray):
            T = len(t)
        else:
            slc, fltr = hist._resolve_slice(t)
            if fltr is None:
                T = slc.start - slc.stop
            else:
                T = len(range(slc.start, slc.stop)[fltr])
        shape = (T,) + shape
    return shape

class ConvolveMixin(CachedOperation):
    # __slots__ = ('_conv_cache',)
        # Setting __slots__ causes TypeError: multiple bases have instance lay-out conflict
        # We use the same pattern as `CachedOperation`
    # _conv_cache : com.OpCache=PrivateAttr(None)  # <-- What we would like to
        # Under the hood, PrivateAttr uses __slots__, so it runs into the same
        # problem as above. It seems that __slots__ and mixins don't mix so well.
    _conv_cache : Any  # Workaround: just declare _conv_cache for purposes of source code documentation
                       # sunder prevents `_conv_cache` being added to the BaseModel's fields
                       # – this is also why we need to use `object.__setattr__` below
            # Store convolutions so they don't have to be recalculated.
            # The dictionary is keyed by the id of the kernels with which
            # convolutions were computed.

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs:
            Passed to base initializer
        """

        super().__init__(*args, **kwargs)

        object.__setattr__(self, '_conv_cache', com.OpCache(self, self.convolve_batch_wrapper))
        # object.__setattr__(self, '_conv_cache',
        #                    com.OpCache(self, self.convolve_batch_wrapper))
        self.cached_ops.append(self._conv_cache)
            # Add this cache to the objects's list of cached ops.
            # This allows it to be pruned later, if one of the op's members changes

    def copy(self, *a, exclude=None, **kw):
        exclude = add_exclude_mask(exclude, {'_conv_cache'})
        m = super().copy(*a, exclude=exclude, **kw)
        object.__setattr__(m, '_conv_cache', com.OpCache(m, m.convolve_batch_wrapper))
        # object.__setattr__(m, '_conv_cache',
        #                    com.OpCache(m, m.convolve_batch_wrapper))
        return m
    @classmethod
    def parse_obj(cls, *a, **kw):
        m = super().parse_obj(*a, **kw)
        object.__setattr__(m, '_conv_cache', com.OpCache(m, m.convolve_batch_wrapper))
        # object.__setattr__(m, '_conv_cache',
        #                    com.OpCache(m, m.convolve_batch_wrapper))
        return m
    def dict(self, *a, exclude=None, **kw):
        exclude = add_exclude_mask(exclude, {'_conv_cache'})
        return super().dict(*a, exclude=exclude, **kw)
    # def json(self, *a, **kw):
    #     excl = kw.pop('exclude', None);
    #     excl = set() if excl is None else set(excl)
    #     excl.add('_conv_cache')
    #     return super().json(*a, exclude=excl, **kw)

    def theano_reset(self):
        self._conv_cache.theano_reset()
        try:
            super().theano_reset()
        except AttributeError as e:
            if "'theano_reset'" in str(e):
                pass
            else:
                raise e

    def convolve_single_t_wrapper(other, t, output_tidx, slc):
        if ((slc.stop is not None and slc.start == slc.stop)
            or len(self) == 0 or len(other) == 0):
            shape = deduce_convolve_result_shape(self, other, t)
            return shim.zeros(shape, dtype=self.dtype)
        try:
            # Use a cached convolution if it exists
            return self._conv_cache.get(other,slc)[output_tidx]
        except KeyError:
            #######################################
            # CUSTOMIZATION: Here is the call to the custom convolution function
            return self._convolve_single_t(other, t, slc)
            #######################################

    def convolve_batch_wrapper(other, slc):
        if ((slc.stop is not None and slc.start == slc.stop)
            or len(self) == 0 or len(other) == 0):
            shape = deduce_convolve_result_shape(self, other)
            return shim.zeros((1,)+shape)
        else:
            #######################################
            # CUSTOMIZATION: Here is the call to the custom convolution function
            return self._convolve_op_batch(other, slc)
            #######################################

    def convolve(self, other, t=slice(None,None), kernel_slice=slice(None,None)):
        """
        Compute the convolution between `self` and `other`, where at least one of the two derives from History. The other is referred to as the "kernel" – it may derive from Kernel, but is not required to.
        `self` is preferentially treated as the history.
        The 'kernel' is truncated to the bounds defined by kernel_slice.
        If `t` is a time index, it refers to the time array of whichever
        between `self` and `other` is considered the 'history'.

        If `t` is a scalar, the convolution at that time is computed.
        If `t` is a slice, the *entire* convolution, for all time lags, is computed,
        even if `t` is a slice of a single element. It's then cached and
        `t` is used to select the appropriate subarray.
        If `t` is unspecified, it is treated as [:], i.e. the convolution
        for all lags is computed and returned.

        Be aware that every call with different kernel bounds may
        trigger a full copy of the cache.

        `kernel_slice` may be specified as a iterable, in which case
        the convolution is computed for each slice. This can be used instead of
        multiple calls to avoid copying the cache.

        If you are going to compute the convolution at most lags, it can be
        worth using slices to trigger the caching mechanism and exploit
        possible optimizations for batch convolutions.

        Either `self._single_t_conv_op` or `self._batch_conv_op` is used to
        compute the convolution operation, irrespective of whether `self` was
        identified as a 'history' or 'kernel'. This allows kernels, which
        may have specially optimized convolution methods, to override the
        more generic history methods.

        Parameters
        ----------
        other: History | Kernel
            The second argument to the convolution operation (first element
            is `self`).
        t : time index (int) | slice
        kernel_slice : slice | iterable of slice

        Returns
        -------
        array-like with shape (*S) or (T, *S)
        list of array-like with shape (*S) or (T, *S)
            If one of the data structures underlying `self` and `other` is
            symbolic, the result is a normal numpy array.
            If `kernel_slice` is an iterable, returns a list of arrays,
            otherwise returns a single array
            If `t` is scalar, returned array(s) have shape S, where S is the
            shape of the identified history (either `self` or `other`). If `t`
            is a slice of length T, the result has one extra initial dimension
            for the time axis.
        """
        # TODO: Use namedtuple for _conv_cache (.data & .idcs) ?
        # TODO: allow 'kernel' to be a plain function
        # FIXME: Don't preemptively cache whole convolution; just the queried times.

        # Determine history and kernel
        if isinstance(self, com.HistoryBase):
            history = self
            kernel = other
        else:
            assert(isinstance(other, com.HistoryBase))
            history = other
            kernel = self

        # Test kernel bounds are specified as slices and wrap them in list if necessary
        try:
            len(kernel_slice)
        except TypeError:
            kernel_slice = [kernel_slice]
            output_scalar = True
        else:
            output_scalar = False
        if not isinstance(kernel_slice[0], slice):
            raise ValueError("Kernel bounds must be specified as a slice.")

        # Hack to get single length time arrays to still return array
        t_is_scalar = shim.isscalar(t)
        if shim.isarray(t) and t.ndim > 0:
            # Convert time array to a scalar if possible, otherwise a slice
            assert(t.ndim == 1)
            if len(t) == 1:
                t = t[0]
            else:
                t = history.time_array_to_slice(t)

        if shim.isscalar(t):
            #tidx = self.get_t_idx(t)
            output_tidx = history.get_t_idx(t) - history.t0idx
                # TODO: This will break if t doesn't exactly correspond to a bin.
                #       Some convolutions don't care about bins (e.g. Spiketimes) –
                #       maybe we want to allow t to be anything, by adding a
                #       "no throw" flag to get_t_idx ?
                #       Then we would probably skip the cache search

            com._convolution_recursion_level += 1
            result = shim.stack( [ convolve_single_t_wrapper(other, t, output_tidx, slc)
                                   for slc in kernel_slice ] )
            com._convolution_recursion_level -= 1
            if not t_is_scalar:
                # t was specified as an array or slice; add time dimension to return value
                result = result[np.newaxis, ...]

        else:
            assert(isinstance(t, slice))

            output_tidx, output_filter = history._resolve_slice(t, exclude_padding=True)
                # Convolution result is necessarily shorter than full history
                # length, and we added exactly enough padding for the result to
                # have the same length as the history.
                # FIXME: What happens if there is MORE padding ? Is is convolution
                # still truncated to the valid history ?


            if ( not history._iterative
                # or (history.use_theano and history._is_batch_computable()) ):
                  or history._is_batch_computable() ):
                # The history can be computed at any time point independently of the others
                # Force computing it entirely, to allow the use of batch operations
                # FIXME conditioning on `use_theano` is meant to make sure that we are
                # computing a Theano graph, but I'm not positive that it's a 100% safe test
                # NOTE Why did I want to restrict to Theano graphs anyway ?
                history._compute_up_to(history.t0idx + len(history) - 1)
            if ( isinstance(kernel, com.HistoryBase) and
                 (not kernel._iterative
                  or kernel._is_batch_computable()) ):
                # The 'kernel' may also be a history
                kernel._compute_up_to(kernel.t0idx + len(kernel) - 1)

            com._convolution_recursion_level += 1
            result = self._conv_cache.ensureget(other, kernel_slice)[:,output_tidx]
                # The extra : is because ensureget returns its result wrapped in an
                # extra dimension (one per kernel slice)
            com._convolution_recursion_level -= 1

            if output_filter is not None:
                result = result[:,output_filter]
            #if step != 1:
            #    result = result[:, ::step]
                # FIXED?: Negative step sizes might be offset, if they are more than one ?
                #result = result[:, ::-1]
                  # extra ':' because ensureget returns the result wrapped in an extra dimension

        # Finalize output shape
        # These steps must only be done on the final result, hence the
        # `_inner_convolution` guard in case we are inside a recursive call
        if com._convolution_recursion_level == 0:
            # Unwrap the TensorWrapper

            # A kernel may involve a dot product, over which we are expected to
            # sum; left-over axes which need to be summed over are labeled
            # "contraction" (other axes are labeled "covariant" or
            # "contravariant", following their use in tensor calculus).
            result = result.array.sum(axis=result.dims.contraction)

            # Determine whether to keep the outer kernel slice dimension
            if output_scalar:
                # Caller only passed a single kernel slice, and so is not
                # expecting the result to be wrapped in a list.
                result = result[0]

        # Return result
        return result


    def _convolve_op_batch(self, other, kernel_slice):
        """Default implementation of batch convolution, which just evaluates
        the single t convolution at all time bins t.
        """
        # Determine history and kernel
        # (We can avoid the overhead of doing this by
        # defining _convolve_op_batch in the class for which
        # `self` is an instance.
        if isinstance(self, com.HistoryBase):
            history = self
            kernel = other
        else:
            assert(isinstance(other, com.HistoryBase))
            history = other
            kernel = self

        return shim.stack( [self._convolve_single_t(other, t, kernel_slice)
                            for t in range(history.t0idx, history.tnidx+1)] )
