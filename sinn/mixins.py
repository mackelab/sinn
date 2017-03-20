# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Author: Alexandre René
"""
import numpy as np

import theano_shim as shim
import sinn.common as com
import sinn.config as config

class CachedOperation:
    """All op mixins which contain an OpCache object should inherit
    from this class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'cached_ops'):
            self.cached_ops = []

    def clear(self):
        # All cached binary ops are now invalid, so delete them
        for op in self.cached_ops:
            op.clear()

        try:
            super().clear()
        except AttributeError:
            pass

# TODO: Only require convolve_shape for descendents of HistoryBase

# This class cannot be in the same module as HistoryBase,
# in order for tests with isinstance to work.
# (Not sure if this is really true)
class ConvolveMixin(CachedOperation):

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs:
            Passed to base initializer
        """

        super().__init__(*args, **kwargs)

        # try:
        #     convolve_shape = kwargs.pop('convolve_shape')
        # except KeyError:
        #     raise TypeError("Unsufficient arguments: ConvolveMixin "
        #                     "requires a `convolve_shape` argument.")
        self._conv_cache = com.OpCache(self, self._convolve_op_batch)
                                       #convolve_shape)
            # Store convolutions so they don't have to be recalculated.
            # The dictionary is keyed by the id of the kernels with which
            # convolutions were computed.
        self.cached_ops.append(self._conv_cache)
            # Add this cache to the objects's list of cached ops.
            # This allows it to be pruned later, if one of the op's members changes

    def theano_reset(self):
        self._conv_cache.theano_reset()
        try:
            super().theano_reset()
        except AttributeError:
            pass

    def convolve(self, other, t=slice(None,None), kernel_slice=slice(None,None)):
        """
        Compute the convolution between `self` and `other`, where at least one of the two derives from History. The other is referred to as the "kernel" – it may derive from Kernel, but is not required to.
        `self` is preferentially treated as the history.
        Compute the convolution with `kernel`, with `kernel` truncated to the bounds
        defined by kernel_slice.

        If `t` is a scalar, the convolution at that time is computed.
        If `t` is a slice, the *entire* convolution, for all time lags, is computed,
        even if `t` is a slice of a single element. It's then cached and
        `t` is used to select the appropriate subarray.
        If `t` is unspecified, it is treated as [:], i.e. the convolution
        for all lags is computed and returned.

        Be aware that every call with different kernel bounds may
        trigger a full copy of the cache.

        DEPRECATED DOCS
        `kernel_start`, `kernel_stop` may be specified as iterables, in which case
        the convolution is computed for each pair of kernel bounds (the two
        iterables should have the same bounds). This can be used instead of
        multiple calls to avoid copying the cache.

        If you are going to compute the convolution at most lags, it can be worth
        using slices to trigger the caching mechanism and exploit possible
        optimizations for batch convolutions.

        `single_t_conv_op` and `batch_conv_op` may be used to specify custom
        convolution operations, otherwise the history class's operations are
        used. This can be useful for example if the kernel has some special
        optimizations.
        """
        # TODO: Use namedtuple for _conv_cache (.data & .idcs) ?
        # TODO: allow 'kernel' to be a plain function

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

        if shim.isscalar(t):
            #tidx = self.get_t_idx(t)
            output_tidx = history.get_t_idx(t) - history.t0idx
                # TODO: This will break if t doesn't exactly correspond to a bin.
                #       Some convolutions don't care about bins (e.g. Spiketimes) –
                #       maybe we want to allow t to be anything, by adding a
                #       "no throw" flag to get_t_idx ?
                #       Then we would probably skip the cache search

            def convolve_single_t(t, slc):
                if slc.stop is not None and slc.start == slc.stop:
                    return 0
                try:
                    # Use a cached convolution if it exists
                    return self._conv_cache.get(other,slc)[output_tidx]
                except KeyError:
                    #######################################
                    # CUSTOMIZATION: Here is the call to the custom convolution function
                    return self._convolve_op_single_t(other, t, slc)
                    #######################################

            retval = shim.lib.stack( [ convolve_single_t(t, slc)
                                       for slc in kernel_slice ] )

        else:
            assert(isinstance(t, slice))
            start = history.t0idx if t.start is None else history.get_t_idx(t.start) - history.t0idx
            stop = history.t0idx + len(history) if t.stop is None else history.get_t_idx(t.stop) - history.t0idx
            output_tidx = slice(start, stop)
            # We have to adjust the index because the 'valid' mode removes
            # time bins at the ends.
            # E.g.: assume kernel.idx_shift = 0. Then (convolution result)[0] corresponds
            # to the convolution evaluated at tarr[kernel.stop]. So to get the result
            # at tarr[tidx], we need (convolution result)[tidx - kernel.stop].

            if ( not history._iterative
                 or history.use_theano and history._is_batch_computable() ):
                # The history can be computed at any time point independently of the others
                # Force computing it entirely, to allow the use of batch operations
                # FIXME conditioning on `use_theano` is meant to make sure that we are
                # computing a Theano graph, but I'm not positive that it's a 100% safe test
                history.compute_up_to(history.t0idx + len(history) - 1)

            retval = self._conv_cache.ensureget(other, kernel_slice)[:,output_tidx]

        if output_scalar:
            # Caller only passed a single kernel slice, and so is not
            # expecting the result to be wrapped in a list.
            return retval[0]
        else:
            return retval


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

        return shim.lib.stack( [self._convolve_op_single_t(other, t, kernel_slice)
                                for t in history._tarr[history.t0idx: history.t0idx + len(history)]] )


