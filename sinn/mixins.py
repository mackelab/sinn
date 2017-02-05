# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Author: Alexandre René
"""
import numpy as np

import sinn.common as com
import sinn.config as config
import sinn.theano_shim as shim
floatX = config.floatX
lib = shim.lib

# This class cannot be in the same module as HistoryBase,
# in order for tests with isinstance to work.
class ConvolveMixin:

    def __init__(self, *args, **kwargs):
        self._conv_cache = com.OpCache(self, self._convolve_op_batch)
            # Store convolutions so they don't have to be recalculated.
            # The dictionary is keyed by the id of the kernels with which
            # convolutions were computed.
        super().__init__(*args, **kwargs)

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

        # Test that kernel bound lists match and wrap them in list if necessary
        try:
            len(kernel_slice)
        except TypeError:
            kernel_slice = [kernel_slice]
        if not isinstance(kernel_slice[0], slice):
            raise ValueError("Kernel bounds must be specified as a slice.")

        if np.isscalar(t):
            #tidx = self.get_t_idx(t)
            output_tidx = history.get_t_idx(t) - history.t0idx
                # TODO: This will break if t doesn't exactly correspond to a bin.
                #       Some convolutions don't care about bins (e.g. Spiketimes) –
                #       maybe we want to allow t to be anything, by adding a
                #       "no throw" flag to get_t_idx ?
                #       Then we would probably skip the cache search

            def convolve_single_t(t, slc):
                try:
                    # Use a cached convolution if it exists
                    return self._conv_cache.get(kernel,slc)[output_tidx]
                except KeyError:
                    #######################################
                    # CUSTOMIZATION: Here is the call to the custom convolution function
                    return self._convolve_op_single_t(other, t, slc)
                    #######################################

            retval = lib.stack( [ convolve_single_t(t, slc)
                                  for slc in kernel_slice ] )

        else:
            start = history.t0idx if t.start is None else history.get_t_idx(t.start) - history.t0idx
            stop = history.t0idx + len(history) if t.stop is None else history.get_t_idx(t.stop) - history.t0idx
            output_tidx = slice(start, stop)
            # We have to adjust the index because the 'valid' mode removes
            # time bins at the ends.
            # E.g.: assume kernel.idx_shift = 0. Then (convolution result)[0] corresponds
            # to the convolution evaluated at tarr[kernel.stop]. So to get the result
            # at tarr[tidx], we need (convolution result)[tidx - kernel.stop].

            retval = self._conv_cache.ensureget(other, kernel_slice)[:,output_tidx]

        if len(retval) == 1:
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

        return lib.stack( [self._convolve_op_single_t(other, t, kernel_slice)
                           for t in history._tarr[history.t0idx: history.t0idx + len(history)]] )


