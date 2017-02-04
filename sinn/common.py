# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Author: Alexandre René
"""
import numpy as np

import sinn.config as config
import sinn.theano_shim as shim
floatX = config.floatX
lib = shim.lib


class HistoryBase:

    def __init__(self, t0, tn):
        self.t0 = config.cast_floatX(t0)
        self.tn = config.cast_floatX(tn)
        self._tarr = None # Implement in child classes

    def get_time(self, t):
        raise NotImplementedError

    def get_t_idx(self, t):
        raise NotImplementedError

class OpCache:

    def __init__(self, source, op):
        """
        Parameters
        ----------
        source: object
            One of two members in an operation, and the one
            to which we are attaching this cache.
        op: callable
            Binary operation we want to cache. Return value
            should be a 1d array of same length as source
        """
        self.cache = {}
        self.old_cache = {}
        self.source = source
        self.op = op

    def theano_reset(self):
        for key in self.old_cache:
            self.old_cache[key] = None

    def sanitize(self, arg):
        """Return a hashable value base on `arg`.
        No-op if `arg` is hashable.
        """
        if isinstance(arg, slice):
            return (arg.start, arg.stop)
        else:
            return arg

    def get(self, other, arg):
        """Will raise KeyError if operation not cached."""
        cache_idx = self.cache[id(other)]['index list'][self.sanitize(arg)]
        return self.cache[id(other)]['data'][cache_idx]

    def ensureget(self, other, args):
        """Will compute and cache the operation if it isn't already."""
        # Create a set of keys for the cache dictionary
        arg_keys = [self.sanitize(arg) for arg in args]

        # Store a list of references to the cached operations we need
        # Storing None indicates that the operation with that argument
        # needs to be calculated
        data_keys = [None]*len(arg_keys)
        if id(other) in self.cache:
            # Retrieve the indices of the already cached convolutions
            for i, key in enumerate(arg_keys):
                try:
                    data_keys[i] = self.cache[id(other)]['index list'][key]
                except KeyError:
                    pass
        else:
            # Create a new empty cache

            # Which of the two members of the convolution – self or other –
            # is a History determines the shape of the result
            if isinstance(self.source, HistoryBase):
                hist = self.source
            else:
                assert( isinstance(other, HistoryBase) )
                hist = other
            entry_shape = (len(hist),) + hist.shape
            self.cache[id(other)] = {
                'data':       shim.shared(np.zeros((0,) + entry_shape)),
                'index list': {}
            }
            self.old_cache[id(other)] = None


        if None in data_keys:
            # There are new convolutions we need to compute
            new_data = lib.stack( [
                #lib.convolve(self[:], dis_kernel[slc], mode='valid')
                ###########################################
                # CUSTOMIZATION: Here is the call to the custom convolution function
                #                      self._convolve_op_batch(kernel, slc)
                self.op(other, arg)
                ###########################################
                for arg, cache_idx in zip(args, data_keys)
                if cache_idx is None ] )
            # Add the indices they will have in the newly augmented cache
            k = self.cache[id(other)]['data'].get_value().shape[0]
            for i, (dkey, akey) in enumerate(zip(data_keys, arg_keys)):
                if dkey is None:
                    data_keys[i] = k
                    self.cache[id(other)]['index list'][akey] = k
                    k += 1
            # Create the new data cache
            if config.use_theano:
                assert(self.old_cache is None)
                # Keep the old cache in memory, otherwise updates mechanism will break
                self.old_cache[id(other)] = self.cache[id(other)]['data']
                self.cache[id(other)]['data'] = lib.concatenate(
                            (self.old_cache[id(other)], data) )
                # This is a shared variable, so add to the updates list
                shim.theano_updates[self.old_cache[id(other)]['data']] = \
                            self.cache[id(other)]['data']
            else:
                # Since we aren't using Theano, we don't need to create old_cache
                # and can allow reuse of cache memory
                self.cache[id(other)]['data'].set_value(
                    lib.concatenate(
                        (self.cache[id(other)]['data'].get_value(), new_data), axis = 0 )
                )
        return self.cache[id(other)]['data'][data_keys]

