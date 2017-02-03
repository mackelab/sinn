# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 2017

Author: Alexandre René
"""

import numpy as np
from collections import deque

import sinn.config as config
import sinn.theano_shim as shim
floatX = config.floatX
lib = shim.lib

class History:
    """
    Generic class for storing a history, serving as a basis for specific history classes.
    On its own it lacks some basic fonctionality, like the ability to retrieve
    data at a certain time point.

    Ensures that if the history at time t is known, than the history at all times
    previous to t is also known (by forcing a computation if necessary).

    Derived classes can safely expect the following attributes to be defined:
        + shape                 : Shape at a single time point, or number of elements in the system
        + t0                    : Time at which history starts
        + tn                    : Time at which history ends
        + dt                    : Timestep size
        + _tarr                 : Ordered array of all time bins
        + _cur_tidx             : Tracker for the latest time bin for which we know history.
        + _update_function : Function taking a time and returning the history
                                  at that time

    Functions deriving from this class **must** implement the following methods:

    def __init__(self, *args):
        '''Initialize a Series instance, derived from History.

        Parameters
        ----------
        *args:
            Arguments required by History.
        '''

    def retrieve(self, key):
        '''A function taking either an index or a splice and returning respectively
        the time point or an interval from the precalculated history.
        It does not check whether history has been calculated sufficiently far.
        '''
        if shim.istype(key, 'int'):
            […]
        elif isintance(key, slice):
            […]
        else:
            raise ValueError("Key must be either an integer or a splice object.")

    def update(self, tidx, value):
        '''Store the a new time slice.

        Parameters
        ----------
        tidx: int
            The time index at which to store the value.
            Should not correspond to more than one bin ahead of _cur_tidx.
        value: timeslice
            The timeslice to store.
        '''

    def pad(self, before, after=0):
        '''Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.

        Parameters
        ----------
        before: float
            Amount of time to add to before t0. If non-zero, All indices
            to this data will be invalidated.
        after: float (default 0)
            Amount of time to add after tn.
        '''
        self.pad_time(before, after)
        […]

    Derived classes **may** also implement the following methods:

    def convolve(self, f, t, begin=None, end=None):
        '''Return the quasi-continuous time convolution with the spike train, i.e.
            ∫ history(t - s) * f(s) ds
        with s ranging from -∞ to ∞  (normally there should be no spikes after t).

        [Class specific comment]

        Parameters
        ----------
        f: function
            Function taking time as input
        t: float
            Time at which to evaluate the convolution
        begin: float
            If specified, the integral begins at this value (rather than -∞).
            This value is inclusive, i.e. f(begin) will be evaluated.
        end: float
            If specified, the integral ends at this value (rather than ∞).
            This value is not inclusive, i.e. f(end) will NOT be evaluated.

        The treatement of integral end points is to ensure that the interval over which
        the integral is evaluated has length 'end' - 'begin'. It also insures that
        integrals over successive intervals can be summed without duplicating the endpoints.
        Returns
        -------
        ndarray of shape `self.shape`
        '''
    """

    def __init__(self, t0, tn, dt, shape, f=None, iterative=True):
        """
        Initialize History object.

        Parameters
        ----------
        t0: float
            Time at which the history starts
        tn: float
            Time at which the history ends
        dt: float
            Timestep
        shape:
            Shape of a history slice at a single point in time.
            E.g. a movie history might store NxN frames in an TxNxN array.
            (N,N) would be the shape, and T would be (tn-t0)/dt.
        f:  function (t) -> shape
            (Optional) Function, which takes a time (float) as argument and computes the
            associated time slice. Self-references to history are permitted,
            for times strictly smaller than t. This function should correspond
            to the mean of the true f over the interval [t, t+dt).
            If this parameter is not specified when constructing the class, then
            the `set_update_function` method must be called before the history
            is required to update itself.
        iterative: bool (default: True)
            (Optional) If true, indicates that f must be computed iteratively. I.e. having
            computed f(t) is required in order to compute f(t+1). When false,
            when computed f for multiple times t, these will be passed as an array
            to f, using only one function call. Default is to force iterative computation.

        Returns
        -------
        None
        """
        self.shape = shape
            # shape at a single time point, or number of elements
            # in the system
        self.t0 = np.array(t0, dtype=floatX)
        self.tn = np.array(np.ceil( (tn - t0)/dt ) * dt + t0, dtype=floatX)
        self.dt = np.array(dt, dtype=floatX)
        self._cur_tidx = -1
            # Tracker for the latest time bin for which we
            # know history.
        if f is None:
            # Set a default function that will raise an error when called
            def f(*arg):
                raise RuntimeError("The update function for {} is not set.".format(self))
        self._update_function = f
        self._compute_range = None
        self._iterative = iterative

        self._tarr = np.arange(self.t0,
                               self.tn + self.dt - config.abs_tolerance,
                               self.dt)
        # 'self.tn+self.dt' ensures the upper bound is inclusive,
        # config.precision avoids numerical rounding errors

        self.t0idx = 0       # the index associated to t0
        self._unpadded_length = len(self._tarr)  # Save this, because _tarr might change with padding
        self._theano_updates = {}
            # Stores a Theano update dictionary. This value can only be
            # changed once, unless a call to self.theano_refresh is made
        self._conv_cache = {}
        self._old_conv_cache = {}
            # Store convolutions so they don't have to be recalculated.
            # The dictionary is keyed by the id of the kernels with which
            # convolutions were computed.

    def __len__(self):
        return self._unpadded_length

    def __getitem__(self, key):
        """
        Ensure that history has been computed far enough to retrieve
        the desired timeslice, and then call the class' `retrieve` method.

        NOTE: key will not be shifted to reflect history padding. So `key = 0`
        may well refer to a time *before* t0.
        """

        if shim.istype(key, ('int', 'float')):
            tidx = self.get_t_idx(key)
                # If `key` is a negative integer, returns as is
                # If `key` is a negative float, returns a positive index integer
            end = shim.ifelse(tidx >= 0,
                             tidx,
                             len(self._tarr) + key)
                # key = -1 returns last element

        elif isinstance(key, slice):
            # Get the latest point queried for in `key`
            if key.start is None:
                start = 0
            else:
                start = self.get_t_idx(key.start)
                start = shim.ifelse(start >= 0,
                                   start,
                                   len(self._tarr) + start)

            if key.stop is None:
                stop = len(self._tarr)
            else:
                stop = self.get_t_idx(key.stop)
                stop = shim.ifelse(stop >= 0,
                                   stop,
                                   len(self._tarr) + start)

            end = shim.max_of_2(start, stop - 1)
                # `stop` is the first point beyond the array

        else:
            raise ValueError("Trying to index using {} ({}). 'key' should be an "
                             "integer, a float, or a slice of integers and floats"
                             .format(key, type(key)))

        if end > self._cur_tidx:
            self.compute_up_to(end)

        return self.retrieve(key)

    def set_update_function(self, func):
        """
        Parameters
        ----------
        func: callable
            The update function. Its signature should be
            `func(t)`
        """
        self._update_function = func

    def set_range_update_function(self, func):
        """
        Parameters
        ----------
        func: callable
            The update function. Its signature should be `func(slice)`
        """
        self._compute_range = func

    def pad_time(self, before, after=0):
        """Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.

        FUTURE WARNING: In the future, will check if the history is
        already padded and only ensure that there is at least as much
        padding time as `before` or `after`.

        Parameters
        ----------
        before: float
            Amount of time to add the time array before t0. It may be
            reduced by an amount at most dt, in order to make it a multiple
            of dt.
        after: float
            Optional - default value is 0. Amount of time to add the time
            array after tn. It may be reduced by an amount at most dt, in
            order to make it a multiple of dt.

        Returns
        -------
        (int, int):
            A integer tuple `(n, m)` where `n` is the number of bins added
            before, and `m` the number of bins added after
        """
        # TODO: Only add as much padding as needed

        before_idx_len = int(before // self.dt)
        after_idx_len = int(after // self.dt)
        before = before_idx_len * self.dt
        after = after_idx_len * self.dt

        before_array = np.arange(self.t0 - before, self.t0, self.dt)
        after_array = np.arange(self.tn + self.dt - config.abs_tolerance,
                                self.tn + self.dt - config.abs_tolerance + after,
                                self.dt)
        self._tarr = np.hstack((before_array, self.get_time_array(), after_array))
        self.t0idx = len(before_array)
        self._cur_tidx += len(before_array)

        return len(before_array), len(after_array)

    def compute_up_to(self, tidx):
        """Compute the history up to `tidx` inclusive."""
        shim.check(shim.istype(tidx, 'int'))
        start = self._cur_tidx + 1
        stop = shim.ifelse(tidx >= 0,
                           tidx + 1,
                           len(self._tarr) + tidx + 1)

        if self._compute_range is not None:
            # A specialized function for computing multiple time points
            # has been defined – use it.
            self._compute_range(self._tarr[slice(start, stop)])

        elif not self._iterative:
            # Computation doesn't depend on history – just compute the whole thing in
            # one go
            self.update(slice(start,stop),
                        self._update_function(self._tarr[time_slice]))
        else:
            for i in lib.arange(start, stop):
                self.update(i, self._update_function(self._tarr[i]))

    def get_time_array(self, include_padding=False):
        """Return the time array.
        By default, the padding portions before and after are not included.
        The flag `include_padding` changes this behaviour:
            - True or 'all'     : include padding at both ends
            - 'begin' or 'start': include the padding before t0
            - 'end'             : include the padding after tn
            - False (default)   : do not include padding
        """

        if include_padding in ['begin', 'start']:
            return self._tarr[: self.t0idx+self._unpadded_length]
        elif include_padding in ['end']:
            return self._tarr[self.t0idx:]
        elif include_padding in [True, 'all']:
            return self._tarr
        else:
            return self._tarr[self.t0idx : self.t0idx+self._unpadded_length]

    def retrieve(self, key):
        raise NotImplementedError  # retrieve function is history type specific

    def update(self, tidx, value):
        raise NotImplementedError  # update function is history type specific

    def get_time(self, t):
        """
        If t is an index (i.e. int), return the time corresponding to t_idx.
        Else just return t
        """
        if shim.istype(t, 'int'):
            return self._tarr[0] + t*self.dt
        else:
            return t

    def get_t_idx(self, t):
         """Return the idx corresponding to time t. Fails if no such index exists.
         It is ok for the t to correspond to a time "in the future",
         and for the data array not yet to contain a point at that time.
         `t` may also be specified as a slice, in which case a slice of time
         indices is returned.
         """
         def _get_idx:
            if shim.istype(t, 'int'):
                # It's an easy error to make, specify a time as an int
                # Print a warning, just in case.
    #             print("Called get_t_idx on an integer ({}). Assuming this to be an INDEX".format(t)
    #                   + " (rather than a time) and returning unchanged.")
                return t
            else:
                if t * config.get_rel_tolerance(t) > self.dt:
                    raise ValueError("You've tried to convert a time (float) into an index "
                                    "(int), but the value is too large to ensure the absence "
                                    "of numerical errors. Try using a higher precision type.")
                t_idx = (t - self._tarr[0]) / self.dt
                if abs(t_idx - round(t_idx)) > config.get_abs_tolerance(t) / self.dt:
                    print("t: {}, t0: {}, t-t0: {}, t_idx: {}, dt: {}"
                        .format(t, self._tarr[0], t - self._tarr[0], t_idx, self.dt) )
                    print("(t0 above is the earliest time, including padding.)")
                    raise ValueError("Tried to obtain the time index of t=" + str(t) + ", but it does not seem to exist.")
                return int(round(t_idx))
        if isinstance(t, slice):
            return slice(_get_tidx(t.start), _get_tidx(t.stop), t.step)
        else:
            return _get_tix(t)

    def theano_reset(self):
        """Allow theano functions to be called again.
        It is assumed that variables in self._theano_updates have been safely
        updated externally.
        """
        for key in self._old_conv_cache:
            self._old_conv_cache[key] = None
        self._theano_updates = {}

class ConvolveMixin:

    def convolve(self, kernel, kernel_slice=slice(None,None), t=slice(None,None)):
        """
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
        # TODO: Move caching to parent history class, and define two methods
        #       in the derived classes: _convolve_op_single_t and _convolve_op_batch

        # TODO: allow 'kernel' to be a plain function

        # Test that kernel bound lists match and wrap them in list if necessary
        try:
            len(kernel_slice)
        except TypeError:
            kernel_slice = [kernel_slice]
        # try:
        #     len(kernel_stop)
        # except TypeError:
        #     kernel_stop = [kernel_stop]
        # if not len(kernel_start) == len(kernel_stop):
        #     raise ValueError("The lists of kernel start and stop bounds must "
        #                      "have the same length")

        # def get_start_idx(t):
        #     return 0 if t is None else dis_kernel.get_t_idx(t)
        # def get_stop_idx(t):
        #     return len(dis_kernel._tarr) if t is None else dis_kernel.get_t_idx(t)

        # kernel_idx_slices = [ slice( get_start_idx(start), get_stop_idx(stop) )
        #                       for start, stop in zip(kernel_start, kernel_stop) ]

        if np.isscalar(t):
            #tidx = self.get_t_idx(t)
            output_tidx = self.get_t_idx(t) - self.t0idx
                # TODO: This will break if t doesn't exactly correspond to a bin.
                #       Some convolutions don't care about bins (e.g. Spiketimes) –
                #       maybe we want to allow t to be anything, by adding a
                #       "no throw" flag to get_t_idx ?
                #       Then we would probably skip the cache search

            def _convolve_single_t(slc, t):
                try:
                    # Use a cached convolution if it exists
                    cache_idx = self._conv_cache[id(kernel)]['index list'][(slc.start, slc.stop)]
                    return self._conv_cache[id(kernel)]['data'][cache_idx][output_tidx]
                except KeyError:
                    #######################################
                    # CUSTOMIZATION: Here is the call to the custom convolution function
                    return self._convolve_op_single_t(kernel, slc, t)
                    #######################################

            retval = self.dt * lib.stack( [ _convolve_single_t(slc, t)
                                            for slc in kernel_slice ] )

        else:
            output_tidx = slice(self.get_t_idx(t.start) - self.t0idx,
                                self.get_t_idx(t.stop) - self.t0idx)
            # We have to adjust the index because the 'valid' mode removes
            # time bins at the ends.
            # E.g.: assume kernel.idx_shift = 0. Then (convolution result)[0] corresponds
            # to the convolution evaluated at tarr[kernel.stop]. So to get the result
            # at tarr[tidx], we need (convolution result)[tidx - kernel.stop].


            #TODO: Use a variable conv_cache = self._conv_cache[id(kernel)]. Maybe also one for 'data'
            cache_idcs = [None]*len(kernel_slice)
            # Store a list of references to the cached convolutions we need
            # Storing None indicates that the convolution with that kernel slice
            # needs to be calculated
            if id(kernel) in self._conv_cache:
                # Retrieve the indices of the already cached convolutions
                for i, slc in enumerate(kernel_slice):
                    try:
                        cache_idcs[i] = self._conv_cache[id(kernel)]['index list'][(slc.start, slc.stop)]
                    except KeyError:
                        pass
            else:
                # Create a new empty cache
                self._conv_cache[id(kernel)] = {
                    'data':       shim.shared(np.zeros((0, len(self)) + self.shape)),
                    'index list': {}
                    }
                self._old_conv_cache[id(kernel)] = None

            if None in cache_idcs:
                # There are new convolutions we need to compute
                new_data = self.dt * lib.stack( [
                      #lib.convolve(self[:], dis_kernel[slc], mode='valid')
                      ###########################################
                      # CUSTOMIZATION: Here is the call to the custom convolution function
                      self._convolve_op_batch(kernel, slc)
                      ###########################################
                      for slc, cache_idx in zip(kernel_slice, cache_idcs)
                      if cache_idx is None ] )
                # Add the indices they will have in the newly augmented cache
                k = self._conv_cache[id(kernel)]['data'].get_value().shape[0]
                for i, slc in enumerate(cache_idcs):
                    if slc is None:
                        cache_idcs[i] = k
                        self._conv_cache[id(kernel)]['index list'][(kernel_slice[i].start, kernel_slice[i].stop)] = k
                        k += 1
                # Create the new data cache
                if config.use_theano:
                    assert(self._old_conv_cache is None)
                    # Keep the old cache in memory, otherwise updates mechanism will break
                    self._old_conv_cache[id(kernel)]['data'] = self._conv_cache[id(kernel)]['data']
                    self._conv_cache[id(kernel)]['data'] = lib.concatenate(
                          (self._old_conv_cache[id(kernel)]['data'], new_data), axis = 0 )
                    # This is a shared variable, so add to the updates list
                    self._theano_updates[self._old_conv_cache[id(kernel)]['data']] = \
                          self._conv_cache[id(kernel)]['data']
                else:
                    # Since we aren't using Theano, we don't need to create _old_conv_cache
                    # and can allow reuse of _conv_cache memory
                    self._conv_cache[id(kernel)]['data'].set_value(
                        lib.concatenate(
                          (self._conv_cache[id(kernel)]['data'].get_value(), new_data), axis = 0 )
                        )

            retval = self._conv_cache[id(kernel)]['data'][cache_idcs][:,output_tidx]

        if len(retval) == 1:
            # Caller only passed a single kernel slice, and so is not
            # expecting the result to be wrapped in a list.
            return retval[0]
        else:
            return retval


    def _convolve_op_batch(self, kernel, kernel_slice):
        """Default implementation of batch convolution, which just evaluates
        the single t convolution at all time bins t.
        """
        return lib.stack( [self._convolve_op_single_t(kernel, kernel_slice, t)
                           for t in self._tarr[self.t0idx: self.t0idx + len(self)]] )


class Spiketimes(History):
    """A class to store spiketrains.
    These are stored as times associated to each spike, so their is
    no well-defined 'shape' of a timeslice. Instead, the `shape` parameter
    is used to indicate the number of neurons in each population.
    """

    def __init__(self, *args):
        """
        Same arguments as `History.__init__`.
        Here `shape` is an array giving the number of neurons in each population
        """
        super().__init__(*args)

    def initialize(self, init_data=-np.inf):
        """
        Parameters
        ----------
        init_data: float or iterable
            Either a scalar (to which each neuron will initialized) or an
            iterable of same length as the number of neurons.
        """

        try:
            init_data[0]
            subscriptable = True
        except:
            subscriptable = False

        # deque incurs a 10-15% cost in iterations compared with lists,
        # but makes adding new spikes O(1) rather than O(log(n)).
        # Testing has shown negligible difference for sims of 500 time bins,
        # and ~3% improvement for sims of 2000 time bins. The payoff is
        # expected to continually improve as simulations get longer.
        if subscriptable:
            if hasattr(init_data, 'shape'):
                shim.check(init_data.shape == self.shape)

            self.spike_times = [ [ deque([init_data[pop_idx][neuron_idx]])
                                   for pop_idx in range(len(self.shape)) ]
                                 for neuron_idx in self.shape[pop_idx] ]
        else:
            self.spike_times = [ [ deque([init_data])
                                   for pop_idx in range(len(self.shape)) ]
                                 for neuron_idx in self.shape[pop_idx] ]

    def retrieve(self, key):
        '''A function taking either an index or a splice and returning respectively
        the time point or an interval from the precalculated history.
        It does not check whether history has been calculated sufficiently far.

        Parameters
        ----------
        key: int, float, or slice

        Returns
        -------
        If `key` is int or float:
            Returns a list of binary vectors, each vector representing a population,
            each element representing a neuron. Values are 1 if the neuron fired in
            this bin, 0 if it didn't fire.
            If key is a float, it must match the bin time exactly. Generally using
            bin indices should be more reliable than bin times.
        If `key` is slice:
            Returns the list of spike times, truncated to the bounds of the slice.
            Slice bounds may be specified as indices (int) or times (float).
            [:] is much more efficient than [0:] if you want all spike times, as
            it just returns the internal list without processing.
        '''
        if shim.istype(key, 'int') or shim.istype(key, 'float'):
            t = self.get_time(key)
            return [ [ 1 if t in spikelist else 0
                       for spikelist in pop ]
                     for pop in self.spike_times ]
        elif isintance(key, slice):
            if (key.start is None) and (key.stop is None):
                return self.spike_times
            else:
                start = -np.inf if key.start is None else self.get_time(key.start)
                end = np.inf if key.end is None else self.get_time(np.inf)
                end =- self.dt  # exclude upper bound, consistent with slicing conv.
                # At present, deque's don't implement slicing. When they do, use that.
                return [ [ list(itertools.islice(spikelist,
                                                 np.searchsorted(spikelist, start),
                                                 np.searchsorted(spikelist, end)))
                           for spikelist in pop ]
                         for pop in self.spike_times ]
        else:
            raise ValueError("Key must be either an integer, float or a splice object.")

    #was :  def update(self, t, pop_idx, spike_arr):
    def update(self, tidx, value):
        '''Add to each neuron specified in `value` the spiketime `tidx`.
        Parameters
        ----------
        tidx: int, float
            The time index of the spike(s). This is converted
            to actual time and saved.
            Can optionally also be a float, in which case no conversion is made.
            Should not correspond to more than one bin ahead of _cur_tidx.
        value: list of iterables
            Should be as many iterables as there are populations.
            Each iterable is a list of neuron indices that fired in this bin.
        '''
        newidx = self.get_t_idx(tidx)
        shim.check(newidx <= self._cur_tidx + 1)

        time = self.get_time(tidx)
        for neuron_lst, spike_times in zip(value, self.spike_times):
            for neuron_idx in neuron_lst:
                spike_times[neuron_idx].append(time)

        self._cur_tidx = newidx  # Set the cur_idx. If tidx was less than the current index,
                                 # then the latter is *reduced*, since we no longer know
                                 # whether later history is valid.

    def pad(self, before, after=0):
        '''Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.
        """
        Parameters
        ----------
        before: float
            Amount of time to add to before t0. If non-zero, All indices
            to this data will be invalidated.
        after: float (default 0)
            Amount of time to add after tn.
        '''
        self.pad_time(before, after)
        # Since data is stored as spike times, no need to update the data


    def _convolve_op_single_t(self, kernel, kernel_slice, t)#begin=None, end=None):
        '''Return the quasi-continuous time convolution with the spike train, i.e.
            ∫ spiketimes(t - s) * kernel(s) ds
        with s ranging from -∞ to ∞  (normally there should be no spikes after t).

        Since spikes are delta functions, effectively what we are doing is
        sum( kernel(t-s) for s in spiketimes )

        Parameters
        ----------
        kernel: class instance or callable
            If a callable, should take time (float) as single input and return a float.
            If a class instance, should have a method `.eval` which satisfies the function requirement.
        DOCS DEPRECATED
        t: float
            Time at which to evaluate the convolution
        begin: float
            If specified, the integral begins at this value (rather than -∞).
            This value is inclusive, i.e. kernel(begin) will be evaluated.
        end: float
            If specified, the integral ends at this value (rather than ∞).
            This value is not inclusive, i.e. kernel(end) will NOT be evaluated.

        The treatement of integral end points is to ensure that the interval
        over which the integral is evaluated has length 'end' - 'begin'. It
        also insures that integrals over successive intervals can be summed
        without duplicating the endpoints.

        Returns
        -------
        ndarray of shape `self.shape`

        '''
        # TODO: To avoid iterating over the entire list, save the last `end`
        #       time and an array (one element per neuron) of the index of the latest
        #       spike before `end`. Iterations starting at `end` can then exclude all
        #       spikes before that point.
        #       Use `np.find` to get the `start` and `end` index, and sum between them

        # TODO: move to callable test ConvolveMixin
        # if callable(kernel):
        #    f = kernel
        # else:
        f = kernel.eval
        begin = kernel_slice.start
        end = kernel_slice.stop

        if begin is None:
            if end is not None:
                raise NotImplementedError

            return  EwiseIter( [
                            EwiseIter( [ np.fromiter( ( sum( f(to_pop_idx, from_pop_idx)(t-s) for s in neuron_spike_times )
                                                        for neuron_spike_times in self.spike_times[from_pop_idx] ),
                                                      dtype=config.floatX )
                                        for from_pop_idx in range(len(self.shape)) ] )
                            for to_pop_idx in range(len(self.shape))] )
        else:
            return EwiseIter( [
                           EwiseIter( [ np.fromiter( ( sum( f(to_pop_idx, from_pop_idx)(t-s) if begin <= s < end else 0 for s in neuron_spike_times )
                                                       for neuron_spike_times in self.spike_times[from_pop_idx] ),
                                                     dtype=config.floatX )
                                        for from_pop_idx in range(len(self.shape)) ] )
                           for to_pop_idx in range(len(self.shape)) ] )



class Series(ConvolveMixin, History):
    """
    Store history as a series, i.e. as an array of dimension T x (shape), where
    T is the number of bins and shape is this history's `shape` attribute.

    Also provides an "infinity bin" – .inf_bin — in which to store the value
    at t = -∞.
    """

    def __init__(self, *args):
        """
        Initialize a Series instance, derived from History.

        Parameters
        ----------
        *args:
            Arguments required by History.
        """

        super().__init__(*args)

        # Migration note: _data was previously called self.array
        self._data = np.zeros(self._tarr.shape + self.shape, dtype=floatX)
        self.inf_bin = np.zeros(self.shape, dtype=floatX)
        self._original_data = None
            # Stores a Theano update dictionary. This value can only be
            # changed once.

    def retrieve(self, key):
        '''A function taking either an index or a splice and returning
        respectively the time point or an interval from the
        precalculated history. It does not check whether history has
        been calculated sufficiently far.

        '''
        return self._data[key]

    def update(self, tidx, value):
        '''Store a new time slice.
        Parameters
        ----------
        tidx: int or slice(int, int)
            The time index at which to store the value.
            If specified as a slice, the length of the range should match
            value.shape[0].
        value: timeslice
            The timeslice to store.
        '''
        assert(not config.use_theano
               or not isinstance(tidx, theano.gof.Variable))
            # time indices must not be variables
        if isinstance(value, tuple):
            # `value` is a theano.scan-style return tuple
            assert(len(value) == 2)
            updates = value[1]
            assert(isinstance(updates, dict))
            value = value[0]
        else:
            updates = None

        if shim.istype(tidx, 'int'):
            end = tidx
            shim.check(tidx <= self._cur_tidx + 1)
                # Ensure that we update at most one step in the future
        else:
            assert(isinstance(tidx, slice))
            shim.check(tidx.end > tidx.start)
            end = tidx.end
            shim.check(tidx.start <= self._cur_tidx + 1)
                # Ensure that we update at most one step in the future

        if config.use_theano and isinstance(value, theano.gof.Variable):
            if self._original_data is not None or self._theano_updates != {}:
                raise RuntimeError("You can only update data once within a "
                                   "Theano computational graph. If you need "
                                   "to update repeatedly, compile a single "
                                   "update as a function, and call that "
                                   "function repeatedly.")
            self._original_data = self._data
                # Persistently store the current _data
                # It's important not to reuse variables after they've
                # been used in set_subtensor, but they must remain
                # in memory.
            self._data = T.set_subtensor(self._original_data[tidx], value)
            if updates is not None:
                self._theano_updates.update(updates)

        else:
            if updates is not None:
                raise RuntimeError("For normal Python and Numpy functions, update variables in place rather than using an update dictionary.")
            self._data[tidx] = value

        self._cur_tidx = end
            # If we updated in the past, this will reduce _cur_tidx
            # – which is what we want

    def pad(self, before, after=0, **kwargs):
        '''Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.
        """
        Parameters
        ----------
        before: float
            Amount of time to add to before t0. If non-zero, All indices
            to this data will be invalidated.
        after: float (default 0)
            Amount of time to add after tn.
        **kwargs:
            Extra keyword arguments are forwarded to `numpy.pad`.
            They may be used to specify how to fill the added time slices.
            Default is to fill with zeros.
        '''

        before_len, after_len = self.pad_time(before, after)

        if not kwargs:
            # No keyword arguments specified – use defaults
            kwargs['mode'] = 'constant'
            kwargs['constant_values'] = 0

        pad_width = ( [(before_len, after_len)]
                      + [(0, 0) for i in range(len(self.shape))] )

        self._data = np.pad(self._data, pad_width, **kwargs)

    # def set_inf_bin(self, value):
    #     """
    #     Set the ∞-bin with `value`.

    #     Parameters
    #     ----------
    #     value: float or ndarray
    #         If a float, every component of the ∞-bin will be set to this value.
    #         If an array, its shape should match that of the history.
    #     """
    #     if hasattr(value, 'shape'):
    #         shim.check(value.shape == self.shape)
    #         self.inf_bin = value
    #     else:
    #         self.inf_bin = np.ones(self.shape) * value

    def zero(self):
        """Zero out the series. The initial data point will NOT be zeroed"""
        self._data[1:] = np.zeros(self._data.shape[1:])

    def get_trace(self, component=None, include_padding='none'):
        """
        Return the series data for the given component.
        If `component` is 'None', return the full multi-dimensional trace

        Parameters
        ----------
        include_padding: 'none' (default) | 'before' | 'left' | 'after' | 'right' | 'all' | 'both'
            'none':
                Don't include the padding bins.
            'before' or 'left':
                Include the padding bins preceding t0.
            'after' or 'right:
                Include the padding bins following tn.
            'all' or 'both':
                Include the padding bins at both ends.
        """
        padding_vals = [ 'none', 'before', 'left', 'after', 'right', 'all', 'both' ]
        if include_padding in ['none', 'after', 'right']:
            start = self.t0idx
        elif include_padding in padding_vals:
            # It's one of the other options
            start = 0
        else:
            raise ValueError("include_padding should be one of {}.".format(padding_vals))

        if include_padding in ['none', 'before', 'left']:
            stop = self.t0idx + len(self)
        elif include_padding in padding_vals:
            stop = len(self._tarr)
        else:
            raise ValueError("include_padding should be one of {}.".format(padding_vals))

        if component is None:
            return self._data[start:stop]
        elif shim.istype(component, 'int'):
            return self._data[start:stop, component]
        elif len(component) == 1:
            return self._data[start:stop, component[0]]
        elif len(component) == 2:
            return self._data[start:stop, component[0], component[1]]
        else:
            raise NotImplementedError("Really, you used more than 2 data dimensions in a series array ? Ok, well let me know and I'll implement that.")

    def set(self, source=None):
        """Set the entire series in one go. `source` may be an array, a function,
        or even another History instance. It's useful for example if we've already
        computed history by some other means, or we specified it as a function
        (common for inputs).

        Accepted types for `source`: functions, arrays, single values
        These are all converted into a time-series with the same time bins as the history.

        If source has the attribute `shape`, than it is checked to be the same as this history's `shape`

        If no source is specified, the series own update function is used, provided
        it has been previously defined.
        can be used to force computation of the whole series.
        """

        data = None

        tarr = self._tarr

        if source is None:
            # Default is to use series' own compute functions
            self._compute_up_to(-1)

        elif shim.istype(source, 'float') or shim.istype(source, 'int'):
            # Constant input
            data = np.ones(tarr.shape + self.shape) * source

        else:
            if hasattr(source, 'shape'):
                # Input specified as an array
                if source.shape != tarr.shape + self.shape:
                    raise ValueError("The given external input series does not match the dimensions of the history")
                data = external_input

            else:
                try:
                    # Input should be specified as a function
                    # TODO: Use integration
                    data = np.concatenate(
                                        [np.asarray(external_input(t),
                                                    dtype=config.floatX)[np.newaxis,...] for t in tarr],
                                        axis=0)

                    # Check that the obtained shape for the input is correct
                    if data.shape != self.shape:
                        raise ValueError("The given external input series does not match the dimensions of the history")
                except ValueError:
                    raise
                except Exception as e:
                    raise Exception("\nExternal input should be specified as either a NumPy array or a function"
                                  ) from e  #.with_traceback(e.__traceback__)

        shim.check(data is not None)
        shim.check(data.shape == self._data.shape)
        shim.check(data.shape[0] == len(tarr))

        self._data = data
        self._cur_tidx = len(tarr) - 1
        return data

    def theano_reset(self, new_data):
        """Refresh data to allow a new call returning theano variables.
        `new_data` should not a complex type, like the result of a `scan`"""
        self._data = new_data
        self._original_data = None
        super.theano_reset()

    def convolve(self, kernel, kernel_slice=slice(None,None), t=slice(None,None),
                 *args, **kwargs):
        """Small wrapper around ConvolveMixin.convolve.
        Discretizes the kernel and converts the kernel_slice into a slice of
        time indices. Also converts t into a slice of indices, so the
        _convolve_op* methods can work with indices.
        """
        # Run the convolution on a discretized kernel
        # TODO: allow 'kernel' to be a plain function

        def get_start_idx(t):
            return 0 if t is None else kernel.get_t_idx(t)
        def get_stop_idx(t):
            return len(kernel._tarr) if t is None else kernel.get_t_idx(t)
        kernel_idx_slices = [ slice( get_start_idx(start), get_stop_idx(stop) )
                              for start, stop in zip(kernel_start, kernel_stop) ]
        tidx = self.get_t_idx(t)

        return super().convolve(self.discretize_kernel(kernel),
                                kernel_idx_slices, tidx, *args, **kwargs)

    def _convolve_op_single_t(self, discretized_kernel, kernel_slice, tidx):
        # When indexing data, make sure to use self[…] rather than self._data[…],
        # to trigger calculations if necessary

        if kernel_slice.start == kernel_slice.stop:
            return 0
        else:
            # Algorithm assumes an increasing kernel_slice
            shim.check(kernel_slice.stop > kernel_slice.start)

            hist_start_idx = tidx - kernel_slice.stop - discretized_kernel.idx_shift
            hist_slice = slice(hist_start_idx, hist_start_idx + kernel_slice.stop - kernel_slice.start)
            shim.check(hist_slice.start >= 0)
            return lib.sum(discretized_kernel[kernel_slice][::-1] * self[hist_slice])

    def _convolve_op_batch(self, discretized_kernel, kernel_slice):
        """Return the convolution at every lag with t0 and tn."""
        # When indexing data, make sure to use self[…] rather than self._data[…],
        # to trigger calculations if necessary

        if kernel_slice.start == kernel_slice.stop:
            return 0
        else:
            # Algorithm assumes an increasing kernel_slice
            shim.check(kernel_slice.stop > kernel_slice.start)

            # We compute the full 'valid' convolution, for all lags and then
            # return just the subarray corresponding to [t0:tn]
            # We have to adjust the index because the 'valid' mode removes
            # time bins at the ends.
            # E.g.: assume kernel.idx_shift = 0. Then (convolution result)[0] corresponds
            # to the convolution evaluated at tarr[kernel.stop + kernel_idx_shift]. So to get the result
            # at tarr[tidx], we need (convolution result)[tidx - kernel.stop - kernel_idx_shift].

            domain_start = self.t0idx - kernel_slice.stop - discretized_kernel.idx_shift
            domain_slice = slice(domain_start, domain_start + len(self))
            shim.check(domain_slice.start >= 0)
                # Check that there is enough padding before t0
            retval = lib.convolve(self[:], discretized_kernel[kernel_slice],
                                  mode='valid')[domain_slice]
            shim.check(len(retval) == len(self))
                # Check that there is enough padding after tn
            return retval

    def discretize_kernel(self, kernel):

        discretization_name = "discrete" + "_" + str(id(self))  # Unique id for discretized kernel

        if hasattr(kernel, discretization_name):
            # TODO: Check that this history (self) hasn't changed
            return getattr(kernel, discretization_name)

        else:
            if config.integration_precision == 1:
                kernel_func = kernel.eval
            elif config.integration_precision == 2:
                # TODO: Avoid recalculating eval at the same places by writing
                #       a _compute_up_to function and passing that to the series
                kernel_func = lambda t: (kernel.eval(t) + kernel.eval(t+self.dt)) / 2
            else:
                # TODO: higher order integration with trapeze or simpson's rule
                raise NotImplementedError

            # The kernel may start at a position other than zero, resulting in a shift
            # of the index corresponding to 't' in the convolution
            idx_shift = int(round(kernel.t0 / self.dt))
                # We don't use shim.round because time indices must be Python numbers
            t0 = idx_shift * self.dt  # Ensure the discretized kernel's t0 is a multiple of dt

            memory_idx_len = int(kernel.memory_time // self.dt) - 1
                # It is essential here to use the same forumla as pad_time
                # We substract one because one bin more or less makes no difference,
                # and doing so ensures that padding with `memory_time` always
                # is sufficient (no dumb numerical precision errors adds a bin)
            full_idx_len = memory_idx_len + idx_shift
                # `memory_time` is the amount of time before t0

            dis_kernel = Series(t0, t0 + full_idx_len*self.dt,
                                self.dt, kernel.shape, kernel_func)
            dis_kernel.idx_shift = idx_shift

            setattr(kernel, discretization_name, dis_kernel)

            return dis_kernel
