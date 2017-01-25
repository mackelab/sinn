# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 2017

Author: Alexandre René
"""

import numpy as np
from collections import deque

import sinn.config as config
floatX = config.floatX

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
        + _tarr      https://www.youtube.com/watch?v=WmVLcj-XKnM           : Ordered array of all time bins
        + _cur_tidx             : Tracker for the latest time bin for which we know history.
        + _ext_compute_function : Function taking a time and returning the history
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
        if isinstance(key, int):
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

    def __init__(self, t0, tn, dt, shape, f, iterative=True):
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
            Function, which takes a time (float) as argument and computes the
            associated time slice. Self-references to history are permitted,
            for times strictly smaller than t. This function should correspond
            to the mean of the true f over the interval [t, t+dt).
        iterative: bool (default: True)
            If true, indicates that f must be computed iteratively. I.e. having
            computed f(t) is required in order to compute f(t+1). When false,
            when computed f for multiple times t, these will be passed as an array
            to f, using only one function call. Default is to force iterative computation.

        Returns
        -------
        None
        """
        self.shape = shape   # shape at a single time point, or number of elements in the system
        self.t0 = np.array(t0, dtype=floatX)
        self.tn = np.array(np.ceil( (tn - t0)/dt ) * dt + t0, dtype=floatX)
        self.dt = np.array(dt, dtype=floatX)
        self._cur_tidx = -1    # Tracker for the latest time bin for which we
                               # know history.
        self._ext_compute_function = f
        self._iterative = iterative

        self._tarr = np.arange(self.t0,
                               self.tn + self.dt - config.abs_tolerance,
                               self.dt)
        # 'self.tn+self.dt' ensures the upper bound is inclusive,
        # config.precision avoids numerical rounding errors

        self.t0idx = 0       # the index associated to t0
        self._unpadded_length = len(self._tarr)  # Save this, because _tarr might change with padding

    def __len__(self):
        return self._unpadded_length

    def __getitem__(self, key):
        """
        Ensure that history has been computed far enough to retrieve
        the desired timeslice, and then call the class' `retrieve` method.

        NOTE: key will *not* be shifted to reflect history padding. So `key = 0`
        may well refer to a time *before* t0.
        """

        if isinstance(key, int):
            end = key
            if end < 0:
                end = len(self._tarr) + end       # a[-1] is the last element of a

        elif isinstance(key, slice):
            # Get the latest point queried for in `key`
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            elif start < 0:
                start = len(self._tarr) + start

            if stop is None:
                stop = len(self._tarr)
            elif stop < 0:
                stop = len(self._tarr) + stop     # a[3:-1] goes up to the *second last* element of a

            end = max(start, stop - 1)  # `stop` is the first point beyond the array

        else:
            raise ValueError("Trying to index using {} ({}). 'key' should be an integer or a slice"
                             .format(key, type(key)))

        if end > self._cur_tidx:
            self.compute_up_to(end)

        return self.retrieve(key)

    def pad_time(self, before, after=0):
        """Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.

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
        #TODO: Check if the module implementing _ext_compute_function, also
        #      has a function for compute_up_to  (might have special optimizations).
        #      Such optimized functions would work well for external inputs, but might
        #      be tricky to code if there are circular dependencies between histories.
        time_slice = slice(self._cur_tidx + 1, tidx + 1)
        if not self._iterative:
            # Computation doesn't depend on history – just compute the whole thing in
            # one go
            raise NotImplementedError # TODO: Allow self.update to take multiple times
            self.update((time_slice.start, time_slice.stop),
                        self._ext_compute_function(self._tarr[time_slice]))
        else:
            for i in range(*time_slice.indices(tidx + 1)):
                self.update(i, self._ext_compute_function(self._tarr[i]))

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
        if isinstance(t, int):
            return self._tarr[0] + t*self.dt
        else:
            return t

    def get_t_idx(self, t):
         """Return the idx corresponding to time t. Fails if no such index exists.
         It is ok for the t to correspond to a time "in the future",
         and for the data array not yet to contain a point at that time.
         """
         if isinstance(t, int):
             # It's an easy error to make, specify a time as an int
             # Print a warning, just in case.
#             print("Called get_t_idx on an integer ({}). Assuming this to be an INDEX".format(t)
#                   + " (rather than a time) and returning unchanged.")
             return t
         else:
             t_idx = (t - self._tarr[0]) / self.dt
             if abs(t_idx - round(t_idx)) > config.abs_tolerance / self.dt:
                 print("t: {}, t0: {}, t-t0: {}, t_idx: {}, dt: {}"
                     .format(t, self._tarr[0], t - self._tarr[0], t_idx, self.dt) )
                 print("(t0 above is the earliest time, including padding.)")
                 raise ValueError("Tried to obtain the time index of t=" + str(t) + ", but it does not seem to exist.")
             return int(round(t_idx))

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
                assert(init_data.shape == self.shape)

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
        if isinstance(key, int) or isinstance(key, float):
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
        assert(newidx <= self._cur_tidx + 1)

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


    def convolve(self, kernel, t, begin=None, end=None):
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
        t: float
            Time at which to evaluate the convolution
        begin: float
            If specified, the integral begins at this value (rather than -∞).
            This value is inclusive, i.e. f(begin) will be evaluated.
        end: float
            If specified, the integral ends at this value (rather than ∞).
            This value is not inclusive, i.e. f(end) will NOT be evaluated.

        The treatement of integral end points is to ensure that the interval
        over which the integral is evaluated has length 'end' - 'begin'. It
        also insures that integrals over successive intervals can be summed
        without duplicating the endpoints.

        Returns
        -------
        ndarray of shape `self.shape`

        '''
        if callable(kernel):
            f = kernel
        else:
            f = kernel.eval

        if begin is None:
            if end is not None:
                raise NotImplementedError

            return  EwiseIter( [
                            EwiseIter( [ np.fromiter( ( sum( f(to_pop_idx, from_pop_idx)(t-s) for s in neuron_spike_times )
                                                        for neuron_spike_times in self.spike_times[from_pop_idx] ),
                                                      dtype=theano.config.floatX )
                                        for from_pop_idx in range(len(self.shape)) ] )
                            for to_pop_idx in range(len(self.shape))] )
        else:
            return EwiseIter( [
                           EwiseIter( [ np.fromiter( ( sum( f(to_pop_idx, from_pop_idx)(t-s) if begin <= s < end else 0 for s in neuron_spike_times )
                                                       for neuron_spike_times in self.spike_times[from_pop_idx] ),
                                                     dtype=theano.config.floatX )
                                        for from_pop_idx in range(len(self.shape)) ] )
                           for to_pop_idx in range(len(self.shape)) ] )



class Series(History):
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


    def retrieve(self, key):
        '''A function taking either an index or a splice and returning respectively
        the time point or an interval from the precalculated history.
        It does not check whether history has been calculated sufficiently far.
        '''
        return self._data[key]

    def update(self, tidx, value):
        '''Store the a new time slice.
        Parameters
        ----------
        tidx: int
            The time index at which to store the value.
        value: timeslice
            The timeslice to store.
        '''
        # TODO: Allow specifying tidx as an array
        assert(isinstance(tidx, int))
        assert(tidx <= self._cur_tidx + 1)  # Ensure that we update at most one step in the future
        self._data[tidx] = value
        self._cur_tidx = tidx    # If we updated in the past, this will reduce _cur_tidx – which is what we want

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

    def set_inf_bin(self, value):
        """
        Set the ∞-bin with `value`.

        Parameters
        ----------
        value: float or ndarray
            If a float, every component of the ∞-bin will be set to this value.
            If an array, its shape should match that of the history.
        """
        if hasattr(value, 'shape'):
            assert(value.shape == self.shape)
            self.inf_bin = value
        else:
            self.inf_bin = np.ones(self.shape) * value

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
        elif isinstance(component, int):
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

        A variety of types for `source` are accepted: functions, arrays, single values
        These are all converted into a time-series with the same time bins as the history.

        If source has the attribute `shape`, than it is checked to be the same as this history's `shape`
        """

        data = None

        tarr = self._tarr

        if source is None:
            # Default to zero input
            data = np.zeros(tarr.shape + self.shape)

        elif isinstance(source, float) or isinstance(source, int):
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
                                                    dtype=theano.config.floatX)[np.newaxis,...] for t in tarr],
                                        axis=0)

                    # Check that the obtained shape for the input is correct
                    if data.shape != self.shape:
                        raise ValueError("The given external input series does not match the dimensions of the history")
                except ValueError:
                    raise
                except Exception as e:
                    raise Exception("\nExternal input should be specified as either a NumPy array or a function"
                                  ) from e  #.with_traceback(e.__traceback__)

        assert(data is not None)
        assert(data.shape == self._data.shape)
        assert(data.shape[0] == len(tarr))

        self._data = data
        self._cur_tidx = len(tarr) - 1
        return data

    def convolve(self, kernel, t, start=None, stop=None):

        # TODO: allow 'kernel' to be a plain function
        dis_kernel = self._discretize_kernel(kernel)

        if start is None:
            start_idx = 0
        else:
            start_idx = dis_kernel.get_t_idx(start)
        if stop is None:
            stop_idx = len(dis_kernel._tarr)
        else:
            stop_idx = dis_kernel.get_t_idx(stop)

        if stop_idx == start_idx:
            # Integrating over a zero-width kernel
            return 0
        assert(stop_idx > start_idx)
        conv_len = stop_idx - start_idx

        if np.isscalar(t):
            tidx = self.get_t_idx(t)
            adjusted_tidx = tidx - dis_kernel.idx_shift
        elif isinstance(t, slice):
            assert(t.step in [1, None])
            tidx = slice(self.get_t_idx(t.start), self.get_t_idx(t.stop))
            output_tidx = slice(self.get_t_idx(t.start) - conv_len - dis_kernel.idx_shift,
                                self.get_t_idx(t.stop) - conv_len - dis_kernel.idx_shift)
            # We have to adjust the index because the 'valid' mode removes
            # time bins at the ends.
            # E.g.: assume kernel.idx_shift = 0. Then (convolution result)[0] corresponds
            # to the convolution evaluated at tarr[conv_len]. So to get the result
            # at tarr[tidx], we need (convolution result)[tidx - conv_len].
        else:
            raise ValueError("'t' should be either a scalar or a slice object.")

        # When indexing data, make sure to use self[…] rather than self._data[…],
        # to trigger calculations if neccesary
        if np.isscalar(t):
            assert(adjusted_tidx >= conv_len)
            return self.dt * np.sum(dis_kernel[start_idx:stop_idx][::-1]
                                    * self[adjusted_tidx - conv_len:adjusted_tidx])

        else:
            # self[:] returns a numpy array
            return self.dt * np.convolve(self[:], dis_kernel[start_idx:stop_idx],
                                         mode='valid')[output_tidx]


    def _discretize_kernel(self, kernel):

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
                kernel_func = lambda t: (kernel.eval(t) + kernel.eval(t+self.dt))
            else:
                # TODO: higher order integration with trapeze or simpson's rule
                raise NotImplementedError

            # The kernel may start at a position other than zero, resulting in a shift
            # of the index corresponding to 't' in the convolution
            idx_shift = int(round(kernel.t0 / self.dt))
            t0 = idx_shift * self.dt  # Ensure the discretized kernel's t0 is a multiple of dt

            dis_kernel = Series(t0, t0 + kernel.memory_time,
                                self.dt, kernel.shape, kernel_func)
            dis_kernel.idx_shift = idx_shift

            setattr(kernel, discretization_name, dis_kernel)

            return dis_kernel
