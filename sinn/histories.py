# -*- coding: utf-8 -*-

"""
Created on Mon Jan 16 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import deque
import itertools
import operator
import logging
logger = logging.getLogger("sinn.history")

import theano_shim as shim
import sinn
import sinn.common as com
import sinn.config as config
import sinn.mixins as mixins
HistoryBase = com.HistoryBase
ConvolveMixin = mixins.ConvolveMixin


class History(HistoryBase):
    """
    Generic class for storing a history, serving as a basis for specific history classes.
    On its own it lacks some basic fonctionality, like the ability to retrieve
    data at a certain time point.

    Ensures that if the history at time t is known, than the history at all times
    previous to t is also known (by forcing a computation if neces#sary).

    Derived classes can safely expect the following attributes to be defined:
        + name             : str. Unique identifying string
        + shape            : int tuple. Shape at a single time point, or number of elements in the syst        # A depends on ρ, so its update function has to be set second
em
        + t0               : float. Time at which history starts
        + tn               : float. Time at which history ends
        + dt               : float. Timestep size
        + locked           : bool. Whether modifications to history are allowed. Modify through method
        + _tarr            : float ndarray. Ordered array of all time bins
        + _cur_tidx        : int. Tracker for the latest time bin for which we know history.
        + _original_tidx   : For Numpy histories, same as _cur_tidx. For Theano histories, a handle to
                             to the tidx variable, which is to be updated with the new value of _cur_tidx
        + _update_function : Function taking a time and returning the history
                             at that time
        + compiled_history : (Theano histories only) Another History instance of same
                             type, where the update function graph has been compiled.
                             Create with `compile` method.
    The following methods are also guaranteed to be defined:
        + compile          : If the update function is a Theano graph, compile it
                             and attach the new history as `compiled_history`.
        + lock             : Set the locked status
    A History may also define
        + _compute_range   : Function taking an array of consecutive times and returning
                             an array-like object of the histhttps://gist.github.com/Newmu/acb738767acb4788bac3ory at those times.
                             NOTE: It is important that the times be consecutive (no skipping)
                             and increasing, as some implementations assume this.

    Classes deriving from History **must** provide the following attributes:
        + _strict_index_rounding : (bool) True => times are only converted
                                   to indices if they are multiples of dt.
        + _data            : Where the actual data is stored. The type can be any numpy variable, but
                             must be wrapped as a shim.shared variable.
        + _original_data   : As _original_tidx, a handle to the original data variable

    and implement the following methods:

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
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

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

    All methods which modify the history (update, set, clear, compute_up_to) must raise a RuntimeError if `lock` is True.

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

    #_strict_index_rounding = None
        # Derived classes should define this as a class attribute
    instance_counter = 0

    def __init__(self, hist=sinn._NoValue, name=None, *args, t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 shape=sinn._NoValue, f=sinn._NoValue, iterative=sinn._NoValue, use_theano=False):
        # TODO: Change `use_theano` to `pure_theano` or something
        """
        Initialize History object.
        Instead of passing the parameters, another History instance may be passed as
        first argument. In this case, t0, tn, dt, shape, f and iterative are taken
        from that instance. These can be overridden by passing also a corresponding keyword
        argument.
        Except for a possible history instance, all parameters should be passed as keywords.

        Parameters
        ----------
        hist: History instance
            Optional. If passed, will used this history's parameters as defaults.
        t0: float
            Time at which the history starts
        tn: float
            Time at which the history ends
        dt: float
            Timestep
        (following are keyword-only arguments)
        shape: int tuple
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
        # The first thing we do is increment the instance counter. This allows us to
        # ensure every class has a unique name.
        History.instance_counter += 1

        if self not in sinn.inputs:
            sinn.inputs[self] = set()

        if name is None:
            name = "history{}".format(self.instance_counter)
        # Grab defaults from the other instance:
        if hist is not sinn._NoValue:
            if not isinstance(hist, History):
                raise ValueError("The first parameter to the History initializer, "
                                 "if given, must be another history instance. "
                                 "All other parameters should use keywords.")
            if t0 is sinn._NoValue:
                t0 = hist.t0
            if tn is sinn._NoValue:
                tn = hist.tn
            if dt is sinn._NoValue:
                dt = hist.dt
            if shape is sinn._NoValue:
                shape = hist.shape
            if f is sinn._NoValue:
                f = hist._update_function
            if iterative is sinn._NoValue:
                iterative = hist._iterative
            if use_theano is sinn._NoValue:
                use_theano = hist.use_theano
        else:
            assert(sinn._NoValue not in [t0, tn, dt, shape])
        if iterative is sinn._NoValue:
            # Default value
            iterative = True

        # Determine whether the series data will use Theano.
        # The default depends on whether Theano is loaded. If it is not loaded
        # and we try to use it for this history, an error is raised.
        if use_theano is sinn._NoValue:
            self.use_theano = sinn.config.use_theano()
        else:
            self.use_theano = use_theano

        super().__init__(t0, np.ceil( (tn - t0)/dt ) * dt + t0)
            # Set t0 and tn, ensuring (tn - t0) is a round multiple of dt

        ############
        # Flags
        self._iterative = iterative
        self._compiling = False
        self.locked = False
            # When this is True, the history is not allowed to be changed
        ############

        self.name = name
        self.shape = shape
            # shape at a single time point, or number of elements
            # in the system
        self.ndim = len(shape)

        self.dt = np.array(dt, dtype=config.floatX)
        #self._cur_tidx = -1
        if self.use_theano and not sinn.config.use_theano():
            raise ValueError("You are attempting to construct a series with Theano "
                             "but it is not loaded. Run `sinn.config.load_theano()` "
                             "before constructing this history.")
        elif self.use_theano:
            self.compiled_history = None
        self._cur_tidx = shim.shared(np.array(-1, dtype='int64'),
                                            name = 't idx (' + name + ')')
            # Tracker for the latest time bin for which we
            # know history.
        self._original_tidx = self._cur_tidx

        if f is sinn._NoValue:
            # TODO Find way to assign useful error message to a missing f, that
            #      works for histories derived from others (i.e. is recognized as the
            #      "missing function" function. Maybe a global function or class,
            #      with which we use isinstance ?
            # Set a default function that will raise an error when called
            def nofunc(*arg):
                raise RuntimeError("The update function for history {} is not set."
                                   .format(self.name))
            self._update_function = nofunc
        else:
            self.set_update_function(f)
        self._compute_range = None

        self._tarr = np.arange(self.t0,
                               self.tn + self.dt - config.abs_tolerance,
                               self.dt)
        # 'self.tn+self.dt' ensures the upper bound is inclusive,
        # -config.abs_tolerance avoids including an extra bin because of rounding errors

        self.t0idx = shim.cast(0, self._cur_tidx.dtype)       # the index associated to t0
        self.tn = self._tarr[-1]                              # remove risk of rounding errors
        self._unpadded_length = len(self._tarr)     # Save this, because _tarr might change with padding

    def __len__(self):
        return self._unpadded_length

    def __lt__(self, other):
        # This allows histories to be sorted. IPyParallel sometimes requires this
        if isinstance(other, History):
            return self.name < other.name
        else:
            raise TypeError("'Lesser than' comparison is not supported between objects of type History and {}."
                            .format(type(other)))

    def raw(self):
        # The raw format is meant for data longevity, and so should
        # seldom, if ever, be changed

        if self.use_theano:
            if self.compiled_history is not None:
                raw = self.compiled_history.raw()
            else:
                raise AttributeError("The `raw` method for uncompiled Theano "
                                     "histories is undefined.")
            raw['name'] = self.name # Replace with non-compiled name
        else:
            raw = {'name': self.name,
                   't0': self.t0,
                   'tn': self.tn,
                   'dt': self.dt,
                   't0idx': self.t0idx,
                   '_unpadded_length': self._unpadded_length,
                   '_cur_tidx': self._original_tidx.get_value(),
                   'shape': self.shape,
                   'ndim': self.ndim,
                   '_tarr': self._tarr,
                   '_data': self._data.get_value(),
                   #'use_theano': self.use_theano,
                   '_iterative': self._iterative,
                   'locked': self.locked
            }
        return raw

    @classmethod
    def from_raw(cls, raw, update_function=sinn._NoValue, use_theano=False, lock=True):
        """
        Parameters
        ----------
        use_theano: bool
            If True, a second Theano history will be constructed, and the loaded
            one attached as its `compiled_history` attribute.
            If unspecified, the behaviour is the same as for the History initializer.
        """
        hist =  cls(name = str(raw['name']),
                    t0 = float(raw['t0']), tn = float(raw['tn']), dt = float(raw['dt']),
                    shape = tuple(raw['shape']),
                    f = update_function,
                    iterative = bool(raw['_iterative']),
                    use_theano = False)
        hist.t0idx = int(raw['t0idx'])
        hist._unpadded_length = int(raw['_unpadded_length'])
        hist.locked = bool(raw['locked'])
            # Could probably be removed, since we set a lock status later, but
            # ensures the history is first loaded in the same state

        hist._tarr = raw['_tarr']
        # Decide whether to wrap the history in another Theano history
        if use_theano is sinn._NoValue:
            use_theano = sinn.config.use_theano()
        if use_theano:
            hist.name = str(raw['name']) + " (compiled)"
            theano_hist = cls(hist, name=str(raw['name']), use_theano=True)
            theano_hist.compiled_history = hist
            rethist = theano_hist
        else:
            hist.name = str(raw['name'])
            rethist = hist
        rethist._original_tidx = shim.shared( np.array(int(raw['_cur_tidx']), dtype='int64'),
                                              name = 't idx (' + rethist.name + ')' )
        rethist._cur_tidx = rethist._original_tidx
        rethist._data = shim.shared(raw['_data'], name = rethist.name + " data")
        if lock:
            rethist.lock()
        else:
            rethist.unlock()

        return rethist


    def __getitem__(self, key):
        """
        Ensure that history has been computed far enough to retrieve
        the desired timeslice, and then call the class' `retrieve` method.

        As a side-effect, adds this history to the list of inputs (it is presumed
        that since we indexing on it, it will appear in the computational graph.)

        NOTE: key will not be shifted to reflect history padding. So `key = 0`
        may well refer to a time *before* t0.

        Parameters
        ----------
        key: int, float, slice or array
            If an array, it must be consecutive (this is not checked).
        """
        if shim.isscalar(key):
            return self._getitem_internal(key)
        elif isinstance(key, slice):
            return self._getitem_internal(key)
        elif shim.isarray(key):
            return shim.ifelse(shim.eq(key.shape[0], 0),
                               self._original_data[0:0], # Empty time slice
                               self._getitem_internal(key))
        else:
            raise RuntimeError("Unrecognized key {} of type {}. (history: {})"
                               .format(key, type(key), self.name))

    def _getitem_internal(self, key):
        """Does the actual work of __getitem__; the latter just excludes special
        cases, like empty keys."""
        if shim.isscalar(key):
            key = self.get_t_idx(key)
                # If `key` is a negative integer, returns as is
                # If `key` is a negative float, returns a positive index integer
            key = shim.ifelse(key >= 0,
                              key,
                              len(self._tarr) + key)
                # key = -1 returns last element
            latest = key
            key_filter = None

        elif isinstance(key, slice):
            # NOTE: If you make changes to the logic here, check mixins.convolve
            #       to see if they should be ported.

            step = 1 if key.step is None else key.step
            step = self.index_interval(step)
                # Make sure we have an index step

            if key.start is None:
                start = shim.ifelse(shim.gt(step, 0),
                                    0,
                                    len(self._tarr) )
            else:
                start = self.get_t_idx(key.start)
                start = shim.ifelse(start >= 0,
                                    start,
                                    len(self._tarr) + start)
            if key.stop is None:
                stop = shim.ifelse(shim.gt(step, 0),
                                   len(self._tarr),
                                   -1)
            else:
                stop = self.get_t_idx(key.stop)
                stop = shim.ifelse(stop >= 0,
                                    stop,
                                    len(self._tarr) + stop)

            # allow to select beyond the end, to be consistent
            # with slicing conventions
            earliest = shim.largest(0, shim.smallest(start, stop - step))
            latest = shim.smallest(len(self._tarr) - 1,
                                   shim.largest(start, stop - step))
            shim.check(earliest >= 0)
            shim.check(latest >= 0)
            # if step > 0:
            #     start = shim.largest(start, 0)
            #     stop = shim.smallest(stop, self.t0idx + len(self) + 1)
            #     latest = stop - 1
            #         # `latest` is the latest time we need to compute
            # else:
            #     start = shim.smallest(start, self.t0idx + len(self))
            #     stop = shim.largest(stop, -1)
            #     latest = start

                # compute_up_to will try to compute to the end if this is false

            key_filter = None if key.step is None else slice(None, None, step)
            key = slice(earliest, latest + 1)
                # applying the step 'filter' afterwards avoids the problem of times
                # inadvertently being converted to negative indices
                # +1 because 'latest' is always > earliest

            # if isinstance(key, slice) and key.stop is None:
            #     # key.stop can't be replaced by a numerical value if step < 0
            #     # for non-slice types, negative values don't really make sense,
            #     # so we treat them as 'None' (typically what happens here is that
            #     # stop = -1
            #     key = slice(start, None, step)
            # else:
            #     key = slice(start, stop, step)

        elif shim.isarray(key):
            #TODO: treat case where this gives a stop at -1
            assert(key.ndim == 1)
            start = self.get_t_idx(key[0])
            end = self.get_t_idx(key[-1])
            earliest = shim.largest(0, shim.smallest(start, end))
            latest = shim.smallest(len(self._tarr) - 1, shim.largest(start, end))
            step = shim.ifelse(shim.eq(key.shape[0], 1),
                               # It's a 1 element array – just set step to 1 and don't worry
                               np.int16(1),  # int16 is the return type of index_interval
                               # Set the step as the difference of the first two elements
                               shim.LazyEval(lambda key: self.index_interval(key[1] - key[0]), (key,) ))
	                           # `LazyEval` prevents Python from greedily executing key[1]
                                   # even when key has length 1
            # Make sure the entire indexing array has the same step
            if not shim.is_theano_object(key) and key.shape[0] > 1:
                assert(np.all(sinn.isclose(key[1:] - key[:-1], key[1]-key[0])))

            key = slice(earliest, latest + 1)
                # +1 because the latest bound is inclusive
            key_filter = slice(None, None, step)

        else:
            raise ValueError("Trying to index using {} ({}). 'key' should be an "
                             "integer, a float, or a slice of integers and floats"
                             .format(key, type(key)))

        if shim.is_theano_object(latest, self._cur_tidx):
            # For theano variables, we can't know in advance if we need to compute
            # or not.
            # TODO: always compute, and let compute_up_to decide ?
            self.compute_up_to(latest)
        elif latest > self._cur_tidx:#.get_value():
            # No get_value() here because after updates, _cur_tidx is no longer a shared var
            if (self.use_theano
                and self.compiled_history is not None
                and self.compiled_history._cur_tidx >= latest):
                # ***** RETURN FORK ******
                # If the history has already been computed to this point,
                # just return that.
                result = self.compiled_history[key]
                if key_filter is None:
                    return result
                else:
                    return result[key_filter]

            self.compute_up_to(latest)

        # Add `self` to list of inputs
        #if self.use_theano and self not in sinn.inputs:
        if self not in sinn.inputs:
            sinn.inputs[self] = set()

        result = self.retrieve(key)
        if key_filter is None:
            return result
        else:
            return result[key_filter]

    def clear(self):
        """
        Invalidate the history data, forcing it to be recomputed the next time its queried.
        Functionally equivalent to clearing the data.

        *Note* If this history is part of a model, you should use that
        model's `clear_history` method instead.
        """
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        self._original_tidx.set_value(self.t0idx - 1)
        self._cur_tidx = self._original_tidx

        if self.use_theano and self.compiled_history is not None:
            self.compiled_history.clear()

        try:
            super().clear()
        except AttributeError:
            pass

    def lock(self, warn=True):
        """
        Lock the history to prevent modifications.
        Raises a warning if the history has not been set up to its end
        (since once locked, it can no longer be updated). This can be disabled
        by setting `warn` to False.
        """
        if shim.is_theano_variable(self._cur_tidx):
            raise RuntimeError("You are trying to lock the history {}, which "
                               "in the midst of building a Theano graph. Reset "
                               "it first".format(self.name))
        if warn and (self._original_tidx.get_value() < self.t0idx + len(self) - 1
                     and (not self.use_theano
                          or self.compiled_history is None)):
            # Only trigger for Theano histories if their compiled histories are unset
            # (If they are set, they will do their own check)
            logger.warning("You are locking the unfilled history {}. Trying to "
                           "evaluate it beyond {} will trigger an error."
                           .format(self.name, self._tarr[self._original_tidx.get_value()]))
        self.locked = True
        if self.use_theano and self.compiled_history is not None:
            self.compiled_history.lock()

    def unlock(self):
        """Remove the history lock."""
        self.locked = False
        if self.use_theano and self.compiled_history is not None:
            self.compiled_history.unlock()

    def set_update_function(self, func):
        """
        Parameters
        ----------
        func: callable
            The update function. Its signature should be
            `func(t)`
        """
        def f(t):
            res = func(t)
            if shim.isscalar(res):
                # update functions need to output an object with a shape
                return shim.add_axes(res, 1)
            else:
                return res
        self._update_function = f

    def set_range_update_function(self, func):
        """
        Parameters
        ----------
        func: callable
            The update function. Its signature should be `func(time array)`
            `time array` will be passed as an array of consecutive times,
            so `func` can safely assume that its input is ordered and that
            it doesn't skip over any time bin.
        """
        self._compute_range = func

    def pad_time(self, before, after=0):
        """Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are increased to make them exact multiples of dt.

        Padding is not cumulative, so
        ```
        history.pad_time(a)
        history.pad_time(b)
        ```
        is equivalent to
        ```
        history.pad_time(max(a,b))
        ```
        Parameters
        ----------
        before: float | int
            Amount of time to add the time array before t0. It may be
            increased by an amount at most dt, in order to make it a multiple
            of dt.
            If specified as an integer, taken to be the number of time bins
            to add. How much this amounts to in time depends on dt.
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
        for t in (before, after):
            if shim.is_theano_object(t):
                raise ValueError("Times must be NumPy or pure Python variables. ")
        if isinstance(before, int):
            before_idx_len = before
        else:
            before_idx_len = int(np.ceil(before / self.dt))
        if isinstance(after, int):
            after_idx_len = after
        else:
            after_idx_len = int(np.ceil(after / self.dt))
        before = before_idx_len * self.dt
        after = after_idx_len * self.dt

        before_array = np.arange(self.t0 - before, self._tarr[0], self.dt)
        after_array = np.arange(self._tarr[-1] + self.dt - config.abs_tolerance,
                                self.tn + self.dt - config.abs_tolerance + after,
                                self.dt)
            # Use of _tarr ensures we don't add more padding than necessary
        self._tarr = np.hstack((before_array,
                                self._tarr[self.t0idx:self.t0idx+len(self)],
                                after_array))
        self.t0idx = len(before_array)

        # Update the current time index
        if self._cur_tidx == self._original_tidx:
            self._original_tidx.set_value( self._original_tidx.get_value() + len(before_array) )
            self._cur_tidx = self._original_tidx
        else:
            # _cur_tidx is already a transformed variable, so don't link it to _original_tidx
            self._original_tidx.set_value( self._original_tidx.get_value() + len(before_array) )
            self._cur_tidx += len(before_array)

        return len(before_array), len(after_array)

    def compute_up_to(self, tidx):
        """
        Compute the history up to `tidx` inclusive.

        Parameters
        ----------
        tidx: int, str
            Index up to which we need to compute. Can also be the string
            'end', in which the entire history (excluding subsequent padding)
            is computed.
            NOTE: The index must be positive. Negative indices are treated as
            before 0, and lead to no computation. (I.e. negative indices are
            not subtracted from the end.)
        """

        if tidx == 'end':
            tidx = self.t0idx + len(self) - 1

        shim.check(shim.istype(tidx, 'int'))

        start = self._original_tidx + 1
        end = tidx
        #end = shim.ifelse(tidx >= 0,
        #                  tidx,
        #                  len(self._tarr) + tidx)

        if self.locked:
            if not shim.is_theano_object(end):
                assert(self._original_tidx.get_value() >= end)
                if not shim.is_theano_object(self._cur_tidx):
                    assert(self._cur_tidx <= end)
            return

        #if shim.is_theano_object(tidx):
        if self.use_theano and (self._compiling
                                or any(hist._compiling for hist in sinn.inputs)):
            # Don't actually compute: just store the current_idx we would need
            # HACK The 'or' line prevents any chaining of Theano graphs
            # FIXME Allow chaining of Theano graphs when possible/specified
            self._cur_tidx = shim.largest(end, self._cur_tidx)
            return

        if (not shim.is_theano_object(end, self._cur_tidx)
            and end <= self._cur_tidx):
            # Nothing to compute
            # We exclude Theano objects because in this case we don't
            # know what value `end` has
            # TODO? Remove this and rely on update ? So that NumPy runs
            # go through the same code as Theano ?
            return

        if (not shim.is_theano_object(end)
            #and hasattr(self, '_theano_cur_tidx')
            and not shim.is_theano_variable(self._cur_tidx)
            and end <= self._cur_tidx.get_value()):
            # Theano computations over fixed time intervals can land here.
            # In these cases we can also safely abort the computation
            return

        if (not shim.is_theano_object(end)
            and self.use_theano
            and self.compiled_history is not None
            and end <= self.compiled_history._cur_tidx):
            # A computed value already exists and has been computed to this point
            return

        #########
        # Did not abort the computation => now let's do the computation

        stop = end + 1    # exclusive upper bound
        # Construct a time array that will work even for Theano tidx
        if shim.is_theano_object(start) or shim.is_theano_object(end):
            tarr = shim.gettheano().shared(self._tarr, borrow=True)
            printlogs = False
        else:
            tarr = self._tarr
            printlogs = True

        if not self._iterative:
            batch_computable = True
        #elif (self.use_theano and self._is_batch_computable()):
        elif self._is_batch_computable():
            batch_computable = True
        else:
            batch_computable = False

        if self._compute_range is not None:
            # A specialized function for computing multiple time points
            # has been defined – use it.
            if printlogs:
                logger.info("Computing {} up to {}. Using custom batch operation."
                            .format(self.name, tarr[tidx]))
            self.update(slice(start, stop),
                        self._compute_range(tarr[slice(start, stop)]))

        elif batch_computable:
            # Computation doesn't depend on history – just compute the whole thing in
            # one go
            if printlogs:
                logger.info("Computing {} from {} to {}. Computing all times simultaneously."
                            .format(self.name, tarr[start], tarr[stop-1]))
            self.update(slice(start,stop),
                        self._update_function(tarr[slice(start,stop)][::-1])[::-1])
                # The order in which _update_function is called is flipped, putting
                # later times first. This ensures that if dependent computations
                # are triggered, they will also batch update.
        else:
            assert(not shim.is_theano_object(tarr))
            logger.info("Iteratively computing {} from {} to {}."
                        .format(self.name, tarr[start], tarr[stop-1]))
            old_percent = 0
            for i in np.arange(start, stop):
                percent = (i*100)//stop
                if percent > old_percent:
                    logger.info("{}%".format(percent))
                    old_percent = percent
                self.update(i, self._update_function(tarr[i]))

        logger.info("Done computing {}.".format(self.name))

    def get_time_array(self, time_slice=slice(None, None), include_padding=False):
        """Return the time array.
        By default, the padding portions before and after are not included.
        Time points which have not yet been computed are also excluded.
        The flag `include_padding` changes this behaviour:
            - True or 'all'     : include padding at both ends
            - 'begin' or 'start': include the padding before t0
            - 'end'             : include the padding after tn
            - False (default)   : do not include padding
        """

        #if not shim.isshared(self._cur_tidx):
        if not self._cur_tidx == self._original_tidx:
            raise RuntimeError("You are in the midst of constructing a Theano graph. "
                               "Reset history {} before trying to obtain its time array."
                               .format(self.name))

        if time_slice.start is None:
            slcidx_start = 0
        else:
            slcidx_start = self.get_t_idx(time_slice.start)
        if time_slice.stop is None:
            slcidx_stop = len(self._tarr)
        else:
            slcidx_stop = self.get_t_idx(time_slice.stop)

        if include_padding in ['begin', 'start']:
            start = slcidx_start
            stop = min(slcidx_stop, self._cur_tidx.get_value()+1, self.t0idx+len(self))
        elif include_padding in ['end']:
            start = max(self.t0idx, slcidx_start)
            stop = min(slcidx_stop, self._cur_tidx.get_value()+1, len(self._tarr))
        elif include_padding in [True, 'all']:
            start = slcidx_start
            stop = min(slcidx_stop, self._cur_tidx.get_value()+1, len(self._tarr))
        else:
            start = max(self.t0idx, slcidx_start)
            stop = min(slcidx_stop, self._cur_tidx.get_value()+1, self.t0idx + len(self))

        if stop < slcidx_stop and slcidx_stop < self.t0idx+len(self):
            logger.warning("You asked for the time array up to {}, but the history "
                            "has only been computed up to {}."
                            .format(self.get_time(slcidx_stop),
                                    self.get_time(self.t0idx+len(self))))

        return self._tarr[start:stop]

    def retrieve(self, key):
        raise NotImplementedError  # retrieve function is history type specific

    def update(self, tidx, value):
        raise NotImplementedError  # update function is history type specific

    def time_interval(self, Δt):
        """
        If Δt is a time (float), do nothing.
        If Δt is an index (int), convert to time by multiplying by dt.
        """
        if shim.istype(Δt, 'int'):
            return Δt*self.dt
        else:
            return Δt

    def index_interval(self, Δt):
        """
        If Δt is a time (float), convert to index interval by multiplying by dt.
        If Δt is an index (int), do nothing.
        OPTIMIZATION NOTE: This is a slower routine than its inverse `time_interval`.
        Avoid it in code that is called repeatedly, unless you know that Δt is an index.
        """
        if shim.istype(Δt, 'int'):
            return Δt
        else:
            try:
                shim.check( Δt * config.get_rel_tolerance(Δt) < self.dt )
            except AssertionError:
                raise ValueError("You've tried to convert a time (float) into an index "
                                 "(int), but the value is too large to ensure the absence "
                                 "of numerical errors. Try using a higher precision type.")
            quotient = Δt / self.dt
            rquotient = shim.round(quotient)
            try:
                shim.check( shim.abs(quotient - rquotient) < config.get_abs_tolerance(Δt) / self.dt )
            except AssertionError:
                logger.error("Δt: {}, dt: {}".format(Δt, self.dt) )
                raise ValueError("Tried to convert t=" + str(Δt) + " to an index interval "
                                 "but its not a multiple of dt.")
            return shim.cast_int16( rquotient )

    def get_time(self, t):
        """
        If t is an index (i.e. int), return the time corresponding to t_idx.
        Else just return t
        """
        # TODO: Is it OK to enforce single precision ?

        if shim.istype(t, 'int'):
            return config.cast_floatX(self._tarr[0] + t*self.dt)
        else:
            return t

    def get_t_idx(self, t):
        """Return the idx corresponding to time t. Fails if no such index exists.
        It is ok for the t to correspond to a time "in the future",
        and for the data array not yet to contain a point at that time.
        `t` may also be specified as a slice, in which case a slice of time
        indices is returned.
        """
        def _get_tidx(t):
            if shim.istype(t, 'int'):
                # It's an easy error to make, specify a time as an int
                # Print a warning, just in case.
                # print("Called get_t_idx on an integer ({}). Assuming this to be an INDEX".format(t)
                #       + " (rather than a time) and returning unchanged.")
                return shim.cast( t, dtype = self._cur_tidx.dtype )
            else:
                try:
                    shim.check(t >= self._tarr[0])
                except AssertionError:
                    raise RuntimeError("You've tried to obtain the time index at t={}, which "
                                       "is outside this history's range. Please add padding."
                                       .format(t))

                if self._strict_index_rounding:
                    # Enforce that times be multiples of dt

                    try:
                        shim.check( t * config.get_rel_tolerance(t) < self.dt )
                    except AssertionError:
                        raise ValueError("You've tried to convert a time (float) into an index "
                                        "(int), but the value is too large to ensure the absence "
                                        "of numerical errors. Try using a higher precision type.")
                    t_idx = (t - self._tarr[0]) / self.dt
                    r_t_idx = shim.round(t_idx)
                    if (not shim.is_theano_object(r_t_idx)
                          and abs(t_idx - r_t_idx) > config.get_abs_tolerance(t) / self.dt):
                        logger.error("t: {}, t0: {}, t-t0: {}, t_idx: {}, dt: {}"
                                     .format(t, self._tarr[0], t - self._tarr[0], t_idx, self.dt) )
                        raise ValueError("Tried to obtain the time index of t=" +
                                        str(t) + ", but it does not seem to exist.")
                    return shim.cast(r_t_idx, dtype = self._cur_tidx.dtype)

                else:
                    # Allow t to take any value, and round down to closest
                    # multiple of dt
                    return shim.cast( (t - self._tarr[0]) // self.dt,
                                      dtype = self._cur_tidx.dtype )

        if isinstance(t, slice):
            start = self.t0idx if t.start is None else _get_tidx(t.start)
            stop = self.t0idx + len(self) if t.stop is None else _get_tidx(t.stop)
            return slice(start, stop, t.step)
        else:
            return _get_tidx(t)

    def make_positive_slice(self, slc):
        def flip(idx):
            assert(shim.istype(idx, 'int'))
            return idx if idx >= 0 else len(self._tarr) + idx
        return slice(0 if slc.start is None else flip(slc.start),
                     len(self._tarr) if slc.stop is None else flip(slc.stop))

    def theano_reset(self):
        """Allow theano functions to be called again.
        It is assumed that variables in self._theano_updates have been safely
        updated externally.
        """
        if self.locked:
            raise RuntimeError("Cannot modify the locked history {}."
                               .format(self.name))

        self._cur_tidx = self._original_tidx
        self._data = self._original_data

        try:
            super().theano_reset()
        except AttributeError:
            pass

    def discretize_kernel(self, kernel):

        dis_attr_name = "discrete_" + str(id(self))  # Unique id for discretized kernel

        if hasattr(kernel, dis_attr_name):
            # TODO: Check that this history (self) hasn't changed
            return getattr(kernel, dis_attr_name)

        else:
            #TODO: Add compability check of the kernel's shape with this history.
            #shim.check(kernel.shape == self.shape*2)
            #    # Ensure the kernel is square and of the right shape for this history

            if config.integration_precision == 1:
                kernel_func = kernel.eval
            elif config.integration_precision == 2:
                # TODO: Avoid recalculating eval at the same places by writing
                #       a compute_up_to function and passing that to the series
                kernel_func = lambda t: (kernel.eval(t) + kernel.eval(t+self.dt)) / 2
            else:
                # TODO: higher order integration with trapeze or simpson's rule
                raise NotImplementedError

            # The kernel may start at a position other than zero, resulting in a shift
            # of the index corresponding to 't' in the convolution
            idx_shift = int(round(kernel.t0 / self.dt))
                # We don't use shim.round because time indices must be Python numbers
            t0 = idx_shift * self.dt  # Ensure the discretized kernel's t0 is a multiple of dt

            memory_idx_len = int(kernel.memory_time // self.dt) - 1 - idx_shift
                # It is essential here to use the same forumla as pad_time
                # We substract one because one bin more or less makes no difference,
                # and doing so ensures that padding with `memory_time` always
                # is sufficient (no dumb numerical precision errors adds a bin)
                # NOTE: kernel.memory_time includes the time between 0 and t0,
                # and so we need to substract idx_shift to keep only the time after t0.

            #full_idx_len = memory_idx_len + idx_shift
            #    # `memory_time` is the amount of time before t0
            dis_name = ("dis_" + kernel.name + " (" + self.name + ")")
            # if shim.is_theano_variable(kernel.eval(0)):
            #     use_theano=True
            # else:
            #     use_theano=False
            use_theano=False
            dis_kernel = Series(t0=t0,
                                tn=t0 + memory_idx_len*self.dt,
                                dt=self.dt,
                                shape=kernel.shape,
                                f=kernel_func,
                                name=dis_name,
                                use_theano=use_theano,
                                iterative=False)
                # Kernels are non-iterative by definition: they only depend on their parameters
            dis_kernel.idx_shift = idx_shift

            setattr(kernel, dis_attr_name, dis_kernel)

            return dis_kernel

    def add_input(self, variable):

        if self not in sinn.inputs:
            sinn.inputs[self] = set()

        if isinstance(variable, str):
            # Not sure why a string would be an input, but it guards against the next line
            sinn.inputs[self].add(x)
            #self._inputs.add(variable)
        try:
            for x in variable:
                sinn.inputs[self].add(x)
        except TypeError:
            # variable is not iterable
            sinn.inputs[self].add(x)
    add_inputs = add_input
        # Synonym to add_input

    def get_input_list(self, inputs=None):
        if inputs is None:
            inputs = []
        try:
            inputs = set(inputs)
        except TypeError:
            inputs = set([inputs])

        sinn.inputs[self] = sinn.inputs[self].union(inputs)
        if self in sinn.inputs[self]:
            assert(self._iterative)
            sinn.inputs[self].remove(self)  # We add `self` separately. We remove it now to
                                            # avoid calling its `compile` method

        input_list = list(sinn.inputs[self])

        input_histories = []
        eval_histories = []
        terminating_histories = []

        # To avoid circular dependencies (e.g. where A[t] depends on B[t] which
        # depends on A[t-1]),  we recursively compile the inputs so that their
        # own graphs don't show up in this one. The `compiling` flag is used
        # to prevent infinite recursion.

            # Typically, calculations take the form A[t-1] -> B[t] -> C[t] -> A[t]
            # In this case, A[t-1] would be the history terminating the cycle started at A[t]
            # These are identified by the fact that they are already in the middle of a
            # compilation (with the `compile` flag).
            # When determining up to which point they need to be computed, we substract one
            # from these (otherwise we would get infinite recursion).
        input_list_for_eval = []
        for i, inp in enumerate(input_list):
            if isinstance(inp, History):
                input_histories.append(inp)
                if inp.use_theano:
                    if inp.compiled_history is None:
                        inp.compile(inputs.difference([inp]))
                    if inp._compiling:
                        terminating_histories.append(inp)
                    eval_histories.append(inp.compiled_history)
                    input_list[i] = inp._data
                else:
                    raise ValueError("Cannot compile a history which has non-Theano "
                                     "dependencies.")

                input_list_for_eval.append(eval_histories[-1]._data)
            else:
                raise ValueError("History compilation with inputs of type {} is unsupported."
                                 .format(type(inp)))
        assert(len(input_list) == len(input_list_for_eval))

        return input_list, input_histories, eval_histories, terminating_histories, input_list_for_eval

    def compile(self, inputs=None):

        assert(not self._compiling)
        self._compiling = True

        if not self.use_theano:
            raise RuntimeError("You cannot compile a Series that does not use Theano.")

        # Create the new history which will contain the compiled update function
        self.compiled_history = self.__class__(self, name=self.name + " (compiled)", use_theano=False)
        # Compiled histories must have exactly the same indices, so we match the padding
        self.compiled_history.pad(self.t0idx, len(self._tarr) - len(self) - self.t0idx)

        # Compile the update function twice: for integers (time indices) and floats (times)
        # We also compile a second function, which returns the time indices up to which
        # each history on which this one depends must be calculated. This is used to
        # precompute each dependent history, (this is not done automatically because we
        # need to use the raw data structure).
        t_idx = shim.getT().scalar('t_idx', dtype='int32')
        assert(len(shim.config.theano_updates) == 0)
        output_res_idx = self._update_function(t_idx)
            # This should populate shim.theano_updates if there are shared variable
            # It might also trigger the creation of some intermediary histories,
            # which is why we wait until here to grab the list of inputs.
        input_list, input_histories, eval_histories, terminating_histories, input_list_for_eval = self.get_input_list(inputs)
        # Convert the inputs to a list to fix the order
        if self._iterative:
            input_self = [self._data]
        else:
            input_self = []

        def get_required_tidx(hist):
            # We cast to Theano variables below because the values may
            # be pure NumPy (if they don't depend on t), and
            # theano.function requires variables
            if hist in terminating_histories:
                return shim.asvariable(hist._cur_tidx - 1)
            else:
                return shim.asvariable(hist._cur_tidx)

        try:
            # TODO? Put this try clause on every theano.function call ?
            update_f_idx = shim.gettheano().function(inputs=[t_idx] + input_list + input_self,
                                                outputs=output_res_idx,
                                                updates=shim.config.theano_updates,
                                                on_unused_input='warn')
        except shim.gettheano().gof.fg.MissingInputError as e:
            err_msg = e.args[0].split('\n')[0] # Remove the stack trace
            raise RuntimeError("\nYou seem to be missing some inputs for compiling the history {}. "
                               "Don't forget that all histories on which {} depends "
                               "must be specified by calling its `add_inputs` method.\n"
                               .format(self.name, self.name)
                               + "The original Theano error was (it should contain the name of the missing input):\n"
                               + err_msg)

        output_tidcs_idx = [get_required_tidx(hist) for hist in input_histories]
        get_tidcs_idx = shim.gettheano().function(inputs=[t_idx] + input_list,
                                             outputs=output_tidcs_idx,
                                             on_unused_input='ignore') # No need for duplicate warnings

        for hist in input_histories:
            hist.theano_reset() # resets the _cur_tidx
        shim.theano_reset()
            # Clear the updates stored in shim.theano_updates

        # Now compile for floats
        t_float = shim.getT().scalar('t', dtype=sinn.config.floatX)
        assert(len(shim.config.theano_updates) == 0)
        output_res_float = self._update_function(t_float)
            # This should populate shim.theano_updates if there are shared variable
        update_f_float = shim.gettheano().function(inputs=[t_float] + input_list + input_self,
                                              outputs=output_res_float,
                                              updates=shim.config.theano_updates,
                                              on_unused_input='warn')
        output_tidcs_float = [get_required_tidx(hist) for hist in input_histories]
        get_tidcs_float = shim.gettheano().function(inputs=[t_float] + input_list,
                                               outputs=output_tidcs_float,
                                               on_unused_input='ignore')
        for hist in input_histories:
            hist.theano_reset() # reset _cur_tidx
        shim.theano_reset()
            # Clear the updates stored in shim.theano_updates

        # The new update function type checks the input and evaluates the proper
        # compiled function
        def new_update_f(t):
            # TODO Compile separate functions for when t is an array
            # TODO Safeguard against infinite recursion

            def single_t(t):
                input_data = [data.get_value(borrow=True) for data in input_list_for_eval]
                    # each stored _data member is a shared variable, but we need to evaluate
                    # on the underlying plain NumPy data
                if self._iterative:
                    f_inputs = [t] + input_data + [self.compiled_history._data.get_value(borrow=True)]
                else:
                    f_inputs = [t] + input_data

                if shim.istype(t, 'int'):
                    t_idcs = get_tidcs_idx(t, *input_data)
                    for t_idx, hist in zip(t_idcs, eval_histories):
                        hist.compute_up_to(t_idx)
                    return update_f_idx(*f_inputs)
                else:
                    t_idcs = get_tidcs_float(t, *input_data)
                    for t_idx, hist in zip(t_idcs, eval_histories):
                        hist.compute_up_to(t_idx)
                        # if t_idx > hist._cur_tidx:
                        #     # Normally compute_up_to checks for this, but if
                        #     # t_idx = -1 it will incorrectly compute up to the end
                        #     hist.compute_up_to(t_idx)
                        # -> Should be fixed now: compute_up_to uses 'end' rather than -1
                    return update_f_float(*f_inputs)

            if shim.isscalar(t):
                return single_t(t)
            else:
                return np.array([single_t(ti) for ti in t])

        self.compiled_history.set_update_function(new_update_f)

        self._compiling = False


    ################################
    # Utility functions
    ################################

    def time_array_to_slice(self, time_array):
        """
        Assumes the time_array is monotonous and evenly spaced.
        """
        # We need to be careful here, because time_array could be empty, and then
        # indexing it with time_array[0] or time_array[-1] would trigger an error
        empty_array = shim.eq( shim.min(time_array.shape), 0 )

        step = shim.ifelse(empty_array,
                           self.dt,
                           shim.largest(time_array[1] - time_array[0], self.dt))

        idxstart = shim.ifelse(empty_array,
                               shim.cast(0, dtype=self._cur_tidx.dtype),
                               self.get_t_idx(time_array[0]))
        if not shim.is_theano_object(time_array) and len(time_array) > 1:
            # Check that time_array is evenly spaced
            assert(np.all(sinn.ismultiple(time_array[1:] - time_array[:-1], step)))

        idxstep = self.index_interval(step)

        idxstop = shim.ifelse(empty_array,
                              shim.cast(0, dtype=self._cur_tidx.dtype),
                              self.get_t_idx(time_array[-1]) + idxstep)
            # We add/subtract dt because the time_array upper bound is inclusive

        # Ensure that idxstop is not negative. This would require replacing it by
        # None, and doing that in a Theano graph, if possible at all, is clumsy.
        # Since negative idxstop is an unnecessary corner case, we rather not support
        # it than add cruft to the Theano graph.
        if not shim.is_theano_object(idxstop):
            assert(idxstop >= 0)

        return slice(idxstart, idxstop, idxstep)


    def _is_batch_computable(self):
        """
        Returns true if the history can be computed at all time points
        simultaneously.
        WARNING: This function is only to be used for the construction of
        a Theano graph. After compilation, sinn.inputs is cleared, and therefore
        the result of this function will no longer be valid.
        HACK: sinn.inputs is no longer cleared, so this function should no longer
        be limited to Theano graphs – hopefully that doesn't break anything else.
        """
        if not self._iterative:
            return True

        # Prevent infinite recursion with a temporary property
        if hasattr(self, '_batch_loop_flag'):
            return False
        else:
            self._batch_loop_flag = True

        # Augment the list of inputs with their compiled forms, if they exist
        all_inputs = set(sinn.inputs.keys()).union(
            set([hist.compiled_history for hist in sinn.inputs
                 if hist.use_theano and hist.compiled_history is not None]) )

        # Get the list of inputs. A compiled history may not be in the input list,
        # so if `self` is not found, we try to find its parent
        input_list = None
        if self in sinn.inputs:
            input_list = sinn.inputs[self]
        else:
            for hist in sinn.inputs:
                if (hist.use_theano and hist.compiled_history is self):
                    input_list = sinn.inputs[hist]
                    break
        assert(input_list is not None)
            # If this history is iterative, it should have at least one input

        if not self._iterative:
            # Batch computable by construction
            retval = True
        elif self not in all_inputs:
            # It's the user's responsibility to specify important inputs;
            # if there are none, we assume this history does not depend on any other.
            # Note that undefined intermediary inputs (such as a dynamically
            # created discretized kernel) are fine, as long as the upstream
            # is included in sinn.inputs.
            retval = True
        elif all( hist.locked or hist._is_batch_computable()
                  for hist in input_list):
            # The potential cyclical dependency chain has been broken
            retval = True
        else:
            retval = False

        del self._batch_loop_flag

        return retval

# Spiketimes is currently really slow for long data traces (> 5000 time bins)
# Development efforts have been moved to Spiketrain; maybe in the future if we need
# to track actual times, we will resurrect development of this class
class Spiketimes(ConvolveMixin, History):
    """A class to store spiketrains.
    These are stored as times associated to each spike, so there is
    no well-defined 'shape' of a timeslice. Instead, the `shape` parameter
    is used to indicate the number of neurons in each population.
    """

    _strict_index_rounding = False

    def __init__(self, hist=sinn._NoValue, name=None, *args, t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 pop_sizes=sinn._NoValue, **kwargs):
        """
        All parameters except `hist` are keyword parameters
        `pop_sizes` is always required
        If `hist` is not specified, `t0`, `tn`, `dt` must all be specified.
        Parameters
        ----------
        hist: History instance
            Optional. If passed, will used this history's parameters as defaults.
        t0: float
            Time at which the history starts
        tn: float
            Time at which the history ends
        dt: float
            Timestep
        pop_sizes: integer tuple
            Number if neurons in each population
        **kwargs:
            Extra keyword arguments are passed on to History's initializer
        """
        if name is None:
            name = "spiketimes{}".format(self.instance_counter + 1)
        try:
            assert(pop_sizes is not sinn._NoValue)
        except AssertionError:
            raise ValueError("'pop_sizes' is a required parameter.")
        try:
            len(pop_sizes)
        except TypeError:
            pop_sizes = (pop_sizes,)
        assert(all( shim.istype(s, 'int') for s in pop_sizes ))
        self.pop_sizes = pop_sizes

        shape = (np.sum(pop_sizes),)

        kwshape = kwargs.pop('shape', None)
        # kwargs should not have shape, as it's calculated
        # Return an error if it's different from the calculated value
        if kwshape is not None and kwshape != shape:
            raise ValueError("Specifying a shape to Spiketimes is "
                             "unecessary, as it's calculated from pop_sizes")

        self.pop_idcs = np.concatenate( [ [i]*size
                                          for i, size in enumerate(pop_sizes) ] )
        self.pop_slices = []
        i = 0
        for pop_size in pop_sizes:
            self.pop_slices.append(slice(i, i+pop_size))
            i += pop_size

        super().__init__(hist, name, t0=t0, tn=tn, dt=dt, shape=shape, **kwargs)

        self.initialize()

    def initialize(self, init_data=-np.inf):
        """
        This method is called without parameters at the end of the class
        instantiation, but can be called again to change the initialization
        of spike times.

        Parameters
        ----------
        init_data: float or iterable
            Either a scalar (to which each neuron will initialized) or an
            iterable of same length as the number of neurons.
        """
        # TODO: Allow either a nested or flattened list for init_data

        try:
            init_data[0]
        except:
            subscriptable = False
        else:
            subscriptable = True

        # deque incurs a 10-15% cost in iterations compared with lists,
        # but makes adding new spikes O(1) rather than O(log(n)).
        # Testing has shown negligible difference for sims of 500 time bins,
        # and ~3% improvement for sims of 2000 time bins. The payoff is
        # expected to continually improve as simulations get longer.
        if subscriptable:
            shim.check(len(init_data) == np.sum(self.pop_sizes))
            #shim.check(all(len(pop_init) == pop_size
            #                for pop_init, pop_size
            #                in zip(init_data, self.pop_sizes)))

            self._data = [ deque([init_data[neuron_idx]])
                                 for neuron_idx in range(np.sum(self.pop_sizes)) ]
        else:
            self._data = [ deque([init_data])
                                 for neuron_idx in range(np.sum(self.pop_sizes)) ]

    def clear(self, init_data=-np.inf):
        """Spiketrains can't just be invalidated, they really have to be cleared."""
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        self.initialize(init_data)
        super().clear()

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
            Returns a binary vector of same size as the total number of neurons, each
            element representing a neuron (populations are flattened). Values are 1
            if the neuron fired in this bin, 0 if it didn't fire.
            If key is a float, it must match the bin time exactly. Generally using
            bin indices should be more reliable than bin times.
        If `key` is slice:
            Returns the list of spike times, truncated to the bounds of the slice.
            Slice bounds may be specified as indices (int) or times (float).
            [:] is much more efficient than [0:] if you want all spike times, as
            it just returns the internal list without processing.
            Note that the key's 'step' attribute will be ignored.
        '''
        if shim.istype(key, 'int') or shim.istype(key, 'float'):
            t = self.get_time(key)
            return np.fromiter( ( True if t in spikelist else False
                                   for spikelist in self._data ),
                                 dtype=bool )
        elif isinstance(key, slice):
            if (key.start is None) and (key.stop is None):
                return self._data
            else:
                start = -shim.inf if key.start is None else self.get_time(key.start)
                stop = shim.inf if key.stop is None else self.get_time(key.stop)
                stop -= self.dt  # exclude upper bound, consistent with slicing conv.
                # At present, deque's don't implement slicing. When they do, use that.
                return [ itertools.islice(spikelist,
                                          int(np.searchsorted(spikelist, start)),
                                          int(np.searchsorted(spikelist, stop)))
                         for spikelist in self._data ]
        else:
            raise ValueError("Key must be either an integer, float or a splice object.")

    #was :  def update(self, t, pop_idx, spike_arr):
    def update(self, tidx, neuron_idcs):
        '''Add to each neuron specified in `value` the spiketime `tidx`.
        Parameters
        ----------
        tidx: int, float
            The time index of the spike(s). This is converted
            to actual time and saved.
            Can optionally also be a float, in which case no conversion is made.
            Should not correspond to more than one bin ahead of _cur_tidx.
        neuron_idcs: iterable
            List of neuron indices that fired in this bin.
        '''
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        newidx = self.get_t_idx(tidx)
        if not theano.is_theano_variable(newidx):
            assert(newidx <= self._original_tidx.get_value() + 1)

        time = self.get_time(tidx)
        for neuron_idx in neuron_idcs:
            self._data[neuron_idx].append(time)

        # for neuron_lst, _data in zip(value, self._data):
        #     for neuron_idx in neuron_lst:
        #         _data[neuron_idx].append(time)


        # Set the cur_idx. If tidx was less than the current index, then the latter
        # is *reduced*, since we no longer know whether later history is valid.
        self._cur_tidx = newidx
        if shim.is_theano_object(self._original_tidx):
            shim.add_update(self._original_tidx, self._cur_tidx)
        else:
            self._original_tidx = self._cur_tidx

    def pad(self, before, after=0):
        '''Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.
        """
        Parameters
        ----------
        before: float | int
            Amount of time to add to before t0. If non-zero, all indices
            to this data will be invalidated.
        after: float (default 0)
            Amount of time to add after tn.
        '''
        self.pad_time(before, after)
        # Since data is stored as spike times, no need to update the data

    def set(self, source=None):
        """Set the entire set of spiketimes in one go. `source` may be a list of arrays, a
        function, or even another History instance. It's useful for
        example if we've already computed history by some other means,
        or we specified it as a function (common for inputs).

        Accepted types for `source`: lists of iterables, functions (not implemented)
        These are converted to a list of spike times.

        If no source is specified, the series' own update function is used,
        provided it has been previously defined. Can be used to force
        computation of the whole series.
        """
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        tarr = self._tarr

        if source is None:
            # Default is to use series' own compute functions
            self.compute_up_to('end')

        elif callable(source):
            raise NotImplementedError

        else:
            assert(len(source) == len(self._data))
            for i, spike_list in enumerate(source):
                self._data[i] = spike_list

        self._original_tidx.set_value(self.t0idx + len(self) - 1)
        self._cur_tidx = self._original_tidx
        return self._data


    def _convolve_op_single_t(self, kernel, t, kernel_slice):
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
        ndarray of shape 'npops' x 'npops'.
            It is indexed as result[from pop idx][to pop idx]

        '''
        # TODO: To avoid iterating over the entire list, save the last `end`
        #       time and an array (one element per neuron) of the index of the latest
        #       spike before `end`. Iterations starting at `end` can then exclude all
        #       spikes before that point.
        #       Use `np.find` to get the `start` and `end` index, and sum between them

        # TODO: allow kernel to return a value for each neuron

        # TODO: move callable test to ConvolveMixin
        # if callable(kernel):
        #    f = kernel
        # else:
        f = kernel.eval

        if kernel_slice.stop is None:
            start = t - kernel.memory_time - kernel.t0
        else:
            if shim.istype(kernel_slice.stop, 'int'):
                start = t - kernel_slice.stop*self.dt - kernel.t0
            else:
                start = t - kernel_slice.stop
        if kernel_slice.start is None:
            stop = t - kernel.t0
        else:
            if shim.istype(kernel_slice.stop, 'int'):
                stop = t - kernel_slice.start*self.dt - kernel.t0
            else:
                stop = t - kernel_slice.start

        shim.check( kernel.ndim <= 2 )
        #shim.check( kernel.shape[0] == len(self.pop_sizes) )
        #shim.check( len(kernel.shape) == 2 and kernel.shape[0] == kernel.shape[1] )

        # For explanations of how summing works, see the IPython
        # notebook in the docs
        _data = self[start:stop]
        if kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1]:
            shim.check(kernel.shape[0] == len(self.pop_sizes))
            return np.stack (
                     np.asarray(np.sum( f(t-s, from_pop_idx)
                             for spike_list in _data[self.pop_slices[from_pop_idx]]
                             for s in spike_list )).reshape(kernel.shape[0:1])
                     for from_pop_idx in range(len(self.pop_sizes)) ).T
                # np.asarray is required because summing over generator
                # expressions uses Python sum(), and thus returns a
                # scalar instead of a NumPy float

        elif kernel.ndim == 1 and kernel.shape[0] == len(self.pop_sizes):
            return shim.lib.concatenate(
                  [ shim.lib.stack( shim.asarray(shim.lib.sum( f(t-s, from_pop_idx) for s in spike_list ))
                               for spike_list in _data[self.pop_slices[from_pop_idx]] )
                    for from_pop_idx in range(len(self.pop_sizes)) ] )

        else:
            raise NotImplementedError

class Spiketrain(ConvolveMixin, History):
    """A class to store spiketrains.
    These are stored in a sparse array where spikes are indicated by 1s.
    The `shape` parameter doesn't exactly correspond to a timeslice; instead
    it is used to indicate the number of neurons in each population.
    These populations are flattened, such that a timeslice is always 2D array;
    if the `shape` is `(N1, N2, N3)`, than the actual shape of a timeslice will
    be `(1 x N1+N2+N3)`.

    Currently we are using a csr array, as Theano only supports csc and csr.
    This means that adding spikes is a relatively costly operation. It also
    means that putting the update function inside a scan (for example to generate
    a whole trace) might lead to a really nasty Theano graph (or maybe not?
    I haven't tried, but the Theano docs warn against this). So in short,
    this implementation is still at the 'just get it working' stage, and may need
    to be changed in the future.
    """

    _strict_index_rounding = True

    def __init__(self, hist=sinn._NoValue, name=None, *args, t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 pop_sizes=sinn._NoValue, **kwargs):
        """
        All parameters except `hist` are keyword parameters
        `pop_sizes` is always required
        If `hist` is not specified, `t0`, `tn`, `dt` must all be specified.
        Parameters
        ----------
        hist: History instance
            Optional. If passed, will used this history's parameters as defaults.
        t0: float
            Time at which the history starts
        tn: float
            Time at which the history ends
        dt: float
            Timestep
        pop_sizes: integer tuple
            Number if neurons in each population
        **kwargs:
            Extra keyword arguments are passed on to History's initializer
        """
        if name is None:
            name = "spiketimes{}".format(self.instance_counter + 1)
        try:
            assert(pop_sizes is not sinn._NoValue)
        except AssertionError:
            raise ValueError("'pop_sizes' is a required parameter.")
        try:
            len(pop_sizes)
        except TypeError:
            pop_sizes = (pop_sizes,)
        assert(all( shim.istype(s, 'int') for s in pop_sizes ))
        self.pop_sizes = pop_sizes

        shape = (np.sum(pop_sizes),)

        kwshape = kwargs.pop('shape', None)
        # kwargs should not have shape, as it's calculated
        # Return an error if it's different from the calculated value
        if kwshape is not None and kwshape != shape:
            raise ValueError("Specifying a shape to Spiketimes is "
                             "unecessary, as it's calculated from pop_sizes")

        # self.pop_idcs is a 1D array with as many entries as there are units
        # The value at each entry is that neuron's population index
        self.pop_idcs = np.concatenate( [ [i]*size
                                          for i, size in enumerate(pop_sizes) ] )

        # self.pop_slices is a list of slices, such that
        # self.data[i][ self.pop_slices[j] ] returns the set of neurons corresponding
        # to population j at time bin i
        self.pop_slices = []
        i = 0
        for pop_size in pop_sizes:
            self.pop_slices.append(slice(i, i+pop_size))
            i += pop_size

        super().__init__(hist, name, t0=t0, tn=tn, dt=dt, shape=shape, **kwargs)

        self.initialize()

    def initialize(self, init_data=None):
        """
        This method is called without parameters at the end of the class
        instantiation, but can be called again to change the initialization
        of spike times.

        Parameters
        ----------
        init_data: ndarray
            n x N array, where n is arbitrary and N is the total number of units.
            The first n time points will be initialized with the data from init_data
            If not specified, the data is initialized to zero.
            Note that the initialized data is set for the first n time indices,
            so if padding is present, those will be before t0. Consequently, the
            typical use is to use n equal to the padding length (i.e. t0idx).
        """
        # TODO: Allow either a nested or flattened list for init_data

        nneurons = np.sum(self.pop_sizes)


        if init_data is None:
            self._data = shim.sparse.coo_matrix('spike train',
                                                shape=(len(self._tarr), nneurons),
                                                dtype='int8')
            # We are just going to store 0s and 1s, so might as well use the smallest
            # available int in Theano
            # Tests have shown that coo_matrix is faster than lil_matrix for the use
            # we make here

        else:
            shim.check(init_data.shape[1] == nneurons)
            n = len(init_data)
            csc_data = shim.sparse.csc_matrix('spike train',
                                              shape=(len(self._tarr), nneurons),
                                              dtype='int8')
            csc_data[:n,:] = shim.sparse.csc_from_dense(init_data.astype('int8'))
            # This may through an efficiency warning, but we can ignore it since
            # self._data is empty
            self._data = csc_data.tocoo()
            # WARNING: This will break with Theano until/if we implement a
            #          coo matrix interface in theano_shim.

    def clear(self, init_data=None):
        """Spiketrains shouldn't just be invalidated, since then multiple runs
        would make them more and more dense."""
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        self.initialize(None)
        super().clear()

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
            Returns a binary vector of same size as the total number of neurons, each
            element representing a neuron (populations are flattened). Values are 1
            if the neuron fired in this bin, 0 if it didn't fire.
            If key is a float, it must match the bin time exactly. Generally using
            bin indices should be more reliable than bin times.
        If `key` is slice:
            Returns the list of spike times, truncated to the bounds of the slice.
            Slice bounds may be specified as indices (int) or times (float).
        '''
        if shim.istype(key, 'int') or shim.istype(key, 'float'):
            tidx = self.get_t_idx(key)
            return self._data[tidx]

        elif isinstance(key, slice):
            start = None if key.start is None else self.get_t_idx(key.start)
            stop  = None if key.stop  is None else self.get_t_idx(key.stop)
            step  = None if key.step  is None else self.index_interval(key.step)

            return shim.asarray(self._data.tocsr()[slice(start, stop, step)].todense())
                # We convert to csr to allow slicing. This is cheap and keeps a
                # sparse representation
                # Converting to dense after the slice avoids allocating a huge
                # data matrix.
                # We call asarray because otherwise the result is a matrix, which behaves
                # differently (in particular, A[0,0].ndim = 2)

        else:
            raise ValueError("Key must be either an integer, float or a splice object.")

    #was :  def update(self, t, pop_idx, spike_arr):
    def update(self, tidx, neuron_idcs):
        '''Add to each neuron specified in `value` the spiketime `tidx`.
        Parameters
        ----------
        tidx: int, float
            The time index of the spike(s). This is converted
            to actual time and saved.
            Can optionally also be a float, in which case no conversion is made.
            Should not correspond to more than one bin ahead of _cur_tidx.
        neuron_idcs: iterable
            List of neuron indices that fired in this bin.
        '''
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        newidx = self.get_t_idx(tidx)
        if not shim.is_theano_variable(newidx):
            assert(newidx <= self._original_tidx.get_value() + 1)
        neuron_idcs = shim.asarray(neuron_idcs)

        onevect = shim.ones(neuron_idcs.shape)
            # vector of ones of the same length as the number of units which fired
        self._data.data = shim.concatenate((self._data.data, onevect))
        self._data.col = shim.concatenate((self._data.col, neuron_idcs))
        self._data.row = shim.concatenate((self._data.row, tidx*onevect))

        # Set the cur_idx. If tidx was less than the current index, then the latter
        # is *reduced*, since we no longer know whether later history is valid.
        self._cur_tidx = newidx
        if shim.is_theano_variable(self._original_tidx):
            shim.add_update(self._original_tidx, self._cur_tidx)
        else:
            self._original_tidx = self._cur_tidx

    def pad(self, before, after=0):
        '''Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.
        """
        Parameters
        ----------
        before: float | int
            Amount of time to add to before t0. If non-zero, all indices
            to this data will be invalidated.
        after: float | int (default 0)
            Amount of time to add after tn.
        '''
        before_len, after_len = self.pad_time(before, after)
        newshape = (len(self._tarr) + before_len + after_len, sum(self.shape))
        self._data.row += before_len
            # increment all time bins by the number that were added
        self._data = sp.sparse.coo_matrix((self._data.data, (self._data.row, self._data.col)),
                                          shape = newshape )

    def set(self, source=None, tslice=None):
        """Set the entire set of spiketimes in one go. `source` may be a list of arrays, a
        function, or even another History instance. It's useful for
        example if we've already computed history by some other means,
        or we specified it as a function (common for inputs).

        Accepted types for `source`: lists of iterables, functions (not implemented)
        These are converted to a list of spike times.

        If `tslice` is specified, only that portion of the data will be set.
        Values later than the slice are discarded (not implemented)

        If no source is specified, the series own update function is used,
        provided it has been previously defined. Can be used to force
        computation of the whole series.
        """
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        tarr = self._tarr

        if source is None:
            # Default is to use series' own compute functions
            self.compute_up_to('end')

        elif callable(source):
            raise NotImplementedError

        else:
            if tslice is None:
                tidxslice = slice(None)
                self._original_tidx.set_value(self.t0idx + len(self) - 1)
                self._cur_tidx = self._original_tidx
                assert(source.shape == self._data.shape)
            else:
                assert( tslice.step is None or tslice.step == 1)
                tidxslice = slice(self.get_t_idx(tslice.start), self.get_t_idx(tslice.stop))
                self._original_tidx.set_value( tidxslice.stop - 1 )
                self._cur_tidx = self._original_tidx
                if not shim.is_theano_object(tidxslice):
                    assert(source.shape == tuple(tidxslice.stop - tidxslice.start,
                                                 self._data.shape[1]))
            csc_data = self._data.tocsc()
            csc_data[tidxslice,:] = shim.sparse.csc_from_dense(source)
            if tidxslice.stop is not None:
                # Clear the invalidated data
                csc_data[tidxslice.stop:, :] = 0
                csc_data.eliminate_zeros()
            self._data = csc_data.tocoo()

        return self._data


    def _convolve_op_single_t(self, discretized_kernel, tidx, kernel_slice):
        '''Return the time convolution with the spike train, i.e.
            ∫ spiketimes(t - s) * kernel(s) ds
        with s ranging from -∞ to ∞  (normally there should be no spikes after t).

        Since spikes are delta functions, effectively what we are doing is
        sum( kernel(t-s) for s in spiketimes )

        Parameters
        ----------
        kernel: class instance or callable
            If a callable, should take time (float) as single input and return a float.
            If a class instance, should have a method `.eval` which satisfies the function requirement.
        tidx: index
            Time index at which to evaluate the convolution
        kernel_slice:

        Returns
        -------
        ndarray of shape 'npops' x 'npops'.
            It is indexed as result[from pop idx][to pop idx]

        '''


        # The setup of slicing is copied from Series._convolve_op_single_t
        if kernel_slice.start == kernel_slice.stop:
            return 0
        tidx = self.get_t_idx(tidx)

        kernel_slice = self.make_positive_slice(kernel_slice)
        # Algorithm assumes an increasing kernel_slice
        shim.check(kernel_slice.stop > kernel_slice.start)

        hist_start_idx = tidx - kernel_slice.stop - discretized_kernel.idx_shift
        hist_slice = slice(hist_start_idx, hist_start_idx + kernel_slice.stop - kernel_slice.start)
        try:
            shim.check(hist_slice.start >= 0)
        except AssertionError:
            raise AssertionError(
                "When trying to compute the convolution at {}, we calculated "
                "a starting point preceding the history's padding. Is it "
                "possible you specified time as an integer instead of a float ? "
                "Floats are treated as times, while integers are treated as time indices. "
                .format(tidx))
        hist_subarray = self._data.tocsc()[hist_slice]

        shim.check( discretized_kernel.ndim <= 2 )
        #shim.check( discretized_kernel.shape[0] == len(self.pop_sizes) )
        #shim.check( len(discretized_kernel.shape) == 2 and discretized_kernel.shape[0] == discretized_kernel.shape[1] )

        # To understand why the convolution is taken this way, consider
        # 1) that sparse arrays are matrices, so * is actually matrix multiplication
        #    (which is why we use the `multiply` method)
        # 2) `multiply` only returns a sparse array if the argument is also 2D
        # 3) that sparse arrays are always 2D, so A[0,0] is 2D, 1x1 matrix
        if discretized_kernel.ndim == 2 and discretized_kernel.shape[0] == discretized_kernel.shape[1]:
            # 2D discretized kernel: each population feeds into every other with a different kernel
            shim.check(discretized_kernel.shape[0] == len(self.pop_sizes))
            return shim.asarray(
                shim.sparse.hstack (
                    [ shim.sparse.vstack (
                        [ hist_subarray[:,self.pop_slices[from_pop_idx]].multiply(
                            discretized_kernel[kernel_slice][::-1, to_pop_idx, from_pop_idx:from_pop_idx+1] ).sum()
                          for from_pop_idx in range(len(self.pop_sizes)) ] )
                        for to_pop_idx in range(len(self.pop_sizes)) ] ).todense() )

        elif discretized_kernel.ndim == 1 and discretized_kernel.shape[0] == len(self.pop_sizes):
            # 1D discretized_kernel: populations only feed back into themselves
            return shim.asarray( shim.sparse.vstack(
                [ hist_subarray[:, self.pop_slices[from_pop_idx]].multiply(
                    discretized_kernel[kernel_slice][::-1, from_pop_idx:from_pop_idx+1] )
                  for from_pop_idx in range(len(self.pop_sizes)) ] ).todense() )

        else:
            raise NotImplementedError


    def convolve(self, kernel, t=slice(None, None), kernel_slice=slice(None,None),
                 *args, **kwargs):
        """Small wrapper around ConvolveMixin.convolve. Discretizes the kernel and converts the kernel_slice into a slice of time indices. Also converts t into a slice of indices, so the
        _convolve_op* methods can work with indices.
        """
        # This copied over from Series.convolve

        # Run the convolution on a discretized kernel
        # TODO: allow 'kernel' to be a plain function

        if not isinstance(kernel_slice, slice):
            raise ValueError("Kernel bounds must be specified as a slice.")

        discretized_kernel = self.discretize_kernel(kernel)
        sinn.add_sibling_input(self, discretized_kernel)
            # This adds the discretized_kernel as an input to any history
            # which already has `self` as an input. It's a compromise solution,
            # because these don't necessarily involve this convolution, but
            # the overeager association avoids the complicated problem of
            # tracking exactly which histories now need `discretized_kernel`
            # as an input.

        def get_start_idx(t):
            return 0 if t is None else discretized_kernel.get_t_idx(t)
        def get_stop_idx(t):
            return len(discretized_kernel._tarr) if t is None else discretized_kernel.get_t_idx(t)
        try:
            len(kernel_slice)
        except TypeError:
            kernel_slice = [kernel_slice]
            single_kernel = True
        else:
            single_kernel = False

        kernel_idx_slices = [ slice( get_start_idx(slc.start), get_stop_idx(slc.stop) )
                              for slc in kernel_slice ]
        tidx = self.get_t_idx(t)

        result = super().convolve(discretized_kernel,
                                  tidx, kernel_idx_slices, *args, **kwargs)
        if single_kernel:
            return result[0]
        else:
            return np.array(result)


class Series(ConvolveMixin, History):
    """
    Store history as a series, i.e. as an array of dimension T x (shape), where
    T is the number of bins and shape is this history's `shape` attribute.

    (DEACTIVATED) Also provides an "infinity bin" – .inf_bin — in which to store the value
    at t = -∞. (Not sure if this is useful after all.)
    """

    _strict_index_rounding = True

    def __init__(self, hist=sinn._NoValue, name=None, *args,
                 t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 shape=sinn._NoValue, **kwargs):
        """
        Initialize a Series instance, derived from History.

        Parameters
        ----------
        *args, **kwargs:
            Arguments required by History and ConvolveMixin
        """
        if name is None:
            name = "series{}".format(self.instance_counter + 1)
        # if shape is None:
        #     raise ValueError("'shape' is a required keyword "
        #                      "for Series intializer.")
        # if 'convolve_shape' in kwargs:
        #     assert(kwargs['convolve_shape'] == shape)
        # else:
        #     kwargs['convolve_shape'] = shape*2

        super().__init__(hist, name, *args,
                         t0=t0, tn=tn, dt=dt,
                         shape=shape, **kwargs)

        if self.use_theano:
            # Make the dimensions where shape is 1 broadcastable
            # (as they would be with NumPy)
            data_tensor_broadcast = tuple(
                [False] + [True if d==1 else 0 for d in self.shape] )
            self.DataType = shim.getT().TensorType(sinn.config.floatX,
                                                   data_tensor_broadcast)
            #self._data = shim.T.zeros(self._tarr.shape + self.shape, dtype=config.floatX)
            self._data = self.DataType(self.name + ' data')
            #self.inf_bin = shim.lib.zeros(self.shape, dtype=config.floatX)
        else:
            self._data = shim.shared(np.zeros(self._tarr.shape + self.shape, dtype=config.floatX),
                                     name = self.name + " data",
                                     borrow = True)

        self._original_data = self._data
            # Stores a handle to the original data variable, which will appear
            # as an input in the Theano graph

    def retrieve(self, key):
        '''A function taking either an index or a splice and returning
        respectively the time point or an interval from the
        precalculated history. It does not check whether history has
        been calculated sufficiently far.

        '''
        assert(not isinstance(key, float))
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
            The timeslice to store. May be a tuple, in which case it has
            the form `(value, updates)`, where `updates` is a Theano
            update dictionary.
        '''
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        # assert(not shim.is_theano_object(tidx))
        #     # time indices must not be variables <-- Should be safe now

        # Check if value includes an update dictionary.
        if isinstance(value, tuple):
            # `value` is a theano.scan-style return tuple
            assert(len(value) == 2)
            updates = value[1]
            assert(isinstance(updates, dict))
            value = value[0]
        else:
            updates = None

        # Adaptations depending on whether tidx is a single bin or a slice
        if shim.istype(tidx, 'int'):
            end = tidx
            if not shim.is_theano_object(tidx):
                assert(tidx <= self._original_tidx.get_value() + 1)
                # Ensure that we update at most one step in the future
        else:
            assert(isinstance(tidx, slice))
            assert(shim.istype(tidx.start, 'int') and shim.istype(tidx.stop, 'int'))
            shim.check(tidx.stop > tidx.start)
            end = tidx.stop - 1
            if not shim.is_theano_object(tidx):
                assert(tidx.start <= self._original_tidx.get_value() + 1)
                    # Ensure that we update at most one step in the future

        end = end
        if shim.is_theano_object(self._data):
            if not shim.is_theano_object(value):
                logger.warning("Updating a Theano array ({}) with a Python value. "
                               "This is likely an error.".format(self.name))
            #if self._original_data is not None or shim.config.theano_updates != {}:
            # if ( self._original_data is not None
            #      and self._original_data in shim.config.theano_updates):
            #     raise RuntimeError("You can only update data once within a "
            #                        "Theano computational graph. If you need "
            #                        "multiple updates, compile a single "
            #                        "update as a function, and call that "
            #                        "function repeatedly.")
            # assert(shim.is_theano_variable(self._data))
            #     # This should be guaranteed by self.use_theano=True
            tmpdata = self._data
            if self._original_data is None:
                self._original_data = self._data
                    # Persistently store the current _data, because that's the handle
                    # to the input that will be used when compiling the function
            self._data = shim.set_subtensor(tmpdata[tidx], value)
            if updates is not None:
                shim.add_updates(updates)

            self._cur_tidx = shim.largest(self._cur_tidx, end)

            if shim.is_theano_object(self._original_data):
                shim.add_update(self._original_data, self._data)
            else:
                self._original_data = self._data
            if shim.is_theano_object(self._original_tidx):
                shim.add_update(self._original_tidx, self._cur_tidx)
            else:
                self._original_tidx = self._cur_tidx

        else:
            if shim.is_theano_object(value):
                raise ValueError("You are trying to update a pure numpy series ({}) "
                                 "with a Theano variable. You need to make the "
                                 "series a Theano variable as well."
                                 .format(self.name))
            if shim.is_theano_object(tidx):
                raise ValueError("You are trying to update a pure numpy series ({}) "
                                 "with a time idx that is a Theano variable. You need "
                                 "to make the series a Theano variable as well."
                                 .format(self.name))

            dataobject = self._data.get_value(borrow=True)

            if updates is not None:
                raise RuntimeError("For normal Python and NumPy functions, update "
                                   "variables in place rather than using an update dictionary.")
            if dataobject[tidx].shape != value.shape:
                raise ValueError("Series '{}': The shape of the update value - {} - does not match "
                                 "the shape of a timeslice(s) - {} -."
                                 .format(self.name, value.shape, dataobject[tidx].shape))

            dataobject[tidx] = value
            self._data.set_value(dataobject, borrow=True)

            #self._data = shim.set_subtensor(self._data[tidx], value)
            #self._data[tidx] = value

            self._original_tidx.set_value(end)
            self._cur_tidx = self._original_tidx
                # If we updated in the past, this will reduce _cur_tidx
                # – which is what we want

    def pad(self, before, after=0, **kwargs):
        '''Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are increased to make them exact multiples of dt.
        See `History.pad_time` for more details.
        """
        Parameters
        ----------
        before: float | int
            Amount of time to add to before t0. If non-zero, All indices
            to this data will be invalidated.
        after: float (default 0) | int
            Amount of time to add after tn.
        **kwargs:
            Extra keyword arguments are forwarded to `numpy.pad`.
            They may be used to specify how to fill the added time slices.
            Default is to fill with zeros.
        '''

        previous_tarr_shape = self._tarr.shape
        before_len, after_len = self.pad_time(before, after)

        if not kwargs:
            # No keyword arguments specified – use defaults
            kwargs['mode'] = 'constant'
            kwargs['constant_values'] = 0

        pad_width = ( [(before_len, after_len)]
                      + [(0, 0) for i in range(len(self.shape))] )

        if self.use_theano:
            if self._original_data is None:
                self._original_data = self._data
                    # Persistently store the current _data, because that's the handle
                    # to the input that will be used when compiling the function
            self._data = shim.pad(self._data, previous_tarr_shape + self.shape,
                                  pad_width, **kwargs)
        else:
            self._data.set_value( shim.pad(self._data, previous_tarr_shape + self.shape,
                                           pad_width, **kwargs) )


    # def generator(self, inputs):
    #     """
    #     Return a function which, given the inputs, returns a pure numpy Series
    #     object equivalent to this one in the current state (i.e. at the current tidx).
    #     This function only makes sense on a Series where the data is a Theano variable.
    #     (For non-Theano data, the returned generator just returns `self`.)
    #     """
    #     if not shim.is_theano_object(self._data):
    #         def gen(self, input_vals):
    #             assert(len(inputs) == len(input_vals))
    #             assert(x == y for x,y in zip(inputs, input_vals))
    #             return self

    #     else:
    #         f = shim.theano.function(inputs, self._data)
    #         def gen(self, input_vals):
    #             new_series = Series(self)
    #             new_series.set(f(input_vals))
    #             return new_series

    #     return gen


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

    def zero(self, mode='all'):
        """Zero out the series. Unless mode='all', the initial data point will NOT be zeroed"""
        if mode == 'all':
            mode_slice = slice(None, None)
        else:
            mode_slice = slice(1, None)

        if self.use_theano:
            if mode == 'all':
                self._data = shim.getT().zeros(self._data.shape)
            else:
                if self._original_data is None:
                    self._original_data = self._data
                      # Persistently store the current _data: it's the handle
                      # to the input that will be used when compiling the function
                self._data = shim.set_subtensor(self._data[mode_slice],
                                                shim.geT().zeros(self._data.shape[mode_slice]))
        else:
            new_data = self._data.get_value(borrow=True)
            new_data[mode_slice] = np.zeros(self._data.get_value(borrow=True).shape[mode_slice])
            self._data.set_value(new_data, borrow=True)

        self.clear()

    def get_trace(self, component=None, include_padding='none'):
        """
        Return the series' computed data for the given component.
        Time points which have not yet been computed are excluded, such that
        the len(series.get_trace(*)) may be smaller than len(series). The
        return value is however guaranteed to be consistent with get_time_array().
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
        if not shim.isshared(self._cur_tidx):
            raise RuntimeError("You are in the midst of constructing a Theano graph. "
                               "Reset history {} before trying to obtain its time array."
                               .format(self.name))

        padding_vals = [ 'none', 'before', 'left', 'after', 'right', 'all', 'both' ]
        if include_padding in ['none', 'after', 'right']:
            start = self.t0idx
        elif include_padding in padding_vals:
            # It's one of the other options
            start = 0
        else:
            raise ValueError("include_padding should be one of {}.".format(padding_vals))

        if include_padding in ['none', 'before', 'left']:
            stop = self._cur_tidx.get_value() + 1
        elif include_padding in padding_vals:
            stop = min(self._cur_tidx.get_value() + 1, len(self._tarr))
        else:
            raise ValueError("include_padding should be one of {}.".format(padding_vals))

        assert(self._cur_tidx.get_value() >= stop - 1)
        if component is None:
            #return self[start:stop]
            return self._data.get_value()[start:stop]
        elif shim.istype(component, 'int'):
            #return self[start:stop, component]
            return self._data.get_value()[start:stop, component]
        elif len(component) == 1:
            #return self[start:stop, component[0]]
            return self._data.get_value()[start:stop, component]
        elif len(component) == 2:
            #return self[start:stop, component[0], component[1]]
            return self._data.get_value()[start:stop, component]
        else:
            raise NotImplementedError("Really, you used more than 2 data dimensions in a series array ? Ok, well let me know and I'll implement that.")

    def set(self, source=None):
        """Set the entire series in one go. `source` may be an array, a
        function, or even another History instance. It's useful for
        example if we've already computed history by some other means,
        or we specified it as a function (common for inputs).

        Accepted types for `source`: functions, arrays, single values.
        These are all converted into a time-series with the same time
        bins as the history.

        If source has the attribute `shape`, than it is checked to be the same
        as this history's `shape`

        Note that this sets the whole data array, including padding. So if
        source is an array, it should match the padded length exactly.

        If no source is specified, the series own update function is used,
        provided it has been previously defined. Can be used to force
        computation of the whole series.
        """
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        data = None

        tarr = self._tarr

        if source is None:
            # Default is to use series' own compute functions
            self.compute_up_to('end')

        elif isinstance(source, History):
            raise NotImplemented

        elif (not hasattr(source, 'shape')
              and (shim.istype(source, 'float')
                   or shim.istype(source, 'int'))):
            # Constant input
            data = np.ones(tarr.shape + self.shape) * source

        else:
            if hasattr(source, 'shape'):
                # Input specified as an array
                if source.shape != tarr.shape + self.shape:
                    raise ValueError("The given external input series does not match the dimensions of the history")
                data = source

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
            shim.check(data.shape == self._data.get_value(borrow=True).shape)
            shim.check(data.shape[0] == len(tarr))

            self._data.set_value(data, borrow=True)

        self._original_tidx.set_value(self.t0idx + len(tarr) - 1)
        self._cur_tidx = self._original_tidx

        return self._data

    def convolve(self, kernel, t=slice(None, None), kernel_slice=slice(None,None),
                 *args, **kwargs):
        """Small wrapper around ConvolveMixin.convolve. Discretizes the kernel and converts the kernel_slice into a slice of time indices. Also converts t into a slice of indices, so the
        _convolve_op* methods can work with indices.
        """
        # Run the convolution on a discretized kernel
        # TODO: allow 'kernel' to be a plain function

        if not isinstance(kernel_slice, slice):
            raise ValueError("Kernel bounds must be specified as a slice.")

        discretized_kernel = self.discretize_kernel(kernel)
        sinn.add_sibling_input(self, discretized_kernel)
            # This adds the discretized_kernel as an input to any history
            # which already has `self` as an input. It's a compromise solution,
            # because these don't necessarily involve this convolution, but
            # the overeager association avoids the complicated problem of
            # tracking exactly which histories now need `discretized_kernel`
            # as an input.

        def get_start_idx(t):
            return 0 if t is None else discretized_kernel.get_t_idx(t)
        def get_stop_idx(t):
            return len(discretized_kernel._tarr) if t is None else discretized_kernel.get_t_idx(t)
        try:
            len(kernel_slice)
        except TypeError:
            kernel_slice = [kernel_slice]
            single_kernel = True
        else:
            single_kernel = False

        kernel_idx_slices = [ slice( get_start_idx(slc.start), get_stop_idx(slc.stop) )
                              for slc in kernel_slice ]
        tidx = self.get_t_idx(t)

        result = super().convolve(discretized_kernel,
                                  tidx, kernel_idx_slices, *args, **kwargs)
        if single_kernel:
            return result[0]
        else:
            return np.array(result)

    def _convolve_op_single_t(self, discretized_kernel, tidx, kernel_slice):
        # When indexing data, make sure to use self[…] rather than self._data[…],
        # to trigger calculations if necessary

        if kernel_slice.start == kernel_slice.stop:
            return 0
        else:
            kernel_slice = self.make_positive_slice(kernel_slice)
            # Algorithm assumes an increasing kernel_slice
            shim.check(kernel_slice.stop > kernel_slice.start)

            hist_start_idx = tidx - kernel_slice.stop - discretized_kernel.idx_shift
            hist_slice = slice(hist_start_idx, hist_start_idx + kernel_slice.stop - kernel_slice.start)
            try:
                shim.check(hist_slice.start >= 0)
            except AssertionError:
                raise AssertionError(
                    "When trying to compute the convolution at {}, we calculated "
                    "a starting point preceding the history's padding. Is it "
                    "possible you specified time as an integer instead of a float ? "
                    "Floats are treated as times, while integers are treated as time indices. "
                    .format(tidx))
            dim_diff = discretized_kernel.ndim - self.ndim
            return self.dt * shim.sum(discretized_kernel[kernel_slice][::-1]
                                      * shim.add_axes(self[hist_slice], dim_diff, -self.ndim),
                                      axis=0)
                # history needs to be augmented by a dimension to match the kernel
                # Since kernels are [to idx][from idx], the augmentation has to be on
                # the second-last axis for broadcasting to be correct.
                # TODO: Untested with multi-dim timeslices (i.e. self.ndim > 1)

    def _convolve_op_batch(self, discretized_kernel, kernel_slice):
        """Return the convolution at every lag within t0 and tn."""
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
            try:
                # Check that there is enough padding before t0
                shim.check(domain_slice.start >= 0)
            except AssertionError:
                raise AssertionError(
                    "When trying to compute the convolution at {}, we calculated "
                    "a starting point preceding the history's padding. Is it "
                    "possible you specified time as an integer rather than scalar ?")
            dis_kernel_shape = (kernel_slice.stop - kernel_slice.start,) + discretized_kernel.shape
            retval = self.dt * shim.conv1d(self[:], discretized_kernel[kernel_slice], len(self._tarr), dis_kernel_shape)[domain_slice]
            shim.check(shim.eq(retval.shape[0], len(self)))
                # Check that there is enough padding after tn
            return retval


    #####################################################
    # Operator definitions
    #####################################################
    # TODO: Operations with two Histories (right now I'm assuming scalars or arrays)

    def _apply_op(self, op, b=None):
        new_series = Series(self)
        if b is None:
            new_series.set_update_function(lambda t: op(self[t]))
            new_series.set_range_update_function(lambda tarr: op(self[self.time_array_to_slice(tarr)]))
            new_series.add_input(self)
        else:
            new_series.set_update_function(lambda t: op(self[t], b))
            new_series.set_range_update_function(lambda tarr: op(self[self.time_array_to_slice(tarr)], b))
            new_series.add_input([self, b])
        return new_series

    def __abs__(self):
        return self._apply_op(operator.abs)
    def __add__(self, other):
        return self._apply_op(operator.add, other)
    def __radd__(self, other):
        return self._apply_op(lambda a,b: b+a, other)
    def __sub__(self, other):
        return self._apply_op(operator.sub, other)
    def __rsub__(self, other):
        return self._apply_op(lambda a,b: b-a, other)
    def __mul__(self, other):
        return self._apply_op(operator.mul, other)
    def __rmul__(self, other):
        return self._apply_op(lambda a,b: b*a, other)
    def __matmul__(self, other):
        return self._apply_op(operator.matmul, other)
    def __rmatmul__(self, other):
        return self._apply_op(lambda a,b: operator.matmul(b,a), other)
            # Using operator.matmul rather than @ prevents import fails on Python <3.5
    def __truediv__(self, other):
        return self._apply_op(operator.truediv, other)
    def __rtruediv__(self, other):
        return self._apply_op(lambda a,b: b/a, other)
    def __floordiv__(self, other):
        return self._apply_op(operator.floordiv, other)
    def __rfloordiv__(self, other):
        return self._apply_op(lambda a,b: b//a, other)
    def __mod__(self, other):
        return self._apply_op(operator.mod, other)

