# -*- coding: utf-8 -*-

"""
Created on Mon Jan 16 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
import collections
from collections import deque, Iterable
from copy import deepcopy
import itertools
import operator
import logging
logger = logging.getLogger("sinn.history")

import theano_shim as shim
import theano_shim.sparse
import sinn
from sinn.common import HistoryBase, PopulationHistoryBase, KernelBase
import sinn.config as config
import sinn.mixins as mixins
from sinn.mixins import ConvolveMixin
import sinn.popterm as popterm


###############
###   Types are registered at end of module
###############

########################
# Exceptions
class LockedHistoryError(RuntimeError):
    pass
########################
# TODO: Remove everything related to compilation – this is better done with Model
#       A lot of compilation logic can be found by search 'self.symbolic', '_compiling'
# TODO: Replace tn attribute by a @property.
# FIXME: Currently `discretize_kernel()` CANNOT be used with Theano – it does
#        not preserve the computational graph

class History(HistoryBase):
    """
    Generic class for storing a history, serving as a basis for specific history classes.
    On its own it lacks some basic fonctionality, like the ability to retrieve
    data at a certain time point.

    Ensures that if the history at time t is known, than the history at all times
    previous to t is also known (by forcing a computation if neces#sary).

    Derived classes can safely expect the following attributes to be defined:
        + name             : str. Unique identifying string
        + shape            : int tuple. Shape at a single time point, or number of elements in the system
        + t0               : floatX. Time at which history starts
        + tn               : floatX. Time at which history ends
        + dt               : floatX. Timestep size
        + dt64             : float64. Timestep size; for some index calculations
                             double precision is sometimes required.
        + idx_dtype        : numpy integer dtype. Type to use for indices within one time slice.
        + tidx_dtype        : numpy integer dtype. Type to use for time indices.
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
                             NOTE: The compilation mechanism is expected to change
                             in the future.
    The following methods are also guaranteed to be defined:
        + compile          : If the update function is a Theano graph, compile it
                             and attach the new history as `compiled_history`.
                             NOTE: The compilation mechanism is expected to change
                             in the future.
        + lock             : Set the locked status
        + get_time_array   : Return time array.
        + time             : (property) Unpadded time array. Calls
                             self.get_time_array() with default arguments.
        + trace            : (property) Unpadded data. Calls self.get_trace()
                             with default arguments.
        + __getitem__      : Calls `self.retrieve()`
        + __setitem__      : Calls `self.update()`, after converting times in a
                             key to time indices, and `None` in a slice to the
                             appropriate time index. Thus `update()` only needs
                             to implement operations on integer time indices.
    A History may also define
        + _compute_range   : Function taking an array of consecutive times and returning
                             an array-like object of the history at those times.
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
        tidx: int | slice | 1D array
            The time index at which to store the value, or an array of such
            time indices. No index should not correspond to more than one
            bin ahead of _cur_tidx.
        value: timeslice
            The timeslice to store. Format is that same as that returned by
            self._update_function
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

    def get_trace(self, **kwargs):
        '''Return unpadded data.'''
        All arguments must be optional. This function is meant for data
        analysis and plotting, so the return value must not be symbolic.
        Typically this means that `get_value()` should be called on `_data`.

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

    def __init__(self, hist=sinn._NoValue, name=None, *args,
                 time_array=sinn._NoValue,  shape=sinn._NoValue,
                 iterative=sinn._NoValue, symbolic=sinn._NoValue,
                 t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 f=sinn._NoValue):
        """
        Initialize History object.
        Instead of passing the parameters, another History instance may be passed as
        first argument. In this case, time_array and shape are taken
        from that instance. These can be overridden by passing also a corresponding keyword
        argument.
        Except for a possible history instance, all parameters should be passed as keywords.
        Note that `iterative` always defaults to True, even when a template history

        Parameters
        ----------
        hist: History instance
            Optional. If passed, will used this history's parameters as defaults.
        time_array: ndarray (float)
            The array of times this history samples. If provided, `t0`, `tn` and `dt` are ignored.
            Note that times should always be provided as 64 bit floats; the array is internally
            convert to floatX, but a full precision version is also kept for some calculations such
            as `index_interval()`.
        t0: float
            Time at which the history starts. /Deprecated: use `time_array`./
        tn: float
            Time at which the history ends. /Deprecated: use `time_array`./
        dt: float
            Timestep. /Deprecated: use `time_array`./
        shape: int tuple
            Shape of a history slice at a single point in time.
            E.g. a movie history might store NxN frames in an TxNxN array.
            (N,N) would be the shape, and T would be (tn-t0)/dt.
        f:  (Deprecated - use set_update_function) function (t) -> shape
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
        symbolic: bool
            A value of `False` indicates that even if a symbolic library is
            loaded, values in this history are treated as data. Only updates
            which do not have any symbolic inputs are permitted, and they are
            immediately calculated using `shim.graph.eval()`.
            IMPORTANT NOTE: This means that any symbolic dependency (e.g. on
            shared parameters) is lost.

        Returns
        -------
        None

        ..Note
        The data type chosen for time indices depends on the length of the time array.
        It is chosen such that it can contain values at least up to twice the length
        of `time_array`; for normal usage this should be plenty, since requiring more
        than this would mean that we have a history with more padding than actual time.
        """
        # The first thing we do is increment the instance counter. This allows us to
        # ensure every class has a unique name.
        History.instance_counter += 1

        # NOTE: We save both 'dt' and 'dt64' variables. 'dt' is of same type as floatX,
        # while 'dt64' is always 64 bit. 'dt' is meant for single step calculations, such
        # as appear in integrators. It should almost always suffice for external usage, and
        # will prevent unwanted upconversions to float64.
        # 'dt64' is meant for specific internal functions that span multiple steps and require
        # the higher precision, such as determining the number of bins in an time interval.

        # if self not in sinn.inputs:
        #    sinn.inputs[self] = set()

        if name is None:
            name = "history{}".format(self.instance_counter)
        # Make sure numerical parameters have members dtype, etc.
        if t0 is not sinn._NoValue: t0 = np.asarray(t0)
        if tn is not sinn._NoValue: tn = np.asarray(tn)
        if dt is not sinn._NoValue: dt = np.asarray(dt)
        # Grab defaults from the other instance:
        if hist is not sinn._NoValue:
            if not isinstance(hist, History):
                raise ValueError("The first parameter to the History initializer, "
                                 "if given, must be another history instance. "
                                 "All other parameters should use keywords.")
            if all(arg is sinn._NoValue for arg in (time_array, t0, tn, dt)):
                # Only set the time_array to the reference history's if no time argument is passed
                time_array = hist._tarr[hist.t0idx : hist.t0idx+len(hist)]
            elif time_array is sinn._NoValue:
                # `time_array` was not passed, so we fall back down to t0, tn and dt.
                # Take the unspecified values from the reference history.
                if t0 is sinn._NoValue:
                    t0 = hist.t0
                if tn is sinn._NoValue:
                    tn = hist.tn
                if dt is sinn._NoValue:
                    dt = hist.dt
            if shape is sinn._NoValue:
                shape = hist.shape
            #if f is sinn._NoValue:
            #    f = hist._update_function
            # if symbolic is sinn._NoValue:
            #     symbolic = hist.symbolic
            # 'hist' does not serve as a default value for 'iterative', since the new history will typically have a different function
            # Same goes for 'symbolic'
        else:
            assert(time_array is not sinn._NoValue
                   or sinn._NoValue not in [t0, tn, dt])
            assert(shape is not sinn._NoValue)
        if iterative is sinn._NoValue:
            # Default value
            iterative = True
        if time_array is not sinn._NoValue:
            # Set time dtype to "smallest" of floatX, time_array.dtype
            if np.can_cast(time_array.dtype, shim.config.floatX):
                t_dtype = time_array.dtype
            else:
                t_dtype = np.dtype(shim.config.floatX)
            time_array = time_array.astype(t_dtype)
            t0 = time_array[0]
            tn = time_array[-1]
            dt64 = sinn.upcast(tn.astype('float64') - t0.astype('float64'),
                               to_dtype=np.float64,
                               from_dtype=np.float32) / (len(time_array)-1)
                 # Esp. if time_array is float32, this should be more precise than time_array[1] - time_array[0]
            dt = dt64.astype(t_dtype)
            assert( np.all(sinn.isclose(time_array[1:] - time_array[:-1], dt)) )
        else:
            t_dtype = np.result_type(t0, tn, np.float32)
                # Don't consider dt – dt should always be passed as double
                # even when we want time dtype to be 32-bit
            if np.can_cast(shim.config.floatX, t_dtype):
                t_dtype = np.dtype(shim.config.floatX)
            # Ensure (tn-t0) is a round multiple of dt
            # t0 = shim.cast(t0, t_dtype)
            # tn = np.ceil( (tn - t0)/dt ) * dt + t0
            # Get time step
            n_steps = np.rint((tn - t0)/dt).astype(np.int)
            if np.can_cast(np.float64, dt.dtype):
                dt64 = dt
            else:
                # Compute dt64 with t0 and tn, which is more precise than
                # upconverting dt
                dt64 = (tn.astype(np.float64) - t0.astype(np.float64))/n_steps
            # time_array = np.arange(t0,
            #                        tn + dt64 - config.abs_tolerance,
            #                        dt64,
            #                        dtype=t_dtype)
            time_array = (np.arange(0, n_steps+1) * dt64).astype(t_dtype)
                # 'self.tn+self.dt' ensures the upper bound is inclusive,
                # -config.abs_tolerance avoids including an extra bin because of rounding errors
            t0 = time_array[0]    # Remove any risk of mismatch due to rounding
            tn = time_array[-1]   # and ensure all dtypes match

        # Deprecated ?
        # Determine whether the series data will use Theano.
        # The default depends on whether Theano is loaded. If it is not loaded
        # and we try to use it for this history, an error is raised.
        if symbolic is sinn._NoValue:
            self.symbolic = shim.config.use_theano
        else:
            self.symbolic = symbolic

        super().__init__(t0, tn)
            # Set t0 and tn

        ############
        # Flags
        self._iterative = iterative
        self._compiling = False
        self.locked = False
            # When this is True, the history is not allowed to be changed
        ############

        self.name = name
        self.shape = tuple(shape)
            # shape at a single time point, or number of elements
            # in the system
            # We call tuple because the shape parameter could be
            # passed e.g. as an ndarray
        self.ndim = len(shape)

        self.dt = shim.cast(dt64, t_dtype)
        self.dt64 = dt64
        self.idx_dtype = np.min_scalar_type(max(self.shape))
        self.tidx_dtype = np.min_scalar_type(-2*len(time_array))
            # Leave enough space in time indices to double the time array
            # Using a negative value forces the type to be 'int' and not 'uint',
            # which we need to store -1
        #self._cur_tidx = -1
        if self.symbolic and not shim.config.use_theano:
            raise ValueError("You are attempting to construct a series with Theano "
                             "but it is not loaded. Run `shim.config.load_theano()` "
                             "before constructing this history.")
        elif self.symbolic:
            self.compiled_history = None
        self._cur_tidx = shim.shared(np.array(-1, dtype=self.tidx_dtype),
                                     name = 't idx (' + name + ')',
                                     symbolic = self.symbolic)
            # Tracker for the latest time bin for which we
            # know history.
        self._original_tidx = self._cur_tidx

        if f is sinn._NoValue:
            # TODO Find way to assign useful error message to a missing f, that
            #      works for histories derived from others (i.e. is recognized as the
            #      "missing function" function. Maybe a global function or class,
            #      with which we use isinstance ?
            #      => If we deprecate this, can just use a flag
            # Set a default function that will raise an error when called
            def nofunc(*arg):
                raise RuntimeError("The update function for history {} is not set."
                                   .format(self.name))
            self._update_function = nofunc
        else:
            self.set_update_function(f)
        self._compute_range = None

        self._tarr = time_array.astype(t_dtype)
        self.t0idx = shim.cast(0, self._cur_tidx.dtype)       # the index associated to t0
        self.tn = tn
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

    def __reduce__(self):
        # Allows pickling
        return (self.from_repr_np, (self.repr_np,))

    @property
    def dtype(self):
        """Data type of the data."""
        return self._data.dtype

    @property
    def tnidx(self):
        return self.t0idx + len(self) - 1

    @property
    def repr_np(self):
        return self.raw()

    @property
    def padding(self):
        """Returns a tuple with the left and right padding."""
        return (self.t0idx, len(self._tarr) - self.tnidx - 1)

    @property
    def cur_tidx(self):
        """Returns the time index up to which the history has been computed.
        Returned index is not corrected for padding; to get the number of bins
        computed beyond t0, do `hist.cur_tidx - hist.t0idx`.
        """
        return self._original_tidx.get_value()

    @classmethod
    def from_repr_np(cls, repr_np):
        return cls.from_raw(repr_np)

    def raw(self, **kwargs):
        # The raw format is meant for data longevity, and so should
        # seldom, be changed
        # Versions:
        #   1 original
        #   2 (31 mai 2018)
        #     + dt64
        """

        Parameters
        ----------
        **kwargs:
            Passed keywords will be added to the raw structure, or replace
            an existing keyword if it is already included.
            This argument exists to allow specialization in derived classes.
            In particular, if a specialized class does not provide a particular
            attribute (e.g. '_original_data'), a keyword can be used to prevent
            `raw()` from trying to retrieve that attribute and triggering an
            error.
        """

        # if self.symbolic:
        #     if self.compiled_history is not None:
        #         raw = self.compiled_history.raw()
        #     else:
        #         raise AttributeError("The `raw` method for uncompiled Theano "
        #                              "histories is undefined.")
        #     raw['name'] = self.name # Replace with non-compiled name
        # else:

        if ((hasattr(self, '_original_data')
             and np.all(self._data != self._original_data))
            or (hasattr(self, '_original_tidx')
                and self._cur_tidx != self._original_tidx)):
            logger.warning("Saving symbolic history '{}'; only the data "
                           "(i.e. what has already been computed) is saved. "
                           "Symbolic state will be discarded.")
        # We do it this way in case kwd substitutions are there to
        # avoid an error (such as _data not having a .get_value() method)
        raw = {}
        raw['version'] = 2
        raw['type'] = sinn.common.find_registered_typename(type(self))
             # find_registered_typename returns the closest registered type name in the hierarchy
             # E.g. if we are saving a subclass of Series, this will return the name under which
             # that class was registered, and if it wasn't registered, the name under which
             # 'Series' is registered. This ensures that ml.iotools.load() is always able to
             # reconstruct the series afterwards.
        raw['name'] = kwargs.pop('name') if 'name' in kwargs else self.name
        raw['t0'] = kwargs.pop('t0') if 't0' in kwargs else self.t0
        raw['tn'] = kwargs.pop('tn') if 'tn' in kwargs else self.tn
        raw['dt'] = kwargs.pop('dt') if 'dt' in kwargs else self.dt
        raw['dt64'] = kwargs.pop('dt') if 'dt' in kwargs else self.dt
        raw['t0idx'] = kwargs.pop('t0idx') if 't0idx' in kwargs else self.t0idx
        raw['_unpadded_length'] = kwargs.pop('_unpadded_length') if '_unpadded_length' in kwargs else self._unpadded_length
        raw['_cur_tidx'] = (kwargs.pop('_cur_tidx') if '_cur_tidx' in kwargs
                            else kwargs.pop('_original_tidx') if '_original_tidx' in kwargs
                            else self._original_tidx.get_value())
        raw['shape'] = kwargs.pop('shape') if 'shape' in kwargs else self.shape
        raw['ndim'] = kwargs.pop('ndim') if 'ndim' in kwargs else self.ndim
        raw['_tarr'] = kwargs.pop('_tarr') if '_tarr' in kwargs else self._tarr
        raw['_data'] = (kwargs.pop('_data') if '_data' in kwargs
                        else kwargs.pop('_original_data') if '_original_data' in kwargs
                        else self._original_data.get_value() if hasattr(self, '_original_data')
                        else self._data.get_value() if hasattr(self._data, 'get_value')
                        else self._data)
            # Pure NumPy histories don't need '_original_data'
        raw['_iterative'] = kwargs.pop('_iterative') if '_iterative' in kwargs else self._iterative
        raw['locked'] = kwargs.pop('locked') if 'locked' in kwargs else self.locked

        raw.update(kwargs)
        # If we write to NpzFile, all entries are converted to arrays;
        # we just do it preemptively, which ensures consistent unpacking whether
        # we saved to NpzFile or not.
        return {key: np.array(value) for key, value in raw.items()}

    @classmethod
    def from_raw(cls, raw, update_function=sinn._NoValue, symbolic=False, lock=True, **kwds):
        """

        Parameters
        ----------
        symbolic: bool
            (Deprecated) If True, a second Theano history will be constructed, and the loaded
            one attached as its `compiled_history` attribute.
            If unspecified, the behaviour is the same as for the History initializer.
        **kwds:
            Will be passed along to the class constructor
            Exists to allow specialization in derived classes.
        """
        if not isinstance(raw, (dict, np.lib.npyio.NpzFile)):
            raise TypeError("'raw' data must be either a dict or a Numpy archive.")
        version = raw['version'] if 'version' in raw else 1
        dt = raw['dt']
        dt64 = raw['dt64'] if version >= 2 else raw['dt']
        t0 = raw['t0'].astype(dt.dtype)
        tn = raw['tn'].astype(dt.dtype)
        hist =  cls(name = str(raw['name']),
                    t0 = t0, tn = tn, dt = dt64,
                    shape = tuple(raw['shape']),
                    f = update_function,
                    iterative = bool(raw['_iterative']),
                    symbolic = False,
                    **kwds)
        # Change dtypes because History.__init__ always initializes to floatX
        # TODO: Initialize hist w/ _tarr instead of t0, tn, dt
        hist.dt = hist.dt.astype(dt.dtype)
        hist.t0 = hist.t0.astype(dt.dtype)
        hist.tn = hist.tn.astype(dt.dtype)
        hist._tarr = raw['_tarr'].astype(dt.dtype)
        hist.t0idx = raw['t0idx'].astype(hist.tidx_dtype)
        hist._unpadded_length = raw['_unpadded_length'].astype(hist.tidx_dtype)
        hist.locked = bool(raw['locked'])
            # Could probably be removed, since we set a lock status later, but
            # ensures the history is first loaded in the same state

        # Decide whether to wrap the history in another Theano history (Deprecated)
        if symbolic is sinn._NoValue:
            symbolic = shim.config.use_theano
        if symbolic:
            hist.name = str(raw['name']) + " (compiled)"
            theano_hist = cls(hist, name=str(raw['name']), symbolic=True)
            theano_hist.compiled_history = hist
            rethist = theano_hist
        else:
            hist.name = str(raw['name'])
            rethist = hist
        rethist._original_tidx = shim.shared( raw['_cur_tidx'].astype(hist.tidx_dtype),
                                              name = 't idx (' + rethist.name + ')' ,
                                              symbolic=symbolic)
        rethist._cur_tidx = rethist._original_tidx
        rethist._original_data = shim.shared(raw['_data'], name = rethist.name + " data")
        rethist._data = rethist._original_data
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
        (This effect will be removed in the future.)

        NOTE: key will not be shifted to reflect history padding. So `key = 0`
        may well refer to a time *before* t0.

        Parameters
        ----------
        key: int, float, slice or array
            If an array, it must be consecutive (this is not checked).
        """

        neg_key_err = ("Negative keys are not supported because they are a "
                       "frequent source of confusing bugs. Instead of `-5`, use "
                       "`[history].tnidx - 5`.")
        if shim.isscalar(key):
            if (not shim.is_theano_object(key) and shim.istype(key, 'int')
                and key < 0):
                raise ValueError(neg_key_err)
            return self._getitem_internal(key)
        elif isinstance(key, slice):
            for t in (key.start, key.stop):
                if (t is not None and not shim.is_theano_object(t)
                    and shim.istype(t, 'int') and t < 0):
                    raise ValueError(neg_key_err)
            return self._getitem_internal(key)
        elif shim.isarray(key):
            # FIXME Empty arrays still go through the 'then' branch somehow
            if not shim.is_theano_object(key) and np.any(key < 0):
                raise ValueError(neg_key_err)
            return shim.ifelse(shim.eq(key.shape[0], 0),
                               self._original_data[0:0], # Empty time slice
                               self._getitem_internal(key))
        else:
            raise RuntimeError("Unrecognized key {} of type {}. (history: {})"
                               .format(key, type(key), self.name))

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start = 0 if key.start is None else self.get_t_idx(key.start)
            stop = len(self._tarr) if key.stop is None else self.get_t_idx(key.stop)
            if start > stop:
                raise NotImplementedError("Assigning to inverted slices not implemented."
                                          "\nhistory: {}\nslice: {}"
                                          .format(self.name, key))
            if key.step is None:
                step = None
                valueshape = (stop-start,) + self.shape
            else:
                step = self.index_interval(key.step)
                valueshape = ((stop-start)//step,) + self.shape
            key = slice(start, stop, step)
            if shim.isscalar(value) or value.shape != valueshape:
                value = shim.broadcast_to(value, valueshape)
        elif shim.isarray(key):
            if not shim.istype(key.dtype, 'int'):
                raise NotImplementedError("__setitem__() not yet implemented "
                                          "for non-integer time arrays. "
                                          "Time array dtype: {}".format(key.dtype))
            valueshape = key.shape + self.shape
            if shim.isscalar(value) or value.shape != valueshape:
                value = shim.broadcast_to(value, valueshape)
        elif shim.isscalar(key):
            key = self.get_t_idx(key)
        else:
            raise ValueError("Unrecognized time key: '{}'".format(key))
        self.update(key, value)

    def _parse_key(self, key):
        """
        Conditionals and tests depending whether a key for __getitem__ is
        a scalar, slice or array.
        """

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
                               shim.cast(1, self.tidx_dtype),
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

        return key, key_filter, latest

    def _getitem_internal(self, key):
        """Does the actual work of __getitem__; the latter just excludes special
        cases, like empty keys."""

        key, key_filter, latest = self._parse_key(key)

        symbolic_return = True
            # Indicates that we allow the return value to be symbolic. If false and return
            # is a graph object, we will force a numerical value by calling 'get_value'
        if not shim.is_theano_object(key, key_filter, latest) and latest <= self._original_tidx.get_value():
            # No need to compute anything, even if _cur_tidx is a graph object
            symbolic_return = False
        elif shim.is_theano_object(latest, self._cur_tidx):
            # For theano variables, we can't know in advance if we need to compute
            # or not.
            # TODO: always compute, and let compute_up_to decide ?
            self.compute_up_to(latest)
        elif latest > self._cur_tidx:#.get_value():
            # No get_value() here because after updates, _cur_tidx is no longer a shared var
            if (self.symbolic
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

        # TODO: Remove. I don't know why we would need to add to inputs.
        # Add `self` to list of inputs
        #if self.symbolic and self not in sinn.inputs:
        # if self not in sinn.inputs:
        #     sinn.inputs[self] = set()

        result = self.retrieve(key)
        if not symbolic_return and shim.is_theano_object(result):
            result = shim.graph.eval(result, max_cost=10)
        if key_filter is None:
            return result
        else:
            return result[key_filter]

    @property
    def time(self):
        return self.get_time_array()

    @property
    def trace(self):
        return self.get_trace()

    def copy(self):
        """Work in progress. At the moment just calls `copy.deepcopy` on itself.
        Will expand to better treat internal data in the future.
        TODO: Return a view instead of actual copy.
        """
        return deepcopy(self)

    def deepcopy(self):
        """Work in progress. At the moment just calls `copy.deepcopy` on itself.
        Will expand to better treat internal data in the future.
        TODO: Allow not copying certain attributes (e.g. '_data')
        """
        return deepcopy(self)

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

        if self.symbolic and self.compiled_history is not None:
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
        if self._cur_tidx != self._original_tidx:
            raise RuntimeError("You are trying to lock the history {}, which "
                               "in the midst of building a Theano graph. Reset "
                               "it first".format(self.name))
        if warn and (self._original_tidx.get_value() < self.t0idx + len(self) - 1
                     and (not self.symbolic
                          or self.compiled_history is None)):
            # Only trigger for Theano histories if their compiled histories are unset
            # (If they are set, they will do their own check)
            logger.warning("You are locking the unfilled history {}. Trying to "
                           "evaluate it beyond {} will trigger an error."
                           .format(self.name, self._tarr[self._original_tidx.get_value()]))
        self.locked = True
        if self.symbolic and self.compiled_history is not None:
            self.compiled_history.lock()

    def unlock(self):
        """Remove the history lock."""
        self.locked = False
        if self.symbolic and self.compiled_history is not None:
            self.compiled_history.unlock()

    def set_update_function(self, func, cast=True, _return_dtype=None):
        """

        Parameters
        ----------
        func: callable
            The update function. Its signature should be
            `func(t)`
        cast: bool
            (Default: True) True indicates to cast the result of a the update
            function to the expected type. Only 'same_kind' casts are permitted.
            This option can be convenient, as it avoids the need to explicitly
            cast in all return functions when using mixed types. For the most
            stringent debugging though this should be set to False.
        _return_dtype: numpy dtype or str equivalent
            The type the function should return. Designed to allow classes to
            override the default type check, which is to check the return
            value against `self.dtype`. E.g. Spiketrain does this because it
            expects indices in its update function.
            In normal usage this should never be set by a user.
        """
        def f(t):
            # TODO: If setting update function in __init__ is deprecated,
            # we can move the assignment to _return_dtype outside of f().
            # NOTE: Don't reassign to _return_dtype: it would put it local
            # scope, and then be undefined when f(t) is called.
            if _return_dtype is None:
                return_dtype = self.dtype
            else:
                return_dtype = _return_dtype
            logger.debug("Compute " + self.name)
            res = func(t)
            logger.debug("Done computing " + self.name)
            if cast:
                res = shim.cast(res, return_dtype, same_kind=True)
                # shim.cast does its own type checking
            else:
                if res.dtype != return_dtype:
                    raise TypeError("Update function for history '{}' returned a "
                                    "value of dtype '{}', but history update "
                                    " expects dtype '{}'."
                                    .format(self.name, res.dtype, return_dtype))
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
        if shim.istype(before, 'int'):
            # before should not be a Theano object here, but we need the condition
            # to return True for both Python and NumPy ints – shim.istype does this
            before_idx_len = before
        else:
            before_idx_len = int(np.ceil(before / self.dt64))
        if shim.istype(after, 'int'):
            after_idx_len = after
        else:
            after_idx_len = int(np.ceil(after / self.dt64))
        # before = before_idx_len * self.dt64
        # after = after_idx_len * self.dt64
        #
        # before_array = np.arange(self.t0 - before,
        #                          self._tarr[0] - config.abs_tolerance,
        #                          self.dt64,
        #                          dtype=self._tarr.dtype)
        # after_array = np.arange(self._tarr[-1] + self.dt - config.abs_tolerance,
        #                         self.tn + self.dt - config.abs_tolerance + after,
        #                         self.dt64,
        #                         dtype=self._tarr.dtype)
            # Use of _tarr ensures we don't add more padding than necessary

        Δbefore_idx = shim.cast(max(before_idx_len - self.padding[0], 0),
                                self.tidx_dtype)
        Δafter_idx = shim.cast(max(after_idx_len - self.padding[1], 0),
                               self.tidx_dtype)
            # Take max because requested padding may be less than what we have
        before_array = (np.arange(-Δbefore_idx, 0) * self.dt64
                        + self._tarr[0]).astype(self._tarr.dtype)
        after_array = (np.arange(1, Δafter_idx+1) * self.dt64
                       + self._tarr[-1]).astype(self._tarr.dtype)

        self._tarr = np.hstack((before_array,
                                self._tarr,#[self.t0idx:self.t0idx+len(self)],
                                after_array))
        self.t0idx += shim.cast(len(before_array), self.tidx_dtype)

        # Update the current time index
        if Δbefore_idx > 0:
            if self._cur_tidx == self._original_tidx:
                self._original_tidx.set_value( self._original_tidx.get_value() + Δbefore_idx )
                self._cur_tidx = self._original_tidx
            else:
                # _cur_tidx is already a transformed variable, so don't link it to _original_tidx
                self._original_tidx.set_value( self._original_tidx.get_value() + Δbefore_idx )
                self._cur_tidx += Δbefore_idx

        # Check that the time index type can still store all time indices
        if not np.can_cast(np.min_scalar_type(-len(self._tarr)), self.tidx_dtype):
            # '-' ensures we don't get a uint as min scalar type
            raise ValueError("With padding, this history now has a length of "
                             "{}, which is too large for the history's time "
                             "index type ({}).\nTo avoid this error, make sure "
                             "total padding does not exceed the length of the "
                             "unpadded history (either by reducing padding, or "
                             "initializing the history with a longer time array.)"
                             .format(len(self._tarr), str(self.tidx_dtype)))

        return Δbefore_idx, Δafter_idx

    def truncate(self, start, end=None, allow_rounding=True, inplace=False):
        """
        Parameters
        ----------
        start: idx | time
            If `None`, no initial truncation. In particular, keeps any padding.
            If `end` is given, initial time of the truncated history.
            If `end` is omitted, value is used for `end` instead. `start` is
            set to `None`.
        end: idx | time
            Latest time of the truncated history.
        allow_rounding: bool
            Whether to allow rounding start and end times to the nearest time
            index. Default is `True`.
        inplace: bool
            Whether to modify the present history inplace, or create and modify
            a copy. Default is to make a copy.

        Returns
        -------
        Series:
            Truncated history.
        """
        # TODO: if inplace=False, return a view of the data
        # TODO: invalidate caches ?
        # TODO: check lock
        # TODO: Theano _data ? _cur_tidx ?
        # TODO: Sparse _data (Spiketrain)
        # TODO: Can't pad (resize) after truncate
        logger.warning("Function `truncate()` is a work in progress.")
        if self._cur_tidx != self._original_tidx:
            raise NotImplementedError  # Building Theano graph

        if end is None:
            end = start
            start = None
        imin = 0 if start is None else self.get_tidx(start)
        imax = len(self._tarr) if end is None else self.get_tidx(end)

        hist = self if inplace else self.deepcopy()
        # TODO: Don't copy _data

        if (isinstance(hist._data, sp.sparse.spmatrix)
            and not isinstance(hist._data, sp.sparse.lil_matrix)):
            hist._data = self._data.tocsr()[imin:imax+1].tocoo()
                # csc matrices can also be indexed by row, but there's no
                # performance hit to converting to csr first.
        else:
            hist._data = self._data[imin:imax+1]  # +1 because imax must be included
        hist._tarr = self._tarr[imin:imax+1]
        if self.t0idx < imin:
            hist.t0 = hist._tarr[0]
            hist.t0idx = shim.cast(0, self.tidx_dtype)
        else:
            hist.t0idx = shim.cast(self.t0idx - imin, self.tidx_dtype)
        if self.tnidx > imax:
            hist.tn = hist._tarr[-1]
            hist._unpadded_length = imax - hist.t0idx

        if self._original_tidx.get_value() > imax:
            hist._original_tidx.set_value(imax)
            hist._cur_tidx.set_value(imax)

        return hist

    def compute_up_to(self, tidx, start='symbolic'):
        """Compute the history up to `tidx` inclusive.

        Parameters
        ----------
        tidx: int, str
            Index up to which we need to compute. Can also be a string, either
            'end' or 'all', in which the entire history is computed. The
            difference between these is that 'end' will compute all values
            starting at the current time, whereas 'all' restarts from 0 and
            computes everything. Padding is also excluded with 'end', while it
            is included with 'all'. When compiling a Theano graph for
            non-iterative histories, 'all' results in a much cleaner graph,
            since the neither computation bound is a Theano variable.
            NOTE: The index must be positive. Negative indices are treated as
            before 0, and lead to no computation. (I.e. negative indices are
            not subtracted from the end.)
        start: str
            Computation runs from one past the currently computed index up to `tidx`.
            The currently computed index may be either
              - 'symbolic': (default) Updating the starting point later will change
                the computation. (Current index is part of the computational graph.)
              - 'numeric': The current value of the current index attribute is
                retrieved and saved. Resulting function will always start from the same
                index.
        """

        if start == 'numeric':
            original_tidx = self._original_tidx.get_value()
        else:
            original_tidx = self._original_tidx
        if tidx == 'end':
            start = original_tidx + 1
            end = self.tnidx
            replace = False
        elif tidx == 'all':
            start = 0
            end = len(self._tarr) - 1
            replace = True
        else:
            shim.check(shim.istype(tidx, 'int'))
            start = original_tidx + 1
            end = tidx
            replace = False


        #end = shim.ifelse(tidx >= 0,
        #                  tidx,
        #                  len(self._tarr) + tidx)

        if self.locked:
            if not shim.is_theano_object(end):
                if ( (self._original_tidx.get_value() < end)
                     or (shim.is_theano_object(self._cur_tidx) and self._cur_tidx < end) ):
                    raise LockedHistoryError("Cannot compute locked history {}."
                                             .format(self.name))
            return

        #if shim.is_theano_object(tidx):
        if self.symbolic and (self._compiling
                                or any(hist._compiling for hist in sinn.inputs)):
            # Don't actually compute: just store the current_idx we would need
            # HACK The 'or' line prevents any chaining of Theano graphs
            # FIXME Allow chaining of Theano graphs when possible/specified
            self._cur_tidx = shim.largest(end, self._cur_tidx)
            return


        # Theano HACK
        if hasattr(self, '_computing'):
            # We are already computing this history (with Theano). Assuming we are
            # correctly only updating up to _original_tidx + 1, there is no need to
            # recursively compute.
            return
        # Theano HACK
        if self._cur_tidx != self._original_tidx:
            # We have already changed the current index once - don't change it again.
            # Again, this assumes that we are only updating up to _original_tidx + 1
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
            and self.symbolic
            and self.compiled_history is not None
            and end <= self.compiled_history._cur_tidx):
            # A computed value already exists and has been computed to this point
            return

        #########
        # Did not abort the computation => now let's do the computation

        stop = end + 1    # exclusive upper bound
        # Construct a time array that will work even for Theano tidx
        # TODO: Remove tarr
        if shim.is_theano_object(start) or shim.is_theano_object(end):
            tarr = shim.gettheano().shared(self._tarr, borrow=True)
            printlogs = False
        else:
            tarr = self._tarr
            printlogs = True

        if not self._iterative:
            batch_computable = True
        #elif (self.symbolic and self._is_batch_computable()):
        elif self._is_batch_computable(up_to=end):
            batch_computable = True
        else:
            batch_computable = False

        if self._compute_range is not None:
            # A specialized function for computing multiple time points
            # has been defined – use it.
            if printlogs:
                logger.monitor("Computing {} up to {}. Using custom batch operation."
                            .format(self.name, tarr[end]))
            self.update(slice(start, stop),
                        # self._compute_range(tarr[slice(start, stop)]))
                        self._compute_range(np.arange(start, stop)))

        elif batch_computable:
            # Computation doesn't depend on history – just compute the whole thing in
            # one go
            if printlogs:
                logger.monitor("Computing {} from {} to {}. Computing all times simultaneously."
                               .format(self.name, tarr[start], tarr[end]))
            if replace:
                # self._data = self._update_function(tarr[::-1])[::-1]
                self._data = self._update_function(np.arange(self.t0idx, self.tnidx+1)[::-1])[::-1]
                if isinstance(self._data, np.ndarray):
                    self._data = shim.shared(self._data, symbolic=self.symbolic)
                self._cur_tidx = end
            else:
                self.update(slice(start,stop),
                            # self._update_function(tarr[slice(start,stop)][::-1])[::-1])
                            self._update_function(shim.arange(start,stop)[::-1])[::-1])
                    # The order in which _update_function is called is flipped, putting
                    # later times first. This ensures that if dependent computations
                    # are triggered, they will also batch update.

        elif shim.is_theano_object(tarr):
            # For non-batch Theano evaluations, we only allow evaluating one time step ahead;
            # here, we simply assume that that is the case.
            # TODO: Throw error if stop is incorrectly higher than start
            logger.monitor("Creating the Theano graph for {}.".format(self.name))

            self._computing = True
                # Temporary flag to prevent infinite recursions
            # self.update(end, self._update_function(tarr[end]))
            self.update(end, self._update_function(end))
            del self._computing

        else:
            assert(not shim.is_theano_object(tarr))
            logger.monitor("Iteratively computing {} from {} to {}."
                        .format(self.name, tarr[start], tarr[stop-1]))
            old_percent = 0
            for i in np.arange(start, stop):
                percent = (i*100)//stop
                if percent > old_percent:
                    logger.monitor("{}%".format(percent))
                    old_percent = percent
                # self.update(i, self._update_function(tarr[i]))
                self.update(i, self._update_function(i))

        logger.monitor("Done computing {}.".format(self.name))

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
        return self._tarr[self._get_time_index_slice(time_slice, include_padding)]

    def _get_time_index_slice(self, time_slice=slice(None, None), include_padding=False):
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

        return slice(start, stop)

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
            return shim.cast(Δt*self.dt64, dtype=self._tarr.dtype)
        else:
            return Δt

    def index_interval(self, Δt, allow_rounding=False):
        """
        If Δt is a time (float), convert to index interval by multiplying by dt.
        If Δt is an index (int), do nothing.
        For very large Δt, this function returns an error because limits in numerical
        precision prevent it from accurately calculating the index interval.
        OPTIMIZATION NOTE: This is a slower routine than its inverse `time_interval`.
        Avoid it in code that is called repeatedly, unless you know that Δt is an index.
        It also always performs calculations with a double precision (64 bit) dt, even
        when floatX is set to float32, because the higher precision is required to compute
        the correct number of time bins.

        FIXME: Make this work with array arguments

        Returns
        -------
        Integer (of type self.tidx_dtype)
        """
        if not shim.is_theano_object(Δt) and abs(Δt) < self.dt64 - config.abs_tolerance:
            if Δt == 0:
                return 0
            else:
                raise ValueError("You've asked for the index interval corresponding to "
                                 "Δt = {}, which is smaller than this history's step size "
                                 "({}).".format(Δt, self.dt64))
        if shim.istype(Δt, 'int'):
            if not shim.istype(Δt, self.tidx_dtype):
                Δt = shim.cast( Δt, self.tidx_dtype )
            return Δt
        else:
            try:
                shim.check( Δt * config.get_rel_tolerance(Δt) < self.dt64 )
            except AssertionError:
                raise ValueError("You've tried to convert a time (float) into an index "
                                 "(int), but the value is too large to ensure the absence "
                                 "of numerical errors. Try using a higher precision type.")
            quotient = Δt / self.dt64
            rquotient = shim.round(quotient)
            if not allow_rounding:
                # try:
                #     shim.check( shim.abs(quotient - rquotient) < config.get_abs_tolerance(Δt) / self.dt64 )
                # except AssertionError:
                if not sinn.isclose(quotient, rquotient, tol=Δt):
                    logger.error("Δt: {}, dt: {}".format(Δt, self.dt64) )
                    raise ValueError("Tried to convert t={} to an index interval "
                                     "but its not a multiple of dt={}."
                                    .format(Δt, self.dt64))
            return shim.cast(rquotient, self.tidx_dtype, same_kind=False)

    def get_time(self, t):
        """
        If t is an index (i.e. int), return the time corresponding to t_idx.
        Else just return t
        """
        # TODO: Is it OK to enforce single precision ?

        if shim.istype(t, 'int'):
            return shim.cast_floatX(self._tarr[0] + t*self.dt)
        else:
            return t

    def get_tidx(self, t, allow_rounding=False):
        """Return the idx corresponding to time t. Fails if no such index exists.
        It is ok for the t to correspond to a time "in the future",
        and for the data array not yet to contain a point at that time; doing
        so triggers the computation of the history up to `t`.
        `t` may also be specified as a slice, in which case a slice of time
        indices is returned.

        Parameters
        ----------
        t: int, float, slice, array
            The time we want to convert to an index. Integers are considered
            indices and returned unchanged.

        allow_rounding: bool (default: False)
            By default, if no time index corresponds to t, a ValueError is raised.
            This behaviour can be changed if allow_rounding is set to True, in
            which case the index corresponding to the time closest to t is returned.
        """
        def _get_tidx(t):
            if shim.istype(t, 'int'):
                # It's an easy error to make, specify a time as an int
                # Print a warning, just in case.
                # print("Called get_t_idx on an integer ({}). Assuming this to be an INDEX".format(t)
                #       + " (rather than a time) and returning unchanged.")
                return shim.cast( t, dtype = self.tidx_dtype )
            else:
                try:
                    shim.check((t >= self._tarr[0]).all())
                except AssertionError:
                    raise RuntimeError("You've tried to obtain the time index at t={}, which "
                                       "is outside this history's range. Please add padding."
                                       .format(t))

                if self._strict_index_rounding:
                    # Enforce that times be multiples of dt

                    try:
                        shim.check( (t * config.get_rel_tolerance(t) < self.dt64).all() )
                    except AssertionError:
                        raise ValueError("You've tried to convert a time (float) "
                                         "for history into an index "
                                         "(int), but the value is too large to ensure the absence "
                                         "of numerical errors. Try using a higher precision type.")
                    t_idx = (t - self._tarr[0]) / self.dt64
                    r_t_idx = shim.round(t_idx)
                    if (not shim.is_theano_object(r_t_idx)
                        and not allow_rounding
                        and (abs(t_idx - r_t_idx) >
                             config.get_abs_tolerance(t_idx, self._tarr[0])/self.dt64
                             ).all() ):
                        logger.error("t: {}, t0: {}, t-t0: {}, t_idx: {}, dt: {}"
                                     .format(t, self._tarr[0], t - self._tarr[0], t_idx, self.dt64) )
                        raise ValueError("Tried to obtain the time index of t=" +
                                        str(t) + ", but it does not seem to exist.")
                    return shim.cast(r_t_idx,
                                     dtype = self.tidx_dtype,
                                     same_kind = False)

                else:
                    # Allow t to take any value, and round down to closest
                    # multiple of dt
                    return shim.cast( (t - self._tarr[0]) // self.dt64,
                                      dtype = self.tidx_dtype )

        if isinstance(t, slice):
            start = self.t0idx if t.start is None else _get_tidx(t.start)
            stop = self.t0idx + len(self) if t.stop is None else _get_tidx(t.stop)
            return slice(start, stop, t.step)
        else:
            return _get_tidx(t)
    get_t_idx = get_tidx
        # For compability with older code

    def get_tidx_for(self, t, target_hist):
        """
        Convert a time or time index into a time index for another history.
        """
        tidx = self.get_tidx(t)
        if self.dt == target_hist.dt:
            if self.t0idx == target_hist.t0idx:
                # Don't add cruft to the computational graph
                return tidx
            else:
                return tidx - self.t0idx + target_hist.t0idx
        else:
            raise NotImplementedError("`get_tidx_for()` is currently only "
                                      "implemented for histories with same "
                                      "timestep 'dt'.")

    def get_t_for(self, t, target_hist):
        """
        Convert a time or time index for indexing into another history.
        The type is preserved, so that if `t` is a time, a time (float) is
        returned, and if `t` is a time index, an time index is returned.
        (In fact, if `t` is a time, it is simply returned as is.)
        """
        if isinstance(t, slice):
            start = None if t.start is None else self.get_t_for(t.start)
            stop = None if t.stop is None else self.get_t_for(t.stop)
            if self.dt != target_hist.dt:
                raise NotImplementedError(
                    "Cannot convert time for history {} to {} because they "
                    "have diffrent steps"
                    .format(self.name, target_hist.name))
        if shim.istype(t, 'float'):
            return t
        else:
            return self.get_tidx_for(t, target_hist)

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

    def recreate_data(self):
        """
        Recreate the internal data by calling `shim.shared` on the current value.
        This 'resets' the state of the data with the current state of `shim`,
        which may be desired if a shimmed library was loaded or unloaded after
        creating the history.
        Will fail if either the data or current index is a symbolic (rather
        than shared) variable.
        This overwrites the original data; consider first making a deepcopy
        if you need to keep the original.
        """
        assert(self._cur_tidx.get_value() == self._original_tidx.get_value())
        assert(np.all(self._data.get_value() == self._original_data.get_value()))
        # for attrname in dir(self):
        for attrname in ['_original_tidx', '_original_data']:
            attr = getattr(self, attrname)
            if isinstance(attr, shim.ShimmedShared):
                setattr(self, attrname, shim.shared(attr.get_value(), name=attr.name))
        self._cur_tidx = self._original_tidx
        self._data = self._original_data

    def discretize_kernel(self, kernel):

        dis_attr_name = "discrete_" + str(id(self))  # Unique id for discretized kernel

        if hasattr(kernel, dis_attr_name):
            # TODO: Check that this history (self) hasn't changed
            return getattr(kernel, dis_attr_name)

        else:
            #TODO: Add compability check of the kernel's shape with this history.
            #shim.check(kernel.shape == self.shape*2)
            #    # Ensure the kernel is square and of the right shape for this history

            # The kernel may start at a position other than zero, resulting in a shift
            # of the index corresponding to 't' in the convolution
            # TODO: if kernel.t0 is not divisible by dt, do an appropriate average
            idx_shift = int(round(kernel.t0 / self.dt64))
                # We don't use shim.round because time indices must be Python numbers
            t0 = shim.cast(idx_shift * self.dt64, shim.config.floatX)
                # Ensure the discretized kernel's t0 is a multiple of dt

            memory_idx_len = int(kernel.memory_time // self.dt64) - 1 - idx_shift
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
            #     symbolic=True
            # else:
            #     symbolic=False
            symbolic=False
            dis_kernel = Series(t0=t0,
                                tn=t0 + memory_idx_len*self.dt64,
                                dt=self.dt64,
                                shape=kernel.shape,
                                # f=kernel_func,
                                name=dis_name,
                                symbolic=symbolic,
                                iterative=False)
                # Kernels are non-iterative by definition: they only depend on their parameters
            dis_kernel.idx_shift = idx_shift

            # Set the update function for the discretized kernel
            if config.integration_precision == 1:
                _kernel_func = kernel.eval
            elif config.integration_precision == 2:
                # TODO: Avoid recalculating eval at the same places by writing
                #       a compute_up_to function and passing that to the series
                _kernel_func = lambda t: (kernel.eval(t) + kernel.eval(t+self.dt)) / 2
            else:
                # TODO: higher order integration with trapeze or simpson's rule
                raise NotImplementedError

            ## Kernel functions can only be defined to take times, so we wrap
            ## the function
            def kernel_func(t):
                t = dis_kernel.get_time(t)
                return _kernel_func(t)

            dis_kernel.set_update_function(kernel_func)

            # Attach the discretization to the kernel instance
            setattr(kernel, dis_attr_name, dis_kernel)

            return dis_kernel

    def add_input(self, variable):

        if self not in sinn.inputs:
            sinn.inputs[self] = set()

        if isinstance(variable, str):
            # Not sure why a string would be an input, but it guards against the next line
            sinn.inputs[self].add(variable)
            #self._inputs.add(variable)
        if isinstance(variable, (HistoryBase, KernelBase)):
            sinn.inputs[self].add(variable)
        elif isinstance(variable, Iterable):
            for x in variable:
                sinn.inputs[self].add(x)
    add_inputs = add_input
        # Synonym to add_input

    def clear_inputs(self):
        if self not in sinn.inputs:
            sinn.inputs[self] = set()
        sinn.inputs[self].clear()

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
                if inp.symbolic:
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

        if not self.symbolic:
            raise RuntimeError("You cannot compile a Series that does not use Theano.")

        # Create the new history which will contain the compiled update function
        self.compiled_history = self.__class__(self, name=self.name + " (compiled)", symbolic=False)
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
        t_float = shim.getT().scalar('t', dtype=shim.config.floatX)
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

    def array_to_slice(self, array, lag=None):
        """
        Assumes the array is monotonous and evenly spaced. This is checked, but
        only if `array` is not symbolic.

        Parameters
        ----------
        array: nparray of floats or ints
            int: time indices
            floats: times
        lag: float or int
            The output slice will be shifted by this amount. If float,
            interpreted as time and first converted to index interval.

        Returns
        -------
        slice of time indexes (integers)
        """
        # We need to be careful here, because array could be empty, and then
        # indexing it with array[0] or array[-1] would trigger an error
        empty_array = shim.eq( shim.min(array.shape), 0 )
        # An array of size 1 would still fail on array[1]
        singleton_array = shim.le( shim.min(array.shape), 1 )

        if ( not shim.is_theano_object(array) and shim.istype(array, 'float')
             and not singleton_array ):
            assert(abs(array[1] - array[0]) >= self.dt - sinn.config.abs_tolerance)

        singleton_dt = self.dt if shim.istype(array, 'float') else 1
        step = shim.ifelse(singleton_array,
                           singleton_dt,
                           shim.LazyEval(lambda: array[1] - array[0]))
                           #shim.largest(array[1] - array[0], self.dt))

        idxstart = shim.ifelse(empty_array,
                               shim.cast(0, dtype=self._cur_tidx.dtype),
                               self.get_t_idx(array[0]))
        if not shim.is_theano_object(array) and not singleton_array:
            # Check that array is evenly spaced
            assert(np.all(sinn.ismultiple(array[1:] - array[:-1], step)))

        idxstep = self.index_interval(step)

        idxstop = shim.ifelse(empty_array,
                              shim.cast(0, dtype=self._cur_tidx.dtype),
                              self.get_t_idx(array[-1]) + idxstep)
            # We add/subtract idxstep because the array upper bound is inclusive

        # Ensure that idxstop is not negative. This would require replacing it by
        # None, and doing that in a Theano graph, if possible at all, is clumsy.
        # Since negative idxstop is an unnecessary corner case, we rather not support
        # it than add cruft to the Theano graph.
        if not shim.is_theano_object(idxstop):
            assert(idxstop >= 0)

        if lag is not None:
            if shim.istype(lag, 'float'):
                lag = self.index_interval(lag)
            else:
                assert(shim.istype(lag, 'int'))
            idxstart += lag
            idxstop += lag

        return slice(idxstart, idxstop, idxstep)
    time_array_to_slice = array_to_slice
        # For compatiblity with older functions
        # TODO: Deprecate and remove time_array_to_slice


    def _is_batch_computable(self, up_to='end'):
        """
        Returns true if the history can be computed at all time points
        simultaneously.
        WARNING: This function is only to be used for the construction of
        a Theano graph. After compilation, sinn.inputs is cleared, and therefore
        the result of this function will no longer be valid.
        HACK: sinn.inputs is no longer cleared, so this function should no longer
        be limited to Theano graphs – hopefully that doesn't break anything else.

        Parameters
        ----------
        up_to: int
            Only check batch computability up to the given time index. Default
            is to check up to the end. Effectively, this adds an additional
            success condition, when the current time index is >= to `up_to`.
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
                 if hist.symbolic and hist.compiled_history is not None]) )

        # Get the list of inputs. A compiled history may not be in the input list,
        # so if `self` is not found, we try to find its parent
        input_list = None
        if self in sinn.inputs:
            input_list = sinn.inputs[self]
        else:
            for hist in sinn.inputs:
                if (hist.symbolic and hist.compiled_history is self):
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
        elif all( hist.locked or hist._is_batch_computable(up_to)
                  for hist in input_list):
            # The potential cyclical dependency chain has been broken
            retval = True
        elif self._original_tidx.get_value() >= self.tnidx:
            return True
        elif shim.is_theano_object(up_to):
            # A symbolic `up_to` can always be higher than _cur_tidx, unless
            # it's actually the same variable
            retval = (up_to == self._cur_tidx)
                # NOTE/TODO: This isn't a perfect test; e.g. if `up_to` is
                # equal to `_cur_tidx - 1`, batch_computable will return False
                # It will also fail if `up_to` is a time (float)
        elif up_to != 'end':
            # A symbolic `up_to` can always be higher than the cur_tidx
            up_to = self.get_t_idx(up_to)
            retval = (up_to <= self._original_tidx.get_value())
        else:
            retval = False

        del self._batch_loop_flag

        return retval

class PopulationHistory(PopulationHistoryBase, History):
    """
    History where traces are organized into populations.
    TODO: At moment just a placeholder history. Eventually the "population" stuff
          (e.g. the redefinition of 'shape') should be moved here.
    """
    pass

# Spiketimes is currently really slow for long data traces (> 5000 time bins)
# Development efforts have been moved to Spiketrain; maybe in the future if we need
# to track actual times, we will resurrect development of this class
class Spiketimes(ConvolveMixin, PopulationHistory):
    """A class to store spiketrains.
    These are stored as times associated to each spike, so there is
    no well-defined 'shape' of a timeslice. Instead, the `shape` parameter
    is used to indicate the number of neurons in each population.

    NOTE: This development of this class has been put on hold as the Spiketrain
    is better suited to its original use case. I think it's a useful concept and
    want to see it developed further; it's just not a current priority.
    Thus expect to need to bring it up to date before using it.
    """

    _strict_index_rounding = False

    def __init__(self, hist=sinn._NoValue, name=None, *args, t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 pop_sizes=sinn._NoValue, dtype=sinn._NoValue, **kwargs):
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

        if dtype is not None:
            logger.warning("Spiketimes class does not support 'dtype'")

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
        """Spiketimes can't just be invalidated, they really have to be cleared."""
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        self.initialize(init_data)
        super().clear()

    def set_update_function(self, func, _return_dtype=None):
        if _return_dtype is None:
            super().set_update_functin(func, self.idx_dtype)
        else:
            super().set_update_function(func, _return_dtype)

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
        """
        Add to each neuron specified in `value` the spiketime `tidx`.

        Parameters
        ----------
        tidx: int, float
            The time index of the spike(s). This is converted
            to actual time and saved.
            Can optionally also be a float, in which case no conversion is made.
            Should not correspond to more than one bin ahead of _cur_tidx.
        neuron_idcs: iterable
            List of neuron indices that fired in this bin.
        """
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
        """
        Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.

        Parameters
        ----------
        before: float | int
            Amount of time to add to before t0. If non-zero, all indices
            to this data will be invalidated.
        after: float (default 0)
            Amount of time to add after tn.
        """
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

        The history is returned, to allow chaining operations.

        Returns
        -------
        This history instance
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
        return self


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
        t = self.get_time(t)

        if kernel_slice.stop is None:
            start = t - kernel.memory_time - kernel.t0
        else:
            if shim.istype(kernel_slice.stop, 'int'):
                start = t - kernel_slice.stop*self.dt64 - kernel.t0
            else:
                start = t - kernel_slice.stop
        if kernel_slice.start is None:
            stop = t - kernel.t0
        else:
            if shim.istype(kernel_slice.stop, 'int'):
                stop = t - kernel_slice.start*self.dt64 - kernel.t0
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

class Spiketrain(ConvolveMixin, PopulationHistory):
    """
    A class to store spiketrains, grouped into populations.

    These are stored in a sparse array where spikes are indicated by ones.
    Instead of the `shape` parameter, we use `pop_slices` to indicate the neurons
    associated to each population, from which `shape` is automatically deduced.
    Only 1d timeslices are currently supported (i.e. all populations are arranged
    along one axis).

    Internally spikes are stored in a coo array, for fast addition of new spikes.
    It is cast to csr for slicing. See the `scipy.sparse` documentation for a
    discussion of different sparse array types.

    Convolutions are specially defined to be summed over populations, and thus
    require the definition of the connectivity matrix. If convolutions are not
    to be used, the connectivity matrix can be left undefined.

    There are still a few unsolved challenges preventing using this class in a
    Theano graph at the moment. Notably, Theano only supports csc and csr. Moreover,
    Theano docs warn against using sparse arrays within `scan()`, as would almost
    certainly be required.

    NOTE: Although this class was designed to simulate and store spike trains,
    since the data type and update functions are arbitrary, it can just as well
    be used for any kind of sparse data. Depending on the needs of the application,
    the convolution functions may need to be redefined.
    """

    _strict_index_rounding = True

    def __init__(self, hist=sinn._NoValue, name=None, *args, time_array=sinn._NoValue,
                 pop_sizes=sinn._NoValue, dtype=sinn._NoValue,
                 t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 **kwargs):
        """
        All parameters except `hist` are keyword parameters.
        `pop_sizes` is always required.
        If `hist` is not specified, `time_array` must be specified.

        Parameters
        ----------
        hist: History instance
            Optional. If passed, will used this history's parameters as defaults.
        time_array: ndarray (float)
            The array of times this history samples. If provided, `t0`, `tn` and `dt` are ignored.
            Note that times should always be provided as 64 bit floats; the array is internally
            convert to floatX, but a full precision version is also kept for some calculations such
            as `index_interval()`.
        pop_sizes: integer tuple
            Number if neurons in each population
        iterative: bool (default: True)
            (Optional) If true, indicates that f must be computed iteratively. I.e. having
            computed f(t) is required in order to compute f(t+1). When false,
            when computed f for multiple times t, these will be passed as an array
            to f, using only one function call. Default is to force iterative computation.
        dtype: numpy dtype
            Type to use for the internal sparse array.
        t0: float
            Time at which the history starts. /Deprecated: use `time_array`./
        tn: float
            Time at which the history ends. /Deprecated: use `time_array`./
        dt: float
            Timestep. /Deprecated: use `time_array`./
        **kwargs:
            Extra keyword arguments are passed on to History's initializer
        """
        self.PopTerm = self.get_popterm
            # FIXME: At present self.PopTerm is not a classes, which
            #        can be confusing. If PopTermMeso/PopTermMicro were implemented as metaclasses,
            #        we could call the metaclass here instead, which would return a proper class

        if name is None:
            name = "spiketimes{}".format(self.instance_counter + 1)
        if hist is not sinn._NoValue and pop_sizes is sinn._NoValue:
            pop_sizes = hist.pop_sizes
        if pop_sizes is sinn._NoValue:
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

        # Force user to explicitly call `set_connectivity` before producing data.
        # It's too easy otherwise to think that this has been correctly set automatically
        self.conn_mats = None

        super().__init__(hist, name, t0=t0, tn=tn, dt=dt, time_array=time_array,
                         shape=shape, **kwargs)

        self.initialize(dtype=dtype)

    def raw(self):
        return super().raw(_data = self._data,
                           pop_sizes = self.pop_sizes)

    @classmethod
    def from_raw(cls, raw, update_function=sinn._NoValue, symbolic=False, lock=True):
        """See History.from_raw"""
        retval = super().from_raw(raw, update_function, symbolic, lock,
                                  pop_sizes=raw['pop_sizes'])
        # data was wrapped with a scalar array then further with shim.shared.
        # This line recovers the original
        retval._data = retval._data.get_value()[()]
        return retval

    @property
    def npops(self):
        """The number of populations."""
        return len(self.pop_slices)

    def get_popterm(self, values):
        if isinstance(values, popterm.PopTerm):
            return values
        else:
            # TODO: Find a way not to instantiate a PopTerm just for 'infer_block_type'
            dummy_popterm = popterm.PopTermMacro(self.pop_sizes, np.zeros(1), ('Macro',))
            block_types = dummy_popterm.infer_block_types(values.shape,
                                                          allow_plain=False)
            cls = popterm.PopTerm.BlockTypes[block_types[0]]
            return cls(self.pop_sizes, values, block_types)
        #elif len(values) == 1:
            #return popterm.PopTermMacro(self.pop_sizes, values)
        #elif len(values) == len(self.pop_sizes):
            #return popterm.PopTermMeso(self.pop_sizes, values)
        #elif len(values) == sum(self.pop_sizes):
            #return popterm.PopTermMicro(self.pop_sizes, values)
        #else:
            #raise ValueError("Provided values (length {}) neither match the number of "
                             #"populations ({}) or of elements ({})."
                             #.format(len(values), len(self.pop_sizes), sum(self.pop_sizes)))

    def set_connectivity(self, w):
        """
        Set the connectivity matrix from neurons to populations. This
        is required for convolutions.

        Parameters
        ----------
        w: ndarray
           Rectangular connectivity matrix: Npops x Nneurons. If w[i,j]
           is non zero, than neuron j projects to population i.
        """
        # TODO: Combine connectivity matrices into a single BroadcastableBlockArray
        self.conn_mats = [ [ (w[to_pop_slice, from_pop_slice])
                             for from_pop_slice in self.pop_slices ]
                           for to_pop_slice in self.pop_slices ]
            #nonzero returns a tuple (one element per dimension - here there is only one dimension)
            #so we need to index with [0] to have just the array of non-zero columns

    def initialize(self, init_data=None, dtype=None):
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
            typical use is to use n equal to the padding length (i.e. self.t0idx).
            If `init_data` does not provide a dtype, it must be given as second
            argument.
        dtype: numpy dtype
            Type to use for storing the data. If both `init_data` and `dtype`
            are provided, data will be cast to `dtype`. If neither is provided,
            the 'int8' type is used, to be compatible with Theano bools.
        """
        # TODO: Allow either a nested or flattened list for init_data

        nneurons = np.sum(self.pop_sizes)

        if init_data is None:
            if dtype is None or dtype is sinn._NoValue:
                dtype = 'int8'
            self._data = shim.sparse.coo_matrix('spike train',
                                                shape=(len(self._tarr), nneurons),
                                                dtype=dtype)
            # We are just going to store 0s and 1s, so might as well use the smallest
            # available int in Theano
            # Tests have shown that coo_matrix is faster than lil_matrix for the use
            # we make here

        else:
            shim.check(init_data.shape[1] == nneurons)
            if dtype is None or dtype is sinn._NoValue:
                try:
                    dtype = init_data.dtype
                except AttributeError:
                    raise ValueError("The provided data of type {} does have a 'dtype' "
                                     "attribute. In this case you must provide it to "
                                     "Spiketrain.initialize().".format(type(init_data)))
            n = len(init_data)
            csc_data = shim.sparse.csc_matrix('spike train',
                                              shape=(len(self._tarr), nneurons),
                                              dtype=dtype)
            csc_data[:n,:] = shim.sparse.csc_from_dense(init_data.astype(dtype))
            # This may throw an efficiency warning, but we can ignore it since
            # self._data is empty
            self._data = csc_data.tocoo()
            # WARNING: This will break with Theano until/if we implement a
            #          coo matrix interface in theano_shim.

        self._original_data = self._data

    def clear(self, init_data=None):
        """Spiketrains shouldn't just be invalidated, since then multiple runs
        would make them more and more dense."""
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        self.initialize(None)
        super().clear()

    def set_update_function(self, func, _return_dtype=None):
        if _return_dtype is None:
            super().set_update_function(func, _return_dtype=self.idx_dtype)
        else:
            super().set_update_function(func, _return_dtype=_return_dtype)


    def get_trace(self, pop=None, neuron=None, include_padding='none', time_slice=None):
        """
        Return the spiketrain's computed data for the given neuron.
        Time points which have not yet been computed are excluded, such that
        the len(series.get_trace(*)) may be smaller than len(series). The
        return value is however guaranteed to be consistent with get_time_array().
        If `component` is 'None', return the full multi-dimensional trace

        Parameters
        ----------
        pop: int
            Index of the population for which we want the trace. If unspecified,
            all neurons are returned, unless otherwise indicated by the 'neuron' parameter.
            Ignored if 'neuron' is specified.

        neuron: int, slice, array of ints
            Index of the neurons to return; takes precedence over 'pop'.

        include_padding: 'none' (default) | 'before' | 'left' | 'after' | 'right' | 'all' | 'both'
            'none':
                Don't include the padding bins.
            'before' or 'left':
                Include the padding bins preceding t0.
            'after' or 'right:
                Include the padding bins following tn.
            'all' or 'both':
                Include the padding bins at both ends.

        time_slice:
            See get_time_array.

        Returns
        -------
        A csr formatted sparse array.
        """
        if not shim.isshared(self._cur_tidx):
            raise RuntimeError("You are in the midst of constructing a Theano graph. "
                               "Reset history {} before trying to obtain its time array."
                               .format(self.name))

        # if start is not None:
        #     start = self.get_t_idx(start)
        # else:
        #     padding_vals = [ 'none', 'before', 'left', 'after', 'right', 'all', 'both' ]
        #     if include_padding in ['none', 'after', 'right']:
        #         start = self.t0idx
        #     elif include_padding in padding_vals:
        #         # It's one of the other options
        #         start = 0
        #     else:
        #         raise ValueError("include_padding should be one of {}.".format(padding_vals))

        # if end is not None:
        #     stop = self.get_t_idx(end) + 1
        # else:
        #     if include_padding in ['none', 'before', 'left']:
        #         stop = self._cur_tidx.get_value() + 1
        #     elif include_padding in padding_vals:
        #         stop = min(self._cur_tidx.get_value() + 1, len(self._tarr))
        #     else:
        #         raise ValueError("include_padding should be one of {}.".format(padding_vals) )

        time_slice = time_slice if time_slice is not None else slice(None, None)
        tslice = self._get_time_index_slice(time_slice, include_padding)

        assert(self._cur_tidx.get_value() >= tslice.stop - 1)

        data_arr = self._data.tocsr()
        if neuron is None:
            if pop is None:
                return data_arr[tslice]
            else:
                return data_arr[tslice, self.pop_slices[pop]]
        elif isinstance(neuron, (int, slice)):
            return data_arr[start:stop, neuron]
        elif isinstance(neuron, Iterable):
            idx = (tslice,) + tuple(component)
            return data_arr[idx]
        else:
            raise ValueError("Unrecognized spiketrain neuron '{}' of type '{}'"
                             .format(neuron, type(neuron)))


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
            return self._data.tocsr()[tidx].todense().A

        elif isinstance(key, slice):
            start = None if key.start is None else self.get_t_idx(key.start)
            stop  = None if key.stop  is None else self.get_t_idx(key.stop)
            step  = None if key.step  is None else self.index_interval(key.step)

            return self._data.tocsr()[slice(start, stop, step)].todense().A
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
        tidx: int, float or 1D array of int, float
            The time index of the spike(s). Can optionally also be a float,
            in which case it is converted to the corresponding bin.
            Should not correspond to more than one bin ahead of _cur_tidx.
        neuron_idcs: iterable
            List of neuron indices that fired in this bin.
        '''
        # TODO: Fix batch update to something less hacky
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        neuron_idcs = shim.asarray(neuron_idcs)
        if neuron_idcs.ndim == 0:
            raise ValueError("Indices of neurons to update must be given as an array.")

        newidx = self.get_t_idx(tidx)
        if shim.isscalar(newidx):
            latestidx = newidx
            if neuron_idcs.ndim > 1:
                raise ValueError("Indices of neurons to update specified as a {}d array. "
                                 "Flatten the array before passing to `update`"
                                 .format(neuron_idcs.ndim))
            tidx = [tidx]
            neuron_idcs = [neuron_idcs]
        else:
            latestidx = shim.max(newidx)
            # if neuron_idcs.ndim == 1:
            #     raise ValueError("You are attempting to update a range of time points, "
            #                      "but specified the indices of neurons to update as only "
            #                      "a 1d array. It must be 2d, where the first axis is time.")
            # elif neuron_idcs.ndim > 2:
            #     raise ValueError("Indices of neurons to update specified as a {}d array. "
            #                      "Flatten the array before passing to `update`"
            #                      .format(neuron_idcs.ndim))
        if not shim.is_theano_variable([newidx, latestidx]):
            assert(latestidx <= self._original_tidx.get_value() + 1)
            if newidx.ndim > 0:
                assert(newidx.shape[0] == neuron_idcs.shape[0])

        for ti, idcs in zip(tidx, neuron_idcs):
            # TODO: Assign in one block
            onevect = shim.ones(idcs.shape, dtype='int8')
                # vector of ones of the same length as the number of units which fired
            self._data.data = shim.concatenate((self._data.data, onevect))
                # Add as many 1 entries as there are new spikes
            self._data.col = shim.concatenate((self._data.col, idcs))
                # Assign those spikes to neurons (col idx corresponds to neuron index)
            self._data.row = shim.concatenate((self._data.row,
                                               (shim.add_axes(ti, 1, 'after')*onevect).flatten()))
                # Assign the spike times (row idx corresponds to time index)

        # Set the cur_idx. If tidx was less than the current index, then the latter
        # is *reduced*, since we no longer know whether later history is valid.
        if not shim.is_theano_object(self._cur_tidx, latestidx):
            if latestidx < self._cur_tidx.get_value():
                logger.warning("Moving the current time index of a Spiketrain "
                               "backwards. Invalidated data is NOT cleared.")
            self._cur_tidx.set_value( latestidx )
        else:
            self._cur_tidx = latestidx
        if shim.is_theano_variable(self._original_tidx):
            shim.add_update(self._original_tidx, self._cur_tidx)
        else:
            self._original_tidx = self._cur_tidx

    def pad(self, before, after=0):
        """
        Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.

        Parameters
        ----------
        before: float | int
            Amount of time to add to before t0. If non-zero, all indices
            to this data will be invalidated.
        after: float | int (default 0)
            Amount of time to add after tn.
        """
        before_len, after_len = self.pad_time(before, after)
        newshape = (len(self._tarr), sum(self.shape))
        self._data.row += before_len
            # increment all time bins by the number that were added
        self._data = sp.sparse.coo_matrix((self._data.data, (self._data.row, self._data.col)),
                                          shape = newshape )

    def set(self, source=None, tslice=None):
        """Set the entire spiketrain in one go. `source` may be a list of arrays, a
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

        The history is returned, to allow chaining operations.

        Returns
        -------
        This history instance
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

        return self


    def _convolve_op_single_t(self, discretized_kernel, tidx, kernel_slice):
        '''Return the time convolution with the spike train, i.e.
            ∫ spiketimes(t - s) * kernel(s) ds
        with s ranging from -∞ to ∞  (normally there should be no spikes after t).
        The result is a 1d array of length Nneurons.
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
        1d ndarray of length Nneurons.
            It is indexed as result[from pop idx][to pop idx]

        FIXME: Need to keep contributions from each pop separate, by returning
               a 2d ndarray of size Nneurons x Npops
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
        #    (But sometimes still returns a dense array ?)
        # 3) that sparse arrays are always 2D, so A[0,0] is a 2D, 1x1 matrix
        # 4) The .A attribute of a matrix returns the underlying array
        # 5) That we do not need to multiply by the step size (dt): This is a discretized
        #    train of Dirac delta peaks, so to get the effective height of the spike
        #    when spread over a bin we should first divide by dt. To take the convolution
        #    we should then also multiply by the dt, cancelling the two.
        if discretized_kernel.ndim == 2 and discretized_kernel.shape[0] == discretized_kernel.shape[1]:
            # 2D discretized kernel: each population feeds into every other with a different kernel
            shim.check(discretized_kernel.shape[0] == len(self.pop_sizes))
            # TODO: remove useless asarray
            return shim.asarray(
                np.concatenate (
                    [ sum (
                          self.conn_mats[to_pop_idx][from_pop_idx].dot(
                            hist_subarray[:, self.pop_slices[from_pop_idx]].multiply(
                                discretized_kernel[kernel_slice][::-1, to_pop_idx, from_pop_idx:from_pop_idx+1]
                            ).sum(axis=0).A.T )  # .T makes a column vector
                          for from_pop_idx in range(len(self.pop_sizes)) )[:,0] # column vec -> 1d array
                      for to_pop_idx in range(len(self.pop_sizes)) ],
                    axis = 0 ) )

        elif discretized_kernel.ndim == 1 and discretized_kernel.shape[0] == len(self.pop_sizes):
            # 1D discretized_kernel: populations only feed back into themselves
            # HACK: Just removed connectivity matrix altogether
            return shim.asarray( np.concatenate(
                [
                    hist_subarray[:, self.pop_slices[from_pop_idx]].multiply(
                        discretized_kernel[kernel_slice][::-1, from_pop_idx:from_pop_idx+1]
                    ).sum(axis=0).A[0, :]  # row vec -> 1d array
                  for from_pop_idx in range(len(self.pop_sizes)) ],
                axis = 0 ) )

        elif discretized_kernel.ndim == 1 and discretized_kernel.shape[0] == sum(self.pop_sizes):
            return hist_subarray.multiply( discretized_kernel[kernel_slice][::-1] ).sum(axis=0).A[0, :]

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

    # TODO: Implement pop_xxx functions as operator methods
    #       This would involve also implementing operators which return a Spiketrain
    def pop_add(self, neuron_term, summand):
        if not shim.is_theano_object(neuron_term, summand):
            assert(len(self.pop_slices) == len(summand))
            return shim.concatenate([neuron_term[..., pop_slice] + sum_el
                                     for pop_slice, sum_el in zip(self.pop_slices, summand)],
                                    axis=-1)
        else:
            raise NotImplementedError

    def pop_radd(self, summand, neuron_term):
        return self.pop_add(neuron_term, summand)

    def pop_mul(self, neuron_term, multiplier):
        if not shim.is_theano_object(neuron_term, multiplier):
            assert(len(self.pop_slices) == len(multiplier))
            return shim.concatenate([neuron_term[..., pop_slice] * mul_el
                                     for pop_slice, mul_el in zip(self.pop_slices, multiplier)],
                                    axis=-1)
        else:
            raise NotImplementedError

    def pop_rmul(self, multiplier, neuron_term):
        return self.pop_mul(neuron_term, multiplier)

    def pop_div(self, neuron_term, divisor):
        if not shim.is_theano_object(neuron_term, divisor):
            assert(len(self.pop_slices) == len(divisor))
            return shim.concatenate( [ neuron_term[..., pop_slice] / div_el
                                       for pop_slice, div_el in zip(self.pop_slices, divisor)],
                                     axis = -1)
        else:
            raise NotImplementedError



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
                 shape=sinn._NoValue, dtype=sinn._NoValue, **kwargs):
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

        # if self.symbolic:
        #     # Make the dimensions where shape is 1 broadcastable
        #     # (as they would be with NumPy)
        #     data_tensor_broadcast = tuple(
        #         [False] + [True if d==1 else 0 for d in self.shape] )
        #     self.DataType = shim.getT().TensorType(shim.config.floatX,
        #                                            data_tensor_broadcast)
        #     #self._data = shim.T.zeros(self._tarr.shape + self.shape, dtype=config.floatX)
        #     self._data = self.DataType(self.name + ' data')
        #     #self.inf_bin = shim.lib.zeros(self.shape, dtype=config.floatX)
        # else:
        if dtype is sinn._NoValue:
            if hist is not sinn._NoValue:
                dtype = hist.dtype
            else:
                dtype = shim.config.floatX
        self._data = shim.shared(np.zeros(self._tarr.shape + self.shape, dtype=dtype),
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
        # Convert Python plain data types to arrays
        if isinstance(value, (int, float)):
            value = np.asarray(value)

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

        if self.symbolic:
            # if not shim.is_theano_object(value):
            #     logger.warning("Updating a Theano array ({}) with a Python value. "
            #                    "This is likely an error.".format(self.name))
            #if self._original_data is not None or shim.config.theano_updates != {}:
            # if ( self._original_data is not None
            #      and self._original_data in shim.config.theano_updates):
            #     raise RuntimeError("You can only update data once within a "
            #                        "Theano computational graph. If you need "
            #                        "multiple updates, compile a single "
            #                        "update as a function, and call that "
            #                        "function repeatedly.")
            # assert(shim.is_theano_variable(self._data))
            #     # This should be guaranteed by self.symbolic=True
            #if self._original_data is None:
            #    self._original_data = self._data
                    # Persistently store the current _data, because that's the handle
                    # to the input that will be used when compiling the function
            # TODO: Decide whether this is the best way to avoid reupdating/recomputing past time points
            # self._data = shim.ifelse(self._cur_tidx < end,
            #                          shim.set_subtensor(tmpdata[tidx], value),
            #                          self._data)

            # self._cur_tidx = shim.largest(self._cur_tidx, end)
            if shim.is_theano_object(end):
                assert(shim.is_theano_object(self._original_tidx))
                self._cur_tidx = end
                shim.add_update(self._original_tidx, self._cur_tidx)
            else:
                self._original_tidx.set_value(shim.cast(end, self.tidx_dtype))
                self._cur_tidx = self._original_tidx

            if (not shim.is_theano_object(end, value)
                and (end == tidx or not shim.is_theano_object(tidx.start))
                and self._cur_tidx == self._original_tidx
                and self._data == self._original_data):
                # There are no symbolic dependencies – update data directly
                tmpdata = self._original_data.get_value(borrow=True)
                tmpdata[tidx] = value
                self._original_data.set_value(tmpdata, borrow=True)
            else:
                tmpdata = self._data
                self._data = shim.set_subtensor(tmpdata[tidx], value)
                if updates is not None:
                    shim.add_updates(updates)
                if shim.is_theano_object(self._original_data):
                    shim.add_update(self._original_data, self._data)
                else:
                    self._original_data = self._data

            # Should only have Theano updates with Theano original data
            assert(shim.is_theano_object(self._original_data)
                   and shim.is_theano_object(self._data))
            assert(shim.is_theano_object(self._original_tidx)
                   and shim.is_theano_object(self._cur_tidx))
        else:
            if shim.is_theano_object(value):
                if not shim.graph.is_computable([value]):
                    raise ValueError("You are trying to update a pure numpy series ({}) "
                                     "with a Theano variable. You need to make the "
                                     "series a Theano variable as well or ensure "
                                     "that `value` does not depend on any "
                                     "symbolic inputs."
                                     .format(self.name))
                value = shim.graph.eval(value, max_cost=None)
            if shim.is_theano_object(tidx):
                if not shim.graph.is_computable([tidx]):
                    raise ValueError("You are trying to update a pure numpy series ({}) "
                                     "with a time idx that is a Theano variable. You need "
                                     "to make the series a Theano variable as well "
                                     "or ensure that `tidx` does not depend "
                                     "on any symbolic inputs."
                                     .format(self.name))
                tidx = shim.graph.eval(tidx, max_cost=None)

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
        """
        Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are increased to make them exact multiples of dt.
        See `History.pad_time` for more details.

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
        """

        previous_tarr_shape = self._tarr.shape
        before_len, after_len = self.pad_time(before, after)

        if not kwargs:
            # No keyword arguments specified – use defaults
            kwargs['mode'] = 'constant'
            kwargs['constant_values'] = 0

        pad_width = ( [(before_len, after_len)]
                      + [(0, 0) for i in range(len(self.shape))] )

        #if self.symbolic:
            #if self._original_data is None:
            #    self._original_data = self._data
                    # Persistently store the current _data, because that's the handle
                    # to the input that will be used when compiling the function
        if self._original_data in shim.get_updates():
            # We've already updated the data array, so we need to update the update
            self._data = shim.pad(self._data, previous_tarr_shape + self.shape,
                                  pad_width, **kwargs)
            shim.add_update(self._original_data, self._data)
        else:
            # _data is still in the original state, so we change the underlying data
            self._data.set_value( shim.pad(self._data.get_value(borrow=True),
                                           previous_tarr_shape + self.shape,
                                           pad_width, **kwargs),
                                  borrow=True)


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

        if self.symbolic:
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

    def get_trace(self, component=None, include_padding='none', time_slice=None):
        """
        Return the series' computed data for the given component.
        Time points which have not yet been computed are excluded, such that
        the len(series.get_trace(*)) may be smaller than len(series). The
        return value is however guaranteed to be consistent with get_time_array().
        If `component` is 'None', return the full multi-dimensional trace

        Parameters
        ----------
        component: int, slice, iterable of ints
            Restrict the returned trace to these components.
        include_padding: 'none' (default) | 'before' | 'left' | 'after' | 'right' | 'all' | 'both'
            'none':
                Don't include the padding bins.
            'before' or 'left':
                Include the padding bins preceding t0.
            'after' or 'right:
                Include the padding bins following tn.
            'all' or 'both':
                Include the padding bins at both ends.

        time_slice:
            See get_time_array.
        """
        if not shim.isshared(self._cur_tidx):
            raise RuntimeError("You are in the midst of constructing a Theano graph. "
                               "Reset history {} before trying to obtain its time array."
                               .format(self.name))

        # if start is not None:
        #     start = self.get_t_idx(start)
        # else:
        #     padding_vals = [ 'none', 'before', 'left', 'after', 'right', 'all', 'both' ]
        #     if include_padding in ['none', 'after', 'right']:
        #         start = self.t0idx
        #     elif include_padding in padding_vals:
        #         # It's one of the other options
        #         start = 0
        #     else:
        #         raise ValueError("include_padding should be one of {}.".format(padding_vals))

        # if end is not None:
        #     stop = self.get_t_idx(end) + 1
        # else:
        #     if include_padding in ['none', 'before', 'left']:
        #         stop = self._cur_tidx.get_value() + 1
        #     elif include_padding in padding_vals:
        #         stop = min(self._cur_tidx.get_value() + 1, len(self._tarr))
        #     else:
        #         raise ValueError("include_padding should be one of {}.".format(padding_vals))
        time_slice = time_slice if time_slice is not None else slice(None, None)
        tslice = self._get_time_index_slice(time_slice, include_padding)

        assert(self._cur_tidx.get_value() >= tslice.stop - 1)

        if component is None:
            return self._original_data.get_value()[tslice]
        elif isinstance(component, (int, slice)):
            return self._original_data.get_value()[tslice, component]
        elif isinstance(component, Iterable):
            idx = (tslice,) + tuple(component)
            return self._original_data.get_value()[idx]
        else:
            raise ValueError("Unrecognized series component '{}' of type '{}'"
                             .format(component, type(component)))

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
        computation of the whole series. Internally, this calls `_compute_up_to('all')`,
        which means every time point (including padding) is recomputed.

        This history is returned to allow chaining operations.

        Returns
        -------
        This history instance
        """

        if ( source is None and self._cur_tidx != self._original_tidx):
            raise RuntimeError("Tried to call '.set()' on the history {} while "
                               "in the midst of building a Theano graph."
                               .format(self.name))

        if ( source is None
             and self._cur_tidx.get_value() >= self.t0idx + len(self) - 1 ):
            # Nothing to do
            return

        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        data = None

        tarr = self._tarr

        if source is None:
            # Default is to use series' own compute functions
            self.compute_up_to('end')

        elif isinstance(source, History):
            raise NotImplementedError
            # See fsGIF.core.get_meso_model for a first attempt at this

        elif (not hasattr(source, 'shape')
              and (shim.istype(source, 'float')
                   or shim.istype(source, 'int'))):
            # Constant input
            data = np.ones(tarr.shape + self.shape) * source

        else:
            if hasattr(source, 'shape'):
                # Input specified as an array
                if ( not shim.is_theano_object(source.shape)
                     and source.shape != tarr.shape + self.shape ):
                    raise ValueError("[Series.set] The given source series does not match the dimensions of this one.\n"
                                     "Source shape: {}\nThis history's shape: {}."
                                     .format(source.shape, tarr.shape + self.shape))
                data = source

            else:
                try:
                    # Input should be specified as a function
                    # TODO: Use integration
                    data = np.concatenate(
                                        [np.asarray(external_input(t),
                                                    dtype=shim.config.floatX)[np.newaxis,...] for t in tarr],
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
            if not shim.is_theano_object(data.shape):
                shim.check(data.shape == self._data.get_value(borrow=True).shape)
                shim.check(data.shape[0] == len(tarr))
                self._data.set_value(data, borrow=True)
            elif shim.isshared(data):
                shim.check(data.get_value(borrow=True).shape == self._data.get_value(borrow=True).shape)
                shim.check(data.get_value(borrow=True).shape[0] == len(tarr))
                self._data = data
            else:
                # We can't check that source shape matches (it's a Theano variable),
                # so we have to trust it
                self._data = data

        if not shim.is_theano_variable(self._data):
            self._original_tidx.set_value(len(tarr) - 1)
            self._cur_tidx = self._original_tidx
        else:
            # HACK If _data is not just a shared variable, its Theano graph
            # almost certainly depends on _original_tidx, so it should not be
            # updated
            self._cur_tidx = shim.shared(len(tarr) - 1,
                                         symbolic=self.symbolic)

        return self

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

            tidx = self.get_tidx(tidx)
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
            dtype = np.result_type(discretized_kernel.dtype, self.dtype)
            return shim.cast(self.dt64 * shim.sum(discretized_kernel[kernel_slice][::-1]
                                                  * shim.add_axes(self[hist_slice], dim_diff,
                                                                  -self.ndim),
                                                  axis=0),
                             dtype=dtype)
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
            dtype = np.result_type(discretized_kernel.dtype, self.dtype)
            retval = shim.cast(self.dt64 * shim.conv1d(self[:], discretized_kernel[kernel_slice],
                                                       len(self._tarr), dis_kernel_shape)[domain_slice],
                               dtype=dtype)
            shim.check(shim.eq(retval.shape[0], len(self)))
                # Check that there is enough padding after tn
            return retval


    #####################################################
    # Operator definitions
    #####################################################

    def _apply_op(self, op, b=None):
        if b is None:
            new_series = Series(self)
            new_series.set_update_function(lambda t: op(self[t]))
            new_series.set_range_update_function(lambda tarr: op(self[self.time_array_to_slice(tarr)]))
            new_series.add_input(self)
        else:
            # HACK Should write function that doesn't create empty arrays
            shape = np.broadcast(np.empty(self.shape), np.empty(b.shape)).shape
            new_series = Series(self, shape=shape)
            if isinstance(b, History):
                new_series.set_update_function(lambda t: op(self[t], b[t]))
                new_series.set_range_update_function(
                    lambda tarr: op(self[self.time_array_to_slice(tarr)],
                                    b[b.time_array_to_slice(tarr)]))
            else:
                new_series.set_update_function(lambda t: op(self[t], b))
                new_series.set_range_update_function(lambda tarr: op(self[self.time_array_to_slice(tarr)], b))
            if isinstance(b, HistoryBase) or shim.is_theano_variable(b):
                new_series.add_input([self, b])
            else:
                new_series.add_input(self)

        if ( self._original_tidx.get_value() >= self.tnidx
             and ( b is None
                   or not isinstance(b, HistoryBase)
                   or b._original_tidx >= b.tnidx ) ):
             # All op members are computed, so filling the result series is 1) possible and 2) cheap
             new_series.set()
        return new_series

    def __abs__(self):
        return self._apply_op(operator.abs)
    def __neg__(self):
        return self._apply_op(operator.neg)
    def __pos__(self):
        return self._apply_op(operator.pos)
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
    def __pow__(self, other, modulo=None):
        return self._apply_op(lambda a,b: pow(a, b, modulo), other)
    def __floordiv__(self, other):
        return self._apply_op(operator.floordiv, other)
    def __rfloordiv__(self, other):
        return self._apply_op(lambda a,b: b//a, other)
    def __mod__(self, other):
        return self._apply_op(operator.mod, other)


#######################################
# Views
#######################################

class DataView(HistoryBase):
    """
    Gives direct access to the history data.
    If the history is not fully computed, the returned history is truncated
    after the latest computed time point.

    Retrieving numerical data varies depending whether a
    history's data is stored as a NumPy, Theano shared or
    Theano variable. This abstracts out the calls, providing
    an interface as though the data was a NumPy object.

    TODO: - In __new__, allow to
            + Subclass proper history
            + Detect if hist is already a DataView, and return itself
              in that case
    """

    def __init__(self, hist):
        self.hist = hist
        # We don't initialize base class, but let __getattr__ take care of undefined attributes
        self._tarr = hist._tarr[:hist._original_tidx.get_value()+1]
        self.tn = self._tarr[-1]
        self._unpadded_length = len(self._tarr) - self.t0idx

    def __getitem__(self, key):
        res = self.hist[key]
        if not shim.is_theano_object(res):
            return res
        elif shim.isshared(res):
            return res.get_value()
        else:
            nwkey, key_filter, latest = self.hist._parse_key(key)
            res = self.hist._original_data.get_value()[nwkey]
            if key_filter is None:
                return res
            else:
                return res[key_filter]

    def __getattr__(self, name):
        return getattr(self.hist, name)

    def __len__(self):
        return len(self.hist)

    # Methods in HistoryBase must be explicitly redirected
    def get_time(self, *args):
        return self.hist.get_time(*args)
    def get_t_idx(self, *args):
        return self.hist.get_t_idx(*args)


sinn.common.register_datatype(History)
sinn.common.register_datatype(PopulationHistory)
sinn.common.register_datatype(Spiketrain)
sinn.common.register_datatype(Series)
