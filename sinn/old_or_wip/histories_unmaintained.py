# -*- coding: utf-8 -*-

"""
This is a placeholder module for incomplete functionality, that turned out
not to be needed but which I may want to resurrect in the future.

Created on Fri Apr 24 2020

Author: Alexandre René
"""



# Spiketimes is currently really slow for long data traces (> 5000 time bins)
# Development efforts have been moved to Spiketrain; maybe in the future if we need
# to track actual times, we will resurrect development of this class
class Spiketimes(ConvolveMixin, PopulationHistory):
    """A class to store spiketrains.
    These are stored as times associated to each spike, so there is
    no well-defined 'shape' of a timeslice. Instead, the `shape` parameter
    is used to indicate the number of neurons in each population.

    .. Warning:: This development of this class has been put on hold as the Spiketrain
    is better suited to its original use case. I think it's a useful concept and
    want to see it developed further; I just don't currently need it.
    Thus expect to need to bring this class up to date before using it.
    """

    _strict_index_rounding = False

    def __init__(self, hist=_NoValue, name=None, *args, time_array=_NoValue,
                 pop_sizes=_NoValue, dtype=_NoValue, **kwargs):
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

        super().__init__(hist, name, time_array=time_array, pop_sizes=pop_sizes,
                         dtype=dtype, **kwargs)

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

    def set_update_function(self, func, *args, _return_dtype=None, **kwargs):
        # FIXME: I don't know when this was written, but shouldn't we return a time type, not index type ?
        #        Could have been copy-pasted from Spiketrain...
        if _return_dtype is None:
            super().set_update_functin(func, *args, self.idx_dtype, **kwargs)
        else:
            super().set_update_function(func, *args, _return_dtype, **kwargs)

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
            Should not correspond to more than one bin ahead of _sym_tidx.
        neuron_idcs: iterable
            List of neuron indices that fired in this bin.
        """
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        newidx = self.get_t_idx(tidx)
        if not theano.is_theano_variable(newidx):
            assert newidx <= self._num_tidx.get_value() + 1

        time = self.get_time(tidx)
        for neuron_idx in neuron_idcs:
            self._data[neuron_idx].append(time)

        # Set the cur_idx. If tidx was less than the current index, then the latter
        # is *reduced*, since we no longer know whether later history is valid.
        object.__setattr__(self, '_sym_tidx', newidx)
        if shim.is_theano_object(self._num_tidx):
            shim.add_update(self._num_tidx, self._sym_tidx)
        else:
            object.__setattr__(self, '_num_tidx', self._sym_tidx)

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
            self._compute_up_to('end')

        elif callable(source):
            raise NotImplementedError

        else:
            assert len(source) == len(self._data)
            for i, spike_list in enumerate(source):
                self._data[i] = spike_list

        self._num_tidx.set_value(self.t0idx + len(self) - 1)
        object.__setattr__(self, '_sym_tidx', self._num_tidx)
        return self


    def _convolve_single_t(self, kernel, t, kernel_slice):
        """Return the quasi-continuous time convolution with the set of
        spike times, i.e.
            ∫ spiketimes(s) * kernel(t-s) ds
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
        kernel_slice: float
            The kernel is considered zero outside `kernel_slice`; use this
            to truncate the kernel. Default is no truncation, or [0, ∞)

        The treatement of integral end points is to ensure that the interval
        over which the integral is evaluated has length 'end' - 'begin'. It
        also insures that integrals over successive intervals can be summed
        without duplicating the endpoints.

        Returns
        -------
        ndarray of shape 'popsize' x 'popsize'.
            It is indexed as result[from pop idx][to pop idx]

        """
        # TODO: To avoid iterating over the entire list, save the last `end`
        #       time and an array (one element per neuron) of the index of the latest
        #       spike before `end`. Iterations starting at `end` can then exclude all
        #       spikes before that point.
        #       Use `np.find` to get the `start` and `end` index, and sum between them

        # TODO: allow kernel to return a value for each neuron

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

        # For explanations of how summing works, see the Jupyter
        # notebook in the docs
        _data = self[start:stop]
        if kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1]:
            shim.check(kernel.shape[0] == len(self.pop_sizes))
            result = np.stack (
                     np.asarray(np.sum( f(t-s, from_pop_idx)
                             for spike_list in _data[self.pop_slices[from_pop_idx]]
                             for s in spike_list )).reshape(kernel.shape[0:1])
                     for from_pop_idx in range(len(self.pop_sizes)) ).T
                # np.asarray is required because summing over generator
                # expressions uses Python sum(), and thus returns a
                # scalar instead of a NumPy float
            return TensorWrapper(result,
                                 TensorDims(contraction=()))
            # TODO: don't contract pop axis here

        elif kernel.ndim == 1 and kernel.shape[0] == len(self.pop_sizes):
            result = shim.concatenate(
                  [ shim.stack( shim.asarray(shim.sum( f(t-s, from_pop_idx) for s in spike_list ))
                               for spike_list in _data[self.pop_slices[from_pop_idx]] )
                    for from_pop_idx in range(len(self.pop_sizes)) ] )
            return TensorWrapper(result, TensorDims())

        else:
            raise NotImplementedError

class SpiketrainCOO(ConvolveMixin, PopulationHistory):
    """
    This is the same as histories.Spiketrain, except that data are stored an
    a COO array.

    A class to store spiketrains, grouped into populations.

    These are stored in a sparse array where spikes are indicated by ones.
    Instead of the `shape` parameter, we use `pop_slices` to indicate the
    neurons associated to each population, from which `shape` is automatically
    deduced. Only 1d timeslices are currently supported (i.e. all populations
    are arranged along one axis).

    Internally spikes are stored in a coo array, for fast addition of new
    spikes. It is cast to csr for slicing. See the :py:mod:`scipy.sparse`
    documentation for a discussion of different sparse array types.

    There are still a few unsolved challenges preventing using this class in a
    Theano graph at the moment. Notably, Theano only supports csc and csr.
    Moreover, Theano docs warn against using sparse arrays within `scan()`, as
    would almost certainly be required. Still, since :py:class:`Spiketrain`
    converts its data to CSR/CSC for mathematical operations, it should suffice
    to add a basic COO class to the backend with only conversion methods, in
    order to get a functioning :py:class:`Spiketrain` class.

    For more serious inference work using sparse arrays, one may want consider
    switching the backend to either PyTorch or TensorFlow, both of which use the
    COO format instead of CSR/CSC for their sparse matrices.

    .. Note::
       Although this class was designed to simulate and store spike trains,
       since the data type and update functions are arbitrary, it can just as
       well be used for any kind of sparse data. Depending on the needs of the
       application, the convolution functions may need to be redefined.

    Parameters
    ----------

    _ Inherited from `History`
        + name     : str
        + time     : TimeAxis
        + :strike:`shape`  : [Removed by :py:class:`PopulationHistory`; see `pop_sizes`]
        + dtype    : numpy dtype
        + iterative: bool
        + symbolic : bool
        + init_data: [Replaced by :py:class:`Spiketrain`]
        + template : History

    pop_sizes: Tuple[int]
        Tuple of population sizes. Since the shape is calculated as
        ``(sum(pop_sizes),)``, this makes the `shape` argument redundant.
    init_data: ndarray, Optional
        n x N array, where n is arbitrary and N is the total number of units.
        The first n time points will be initialized with the data from init_data
        If not specified, the data is initialized to zero.
        Note that the initialized data is set for the first n time indices,
        so if padding is present, those will be before t0. Consequently, the
        typical use is to use n equal to the padding length (i.e. self.t0idx).
        If `init_data` does not provide a dtype, it must be given as argument.
    dtype: numpy dtype, Optional (default: 'int8')
        Type to use for storing the data. If both `init_data` and `dtype`
        are provided, data will be cast to `dtype`. If neither is provided,
        the 'int8' type is used, to be compatible with Theano bools.
    """

    dtype       : DType = np.dtype('int8')

    @History.update_function.setter
    def update_function(cls, f :HistoryUpdateFunction):
        """
        Spiketrain expects an index for its update function
        -> different return dtype than History.attach_update_function
        """
        assert isinstance(f, HistoryUpdateFunction)
        f._return_dtype = self.idx_dtype
        return f

    # Called by History.__init__
    # Like an @validator, returns the value instead of setting attribute directly
    def initialized_data(self, init_data=None):
        """
        Parameters
        ----------
        init_data: ndarray, optional
            If provided, `init_data` must be a 2d array with first axis
            corresponding to time and second to neurons. First axis may be
            shorter then the time array; in this case it is padded on the right
            with weros until its length is equal to `self.time.padded_length`.
            The second axis must match the number of neurons exactly.

            Must not be a symbolic value. If a shared value, no padding is
            performed, so the time dimension must match
            `self.time.padded_length` exactly.

        Returns
        -------
        Shared[ndarray]
            Value to store as `self._num_data`
        Shared[AxisIndex]
            Value to store as `self._num_tidx`
        """
        # shape = cls.get_shape_from_values(values)
        # pop_sizes, dtype, time = (values.get(x, None) for x in
        #     ('pop_sizes', 'dtype', 'time'))
        shape, pop_sizes, dtype, time = (
            self.shape, self.pop_sizes, self.dtype, self.time)
        nneurons = np.sum(pop_sizes)
        if init_data is None:
            data = scipy.sparse.coo_matrix((time.padded_length, nneurons),
                                           dtype=dtype)
            tidx_val = self.time.t0idx - 1
        else:
            assert shim.eval(init_data.shape[1]) == nneurons
            data_dtype = getattr(init_data, 'dtype', None)
            if dtype is None:
                if data_dtype is None:
                    raise ValueError("The provided data of type {} does have a 'dtype' "
                                     "attribute. In this case you must provide it to "
                                     "Spiketrain.initialize().".format(type(init_data)))
                else:
                    dtype = np.dtype(data_dtype)
            elif not np.can_cast(init_data.dtype, dtype):
                raise TypeError(f"Data (type {init_data.dtype}) cannot be "
                                f"cast to the specified dtype ({dtype}).")

            tidx_val = len(init_data)
            csc_data = shim.sparse.csc_matrix('spike train',
                                              shape=(time.padded_length, nneurons),
                                              dtype=dtype)
            csc_data[:tidx_val,:] = shim.sparse.csc_from_dense(init_data.astype(dtype))
            # This may throw an efficiency warning, but we can ignore it since
            # self._sym_data is empty
            csc_data.eliminate_zeros()
            data = csc_data.tocoo()
            # WARNING: This will break with Theano until/if we implement a
            #          coo matrix interface in theano_shim.
        cur_tidx = shim.shared(np.array(tidx_val, dtype=self.time.index_nptype),
                               name = 't idx (' + self.name + ')',
                               symbolic = self.symbolic)
        return data, cur_tidx

    def clear(self, init_data=None):
        """Spiketrains shouldn't just be invalidated, since then multiple runs
        would make them more and more dense."""
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        assert not shim.pending_update(self._num_data)
        object.__setattr__(self, '_sym_data', self.initialized_data(None))
        assert shim.graph.is_computable(self._sym_data)
        object.__setattr__(self, '_num_data', self._sym_data)
        super().clear()

    def get_data_trace(self, pop=None, neuron=None,
                  time_slice=slice(None, None), include_padding=False):
        """
        Return the spiketrain's computed data for the given neuron.
        Time points which have not yet been computed are excluded, such that
        the len(series.get_data_trace(*)) may be smaller than len(series). The
        return value is however guaranteed to be consistent with get_time_stops().
        If `component` is 'None', return the full multi-dimensional trace

        Parameters
        ----------
        pop: int
            Index of the population for which we want the trace. If unspecified,
            all neurons are returned, unless otherwise indicated by the 'neuron' parameter.
            Ignored if 'neuron' is specified.

        neuron: int, slice, array of ints
            Index of the neurons to return; takes precedence over 'pop'.

        time_slice:
        include_padding:
            See `DiscretizedAxis.data_index_slice`.

        Returns
        -------
        A csr formatted sparse array.
        """
        if self._sym_tidx is not self._num_tidx:
            raise RuntimeError("You are in the midst of constructing a Theano graph. "
                               "Reset history {} before trying to obtain its time array."
                               .format(self.name))

        tslice = self.time.data_index_slice(time_slice,
                                            include_padding=include_padding)

        tslice = slice(tslice.start, min(self.cur_tidx.data_index+1, tslice.stop))

        data_arr = self._num_data.tocsr()
        if neuron is None:
            if pop is None:
                return data_arr[tslice]
            else:
                return data_arr[tslice, self.pop_slices[pop]]
        elif isinstance(neuron, (int, slice)):
            return data_arr[tslice, neuron]
        elif isinstance(neuron, Iterable):
            idx = (tslice,) + tuple(component)
            return data_arr[idx]
        else:
            raise ValueError("Unrecognized spiketrain neuron '{}' of type '{}'"
                             .format(neuron, type(neuron)))


    def _getitem_internal(self, axis_index):
        """
        A function taking either an index or a slice and returning respectively
        the time point or an interval from the precalculated history.
        It does not check whether history has been calculated sufficiently far.

        .. Note:: This is an internal function – it implements the
           indexing interface. For most uses, one should index the history
           directly: ``hist[axis_index]``, which will check that the index is
           valid before calling this function.

        Parameters
        ----------
        axis_index: Axis index (int) | slice
            AxisIndex of the position to retrieve, or slice where start & stop
            are axis indices.

        Returns
        -------
        ndarray
            A binary array with last dimension equal to total number of neurons.
            Each element represents a neuron (populations are flattened).
            Values are 1 if the neuron fired in this bin, 0 if it didn't fire.
            If `key` is a scalar, array is 1D.
            If `key` is a slice or array, array is 2D. First dimension is time.
        """
        if shim.isscalar(axis_index):
            return self._sym_data.tocsr()[
                self.time.axis_to_data_index(axis_index)].todense().A[0]
        else:
            return self._sym_data.tocsr()[
                self.time.axis_to_data_index(axis_index)].todense().A

    def update(self, tidx, neuron_idcs):
        """
        Add to each neuron specified in `value` the spiketime `tidx`.

        Parameters
        ----------
        tidx: AxisIndex | Slice[AxisIndex] | Array[AxisIndex]. Possibly symbolic.
            The time index of the spike(s).
            The lowest `tidx` should not correspond to more than one bin ahead
            of _sym_tidx.
            The indices themselves may be symbolic, but the _number_ of indices
            must not be.
        neuron_idcs: iterable
            List of neuron indices that fired in this bin. May be a
            2D numeric array, a list of 1D numeric arrays, or a list of 1D
            symbolic arrays, but not a 2D symbolic array: the outer dimension
            must be not be symbolic.
            For convenience, also accepts a 1D array – this is understood as
            an array of indices, and it is wrapped with a list to add the
            time dimension.

        **Side-effects**
            If either `tidx` or `neuron_idcs` is symbolic, adds symbolic updates
            in :py:mod:`shim`'s :py:attr:`symbolic_updates` dictionary  for
            `_num_tidx` and `_num_data`.
        """

        # TODO: Fix batch update to something less hacky
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        time = self.time

        # neuron_idcs = shim.asarray(neuron_idcs)
        if shim.isscalar(neuron_idcs):
            raise ValueError(
                "Indices of neurons to update must be given as a "
                f"list of 1D arrays.\nIndices: {repr(neuron_idcs)}.")
        else:
            if shim.isarray(neuron_idcs):
                if neuron_idcs.ndim == 1:
                    # Single array of indices passed without the time dimension
                    # => add time dimension
                    neuron_idcs = [neuron_idcs]
                elif neuron_idcs.ndim != 2:
                    raise ValueError(
                        "Indices of neurons to update must be given as a "
                        f"list of 1D arrays.\nIndices: {repr(neuron_idcs)}.")
            neuron_idcs = [shim.atleast_1d(ni) for ni in neuron_idcs]
            if not all(ni.ndim == 1 for ni in neuron_idcs):
                raise ValueError(
                    "Indices of neurons to update must be given as a "
                    f"list of 1D arrays.\nIndices: {repr(neuron_idcs)}.")
        # From this point on we can assume that neuron_idcs can be treated as
        # a list of 1D arrays.
        # In particular, `len(neuron_idcs)` is valid and should correspond
        # to the number of time indices.

        # _orig_tidx = tidx
        if shim.isscalar(tidx):
            assert isinstance(tidx, time.Index)
            earliestidx = latestidx = tidx
            assert len(neuron_idcs) == 1
            tidx = shim.add_axes(tidx, 1)
            # neuron_idcs = [neuron_idcs]
        elif isinstance(tidx, slice):
            assert (isinstance(tidx.start, time.Index)
                    and isinstance(tidx.stop, time.Index))
            earliestidx = tidx.start
            latestidx = tidx.stop-1
            assert (len(neuron_idcs)
                    == shim.eval(tidx.stop) - shim.eval(tidx.start))
            tidx = shim.arange(tidx.start, tidx.stop, dtype=time.Index.nptype)
                # Calling `eval` on just start or stop makes better use of
                # its compilation cache.
        else:
            assert shim.isarray(tidx)
            earliestidx = shim.min(tidx)
            latestidx = shim.max(tidx)
            try:
                assert len(neuron_idcs) == shim.eval(tidx.shape[0])
            except shim.graph.TooCostly:
                pass
        try:
            assert shim.eval(earliestidx) <= shim.eval(self._sym_tidx) + 1
        except shim.graph.TooCostly:
            pass

        # Clear any invalidated data
        if shim.eval(earliestidx) <= shim.eval(self._sym_tidx):
            if shim.is_symbolic(tidx):
                raise TypeError("Overwriting data (i.e. updating in the past) "
                                "only works with non-symbolic time indices. "
                                f"Provided time index: {tidx}.")
            if shim.pending_update():
                raise TypeError(
                    "Overwriting data (i.e. updating in the past) only works "
                    "when the symbolic updates dict is empty. Current values "
                    f"in the updates dictionary: {shim.get_updates().keys()}.")
            # _orig_dataidx = self.time.data_index(_orig_tidx)
                # _orig_tidx is not artifically converted to index array
            csc_data = self._num_data.tocsc()
            csc_data[earliestidx.data_index+1:, :] = 0
            csc_data.eliminate_zeros()
            object.__setattr__(self, '_num_data', csc_data.tocoo())
            object.__setattr__(self, '_sym_data', self._num_data)

        dataidx = self.time.data_index(tidx)
        # FIXME: NumPy only
        #        Code below is COO-specific
        #        A symbolic COO interface storing data, col & row as symbolic
        #        tensors in a `_sym_data` container would have to define
        #        symbolic updates to the container (`shim.add_update(self_num_data)`)
        # assert len(dataidx) == len(neuron_idcs)
        if shim.config.use_theano:
            raise NotImplementedError
        for ti, idcs in zip(dataidx, neuron_idcs):
            # TODO: Assign in one block
            onevect = shim.ones(idcs.shape, dtype='int8')
                # vector of ones of the same length as the number of units which fired
            self._sym_data.data = shim.concatenate((self._sym_data.data, onevect))
                # Add as many 1 entries as there are new spikes
            self._sym_data.col = shim.concatenate((self._sym_data.col, idcs))
                # Assign those spikes to neurons (col idx corresponds to neuron index)
            self._sym_data.row = shim.concatenate((self._sym_data.row, ti*onevect))
                                               # (shim.add_axes(ti, 1, 'after')*onevect).flatten()))
                # Assign the spike times (row idx corresponds to time index)
        # Set the cur_idx. If tidx was less than the current index, then the latter
        # is *reduced*, since we no longer know whether later history is valid.
        if shim.eval(latestidx) < shim.eval(self._sym_tidx):
            # I can't imagine a legitimate reason to be here with a symbolic
            # time index
            assert not shim.is_graph_object(latest)
            logger.warning("Moving the current time index of a Spiketrain "
                           "backwards. Invalidated data is NOT cleared.")
        #     self._num_tidx.set_value( latestidx )
        #     assert self._sym_tidx is self._num_tidx
        # else:
        #     self._sym_tidx = latestidx
        # self._sym_tidx = latestidx

        # Add symbolic updates to updates dict
        data_is_symb = shim.is_symbolic(self._sym_data)
        tidx_is_symb = shim.is_symbolic(latestidx)
        if tidx_is_symb:
            assert data_is_symb  # Should never have symbolic tidx w/out symbolic data
            # assert self._num_tidx is not self._sym_tidx
            object.__setattr__(self, '_sym_tidx', latestidx)
            shim.add_update(self._num_tidx, self._sym_tidx)
        else:
            self._num_tidx.set_value(latestidx)
            object.__setattr__(self, '_sym_tidx', self._num_tidx)

        if data_is_symb:
            # But we *can* have symbolic data w/out symbolic tidx
            assert self._sym_data is not self._num_data
            shim.add_update(self._num_data, self._sym_data)
        else:
            object.__setattr__(self, '_num_data', self._sym_data)

    def pad(self, pad_left, pad_right=0):
        """
        Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.

        Parameters
        ----------
        pad_left: AxisIndexDelta (int) | value type
            Amount of time to add to before t0. If non-zero, all indices
            to this data will be invalidated.
        pad_right: AxisIndexDelta (int) | value type (default 0)
            Amount of time to add after tn.
        """
        if shim.is_graph_object(pad_left, pad_right):
            raise TypeError("Can only pad with non-symbolic values.")
        if shim.pending_update():
            raise RuntimeError("Cannot add padding while computing symbolic "
                               "updates.")
        assert self._sym_data is self._num_data

        before_len, after_len = self.time.pad(pad_left, pad_right)

        if before_len > 0 and self.cur_tidx >= self.t0idx:
            warn("Added non-zero left padding - invalidating the data "
                 f"associated with history {self.name}.")
            self.clear()
        newshape = (self.time.padded_length,) + self.shape
        data = self._num_data
        data.row += before_len
            # increment all time bins by the number that were added
        newdata = shim.sparse.coo_matrix(
            (data.data, (data.row, data.col)), shape = newshape )
        object.__setattr__(self, '_num_data', newdata)
        object.__setattr__(self, '_sym_data', self._num_data)

    def _convolve_single_t(self, discretized_kernel, tidx, kernel_slice):
        """
        Return the time convolution with the spike train, i.e.
            ∫ spiketrain(t - s) * kernel(s) ds
        with s ranging from -∞ to ∞  (normally there should be no spikes after t).
        The result is a 1d array of length Nneurons.
        Since spikes are delta functions, effectively what we are doing is
        sum( kernel(t-s) for s in spiketrain if s == 1 )

        .. Hint:: Typically the number of neurons is very large, but the
        kernel only depends on the population to which a neuron belongs.
        In this case consider using using a FactoredKernel, which when used in
        tandem with Spiketrain, more efficiently gets around the limitations
        of the scipy.sparse array.

        .. Note:: This method is an internal hook to allow histories to define
        the specifics of a convolution operation; it is called within the
        public-facing method `~History.convolve()`. There should be no reason
        for user code to call it directly instead of `~History.convolve()`.

        Parameters
        ----------
        discretized_kernel: History
            History, as returned by `self.discretize_kernel()`.
        tidx: Axis Index
            Time index at which to evaluate the convolution.
            This must be an axis index, not a data index.
        kernel_slice: slice
            The kernel is truncated to the bounds specified by
            this slice (thus implicitly set to zero outside these bounds).
            This achieved simply by indexing the kernel:
            ``discretized_kernel[kernel_slice]``.
        Returns
        -------
        TensorWrapper
        """
        # The setup of slicing is copied from Series._convolve_single_t

        assert isinstance(tidx, self.time.Index)
        kernel_slice = discretized_kernel.time.data_index(kernel_slice)
        assert shim.eval(kernel_slice.stop > kernel_slice.start)

        # tidx = self.get_t_idx(tidx)
        #
        # # Convert None & negative slices into positive start & stop
        # kernel_slice = self.slice_indices(kernel_slice)
        # # Algorithm assumes an increasing kernel_slice
        # shim.check(kernel_slice.stop > kernel_slice.start)

        hist_start_idx = (tidx.data_index
                          - kernel_slice.stop - discretized_kernel.idx_shift)
        hist_slice = slice(
            hist_start_idx,
            hist_start_idx + kernel_slice.stop - kernel_slice.start)
        assert shim.eval(hist_slice.start) >= 0

        hist_subarray = self._sym_data.tocsc()[hist_slice]

        assert shim.eval(discretized_kernel.ndim) <= 2
        sliced_kernel = discretized_kernel[kernel_slice]

        # To understand why the convolution is taken this way, consider
        # 1) That sparse arrays are matrices, so * is actually matrix multiplication
        #    (which is why we use the `multiply` method)
        # 2) `multiply` only returns a sparse array if the argument is also 2D
        #    (But sometimes still returns a dense array ?)
        # 3) That sparse arrays are always 2D, so A[0,0] is a 2D, 1x1 matrix
        #    Moreover, one can't add a 3rd dimension to a sparse array to
        #    broadcast along that dimension
        # 4) The .A attribute of a matrix returns the underlying array
        # 5) That we do not need to multiply by the step size (dt): This is a discretized
        #    train of Dirac delta peaks, so to get the effective height of the spike
        #    when spread over a bin we should first divide by dt. To take the convolution
        #    we should then also multiply by the dt, cancelling the two.
        # Addendum) Points 1-4 are specific to scipy.sparse (and by corollary
        #    theano.sparse). If we used pydata's sparse (which is COO), along
        #    with the sparse formats in TensorFlow or PyTorch (also COO), we
        #    could implement this more transparently.

        # We are currently limited to 2D kernels
        assert discretized_kernel.ndim in (1,2)

        if discretized_kernel.shape[-1] != self.shape[-1]:
            result = discretized_kernel.get_empty_convolve_result()
            # result = shim.tensor(discretized_kernel.shape, dtype=self.dtype)

            for inslc, outslc, kernslc in sliced_kernel.block_iterator(
                kernel_dims=None, include_time_slice=True):
                # Examples:
                # kernslc = (:, i1:i2)          # include_time_slice prepends ':'
                # outslc = (..., i1:i2, i2:i3)  # all in/outslc prepend '...'
                s = hist_subarray[inscl]
                κ = sliced_kernel[kernslc]
                result[outslc] = s.multiply(κi[::-1])
        else:
            result = hist_subarray.multiply(sliced_kernel[::-1])

        return TensorWrapper(result,
            TensorDims(contraction=discretized_kernel.contravariant_axes))
