.. _histories_devdocs:

*********
Histories
*********

.. highlight:: python

The generic History class provides a general interface for storing data in various formats along with a time axis, along with an update function. The data structure is initialized empty, and filled as needed by calling the update function when a data point is requested.

A key assumption is that the update function is **causal**: it may depend on earlier data points (which should then be initialized before the first update call), but must not depend on the value at later points.

:py:class:`History` tracks its current time with integer :py:attr:`cur_tidx` corresponding to its time axis: data values up to and included that time point are considered computed, and are simply retrieved from storage when requested. Accessing a data value later than :py:attr:`cur_tidx` triggers computation of **all** values **up to** `cur_tidx`. (The planned :py:class:`LaggedSeries` subclasses would relax this.)

In general one does not instantiate the :py:class:`History` class directly, but one of its subclasses. Currently the following are available:

  - :py:class:`Series`: Store data in a Numpy array
    Useful for data and post mortem analysis, but gets really heavy when there are many histories.
  - :py:class:`Spiketrain`: Store data in ``bool`` in a sparse array.
  - (Planned) :py:class:`LagFreeSeries`: Store only the most recently computed time.
  - (Planned) :py:class:`LagSeries`: Store only the last ``n`` time steps.

If none of these suit your need, you can also [define your own history type](#defining-your-own-history-type).


Usage tutorial
==============

.. TODO:: This section is completely out of date.

The History class, and the specialized classes that inherit from it, serves the purpose of keeping two data structures in sync:

  - A data structure `data`
  - A set of times `time`, stored as an :class:`~sinn.axis.Axis`.

Time always corresponds to the data's first dimension, such that :code:`len(data) == len(time_array)`. The data at a particular point ``t`` in time, i.e. :code:`data[t]`, is referred to as a *timeslice* within this documentation.

Histories provide methods for time-value and time-index access to the data, automatic computation, padding, convolutions, exporting to file-writable types, and other subtle conveniences that ultimately save you from error-prone indexing operations.

The basic use case is to use a history to generate data. To do this, we first create a history,
providing the :py:attr:`time_array` as argument

.. code-block:: python

  spikes = Spiketrain(name = 'spikes',
                      time_array = np.arange(0, 10, 0.01),
                      pop_sizes = (200, 100, 400),
                      dtype = 'bool')

For the purpose of this example let's define another history

.. code-block:: python

   rate = Series(spikes,
                 name = 'rate',
                 shape = (3,)
                 dtype = 'float64')

Here we've used the first history as template to the second; this ensures they both share the
time array.

We then define an update function. An update function always has the same signature: it expects
one variable (`t`: time), and should be able to accept either a scalar `t` or an array
of time values. The value of `t` may represent either an index or an actual time; if you need
either quantity, use the `get_t_idx()` or `get_time()` methods to ensure it is of the right form.
The update function may access any variable within the scope. This includes circular dependencies
that loop back to the history itself, as long the it is indexed at an earlier time. For instance,
the following are all acceptable update functions::

   def rate_update1(t):
       return [spikes[t-0.1][slc].mean() for slc in spikes.pop_slices]

   def rate_update2(t):
       tidx = spikes.get_t_idx(t)
       return [spikes[tidx-10][slc].mean() for slc in spikes.pop_slices]

   def spike_update1(t):
       return np.concatenate( [ np.random.binomial(1, r, size)
                                for r, size in zip(rate[t], spikes.pop_sizes ] )

However, the following would introduce an unresolvable dependency when combined with :py:func:`spike_update1`::

   def rate_update3(t):
       return [spikes[t][slc].mean() for slc in spikes.pop_slices]

Note the use of both floating point and integer indexing in the examples above. Integers are
interpreted as indices into the array, while floats are interpreted as times, which are converted
to indices using the internal ``time_array``. You may use whichever is most convenient in a particular
situation. We recommend avoiding large numbers of time<->index conversions (e.g. at every simulation
time point), as there is a small cost associated with it.

Accessing a history timeslice that hasn't already been computed automatically triggers computation,
so one can do::

   sum_up_to_7 = spikes[:7.0].sum()

and be assured that this will return the correct value, triggering any required computation.
This may take a long time on the first call, but is just as fast as indexing on any subsequent
call. To compute a history at all time points, we can use any of the two following forms::

   spikes._compute_up_to('end')
   spikes.set()

(:code:`spikes.set()` internally calls :code:`_compute_up_to('end')` when called with no argument).

Retrieval vs evaluation
-----------------------

A history ``hist`` distinguishes between *retrieval* (indicated by square brackets ``[]``) and *evaluation* (indicated by round brackets ``()``)

  - ``hist[tidx]`` will return the value of ``hist`` at position ``tidx``. If that value has not already been computed, an `IndexError` is raised.
  - ``hist(tidx)`` will also return the value of ``hist``. If the value has already been computed, it is simply returned – in this case ``hist(tidx)`` is equivalent to ``hist(tidx)``. However, in the opposite case, instead of raising an error, the history's :meth:`update_function()` is called to fill it up to `tidx`, and then the value is retrieved. The round brackets are meant as a indicator that this may trigger an expensive function call.

In general, it is recommended using ``()`` when specifying a model's update equations, and ``[]`` in post-simualtion analysis. This communicates intent, and avoids or catches errors.

.. _hist-tutorial-pub-api:

Public API
==========
The following attributes and methods are provided by `History` and thus
guaranteed to be defined in all subclasses:

Attributes
  name       : str
      Unique identifying string
  shape      : int tuple
      Shape at a single time point. Full data shape should be `(T,) + shape`,
      where `T` is the number of time steps.
  time       : `TimeAxis`
      Underlying `Axis` object describing the time axis.
  idx_dtype    : numpy integer dtype.
      Type to use for indices within one time slice.
  locked       : bool
      Whether modifications to history are allowed. Modify through method

Properties
  trace          :
      Unpadded data. Calls self.get_data_trace() with default arguments.
  time_stops     : `ndarray`
      Unpadded time array. Calls `self.time.stops_array(padded=False)`.
  Access to specified `time` properties:
      t0           : floatX
          Time at which history starts
      tn           : floatX
          Time at which history ends
      dt           : floatX
          Timestep size
      dt64         : float64.
          Timestep size; guaranteed double precision, useful for index calculations
      tidx_dtype   : numpy integer dtype
          Type to use for time indices.

Methods
  lock             : Set the locked status to `True`
  unlock           : Set the locked status to `False`


Defining your own history type
==============================

If you want to subclass one of existing `History` subclasses (e.g. `Series`),
just proceed as usual by inheriting from the base class and and adding your
desired methods. If you want to add initialization parameters, note that
histories are implemented as Pydantic_ models, so additional parameters
should be specified as class attributes (rather than by specializing
`__init__()`).

If subclassing `History` directly, the following attributes and methods **must**
be provided (see :ref:`below <subclass_template>`) for method templates):

- Attributes:


- Methods:

  - `initialized_data()`
  - `_getitem_internal()`
  - `update()`
  - `pad()`
  - `get_data_trace()`
  - `_compute_up_to()`

The following methods **may** also be provided:

  - `_compute_range()`
  - `_convolve_single_t()` (Required for convolutions)
  - `_convolve_batch()`

.. Note::
   All methods which modify the history (update, set, clear, _compute_up_to)
   must raise a RuntimeError if `lock` is True.

In addition to the attributes listed in the :ref:`public API <hist-tutorial-pub-api>`, the
following attributes and methods are also made available by the `History`
base class:

- Attributes:

  + `_sym_tidx`   : `time.tidx_dtype`
     Tracker for the latest time bin for which we know history.
  + `_num_tidx`   :
     | For Numpy histories, same as `_sym_tidx`.
     | For Symbolic histories, a handle to the tidx variable, which is to be updated with the new value of `_sym_tidx`. (See `_num_data` above, and :ref:`symb-upds` below.)
  + `_sym_data`: Where the actual data is stored.
     A shared variable. The type can be any NumPy variable, but NumPy arrays must be wrapped as a shim.shared variable. Some histories implement this as a tuple of arrays (e.g. `~sinn.histories.Spiketrain`)
  + `_num_data`: For NumPy histories, the same as `_sym_data`.
     For symbolic histories, an handle to the shared variable, which is to be updated with the new value of `_sym_tidx`. (See :ref:`symb-upds`.)
  + `update_function` : `~sinn.history.HistoryUpdateFunction`
     Function taking a time and returning the history at that time

- Methods:

  + __getitem__
     Calls `self.retrieve()`
  + __setitem__
     Calls `self.update()`, after converting times in a
     key to time (axis) indices, and `None` in a slice to the
     appropriate axis index. Thus `update()` only needs
     to implement operations on axis (not data !) indices.

.. _symb-upds:

Symbolic updates
----------------

  **TODO** `_num_tidx`/`_num_data` always points to numeric data.
  Symbolic updates are accumulated in `_sym_tidx`/`_sym_data`.

.. _Pydantic: https://pydantic-docs.helpmanual.io/usage/models/


.. _subclass_template:

Subclass method templates
-------------------------


The following method templates are intended as a guide; although I try to keep them up to date, if in doubt, always have a look at how these method are implemented in the existing history subclasses. :py:class:`~sinn.histories.Series` is generally the most highly tested.


.. code-block:: python
   :force:

    def initialized_data(self, data=None):
        """
        Create and return the structure which will hold the data.
        If `data` is provided, use that to initialize the structure.

        Parameters
        ----------
        data: hist data type | other coercible type(s)
          Must accept a fully formed data object, as would happen when calling
          :code:`hist(**otherhist)`. It is OK (and probably necessary) to
          create a new symbolic variable to which to attach the data values.
          May also accept additional types (for example, initialization
          data in a list)

        Returns
        -------
        Shared[hist data object]
          Use :py:func:`shim.shared` to return an appropriate shared variable.
        """
        […]

    def retrieve(self, key):
        """
        A function taking either an index or a slice and returning respectively
        the time point or an interval from the precalculated history.
        It does not check whether history has been calculated sufficiently far.

        .. Note:: This is considered an internal function – it implements the
        indexing interface. For most uses, one should index the history
        directly: ``hist[axis_index]`` instead of ``hist.retrieve(axis_index)``.

        Parameters
        ----------
        axis_index: Axis index (int) | slice
            AxisIndex of the position to retrieve, or slice where start & stop
            are axis indices.

        Returns
        -------
        ndarray
        """
        if shim.istype(key, 'int'):
            […]
        elif isintance(key, slice):
            […]
        else:
            raise ValueError(
              "Key must be either an integer or a splice object.")

    def update(self, tidx, value):
        """Store the a new time slice.

        The implementation of this function must

        1) Update the value of `self._sym_data`
        2) Update the value of `self._sym_tidx`
        3) Call `shim.add_update(self._num_data, self._sym_data)` and
           `shim.add_update(self._num_tidx, self._sym_data)`.

        Parameters
        ----------
        tidx: AxisIndex or slice(int, int). Possibly symbolic.
            The time index at which to store the value.
            If specified as a slice, the length of the range should match
            value.shape[0].
        value: timeslice
            The timeslice to store. Format is that same as that returned by
            self._update_function


        **Side-effects**
            If either `tidx` or `neuron_idcs` is symbolic, adds symbolic updates
            in :py:mod:`shim`'s :py:attr:`symbolic_updates` dictionary  for
            `_num_tidx` and `_num_data`.
        """
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        […]

    def pad(self, before, after=0):
        """
        Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.

        Parameters
        ----------
        before: float
            Amount of time to add to before t0. If non-zero, All indices
            to this data will be invalidated.
        after: float (default 0)
            Amount of time to add after tn.
        """
        self.pad_time(before, after)
        […]

    def get_data_trace(self, **kwargs):
        """
        Return unpadded data.

        All arguments must be optional. This function is meant for data
        analysis and plotting, so the return value must not be symbolic.
        Typically this means that `get_value()` should be called on `_data`.
        """
        […]

    def _convolve_single_t(self, discretized_kernel, tidx, kernel_slice):
        """
        Returns
        -------
        TensorWrapper
        """
        […]

    # TODO
    def _compute_range([…]):  # See `__compute_up_to`
        """
        Function taking an array of consecutive times and returning an
        array-like object of the history at those times.
        NOTE: It is important that the times be consecutive (no skipping)
        and increasing, as some implementations assume this.
        """
        […]
