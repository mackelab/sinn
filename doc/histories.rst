History class
=============

Tutorial
--------
The History class, and the specialized classes that inherit from it, serves the purpose
of keeping two data structures in sync:

  - A data structure `data`
  - An array of times `time_array`

Time always corresponds to the data's first dimension, such that `len(data) == len(time_array)`.
The data at a particular point `t` in time, i.e. `data[t]`, is referred to as a *timeslice* within
this documentation.

Histories provide methods for time-value and time-index access to the data, automatic computation,
padding, convolutions, exporting to file-writable types, and other subtle conveniences that
ultimately save you from error-prone indexing operations.

The basic use case is to use a history to generate data. To do this, we first creates a history,
providing the `time_array` as argument::

    spikes = Spiketrain(name = 'spikes',
                        time_array = np.arange(0, 10, 0.01),
                        pop_sizes = (200, 100, 400),
                        dtype = 'bool')

For the purpose of this example let's define another history::

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

However, the following would introduce an unresolvable dependency when combined with `spike_update1`::
      
  def rate_update3(t):
      return [spikes[t][slc].mean() for slc in spikes.pop_slices]

Note the use of both floating point and integer indexing in the examples above. Integers are
interpreted as indices into the array, while floats are interpreted as times, which are converted
to indices using the internal `time_array`. You may use whichever is most convenient in a particular
situation. We recommend avoiding large numbers of time<->index conversions (e.g. at every simulation
time point), as there is a small cost associated with it.

Accessing a history timeslice that hasn't already been computed automatically triggers computation,
so one can do::

  sum_up_to_7 = spikes[:7.0].sum()

and be assured that this will return the correct value, triggering any required computation.
This may take a long time on the first call, but is just as fast as indexing on any subsequent
call. To compute a history at all time points, we can use any of the two following forms::

  spikes.compute_up_to('end')
  spikes.set()

(`spikes.set()` internally calls `compute_up_to('end')` when called with no argument).


Reference
---------

.. autoclass:: sinn.histories.Spiketrain()
   :members:
   :inherited-members:

   .. automethod:: __init__
   .. automethod:: __getitem__

   .. attribute:: pop_slices
      List of slices selecting population within a timeslice.
   .. attribute:: pop_sizes
      Tuple of population sizes.
   .. method:: PopTerm
      Call as `PopTerm()` on a correctly sized array to make it
      broadcastable with population timeslices.
                   
