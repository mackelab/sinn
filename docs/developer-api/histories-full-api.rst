:py:mod:`histories` module
===========================

Hints and guidelines
--------------------

Time index tests
^^^^^^^^^^^^^^^^

A key function of the `History` class is to compare a requested time index with its internal cache, and determine whether the data at that time index needs to be computed. How does this work with symbolic values ? The answer is that only the *data* are symbolic , not the *axes*. Symbolic time indices are allowed to have only one symbolic input – `self._num_tidx` – which is in fact a shared variable (i.e. it is associated to a concrete numerical value). So any time index can be evaluated with `shim.eval()` – the obtained value will depend on the current value of `self._num_tidx`, but *differences* between these values do not. So tests like::

    shim.eval(symbolic_idx) <= shim.eval(self._sym_tidx)

always make sense: the difference between the left and right-hand side is conserved, even if their numerical values change.

One might wonder whether calling `shim.eval` everywhere like this is wise, given the cost of compilation. This is actually fine for a few reasons:
  - Only simply index arithmetic is allowed, so the compilation time is quite short (although still perceptible).
  - Theano caches compilations, so after the first one this is as fast as a normal function call. This is the reason it is preferable to do ``shim.eval(_sym_tidx) - 1`` over ``shim.eval(_sym_tidx - 1)``.
  - These compilation costs are only incurred when building the graph – they don't show up in the final computational graph, and so don't affect runtime.

Full API
--------

.. See https://stackoverflow.com/a/30783465

.. rubric:: History Classes

.. autosummary::
   :toctree: _autosummary

   sinn.histories.History
   sinn.histories.Series
   sinn.histories.Spiketrain

.. rubric:: Other classes

.. autosummary::
   :toctree: _autosummary

   sinn.histories.TimeAxis
   sinn.histories.HistoryUpdateFunction

.. rubric:: History methods

.. autosummary::
   :toctree: _autosummary

   sinn.histories.History.__call__
   sinn.histories.History.__getitem__
   sinn.histories.History.__setitem__
   sinn.histories.History.update
   sinn.histories.History.clear
   sinn.histories.History.lock
   sinn.histories.History.unlock
   sinn.histories.History.theano_reset
   sinn.histories.History.truncate
   sinn.histories.History.align_to
   sinn.histories.History.interpolate
   sinn.histories.History.time_interval
   sinn.histories.History.index_interval
   sinn.histories.History.get_time
   sinn.histories.History.get_tidx
   sinn.histories.History.convolve

   sinn.histories.History._compute_up_to
   sinn.histories.History.get_time_stops
   sinn.histories.History._getitem_internal
   sinn.histories.History._is_batch_computable





.. rubric:: History attributes

.. autosummary::
   :toctree: _autosummary

   sinn.histories.History.copy
   sinn.histories.History.copy


.. rubric:: History: Pydantic methods and validators

.. autosummary::
   :toctree: _autosummary

   sinn.histories.History.copy
   sinn.histories.History.parse_obj
   sinn.histories.History.default_name
   sinn.histories.History.normalize_dtype
   sinn.histories.History.default_symbolic
   sinn.histories.History.initialized_data


.. .. autoclass:: sinn.histories.TimeAxis()
..    :members:
..
..    .. automethod:: __init__
..
.. .. autoclass:: sinn.histories.HistoryUpdateFunction()
..    :members:
..


   .. rubric:: Attributes


   .. rubric:: Methods

   .. autoautosummary:: sinn.histories.History
      :methods:

.. .. automethod:: __init__
.. .. automethod:: __getitem__
.. .. automethod:: __call__

   .. auto
..
.. .. autoclass:: sinn.histories.Series(History)
..    :members:
..
.. .. autoclass:: sinn.histories.PopulationHistory(History)
..    :members:
..
..    .. automethod:: __init__
..
.. .. autoclass:: sinn.histories.Spiketrain(PopulationHistory)
..    :members:
..
..    .. automethod:: __getitem__

   .. .. attribute:: pop_slices
   ..    List of slices selecting population within a timeslice.
   .. .. attribute:: pop_sizes
   ..    Tuple of population sizes.
   .. .. method:: PopTerm
   ..    Call as `PopTerm()` on a correctly sized array to make it
   ..    broadcastable with population timeslices.
