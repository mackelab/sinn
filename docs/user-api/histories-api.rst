User API – Histories
===========================

.. autoclass:: sinn.histories.TimeAxis()
   :noindex:
.. autoclass:: sinn.histories.HistoryUpdateFunction()
   :noindex:
.. autoclass:: sinn.histories.History()
   :noindex:
.. autoclass:: sinn.histories.Series(History)
   :noindex:

   .. automethod:: initialized_data
      :noindex:
   .. automethod:: update
      :noindex:

.. autoclass:: sinn.histories.PopulationHistory(History)
   :noindex:
.. autoclass:: sinn.histories.Spiketrain(PopulationHistory)
   :noindex:


.. .. autoclass:: sinn.histories.TimeAxis()
..    :members:
..    :inherited-members:
..
..    .. automethod:: __init__
..
.. .. autoclass:: sinn.histories.HistoryUpdateFunction()
..    :members:
..
.. .. autoclass:: sinn.histories.History()
..    :members:
..    :inherited-members:
..
..    .. automethod:: __init__
..    .. automethod:: __getitem__
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
