*************
Serialization
*************

What should be serialized ?
===========================

In short, almost everything. :py:mod:`sinn` is intended to be used within larger data processing workflows composed of many distinct tasks. Being able to serialize and deserialize any :py:mod:`sinn` object makes it much easier to build modular workflows, which can be saved and restarted from disk at any point between tasks. Serialized objects are also trivially picklable, which greatly simplifies parallelization, since most libaries use :py:mod:`pickle` to send data to parallelized subprocesses.

The desire for easily serializable objects was the primary motivation for the move to using Pydantic_ throughout the code base: :py:mod:`pydantic`'s :py:class:`BaseModel` comes with a :py:meth:`json()` method which attempts to export a model to a JSON string. :py:class:`BaseModel` already knows who to serialize most common data types, takes care of recursively serializing and deserializing nested objects, and provides hooks to configure custom JSON encoders and decoders. Thus the only thing we need to add are a few custom encoders for types not already recognized by :py:mod:`pydantic`. These can be found in :py:mod:`mackelab_toolbox.typing` and are described below.

.. Note:: The custom JSON encoders are defined in the upstream package mackelab_toolbox_, since they can also be of use for other packages.

.. Note:: CAPITAL letters indicate quantities that are filled in. Quantities in [BRACKETS] are omitted if unspecified or equal to their default value.

.. _Pydantic: https://pydantic-docs.helpmanual.io
.. _mackelab_toolbox: https://github.com/mackelab/mackelab-toolbox

Custom serializations
=====================

Range
-----

Description
  A value returned by :py:func:`range()`.

JSON Schema

  .. code-block:: JSON

     {'type': "array",
      'description': "('range', START, STOP, [STEP])"}

Implementation
--------------

  .. code-block:: python

    def json_encoder(v):
        if v.step is None:
            args = (v.start, v.stop)
        else:
            args = (v.start, v.stop, v.step)
        return ("range", args)
      'description': "('range', START, STOP, [STEP])"

Slice
-----

Description
  A value returned by :py:func:`slice()`.

JSON Schema

  .. code-block:: JSON

     {'type': "array",
      'description': "('slice', START, STOP, [STEP])"}

Implementation

  .. code-block:: python

     def json_encoder(v):
         if v.step is None:
             args = (v.start, v.stop)
         else:
             args = (v.start, v.stop, v.step)
         return ("slice", args)

PintValue
----------

Description
  A numerical value with associated unit defined by Pint_.

Implementation

  .. code-block:: python

     def json_encoder(v):
         return ("PintValue", v.to_tuple())

QuantitiesValue
---------------

Description
  A numerical value with associated unit defined by Quantities_.

Implementation

  .. code-block:: python

     def json_encoder(v):
         return ("QuantitiesValue", (v.magnitude, str(v.dimensionality)))

DType
-----

Description
  An NumPy `data type object`_.

JSON Schema

  .. code-block:: JSON

     {'type': "str"}

Implementation

  .. code-block:: python

     def json_encoder(cls, value):
         return str(value)

NPType
------

Description
  A NumPy value, created for example with :code:`np.int8()` or :code:`np.float64()`.

JSON Schema

  .. code-block:: JSON

     {'type': "integer"|"number"}

Implementation

  .. code-block:: python

     def json_encoder(cls, value):
         return value.item()  #  Convert Numpy to native Python type

Array
-----

Description
  A NumPy array.

Design decisions
  NumPy arrays can grow quite large, and simply storing them as strings is not only wasteful but also not entirely robust (for example, NumPy's algorithm for converting arrays to strings changed between versions 1.13 and 1.14. [#fnpstr]_). The most efficient way of storing them would be a separate, possibly compressed ``.npy`` file. The disadvantage is that we then need a way for a serialized :py:mod:`sinn` object to point to this file, and retrieve it during serialization. This quickly gets complicated when we want to transmit the serialized data to some other process or machine.

  It's a lot easier if all the data stays in a single JSON file. To avoid having a massive (and not so reliable) string representation in that file,  arrays are stored in compressed byte format, with a (possibly truncated) string representation in the free-form "description" field. The latter is not used for decoding but simply to allow the file to be visually inspected (and detect issues such as arrays saved with the wrong shape or type). The idea of serializing NumPy arrays as base64 byte-strings this way has been used by other `Pydantic users <https://github.com/samuelcolvin/pydantic/issues/951>`_, and suggested by the `developers <https://github.com/samuelcolvin/pydantic/issues/692#issuecomment-515565389>`_.

  Byte conversion is done using NumPy's own :py:func:`~numpy.save` function. (:py:func:`~numpy.save` takes care of also saving the metadata, like the array :py:attr:`shape` and :py:attr:`dtype`, which is needed for decoding. Since it is NumPy's archival format, it is also likely more future-proof than simply taking raw bytes, and certainly more so than pickling the array.) This is then compressed using `blosc`_ [#f1]_, and the result converted to a string with :py:mod:`base64`. This procedure is reversed during decoding. A comparison of different encoding options is shown in :download:`numpy-serialization.nb.html`.

  .. Note:: Because the string encodings are ASCII based, it's important to save JSON files in ASCII format to avoid wasting the compression.

Implementation
--------------

  WIP


(The result of :py:func:`base64.b85encode` is ~6% more compact than :py:func:`base64.b64encode`.)

Custom JSON types
=================

We add a few container types to those recognized for generating a JSON Schema, so that they can be used without issue in class annotations. These types already have an encoder.

Number
------

Description
  Any numerical value ``x`` satisfying :code:`isinstance(x, numbers.Number)`.

JSON Schema

  .. code-block:: JSON

     {'type': "number"}

Integral
--------

Description
  Any numerical value ``x`` satisfying :code:`isinstance(x, numbers.Integral)`.

JSON Schema

  .. code-block:: JSON

     {'type': "integer"}

Real
----

Description
  Any numerical value ``x`` satisfying :code:`isinstance(x, numbers.Real)`.

JSON Schema

  .. code-block:: JSON

     {'type': "number"}


.. rubric:: Footnotes

.. [#fnpstr] NumPy v1.14.0 Release Notes, `“Many changes to array printing…” <https://docs.scipy.org/doc/numpy-1.16.0/release.html#many-changes-to-array-printing-disableable-with-the-new-legacy-printing-mode>`_
.. [#f1] For my purposes, the standard :py:mod:`zlib` would likely suffice, but since :py:mod:`blocsc` achieves 30x performance gain for no extra effort, I see no reason not to use it. In terms of compression ratio, with default arguments, :py:mod:`blosc` seems to do 30% worse than :py:mod:`zlib` on integer arrays, but 5% better on floating point arrays. (See :download:`numpy-serialization.nb.html`.) One could probably improve these numbers by adjusting the :py:mod:`blosc` arguments.



.. _Pint: https:pint.readthedocs.io
.. _Quantities: https://github.com/python-quantities/python-quantities
.. _`data type object`: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes
.. _`NumPy type`: https://docs.scipy.org/doc/numpy/user/basics.types.html
.. _`blosc`: http://python-blosc.blosc.org/
