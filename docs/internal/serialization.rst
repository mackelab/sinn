*************
Serialization
*************

What should be serialized ?
===========================

In short, almost everything. :py:mod:`sinn` is intended to be used within larger data processing workflows composed of many distinct tasks. Being able to serialize and deserialize any :py:mod:`sinn` object makes it much easier to build modular workflows, which can be saved and restarted from disk at any point between tasks. Serialized objects are also trivially picklable, which greatly simplifies parallelization, since most libaries use :py:mod:`pickle` to send data to parallelized subprocesses.

The desire for easily serializable objects was the primary motivation for the move to using Pydantic_ throughout the code base: :py:mod:`pydantic`'s :py:class:`BaseModel` comes with a :py:meth:`json()` method which exports a model to a JSON string. :py:class:`BaseModel` already knows how to serialize most common data types, takes care of recursively serializing and deserializing nested objects, and provides hooks to configure custom JSON encoders and decoders. Thus the only thing we need to add are a few custom encoders for types not already recognized by :py:mod:`pydantic`. These can be found in :py:mod:`mackelab_toolbox.typing`.

The documentation for the workflow management package smttask_, which uses the same serialization mechanisms, provides descriptions for the customized serialization routines.

.. Note:: The custom JSON encoders are defined in the upstream package mackelab_toolbox_, since they can also be of use for other packages.

.. Note:: CAPITAL letters indicate quantities that are filled in. Quantities in [BRACKETS] are omitted if unspecified or equal to their default value.

.. _Pydantic: https://pydantic-docs.helpmanual.io
.. _mackelab_toolbox: https://github.com/mackelab/mackelab-toolbox
.. _smttask: https://github.com/alcrene/smttask/
