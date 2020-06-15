******
Models
******

Models are defined by subclassing :class:`sinn.models.Model`.

Model parameters
================

Models *must* define a :attr:`time` attribute of type `~sinn.axis.DiscretizedAxis`. Additional attributes can be divided into the following categories:

- parameters:
  Numeric or symbolic (shared) variables.
- kernels
  Instances of `~sinn.kernels.Kernel`.
- histories
  Instances of `~sinn.histories.History`.
- state variables
  A subset *histories*; if all these histories are computed up to ``t``, then the model can compute ``t+1``. See :ref:`model-state`.

These are defined by special classes at the top of your model::

  import theano_shim as shim
  import sinn.models as models
  from sinn.histories import TimeAxis, Series, AutoHist

  class MyModel(models.Model):
    class Config:
      rng = shim.typing.RandomStream

    time: TimeAxis

    class Parameters:
      a :float
      b :float = default_value
    params: Parameters

    class Kernels:
      κ :ExpKernel

    class Histories:
      x :Series  # Any History type is valid
      y :Series = AutoHist(name='y', template='x')

    class State:
      # Types must match those in Histories
      x :Series

A few things to note:
  - `sinn.models.Model` inherits from :ref:`Pydantic's <what-is-pydantic>` :class:`~pydantic.BaseModel` and uses the same format for defining model attributes. It changes the following defaults:

    + :attr:`extra` : ``'allow'``

  - :class:`Config` can be used to pass any additional configuration value to `pydantic.BaseModel.Config`, plus :ref:`a few others <model-cat-config>`.
  - :class:`Parameters`, :class:`Kernels`, :class:`Histories` and :class:`State` all implicitly inherit from Pydantic_'s :class:`~pydantic.BaseModel` (`sinn` does this for you).
  - The attributes :attr:`params`, :attr:`histories` and :attr:`kernels` are reserved: they are used to access the values associated to their associate categories. These values are also added to the model's global namespace, so that ``model.histories.x`` and ``model.x`` refers to the same object.
  - :class:`~sinn.histories.AutoHist` is a special `~sinn.Histories.History` type which is only valid within a model definition; it exists to get around the fact that histories depend on the time discretization, which is only known once a model is instantiated. It accepts the same parameters as the associated history type (in the example above, this would be `~sinn.histories.Series`). In addition, uses the model's :attr:`time` attribute to set the histories :attr:`time` axis.
  - The :data:`time` and :data:`params` attributes are added automatically if omitted, but for clarity it is recommended to include them in your model.

About names:
  - The variable in :class:`Parameters`, :class:`Kernels`, :class:`Histories` are added to the :class:`Model` global namespace, so make sure their names all differ.
  - The names :attr:`values`, :attr:`config` and :attr:`field` are reserved by *Pydantic* and should not be used to define model attributes.

Brief aside
-----------

.. _model-state:

State
^^^^^

TODO: What is state, and why do we need it.

.. _what-is-pydantic:

Pydantic
^^^^^^^^

TODO: Brief overview: annotations, declarative, later params can depend on earlier params, link to docs.

.. _Pydantic: https://pydantic-docs.helpmanual.io/

Special model classes
---------------------

Model categories define an implicit definition order, which satisfy the following requirements:

  - Kernels may depend on parameters;
  - Histories may depend on both parameters and kernels.

Attributes are therefore defined as follows:

  - All attributes in `Parameters`
  - All attributes in `Kernels`
  - All top-level model attributes
    This includes the mandatory `time` attribute.
  - All attributes in `Histories`

Within each of these categories, attributes are defined sequentially as in *Pydantic*.

.. _model-cat-config:

Config
^^^^^^

.. _model-cat-params:

- rng:


Parameters
^^^^^^^^^^

Kernels
^^^^^^^

Histories
^^^^^^^^^

Model methods
=============

Models should define an `initialize` method; this is called automatically on model creation after all parameters, kernels and histories have been set. It can also be called to reset a model, for example to start a new optimization run.
Models should also define an update function for each of their histories; continuing the example from above::

.. code-block:: python
   :force:

   […]
   from sinn.models import update_function

   class MyModel(models.Model):
     […]
     def initialize(self):
       self.x.pad(1)

     @update_function('y', inputs=['y']):
     def y_upd(self, tidx):
       return self.y[tidx-1] + self.x[tidx-1]*self.time.dt
     @update_function('x', inputs=['x']):
     def x_upd(self, tidx):
       return self.x[tidx-1] - self.y[tidx]*self.time.dt

`initialize`: optionally include :keyword:`initializer` keyword.
