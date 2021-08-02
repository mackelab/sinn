.. _models_devdocs:

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
- a random number generator (RNG)
  Update functions which require random numbers should expect this RNG as argument.
  (Note: Multiple RNGs are rarely needed, and are not well supported.)
- other models

Parameters and state are defined by special classes at the top of your model, while histories, kernels, RNG and submodels are listed at the top level. Parameters should be defined first, so that they are available when defining histories and kernels::

  import theano_shim as shim
  from sinn.models import Model, ModelParams
  from sinn.histories import TimeAxis, Series, AutoHist
  from mackelab_toolbox import typing

  class MyModel(Model):
    time: TimeAxis
    rng : typing.AnyRNG

    class Parameters(ModelParams):
      a: float
      b: float = default_value
    params: Parameters

    κ: ExpKernel

    x: Series  # Any History type is valid
    y: Series = AutoHist(name='y', template='x')

    class State:
      # Types must be Any
      x: Any
      y: Any

A few things to note:
  - `sinn.models.Model` inherits from :ref:`Pydantic's <what-is-pydantic>` :class:`~pydantic.BaseModel` and uses the same format for defining model attributes. It changes the following defaults:

    + :attr:`extra` : ``'allow'``

  - :class:`Config` can be used to pass any additional configuration value to `pydantic.BaseModel.Config`.
  - :class:`Parameters` *must* inherit from :class:`~sinn.models.ModelParams`, which itself inherits from Pydantic_'s :class:`~pydantic.BaseModel`.
  - The attribute :attr:`params` is reserved: it is used to access parameter values. Unless it clashes with an attribute in the model's global namespace, a parameter ``a`` can be equivalently accessed as either ``model.params.a`` or ``model.a``.
  - :class:`~sinn.histories.AutoHist` is a special `~sinn.Histories.History` type which is only valid within a model definition; it exists to get around the fact that histories depend on the time discretization, which is only known once a model is instantiated. It accepts the same parameters as the associated history type (in the example above, this would be `~sinn.histories.Series`). In addition, it uses the model's :attr:`time` attribute to set the histories :attr:`time` axis.
  - The :data:`time` and :data:`params` attributes are added automatically if omitted, but for clarity it is recommended to include them in your model.

About names:
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

The :class:`ModelParams` class
------------------------------

TODO


Model methods
=============

Models should define an :meth:`initialize` method; this is called automatically on model creation after all parameters, kernels and histories have been set. It can also be called to reset a model, for example to start a new optimization run. The :meth:`initialize` method must take one optional free-form argument :keyword:`initializer`; this can be e.g. a flag to select between initialization algorithms, or a dictionary of initialization values. You are free to ignore this value, but it should be in the signature, and the default should be ``None``.

Models should also define an update function for each of their histories. Continuing the example from above, this could look like::

.. code-block:: python
   :force:

   […]
   from sinn.models import update_function

   class MyModel(Model):
     […]
     def initialize(self, initializer=None):
       self.x.pad(1)

     @update_function('y', inputs=['y']):
     def y_upd(self, tidx):
       return self.y[tidx-1] + self.x[tidx-1]*self.time.dt
     @update_function('x', inputs=['x']):
     def x_upd(self, tidx):
       return self.x[tidx-1] - self.y[tidx]*self.time.dt

.. important:: If a model contains a left-padded history (a history with time points before ``t0``), that model must define an :meth:`initialize` method which fills all left-padded histories with data. This method can also be used to pre-compute kernels, or anything else which should be done when parameters change.
   After calling :meth:`initialize`, one should have ``model.cur_tidx == -1``.

Default values and initializers
===============================

TODO: Pydantic provides initialization in the form of the `@validator` decorator.

The :func:`@initializer` decorator
----------------------------------

TODO

:class:`AutoHist`
-----------------

TODO: Already mentioned above is the special :class:`AutoHist` default; for histories without dependencies on parameters, this allows to avoid the more verbose definition using the :func:`@initializer` decorator. [continue…]

Model instantiation
===================

TODO

.. code-block:: python
   θ = MyModel.Parameters(a=1, b=0.2)
   model = MyModel(params=θ)

Composing models
================

Multiple models can be combined. For example, we may want to model the external inputs separately from the dynamics. For this example, let's suppose that :class:`MyModel` defined above describes our dynamics. Then we can do::

.. code-block:: python
   class WhiteNoise(Model):
     time: TimeAxis
     rng: typing.AnyRNG

     class Parameters:
       σ: typing.FloatX

     ξ: AutoHist(name='ξ', shape=(1,), dtype='float64')

     @update_function('ξ')
     def ξ_upd(self, k):
       σ=self.σ; dt=self.ξ.dt
       rng.normal(avg=0, std=σ*shim.sqrt(dt))

   class FullModel(Model):
     external: WhiteNoise
     dynamics: MyModel

Note:

- That each model defines its own :class:`TimeAxis`.

TODO: Tie the two models together (atm external input is not seen by MyModel).

The instantiation is as you would expect::

.. code-block:: python
   θ_ext = WhiteNoise.Parameters(params=σ=1)
   ext_input = WhiteNoise(params=θ_ext)
   dyn_model = MyModel(params=θ, I=ext_input.ξ)
   model = FullModel(external=ext_input, dynamics=dyn_model)

Note how we tied the history :attr:`ξ` of :class:`ext_input` with history :attr:`I` of :class:`dyn_model`.

This way of combining submodels is quite flexible, and makes it easy to change for example the form of the input, without redefining an entirely new model.

.. Note::
   Models that are passed as arguments are *shallow-copied*. This means that their histories are untouched, so in the example above, ``model.dynamics.x is dyn_model.x`` would evaluate to ``True``, unless another value is assigned to ``model.x`` after the copy. However, the models themselves differ: ``model.dynamics is dyn_model`` always returns ``False``.
   There may be lingering issues with symbolic and compilation variables, as we progressively figure out the most intuitive way those should behave when copying. At present these are not preserved across copies, so for example ``model.dynamics.curtidx_var is not dyn_model.curtidx_var``.
