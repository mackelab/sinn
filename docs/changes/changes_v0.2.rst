# Version 0.2: The big Pydantic migration

This commit inaugurates the transition to sinn v0.2. It is a near complete rewrite of the entire stack, based on the lessons learned from v0.1. The because fundamental changes are:

  - Integration of `Pydantic <https://pydantic-docs.helpmanual.io>`_ throughout.
  - Extension of a few, dispersed axis functions into a coherent, full-featured axis & indexing module.
  - Consistent automatic compilation
  - Addition of unit tests
  - A documentation skeleton

## Integration of Pydantic

Every core object is now a Pydantic `BaseModel`. This has benefits in terms of specifying and validating intialization inputs, and allows a declarative specification of parameters. Declarative specifications are especially especially natural for `Model`, where composability and extensability are important features. The definition of model components (`History`, `Axis`, `Kernel`) as subclasses of `BaseModel` allows us to make full use of this composability.

However, the biggest advantages of using Pydantic are when embedding models within larger workflows. Pydantic objects can be easily (de)serialized, making it simple and reproducible to read/write parameter sets, model components or entire models to disk. Any special packing/unpacking code is within the object itself, allowing standardized codes for passing parameters and results between data processing stages.

## Axis module

The axis module was created to address a few issues:

  1. Sinn v0.1 defined different incompatible axis concepts, for manipulating time arrays (`History`, simulations) and rescaling, rebinning data (`AxisData`, Bayesian inference, data analysis).
  2. The time axis was fully enmeshed in the `History` class. It started as a simple second array, but grew into multiple specialized methods as needs arose.
  3. Conversions between time axes of different histories was unintuitive and error-prone.
  4. Being able to index a history by value instead of index was really useful: it allows to index different histories consistently without worrying about padding, and is convenient in interactive analysis sessions. However conversion to and from values brings a whole host of numerical rounding issues in simulation code. (In interactive sessions these weren't too important.)

The *axis.py* module collects all the functionality for time and data axes into a single consistent class. The more focussed module allowed for more precisely defined behaviour, and lightened the `History` class which had grown too large. The collection of pre-existing functionality represents maybe 20% of the new *axis.py* module.

The rest of the 80% adds the important new functionality of `AxisIndex` classes. Axis indices are structured in a hierarchy of types: similar to how Python's :mod:`datetime` defines *time* and *timedelta* objects, we have *axis* and *axisdelta* objects. We also distinguish between numeric and symbolic indices, a requirement for the redesigned Theano compilation algorithm. Finally, each axis defines its own index types ((absolute, delta) x (numeric, symbolic)) – this allows histories to transparently convert indices as required, solving point 3 above.

The complex relationships between all axis index types is captured with an abstract class hierarchy, enabling easy checking of index type at any level of granularity (any delta index type vs any index type corresponding to *this* axis).

The concepts of “axis index” and “data index” are now clearly separated. For an axis index, there is no wrapping: the position at `-1` is simply more to the left than `0`, provided it exists. This allowed us to avoid the issues with numerical errors, without losing the convenience of value indexing in interactive sessions.

## Consistent automatic compilation

Version 0.1 of sinn saw multiple iterations of algorithms for constructing the Theano functions declaratively defined in models. This culminated in the pattern of “anchored computational graphs“ and connected/disconnected histories. In version 0.2, all obsolete compilation code was removed, and the pattern made coherent throughout. This lead to massive simplifications to the code underlying `Model.advance()`, and huge improvements in the reliability of that code.

## Unit tests

All new code is now covered by unit tests. This both massively improves reliability, and provides for a basic form of documentation of the API.

## Documentation

Documentation is now a an incoherent set of pages written with varying levels of completeness. This is a big improvement over the previous state of no documentation to speak of.
