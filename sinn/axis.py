"""
The `Axis` objects serve as the link between physical and digital dimensions.
They form the 'frame' around a data object, and provide specialized functions
for index arithmetic. For more information, see :doc:`/axis`.

Important notes for implementers
================================

Regarding symbolic arguments
----------------------------

In contrast to much of `sinn`, the `index_range` underlying a `*Mapping` **must
not be symbolic**. This means that you can safely assume that operations like
``mapping.index_range[0]`` return a concrete value. Having concrete axes even
on otherwise symbolic data tremendously simplifies index arithmetic (which is
already hard enough as it is) and makes the compiler's job easier. Essentially,
any internal variable to a mapping or axis is concrete.

Nevertheless, external arguments may still be symbolic. So for the following
functions, make sure that operations on arguments are either appropriately
passed through `shim` or explicitly disallowed and guarded by a check using
`shim.is_graph_object()`:

    - `data_index()`
    - `index()`
    - `index_interval()`

Todo
^^^^
I would really like ``mapping.index_range[i]`` to guarantee that the return
type is `AxisIndex` (rather than plain ``int``).

The `AxisIndex` type
--------------------
The `AxisIndex` type (as retrieved by `mapping.Index`) is in fact a (nearly)
empty abstract base class. Two index types, for concrete and symbolic indices,
are registered as subclasses to `mapping.Index`; this allows `isinstance`
checks to work as expected.
The abstract `Index` implements `__new__`, which checks whether any argument
is symbolic and calls the constructor of the appropriate index type. This
allows for transparent casting to the index type: `mapping.Index(x)` will
work for both symbolic and numeric `x`.

(Planned) The `AxisIndexDelta` and `AxisIndex` are mixin classes, and the
construction of an `Index` type attached to a mapping is equivalent to:

    class Index(AxisIndex, np.int32):
        […]

(Current) The `AbstractAxisIndex` class is an abstract class, to which newly
created `Index` types are registered.

    class Index(np.int32):
        […]
    AbstractAxisIndex.register(Index)

Todo
^^^^
- An `AxisIndex` points back to its associated `Axis` (through its `Axis`
  attribute), but it would make a lot more sense for it to point to the
  associated `Mapping` instead. Because now `AxisIndex`, `Mapping` and `Axis`
  are all mutually entangled (and need special `copy()` functions to maintain
  their references), but at least `Axis` could be disentangled from that lot.
- It would be really helpful when debugging if the printed representation of
  an Axis included what it was attached to (i.e. model or history name).

"""

# FIXME: There is a lot of duplication between `index()` and `data_index()`

from __future__ import annotations
from warnings import warn
import abc
from abc import abstractmethod
from enum import Enum, auto
from collections import namedtuple
import inspect
import numbers
import numpy as np

import mackelab_toolbox as mtb
import mackelab_toolbox.units
from typing import Callable, Optional, Any, Union, Type, ClassVar
# from types import FunctionType
from mackelab_toolbox.typing import (
    Range, Sequence, Number, Integral, Real, NPType, Array, FloatX, AnyUnitType)
from pydantic import (
    validator, root_validator, BaseModel, ValidationError, Field)

import theano_shim as shim

from mackelab_toolbox.utils import (
    comparedicts, int_if_close, rgetattr, Singleton, flatten)
from mackelab_toolbox.transform import Transform, Bijection
from sinn import config
import sinn

# The unitless object is assigned as the `unit` to an axis with no units.
# This allows us to test for presence of units simply writing
# ``axis.units is unitless``.
# Instantiating with ``1`` allows us to use ``value * self.unit`` wherever
# we want.
# The use of the :class:`~utils.Singleton` metaclass ensures that ``is``
# checks are always valid. It is important to use ``is`` and not ``==``
# because a lot of things can be equal to 1.
UnitlessT = mtb.units.UnitlessT
unitless = mtb.units.unitless

# Unit conversion methods; one `check` and one `convert` method are attached
# to the the Axis object by `set_unit_methods`
# NOTE: I ran into problems when I put these functions into the Axis class
def _unit_check_factory(axis, unit_library):
    """
    The `unit_check` method returns True if the provided value is a float,
    and compatible with the axis unit type.
    """
    axisunit = axis.unit
    if unit_library == 'pint':
        if hasattr(axisunit, 'u'):
            # Replace Quantity by Unit
            axisunit = axisunit.u
        def unit_check(*values):
            return all(hasattr(v, 'compatible_units')
                       and axisunit in v.compatible_units() for v in values)
    elif unit_library == 'quantities':
        axisunitsimpdim = axisunit.simplified.dimensionality
        def unit_check(*values):
            return all(hasattr(v, 'simplified')
                       and (v.simplified.dimensionality == axisunitsimpdim)
                       for v in values)
    else:
        assert axisunit is unitless
        def unit_check(*values):
            """Returns False if any value is of a recognize unit type."""
            return all(not isinstance(v, numbers.Integral)  # Integers reserved for index
                       and not hasattr(v, 'magnitude')
                       for v in values)
            # raise TypeError("Unable to check units: axis unit '{}' unrecognized."
            #                 .format(type(axisunit)))
    return unit_check
def _unit_convert_factory(axis, unit_library):
    axisunit = axis.unit
    if unit_library == 'pint':
        def unit_convert(value):
            if isinstance(value, list):
                return [unit_convert(v for v in value)]
            elif isinstance(value, tuple):
                return tuple(unit_convert(v for v in value))
            return value.to(axisunit)
    elif unit_library == 'quantities':
        def unit_convert(value):
            if isinstance(value, list):
                return [unit_convert(v for v in value)]
            elif isinstance(value, tuple):
                return tuple(unit_convert(v for v in value))
            return value.rescale(axisunit)
    else:
        assert axisunit is unitless
        def unit_convert(value):
            # No conversion; as for other conversion methods, we presume that
            # `unitcheck` was called on `value`, and therefore it has no units
            return value
            # raise TypeError("Unable to convert units: axis unit '{}' unrecognized."
            #                 .format(type(axisunit)))
    return unit_convert

class Axis(BaseModel, abc.ABC):
    """
    Abstract axis class. Only stores label, unit, and optional min and max
    values. For an axis with defined stops (as needed for must numerical
    applications), use `DiscretizedAxis`.

    **Transformed axes**
       It is possible to attach a
       :py:class:`mackelab_toolbox.transform.Bijection` (φ,φ⁻¹) to the axis, such
       that if x1, x2 are stops along the axis, `axis.transformed` has stops
       φ(x1), φ(x2)… If no sensible unit can be assigned to the transformed axis
       (e.g. if the φ=log), then it will units of 'None'.
       axis.transformed.transformed is guaranteed to be idempotent.

       The motivation case for this functionality is parameter optimization,
       where a log transform is often applied to strictly positive data to make
       their domain the entire real line. In particular, we want to be able to
       convert between bin edges and centers in the transformed space (where
       optimization was performed, and where the distribution of data
       is ostensibly less pathological).

    Parameters
    ----------
    label: str or a valid description for Bijection.
           (from mackelab_toolbox.transform.Bijection)

    unit : Pint unit | Quantities unit | 1
        The special value of 1 is used to indicate no units.

    Prefer using a proper unit type (as provided by `pint` or `quantities`)
    instead of specifying a unit_label.
    Using a unit_label does not allow to obtain both long and short forms
    (e.g. 'meter' and 'm') and is ignored for the transformed axis.

    transformed:
        By default, initializer infers from `label` if it specifies a
        transformation. One can force `label` to be treated as a label
        or a transformation by passing `transformed=True` or
        `transformed=False`.
    """
    __slots__ = ('_transformed','_unit_check','_unit_convert')
    # class Parameters(ParameterSpec):
    #     schema = {'label', 'unit', 'unit_label', 'transformed', 'transform',
    #               '_min', '_max'}

    @property
    @abstractmethod          # Derived class would set this as a normal
    def transformed_type(self):  # class attribute.
        return                   # @property is just to have an abstract attr.

    transform : Optional[Bijection] = Field(
        None,
        description = "Specify either `transform` or `label`. Any valid "
                      "initializer for `Bijection` is also accepted.")
    label     : str = Field(
        None,
        description = "Specify either `transform` or `label`. If `label` is "
                       "not provided, it's value deduced from `transform`. "
                       "`label` is required if `transform` is not provided.")
    min_      : Optional[Number] = Field(None, alias='min')
    max_      : Optional[Number] = Field(None, alias='max')
    unit      : Union[AnyUnitType] = unitless
    unit_label: Optional[str]

    class Config:
        json_encoders = {**mtb.typing.json_encoders,
                         UnitlessT: lambda u: None }  # FIXME: Doesn't work

    # ----------------
    # Validators and initializer

    def __init__(self, *, _transformed_axis=None, **kwargs):
        if _transformed_axis is not None:
            if 'transform' in kwargs:
                raise ValueError(
                    "Specifying both a bijection and"
                    "`_transformed_axis` is ill-defined. Also "
                    "`_transformed_axis` is reserved for "
                    "internal use by the `Axis` class.")
            object.__setattr__(self, '_transformed', _transformed_axis)
            kwargs['transform'] = _transformed_axis.transform.inverse
        super().__init__(**kwargs)
        self.set_unit_methods(self.unit)

    def copy(self, *args, **kwargs):
        # Private attributes aren't copied over with _construct, and
        # `__init__` is not executed by `copy()`
        m = super().copy(*args, **kwargs)
        # HACK: Pydantic doesn't recursively call `copy()`. Here we throw away
        #       the copy made by Pydantic, and replace with our own <- wasteful
        if isinstance(self.transform, Bijection):
            m.transform = self.transform.copy(*args, **kwargs)
        # END HACK
        if hasattr(self, '_transformed'):
            object.__setattr__(m, '_transformed', self._transformed)
        object.__setattr__(m, '_unit_check', self._unit_check)
        object.__setattr__(m, '_unit_convert', self._unit_convert)
        return m

    @root_validator
    def label_or_transform(cls, values):
        label, transform = (values.get(x, None) for x in
                            ('label', 'transform'))
        if (label is None) and (transform is None):
            raise TypeError("Either `label` or `transform` must be specified.")
        return values

    @validator('transform', pre=True)
    def set_transform(cls, transform):
        if transform is not None:
            transform = Bijection(transform)
        return transform

    @validator('label', pre=True)
    def set_label(cls, label, values):
        """If label isn't given, try to deduce form transform."""
        if label is None:
            transform = values.get('transform', None)
            label = getattr(transform, 'xname', None)
        return label

    @validator('max_')  # max_ after min_ => min_ in `values`
    def check_axis_minmax_consistent(cls, max_, values):
        min_ = values.get('min', None)
        if min_ is not None and max_ is not None:
            if min_ >= max_:
                raise ValueError("Axis `min` must be smaller than `max`.\n"
                                 f"min: {min_}\nmax: {max_}")
        return max_

    @validator('unit')
    def default_unit(cls, unit):
        """Convert plain values of '1' (int, float, numpy type) to `unitless`."""
        # Since `unitless` subclasses int, it may be written out as `1`.
        if mtb.units.detect_unit_library(unit) == 'none':
            if isinstance(unit, numbers.Number):
                assert unit == 1
            unit = unitless
        return unit

    # ------------
    # __str__ and __repr__ methods

    def __str__(self):
        s = type(self).__name__
        if self.label is not None:
            s += f" '{self.label}'"
        unit = "" if self.unit is unitless else f"{self.unit}, "
        s += f" ({unit}{self.min}..{self.max})"
        return s

    # ------------
    # Unit methods

    # Dynamically set unit & conversion methods based on `unit` argument to
    # initializer. We detect types with duck typing rather than testing
    # against the actual types for two reasons:
    #   - Testing against types would force importing all quantities
    #     libraries, and therefore installing them.
    #   - In theory other libraries could implement these methods, and
    #     they would work as well.
    # @validator('unit')
    def set_unit_methods(self, unit):
        unitlib = mtb.units.detect_unit_library(unit)
        object.__setattr__(self, '_unit_check', _unit_check_factory(self, unitlib))
        object.__setattr__(self, '_unit_convert', _unit_convert_factory(self, unitlib))
        # if hasattr(unit, 'compatible_units') and hasattr(1*unit, 'to'):
        #     object.__setattr__(self, '_unit_check', _unit_check_factory(self, 'pint'))
        #     object.__setattr__(self, '_unit_convert', _unit_convert_factory(self, 'pint'))
        # elif (hasattr(unit, 'simplified') and hasattr(unit, 'dimensionality')
        #       and hasattr(unit, 'rescale')):
        #     object.__setattr__(self, '_unit_check', _unit_check_factory(self, 'quantities'))
        #     object.__setattr__(self, '_unit_convert', _unit_convert_factory(self, 'quantities'))
        # else:
        #     object.__setattr__(self, '_unit_check', _unit_check_factory(self, None))
        #     object.__setattr__(self, '_unit_convert', _unit_convert_factory(self, None))

    # Placeholder unit check, unit convert methods
    def unit_convert(self, value :float):
        """
        Convert a float value with units into the reference units for this axis.

        Parameters
        ----------
        value: float [equivalent unit]
            A real value with units. Units must be equivalent to the axis'
            (e.g., ms to s, m to km, etc.)

        Returns
        -------
        float [axis units]
        """
        return self._unit_convert(value)
    def unit_check(self, *values):
        """
        Return True if all values have units equivalent to the axis'

        Parameters
        ----------
        values: floats [with units]
            Real values with units.

        Returns
        -------
        bool
        """
        return self._unit_check(*values)

    def unit_remove(self, value):
        """
        Returns the magnitude of `value`, first converting to the axis unit
        if necessary.
        If `value` is already unitless, simply returns it.
        """
        if hasattr(value, 'magnitude'):
            return self._unit_convert(value).magnitude
        else:
            return value

    # ----------------
    # Public API

    def is_compatible_value(self, value):
        """
        Return True if `value` has the same units as the stop values, and has
        the same kind (as obtained with np.dtype().kind) and is safely
        castable to their dtype.

        .. Note:: There is currently an asymmetry with `is_compatible_index`,
           because indexing with lists of indices is supported, but not
           indexing with lists of values.

        """
        # For theano vars, we can't use `np.dtype(type(value))`, and dtype
        # is ust a string
        if hasattr(value, 'dtype'):
            kind = np.dtype(value.dtype).kind
        else:
            kind = np.dtype(type(value)).kind
        return (
            kind == np.dtype(self.stops_dtype).kind
            and mtb.units.detect_unit_library(value) == mtb.units.detect_unit_library(self.unit)
            and np.can_cast(value, self.stops_dtype))

    def is_compatible_index(self, value):
        """
        Return True if `value` is an integral type, not an AxisIndex attached
        to another axis, and safely castable to this axis' idx_dtype.

        .. Note:: There is currently an asymmetry with `is_compatible_value`,
           because indexing with lists of indices is supported, but not
           indexing with lists of values.

        TODO: Redundant with "more correct" method `axis.Index.is_compatible`
        """
        # For theano vars, we can't use `np.dtype(type(value))`, and dtype
        # is ust a string
        return self.Index.is_compatible(value)
        # if hasattr(value, 'dtype'):
        #     kind = np.dtype(value.dtype).kind
        # else:
        #     kind = np.dtype(type(value)).kind
        # return (
        #     kind == 'i'  # Only allow integers
        #     and mtb.units.detect_unit_library(value) == 'none'  # Disallow units (redundant?)
        #     and (not isinstance(value, (AbstractAxisIndex, AbstractAxisIndexDelta))
        #          or isinstance(self.stops.Index.Delta, self.stops.Index))
        #          # Only allow axis types from this axis
        #     and np.can_cast(value, self.index_dtype)
        #     )


    @property
    @abstractmethod
    def stops_array(self):
        """
        An axis may generate stops as required, to avoid saving the whole
        array. This attribute guarantees that the returned stops are an array.
        """
        raise NotImplementedError

    @property
    def transformed(self):
        try:
            return self._transformed
        except AttributeError:
            # On first execution create the transformed axis
            desc = self.transformed_desc
            # Remove the transform kwarg to avoid error
            del desc['transform']
            # Add this axis as transformed axis
            desc['_transformed_axis'] = self
            # Call the pydantic object parser. This is designed to be called
            # on a dictionary of attribute values, and so is more robust
            # simply using self.transformed_type(**desc)
            object.__setattr__(self, '_transformed',
                               self.transformed_type.parse_obj(desc))
            return self._transformed

    def transformed_dict(self):
        # Used within `self.transformed`, so don't call self.transformed here ;)
        if self.transform is None:
            raise AttributeError("This axis does not define a transformation.")
        desc = self.dict()
        min_ = self.min_; max_ = self.max_
        desc['transform'] = self.transform.inverse.dict()
        desc['stops'] = self.stops.transform(self.transform.map).dict()
        desc['label'] = self.transform.inverse_map.xname
        desc['unit'] = self.transformed_unit
        desc['min'] = min_ if min_ is None else self.transform(min_)
        desc['max'] = max_ if max_ is None else self.transform(max_)
        del desc['min_']  # Remove the internal min/max variables
        del desc['max_']
        return desc

    @property
    def transformed_unit(self):
        if self.transform is None:
            raise AttributeError("This axis does not define a transformation.")
        try:
            transformed_unit = self.transform(self.unit)
        except TypeError:
            transformed_unit = unitless
        return transformed_unit
    # Saving None allows to distinguish 'not set' from 'set to infinity'
    @property
    def min(self):
        return -np.inf if self.min_ is None else self.min_
    @property
    def max(self):
        return np.inf if self.max_ is None else self.max_
    @property
    def limits(self):
        return (self.min, self.max)
    @property
    def desc(self):
        return self.dict()
    @property
    def transformed_desc(self):
        return self.transformed_dict()

    def discretize(self, stops, format='centers', **kwargs):
        return DiscretizedAxis(
            **self.desc, stops=stops, format=format, **kwargs)

class Axes(tuple):
    """Set of multiple Axis instances"""
    __slots__ = ()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(not isinstance(ax, Axis) for ax in self):
            raise TypeError(
                "All elements of an `Axes` must be instances of `Axis`.")


###########################
# Discretized axis objects
#
# Defines the following mapping classes:
#   - SequenceMapping(Sequence)
#   - RangeMapping(SequenceMapping)
# And the following axis classes
#   - DiscretizedAxis(Axis)
#     + abstract base class; instantiate one of the subclasses
#   - ArrayAxis
#     + Stores all stops as an array
#   - MapAxis(DiscretizedAxis)
#     + Stores stops as a mapping from index to value
#     + Uses SequenceMapping
#     + O(n) retrieval of index values because stops have to all be calculated
#       and then searched.
#     + Converting indices between two axis instances requires them to be
#       converted to values.
#   - RangeAxis(MapAxis)
#     + Preset-mapping for an axis with regular steps
#     + Uses RangeMapping
#     + O(1) retrieval of index values
#     + Direct index conversion between two RangeAxis instances with
#       commensurate step size.
# Each axis instance defines its own Index type, constructed with
# `get_AxisIndex` which ensures that arithmetic operations are only performed
# between indices of the same axis.
# Specialized conversions allow operations between indices of two RegularAxes,
# as long as those indexes are commensurate.
##########################

class IndexTypeOperationError(TypeError):
    __slots__ = ()
    def __init__(self, message=None):
        if message is None:
            message = "Operations are only supported between indices of the same IndexType."
        super().__init__(message)

class AbstractAxisIndexDelta(abc.ABC):
    pass
class AbstractAxisIndex(abc.ABC):
    pass
class NumericAbstractAxisIndexDelta(abc.ABC):
    _created_indexes = {}
class NumericAbstractAxisIndex(abc.ABC):
    _created_indexes = {}
class SymbolicAbstractAxisIndexDelta(abc.ABC):
    _created_indexes = {}
class SymbolicAbstractAxisIndex(abc.ABC):
    _created_indexes = {}
AbstractAxisIndexDelta.register(NumericAbstractAxisIndexDelta)
AbstractAxisIndexDelta.register(SymbolicAbstractAxisIndexDelta)
AbstractAxisIndex.register(NumericAbstractAxisIndex)
AbstractAxisIndex.register(SymbolicAbstractAxisIndex)

class AxisIndexMeta(abc.ABCMeta):
    """
    This is the virtual base class attached to :class:`axis.Index`; when one
    calls ``axis.Index(i)``, either a :class:`NumericAxisIndex` or a
    :class:`SymbolicAxisIndex` is returned, depending on whether `i` is
    symbolic.

    Note that `bases` is currently redundant and can be passed as an empty
    tuple ``()``.
    """
    def __new__(metacls, name, bases, namespace, axis, numeric, symbolic):
        # HACK: parsing `name` is fragile
        #       Maybe deducing from `bases` would make more sense ?
        deltatype = "delta" in name.lower()
        if deltatype:
            _bases = (AbstractAxisIndexDelta,)
            if len(bases) > 0:
                assert bases == _bases
            else:
                bases = _bases
            assert issubclass(numeric, NumericAbstractAxisIndexDelta)
            assert issubclass(symbolic, SymbolicAbstractAxisIndexDelta)
        else:
            # Assertion: `bases` composed of AbstractAxisIndex and the associated Delta index
            assert len(bases) == 2
            assert bases[0] is AbstractAxisIndex
            assert issubclass(bases[1], AbstractAxisIndexDelta)
            assert issubclass(numeric, NumericAbstractAxisIndex)
            assert issubclass(symbolic, SymbolicAbstractAxisIndex)
        assert numeric.nptype is symbolic.nptype
        assert numeric.step == symbolic.step
        nptype = numeric.nptype
        step   = numeric.step
        assert nptype is symbolic.nptype
        class_attrs = {
            '__slots__' : (),
            'nptype'    : nptype,
            'dtype'     : np.dtype(nptype),
            'axis'      : axis,
            'step'      : step,
            'Numeric'   : numeric,
            'Symbolic'  : symbolic
        }
        properties = {}
        methods = {
            '__new__': metacls.__clsnew__,
        }
        static_methods = {
            'check_in_bounds': staticmethod(metacls._check_in_bounds)
        }
        class_methods = {
            'attach_axis'  : classmethod(metacls._class_attach_axis),
            'make_index'   : classmethod(metacls._class_make_index),
            'is_compatible': classmethod(metacls._class_is_compatible)
        }
        # Assert no clashing names
        assert not (   (set(class_attrs) ^ set(properties) ^ set(methods) ^ set(static_methods) ^ set(class_methods))
                     - (set(class_attrs) | set(properties) | set(methods) | set(static_methods) | set(class_methods)) )

        T = super().__new__(metacls,
                            name,
                            bases,
                            {**class_attrs, **properties, **methods, **static_methods, **class_methods, **namespace},
                            )
        assert issubclass(T, abc.ABC)
        T.register(numeric)
        T.register(symbolic)
        return T

    def __init__(cls, name, bases, namespace, axis, numeric, symbolic):
        # We want the absolute and relative index types to be associated
        # – the easiest way is to create them together.
        if not hasattr(cls, 'Delta'):
            assert not issubclass(cls, AbstractAxisIndex)
                # Assert that cls is an AxisDelta
            cls.Delta = cls
        if not hasattr(cls, 'Absolute'):
            # `type` below will call `[S|N]AxisIndexMeta.__new__`
            cls.Absolute = type(cls.__name__.replace("Delta", ""),
                                (AbstractAxisIndex, cls),
                                {'Delta'    : cls,
                                 'Absolute' : None},
                                 axis = cls.axis,
                                 numeric = numeric.Absolute,
                                 symbolic = symbolic.Absolute
                                )
            cls.Absolute.Absolute = cls.Absolute

    @staticmethod
    def __clsnew__(cls, x, allow_rounding=False):
        # if not shim.isscalar(x):
        #     raise ValueError("Only scalars can be converted to indices.")
        if issubclass(cls, (NumericAbstractAxisIndexDelta,
                            SymbolicAbstractAxisIndexDelta)):
            # No need to redirect to appropriate Index type; proceed
            # with super().__new__
            # (Note: as far as I know, no code path should end up here,
            #  because Symbolic & Numeric aren't proper subclasses)
            return super().__new__(x)
        elif shim.is_symbolic(x):
            if allow_rounding:
                warn("AxisIndex: `allow_rounding` is ignored for symbolic "
                     "indices.")
            return cls.Symbolic(x)
        else:
            if shim.is_graph_object(x):
                # TensorConstants end up here
                x = shim.eval(x)
            return cls.Numeric(x, allow_rounding=allow_rounding)

    # Actual static method
    @staticmethod
    def _check_in_bounds(index, imin, imax, max_cost=30):
        """
        Return True if `index` is in the closed interval ``[imin, imax]``.

        .. Note:: Both `imin` and `imax` are inclusive.
        """
        if not shim.istype(index, 'int'):
            raise TypeError("`index` must be in integer.")
        if shim.graph.is_computable(index):
            evalidx = shim.eval(index, max_cost=max_cost)
            if np.any(evalidx < imin) or np.any(evalidx > imax):
                raise IndexError("Provided index `{}` exceeds the mapping's "
                                 "range.".format(index))
        else:
            raise ValueError("`index` is not computable.")

    # Class method
    @staticmethod
    def _class_make_index(cls, value):
        """
        Ensure that `value` is an index. If `value` is an
        instance of `cls`, it is simply returned. If it is a plain
        integer, it is converted to an index. Any other input is
        rejected.

        Parameters
        ----------
        value: AxisIndex[self] | Integral | Symbolic integer
        """
        if isinstance(value, cls.Delta):  # Includes both abs and delta
            return value
        elif isinstance(value,  (AbstractAxisIndex, AbstractAxisIndexDelta)):
            raise IndexTypeOperationError(
                "`AxisIndex.make_index` cannot convert "
                "between indexes attached to different dimensions.")
        # elif (isinstance(value, numbers.Integral)  # <-- does not include 0-dim arrays
        #       or shim.is_symbolic(value) and shim.is_type(value, 'int')):
        elif (shim.isscalar(value) and shim.istype(value, 'int')):
            # It's a plain value, not an index.
            # cls() will return either a Numeric or Symbolic Index
            return cls(value)
        else:
            raise ValueError("`AxisIndex.make_index` only accepts "
                             "integer arguments.")

    # Class method
    @staticmethod
    def _class_attach_axis(cls, axis):
        """
        Attach this index class to an axis. If you passed `axis` to
        `get_AxisIndex`, this was done automatically.
        (NOTE: At present it's generally better to attach axes
        afterwards, to ensure `axis.stops` is defined (otherwise
        `step` can't be retrieved and defaults to `None`))

        When called on `axis.Index`, this method attaches all six index types:
          - `Index`
          - `Index.Delta`
          - `Index.Numeric`
          - `Index.Numeric.Delta`
          - `Index.Symbolic`
          - `Index.Symbolic.Delta`
        """
        if cls.axis is not axis:
            if cls.axis is not None:
                raise RuntimeError(f"{str(cls)} is already attached "
                                   f"to {str(cls.axis)}.")
            cls.axis = axis
            cls.step = rgetattr(axis, 'stops.step', None)
        else:
            assert cls.step == rgetattr(axis, 'stops.step', None)
        cls.Numeric.attach_axis(axis)
        cls.Symbolic.attach_axis(axis)
        if issubclass(cls, AbstractAxisIndex):
            cls.Delta.attach_axis(axis)

    # Class method
    @staticmethod
    def _class_is_compatible(cls, *idx):
        """
        Return True if `idx` can be used in arithmetic operations
        with `cls`.
        `idx` can be either an index type (AxisIndex) or an index value; in
        the latter case it is first converted to a type.
        Valid combinations:
          - `idx` is either an AxisIndex or AxisIndexDelta of same
            type as `cls`.
          - `idx` is an index delta of another type, but with the same
            step size as `cls`.
          - `idx` is a plain integer, which is castable to `cls.dtype`. Note
            that we compare types, so `np.int64(2)` is still not castable to
            `int32`.
        """
        idx_types = [_idx if isinstance(_idx, type) and issubclass(_idx, AbstractAxisIndexDelta)
                     else type(_idx) if isinstance(_idx, AbstractAxisIndexDelta)
                     else cls.nptype if shim.istype(_idx, 'int')
                     else False
                     for _idx in flatten(idx, terminate=shim.config.TerminatingTypes)]
        if False in idx_types:
            return False
        return all(issubclass(T, cls.Delta)  # num & sym, abs & rel of this index type
                   or (issubclass(T, AbstractAxisIndexDelta)  # rel of other index types
                       and not issubclass(T, AbstractAxisIndex)
                       and T.step is not None and T.step == cls.step)
                   or (not issubclass(T, AbstractAxisIndexDelta)
                       and np.can_cast(T, cls.dtype)) # plain ints
                   for T in idx_types)

class SpecAxisIndexMeta(type):
    """
    Axis index type. Will only accept to instantiate values within
    the index bounds (plus the special values ``imin-1``, ``imax+1``,
    which are used to indicate points outside the axis, for example the
    "current position" in a history with no calculated value, or the
    "stop" attribute of a slice going up to the last position.
    """

    def __new__(metacls, name, bases, namespace, axis, nptype):
        if metacls is SpecAxisIndexMeta:
            raise RuntimeError("Don't use SpecAxisIndexMeta directly. Use either "
                               "NumericAxisIndexMeta or SymbolicAxisIndexMeta.")
        # HACK: parsing `name` is fragile
        #       Maybe deducing from `bases` would make more sense ?
        deltatype = "delta" in name.lower()
        if issubclass(metacls, NumericAxisIndexMeta):
            if deltatype:
                abstract_class = NumericAbstractAxisIndexDelta
            else:
                abstract_class = NumericAbstractAxisIndex
        else:
            assert issubclass(metacls, SymbolicAxisIndexMeta)
            if deltatype:
                abstract_class = SymbolicAbstractAxisIndexDelta
            else:
                abstract_class = SymbolicAbstractAxisIndex


        # if not (len(bases) == 1 and np.issubdtype(bases[0], np.integer)):
        #     raise TypeError("`bases` must comprise exactly one type, and "
        #                     "it must be a NumPy integer.")

        # # AxisIndex is a subclass of AxisIndexDelta
        # nptype = bases[0]
        # while issubclass(nptype, AbstractAxisIndexDelta):
        #     assert len(nptype.__bases__) == 1
        #     nptype = nptype.__bases__[0]
        if not (len(bases) == 1):
            raise TypeError("`bases` must comprise exactly one type")
        if not np.issubdtype(nptype, np.integer):
            raise TypeError("`nptype` must be a NumPy integer type.")

        # TODO: I'm not sure we really should save `step` as an Index attribute
        step = None if axis is None else rgetattr(axis, 'stops.step', None)
        class_attrs = {
            '__slots__'       : ('_plain', '_data_index'),
            '_created_indexes': abstract_class._created_indexes,
            'nptype'          : nptype,
            'axis'            : axis,
            'step'            : step
        }
        properties = {
            'plain'     : property(metacls._instance_plain),
            'data_index': property(metacls._instance_data_index),
            'in_bounds' : property(metacls._instance_in_bounds)
        }
        methods = {
            'convert'         : metacls._instance_convert,
            'get_abs_rel_args': metacls._instance_get_abs_rel_args,
            '__new__'         : metacls.__clsnew__,
            '__init__'        : metacls.__clsinit__,
            '__eq__'          : metacls._instance___eq__,
            '__ne__'         : metacls._instance___ne__,
            '__add__'         : metacls._instance___add__,
            '__sub__'         : metacls._instance___sub__,
            '__mul__'         : metacls._instance___mul__,
            '__truediv__'     : metacls._instance___truediv__,
        }
        if hasattr(metacls, '_instance___str__'):
            methods['__str__'] = metacls._instance___str__
        static_methods = {
            '_check_in_bounds': staticmethod(AxisIndexMeta._check_in_bounds)
        }
        class_methods = {
             '__hash__'     : classmethod(metacls._class___hash__),
             'convert_index': classmethod(metacls._class_convert_index),
             'make_index'   : classmethod(metacls._class_make_index),
             'attach_axis'  : classmethod(metacls._class_attach_axis),
             'is_compatible': classmethod(metacls._class_is_compatible)
        }

        # Assert no clashing names
        assert not (   (set(class_attrs) ^ set(properties) ^ set(methods) ^ set(static_methods) ^ set(class_methods))
                     - (set(class_attrs) | set(properties) | set(methods) | set(static_methods) | set(class_methods)) )

        suffix = "" if axis is None else f" ({str(axis)})"

        T = super().__new__(metacls,
                            name + suffix,
                            bases,
                            {**class_attrs, **properties, **methods, **static_methods, **class_methods,
                             **namespace}  # Let namespace override attrs
                            )
        # # HACK
        # if "delta" in name.lower():
        #     NumericAbstractAxisIndexDelta.register(T)
        # else:
        #     NumericAbstractAxisIndex.register(T)

        if axis is not None and id(axis) not in T._created_indexes:
            T._created_indexes[id(axis)] = T

        return T

    def __init__(cls, name, bases, namespace, axis, nptype):
        # We want the absolute and relative index types to be associated
        # – the easiest way is to create them together.
        if not hasattr(cls, 'Delta'):
            assert not issubclass(cls, AbstractAxisIndex)
                # Assert that cls is an AxisDelta
            cls.Delta = cls
        if not hasattr(cls, 'Absolute'):
            # `type` below will call `[S|N]AxisIndexMeta.__new__`
            cls.Absolute = type(cls.__name__.replace("Delta", ""),
                                (cls,),
                                {'Delta'    : cls,
                                 'Absolute' : None},
                                 axis = cls.axis,
                                 nptype = nptype
                                )
            cls.Absolute.Absolute = cls.Absolute

    # @property
    # @abstractmethod
    # def nptype(self): raise AttributeError
    # @property
    # @abstractmethod
    # def axis(self): raise AttributeError

    # The __new__ method for the generated class
    # Doesn't actually do anything, but allows subclasses to specialize __init__
    @staticmethod
    def __clsnew__(cls, *a, **kw):
        # Problem: super(cls, cls) -> Infinite recursion
        # HACK: Find the first cls in MRO for which the *next* cls is not AxisIndex
        #       (super() starts looking at the next class)
        base = cls  # First entry in cls.mro()
        for C in cls.mro()[1:]:
            if not issubclass(C, AbstractAxisIndexDelta):
                break
            base = C
        new = super(base, cls).__new__
        if new is object.__new__:
            # object.__new__ takes no argument
            return new(cls)
        else:
            # HACK: Remove any keyword args not expected by the parent's `new`
            #       Note: we assume that all positional args need to be forwarded
            sig = inspect.signature(new)
            kw = {k:v for k,v in kw.items() if k in sig.parameters}
            return new(cls, *a, **kw)

    # The __init__ method for the generated class
    @staticmethod
    def __clsinit__(self, x, *args, **kwargs):
        if (isinstance(x, (AbstractAxisIndex, AbstractAxisIndexDelta))
            and type(x) is not type(self)):
            raise TypeError("Tried to construct an "
                            f"{type(self).__name__} from a different "
                            "AxisIndex type.")
        # if allow_rounding:
        #     # Cast before checking bounds
        #     x = self.nptype(x)
        # FIXME: Accessing `index_range` through `axis` is dumb
        if (self.axis is not None
            and isinstance(self, AbstractAxisIndex)
            and not isinstance(x, shim.config.TensorDescType)): #HACK
            # Bounds checking doesn't really make sense for Δidx
            # Theano may trigger a clone, and have `x` be a TensorDescType
            index_range = self.axis.stops.index_range
            imin = index_range[0]
            imax = index_range[-1]
            try:
                self._check_in_bounds(shim.eval(x, max_cost=20),
                                      imin-1, imax+1)
            except shim.graph.TooCostly:
                pass
        # HACK: Neither `int` nor `np.integer` implement __init__
        #       (everything is done in __new__), so super().__init__() resolves
        #       to object._init__(), which expects no arguments
        # HACK: super() gets confused within metaclass, so we use
        #       recurse through MRO until we find non-axis type
        for base in type(self).mro():
            if not issubclass(base, AbstractAxisIndexDelta):
                break
        if isinstance(self, numbers.Integral):
            base.__init__(self)  # Either int or np.int base
        else:
            base.__init__(self, x, *args, **kwargs)

    #Instance: @property
    @staticmethod
    def _instance_plain(self):
        """
        Remove the 'index' identity by casting `self` to `nptype`.
        """
        # We cache the return value so that symbolic values are always the same
        # TODO?: Store as attribute when creating the instance ?
        #        => See also SymbolicAxisIndexMeta._instance_plain
        if not hasattr(self, '_plain'):
            self._plain = self.astype(self.dtype)
        else:
            try:
                assert shim.eval(self._plain) == shim.eval(self.astype(self.dtype))
            except shim.graph.TooCostly:
                pass
        return self._plain

    #Instance: @property
    @staticmethod
    def _instance_data_index(self):
        """
        Return an index value such as 0 corresponds to the left-most
        stop, including padding.

        This is intended for indexing into an associated data
        structure, and thus returns a plain integer.
        (I.e. an instant of the index's parent type.)
        """
        # FIXME: Accessing `mapping` through `axis` is dumb
        # We cache the return value so that symbolic values are always the same
        # TODO?: Store as attribute when creating the instance ?
        imin = self.nptype(self.axis.stops.index_range[0])
        self._check_in_bounds(self, imin, np.inf)
            # We don't use self.in_bounds here, because we only check lower bound
            # (Slicing up to upper bound inclusive requires upper bound + 1)
        if not hasattr(self, '_data_index'):
            # self._data_index = shim.cast(self, self.nptype) - imin
            self._data_index = self.plain - imin
        else:
            try:
                # assert shim.eval(self._data_index) == shim.eval(shim.cast(self, self.nptype) - imin)
                assert shim.eval(self._data_index) == shim.eval(self.plain - imin)
            except shim.graph.TooCostly:
                pass
        return self._data_index

    #Instance: @property
    @staticmethod
    def _instance_in_bounds(self):
        """Return True iff the index value is within the axis bounds."""
        index_range = self.axis.stops.index_range
        try:
            self._check_in_bounds(self, index_range[0], index_range[-1])
        except IndexError:
            return False
        else:
            return True

    # # Actual static method
    # @staticmethod
    # def _check_in_bounds(index, imin, imax):
    #     """
    #     Return True if `index` is in the closed interval ``[imin, imax]``.
    #
    #     .. Note:: Both `imin` and `imax` are inclusive.
    #     """
    #     if not shim.istype(index, 'int'):
    #         raise TypeError("`index` must be in integer.")
    #     if shim.graph.is_computable(index):
    #         evalidx = shim.eval(index)
    #         if np.any(evalidx < imin) or np.any(evalidx > imax):
    #             raise IndexError("Provided index `{}` exceeds the mapping's "
    #                              "range.".format(index))
    #     else:
    #         raise ValueError("`index` is not computable.")

    # Class method
    @staticmethod
    def _class___hash__(cls):
        return id(cls)

    # Class method
    @staticmethod
    def _class_make_index(cls, value):
        """
        Ensure that `value` is an index. If `value` is an
        instance of `cls`, it is simply returned. If it is a plain
        integer, it is converted to an index. Any other input is
        rejected.

        Parameters
        ----------
        value: AxisIndex[self] | Integral | Symbolic integer
        """
        return cls.Base.make_index(value)

    # Class method
    @staticmethod
    def _class_attach_axis(cls, axis):
        # FIXME: We should attach a Mapping, not an Axis
        """
        Attach this index class to an axis. If you passed `axis` to
        `get_AxisIndex`, this was done automatically.
        (NOTE: At present it's generally better to attach axes
        afterwards, to ensure `axis.stops` is defined (otherwise
        `step` can't be retrieved and defaults to `None`))
        """
        if cls.axis is not axis:
            if cls.axis is not None:
                raise RuntimeError(f"{str(cls)} is already attached "
                                   f"to {str(cls.axis)}.")
            cls.axis = axis
            cls.step = rgetattr(axis, 'stops.step', None)
            cls._created_indexes[id(axis)] = cls
        else:
            assert cls.step == rgetattr(axis, 'stops.step', None)
            assert id(axis) in cls._created_indexes
        if issubclass(cls, AbstractAxisIndex):
            cls.Delta.attach_axis(axis)

    # Class method
    @staticmethod
    def _class_is_compatible(cls, *idx):
        """
        Return True if `idx` can be used in arithmetic operations
        with `cls`.
        Valid combinations:
          - `idx` is either an AxisIndex or AxisIndexDelta of same
            type as `cls`.
          - `idx` is an index delta of another type, but with the same
            step size as `cls`.
        """
        return cls.Base.is_compatible(*idx)
        # if any(not shim.istype(_idx, 'int') for _idx in idx):
        #     return False
        # return all(isinstance(_idx, cls.Base.Delta)  # num & sym, abs & rel of this index type
        #            or (isinstance(_idx, AbstractAxisIndexDelta)  # rel of other index types
        #                and not isinstance(_idx, AbstractAxisIndex)
        #                and _idx.step is not None and _idx.step == cls.step)
        #            or not isinstance(_idx, (AbstractAxisIndex, AbstractAxisIndexDelta)) # plain ints
        #            for _idx in idx)

    # Class method
    @staticmethod
    def _class_convert_index(cls, index, new_mapping):
        """
        Convert this index into one for the new mapping.

        .. Note:: For most uses, calling :meth:`convert()` on the index
           instance is the most appropriate; this method is mostly for
           internal use.

        new_mapping : SequenceMapping | DiscretizedAxis | Index
        index: AxisIndex (instance of `cls`)
        """
        if not isinstance(index, cls):
            raise TypeError("`index` argument to `cls.convert_index` must "
                            "be an instance of `cls`.")

        this_mapping = cls.axis.stops
        if isinstance(new_mapping, Axis):
            new_mapping = new_mapping.stops
            NewIndex   = new_mapping.Index
        elif isinstance(new_mapping, SequenceMapping):
            NewIndex   = new_mapping.Index
        elif (isinstance(new_mapping, type)
              and issubclass(new_mapping, (AbstractAxisIndex, AbstractAxisIndexDelta))):
            new_mapping = new_mapping.axis.stops
            NewIndex   = new_mapping.Index  # Ensure its not an AxisDelta
        if not isinstance(new_mapping, SequenceMapping):
            raise TypeError(f"{new_mapping} is not a SequenceMapping.")
        if (not isinstance(this_mapping, RangeMapping)
            or not isinstance(new_mapping, RangeMapping)):
            raise TypeError("Index conversion only defined between "
                            "RangeMappings")
        if cls.step is None or NewIndex.step is None:
            raise TypeError("Index conversion only allowed between "
                            "regular mappings with a `step` attribute.")
        if cls.step != NewIndex.step:
            raise TypeError("Cannot convert index between mappings "
                            "with different steps.")
        Δi0 = new_mapping.i0.plain - this_mapping.i0.plain
            # Difference in anchor points. `plain` required b/c of ≠ idx types
        Δx0 = this_mapping.index_interval(new_mapping.x0 - this_mapping.x0)
            # Difference in the values at the anchor points
        # For an explanation of the index arithmetic, see
        # .. image::/docs/index_shift.svg
        return NewIndex(index.plain + Δi0 - Δx0)

    # Instance method
    @staticmethod
    def _instance_convert(self, other_mapping):
        """
        Convert this index into the corresponding index in the other mapping.
        “Corresponding” here means that both indices map to the same value.

        Parameters
        ----------
        other_mapping: RangeMapping
        """
        return self.convert_index(self, other_mapping)

    # Instance method
    @staticmethod
    def _instance_get_abs_rel_args(self, other
        ) -> Tuple[Tuple["AxisIndex"], Tuple[AxisIndexDelta]]:
        """
        Classify `self` and `other` into absolute and relative indices.
        First tuple contains absolute indices.
        Second tuple anything which isn't an absolute index. This
        includes index deltas of any type, but also plain ints.

        Returns
        -------
        Tuple[Tuple[AxisIndex], Tuple[AxisIndexDelta]]
        """
        res = (tuple(C for C in (self, other) if isinstance(C, AbstractAxisIndex)),
               tuple(C for C in (self, other) if not isinstance(C, (AbstractAxisIndex))))
        assert len(res[0]) + len(res[1]) == 2
        return res

    # Instance method
    @staticmethod
    def _instance___eq__(self, other):
        if shim.graph.pure_symbolic_inputs([other]):
            # We know that pure symbolics can't be made into indices, so this
            # must return False. The eval checks in `make_index` would throw
            # MissingInputError
            return False
        try:
            other = self.make_index(other)
        except (IndexTypeOperationError, ValueError, TypeError):
            return False
        else:
            return self.plain == other.plain
    # Instance method
    @staticmethod
    def _instance___ne__(self, other):
        return not self.__eq__(other)
    # Instance method
    @staticmethod
    def _instance___add__(self, other):
        if not self.is_compatible(other): raise IndexTypeOperationError
        other = self.Delta.make_index(other)
            # Use Delta so that plain ints are converted to a Δ
        absidx, relidx = self.get_abs_rel_args(other)
        if len(absidx) == 2:
            raise ValueError("Can't add two absolute indices. At "
                             "least one should be an index delta.")
        elif len(absidx) == 1:
            rettype = type(absidx[0]).Base
                # => other.Absolute may differ from self.Absolute
        else:
            rettype = self.Delta.Base
        return rettype(self.plain + other.plain)
    # Instance method
    @staticmethod
    def _instance___sub__(self, other):
        if not self.is_compatible(other): raise IndexTypeOperationError
        other = self.Delta.make_index(other)
            # Use Delta so that plain ints are converted to a Δ
        absidx, relidx = self.get_abs_rel_args(other)
        if len(absidx) == 2:
            rettype = self.Delta.Base
        elif len(absidx) == 1:
            rettype = type(absidx[0]).Base
        else:
            rettype = self.Delta.Base
        return rettype(self.plain - other.plain)
    # mul and div are allowed with arbitrary types, but just return
    # a plain type then. This allows things like `Δi * step`.
    # Instance method
    @staticmethod
    def _instance___mul__(self, other):
        if not self.is_compatible(other):
            return self.plain * other
        else:
            absidx, relidx = self.get_abs_rel_args(other)
            if len(absidx) > 0:
                raise ValueError("Can't multiply absolute indices. "
                                 "Use index deltas.")
            rettype = self.Delta.Base
            return rettype(self.plain * other)
    # Instance method
    @staticmethod
    def _instance___truediv__(self, other):
        if not self.is_compatible(other):
            return self.plain / other
        else:
            div = self.plain / other
            if not div.is_integer():
                raise ValueError(
                    "Index division must return an integer.")
            return self.nptype(div)

class NumericAxisIndexMeta(SpecAxisIndexMeta):
    def __new__(metacls, name, bases, namespace, axis, nptype=None):
        # nptype is superfluous, but allowing it keeps signatures consistent
        if not (len(bases) == 1 and np.issubdtype(bases[0], np.integer)):
            raise TypeError("`bases` must comprise exactly one type, and "
                            "it must be a NumPy integer type.")

        # AxisIndex is a subclass of AxisIndexDelta
        _nptype = bases[0]
        while issubclass(_nptype, AbstractAxisIndexDelta):
            assert len(_nptype.__bases__) == 1
            _nptype = _nptype.__bases__[0]
        if nptype is not None and nptype is not _nptype:
            raise ValueError(
                "`bases` and `nptype` are inconsistent. "
                f"bases.mro(): {bases[0].mro()}\nnptype.mro(): {nptype.mro()}")

        T = super().__new__(metacls, name, bases, namespace, axis, nptype)

        # HACK
        if "delta" in name.lower():
            NumericAbstractAxisIndexDelta.register(T)
        else:
            NumericAbstractAxisIndex.register(T)

        return T

    @staticmethod
    def __clsinit__(self, x, allow_rounding=False):
        if allow_rounding:
            # Cast before checking bounds
            x = self.nptype(x)
        # Neither `int` nor `np.integer` implement __init__ (everything is done in __new__)
        SpecAxisIndexMeta.__clsinit__(self, x)

    @staticmethod
    def _instance___str__(self):
        s = "NumericIndex"
        if hasattr(self, 'axis'):
           s += f" (axis: {self.axis})"
        return s

class SymbolicAxisIndexMeta(SpecAxisIndexMeta, shim.graph.GraphExpressionMeta):
    def __new__(metacls, name, bases, namespace, axis, nptype):
        T = super().__new__(metacls, name, bases, namespace, axis, nptype)

        # HACK
        if "delta" in name.lower():
            SymbolicAbstractAxisIndexDelta.register(T)
        else:
            SymbolicAbstractAxisIndex.register(T)

        return T

    # @staticmethod
    # def __clsnew__(cls, type, owner=None, index=None, name=None):
    #     """
    #     The expectation is that `x` should already be a symbolic expression.
    #     All we do is check that `x` is actually symbolic, and that its
    #     dtype is consistent with the axis' index type.
    #     """
    #     return idx
    #
    @staticmethod
    def __clsinit__(self, expr_or_type, owner=None, index=None, name=None):
        """
        The expectation is that `x` should already be a symbolic expression.
        We check that `x` is actually symbolic, and that its
        dtype is consistent with the axis' index type.
        """
        if not isinstance(expr_or_type, shim.config.SymbolicExpressionType):
            # Assume this is the usual symbolic initializer, and just forward
            # arguments
            SpecAxisIndexMeta.__clsinit__(self, expr_or_type, owner, index, name)
                # super() seems to get confused within metaclasses
        else:
            x = expr_or_type
            assert all(a is None for a in (owner, index))
            if not shim.graph.is_computable(x):
                raise TypeError("Symbolic indices must still be computable.")
            if (isinstance(x, AbstractAxisIndexDelta)
                and not isinstance(x, type(self))):
                raise TypeError(
                    f"Value {x} (type {type(x)}) is associated to another "
                    "axis, or otherwise not a subtype of the AxisIndex type.\n"
                    "Axis: {str(self.axis)}\nIndex type: {str(type(self))}")
            if np.can_cast(x.dtype, self.nptype):
                x = shim.cast(x, self.nptype)  # Only casts if necessary
            else:
                raise TypeError(f"Axis value {x} has dtype {x.dtype}, while this "
                                f"axis expects {np.dtype(self.nptype)}.")
            # TODO: Make a generic function in shim.graph
            name = x.name
            if name is None:
                name = f"Idx ({str(self.axis):.10})"
            else:
                name += f" (idx, {str(self.axis):.10})"
            SpecAxisIndexMeta.__clsinit__(self, x, name=name)
                # super() seems to get confused within metaclasses

    @staticmethod
    def _instance___str__(self):
        s = "SymbolicIndex"
        if hasattr(self, 'axis'):
           s += f" (axis: {self.axis})"
        return s

    # Instance property
    @staticmethod
    def _instance_plain(self):
        """
        Remove the 'index' identity by casting `self` to `nptype`.
        """
        if not hasattr(self, '_plain'):
            self._plain = self.copy()
            self._plain.name = self.name
        return self._plain
            # We can't use `astype`, because that still returns an AxisIndex
            # Here `copy` adds an identity operation which has `self` as input
            # There may be ways to remove the `self` node from the graph
            # instead, but that seems much more fragile.

# TODO: Replace this function by the metaclass for AxisIndex
def get_AxisIndex(axis, dtype=None):
    """
    Create a new index *type* attached to an axis. The new type
    subclasses `int` (or `np.int`) and can be operated on like
    an integer, with the exception that only operations between
    instances of the *same type* are permitted.

    Parameters
    ----------
    axis: DiscretizedAxis
        The axis instance to attach the index to. The index type
        can be set by first setting `axis.index_dtype`; this must
        be an integer type, and should subclass `int` or `np.integer`.
        If `axis` does not specify an index type, `int` is used.

    dtype: numpy dtype, Optional (default: `int32`)
        The type to use to store index values.

    Example
    -------
    >>> from sinn.axis import DiscretizedAxis, get_AxisIndex
    >>> class MyAxis(DiscretizedAxis):
    >>>     […]
    >>> Class OtherAxis(DiscretizedAxis):
    >>>     […]
    >>>
    >>> # Create index types attached to this axes
    >>> # (For example only: Axis already provides its type)
    >>> MyIndex = get_AxisIndex(MyAxis)
    >>> OtherIndex = get_AxisIndex(OtherAxis)
    >>>
    >>> # Create index instances
    >>> i1 = MyIndex(5)
    >>> i2 = MyIndex(9)
    >>> j1 = OtherIndex(5)
    >>>
    >>> # Can add same type indices
    >>> i1 + i2  # Returns 14
    >>> # Cannot add different type indices
    >>> # i1 + j1  # Raises IndexTypeOperationError
    >>>
    >>> # The axes in fact already provide an index type
    >>> k1 = MyAxis.Index(3)
    >>> k2 = MyAxis.Index(8)
    >>> k1 + k2  # Returns 11
    >>>
    >>> # It is the index type and not the axis instance which is checked,
    >>> # so the following fails:
    >>> # i1 + k1   #  Raises IndexTypeOperationError
    """
    if axis is None or id(axis) not in NumericAbstractAxisIndex._created_indexes:
        if dtype is None:
            if axis is None:
                dtype = np.int32
            else:
                dtype = getattr(axis, 'index_dtype', np.int32)
        nptype = np.dtype(dtype).type
        if not issubclass(nptype, (numbers.Integral)):
            raise TypeError("Axis' `index_dtype` is {}, but must be an integral type."
                            .format(nptype))
        # class AxisIndex(AxisIndexDelta):
        #     __slots__ = ('IndexDelta')
        #     Delta = AxisIndexDelta
        #     _store =  AbstractAxisIndex

        # Symbolic indices are either shared varibles, or computable expressions
        # with shared variables.
        symbtype = shim.graph.GraphExpression
            # Base symbolic type for tensors, shared vars, symbolic expressions
            # If Theano isn't loaded, returns an empty tuple

        suffix = "" if axis is None else f" ({str(axis)})"

        NumericAxisIndexDelta = NumericAxisIndexMeta(
            "NumericAxisIndexDelta" + suffix,
            (nptype,),
            {},
            axis=axis,
            nptype=nptype
            )
        if symbtype == ():
            # No symbolic library loaded
            SymbolicAxisIndexDelta = NumericAxisIndexDelta
            # Also register this as the "symbolic" type to pass assertions
            SymbolicAbstractAxisIndex.register(NumericAxisIndexDelta.Absolute)
            SymbolicAbstractAxisIndexDelta.register(NumericAxisIndexDelta)
        else:
            assert not isinstance(symbtype, tuple)
                # Not all that important, but line below assumes it atm
            SymbolicAxisIndexDelta = SymbolicAxisIndexMeta(
                "SymbolicAxisIndexDelta" + suffix,
                (symbtype,),
                {},
                axis=axis,
                nptype=nptype
                )
        # AxisIndex = NumericAxisIndexMeta(nptype=nptype, axis=axis, Delta=AxisIndexDelta)
        # AxisIndexDelta = type("AxisIndexDelta" + suffix,
        #                       (NumericAxisIndexDelta, nptype),
        #                       {'nptype': nptype,
        #                        'axis'  : None})
        # AxisIndex = type("AxisIndex" + suffix,
        #                  (NumericAxisIndex, nptype),
        #                  {'Delta' : AxisIndexDelta,
        #                   'nptype': nptype,
        #                   'axis'  : None}
        #                  )
        # AbstractAxisIndexDelta.register(AxisIndexDelta)
        # AbstractAxisIndex.register(AxisIndex)


        AxisIndexDelta = AxisIndexMeta(f'AxisIndexDelta' + suffix,
                                       (AbstractAxisIndexDelta,),
                                       {},
                                       axis = axis,
                                       numeric = NumericAxisIndexDelta,
                                       symbolic = SymbolicAxisIndexDelta
                                       )
        AxisIndex = AxisIndexDelta.Absolute

        NumericAxisIndexDelta.Base = AxisIndexDelta
        NumericAxisIndexDelta.Absolute.Base = AxisIndex
        SymbolicAxisIndexDelta.Base = AxisIndexDelta
        SymbolicAxisIndexDelta.Absolute.Base = AxisIndex

        if axis is not None:
            assert AxisIndexDelta.axis is axis
            assert AxisIndex.axis is axis
            AxisIndex.attach_axis(axis)
            # AxisIndexDelta.attach_axis(axis)  # Called by AxisIndex

        return AxisIndex

    else:
        assert id(axis) in NumericAbstractAxisIndexDelta._created_indexes
            # Ensure AxisIndex & AxisIndexDelta are in sync
        return NumericAbstractAxisIndex._created_indexes[id(axis)]

# ---------------------
# Indexed mappings

class SequenceMapping(BaseModel):
    """
    The idea here is that the provided `index_range` can be an object like
    `range`, which provides O(1) memory, bounds checking and presence checking
    (`x in range`). However this is _not_ assumed (and for this reason one
    should use the `RangeMapping` subclass when possible).

    All parameters are keyword-only.

    .. Note:: Although any starting point to `index_range` is valid, this is
    an artifact of the implementation and does not add flexibility (one can
    always shift an index inside `index_map`). It is thus _highly_ recommended
    that this range begin at 0. (Or more precisely, that
    `index_range[pad_left] == 0`.) This will avoid surprises from
    methods like `index()`, which return values with which to index into
    data (and thus must begin at 0).

    Parameters
    ----------
    index_range: Sequence
        The range of indices. Usually an instance of `range` or `ndarray`.
    index_map : Transform (int -> float) | ndarray
        Function converting indices into values. It must be monotone increasing.
        (If needed, this class could be generalized to also accomodate
        monotone decreasing map functions.)
    index_type : Type[Integral] | None
        The index type, used to do comparisons/arithmetic between
        SequenceMappings. Typically one obtains this by calling `get_AxisIndex`.
        The value `None` indicates that a mapping is not attached to any axis;
        avoid this when possible because this option is still experimental.
    """
    __slots__ = ('_stops_array', '_x_dtype')

    # Private attributes (excluded from serialization in Config)
    Index      : Optional[Type] = Field(..., alias='index_type')
        # TODO: Replace 'Type' by AxisIndex type
        # Index is not part of schema: it stores a specific instantiated,
        # AxisIndex type, so doesn't make sense to serialize.
        # The AxisIndex must be provided at each runtime.
        # We need Optional here because sometimes (at the moment, only in
        # SequenceMapping.transform) we need a mapping with no axis.
        # Recall: 'Optional' just allows the value to be None, but `None` still
        #         needs to be provided.

    # Maximum allowable length
    # This is a heuristic, and could be changed if needed
    max_len: ClassVar[Integral] = int(np.iinfo(np.int32).max / 2)

    index_range: Sequence
    index_map  : Transform  #Callable[[Integral], Real]
    # Index      : Type[Integral] = Field(None, alias='index_type')
    # 'pad': number of index values outside of the range [imin, imax]
    pad_left   : Integral = 0    # AxisIndexDelta
    pad_right  : Integral = 0    # AxisIndexDelta

    class Config:
        keep_untouched = (type,)  # FIXME: This doesn't work with `type`
        json_encoders = mtb.typing.json_encoders

    # HACK: Because `keep_untouched` doesn't work, we forcefully remove
    # 'Index' from the returned dict.
    def dict(self, *args, **kwargs):
        excl = kwargs.pop('exclude', None)
        excl = set() if excl is None else set(excl)
        excl.add('Index')
        d = super().dict(*args, exclude=excl, **kwargs)
        # Cast pad_left, pad_right to plain integers to detach them from Index
        for pad_side in ('pad_left', 'pad_right'):
            pad = d[pad_side]
            if isinstance(pad, AbstractAxisIndexDelta):
                d[pad_side] = pad.astype(pad.nptype)
        return d

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def copy(self, *args, **kwargs):
        """
        Copy the mapping. If `self` is attached to an axis, the copy will be
        attached to the same axis.
        However, using `axis.copy()` will call this method internally, and
        then replace the associated axis with the new copy.
        """
        # Method required b/c private attributes aren't copied over with _construct.
        m = super().copy(*args, **kwargs)
        # HACK: Pydantic doesn't recursively call `copy()`. Here we throw away
        #       the copy made by Pydantic, and replace with our own <- wasteful
        if isinstance(self.index_map, Transform):
            m.index_map = self.index_map.copy(*args, **kwargs)
        # END HACK
        object.__setattr__(m, 'Index', self.Index)
        return m

    @classmethod
    def parse_obj(self, *args, **kwargs):
        raise RuntimeError(
            "A *Mapping must be tied to an axis, and is not instantiatable on "
            "its own. Instead, use `parse_obj` or `parse_raw` from one of the "
            "*Axis classes.")

    @validator('Index')
    def set_index_type(cls, index_type):
        if index_type is None:
            pass
        elif not issubclass(index_type, AbstractAxisIndex):
            raise TypeError("`index_type` must be created with the "
                            "`get_AxisIndex` function.")
        return index_type

    @validator('index_range')
    def check_index_bounds(cls, index_range):
        if len(index_range) > cls.max_len:
            raise ValueError(
                "Length of `index_range` exceeds allowed maximum. If you need "
                "a longer index, change the value of `SequenceMapping.max_len`."
                " len(index_range) = {}".format(len(index_range)))
        return index_range

    @validator('index_range')
    def check_index_no_skips(cls, index_range):
        if isinstance(index_range, range) and index_range.step not in (None, 1):
            raise ValueError("`index_range` must have a step of 1.")
        elif not all(np.diff(np.array(index_range)) == 1):
            raise ValueError("`index_range` must have a step of 1.")
        return index_range

    @validator('index_map')
    def check_map_monotone_increasing(cls, m, values):
        """Implementation evalutes mapping at each index."""
        # NOTE: If this is changed to accept monotone decreasing, make sure
        #       `self.index` still works.
        index_range = values.get('index_range', None)
        if index_range is None:
            return
        # OPTIMIZE?: If testing the entire index like this becomes a problem,
        #            we could try sampling say 30 indices and checking that
        #            the function is monotone on those.
        if not np.all(np.diff(m(np.array(index_range))) > 0):
            raise ValueError("`index_map` is not monotone increasing.")
        return m

    @validator('pad_left', always=True)
    def cast_pad_left(cls, v, values):
        Index = values.get('Index', None)
        if Index is not None:
            v = Index.Delta(v)
        return v
    @validator('pad_right', always=True)
    def cast_pad_right(cls, v, values):
        Index = values.get('Index', None)
        if Index is not None:
            v = Index.Delta(v)
        return v

    @root_validator()
    def check_index_alignment(cls, values):
        index_range = values.get('index_range', None)
        pad_left = values.get('pad_left', None)
        if index_range is not None and pad_left is not None:
            if index_range[pad_left] != 0:
                warn(
                    "It is highly recommended that the zero point of your "
                    "index range actually be zero. Your values:\n"
                    f"Left padding: {pad_left}\nIndex_range: {index_range}.")
        return values

    # ------------------
    # __str__ and __repr__

    def __str__(self):
        s = type(self).__name__
        s += (f" (i={self.x0idx}..{self.xnidx}, x={self.index_map[self.x0idx]}"
              f"..{self.index_map[self.xnidx]})")
        return s

    # ------------------
    # Sequence interface

    def __getitem__(self, index):
        if not isinstance(index, slice):
            return self.index_map(self.index(index))
        else:
            index = self.index_range[self.data_index(index)]
            return self.index_map(np.array(index))

    def __iter__(self):
        return (self.index_map(i) for i in self.index_range)

    def __len__(self):
        return len(self.index_range)

    # ----------------
    # Equality testing

    def __eq__(self, other :Union[SequenceMapping,Array]):
        """
        Return True if both the index and stop values of `other` match
        those of `self`.

        Parameters
        ----------
        other: SequenceMapping | ndarray
            SequenceMapping: Checks equality of index ranges and stop values.
                Avoids creating an array from index range if possible
                FIXME: Currently calls `.stops_array` on both arguments. It
                would be nice to avoid this if possible by comparing mappings.
            ndarray: Checks that size and values match `self`'s stop values.

        Returns
        -------
        bool
        """
        if isinstance(other, SequenceMapping):
            selfrange = self.index_range; otherrange = other.index_range
            if (issubclass(type(selfrange), type(otherrange))
                or issubclass(type(otherrange), type(selfrange))):
                # If ranges are of same type, compare them directly
                # We especially want to avoid instantiating ndarray if possible
                range_compare = (selfrange == otherrange)
            else:
                range_compare = np.all(np.array(selfrange) == np.array(otherrange))
            return (range_compare
                    and np.all(self.stops_array == other.stops_array))
        elif isinstance(other, np.ndarray):
            return (other.ndim == 1 and len(other) == len(self.index_range)
                    and np.all(self.stops_array == other))
        else:
            return False

    def isclose(self, other):
        """
        Like `__eq__`, but uses `np.isclose` instead of `==` to compare stop
        values. Indexes of `self` and `other` must still match exactly.

        Parameters
        ----------
        other: SequenceMapping | ndarray
            SequenceMapping: Checks equality of index ranges and closeness
                of stop values (using `sinn.isclose`).
                Avoids creating an array from index range if possible
                FIXME: Currently calls `.stops_array` on both arguments. It
                would be nice to avoid this if possible by comparing mappings.
            ndarray: Checks that size and values match `self`'s stop values.

        Returns
        -------
        bool
        """

        if isinstance(other, SequenceMapping):
            selfrange = self.index_range; otherrange = other.index_range
            if (issubclass(type(selfrange), type(otherrange))
                or issubclass(type(otherrange), type(selfrange))):
                # If ranges are of same type, compare them directly
                # We especially want to avoid instantiating ndarray if possible
                range_compare = (selfrange == otherrange)
            else:
                range_compare = np.all(np.array(selfrange) == np.array(otherrange))
            return (range_compare
                    and np.all(sinn.isclose(self.stops_array, other.stops_array)))
        elif isinstance(other, np.ndarray):
            return (other.ndim == 1 and len(other) == len(self.index_range)
                    and np.all(sinn.isclose(self.stops_array, other)))
        else:
            return False


    # -------------
    # API

    @property
    def padded_stops_array(self):
        try:
            return self._stops_array
        except AttributeError:
            object.__setattr__(self, '_stops_array',
                               self.index_map(np.array(self.index_range)))
        return self._stops_array
    @property
    def unpadded_stops_array(self):
        stop = None if self.pad_right == 0 else -self.pad_right
        return self.padded_stops_array[self.pad_left:stop]
    stops_array = unpadded_stops_array

    @property
    def x_dtype(self):
        try:
            return self._x_dtype
        except AttributeError:
            object.__setattr__(self, '_x_dtype',
                               np.array(self.index_map(self.x0idx)).dtype)
        return self._x_dtype
    #
    # @property
    # def data_index_min(self):
    #     return self.index_range[0]
    # @property
    # def data_index_max(self):
    #     return self.index_range[-1]

    def data_index(self, value) -> Integral:
        """
        Return the index corresponding to `value`. Currently implemented with
        `stops_array` and `numpy.searchsorted`, which isn't particularly
        efficient.

        .. Note:: This corresponds to the index in the padded array. Also,
           If the `index_range` starts at something else than 0, this will not
           be the corresponding padding-shifted 'index' in `index_range`.
           (One reason why starting `index_range` at 0 is recommended.)

        Parameters
        ----------
        value: x_dtype | Index (int) | slice
            Index: Return result of `value.data_index`.
                   Plain integers are treated as an `Index` if they are
                   castable to `self.Index.dtype`
                   (Unless their type matches `x_dtype`).
            x_dtype: Search for index with `searchsorted`. Symbolic values
                   disallowed.
            slice: Recursively call `data_index` on slice bounds.
                   `None` values still exclude padding.

        Returns
        -------
        Integral  (Parent type to `self.Index.Delta`)
        """
        # FIXME: Lots of duplication with data_index
        # Branches:
        #  - Slice -> Recursively call on slice's start, stop
        #  - Already Index -> return
        #  - Index of wrong type -> raise TypeError
        #  - Type ≠ x_dtype & not castable as index -> raise TypeError
        #  - Castable as index -> cast to Index and return
        #  - Type = x_dtype -> Find index and return
        value_type = np.dtype(getattr(value, 'dtype', type(value))).type
            # The ugliness of np.dtype(…).type is because Theano stores dtype
            # as a string
        if isinstance(value, slice):
            # Recursive call to construct slice
            start = value.start
            if start is None: start = self.Index(
                self.index_range[self.pad_left])
            stop  = value.stop
            if stop is None:  stop  = self.Index(
                self.index_range[-self.pad_right-1])
            assert value.step is None or value.step == 1
            return slice(self.data_index(start), self.data_index(stop))
        elif isinstance(value, list):
            return [self.data_index(v) for v in value]
        elif isinstance(value, tuple):
            return tuple(self.data_index(v) for v in value)
        elif isinstance(value, self.Index):
            # Idempotent on indices
            return value.data_index
        elif isinstance(value, (AbstractAxisIndex, AbstractAxisIndexDelta)):
            # Illegal on deltas and indices from different axes
            if isinstance(value, AbstractAxisIndexDelta):
                raise TypeError("`index()` can't determine an absolute index "
                                "from an `AxisIndexDelta`.")
            elif isinstance(value, AbstractAxisIndex):
                raise TypeError("`index()` called on an index of a different "
                                "axis.")
        elif not issubclass(value_type, self.x_dtype.type):
            # Allow indexing with plain integers, unless x_dtype is integer (would be weird, but who knows ?)
            if shim.can_cast(value, self.Index.nptype):
                return self.Index(value).data_index
            raise TypeError(
                "SequenceMapping.index(): Type of argument ({}) does not "
                "match type of stops ({}) or of indices ({})."
                .format(value_type, self.x_dtype.type, self.Index.nptype))
        else:
            # Type of `value` matches `x_dtype` – search for corresponding index
            if shim.is_graph_object(value):
                raise TypeError(
                    "`SequenceMapping` does not support inverting a stop "
                    "value to its index. Try `RangeMapping`.")
            stops = self.padded_stops_array
            i = np.searchsorted(stops, value)
            if not np.all(np.isclose(stops[i], value)):
                raise ValueError(f"{value} not present in indexed mapping. "
                                 f"Closest value: {stops[i]} (index: {i}).")
            return i

    def index(self, value) -> "AxisIndex":
        """
        Return the index corresponding to `value`.

        .. Note:: This corresponds to the index as defined by `index_range`.
           If that range does not start with zero, the result will not match
           the actual (pad-shifted) position of the stop.

        Parameters
        ----------
        value: x_dtype | Index (int) | symbolic | slice
            Index: Return `value`.
                   Plain integers are treated as an `Index` if they are
                   castable to `self.Index.nptype`
                   (Unless their type matches `x_dtype`).
            x_dtype: Search for index with `searchsorted`.
            slice: Recursively call `index` on slice bounds.

        Returns
        -------
        AxisIndex[self] | slice

        Raises
        ------
        TypeError:
            - If value is an IndexDelta or Index from a different Axis
            - If value's type doesn't match `self.x_dtype` (type of stops), and
              cannot be cast as an index.
        """
        # FIXME: Lots of duplication with data_index
        # Branches:
        #  - Slice -> Recursively call on slice's start, stop
        #  - Already Index -> return
        #  - Index of wrong type -> raise TypeError
        #  - Type ≠ x_dtype & not castable as index -> raise TypeError
        #  - Castable as index -> cast to Index and return
        #  - Type = x_dtype -> Find index and return
        value_type = np.dtype(getattr(value, 'dtype', type(value))).type
            # The ugliness of np.dtype(…).type is because theano stores dtype
            # as a string
        if isinstance(value, slice):
            # Recursive call to construct slice
            start = value.start
            if start is None: start = self.Index(
                self.index_range[self.pad_left])
            stop  = value.stop
            if stop is None:  stop  = self.Index(
                self.index_range[-self.pad_right-1])
            assert value.step is None or value.step == 1
            return slice(self.index(start), self.index(stop))
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                return np.array(self.index(x[()]))
            else:
                warn("Calling `index` on a `SequenceMapping` generates an "
                     "array on every call. Consider using `ArrayMapping`. "
                     "(Or improving `SequenceMapping` ;-P)")
                # `index()` is undefined with higher dim arrays
                assert value.ndim == 1
                return np.array([self.index(x) for x in value])
        elif isinstance(value, self.Index):
            # Idempotent on indices
            return value
        elif isinstance(value, (AbstractAxisIndex, AbstractAxisIndexDelta)):
            # Illegal on deltas and indices from different axes
            if isinstance(value, AbstractAxisIndexDelta):
                raise TypeError("`index()` can't determine an absolute index "
                                "from an `AxisIndexDelta`.")
            elif isinstance(value, AbstractAxisIndex):
                raise TypeError("`index()` called on an index of a different "
                                "axis.")
        elif not issubclass(value_type, self.x_dtype.type):
            # Allow indexing with plain integers, unless x_dtype is integer (would be weird, but who knows ?)
            if shim.can_cast(value, self.Index.nptype):
                return self.Index(value)
            raise TypeError("[data_index] Type of argument ({}) does not "
                            "match type of stops ({}) or of indices ({})."
                            .format(value_type, self.x_dtype.type,
                                    self.Index.nptype))
        else:
            # Type of `value` matches `x_dtype` – search for corresponding index
            # FIXME: All the tests above will be repeated in data_index
            if isinstance(value, np.ndarray) and value.ndim > 0:
                index_array = np.array(self.index_range)
                return self.Index(index_array[self.data_index(value)])
            else:
                return self.Index(self.index_range[self.data_index(value)])
        # imin = self.index_range[0]
        # imax = self.index_range[-1]
        # _check_in_bounds = self._check_in_bounds
        # if isinstance(index, slice):
        #     start, stop, step = slice_indices(index, len(self.index_range))
        #     _check_in_bounds(start, imin, imax)
        #     _check_in_bounds(stop-1, imin, imax)
        #     start -= imin
        #     stop  -= imin
        #     return slice(start, stop, step)
        # else:
        #     _check_in_bounds(index, imin, imax)
        #     return index - imin

    def data_to_axis_index(self, data_index :Integral):
        """
        Convert a data index into an axis index.
        This function assumes that it receives a data index, which makes it
        very cheap, but will return an incorrect value on inputs which are
        already in axis space.

        Parameters
        ----------
        data_index : Integral | slice
            There are at present no Axis types associated to the data space,
            so only plain integers are allowed.

        Returns
        -------
        AxisIndex

        Raises
        ------
        TypeError:
            - If input is an AxisIndex or AxisIndexDelta
            - If input is not an integer
        """
        if isinstance(data_index, slice):
            assert data_index.step is None or data_index.step == 1
            return slice(self.data_to_axis_index(data_index.start),
                         self.data_to_axis_index(data_index.stop))
        else:
            if isinstance(data_index, (AbstractAxisIndex, AbstractAxisIndexDelta)):
                raise TypeError(
                    "`data_to_axis_index` is not idempotent on axis indices. "
                    f"Argument:{data_index}, of type {type(data_index)}.")
            elif not shim.istype(data_index, 'int'):
                raise TypeError(
                    "`data_to_axis_index` expects an integer. "
                    f"Argument:{data_index}, of type {type(data_index)}.")
            return self.Index(self.index_range[data_index])

    def axis_to_data_index(self,
                           axis_index :Union[Integral, AbstractAxisIndex, Slice]):
        """
        Convert an axis index to a data index. This function can be used when
        the inputs are known to be axis indices, to avoid the heuristics
        employed by `~SequenceMapping.data_index` to determine its input type.

        Parameters
        ----------
        axis_index : AxisIndex | int | Slice[AxisIndex]
            If a slice, applied recursively on its bounds.

        Returns
        -------
        Plain integral (self.Index.dtype)

        Raises
        ------
        TypeError:
            - If input is an AxisIndexDelta
            - If input is not an integer
        """
        if isinstance(axis_index, slice):
            assert axis_index.step is None or axis_index.step == 1
            return slice(self.axis_to_data_index(axis_index.start),
                         self.axis_to_data_index(axis_index.stop))
        else:
            if (isinstance(axis_index, AbstractAxisIndexDelta)
                and not isinstance(axis_index, AbstractAxisIndex)):
                raise TypeError(
                    "`axis_to_data_index` is undefined on index deltas. "
                    f"Argument:{axis_index}, of type {type(axis_index)}.")
            elif not shim.istype(axis_index, 'int'):
                raise TypeError(
                    "`axis_to_data_index` expects an integer. "
                    f"Argument:{axis_index}, of type {type(axis_index)}.")
            if shim.isscalar(axis_index):
                return self.Index(axis_index).data_index
            else:
                if not isinstance(axis_index, (list, tuple, np.ndarray)):
                    raise TypeError(
                        "When calling `SequenceMapping.axis_to_data_index` on "
                        "multiple values, only lists, tuples and Numpy arrays "
                        "are permitted.")
                warn("Calling `SequenceMapping.axis_to_data_index` on an array "
                     "is inefficient.")
                return np.array([self.Index(idx).data_index
                                 for idx in axis_index])

    def index_interval(self, value, value2=None,
                       allow_rounding=False, cast=True) -> "AxisIndexDelta":
        """
        An index delta is only sensible with constant steps. Use `RangeMapping`.

        Would return
        ------------
        AxisIndexDelta
        """
        raise NotImplementedError

    def data_index_slice(self, axis_slice, include_padding=False):
        """
        Convert an axis to a data slice, optionally including padding.

        Parameters
        ----------
        time_slice: slice[AxisIndex]
            Slice of axis indices.
        include_padding : bool | 'all' | 'begin' | 'start' | 'end'
            - True or 'all'     : include padding at both ends
            - 'begin' or 'start': include the padding before t0
            - 'end'             : include the padding after tn
            - False (default)   : do not include padding

        Returns
        -------
        slice
        """
        assert (isinstance(include_padding, bool)
                or include_padding in ('all', 'begin', 'start', 'end'))
        start = (0 if axis_slice.start is None
                 else self.data_index(axis_slice.start))
        stop  = (len(self.index_range) if axis_slice.stop is None
                 else self.data_index(axis_slice.stop))
        if include_padding in (False, 'end'):
            start = max(start, self.pad_left)
        if include_padding in (False, 'start', 'begin'):
            stop  = min(stop, self.padded_length - self.pad_right)
        return slice(start, stop, axis_slice.step)

    @property
    def x0idx(self):
        """
        The index of the origin before adding any padding. This is *not*
        affected by padding, and if the recommendations are followed, is
        always zero.

        Returns `AxisIndex`

        Returns
        -------
        AxisIndex
        """
        return self.Index(self.index_range[self.pad_left])
    @property
    def xnidx(self):
        """
        Index corresponding to the end of the mapping.

        Calculated as len(mapping) - self.pad_right - 1.

        Returns
        -------
        AxisIndex
        """
        return self.Index(self.index_range[-self.pad_right-1])
    @property
    def unpadded_length(self):
        return len(self.index_range) - self.pad_left - self.pad_right
    @property
    def padded_length(self):
        """Same as __len__; provided for interface consistency."""
        return len(self.index_range)

    def pad(self, pad_left :Integral, pad_right: Integral=0):
        """
        Already present padding is subtracted, so that
        >>> mapping.pad(2)
        >>> mapping.pad(3)
        has a padding of 3, and not 5.

        Typically some data structure is associated with the axis, and it
        should also be padded. For this reason the method returns a tuple of
        integers, indicating the number of bins it actually added. So in the
        example above, the first line would return `(2,0)`, and the second,
        `(1,0)`.

        .. Note:: This method simply increases the range of indexes.
        It is the user's responsibility to ensure that `index_map` is valid
        over the expanded range. (With RangeAxis this is always true.)

        Parameters
        ----------
        pad_left
        pad_right: int | AxisIndexDelta
            Number of bins to add to the before (after) the axis.

        Return
        ------
        tuple of int
            2-element tuple giving the number of bins added to the left and
            right.

        Raises
        ------
        ValueError:
           If the new axis is too large for the index type.
        """
        if shim.is_graph_object(pad_left, pad_right):
            raise TypeError("`pad()` doesn't accept symbolic values.")
        # raise NotImplementedError(
        #     "`SequenceMapping` does not support modifying padding because "
        #     "it doesn't know the format of the underlying `index_range`. "
        #     "Use `RangeMapping` or `ArrayMapping`.")
        for p, s in zip((pad_left, pad_right), ('left', 'right')):
            if not self.Index.is_compatible(p):
                raise IndexTypeOperationError(f"`pad_{s}` is incompatible "
                                              f"with {self}'s index.")
            # if not shim.istype(p, 'int'):
            #     raise TypeError("`pad` accepts only integer values")
            # if (isinstance(p, AbstractAxisIndexDelta)
            #     and p.step != self.Index.Delta.step):
            #     mystep = self.Index.Delta.step
            #     raise TypeError("Value for `pad_{s}` is an index delta "
            #                     "with different step size (received: "
            #                     f"{p.step}, expected: {mystep}).")
        pad_left  = max(pad_left - self.pad_left, 0)
        pad_right = max(pad_right- self.pad_right, 0)
        idx_pad_left  = self.Index.Delta(pad_left)
        idx_pad_right = self.Index.Delta(pad_right)
        if pad_left != 0 or pad_right != 0:
            if isinstance(self.index_range, range):
                start = self.index_range.start
                stop  = self.index_range.stop
                self.index_range = range(start-pad_left, stop+pad_right)
            else:
                start = self.index_range[0]
                stop  = self.index_range[-1] + 1
                self.index_range = np.arange(start-pad_left, stop+pad_right)
            # Padding has invalidated the stops array
            if hasattr(self, '_stops_array'):
                del self._stops_array
            # Map may no longer be monotone increasing
            self.check_map_monotone_increasing(
                self.index_map, {'index_range': self.index_range})
            self.pad_left  += idx_pad_left
            self.pad_right += idx_pad_right

        # Check that the time index type can still store all time indices
        maxidx = abs(max(self.index_range))
        if not np.can_cast(np.min_scalar_type(-maxidx), self.Index):
            # '-' ensures we don't get a uint as min scalar type
            raise ValueError(
                "With padding, this axis' indices can now reach {}, which is "
                "too large for the history's time index type ({}).\nTo avoid "
                "this error, make sure total padding does not exceed the "
                "length of the unpadded axis (either by reducing padding, or "
                "by initializing the axis with a longer time array.)"
                .format(maxidx, str(self.Index)))

        return (idx_pad_left, idx_pad_right)

    def transform(self, f):
         """Transform the index_map by applying `f` to every point."""
         # def new_index_map(i): return f(self.index_map(i))
         new_index_map = Transform(f).compose(self.index_map)
         return SequenceMapping(index_range=self.index_range,
                                index_map=new_index_map,
                                index_type=None)

# class Computed_index_map:
#     __slots__ = ()
#     def __call__(self, *args, **kwargs):
#         return self._index_map(*args, **kwargs)

class RangeMapping(SequenceMapping):
    """
    Note: To prevent rounding point errors, `i0` should not be changed.
    Provides same functionality as `SequenceMapping`, but with potentially
    much smaller memory footprint, using the same idea as Python's `range`.

    Implemented as follows: If `x(i)` is the axis value at
    index `i`, then `x(i) = self.x0 + (i-self.i0)*self.step`.
    `x0` is initialized at `min`, `i0` is initialized at 0. These values
    may change when adding padding to the axis.
    It is recommended not to change `i0` in the middle of calculations, to
    avoid issues due to rounding errors.

    Parameters
    ----------
    index_range
    index_map
    index_type
    pad_left
    pad_right :
        See SequenceMapping

    x0: Value of the index at i0.
    i0: Lowest(?) index. Can be negative.
    step: Step size; always pass as float64; it will be downcasted appropriately

    (Automatically calculated)
    step64: High precision step size.
        As long as `step` is passed as double, there is no need to also
        specify `step64`.
    """
    # HACK: We need to hide the `index_map` attribute for the parent class
    #       (its unnecessary, and the recursion crashes repr)
    #       Approach:
    #         - Add to __slots__ so pydantic lets us assign the name
    #         - After class creation, del RangeMapping.__fields__['index_map']
    #           (Super Hacky !!)
    #         - In __init__, assign `self._index_map` to `self.index_map`
    #         - Do the same in copy()
    __slots__ = ('index_map',)# + ('step_dtype', 'imin', 'imax')
        # `step_dtype` provides fast access to self.__fields__['step'].type_

    # step_dtype : ClassVar[NPType[np.floating]] = np.float64
    #     # Use a high-precision type for index calculations

    index_range: Range  # No longer any Sequence allowed: must also be regular
    i0         : Integral    # AxisIndex
    x0         : Real
    # Step is *not* the step size of `index_range` (which is always 1), but
    # the size of the mapped-to values.
    # It's type should be compatible with `x0`.
    step       : Real
    # Use a high-precision type for index calculations
    # Value taken from `step`, as long as it's a 64-bit float.
    step64     : NPType[np.float64]  = None

    # index_map overridden just to remove it from json, dict
    # (default values are not exported by default)
    # The 'index_map' is always the same and stored as `self._index_map`
    index_map  : Optional[Callable[[int],float]] = None
    #
    # class Config:
    #     keep_untouched = (Computed_index_map,)

    # ---------------------------
    # Validators and constructors

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        # Cast i0 to plain integer to detach them from Index
        i0 = d['i0']
        if isinstance(i0, AbstractAxisIndex):
            d['i0'] = i0.astype(i0.nptype)
        return d

    def __init__(self, **kwargs):
        # if i0 not in index_range:
        #     raise ValueError("`i0` must be within `index_range`.")
        # self.i0 = i0
        # self.x0 = x0
        # object.__setattr__(self, 'step_dtype', self.__fields__['step'].type_)
        if kwargs.get('index_map', None) is not None:
            raise ValueError("RangeMapping does not take an `index_map` "
                             "argument.")
        super().__init__(**kwargs)
        # Cast `step`
        # TODO: Any way to do this in validators ? How do we get Index.type ?
        # step = self.step; step_dtype = self.Index.nptype
        # if step is not None and not isinstance(step, step_dtype):
        #     if np.can_cast(step, step_dtype):
        #         self.step = step_dtype(step)
        #     else:
        #         raise TypeError(f"Can't cast `step` (value: {step}, type: "
        #                         f"{type(step)}) to {step_dtype}.")
        # Assign private vars
        object.__setattr__(self, 'index_map', self._index_map)
        # object.__setattr__(self, 'imin', self.index_range[0])
        # object.__setattr__(self, 'imax', self.index_range[-1])

    def copy(self, *args, **kwargs):
        """Private attributes aren't copied over with _construct."""
        m = super().copy(*args, **kwargs)
        object.__setattr__(m, 'index_map', m._index_map)
        # object.__setattr__(m, 'step_dtype', self.step_dtype)
        # object.__setattr__(m, 'imin', self.imin)
        # object.__setattr__(m, 'imax', self.imax)
        return m

    @validator('i0')
    def check_i0_in_bounds(cls, i0, values):
        index_type = values.get('Index', None)
        index_range = values.get('index_range', None)
        if index_range is not None and i0 not in index_range:
            # For a Python range, `i0 in range` is O(1)
            raise ValueError("`i0` must be within `index_range`.")
        if index_type is not None:
            i0 = index_type(i0)
        return i0

    @validator('step64', pre=True, always=True)
    def get_step64(cls, step64, values):
        """
        Get `step64` from `step`.
        """
        step = values.get('step', None)
        if not isinstance(step, (np.float64, float)) and step is not None:
            warn("[RangeMapping] `step` should be given as float64. "
                 f"(Type received: {type(step)}")
        if step64 is None and step is not None:
            step64 = step
        return step64

    @validator('index_map')
    def no_index_map(cls, v):
        raise AssertionError(
            "RangeMapping does not take an `index_map` argument.")

    # IMPORTANT: `step` must not have been downcasted before the 'step64'
    # validator runs.
    @root_validator
    def upcast_x0_step(cls, values):
        x0   = values.get('x0',   None)
        step = values.get('step', None)
        if None not in (x0, step):
            if np.dtype(type(x0)).kind != np.dtype(type(step)).kind:
                raise TypeError("`x0` and `step` values have incompatible "
                                f"types (they are {type(x0)} and {type(step)}).")
            uptype = np.result_type(x0, step).type
            values['x0']   = uptype(x0)
            values['step'] = uptype(step)
        return values

    # ------------------
    # Sequence interface

    def __getitem__(self, index):
        # self._zero_shifted_index(index)
            # Executed just for the bounds checking
            # => Avoid numerical errors by always using i0 as reference
        # index = np.array(self.index_range[index])
        # if not shim.istype(index, 'int'):
        #     raise TypeError("`index` must be in integer.")
        symb = shim.is_graph_object(self, index)
        idx_range = self.index_range
        if symb:
            idx_range = shim.arange(idx_range.start, idx_range.stop,
                                    idx_range.step, symbolic=True)
        if isinstance(index, list):
            return [self[i] for i in index]
        elif isinstance(index, tuple):
            return tuple(self[i] for i in index)
        elif shim.isarray(index):
            if not symb:
                warn("Creating a throwaway index array; consider ArrayMapping.")
                idx_range = np.asarray(idx_range)
            index = idx_range[self.data_index(index)]
        else:
            index = shim.asarray(idx_range[self.data_index(index)])
        i0 = self.i0
        x0 = self.x0
        imin = self.index_range[0]
        imax = self.index_range[-1]
        step = self.step64
        try:
            numidx = shim.eval(index)
            if np.any(numidx < imin) or np.any(numidx > imax):
                raise IndexError("Provided index `{}` exceeds the mapping's "
                                 "range.".format(index))
        except shim.graph.TooCostly:
            pass
        # We don't call self.mapping for efficiency
        return x0 + (index-i0)*step

    # ----------------
    # Equality testing

    def __eq__(self, other):
        def plain(x):
            return getattr(x, 'plain', x)
        if isinstance(other, RangeMapping):
            return (    plain(self.i0)    == plain(other.i0)
                    and plain(self.x0)    == plain(other.x0)
                    and self.step         == other.step
                    and self.index_range  == other.index_range)
        else:
            return super().__eq__(other)

    # ------------------
    # Internal functions

    # Need `index_map` function to be consistent with SequenceMapping
    def _index_map(self, index):
        i0 = self.i0
        x0 = self.x0
        step = self.step64
        return x0 + (index-i0)*step

    def floating_point_error_check(self, value):
        if np.any(value * config.get_rel_tolerance(value) > self.step):
            raise ValueError(
                "You've tried to convert the value {} into an index, "
                "but the value is too large to ensure the absence of "
                "numerical errors. Try using a higher precision type.")

    # ------------------
    # Public API

    def data_index(self, value, allow_rounding=False):
        res = self.index(value, allow_rounding=False)
        if isinstance(res, slice):
            return slice(res.start.data_index, res.stop.data_index, res.step)
        elif shim.isarray(res):
            # Copied from AxisIndexDelta.data_index
            imin = self.index_range[0]
            restype = str(self.Index.dtype)
            return (res if imin == 0 else res - imin).astype(restype)
        elif isinstance(res, list):
            return [r.data_index for r in res]
        elif isinstance(res, tuple):
            return tuple(r.data_index for r in res)
        else:
            return res.data_index

    def index(self, value, allow_rounding=False):
        """
        Return the index corresponding to `value`.

        .. Note:: This corresponds to the index as defined by `index_range`.
        If that range does not start with zero, the result will not match
        the actual (pad-shifted) position of the stop.

        Parameters
        ----------
        value: x_dtype | Index (int) | symbolic | slice
            Index: Return `value`.
                   Plain integers are treated as an `Index` if they are
                   castable to `self.Index.nptype`
                   (Unless their type matches `x_dtype`).
            x_dtype: Search for index with `searchsorted`.
            slice: Recursively call `index` on slice bounds.
        allow_rounding: bool
            True: If `value` does not correspond to an index, round to the
            nearest index. Can also be used to skip the `isclose` check.
        """
        # Branches:
        #  - Slice -> Recursively call on slice's start, stop
        #  - list,tuple -> Recursively call on elements and return list|tuple
        #  - Already Index -> return
        #  - Index of wrong type -> raise TypeError
        #  - Type ≠ x_dtype & not castable as index -> raise TypeError
        #  - Castable as index -> cast to Index and return
        #  - Type = x_dtype -> Find index and return
        value_type = np.dtype(getattr(value, 'dtype', type(value))).type
            # The ugliness of np.dtype(…).type is because Theano stores dtype
            # as a string
        if isinstance(value, slice):
            start = value.start
            if start is None: start = self.i0
            stop  = value.stop
            if stop is None:  stop  = self.i0 + self.unpadded_length
            step  = value.step
            # Step must be an int but not an index (but self.Index.Delta is OK)
            if step is None: step = 1
            assert (shim.istype(step, 'int')
                    and (not isinstance(step, (AbstractAxisIndex,
                                               AbstractAxisIndexDelta))
                         or isinstance(step, self.Index.Delta)))
            # assert value.step is None or value.step == 1
            return slice(self.index(start), self.index(stop),
                         self.Index.Delta(step))
        elif isinstance(value, list):
            return [self.index(v) for v in value]
        elif isinstance(value, tuple):
            return tuple(self.index(v) for v in value)
        elif isinstance(value, self.Index):
            # Idempotent on indices
            return value
        elif isinstance(value, (AbstractAxisIndex, AbstractAxisIndexDelta)):
            # Illegal on deltas and indices from different axes
            if (isinstance(value, AbstractAxisIndexDelta)
                and not isinstance(value, AbstractAxisIndex)):
                raise TypeError("`index()` can't determine an absolute index "
                                "from an `AxisIndexDelta`.")
            elif isinstance(value, AbstractAxisIndex):
                return value.convert(self)
                # raise TypeError("`index()` called on an index of a different "
                #                 "axis.")
        elif not issubclass(value_type, self.x_dtype.type):
            # Allow indexing with plain integers, unless x_dtype is integer (would be weird, but who knows ?)
            if shim.can_cast(value, self.Index.nptype):
                return self.Index(value)
            raise TypeError(
                "RangeMapping.index(): Type of argument ({}) does not "
                "match type of stops ({}) or of indices ({})."
                .format(value_type, self.x_dtype.type, self.Index.nptype))
        else:
            # Type of `value` matches `x_dtype` – invert mapping
            # to get corresponding index
            self.floating_point_error_check(value)
                # Ensure `value` has high enough precision to compute index
            i0 = self.i0
            x0 = self.x0
            step = self.step64
            i = self.Index((value - x0) / step + 0.5 + i0,
                           allow_rounding=True)
                # + 0.5 is so 9.99 rounds to 10 instead of 9
            if allow_rounding or shim.is_graph_object(i):
                return self.Index(i)
            else:
                if np.all(np.isclose((value - x0)/step + i0, i)):
                    return self.Index(i)
                else:
                    raise ValueError(
                        "This axis has no index corresponding to the value {}."
                        .format(value))

    # TODO: remove duplicated input type checks from SequenceMapping
    def data_to_axis_index(self, data_index :Integral) -> "AxisIndex":
        if isinstance(data_index, (AbstractAxisIndex, AbstractAxisIndexDelta)):
            raise TypeError(
                "`data_to_axis_index` is not idempotent on axis indices. "
                f"Argument:{data_index}, of type {type(data_index)}.")
        elif not shim.istype(data_index, 'int'):
            raise TypeError(
                "`data_to_axis_index` expects an integer. "
                f"Argument:{data_index}, of type {type(data_index)}.")
        return self.Index(data_index - self.pad_left + self.i0)
    data_to_axis_index.__doc__ = SequenceMapping.data_to_axis_index.__doc__

    def index_interval(self, value, value2=None,
                       allow_rounding=False, cast=True):
        """
        Convert `value` to the corresponding number of steps.
        If `value2` is passed, convert `value2 - value1`.

        Parameters
        ----------
        value: float
        value2: float | None
        allow_rounding: bool
            If set to True, function will not throw errors if the value
            is not commensurate with the step size.
        cast: bool
            When True, cast the result to `self.Index`.
            When False, return a float (or whatever the result of
            `value/self.stop` is), possibly with fractional part if
            the value difference is not a multiple of the step.

        Returns
        -------
        If `cast` == True,  AxisIndexDelta (self.stops.step_type)
        If `cast` == False, result of `value/self.step`; should be compatible
        with `x_dtype`.
        """
        step = self.step64
        if value2 is not None:
            value = value2 - value
        self.floating_point_error_check(value)
        Δi = self.Index.Delta(value / step + 0.5, allow_rounding=True)
        di = value / step
        if not allow_rounding and not shim.is_symbolic(di) and not np.isclose(Δi, di):
            raise ValueError("Value {} is not commensurate with step size {}."
                             .format(value, step))
        return Δi if cast else di

    #
    # def pad(self, pad_left, pad_right=0):
    #     pad_left, pad_right = super().pad(pad_left, pad_right)
    #     self.i0 += pad_left
    #     return pad_left, pad_right

    def transform(self, f):
        # Compose needs a Transform object
        _index_map = Transform(f"i -> {self.x0} + (i-{self.i0})*{self.step64}")
            # Translated from self._index_map
        new_index_map = Transform(f).compose(_index_map)
        return SequenceMapping(index_range=self.index_range,
                               index_map=new_index_map,
                               index_type=None)

# HACK: Super hacky trick to remove 'index_map' from schema. See above for rest
del RangeMapping.__fields__['index_map']

class ArrayMapping(SequenceMapping):
    """
    Basically the same as SequenceMapping, but instead of a Transform, stores
    an array with the value at each position.

    All parameters are keyword-only.

    Parameters
    ----------
    index_range: Sequence
        The range of indices. Usually an instance of `range`.
    index_map : ndarray
        Size must match `index_range`.
    index_type : Type[Integral] | None
        The index type, used to do comparisons/arithmetic between
        SequenceMappings. Typically one obtains this by calling `get_AxisIndex`.
        The value `None` indicates that a mapping is not attached to any axis;
        avoid this when possible because this option is still experimental.
    """
    index_map : Array[np.floating]

    @validator('index_map')
    def check_index_map_size(cls, im, values):
        index_range = values.get('index_range', None)
        if im.ndim != 1:
            raise ValueError("`index_map` must be 1-dimension.")
        if index_range is None:
            return
        if len(im) != len(index_range):
            raise ValueError(f"`index_map` has length {len(im)}, but "
                             f"`index_range` has length {len(index_range)}.")
        return im

    @validator('index_map')
    def check_map_monotone_increasing(cls, m, values):
        """This validator shadows the one in SequenceMapping."""
        if not np.all(np.diff(m) > 0):
            raise ValueError("`index_map` is not monotone increasing.")
        return m

    # -------------
    # Sequence interface

    def __getitem__(self, index):
        return self.index_map[self.data_index(index)]

    def __iter__(self):
        return iter(self.index_map)

    # -------------
    # API

    @property
    def x_dtype(self):
        return self.index_map.dtype

    @property
    def padded_stops_array(self):
        return self.index_map
    # @property
    # def stops_array(self):
    #     return self.index_map

    def transform(self, f):
        new_index_map = Transform(f)(self.index_map)
        return ArrayMapping(index_range=self.index_range,
                            index_map  =new_index_map,
                            index_type =None)

    def pad(self, pad_left, pad_right=0):
        """
        TODO: Write algorithm for padding ArrayMapping which works at
        least for regular steps by inferring step from array.
        """
        raise NotImplementedError(
            "TODO: Write algorithm for padding ArrayMapping which works at "
            "least for regular steps by inferring step from array.")
        pad_left, pad_right = super().pad(pad_left, pad_right)
        return pad_left, pad_right

# ---------------------
# Discretized axes

class BinAlign(Enum):
    """The interpretation of stops as either bin centers or bin edges."""
    CENTERS = auto()
    EDGES = auto()
    def get(align):
        if isinstance(align, str):
            return BinAlign[align.upper()]
        elif isinstance(align, BinAlign):
            return align
        else:
            raise ValueError(
                "`align` must be a string or Enum type compatible with {}."
                .format(BinAlign.__qualname__))
class BinRef(Enum):
    """Whether bin alignment is based on this or the transformed axis."""
    SELF = auto()
    TRANSFORMED = auto()
    def get(ref):
        if isinstance(ref, str):
            return BinRef[ref.upper()]
        elif isinstance(ref, BinRef):
            return ref
        else:
            raise ValueError(
                "`bin_ref` must be a string or Enum type compatible with "
                "{}.".format(BinRef.__qualname__))

def determine_index_dtype(stops_len):
    return np.min_scalar_type(-2*max(stops_len, 1))
        # Leave enough space in time indices to double the time array
        # Using a negative value forces the type to be `int` and not
        # `uint`, which we need to store negative indices (esp. -1)

class DiscretizedAxis(Axis):
    """
    Parameters
    ----------
        [See Axis]

        bin_align: DiscretizedAxis.BinAlign | str

        bin_ref: DiscretizedAxis.BinRef | str
            SELF: Use this axis when converting from bins to edges.
            TRANSFORMED: Use the transformed axis when converting from bins
            to edges.
            Can also specify as a string.
        stops_dtype: Dtype, optional. (default: `shim.config.floatX`)
            Type to use for stop values
        index_dtype: NPType[integer], optional (default: computed)
            Type to use for index. Generally not necessary to specify: it is
            determined automatically from the length of `stops`.
            Specifically: the smallest possible type is chosen which allows
            for values twice the length of `stops` (this allows space
            for padding).

    """
    __slots__ = ('_edges_axis', '_centers_axis')

    transformed_type: ClassVar[type]# = DiscretizedAxis
        # The AxisType resulting from a call to `.transform`.
        # Assigned below because DiscretizedAxis not yet defined

    stops    : SequenceMapping
    bin_align: BinAlign = 'centers'
    bin_ref  : BinRef   = 'self'
    stops_dtype     : NPType[np.generic] = np.dtype(shim.config.floatX).type
        # Type to use for stop values
    index_dtype     : Type[np.integer] = None
        # Type to use for index.

    class Config:
        keep_untouched = (type,)  # FIXME: This doesn't work with `type`

    # HACK: Because `keep_untouched` doesn't work, we forcefully remove
    # 'index_dtype' from the returned dict.
    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        del d['index_dtype']
        return d

    @validator('bin_align', pre=True, always=True)
    def set_bin_align(cls, v):
        if isinstance(v, str):
            v = BinAlign.get(v)
        return v
    @validator('bin_ref', pre=True, always=True)
    def set_bin_ref(cls, v):
        if isinstance(v, str):
            v = BinRef.get(v)
        return v

    @validator('stops')
    def check_stops_within_bounds(cls, stops, values):
        min_, max_ = (values.get(x, None) for x in ('min', 'max'))
        if min_ is not None and min(stops) < min_:
            warn("Axis stops exceed its stated minimum")
        if max_ is not None and max(stops) > max_:
            warn("Axis stops exceed its stated maximum")
        return stops

    @validator('index_dtype', pre=True, always=True)
    def determine_index_dtype(cls, v, values):
        stops = values.get('stops', None)
        if v is None and stops is not None:
            v = determine_index_dtype(len(stops)).type
        return v

    # def __init__(self, **kwargs):
    #     if not hasattr(self, 'Index'):  #  Avoid overwriting subclass index
    #         object.__setattr__(self, 'Index', get_AxisIndex(self))
    #     super().__init__(**kwargs)
    #     # self.stops = stops
    #     # self.bin_align = self.BinAlign.get(bin_align)
    #     # self.bin_ref = self.BinRef.get(bin_ref)
    #     # super().__init__(label=label, unit=unit, unit_label=unit_label,
    #     #                  min=min, max=max,
    #     #                  transformed=transformed,
    #     #                  _transformed_axis=_transformed_axis)
    #
    #     # if min(stops) < self.min:
    #     #     warn("Axis stops exceed its stated minimum")
    #     # if max(stops) < self.max:
    #     #     warn("Axis stops exceed its stated maximum")
    #
    #     # if isinstance(format, Callable):
    #     #     format = format()  # In case we pass the format method rather than its evaluation
    #     # if format not in self.formats:
    #     #     raise ValueError(
    #     #         "`format` must be one of {}. It is '{}'."
    #     #          .format(', '.join(["'"+f+"'" for f in self.formats]), format))
    #     # else:
    #     #     self._format_str = format

    def copy(self, *a, **kw):
        """Private attributes aren't copied over with BaseModel.copy."""
        m = super().copy(*a, **kw)
        # HACK: Pydantic doesn't recursively call `copy()`. Here we throw away
        #       the copy made by Pydantic, and replace with our own <- wasteful
        m.stops = self.stops.copy(*a, **kw)
        # Create a new AxisIndex
        # TODO: All Index class management should go to `Mapping` class
        m.stops.Index = get_AxisIndex(m, dtype=self.stops.Index.nptype)
        # Cast index attributes to the new AxisIndex type
        # TODO?: Create validator to cast all index attributes ?
        m.stops.pad_left = m.stops.Index.Delta(m.stops.pad_left.plain)
        m.stops.pad_right = m.stops.Index.Delta(m.stops.pad_right.plain)
        # Recall: _edges_axis,_centers_axis may point to `self`, so copy should
        #         point to the new object (and avoids infinite recursion !)
        if hasattr(self, '_edges_axis'):
            if self._edges_axis is self:
                object.__setattr__(m, '_edges_axis', m)
            else:
                object.__setattr__(m, '_edges_axis',
                                   self._edges_axis.copy(*a, **kw))
        if hasattr(self, '_centers_axis'):
            if self._centers_axis is self:
                object.__setattr__(m, '_centers_axis', m)
            else:
                object.__setattr__(m, '_centers_axis',
                                   self._centers_axis.copy(*a, **kw))
        # END HACK
        return m

    def __len__(self):
        return len(self.stops)
    # def __str__(self):
    #     try:
    #         return (self.label + ' ({:.3}:{.3}, {} stops)'
    #                              .format(*self.limits, len(self)))
    #     except: # Don't let an exception prevent from printing
    #             # -> half the time we use this is when debugging.
    #         return super().__str__()
    def __eq__(self, other):
        # We try to short-circuit comparison by starting with cheap tests
        if self is other: return True
        # if (self.transform is None) != (other.transform is None): return False
        if not isinstance(other, DiscretizedAxis): return False
        # if not comparedicts(self.desc, other.desc): return False
        if len(self) != len(other): return False
        if self.unit != 1 and other.unit != 1:
            # FIXME: Allow conversion between compatible units
            if self.unit != other.unit: return False
        # if (self.transform is not None and
        #     not comparedicts(self.transformed_desc, other.transformed_desc)):
        #     return False
        # return True
        return self.stops == other.stops

    def __getitem__(self, index):
        v = self.stops[self.index(index)]
        # if self.unit is not unitless:
        #     v *= self.unit
        return v * self.unit

    # @property
    # def desc(self):
    #     desc = super().desc
    #     desc['bin_align'] = self.format
    #     desc['stops'] = self.stops
    #     dest['bin_ref'] = self.bin_ref
    #     return desc

    # @property
    # @abstractmethod
    # def transformed_desc(self):
    #     desc = super().transformed_desc
    #     desc['alignment'] = self.bin_align
    #     desc['stops'] = self.transformed_stops
    #     if self.bin_ref is self.BinRef.SELF:
    #         desc['bin_ref'] = self.BinRef.TRANSFORMED
    #     else:
    #         assert self.bin_ref is self.BinRef.TRANSFORMED
    #         desc['bin_ref'] = self.BinRef.SELF
    #     return desc

    # @property
    # def name(self):
    #     """Synonym for `label.name`."""
    #     return self.label.name
    #
    # @property
    # def start(self):
    #     """
    #     Return the start of the axis. Can be used to set limits on a plot.
    #     """
    #     return self.edges.stops[0]
    #
    # @property
    # def end(self):
    #     """
    #     Return the end of the axis. Can be used to set limits on a plot.
    #     """
    #     return self.edges.stops[-1]

    @property
    def Index(self):
        return self.stops.Index

    @property
    def padded_stops_array(self):
        return self.stops.padded_stops_array
    @property
    def unpadded_stops_array(self):
        return self.stops.unpadded_stops_array
    @property
    def stops_array(self):
        return self.stops.stops_array

    @property
    def padded_length(self):
        return self.stops.padded_length
    @property
    def unpadded_length(self):
        return self.stops.unpadded_length

    @property
    def pad_left(self):
        return self.stops.pad_left
    @property
    def pad_right(self):
        return self.stops.pad_right

    @property
    @abstractmethod
    def transformed_stops(self):
        """Return the stops of the transformed axis."""
        raise NotImplementedError

    @property
    def x0idx(self):
        return self.stops.x0idx
    @property
    def xnidx(self):
        return self.stops.xnidx

    @property
    def limits(self):
        """
        Return a (start, end) giving the bounds of the axis. Can be used to "
        "set limits on a plot.
        """
        stops = self.edges.stops
        return (stops[0], stops[-1])

    @property
    def nbins(self):
        if   self.bin_align is DiscretizedAxis.BinAlign.EDGES:
            return self.Index(len(self) - 1)
        elif self.bin_align is DiscretizedAxis.BinAlign.CENTERS:
            return self.Index(len(self))
        else:
            raise RuntimeError

    @property
    def widths(self):
        """
        Return an ndarray of same length as `centers` giving each bin's width.
        """
        edges = np.array(self.edges.stops)
            # `stops` may be stored as an iterator
        return abs(edges[1:] - edges[:-1])

    def data_index(self, value):
        """
        Return the index corresponding to `value`.
        This function must evaluate the index_map at every point; if you need
        to do this, consider using ArrayAxis or RangeAxis.

        Parameters
        ----------
        value: float | slice(float, float)
            - float: Units must match those of the axis.
            - slice: start and end points must be specified with the correct
              units. Because the index points are not regularly spaced,
              a slice cannot specify a step.
        """
        if isinstance(value, slice):
            start = self.data_index(value.start)
            stop  = self.data_index(value.stop)
            if value.step is not None:
                raise ValueError("`DiscretizedAxis.index()` doesn't support  "
                                 "the `step` argument. If your axis is "
                                 "regular, try `RangeAxis` or `ArrayAxis`.")
            return slice(start, stop)
        else:
            if hasattr(value, 'magnitude'):
                # value_dtype = np.array(value.magnitude).dtype.type
                if not self.unit_check(value):
                    raise TypeError("Provided value ({}) does not have the "
                                    "expected units ({})."
                                    .format(value, self.unit))
                value = self.unit_remove(self.unit_convert(value))
                    # We explicitly cast to the stops_dtype, so that Index
                    # recognizes as a stop value.
                    # Otherwise things like 1*ureg.s get sent as int, and Index
                    # thinks they are indices.
                if not np.can_cast(value, self.stops_dtype):
                    raise TypeError(
                        "`index` expects an input of type "
                        f"{self.stops_dtype} (received {type(value)}).")
                value = self.stops_dtype(value)
            # else:
            #     value_dtype = np.array(value).dtype.type
            return self.stops.data_index(value)

    def index(self, value):
        """
        Return the index corresponding to `value`.
        This function must evaluate the index_map at every point; if you need
        to do this, consider using ArrayAxis or RangeAxis.

        Parameters
        ----------
        value: float | slice(float, float)
            - float: Units must match those of the axis.
            - slice: start and end points must be specified with the correct
              units. Because the index points are not regularly spaced,
              a slice cannot specify a step.
        """
        if isinstance(value, slice):
            if value.start is None:
                start = self.stops.Index(self.stops.index_range[0])
            else:
                start = self.index(value.start)
            if value.stop is None:
                stop  = self.stops.Index(self.stops.index_range[-1]+1)
            else:
                stop  = self.index(value.stop)
            if value.step is not None:
                raise ValueError("`DiscretizedAxis.index()` doesn't support  "
                                 "the `step` argument. If your axis is "
                                 "regular, try `RangeAxis` or `ArrayAxis`.")
            return slice(start, stop)
        else:
            if hasattr(value, 'magnitude'):
                # value_dtype = np.array(value.magnitude).dtype.type
                if not self.unit_check(value):
                    raise TypeError("Provided value ({}) does not have the "
                                    "expected units ({})."
                                    .format(value, self.unit))
                value = self.unit_remove(self.unit_convert(value))
                    # We explicitly cast to the stops_dtype, so that Index
                    # recognizes as a stop value.
                    # Otherwise things like 1*ureg.s get sent as int, and Index
                    # thinks they are indices.
                if not np.can_cast(value, self.stops_dtype):
                    raise TypeError(
                        "`index` expects an input of type "
                        f"{self.stops_dtype} (received {type(value)}).")
                value = self.stops_dtype(value)
            # else:
            #     value_dtype = np.array(value).dtype.type
            return self.stops.index(value)

    def data_to_axis_index(self, data_index :Integral) -> "AxisIndex":
        return self.stops.data_to_axis_index(data_index)
    data_to_axis_index.__doc__ = SequenceMapping.data_to_axis_index.__doc__

    def axis_to_data_index(self,
                           axis_index :Union[Integral, AbstractAxisIndex]):
        return self.stops.axis_to_data_index(axis_index)

    def index_interval(self, value1, value2):
        """
        Return the number of stops between two values.
        Implemented as `self.index(value2) - self.index(value1)`.
        """
        return self.index(value2) - self.index(value1)

    def data_index_slice(self, axis_slice, include_padding=False):
        if self.is_compatible_value(axis_slice.start):
            axis_slice.start = self.index(axis_slice.start)
        if self.is_compatible_value(axis_slice.stop):
            axis_slice.stop = self.index(axis_slice.stop)
        return self.stops.data_index_slice(axis_slice, include_padding)

    def pad(self, pad_left, pad_right=0):
        """
        Already present padding is subtracted, so that
        >>> axis.pad(2)
        >>> axis.pad(3)
        has a padding of 3, and not 5.

        Typically some data structure is associated with the axis, and it
        should also be padded. For this reason the method returns a tuple of
        integers, indicating the number of bins it actually added. So in the
        example above, the first line would return `(2,0)`, and the second,
        `(1,0)`.

        .. Note:: This method simply increases the range of indexes.
           It is the user's responsibility to ensure that underlying
           `index_map` is valid over the expanded range. (With RangeAxis this
           is always true.)

        Parameters
        ----------
        pad_left
        pad_right: int | float [units] (not all Axes) | AxisIndexDelta
            Number of bins to add to the before (after) the axis.
            Integer and AxisIndexDelta specify number of bins directly.
            Float values must have units matching those of the axis.
            Axis must support single argument `index_interval` for this
            method to accept floats (`RangeAxis` does so).

        Return
        ------
        tuple of int
            2-element tuple giving the number of bins added to the left and
            right.
        """
        if self.unit_check(pad_left):
            try:
                pad_left = self.index_interval(self.unit_convert(pad_left))
            except TypeError as e:
                raise TypeError(
                    "Calling `pad` with floating point values is only "
                    "possible for axes with a fixed step (like `RangeAxis`)."
                    "\nThe internal error raised by `index_interval` printed "
                    f"produced the following message:\n\t\"{e}\"")
        elif not self.Index.is_compatible(pad_left):
            raise TypeError("`pad_left` is neither a float with compatible "
                            "units, nor compatible with the axis index type.")
        if self.unit_check(pad_right):
            try:
                pad_right = self.index_interval(self.unit_convert(pad_right))
            except TypeError as e:
                raise TypeError(
                    "Calling `pad` with floating point values is only "
                    "possible for axes with a fixed step (like `RangeAxis`)."
                    "\nThe internal error raised by `index_interval` printed "
                    f"produced the following message:\n\t\"{e}\"")
        elif not self.Index.is_compatible(pad_right):
            raise TypeError("`pad_right` is neither a float with compatible "
                            "units, nor compatible with the axis index type.")
        return self.stops.pad(pad_left, pad_right)

    def realign(self, new_align):
        """
        Align the axis to either edges or centers, based on the value of
        `new_align`. If the current axis is already aligned accordingly, it
        is simply returned, otherwise a new axis is created.

        Parameters
        ----------
        new_align: str | BinAlign

        Returns
        -------
        DiscretizedAxis
        """
        new_align = BinAlign.get(new_align)
        if new_align is self.bin_align:
            return self
        elif new_align is BinAlign.EDGES:
            return self.edges
        elif new_align is BinAlign.CENTERS:
            return self.centers
        else:
            raise RuntimeError

    @property
    @abstractmethod       # Force derived classes to implement, even if it's
    def edges(self):      # just super().edges
        """
        Return an Axis instance where stops correspond to bin edges.
        If this is already this axis' alignment, it is simply returned;
        otherwise, it is converted. E.g. if the current alignment is 'centers',
        the returned axis will have stops such that produced bins are
        centered around the current stops.
        Whether "centered" means in the current or transformed space depends
        on the value of `self.bin_ref`.
        **NOTE** Since there are n bins centers but n+1 bin edges, there is
        ambiguity on which 'side' to center bins. The implementation here
        computes bin widths as the distance between each center, using the
        distance between the last two centers twice (for both last and second
        to last bin)
        """
        # BinAlign = DiscretizedAxis.BinAlign
        # BinRef   = DiscretizedAxis.BinRef
        try:
            return self._edges_axis
        except AttributeError:
            pass
        if self.bin_align is BinAlign.EDGES:
            object.__setattr__(self, '_edges_axis', self)
        else:
            assert self.bin_align is BinAlign.CENTERS
            if self.bin_ref is BinRef.SELF:
                desc = self.desc
                stops = np.array(self.stops)
            else:
                assert self.bin_ref is BinRef.TRANSFORMED
                desc = self.transformed_desc
                stops = np.array(self.transformed_stops)
            # We just create the arrays: the conditional needed for the extra
            # bin would add too much overhead  (b/c arrays are pretty efficient)
            dxs = np.diff(stops)/2
            desc['stops'] = np.concatenate(((stops[0]-dxs[0],),
                                            (stops[1:] - dxs),
                                            (stops[-1] + dxs[-1],)))
            desc['bin_align'] = BinAlign.EDGES
            axis = ArrayAxis(**desc)
            if self.bin_ref is BinRef.TRANSFORMED:
                axis = axis.transformed
            object.__setattr__(self, '_edges_axis', axis)

        return self._edges_axis

    @property
    @abstractmethod       # Force derived classes to implement, even if it's
    def centers(self):    # just super().edges
        """
        Return an Axis instance where stops correspond to bin centres.
        If this is already this axis' alignment, it is simply returned;
        otherwise, it is converted. E.g. if the current alignment is 'edges',
        the returned axis will have stops at the center of each bin.
        Whether the stops are centered in the current or transformed space
        depends on the value of `self.bin_ref`.
        """
        # BinAlign = DiscretizedAxis.BinAlign
        # BinRef   = DiscretizedAxis.BinRef
        try:
            return self._centers_axis
        except AttributeError:
            pass
        if self.bin_align is BinAlign.CENTERS:
            object.__setattr__(self, '_centers_axis', self)
        else:
            assert self.bin_align is BinAlign.EDGES
            if self.bin_ref is BinRef.SELF:
                desc = self.desc
                stops = self.padded_stops_array
            else:
                assert self.bin_ref is BinRef.TRANSFORMED
                desc = self.transformed_desc
                stops = self.transformed_stops.padded_stops_array
            desc['stops'] = (stops[1:]+stops[:-1])/2
            desc['bin_align'] = 'centers'
            axis = ArrayAxis(**desc)
            if self.bin_ref is BinRef.TRANSFORMED:
                axis = axis.transformed
            object.__setattr__(self, '_centers_axis', axis)

        return self._centers_axis

DiscretizedAxis.transformed_type = DiscretizedAxis


class MapAxis(DiscretizedAxis):
    """
    A memory-efficient axis which only stores a mapping between integers and
    stop values.

    Parameters
    ----------
    label
    unit
    unit_label
    min
    max
    transformed: See `Axis`.

    bin_align
    bin_ref: See `DiscretizedAxis`

    index_map: Transform
        Instance of `mackelab_toolbox.transform.Transform` which takes
        integers and returns the corresponding stop values.
        It should be monotone increasing.
    index_range: Sequence
        The set of indices corresponding to stops.
    stops: SequenceMapping
        Instead of specifying `index_map` and `index_range`, one can also
        provide the SequenceMapping directly. This is especially convenient
        to build a MapAxis from a definition, since it allows
        >>> MapAxis(**mapaxis.dict())
        to return a valid axis.
    """
    # __slots__ = ('Index',)

    # transformed_type: ClassVar[Type] = MapAxis
        # Assigned below because DiscretizedAxis not yet defined

    stops : SequenceMapping

    def __init__(self, *,
                 index_range: Sequence=None, index_map: Transform=None,
                 **kwargs):
        # if not hasattr(self, 'Index'):  #  Avoid overwriting subclass index
        #     object.__setattr__(self, 'Index', get_AxisIndex(self))
        #         # __setattr__ required because of pydantic – see Issue #655
        stops = kwargs.get('stops', None)
        if (index_map is None or index_range is None) == (stops is None):
            raise ValueError(
                "You must specify either `index_map` and `index_range`, or "
                "`stops`.")
        if stops is None:
            dtype = determine_index_dtype(len(index_range))
            Index = get_AxisIndex(None, dtype=dtype)
            v = Index.Delta(0)  # DEBUG
            stops = SequenceMapping(index_range = index_range,
                                   index_map   = index_map,
                                   index_type  = Index)
            kwargs['stops'] = stops
            Index.attach_axis(self)  # After self.stops has been defined

        super().__init__(**kwargs)

    @classmethod
    def parse_obj(cls, obj):
        """Small wrapper which adds the reference to self.Index
        (at `obj['stops']['index_type']`). If a reference to an Index is present
        it is replaced by a new one pointing to the newly created object.
        """
        if isinstance(obj, dict):
            dtype = obj['stops'].get('index_dtype', None)
        else:
            dtype = None
        Index = get_AxisIndex(None, dtype=dtype)
        if isinstance(obj, dict) and 'stops' in obj:
            obj['stops']['index_type'] = Index
        axis = super().parse_obj(obj)
        Index.attach_axis(axis)
        return axis

    # def __init__(self, label, index_map=None, index_range=None, stops=None,
    #              bin_align='centers', bin_ref='self',
    #              unit=None, unit_label=None, *args,
    #              min=None, max=None,
    #              transformed=None, _transformed_axis=None):
    #     # Below: (index_map & index_range) XOR stops
    #     if (index_map is None or index_range is None) == stops is None:
    #         raise ValueError(
    #             "You must specify either `index_map` and `index_range`, or "
    #             "`stops`.")
    #     if stops is not None:
    #         if not isinstance(stops, SequenceMapping):
    #             raise ValueError("`stops` must be an `SequenceMapping`.")
    #     else:
    #         stops = SequenceMapping(index_range, index_map, index_type=self.Index)
    #     super().__init__(label=label,
    #                      stops=stops,
    #                      bin_align=bin_align,
    #                      bin_ref=bin_ref,
    #                      unit=unit, unit_label=unit_label,
    #                      min=min, max=max, transformed=transformed,
    #                      _transformed_axis=_transformed_axis)

    def __len__(self):
        return len(self.stops)
    #
    # @property
    # def transformed_desc(self):
    #     desc = super().transformed_desc

    @property
    def transformed_stops(self):
        try:
            return self._transformed_stops
        except AttributeError:
            self._transformed_stops = self.stops.transform(self.transform)
            return self._transformed_stops

    @property
    def edges(self):
        return super().edges

    @property
    def centers(self):
        return super().centers

    # def index(self, value, **kwargs):
    #     return self.stops.index(self.unit_remove(value), **kwargs)

MapAxis.transformed_type = MapAxis

class RangeAxis(MapAxis):
    """
    A memory-efficient axis where stops are at regular intervals.
    Implemented as a :class:`MapAxis`, where the underlying index is a
    :class:`RangeMapping`.

    Implemented as follows: If ``x(i)`` is the axis value at
    index ``i``, then ``x(i) = self.x0 + (i-self.i0)*self.step``.
    ``x0`` is initialized at ``min``, ``i0`` is initialized at 0. These values
    may change when adding padding to the axis.

    Required parameters:
        label, min, max, step
    or
        label, stops

    Parameters
    ----------
    label
    unit
    unit_label
    min
    max
    transformed: See :py:class:`Axis`.

    bin_align
    bin_ref: See :py:class:`DiscretizedAxis`

    stops: :class:`RangeMapping`

    """
    __slots__ = ()

    transformed_type: ClassVar[Type] = MapAxis

    stops: RangeMapping
        # Override MaxAxis' type (was SequenceMapping)

    # ------------------------
    # Initializer & validators

    def __init__(self, step=None, **kwargs):
        stops, min, max = (kwargs.get(x, None) for x in
                           ('stops', 'min', 'max'))
        if (step is None) == (stops is None):
            raise ValueError(
                "You must specify either `step` or `stops`, and not both.")

        # Extract 'unit' from keyword args and emulate Pydantic parsing
        # If we weren't doing hackery with Index, we could do the min, max
        # stuff with validators, and we wouldn't have to parse `unit`
        unit = kwargs.get('unit', None)
        parsedunit = None
        for unitT in mtb.typing._AllUnitTypes:
            for validator in unitT.__get_validators__():
                try:
                    parsedunit = validator(unit)
                except (ValueError, TypeError, AssertionError):
                    pass
                else:
                    break
            if parsedunit is not None:
                break
        unit = parsedunit
        assert unit is not None

        if stops is None:
            if None in (min, max, step):
                raise ValueError("You must specify all of `min`, `max` and "
                                 "`step`.")
            nsteps = (max-min) / step
            if not mtb.units.is_dimensionless(nsteps):
                warn("While creating a RangeAxis, the value of `(max-min)/step` "
                     "should be dimensionless, but it has dimensions of "
                     f"{nsteps.dimensionality}. The units will be dropped, but "
                     "you should fix the parameters to ensure their units cancel.")
                nsteps = nsteps.magnitude
            nsteps = np.float64(nsteps)  # One of many ways to cast 'nsteps' to a proper NumPy type
            nsteps = int_if_close(nsteps, allow_power10=False)
                # If we just use `int` to apply the floor function, we run into
                # issues, e.g. int(10/0.1) == 99 (because 0.1 == 0.1+ε in binary)
                # int_if_close ensures small differences are rounded towards
                # the integer.
            L = int(nsteps) + 1
            # Remove all units from arguments – already stored in the axis 'unit' attribute
            if unit is not unitless:
                min = mtb.units.unit_convert(min, unit).magnitude
                max = mtb.units.unit_convert(max, unit).magnitude
                step = mtb.units.unit_convert(step, unit).magnitude
            kwargs['min'] = min; kwargs['max'] = max; kwargs['step'] = step
            idx_dtype = determine_index_dtype(L)
            Index = get_AxisIndex(None, dtype=idx_dtype)
            stops = RangeMapping(index_range=range(0, L),
                                 i0=0, x0=min, step=step,
                                 index_type=Index)
            kwargs['stops'] = stops
            super().__init__(**kwargs)
            Index.attach_axis(self)  # After self.stops has been defined
        elif isinstance(stops, dict) and 'index_type' not in stops:
            # We reach this point when parsing the dict or json export of an Axis
            # Export doesn't include 'Index' because it is tied to the Axis
            # (see SequenceMapping)
            index_range = stops['index_range']
            if isinstance(index_range, (list, tuple)):
                # Was not parsed by Pydantic
                # FIXME: Should be parsed by Pydantic, no ?
                index_range = Range.validate(index_range)
            assert isinstance(index_range, range)
            L = len(index_range)
            idx_dtype = determine_index_dtype(L)
            Index = get_AxisIndex(None, dtype=idx_dtype)
            stops = RangeMapping(**stops, index_type=Index)
            kwargs['stops'] = stops
            super().__init__(**kwargs)
            Index.attach_axis(self)  # After self.stops has been defined
        else:
            super().__init__(**kwargs)

    def copy(self, *a, **kw):
        m = super().copy(*a, **kw)
        # Update i0 so it is consistent with the new Index type
        # TODO: This should be done in RangeMapping
        Index = m.stops.Index
        m.stops.i0 = Index(Index.nptype(m.stops.i0))
        return m

    # ---------
    # Properties exposing RangeMapping

    @property
    def i0(self):
        return self.stops.i0
    @property
    def step(self):
        return self.stops.step * self.unit

    # ----------
    # Methods relative to binning

    @property
    def widths(self):
        """
        Return an ndarray of same length as `centers` giving each bin's width.
        """
        return np.broadcast_to(1, self.nbins) * self.step

    # ---------
    # Indexing methods

    def index_interval(self, value, value2=None,
                       allow_rounding=False, cast=True):
        """
        Convert `value` to the corresponding number of steps.
        If `value2` is passed, convert `value2 - value1`.
        Values with units are converted appropriately.

        Parameters
        ----------
        value: float
        value2: float | None
        allow_rounding: bool
            If set to True, function will not throw errors if the value
            is not commensurate with the step size.
        cast: bool
            When True, cast the result to `self.Index`.
            When False, return a float (or whatever the result of
            `value/self.stop` is), possibly with fractional part if
            the value difference is not a multiple of the step.

        Returns
        -------
        - (`cast` == True)  AxisIndexDelta (self.stops.step_type)
        - (`cast` == False) result of `value/self.step`; should be compatible
          with `x_dtype`.
        """
        if value2 is None:
            if not self.unit_check(value):
                raise TypeError("Provided value ({}) do not have the expected "
                                "units ({}).".format(value, self.unit))
            value = self.unit_remove(self.unit_convert(value))
        else:
            if not self.unit_check(value, value2):
                values = (value, value2)
                raise TypeError("Provided values ({}) do not have the expected "
                                "units ({}).".format(values, self.unit))
            value  = self.unit_remove(self.unit_convert(value))
            value2 = self.unit_remove(self.unit_convert(value2))
        return self.stops.index_interval(value, value2,
                                         allow_rounding=allow_rounding,
                                         cast=cast)

    def index(self, value, allow_rounding=False):
        ar = allow_rounding
        if isinstance(value, slice):
            if value.start is None:
                start = self.stops.Index(self.stops.index_range[0])
            else:
                start = self.index(value.start, allow_rounding=ar)
            if value.stop is None:
                stop  = self.stops.Index(self.stops.index_range[-1]+1)
            else:
                stop  = self.index(value.stop, allow_rounding=ar)
            if value.step is None:
                step = None
            elif shim.istype(value.step, 'int'):
                step = self.Index.Delta.make_index(value.step)
            else:
                step = self.index_interval(value, allow_rounding=ar)
            return slice(start, stop, step)
        else:
            # TODO: Warn if `allow_rounding` was set since it is discarded here
            return super().index(value)


class ArrayAxis(DiscretizedAxis):
    """
    In contrast to MapAxis, stores all stops as an array.

    Parameters
    ----------
    [See SequenceAxis]

    stops: ndarray | ArrayMapping
        If an array, used to construct an ArrayMapping.
    """
    __slots__ = ()

    # transformed_type: ClassVar[Type] = ArrayAxis

    stops: ArrayMapping

    def __init__(self, *, stops, **kwargs):
        # if not hasattr(self, 'Index'):  #  Avoid overwriting subclass index
        #     object.__setattr__(self, 'Index', get_AxisIndex(self))
        if isinstance(stops, np.ndarray):
            dtype = determine_index_dtype(len(stops))
            Index = get_AxisIndex(None, dtype)
            stops = ArrayMapping(index_range = range(len(stops)),
                                  index_map   = stops,
                                  index_type  = Index)
        kwargs['stops'] = stops
        super().__init__(**kwargs)
        Index.attach_axis(self)  # After self.stops has been defined

    # @property
    # def transformed_desc(self):
    #     return super().transformed_desc
    #
    # @property
    # def stops_array(self):
    #     return self.stops

    @property
    def transformed_stops(self):
        """Apply `self.transform` to the array of stops."""
        return self.transform(self.stops)

    @property
    def edges(self):
        return super().edges
    edges.__doc__ = DiscretizedAxis.edges.__doc__

    @property
    def centers(self):
        return super().centers
    centers.__doc__ = DiscretizedAxis.centers.__doc__

ArrayAxis.transformed_type = ArrayAxis

DiscretizedAxis.update_forward_refs()
MapAxis.update_forward_refs()
RangeAxis.update_forward_refs()
ArrayAxis.update_forward_refs()
