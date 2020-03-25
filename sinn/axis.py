import numpy as np
from enum import Enum, auto
from collections import namedtuple, Callable, Sequence
import numbers
import abc
from mackelab_toolbox.utils import comparedicts
from mackelab_toolbox.transform import Transform
from sinn import config

class Axis:
    """
    Abstract axis class. Only stores label, unit, and optional min and max
    values. For an axis with defined stops (as needed for must numerical
    applications), use `DiscretizedAxis`.
    """
    # class Parameters(ParameterSpec):
    #     schema = {'label', 'unit', 'unit_label', 'transformed', 'transform',
    #               '_min', '_max'}

    @property
    @abc.abstractmethod          # Derived class would set this as a normal
    def transformed_type(self):  # class attribute.
        return                   # @property is just to have an abstract attr.

    def __init__(self, label, unit=None, unit_label=None,
                 min=None, max=None,
                 transformed=None, _transformed_axis=None):
        """
        Parameters
        ----------
        label: str or a valid description for Bijection.
               (from mackelab_toolbox.transform.Bijection)

        …
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
        if _transformed_axis is not None:
            self.transformed = _transformed_axis
            if (isinstance(label, Bijection)
                or Bijection.Parameters.castable(label)):
                raise ValueError(
                    "Specifying both a transformation in `label` and"
                    "`_transformed_axis` is ill-defined. Also "
                    "`_transformed_axis` is reserved for "
                    "internal use by the `Axis` class.")
            transformed = False  # Use same code path as no transform,
                                 # since self.transformed is already set
        elif transformed is None:
            transformed = Bijection.Parameters.castable(label)
        if not transformed:
            self.label = label
            self.transformed = False
            self.transform = None
        else:
            self.transform = Bijection(label)
            transformed_min = min if min is None else self.transform(min)
            transformed_max = max if max is None else self.transform(max)
            self.label = self.transform.map.xname
            # self.transform = transform
        self.unit = unit
        self.unit_label = unit_label
        self._min = min
        self._max = max
        # Set unit & conversion methods. We detect types with duck typing
        # rather than testing against the actual types for two reasons:
        #   - Testing against types would force importing all quantities
        #     libraries, and therefore installing them.
        #   - In theory other libraries could implement these methods, and
        #     they would work as well.
        if hasattr(unit, 'compatible_units') and hasattr(1*unit, 'to'):
            self.unit_check = self._pint_check
            self.unit_check = self._pint_check
        elif (hasattr(unit, 'simplified') and hasattr(unit, 'dimensionality')
              and hasattr(unit, 'rescale')):
            self.unit_check = self._quantities_check
        else:
            self.unit_check = self._unknown_check
            self.unit_convert = self._unknown_convert

    def _pint_check(self, value):
        return self.unit.dimensionality in value.compatble_units()
    def _quantities_check(self, value):
        return (value.simplified.dimensionality
                == self.unit.simplified.dimensionality)
    def _unknown_check(self, value):
        raise TypeError("Unable to check units: axis unit '{}' unrecognized."
                        .format(type(self.unit)))
    def _pint_convert(self, value):
        return value.to(self.unit)
    def _quantities_convert(self, value):
        return value.rescale(self.unit)
    def _unknown_convert(self, value):
        raise TypeError("Unable to convert units: axis unit '{}' unrecognized."
                        .format(type(self.unit)))

    @property
    @abc.abstractmethod
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
            desc['label'] = self.transform.inverse_map.xname
                # Remove the transform description from label to avoid error
            # Will call derived class constructor if needed
            self.transformed = self.transformed_type(
                **desc, _transformed_axis=self)

    def __repr__(self):
        return str(type(self)) + '(' + self.desc + ')'

    @property
    def transformed_unit(self):
        if self.transform is None:
            raise AttributeError("This axis does not define a transformation.")
        try:
            transformed_unit = self.transform(unit)
        except TypeError:
            transformed_unit = None
    # Saving None allows to distinguish 'not set' from 'set to infinity'
    @property
    def min(self):
        return -np.inf if self._min is None else self._min
    @property
    def max(self):
        return np.inf if self._max is None else self._max
    @property
    def limits(self):
        return (self.min, self.max)
    @property
    def desc(self):
        desc = {}
        if self.transform is None:
            desc['label'] = self.label
        else:
            desc['label'] = self.transform.desc
        for attr in ['unit', 'unit_label', 'transformed']:
            desc[attr] = getattr(self, attr)
        desc['min'] = self._min
        desc['max'] = self._max
        return ParameterSet(**desc)
    @property
    def transformed_desc(self):
        if self.transform is None:
            raise AttributeError("This axis does not define a transformation.")
        desc = self.desc  # Ensures we don't forget attributes ?
        desc['label'] = self.transform.inverse.desc
        desc['unit'] = self.transformed_unit
        desc['min'] = min if min is None else self.transform(min)
        desc['max'] = max if max is None else self.transform(max)
        return desc

    def discretize(self, stops, format='centers', **kwargs):
        return DiscretizedAxis(
            **self.desc, stops=stops, format=format, **kwargs)

class Axes(tuple):
    """Set of multiple Axis instances"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(not isinstance(ax, Axis) for ax in self):
            raise TypeError(
                "All elements of an `Axes` must be instances of `Axis`.")


###########################
# Discretized axis objects
#
# Defines the following mapping classes:
#   - IndexedMapping(Sequence)
#   - RegularIndexedMapping(IndexedMapping)
# And the following axis classes
#   - DiscretizedAxis(Axis)
#     + abstract base class; instantiate one of the subclasses
#   - ArrayAxis
#     + Stores all stops as an array
#   - MapAxis(DiscretizedAxis)
#     + Stores stops as a mapping from index to value
#     + Uses IndexedMapping
#     + O(n) retrieval of index values because stops have to all be calculated
#       and then searched.
#     + Converting indices between two axis instances requires them to be
#       converted to values.
#   - RegularAxis(MapAxis)
#     + Preset-mapping for an axis with regular steps
#     + Uses RegularIndexedMapping
#     + O(1) retrieval of index values
#     + Direct index conversion between two RegularAxis instances with
#       commensurate step size.
# Each axis instance defines its own Index type, constructed with
# `AxisIndexMeta` which ensures that arithmetic operations are only performed
# between indices of the same axis.
# Specialized conversions allow operations between indices of two RegularAxes,
# as long as those indexes are commensurate.
##########################

class IndexTypeOperationError(TypeError):
    def __init__(self, message=None):
        if message is None:
            message = "Operations are only supported between indices of the same IndexType."
        super().__init__(message)

class AxisIndexMeta(type):
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

    Example
    -------
    >>> from sinn.axis import DiscretizedAxis, AxisIndexMeta
    >>> class MyAxis(DiscretizedAxis):
    >>>     index_dtype = np.int32
    >>> Class OtherAxis(DiscretizedAxis):
    >>>     index_dtype = np.int32
    >>>
    >>> # Create index types attached to this axes
    >>> # (For example only: Axis already provides its type)
    >>> MyIndex = AxisIndexMeta(MyAxis)
    >>> OtherIndex = AxisIndexMeta(OtherAxis)
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
    def __new__(cls, axis):
        dtype = getattr(axis, 'index_dtype', int)
        if not issubclass(dtype, (numbers.Integral)):
            raise TypeError("Axis' `index_dtype` is {}, but must be an integral type."
                            .format(dtype))
        class AxisIndex(dtype):
            axis = axis
            def __init__(self, x, base=10):
                # Neither `int` nor `np.integer` implement __init__ (everything is done in __new__)
                # `super()` thus proceeds to `object.__init__`, which expects no arguments.
                super().__init__()
            def __add__(self, other):
                if not type(self) is type(other): raise IndexTypeOperationError
                return type(self)(super().__add__(other))
            def __sub__(self, other):
                if not type(self) is type(other): raise IndexTypeOperationError
                return type(self)(super().__sub__(other))
            def __mul__(self, other):
                if not type(self) is type(other): raise IndexTypeOperationError
                return type(self)(super().__mul__(other))
            def __truediv__(self, other):
                if not type(self) is type(other): raise IndexTypeOperationError
                div = super().__truediv__(other)
                if not div.is_integer(): raise ValueError("Division must return an integer.")
                return type(self)(div)
        return AxisIndex

class IndexedMapping(Sequence):
    # Maximum allowable length
    # This is a heuristic, and could be changed if needed
    max_len = int(np.iinfo(np.int32).max / 2)

    def __init__(self, index_range, mapping, index_type):
        if not issubclass(index_type, numbers.Integral):
            raise TypeError("`index_type` must be an integral type.")
        self.Index = index_type

        if not isinstance(index_range, Sequence):
            raise ValueError("`index_range` must be a Sequence.")
        if len(index_range) > self.max_len:
            raise ValueError(
                "Length of `index_range` exceeds allowed maximum. If you need "
                "a longer index, change the value of `IndexedMapping.max_len`. "
                "len(index_range) = {}".format(len(index_range)))
        self.index_range = index_range

        if mapping is not None:
            # Subclasses may define mapping themselves
            if not isinstance(mapping, Callable):
                raise ValueError("`mapping` must be a Callable.")
            self.mapping = mapping

    def __getitem__(self, index):
        if not isinstance(index, Integral):
            raise TypeError("`index` must be in integer.")
        i0 = self.index_range[0]
        ik = self.index_range[-1]
        if isinstance(index, slice):
            step = index.step;
            if step is None: step = 1
            start = index.start;
            if start is None: start = i0
            stop = index.stop;
            if stop is None: stop = ik
            index = self.index_range[i0:ik:step]
        elif not shim.is_symbolic(index):
            if shim.isscalar(index) and (index < io or ik < index):
                raise IndexError("Provided index `{}` exceeds the mapping's "
                                 "range.".format(index))
        else:
            if np.any(index < i0) or np.any(index > ik):
                raise IndexError("Provided index `{}` exceeds the mapping's "
                                 "range.".format(index))
        return self.mapping(index)

    def __len__(self):
        return len(self.index_range)

    def __eq__(self, other):
        if isinstance(other, IndexedMapping):
            return (self.index_range == other.index_range
                    and np.all(self.stops_array == other.stops_array))
        else:
            return False

    @property
    def stops_array(self, t):
        try:
            return _stops_array
        except AttributeError:
            self._stops_array = self.mapping(self.array(self.index_range))
        return self._stops_array

    def transform(self, f):
        """Transform the mapping by applying `f` to every point."""
        def new_mapping(i): return f(self.mapping(i))
        return IndexedMapping(self.index_range, new_mapping)

class RegularIndexedMapping(IndexedMapping):
    """
    Note: To prevent rounding point errors, `i0` should not be changed.
    """
    step_dtype = np.float64  # Use a high-precision type for index calculations

    def __init__(self, index_range, i0, x0, **kwargs):
        """Value of the index at i0."""
        if i0 not in index_range:
            raise ValueError("`i0` must be within `index_range`.")
        self.i0 = i0
        self.x0 = x0
        self.step = self.step_dtype(index_range.step)
        super().__init__(index_range, None, **kwargs)

    def __getitem__(self, index):
        if not isinstance(index, Integral):
            raise TypeError("`index` must be in integer.")
        i0 = self.i0
        x0 = self.x0
        imin = self.index_range[0]
        imax = self.index_range[-1]
        step = self.step
        if np.any(index < imin) or np.any(index > imax):
            raise IndexError("Provided index `{}` exceeds the mapping's range."
                             .format(index))
        # We don't call self.mapping for efficiency
        return x0 + (index-i0)*step

    def __eq__(self, other):
        if isinstance(other, RegularIndexedMapping):
            return (self.i0 == other.i0
                    and self.x0 == other.x0
                    and self.step == other.step
                    and self.index_range == other.index_range)
        else:
            return super().__eq__(self, other)

    # Need `mapping` function to be consistent with IndexedMapping
    def mapping(self, index):
        i0 = self.i0
        x0 = self.x0
        step = self.step
        return x0 + (index-i0)*step

    def floating_point_error_check(self, value):
        if np.any(value * config.get_rel_tolerance(value) > self.step):
            raise ValueError(
                "You've tried to convert the value {} into an index, "
                "but the value is too large to ensure the absence of "
                "numerical errors. Try using a higher precision type.")

    def index(self, value, allow_rounding=False):
        """
        `allow_rounding`:
            If `value` does not correspond to an index, round to the nearest
            index. Can also be used to skip the `isclose` check.
        """
        self.floating_point_error_check(value)
        i0 = self.i0
        x0 = self.x0
        step = self.step
        i = self.Index((value - x0) / step + 0.5 + i0)
            # + 0.5 is so 9.99 rounds to 10 instead of 9
        if allow_rounding:
            return i
        else:
            if np.all(np.isclose((value - x0)/step + i0, i)):
                return i
            else:
                raise ValueError(
                    "This axis has no index corresponding to the value {}."
                    .format(value))

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

        Returns
        -------
        Integral (self.Index)
        """
        step = self.step
        if value2 is not None:
            value = value2 - value
        self.floating_point_error_check(value)
        Δi = self.Index(value / step + 0.5)
        di = value / step
        if not allow_rounding and not np.isclose(Δi, di):
            raise ValueError("Value {} is not commensurate with step size {}."
                             .format(value, step))
        return Δi if cast else di

class DiscretizedAxis(abc.ABC, Axis):
    # transformed_type = DiscretizedAxis
        # Set below, because DiscretizedAxis class must be created first
    index_dtype = np.int32  # Default value to use for index.
    class BinAlign(Enum):
        """The interpretation of stops as either bin centers or bin edges."""
        CENTERS = auto()
        EDGES = auto()
        def get(align):
            if isinstance(align, str):
                return DiscretizedAxis.BinAlign[align.upper()]
            elif isinstance(align, DiscretizedAxis.BinAlign):
                return align
            else:
                raise ValueError(
                    "`align` must be a string or Enum type compatible with {}."
                    .format(DiscretizedAxis.BinAlign.__qualname__))
    class BinRef(Enum):
        """Whether bin alignment is based on this or the transformed axis."""
        SELF = auto()
        TRANSFORMED = auto()
        def get(ref):
            if isinstance(ref, str):
                return DiscretizedAxis.BinRef[ref.upper()]
            elif isinstance(ref, DiscretizedAxis.BinRef):
                return ref
            else:
                raise ValueError(
                    "`bin_ref` must be a string or Enum type compatible with "
                    "{}.".format(DiscretizedAxis.BinRef.__qualname__))

    def __init__(self, label, stops,
                 bin_align='centers', bin_ref='self',
                 unit=None, unit_label=None, *args,
                 min=None, max=None,
                 transformed=None, _transformed_axis=None):
        """
        Parameters
        ----------
        label
        unit
        unit_label
        min
        max
        transformed: See `Axis`.

        format: str

        bin_ref: DiscretizedAxis.BinRef
            SELF: Use this axis when converting from bins to edges.
            TRANSFORMED: Use the transformed axis when converting from bins
            to edges.
            Can also specify as a string.

        """
        if not hasattr(self, 'Index'):  #  Avoid overwriting subclass index
            self.Index = AxisIndexMeta(self)
        self.stops = stops
        self.bin_align = self.BinAlign.get(bin_align)
        self.bin_ref = self.BinRef.get(bin_ref)
        super().__init__(label=label, unit=unit, unit_label=unit_label,
                         min=min, max=max,
                         transformed=transformed,
                         _transformed_axis=_transformed_axis)

        if min(stops) < self.min:
            logger.warning("Axis stops exceed its stated minimum")
        if max(stops) < self.max:
            logger.warning("Axis stops exceed its stated maximum")

        # if isinstance(format, Callable):
        #     format = format()  # In case we pass the format method rather than its evaluation
        # if format not in self.formats:
        #     raise ValueError(
        #         "`format` must be one of {}. It is '{}'."
        #          .format(', '.join(["'"+f+"'" for f in self.formats]), format))
        # else:
        #     self._format_str = format

    def __len__(self):
        return len(self.stops)
    def __str__(self):
        return (self.label + ' ({:.3}:{.3}, {} stops)'
                             .format(*self.limits, len(self)))
    def __eq__(self, other):
        # We try to short-circuit comparison by starting with cheap tests
        if self is other: return True
        if (self.transform is None) != (other.transform is None): return False
        if not isinstance(other, DiscretizedAxis): return False
        if not comparedicts(self.desc, other.desc): return False
        if (self.transform is not None and
            not comparedicts(self.transformed_desc, other.transformed_desc)):
            return False
        return True

    @property
    def desc(self):
        desc = super().desc
        desc['format'] = self.format
        desc['stops'] = self.stops
        dest['bin_ref'] = self.bin_ref
        return desc

    @property
    @abc.abstractmethod
    def transformed_desc(self):
        desc = super().transformed_desc
        desc['format'] = self.format
        desc['stops'] = self.transformed_stops
        if self.bin_ref is self.BinRef.SELF:
            desc['bin_ref'] = self.BinRef.TRANSFORMED
        else:
            assert self.bin_ref is self.BinRef.TRANSFORMED
            desc['bin_ref'] = self.BinRef.SELF
        return desc

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
    @abc.abstractmethod
    def transformed_stops(self):
        """Return the stops of the transformed axis."""
        raise NotImplementedError

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

    def index(self, value):
        """
        Return the index corresponding to `value`.
        This function must evaluate the mapping at every point; if you need
        to do this, consider using ArrayAxis or RegularAxis.
        """
        i = np.searchsorted(self.stops_array, value)
        if i > len(self) or self[i] != value:
            raise ValueError("{} not in indexed mapping.".format(value))
        else:
            return self.Index(i)

    def index_interval(self, value1, value2):
        """
        Return the number of stops between two values.
        Implemented as `self.index[value2] - self.index[value1]`.
        """
        return self.index[value2] - self.index[value1]

    # Previously called `format()`
    def realign(self, new_align):
        BinAlign = DiscretizedAxis.BinAlign
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
    @abc.abstractmethod   # Force derived classes to implement, even if its
    def edges(self):      # just super().edges
        """
        Return an Axis instance where stops correspond to bin edges.
        If this is already this axis' format, it is simply returned;
        otherwise, it is converted. E.g. if the current format is 'centers',
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
        BinAlign = DiscretizedAxis.BinAlign
        BinRef   = DiscretizedAxis.BinRef
        try:
            return self._edge_axis
        except AttributeError:
            pass
        if self.bin_align is BinAlign.EDGES:
            self._edge_axis = self
        else:
            assert self.bin_align is BinAlign.CENTERS
            if self.bin_ref is self.BinRef.SELF:
                desc = self.desc
                stops = np.array(self.stops)
            else:
                assert self.bin_ref is self.BinRef.TRANSFORMED
                desc = self.transformed_desc
                stops = np.array(self.transformed_stops)
            # We just create the arrays: the conditional needed for the extra
            # bin would add too much overhead  (b/c arrays are pretty efficient)
            dxs = (stops[1:]-stops[:-1])/2
            desc['stops'] = np.concatenate(((stops[0]-dxs[0],),
                                            (stops[1:] - dxs),
                                            (stops[-1] + dxs[-1],)))
            desc['format'] = BinAlign.EDGES
            axis = ArrayAxis(**desc)
            if self.bin_ref is self.BinRef.TRANSFORMED:
                axis = axis.transformed
            self._edge_axis = axis

        return self._edge_axis

    @property
    @abc.abstractmethod   # Force derived classes to implement, even if its
    def centers(self):    # just super().edges
        """
        Return an Axis instance where stops correspond to bin centres.
        If this is already this axis' format, it is simply returned;
        otherwise, it is converted. E.g. if the current format is 'edges',
        the returned axis will have stops at the center of each bin.
        Whether the stops are centered in the current or transformed space
        depends on the value of `self.bin_ref`.
        """
        BinAlign = DiscretizedAxis.BinAlign
        BinRef   = DiscretizedAxis.BinRef
        try:
            return self._edge_axis
        except AttributeError:
            pass
        if self.bin_align is BinAlign.CENTERS:
            self._edge_axis = self
        else:
            assert self.bin_align is BinAlign.EDGES
            if self.bin_ref is self.BinRef.SELF:
                desc = self.desc
                stops = self.stops
            else:
                assert self.bin_ref is self.BinRef.TRANSFORMED
                desc = self.transformed_desc
                stops = self.transformed_stops
            desc['stops'] = (stops[1:]+stops[:-1])/2
            desc['format'] = 'centers'
            axis = ArrayAxis(**desc)
            if self.bin_ref is self.BinRef.TRANSFORMED:
                axis = axis.transformed
            return axis
        return self._edge_axis

DiscretizedAxis.transformed_type = DiscretizedAxis

class MapAxis(DiscretizedAxis):
    """
    A memory-efficient axis which only stores a mapping between integers and
    stop values.
    """
    # transformed_type = MapAxis
        # Set below, because DiscretizedAxis class must be created first

    def __init__(self, label, mapping=None, index_range=None, stops=None,
                 format='centers', bin_ref=None,
                 unit=None, unit_label=None, *args,
                 min=None, max=None,
                 transformed=None, _transformed_axis=None):
        """
        Parameters
        ----------
        label
        unit
        unit_label
        min
        max
        transformed: See `Axis`.

        format
        bin_ref: See `DiscretizedAxis`

        mapping: callable
            Function which takes integers and returns the corresponding stop
            values.
        index_range: Sequence
            The set of indices corresponding to stops.
        stops: IndexedMapping
            Instead of specifying `mapping` and `index_range`, one can also
            provide the IndexedMapping directly
        """
        if not hasattr(self, 'Index'):  #  Avoid overwriting subclass index
            self.Index = AxisIndexMeta(self)
        # Allowing `stops` as an alternative argument makes it easier to
        # reuse methods from the parent class DiscretizedAxis.
        # Below: (mapping & index_range) XOR stops
        if (mapping is None or index_range is None) != stops is None:
            raise ValueError(
                "You must specify either `mapping` and `index_range`, or "
                "`stops`.")
        if stops is not None:
            if not isinstance(stops, IndexedMapping):
                raise ValueError("`stops` must be an `IndexedMapping`.")
        else:
            stops = IndexedMapping(index_range, mapping, index_type=self.Index)
        super().__init__(label=label,
                         stops=stops,
                         format=format,
                         bin_ref=bin_ref,
                         unit=unit, unit_label=unit_label,
                         min=min, max=max, transformed=transformed,
                         _transformed_axis=_transformed_axis)

    def __len__(self):
        return len(self.index_mapping)

    @property
    def transformed_desc(self):
        desc = super().transformed_desc

    @property
    def transformed_stops(self):
        try:
            return self._transformed_stops
        except AttributeError:
            self._transformed_stops = self.stops.transform(self.transform)
            return self._transformed_stops

    @property
    def edges(self):
        super().edges

    @property
    def centers(self):
        super().centers

    def index(self, value, **kwargs):
        if not self.unit_check(value):
            raise TypeError("Provided value ({}) does not have the expected "
                            "units ({}).".format(value, self.unit))
        return self.stops.index(self.unit_convert(value), **kwargs)

MapAxis.transformed_type = MapAxis

class RegularAxis(MapAxis):
    """
    A memory-efficient axis where stops are at regular intervals.
    Implemented as a MapAxis.
    """
    transformed_type = MapAxis

    def __init__(self, label,
                 step=None, min=None, max=None, stops=None,
                 format='centers', bin_ref=None,
                 unit=None, unit_label=None, *args,
                 transformed=None, _transformed_axis=None):
        """
        Creates regularly discretized axis. If `x(i)` is the axis value at
        index `i`, then `x(i) = self.x0 + (i-self.i0)*self.step`.
        `x0` is initialized at `min`, `i0` is initialized at 0. These values
        may change when adding padding to the axis.

        Required parameters:
            label, step, min, max
        or
            label, stops

        Parameters
        ----------
        label
        unit
        unit_label
        min
        max
        transformed: See `Axis`.

        format
        bin_ref: See `DiscretizedAxis`

        step: float
            distance between stops
        """
        if not hasattr(self, 'Index'):  #  Avoid overwriting subclass index
            self.Index = AxisIndexMeta(self)
        # Allowing `stops` as an alternative argument makes it easier to
        # reuse methods from the parent class DiscretizedAxis.
        # Below: (step & min & max) XOR stops
        if (step is None) != stops is None:
            raise ValueError(
                "You must specify either `step` or `stops`.")
        if stops is not None:
            if not isinstance(stops, RegularIndexedMapping):
                raise ValueError("`stops` must be an `IndexedMapping`.")
        else:
            if None in (min, max, step):
                raise ValueError("You must specify all of `min`, `max` and "
                                 "`stop`.")
            L = ((max - min) // step) + 1
            stops = RegularIndexedMapping(range(0, L), 0, min,
                                          index_type=self.Index)
        if max <= min:
            raise ValueError("`max` must be larger than `min`.")
        super().__init__(label=label,
                         stops=stops,
                         format=format, bin_ref=bin_ref,
                         unit=unit, unit_label=unit_label,
                         min=min, max=max, transformed=transformed,
                         _transformed_axis=_transformed_axis)

    @property
    def widths(self):
        """
        Return an ndarray of same length as `centers` giving each bin's width.
        """
        return np.broadcast_to(1, self.nbins) * self.step

    def index_interval(self, value, value2=None, **kwargs):
        """
        Convert `value` to the corresponding number of steps.
        If `value2` is passed, convert `value2 - value1`.

        Parameters
        ----------
        value: float
        value2: float | None
        **kwargs:
            Passed on to RegularIndexedMapping.index_interval.

        Returns
        -------
        Integral (self.stops.step_type)
        """
        value = self.unit_convert(value)
        if value2 is not None: value2 = self.unit_convert(value2)
        return self.stops.index_interval(value, value2, **kwargs)

class ArrayAxis(DiscretizedAxis):
    """
    In contrast to MapAxis, stores all stops as an array.
    """
    # transformed_type = ArrayAxis
        # Set below, because ArrayAxis class must be created first

    def __init__(self, label, stops,
                 format='centers', bin_ref=None,
                 unit=None, unit_label=None, *args,
                 min=None, max=None,
                 transformed=None, _transformed_axis=None):
        """
        Parameters
        ----------
        label
        unit
        unit_label
        min
        max
        transformed: See `Axis`.

        format
        bin_ref: See `DiscretizedAxis`

        stops: ndarray
        """
        super().__init__(label=label,
                         stops=stops, format=format, bin_ref=bin_ref,
                         unit=unit, unit_label=unit_label,
                         min=min, max=max, transformed=transformed,
                         _transformed_axis=_transformed_axis)
        self.stops_array = self.stops

    @property
    def transformed_desc(self):
        return super().transformed_desc

    @property
    def transformed_stops(self):
        """Apply `self.transform` to the array of stops."""
        return self.transform(self.stops)

    @property
    def edges(self):
        """
        Return an Axis instance where stops correspond to bin edges.
        If this is already this axis' format, it is simply returned;
        otherwise, it is converted. E.g. if the current format is 'centers',
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
        return super().edges

    @property
    def centers(self):
        """
        Return an Axis instance where stops correspond to bin centres.
        If this is already this axis' format, it is simply returned;
        otherwise, it is converted. E.g. if the current format is 'edges',
        the returned axis will have stops at the center of each bin
        in **transformed** space.
        """
        return super().centers

ArrayAxis.transformed_type = ArrayAxis
