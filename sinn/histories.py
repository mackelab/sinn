# -*- coding: utf-8 -*-

"""
Created on Mon Jan 16 2017

Copyright 2017-2021 Alexandre René
"""
from __future__ import annotations
    # Postponed annotations allow cleaner self-referencing models

from warnings import warn
import warnings
import abc
from abc import abstractmethod
import collections
from collections import deque
from collections.abc import Iterable, Sized, Callable
from odictliteral import odict
from copy import deepcopy
import itertools
import operator
import logging
logger = logging.getLogger("sinn.history")
import hashlib
from functools import lru_cache, partial
from enum import Enum, auto
import builtins  # Only for __setattr__ hack in History
from types import SimpleNamespace

import numpy as np
import scipy.sparse

import theano_shim as shim
import theano_shim.sparse

from types import FunctionType, MethodType
from typing import ClassVar, Tuple, Set, Optional, Any, Union
from pydantic import validator, root_validator, BaseModel, Field
from pydantic.typing import Type
import mackelab_toolbox as mtb
import mackelab_toolbox.typing
import mackelab_toolbox.iotools
from mackelab_toolbox.transform import Transform
from mackelab_toolbox.typing import (
    Array, NPValue, DType, Number, Integral, Real, PintValue, QuantitiesValue,
    FloatX)

from inspect import signature, isabstract

import mackelab_toolbox as mtb
from mackelab_toolbox.utils import (
    Stash, fully_qualified_name, broadcast_shapes,
    stablehexdigest, stableintdigest)
import mackelab_toolbox.serialize
import sinn
from sinn.common import HistoryBase, PopulationHistoryBase, KernelBase, TensorWrapper, TensorDims
import sinn.config as config
import sinn.mixins as mixins
from sinn.axis import unitless, DiscretizedAxis, RangeAxis, ArrayAxis
from sinn.convolution import ConvolveMixin, deduce_convolve_result_shape
import sinn.popterm as popterm
from sinn.utils.pydantic import initializer, add_exclude_mask


_NoValue     = sinn._NoValue
SinnOptional = config.SinnOptional

###############
###   Types are registered at end of module
###############

########################
# Exceptions & enums
class LockedHistoryError(RuntimeError):
    pass
class AxisMismatchError(TypeError):
    pass
class NotComputed(Enum):
    """The return types for a failed attempt at indexing a History."""
    NotYetComputed = auto()
    NoLongerComputed = auto()
    Expired = auto()

########################

class TimeAxis(RangeAxis):
    """
    Subclass of :ref:`RangeAxis <API-RangeAxis-User>`
    with following defaults:

      - `~RangeAxis.label`: ``'t'``
      - `~RangeAxis.unit` : `TimeAxis.time_unit`
      - `~RangeAxis.step` : `TimeAxis.time_step`

    `TimeAxis.time_unit` and `TimeAxis.time_step` can be changed by assigning
    values directly to the class (i.e. ``TimeAxis.time_step = 0.1``). This
    may be more convenient than specifying them when creating a `TimeAxis`,
    since most of the time histories will all share the same time units and
    steps. Values are initially set to:

      - `TimeAxis.time_step` : ``None``  (invalid – must be set)
      - `TimeAxis.time_unit` : ``None``  (leave unchanged to ignore units)

    .. note:: In a fully consistent naming convention, this class would be
       named :class:`TimeRangeAxis`. Since this is should be viewed as the
       default, and the one we want to use 99% of the time, we choose to hide
       the `RangeAxis` implementation detail and just name this `TimeAxis`.
    """
    ## TimeAxisMixin ##
    # This is stuff shared with TimeArrayAxis, and originally was in the form
    # of a mixin class. However, that caused hard to track diamond inheritance
    # bugs with Pydantic, and even if I could find a way to get it to work
    # nicely, it would obscure the inheritance more than it is useful to.
    label: str = 't'
    time_unit: ClassVar[Union[PintValue,QuantitiesValue]] = unitless
    unit : Optional[mtb.typing.AnyUnitType]
    @validator('unit', pre=True, always=True)
    def set_unit_default(cls, unit):
        if unit is None:
            unit = cls.time_unit
        return unit

    # Interface change to make it time-like
    @property
    def t0idx(self): return self.x0idx
    @property
    def tnidx(self): return self.xnidx
    @property
    def dt(self): return self.step
    @property
    def t0(self): return self[self.x0idx]
    @property
    def tn(self): return self[self.xnidx]
    pass

    ## TimeRangeAxis ##
    time_step: ClassVar[Real] = None
    def __init__(self, step=None, **kwargs):
        if 'stops' not in kwargs and step is None:
            step = TimeAxis.time_step
        super().__init__(step=step, **kwargs)

class TimeArrayAxis(ArrayAxis):
    """
    Subclass of :ref:`ArrayAxis <API-ArrayAxis-User>` with the following defaults:
        - `~RangeAxis.label`: ``'t'``
        - `~RangeAxis.unit` : `TimeAxis.time_unit`

    `TimeArrayAxis.time_unit` can be changed by assigning values directly to the
    class (i.e. ``TimeArrayAxis.time_unit = ureg.s``). This is may be more
    convenient than specifying them when creating a `TimeArrayAxis`, since most
    of the time histories will all share the same time units.
    Values is initially set to:
      - `TimeArrayAxis.time_unit` : ``None``  (leave unchanged to ignore units)
    """
    ## TimeAxisMixin ##
    # See comment in TimeAxis
    label: str = 't'
    time_unit: ClassVar[Union[PintValue,QuantitiesValue]] = unitless
    unit : Optional[Any]
    @validator('unit', pre=True, always=True)
    def set_unit_default(cls, unit):
        if unit is None:
            unit = cls.time_unit
        return unit

    # Interface change to make it time-like
    @property
    def t0idx(self): return self.x0idx
    @property
    def tnidx(self): return self.xnidx
    @property
    def dt(self): return self.step
    @property
    def t0(self): return self[self.x0idx]
    @property
    def tn(self): return self[self.xnidx]
    pass

# TODO: Move `batch_computable` attribute to HistoryUpdateFunction
# TODO: It would be nicer if _sym_tidx and _num_data were guaranteed of
#       Index type, then we wouldn't need to cast them to Index all the time.
class HistoryUpdateFunction(BaseModel):
    """
    Inputs are stored as string names (rather than references to the actual
    `History` instance), to allow serialization. This means we need to specify
    both the identifiers (as strings) for each input history, and a namespace
    where those identifiers are defined.

    It would be quite tricky and fragile to store the namespace during
    serialization, so it must be specified again when deserializing.
    Note that this only applies to JSON serialization – the export to a dict
    still includes the namespace by default.

    .. note:: With regards to deserialization, two different local scopes may
       come into play: the one within the function (active during function
       execution) and the one outside the function (active during function
       definition). The former is what we termed 'namespace', and can be
       determined by setting the correspondingly named parameter. It is passed
       on to the function through its ``self`` argument, same as a class
       namespace is passed on to a method. The latter (scope outside function)
       is only relevant if there are custom decorators. If necessary, it be set
       by modifying the class variable `_deserialization_locals`.

    Parameters
    ----------
    func: callable,  (namespace, Index) -> data_type  or  (Index) -> data_type
        The update function. Its signature should be ``func(self, tidx)`` or
        ``func(tidx)``. The value of `namespace` is passed to the ``self``
        argument, when it is included.
    input_names: list of history names  (List[str])
        List of histories on which the update function depends.
        Use an empty list to indicate no dependencies.
        The names should match histories in `namespace`.
        As a convenience, it is also possible to specify the histories
        themselves.
    namespace: Model | SimpleNamespace
        Any object which permits attribute access. This is where the inputs
        are retrieved from. Its value is be passed as first argument to `func`.
    cast: bool
        (Default: True) True indicates to cast the result of a the update
        function to the expected type. Only 'same_kind' casts are permitted.
        This option can be convenient, as it avoids the need to explicitly
        cast in all return functions when using mixed types. For the most
        stringent debugging though this should be set to False.
        Note: `cast` is ignored if `return_dtype` is `None`.
    return_dtype: numpy dtype or str equivalent
        The type the function should return. Designed to allow classes to
        override the default type check, which is to check the return
        value against `self.dtype`. E.g. Spiketrain does this because it
        expects indices in its update function.
        In normal usage this should never be set by a user: It is set
        automatically when assigning the update_function to a history.

    **Todo**
       - Enforce / support units in __call__ ? So we can do f(1*second) and f(3*tidx).

    Examples
    >>> from sinn.histories import HistoryUpdateFunction, Series
    >>> from types import SimpleNamespace
    >>> s = Series()
    >>> hists = SimpleNamespace(s=s)
    >>> HistoryUpdateFunction.namespace = hists
    >>> s_upd = HistoryUpdateFunction(func=Transform('tidx -> tidx**2 * s.dt'),
                                      inputs=None)
    >>> s.update_function = s_upd
    """
    ## Not intended for use by users — Set by Model to make the `updatefunction`
    ## decorator available when deserializing a function.
    ## Conceivably be used as a hook to place other variables in scope
    _deserialization_locals: dict = {}

    namespace    : Any
    input_names  : Set[str] = Field(..., alias='inputs')
    func         : Callable
    cast         : bool = True
    return_dtype : Optional[DType]
    ## Not intended for use by users
    include_namespace: bool = None  # Set by validators

    class Config:
        validate_assignment = True  # In order to re-run update_Transform_namespace
        allow_population_by_field_name = True  # Make serialize->deserialize work
        json_encoders = {**mtb.serialize.json_encoders,
                         **mtb.typing.json_encoders}

    # ---------------
    # Initializers & validators

    # def __init__(self, **kw):
    #     super().__init__(**kw)
    #     # self.validate_namespace(namespace)
    #     # object.__setattr__(self, '_history_namespace', namespace)
    #     # namespace = kw.pop('namespace', None)
    #     # if namespace is not None:
    #     #     if self.namespace == SimpleNamespace():
    #     #         # Initialize the namespace
    #     #         HistoryUpdateFunction.namespace = namespace
    #     #     elif self.namespace is not namespace:
    #     #         raise RuntimeError(
    #     #             "`HistoryUpdateFunction.namespace` is a class variable "
    #     #             "and can only be set once.")
    #     self.update_Transform_namespace()
    # def copy(self, *a, **kw):
    #     # Make a shallow copy of the namespace; in general namespace is a
    #     # module, class or Model, and we don't want to copy it.
    #     # In particular if it is a Model, this prevents ∞-recursion
    #     excl = kw.pop('exclude', None)
    #     excl = set() if excl is None else set(excl)
    #     excl.add('namespace')
    #     d = super().dict(*a, exclude=excl, **kw)
    #     d['namespace'] = self.namespace
    #     return d
    def dict(self, *a, **kw):
        """
        `namespace` is excluded by default, unless it is explicitely requested
        with the `include` keyword. This prevents it from being included in
        serializations (which would typically lead to inifinite recursion).
        """
        # Make a shallow copy of the namespace; in general namespace is a
        # module, class or Model, and we don't want to copy it.
        # In particular if it is a Model, this prevents ∞-recursion

        # if not isinstance(self.func, Transform):
        #     warn(
        #         f"The update function `{self}` is of type {type(self.func)}, "
        #         " which is not serializable. If you are attempting to export "
        #         " to JSON, use a Transform object, which supports serialization.")
        excl = kw.pop('exclude', None)
        # exclude_namespace = excl is not None and ('namespace' in excl)
        include_namespace = (kw.get('include', None) is not None
                             and 'namespace' in kw['include'])
        excl = add_exclude_mask(excl, {'namespace'})
        d = super().dict(*a, exclude=excl, **kw)
        if include_namespace:
            d['namespace'] = self.namespace
        # The input_names are stored as a set, therefore have undefined order
        # smttask relies on having consistent hashes of the serialization,
        # therefore we fix the order.
        d['input_names'] = sorted(d['input_names'])
        return d
    def json(self, *a, **kw):
        # Exclude namespace from JSON
        excl = kw.pop('exclude', None)
        excl = add_exclude_mask(excl, {'namespace'})
        # if 'func' not in excl and not isinstance(self.func, Transform):
        #     raise RuntimeError(
        #         f"Cannot export the update function `{self}` to JSON: the "
        #         f"function is of type {type(self.func)}, which is not "
        #         "serializable. To allow serialization, use a Transform object.")
        return super().json(*a, exclude=excl, **kw)

    # @classmethod
    # def parse_obj(self, *a, **kw):
    #     m = super().parse_obj(*a, **kw)
    #     m.update_Transform_namespace()
    #     return m

    @validator('input_names', pre=True)
    def set_input_names(cls, input_names):
        """
        Allow histories to be passed as inputs – replace them by their name.
        """
        input_names = set([inp.name if isinstance(inp, History) else inp
                           for inp in input_names])
        return input_names

    @validator('input_names')
    def validate_namespace(cls, input_names, values):
        """
        Raises `ValueError` if `HistoryUpdateFunction.namespace` does not
        contain all the histories in `self.inputs`.
        Not called automatically during initialization, to allow specifying
        an update function before the namespace is complete.
        """
        namespace = values.get('namespace', None)
        for nm in input_names:
            if not hasattr(namespace, nm):
                raise ValueError(f"History {nm} is not defined in the "
                                 "provided namespace.")
        return input_names

    @initializer('func', always=True)
    def parse_func_from_str(cls, func, namespace):
        """
        Allows initializing `func` with a string representation. If `func` is
        not passed as a string, this has no effect.
        There are two possible string formats:

        - Serialized `~mackelab_toolbox.transform.Transform`.
          Recognized by having no newlines, and containing the string '->'.
        - Python source.
          Recognized by starting with the string 'def ', discounting preceding
          whitespace and decorators.

        .. remark::
           Transforms are executed with `simpleeval` and should be reasonably
           safe. Python source is executed with `exec`, which with untrusted
           inputs is a major security risk. For this reason, it is only
           attempted if the configuration flag `sinn.config.trust_all_inputs`
           is set to ``True``.
        """
        if isinstance(func, str):
            if "->" in func.split('\n', 1)[0]:
                func = Transform(func)
            else:
                func = mtb.serialize.deserialize_function(
                    func, globals={}, locals=cls._deserialization_locals)
        return func

    @initializer('func', pre=False, always=True)
    def update_Transform_namespace(self, func, namespace, input_names):
        """
        If `func` is a Transform, the input histories need to be added
        to its namespace.
        """
        if isinstance(func, Transform):
            for nm in input_names:
                if hasattr(namespace, nm):
                    # This hasattr condition is there because we want to allow
                    # specifying the update function before the namespace
                    simple = func.simple
                    hist = getattr(namespace, nm)
                    if nm not in simple.names:
                        simple.names[nm] = hist  # Allow for indexing []
                    if nm not in simple.functions:
                        simple.functions[nm] = hist  # Allow for calling ()
        return func

    @initializer('include_namespace')
    def check_func_signature(cls, include_namespace, func, namespace):
        if isinstance(func, (FunctionType, partial)):
            sig = signature(func)
            if len(sig.parameters) == 1:
                warn(f"The function {func} (signature: {sig}) "
                     "defines only a time index argument, meaning that the "
                     "`namespace` attribute is ignored. To access variables in "
                     "the namespace, define the function with two arguments "
                     "(typically ``(self, tidx)``), and access namespace variables "
                     "with ``self.varname``.")
                include_namespace = False
            elif len(sig.parameters) == 2:
                include_namespace = True
            else:
                raise ValueError(
                    f"The function {func.__qualname__} (signature: {sig}) "
                    "should accept two arguments (typically ``(self, tidx)``), "
                    f"but is defined with {len(sig.parameters)}.")
        elif isinstance(func, MethodType):
            raise NotImplementedError(
                "You seem to be trying to use a method as an update function "
                "without decorating it with `@updatefunction`. This is not "
                "currently supported, although it could be if we had a clear "
                "use case for it.")
            # # DO NOT DELETE THE CODE BELOW (yet)
            # # From my testing this branch is never reached, but I think the
            # # idea is worthwhile, if we can find a use case for it.
            # if not isinstance(namespace, func.__class__):
            #     # If they aren't the same, should we use the object or
            #     # `namespace` as namespace ? Best to disallow the possibility.
            #     raise ValueError("When a method used as an update function, "
            #                      "the specified namespace must be the object "
            #                      "containing that method.\n"
            #                      f"Namespace: {namespace}\nMethod: {func}\n"
            #                      f"Class containing method: {func.__class__}")
            # elif func not in namespace.__dict__:
            #     # We passed a method, but are rebinding it to a new class instance
            #     # This happens during model copy.
            #     include_names = True
            #     func = func.__func__  # Extract bare function from method
            # else:
            #     # The 'self' argument is already set by the method
            #     include_names = False
        else:
            if not isinstance(func, Transform):
                raise TypeError(f"Argument `func` (value: {func}, type: "
                                f"{type(func)}) is neither a function, a "
                                "method nor a Transform.")
            include_namespace = False
        return include_namespace

    # ---------------
    # Standard dunders

    def __str__(self):
        input_names = ",".join(self.input_names)
        return f"f({input_names}) -> {self.return_dtype}"
    def __repr__(self):
        s = str(self)
        return f"{s} (within {self.namespace})"

    # ---------------
    # Properties

    @property
    def inputs(self):
        return [getattr(self.namespace, n) for n in self.input_names]

    # -------------
    # Function call

    def __call__(self, tidx: Integral):
        """
        Small wrapper around `func`; serves to normalize the input to a
        history's `update` method. Does two things:
        1. cast the result of `func` to the expected dtype
            This is the type stored as `return_dtype`.
            If the result of `update_function` is a list or tuple, iterates
            through the list/tuple and casts each element individually.
            This allows e.g. Spiketrain to return a list of different sized
            vectors of neuron indices.
        2. Ensure the result is at least 1d
            Update functions need output objects to have a shape.
        """
        def cast_result(result):
            if isinstance(result, list):
                result = [cast_result(r) for r in result]
            elif isinstance(result, tuple):
                result = tuple(cast_result(r) for r in result)
            elif self.return_dtype is None:
                pass
            elif self.cast:
                # Normally set 'same_kind' to True, with the exception that
                # we allow int to be cast to uint.
                same_kind = not (shim.istype(result, 'int')
                                 and np.issubdtype(self.return_dtype, np.integer))
                result = shim.cast(result, self.return_dtype.type, same_kind=same_kind)
            else:
                if result.dtype != self.return_dtype:
                    raise TypeError("Update function for history '{}' returned a "
                                    "value of dtype '{}', but history update "
                                    " expects dtype '{}'."
                                    .format(self.name, result.dtype, return_dtype))
            return result

        if self.include_namespace:
            return shim.atleast_1d(cast_result(self.func(self.namespace, tidx)))
        else:
            return shim.atleast_1d(cast_result(self.func(tidx)))

# FIXME: Currently `discretize_kernel()` CANNOT be used with Theano – it does
#        not preserve the computational graph

class History(HistoryBase, abc.ABC):
    """
    Generic class for storing a history, serving as a basis for specific history classes.
    On its own it lacks some basic fonctionality, like the ability to retrieve
    data at a certain time point.

    Ensures that if the history at time t is known, than the history at all times
    previous to t is also known (by forcing a computation if necessary).

    The special `template` parameter can be used when creating multiple
    commensurate histories.

    .. Note::**Setting the update function**
       After creation of a history, assign a HistoryUpdateFunction
       to its `update_function` attribute. This calls a setter method, which
       ensures that the update function's `return_dtype` is set correctly.
       This approach allows for defining functions for which the history itself
       is an input.
       Assigning `None` unsets the update function.

       The `~sinn.models.Model` class provides the `~sinn.models.updatefunction`
       decorator, which makes defining update functions more convenient within
       models.

    The `range_update_function` is set in the same way.

    Parameters
    ----------
    name: str
    time: TimeAxis
        The array of times this history samples.
    shape: tuple[int]
        Shape of a history slice at a single point in time.
        E.g. a movie history might store NxN frames in an TxNxN array.
        (N,N) would be the shape, and T would be (tn-t0)/dt.
    dtype: numpy dtype
    iterative: bool, optional (default: True)
        (Optional) If true, indicates that f must be computed iteratively.
        I.e. having computed f(t) is required in order to compute f(t+1).
        When false, when computed f for multiple times t, these will be passed
        as an array to f, using only one function call. Default is to force
        iterative computation.
    symbolic:  bool, optional (default: `shim.config.use_theano`)
        A value of `False` indicates that even if a symbolic library is
        loaded, values in this history are treated as data. Only updates
        which do not have any symbolic inputs are permitted, and they are
        immediately calculated using `shim.graph.eval()`.
        A value of `None` will use the value from
        `theano_shim.config.use_theano`.
        .. WARNING:: This means that any symbolic dependency (e.g. on
        shared parameters) is lost.
    template: History (optional)
        If provided, its values for `time` and `shape` are used as defaults
        for those parameters. If used, the `Axis` associated to `time` is
        copied, so as to not mix axis indices.
        # TODO: This copy should not include padding.


    The following parameters are also accepted, but should be reserved for
    internal and special usage, such as the copying histories.


    Other Parameters
    ----------------
    locked: bool
      History's locked status
    data : [depends on subclass]
      Initialization data.


    .. Note:: Each `History` must be associated to a different `TimeAxis`.
       Failing to do so can lead to incorrectly calculated indices, as padding
       between axis and data falls out of sync.
       Eventually there should be a mechanism to prevent the user from doing
       this, but for now it is not enforced. That said, because `Pydantic`
       always copies inputs, one would have to go out of their way to
       associate one `TimeAxis` to two `History` instances.

    .. Note:: Histories are **not** copied when passed as arguments to a
       `pydantic.BaseModel` (e.g. ~`sinn.models.Model`). Although this differs
       from Pydantic's conventions, for most users this is likely the most
       intuitive behaviour, and it allows the important use case of setting
       model histories by passing them as arguments.

    """
    instance_counter: ClassVar[int] = 0
    __slots__ = ('ndim', 'stash',
                 '_sym_data', '_num_data',  # Symbolic and numeric (shared) data
                 '_sym_tidx', '_num_tidx',  # Trackers for the current symbolic & concrete time indices
                 '_update_pos', '_update_pos_stop',  # See _compute_up_to
                 '_batch_loop_flag',        # See _is_batch_computable
                 '_update_function', '_range_update_function'
                 )
    @property
    def input_list(self):
        raise DeprecationWarning

    name        : str  = None
    time        : TimeAxis
    shape       : Tuple[Union[Integral,NPValue[np.integer],Array[np.integer,0]]]
    dtype       : DType
      # WARNING: Don't use actual dtypes for defaults, just strings.
      #   dtype's __eq__ is broken (https://github.com/numpy/numpy/issues/5345)
      #   and makes reasonable Pydantic code raise an exception

    iterative   : bool = True
    symbolic    : bool = None
    # Attributes which are not generally set with __init__
    # (but may be used when reconstructing from a dict())
    locked      : bool = False

    # ------------
    # Hack to allow property setters
    # (otherwise Pydantic overrides them and says the attribute is undefined)
    # Also, allows assigning to attributes in __slots__ without Pydantic
    # complaining about them being undefined
    # (required by implementation of Stash)

    def __setattr__(self, attr, v):
        clsattr = getattr(type(self), attr, None)
        if isinstance(clsattr, builtins.property):
            fset = clsattr.fset
            if fset is None:
                raise AttributeError("can't set attribute")
            else:
                return clsattr.fset(self, v)
        elif attr in set().union(*(getattr(C, '__slots__', set())
                                   for C in type(self).mro())):
            object.__setattr__(self, attr, v)
        else:
            super().__setattr__(attr, v)

    # ------------
    # Initializer and validators

    class Config:
        # # TODO: Remove extra: 'allow'
        # extra = 'allow'
        json_encoders = {**HistoryUpdateFunction.Config.json_encoders,
                         **mtb.typing.json_encoders}

    # Register subclasses so they can be deserialized
    def __init_subclass__(cls):
        if not isabstract(cls):
            mtb.iotools.register_datatype(cls)

    def __init__(self, *,
                 template :History=None,
                 data :Any=None,
                 update_function=None, range_update_function=None,
                 **kwargs):
        if template is not None:
            if kwargs.get('time', None) is None:
                kwargs['time'] = template.time.copy()
            if kwargs.get('shape', None) is None:
                kwargs['shape'] = template.shape
        super().__init__(**kwargs)
        # Set internal variables

        # The update_function setter takes care of attaching history
        # (This clears the data, so do it before assigning to tidx & data)
        msg = ("Attempting to recreate a serialized update function. "
               "Typically this is more reliably done by deserializing "
               "a Model, which appropriately links the model instance.")
        if isinstance(update_function, dict):
            warn(msg)
            update_function = HistoryUpdateFunction.parse_obj(update_function)
        self.update_function = update_function
        if isinstance(range_update_function, dict):
            warn(msg)
            range_update_function = HistoryUpdateFunction.parse_obj(
                range_update_function)
        self.range_update_function = range_update_function

        # Initialize symbolic & concrete data & index
        object.__setattr__(self, 'ndim', len(self.shape))
        data, tidx = self.initialized_data(data)
        # Data storage. Values > _num_tidx are undefined
        object.__setattr__(self, '_num_data', data)
        object.__setattr__(self, '_sym_data', self._num_data)
        # Tracker for the latest time bin for which we know history
        object.__setattr__(self, '_num_tidx', tidx)
        object.__setattr__(self, '_sym_tidx', self._num_tidx)
        # Allow for shared variables & tuples of shared vars, but not symbolic tensors
        if shim.is_pure_symbolic(self._num_data):
            raise AssertionError(f"{cls.__qualname__}.parse_obj: `_num_data` "
                                 "must not be a symbolic variable.")
        # Ensure _num_tidx is a shared variable
        if not shim.isshared(self._num_tidx):
            raise AssertionError(f"{cls.__qualname__}.parse_obj: `_num_tidx` "
                                 "must be a shared variable.")
        # Trackers for the dynamic updates – These are always initialized to
        # `None`, same as _num_tidx is always initialized to _sym_tidx
        object.__setattr__(self, '_update_pos'     , None)
        object.__setattr__(self, '_update_pos_stop', None)

        # Following allows to stash Theano updates
        object.__setattr__(
            self, 'stash', Stash(self, ('_sym_tidx', '_num_tidx'),
                                       ('_sym_data', '_num_data')))

    def copy(self, *args, **kwargs):
        """
        .. Note:: The copied history has no associated update function.
           If needed, attach a new update function to the copied history; note
           that this will clear the data.

           Rational:
           This is because we allow arbitrary functions for updates, and it
           is not always possible to replace the possible references to `self`
           in such a function. We could do so for non-iterative histories, or
           functions defined with Transform, but copying update functions only
           some of the time would be confusing.
           Besides, if there are other input histories, there is no way of
           knowing whether some of them were copied, and whether those terms
           in the function should also be replaced by the copied histories.

           Conserving the update function when copying Models, however, should
           be feasible and could be implemented if it found a use.
        """
        if kwargs.get('deep', False):
            raise NotImplementedError(
                "`copy` with `deep=True` is not currently supported with "
                "Histories. Note that even a normal copy does a deep copy "
                "of the underlying data.")
        # Private attributes aren't copied over with _construct, and
        # `__init__` is not executed by `copy()`
        m = super().copy(*args, **kwargs)
        object.__setattr__(m, 'ndim', self.ndim)

        # Remove update function
        # The update_function setter takes care of attaching history
        # Set before the tidx, because setting update_function class clear()
        # (not an issue if we set underlying _update_function instead)
        object.__setattr__(m, '_update_function', None)
        object.__setattr__(m, '_range_update_function', None)

        # Don't copy trackers for dynamic updates
        object.__setattr__(m, '_update_pos'     , None)
        object.__setattr__(m, '_update_pos_stop', None)

        # Copy the data and time index tracker
        # Simply copying the shared var would still point to the same
        # memory location
        # data may be stored as an array (Series) or tuple of arrays (Spiketrain)
        if shim.isarray(self._num_data):
            num_copy = deepcopy(self._num_data)
        else:
            assert isinstance(self._num_data, tuple)
            num_copy = tuple(deepcopy(v) for v in self._num_data)
        object.__setattr__(m, '_num_data', num_copy)
        object.__setattr__(m, '_sym_data', m._num_data)
        object.__setattr__(m, '_num_tidx',
                           shim.shared(np.array(self.cur_tidx.copy(),
                                                dtype=m.time.index_dtype),
                                       name = ' t idx (' + m.name + ')',
                                       symbolic = m.symbolic))
        object.__setattr__(m, '_sym_tidx'      , m._num_tidx)

        # Setup stash
        object.__setattr__(
            m, 'stash', Stash(m, ('_sym_tidx', '_num_tidx'),
                                 ('_sym_data', '_num_data')))
        return m

    def dict(self, *args, exclude=None, **kwargs):
        d = super().dict(*args, exclude=exclude, **kwargs)
        if isinstance(exclude, set):
            # Need to convert to dict for nested 'exclude' arg
            exclude = {attr:... for attr in exclude}
        upd_fn = self.update_function
        if upd_fn is None:
            d['update_function'] = upd_fn
        else:
            d['update_function'] = upd_fn.dict(
                *args, exclude=exclude.get('update_function', None), **kwargs)
        rg_upd_fn = self.range_update_function
        if rg_upd_fn is None:
            d['range_update_function'] = rg_upd_fn
        else:
            d['range_update_function'] = rg_upd_fn.dict(
                *args, exclude=exclude.get('range_update_function', None), **kwargs)
        d['data'] = self.get_data_trace(include_padding=True)
        return d

    @classmethod
    def parse_obj(cls, obj):
        if not isinstance(obj, dict):
            raise ValueError(f"{cls.__qualname__}.parse_obj expects a dictionary. "
                             f"Received {str(obj)[:25]}{'...' if len(obj) > 25 else ''} "
                             f"(type: {type(obj)}).")
        obj = obj.copy()  # Don't modify the original
        data       = obj.pop('data')
        upd_f      = obj.pop('update_function')
        rg_upd_f   = obj.pop('range_update_function')
        m = super().parse_obj(obj)
        name = m.name
        time = m.time
        shape = m.shape
        ndim  = len(shape)
        object.__setattr__(m, 'ndim', ndim)
        # The update_function setter takes care of attaching history
        # (This clears the data, so do it before assigning to tidx & data)
        if upd_f is not None:
            m.update_function = HistoryUpdateFunction.parse_obj(upd_f)
        if rg_upd_f is not None:
            m.range_update_function = HistoryUpdateFunction.parse_obj(rg_upd_f)
        # Initialize symbolic & concrete data & index
        data, tidx = m.initialized_data(data)
        object.__setattr__(m, '_num_data', data)
        object.__setattr__(m, '_sym_data', m._num_data)
        object.__setattr__(m, '_num_tidx', tidx)
        object.__setattr__(m, '_sym_tidx', m._num_tidx)
        if shim.is_pure_symbolic(m._num_data):
            raise AssertionError(f"{cls.__qualname__}.parse_obj: `_num_data` "
                                 "must not be a symbolic variable.")
        if not shim.isshared(m._num_tidx):
            raise AssertionError(f"{cls.__qualname__}.parse_obj: `_num_tidx` "
                                 "must be a shared variable.")
        # Initialize dynamic update trackers
        object.__setattr__(m, '_update_pos'     , None)
        object.__setattr__(m, '_update_pos_stop', None)
        # Stash
        object.__setattr__(
            m, 'stash', Stash(m, ('_sym_tidx', '_num_tidx'),
                                 ('_sym_data', '_num_data')))
        return m

    def json(self, *a, **kw):
        # Exclude namespace from update_function JSON
        excl = kw.pop('exclude', None)
        if excl is not None and not isinstance(excl, (dict, set)):
            raise TypeError(f"Method 'json' of history '{self}': argument "
                            "'exclude' should be either a set or dict.")
        excl = add_exclude_mask(excl, {'update_function': {'namespace'}})
        return super().json(*a, exclude=excl, **kw)

    # Overwrite `validate` so that Histories are not copied when passed as arguments
    # See https://github.com/samuelcolvin/pydantic/issues/1246#issuecomment-623399241
    @classmethod
    def validate(cls: Type['Model'], value: Any) -> 'Model':
        if isinstance(value, cls):
            return value
        else:
            return super().validate(value)

    @validator('name', pre=True, always=True)
    def default_name(cls, v):
        if v is _NoValue:
            cls.instance_counter += 1
            v = f"{cls}{cls.instance_counter}"
        return v

    @validator('shape')
    def only_ints_for_shape(cls, v):
        "Shape arguments must be composed of plain its, not scalar arrays."
        return tuple(int(s) for s in v)

    @validator('dtype', pre=True, always=True)
    def normalize_dtype(cls, v):
        if v is not None:
            v = np.dtype(v)
        return v

    @validator('symbolic', pre=True, always=True)
    def default_symbolic(cls, v):
        if v is None:
            v = shim.config.use_theano
        elif v and not shim.config.use_theano:
            raise ValueError(
                "You are attempting to construct a symbolic series but the"
                "symbolic library it is not loaded. Run "
                "`shim.load('theano')` before constructing this history.")
        return v

    # Derived classes must define how they store data
    @abstractmethod
    def initialized_data(self, init_data=None):
        raise NotImplementedError(
            "Derived classes must define their data storage")

    # ------------------
    # __str__ and __repr__

    def __str__(self):
        s = f"{type(self).__name__} '{self.name}'"
        s += f" (cur_tidx:{self.cur_tidx}, time:{self.time})"
        return s

    # ----------
    # Properties

    # Type definitions
    @property
    def t_dtype(self):
        return self.time.dtype
    @property
    def tidx_dtype(self):
        return self.time.index_dtype
    @property
    def idx_dtype(self):
        return np.min_scalar_type(max(self.shape))
    # I don't use `idx_dtype` very often, and a property is easier to manage
    # than a private attribute with pydantic.

    @property
    def t0idx(self):
        return self.time.t0idx
    @property
    def tnidx(self):
        return self.time.tnidx
    @property
    def t0(self):
        return self.time.t0
    @property
    def tn(self):
        return self.time.tn
    @property
    def dt(self):
        return self.time.dt

    @property
    def pad_left(self):
        return self.time.pad_left
    @property
    def pad_right(self):
        return self.time.pad_right
    @property
    def padding(self):
        """Returns a tuple with the left and right padding."""
        warn("Use `.pad_left` and/or `.pad_right`", DeprecationWarning)
        return (self.time.pad_left, self.time.pad_right)

    @property
    def timeaxis(self):
        warn("Use `.time`.", DeprecationWarning)
        return self.time

    @property
    def cur_tidx(self):
        """
        Returns the time index up to which the history has been computed.
        Returned index is not corrected for padding; to get the number of bins
        computed beyond t0, do ``hist.cur_tidx - hist.t0idx + 1``.
        """
        curtidx = self._num_tidx.get_value()
        if curtidx > self.tnidx:
            logger.warning("Current time of history {} index exceeds its "
                           "time array. Using highest valid value. "
                           "({} instead of {})"
                           .format(self.name, self.tnidx, curtidx))
            curtidx = self.tnidx
        return self.time.Index(curtidx)

    @property
    def cur_t(self):
        """
        Returns the time up to which the history has been computed.
        Equivalent to `self._tarr[self.cur_tidx]`.
        """
        return self.time[self.cur_tidx]

    # We use getter+setter for update functions to ensure they are correctly
    # attached to the history (i.e. their return type matches)
    @property
    def update_function(self):
        """
        The `update_function` is called when a history must update a time
        point. Occasionally it may not be required (e.g. if a history serves
        only to view pre-computed data), but in most cases it is.

        **Side-effects**
            When assigning to this property,
            `self.clear()` is called to invalidate preexisting data.
        """
        return self._update_function

    # We split the setter into two methods, so that Model._base_initialize can
    # call _set_update_function without the side-effect of clearing the history
    def _set_update_function(self, f: HistoryUpdateFunction):
        assert isinstance(f, HistoryUpdateFunction) or f is None, "`f` must be a HistoryUpdateFunction"
        if shim.pending_updates():
            symb_upds = shim.get_updates().keys()
            raise RuntimeError(
                f"Can't assign a new update function to history '{self.name}' "
                f"while there are pending updates: {symb_upds}.")
        if f is not None:
            f.return_dtype = self.dtype
        object.__setattr__(self, '_update_function', f)
    @update_function.setter
    def update_function(self, f: HistoryUpdateFunction):
        self._set_update_function(f)
        # Invalidate any existing data
        if hasattr(self, '_num_data'):
            # During instantiation, update function is assigned before creating
            # the data structure; nothing to clear in that case.
            self.clear()

    @property
    def range_update_function(self):
        """
        A range update function is optional. When a history must update
        multiple time points at once, it will prefer this function over
        making multiple calls to `~History.update_function()`.

        **Side-effects**
            When assigning to this property,
            `self.clear()` is called to invalidate preexisting data.
        """
        return self._range_update_function
    
    def _set_range_update_function(self, f :HistoryUpdateFunction):
        assert isinstance(f, HistoryUpdateFunction) or f is None
        if shim.pending_updates():
            symb_upds = shim.get_updates().keys()
            raise RuntimeError(
                f"Can't assign a new update function to history '{self.name}' "
                f"while there are pending updates: {symb_upds}.")
        if f is not None:
            f.return_dtype = self.dtype
        object.__setattr__(self, '_range_update_function', f)
    @range_update_function.setter
    def range_update_function(self, f :HistoryUpdateFunction):
        self._set_range_update_function(f)
        # Invalidate any existing data
        if hasattr(self, '_num_data'):
            # During instantiation, update function is assigned before creating
            # the data structure; nothing to clear in that case.
            self.clear()


    def __hash__(self):
        """
        Current implementation just returns the object id.
        I'm still not convinced this is a good idea, but it allows Histories
        to used in sets (Model) and tested for inclusion in sets (SimpleEval).
        """
        return id(self)
        # return stableintdigest(self.name)
            # Hashing the name is fragile – nested models may contain several
            # submodels with histories of the same name.
    @property
    def digest(self):
        return stablehexdigest(self.json())

    def __len__(self):
        """Length includes padding; equivalent to `padded_length`."""
        return len(self.time)

    def __lt__(self, other):
        # This allows histories to be sorted. IPyParallel sometimes requires this
        if isinstance(other, History):
            return self.name < other.name
        else:
            raise TypeError("'Lesser than' comparison is not supported between objects of type History and {}."
                            .format(type(other)))

    # Pickling infrastructure
    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError

    @property
    def repr_np(self):
        return self.raw()

    @classmethod
    def from_repr_np(cls, repr_np):
        return cls.from_raw(repr_np)

    def __call__(self, key):
        """
        Return the time slice at position(s) `axis_index`, triggering
        computation if necessary.

        .. Note:: The key origin is the one set by ``self.t0idx``. Indices will
           be shifted by both ``self.t0idx`` and ``self.pad_left`` in order to
           align them to the data – indices smaller than `~History.t0idx`
           (which may be negative) are treated as “more to the left” than
           `~sinn.histories.History.t0idx`.

        Parameters
        ----------
        key: axis index (AxisIndex, int) | axis value (float) | slice | array
            If an array, it must be consecutive (this is not checked).

        Returns
        -------
        ndarray (although depends on History subclass)

        Raises
        ------
        IndexError
            - If `axis_index` exceeds the limits of the time axis.
        RuntimeError
            - If `key` is an empty array.
            - If `key` is neither a scalar, a slice, or an ndarray.
        """
        # TODO: Reduce code duplication with __getitem__
        if isinstance(key, str):
            if key == 'end':
                key = self.tnidx
            # TODO?: 'all': reset history and compute from 0.
            #        Be consistent with _compute_up_to regarding padding
            else:
                accepted = ['end']
                if len(accepted) > 0:
                    accepted_str = "The accepted string values are "
                    accepted_str += ", ".join(*[f"'{s}'" for s in accepted[:-1]])
                    accepted_str += f" and '{accepted[-1]}'."
                else:
                    accepted_str = ("The only accepted string value is "
                                    f"'{accepted[0]}'.")
                raise ValueError("Unrecognized string argument. "
                                 + accepted_str)

        # Normalize index
        axis_index = self.time.index(key)
            # axis_index may be scalar, slice or array, but always positive.
        # Retrieve earliest & latest indices for bounds checking
        if shim.isscalar(axis_index):
            latest = axis_index
            earliest = axis_index
        elif isinstance(axis_index, slice):
            latest = axis_index.stop - 1
            earliest = axis_index.start
        elif shim.isarray(axis_index):
            if not shim.is_symbolic(axis_index) and len(axis_index) == 0:
                raise RuntimeError(
                    "Indexing a history with an empty array is disallowed "
                    "because Theano can't handle it. "
                    f"History: {self.name}, index: {key}.")
            latest = shim.largest(axis_index)
            earliest = shim.smallest(axis_index)
        else:
            raise RuntimeError("Unrecognized key {} of type {}. (history: {})"
                               .format(key, type(key), self.name))
        # Check bounds
        try:
            if not (earliest.in_bounds and latest.in_bounds):
                raise IndexError(f"Index {axis_index} is out of bounds for "
                                 f"history {self.name}.")
        except shim.graph.TooCostly:
            pass
        # Ensure axis has been sufficiently computed
        if self.update_function is None:
            raise RuntimeError(
                f"No update function was assigned to history {self.name}. If "
                "you only need to access values, use indexing syntax (`[]`) "
                "instead of calling syntax (`()`).")
        self._compute_up_to(latest)
        # Now return the requested index
        return self._getitem_internal(axis_index)

    def __getitem__(self, axis_index):
        """
        Check that history has been computed far enough to retrieve
        the desired timeslice, and if so retrieve that timeslice.

        .. Note:: The key origin is the one set by ``self.t0idx``. Indices will
           be shifted by both ``self.t0idx`` and ``self.pad_left`` in order to
           align them to the data – indices smaller than `~History.t0idx`
           (which may be negative) are treated as “more to the left” than
           `~History.t0idx`.

        Parameters
        ----------
        axis_index: AxisIndex (int) | slice | Real (experimental)
            AxisIndex of the position to retrieve, or slice where start & stop
            are axis indices.
            Indexing with real values (which are converted to AxisIndex) should
            work but is not exhaustively tested.

        Returns
        -------
        ndarray (if successful; exact type depends on History subclass)
        NotComputed (if unsuccessful)
            One of:

            + `NotComputed.NotYetComputed`: This time point was never computed
            + `NotComputed.NoLongerComputed`: This time point was previously
              computed, but no longer accessible. It would need to be
              recomputed, which may invalidate later time points.
            + `NotComputed.Expired`: This time point was previously computed,
              is no longer accessible, and cannot be computed again. This
              is the case if discarded computations depend on a randon number
              generator.

        Raises
        ------
        IndexError
            - If `axis_index` exceeds the limits of the time axis.
        RuntimeError
            - If `axis_index` is an empty array.
            - If `axis_index` is neither a scalar, a slice, or an ndarray.
        """
        # Normalize index
        axis_index = self.time.index(axis_index, allow_rounding=True)
            # Allowing rounding makes indexing with real time much more convenient
        # Retrieve earliest & latest indices for bounds checking
        if shim.isscalar(axis_index):
            latest = axis_index
            earliest = axis_index
        elif isinstance(axis_index, slice):
            latest = axis_index.stop - 1
            earliest = axis_index.start
        elif shim.isarray(axis_index):
            if not shim.is_symbolic(axis_index) and len(axis_index) == 0:
                raise RuntimeError(
                    "Indexing a history with an empty array is disallowed "
                    "because Theano can't handle it. "
                    f"History: {self.name}, index: {axis_index}.")
            latest = self.time.Index(shim.max(axis_index))
            earliest = self.time.Index(shim.min(axis_index))
        else:
            raise RuntimeError("Unrecognized axis_index {} of type {}. (history: {})"
                               .format(axis_index, type(axis_index), self.name))
        # Check bounds
        if not (earliest.in_bounds and latest.in_bounds):
            raise IndexError(f"Index {axis_index} is out of bounds for "
                             f"history {self.name}.")
        if shim.eval(latest, max_cost=None) > shim.eval(self._sym_tidx, max_cost=None):
            return NotComputed.NotYetComputed
        # Within bounds => retrieve the cached value
        else:
            return self._getitem_internal(axis_index)

    def __setitem__(self, key, value):
        """
        Essentially this is a wrapper for `self.update(key, value)`.
        The history may implement `update` such that it automatically broadcasts
        `value`, to allow expressions like ``history[0] = 1``.

        Parameters
        ----------
        key: index (int) | x_dtype | slice | array
            Only specify the time index; data indices are implicitly ``...``.
            Restrictions to maintain the causal assumption:
              + Assigning to partial time slices is not possible (i.e. no data
                indices)
              + slice and array keys should be continuous (`step` == 1 or None)
                and any values preceding them should already be computed.
            `key` is converted to a data index with `self.time.index`.
        value: x_dtype
            Any value broadcastable to `self.shape`. If `key` is an array or
            a slice, `value` should have a corresponding time dimension.

        Raises
        ------
        ValueError:
            - `key`: slice with step ≠ None, 1.
            - `key` is symbolic
        RuntimeError:
            - If there are pending symbolic updates.
        """
        axis_index = self.time.index(key)
            # axis_index may be scalar, slice or array, but always positive.
        if isinstance(axis_index, slice) and axis_index.start == axis_index.stop:
            # Empty slice => nothing to do
            return
        elif (hasattr(axis_index, 'shape')
              and len(axis_index.shape) > 0 and axis_index.shape[0] == 0):
            # Empty array => nothing to do
            return
        if (self._sym_data is not self._num_data
            or self._sym_tidx is not self._num_tidx):
            raise RuntimeError(
                f"Can't assign to the data of history '{self.name}' "
                "while there it has pending updates.")
        if shim.is_graph_object(axis_index):
            raise ValueError("Cannot assign a value to a symbolic time index. "
                             f"Time index: {key}, value: {value}.")
        # Update the history with the new value
        self.update(axis_index, value)
            # Histories which support broadcasting `value` implement it in `update`
        # update() places the new values in _sym_data, _sym_tidx; we still
        # need to apply them to _num_data/_num_tidx
        # `eval()` should always be cheap, since `value` is numeric
        self.eval()

    @property
    def time_stops(self):
        return self.get_time_stops()
        # return self.time.unpadded_stops_array
    times = time_stops  # In interactive sessions `time_stops` is too verbose

    @property
    def traces(self):
        """
        Return a list of traces, one per history component.
        Each trace is a list of (t, h_α(t)) tuples, where t is a time,
        and h_α(t) is the value of component α of the history at time t.
        The list of traces is flat, independent of the histories dimensions.
        To obtain the list of component indices, in the right order, do

            np.ndindex(self.shape)

        .. Note:: Values are returned as lists to avoid unnecessary copying.

        Returns
        -------
        List[List[Tuple[float, float]]]
            List of traces.
            Each trace: list of tuples.
            Each tuple: (time, value)
        """
        # NOTE: One could probably do something fancy with the duplicate t
        #       to minimize memory footprint, but that beyond the current scope
        data = self.get_data_trace(include_padding=False)
        times = self.time.unpadded_stops_array
        return [[(t, data[(tidx, *xidx)]) for tidx, t in enumerate(times)]
                for xidx in np.ndindex(self.shape)]

    # `trace` was never useful, because a) of history components
    # and b) the returned shape didn't plot well anyway
    # @property
    # def trace(self):
    #     """
    #     Return the tuple (`time_array`, `data`).
    #     The time stops of `data` matches those of `time_array`.
    #     """
    #     # slc = slice(self.t0idx, self.cur_tidx+1)
    #     return (self.get_time_stops(),
    #             self.get_data_trace())


    @property
    def data(self):
        """
        Returns the data which has been computed. This does not include
        in-progress symbolic updates.
        """
        return self.get_data_trace()

    @property
    def padded_length(self):
        """Same as __len__; provided for interface consistency."""
        return self.time.padded_length
    @property
    def unpadded_length(self):
        return self.time.unpadded_length


    def clear(self,after=None):
        """
        Invalidate the history data, forcing it to be recomputed the next time it is queried.
        Functionally equivalent to clearing the data, keeping the padding.
        Discards symbolic updates by calling `~History.theano_reset()`.

        *Note* If this history is part of a model, you should use that
        model's :py:meth:`~Model.clear_history()` method instead.

        Parameters
        ----------
        after: AxisIndex
            If given, history will only be cleared after this point.
            `cur_tidx` will be set to `after`, rather than `t0idx-1`.
        """
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        if after is not None:
            after = self.time.Index(after)
            self._num_tidx.set_value(after)
        else:
            self._num_tidx.set_value(self.t0idx - 1)
        object.__setattr__(self, '_sym_tidx', self._num_tidx)

        self.theano_reset()

        try:
            super().clear()
        except AttributeError:
            pass

    def lock(self, warn=True):
        """
        Lock the history to prevent modifications.
        Raises a warning if the history has not been set up to its end
        (since once locked, it can no longer be updated). This can be disabled
        by setting `warn` to False.
        """
        if self._sym_tidx != self._num_tidx:
            raise RuntimeError("You are trying to lock the history {}, which "
                               "in the midst of building a Theano graph. Reset "
                               "it first".format(self.name))

        if warn and self.cur_tidx <= self.t0idx:
            warnings.warn("You are locking the empty history {}. Trying to "
                           "evaluate it at any point will trigger an error."
                           .format(self.name))
        elif warn and (self.cur_tidx < self.tnidx
                       and not self.symbolic):
            warnings.warn("You are locking the unfilled history {}. Trying to "
                           "evaluate it beyond {} will trigger an error."
                           .format(self.name, self.cur_tidx))
        self.locked = True

    def unlock(self):
        """Remove the history lock."""
        # warn(f"Unlocking history {self.name}. Note that unless you are "
        #      "debugging, you shouldn't call this function, since sinn assumes "
        #      "that locked histories will never change.")
        if shim.pending_updates():
            warn(f"Failed unlocking history {self.name}. Histories cannot be "
                 "unlocked while there are pending updates.\n"
                 f"Pending updates: {shim.get_updates().keys()}")
        else:
            self.locked = False

    def truncate(self, start, end=None, allow_rounding=True, inplace=False):
        """
        .. Note:: Padding is always removed.

        .. FIXME:: `allow_rounding` is currently ignored and effectively `False`

        Parameters
        ---------   -
        start: idx | time
            If `None`, no initial truncation. In particular, keeps any padding.
            If `end` is given, initial time of the truncated history.
            If `end` is omitted, value is used for `end` instead. `start` is
            set to `None`.
        end: idx | time
            Latest time of the truncated history.
        allow_rounding: bool
            Whether to allow rounding start and end times to the nearest time
            index. Default is `True`.
        inplace: bool
            Whether to modify the present history inplace, or create and modify
            a copy. Default is to make a copy.

        Returns
        -------
        Series:
            Truncated history.
        """
        # TODO: if inplace=False, return a view of the data
        # TODO: invalidate caches ?
        # TODO: check lock
        # TODO: Theano _sym_data ? _sym_tidx ?
        # TODO: Sparse _sym_data (Spiketrain)
        # TODO: Can't pad (resize) after truncate
        warn("Function `truncate()` is a work in progress.")
        if self._sym_tidx != self._num_tidx:
            raise NotImplementedError("There are uncommitted symbolic updates.")

        if end is None:
            end = start
            start = None

        new_data_index = self.time.data_index(slice(start, end))
        imin = new_data_index.start
        imax = new_data_index.stop - 1

        # imin = 0 if start is None else self.get_tidx(start)
        # imax = len(self._tarr) if end is None else self.get_tidx(end)

        hist = self if inplace else self.deepcopy()
        # TODO: Don't copy _sym_data

        if (shim.issparse(hist._sym_data)
            and not isinstance(hist._sym_data, sp.sparse.lil_matrix)):
            object.__setattr__(hist, '_sym_data', self._sym_data.tocsr()[imin:imax+1].tocoo())
                # CSC matrices can also be indexed by row, but even if _sym_data
                # is CSC, converting to CSR first doesn't impact performance
        else:
            object.__setattr__(hist, '_sym_data', self._sym_data[imin:imax+1])  # +1 because imax must be included
        # TODO: use `set_value` ?
        object.__setattr__(hist, '_num_data', hist._sym_data)
        hist._tarr = self._tarr[imin:imax+1]
        if self.t0idx < imin:
            hist.t0 = hist._tarr[0]
            hist.t0idx = shim.cast(0, self.tidx_dtype)
        else:
            hist.t0idx = shim.cast(self.t0idx - imin, self.tidx_dtype)
        if self.tnidx > imax:
            hist.tn = hist._tarr[-1]

        if self._num_tidx.get_value() < imin:
            hist._num_tidx.set_value(-1)
            hist._sym_tidx.set_value(-1)
        elif self._num_tidx.get_value() > imax:
            hist._num_tidx.set_value(imax)
            hist._sym_tidx.set_value(imax)
        else:
            # If truncation is done inplace, setting _num_tidx can
            # change _sym_tidx  => Get indices first
            oidx = self._num_tidx.get_value()
            cidx = self._sym_tidx.get_value()
            hist._num_tidx.set_value(oidx - imin)
            hist._sym_tidx.set_value(cidx - imin)

        return hist

    def align_to(self, hist):
        """
        Return a new history which has the same time stops as `hist`, truncated
        to not go beyond this history's times.
        """
        ε = sinn.config.get_abs_tolerance(self.dt)
        # Target interpolation times are taken from the target history
        # We make them as wide as possible, from a half step before the
        # beginning of `hist`, to a half step after the end
        # TODO: Use np.searchsorted to find bounds instead of bool indexing
        dt = min(self.dt, hist.dt)
        interp_times = hist.time_stops[
            np.logical_and(hist.time >= self.time[0] - dt/2 - ε,
                           hist.time <= self.time[-1] + dt/2 + ε)]
        return self.interpolate(interp_times)

    def interpolate(self, interp_times):
        raise NotImplementedError("`interpolate` not implemented for histories "
                                  "of type '{}'.".format(type(self).__name__))

    def _compute_up_to(self, tidx, start='symbolic'):
        """Compute the history up to `tidx` inclusive.

        .. Warning::
        For symbolic `tidx`, the logic in this function assumes them to be
        constant offsets from the current evaluated point. Things like
        ``hist._compute_up_to(5*hist._sym_tidx)`` may pass the input check,
        but will not work.

        .. Treatment of symbolic time points::
        Symbolic time indices are computable relative to an anchor time point,
        which is set to ``self._num_tidx`` (the latest non-symbolic time
        point). This is the only symbolic dependency allowed in the expressions
        for ``tidx``. Thus expressions like
        ``hist._compute_up_to(hist._sym_tidx + 1)`` are valid, but
        ``hist._compute_up_to(hist._sym_tidx + a)``, where ``a`` is symbolic,
        are not.
        To create a ``scan`` which iterates a ``scan_tidx`` through time to
        fill histories, one can then replace ``_num_tidx`` in the final
        graph by ``scan_tidx``, and all indices become relative to that moving
        time point. This is the approach used by :py:class:`Model` to generate
        the :py:meth:`Model.advance` method.

        Parameters
        ----------
        tidx: Axis index | 'end' | 'all'
            Index up to which we need to compute.
            (see :ref:`sinn-indexing` in the docs)
            Can also be a string, either
            'end' or 'all', in which case the entire history is computed. The
            difference between these is that 'end' will compute all values
            starting at the current time, whereas 'all' restarts from 0 and
            computes everything. Padding is also excluded with 'end', while it
            is included with 'all'. When compiling a Theano graph for
            non-iterative histories, 'all' results in a much cleaner graph,
            since the the computation bounds are not Theano variables.
        start: str
            Computation runs from one past the currently computed index up to `tidx`.
            The currently computed index may be either
              - 'symbolic': (default) Updating the starting point later will
                change the computation. (Current index is part of the
                computational graph.)
              - 'numeric': The current value of the current index attribute is
                retrieved (with ``shim.graph.eval(self._sym_tidx)`` and stored.
                Resulting function will always start from exactly that index
                value.
            This parameter is only meaningful if `tidx` is symbolic.

        Todo
        ----
        Batch updates are not yet supported – all updates are done
        sequentially, one time step at a time.
        """

        cur_tidx = self.time.Index(self._sym_tidx)
        if start == 'numeric':
            assert shim.graph.is_computable([cur_tidx])
            cur_tidx = shim.graph.eval(cur_idx, if_too_costly='ignore')
        else:
            assert start == 'symbolic'
            if self.locked:
                return
            elif not shim.graph.is_computable([tidx, cur_tidx],
                                            with_inputs=[self._num_tidx]):
                # Recall: _num_tidx is a shared var with current time
                # tidx and cur_tidx may be symbolic.
                raise TypeError(
                    "We cannot construct the computational graph for updates "
                    "up to arbitrary symbolic times. Use expressions such as "
                    "`hist._compute_up_to(hist._sym_tidx + 1)`. The only "
                    "allowed symbolic dependency is `self._num_tidx`."
                    "For constructing the graph filling a history up to some "
                    "arbitrary time, see `models.Model.advance`.")
        if tidx == 'end':
            start = cur_tidx + 1
            end = self.tnidx  # Exclude right padding
            replace = False
        elif tidx == 'all':
            start = 0
            end = self.time.padded_length - 1
            replace = True
        else:
            assert shim.istype(tidx, 'int')
            start = cur_tidx + 1
            end = tidx
            replace = False
        stop = end + 1    # exclusive upper bound

        # Compute the index difference between the current index and the point
        # up to which we want to compute.
        # We can always do this, even with symbolics, where there are two cases:
        # 1) Both start and end are offsets from `_num_tidx`; in this
        #    case the difference is independent of `_num_tidx` and we
        #    don't care what value it has.
        # 2) `end` does not depend on `_num_tidx` (b/c tidx == 'end',
        #    'all'). In this case the difference does depend on,
        #    `_num_tidx`, and we want it to, so we use the value stored
        #    in its shared storage.
        # Since it is useful to have non-moving points when dealing with
        # recursion, we calculate absolute positions and subtract the concrete
        # integers to get Δidx
        absstart = shim.graph.eval(start.plain, max_cost=None)  # _num_tidx is shared var,
        absstop  = shim.graph.eval(stop.plain, max_cost=None)   # so eval() works fine
        Δidx = absstop - absstart  # No dependence on _num_tidx if case 1)
        symbolic_times = shim.is_graph_object(start, end)
            # True if either `start` or `end` is symbolic
        if Δidx <= 0:
            # Nothing to compute => exit
            return
        elif symbolic_times and Δidx > 100:
            # 100 is obviously arbitrary
            warn("Theano history updates are not meant for filling them "
                 "recursively, but rather for generating an update graph "
                 "(which can be fed into a scan). You are asking to create "
                 f"an update graph with {Δidx} steps; be advised that this may "
                 "require large amounts of memory and extremely long "
                 "compilation times.")

        # From this point on we can assume somthing needs to be calculated

        if self.locked:
            raise LockedHistoryError("Cannot compute locked history {}."
                                     .format(self.name))

        #########
        # Did not abort the computation => now let's do the computation

        if self._update_pos is not None:
            assert self._update_pos_stop is not None
            # We recursed back into this function. Under the causal assumption,
            # we must have:
            #   - That we are evaluating at a point we have already calculated
            #   - That the end point is no further than a point we have already calculated
            if (absstart >= self._update_pos      # Yes, these are redundant;
                or absstop  >= self._update_pos): # that's experimental code for you: verbosity + easier to track errors
                raise AssertionError(
                    f"The update function for history '{self.name}' seems to "
                    "break the causality assumption. This can happen if the "
                    "update for time point 't' requires the value at 't'."
                )
            # Since absstart < self.update_pos, there's nothing to do.
            # => skip clean-up of recursion variables since we didn't set them.
        else:
            # We set _update_pos here to detect any (possibly accidental) recursion
            object.__setattr__(self, '_update_pos', absstart)
                # Since this variable depends on the current value of
                # `_num_tidx`, it must not appear in the final graph.
                # It is used only to track the iteration through recursions.
            object.__setattr__(self, '_update_pos_stop', absstop)
                # At some point I needed (or thought I needed) to push forward the
                # end point from within an recursed call. I leave this here in
                # case that need comes up again.
            if (self.range_update_function is not None
                and self._is_batch_computable(up_to=self.time.axis_to_data_index(stop))):
                raise NotImplementedError
                # Computation doesn't depend on history – just compute the whole
                # thing in one go
                # The time index array is flipped, putting
                # later times first. This ensures that if dependent computations
                # are triggered (which we know must be batch_computable),
                # they will also batch update.
                if replace:
                    assert (not shim.is_graph_object(self._num_data)
                            or self._num_data not in shim.config.symbolic_updates)
                    # upd_val = self.update_function(
                    #     np.array(self.times.index_range)[::-1])[::-1]
                    # if shim.is_symbolic(upd_val):
                    #     # Symbolic update => detach _sym_data from _num_data
                    #     self._sym_data = upd_val
                    #     self._sym_tidx = self.times.index_range[-1]
                    #     shim.add_update(self._num_data, self._sym_data)
                    #     shim.add_update(self._num_tidx, self._sym_tidx)
                    # else:
                    #     if shim.isshared(self._num_data):
                    #         self._num_data.set_value(upd_val)
                    #     else:
                    #         self._num_data = upd_val
                    #     self._sym_data = self._num_data
                    #     self._num_tidx.set_value(self.times.index_range[-1])
                    #     self._sym_tidx = self._num_tidx
                Index = self.time.Index
                idxlst = [Index(start+i) for i in range(Δidx)]
                    # Using a list here instead of shim.arange is preferred,
                    # because it allows the update function to know the number
                    # of time steps.
                    # Also, these lists are supposed to be short (< 10 steps)
                    # so the advantages of an array aren't so great.
                self.update(slice(start,stop),
                            self.range_update_function(idxlst[::-1])[::-1])
                                # Index(shim.arange(start,stop,dtype=Index.nptype))[::-1]
                                # )[::-1])
            else:
                i = 0
                while absstart + i < self._update_pos_stop:
                    cur_tidx = start + i       # start may be symbolic
                        # make it easier for compiler by having update and
                        # evaluation positions point to the same symbol `cur_tidx`
                    self._update_pos = absstart + i
                    try:
                        self.update(cur_tidx, self.update_function(cur_tidx))
                    except:
                        # Clean-up _update_pos before raising
                        # If we don't do this, and the calling code catches
                        # the exception and continues (or we are in an
                        # interactive session), the internal state would be
                        # inconsistent.
                        object.__setattr__(self, '_update_pos', None)
                        object.__setattr__(self, '_update_pos_stop', None)
                        raise
                    i += 1

            # We've exited the recursion – clean up variables
            object.__setattr__(self, '_update_pos', None)
            object.__setattr__(self, '_update_pos_stop', None)

    def get_time_stops(self, time_slice=slice(None, None), include_padding=False):
        """Return the time array.
        By default, the padding portions before and after are not included.
        Time points which have not yet been computed are also excluded.

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
        ndarray
        """
        slc = self.time.data_index_slice(time_slice, include_padding=include_padding)
        start = slc.start
        try:
            stop  = min(slc.stop , self.cur_tidx.data_index+1)
            if start+1 == stop or start > self.cur_tidx.data_index:
                raise IndexError
        except IndexError:
            # Can arrive here either by self.cur_tidx.data_index raising,
            # or the explicite raise IndexError above
            warn("`get_time_stops` returned an empty slice, presumably "
                "because the history hasn't been computed.\n"
                f"History: {self.name}, current tidx: {self.cur_tidx}.")
            stop = start
        return self.time.padded_stops_array[slice(start, stop)]
        # return self._tarr[self._get_time_index_slice(time_slice, include_padding)]

    @abstractmethod
    def _getitem_internal(self, key):
        raise NotImplementedError  # __getitem__ method is history type specific

    @abstractmethod
    def update(self, tidx, value):
        """Abstract method; defined in subclass."""
        raise NotImplementedError  # update function is history type specific

    def eval(self, max_cost :Optional[int]=None, if_too_costly :str='raise'):
        """
        Apply symbolic updates onto this history.

        Parameters
        ----------
        max_cost: int | None (default: None)
            Passed on to :func:`theano_shim.graph.eval`. This is a heuristic
            to guard againts accidentally expensive function compilations.
            Value corresponds to the maximum number of nodes in the
            computational graph. With ``None``, any graph is evaluated.

        if_too_costly: 'raise' | 'ignore'
            Passed on to :func:`theano_shim.graph.eval`.
            What to do if `max_cost` is exceeded.

        Returns
        -------
        None
            Updates are done in place.

        **Side-effects**
            Removes updates from :attr:`theano_shim.config.symbolic_updates`.
        """
        # Adapted from History.eval. See History.eval for docstring
        if self._num_data is self._sym_data:
            assert self._num_tidx is self._sym_tidx
            # Nothing to do
            return
        # Note: permissible to have symbolic data & numeric tidx, but
        #       not permissible to have numeric data & symbolic tidx
        kwargs = {'max_cost': max_cost, 'if_too_costly': if_too_costly}
        updates = shim.get_updates()
        # All symbolic updates should be in shim's updates dict
        if self._num_tidx in updates:
            assert self._sym_tidx is updates[self._num_tidx]
        assert self._num_data in updates
        assert self._sym_tidx is updates[self._num_tidx]
        assert self._sym_data is updates[self._num_data]
        # TODO: tidx, data = shim.eval([self._sym_tidx, self._sym_data], **kwargs)
        tidx = shim.eval(self._sym_tidx, **kwargs)
        data = shim.eval(self._sym_data, **kwargs)
        self._num_tidx.set_value(tidx)
        self._num_data.set_value(data)
        object.__setattr__(self, '_sym_tidx', self._num_tidx)
        object.__setattr__(self, '_sym_data', self._num_data)
        if self._num_tidx in updates:
            del updates[self._num_tidx]
        del updates[self._num_data]

    def time_interval(self, Δt):
        """
        If Δt is a time (float), do nothing.
        If Δt is an index (int), convert to time by multiplying by dt.
        """
        if shim.istype(Δt, 'int'):
            return shim.cast(Δt*self.dt64, dtype=self._tarr.dtype)
        else:
            return Δt

    def index_interval(self, value, value2, allow_rounding=False, cast=True):
        return self.time.index_interval(value, value2,
                                        allow_rounding=allow_rounding,
                                        cast=cast)
    index_interval.__doc__ = TimeAxis.index_interval.__doc__

    def get_time(self, t):
        """
        If t is an index (i.e. int), return the time corresponding to t_idx.
        Else just return t

        Parameters
        ----------
        t: AxisIndex | time
            If a time, `t` must have the appropriate units.
        """
        # NOTE: Copy changes to this function in Model.get_time()
        # FIXME: Should AxisIndex have a check for "index-like" ?
        #        Experience has taught us that relying on the int type for this
        #        is fraught with issues...
        #        is_compatible_index() is wrong, because it checks for suitability
        #        for arithmetic operations (so absolute conversions are disallowed,
        #        but relative conversions are allowed, which is the opposite
        #        of what we want)
        if shim.istype(t, 'int'):
            # Either we have a bare int, or an AxisIndex
            if isinstance(t, sinn.axis.AbstractAxisIndex):
                t = t.convert(self.time)
            elif isinstance(t, sinn.axis.AbstractAxisIndexDelta):
                raise TypeError(f"Can't retrieve the time corresponding to {t}: "
                                "it's a relative, not absolute, time index.")
            return self.time[t]
        else:
            assert self.time.is_compatible_value(t)
            # `t` is already a time value -> just return it
            return t

    def get_tidx(self, t, allow_rounding=False):
        """Return the idx corresponding to time t. Fails if no such index exists.
        It is ok for the t to correspond to a time "in the future",
        and for the data array not yet to contain a point at that time
        `t` may also be specified as a slice, in which case a slice of time
        indices is returned.

        Parameters
        ----------
        t: int, float, slice, array
            The time we want to convert to an index. Integers are considered
            indices and returned unchanged.

        allow_rounding: bool (default: False)
            By default, if no time index corresponds to t, a ValueError is raised.
            This behaviour can be changed if allow_rounding is set to True, in
            which case the index corresponding to the time closest to t is returned.
        """
        # NOTE: Copy changes to this function in Model.get_tidx()
        if self.time.is_compatible_value(t):
            return self.time.index(t, allow_rounding=allow_rounding)
        else:
            # assert self.time.is_compatible_index(t)
            assert shim.istype(t, 'int')
            # FIXME: See fixme in `get_time` regarding "index-like" test.
            return self.time.Index(t)

    def get_tidx_for(self, t, target_hist, allow_fractional=False):
        """
        Convert a time or time index into a time index for another history.
        """
        raise DeprecationWarning("Cast `t` with `hist.time.Index(t)`.")


    # def get_t_for(self, t, target_hist):
    #     """
    #     Convert a time or time index for indexing into another history.
    #     The type is preserved, so that if `t` is a time, a time (float) is
    #     returned, and if `t` is a time index, an time index is returned.
    #     (In fact, if `t` is a time, it is simply returned as is.)
    #     """
    #     if isinstance(t, slice):
    #         start = None if t.start is None else self.get_t_for(t.start, target_hist)
    #         stop = None if t.stop is None else self.get_t_for(t.stop, target_hist)
    #         if self.dt != target_hist.dt:
    #             raise NotImplementedError(
    #                 "Cannot convert time for history {} to {} because they "
    #                 "have diffrent steps"
    #                 .format(self.name, target_hist.name))
    #     if shim.istype(t, 'float'):
    #         return t
    #     else:
    #         return self.get_tidx_for(t, target_hist)

    def theano_reset(self):
        """Allow theano functions to be called again.
        TODO: Add argument to clear shim.config.symbolic_updates ?
        """
        if shim.pending_update(self._num_tidx, self._num_data):
            raise RuntimeError("There are pending updates in `theano_shim`. "
                               "Clear those first with `shim.reset_updates`.")
        if self.locked:
            raise RuntimeError("Cannot modify the locked history {}."
                               .format(self.name))

        object.__setattr__(self, '_sym_tidx', self._num_tidx)
        object.__setattr__(self, '_sym_data', self._num_data)
        # We shouldn't be in the middle of a compute_up_to recursion
        assert self._update_pos is None
        assert self._update_pos_stop is None

        try:
            super().theano_reset()
        except AttributeError:
            pass

    def convolve(self, kernel, t=slice(None, None),
                 kernel_slice=slice(None,None),
                 *args, **kwargs):
        """
        Small wrapper around ConvolveMixin.convolve. Discretizes the kernel and
        converts the kernel_slice into a slice of time indices. Also converts
        `t` into a slice of indices, so the :py:meth:`_convolve_*` methods can
        work with indices.

        Parameters
        ----------
        See mixins.ConvolveMixin.convolve

        Returns
        -------
        Array of shape ``shape``:
          - If `t` is a scalar and only one kernel slice is given, then
            ``shape = (self.shape)``
          - If `t` is a slice or an array, and only one kernel slice is
            given, then ``shape = (len(t), self.shape)``
          - If `t` is a scalar and multiple kernel slices are given, then
            ``shape = (K, self.shape)``, where ``K`` is the number of kernel
            slices
          - If `t` is a slice or an array, and multiple kernel slices
            are given, then `shape = (K, len(T), self.shape)`
        """
        # This copied over from Series.convolve

        # Run the convolution on a discretized kernel
        # TODO: allow 'kernel' to be a plain function

        # Resolve kernel slice
        if not isinstance(kernel_slice, slice):
            raise ValueError("Kernel bounds must be specified as a slice.")
        try:
            len(kernel_slice)
        except TypeError:
            kernel_slice = [kernel_slice]
            single_kernel = True
        else:
            single_kernel = False

        discretized_kernel = kernel.discretize_kernel(self)
        if len(discretized_kernel) == 0:
            shape = deduce_convolve_result_shape(self, kernel, t)
            if not single_kernel:
                shape = (len(kernel_slice),) + shape
            return np.zeros(shape)

        kernel_idx_slices = [ slice( *discretized_kernel.time.index(slc) )
                              for slc in kernel_slice ]

        # Convert t to time index if necessary
        tidx = self.time.index(t)

        result = super().convolve(discretized_kernel,
                                  tidx, kernel_idx_slices, *args, **kwargs)
        if single_kernel:
            return result[0]
        else:
            return shim.asarray(result)

    # # With the `freeze_types`, this method isn't useful anymore
    # def recreate_data(self):
    #     """
    #     Recreate the internal data by calling `shim.shared` on the current value.
    #     This 'resets' the state of the data with the current state of `shim`,
    #     which may be desired if a shimmed library was loaded or unloaded after
    #     creating the history.
    #     Will fail if either the data or current index is a symbolic (rather
    #     than shared) variable.
    #     This overwrites the original data; consider first making a deepcopy
    #     if you need to keep the original.
    #     """
    #     assert self._sym_tidx.get_value() == self._num_tidx.get_value()
    #     assert np.all(self._sym_data.get_value() == self._num_data.get_value())
    #     # for attrname in dir(self):
    #     for attrname in ['_num_tidx', '_num_data']:
    #         attr = getattr(self, attrname)
    #         if isinstance(attr, shim.ShimmedTensorShared):
    #             setattr(self, attrname, shim.shared(attr.get_value(), name=attr.name))
    #     self._sym_tidx = self._num_tidx
    #     self._sym_data = self._num_data

    ################################
    # Utility functions
    ################################

    def _is_batch_computable(self, up_to='end'):
        """
        Returns true if the history can be computed at all time points
        simultaneously.
        WARNING: This function is only to be used for the construction of
        a Theano graph. After compilation, sinn.inputs is cleared, and therefore
        the result of this function will no longer be valid.
        HACK: sinn.inputs is no longer cleared, so this function should no longer
        be limited to Theano graphs – hopefully that doesn't break anything else.

        Parameters
        ----------
        up_to: Data index
            Only check batch computability up to the given time index. Default
            is to check up to the end. Effectively, this adds an additional
            success condition, when the current time index is >= to `up_to`.
            See :ref:`sinn-indexing` for the difference between data index
            and axis index.
        """
        if not self.iterative:
            # Batch computable by construction
            return True
        elif self.update_function is None:
            raise RuntimeError("The update function for history {} is not set."
                               .format(self.name))
        elif shim.is_symbolic(up_to):
            # If `up_to` is symbolic, it's value could change and is therefore irrelevant
            up_to = 'end'

        # Prevent infinite recursion
        if getattr(self, '_batch_loop_flag', True):
            return False
        else:
            self._batch_loop_flag = True

        # Augment the list of inputs with their compiled forms, if they exist
        # all_inputs = set(sinn.inputs.keys())

        # Get the list of inputs.
        input_list = [h for h in self.update_function.inputs
                      if not isinstance(h, shim.config.RNGTypes)]
        if len(input_list) == 0:
            # If this history is iterative, it should have at least one input
            warn(f"The update function for history {self.name} lists no "
                 "inputs. If this is correct, you should specify it by "
                 "passing `iterative=False` when creating the history.")
            return True

        if up_to == 'end':
            up_to = self.time.padded_length-1  # padded tnidx

        assert shim.istype(up_to, 'int')

        if all( hist.locked or hist._is_batch_computable(up_to)
                  for hist in input_list):
            # The potential cyclical dependency chain has been broken
            retval = True
        elif shim.eval(up_to - self.cur_tidx) <= 0:
            return True
        # elif shim.is_graph_object(up_to):
        #     # A symbolic `up_to` can always be higher than _sym_tidx, unless
        #     # it's actually the same variable
        #     retval = (up_to == self._sym_tidx)
        # elif up_to != 'end':
        #     # A symbolic `up_to` can always be higher than the cur_tidx
        #     up_to = self.get_t_idx(up_to)
        #     retval = (up_to <= self._num_tidx.get_value())
        else:
            retval = False

        self._batch_loop_flag = False

        return retval

class PopulationHistory(PopulationHistoryBase, History):
    """
    History where traces are organized into populations.
    This is a base class collecting common functionality; in practice one
    should use one of the derived classes, like Spiketrain or PopulationSeries.
    """
    pop_sizes:  Tuple[Integral, ...]   # ... indicates variable length
    shape    : Optional[Tuple[Integral, ...]]

    def __init__(self, *, template :History=None, **kwargs):
        if template is not None:
            if kwargs.get('pop_sizes', None) is None:
                kwargs['pop_sizes'] = template.pop_sizes
        super().__init__(template=template, **kwargs)

    __hash__ = History.__hash__  # Not inherited; see https://github.com/samuelcolvin/pydantic/issues/2422#issuecomment-828439446

    @validator('pop_sizes')
    def check_pop_sizes(cls, pop_sizes):
        if not isinstance(pop_sizes, Sized):  # Tests if len() is valid
            pop_sizes = (pop_sizes,)
        if not all( shim.istype(s, 'int') for s in pop_sizes ):
            raise ValueError("'pop_sizes' must be a tuple of integers.")
        return pop_sizes

    @classmethod
    def get_shape_from_values(cls, values):
        pop_sizes = values.get('pop_sizes', None)
        if pop_sizes is not None:
            shape = (np.sum(pop_sizes),)
        else:
            shape = None
        return shape
    @root_validator
    def set_shape(cls, values):
        shape = cls.get_shape_from_values(values)
        kwshape = values.get('shape', None)
        if kwshape is not None and kwshape != shape:
            raise ValueError("Specifying a shape to Spiketimes is "
                             "unecessary, as it's calculated from pop_sizes")
        values['shape'] = shape
        return values

    @property
    @lru_cache(maxsize=1)
    def pop_idcs(self):
        """
        self.pop_idcs is a 1D array with as many entries as there are units.
        The value at each entry is that unit's population index, and so the
        the result looks something like [0,0,0,...,1,1,1,...,2,2,2,...].
        """
        return np.concatenate(
            [ [i]*size for i, size in enumerate(pop_sizes) ] )

    @property
    @lru_cache(maxsize=1)
    def pop_slices(self):
        """
        self.pop_slices is a list of slices, such that
        self.data[i][ self.pop_slices[j] ] returns the set of neurons corresponding
        to population j at time bin i
        """
        pop_slices = []
        i = 0
        for pop_size in self.pop_sizes:
            pop_slices.append(slice(i, i+pop_size))
            i += pop_size
        return pop_slices

    @property
    def npops(self):
        """The number of populations."""
        return len(self.pop_sizes)

    # FIXME: At present self.PopTerm is not a class, which
    #        can be confusing. If PopTermMeso/PopTermMicro were implemented as metaclasses,
    #        we could call the metaclass here instead, which would return a proper class
    def PopTerm(self, values):
        if isinstance(values, popterm.PopTerm):
            return values
        else:
            # TODO: Find a way w/out instantiating a PopTerm just for 'infer_block_type'
            dummy_popterm = popterm.PopTermMacro(
                self.pop_sizes, np.zeros(1), ('Macro',))
            if shim.isshared(values):
                shape = values.get_value().shape
            else:
                shape = values.shape
            block_types = dummy_popterm.infer_block_types(
                shape, allow_plain=False)
            cls = popterm.PopTerm.BlockTypes[block_types[0]]
            if shim.isshared(values):
                cls = cls.shim_class
            return cls(self.pop_sizes, values, block_types)

    # TODO: Implement pop_xxx functions as operator methods
    #       This would involve also implementing operators which return a Spiketrain
    def pop_add(self, neuron_term, summand):
        if not shim.is_theano_object(neuron_term, summand):
            assert len(self.pop_slices) == len(summand)
            return shim.concatenate([neuron_term[..., pop_slice] + sum_el
                                     for pop_slice, sum_el in zip(self.pop_slices, summand)],
                                    axis=-1)
        else:
            raise NotImplementedError

    def pop_radd(self, summand, neuron_term):
        return self.pop_add(neuron_term, summand)

    def pop_mul(self, neuron_term, multiplier):
        if not shim.is_theano_object(neuron_term, multiplier):
            assert len(self.pop_slices) == len(multiplier)
            return shim.concatenate([neuron_term[..., pop_slice] * mul_el
                                     for pop_slice, mul_el in zip(self.pop_slices, multiplier)],
                                    axis=-1)
        else:
            raise NotImplementedError

    def pop_rmul(self, multiplier, neuron_term):
        return self.pop_mul(neuron_term, multiplier)

    def pop_div(self, neuron_term, divisor):
        if not shim.is_theano_object(neuron_term, divisor):
            assert len(self.pop_slices) == len(divisor)
            return shim.concatenate( [ neuron_term[..., pop_slice] / div_el
                                       for pop_slice, div_el in zip(self.pop_slices, divisor)],
                                     axis = -1)
        else:
            raise NotImplementedError

class Spiketrain(ConvolveMixin, PopulationHistory):
    """
    A class to store spiketrains, grouped into populations.

    These are stored in a sparse array where spikes are indicated by ones.
    Instead of the `shape` parameter, we use `pop_slices` to indicate the
    neurons associated to each population, from which `shape` is automatically
    deduced. Only 1d timeslices are currently supported (i.e. all populations
    are arranged along one axis).

    Internally spikes are stored in a CSR format, which is fast for appending
    rows and allows slicing and most mathematical operations. See the
    :py:mod:`scipy.sparse` documentation for a discussion of different sparse
    array types. We manage the three arrays of the CSR format ourselves, so
    no special sparse support from the auto-differentiation library is needed
    for adding new data. For convolutions, support of sparse vector products
    is needed.

    .. Note:
       Although this class was designed to simulate and store spike trains,
       since the data type and update functions are arbitrary, it can just as
       well be used for any kind of sparse data. Depending on the needs of the
       application, the convolution functions may need to be redefined and the
       dtype of ``int8`` may need to be changed.

    **Note for developers**: The tendency in more recent ML libraries seems to
    be to support the COO format exclusively; you can find a COO version of
    this class in `sinn.histories_unmaintained`.

    Parameters
    ----------

    _ Inherited from `History`
        + name     : str
        + time     : TimeAxis
        + :strike:`shape`  : [Removed by :py:class:`PopulationHistory`; see `pop_sizes`]
        + dtype    : numpy dtype
        + iterative: bool
        + symbolic : bool
        + init_data: [Replaced by :py:class:`Spiketrain`]
        + template : History

    pop_sizes: Tuple[int]
        Tuple of population sizes. Since the shape is calculated as
        ``(sum(pop_sizes),)``, this makes the `shape` argument redundant.
    init_data: ndarray, Optional
        n x N array, where n is arbitrary and N is the total number of units.
        The first n time points will be initialized with the data from init_data
        If not specified, the data is initialized to zero.
        Note that the initialized data is set for the first n time indices,
        so if padding is present, those will be before t0. Consequently, the
        typical use is to use n equal to the padding length (i.e. self.t0idx).
        If `init_data` does not provide a dtype, it must be given as argument.
    dtype: numpy dtype, Optional (default: 'int8')
        Type to use for storing the data. If both `init_data` and `dtype`
        are provided, data will be cast to `dtype`. If neither is provided,
        the 'int8' type is used, to be compatible with Theano bools.
    """
    ######################################
    # Theano-compatible CSR implementation
    #
    # Scipy names for CSR arrays: data, indices, indptr
    #
    # For accumulating updates, we need all data to be stored in shared variables
    # So we set _num_data to be a tuple of three shared arrays (data, indices
    # and indptr). _sym_data is a corresponding tuple of (possibly symbolic) arrays.
    #
    # Issue 1: While we can pass shared arrays to theano.sparse.CSR(), later
    #   compilation will fail. This is because theano.sparse expects the
    #   arguments to CSR to derive from `CSMProperty` graph node, of the type
    #   obtained with `theano.sparse.csm_properties(sparse_array)`.
    #
    # Solution 1: Create an empty sparse array on which we call csm_properties;
    #   this gives us empty arrays for data and indices, and a vector of 0 for
    #   indptr. We save these as attributes to the history as the tuple
    #   `_empty_data`, alongside `_num_data` and `_sym_data`.
    #   empty_indices, empty_indptr), in addition to the shared arrays.
    #   Whenever we need an actual sparse array, we concatenate the empty arrays
    #   with the values of either _num_data or _sym_data, and call CSR(*)
    #
    # Issue 2: When adding n entries to a the sparse array at row i, all values
    #   in `indices` after `i` need to be incremented by `n`. This is
    #   potentially more costly than simply appending to the list of indices,
    #   as we would do with a COO array. On the other hand, appending means
    #   reallocating memory since NumPy arrays must be contiguous.
    #
    # Solution 2: Since we haven't done a cost/benefit analysis, we use the
    #    simplest approach: incrementing `indices` when we adding spikes.
    #
    ######################################

    # _DataFormat = namedtuple('SpiketrainDataFormat',
    #                          ['data', 'neuron_idcs', 'indptr'])

    dtype       : DType = 'int8'
        # Using `np.dtype('int8')` causes errors (see History.dtype)

    class Config:
        # json_encoders overrides (is not merged) with parent option
        json_encoders = {**History.Config.json_encoders,
                         shim.sparse.csr_matrix_wrapper:
                            shim.sparse.csr_matrix_wrapper.json_encoder}

    __hash__ = History.__hash__  # Not inherited; see https://github.com/samuelcolvin/pydantic/issues/2422#issuecomment-828439446

    @History.update_function.setter
    def update_function(self, f: Optional[HistoryUpdateFunction]):
        """
        Spiketrain expects an index for its update function
        -> different return dtype than History.attach_update_function
        """
        cls = type(self)
        super(cls, cls).update_function.fset(self, f)
        if f is not None:
            self.update_function.return_dtype = self.idx_dtype

    # Called by History.__init__
    # Like a @validator, returns the value instead of setting attribute directly
    def initialized_data(self, init_data=None):
        """
        Create the data storage for the history, initializing with `init_data`
        if it is provided. Does the following:

        - Deserialize `init_data` if necessary
        - Validate that its shape and dtype match that of the History
        - Create the data storage:
          + (data) A tuple of three shared arrays for data, indices and indptr
          + (tidx) A shared scalar array

        Parameters
        ----------
        init_data: ndarray, optional
            If provided, `init_data` must be a 2d array with first axis
            corresponding to time and second to neurons. First axis may be
            shorter then the time array; in this case it is padded on the right
            with weros until its length is equal to `self.time.padded_length`.
            The second axis must match the number of neurons exactly.

            Must not be a symbolic value. If a shared value, no padding is
            performed, so the time dimension must match
            `self.time.padded_length` exactly.

        Returns
        -------
        tuple
            Value to store as `self._num_data`. This is a tuple of the three
            CSR arrays: data, (neuron) indices and indptr.
            The arrays may be numeric or symbolic.
        Shared[AxisIndex]
            Value to store as `self._num_tidx`
        """
        # shape = cls.get_shape_from_values(values)
        # pop_sizes, dtype, time = (values.get(x, None) for x in
        #     ('pop_sizes', 'dtype', 'time'))
        shape, pop_sizes, dtype, time = (
            self.shape, self.pop_sizes, self.dtype, self.time)
        nneurons = np.sum(pop_sizes)

        # Determine the data type
        if init_data is not None:
            data_dtype = getattr(init_data, 'dtype', None)
            if data_dtype is None and dtype is not None:
                pass
            elif dtype is None and data_dtype is None:
                raise ValueError("The provided data of type {} does have a 'dtype' "
                                 "attribute. In this case you must provide it to "
                                 "Spiketrain.initialize().".format(type(init_data)))
            elif not np.can_cast(data_dtype, dtype):
                raise TypeError(f"Data (type {init_data.dtype}) cannot be "
                                f"cast to the specified dtype ({dtype}).")
            else:
                dtype = np.dtype(data_dtype)
        if dtype is None:
            raise ValueError(f"Spiketrain {self.name}: `dtype` is inspecified.")

        # Initialize the data
        if (init_data is not None
            and not shim.issparse(init_data)
            and not isinstance(init_data, np.ndarray)):
            # Because of parse_obj hackery, we need to emulate Pydantic coercion
            # It may be a list of args to CSR, rather than a (sparse) array,
            # and these args may be serialized arrays.
            # Detect two types of serialization:
            # a) single dense array: ['Array', {'data':…}]
            # b) CSR sparse data: 3 arrays + shape: [['Array', …], ['Array'…]
            array_count = str(init_data).count("Array")
            if array_count == 1:
                init_data = Array.validate(init_data)
            elif array_count == 3 and isinstance(init_data, Iterable):
                init_data = tuple(Array.validate(v) for v in init_data)
            elif array_count:
                raise ValueError("Unrecognized format for Spiketrain "
                                 "initialization data:\n{init_data}")
            init_data = shim.sparse.csr_matrix_wrapper.validate(
                init_data, field=SimpleNamespace(name='init_data'))
        shape = (time.padded_length, nneurons)
        empty_csr = shim.sparse.csr_matrix(f'{self.name} - empty seed array',
                                           shape,
                                           dtype = dtype,
                                           symbolic = self.symbolic)
        object.__setattr__(self, '_empty_data',
                           shim.sparse.csm_properties(empty_csr)[:3])
            # Last returned value is shape, which we don't need

        init_csr = scipy.sparse.csr_matrix((time.padded_length, nneurons),
                                           dtype=dtype)
            # We use `scipy` here because we want true numpy arrays
        if init_data is None:
            # csrdata, csridcs, csrindptr = mtydata, mtyidcs, mtyindptr
            cur_datatidx = self.time.t0idx - 1
        elif shim.issparse(init_data):
            assert shim.eval(init_data.shape[1]) == nneurons, \
                "Spiketrain.initialized_data: The second dimension of " \
                "`init_data` must  match the number of neurons. " \
                f"init_data.shape: {init_data.shape}\n No. neurons: {nneurons}"
            cur_datatidx = init_data.shape[0] - 1
            init_csr[:cur_datatidx+1,:] = init_data
        else:
            assert shim.eval(init_data.shape[1]) == nneurons, \
                "Spiketrain.initialized_data: The second dimension of " \
                "`init_data` must  match the number of neurons. " \
                f"init_data.shape: {init_data.shape}\n No. neurons: {nneurons}"
            cur_datatidx = len(init_data) - 1
            init_csr[:cur_datatidx+1,:] = scipy.sparse.csr_from_dense(init_data.astype(dtype))
            # This may throw an efficiency warning, but we can ignore it since
            # init_csr is empty
            init_csr.eliminate_zeros()
        data = (shim.shared(init_csr.data, name=f'{self.name} - data'),
                shim.shared(init_csr.indices, name=f'{self.name} - indices'),
                shim.shared(init_csr.indptr, name=f'{self.name} - indptr')
                )
        cur_tidx = shim.shared(np.array(cur_datatidx-self.time.t0idx,
                                        dtype=self.time.index_dtype),
                               name = 't idx (' + self.name + ')',
                               symbolic = self.symbolic)
        return data, cur_tidx

    def clear(self, init_data=None, after=None):
        """Spiketrains shouldn't just be invalidated, since then multiple runs
        would make them more and more dense."""
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))
        if init_data is not None and after is not None:
            raise ValueError("[Spiketrain.clear()]: Cannot specify both "
                             "`init_data` and `after`.")
        elif after is not None:
            after = self.time.Index(after)
            init_data = self.get_data_trace(time_slice=np.s_[:after],
                                       include_padding=True)
        assert not shim.pending_update(self._num_data)
        data, tidx = self.initialized_data(init_data)
        if after is not None:
            assert shim.eval(tidx) == after
        else:
            assert shim.eval(tidx) == self.t0idx - 1
        object.__setattr__(self, '_sym_data', data)
        assert shim.graph.is_computable(self._sym_data)
        object.__setattr__(self, '_num_data', self._sym_data)
        super().clear(after=after)

    def get_data_trace(self, pop=None, neuron=None,
                  time_slice=slice(None, None), include_padding=False):
        """
        Return the spiketrain's computed data for the given neuron.
        Time points which have not yet been computed are excluded, such that
        the len(series.get_data_trace(*)) may be smaller than len(series). The
        return value is however guaranteed to be consistent with get_time_stops().
        If `component` is 'None', return the full multi-dimensional trace

        Parameters
        ----------
        pop: int
            Index of the population for which we want the trace. If unspecified,
            all neurons are returned, unless otherwise indicated by the 'neuron' parameter.
            Ignored if 'neuron' is specified.

        neuron: int, slice, array of ints
            Index of the neurons to return; takes precedence over 'pop'.

        time_slice:
        include_padding:
            See `DiscretizedAxis.data_index_slice`.

        Returns
        -------
        A csr formatted sparse array.
        """
        if self._sym_tidx is not self._num_tidx:
            raise RuntimeError("You are in the midst of constructing a Theano graph. "
                               "Reset history {} before trying to obtain its trace."
                               .format(self.name))

        if not self.cur_tidx.in_bounds:
            tslice = slice(self.time.t0idx, self.time.t0idx)
        else:
            if time_slice is None:
                time_slice = slice(None)
            tslice = self.time.data_index_slice(time_slice,
                                                include_padding=include_padding)
            if self.cur_tidx.data_index < tslice.stop - 1:
                tslice = slice(tslice.start, self.cur_tidx.data_index+1)

        # shape = (self.time.padded_length, self.shape[0])
        # data_arr = scipy.sparse.csr_matrix(tuple(shim.eval(self._num_data)),
        #                                    shape)
        data_arr = self._get_num_csr()
        # data_arr = self._num_data.tocsr()
        if neuron is None:
            if pop is None:
                return data_arr[tslice]
            else:
                return data_arr[tslice, self.pop_slices[pop]]
        elif isinstance(neuron, (int, slice)):
            return data_arr[tslice, neuron]
        elif isinstance(neuron, Iterable):
            idx = (tslice,) + tuple(component)
            return data_arr[idx]
        else:
            raise ValueError("Unrecognized spiketrain neuron '{}' of type '{}'"
                             .format(neuron, type(neuron)))

    def _get_num_csr(self):
        """
        Construct the numeric sparse array based on the current numeric
        variables.

        Returns
        -------
        Scipy sparse array
            The array's `*` (multiplication) operation has been remapped to
            element-wise multiplication to match Theano.
            This is equivalent to the raw data, so indexing corresponds to
            data indices: i in [0, ..., self.padded_length].
        """
        shape = (self.time.padded_length, self.shape[0])
        return shim.sparse.CSR(*tuple(shim.eval(self._num_data)),
                               shape=shape, symbolic=False)
    def _get_sym_csr(self):
        """
        Construct the symbolic sparse array based on the current symbolic
        variables.

        Returns
        -------
        Scipy sparse array
            If symbolic is False.
            The array's `*` (multiplication) operation has been remapped to
            element-wise multiplication to match Theano.
        Theano sparse array
            If symbolic is True.
        """
        shape = (self.time.padded_length, self.shape[0])
        return shim.sparse.CSR(*self._sym_data, shape=shape)

    def _getitem_internal(self, axis_index):
        """
        A function taking either an index or a slice and returning respectively
        the time point or an interval from the precalculated history.
        It does not check whether history has been calculated sufficiently far.

        .. Note:: This is an internal function – it implements the
           indexing interface. For most uses, one should index the history
           directly: ``hist[axis_index]``, which will check that the index is
           valid before calling this function.

        Parameters
        ----------
        axis_index: Axis index (int) | slice
            AxisIndex of the position to retrieve, or slice where start & stop
            are axis indices.

        Returns
        -------
        ndarray
            A binary array with last dimension equal to total number of neurons.
            Each element represents a neuron (populations are flattened).
            Values are 1 if the neuron fired in this bin, 0 if it didn't fire.
            If `key` is a scalar, array is 1D.
            If `key` is a slice or array, array is 2D. First dimension is time.
        """
        data_index = self.time.axis_to_data_index(axis_index)
        if not isinstance(data_index, slice):
            # Theano requires indices to sparse arrays to be slices
            # This also makes scipy.spare's indexing less surprising
            data_index = slice(data_index, data_index+1)

        shape = (self.time.padded_length, self.shape[0])
        csr = shim.sparse.CSR(*self._sym_data, shape=shape)
        if shim.isscalar(axis_index):
            return shim.sparse.dense_from_sparse(csr[data_index])[0]
            # return self._sym_data.tocsr()[
            #     self.time.axis_to_data_index(axis_index)].todense().A[0]
        else:
            assert isinstance(axis_index, slice)
            return shim.sparse.dense_from_sparse(csr[data_index])
            # return self._sym_data.tocsr()[
            #     self.time.axis_to_data_index(axis_index)].todense().A

    def update(self, tidx, neuron_idcs):
        """
        Add to each neuron specified in `value` the spiketime `tidx`.

        Parameters
        ----------
        tidx: AxisIndex | Slice[AxisIndex] | Array[AxisIndex]. Possibly symbolic.
            The time index of the spike(s).
            The lowest `tidx` should not correspond to more than one bin ahead
            of _sym_tidx.
            The indices themselves may be symbolic, but the _number_ of indices
            must not be.
        neuron_idcs: iterable
            List of neuron indices that fired in this bin. May be a
            2D numeric array, a list of 1D numeric arrays, or a list of 1D
            symbolic arrays, but not a 2D symbolic array: the outer dimension
            must be not be symbolic.
            For convenience, also accepts a 1D array – this is understood as
            an array of indices, and it is wrapped with a list to add the
            time dimension.

        **Side-effects**
            If either `tidx` or `neuron_idcs` is symbolic, adds symbolic updates
            in :py:mod:`shim`'s :py:attr:`symbolic_updates` dictionary  for
            `_num_tidx` and `_num_data`.
        """

        # TODO: Fix batch update to something less hacky
        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        time = self.time

        # neuron_idcs = shim.asarray(neuron_idcs)
        if shim.isscalar(neuron_idcs):
            raise ValueError(
                "Indices of neurons to update must be given as a "
                f"list of 1D arrays.\nIndices: {repr(neuron_idcs)}.")
        else:
            if shim.isarray(neuron_idcs):
                if neuron_idcs.ndim == 1:
                    # Single array of indices passed without the time dimension
                    # => add time dimension
                    neuron_idcs = [neuron_idcs]
                elif neuron_idcs.ndim != 2:
                    raise ValueError(
                        "Indices of neurons to update must be given as a "
                        f"list of 1D arrays.\nIndices: {repr(neuron_idcs)}.")
            neuron_idcs = [shim.atleast_1d(ni) for ni in neuron_idcs]
            if not all(getattr(ni, 'ndim', None) == 1 for ni in neuron_idcs):
                raise ValueError(
                    "Indices of neurons to update must be given as a "
                    f"list of 1D arrays.\nIndices: {repr(neuron_idcs)}.")
        # From this point on we can assume that neuron_idcs can be treated as
        # a list of 1D arrays.
        # In particular, `len(neuron_idcs)` is valid and should correspond
        # to the number of time indices.

        # _orig_tidx = tidx
        if shim.isscalar(tidx):
            assert isinstance(tidx, time.Index)
            earliestidx = latestidx = tidx
            assert len(neuron_idcs) == 1
            tidx = shim.add_axes(tidx, 1)
            # neuron_idcs = [neuron_idcs]
        elif isinstance(tidx, slice):
            assert (isinstance(tidx.start, time.Index)
                    and isinstance(tidx.stop, time.Index))
            earliestidx = tidx.start
            latestidx = tidx.stop-1
            assert (len(neuron_idcs)
                    == shim.eval(tidx.stop) - shim.eval(tidx.start))
            tidx = shim.arange(tidx.start, tidx.stop, dtype=time.Index.nptype)
                # Calling `eval` on just start or stop makes better use of
                # its compilation cache.
        else:
            assert shim.isarray(tidx)
            earliestidx = shim.min(tidx)
            latestidx = shim.max(tidx)
            try:
                assert len(neuron_idcs) == shim.eval(tidx.shape[0])
            except shim.graph.TooCostly:
                pass
        try:
            assert shim.eval(earliestidx) <= shim.eval(self._sym_tidx) + 1
        except shim.graph.TooCostly:
            pass

        # Clear any invalidated data
        if (shim.eval(earliestidx, max_cost=None)
            <= shim.eval(self._sym_tidx, max_cost=None)):
            if shim.is_symbolic(tidx):
                raise TypeError("Overwriting data (i.e. updating in the past) "
                                "only works with non-symbolic time indices. "
                                f"Provided time index: {tidx}.")
            if shim.pending_update():
                raise TypeError(
                    "Overwriting data (i.e. updating in the past) only works "
                    "when the symbolic updates dict is empty. Current values "
                    f"in the updates dictionary: {shim.get_updates().keys()}.")
            # _orig_dataidx = self.time.data_index(_orig_tidx)
                # _orig_tidx is not artifically converted to index array
            shape = (self.time.padded_length, self.shape[0])
            csr = scipy.sparse.csr_matrix(shim.eval(self._num_data), shape=shape)
            csr[earliestidx.data_index+1:, :] = 0
            csr.eliminate_zeros()
            self._num_data[0].set_value(csr.data)
            self._num_data[1].set_value(csr.indices)
            self._num_data[2].set_value(csr.indptr)

            # object.__setattr__(self, '_num_data', csc_data.tocoo())
            object.__setattr__(self, '_sym_data', self._num_data)

        dataidx = self.time.data_index(tidx)
        # assert len(dataidx) == len(neuron_idcs)
        for ti, idcs in zip(dataidx, neuron_idcs):
            # TODO: Assign in one block
            onevect = shim.ones(idcs.shape, dtype='int8')
                # vector of ones of the same length as the number of units which fired
            data, indices, indptr = self._sym_data
            object.__setattr__(
                self, '_sym_data',
                (shim.concatenate((data, onevect)),
                    # Add as many 1 entries as
                 shim.concatenate((indices, idcs)),
                    # Assign those spikes to neurons (col idx corresponds to neuron index) there are new spikes
                 shim.inc_subtensor(indptr[ti+1:], idcs.shape[0])
                    # Increment all the index pointers for time bins after ti
                    # by the number of spikes we added
                )
            )
        # Set the cur_idx. If tidx was less than the current index, then the latter
        # is *reduced*, since we no longer know whether later history is valid.
        if (shim.eval(latestidx, max_cost=None)
            < shim.eval(self._sym_tidx, max_cost=None)):
            # I can't imagine a legitimate reason to be here with a symbolic
            # time index
            assert not shim.is_graph_object(latest)
            warn("Moving the current time index of a Spiketrain "
                 "backwards. Invalidated data is NOT cleared.")
        #     self._num_tidx.set_value( latestidx )
        #     assert self._sym_tidx is self._num_tidx
        # else:
        #     self._sym_tidx = latestidx
        # self._sym_tidx = latestidx

        # Add symbolic updates to updates dict
        data_is_symb = shim.is_symbolic(self._sym_data)
        tidx_is_symb = shim.is_symbolic(latestidx)
        if tidx_is_symb:
            assert data_is_symb  # Should never have symbolic tidx w/out symbolic data
            # assert self._num_tidx is not self._sym_tidx
            object.__setattr__(self, '_sym_tidx', latestidx)
            shim.add_update(self._num_tidx, self._sym_tidx)
        else:
            self._num_tidx.set_value(latestidx)
            object.__setattr__(self, '_sym_tidx', self._num_tidx)

        if data_is_symb:
            # But we *can* have symbolic data w/out symbolic tidx
            assert self._sym_data is not self._num_data
            for num,sym in zip(self._num_data, self._sym_data):
                shim.add_update(num, sym)
        else:
            for nv, sv in zip(self._num_data, self._sym_data):
                nv.set_value(sv)
            object.__setattr__(self, '_sym_data', self._num_data)

    def pad(self, pad_left, pad_right=0):
        """
        Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are reduced to make them exact multiples of dt.

        Parameters
        ----------
        pad_left: AxisIndexDelta (int) | value type
            Amount of time to add to before t0. If non-zero, all indices
            to this data will be invalidated.
        pad_right: AxisIndexDelta (int) | value type (default 0)
            Amount of time to add after tn.
        """
        if shim.is_graph_object(pad_left, pad_right):
            raise TypeError("Can only pad with non-symbolic values.")
        if shim.pending_update():
            raise RuntimeError("Cannot add padding while computing symbolic "
                               "updates.")
        assert self._sym_data is self._num_data

        before_len, after_len = self.time.pad(pad_left, pad_right)

        indptr = self._num_data[2]
        if before_len > 0 and self.cur_tidx >= self.t0idx:
            warn("Added non-zero left padding - invalidating the data "
                 f"associated with history {self.name}.")
            self.clear()
            # clearing will already update the indptr, because self.time has
            # already been updated, so no need for 'concatenate'
            data, indices, indptr = self._num_data
        else:
            data, indices, indptr = self._num_data
            indptr = np.concatenate((np.zeros(before_len, dtype=indptr.dtype),
                                     indptr.get_value(borrow=True)))
                # move index pointers forward by the number that were added
                # index pointers themselves don't change, since data didn't change
        assert shim.eval(indptr).shape[0] == self.time.padded_length + 1
        self._num_data[2].set_value(indptr, borrow=True)
        assert self._sym_data is self._num_data
        # object.__setattr__(self, '_num_data', (data, indices, indptr))
        # object.__setattr__(self, '_sym_data', self._num_data)
        self._sym_tidx.set_value(self._sym_tidx.get_value(borrow=True) - before_len, borrow=True)
        assert self._num_tidx is self._sym_tidx

    def eval(self, max_cost :Optional[int]=None, if_too_costly :str='raise'):
        # Adapted from History.eval. See History.eval for docstring
        if self._num_data is self._sym_data:
            assert self._num_tidx is self._sym_tidx
            # Nothing to do
            return
        # Note: permissible to have symbolic data & numeric tidx, but
        #       not permissible to have numeric data & symbolic tidx
        kwargs = {'max_cost': max_cost, 'if_too_costly': if_too_costly}
        updates = shim.get_updates()
        # All symbolic updates should be in shim's updates dict
        if self._num_tidx in updates:
            assert self._sym_tidx is updates[self._num_tidx]
        for nv, sv in zip(self._num_data, self._sym_data):
            assert nv in updates
            assert updates[nv] is sv
        tidx, data, indices, indptr = shim.eval(
            (self._sym_tidx, *self._sym_data), **kwargs)
        self._num_tidx.set_value(tidx)
        self._num_data[0].set_value(data)
        self._num_data[1].set_value(indices)
        self._num_data[2].set_value(indptr)
        object.__setattr__(self, '_sym_data', self._num_data)
        object.__setattr__(self, '_sym_tidx', self._num_tidx)
        if self._num_tidx in updates:
            del updates[self._num_tidx]
        for nv in self._num_data:
            del updates[nv]

    def _convolve_single_t(self, discretized_kernel, tidx, kernel_slice):
        """
        Return the time convolution with the spike train, i.e.
            ∫ spiketrain(t - s) * kernel(s) ds
        with s ranging from -∞ to ∞  (normally there should be no spikes after t).
        The result is a 1d array of length Nneurons.
        Since spikes are delta functions, effectively what we are doing is
        sum( kernel(t-s) for s in spiketrain if s == 1 )

        .. Hint:: Typically the number of neurons is very large, but the
        kernel only depends on the population to which a neuron belongs.
        In this case consider using using a FactoredKernel, which when used in
        tandem with Spiketrain, more efficiently gets around the limitations
        of the scipy.sparse array.

        .. Note:: This method is an internal hook to allow histories to define
        the specifics of a convolution operation; it is called within the
        public-facing method `~History.convolve()`. There should be no reason
        for user code to call it directly instead of `~History.convolve()`.

        Parameters
        ----------
        discretized_kernel: History
            History, as returned by `self.discretize_kernel()`.
        tidx: Axis Index
            Time index at which to evaluate the convolution.
            This must be an axis index, not a data index.
        kernel_slice: slice
            The kernel is truncated to the bounds specified by
            this slice (thus implicitly set to zero outside these bounds).
            This achieved simply by indexing the kernel:
            ``discretized_kernel[kernel_slice]``.
        Returns
        -------
        TensorWrapper
        """
        # The setup of slicing is copied from Series._convolve_single_t

        assert isinstance(tidx, self.time.Index)
        kernel_slice = discretized_kernel.time.data_index(kernel_slice)
        assert shim.eval(kernel_slice.stop > kernel_slice.start)

        # tidx = self.get_t_idx(tidx)
        #
        # # Convert None & negative slices into positive start & stop
        # kernel_slice = self.slice_indices(kernel_slice)
        # # Algorithm assumes an increasing kernel_slice
        # shim.check(kernel_slice.stop > kernel_slice.start)

        hist_start_idx = (tidx.data_index
                          - kernel_slice.stop - discretized_kernel.idx_shift)
        hist_slice = slice(
            hist_start_idx,
            hist_start_idx + kernel_slice.stop - kernel_slice.start)
        assert shim.eval(hist_slice.start) >= 0

        hist_subarray = self._sym_csr()[hist_slice]

        assert shim.eval(discretized_kernel.ndim) <= 2
        sliced_kernel = discretized_kernel[kernel_slice]

        # To understand why the convolution is taken this way, consider
        # 1) That the Theano (and also the shimmed Scipy) sparse arrays treat
        #    * as an elementwise product, in contrast to standard Scipy sparse
        #    matrices which treat it as matrix multiplication.
        #    (The shimmed arrays redefine * to their `multiply` method.)
        # 2) `multiply` only returns a sparse array if the argument is also 2D
        #    (But sometimes still returns a dense array ?)
        # 3) That sparse arrays are always 2D, so A[0,0] is a 2D, 1x1 matrix
        #    Moreover, one can't add a 3rd dimension to a sparse array to
        #    broadcast along that dimension
        # 4) The .A attribute of a matrix returns the underlying array
        # 5) That we do not need to multiply by the step size (dt): This is a discretized
        #    train of Dirac delta peaks, so to get the effective height of the spike
        #    when spread over a bin we should first divide by dt. To take the convolution
        #    we should then also multiply by the dt, cancelling the two.
        # Addendum) Points 1-4 are specific to scipy.sparse (and by corollary
        #    theano.sparse). If we used pydata's sparse (which is COO), along
        #    with the sparse formats in TensorFlow or PyTorch (also COO), we
        #    could implement this more transparently.

        # We are currently limited to 2D kernels
        assert discretized_kernel.ndim in (1,2)

        if discretized_kernel.shape[-1] != self.shape[-1]:
            result = discretized_kernel.get_empty_convolve_result()
            # result = shim.tensor(discretized_kernel.shape, dtype=self.dtype)

            for inslc, outslc, kernslc in sliced_kernel.block_iterator(
                kernel_dims=None, include_time_slice=True):
                # Examples:
                # kernslc = (:, i1:i2)          # include_time_slice prepends ':'
                # outslc = (..., i1:i2, i2:i3)  # all in/outslc prepend '...'
                s = hist_subarray[inscl]
                κ = sliced_kernel[kernslc]
                result[outslc] = s * κi[::-1]
                # result[outslc] = s.multiply(κi[::-1])
        else:
            result = hist_subarray * sliced_kernel[::-1]
            # result = hist_subarray.multiply(sliced_kernel[::-1])

        return TensorWrapper(result,
            TensorDims(contraction=discretized_kernel.contravariant_axes))


class Series(ConvolveMixin, History):
    """
    Store history as a series, i.e. as an array of dimension ``T x (shape)``,
    where ``T`` is the number of bins and shape is this history's `shape`
    attribute.
    """

    # _strict_index_rounding = True
    dtype : DType = shim.config.floatX  # Set default dtype

    __hash__ = History.__hash__  # Not inherited; see https://github.com/samuelcolvin/pydantic/issues/2422#issuecomment-828439446

    def initialized_data(self, init_data=None):
        """
        Create the data storage for the history, initializing with `init_data`
        if it is provided. Does the following:

        - Deserialize `init_data` if necessary
        - Validate that its shape and dtype match that of the History
        - Create the data storage:
          + (data) A shared array of size (T,)+shape
          + (tidx) A shared scalar array

        Parameters
        ----------
        init_data: ndarray, optional
            If provided, shape must match `self.shape`. First axis must
            correspond to time (so :code:`init_data.ndim == self.ndim + 1`),
            but may be shorter then the time array; in this case it is padded
            on the right with zeros until its length is equal to
            `self.time.padded_length`.

            Special case: An empty array is allowed, independent of its shape,
            and treated the same as `None`.

            Must not be a symbolic value. If a shared value, no padding is
            performed, so the time dimension must match
            `self.time.padded_length`.

        Returns
        -------
        Shared[ndarray]
            Value to store as `self._num_data`
        Shared[AxisIndex]
            Value to store as `self._num_tidx`
        """
        # NOTE: Since this method is called within Pydantic’s parsing, exceptions
        #       should be extra-informative (Pydantic removes the stack trace).
        #       In particular, asserts should always have description messages
        # shape, dtype = (values.get(x, None) for x in ('shape', 'dtype'))
        shape, dtype = self.shape, self.dtype
        if mtb.typing.json_like(init_data, 'Array'):
            init_data = Array.validate(init_data)
        if getattr(init_data, 'size', None) == 0:
            init_data = None
        if init_data is None:
            assert shape is not None and dtype is not None, \
                "Series.initialized_data: If `init_data` is None, `shape` " \
                "and `dtype` must also be None. " \
                f"Received shape {shape} and dtype {dtype}."
            data = shim.shared(
                np.zeros((self.time.padded_length,) + self.shape, dtype=dtype),
                name = self.name + " data",
                borrow = True
                )
            tidx_val = self.time.t0idx-1
        else:
            assert not shim.is_symbolic(init_data), \
                "Series.initialized_data: `init_data` must not be symbolic."
            if not shim.isarray(init_data):
                if len(init_data) == 0:
                    # 0 length arrays are flattened when exporting
                    init_data = np.zeros((0,*self.shape))
                else:
                    # Because of parse_obj hackery, must emulate Pydantic coercion
                    init_data = mtb.typing.Array[self.dtype].validate(
                        init_data, field=SimpleNamespace(name='init_data'))
            assert shim.eval(init_data.shape[1:]) == self.shape, \
                "Series.initialized_data: initialization data does not match " \
                "the history's shape.\n" \
                f"Data shape:{init_data.shape}\nHistory's shape: {self.shape}"
            if shim.isshared(init_data):
                # Use shared variables as-is
                data_length = len(init_data.get_value())
                Δtidx = self.time.padded_length - data_length
                assert Δtidx == 0
                data = init_data
            else:
                data_length = len(init_data)
                Δtidx = self.time.padded_length - data_length
                pad_width = [(0, Δtidx)] + [(0,0)]*self.ndim
                npdata = np.pad(init_data, pad_width)
                data = shim.shared(
                    npdata,
                    name = self.name + " data",
                    borrow = True
                    )
            tidx_val = self.time.data_to_axis_index(data_length - 1)
        cur_tidx = shim.shared(np.array(tidx_val, dtype=self.time.index_dtype),
                               name = 't idx (' + self.name + ')',
                               symbolic = self.symbolic)
        return data, cur_tidx

    def _getitem_internal(self, axis_index):
        """
        A function taking either an index or a slice and returning respectively
        the time point or an interval from the precalculated history.
        It does not check whether history has been calculated sufficiently far.

        .. Note:: This is an internal function – it implements the
           indexing interface. For most uses, one should index the history
           directly: ``hist[axis_index]``, which will check that the index is
           valid before calling this function.

        Parameters
        ----------
        axis_index: Axis index (int) | slice
            AxisIndex of the position to retrieve, or slice where start & stop
            are axis indices.

        Returns
        -------
        ndarray
        """
        return self._sym_data[self.time.axis_to_data_index(axis_index)]

    def update(self, tidx, value):
        """
        Store a new time slice.

        As a convenience for initializing data, if both `tidx` and `value`
        are non-symbolic, `value` can omit the time dimension, or be specified
        as a single scalar, in which case it is broadcast to
        ``(len(tidx), self.shape)``.

        Parameters
        ----------
        tidx: AxisIndex | Slice[AxisIndex] | Array[AxisIndex]. Possibly symbolic.
            The time index at which to store the value.
            If specified as a slice, the length of the range should match
            value.shape[0].
        value: timeslice
            The timeslice to store. Shape must match ``self.shape``.


        **Side-effects**
            If either `tidx` or `neuron_idcs` is symbolic, adds symbolic updates
            in :py:mod:`shim`'s :py:attr:`symbolic_updates` dictionary  for
            `_num_tidx` and `_num_data`.
        """

        # If both tidx and value are numeric, figure out the expected
        # shape of `value` and broadcast if necessary
        if not shim.is_graph_object(tidx, value):
            # First check if value is a scalar – this is common when initializing
            if shim.isscalar(value):
                value = np.broadcast_to(value, self.shape)
            # Now broadcast to the time dimension
            if isinstance(tidx, slice):
                if tidx.step is not None and key.step != 1:
                    raise ValueError("Slices must have steps of `None` or 1.")
                valueshape = (tidx.stop-tidx.start,) + self.shape
                value = shim.broadcast_to(value, valueshape)
            elif shim.isarray(tidx):
                assert tidx.ndim == 1
                valueshape = tidx.shape + self.shape
                if shim.isscalar(value) or value.shape != valueshape:
                    value = shim.broadcast_to(value, valueshape)


        if self.locked:
            raise RuntimeError("Tried to modify locked history {}."
                               .format(self.name))

        time = self.time

        # Convert constants to TensorConstant if history is symbolic
        if self.symbolic and not isinstance(value, shim.cf.GraphTypes):
            value = shim.asvariable(value)

        # Adaptations depending on whether tidx is a single bin or a slice
        if shim.isscalar(tidx):
            assert isinstance(tidx, time.Index)
            earliestidx = latestidx = tidx
        elif isinstance(tidx, slice):
            assert (isinstance(tidx.start, time.Index)
                    and isinstance(tidx.stop, time.Index))
            assert shim.eval(tidx.start) < shim.eval(tidx.stop)
            earliestidx = tidx.start
            latestidx = tidx.stop - 1
            if shim.graph.is_computable(value.shape):
                try:
                    assert (shim.eval(value.shape[0], max_cost=50)
                            == shim.eval(tidx.stop) - shim.eval(tidx.start))
                    # Calling `eval` on just start or stop makes better use of
                    # its compilation cache.
                except shim.graph.TooCostly:
                    pass
        else:
            assert shim.isarray(tidx)
            earliestidx = shim.min(tidx)
            latestidx = shim.max(tidx)
            if shim.graph.is_computable(value.shape):
                assert shim.eval(value.shape[0]) == shim.eval(tidx.shape[0])
        # Ensure that we don't break causal assumption by updating too far
        # into the future.
        try:
            assert (shim.eval(earliestidx, max_cost=30)
                    <= shim.eval(self._sym_tidx, max_cost=30) + 1)
        except shim.graph.TooCostly:
            pass

        datatidx = self.time.axis_to_data_index(tidx)

        if self.symbolic:

            # There are two possibilities:
            # 1. Neither the new value nor time indices are symbolic, AND
            #    the internal running _sym_data and _sym_tidx have not been
            #    symbolically updated. In this case we update the underlying
            #    shared variables and it behaves much like a normal Numpy
            #    update. This typically happens when intializing the history.
            # 2. At least one of the conditions of 1. is not met. In this case
            #    the running _sym_data and _sym_tidx are disconnected from
            #    the concrete variables _num_data and _num_tidx, and we
            #    perform a symbolic update.

            # Should only have Theano updates with Theano original data
            assert isinstance(self._num_data, shim.config.SymbolicSharedType)
            assert isinstance(self._num_tidx, shim.config.SymbolicSharedType)

            if (not shim.is_theano_object(latestidx, value)
                and (latestidx == tidx or not shim.is_theano_object(tidx.start))
                and self._sym_tidx == self._num_tidx
                and self._sym_data == self._num_data):
                # 1 : There are no symbolic dependencies – update data directly

                # Not clear how to resolve an update following one where _sym_tidx
                # was made non-computable, since then we can't know how to compare
                # the two time points compare.
                assert self._sym_tidx is self._num_tidx
                assert self._sym_data is self._num_data

                tmpdata = self._num_data.get_value(borrow=True)
                tmpdata[datatidx] = value
                self._num_data.set_value(tmpdata, borrow=True)
                # Update both the running/symbolic and base time indices
                self._num_tidx.set_value(shim.cast(latestidx, self.tidx_dtype))
                object.__setattr__(self, '_sym_tidx', self._num_tidx)
            else:
                # 2 : There are symbolic dependencies => update just the running
                # vars (_sym_tidx & _sym_data) and add to the update dictionary
                # Update the data
                tmpdata = self._sym_data
                object.__setattr__(self, '_sym_data',
                                   shim.set_subtensor(tmpdata[datatidx], value))
                # if updates is not None:
                #     shim.add_updates(updates)
                shim.add_update(self._num_data, self._sym_data)
                # Update the time index
                assert shim.is_theano_object(self._num_tidx)
                if not isinstance(latestidx, shim.cf.GraphTypes):
                    name = (self._num_tidx.name
                            + '[update to {}]'.format(latestidx))
                    latestidx = shim.asvariable(latestidx, name=name)
                object.__setattr__(self, '_sym_tidx', latestidx)
                shim.add_update(self._num_tidx, self._sym_tidx)

        else:
            if shim.is_theano_object(value):
                if not shim.graph.is_computable([value]):
                    raise ValueError("You are trying to update a pure numpy series ({}) "
                                     "with a Theano variable. You need to make the "
                                     "series a Theano variable as well or ensure "
                                     "that `value` does not depend on any "
                                     "symbolic inputs."
                                     .format(self.name))
                value = shim.graph.eval(value, max_cost=None)
            if shim.is_theano_object(datatidx):
                if not shim.graph.is_computable([datatidx]):
                    raise ValueError("You are trying to update a pure numpy series ({}) "
                                     "with a time idx that is a Theano variable. You need "
                                     "to make the series a Theano variable as well "
                                     "or ensure that `tidx` does not depend "
                                     "on any symbolic inputs."
                                     .format(self.name))
                datatidx = shim.graph.eval(datatidx, max_cost=None)

            dataobject = self._num_data.get_value(borrow=True)

            # if updates is not None:
            #     raise RuntimeError("For normal Python and NumPy functions, update "
            #                        "variables in place rather than using an update dictionary.")
            if dataobject[datatidx].shape != value.shape:
                raise ValueError("Series '{}': The shape of the update value - {} - does not match "
                                 "the shape of a timeslice(s) - {} -."
                                 .format(self.name, value.shape, dataobject[datatidx].shape))

            dataobject[datatidx] = value
            self._num_data.set_value(dataobject, borrow=True)
            self._num_tidx.set_value(latestidx)

            assert self._sym_data is self._num_data
            assert self._sym_tidx is self._num_tidx
            # oject.__setattr__(self, '_sym_tidx', self._num_tidx)
                # If we updated in the past, this will reduce _sym_tidx
                # – which is what we want

    def pad(self, pad_left, pad_right=0, **kwargs):
        """
        Extend the time array before and after the history. If called
        with one argument, array is only padded before. If necessary,
        padding amounts are increased to make them exact multiples of dt.
        See `DiscretizedAxis.pad` for more details.

        Parameters
        ----------
        pad_left: AxisIndexDelta (int) | value type
            Amount of time to add to before t0. If non-zero, all indices
            to this data will be invalidated.
        pad_right: AxisIndexDelta (int) | value type (default 0)
            Amount of time to add after tn.
        **kwargs:
            Extra keyword arguments are forwarded to `numpy.pad`.
            They may be used to specify how to fill the added time slices.
            Default is to fill with zeros.
        """

        if not kwargs:
            # No keyword arguments specified – use defaults
            kwargs['mode'] = 'constant'
            kwargs['constant_values'] = 0

        if shim.is_graph_object(pad_left, pad_right):
            raise TypeError("Can only pad with non-symbolic values.")
        if self._sym_data is not self._num_data:
            raise RuntimeError("Cannot add padding while computing symbolic "
                               "updates.")

        previous_tarr_shape = (self.time.padded_length,)
        before_len, after_len = self.time.pad(pad_left, pad_right)

        if before_len > 0 and self.cur_tidx >= self.t0idx:
            warn("Added non-zero left padding - invalidating the data "
                 f"associated with history {self.name}.")
            self.clear()

        pad_width = [(before_len, after_len)] + [(0, 0)]*self.ndim
                      # + [(0, 0) for i in range(len(self.shape))] )

        self._sym_data.set_value(shim.pad(self._sym_data.get_value(borrow=True),
                                           previous_tarr_shape + self.shape,
                                           pad_width, **kwargs),
                                  borrow=True)
        self._sym_tidx.set_value(self._sym_tidx.get_value(borrow=True) - before_len, borrow=True)
        assert self._num_tidx is self._sym_tidx

    def zero(self, mode='all'):
        """Zero out the series. Unless mode='all', the initial data point will NOT be zeroed"""
        if mode == 'all':
            mode_slice = slice(None, None)
        else:
            mode_slice = slice(1, None)

        if shim.pending_update():
            raise RuntimeError("Cannot add zero data while computing symbolic "
                               "updates.")

        new_data = self._conc_shared.get_value(borrow=True)
        new_data[mode_slice] = np.zeros(new_data.shape[mode_slice])
        self._sym_data.set_value(new_data, borrow=True)
        assert self._sym_data is self._num_data

        self.clear()
            # Invalidate time indices and set _symb_* = _conc_*

    def get_data_trace(self, component=None, include_padding=False, time_slice=None):
        """
        Return the series' computed data for the given component.
        Time points which have not yet been computed are excluded, such that
        the len(series.get_data_trace(*)) may be smaller than len(series). The
        return value is however guaranteed to be consistent with get_time_stops().
        If `component` is 'None', return the full multi-dimensional trace

        Parameters
        ----------
        component: int, slice, iterable of ints
            Restrict the returned trace to these components.
        include_padding : bool | 'all' | 'begin' | 'start' | 'end'
            - True or 'all'     : include padding at both ends
            - 'begin' or 'start': include the padding before t0
            - 'end'             : include the padding after tn
            - False (default)   : do not include padding

        time_slice:
            See get_time_stops.
        """
        if self._sym_tidx is not self._num_tidx:
            raise RuntimeError("You are in the midst of constructing a Theano graph. "
                               "Reset history {} before trying to obtain its time array."
                               .format(self.name))

        if not self.cur_tidx.in_bounds:
            tslice = slice(self.time.t0idx, self.time.t0idx)
        else:
            if time_slice is None:
                time_slice = slice(None)
            tslice = self.time.data_index_slice(time_slice,
                                                include_padding=include_padding)
            if self.cur_tidx.data_index < tslice.stop - 1:
                tslice = slice(tslice.start, self.cur_tidx.data_index+1)

        data = self._num_data.get_value()
        if component is None:
            return data[tslice]
        elif isinstance(component, (int, slice)):
            return data[tslice, component]
        elif isinstance(component, Iterable):
            idx = (tslice,) + tuple(component)
            return data[idx]
        else:
            raise ValueError("Unrecognized series component '{}' of type '{}'"
                             .format(component, type(component)))

    def interpolate(self, interp_times):
        """
        Interpolate a history at the given interp_times. Returns a new
        History object of same type as `self`.

        This is implemented by looping through components, and calculating
        1d interpolation onto the new times for each.

        Parameters
        ----------
        interp_times: 1d-array
            Array of times. These will be the stops of the returned History.

        Returns
        -------
        Series (or subclass)
            Returns a History of same type as `self`.
        """

        if self.symbolic:
            raise NotImplementedError("Interpolation of symbolic histories "
                                      "is not yet implemented.")

        assert interp_times.ndim == 1

        # Create an empty array for the interpolated data
        interp_data = np.empty(interp_times.shape + self.shape)
        time_stops = self.time.stops_array
        trace = self.trace
        # Fill the array by computing 1-D interpolation for each component
        for idx in np.ndindex(self.shape):
            interp_data[(slice(None),)+idx] = \
                np.interp(interp_times, time_stops, trace[(slice(None),)+idx])

        # Grab dict description of `self` to serve as template for new history
        d = self.dict()
        # Replace time stops with `interp_times`
        # FIXME: Is padding treated correctly ?
        d['time']['stops'] = interp_times
        d['time'] = TimeArrayAxis(**d['time'])
        # Replace old with new data
        d['data'] = interp_data
        # Remove custom dict entries which aren't expected in initializer
        del d['update_function']  # Not guaranteed the update functions would
        del d['range_update_function']  # be valid on the interpolated history
        # Give a new, understandable name
        d['name'] = self.name + '_interp'
        # Finally, create the new interpolated history
        interp_hist = type(self)(**d)

        return interp_hist

    def _convolve_single_t(self, discretized_kernel, tidx, kernel_slice):
        """
        Return the time convolution between this history and a causal kernel, i.e.
            ∫ series(t - s) * kernel(s) ds
        with s ranging from 0 to ∞.

        .. Note:: This method is an internal hook to allow histories to define
        the specifics of a convolution operation; it is called within the
        public-facing method `~History.convolve()`. There should be no reason
        for user code to call it directly instead of `~History.convolve()`.

        Parameters
        ----------
        discretized_kernel: History
            History, as returned by `self.discretize_kernel()`.
        tidx: AxisIndex (self.time.Index)
            Time index at which to evaluate the convolution.
            This must be an axis index, not a data index.
        kernel_slice: slice
            The kernel is truncated to the bounds specified by
            this slice (thus implicitly set to zero outside these bounds).
            This achieved simply by indexing the kernel:
            ``discretized_kernel[kernel_slice]``.
        Returns
        -------
        TensorWrapper
        """
        # When indexing data, make sure to use self[…] rather than
        # self._sym_data[…], to trigger calculations if necessary

        assert isinstance(tidx, self.time.Index)
        kernel_slice = discretized_kernel.time.data_index(kernel_slice)
        assert shim.eval(kernel_slice.stop > kernel_slice.start)

        # # Convert None & negative slices into positive start & stop
        # kernel_slice = slice(*kernel.slices.indices(discretized_kernel.padded_length))
        # # kernel_slice = discretized_kernel.slice_indices(kernel_slice)
        # # Algorithm assumes an increasing kernel_slice
        # shim.check(kernel_slice.stop > kernel_slice.start)

        # tidx = self.get_tidx(tidx)

        hist_start_idx = (tidx.data_index
                          - kernel_slice.stop - discretized_kernel.idx_shift)
        hist_slice = slice(
            hist_start_idx,
            hist_start_idx + kernel_slice.stop - kernel_slice.start)
        assert shim.eval(hist_slice.start >= 0)

        dim_diff = discretized_kernel.ndim - self.ndim
        dtype = np.result_type(discretized_kernel.dtype, self.dtype)
        kflipped = discretized_kernel[kernel_slice][::-1]
        hist = shim.add_axes(self[hist_slice], dim_diff, -self.ndim)

        if not (all(kflipped.shape) and all(hist.shape)):
            # If either the kernel or history slice is empty, its shape
            # will contain 0 and we arrive here.
            # In this case broadcasting won't work, but we know the result
            # must be zero
            result = np.zeros(discretized_kernel.shape, dtype=dtype)
        else:
            result = shim.cast(
                self.dt64 * shim.sum(kflipped * hist, axis=(0,)),
                dtype=dtype)
            # history needs to be augmented by a dimension to match the kernel
            # Since kernels are [to idx][from idx], the augmentation has to be on
            # the second-last axis for broadcasting to be correct.
            # TODO: Untested with multi-dim timeslices (i.e. self.ndim > 1)
        return TensorWrapper(result,
             TensorDims(contraction=discretized_kernel.contravariant_axes))

    def _convolve_batch(self, discretized_kernel, kernel_slice):
        """Return the convolution at every lag within t0 and tn."""
        # When indexing data, make sure to use self[…] rather than self._sym_data[…],
        # to trigger calculations if necessary

        kernel_slice = discretized_kernel.time.data_index(kernel_slice)
        if (kernel_slice.start == kernel_slice.stop
            or len(discretized_kernel) == 0):
            return shim.zeros((1,)+self.shape, dtype=self.dtype)
        else:
            # Algorithm assumes an increasing kernel_slice
            assert shim.eval(kernel_slice.stop) > shim.eval(kernel_slice.start)
            disksliced = discretized_kernel[kernel_slice]

            # We compute the full 'valid' convolution, for all lags and then
            # return just the subarray corresponding to [t0:tn]
            # We have to adjust the index because the 'valid' mode removes
            # time bins at the ends.
            # E.g.: assume kernel.idx_shift = 0. Then (convolution result)[0] corresponds
            # to the convolution evaluated at tarr[kernel.stop + kernel_idx_shift]. So to get the result
            # at tarr[tidx], we need (convolution result)[tidx - kernel.stop - kernel_idx_shift].

            domain_start = (self.t0idx.data_index
                            - kernel_slice.stop - discretized_kernel.idx_shift)
            domain_slice = slice(domain_start, domain_start + len(self))
            # Check that there is enough padding before t0
            assert shim.eval(domain_slice.start >= 0)

            dis_kernel_shape = (kernel_slice.stop - kernel_slice.start,) \
                               + discretized_kernel.shape
            dtype = np.result_type(discretized_kernel.dtype, self.dtype)
            retval = shim.cast(self.dt64 * shim.conv1d(
                self[:], disksliced, len(self._tarr), dis_kernel_shape)[domain_slice],
                               dtype=dtype).sum(axis=discretized_kernel.contravariant_axes)
            if shim.graph.is_computable(retval.shape):
                # Check that there is enough padding after tn
                assert shim.eval(retval.shape[0]) == len(self)
            return retval


    #####################################################
    # Operator definitions
    #####################################################

    def _apply_op(self, op, b=None):
        raise NotImplementedError("History operations still need to be ported.")
        try:
            opname = str(op)
        except Exception:
            opname = "f"
        if b is None:
            new_series = self.copy()
            new_series.name = f"{opname}({self.name})"
            new_series.update_function = HistoryUpdateFunction(
                func = lambda t: op(self[t]),
                inputs = [self]
                )
            new_series.range_update_function = HistoryUpdateFunction(
                func = lambda tarr: op(self[self.time_array_to_slice(tarr)]),
                inputs = [self]
                )
            # new_series.add_input(self)
        elif isinstance(b, HistoryBase):
            # HACK Should write function that doesn't create empty arrays
            # Get a dictionary description of `self`
            desc = self.dict()
            # Update name
            desc['name'] = f"{opname}({self.name}, {b.name})"
            # Truncate its time array according to arguments
            tnidx = min(self.tnidx, b.tnidx.convert(self.time))
            desc['time']['stops'] = self.time.padded_stops_array[:tnidx+1]
            desc['time'] = type(self.time)(**desc['time'])
                # type(self.time) -> either TimeAxis or TimeArrayAxis
            # Update its shape according to arguments
            shape = np.broadcast(np.empty(self.shape), np.empty(b.shape)).shape
            desc['shape'] = shape
            # FIXME: Is padding treated correctly ?
            new_series = Series(**desc)
            new_series.update_function = HistoryUpdateFunction(
                func = lambda t: op(self[t], b[t]),
                inputs = [self, b]
                )
            new_series.range_update_function = HistoryUpdateFunction(
                func = lambda tarr: op(self[self.time_array_to_slice(tarr)],
                                       b[b.time_array_to_slice(tarr)]),
                inputs = [self, b]
                )
            #new_series.add_input(self)
            computable_tidx = min(
                min(self.cur_tidx, self.tnidx).convert(new_series),
                min(b.cur_tidx, b.tnidx).convert(new_series))
        else:
            # Get a dictionary description of `self`
            desc = self.dict()
            # Update name
            bname = getattr(b, 'name', str(b))
            desc['name'] = f"{opname}({self.name}, {bname})"
            # Update its shape according to arguments
            if hasattr(b, 'shape'):
                shape = np.broadcast(np.empty(self.shape),
                                     np.empty(b.shape)).shape
            else:
                shape = self.shape
            desc['shape'] = shape
            # # Deprecated? I'm not sure why I would have needed symbolic vars as inputs
            # if shim.is_theano_variable(b):
            #     inputs = [self, b]
            # else:
            #     inputs = [self]
            new_series = Series(**desc)
            new_series.update_function = HistoryUpdateFunction(
                func = lambda t: op(self[t], b),
                inputs = [self]
                )
            new_series.range_update_function = HistoryUpdateFunction(
                func = lambda tarr: op(self[self.time_array_to_slice(tarr)], b),
                inputs = [self]
                )
            computable_tidx = min(self.cur_tidx, self.tnidx).convert(new_series)

        # Since we assume the op is cheap, calculate as much as possible
        # without triggering updates in the inputs
        new_series._compute_up_to(computable_tidx)

        return new_series

    def __abs__(self):
        return self._apply_op(operator.abs)
    def __neg__(self):
        return self._apply_op(operator.neg)
    def __pos__(self):
        return self._apply_op(operator.pos)
    def __add__(self, other):
        return self._apply_op(operator.add, other)
    def __radd__(self, other):
        return self._apply_op(lambda a,b: b+a, other)
    def __sub__(self, other):
        return self._apply_op(operator.sub, other)
    def __rsub__(self, other):
        return self._apply_op(lambda a,b: b-a, other)
    def __mul__(self, other):
        return self._apply_op(operator.mul, other)
    def __rmul__(self, other):
        return self._apply_op(lambda a,b: b*a, other)
    def __matmul__(self, other):
        return self._apply_op(operator.matmul, other)
    def __rmatmul__(self, other):
        return self._apply_op(lambda a,b: b@a, other)
    def __truediv__(self, other):
        return self._apply_op(operator.truediv, other)
    def __rtruediv__(self, other):
        return self._apply_op(lambda a,b: b/a, other)
    def __pow__(self, other, modulo=None):
        return self._apply_op(lambda a,b: pow(a, b, modulo), other)
    def __floordiv__(self, other):
        return self._apply_op(operator.floordiv, other)
    def __rfloordiv__(self, other):
        return self._apply_op(lambda a,b: b//a, other)
    def __mod__(self, other):
        return self._apply_op(operator.mod, other)

class PopulationSeries(PopulationHistory, Series):
    """
    Similar to a Series, with the following differences:
        - There is only one data dimension. Each position along that dimension
          is referred to as a 'unit'.
        - Units can be grouped into arbitrary sized populations, with the
          constraint being that the sum of population sizes be the same as the
          total number of units.
    As for Spiketrain, the `shape` argument is replaced by `pop_sizes`.
    """
    __hash__ = History.__hash__  # Not inherited; see https://github.com/samuelcolvin/pydantic/issues/2422#issuecomment-828439446

class AutoHist:
    """
    Placeholder history; valid only inside a `~sinn.models.Model`.
    Simply stores the initialization values – the `Model` initializer then
    uses these to create the History, adding the model's `time` attribute.

    Keyword arguments only.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs


#######################################
# Views
#######################################

class DataView(HistoryBase):
    """
    Gives direct access to the history data.
    If the history is not fully computed, the returned history is truncated
    after the latest computed time point.

    Retrieving numerical data varies depending whether a
    history's data is stored as a NumPy, Theano shared or
    Theano variable. This abstracts out the calls, providing
    an interface as though the data was a NumPy object.

    Todo
    ----
    - In __new__, allow to
        + Subclass proper history
        + Detect if hist is already a DataView, and return itself
          in that case
    - Add appropriate method redirections when they are needed.
    """

    def __init__(self, hist):
        self.hist = hist
        # We don't initialize base class, but let __getattr__ take care of undefined attributes
        self.time = hist.time.stops_array[:hist.cur_tidx+1]
        self.tn = hist.cur_tidx

    def __getitem__(self, key):
        return self.hist[shim.eval(key)]

    def __getattr__(self, name):
        return getattr(self.hist, name)

    def __len__(self):
        return len(self.hist)

    # Methods in HistoryBase must be explicitly redirected
    # TODO

#############################
# Registrations
#############################

TimeAxis.update_forward_refs()
TimeArrayAxis.update_forward_refs()
HistoryUpdateFunction.update_forward_refs()
History.update_forward_refs()
    # This is required for the `hist: History` annotation at beginning of model
PopulationHistory.update_forward_refs()
Spiketrain.update_forward_refs()
Series.update_forward_refs()
