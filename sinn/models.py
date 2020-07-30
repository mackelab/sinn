# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René

TODO
----

- At present, things link ``h for h not in model.statehists`` will use __eq__
  for the ``in`` check, which for histories creates a dict and compares.
  Iterables like `statehists` should be something like a ``set``, such that
  this check then just compares identity / hashes.
- Have one dictionary/SimpleNamespace storing all compilation variables.
  See comment under `class Model`
"""

import numpy as np
import scipy as sp
#from scipy.integrate import quad
#from collections import namedtuple
from warnings import warn
import logging
logger = logging.getLogger(__name__)
import abc
from collections import namedtuple, OrderedDict, ChainMap
from collections.abc import Sequence, Iterable, Callable
import inspect
from inspect import isclass
from itertools import chain
from functools import partial
import copy
import sys

import pydantic
from pydantic import BaseModel, validator, root_validator
from pydantic.typing import AnyCallable
from typing import Any, Optional, Union, Tuple, Callable as CallableT
from inspect import signature

import theano_shim as shim
import mackelab_toolbox as mtb
import mackelab_toolbox.utils as utils
import mackelab_toolbox.theano
import mackelab_toolbox.typing
from mackelab_toolbox.theano import GraphCache, CompiledGraphCache
from mackelab_toolbox.utils import class_or_instance_method

import sinn
import sinn.config as config
import sinn.common as com
from sinn.axis import DiscretizedAxis
from sinn.histories import (
    History, TimeAxis, AutoHist, HistoryUpdateFunction, Series, Spiketrain)
from sinn.kernels import Kernel
import sinn.diskcache as diskcache

_models = {}
registered_models = _models.keys()
    # I don't really like this, but it works. Ideally it would be some kind
    # of read-only property of the module.
    # Registering models allows us to save the model name within a parameter
    # file, and write a function which can build the correct model
    # automatically based on only on that parameter file.

expensive_asserts = True

failed_build_msg = (
        "Failed to build the symbolic update. Make sure that the "
        "model's definition of State is correct: it should include "
        "enough histories to fully define the model's state and "
        "allow forward integration. If you are sure the problem is "
        "not your model, you may need to workaround the issue by "
        "defining a `symbolic_update` in your model class. "
        "Automatic construction of symbolic updates is still work in "
        "progress and not always possible.")

def register_model(model, modelname=None):
    """
    Register a subclass of Model.
    Typically this is called from the module which implements the subclass.
    If `modelname` is unspecified, `model`'s class name is used.
    """
    global _models
    assert(isclass(model))
    if modelname is None:
        modelname = model.__name__
    assert(isinstance(modelname, str))
    _models[modelname] = model

def is_registered(modelname):
    """Returns True if a model is registered with this name."""
    global _models
    return modelname in _models

def get_model(modelname, *args, **kwargs):
    """Retrieves the model associated with the model name. Same arguments as dict.get()."""
    global _models
    return _models.get(modelname, *args, **kwargs)

# Parameters

class ModelParams(BaseModel):
    class Config:
        json_encoders = mtb.typing.json_encoders

# Model decorators

## updatefunction decorator

PendingUpdateFunction = namedtuple('PendingUpdateFunction',
                                   ['hist_nm', 'inputs', 'upd_fn'])
def updatefunction(hist_nm, inputs):
    """
    Decorator. Attaches the following function to the specified history,
    once the model is initialized.
    """
    if not isinstance(hist_nm, str) or not isinstance(inputs, (list, tuple)):
        raise ValueError("updatefunction decorator must be used as \n"
                         "  @updatefunction('hist_nm', inputs=['hist1', hist2'])\n"
                         "  def hist_update(self, tidx):\n""    …")
    def dec(upd_fn):
        return PendingUpdateFunction(hist_nm, inputs, upd_fn)
    return dec

## initializer decorator
def initializer(
    *fields, unintialized=None, pre=True, always=False, **dec_kwargs
) -> CallableT[[AnyCallable], classmethod]:
    """
    Specialized validator for writing more complex default initializers with
    less boilerplate. Does two things:

    - Changes the default for `pre` to ``True``.
    - Always sets `always=True` (the `always` parameter is still accepted,
      but with a slightly different meaning; see note below).
    - Allows model parameters to be specified as keyword arguments in the
      validator signature. This works with both model-level parameters, and
      the parameters defined in the `Parameters` subclass.

    .. Note:: The point of an initializer is to replace a default value, so
       it doesn't make sense to set `always=True`. However, by default an
       initializer will *not* execute if a value is already provided.
       (The logic being that if a value is provided, it doesn't need to be
       initialized.) Thus, in analogy with `~pydantic.validator`, the `always`
       keyword is provided to specify that an initializer should be run even if
       a value for that parameter is provided.

    Example
    -------

    The following

    >>> class Model(BaseModel):
    >>>   a: float
    >>>   t: float = None
    >>>   @initializer('t'):
    >>>   def set_t(t, a):
    >>>     return a/4

    is equivalent to

    >>> class Model(BaseModel):
    >>>   a: float
    >>>   t: float = None
    >>>   @validator('t', pre=True, always=True):
    >>>   def set_t(t, values):
    >>>     if t is not None:
    >>>       return t
    >>>     a = values.get('a', None)
    >>>     if a is None:
    >>>       raise AssertionError(
    >>>         "'a' cannot be found within the model parameters. This may be "
    >>>         "because it is defined after 't' in the list of parameters, "
    >>>         "or because its own validation failed.")
    >>>     return a/4

    Parameters
    ----------
    *fields
    pre (default: True)
    each_item
    check_fields
    allow_reuse: As in `pydantic.validator`, although some arguments may not
        be so relevant.

    always: bool
        - `True`: Always run the initializer. This is the same as setting
          `always=True` with a Pydantic `~pydantic.validator`.
        - `False` (default): Only run the initializer when the value is **not**
          provided. Note that this is the opposite effect to setting
          `always=False` with a Pydantic `~pydantic.validator`.

    uninitialized: Any (default: None)
        The initializer is only executed when the parameter is equal to this
        value.
    """

    val_fn = validator(*fields, pre=pre, always=True, **dec_kwargs)

    # Refer to pydantic.class_validators.make_generic_validator
    def dec(f: AnyCallable) -> classmethod:
        sig = signature(f)
        args = list(sig.parameters.keys())
        # 'value' is the first argument != from 'self', 'cls'
        # It is positional, and the only required argument
        if args[0] in ('self', 'cls'):
            req_val_args = args[:2]
            opt_val_args = set(args[2:])  # Remove cls and value
        else:
            req_val_args = args[:1]
            opt_val_args = set(args[1:])  # Remove value
        # opt_validator_args will store the list of arguments recognized
        # by pydantic.validator. Everything else is assumed to match an earlier
        # parameter.
        param_args = set()
        for arg in opt_val_args:
            if arg not in ('values', 'config', 'field', '**kwargs'):
                param_args.add(arg)
        for arg in param_args:
            opt_val_args.remove(arg)
        def new_f(cls, v, values, field, config):
            if not always and v is not unintialized:
                return v
            param_kwargs = {}
            params = values.get('params', None)
            if not isinstance(params, BaseModel):
                params = None  # We must not be within a sinn Model => 'params' does not have special meaning
            for p in param_args:
                if p in values:  # Try module-level param first
                    pval = values.get(p)
                elif params is not None and hasattr(params, p):
                    pval = getattr(params, p)
                else:
                    raise AssertionError(
                      f"'{p}' cannot be found within the model parameters. "
                      "This may be because it is "
                      f"defined after '{field.name}' in the list of parameters, "
                      "or because its own validation failed.")
                param_kwargs[p] = pval

            # Now assemble the expected standard arguments
            if len(req_val_args) == 2:
                val_args = (cls, v)
            else:
                val_args = (v,)
            val_kwargs = {}
            if 'values' in opt_val_args: val_kwargs['values'] = values
            if 'field' in opt_val_args: val_kwargs['field'] = field
            if 'config' in opt_val_args: val_kwargs['config'] = config

            return f(*val_args, **val_kwargs, **param_kwargs)

        # Can't use @wraps because we changed the signature
        new_f.__name__ = f.__name__
        # Having a different qualname is required to avoid overwriting validators
        # (Pydantic identifies them by name, and otherwise they all have `new_f`)
        new_f.__qualname__ = f.__qualname__
        new_f.__doc__ = f.__doc__

        return val_fn(new_f)

    return dec

class ModelMetaclass(pydantic.main.ModelMetaclass):
    # Partial list of magical things done by this metaclass:
    # - Transform a plain `Parameters` class into a pydantic BaseModel
    #   (specifically, ModelParams)
    # - Add the `params: Parameters` attribute to the annotatons.
    #   Ensure in inherits from the changed `Parameters`
    # - Move `params` to the front of the annotations list, so it is available
    #   to validators.
    # - Add the `time` annotation if it isn't already present.
    def __new__(metacls, cls, bases, namespace):
        # MRO resolution
        # We will need to retrieve attributes which may be anywhere in the MRO.
        # USE: Sanity checks, retrieve `Parameters`, inherited annotations
        # NOTE: type.mro(metacls) would return the MRO of the metaclass, while
        #     we want that of the still uncreated class.
        # Option 1: Create a new throwaway class, and use its `mro()` method
        #     Implementation: type('temp', bases, namespace)
        #     Problem: Infinite recursion if Model is within `bases`
        # Option 2: Force `bases` to have one element, and call its `mro()`
        #     Implementation: bases[0].mro()
        #     Problem: Multiple inheritance of models is no longer possible.
        # At the moment I begrudgingly went with Option 1, because I'm not sure
        # of a use case for multiple inheritance.
        nonabcbases = tuple(b for b in bases if b is not abc.ABC)
        if len(nonabcbases) != 1:
            from inspect import currentframe, getframeinfo
            import pathlib
            path = pathlib.Path(__file__).absolute()
            frameinfo = getframeinfo(currentframe())
            info = f"{path}::line {frameinfo.lineno}"
            basesstr = ', '.join(str(b) for b in nonabcbases)
            raise TypeError(
                f"Model {cls} has either no or multiple parents: {basesstr}.\n"
                "Models must inherit from exactly one class (eventually "
                "`sinn.models.Model`). This is a technical limitation rather "
                "than a fundamental one. If you need multiple inheritance for "
                f"your model, have a look at the comments above {info}, and "
                "see if you can contribute a better solution.")
        mro = bases[0].mro()

        # Existing attributes
        # (ChainMap gives precedence to elements earlier in the list)
        annotations = namespace.get('__annotations__', {})
        inherited_annotations = ChainMap(*(getattr(b, '__annotations__', {})
                                           for b in mro))
        all_annotations = ChainMap(annotations, inherited_annotations)
        # (We only iterate over immediate bases, since those already contain
        #  the values for their parents)
        inherited_kernel_identifiers = set(chain.from_iterable(
            getattr(b, '_kernel_identifiers', []) for b in bases))
        inherited_hist_identifiers = set(chain.from_iterable(
            getattr(b, '_hist_identifiers', []) for b in bases))
        inherited_pending_updates = dict(ChainMap(
            *({obj.hist_nm: obj for obj in getattr(b, '_pending_update_functions', [])}
              for b in bases)))

        # Structures which accumulate the new class attributes
        new_annotations = {}
        _kernel_identifiers = inherited_kernel_identifiers
        _hist_identifiers = inherited_hist_identifiers
        _model_identifiers = set()
        _pending_update_functions = inherited_pending_updates
            # pending updates use a dict to allow derived classes to override

        # Model validation
        ## `time` parameter
        if 'time' not in annotations:
            annotations['time'] = TimeAxis
        else:
            if (not isinstance(annotations['time'], type)
                or not issubclass(annotations['time'], DiscretizedAxis)):
                raise TypeError(
                    "`time` attribute must be an instance of `DiscretizedAxis`; "
                    "in general `histories.TimeAxis` is an appropriate type.")
        # ## 'initialize' method
        # if not isinstance(namespace.get('initialize', None), Callable):
        #     raise TypeError(f"Model {cls} does not define an `initialize` "
        #                     "method.")

        # Add module-level annotations

        # TODO?: Allow derived classes to redefine histories ?
        #        We would just need to add the inherited kernel/hists after this loop
        for nm, T in annotations.items():
            if nm in new_annotations:
                raise TypeError("Name clash in {cls} definition: '{nm}'")
            new_annotations[nm] = T
            if isinstance(T, type) and issubclass(T, History):
                _hist_identifiers.add(nm)
            elif isinstance(T, type) and issubclass(T, Kernel):
                _kernel_identifiers.add(nm)
            elif isinstance(T, type) and cls != 'Model' and issubclass(T, Model):
                _model_identifiers.add(nm)
            elif isinstance(T, PendingUpdateFunction):
                # FIXME: Remove. I don't remember why I originally put this branch
                assert False
                # _pending_update_functions.append(obj)

        # Sanity check State subclass
        State = namespace.get('State', None)
        if abc.ABC in bases:
            # Deactivate warnings for abstract models
            pass
        elif State is None:
            if len(_hist_identifiers) > 0:
                # If there are no histories, it makes sense not to specify a State
                # This can happen for models which just combine submodels
                warn(f"Model {cls} does not define a set of state variables.")
            State = type('State', (), {'__annotations__': {}})
            namespace['State'] = State
        elif len(getattr(State, '__annotations__', {})) == 0:
            warn(f"Model {cls} has an empty `State` class.")
        else:
            if issubclass(State, BaseModel):
                raise TypeError(f"Model {cls} `State` must be a plain class, "
                                f"not a Pydantic BaseModel.")
            # if len(_hist_identifiers) == 0:
            #     raise TypeError(
            #         f"Model {cls}: Variables declared in `State` are not all "
            #         "declared as histories in the model.")
            for nm, T in State.__annotations__.items():
                histT = all_annotations.get(nm, None)
                if histT is None:
                    raise TypeError(
                        f"Model {cls}: `State` defines '{nm}', which is not "
                        "defined in the model.")
                if T is not Any:
                    raise TypeError(
                        "At the moment, all attributes of the `State` class "
                        "should have type `Any`. In the future we may add "
                        "the possibility to be more specific.")
                # NOTE: If checking the type histT, remember that it may be
                #       something else than a History (e.g. RNG).
                # elif T is not histT:
                #     raise TypeError(
                #         f"Model {cls}: `State` defines '{nm}' with type '{T}', "
                #         f"while the model defines it with type '{histT}'.")

        # Get Parameters and Config: iterate through MRO and stop on 1st hit
        Parameters = namespace.get('Parameters', None)  # First check namespace
        Config = namespace.get('Config', None)
        if Config is None:
            Config = type("Config", (), {'keep_untouched': ()})
        for C in mro:
            if Parameters is not None:  # and Config is not None:
                break  # We've found the Parameters classes; no need to look further
            if Parameters is None:
                Parameters = getattr(C, 'Parameters', None)
            # if Config is None:
            #     Config = getattr(C, 'Config', None)

        # Sanity check Parameters subclass
        # Parameters = namespace.get('Parameters', None)
        if not isinstance(Parameters, type):
            raise TypeError(f"Model {cls}: `Parameters` must be a class.")
        # if (not isinstance(Parameters, type)
        #     or not issubclass(Parameters, BaseModel)):
        #     raise TypeError(
        #         f"Model {cls}: `Parameters` must inherit from pydantic."
        #         f"BaseModel. `{cls}.Parameters.mro()`: {Parameters.mro()}.")

        # Ensure `Parameters` inherits from ModelParams
        if not issubclass(Parameters, ModelParams):
            # We can't have multiple inheritance if the parents don't have the
            # same metaclass. Standard solution: Create a new metaclass, which
            # inherits from the other two, and use *that* as a metaclass.
            paramsmeta = type(ModelParams)
            if paramsmeta is not type(Parameters):
                paramsmeta = type("ParametersMeta_Subclass",
                                  (paramsmeta, type(Parameters)),
                                  {})
            OldParameters = Parameters
            if (not issubclass(Parameters, BaseModel)
                and hasattr(Parameters, '__annotations__')):
                # If `Parameters` is not even a subclass of BaseModel, we have
                # to move its annotations to the derived class
                params_namespace = {'__annotations__': Parameters.__annotations__}
                # Also move any defined fields or default values
                # We also need to catch validators, etc.
                # FIXME: I haven't figured out a way to move validators. They
                #        need to be of type 'classmethod' when pydantic sees
                #        them, they only remain such during the creation of
                #        the class. Afterwards, they are just 'methods'.
                #        So, current approach is, if there are any non-dunder
                #        attributes, we raise an error.
                if [attr for attr in dir(Parameters) if attr[:2] != '__']:
                    raise TypeError("A model's `Parameters` must subclass "
                                    "`sinn.models.ModelParams`.")
                # for attr in dir(Parameters):
                #     if attr[:2] != '__':
                #         params_namespace[attr] = getattr(Parameters, attr)
                #         delattr(Parameters, attr)
                Parameters.__annotations__= {}
            else:
                params_namespace = {}
            Parameters = paramsmeta("Parameters", (OldParameters, ModelParams),
                                    params_namespace)

        # Sanity check Config subclass, if present
        # Config = namespace.get('Config', None)
        # No warnings for Config: not required
        if not isinstance(Config, type):
            raise TypeError(f"Model {cls} `Config` must be a class, "
                            f"not {type(Config)}.")

        # Rename State and Parameters classes for easier debugging/inspection
        # Don't just append the class name, because classes can be nested and
        # that could lead to multiple names
        # It seems prudent to leave __qualname__ untouched
        if getattr(State, '__name__', None) == "State":
            State.__name__ = f"State ({cls})"
        if getattr(Parameters, '__name__', None) == "Parameters":
            Parameters.__name__ == f"Parameters ({cls})"

        # Add 'params' variable if it isn't present, and place first in
        # the list of variables so that initializers can find it.
        if 'params' in new_annotations:
            ann_params = new_annotations['params']
            if new_annotations['params'] is not Parameters:
                if isinstance(ann_params, type) and issubclass(Parameters, ann_params):
                    new_annotations['params'] = Parameters
                else:
                    raise TypeError(f"Model {cls} defines `params` but it is "
                                    f"not of type `{cls}.Parameters`")
            new_annotations = {'params': new_annotations.pop('params'),
                               **new_annotations}
        elif issubclass(Parameters, abc.ABC):
            # Most likely no Parameters class was defined, and we retrieved
            # the abstract definition above.
            # => We skip creating the 'params' attribute, since there is no
            #    point in requiring it when the model defines no parameters
            if len(_hist_identifiers) > 0:
                # If there are no histories, it makes sense not to specify a State
                # This can happen for models which just combine submodels
                warn(f"Model '{type(self).__name__}' does not define a `Parameters` "
                     "class, or its `Parameters` class inherits from `abc.ABC`.")
            # Create a new type to minimize the risk of clobbering the base Parameters type
            Parameters = type('Parameters', (Parameters,), {})
        else:
            new_annotations = {'params': Parameters, **new_annotations}

        # Add update functions to list
        for obj in namespace.values():
            if isinstance(obj, PendingUpdateFunction):
                if obj.hist_nm not in _hist_identifiers:
                    raise TypeError(
                        f"Update function {obj.upd_fn} is intended for history "
                        f"{obj.hist_nm}, but it is not defined in the model.")
                _pending_update_functions[obj.hist_nm] = obj

        # Add AutoHist validators
        for nm, obj in list(namespace.items()):
            if isinstance(obj, AutoHist):
                T = annotations.get(nm, None)
                # Determine a name for the initialization validator which
                # doesn't match something already in `namespace`
                fn_nm = f"autohist_{nm}"
                if fn_nm in namespace:
                    # I honestly don't know why someone would define a model
                    # with clashing names, but just in case.
                    for i in range(10):
                        fn_nm = f"autohist_{nm}_{i}"
                        if fn_nm not in namespace:
                            break
                    assert fn_nm not in namespace
                if T is None:
                    raise TypeError("`AutoHist` must follow a type annotation.")
                if T is Series:
                    namespace[fn_nm] = validator(
                        nm, allow_reuse=True, always=True, pre=True)(
                            init_autoseries)
                elif T is Spiketrain:
                    namespace[fn_nm] = validator(
                        nm, allow_reuse=True, always=True, pre=True)(
                            init_autospiketrain)
                else:
                    raise TypeError("Unrecognized history type; recognized "
                        "types:\n    Series, Spiketrain")

        # Update namespace
        namespace['Config'] = Config
        namespace['Parameters'] = Parameters
        namespace['_kernel_identifiers'] = list(_kernel_identifiers)
        namespace['_hist_identifiers'] = list(_hist_identifiers)
        namespace['_model_identifiers'] = list(_model_identifiers)
        namespace['_pending_update_functions'] = list(_pending_update_functions.values())
        namespace['__annotations__'] = new_annotations

        return super().__new__(metacls, cls, bases, namespace)

    # TODO: Recognize RNG as input

def init_autospiketrain(cls, autohist: AutoHist, values) -> Spiketrain:
    time = values.get('time', None)
    if time is None: return autohist
    return Spiketrain(time=time, **autohist.kwargs)

def init_autoseries(cls, autohist: AutoHist, values) -> Series:
    time = values.get('time', None)
    if time is None: return autohist
    return Series(time=time, **autohist.kwargs)

class Model(pydantic.BaseModel, abc.ABC, metaclass=ModelMetaclass):
    """Abstract model class.

    A model implementations should derive from this class.
    It must minimally provide:
    - A `Parameter_info` dictionary of the form:
        (See sinn.common.Parameterize)
        ```
        Parameter_info = OrderedDict{ 'param_name': Parameter([cast function], [default value]),
                                      ... }
        ```
    - A class-level (outside any method) call
        `Parameters = com.define_parameters(Parameter_info)`

    Implementations may also provide class methods to aid inference:
    - likelihood: (params) -> float
    - likelihood_gradient: (params) -> vector
    If not provided, `likelihood_gradient` will be calculated by appyling theano's
    grad method to `likelihood`. (TODO)
    As class methods, these don't require an instance – they can be called on the class directly.

    .. Hint::
       If you are subclassing Model to create another abstract class (a class
       meant to be subclassed rather than used on its own), add `abc.ABC` to
       its parents – this will identify your model as abstract, and disable
       warnings about missing attributes like `State`.
    """
    # ALTERNATIVE: Have a SimpleNamespace attribute `priv` in which to place
    #              private attributes, instead of listing them all here.
    __slots__ = ('graph_cache', 'compile_cache', '_pymc', 'batch_start_var', #'batch_size_var',
                 '_num_tidx', '_curtidx', '_stoptidx_var', '_batchsize_var',
                 '_advance_updates', '_compiled_advance_fns')

    class Config:
        # Allow assigning other attributes during initialization.
        extra = 'allow'
        keep_untouched = (ModelParams, PendingUpdateFunction, class_or_instance_method)

    class Parameters(abc.ABC, ModelParams):
        """
        Models must define a `Parameters` class within their namespace.
        Don't inherit from this class; i.e. do::

            class MyModel(Model):
                class Parameters:
                    ...
        """
        pass

    def __init__(self, initializer=None, **kwargs):
        # Initialize attributes with Pydantic
        super().__init__(**kwargs)
        # Attach update functions to histories, and set up __slots__
        self._base_initialize()
        # Run the model initializer
        self.initialize(initializer)

    def copy(self, *args, deep=False, **kwargs):
        m = super().copy(*args, deep=deep, **kwargs)
        m._base_initialize(shallow_copy=not deep)
        m.initialize()
        return m

    @classmethod
    def parse_obj(cls, obj):
        m = super().parse_obj(obj)
        m._base_initialize()
        m.initialize()
        return m

    def _base_initialize(self, shallow_copy=False):
        """
        Collects initialization that should be done in __init__, copy & parse_obj.
        """
        # Attach history updates (don't do during shallow copy – histories are preserved then)
        if not shallow_copy:
            for obj in self._pending_update_functions:
                hist = getattr(self, obj.hist_nm)
                hist.update_function = HistoryUpdateFunction(
                    namespace    = self,
                    func         = partial(obj.upd_fn, self),  # Wrap history so `self` points to the model
                    inputs       = obj.inputs,
                    parent_model = self
                )
        object.__setattr__(self, 'graph_cache',
                           GraphCache('.sinn.graphcache/models', type(self),
                                      modules=('sinn.models',)))
        object.__setattr__(self, 'compile_cache',
                           CompiledGraphCache('.sinn.graphcache/models.compilecache'))
            # TODO: Add other dependencies within `sinn.models` ?
        object.__setattr__(self, '_advance_updates', {})
        object.__setattr__(self, '_compiled_advance_fns', {})
            # Keys of these dictionary are tuples of histories passed to `integrate(histories=…)`,
            # i.e. extra histories to integrate along with the state.
            # Values of the first are update dictionaries
            # Values of the second are compiled Theano functions.
        # Create symbolic variables for batches
        if shim.cf.use_theano:
            # # Any symbolic function on batches should use these, that way
            # # other functions can retrieve the symbolic input variables.
            start = np.array(1).astype(self.tidx_dtype)
            object.__setattr__(self, 'batch_start_var',
                               shim.shared(start, name='batch_start'))
            #     # Must be large enough so that test_value slices are not empty
            size = np.array(2).astype(self.tidx_dtype)
            # object.__setattr__(self, 'batch_size_var',
            #                    shim.shared(size, name='batch_size'))
            # #     # Must be large enough so that test_value slices are not empty

    @abc.abstractmethod
    def initialize(self, initializer: Any=None):
        """
        Models must define an `initialize` method. This is where you can add
        padding to histories, pre-compute kernels, etc. – anything which should
        be done whenever parameters changed.

        It takes one optional keyword argument, `initializer`, which can be of
        any form; the model will accept an `initializer` argument at
        instantiation and pass it along to this method.
        This argument can be e.g. a string flag, to indicate one of multiple
        initialization protocols, or a dictionary with multiple initialization
        parameters.

        .. important:: Any left-padded history should be filled up to -1 after
           a call to `initialize`.

        Arguably this could be implemented as a `root_validator`, but for at
        least for now having a method with exactly this name is required.
        """
        pass


    # @root_validator
    # def check_same_dt(cls, values):
    #     hists = [h for h in values if isinstance(h, History)]
    #     if any(h1.dt != h2.dt for h1,h2 in zip(hists[:-1], hists[1:])):
    #         steps = ", ".join(f"{str(h)} (dt={h.dt})" for h in hists)
    #         raise ValueError(
    #             f"Histories do not all have the same time step.\n{steps}")
    #     return values

    @root_validator
    def consistent_times(cls, values):
        time = values.get('time', None)
        hists = (v for v in values if isinstance(v, History))
        if time is not None:
            for hist in hists:
                if not time.Index.is_compatible(hist.time.Index):
                    raise ValueError(
                        "History and model have incompatible time indexes.\n"
                        f"History time index: {hist.time}\n"
                        f"Model time index: {time}")
        return values


    # Called by validators in model implementations
    @classmethod
    def check_same_shape(cls, hists):
        if any(h1.shape != h2.shape for h1,h2 in zip(hists[:-1], hists[1:])):
            shapes = ", ".join(f"{str(h)} (dt={h.shape})" for h in hists)
            raise ValueError(
                f"Histories do not all have the same time shape.\n{shapes}")
        return None

    @root_validator
    def sanity_rng(cls, values):
        hists = values.get('histories', None)
        if hists is None: return values
        for h in hists:
            input_rng = [inp for inp in h.update_function.inputs
                         if isinstance(inp, mtb.typing.AnyRNG)]
            if len(input_rng) > 0:
                Model.output_rng(h, input_rng)
        return values

    @staticmethod
    def output_rng(outputs, rngs):
        """
        Check that all required random number generators (rngs) are specified.
        This function is just a sanity check: it only ensures that the RNGs
        are not None, if any of the outputs are uncomputed.

        Parameters
        ----------
        outputs: History
            Can also be a list of Histories
        rngs: random stream, or list of random streams
            The random stream(s) required to generate the histories in
            `outputs`

        Raises
        ------
        ValueError:
            If at least one RNG is `None` and at least one of the `outputs`
            is both unlocked and not fully computed.

        Warnings
        -------
        UserWarning:
            If all histories are already computed but an RNG is specified,
            since in this case the RNG is not used.
        """
        if isinstance(outputs, History):
            outputs = [outputs]
        else:
            assert(all(isinstance(output, History) for output in outputs))
        try:
            len(rngs)
        except TypeError:
            rngs = [rngs]

        # if any( not shim.isshared(outhist._sym_data) for outhist in outputs ):
        #     # Bypass test for Theano data
        #     return

        unlocked_hists = [h for h in outputs if not h.locked]
        hists_with_missing_rng = []
        hists_with_useless_rng = []
        for h in outputs:
            hinputs = h.update_function.inputs
            if not h.locked and h.cur_tidx < h.tnidx:
                missing_inputs = ", ".join(
                    nm for nm, inp in zip(
                        h.update_function.input_names, hinputs)
                    if inp is None)
                if len(missing_inputs) > 0:
                    hists_with_missing_rng.append(f"{h.name}: {missing_inputs}")
            else:
                useless_rng = ", ".join(
                    nm for nm, inp in zip(
                        h.update_function.input_names, hinputs)
                    if inp is shim.config.RNGTypes)
                if len(useless_rng) > 0:
                    hists_with_useless_rng.append(f"{h.name}: {useless_rng}")
        if len(hists_with_missing_rng):
            missing = "\n".join(hists_with_missing_rng)
            raise ValueError(
                "The following histories are missing the following inputs:\n"
                + missing)
        if len(hists_with_useless_rng):
            useless = "\n".join(hists_with_missing_rng)
            warn("The random inputs to the following histories will be "
                 "ignored, since the histories are already computed:\n"
                 "(hist name: random input)\n" + useless)

    #############
    # Specializations of standard dunder methods

    def __str__(self):
        name = getattr(self, '__name__', type(self).__name__)
        return "Model '{}' (t0: {}, tn: {}, dt: {})" \
            .format(name, self.t0, self.tn, self.dt)

    def __repr__(self):
        return self.summarize()

    def __getattribute__(self, attr):
        """
        Retrieve parameters if their name does not clash with an attribute.
        """
        # Use __getattribute__ to maintain current stack trace on exceptions
        # https://stackoverflow.com/q/36575068
        # if (attr != 'params'
        #     and attr in self.params.__fields__):
        if (attr != 'params'              # Prevent infinite recursion
            # and attr not in self.__dict__ # Prevent shadowing
            and hasattr(self, 'params') and attr in self.params.__fields__):
            # Return either a PyMC3 prior (if in PyMC3 context) or shared var
            # Using sys.get() avoids the need to load pymc3 if it isn't already
            pymc3 = sys.modules.get('pymc3', None)
            param = getattr(self.params, attr)
            if not hasattr(param, 'prior') or pymc3 is None:
                return param
            else:
                try:
                    pymc3.Model.get_context()
                except TypeError:
                    # No PyMC3 model on stack – return plain param
                    return param
                else:
                    # Return PyMC3 prior
                    return param.prior

        else:
            return super().__getattribute__(attr)

    # Pickling infrastructure
    # Reference: https://docs.python.org/3/library/pickle.html#pickle-state
    def __getstate__(self):
        raise NotImplementedError
        # # Clear the nonstate histories: by definition these aren't necessary
        # # to recover the state, but they could store huge intermediate date.
        # # Pickling only stores history data up to their `cur_tidx`, so by
        # # clearing them we store no data at all.
        # state = self.__dict__.copy()
        # for attr, val in state.items():
        #     if val in self.nonstatehists:
        #         # TODO: Find way to store an empty history without clearing
        #         #       the one in memory. The following would work if it
        #         #       didn't break all references in update functions.
        #         # state[attr] = val.copy_empty()
        #         val.clear()
        # return state

    ###################
    # Inspection / description methods
    # These methods do not modify the model

    # TODO: Some redundancy here, and we could probably store the
    # hist & kernel lists after object creation – just ensure this is
    # also done after copy() and parse_obj()
    # FIXME: what to do with histories which aren't part of model (e.g.
    #        returned from an operation between hists) ?
    @property
    def nonnested_histories(self):
        return {nm: getattr(self, nm) for nm in self._hist_identifiers}
    @property
    def nonnested_history_set(self):
        return {getattr(self, nm) for nm in self._hist_identifiers}
    @property
    def history_set(self):
        return set(chain(self.nonnested_history_set,
                         *(m.history_set for m in self.nested_models_list)))
    @property
    def nonnested_kernels(self):
        return {nm: getattr(self, nm) for nm in self._kernel_identifiers}
    @property
    def nonnested_kernel_list(self):
        return [getattr(self, nm) for nm in self._kernel_identifiers]
    @property
    def kernel_list(self):
        return list(chain(self.nonnested_kernel_list,
                          *(m.kernel_list for m in self.nested_models_list)))
    @property
    def nested_models(self):
        return {nm: getattr(self, nm) for nm in self._model_identifiers}
    @property
    def nested_models_list(self):
        return [getattr(self, nm) for nm in self._model_identifiers]

    @property
    def t0(self):
        return self.time.t0
    @property
    def tn(self):
        return self.time.tn
    @property
    def t0idx(self):
        return self.time.t0idx
    @property
    def tnidx(self):
        return self.time.tnidx
    @property
    def tidx_dtype(self):
        return self.time.Index.dtype
    @property
    def dt(self):
        return self.time.dt

    @property
    def statehists(self):
        nested_len = sum(len(m.statehists) for m in self.nested_models_list)
        return utils.FixedGenerator(
            chain(
                (getattr(self, varname) for varname in self.State.__annotations__),
                *(m.statehists for m in self.nested_models_list)),
            nested_len + len(self.State.__annotations__) )

    @property
    def unlocked_statehists(self):
        return (h for h in self.statehists if not h.locked)

    @property
    def locked_statehists(self):
        return (h for h in self.statehists if h.locked)

    @property
    def unlocked_histories(self):
        return (h for h in self.history_set if not h.locked)

    @property
    def nonstatehists(self):
        statehists = list(self.statehists)
        return utils.FixedGenerator(
            (h for h in self.history_set if h not in statehists),
            len(self.history_set) - len(self.statehists) )

    @property
    def unlocked_nonstatehists(self):
        return (h for h in self.nonstatehists if not h.locked)

    @property
    def rng_inputs(self):
        rng_inputs = []
        for h in self.unlocked_histories:
            for nm in h.update_function.input_names:
                inp = getattr(h.update_function.namespace, nm)
                if (isinstance(inp, shim.config.RNGTypes)
                    and inp not in rng_inputs):
                    rng_inputs.append(inp)
        return rng_inputs

    @property
    def rng_hists(self):
        rng_hists = []
        for h in self.unlocked_histories:
            for nm in h.update_function.input_names:
                inp = getattr(h.update_function.namespace, nm)
                if isinstance(inp, shim.config.RNGTypes):
                    rng_hists.append(h)
                    break
        return rng_hists

    @property
    def cur_tidx(self):
        """
        Return the earliest time index for which all histories are computed.
        """
        try:
            return min(h.cur_tidx.convert(self.time.Index)
                       for h in self.statehists)
        except IndexError as e:
            raise IndexError(
                "Unable to determine a current index for "
                f"{type(self).__name__}. This usually happens accessing "
                "`cur_tidx` before a model is initialized.") from e

    # Symbolic variables for use when compiling unanchored functions
    # Building as `shim.tensor(np.array(...))` assigns a test value to the
    # variable, allowing models to work with compute_test_value != 'ignore'
    # (As is required for PyMC3)
    # Stop test value should be at least 2 more than _curtidx, because scan runs
    # from `_curtidx + 1` to `stoptidx`.
    @property
    def curtidx_var(self):
        """
        Return a purely symbolic variable intended to represent the current
        time index of the model (i.e. all state histories have been computed
        up to this point inclusively).

        Always returns the same object, so that it can be substituted in
        computational graphs.

        .. Note:: Like all user-facing indices, this should be treated as an
           *axis index*, not a data index.
        """
        # It's important to guard with hasattr, because `self.curtidx_var`
        # must always return the same variable.
        if not hasattr(self, '_curtidx_var'):
            object.__setattr__(self, '_curtidx_var',
                               shim.tensor(np.array(1, dtype=self.tidx_dtype),
                                           name='curtidx (model)'))
        return self._curtidx_var
    @property
    def stoptidx_var(self):
        """
        Return a purely symbolic variable intended to represent the end point
        (exclusive) of a computation.

        Always returns the same object, so that it can be substituted in
        computational graphs.

        .. Note:: Like all user-facing indices, this should be treated as an
           *axis index*, not a data index.
        """
        if not hasattr(self, '_stoptidx_var'):
            object.__setattr__(self, '_stoptidx_var',
                               shim.tensor(np.array(3, dtype=self.tidx_dtype),
                                           name='stoptidx (model)'))
        return self._stoptidx_var
    @property
    def batchsize_var(self):
        """
        Return a purely symbolic variable intended to represent the batch size.
        This is sometimes more convenient in functions than specifying the end
        point.

        Always returns the same object, so that it can be substituted in
        computational graphs.
        """
        if not hasattr(self, '_batchsize_var'):
            object.__setattr__(self, '_batchsize_var',
                               shim.tensor(np.array(2, dtype=self.tidx_dtype),
                                           name='batchsize (model)'))
        return self._batchsize_var

    @property
    def num_tidx(self):
        """
        A shared variable corresponding to the current time point of
        the model. This is only defined if all histories are synchronized.

        Always returns the same object, so that it can be substituted in
        computational graphs.

        Raises `RuntimeError` if histories are not all synchronized.

        .. WARNING::
           This does not return an AxisIndex, so always wrap this variable
           with [model].time.Index or [model].time.Index.Delta before using
           it to index into a history.

        .. Dev note::
           If someone can suggest a way to make SymbolicAxisIndexMeta._instance_plain
           return the original underlying symbolic variable (the one which
           appears in graphs), I will gladly change this method to return a
           proper symbolic index.
        """
        if not self.histories_are_synchronized():
            raise RuntimeError(
                f"Histories for the {type(self).__name__}) are not all "
                "computed up to the same point. The compilation of the "
                "model's integration function is ill-defined in this case.")
        if not hasattr(self, '_num_tidx'):
            object.__setattr__(
                self, '_num_tidx',
                shim.shared(np.array(self.cur_tidx, dtype=self.tidx_dtype),
                            f"t_idx ({type(self).__name__})") )
        else:
            self._num_tidx.set_value(self.cur_tidx)
        return self._num_tidx

    def histories_are_synchronized(self):
        """
        Return True if all unlocked state hists are computed to the same time
        point, and all locked histories at least up to that point.
        """
        try:
            tidcs = [h.cur_tidx.convert(self.time.Index)
                     for h in self.unlocked_statehists]
        except IndexError:
            # If conversion fails, its because the converted index would be out
            # of the range of the new hist => hists clearly not synchronized
            return False
        locked_tidcs = [h.cur_tidx.convert(self.time.Index)
                        for h in self.locked_statehists]
        if len(tidcs) == 0:
            return True
        earliest = min(tidcs)
        latest = max(tidcs)
        if earliest != latest:
            return False
        elif any(ti < earliest for ti in locked_tidcs):
            return False
        else:
            return True

    @class_or_instance_method
    def summarize(self, hists=None):
        nested_models = self._model_identifiers
        if isinstance(self, type):
            nameline = "Model '{}'".format(self.__name__)
            paramline = "Parameters: " + ', '.join(self.Parameters.__annotations__) + "\n"
            if len(nested_models) == 0:
                nestedline = ""
            else:
                nestedline = "Nested models:\n    " + '\n    '.join(nested_models)
                nestedline = nestedline + "\n"
            nested_summaries = []
        else:
            assert isinstance(self, Model)
            nameline = str(self)
            if hasattr(self, 'params'):
                paramline = f"Parameters: {self.params}\n"
            else:
                paramline = "Parameters: None\n"
            if len(nested_models) == 0:
                nestedline = ""
            else:
                nestedlines = [f"    {attr} -> {type(cls).__name__}"
                               for attr, cls in self.nested_models.items()]
                nestedline = "Nested models:\n" + '\n'.join(nestedlines) + "\n"
            nested_summaries = [model.summarize(hists)
                                for model in self.nested_models.values()]
        nameline += '\n' + '-'*len(nameline) + '\n'  # Add separating line under name
        stateline = "State variables: " + ', '.join(self.State.__annotations__)
        stateline = stateline + "\n"
        summary = (nameline + stateline + paramline + nestedline
                   + '\n' + self.summarize_updates(hists))
        return '\n'.join((summary, *nested_summaries))

    @class_or_instance_method
    def summarize_updates(self, hists=None):
        """
        Return a string summarizing the update function. By default, returns those of `self.state`.
        May be called on the class itself, or an instance.

        FIXME
        -----
        hists

        Parameters
        ----------
        hists: list|tuple of histories|str
            List of histories for which to print the update.
            For each history given, retrieves its update function.
            Alternatively, a history's name can be given as a string
        """
        # Default for when `hists=None`
        if hists is None:
            hists = list(self.State.__annotations__.keys())
        # Normalize to all strings; take care that identifiers may differ from the history's name
        histsdict = {}
        nonnested_hists = getattr(self, 'nonnested_histories', None)
        for h in hists:
            if isinstance(h, History):
                h_id = None
                for nm, hist in nonnested_hists.items():
                    if hist is h:
                        h_id = nm
                        break
                if h_id is None:
                    continue
                h_nm = h.name
            else:
                assert isinstance(h, str)
                if h not in self.__annotations__:
                    continue
                h_id = h
                h_nm = h
            histsdict[h_id] = h_nm
        hists = histsdict
        funcs = {pending.hist_nm: pending.upd_fn
                 for pending in self._pending_update_functions}
        if not all(isinstance(h, str) for h in hists):
            raise ValueError(
                "`hists` must be a list of histories or history names.")

        # For each function, retrieve its source
        srcs = []
        for hist_id, hist_nm in hists.items():
            fn = funcs.get(hist_id, None)
            if fn is None:
                continue
            src = inspect.getsource(fn)
            # Check that the source defines a function as expected:
            # first non-decorator line should start with 'def'
            for line in src.splitlines():
                if line.strip()[0] == '@':
                    continue
                elif line.strip()[:3] != 'def':
                    raise RuntimeError(
                        "Something went wrong when retrieve an update function's source. "
                        "Make sure the source file is saved and try reloading the Jupyter "
                        "notebook. Source should start with `def`, but we got:\n" + src)
                else:
                    break
            # TODO: Remove indentation common to all lines
            if hist_id != hist_nm:
                hist_desc = f"{hist_id} ({hist_nm})"
            else:
                hist_desc = hist_id
            # Replace the `def` line by a more explicit string
            src = f"Update function for {hist_desc}:\n" + src.split('\n', 1)[1]
            srcs.append(src)

        # Join everything together and return
        return '\n\n'.join(srcs)

    def get_tidx(self, t, allow_rounding=False):
        # Copied from History.get_tidx
        if self.time.is_compatible_value(t):
            return self.time.index(t, allow_rounding=allow_rounding)
        else:
            assert self.time.is_compatible_index(t)
            if (isinstance(t, sinn.axis.AbstractAxisIndexDelta)
                and not isinstance(t, sinn.axis.AbstractAxisIndex)):
                raise TypeError(
                    "Attempted to get the absolute time index corresponding to "
                    f"{t}, but it is an index delta.")
            return self.time.Index(t)
    get_tidx.__doc__ = History.get_tidx.__doc__

    def get_tidx_for(self, t, target_hist, allow_fractional=False):
        raise DeprecationWarning("Use the `convert` method attached to the AxisIndex.")

    def index_interval(self, Δt, allow_rounding=False):
        return self.time.index_interval(value, value2,
                                        allow_rounding=allow_rounding,
                                        cast=cast)
    index_interval.__doc__ = TimeAxis.index_interval.__doc__

    def get_time(self, t):
        # Copied from History
        # TODO: Is it OK to enforce single precision ?
        if self.time.is_compatible_index(t):
            return self.time[t]
        else:
            assert self.time.is_compatible_value(t)
            # `t` is already a time value -> just return it
            return t

    get_time.__doc__ = History.get_time.__doc__

    ###################
    # Methods which modify the model

    def lock(self):
        for hist in self.history_set:
            hist.lock()
    def clear(self):
        shim.reset_updates()
        for hist in self.unlocked_histories:
            hist.clear()

    def eval(self, max_cost :Optional[int]=None, if_too_costly :str='raise'):
        """
        Parameters
        ----------
        max_cost: int | None (default: None)
            Passed on to :func:`theano_shim.graph.eval`. This is a heuristic
            to guard againts accidentally expensive function compilations.
            Value corresponds to the maximum number of nodes in the
            computational graph. With ``None``, any graph is evaluated.
            The cost is evaluated per history.

        if_too_costly: 'raise' | 'ignore'
            Passed on to :func:`theano_shim.graph.eval`.
            What to do if `max_cost` is exceeded.

        Remove all symbolic dependencies by evaluating all ongoing updates.
        If the update is present in `shim`'s update dictionary, it's removed
        from there.

        Returns
        -------
        None
            Updates are done in place.

        **Side-effects**
            Removes updates from :attr:`theano_shim.config.symbolic_updates`.

        .. Todo:: Currently each symbolic variable is compiled and evaluated
           separately with shim.eval(). Wouldn't it be better to compile a
           single update function ?
        """
        for h in self.history_set:
            h.eval(max_cost, if_too_costly)
        # # Get the updates applied to the histories
        # tidx_updates = {h._num_tidx: (h, h._sym_tidx)
        #                 for h in self.history_set
        #                 if h._num_tidx is not h._sym_tidx}
        # data_updates = {h._num_data: (h, h._sym_data)
        #                 for h in self.history_set
        #                 if h._num_data is not h._sym_data}
        # updates = OrderedDict( (k, v[1])
        #                        for k, v in chain(tidx_updates.items(),
        #                                          data_updates.items()) )
        # # Check that there are no dependencies
        # if not shim.graph.is_computable(updates.values()):
        #     non_comp = [str(var) for var, upd in updates.items()
        #                          if not shim.graph.is_computable(upd)]
        #     raise ValueError("A model can only be `eval`ed when all updates "
        #                      "applied to its histories are computable.\n"
        #                      "The updates to the following variables have "
        #                      "symbolic dependencies: {}.".format(non_compu))
        # # Get the comp graph update dictionary
        # shimupdates = shim.get_updates()
        # for var, upd in updates.items():
        #     logger.debug("Evaluating update applied to {}.".format(var))
        #     if var in shimupdates:
        #         if shimupdates[var] == upd:
        #             logger.debug("Removing update from CG update dictionary.")
        #             del shimupdates[var]
        #         else:
        #             logger.debug("Update differs from the one in CG update "
        #                          "dictionary: leaving the latter untouched.")
        #     var.set_value(shim.eval(shim.cast(upd, var.dtype)))
        # # Update the histories
        # for orig in tidx_updates.values():
        #     h = orig[0]
        #     h._sym_tidx = h._num_tidx
        # for orig in data_updates.values():
        #     h = orig[0]
        #     h._sym_data = h._num_data
        #
        # # Ensure that we actually removed updates from the update dictionary
        # assert len(shimupdates) == len(shim.get_updates())

    def theano_state_is_clean(self):
        if shim.pending_updates():
            return False
        for hist in self.unlocked_histories:
            if (hist._num_tidx is not hist._sym_tidx
                or hist._num_data is not hist._sym_data):
                return False
        for rng in self.rng_inputs:
            if (isinstance(rng, shim.config.SymbolicRNGType)
                and len(rng.state_updates) > 0):
                return False
        return True

    def theano_reset(self, warn_rng=True):
        """
        Put model back into a clean state, to allow building a new Theano graph.
        :param warn_rng: If True (default), emit a warning if updates to a
            random number generator were cleared.

        **Side-effecs**: Clears all shim symbolic updates in shim.
        """
        shim.reset_updates()

        for hist in self.unlocked_histories:
            hist.theano_reset()
        for kernel in self.kernel_list:
            kernel.theano_reset()

        for rng in self.rng_inputs:
            # FIXME: `.state_updates` is Theano-only
            if (isinstance(rng, shim.config.SymbolicRNGType)
                and len(rng.state_updates) > 0 and warn_rng):
                rng_name = getattr(rng, 'name', str(rng))
                if rng_name is None: rng_name = str(rng)
                warn("Erasing random number generator updates. Any "
                     "other graphs using this generator are likely "
                     "invalidated.\n"
                     "RNG: {}".format(rng))
            rng.state_updates = []

    def update_params(self, new_params, **kwargs):
        """
        Update model parameters. Clears all histories except those whose `locked`
        attribute is True, as well as any kernel which depends on these parameters.

        TODO: Make `new_params` a dict and just update parameters in the dict.

        Parameters
        ----------
        new_params: self.Parameters | dict
            New parameter values.
        **kwargs:
            Alternative to specifying parameters with `new_params`
            Keyword arguments take precedence over values in `new_params`.
        """
        if isinstance(new_params, self.Parameters):
            new_params = new_params.dict()
        if len(kwargs) > 0:
            new_params = {**new_params, **kwargs}
        pending_params = self.Parameters.parse_obj(
            {**self.params.dict(), **new_params})
            # Calling `Parameters` validates all new parameters
            # Wait until kernels have been cached before updating model params

        # We don't need to clear the advance function if all new parameters
        # are just the same Theano objects with new values.
        clear_advance_fn = any(id(getattr(self.params, p)) != id(newp)
                               for p, newp in pending_params.dict().items())

        # def gettype(param):
        #     return type(param.get_value()) if shim.isshared(param) else type(param)
        # if isinstance(new_params, self.Parameters):
        #     assert(all( gettype(param) == gettype(new_param)
        #                 for param, new_param in zip(self.params, new_params) ))
        # elif isinstance(new_params, dict):
        #     assert(all( gettype(val) == gettype(getattr(self.params, name))
        #                 for name, val in new_params.items() ))
        # else:
        #     raise NotImplementedError

        # # HACK Make sure sinn.inputs and models.history_inputs coincide
        # sinn.inputs.union(self.history_inputs)
        # self.history_inputs.union(sinn.inputs)

        # Determine the kernels for which parameters have changed
        kernels_to_update = []
        for kernel in self.kernel_list:
            if set(kernel.__fields__) & set(new_params.keys()):
                kernels_to_update.append(kernel)

        # if isinstance(new_params, self.Parameters):
        #     for kernel in self.kernel_list:
        #         if not sinn.params_are_equal(
        #                 kernel.get_parameter_subset(new_params), kernel.params):
        #             # Grab the subset of the new parameters relevant to this kernel,
        #             # and compare to the kernel's current parameters. If any of
        #             # them differ, add the kernel to the list of kernels to update.
        #             kernels_to_update.append(kernel)
        # else:
        #     assert(isinstance(new_params, dict))
        #     for kernel in self.kernel_list:
        #         if any(param_name in kernel.Parameters._fields
        #                for param_name in new_params):
        #             kernels_to_update.append(kernel)

        # # Now update parameters. This must be done after the check above,
        # # because Theano parameters automatically propagate to the kernels.
        # sinn.set_parameters(self.params, new_params)

        # Loop over the list of kernels and do the following:
        # - Remove any cached binary op that involves a kernel whose parameters
        #   have changed (And write it to disk for later retrieval if these
        #   parameters are reused.)
        # Once this is done, go through the list of kernels to update and
        # update them
        for obj in self.kernel_list:
            if obj not in kernels_to_update:  # Updated kernels cached below
                for op in obj.cached_ops:
                    for kernel in kernels_to_update:
                        if hash(kernel) in op.cache:
                            diskcache.save(op.cache[hash(kernel)])
                            # TODO subclass op[other] and define __hash__
                            logger.monitor("Removing cache for binary op {} ({},{}) from heap."
                                        .format(str(op), obj.name, kernel.name))
                            del op.cache[hash(kernel)]

        for kernel in kernels_to_update:
            diskcache.save(kernel)
            kernel.update_params(**pending_params.dict())

        # Update self.params in place; TODO: there's likely a cleaner way
        for nm in pending_params.__fields__:
            setattr(self.params, getattr(pending_params, nm))
        logger.monitor("Model params are now {}.".format(self.params))

        self.clear_unlocked_histories()
        if clear_advance_fn:
            # Only clear advance function when necessary, since it forces a
            # recompilation of the graph.
            # We need to do this if any of the parameters change identity (e.g.
            # replaced by another shared variable).
            self._compiled_advance_fns.clear()

    def clear_unlocked_histories(self):
        """Clear all histories that have not been explicitly locked."""
        #for hist in self.history_inputs.union(sinn.inputs):
        for hist in self.history_inputs:
            # HACK: Removal of sinn.inputs is a more drastic version attempt
            #       at correcting the same problem as fsgif.remove_other_histories
            if not hist.locked:
                self.clear_history(hist)

    # def apply_updates(self, update_dict):
    #     """
    #     Theano functions which produce updates (like scan) naturally will not
    #     update the history data structures. This method applies those updates
    #     by replacing the internal _sym_data and _sym_tidx attributes of the history
    #     with the symbolic expression of the updates, allowing histories to be
    #     used in subsequent calculations.
    #
    #     .. Note:: The necessity of this function is unclear, given the improved
    #        compilation in sinn v0.2.
    #     """
    #     # Update the history data
    #     for history in self.history_set:
    #         if history._num_tidx in update_dict:
    #             assert(history._num_data in update_dict)
    #                 # If you are changing tidx, then surely you must change _sym_data as well
    #             object.__setattr__(history, '_sym_tidx', update_dict[history._num_tidx])
    #             object.__setattr__(history, '_sym_data', update_dict[history._num_data])
    #         elif history._num_data in update_dict:
    #             object.__setattr__(history, '_sym_data', update_dict[history._num_data])
    #
    #     # Update the shim update dictionary
    #     shim.add_updates(update_dict)

    def eval_updates(self, givens=None):
        """
        Compile and evaluate a function evaluating the `shim` update
        dictionary. Histories' internal _sym_data and _sym_tidx are reset
        to be equal to _num_tidx and _num_data.
        If the updates have symbolic inputs, provide values for them through
        the `givens` argument.
        If there are no updates, no function is compiled, so you can use this
        as a safeguard at the top of a function to ensure there are no
        unapplied updates, without worrying about the cost of repeated calls.
        """
        upds = shim.get_updates()
        if len(upds) > 0:
            f = shim.graph.compile([], [], updates=upds, givens=givens)
            f()
            for h in self.history_set:
                if h._sym_tidx != h._num_tidx:
                    object.__setattr__(h, '_sym_tidx', h._num_tidx)
                if h._sym_data != h._num_data:
                    object.__setattr__(h, '_sym_data', h._num_data)

    # def get_loglikelihood(self, *args, **kwargs):
    #
    #     # Sanity check – it's easy to forget to clear histories in an interactive session
    #     uncleared_histories = []
    #     # HACK Shouldn't need to combine sinn.inputs
    #     # TODO Make separate function, so that it can be called within loglikelihood instead
    #     for hist in self.history_inputs.union(sinn.inputs):
    #         if ( not hist.locked and ( ( hist.use_theano and hist.compiled_history is not None
    #                                      and hist.compiled_history._sym_tidx.get_value() >= hist.t0idx )
    #                                    or (not hist.use_theano and hist._sym_tidx.get_value() >= hist.t0idx) ) ):
    #             uncleared_histories.append(hist)
    #     if len(uncleared_histories) > 0:
    #         raise RuntimeError("You are trying to produce a cost function graph, but have "
    #                            "uncleared histories. Either lock them (with their .lock() "
    #                            "method) or clear them (with their individual .clear() method "
    #                            "or the model's .clear_unlocked_histories() method). The latter "
    #                            "will delete data.\nUncleared histories: "
    #                            + str([hist.name for hist in uncleared_histories]))
    #
    #     if sinn.config.use_theano():
    #         # TODO Precompile function
    #         def likelihood_f(model):
    #             if 'loglikelihood' not in self.compiled:
    #                 self.theano()
    #                     # Make clean slate (in particular, clear the list of inputs)
    #                 logL = model.loglikelihood(*args, **kwargs)
    #                     # Calling logL sets the sinn.inputs, which we need
    #                     # before calling get_input_list
    #                 # DEBUG
    #                 # with open("logL_graph", 'w') as f:
    #                 #     theano.printing.debugprint(logL, file=f)
    #                 input_list, input_vals = self.get_input_list()
    #                 self.compiled['loglikelihood'] = {
    #                     'function': theano.function(input_list, logL,
    #                                                 on_unused_input='warn'),
    #                     'inputs'  : input_vals }
    #                 self.theano_reset()
    #
    #             return self.compiled['loglikelihood']['function'](
    #                 *self.compiled['loglikelihood']['inputs'] )
    #                 # * is there to expand the list of inputs
    #     else:
    #         def likelihood_f(model):
    #             return model.loglikelihood(*args, **kwargs)
    #     return likelihood_f

    # ==============================================
    # Model advancing code
    #
    # This code isn't 100% generic yet;
    # look for TODO tags for model-specific hacks
    #
    # Function overview:  (NEEDS UPDATE)
    # - advance(self, stop): User-facing function
    # - _advance(self): Returns a function; use as `self._advance(stop)`:
    #   `self._advance` is a property which memoizes the compiled function.
    # - compile_advance_function(self): Function called by `_advance` the first
    #   time to do the compilation. Could conceivably also be used by a user.
    #   Returns a compiled function.
    # - advance_updates(self, stoptidx): Function used by
    #   `compile_advance_function` to retrieve the set of symbolic updates.
    # ==============================================
    def get_state(self, tidx=None):
        """
        Return a State object corresponding to the state at time `tidx`
        If no tidx is given, uses `self.cur_tidx` to return the current state
        TODO: Add support for >1 lags.
        """
        ti = self.cur_tidx
        return self.State(*(h[ti-self.t0idx+h.t0idx] for h in self.statehists))

    def advance(self,
                upto: Union[int, float],
                histories: Union[Tuple[History],History]=()):
        """
        Advance (i.e. integrate) a model.
        For a non-symbolic model the usual recursion is used – it's the
        same as calling `hist[end]` on each history in the model.
        For a symbolic model, the function constructs the symbolic update
        function, compiles it, and then evaluates it with `end` as argument.
        The update function is compiled only once, so subsequent calls to
        `advance` are much faster and benefit from the acceleration of running
        on compiled code.

        Parameters
        ----------
        upto: int, float
            Compute history up to this point (inclusive).
            May also be the string 'end'
        histories: Tuple of histories to integrate. State histories do not need to
            be included; they are added automatically.
            If only one history is specified, it doesn't need to be wrapped in
            a tuple.
        """
        end = upto
        if isinstance(histories, History):
            histories = (histories,)
        # Remove any locked histories
        if any(h.locked for h in histories):
            locked_hists = tuple
            warn("You requested to integrate the following histories, but they "
                 f"are locked: {[h.name for h in histories if h.locked]}.")
            histories = tuple(h for h in histories if not h.locked)
        # Remove redundant histories, so cache keys are consistent
        histories = tuple(h for h in histories if h not in self.statehists)


        # TODO: Rename endtidx -> endidx
        if end == 'end':
            endtidx = self.tnidx
        else:
            endtidx = self.get_tidx(end)

        # Make sure we don't go beyond given data
        for hist in self.history_set:
            if hist.locked:
                tnidx = hist._num_tidx.get_value()
                if tnidx < endtidx.convert(hist.time):
                    endtidx = hist.time.Index(tnidx).convert(self.time)
                    if not endtidx.in_bounds:
                        assert endtidx < hist.time.t0idx  # I don't see how we could exceed the upper bound
                        warn("History '{}' was locked before being computed. "
                             "Integration aborted.".format(hist.name))
                    else:
                        warn("Locked history '{}' is only provided "
                             "up to t={}. Output will be truncated."
                             .format(hist.name, self.get_time(endtidx)))

        if not shim.config.use_theano:
            for hist in self.statehists:
                hist._compute_up_to(endtidx.convert(hist.time))
            for hist in histories:
                hist._compute_up_to(endtidx.convert(hist.time))

        else:
            if not shim.graph.is_computable(
                [hist._sym_tidx for hist in self.statehists]):
                raise TypeError("Advancing models is only implemented for "
                                "histories with a computable current time "
                                "index (i.e. the value of `hist._sym_tidx` "
                                "must only depend on symbolic constants and "
                                "shared vars).")
            # if shim.pending_updates():
            #     raise RuntimeError("There "
            #         "are symbolic inputs to the already present updates:"
            #         "\n{}.\nEither discard them with `theano_reset()` or "
            #         "evaluate them with `eval_updates` (providing values "
            #         "with the `givens` argument) before advancing the model."
            #         .format(shim.graph.inputs(shim.get_updates().values())))
            curtidx = self.cur_tidx
            assert(curtidx >= -1)

            if curtidx < endtidx:
                self._advance(histories)(curtidx, endtidx+1)
                # _advance applies the updates, so should get rid of them
                self.theano_reset()
    integrate = advance

    @property
    def no_updates(self):
        """
        Return `True` if none of the model's histories have unevaluated
        symbolic updates.

        Deprecated ? Should be replaceable by `theano_shim.pending_updates()`.
        """
        no_updates = all(h._sym_tidx is h._num_tidx
                         and h._sym_data is h._num_data
                         for h in self.history_set)
        if no_updates and len(shim.get_updates()) > 0:
            raise RuntimeError(
                "Unconsistent state: there are symbolic theano updates "
                " (`shim.get_updates()`) but none of the model's histories "
                "has a symbolic update.")
        elif not no_updates and len(shim.get_updates()) == 0:
            hlist = {h.name: (h._sym_tidx, h._sym_data) for h in self.history_set
                     if h._sym_tidx is not h._num_tidx
                        and h._sym_data is not h._num_data}
            raise RuntimeError(
                "Unconsistent state: some histories have a symbolic update "
                "({}), but there are none in the update dictionary "
                "(`shim.get_updates()`)".forma(hlist))
        return no_updates

    def _advance(self, histories=()):
        """
        Attribute which memoizes the compilation of the advance function.

        Parameters
        ----------
        histories: Set of histories to update. State histories do not need to
            be included; they are added automatically.
        """
        if histories not in self._advance_updates:
            self._advance_updates[histories] = self.get_advance_updates(histories)
        _advance_updates = self._advance_updates[histories]
            # DEBUG
            # for i, s in enumerate(['base', 'value', 'start', 'stop']):
            #     self._advance_updates[self.V._num_data].owner.inputs[i] = \
            #         shim.print(self._advance_updates[self.V._num_data]
            #                    .owner.inputs[i], s + ' V')
            #     self._advance_updates[self.n._num_data].owner.inputs[i] = \
            #         shim.print(self._advance_updates[self.n._num_data]
            #                    .owner.inputs[i], s + ' n')
        if self.no_updates:
            if histories not in self._compiled_advance_fns:
                logger.info("Compiling the update function")
                self._compiled_advance_fns[histories] = self.cached_compile(
                    [self.curtidx_var, self.stoptidx_var], [], _advance_updates)
                logger.info("Done.")
            _advance_fn = self._compiled_advance_fns[histories]
        else:
            # TODO: Use _compiled_advance_fns to cache these compilations
            # We would need to cache the compilation for each different
            # set of symbolic updates.
            if histories != ():
                raise NotImplementedError(
                    "Not hard to implement; I am just waiting for a case where this is needed.")
            advance_updates = OrderedDict(
                (var, shim.graph.clone(upd, replace=shim.get_updates()))
                for var, upd in _advance_updates.items())

            logger.info("Compiling the update function")
            _advance_fn = self.cached_compile(
                [self.curtidx_var, self.stoptidx_var], [], advance_updates)
            logger.info("Done.")

        return lambda curtidx, stoptidx: _advance_fn(curtidx, stoptidx)

    def get_advance_updates(self, histories=()):
        """
        Returns a 'blank' update dictionary. Update graphs do not include
        any dependencies from the current state, such as symbolic/transformed
        initial conditions.

        Parameters
        ----------
        histories: Set of histories to update. State histories do not need to
            be included; they are added automatically.
        """
        cur_tidx = self.cur_tidx

        logger.info("Constructing the update graph.")
        # Stash current symbolic updates
        for h in self.statehists:
            h.stash()  # Stash unfinished symbolic updates
        updates_stash = shim.get_updates()
        shim.reset_updates()

        # Get advance updates
        updates = self.advance_updates(self.curtidx_var, self.stoptidx_var, histories)
        # Reset symbolic updates to their previous state
        shim.reset_updates()
        self.theano_reset(warn_rng=False)
            # theano_reset() is half redundant with reset_updates(), but we
            # still need to reset the RNG updates
        for h in self.statehists:
            h.stash.pop()
        shim.config.symbolic_updates = updates_stash
        logger.info("Done.")
        return updates

    def cached_compile(self, inputs, outputs, updates, **kwargs):
        """
        A wrapper around `shim.graph.compile` which caches the result to disk
        and retrieves it when possible.

        .. Note:: Although all arguments to `shim.graph.compile` are supported,
           caching is disabled when arguments other than `inputs`, `outputs`
           and `updates` are specified.
           With additional development and testing, it should be possible to
           support other arguments.
        """

        if kwargs:
            warn("Compilation caching is disabled when keyword arguments other "
                 "than `inputs`, `outputs` and `updates` are specified.")
            fn = None
            cache = False
        else:
            fn = self.compile_cache.get(outputs, updates, rng=self.rng_inputs)
            cache = True
        if fn is None:
            fn = shim.graph.compile([self.curtidx_var, self.stoptidx_var],
                                    outputs,
                                    updates = updates, **kwargs)
            if cache:
                self.compile_cache.set(outputs, updates, fn, rng=self.rng_inputs)
        else:
            logger.info("Compiled advance function loaded from cache.")
        return fn

    def advance_updates(self, curtidx, stoptidx, histories=()):
        """
        Compute model updates from curtidx to stoptidx.

        Parameters
        ----------
        curtidx: symbolic (int):
            We want to compute the model starting from this point exclusive.
            (`curtidx` is the latest already computed point, so we start at
            ``curtidx + 1``)
        stoptidx: symbolic (int)
            We want to compute the model up to this point exclusive.
        histories: Set of histories to update. State histories do not need to
            be included; they are added automatically.

        Returns
        -------
        Update dictionary:
            Compiling a function and providing this dictionary as 'updates' will return a function
            which fills in the histories up to `stoptidx`.

        .. rubric:: For developers
           You can find a documented explanation of the function's algorithm
           in the internal documentation: :doc:`/docs/internal/Building an integration graph.ipynb`.
        """

        histories += tuple(h for h in self.unlocked_statehists if h not in histories)
        # Build the update dictionary by computing each history forward by one step.
        for h in histories:
            h(h._num_tidx+1)
            # The way we do this, at time point k, we evaluate k+1
            # => Start iterations at the _current_ k, and stop one early
            # => curtidx is included below, and we do stoptidx-1
        return self.convert_point_updates_to_scan(shim.get_updates(), curtidx, stoptidx-1)

    def convert_point_updates_to_scan(self, updates=None, curtidx=None, stoptidx=None):
        """
        Convert the current point-wise updates in the global shim update
        dictionary, and convert them to an scan graph with start and end given
        by ``curtidx + 1`` and `stoptidx`. The result is thus unanchored
        from the possible `_num_tidx` dependencies in the updates.

        .. Important:: This function will fail if the updates dictionary is
           empty. Moreover, these updates must depend on
           at least one history time index which can be related to the model's
           time index.

        Parameters
        ----------
        updates: dict (update dictionary)
            Default: `shim.get_updates()`
            A Theano update dictionary. The updates should correspond to one
            time point update from ``self.num_tidx`` to ``self.num_tidx+1``.
            This function will then iterate over these updates from
            `curtidx` to `stoptidx`.
        curtidx: symbolic (int):
            Default: `self.curtidx_var`
            We want to compute the model starting from this point inclusive.
        stoptidx: symbolic (int)
            Default: `self.stoptidx_var`
            We want to compute the model up to this point exclusive.

        Returns
        -------
        Update dictionary:
            Compiling a function and providing this dictionary as 'updates' will return a function
            which fills in the histories up to `stoptidx`.

        Raises
        ------
        TypeError:
            - If `curtidx` or `stoptidx` cannot be casted to the index type of
              the histories.
        NotImplementedError:
            - If there are no unlocked histories.
        RuntimeError:
            - If the update dictionary is empty.
            - If the update dictionary has no dependency on any time index.
              (Or all such dependencies are to locked histories.)

        .. rubric:: For developers
           You can find a documented explanation of the function's algorithm
           in the internal documentation: :doc:`/docs/internal/Building an integration graph.ipynb`.
        """
        # TODO: Replace curtidx by startidx everywhere appropriate
        # Default values
        if updates  is None: updates = shim.get_updates()
        if curtidx  is None: curtidx = self.curtidx_var
        if stoptidx is None: stoptidx = self.stoptidx_var

        if not all(np.can_cast(stoptidx.dtype, hist.tidx_dtype)
                   for hist in self.statehists):
            raise TypeError("`stoptidx` cannot be safely cast to a time index. "
                            "This can happen if e.g. a history uses `int32` for "
                            "its time indices while `stoptidx` is `int64`.")

        if len(list(self.unlocked_histories)) == 0:
            pass
            # raise NotImplementedError(
            #     "Cannot build a function iterating over time points if there "
            #     "are no unlocked histories.")
        try:
            assert( shim.get_test_value(curtidx) >= -1 )
                # Iteration starts at startidx + 1, and will break for indices < 0
        except AttributeError:
            # Unable to find test value; just skip check
            pass

        # First, declare a “anchor” time index
        anchor_tidx = self.num_tidx
        if not self.time.Index(anchor_tidx+1).in_bounds:
            raise RuntimeError(
                "In order to compute a scan function, the model must "
                "have at least one uncomputed time point.")
        # Build the substitution dictionary to convert all history time indices
        # to that of the model. This requires that histories be synchronized.
        assert self.histories_are_synchronized()
        anchor_tidx_typed = self.time.Index(anchor_tidx)  # Do only once to keep graph as clean as possible
        tidxsubs = {h._num_tidx: anchor_tidx_typed.convert(h.time)
                    for h in self.unlocked_histories}

        # Now we recover the global updates, and replace the multiple history
        # time indices by the time index of the model.
        if len(updates) == 0:
            raise RuntimeError("The updates dictionary is empty. Cannot build "
                               "a scan graph.")

        if not ( set(shim.graph.symbolic_inputs(updates.values()))
                & (set(h._num_tidx for h in self.unlocked_histories)  | set([anchor_tidx])) ):
            raise RuntimeError("The updates dictionary has no dependency on "
                               "any time index. Cannot build a scan graph.")
        anchored_updates = {k: shim.graph.clone(g, replace=tidxsubs)
                            for k,g in updates.items()}

        # Now we build the step function that will be passed to `scan`.
        def onestep(tidx):
            step_updates = OrderedDict(
                (k, shim.graph.clone(g, replace={anchor_tidx: tidx}))
                for k,g in anchored_updates.items())
            return [], step_updates

        # Check that there are no missing inputs: Theano's error messages
        # for this are nasty. We want to catch the mistake first and print
        # better information
        symbinputs = shim.graph.pure_symbolic_inputs(anchored_updates.values())
        if symbinputs:
            raise shim.graph.MissingInputError(
                "The following purely symbolic variables are present in the "
                "computational graph. They were probably added accidentally; "
                "the only supported symbolic variables are shared variables.\n"
                f"Pure symbolic inputs: {symbinputs}.")

        # Now we can construct the scan graph.
        # We discard outputs since everything is in the updates
        _, upds = shim.scan(onestep,
                           sequences = [shim.arange(curtidx, stoptidx,
                                                    dtype=self.tidx_dtype)],
#                           outputs_info = [curtidx],
                           name = f"scan ({type(self).__name__})")

        return upds

    # ---------------------------------------------
    # Helper decorators for building cost functions

    def accumulate_with_offset(self, start_offset):
        """
        .. Note:: In most cases, it will is easier to use one of the helper functions,
           either `accumulate` or `static_accumulate`.

        Construct a cost graph from a pointwise cost function.
        A function is “pointwise” if it can be computed entirely from the histories
        (i.e. it does not depend on its own value at other time points).

        For a cost function ``f``, the resulting graph corresponds to

        .. math::

            \sum_{i=t0}^tn f(i)

        where *t0* and *tn* are respectively ``curtidx+start_offset``
        and ``curtidx+start_offset+batch_size``.

        The returned function r takes two arguments, ``curtidx`` and ``batch_size``

        **Side-effects**
            State updates are added to the global update dictionary.

        Parameters
        ----------
        start_offset: Axis index delta | int
            Index offset relative to the current time index.
            There are mainly two intended values:
              - 1: Evaluate one time index forward, relative to the current time.
                   This will trigger all required history updates to integrate
                   the model.
                   Equivalent to `accumulate`.
              - 0: Evaluate at the current time index. This should not trigger
                   any history updates. This means dependencies on parameters are
                   likely lost.
                   Equivalent to `static_accumulate`.
        """
        def wrapped_acc(f):
            def accumulated_function(curtidx, batchsize):
                # Ensuring this works correctly with shared variables would require more testing
                assert shim.is_pure_symbolic(curtidx) and shim.is_pure_symbolic(batchsize)
                # TODO: Make cost a symbolic variable, and pass it to scan as `outputs_info`
                # time index anchor
                numtidx = self.num_tidx  # Raises RuntimeError if errors are not synchronized
                if not self.time.Index(numtidx+start_offset).in_bounds:
                    raise RuntimeError(
                        "In order to compute a scan function, the model must "
                        f"have at least {start_offset} uncomputed time point.")
                # Accumulator variable for the cost. Initialized to zero.
                cost = shim.shared(np.array(0, dtype='float64'), f"accumulator ({f.__name__})")
                # Build the computational graph for the step update of the cost
                shim.add_update(cost, cost + f(self.time.Index(numtidx+start_offset)))
                    # f(…) triggers required history update computations
                # Convert the step update to an iteration
                updates = self.convert_point_updates_to_scan(
                    shim.get_updates(), curtidx, curtidx+batchsize)
                # Return the final cost, along with the rest of the updates
                # Caller can decide if they want to apply updates or discard them
                cost_total = updates.pop(cost)
                shim.add_updates(updates)
                return cost_total, shim.get_updates()
            accumulated_function.__name__ = f"accumulated_{f.__name__}"
            accumulated_function.__doc__ = ("This function accumulates (sums) the values "
                                            f"of the function `{f.__name__}` from "
                                            f"`curtidx+{start_offset}` to `curtidx+{start_offset}+stoptidx`.\n\n"
                                            "--------------------------------------------\n"
                                            f"Docstring for {f.__name__}:\n\n{f.__doc__}")
            return accumulated_function
        return wrapped_acc

    def accumulate(self, f):
        """
        Accumulate (sum) the function f. Histories are integrated along
        with the accumulator.

        If you need a differentiable cost function, this is almost always
        the decorator to use.

        Intended uses:
            - Training a model with back propagation through time.

        Example
        -------
        Mean-squared error between a target `y` and the variable `x` of a model
        `model`.
        Note the use of round brackets (`model.x(tidx)`) to ensure that the
        update computations for `x` are triggered.
        >>> model = Model(...)
        >>> y = np.array(...)
        >>> @model.accumulate
        >>> def mse(tidx):
        >>>     return (y[tidx] - model.x(tidx))**2
        """
        acc_f = self.accumulate_with_offset(1)(f)
        acc_f._accumulator = 'accumulate'
        return acc_f

    def static_accumulate(self, f):
        """
        Accumulate (sum) the function f without updating histories.
        In `f`, use [] indexing to make sure you don't accidentally
        trigger computations.

        .. Warning:: Since update computations are not triggered,
           any dependencies of those computations on parameters
           will not show up in accumulator's graph.

        Intended for:
            - Evaluating a function on an already computed history.
            - Optimizating the evolution of a latent variable.

        *Not* intended for:
            - Training the model dynamics.
        """
        acc_f = self.accumulate_with_offset(0)(f)
        acc_f._accumulator = 'static_accumulate'
        return acc_f

class BinomialLogpMixin:
    """
    Mixin class for models with binomial likelihoods.

    .. Note::
       Not yet tested with v0.2.
    """
    def make_binomial_loglikelihood(self, n, N, p, approx=None):
        """
        Parameters
        ----------
        n: History
            Number of successful samples
        N: array of ints:
            Total number of samples.
            Must have n.shape == N.shape
        p: History
            Probability of success; first dimension is time.
            Must have n.shape == p.shape and len(n) == len(p).
        approx: str
            (Optional) If specified, one of:
            - 'low p': The probability is always very low; the Stirling approximation
              is used for the contribution from (1-p) to ensure numerical stability
            - 'high p': The probability is always very high; the Stirling approximation
              is used for the contribution from (p) to ensure numerical stability
            - 'Stirling': Use the Stirling approximation for both the contribution from
              (p) and (1-p) to ensure numerical stability.
            - 'low n' or `None`: Don't use any approximation. Make sure `n` is really low
              (n < 20) to avoid numerical issues.
        """

        def loglikelihood(start=None, stop=None):

            hist_type_msg = ("To compute the loglikelihood, you need to use a NumPy "
                             "history for the {}, or compile the history beforehand.")
            if n.use_theano:
                if n.compiled_history is None:
                    raise RuntimeError(hist_type_msg.format("events"))
                else:
                    nhist = n.compiled_history
            else:
                nhist = n

            phist = p
            # We deliberately use times here (instead of indices) for start/
            # stop so that they remain consistent across different histories
            if start is None:
                start = nhist.t0
            else:
                start = nhist.get_time(start)
            if stop is None:
                stop = nhist.tn
            else:
                stop = nhist.get_time(stop)

            n_arr_floats = nhist[start:stop]
            p_arr = phist[start:stop]

            # FIXME: This would break the Theano graph, no ?
            if shim.isshared(n_arr_floats):
                n_arr_floats = n_arr_floats.get_value()
            if shim.isshared(p_arr):
                p_arr = p_arr.get_value()

            p_arr = sinn.clip_probabilities(p_arr)

            if not shim.is_theano_object(n_arr_floats):
                assert(sinn.ismultiple(n_arr_floats, 1).all())
            n_arr = shim.cast(n_arr_floats, 'int32')

            #loglikelihood: -log n! - log (N-n)! + n log p + (N-n) log (1-p) + cst

            if approx == 'low p':
                # We use the Stirling approximation for the second log
                l = shim.sum( -shim.log(shim.factorial(n_arr, exact=False))
                              -(N-n_arr)*shim.log(N - n_arr) + N-n_arr + n_arr*shim.log(p_arr)
                              + (N-n_arr)*shim.log(1-p_arr) )
                    # with exact=True, factorial is computed only once for whole array
                    # but n_arr must not contain any elements greater than 20, as
                    # 21! > int64 (NumPy is then forced to cast to 'object', which
                    # does not play nice with numerical ops)
            else:
                raise NotImplementedError

            return l

        return loglikelihood

class PyMC3Model(Model, abc.ABC):
    """
    A specialized model that is compatible with PyMC3.

    .. Note:
       Still needs to be ported to v0.2.
    """

    @property
    def pymc(self, **kwargs):
        """
        The first access should be done as `model.pymc()` to instantiate the
        model. Subsequent access, which retrieves the already instantiated
        model, should use `model.pymc`.
        """
        if getattr(self, '_pymc', None) is None:
            # Don't require that PyMC3 be installed unless we need it
            import sinn.models.pymc3
            # Store kwargs so we can throw a warning if they change
            self._pymc_kwargs = copy.deepcopy(kwargs)
            # PyMC3ModelWrapper assigns the PyMC3 model to `self._pymc`
            return sinn.models.pymc3.PyMC3ModelWrapper(self, **kwargs)
        else:
            for key, val in kwargs.items():
                if key not in self._pymc_kwargs:
                    logger.warning(
                        "Keyword parameter '{}' was not used when creating the "
                        "model and will be ignored".format(key))
                elif self._pymc_kwargs[key] != value:
                    logger.warning(
                        "Keyword parameter '{}' had a different value ({}) "
                        "when creating the model.".format(key, value))
        return self._pymc
