# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René

DIFFERENCE WITH models.py IN THE MAIN CODE BASE:
    In the `advance_updates` method, this module builds `scan` differently.
    Instead of putting everything in shared variables, the history updates
    are accumulated in the `outputs_info` argument (the "proper" Theano way).
    This makes for a much more complicated algorithm, possibly less flexible
    and robust, but also possibly more efficient with Theano.
"""

# TODO
#   - Remove any use of '_refhist'. Package the (t0, tn, dt) -> time_array
#     code (which is in histories.History) into an Axis class, and then give
#     a proper axis to Model.
#   - Have one dictionary/SimpleNamespace storing all compilation variables.
#     See comment under `class Model`

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
from typing import Any, Optional
from inspect import signature

import theano_shim as shim
import mackelab_toolbox.utils as utils
import mackelab_toolbox.theano
from mackelab_toolbox.theano import GraphCache, CompiledGraphCache
from mackelab_toolbox.utils import class_or_instance_method
from mackelab_toolbox.cgshim import typing as cgtyping

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

def make_placeholder(history, name_suffix=' placeholder'):
    """
    Return a symbolic variable representating a time slice of
    :param:history.
    TODO: Add support for >1 lags.
    """
    return shim.tensor(history.shape, history.name + name_suffix, history.dtype)

def _graph_batch_bounds(model, start, stop, batch_size):
    """ Internal function for :func:batch_function_scan. """
    start = model.get_tidx(start)
    if stop is None and batch_size is None:
        raise TypeError("Batch function requires that `start` and"
                        "one of `stop`, `batch_size` be specified.")
    elif batch_size is None:
        stop = model.get_tidx(stop)
        batch_size = stop - start
    elif stop is None:
        batch_size = model.index_interval(batch_size)
        stop = start + batch_size
    else:
        logger.warning("Both `stop` and `batch_size` were provided "
                       "to a batch function. This is probably an "
                       "error, and if it isn't, make sure that "
                       "they are consistent.")
    return start, stop, batch_size

def batch_function_scan(*inputs):
    """
    To be used as a decorator. Uses `scan` to construct the vectors
    of values constituting a batch, iterating from `start` to `stop`.

    Parameters
    ----------
    inputs: list of str
        Each string must correspond exactly to the identifier for one
        of the model's histories. (They are retrieved with `gettattr`.)
        The slice corresponding to a batch for each defined input will
        be passed to the function, in the order defined by :param:inputs.

    Example
    -------

    >>> import numpy as np
    >>> from Collections import namedtuple
    >>> from odictliteral import odict
    >>> import sinn
    >>> from sinn.histories import Series
    >>> from sinn.models import Model
    >>> class MyModel(Model):
            requires_rng = True
            Parameter_info = odict['θ': 'floatX']
            Parameters = sinn.define_parameters(Parameter_info)
            State = namedtuple()
            def __init__(self, params, random_stream=None):
                self.A = Series('A', shape=(1,), time_array=np.arange(1000))
                self.rndstream = random_stream
                super().__init__(params=params, reference_history=self.A)
                self.A.set_update_function(
                    lambda t: self.rndstream.normal(avg=self.θ))

            @batch_function_scan('A')
            def logp(A):
                # Squared error
                return ((A - self.θ)**2).sum()

    >>> θ = 0
    >>> model = MyModel(MyModel.Parameters(θ=θ))
    >>> model.A.set(np.random.normal(loc=θ)
    >>> model.logp(start=200, batch_size=500)
    >>> model.params.θ.set_value(1)
    >>> model.logp(start=200, batch_size=500)

    """
    def decorator(f):
        def wrapped_f(self, start, stop=None, batch_size=None):
            """Either :param:stop or :param:batch are required."""
            start, stop, batch_size = \
                _graph_batch_bounds(self, start, stop, batch_size)
                # Returns integer indices

            # Define a bunch of lists of histories and indices to be able to
            # permute inputs between the order of `self.State` and that defined
            # by `inputs`.
            statehists = list(self.unlocked_statehists)
            locked_statehists = list(self.locked_statehists)
            inputhists = [getattr(self, name) for name in inputs]
                # The set of histories which are included in the function
                # May include both state and non-state histories
            lockedstateinputs = [h for h in inputhists
                                   if h in locked_statehists]
            nonstateinputs  = [h for h in inputhists
                                 if h not in self.statehists]
            stateinputs     = [h for h in inputhists if h in statehists]
            stateinput_idcs = [(i, statehists.index(h))
                               for i, h in enumerate(inputhists)
                               if h in statehists]

            # Construct the initial values
            if not shim.is_theano_object(start):
                unupdatedhists = [h for h in chain(nonstateinputs, statehists)
                                 if h.cur_tidx < self.get_tidx_for(start-1, h)]
                unfilledhists = [h for h in locked_statehists
                                 if h.cur_tidx < self.get_tidx_for(stop-1, h)]
                if len(unupdatedhists) > 0:
                    raise RuntimeError(
                        "Use the `advance()` method to ensure all histories "
                        "have been integrated up to the start of the batch. "
                        "Offending histories are: {}."
                        "If you are calling `advance()`, then it may be that "
                        "the update functions for these histories are actually "
                        "disconnected for those of the model."
                        .format(', '.join([repr(h) for h in unupdatedhists])))
                if len(unfilledhists) > 0:
                    raise RuntimeError(
                        "Locked histories must already be filled up to the end "
                        "of the batch, since they cannot be updated."
                        "Offending histories are: {}"
                        .format(', '.join([repr(h) for h in unfilledhists])))
            initial_values = [h._sym_data[self.get_tidx_for(start-1, h)]
                              for h in chain(nonstateinputs, statehists)]

            if shim.cf.use_theano:
                def onestep(tidx, *args):
                    for x, name in zip(
                        utils.flatten(
                            tidx, *args, terminate=shim.cf._TerminatingTypes),
                        utils.flatten(
                            'tidx (scan)',
                            #(h.name + ' (scan)' for h in lockedstateinputs),
                            (h.name + ' (scan)' for h in nonstateinputs),
                            (h.name + ' (scan)' for h in statehists),
                            terminate=shim.cf._TerminatingTypes)):
                        if getattr(x, 'name', None) is None:
                            x.name = name
                    m = len(nonstateinputs)
                    _nonstate = args[:m]
                    _state = args[m:]
                    assert len(_state) == len(statehists)
                    _stateinputs = [_state[j] for i,j in stateinput_idcs]
                    state_outputs, updates = self.symbolic_update(tidx, *_state)
                    nonstate_outputs, nonstate_updates = self.nonstate_symbolic_update(
                        tidx, nonstateinputs, _state, _nonstate, state_outputs)
                    assert len(set(updates).intersection(nonstate_updates)) == 0
                    updates.update(nonstate_updates)
                    return nonstate_outputs + state_outputs, updates

            else:
                def onestep(tidx, *args):
                    # There are no symbolic state updates if we are using NumPy
                    return ([h[self.get_tidx_for(tidx, h)] for h in inputhists],
                            OrderedDict())

            # Accumulate over the batch
            if batch_size == 1:
                # No need for scan
                outputs, updates = onestep(start, *initial_values)
                # Add the batch dimension which scan would have created
                outputs = [o[np.newaxis,...] for o in outputs]
            else:
                outputs, updates = shim.scan(
                    onestep, sequences=shim.arange(start, stop),
                    outputs_info=initial_values,
                    return_list=True)
            assert(len(outputs) == len(nonstateinputs) + len(statehists))

            # Permute the outputs so they are in the order expected by `f`
            finputs = [None]*len(inputs)
            m = len(nonstateinputs)
            for h in lockedstateinputs:
                i = inputhists.index(h)
                _start = self.get_tidx_for(start, h)
                _stop = self.get_tidx_for(stop, h)
                finputs[i] = h._sym_data[_start:_stop]
            for h, o in zip(nonstateinputs, outputs[:m]):
                i = inputhists.index(h)
                finputs[i] = o
            for h, o in zip(statehists, outputs[m:]):
                if h in inputhists:
                    i = inputhists.index(h)
                    finputs[i] = o
            assert all(i is not None for i in finputs)

            # Evaluate `f`.
            return f(self, *finputs)
                # `f` is still a function while being decoratod, so we need
                # to explicitly pass `self`

        return wrapped_f
    return decorator

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

def initializer(*fields, unintialized=None, pre=True, always=True, **dec_kwargs):
    """
    Specialized validator for writing more complex default initializers with
    less boilerplate. Does two things:

    - Changes defaults for `pre` and `always` to ``True``.
    - Allows model parameters to be specified as keyword arguments in the
      validator signature. This works with both model-level parameters, and
      the parameters defined in the `Parameters` subclass.

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
    always (default: True)
    each_item
    check_fields
    allow_reuse: As in `pydantic.validator`, although some arguments may not
        be so relevant.

    uninitialized: Any (default: None)
        The initializer is only executed when the parameter is equal to this
        value.
    """

    val_fn = validator(*fields, pre=pre, always=always, **dec_kwargs)

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
        if len(param_args) == 0:
            # No param args => nothing to do
            new_f = f
        else:
            def new_f(cls, v, values, field, config):
                if v is not unintialized:
                    return v
                param_kwargs = {}
                params = values.get('params', None)
                if not isinstance(params, BaseModel):
                    params = None  # We must not be within a sinn Model => 'params' does not have special meaning
                for p in param_args:
                    pval = values.get(p, None)  # Try module-level param
                    if pval is None and params is not None:
                        pval = getattr(params, p, None)
                    if pval is None:
                        raise AssertionError(
                          f"'{p}' cannot be found within the model parameters. "
                          "This may be because it is defined after "
                          f"'{field.name}' in the list of parameters, or "
                          "because its own validation failed.")
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
            new_f.__doc__ = f.__doc__
            return val_fn(new_f)

    return dec

class ModelMetaclass(pydantic.main.ModelMetaclass):
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
        Parameters = namespace.get('Parameters', None)  # First check namespace
        Config = namespace.get('Config', None)
        for C in mro:
            if Parameters is not None and Config is not None:
                break  # We've found both classes; no need to look further
            if Parameters is None:
                Parameters = getattr(C, 'Parameters', None)
            if Config is None:
                Config = getattr(C, 'Config', None)

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


        # Sanity check Parameters subclass
        # Parameters = namespace.get('Parameters', None)
        if (not isinstance(Parameters, type)
            or not issubclass(Parameters, BaseModel)):
            raise TypeError(
                f"Model {cls}: `Parameters` must inherit from pydantic."
                f"BaseModel. `{cls}.Parameters.mro()`: {Parameters.mro()}.")

        # Sanity check Config subclass, if present
        # Config = namespace.get('Config', None)
        # No warnings for Config: not required
        if not isinstance(Config, type):
            raise TypeError(f"Model {cls} `Config` must be a class, "
                            f"not {type(Config)}.")

        # Add 'params' variable if it isn't present, and place first in
        # the list of variables so that initializers can find it.
        if 'params' in annotations:
            if annotations['params'] is not Parameters:
                raise TypeError(f"Model {cls} defines `params` but it is not "
                                f"of type `{cls}.Parameters`")
            new_annotations['params'] = annotations.pop('params')
        else:
            new_annotations['params'] = Parameters

        # TODO?: Allow derived classes to redefine histories ?
        #        We would just need to add the inherited kernel/hists after this loop

        # Add module-level annotations
        for nm, T in annotations.items():
            if nm in new_annotations:
                raise TypeError("Name clash in {cls} definition: '{nm}'")
            new_annotations[nm] = T
            if isinstance(T, type) and issubclass(T, History):
                _hist_identifiers.add(nm)
            elif isinstance(T, type) and issubclass(T, Kernel):
                _kernel_identifiers.add(nm)
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
            warn(f"Model {cls} does not define a set of state variables.")
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
                        "types:\n\tSeries, Spiketrain")

        # Update namespace
        namespace['_kernel_identifiers'] = list(_kernel_identifiers)
        namespace['_hist_identifiers'] = list(_hist_identifiers)
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
    __slots__ = ('graph_cache', 'compile_cache', '_pymc', 'batch_start_var', 'batch_size_var',
                 '_curtidx', '_stoptidx_var', '_advance_updates')

    class Config:
        # Allow assigning other attributes during initialization.
        extra = 'allow'
        keep_untouched = (PendingUpdateFunction, class_or_instance_method)

    class Parameters(abc.ABC, BaseModel):
        """
        Models must define a `Parameters` class within their namespace.
        Don't inherit from this class, just `BaseModel`; i.e. do::

            class MyModel(Model):
                class Parameters(BaseModel):
                    ...
        """
        pass

    def __init__(self, initializer=None, **kwargs):
        # Sanity check Parameters subclass
        Parameters = type(self).Parameters  # `self` not properly initialized yet
        assert isinstance(Parameters, type)  # Assured by metaclass
        if issubclass(Parameters, abc.ABC):
            # Most likely no Parameters class was defined, and we retrieved
            # the abstract definition above.
            raise ValueError(
                f"Model {cls} does not define a `Parameters` class, or its "
                "`Parameters` class inherits from `abc.ABC`.")
        # Initialize attributes with Pydantic
        super().__init__(**kwargs)
        # Attach update functions to histories, and set up __slots__
        self._base_initialize()
        # Run the model initializer
        self.initialize(initializer)

    def copy(self, *args, **kwargs):
        m = super().copy(*args, **kwargs)
        m._base_initialize()
        m.initialize()
        return m

    @classmethod
    def parse_obj(cls, obj):
        m = super().parse_obj(obj)
        m._base_initialize()
        m.initialize()
        return m

    def _base_initialize(self):
        """
        Collects initialization that should be done in __init__, copy & parse_obj.
        """
        # Attach history updates
        HistoryUpdateFunction.namespace = self
        for obj in self._pending_update_functions:
            hist = getattr(self, obj.hist_nm)
            hist.update_function = HistoryUpdateFunction(
                func = partial(obj.upd_fn, self),  # Wrap history so `self` points to the model
                inputs = obj.inputs,
            )
        object.__setattr__(self, 'graph_cache',
                           GraphCache('sinn.models', type(self),
                                      modules=('sinn.models',)))
        object.__setattr__(self, 'compile_cache',
                           CompiledGraphCache('sinn.models.compilecache'))
            # TODO: Add other dependencies within `sinn.models` ?
        # Create symbolic variables for batches
        if shim.cf.use_theano:
            # # Any symbolic function on batches should use these, that way
            # # other functions can retrieve the symbolic input variables.
            start = np.array(1).astype(self.tidx_dtype)
            object.__setattr__(self, 'batch_start_var',
                               shim.shared(start, name='batch_start'))
            #     # Must be large enough so that test_value slices are not empty
            size = np.array(2).astype(self.tidx_dtype)
            object.__setattr__(self, 'batch_size_var',
                               shim.shared(size, name='batch_size'))
            #     # Must be large enough so that test_value slices are not empty

    @abc.abstractmethod
    def initialize(self, initializer :Any=None):
        """
        Models must define an `initialize` method. This is where you can add
        padding to histories, pre-compute kernels, etc. – anything which should
        be done whenever parameters changed.

        It takes one optional keyword argument, `initializer`, which can be of
        any form. This could be e.g. a string flag, to indicate one of multiple
        initialization protocols, or a dictionary with multiple initialization
        parameters.

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
                         if isinstance(inp, cgshim.typing.RNG)]
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

    # TODO: Some redundancy here, and we could probably store the
    # hist & kernel lists after object creation – just ensure this is
    # also done after copy() and parse_obj()
    # FIXME: what to do with histories which aren't part of model (e.g.
    #        returned from an operation between hists) ?
    @property
    def histories(self):
        return {nm: getattr(self, nm) for nm in self._hist_identifiers}
    @property
    def history_set(self):
        return {getattr(self, nm) for nm in self._hist_identifiers}
    @property
    def kernels(self):
        return {nm: getattr(self, nm) for nm in self._kernel_identifiers}
    @property
    def kernel_list(self):
        return [getattr(self, nm) for nm in self._kernel_identifiers]

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

    # def set_reference_history(self, reference_history):
    #     if (reference_history is not None
    #         and getattr(self, '_refhist', None) is not None):
    #         raise RuntimeError("Reference history for this model is already set.")
    #     self._refhist = reference_history

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

    @property
    def statehists(self):
        return utils.FixedGenerator(
            (getattr(self, varname) for varname in self.State.__annotations__),
            len(self.State.__annotations__) )

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
        all_input_names = set().union(*(set(h.update_function.input_names)
                                        for h in self.unlocked_histories))
        all_inputs = [getattr(self, nm) for nm in all_input_names]
        return [inp for inp in all_inputs
                if isinstance(inp, shim.config.RNGTypes)]

    @property
    def cur_tidx(self):
        """
        Return the earliest time index for which all histories are computed.
        """
        return min(h.cur_tidx.convert(self.time.Index)
                   for h in self.statehists)

    @property
    def _num_tidx(self):
        if not self.histories_are_synchronized():
            raise RuntimeError(
                f"Histories for the {type(self).__name__}) are not all "
                "computed up to the same point. The compilation of the "
                "model's integration function is ill-defined in this case.")
        return shim.shared(np.array(self.cur_tidx, dtype=self.tidx_dtype),
                           f"t_idx ({type(self).__name__})")

    def histories_are_synchronized(self):
        """
        Return True if all unlocked state hists are computed to the same time
        point, and all locked histories at least up to that point.
        """
        tidcs = [h.cur_tidx.convert(self.time.Index)
                 for h in self.unlocked_statehists]
        locked_tidcs = [h.cur_tidx.convert(self.time.Index)
                        for h in self.locked_statehists]
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
        if isinstance(self, type):
            nameline = "Model '{}'".format(self.__name__)
            paramline = "Parameters: " + ', '.join(self.Parameters.__annotations__)
        else:
            assert isinstance(self, Model)
            name = getattr(self, '__name__', type(self).__name__)
            nameline = "Model '{}' (t0: {}, tn: {}, dt: {})" \
                .format(name, self.t0, self.tn, self.dt)
            paramline = str(self.params)
        nameline += '\n' + '-'*len(nameline)  # Add separating line under name
        stateline = "State variables: " + ', '.join(self.State.__annotations__)
        return (nameline + '\n' + stateline + '\n' + paramline + '\n\n'
                + self.update_summary(hists))

    @class_or_instance_method
    def update_summary(self, hists=None):
        """
        Return a string summarizing the update function. By default, returns those of `self.state`.
        May be called on the class itself, or an instance.

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
        # Normalize to all strings
        names = [h.name if isinstance(h, History) else h for h in hists]
        if not all(isinstance(h, str) for h in hists):
            raise ValueError(
                "`hists` must be a list of histories or history names.")
        funcs = [self._pending_update_functions[h] for h in statehists]

        # if hists is None:
        #     if isinstance(self, Model):
        #         # Just grab the histories from self.statehists
        #         # Converted to functions below
        #         hists = self.statehists
        #     else:
        #         # Get the hist names from the State class
        #         hists = self.State.__annotations__.keys()
        #
        # # Create list of update functions and their corresponding names
        # hists = list(hists)
        # funcs = []
        # names = []
        # for i, hist in enumerate(hists):
        #     if isinstance(hist, History):
        #         funcs.append(hist._update_function_dict['func'])
        #         names.append(hist.name)
        #     elif isinstance(hist, str):
        #         try:
        #             fn = getattr(self, hist + '_fn')
        #         except AttributeError:
        #             raise AttributeError("Unable to find update function for `{}`. "
        #                                  "Naming convention expects it to be `{}`."
        #                                  .format(hist, hist + '_fn'))
        #         funcs.append(fn)
        #         names.append(hist)
        #     else:
        #         assert isintance(hist, Callable)
        #         funcs.append(hist)
        #         names.append(None)

        # For each function, retrieve its source
        srcs = []
        for name, fn in zip(names, funcs):
            src = inspect.getsource(fn)
            if src.strip()[:3] != 'def':
                raise RuntimeError(
                    "Something went wrong when retrieve an update function's source. "
                    "Make sure the source file is saved and try reloading the Jupyter "
                    "notebook. Source should start with `def`, but we got:\n" + src)
            # TODO: Remove indentation common to all lines
            if name is not None:
                # Replace the `def` line by a more explicit string
                src = "Update function for {}:".format(name) + '\n' + src.split('\n', 1)[1]
            srcs.append(src)

        # Join everything together and return
        return '\n\n'.join(srcs)

    def get_tidx(self, t, allow_rounding=False):
        # Copied from History.get_tidx
        if self.time.is_compatible_value(t):
            return self.time.index(t, allow_rounding=allow_rounding)
        else:
            assert self.time.is_compatible_index(t)
            return self.time.Index(t)
        # if self._refhist is not None:
        #     if shim.istype(t, 'int'):
        #         return t
        #     else:
        #         return self._refhist.get_tidx(t, allow_rounding) - self._refhist.t0idx + self.t0idx
        # else:
        #     raise AttributeError("The reference history for this model was not set.")
    get_tidx.__doc__ = History.get_tidx.__doc__

    def get_tidx_for(self, t, target_hist, allow_fractional=False):
        raise DeprecationWarning("Use the `convert` method attached to the AxisIndex.")
        # if self._refhist is not None:
        #     ref_tidx = self.get_tidx(t) - self.t0idx + self._refhist.t0idx
        #     return self._refhist.get_tidx_for(
        #         ref_tidx, target_hist, allow_fractional=allow_fractional)
        # else:
        #     raise AttributeError("Reference history for this model is not set.")

    def index_interval(self, Δt, allow_rounding=False):
        return self.time.index_interval(value, value2,
                                        allow_rounding=allow_rounding,
                                        cast=cast)
        # if self._refhist is not None:
        #     return self._refhist.index_interval(Δt, allow_rounding)
        # else:
        #     raise AttributeError("The reference history for this model was not set.")
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

        # if self._refhist is not None:
        #     if shim.istype(t, 'float'):
        #         return t
        #     else:
        #         assert(shim.istype(t, 'int'))
        #         tidx = t - self.t0idx + self._refhist.t0idx
        #         return self._refhist.get_time(tidx)
        # else:
        #     raise AttributeError("The reference history for this model was not set.")
    get_time.__doc__ = History.get_time.__doc__

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

    def theano_reset(self):
        """Put model back into a clean state, to allow building a new Theano graph."""
        for hist in self.unlocked_histories:
            hist.theano_reset()
        for kernel in self.kernel_list:
            kernel.theano_reset()

        for rng in self.rng_inputs:
            # FIXME: `.state_updates` is Theano-only
            if (isinstance(rng, shim.config.SymbolicRNGType)
                and len(self.rng.state_updates) > 0):
                logger.warning("Erasing random number generator updates. Any "
                               "other graphs using this generator are likely "
                               "invalidated.\n"
                               "RNG: {}".format(self.rng))
            rng.state_updates = []
        #sinn.theano_reset() # theano_reset on histories will be called twice,
                            # but there's not much harm
        shim.reset_updates()

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
            # Only clear advance functio when necessary, since it forces a
            # recompilation of the graph.
            self.clear_advance_function()

    def clear_unlocked_histories(self):
        """Clear all histories that have not been explicitly locked."""
        #for hist in self.history_inputs.union(sinn.inputs):
        for hist in self.history_inputs:
            # HACK: Removal of sinn.inputs is a more drastic version attempt
            #       at correcting the same problem as fsgif.remove_other_histories
            if not hist.locked:
                self.clear_history(hist)

    def clear_advance_function(self):
        """
        Removes the compiled advance function, if present, forcing it to be
        recompiled if called again.
        We need to do this if any of the parameters change identity (e.g.
        replaced by another shared variable).
        """
        if hasattr(self, '_advance_fn'):
            del self._advance_fn

    def clear_other_histories(self):
        """
        Clear unlocked histories that are not explicitly part of this model
        (but may be inputs).
        """
        # Implemented as a wrapper around clear_unlocked_histories:
        # first lock of this model's histories, clear histories, and then
        # revert to the original locked/unlocked status
        old_status = {hist: hist.locked for hist in self.history_set}
        for hist in self.history_set:
            if not hist.locked:
                hist.lock(warn=False)
        self.clear_unlocked_histories()
        for hist, status in old_status.items():
            if status == False:
                hist.unlock()

    def clear_history(self, history):
        # Clear the history, and remove any cached operations related to it
        # In contrast to `update_params`, we don't write these operations to
        # disk, because histories are data structures: there's no way of knowing
        # if they're equivalent to some already computed case other than comparing
        # the entire data.
        logger.monitor("Clearing history " + history.name)
        history.clear()
        #if history in self.history_inputs.union(sinn.inputs):
        if history in self.history_inputs:
            # HACK: Removal of sinn.inputs is a more drastic version attempt
            #       at correcting the same problem as fsgif.remove_other_histories
            for obj in list(self.history_inputs) + self.kernel_list:
                for op in obj.cached_ops:
                    if hash(history) in op.cache:
                        del op.cache[hash(history)]
        else:
            for obj in list(self.history_inputs) + self.kernel_list:
                for op in obj.cached_ops:
                    if hash(history) in op.cache:
                        logger.error("Uncached history {} is member of cached "
                                     "op {}. This may indicate a memory leak."
                                     .format(history.name, str(op)))

    # def remove_other_histories(self):
    #     """HACK: Remove histories from sinn.inputs that are not in this model.
    #     Can remove this once we store dependencies in histories rather than in
    #     sinn.inputs."""
    #     histnames = [h.name for h in self.history_set]
    #     dellist = []
    #     for h in sinn.inputs:
    #         if h.name not in histnames:
    #             dellist.append(h)
    #     for h in dellist:
    #         del sinn.inputs[h]

    def apply_updates(self, update_dict):
        """
        Theano functions which produce updates (like scan) naturally will not
        update the history data structures. This method applies those updates
        by replacing the internal _sym_data and _sym_tidx attributes of the history
        with the symbolic expression of the updates, allowing histories to be
        used in subsequent calculations.
        """
        # Update the history data
        for history in self.history_set:
            if history._num_tidx in update_dict:
                assert(history._num_data in update_dict)
                    # If you are changing tidx, then surely you must change _sym_data as well
                object.__setattr__(history, '_sym_tidx', update_dict[history._num_tidx])
                object.__setattr__(history, '_sym_data', update_dict[history._num_data])
            elif history._num_data in update_dict:
                object.__setattr__(history, '_sym_data', update_dict[history._num_data])

        # Update the shim update dictionary
        shim.add_updates(update_dict)

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
                    h._sym_tidx = h._num_tidx
                if h._sym_data != h._num_data:
                    h._sym_data = h._num_data

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
    # Function overview:
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

    def get_state_placeholder(self, name_suffix=' placeholder'):
        """
        Return a State object populated with symbolic placeholder variables.
        TODO: Add support for >1 lags.
        """
        return self.State(*(make_placeholder(h, name_suffix)
                            for h in self.statehists))

    def advance(self, stop):
        """
        Advance (i.e. integrate) a model.
        For a non-symbolic model the usual recursion is used – it's the
        same as calling `hist[stop]` on each history in the model.
        For a symbolic model, the function constructs the symbolic update
        function, compiles it, and then evaluates it with `stop` as argument.
        The update function is compiled only once, so subsequent calls to
        `advance` are much faster and benefit from the acceleration of running
        on compiled code.

        Parameters
        ----------
        stop: int, float
            Compute history up to this point (inclusive).
        """

        # TODO: Rename stoptidx -> endidx
        if stop == 'end':
            stoptidx = self.tnidx
        else:
            stoptidx = self.get_tidx(stop)

        # Make sure we don't go beyond given data
        for hist in self.history_set:
            if hist.locked:
                tnidx = hist._num_tidx.get_value()
                if tnidx < stoptidx.convert(hist.time):
                    stoptidx = tnidx.convert(self.time)
                    logger.warning("Locked history '{}' is only provided "
                                   "up to t={}. Output will be truncated."
                                   .format(hist.name, self.get_time(stoptidx)))

        if not shim.config.use_theano:
            for hist in self.statehists:
                hist._compute_up_to(stoptidx.convert(hist.time))

        else:
            if not shim.graph.is_computable(
                [hist._sym_tidx for hist in self.statehists]):
                raise TypeError("Advancing models is only implemented for "
                                "histories with a computable current time "
                                "index (i.e. the value of `hist._sym_tidx` "
                                "must only depend on symbolic constants and "
                                "shared vars).")
            # try:
            #     self.eval_updates()
            # except shim.graph.MissingInputError:
            #     raise shim.graph.MissingInputError("There "
            #         "are symbolic inputs to the already present updates:"
            #         "\n{}.\nEither discard them with `theano_reset()` or "
            #         "evaluate them with `eval_updates` (providing values "
            #         "with the `givens` argument) before advancing the model."
            #         .format(shim.graph.inputs(shim.get_updates().values())))
            curtidx = min( hist.time.Index(
                               shim.graph.eval(hist._sym_tidx, max_cost=50)
                           ).convert(self.time)
                           for hist in self.statehists )
            assert(curtidx >= -1)

            if curtidx < stoptidx:
                self._advance(curtidx, stoptidx+1)
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

    @property
    def _advance(self):
        """
        Attribute which caches the compilation of the advance function.
        """
        if not hasattr(self, '_advance_updates'):
            object.__setattr__(self, '_advance_updates',
                               self.get_advance_updates())
            # DEBUG
            # for i, s in enumerate(['base', 'value', 'start', 'stop']):
            #     self._advance_updates[self.V._num_data].owner.inputs[i] = \
            #         shim.print(self._advance_updates[self.V._num_data]
            #                    .owner.inputs[i], s + ' V')
            #     self._advance_updates[self.n._num_data].owner.inputs[i] = \
            #         shim.print(self._advance_updates[self.n._num_data]
            #                    .owner.inputs[i], s + ' n')
        if self.no_updates:
            if not hasattr(self, '_advance_fn'):
                logger.info("Compiling the update function")
                self._advance_fn = self.compile_advance_function(
                    self._advance_updates)
                logger.info("Done.")
            _advance_fn = self._advance_fn
        else:
            # TODO: Find reasonable way of caching these compilations ?
            # We would need to cache the compilation for each different
            # set of symbolic updates.
            advance_updates = OrderedDict(
                (var, shim.graph.clone(upd, replace=shim.get_updates()))
                for var, upd in self._advance_updates.items())

            logger.info("Compiling the update function")
            _advance_fn = self.compile_advance_function(advance_updates)
            logger.info("Done.")

        return _advance_fn

    def get_advance_updates(self):
        """
        Returns a 'blank' update dictionary. Update graphs do not include
        any dependencies from the current state, such as symbolic/transformed
        initial conditions.
        """
        cur_tidx = self.cur_tidx
        if not hasattr(self, '_curtidx_var'):
            object.__setattr__(self, '_curtidx_var',
                               shim.tensor(np.array(1, dtype=scs.tidx_dtype),
                                           name='curtidx (model)'))
            #                   shim.shared(np.array(cur_tidx+1, dtype=self.tidx_dtype)))
                                # shim.getT().scalar('curtidx (model)',
                                #                    dtype=self.tidx_dtype))
            # self._curtidx_var.tag.test_value = 1
        if not hasattr(self, '_stoptidx_var'):
            object.__setattr__(self, '_stoptidx_var',
                               shim.tensor(np.array(3, dtype=scs.tidx_dtype),
                                           name='curtidx (model)'))
            #                   shim.shared(np.array(cur_tidx+3, dtype=self.tidx_dtype)))
                                # shim.getT().scalar('stoptidx (model)',
                                #                    dtype=self.tidx_dtype))
            # self._stoptidx_var.tag.test_value = 3
                # Allow model to work with compute_test_value != 'ignore'
                # Should be at least 2 more than _curtidx, because scan runs
                # from `_curtidx + 1` to `stoptidx`.
        logger.info("Constructing the update graph.")
        # Stash current symbolic updates
        for h in self.statehists:
            h.stash()  # Stash unfinished symbolic updates
        updates_stash = shim.get_updates()
        shim.reset_updates()

        # Get advance updates
        updates = self.advance_updates(self._curtidx_var, self._stoptidx_var)
        # Reset symbolic updates to their previous state
        self.theano_reset()
        for h in self.statehists:
            h.stash.pop()
        shim.config.symbolic_updates = updates_stash
        logger.info("Done.")
        return updates

    def compile_advance_function(self, updates):
        self._debug_ag = updates
        fn = self.compile_cache.get([], updates, self.rng)
        if fn is None:
            fn = shim.graph.compile([self._curtidx_var, self._stoptidx_var], [],
                                    updates = updates)
            self.compile_cache.set([], updates, fn, self.rng)
        else:
            logger.info("Compiled advance function loaded from cache.")
        return fn

    def advance_updates(self, curtidx, stoptidx):
        """
        Compute model updates from curtidx to stoptidx.

        Parameters
        ----------
        curtidx: symbolic (int):
            We want to compute the model starting from this point.
        stoptidx: symbolic (int)
            We want to compute the model up to this point.

        Returns
        -------
        Update dictionary:
            Compiling a function and providing this dictionary as 'updates' will return a function
            which fills in the histories up to `stoptidx`.
        """
        # self.remove_other_histories()  # HACK
        # self.clear_unlocked_histories()
        # self.theano_reset()
        if not all(np.can_cast(stoptidx.dtype, hist.tidx_dtype)
                   for hist in self.statehists):
            raise TypeError("`stoptidx` cannot be safely cast to a time index. "
                            "This can happen if e.g. a history uses `int32` for "
                            "its time indices while `stoptidx` is `int64`.")

        if len(list(self.unlocked_statehists)) == 0:
            raise NotImplementedError
        # elif len(self.statehists) == 1:
        #     hist = next(iter(self.statehists))
        #     startidx = hist._num_tidx - hist.t0idx + self.t0idx
        # else:
        #     startidx = shim.smallest( *( hist._num_tidx - hist.t0idx + self.t0idx
        #                                 for hist in self.statehists ) )
        try:
            assert( shim.get_test_value(curtidx) >= -1 )
                # Iteration starts at startidx + 1, and will break for indices < 0
        except AttributeError:
            # Unable to find test value; just skip check
            pass

        # `onestep` is the function that will be passed to `scan`.
        # The signature is determined by Theano:
        # first the time indices, then the output variables.
        # We add the convention that within th output variables, nonstate
        # histories appear before state histories.
        def onestep(tidx, *args):
            # To help with debugging, assign a name to the symbolic variables
            # created by `scan`
            unlocked_statevar_names = [s + ' (scan)'
                                       for s, h in zip(self.State.__annotations__,
                                                       self.unlocked_statehists)]
            unlocked_nonstate_names = [h.name + ' (scan)'
                                       for h in self.unlocked_nonstatehists]
            for x, name in zip(
                utils.flatten(tidx, *args, terminate=shim.cf._TerminatingTypes),
                utils.flatten('tidx (scan)',
                              unlocked_nonstate_names,
                              unlocked_statevar_names,
                              terminate=shim.cf._TerminatingTypes)):
                if getattr(x, 'name', None) is None:
                    x.name = name
            # Decompose the scan argument. Following our chosen convention,
            # all non state variables come first.
            m = len(unlocked_nonstate_names)
            _nonstate = args[:m]
            _state = args[m:]
            assert len(_state) == len(unlocked_statevar_names)
            # Now call the method which, starting from a symbolic current
            # state, builds the graph for the state at the next time point.
            state_outputs, updates = self.symbolic_update(tidx, *_state)
            assert len(state_outputs) == len(list(self.unlocked_statehists))
            # Once the state variables are updated, non-state variables
            # should be simple derived quantities
            nonstate_outputs, nonstate_updates = self.nonstate_symbolic_update(
                tidx, list(self.unlocked_nonstatehists),
                _state, _nonstate, state_outputs)
            assert len(set(updates).intersection(nonstate_updates)) == 0
            # Add the nonstate updates to the state updates
            updates.update(nonstate_updates)
            # Cast the outputs so they are of the expected type
            # TODO: Is this still necessary ? History already casts function results
            for i, statehist in enumerate(self.unlocked_statehists):
                state_outputs[i] = shim.cast(state_outputs[i],
                                             statehist.dtype)
            for i, hist in enumerate(self.unlocked_nonstatehists):
                nonstate_outputs[i] = shim.cast(nonstate_outputs[i],
                                                hist.dtype)
            # Finally, return all the outputs following the convention:
            # nonstate before state outputs
            return nonstate_outputs + state_outputs, updates
            #return list(state_outputs.values()), updates

        outputs_info = []
        for hist in self.unlocked_nonstatehists:
            tidx = hist.time.Index(curtidx).convert(hist.time)
            outputs_info.append(sinn.upcast(hist._sym_data[tidx],
                                            to_dtype=hist.dtype,
                                            same_kind=True,
                                            disable_rounding=True))
        for hist in self.unlocked_statehists:
            # TODO: Generalize
            maxlag = hist.pad_left.plain
            # HACK/FIXME: We should query history for its lags
            if maxlag > 1:
                lags = [-maxlag, -1]
            else:
                lags = [-1]
            tidx = curtidx.convert(hist.time)
            # assert(maxlag <= hist.t0idx)
            #     # FIXME Maybe not necessary if built into lag history
            if len(lags) == 1:
                assert(maxlag == 1)
                outputs_info.append( sinn.upcast(hist._sym_data[tidx],
                                                 to_dtype=hist.dtype,
                                                 same_kind=True,
                                                 disable_rounding=True))
            else:
                outputs_info.append(
                    {'initial': sinn.upcast(hist._sym_data[tidx+1-maxlag:tidx+1],
                                            to_dtype=hist.dtype,
                                            same_kind=True,
                                            disable_rounding=True),
                     'taps': lags})


        outputs, upds = shim.scan(onestep,
                                  sequences = shim.arange(curtidx+1, stoptidx),
                                  outputs_info = outputs_info,
                                  return_list = True)

        # Remove histories from updates
        # FIXME: For state histories this is don in _get_symbolic_update
        # should we also remove the nonstate updates there instead ?
        # if len(updates) > 0:
        #     for h in self.history_set:
        #         if h._num_data in updates:
        #             assert not h.locked
        #             del updates[h._num_data]
        #         if h._num_tidx in updates:
        #             assert not h.locked
        #             del updates[h._num_tidx]

        m = len(list(self.unlocked_nonstatehists))
        nonstate_outputs = outputs[m:]
        state_outputs = outputs[m:]
        # Ensure that all updates are of the right type
        # Theano can add updates for variables that don't have a dtype, e.g.
        # a RandomStateType variable, which is why we include the hasattr guard
        upds = OrderedDict([(orig_var,
                             (sinn.upcast(upd, to_dtype=orig_var.dtype,
                                          same_kind=True, disable_rounding=True))
                              if hasattr(orig_var, 'dtype') else upd)
                            for orig_var, upd in upds.items()])
        self.apply_updates(upds)
            # Applying updates ensures we remove the iteration variable
            # scan introduces from the shim updates dictionary
            # FIXME: This sounds pretty hacky, although it seems like a good
            # idea to update the intermediate state of all the histories in
            # case there are subsequent operations.

        # Update the state variables
        # These are stripped from the update dictionary within
        # `_get_symbolic_update` because we want to update them with a slice
        # rather than with a long sequence of nested `IncSubtensor` ops.
        for h in self.history_set:
            h.stash()
        updates_stash = shim.get_updates()
        self.theano_reset()
        for hist, output in zip(chain(self.unlocked_nonstatehists,
                                      self.unlocked_statehists),
                                outputs):
            assert hist._num_data not in upds
            valslice = slice(curtidx.convert(hist.time) + 1,
                             stoptidx.convert(hist.time))
            # odata = hist._num_data
            # upd = shim.set_subtensor(hist._sym_data[valslice], output)
            upd = sinn.upcast(output, to_dtype=hist.dtype,
                              same_kind=True, disable_rounding=True)
            hist.update(valslice, upd)
                # `update` applies the update and adds it to shim's update dict
        hist_upds = shim.get_updates()
        for h in self.history_set:
            h.stash.pop()
        shim.config.symbolic_updates = updates_stash

        # hist_upds = shim.get_updates()
        # # Ensure that all updates are of the right type
        # # Theano can add updates for variables that don't have a dtype, e.g.
        # # a RandomStateType variable, which is why we include the hasattr guard
        # hist_upds = OrderedDict([(orig_var,
        #                           (sinn.upcast(upd, to_dtype=orig_var.dtype,
        #                                       same_kind=True, disable_rounding=True))
        #                            if hasattr(orig_var, 'dtype') else upd)
        #                          for orig_var, upd in hist_upds.items()])

        assert len(set(upds).intersection(hist_upds)) == 0
        upds.update(hist_upds)
        return upds

    def symbolic_update(self, tidx, *statevars):
        """
        Attempts to build a symbolic update automatically. This is work in
        progress, so for the time being will only work on simpler models.
        An error is thrown if the function suspects the output to be wrong.
        For more complicated models you can define the `symbolic_update`
        method yourself in the model's class.
        Creating the graph is quite slow, but the result is cached to disk, so
        subsequent calls don't need to recreate it.

        Parameters
        ----------
        tidx: symbolic int
            The symbolic integer representing the "next" time index.

        *statevars: symbolic expressions
            All subsequent variables should match the shape and type of a
            time slice from each *unlocked* history in `self.statehists`, in
            order. Histories `h` for which `h.locked is True` don't need to
            be updated and should not be passed as arguments.
        """
        # TODO: if module attribute cache is removed, remove the
        # placeholder variable and move the on-disk cache to this
        # function.

        # This function is actually a wrapper which caches the result of
        # `_get_symbolic_update`, to avoid constructing the graph twice.
        # However, it is `symbolic_update` that is part of the API
        # and which should be overloaded by a derived class, so this is the
        # one we document.
        l = len(list(self.unlocked_statehists))
        if (len(statevars) > 0
            and not isinstance(statevars[0], shim.cf.GraphTypes)):
            raise TypeError("state variables must be passed separately to "
                            "`symbolic_update`, not as a tuple or list.")
        elif len(statevars) < l:
            raise TypeError("There are {} unlocked state histories, but only "
                            "{} state variables were passed to "
                            "`symbolic_update`.".format(len(statevars), l))
        elif len(statevars) > l:
            raise TypeError("There are {} unlocked state histories, but "
                            "{} state variables were passed to "
                            "`symbolic_update`. Remember that variables should "
                            "not be passed for locked state histories."
                            .format(l, len(statevars)))
        return self._get_symbolic_update(tidx, *statevars)
        # if not hasattr(self, '_symbolic_update_graph'):
        # if True:
        #     stateph = self.get_state_placeholder()
        #     symbupd = self._get_symbolic_update(tidx, *stateph)
        # #    self._symbolic_update_graph = (stateph, symbupd)
        # # else:
        # #     stateph, symbupd = self._symbolic_update_graph
        # # symbupd: ([state xt vars], odict(shared var updates))
        # subs = OrderedDict((xph, x) for xph, x in zip(stateph, statevars))
        # outputs = [shim.graph.clone(xt, replace=subs) for xt in symbupd[0]]
        # updates = OrderedDict((var, shim.graph.clone(upd, replace=subs))
        #                       for var, upd in symbupd[1].items())
        # return outputs, updates

    def _get_symbolic_update(self, tidx, *statevars):
        """See :method:symbolic_update."""
        # Stash current symbolic updates
        assert set(self.statehists).issubset(self.history_set)
        for h in self.history_set:
            h.stash()  # Stash unfinished symbolic updates
        updates_stash = shim.get_updates()
        self.theano_reset()

        # It doesn't really matter which time point we use, we just want the
        # t -> t+1 update. But a graph update will be created for every time
        # point between _num_tidx and the chosen _tidx, so making it
        # large can be really costly.
        # Can't just use self._refhist because it could filled while
        # others are empty (e.g. if it is filled with data)
        ush = list(self.unlocked_statehists)
        refhist_idx =  np.argmax([h.cur_tidx - h.t0idx + self._refhist.t0idx
                                  for h in ush])
        refhist = ush[refhist_idx]
        ref_tidx = refhist._num_tidx
        tidcs = [ref_tidx - refhist.t0idx + h.t0idx
                 for h in self.unlocked_statehists]
        # tidxvals = [shim.graph.eval(ti) for ti in tidcs]
        # Get the placeholder current state
        # Get the placeholder new state
        # St = [(h[ti+1], False) for h, ti in zip(self.statehists, tidcs)]
        St = [(h._update_function(ti+1), False)
              for h, ti in zip(self.unlocked_statehists, tidcs)]
            # We exclude locked histories because those shouldn't be modified
        # Bool is flag indicating whether history graph is fully substituted
        # When they are all True, we stop substitutions

        # Replace any PyMC3 prior by the corresponding shared variable
        prior_subs = {p.prior: p for p in self.params if hasattr(p, 'prior')}
        if len(prior_subs) > 0:
            St = [(shim.graph.clone(xt[0], replace=prior_subs), xt[1])
                  for xt in St]

        # FIXME: Assumes no dependencies beyond a lag 1 for every one
        # Get S0 after St: don't need to update _data the second time, so it
        # will be the same _data which is indexed for both.
        #S0 = [h[ti] for h, ti in zip(self.statehists, tidcs)]
        # assert(len(S0) == len(St))
        # assert(len(St) == len(statevars))

        # Check if this is in the disk cache
        St_graphs_original = [xt[0] for xt in St]
        updates_original = shim.get_updates()
        St_graphs, updates = self.graph_cache.get(
            St_graphs_original, updates_original,
            other_inputs = statevars + (tidx,), rng = self.rng)

        if St_graphs is not None:
            logger.info("Symbolic update graphs loaded from cache.")
        else:
            # It's not in the cache, so we have to do the substitutions
            # TODO: Move to own function, or combine with `batch_function_decorator`
            for recursion_count in range(5):  # 5: max recursion
                # # ---------------------------
                # # Debugging code
                # # Update variables which still have symbolic inputs
                # odatas = [h._num_data for h in self.statehists]
                # xvars = [(h.name, xt[0]) for h, xt in zip(self.statehists, St)
                #          if any(y in shim.graph.variables([xt[0]])
                #                 for y in odatas + [ref_tidx])]
                # # The unsubstituted symbolic inputs to the above update variables
                # ivars = [[(h.name, h._num_data) for h in self.statehists
                #             if h._num_data in shim.graph.variables([xt])]
                #          for _, xt in xvars]
                # if len(xvars) > 0:
                #     # Locating the first unsubstituted symbolic input in the graph
                #     upd = xvars[0][1]
                #     xin = ivars[0][0][1]
                #     child1 = [v for v in shim.graph.variables([upd])
                #                 if v.owner is not None and xin in v.owner.inputs]
                #     child2 = [v for v in shim.graph.variables([upd])
                #                 if v.owner is not None
                #                 and v.owner.inputs[0].owner is not None
                #                 and xin in v.owner.inputs[0].owner.inputs]
                # import pdb; pdb.set_trace()
                # # ---------------------------
                if all(xt[1] for xt in St):
                    # All placeholders are substituted
                    break
                for i in range(len(St)):
                    # Don't use list comprehension, that way if earlier states
                    # appear in the updates for later states, their substitutions
                    # are already applied. This should save recursion loops.
                    _St = [xt[0] for xt in St]
                    St[i] = self.sub_states_into_graph(
                        St[i][0], self.unlocked_statehists,
                        statevars, _St, tidx, ref_tidx, refhist)

            assert all(xt[1] for xt in St)
                # All update graphs report as successfully substituted

            # Also substitute updates
            # We shouldn't need recursion for these
            updates = shim.get_updates()
            # Remove the updates to histories: those are done by applying
            # the St graphs
            if len(updates) > 0:
                for h in self.statehists:
                    if h._num_data in updates:
                        assert not h.locked
                        del updates[h._num_data]
                    if h._num_tidx in updates:
                        assert not h.locked
                        del updates[h._num_tidx]
                subbed_updates = [self.sub_states_into_graph(
                                    upd, self.unlocked_statehists,
                                    statevars, _St, tidx, ref_tidx, refhist)
                                  for upd in updates.values()]
                updvals, updsuccess = zip(*subbed_updates)
                    # Transpose `subbed_updates`
                assert all(updsuccess)
                    # All updates report as successfully substituted
                for var, upd in zip(updates, updvals):
                    updates[var] = upd
            assert all(u1 is u2 for u1, u2 in zip(updates.values(),
                                                  shim.get_updates().values()))

            St_graphs = [xt[0] for xt in St]
            # Sanity checks
            try:
                assert(shim.graph.is_computable(St_graphs,
                                                with_inputs=statevars+(tidx,)))
                    # If we have >1 time lags, this should catch it
                all_graphs = St_graphs + list(updates.values())
                inputs = shim.graph.inputs(all_graphs)
                # vs = [v for v in shim.graph.variables(inputs, St) if hasattr(v.owner, 'inputs') and any(i.name is not None and 'data' in i.name for i in v.owner.inputs)]  # DEBuG
                assert(ref_tidx not in inputs)
                    # Still test this: if ref_tidx is shared, it's computable
                # assert(not any(x0 in inputs for x0 in S0))
                assert not any(h._num_data in inputs
                               for h in self.unlocked_statehists)
                    # Symbolic update should only depend on `statevars` and `tidx`
            except AssertionError as e:
                raise (AssertionError(failed_build_msg)
                        .with_traceback(e.__traceback__))

            self.graph_cache.set(St_graphs_original,updates_original,
                                 St_graphs, updates, self.rng)

        # Reset symbolic updates to their previous state
        for h in self.history_set:
            h.stash.pop()
        shim.config.symbolic_updates = updates_stash

        # Return the new state
        return St_graphs, updates

    def nonstate_symbolic_update(self, tidx, hists, curstatevars,
                                 curnonstatevars, newstatevars):
        # TODO: Combine more with _get_symbolic_update ?

        assert all(h not in self.statehists for h in hists)
        assert set(self.statehists).issubset(self.history_set)
            # Basic check that all histories were properly attached to the model
            # This is a necessary but not sufficient condition
        # Stash any current symbolic update
        for h in self.history_set:
            h.stash()
        updates_stash = shim.get_updates()
        self.theano_reset()

        # Can't just use self._refhist because it could filled while
        # others are empty (e.g. if it is filled with data)
        refhist_idx =  np.argmax([h.cur_tidx - h.t0idx + self._refhist.t0idx
                                  for h in hists])
        refhist = hists[refhist_idx]
        ref_tidx = refhist._num_tidx
        tidcs = [ref_tidx - refhist.t0idx + h.t0idx
                  for h in hists]
        # statetidcs = [ref_tidx - self._refhist.t0idx + h.t0idx
        #               for h in self.statehists]
        # statetidxvals = [shim.graph.eval(ti) for ti in tidcs]

        ht = [(h._update_function(ti+1), False)
              for h, ti in zip(hists, tidcs)]
        assert len(ht) == len(hists)

        # Replace any PyMC3 prior by the corresponding shared variable
        prior_subs = {p.prior: p for p in self.params if hasattr(p, 'prior')}
        ht = [(shim.graph.clone(xt[0], replace=prior_subs), xt[1]) for xt in ht]

        # Remove the locked histories from the variables we want to substitute:
        # those don't need to be computed (typically they contain they
        # observation data) and so should stay in the graphs.
        curstatevars = tuple(sv for sv, h
                                in zip(curstatevars, self.unlocked_statehists))
        newstatevars = tuple(sv for sv, h
                                in zip(newstatevars, self.unlocked_statehists))
        statehists   = tuple(self.unlocked_statehists)
        assert len(curstatevars) == len(newstatevars) == len(statehists)

        newnonstatevars = tuple(nsv for nsv,f in ht)

        # First sub the state variables. This should work with a single pass.
        ht = [self.sub_states_into_graph(
                xt[0], statehists,
                curstatevars,
                newstatevars,
                tidx, ref_tidx, refhist)
              for xt in ht]
        # Now sub the non-state variables. On every pass, the final one should
        # not contain any more of the original variables.
        for i in range(len(ht)):
            for j in range(len(ht)-i):
                _ht = [xt[0] for xt in ht]
                ht[i] = self.sub_states_into_graph(
                    _ht[j], hists,
                    curnonstatevars,
                    _ht,
                    tidx, ref_tidx, refhist)
                if all(xt[1] for xt in ht):
                    break
            assert ht[len(ht)-i-1][1]  # Check that the last one entry is substituted
        assert all(xt[1] for xt in ht)
            # All update graphs report as successfully substituted
        graphs = [xt[0] for xt in ht]

        # Drop all updates: this is a side-effect-free calculation, and state
        # variables are taken care of by `scan`'s output variables.
        updates = OrderedDict()
        # # Also substitute updates
        # updates = shim.get_updates()
        # # Ensure that we are not updating histories outside of `hists`
        # if len(updates) > 0:
        #     for h in self.history_set.difference(hists):
        #         assert h._num_data not in updates
        #         assert h._num_tidx not in updates
        #     subbed_updates = [self.sub_states_into_graph(
        #                         upd, statehists, curstatevars, newstatevars, tidx)
        #                       for upd in updates.values()]
        #     updvals, updsuccess = zip(*subbed_updates)  # Transpose `subbed_updates`
        #     assert all(updsuccess)
        #         # All updates report as successfully substituted
        #     for var, upd in zip(updates, updvals):
        #         updates[var] = upd
        # assert all(u1 is u2 for u1, u2 in zip(updates.values(),
        #                                       shim.get_updates().values()))

        # Sanity checks
        try:
            all_graphs = graphs + list(updates.values())
            assert shim.graph.is_computable(
                all_graphs, with_inputs=curstatevars+(tidx,))
                # If we have >1 time lags, this should catch it
            inputs = shim.graph.inputs(all_graphs)
            # vs = [v for v in shim.graph.variables(inputs, St) if hasattr(v.owner, 'inputs') and any(i.name is not None and 'data' in i.name for i in v.owner.inputs)]  # DEBuG
            assert ref_tidx not in inputs
                # Still test this: if ref_tidx is shared, it's computable
            # assert(not any(x0 in inputs for x0 in S0))
            assert not any(h._num_data in inputs for h in statehists)
            assert not any(h._num_data in inputs for h in hists)
                # Symbolic update should only depend on `statevars` and `tidx`
        except AssertionError as e:
            raise (AssertionError(failed_build_msg)
                    .with_traceback(e.__traceback__))

        # Reset symbolic updates to their previous state
        for h in self.history_set:
            h.stash.pop()
        shim.config.symbolic_updates = updates_stash

        # Return the new values
        return graphs, updates

    def sub_states_into_graph(self, graph, hists, cur_histvars, new_histvars,
                              new_tidx, ref_tidx, refhist):
        """
        Substitute nodes in `graph` where a state variable is indexed
        by the corresponding variable in `statevars`.
        This returns a correctly substituted :param:graph for `t -> t+1`
        calculation where :param:cur_histvars corresponds to `t`and
        :param:new_histvars corresponds to `t+1`.

        Parameters
        ----------
        graph: computational graph
            Computational graph we wish to traverse.
        hists: list of Histories
            The histories for which we want to replace references to their data
            by virtual variables.
        cur_histvars: list of symbolic variables
            The symbolic variables to use for replacing the current state of
            the histories in :param:hists.
        new_statevars: list of symbolic expressions
            The symbolic variables to use for replacing the new / next state
            of the histories in :param:hists.
        new_tidx: symbolic index
            Time index relative to `self.t0idx` corresponding to the new state.
        """
        theano = shim.gettheano()
        # TODO: Allow lists of graphs ?
        # cur_statevars and new_statevars are typically virtual state variables,
        # e.g. symbolics appearing within a scan or which will be replaced
        # by another variable before compilation.

        # TODO: apply varname changes in method's code
        cur_statevars = cur_histvars
        new_statevars = new_histvars

        # If history lists are generators, turn them into lists because
        # we will iterate over them more than once
        if not isinstance(hists, (list, tuple)): hists = list(hists)
        if not isinstance(cur_histvars, (list, tuple)):
            cur_histvars = list(cur_histvars)
        if not isinstance(new_histvars, (list, tuple)):
            new_histvars = list(new_histvars)
        assert len(hists) == len(cur_histvars) == len(new_histvars)

        # # Can't just use self._refhist because it could filled while
        # # others are empty (e.g. if it is filled with data)
        # refhist_idx =  np.argmax([h.cur_tidx - h.t0idx + self._refhist.t0idx
        #                           for h in hists])
        # refhist = hists[refhist_idx]
        # ref_tidx = refhist._num_tidx
        new_tidx = new_tidx - self.t0idx + refhist.t0idx
            # Convert time index to be relative to reference history
        _tidcs = [ref_tidx - refhist.t0idx + h.t0idx
                  for h in hists]
        ref_tidxvals = [shim.graph.eval(ti) for ti in _tidcs]
        inputs = shim.graph.inputs([graph])
        variables = shim.graph.variables([graph])
        odatas = [h._num_data for h in hists]
        statevars = cur_statevars
        St = new_statevars

        # Check if this graph is already fully substituted
        if (not any(odata in inputs for odata in odatas)
            and ref_tidx not in inputs):
            fully_substituted = True
            return graph, fully_substituted

        # Proceed with substitution
        replace = {}
        # Loop over graph nodes, replacing any instance where we index
        # into _num_data by the appropriate virtual state variable
        for y in variables:
            if (shim.graph.symbolic_inputs(y) == [ref_tidx]
                and shim.graph.is_same_graph(y, ref_tidx)):
                # `is_same_graph` is expensive: avoid computing it when possible
                replace[y] = shim.cast(new_tidx - 1, y.dtype)
            elif (y.owner is None
                  or (not any(odata in shim.graph.symbolic_inputs(y)
                              for odata in odatas))):
                  # Terminate recursion if at end, or if all remaining
                  # variables are virtual variables added for the scan
                continue
            elif isinstance(y.owner.op, theano.tensor.Subtensor):
                if (y.owner.inputs[0].owner is not None
                    and isinstance(y.owner.inputs[0].owner.op,
                                   theano.tensor.IncSubtensor)):
                    for xs, xt2, odata, tival in zip(statevars, St, odatas, ref_tidxvals):
                        if y.owner.inputs[0].owner.inputs[0] is odata:
                            # if tival != 0:
                            #     # I don't remember the logic for this condition,
                            #     # but without it things don't work
                            #     continue
                            iy = shim.graph.eval(y.owner.inputs[1],
                                                 max_cost=50)
                            # Subtensor op corresponds to selecting the
                            # updated state variable
                            # odata is a shared variable attached to a hist
                            # xs is the symbolic variable for the current state
                            # xt2 is the symbolic variable for the once-updated
                            # state
                            if iy == tival:
                                # Even though we are indexing the updated
                                # variable, we are indexing it at the original
                                # index value
                                replace[y] = xs
                            elif iy == tival+1:
                                # Indexing 1 past the original index value
                                # => use the updated state variable
                                replace[y] = xt2
                            else:
                                raise RuntimeError(
                                    "Graph substitution got confused. This may "
                                    "be due to indexixng more than one time-"
                                    "step in the past.")

                            # if tival == 0 and expensive_asserts:
                            #     assert(iy == tival+1)
                                # assert(shim.graph.eval(
                                #   y.owner.inputs[0].owner.inputs[1]-xt2,
                                #   max_cost = 1000,
                                #   givens={new_tidx:ref_tidx+1}) == 0 )
                                #   # FIXME: `givens` assume lag of 1
                            # replace[y] = xt2
                            break
                else:
                    for xs, xt2, odata, tival in zip(statevars, St,
                                                     odatas, ref_tidxvals):
                        if y.owner.inputs[0] is odata:
                            # args: data, index
                            i = shim.eval(y.owner.inputs[1])
                            if i == tival:
                                replace[y] = xs
                            elif i == tival+1:
                                # We can land here if the histories'
                                # original tindices aren't synchronized
                                assert(xt2 != graph)
                                replace[y] = xt2
                            break
                    else:
                        # We can land here e.g. when indexing a hist
                        # which is not a state variable (like input),
                        # but is there another way ? Should we throw
                        # a warning ?
                        pass
                # Any situation with different `i` is unsupported
                # and will be caught in the asserts below
                # FIXME: This is where to add support for multiple lags

        new_graph = shim.graph.clone(graph, replace)

        # Check if this graph is fully substituted
        new_inputs = shim.graph.inputs([new_graph])
        if (not any(odata in new_inputs for odata in odatas)
            and ref_tidx not in new_inputs):
            # We replaced all placeholder variables with virtual ones
            fully_substituted = True
        elif (len(replace) == 0):
            # There are placeholder variables in the graph that we are unable
            # to substitute
            raise RuntimeError(failed_build_msg)
        else:
            fully_substituted = False

        return new_graph, fully_substituted

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
