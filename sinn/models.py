# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Copyright 2017-2021 Alexandre René

TODO
----

- At present, things link ``h for h in model.statehists`` will use __eq__
  for the ``in`` check, which for histories creates a dict and compares.
  Iterables like `statehists` should be something like a ``set``, such that
  this check then just compares identity / hashes.
- Have one dictionary/SimpleNamespace storing all compilation variables.
  See comment under `class Model`
"""
from __future__ import annotations

import numpy as np
import scipy as sp
from warnings import warn
import logging
logger = logging.getLogger(__name__)
import abc
from collections import namedtuple, OrderedDict, ChainMap, defaultdict
from collections.abc import Sequence, Iterable, Callable
import inspect
from inspect import isclass
from itertools import chain
from functools import partial
import copy
import sys

import pydantic
from pydantic import BaseModel, PrivateAttr, validator, root_validator
from pydantic.typing import AnyCallable
from typing import (
    Any, Optional, Union, Tuple, Sequence, List, Dict, Set, Generator,
    Callable as CallableT)
from types import SimpleNamespace
from dataclasses import dataclass
from parameters import ParameterSet

import theano_shim as shim
import mackelab_toolbox as mtb
import mackelab_toolbox.utils as utils
import mackelab_toolbox.theano
import mackelab_toolbox.typing
import mackelab_toolbox.iotools
from mackelab_toolbox.theano import GraphCache, CompiledGraphCache
from mackelab_toolbox.utils import class_or_instance_method

import sinn
import sinn.config as config
import sinn.common as com
from sinn.axis import DiscretizedAxis, AbstractAxisIndex
from sinn.histories import (
    History, TimeAxis, AutoHist, HistoryUpdateFunction, Series, Spiketrain)
from sinn.kernels import Kernel
from sinn.diskcache import diskcache

# Import into namespace, so user code doesn't need to import utils
from sinn.utils.pydantic import initializer, add_exclude_mask

_models = {}
registered_models = _models.keys()
    # I don't really like this, but it works. Ideally it would be some kind
    # of read-only property of the module.
    # Registering models allows us to save the model name within a parameter
    # file, and write a function which can build the correct model
    # automatically based on only on that parameter file.
    # NOTE: sinn-full uses a much nicer mechanism of TaggedCollections
    # which mostly makes _models (and the functions below) obsolete.

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

# def nesteddictwalk(d, separator='.', terminate=()):
#     """
#     Walk a nested dict structure, using a generator.
#      Values associated to keys matching one of the values in `terminate` are not
#      recursed into, even if they are dictionaries.
# 
#     Composite keys are created by joining each key to the key of the parent dict
#     using `separator`.
# 
#     .. Note:: Copied from the parameters package (parameters.__init__.py), to
#        add the `terminate` argument.
#     """
#     # Used in Model.__init__ to delay creation of update functions until all
#     # histories are created.
#     if isinstance(terminate, str):
#         terminate = [terminate]
#     for key1, value1 in d.items():
#         if isinstance(value1, dict) and key1 not in terminate:
#             for key2, value2 in nesteddictwalk(value1, separator, terminate=terminate):  # recurse into subdict
#                     yield "%s%s%s" % (key1, separator, key2), value2
#         else:
#             yield key1, value1

# Parameters

# TODO: A NumericModelParams type, automatically constructed with the same
#      fields as a model's ModelParams, but using non-symbolic types.
#      Replace all occurrences of IndexableNamespace with NumericModelParams
from mackelab_toolbox.typing import IndexableNamespace

class ModelParams(BaseModel):

    class Config:
        json_encoders = mtb.typing.json_encoders

    def __init__(self, *args, **kwargs):
        # Reconstruct hierarchies from dotted parameter names
        kwargs = ParameterSet(kwargs)
        super().__init__(*args, **kwargs)

    def get_values(self):  # -> NumericModelParams
        """
        Helper method which calls `get_value` on each parameter.
        Returns a copy of the ModelParams, where each shared variable has
        been replaced by its current numeric value.
        """
        d = {k: v.get_values() if isinstance(v, ModelParams) else
                v.get_value() if shim.isshared(v) else
                v
             for k,v in self}
        return IndexableNamespace(**d)

    def set_values(self, values: ModelParams,  # TODO -> NumericModelParams
                   must_set_all_params: bool=False,
                   must_not_set_other_params: bool=True):
        """
        TODO: The annotation is not quite correct. Values should be Params-like,
              but contain no symbolic variables.
              (Typically, ModelParams subclasses _enforce_ certain params to be
              shared vars)
        Helper method which calls `set_value` on each parameter.
        For non-shared variables, the value is simply updated.
        Values are updated in place.

        Parameters
        ----------
        values: Dictionary of values to set
        must_set_all_params:
            Whether the model params must be a subset of `values`.
        must_not_set_other_params:
            Whether `values` must be a subset of the model params.
        """
        if isinstance(values, dict):
            values_dict = values
        elif isinstance(values, SimpleNamespace):
            values_dict = values.__dict__
        elif isinstance(values, type(self)):
            values_dict = {k: v for k,v in values}
        else:
            # `self` is always a subclass, and we want `values` to be instance of that subclass
            raise TypeError(f"{type(self)}.set_values: `values` must be an "
                            f"instance of {type(self)}, but is rather of type "
                            f"{type(values)}.")
        if must_set_all_params and not set(self.__fields__) <= set(values_dict):
            raise ValueError("The following parameters are missing: "
                             f"{set(self.__fields__) - set(values_dict)}")
        if must_not_set_other_params and not set(values_dict) <= set(self.__fields__):
            raise ValueError("The following parameters are not recognized by the model: "
                             f"{set(values_dict) - set(self.__fields__)}")
        # Build hierarchies, in case `values` uses dotted keys
        values_dict = ParameterSet(values_dict)
        for k, v in values_dict.items():
            # Special track for nested models
            if isinstance(v, (dict, SimpleNamespace)):
                subΘ = getattr(self, k)
                assert isinstance(subΘ, ModelParams)
                subΘ.set_values(v, must_not_set_other_params=False)
                    # At present, nested model params don't know what model
                    # they attach to, and therefore can't validate their inputs
            # Normal track
            else:
                self_v = getattr(self, k)
                if shim.isshared(self_v):
                    self_v.set_value(v)
                else:
                    setattr(self, k, v)
ModelParams.update_forward_refs()

# Model decorators

## updatefunction decorator

# PendingUpdateFunction = namedtuple('PendingUpdateFunction',
#                                    ['hist_nm', 'inputs', 'upd_fn'])
@dataclass
class PendingUpdateFunction:
    hist_nm: str
    inputs : List[str]
    upd_fn : CallableT
    def __call__(self, model, k):
        return self.upd_fn(model, k)

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
        if '__throwaway_class' in namespace:
            # During MRO resolution, we need to create a throwaway class to
            # determine the MRO of the final class. In that case we want to
            # skip all the metaclass activity because
            # 1) It's wasteful to do all the annotation/parameter parsing
            # 2) It would attempt to create its own throwaway class,
            #    leading to infinite recursion
            return super().__new__(metacls, cls, bases, namespace)

        # Make a copy of namespace, so we can modify it below
        namespace = namespace.copy()

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
        # # At the moment I begrudgingly went with Option 2, because I'm not sure
        # # of a use case for multiple inheritance.
        # nonabcbases = tuple(b for b in bases if b is not abc.ABC)
        # if len(nonabcbases) != 1:
        #     from inspect import currentframe, getframeinfo
        #     import pathlib
        #     path = pathlib.Path(__file__).absolute()
        #     frameinfo = getframeinfo(currentframe())
        #     info = f"{path}::line {frameinfo.lineno}"
        #     basesstr = ', '.join(str(b) for b in nonabcbases)
        #     raise TypeError(
        #         f"Model {cls} has either no or multiple parents: {basesstr}.\n"
        #         "Models must inherit from exactly one class (eventually "
        #         "`sinn.models.Model`). This is a technical limitation rather "
        #         "than a fundamental one. If you need multiple inheritance for "
        #         f"your model, have a look at the comments above {info}, and "
        #         "see if you can contribute a better solution.")
        # mro = bases[0].mro()
        # The solution below at least allows for multiple inheritance with
        # mixin classes
        try:
            Tmp = type('Tmp', bases, {**namespace, **{'__throwaway_class': None}})
            mro = Tmp.mro()[1:]  # Exclude 'temp' throwaway class from MRO
        except TypeError as e:
            nonabcbases = tuple(b for b in bases if b is not abc.ABC)
            basesstr = ', '.join(str(b) for b in nonabcbases)
            if "multiple bases have instance lay-out conflict" in str(e):
                raise TypeError(
                    f"Model {cls} may have multiple parents inheriting "
                    "from sinn.Model. Because sinn Models use __slots__, they "
                    "cannot be used this way. If your goal is to define "
                    "variants by changing some of the definitions of a model, "
                    "consider defining those changes in a mixin class instead. "
                    "As long as the mixin doesn't define __slots__, multiple "
                    "inheritance with it should work."
                    ) from e
            else:
                raise e

        # Existing attributes
        # (ChainMap gives precedence to elements earlier in the list)
        annotations = namespace.get('__annotations__', {})
        inherited_annotations = ChainMap(*(getattr(b, '__annotations__', {})
                                           for b in mro))
        all_annotations = ChainMap(annotations, inherited_annotations)
        inherited_kernel_identifiers = set(chain.from_iterable(
            _get_inherited_kernels(b) for b in bases))
        inherited_hist_identifiers = set(chain.from_iterable(
            _get_inherited_hists(b) for b in bases))
        inherited_pending_updates = dict(ChainMap(
            *({obj.hist_nm: obj for obj in _get_inherited_updates(b)}
              for b in bases)))
            # NB: Although the _order_ of a ChainMap is set by iterating over
            #     bases right-to-left, the _precedence_ of values goes
            #     left-to-right.
            #     This makes it comparable to a chain of .update() calls.
        # Old version which didn't work with mixins:
        # # (We only iterate over immediate bases, since those already contain
        # #  the values for their parents.)
        # inherited_kernel_identifiers = set(chain.from_iterable(
        #     getattr(b, '_kernel_identifiers', []) for b in bases[::-1]))
        # inherited_hist_identifiers = set(chain.from_iterable(
        #     getattr(b, '_hist_identifiers', []) for b in bases[::-1]))
        # inherited_pending_updates = dict(ChainMap(
        #     *({obj.hist_nm: obj for obj in getattr(b, '_pending_update_functions', [])}
        #       for b in bases[::-1])))

        # Resolve string annotations (for 3.9+, annotations are always strings)
        for nm, annot in annotations.items():
            if isinstance(annot, str):
                # To support forward refs, we leave as-is if undefined
                # Certain annotations (e.g. time) we need immediately, which
                # is why we do this before the class is defined
                annotations[nm] = sys.modules[metacls.__module__].__dict__.get(annot, annot)

        # Structures which accumulate the new class attributes
        new_annotations = {}
        _kernel_identifiers = inherited_kernel_identifiers
        _hist_identifiers = inherited_hist_identifiers
        _model_identifiers = {}  # Use dict as an ordered set
        _pending_update_functions = inherited_pending_updates
            # pending updates use a dict to allow derived classes to override

        # Model validation
        ## Disallowed attributes
        for attr in ['ModelClass']:
            if attr in annotations:
                raise TypeError(f"The attribute '{attr}' is disallowed for "
                                "subclasses of 'sinn.models.Model'.")

        # Replace any unnecessarily 'stringified' types
        # (Sometimes, even if types are in the namespace, they still
        #  get set as strings in the annotations)
        # c.f. pydantic.main.update_forward_refs & pydantic.typing.evaluate_forwardref
        globalns = sys.modules[namespace['__module__']].__dict__
        for nm, T in annotations.items():
            if isinstance(T, str):
                annotations[nm] = globalns.get(T, T)

        ## `time` parameter
        if 'time' not in annotations:
            annotations['time'] = TimeAxis
        else:
            if (not isinstance(annotations['time'], type)
                or not issubclass(annotations['time'], DiscretizedAxis)):
                raise TypeError(
                    "`time` attribute must be an instance of `DiscretizedAxis` "
                    f"(it has type {type(annotations['time'])}); "
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
                raise TypeError(f"Name clash in {cls} definition: '{nm}'")
            new_annotations[nm] = T
            if isinstance(T, type) and issubclass(T, History):
                _hist_identifiers.add(nm)
            elif isinstance(T, type) and issubclass(T, Kernel):
                _kernel_identifiers.add(nm)
            elif isinstance(T, type) and cls != 'Model' and issubclass(T, Model):
                _model_identifiers[nm] = None  # We use dict as an ordered set
            elif isinstance(T, PendingUpdateFunction):
                # FIXME: Remove. I don't remember why I originally put this branch
                raise AssertionError("Leftover PendingUpdateFunction")
                # _pending_update_functions.append(obj)

        # Sanity check State subclass
        State = namespace.get('State', None)
        if State is None:
            for C in mro:
                State = getattr(C, 'State', None)
                if State is not None:
                    break

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
        elif not hasattr(State, '__annotations__'):
            # We end up with sate-less histories, which are indicated with an empty State class
            State.__annotations__ = {}
        elif len(getattr(State, '__annotations__', {})) == 0:
            pass
            # warn(f"Model {cls} has an empty `State` class.")
        else:
            if issubclass(State, BaseModel):
                raise TypeError(f"Model {cls} `State` must be a plain class, "
                                f"not a Pydantic BaseModel.")
            # if len(_hist_identifiers) == 0:
            #     raise TypeError(
            #         f"Model {cls}: Variables declared in `State` are not all "
            #         "declared as histories in the model.")
            # Resolve string annotations (for 3.9+, annotations are always strings)
            for nm, annot in State.__annotations__.items():
                if isinstance(annot, str):
                    # State.__annotations__[nm] = sys.modules[metacls.__module__].__dict__.get(annot, annot)
                    State.__annotations__[nm] = globalns.get(annot, annot)
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
        # FIXME: When doing this, ensure Parameters.__module__ is the same as
        #        originally (and not 'abc')
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
            if new_annotations['params'] == 'Parameters':
                new_annotations['params'] = Parameters
            elif new_annotations['params'] is not Parameters:
                if isinstance(ann_params, type) and issubclass(Parameters, ann_params):
                    new_annotations['params'] = Parameters
                else:
                    raise TypeError(f"Model {cls} defines `params` but it is "
                                    f"not of type `{cls}.Parameters`. "
                                    f"(Instead it is of type `{type(ann_params)}`)")
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
                    if fn_nm in namespace:
                        raise AssertionError("ModelMetaclass.__new__: The function "
                                             "'{fn_nm}' is already in the namespace.")
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

        # Add connected hists validators
        for submodel_nm in _model_identifiers:
            namespace[f"_connect_submodel_{submodel_nm}"] = \
                connect_submodel_validator(submodel_nm)

        # Update namespace
        namespace['Config'] = Config
        namespace['Parameters'] = Parameters
        # It's not essential to have sorted identifiers (`list` would also work)
        # but predictability helps with debugging, and may affect serialization.
        namespace['_kernel_identifiers'] = sorted(_kernel_identifiers)
        namespace['_hist_identifiers'] = sorted(_hist_identifiers)
        namespace['_model_identifiers'] = list(_model_identifiers)  # Must not be sorted: order sets precedence
        namespace['_pending_update_functions'] = list(_pending_update_functions.values())
        namespace['__annotations__'] = new_annotations

        return super().__new__(metacls, cls, bases, namespace)

    # TODO: Recognize RNG as input

# Retrieve kernel_identifiers, hist_identifiers and update_functions
# from an existing class.
# This class may or may not subclass Model; in the former case there is
# nothing beyond retrieving the attributes already set. In the latter case,
# we reproduce how ModelMetaclass would extract them from annotations.
# TODO: I'm not super happy with these functions; there should be a way to do
#       this with less repeated code
# TODO: We lose here for mixins some sanity checks done in ModelMetaclass:
#       - That no annotations are duplicated
#       - That all update functions correspond to a hist_identifier
def _get_inherited_kernels(cls):
    kernel_identifiers = getattr(cls, '_kernel_identifiers', None)
    if kernel_identifiers is None:
        # cls doesn't inherit from Model – probably a mixin
        kernel_identifiers = set()
        for C in cls.mro()[::-1]:
            C_kernel_identifiers = getattr(C, '_kernel_identifiers', None)
            if C_kernel_identifiers is None:
                for nm, T in getattr(C, '__annotations__', {}).items():
                    if isinstance(T, type) and issubclass(T, Kernel):
                        kernel_identifiers.update(nm)
        for nm, T in getattr(cls, '__annotations__', {}).items():
            if isinstance(T, type) and issubclass(T, Kernel):
                kernel_identifiers.add(nm)
    return kernel_identifiers
def _get_inherited_hists(cls):
    hist_identifiers = getattr(cls, '_hist_identifiers', None)
    if hist_identifiers is None:
        # cls doesn't inherit from Model – probably a mixin
        hist_identifiers = set()
        for C in cls.mro()[::-1]:
            C_hist_identifiers = getattr(C, '_hist_identifiers', None)
            if C_hist_identifiers is None:
                for nm, T in getattr(C, '__annotations__', {}).items():
                    if isinstance(T, type) and issubclass(T, History):
                        hist_identifiers.update(nm)
        for nm, T in getattr(cls, '__annotations__', {}).items():
            if isinstance(T, type) and issubclass(T, History):
                hist_identifiers.add(nm)
    return hist_identifiers
def _get_inherited_updates(cls):
    pending_update_functions = getattr(cls, '_pending_update_functions', None)
    if pending_update_functions is None:
        # cls doesn't inherit from Model – probably a mixin
        pending_update_functions = []
        for obj in cls.__dict__.values():
            if isinstance(obj, PendingUpdateFunction):
                pending_update_functions.append(obj)
    return pending_update_functions

def init_autoseries(cls, autohist: AutoHist, values) -> Series:
    time = values.get('time', None)
    if time is None: return autohist
    if isinstance(autohist, dict):
        # We end up here when constructing a model from a dict,
        # which happens when we deserialize a model representation
        # `autohist` then is a complete definition for a history, but we
        # still make sure its time axis is consistent with the model's
        hist = Series.parse_obj(autohist)
        # Test compatibility by converting the two endpoints of the model's time axis
        try:
            time.t0idx.convert(hist.time)
            time.tnidx.convert(hist.time)
        except (TypeError, IndexError):
            raise AssertionError(f"The time array of {hist.name} does not "
                                 "match that of the model.\n"
                                 f"{hist.name}.time: {hist.time}\n"
                                 f"Model.time: {time}")
        return hist
    return Series(time=time, **autohist.kwargs)

def init_autospiketrain(cls, autohist: AutoHist, values) -> Spiketrain:
    time = values.get('time', None)
    if time is None: return autohist
    if isinstance(autohist, dict):
        # See comment in init_autoseries
        hist = Spiketrain.parse_obj(autohist)
        try:
            time.t0idx.convert(hist.time)
            time.tnidx.convert(hist.time)
        except (TypeError, IndexError):
            raise AssertionError(f"The time array of {hist.name} does not "
                                 "match that of the model.\n"
                                 f"{hist.name}.time: {hist.time}\n"
                                 f"Model.time: {time}")
        return hist
    else:
        return Spiketrain(time=time, **autohist.kwargs)

def connect_submodel_validator(submodel_name):
    """
    One of these validators is added for each submodel in a composite model.
    When a history is part of multiple submodels, it is only serialized once.
    This validator adds histories back into the dictionaries for submodels
    where they are missing.
    
    Demonstrative usage::
    >>> class A(Model):
          ha: History
        class B(Model):
          hb: History
        class C(Model):
          a: A
          b: B
          _connect_submodel_a = connect_submodel_validator('a')
          _connect_submodel_b = connect_submodel_validator('b')
          
    Actual usage: The Model metaclass adds these validators automatically,
      so users do not need to do so in their models.
    """
    def f(submodel, values):
        if not isinstance(submodel, dict):
            # This function only makes sense if `submodel` is not yet instantiated
            # but instead is a dict of attribute values
            # TODO: For instantiated models, we should still check that histories
            #       are correctly connected
            return submodel
        conns = values.get('connections')
        if conns:
            for lowerobj, upperobj in conns.items():
                if upperobj.count('.') > 1:
                    raise NotImplementedError(
                        "This function _might_ work when `upperobj` references "
                        "more deeply nested models, but it should be tested first.")
                uppermodel, histname = upperobj.rsplit('.')
                if uppermodel != submodel_name:
                    continue
                if histname in submodel:
                    raise ValueError(f"Connect history '{lowerobj}' to "
                                     f"'{upperobj}': '{upperobj}' is already "
                                     "defined.")
                # Recursively retrieve the lowerobj. Objects at different levels
                # may be both dicts and Models, so we need to support both `get`
                # and`getattr` (for the same reason, relying on ParameterSet
                # to do this won't work, since it won't recurse into objects.
                o = values
                for attr in lowerobj.split('.'):
                    try:
                        o = getattr(o, attr)
                    except AttributeError:
                        o = o[attr]
                # Connect the lower history to the upper one
                submodel[histname] = o
        return submodel
    f.__name__ = f"connect_submodel_{submodel_name.replace('.', '_')}"
    return validator(submodel_name, pre=True, allow_reuse=True)(f)

class Model(pydantic.BaseModel, abc.ABC, metaclass=ModelMetaclass):
    """Abstract model class.

    A model implementation should derive from this class.

    Models **must**:

    - Inherit from `~sinn.models.Model`
    - Define a ``time`` attribute, of type `~sinn.histories.TimeAxis`.
    - Define define a `Parameters` class within their namespace, which inherits
      from `~pydantic.BaseModel`. This class should define all (and only) model
      parameters as annotations, using the syntax provided by `pydantic`.
    - Define an :meth:`initialize(self, initializer=None)` method. This method
      is intended to be called to reset the model (e.g. after a run / fit).
      It is also called during initialization to set up the model, unless the
      special value ``initializer='do not initialize'`` is passed.
      It's second argument must be ``initializer``, it must be optional, and
      the optional value must be ``None``.

    In practice, they will also:

    - Define histories and kernels as annotations to the model.
    - Define any number of validation/initialization methods, using either the
      `@pydantic.validators` or `@sinn.models.initializer` decorators.
    - Define update functions for the histories using the `@sinn.models.updatefunction`
      decorator.

    A typical class definition should look like::

        from sinn.histories import TimeAxis, Series
        from sinn.models import Model
        from pydantic import BaseModel

        class MyModel(Model):
            time: TimeAxis
            class Parameters(BaseModel):
                ...
            x: Series

            @initializer('x')
            def init_x(cls, x, time):
                return Series(time=time, ...)

            @updatefunction('x', inputs=())
            def upd_x(self, tidx):
                return ...

    .. Hint::
       If you are subclassing Model to create another abstract class (a class
       meant to be subclassed rather than used on its own), add `abc.ABC` to
       its parents – this will identify your model as abstract, and disable
       warnings about missing attributes like `State`.

    .. Warning::
       If you need to set the model to a predictable state, use the provided
       `~Model.reseed_rngs` method. Simply setting the state of `self.rng`
       will work for NumPy RNGs, but not symbolic ones.
    """
    # TODO: Use the new PrivateAttr instead of __slots__
    __slots__ = ('graph_cache', 'compile_cache', '_pymc', #'batch_start_var', 'batch_size_var',
                 '_num_tidx', '_curtidx_var', '_stoptidx_var', '_batchsize_var',
                 '_advance_updates', '_compiled_advance_fns')
    _num_tidx_objs: dict=PrivateAttr({})
        # _num_tidx objects are shared variables which can be used in
        # computational graphs. This caching attribute ensures the same object
        # is always returned, so that it can be properly substituted in the
        # computational graph. Since the variable depends on which state
        # histories are unlocked, a different object is returned for different
        # unlocked history combinations (hence the need for a dict)
    _rng_updates  : defaultdict=PrivateAttr(
        defaultdict(lambda: defaultdict(lambda: [])))
        # Store RNG updates generated in `get_advance_updates`, so that
        # `reseed_RNGs` knows which RNG need to be seeded.
        # Structure: _rng_updates[cache_key][rng] = [upd1, upd2, ...]
    connections: Optional[Dict[str,str]]
        # Used to connect histories between submodels; mostly used for deserialization
        # Defines connection pairs for histories in different submodels
        # Order: {lower model.hist name : upper model.hist name},
        # where the "lower" model is the one which can be instantiated first
        # IMPORTANT: It is not required to set this variable in order to connect
        #    histories; simply reusing the same history instance will do.
        #    Once a model is constructed, its `connections` attribute
        #    is updated to reflect actual history connections.

    class Config:
        # Allow assigning other attributes during initialization.
        extra = 'allow'
        keep_untouched = (ModelParams, PendingUpdateFunction, class_or_instance_method)
        json_encoders = {**mtb.typing.json_encoders,
                         **History.Config.json_encoders}

    class Parameters(abc.ABC, ModelParams):
        """
        Models must define a `Parameters` class within their namespace.
        Don't inherit from this class; i.e. do::

            class MyModel(Model):
                class Parameters(BaseModel):
                    ...
        """
        pass

    # Register subclasses so they can be deserialized
    def __init_subclass__(cls):
        if not inspect.isabstract(cls) and '__throwaway_class' not in cls.__dict__:
            mtb.iotools.register_datatype(cls)

    def __new__(cls, ModelClass=None, **kwargs):
        """
        Allow instantiating a more specialized Model from the base class.
        If `ModelClass` is passed (usually a string from the serialized model),
        the corresponding model type is loaded from those registered with
        mtb.iotools. Class construction is then diverted to that subclass.
        `ModelClass` also serves as a flag, to indicate that we are
        reconstructing a model (either from memory or disk), and so that it
        should not be initialized.
        """
        if ModelClass is not None:
            if isinstance(ModelClass, str):
                try:
                    # TODO: Fix API so we don't need to use private _load_types
                    ModelClass = mtb.iotools._load_types[ModelClass]
                except KeyError as e:
                    raise ValueError(f"Unrecognized model type '{ModelClass}'."
                                     ) from e
            if not (isinstance(ModelClass, type) and issubclass(ModelClass, cls)):
                raise AssertionError(f"Model.__new__: {ModelClass} is not a subclass of {cls}.")
            if 'initializer' not in kwargs:
                kwargs['initializer'] = 'do not initialize'
            # IMPORTANT: __init__ will still be called with original sig,
            # so setting 'initializer' needs to be done there too.
            # If there are more sig. changes, we may want to call __init__
            # here ourselves, and e.g. set a '_skip_init' attribute
            return ModelClass.__new__(ModelClass, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self, initializer=None, ModelClass=None, **kwargs):
        # Recognize if being deserialized, as we do in __new__
        if ModelClass is not None and initializer is None:
            initializer = 'do not initialize'
        # Any update function specification passed as argument needs to be
        # extracted and passed to _base_initialize, because update functions
        # can't be created until the model (namespace) exists
        update_functions = {}
        replace_in_dict = {}
        kwargs = copy.copy(kwargs)
        # NB: We specifically don't recurse into kwargs for submodels:
        #     the update_function specs for submodels need to remain in kwargs
        #     until they reach their submodel, so that it can attach it correctly
        for attr, v in kwargs.items():
            # Deals with the case where we initialize from a .dict() export
            if isinstance(v, dict) and 'update_function' in v:
                update_functions[f"{attr}.update_function"] = v['update_function']
                update_functions[f"{attr}.range_update_function"] = v.get('range_update_function', None)
                replace_in_dict[attr] = copy.copy(v)
                replace_in_dict[attr]['update_function'] = None
                replace_in_dict[attr]['range_update_function'] = None
        # We do it this way to avoid mutating the kwargs
        for attr, v in replace_in_dict.items():
            kwargs[attr] = v
        # Initialize attributes with Pydantic
        super().__init__(**kwargs)
        # Attach update functions to histories, and set up __slots__
        self._base_initialize(update_functions=update_functions)
        # Run the model initializer
        if not isinstance(initializer, str) or initializer != 'do not initialize':
            self.initialize(initializer)

    def copy(self, *args, deep=False, **kwargs):
        m = super().copy(*args, deep=deep, **kwargs)
        m._base_initialize(shallow_copy=not deep)
        # m.initialize()
        return m

    def dict(self, *args, exclude=None, **kwargs):
        # Remove pending update functions from the dict – they are only used
        # to pass information from the metaclass __new__ to the class __init__,
        # and at this point already attached to the histories. Moreover, they
        # are already included in the serialization of HistoryUpdateFunctions
        exclude = add_exclude_mask(
            exclude,
            {attr for attr, value in self.__dict__.items()
             if isinstance(value, PendingUpdateFunction)}
        )
        # When serialized, HistoryUpdateFunctions include the namespace.
        # Remove this, since it is the same as `self`, and inserts a circular
        # dependence.
        hists = {attr: hist for attr,hist in self.__dict__.items()
                 if isinstance(hist, History)}
        # TODO: Any way to assert the integrity of the namespace ? We would
        #       to execute the assertion below, but during a shallow copy, a
        #       2nd model is created with the same histories; since `namespace`
        #       can't point to both, the assertion fails.
        # assert all(h.update_function.namespace is self
        #            for h in hists.values())
        excl_nmspc = {attr: {'update_function': {'namespace'}}
                      for attr in hists}
        # Remove connected objects which are actually part of another submodel
        excl_conn = {conn_obj: ... for conn_obj in self.connections.values()}
        exclude = add_exclude_mask(exclude, {**excl_nmspc, **excl_conn})
        # Proceed with parent's dict method
        obj = super().dict(*args, exclude=exclude, **kwargs)
        # Add the model name
        obj['ModelClass'] = mtb.iotools.find_registered_typename(self)
        
        return obj

    @classmethod
    def parse_obj(cls, obj):
        # Add `initializer` keyword arg to skip initialization
        if 'initializer' not in obj:
            obj['initializer'] = 'do not initialize'
        m = super().parse_obj(obj)
        return m

    def set_submodel_params(self):
        """
        Keep submodel parameters in sync with the container model.
        This method has no effect if there are no submodels.
        
        This method has the following effect (`submodel.params` is the
        parameter set of the submodel, and `model.subparams` is the
        corresponding subset of parameters of the container model)::
        
            params_values = submodel.Parameters(model.subparams).get_values()
            submodel.params.set_values(param_values)
            model.subparams = submodel.params
        
        The reasoning is as follows:
        
        - We normally set parameter values with `model.params.set_values(value
          dict)`. The values then need to be propagated to the matching
          parameters of submodels.
        - submodel.Parameters may implement validation or normalization, which
          the model expects to be pre-applied to its parameter set.
        - Theano graphs will depend submodel.params, so
          a) we don't want to change the identity of those variables
          b) we want `model.subparams` and `submodel.params` to point to the
             same instances, so that either can be used in a graph.
        """
        for submodelname, submodel in self.nested_models.items():
            subparams = getattr(self.params, submodelname)
            # Apply submodel's Parameters validator
            subparam_vals = submodel.Parameters.parse_obj(subparams).get_values()
            # Transfer parameter values to submodel
            submodel.params.set_values(subparam_vals)
            # Replace container parameter by corresponding instances of submodel
            for subθname, θ in submodel.params:
                assert subθname in subparams.__dict__, (
                    f"Subparameter set for submodel '{submodelname}' does not contain a parameter '{subθname}'.")
                setattr(subparams, subθname, θ)
                # Update the variable name to include the parent model.
                # (with a guard in case we run this twice)
                if hasattr(θ, 'name') and submodelname not in θ.name:
                    # (There are ways the test above can fail (e.g. if the 
                    # parameter's name is the same as the submodel's), but
                    # those seem quite unlikely.
                    θ.name = submodelname + "." + θ.name

    def _base_initialize(self,
                         shallow_copy: bool=False,
                         update_functions: Optional[dict]=None):
        """
        Collects initialization that should be done in __init__, copy & parse_obj.

        Both arguments are meant for internal use and documented with comments
        in the source code.
        """
        self.set_submodel_params()
        
        if update_functions is not None:
            # 1) update_functions should be a dict, and will override update function
            # defs from the model declaration.
            # This is used when deserializing a model; not really intended as a
            # user-facing option.
            # 2) Add @updatefunction decorator to those recognized by function deserializer
            # To avoid user surprises, we cache the current state of
            # HistoryUpdateFunction._deserialization_locals, update the variable,
            # then return to the original state once we are done
            # Remark: For explicitly passed update functions, we don't use the
            #    'PendingUpdateFunction' mechanism, so in fact
            #    we just replace @updatefunction by an idempotent function.
            def idempotent(hist_nm, inputs=None):
                def dec(f):
                    return f
                return dec
            # Stash _deserialization_locals
            stored_locals = HistoryUpdateFunction._deserialization_locals
            # Insert substitute @updatefunction decorator
            if 'updatefunction' not in HistoryUpdateFunction._deserialization_locals:
                HistoryUpdateFunction._deserialization_locals = HistoryUpdateFunction._deserialization_locals.copy()
                HistoryUpdateFunction._deserialization_locals['updatefunction'] = idempotent
            # Attach all explicitly passed update functions
            for upd_fn_key, upd_fn in update_functions.items():
                if upd_fn is None:
                    continue
                hist_name, method_name = upd_fn_key.rsplit('.', 1)
                hist = getattr(self, hist_name)
                ns = upd_fn.get('namespace', self)
                if ns is not self:
                    raise ValueError(
                        "Specifying the namespace of an update function is "
                        "not necessary, and if done, should match the model "
                        "instance where it is defined.")
                upd_fn['namespace'] = self
                if method_name == "update_function":
                    hist._set_update_function(HistoryUpdateFunction.parse_obj(upd_fn))
                elif method_name == "range_update_function":
                    hist._set_range_update_function(HistoryUpdateFunction.parse_obj(upd_fn))
                else:
                    raise ValueError(f"Unknown update function '{method_name}'. "
                                     "Recognized values: 'update_function', 'range_update_function'.")
            # Reset deserializaton locals
            HistoryUpdateFunction._deserialization_locals = stored_locals

        # Otherwise, create history updates from the list of pending update
        # functions created by metaclass (don't do during shallow copy – histories are preserved then)
        if not shallow_copy:
            for obj in self._pending_update_functions:
                if obj.hist_nm in update_functions:
                    # Update function is already set
                    continue
                hist = getattr(self, obj.hist_nm)
                hist._set_update_function(HistoryUpdateFunction(
                    namespace    = self,
                    func         = obj.upd_fn,  # HistoryUpdateFunction ensures `self` points to the model
                    inputs       = obj.inputs,
                    # parent_model = self
                ))
        self._pending_update_functions = {}
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
        # # Create symbolic variables for batches
        # if shim.cf.use_theano:
        #     # # Any symbolic function on batches should use these, that way
        #     # # other functions can retrieve the symbolic input variables.
        #     start = np.array(1).astype(self.tidx_dtype)
        #     object.__setattr__(self, 'batch_start_var',
        #                        shim.shared(start, name='batch_start'))
        #     #     # Must be large enough so that test_value slices are not empty
        #     # size = np.array(2).astype(self.tidx_dtype)
        #     # object.__setattr__(self, 'batch_size_var',
        #     #                    shim.shared(size, name='batch_size'))
        #     # #     # Must be large enough so that test_value slices are not empty
        
        # Set the value of `connections` so that it reflects actual connections
        # TODO: Put in a root_validator (requires `nested_models` which works as class method)
        already_seen = {}  # Keeps track of histories that have been seen
        conn_objs = {}     # Connection pairs for objects in different submodels
                           # Order: lower model.obj name : upper model.obj name,
                           # where the "lower" model is the one which can be instantiated first
        # NB: The order in which we iterate over nested_models is set by  
        #     the order in which submodels are defined in the class
        for subnm, submodel in self.nested_models.items():
            for objnm, obj in ChainMap(submodel.nested_histories,
                                       submodel.nested_rngs      ).items():
                if id(obj) in already_seen:
                    conn_objs[already_seen[id(obj)]] = f"{subnm}.{objnm}"
                else:
                    already_seen[id(obj)] = f"{subnm}.{objnm}"
        self.connections = conn_objs

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
            if not all(isinstance(output, History) for output in outputs):
                raise AssertionError("Model.output_rng: not all listed outputs "
                                     f"are histories.\nOutputs: {outputs}.")
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
        name = self.name
        return "Model '{}' (t0: {}, tn: {}, dt: {})" \
            .format(name, self.t0, self.tn, self.dt)

    def __repr__(self):
        return f"<{str(self)}>"

    def _repr_html_(self):
        summary = self.get_summary()
        topdivline = '<div style="margin-bottom: 0; border-left: solid black 1px; padding-left: 10px">'
        modelline = f'<p style="margin-bottom:0">Model <b>{summary.model_name}</b>'
        if isinstance(self, Model):  # Eventually we want to be able to be able to call this on the class
            modelline += f' (<code>t0: {self.t0}</code>, <code>tn: {self.tn}</code>, <code>dt: {self.dt}</code>)'
        modelline += '</p>'
        topulline = '<ul style="margin-top: 0;">'
        stateline = '<li>State variables: '
        stateline += ', '.join([f'<code>{v}</code>' for v in summary.state_vars])
        stateline += '</li>'
        paramblock = '<li>Parameters\n<ul>\n'
        if isinstance(summary.params, IndexableNamespace):
            for name, val in summary.params:
                paramblock += f'<li><code> {name}={val}</code></li>\n'
        else:
            for name in summary.params:
                paramblock += f'<li><code> {name}</code></li>\n'
        paramblock += '</ul>\n</li>'
        updfnblock = ""
        for varname, upd_code in summary.update_functions.items():
            updfnblock += f'<li>Update function for <code>{varname}</code>\n'
            updfnblock += f'<pre>{upd_code}</pre>\n</li>\n'
        if not summary.nested_models:
            nestedblock = ""
        else:
            nestedblock = '<li>Nested models:\n<ul>'
            for nestedmodel in summary.nested_models:
                nestedblock += f'<li style="margin-bottom:.6em">{nestedmodel._repr_html_()}</li>'
            nestedblock += '</ul></li>\n'

        return "\n".join([
            topdivline, modelline, topulline, stateline,
            paramblock, updfnblock, nestedblock,
            "</ul>", "</div"
        ])

    def __getattribute__(self, attr):
        """
        Retrieve parameters if their name does not clash with an attribute.
        """
        # Use __getattribute__ to maintain current stack trace on exceptions
        # https://stackoverflow.com/q/36575068
        if (not attr.startswith('_')      # Hide private attrs and prevent infinite recursion with '__dict__'
            and attr != 'params'          # Prevent infinite recursion
            and attr not in dir(self)     # Prevent shadowing; not everything is in __dict__
            # Checking dir(self.params) instead of self.params.__fields__ allows
            # Parameter classes to use @property to define transformed parameters
            and hasattr(self, 'params') and attr in dir(self.params)):
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

        # Return attributes from nested models
        elif '.' in attr:
            submodel, subattr = attr.split('.', 1)
            return getattr(getattr(self, submodel), subattr)

        else:
            return super().__getattribute__(attr)

    # Pickling infrastructure
    # Reference: https://docs.python.org/3/library/pickle.html#pickle-state
    def __getstate__(self):
        raise NotImplementedError
        # # Clear the nonstate histories: by definition these aren't necessary
        # # to recover the state, but they could store huge intermediate data.
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
    def name(self) -> str:
        return getattr(self, '__name__', type(self).__name__)

    @property
    def nonnested_histories(self) -> Dict[str, History]:
        return {nm: getattr(self, nm) for nm in self._hist_identifiers}
    @property
    def nonnested_history_set(self) -> Set[History]:
        return {getattr(self, nm) for nm in self._hist_identifiers}
    @property
    def nested_histories(self) -> Dict[str,History]:
        """
        Return the model's histories as {name: History} pairs, with the names
        of nested histories prefixed by the their submodel name: "{submodel}.{name}".
        """
        return {**self.nonnested_histories,
                **{f"{submodel_nm}.{hist_nm}": hist
                   for submodel_nm, submodel in self.nested_models.items()
                   for hist_nm, hist in submodel.nested_histories.items()}}
    @property
    def history_set(self) -> Set[History]:
        return set(chain(self.nonnested_history_set,
                         *(m.history_set for m in self.nested_models_list)))
    @property
    def nonnested_kernels(self) -> Dict[str, Kernel]:
        return {nm: getattr(self, nm) for nm in self._kernel_identifiers}
    @property
    def nonnested_kernel_list(self) -> List[Kernel]:
        return [getattr(self, nm) for nm in self._kernel_identifiers]
    @property
    def kernel_list(self) -> List[Kernel]:
        return list(chain(self.nonnested_kernel_list,
                          *(m.kernel_list for m in self.nested_models_list)))
    @property
    def nested_models(self) -> Dict[str, Model]:
        return {nm: getattr(self, nm) for nm in self._model_identifiers}
    @property
    def nested_models_list(self) -> List[Model]:
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
    def statehists(self) -> utils.FixedGenerator:
        nested_len = sum(len(m.statehists) for m in self.nested_models_list)
        return utils.FixedGenerator(
            chain(
                (getattr(self, varname) for varname in self.State.__annotations__),
                *(m.statehists for m in self.nested_models_list)),
            nested_len + len(self.State.__annotations__) )

    @property
    def unlocked_statehists(self) -> Generator[History]:
        return (h for h in self.statehists if not h.locked)

    @property
    def locked_statehists(self) -> Generator[History]:
        return (h for h in self.statehists if h.locked)

    @property
    def locked_histories(self) -> Generator[History]:
        return (h for h in self.history_set if h.locked)

    @property
    def unlocked_histories(self) -> Generator[History]:
        return (h for h in self.history_set if not h.locked)

    @property
    def nonstatehists(self) -> utils.FixedGenerator:
        statehists = list(self.statehists)
        return utils.FixedGenerator(
            (h for h in self.history_set if h not in statehists),
            len(self.history_set) - len(self.statehists) )

    @property
    def unlocked_nonstatehists(self):
        return (h for h in self.nonstatehists if not h.locked)

    @property
    def rng_inputs(self) -> List[mtb.typing.AnyRNG]:
        """
        Return a list of all RNGs on which _unlocked_ histories depend.
        RNGs which only appear as inputs for _locked_ histories are excluded;
        thus the returned list can usually be assumed to be those RNGs which
        would be triggered by integration, with the current lock statuses.
        """
        rng_inputs = []
        for h in self.unlocked_histories:
            for nm in h.update_function.input_names:
                inp = getattr(h.update_function.namespace, nm)
                if (isinstance(inp, shim.config.RNGTypes)
                    and inp not in rng_inputs):
                    rng_inputs.append(inp)
        return rng_inputs
        
    @property
    def nonnested_rngs(self) -> Dict[str, mtb.typing.AnyRNG]:
        d = {}
        for attr in self.__fields__:
            obj = getattr(self, attr, None)
            if isinstance(obj, shim.config.RNGTypes):
                d[attr] = obj
        return d
    @property
    def nested_rngs(self) -> Dict[str, mtb.typing.AnyRNG]:
        return {**self.nonnested_rngs,
                **{f"{submodel_nm}.{rng_nm}": rng
                   for submodel_nm, submodel in self.nested_models.items()
                   for rng_nm, rng in submodel.nested_rngs.items()}}

    @property
    def rng_hists(self) -> List[History]:
        """
        Return a list of stochastic histories (those with an RNG as input.)
        """
        rng_hists = []
        for h in self.unlocked_histories:
            for nm in h.update_function.input_names:
                inp = getattr(h.update_function.namespace, nm)
                if isinstance(inp, shim.config.RNGTypes):
                    rng_hists.append(h)
                    break
        return rng_hists

    def get_min_tidx(self, histories: Sequence[History]):
        """
        Return the earliest time index for which all histories are computed.
        """
        try:
            return min(h.cur_tidx.convert(self.time.Index)
                       for h in histories)
        except IndexError as e:
            raise IndexError(
                "Unable to determine a current index for "
                f"{self.name}. This usually happens accessing "
                "`cur_tidx` before a model is initialized.") from e

    @property
    def cur_tidx(self):
        """
        Return the earliest time index for which all state histories are computed.
        """
        if not self.statehists:
            raise RuntimeError("`cur_tidx` is undefined for a model with no "
                               "state histories, since any time point can be "
                               "computed at any time.\nIf you need an anchor "
                               "time point for building a computational graph, "
                               "use `num_tidx` instead.")
        return self.get_min_tidx(self.statehists)

    @property
    def cur_t(self):
        """
        Returns the time up to which all state histories are computed computed.
        Equivalent to `self.time[self.cur_tidx]`.
        """
        return self.time[self.cur_tidx]

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

    def get_num_tidx(self, histories: Sequence[History]):
        """
        A shared variable corresponding to the current time point of
        the model. This is only defined if all histories are synchronized.
        (*Locked* histories need not be synchronized, but must be computed at
        least as far as unlocked histories.)
        
        Special case: if all listed histories are locked (or `histories` is
        empty), then a tidx corresponding to `self.t0idx` is returned.
        Rationale: the time point of locked histories is meaningless, since
        they cannot be updated, and it makes no sense to force them to be
        synchronized. The only thing we expect is that whatever value is
        returned by `num_tidx` is already computed for all locked histories.
        Since the purpose of this method is to return a time index suitable
        for constructing a graph, if there are no histories to update, the time
        point itself does not matter. Thus we return the index corresponding to
        t0, since this requires the least data.

        Always returns the same object, so that it can be substituted in
        computational graphs.

        Raises `RuntimeError` if histories are not all synchronized.

        .. WARNING::
           This does not return an AxisIndex, so always wrap this variable
           with [model].time.Index or [model].time.Index.Delta before using
           it to index into a history.
           
        .. Warning::
           Each computational graph should only involve calls to `get_num_tidx`
           with the same list of histories. Different lists of histories
           may return different objects. Different objects will also be
           returned if the lock status of some histories changes.

        .. Dev note::
           If we can find a way to make SymbolicAxisIndexMeta._instance_plain
           return the original underlying symbolic variable (the one which
           appears in graphs), I will gladly change this method to return a
           proper symbolic index.

        Parameters
        ----------
        histories: Set of histories
        """
        # TODO: Since we assert that all unlocked state histories are
        # synchronized, I'm not sure how useful it is to also allow `histories`
        # to be specified (we could just always use unlocked state hists)
        if not self.histories_are_synchronized():
            L = max(len(hnm) for hnm in self.nested_histories)
            tidcs = "\n".join(f"{hnm:<{L}}: {h.cur_tidx.convert(self.time.Index)}"
                              for hnm, h in self.nested_histories.items()
                              if not h.locked)
            raise RuntimeError(
                f"Unlocked histories for the {self.name} model are not all "
                "computed up to the same point, or further than some locked "
                "histories. The compilation of the model's integration function "
                "is ill-defined in this case.\nTime indexes for unlocked "
                f"histories (converted to model's time axis):\n{tidcs}")
        unlocked_histories = [h for h in histories if not h.locked]
        if unlocked_histories:
            key = tuple(id(h) for h in unlocked_histories)
            tidx = self.get_min_tidx(unlocked_histories)
        else:
            # DEV NOTE: I put this restriction here because all current use cases
            #    for `num_tidx` are to index into the history, hence we need
            #    a value at which the histories are computed. But theoretically
            #    it could also return a negative index if nothing is computed,
            #    like History.cur_tidx does. The issue then is what is the most
            #    sensible thing to do if histories have different left padding.
            _locked_histories = list(self.locked_histories)  # Consume generator
            if _locked_histories:
                assert self.get_min_tidx(_locked_histories) >= self.t0idx, \
                    "`get_num_tidx` requires histories be computed at least up to `t0idx`"
            key = ()
            tidx = self.t0idx
        if key not in self._num_tidx_objs:
            self._num_tidx_objs[key] = shim.shared(
                np.array(tidx, dtype=self.tidx_dtype), f"t_idx ({self.name})")
        else:
            self._num_tidx_objs[key].set_value(tidx)
        return self._num_tidx_objs[key]

    @property
    def num_tidx(self):
        """
        Return a shared variable suitable to use as an anchor time index for
        building computational graphs.
        """
        return self.get_num_tidx(self.statehists)

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
    def get_summary(self, hists=None):
        summary = SimpleNamespace()
        summary.model_name = self.name
        State = getattr(self, 'State', None)
        if State:
            summary.state_vars = list(State.__annotations__)
        else:
            summary.state_vars = []
        if isinstance(self, type):
            Parameters = getattr(self, 'Parameters', None)
            if Parameters:
                summary.params = list(Parameters.__annotations__)
            else:
                summary.params = []
        else:
            params = getattr(self, 'params', None)
            if params:
                summary.params = params.get_values()
            else:
                summary.params = {}
        summary.update_functions = self.get_update_summaries(hists)
        summary.nested_models = list(self.nested_models.values())
        return summary

    @class_or_instance_method
    def get_update_summaries(self, hists=None) -> Dict[str,str]:
        """
        For selected histories, return a string summarizing the update function.
        By default, the histories summarized are those of `self.state`.
        May be called on the class itself, or an instance.

        Parameters
        ----------
        hists: list | tuple of histories or str
            List of histories to summarize.
            For each history given, retrieves its update function.
            Alternatively, a history's name can be given as a string

        Returns
        -------
        Dict[str,str]: {hist name: hist upd code}
        """
        # Default for when `hists=None`
        if hists is None:
            State = getattr(self, 'State', None)
            if State:
                hists = list(self.State.__annotations__.keys())
            else:
                hists = []        # Normalize to all strings; take care that identifiers may differ from the history's name
        histsdict = {}
        nonnested_hists = getattr(self, 'nonnested_histories', {})
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
        srcs = {}
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
            srcs[hist_desc] = src.split('\n', 1)[1]
                # 'split' is used to replace the `def` line: callers use
                # the `hist_desc` value to create a more explicit string
        return srcs

    @class_or_instance_method
    def summarize(self, hists=None):
        nested_models = self._model_identifiers
        if isinstance(self, type):
            nameline = "Model '{}'".format(self.name)
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
        updateblock = '\n\n'.join([
            f"Update function for {hist_desc}:\n" + upd_src
            for hist_desc, upd_src in self.get_update_summaries(hists).items()])
        summary = (nameline + stateline + paramline + nestedline
                   + '\n' + updateblock)
        return '\n'.join((summary, *nested_summaries))

    def get_tidx(self, t, allow_rounding=False):
        # Copied from History.get_tidx
        if self.time.is_compatible_value(t):
            return self.time.index(t, allow_rounding=allow_rounding)
        else:
            # assert self.time.is_compatible_index(t)
            assert shim.istype(t, 'int')
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
        # NOTE: Copy changes to this function in Model.get_time()
        # TODO: Is it OK to enforce single precision ?
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

    get_time.__doc__ = History.get_time.__doc__

    ###################
    # Methods which modify the model

    def lock(self):
        for hist in self.history_set:
            hist.lock()
    def clear(self,after=None):
        """
        Invalidate the model data, forcing histories to be recomputed the next
        time they are queried.
        Functionally equivalent to clearing the data, keeping the padding.
        Discards symbolic updates by calling `shim.reset_updates()` and
        `~History.theano_reset()`.

        Parameters
        ----------
        after: AxisIndex
            If given, history will only be cleared after this point.
            `cur_tidx` will be set to `after`, rather than `t0idx-1`.
            """
        if shim.is_symbolic(after):
            raise TypeError(
                "`clear` requires a numeric (not symbolic) time index.")
        shim.reset_updates()
        if after is not None:
            after = self.time.Index(after)
            for hist in self.unlocked_histories:
                hist.clear(after=after.convert(hist.time.Index))
        else:
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

    def theano_state_is_clean(self):
        if shim.pending_updates():
            return False
        for hist in self.unlocked_histories:
            # if (hist._num_tidx is not hist._sym_tidx
            #     or hist._num_data is not hist._sym_data):
            if hist._taps:
                return False
        for rng in self.rng_inputs:
            if (isinstance(rng, shim.config.SymbolicRNGTypes)
                and len(rng.state_updates) > 0):
                return False
        return True

    def theano_reset(self, warn_rng=True):
        """
        Put model back into a clean state, to allow building a new Theano graph.

        .. warning:: If there are dependencies on RNGs, those will be removed.
           This is necessary to avoid having disconnected RNGs appear in the
           symbolic graph. For NumPy RNGs, this is inconsequential.
           However, for Theano RNGs, this means that `self.rng.seed(0)`
           **will not set the simulator in a predictable state**. The reason
           is that when using Theano, `self.rng` is in fact a *collection* of
           RNGs. When we remove an RNG from this collection, `self.seed` then
           has no way of knowing it should reset it.

           The method `self.reseed_rngs` is provided to reseed the RNGs created
           and disconnected by `get_advance_updates`.

        :param warn_rng: If True (default), emit a warning if updates to a
            random number generator were cleared.

        **Side-effects**: Clears all shim symbolic updates in shim.
        """
        shim.reset_updates()

        for hist in self.unlocked_histories:
            hist.theano_reset()
        for kernel in self.kernel_list:
            kernel.theano_reset()

        for rng in self.rng_inputs:
            if (isinstance(rng, shim.config.SymbolicRNGTypes)
                and len(rng.state_updates) > 0 and warn_rng):
                rng_name = getattr(rng, 'name', str(rng))
                if rng_name is None: rng_name = str(rng)
                warn("Erasing random number generator updates. Any "
                     "other graphs using this generator are likely "
                     "invalidated.\n"
                     "RNG: {}".format(rng))
            rng.state_updates = []

    def reseed_rngs(self, seed):
        """
        Reset all model RNGs into a predictable state.
        """
        # TODO?: Allow resetting only certain entries in _advance_updates ?
        #        Or: Option for each entry to be seeded the same way ?
        rngs = {id(rng): rng for rng in self.rng_inputs}
            # Dictionary allows multiple submodels to have an RNG, as long
            # as it is shared.
        if len(rngs) == 0:
            logger.warning(f"Model {self} has no RNGs; nothing to reseed.")
        if len(rngs) > 1:
            raise NotImplementedError("`reseed_rngs` doesn't support models "
                                      "with multiple random number generators.")
        rng = next(iter(rngs.values()))
        cur_state_updates = getattr(rng, 'updates', None)
        if cur_state_updates:
            cur_state_updates = cur_state_updates()
        # Each compiled function maps to a different cache_key in _rng_updates
        # Iterate over each set of updates and reseed the RNGs
        if len(self._rng_updates) > 1:
            warn("Support for multiple compiled integration functions is still "
                 "WIP. For reliably deterministic reseeding of RNGs, ensure "
                 "only one function is compiled.")
        _num_updates = None
        for update_stash in self._rng_updates.values():
            stashed_state_updates = update_stash.get(rng, [])
            if cur_state_updates is None and stashed_state_updates:
                raise RuntimeError(f"The model {self.name} seems to have recorded "
                                   "Theano state updates, but uses a plain NumPy "
                                   "RNG. This is likely a bug in sinn.Model")
            elif stashed_state_updates:
                # We have a Theano RNG, either RandomStream or MRG
                # We know cur_state_updates is a list, because we didn't raise RuntimeError
                # Assumption: Any stashed state updates should come before current ones
                # (in fact, `cur_state_updates` should probably always be empty)
                rng.state_updates = stashed_state_updates + cur_state_updates
            if _num_updates is None:
                _num_updates = len(rng.state_updates)
            elif _num_updates != len(rng.state_updates):
                warn("There are multiple compiled integration functions, and they "
                     "update the RNG state a different number of times. This "
                     "will still work, in the sense that functions the the model "
                     "can be integrated without errors. However, we have not "
                     "confirmed that result will be is entirely deterministic: "
                     "they may depend on the order in which the functions are "
                     "compiled, or whether another function is compiled or not.")
            # Having reattached state updates, rng.seed will update them all
            shim.reseed_rng(rng, seed)
            # Detach the state updates to return the graph in the state it was before
            if stashed_state_updates:
                rng.state_updates = rng.state_updates[len(stashed_state_updates):]
                    # NB: `reseed_rng` has changed the state updates, so we can't
                    #     just reuse cur_state_updates.
            
        # The code below did not require stashing RNG updates, but only works
        # with RandomStream (MRG updates cannot as easily be identified in
        # the symbolic updates, because they appear as complex expressions
        # rather than special RNG types)
        # # Find all the Theano RNG streams and reseed them.
        # # This is analogous to what `seed` does with the streams listed in `rng.state_updates`
        # srs = sys.modules.get('theano.tensor.shared_RandomStream', None)
        # if srs is not None:
        #     seedgen = rng.gen_seedgen  # Was seeded by `reseed_rng` above
        #     for update_dict in self._advance_updates.values():
        #         for v in update_dict:
        #             if isinstance(v, srs.RandomStateSharedVariable):
        #                 old_r_seed = seedgen.randint(2 ** 30)
        #                 v.set_value(np.random.RandomState(int(old_r_seed)),
        #                             borrow=True)

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
            {**self.params.dict(), **new_params}
            ).get_values()
            # Calling `Parameters` validates all new parameters but converts them to shared vars
            # Calling `get_values` converts them back to numeric values
            # TODO: Simply cast as NumericModelParams
            # Wait until kernels have been cached before updating model params

        # TODO: Everything with old_ids & clear_advance_fn should be deprecatable
        # We don't need to clear the advance function if all new parameters
        # are just the same Theano objects with new values.
        old_ids = {name: id(val) for name, val in self.params}
        # clear_advance_fn = any(id(getattr(self.params, p)) != id(newp)
        #                        for p, newp in pending_params.dict().items())

        # Determine the kernels for which parameters have changed
        kernels_to_update = []
        for kernel in self.kernel_list:
            if set(kernel.__fields__) & set(new_params.keys()):
                kernels_to_update.append(kernel)

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
                            # FIXME: Should use newer mtb.utils.stablehash
                            obj = op.cache[hash(kernel)]
                            diskcache.save(str(hash(obj)), obj)
                            # TODO subclass op[other] and define __hash__
                            logger.monitor("Removing cache for binary op {} ({},{}) from heap."
                                        .format(str(op), obj.name, kernel.name))
                            del op.cache[hash(kernel)]

        for kernel in kernels_to_update:
            # FIXME: Should use newer mtb.utils.stablehash
            diskcache.save(str(hash(kernel)), kernel)
            kernel.update_params(**pending_params.dict())

        # Update self.params in place
        self.params.set_values(pending_params)
        self.set_submodel_params()

        # NB: I think the only way params can change identity if if we have
        #     a nested model and `set_submodel_params` reassigns them.
        clear_advance_fn = any(id(val) != old_ids[name]
                               for name, val in self.params)

        logger.debug("Model params are now {}.".format(self.params))

        self.clear()
        if clear_advance_fn:
            # Only clear advance function when necessary, since it forces a
            # recompilation of the graph.
            # We need to do this if any of the parameters change identity (e.g.
            # replaced by another shared variable).
            self._compiled_advance_fns.clear()

    # FIXME: This function was originally written when the update dictionary
    # would also include all symbolic updates to histories (through _sym_tidx
    # and _sym_data).
    # It also does not seem used (or at least is not covered by a test).
    # We should see if it appears in usage, and if so, make a test for it.
    # Otherwise removal is probably advisable.
    def eval_updates(self, givens=None):
        """
        Compile and evaluate a function evaluating the `shim` update
        dictionary. Histories' internal variables tracking symbolic updates
        are unused and untouched.
        If the updates have symbolic inputs, provide values for them through
        the `givens` argument.
        If there are pending symbolic updates, no function is compiled, so you
        can use this as a safeguard at the top of a function to ensure there are
        no unapplied updates, without worrying about the cost of repeated calls.
        """
        warn("The way symbolic updates are tracked has changed, but since the "
             "method `eval_updates` had no clear use case, it was not updated. "
             "Please report your use case, so we can determine whether this "
             "method is still relevant. In the affirmative, adding a test for "
             "it would also be nice.")
        # TODO: Should this function also evaluate symbolic updates in _taps ?
        # Depends what it's meant for. (See also eval(), which would do something
        # similar without the non-tap updates.)
        upds = shim.get_updates()
        if len(upds) > 0:
            f = shim.graph.compile([], [], updates=upds, givens=givens)
            f()
            # for h in self.history_set:
            #     if h._sym_tidx != h._num_tidx:
            #         object.__setattr__(h, '_sym_tidx', h._num_tidx)
            #     if h._sym_data != h._num_data:
            #         object.__setattr__(h, '_sym_data', h._num_data)

    # ==============================================
    # Model advancing code
    #
    # This code isn't 100% generic yet;
    # look for TODO tags for model-specific hacks
    #
    # Function overview:  (NEEDS UPDATE)
    # - integrate(self, stop): User-facing function. Synonyme: `advance`
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

    def integrate(self,
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
        upto: int, float, AxisIndex
            Compute history up to this point (inclusive).
            May also be the string 'end'
            If an AxisIndex, it must be compatible with ``self.time``.
        histories: Tuple of histories to integrate. State histories do not need to
            be included; they are added automatically.
            If only one history is specified, it doesn't need to be wrapped in
            a tuple.
            The value 'all' can be used to specify integrating all histories.
        """
        end = upto
        if histories == 'all':
            histories = tuple(h for h in self.unlocked_nonstatehists)
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
            if isinstance(end, AbstractAxisIndex):
                end = end.convert(self.time)
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
            # if not shim.graph.is_computable(
            #     [hist._sym_tidx for hist in self.statehists]):
            #     raise TypeError("Integrating models is only implemented for "
            #                     "histories with a computable current time "
            #                     "index (i.e. the value of `hist._sym_tidx` "
            #                     "must only depend on symbolic constants and "
            #                     "shared vars).")
            # if shim.pending_updates():
            #     raise RuntimeError("There "
            #         "are symbolic inputs to the already present updates:"
            #         "\n{}.\nEither discard them with `theano_reset()` or "
            #         "evaluate them with `eval_updates` (providing values "
            #         "with the `givens` argument) before advancing the model."
            #         .format(shim.graph.inputs(shim.get_updates().values())))
            curtidx = self.get_min_tidx(histories + tuple(self.statehists))
            assert(curtidx >= -1)

            if curtidx < endtidx:
                self._advance(histories)(curtidx, endtidx+1)
                # _advance applies the updates, so should get rid of them
                self.theano_reset()

    @property
    def no_histories_have_updates(self):
        """
        Return `True` if none of the model's histories have unevaluated
        symbolic updates.
        """
        # no_updates = all(h._sym_tidx is h._num_tidx
        #                  and h._sym_data is h._num_data
        no_updates = all(h._latest_tap == 0
                         for h in self.history_set)
        # if no_updates and len(shim.get_updates()) > 0:
        #     raise RuntimeError(
        #         "Unconsistent state: there are symbolic theano updates "
        #         " (`shim.get_updates()`) but none of the model's histories "
        #         "has a symbolic update.")
        # elif not no_updates and len(shim.get_updates()) == 0:
        #     hlist = {h.name: (h._sym_tidx, h._sym_data) for h in self.history_set
        #              if h._sym_tidx is not h._num_tidx
        #                 and h._sym_data is not h._num_data}
        #     raise RuntimeError(
        #         "Unconsistent state: some histories have a symbolic update "
        #         "({}), but there are none in the update dictionary "
        #         "(`shim.get_updates()`)".forma(hlist))
        return no_updates
        
    def _get_cache_key(self, histories=()):
        # The compiled function depends both on the histories we want to update,
        # and on which histories are locked.
        return (histories, tuple(sorted(self.locked_histories)))

    def _advance(self, histories=()):
        """
        Attribute which memoizes the compilation of the advance function.

        Parameters
        ----------
        histories: Set of histories to update. State histories do not need to
            be included; they are added automatically.
        """
        cache_key = self._get_cache_key(histories)
        if cache_key not in self._advance_updates:
            self._advance_updates[cache_key] = self.get_advance_updates(histories)
        _advance_updates = self._advance_updates[cache_key]
            # DEBUG
            # for i, s in enumerate(['base', 'value', 'start', 'stop']):
            #     self._advance_updates[self.V._num_data].owner.inputs[i] = \
            #         shim.print(self._advance_updates[self.V._num_data]
            #                    .owner.inputs[i], s + ' V')
            #     self._advance_updates[self.n._num_data].owner.inputs[i] = \
            #         shim.print(self._advance_updates[self.n._num_data]
            #                    .owner.inputs[i], s + ' n')
        if self.no_histories_have_updates:
            if cache_key not in self._compiled_advance_fns:
                logger.info("Compiling the update function")
                self._compiled_advance_fns[cache_key] = self.cached_compile(
                    [self.curtidx_var, self.stoptidx_var], [], _advance_updates)
                logger.info("Done.")
            _advance_fn = self._compiled_advance_fns[cache_key]
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

        # I think the point of this lambda is to allow keyword args
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
        cache_key = self._get_cache_key(histories)

        logger.info("Constructing the update graph.")
        # Stash current symbolic updates
        # for h in self.statehists:
        for h in self.history_set:
            h._stash()  # Stash unfinished symbolic updates
        updates_stash = shim.get_updates()
        assert len(self._rng_updates[cache_key]) == 0, "`sinn.Model._rng_updates should only be modified by `get_advance_updates`"
        shim.reset_updates()

        # Get advance updates
        updates = self.advance_updates(self.curtidx_var, self.stoptidx_var, histories)
        # Store the RNG updates so they can be used for reseeding RNGs
        for rng in self.rng_inputs:
            self._rng_updates[cache_key][rng] = rng.updates()
        # Reset symbolic updates to their previous state
        shim.reset_updates()
        self.theano_reset(warn_rng=False)
            # theano_reset() is half redundant with reset_updates(), but we
            # still need to reset the RNG updates
        # for h in self.statehists:
        for h in self.history_set:
            h._stash.pop()
        shim.config.symbolic_updates = updates_stash
        logger.info("Done constructing the update graph.")
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
            fn = shim.graph.compile(inputs, outputs, updates=updates, **kwargs)
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
        return self.convert_point_updates_to_scan(shim.get_updates(), curtidx, stoptidx-1, histories)

    def convert_point_updates_to_scan(self, updates=None, curtidx=None, stoptidx=None, histories=None):
        """
        Convert the current point-wise updates in the global shim update
        dictionary, and convert them to an scan graph with start and end given
        by ``curtidx + 1`` and `stoptidx`. The result is thus unanchored
        from the possible `_num_tidx` dependencies in the updates.

        .. Important:: This function will fail if there are neither pending
           symbolic updates on the model's histories nor entries in the
           `updates` dictionary. Moreover, these updates must depend on at least
           one history time index which can be related to the model's time
           index.
           
        .. Remark:: `update` dependencies across time steps are *integrable* but
           not *differentiable*. Example: since histories store their data as a
           shared variable `_num_data`, `updates` could in theory contain a pair
           {_num_data: updated data at t+1}. The `integrate` method in this case
           would work, since at each time step updates are applied as prescribed.
           However Theano would then not include `_num_data` when unrolling the
           graph for backpropagation.

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
        histories: Sequence[History]
            Default: `self.unlocked_statehists`
            The chosen anchor_tidx will allow each of these histories to be computed.

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
        # TODO: Accumulated values would be better treated as outputs rather
        #       than updates. Currently, we have:
        #           cost = 0; scan update: {cost += step_cost}  (defined in `updates`)
        #       Better would be:
        #           outs, upds = scan(…); cost = outs[0].sum()
        # TODO: Replace curtidx by startidx everywhere appropriate
        # Default values
        if updates  is None: updates = shim.get_updates()
        if curtidx  is None: curtidx = self.curtidx_var
        if stoptidx is None: stoptidx = self.stoptidx_var
        if histories is None: histories = tuple(self.unlocked_statehists)

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
        # NB: num_tidx normally returns the earliest cur_tidx among unlocked state histories
        #     If there are no state histories, or they are all locked, it returns self.t0idx
        #     In both cases, this should constitute an appropriate anchor time index,
        #     assuming that histories are synchronized (i.e. that all unlocked
        #     state histories have the same cur_tidx). This is checked when
        #     `num_tidx` is retrieved.
        anchor_tidx = self.num_tidx
            # `get_num_tidx` asserts that histories are synchronized
        if not self.time.Index(anchor_tidx+1).in_bounds:
            raise RuntimeError(
                "In order to compute a scan function, the model must "
                "have at least one uncomputed time point.")
        # Build the substitution dictionary to convert all history time indices
        # to that of the model. This requires that histories be synchronized.
        anchor_tidx_typed = self.time.Index(anchor_tidx)
        tidxsubs = {h._num_tidx: anchor_tidx_typed.convert(h.time).plain
                    for h in self.unlocked_histories}

        # Now we recover the global updates, and replace the multiple history
        # time indices by the time index of the model.
        output_hists = [h for h in self.history_set
                        if h._latest_tap > 0 and not h.locked]
        if len(output_hists) + len(updates) == 0:
            raise RuntimeError("No history has been updated symbolically and "
                               "the list of updates is empty. "
                               "Cannot build a scan graph.")
        if any(h._latest_tap > 1 for h in output_hists):
            problem_hists = ", ".join(f"{h.name} (Δk: {h._latest_tap})"
                                      for h in output_hists if h._latest_tap > 1)
            raise NotImplementedError(
                "`convert_point_updates_to_scan` requires that symbolic "
                "updates only look forward one time step.\nProblematic "
                f"histories: {problem_hists}")

        # Collect the information for the variables that are iterated forward
        # We split taps into forward/positive (> 0) and backward/negative (≤ 0).
        # Backward taps will become the `outputs_info` variable to scan.
        # Forward taps define the update function; we only allow one forward tap.
        # We only need the backward taps of histories which also have forward taps:
        # since the others don't change during iteration, we can keep indexing
        # directly into their `_num_data`.
        
        # Create the list of output variables for the scan step
        # Negative tap variables (which appear within the graph of positive 
        # tap variables) are substituted in by the `onestep` function below
        # `clone` replaces time index vars by expressions depending on the anchor
        # outputs = [shim.graph.clone(h[h._num_tidx+1], replace=tidxsubs)
        outputs = [shim.graph.clone(h._taps[h.time.Index.Delta(1)], replace=tidxsubs)
                   for h in output_hists]
        for o, h in zip(outputs, output_hists):
            o.name = f"{h.name}[k] (outside scan)"  # Corresponding var inside scan is named "{h.name}[k]"
        
        # Construct the outputs info from the negative taps.
        # NB: Initial data must be contiguous, even if negative taps are not,
        #     so taps [-3, -1] still require [k-3, k-2, k-1, k], where k is the
        #     current time index.
        # If there are no negative taps, we just use the current value as
        # initialization (equivalent to having a tap of 0).
        # IMPORTANT: In Theano's tap indexing, taps are relative _to the time
        #    point **being** computed_, while in Sinn, they are relative _to the
        #    lastest **computed** time point_. This means that we need to
        #    subtract 1 from each tap to construct `outputs_info`.
        outputs_info = []
        output_taps_replace = []  # Flattened list of taps. Used to construct the
                                  # substitution dict in onestep
        tap_names = []  # Names to assign to tap variables. This must be done
                        # inside `onestep`, when the tap variables are accessible
        for h in output_hists:
            negtaps = [tapk for tapk in sorted(h._taps) if tapk <= 0]
            if len(negtaps) == 0:
                # History is not needed to evaluate any graph. 
                # In theory there should be no need for an init val and the
                # line below should be the most appropriate
                # > outputs_info.append({'taps': None})
                # However, scan doesn't behave quite correctly with tapless variables
                # (see https://github.com/aesara-devs/aesara/issues/500), so
                # we use the workaround of defining dummy variables.
                init_val = shim.ones(h.shape, dtype=h.dtype, symbolic=True)
                output_taps_replace.append(init_val)
                init_val.name = f"{h.name}[scan init, dummy]"
                tap_names.append(f"{h.name}[k-1]")
                outputs_info.append({'initial': init_val, 'taps': [-1]})
            else:
                scantaps = [k.plain-1 for k in negtaps]
                init_length = -min(scantaps)
                assert init_length > 0, "create scan -> output_info -> init length must be positive"
                # NB: This will add any intermediate tap to the history's _taps;
                #     I'm undecided on whether this is desired or not.
                # NB: We take care here that the instances in output_taps_replace
                #     are the same ones as in init_vals
                # init_vals = [shim.graph.clone(h[h._num_tidx+Δk], replace=tidxsubs)
                init_vals = [shim.graph.clone(h._taps[h.time.Index.Delta(Δk)], replace=tidxsubs)
                    # NB: The use of _taps[…] is temporary, to help detect bugs (almost always, all previous taps should already have been calculated)
                             for Δk in range(-init_length+1, 1)]
                if scantaps == [-1]:
                    # To avoid requiring extra dimensions when no taps are used,
                    # Theano special-cases the case with only the -1 tap and
                    # removes the outer dimension.
                    init_val = init_vals[0]
                else:
                    init_val = shim.stack(init_vals)
                    init_val.name = f"{h.name}[scan init]"
                # init_val = init_vals  # DEBUG
                init_tap_idx = [k+init_length for k in scantaps]
                assert init_tap_idx[0] == 0, "Bug in index arithmetic for scan outputs_info"
                output_taps_replace.extend([init_vals[Δk] for Δk in init_tap_idx])
                # Add names for debugability
                for iv, Δk in zip(init_vals, scantaps):
                    iv.name = f"{h.name}[k-{abs(Δk)}] (outside scan)"
                tap_names.extend(f"{h.name}[k-{abs(Δk)}]" for Δk in scantaps)
                # Add to outputs_info
                outputs_info.append({'initial': init_val, 'taps': scantaps})
        
        # Replace all time indices in updates by expressions depending on anchor
        updates = {k: shim.graph.clone(g, replace=tidxsubs)
                   for k,g in updates.items()}
        # Ensure that we actually do have a time dependence somewhere in the
        # graph we send to scan
        all_exprs = chain(outputs, output_taps_replace, updates.values())    
        if not set(shim.graph.symbolic_inputs(all_exprs)) & set([anchor_tidx]):
                # & (set(h._num_tidx for h in self.unlocked_histories)
                #    | set([anchor_tidx])) ):
            raise RuntimeError("The updates dictionary has no dependency on "
                               "any time index. Cannot build a scan graph.")

        # Grab the list of shared vars to pass as non_sequences
        # NB: Providing the list of shared variables is not required, but
        #    supposedly helps keep the graph cleaner and more efficient – which
        #    might be especially beneficial with complex history update functions.
        #    (https://theano-pymc.readthedocs.io/en/latest/library/scan.html?highlight=scan#using-shared-variables-gibbs-sampling)
        #    I don't know how much of a difference this makes in practice, since 
        #    I generally don't see a difference in the output of debug print.
        shared_vars_in_graph = [
            v for v in shim.graph.shared_inputs(updates.values())
            if v is not anchor_tidx]  # anchor_tidx is replaced inside onestep
        n_shared_vars = len(shared_vars_in_graph)
            
        # Now we build the step function that will be passed to `scan`.
        def onestep(tidx, *args,
                    outputs=outputs, output_taps_replace=output_taps_replace,
                    updates=updates, output_hists=output_hists, tap_names=tap_names,
                    n_shared_vars=n_shared_vars):
            assert len(args) == len(output_taps_replace) + n_shared_vars, \
                "Unexpected number of scan arguments. Unpacking relies on alignment of `args`, and thus will almost certainly fail"
            # Construct the replace dict to substitute the scan vars passed in *args
            output_taps = args[:len(output_taps_replace)]
            taps_replace_dict = {orig: new for orig, new
                                 in zip(output_taps_replace, output_taps)}
            replace_dict = {anchor_tidx: tidx, **taps_replace_dict}
            # Construct the output by substituting the scan args
            step_outputs = [shim.graph.clone(o, replace=replace_dict)
                            for o in outputs]
            step_updates = OrderedDict(
                (k, shim.graph.clone(g, replace=replace_dict))
                for k,g in updates.items())
            # Add var names for easier debugging
            assert len(output_taps) == len(tap_names)
            for o, h in zip(step_outputs, output_hists):
                o.name = f"{h.name}[k]"
            for o, tap_name in zip(output_taps, tap_names):
                o.name = tap_name
            # Return outputs and updates
            return step_outputs, step_updates

        # Check that there are no missing inputs: Theano's error messages
        # for this are nasty. We want to catch the mistake first and print
        # better information
        all_exprs = chain(outputs, output_taps_replace, updates.values())    
        symbinputs = shim.graph.pure_symbolic_inputs(all_exprs)
        if symbinputs:
            raise shim.graph.MissingInputError(
                "The following purely symbolic variables are present in the "
                "computational graph. They were probably added accidentally; "
                "the only supported symbolic variables are shared variables.\n"
                f"Pure symbolic inputs: {symbinputs}.")

        # Now we can construct the scan graph.
        # NB: tidx will be replaced by anchor_tidx (i.e. num_tidx), so it should be k-1
        #     if we want to compute k. Ergo, tidx range must go from curtidx to stoptidx-1.
        outs, upds = shim.scan(onestep,
                               sequences = [shim.arange(curtidx, stoptidx,
                                                        dtype=self.tidx_dtype)],
                               outputs_info = outputs_info,
                               non_sequences=shared_vars_in_graph,
                               name = f"scan ({self.name})",
                               return_list=True
                               )

        # At least for now, we return everything as a update dictionary; this
        # matches the old logic, so doesn't require changes outside this method.
        # To do this, we create a new update dictionary for the shared data
        # underlying updated histories, applying the values in `outs`,
        # and combine it with `upds`.
        
        if outs is None: outs = []
        assert len(output_hists) == len(outs)
        # curtidx, stoptidx may be involved in arithmetic ops, but they should have one symbolic input each
        inp_curtidx = shim.graph.pure_symbolic_inputs(curtidx)
        if len(inp_curtidx) != 1:
            warn("When creating the `scan`, we expect the symbolic curtidx (start index of the scan) "
                 f"to depend on one purely symbolic input, but in fact it depends on {len(inp_curtidx)}.")
        inp_stoptidx = shim.graph.pure_symbolic_inputs(stoptidx)
        inp_stoptidx = [i for i in inp_stoptidx if not any(i is ic for ic in inp_curtidx)]
        if len(inp_curtidx) != 1:
            warn("When creating the `scan`, we expect the symbolic stoptidx (stop index of the scan) "
                 f"to depend on one purely symbolic input, but in fact it depends on {len(inp_stoptidx)}.")
        if len(inp_curtidx) == 1 and len(inp_stoptidx) == 1:
            # Test values for the assertions
            # TODO?: Use actual test values ?
            inp = {inp_curtidx[0]: 0, inp_stoptidx[0]: 10}
        else:
            inp = None
        for h, out in zip(output_hists, outs):
            outslc = slice(h._num_tidx + 1, h._num_tidx + 1 + (stoptidx-curtidx))
            outslc = slice(shim.graph.clone(outslc.start, replace=tidxsubs),
                           shim.graph.clone(outslc.stop,  replace=tidxsubs))
            # Sanity check: outslc resolves to a non-empty slice of same shape as out
            evalslc = None if not inp else shim.eval(outslc, givens=inp, if_too_costly='ignore')
            if evalslc:
                assert evalslc.stop > evalslc.start, f"Model.convert_point_updates_to_scan: negative or zero length output slice for history {h.name}. " \
                    f"After the scan op, the generated data for this history would be of shape {out.shape.eval(inp)}, but the time slice (first time dimension) has end points\nStart: {evalslc.start}\nStop:  {evalslc.stop}"
                try:
                    outlen = shim.eval(out.shape[0], givens=inp)
                except shim.graph.TooCostly:
                    pass
                else:
                    if outlen and evalslc.stop - evalslc.start != outlen:
                        raise AssertionError(
                            f"After the scan op, the generated data for the history '{h.name}' would be of shape {out.shape.eval(inp)}, "
                            f"but the time slice (first time dimension) has end points\nStart: {evalslc.start}\nStop:  {evalslc.stop}")
            # Add updates for this history
            upds = {**upds, **h._get_symbolic_update(outslc, out)}
                
        # out_upds = {h._num_data: shim.set_subtensor(h._num_data[curtidx:stoptidx], out)
        #             for h, out in zip(output_hists, outs)}
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

        Raises
        ------
        ValueError
            : If the wrapped function has variadic arguments (*args or **kwargs).
            : If the wrapped function does not have 1 or 2 arguments.
        RuntimeError
            : (when calling the returned accumulator) if the histories on which.
              the accumulator depends do not have at least one non-computed time point.

        Warns
        -----
        UserWarning
            : If the wrapped function has 2 arguments and the first is not 'self'.
        """
        def wrapped_acc(f):
            # Inspect the signature of `f` to see if there is a `self` argument
            # If so, attach the model to it.
            # This is similar to the same pattern used in histories.HistoryUpdateFunction
            sig = inspect.signature(f)
            if any(p.kind in (inspect._ParameterKind.VAR_POSITIONAL,)
                              #inspect._ParameterKind.VAR_KEYWORD)
                   for p in sig.parameters.values()):
               raise ValueError("Sinn accumulators don't support variadic "
                               "arguments at this time.")
            extra_required_args = [p for p in list(sig.parameters.values())[2:]
                                   if p.default is not inspect._empty]
            if extra_required_args:
                raise ValueError(
                    f"The function {func.__qualname__} (signature: {sig}) "
                    "should require at most two arguments (typically ``(model, tidx)``), "
                    f"but it also requires {extra_required_args}.")
            if len(sig.parameters) == 1:
                # Do not include `self`
                firstarg = ()
            elif len(sig.parameters) >= 2:
                first_arg_name = next(iter(inspect.signature(f).parameters.keys()))
                if first_arg_name not in ("self", "model"):
                    warn("If an accumulator function accepts two arguments, "
                         "arguments, the first should be 'self' or 'model'. "
                         f"Received '{first_arg_name}'.")
                firstarg = (self,)
            else:
                raise ValueError(
                    f"The function {func.__qualname__} (signature: {sig}) "
                    "should accept two arguments (typically ``(model, tidx)``), "
                    f"but is defined with {len(sig.parameters)}.")
            # Create the accumulator function
            def accumulated_function(curtidx, batchsize):
                # Ensuring this works correctly with shared variables would require more testing
                assert shim.is_pure_symbolic(curtidx) and shim.is_pure_symbolic(batchsize)
                # TODO: Make cost a symbolic variable, and pass it to scan as `outputs_info`
                # time index anchor
                numtidx = self.num_tidx  # NB: Must be the same as used to create
                                         #     anchor_tidx in convert_point_updates_to_scan
                if not self.time.Index(numtidx+start_offset).in_bounds:
                    raise RuntimeError(
                        "In order to compute a scan function, the model must "
                        f"have at least {start_offset} uncomputed time point.")
                # Accumulator variable for the cost. Initialized to zero.
                cost = shim.shared(np.array(0, dtype='float64'), f"accumulator ({f.__name__})")
                # Build the computational graph for the step update of the cost
                shim.add_update(cost, cost + f(*firstarg, self.time.Index(numtidx+start_offset)))
                    # f(…) triggers required history update computations
                # Convert the step update to an iteration
                updates = self.convert_point_updates_to_scan(
                    shim.get_updates(), curtidx, curtidx+batchsize)
                # Return the final cost, along with the rest of the updates
                # Caller can decide if they want to apply updates or discard them
                cost_total = updates.pop(cost)
                shim.remove_update(cost)
                shim.add_updates(updates)
                return cost_total, shim.get_updates()
            # Attach the offset, so functions can determine by how much they
            # need to shift arguments
            accumulated_function.start_offset = start_offset
            # Attach the original function, so we can use it for serialization
            accumulated_function.__func__ = f
            # Define name and docstring of the new function based on the original
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
            - Optimizing the evolution of a latent variable.

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
