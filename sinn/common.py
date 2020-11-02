# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Copyright 2017-2020 Alexandre René
"""
import abc
import os
import sys
import logging
from enum import IntEnum, Enum
import numpy as np
from collections import namedtuple, deque
from types import SimpleNamespace

from typing import Tuple
import dataclasses
import pydantic
import mackelab_toolbox as mtb
from mackelab_toolbox.typing import FloatX

import theano_shim as shim
from ._globals import *
from . import config
#import sinn.config as config
from sinn.diskcache import diskcache

#########################
# Configure logger

# Add custom logger levels
# https://stackoverflow.com/a/22586200
class LoggingLevels(IntEnum):
    MONITOR = 17

class SinnLogger(logging.getLoggerClass()):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        logging.addLevelName(LoggingLevels.MONITOR, "MONITOR")

    def monitor(self, msg, *args, **kwargs):
        if self.isEnabledFor(LoggingLevels.MONITOR):
            self._log(LoggingLevels.MONITOR, msg, args, **kwargs)

logging.setLoggerClass(SinnLogger)
    # Make sure that the logger class is set before constructing
    # any logging instance

_msg_queue = deque()
def log_queue(logfn, msg):
    # TODO: use log level string and/or number instead of function
    """Add a log message to the stack. If the same message is already
    present, it is not added again."""
    global _msg_queue
    if (logfn, msg) not in _msg_queue:
        _msg_queue.append( (logfn, msg) )

def flush_log_queue():
    global _msg_queue
    while len(_msg_queue) > 0:
        logfn, msg = _msg_queue.popleft()
        logfn(msg)

# Create logging instance for this module
logger = logging.getLogger(__name__)

###########################
# Type registration
#
# Registering a type allows a load function to reconstruct it from its numpy_repr.
# Types are registered under a typename, which is by default should be the type's __name__ attribute.

# We use import guards to avoid making 'mackelab' a hard dependency: it's just the provider of the
# save/load functions
try:
    from mackelab_toolbox.iotools import register_datatype, find_registered_typename
except ImportError:
    def register_datatype(type):
        return
    def find_registered_typename(type):
        return type.__name__

###########################
# Configure defaults of imported libraries

import mackelab_toolbox.serialize
mackelab_toolbox.serialize.config.default_namespace.update(
    shim=shim
)

###########################
# Internal state variables
#
# These are mostly used as sentinels to recognize recursion within a function
# Such global state variables should be avoided, but sometimes I haven't yet
# found a better solution.

_convolution_recursion_level = 0


###########################
# Special types
#
# Especially small wrapper types, which associate metadata to an object.

# These objects can get created inside iterated convolution calls, so we prefer
# the lighter `dataclasses` structure with no validation over the pydantic one
@dataclasses.dataclass
class TensorDims:
    covariant    : Tuple = ()
    contravariant: Tuple = ()
    contraction  : Tuple = ()

    # If we didn't care about performance, we could use the following validator:
    # @root_validator
    # def no_overlap(cls, values):
    #     covary, contravary, contract = (values.get(x, None)
    #         for x in ('covariant', 'contravariant', 'contraction'))
    #     if None not in (covary, contravary, contract):
    #         assert len(set(covary) & set(contravary)) == 0
    #         assert len(set(covary) & set(contract)) == 0
    #         assert len(set(contravary) & set(contract)) == 0
    #     return values

@dataclasses.dataclass
class TensorWrapper:
    """
    Package an array along with some metadata indicating which axes are
    covariant/contravariant, and which are contractions.
    """
    array: mtb.typing.Tensor
    dims: TensorDims

    # # If we need to normalize inputs in the future, we can consider if the
    # # following is worth doing.
    # def __post_init__(self):
    #     """Unlabled axes default to 'covariant'."""
    #     axes = np.arange(self.array.ndim)
    #     labeled_axes = set(self.dims.covariant) | set(self.dims.contravariant) | set(self.dims.contraction)
    #     if len(axes) > len(labeled_axes):
    #         self.dims.covariant = Tuple(ax for ax in axes if ax in self.dims.covariant or ax not in labeled_axes):

# TODO: Replace with NumericModelParams (c.f. sinn.models)
class IndexableNamespace(SimpleNamespace):
    def __getitem__(self, key):
        return self.__dict__[key]

    # Required to behave like a mapping, otherwise Pydantic gets confused
    def __iter__(self):
        return iter(self.__dict__)
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()

    # Encoder/decoder required for use within a Pydantic model
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value):
        # `value` can be any mapping
        return cls(**value)
    @classmethod
    def json_encoder(cls, value, **kwargs):
        if not isinstance(value, IndexableNamespace):
            logger.error("`IndexableNamespace.json_encoder` expects an "
                         f"IndexableNamespace as `value`; received {value} "
                         f"(type: {type(value)}). Continuing, but behaviour "
                         "is undefined.")
        return value.__dict__
    def json(self):
        return self.json_encoder(self)
mtb.typing.add_json_encoder(IndexableNamespace, IndexableNamespace.json_encoder)

###########################
# Utility functions

def clip_probabilities(prob_array, min_prob = None, max_prob = None):
    # For float64 variables, we should clip at least 1e-8 away from the bounds
    # For float32 this seems to work as well
    # Float16 can't resolve anything lower than 6e-8, but I've never
    # run code with single precision floats, so a higher value
    # may be needed

    # Set default clipping bounds
    dtype = getattr(prob_array, 'dtype', shim.config.floatX)
    precision = (16 if '16' in str(dtype)
                 else 32 if '32' in str(dtype)
                 else 64 if '64' in str(dtype)
                 else None)
    if precision is None:
        raise TypeError("Probabilities to clip must be of float type, but "
                        "they are {}.".format(dtype))
    if precision == 16:
        logger.warning("Half-precision is insufficient for most probability "
                       "application: it only holds ~3 significant digits.")
    if min_prob is None:
        min_prob = (1e-7 if precision == 16
                    else 1e-8 if precision in (32, 64)
                    else None)
    if max_prob is None:
        max_prob = (1.-1e-3 if precision == 16
                    else 1.-1e-7 if precision == 32
                    else 1.-1e-8 if precision == 64
                    else None)
    assert(None not in (min_prob, max_prob))
    min_prob = np.asarray(min_prob).astype(dtype)
    max_prob = np.asarray(max_prob).astype(dtype)
    if shim.config.library == 'numpy':
        if np.any(prob_array > 1) or np.any(prob_array < 1):
            log_queue(logger.warning, "Some probabilities were clipped.")
        elif np.any(prob_array > max_prob) or np.any(prob_array < min_prob):
            log_queue(logger.debug,
                      "Some valid probabilities were clipped for being "
                      "too close to 0 or 1.")

    return shim.clip(prob_array, min_prob, max_prob)
        # Clipping a little bit within the interval [0,1] avoids problems
        # with likelihoods (which tend to blow up when p = 0 or 1)

def isclose(a, b, tol=None, rtol=None, atol=None, equal_nan=False):
    """
    Wrapper around numpy.isclose, which uses the sinn.config tolerances.
    Tolerance is determined based on the smallest dtype between a and b.
    Use `tol` to set both `atol` and `rtol` simultaneously.
    """
    # Use `tol` as default value for rtol, atol
    if rtol is None: rtol = tol
    if atol is None: atol = tol
    dtype = np.result_type(a, b)
    if shim.is_theano_object(a, b):
        logger.warning("Called `sinn.isclose` on a Theano object. This always returns True.")
        return True
    else:
        if rtol is None:
            rtol = config.get_rel_tolerance(dtype)
        elif isinstance(rtol, (np.dtype, str)):
            # rtol is actually a dtype
            rtol = config.get_rel_tolerance(rtol)
        if atol is None:
            atol = config.get_abs_tolerance(dtype)
        elif isinstance(atol, (np.dtype, str)):
            # atol is actually a dtype
            atol = config.get_rel_tolerance(atol)
        return np.isclose(a, b, rtol, atol, equal_nan)

def ismultiple(x, base, rtol=None, atol=None):
    """Returns True if `x` is a multiple of `base`, up to the given numerical precision."""
    if shim.is_theano_object(x, base):
        logger.warning("Called `sinn.ismultiple` on a Theano object. This always returns True.")
        return True
    else:
        # Get lowest precision type
        x = abs(np.asarray(x))
        base = abs(np.asarray(base))
        if (np.issubdtype(x.dtype, np.integer)
            != np.issubdtype(base.dtype, np.integer)):
            # One is a float, the other an integer. Take float as dtype.
            if np.issubdtype(x.dtype, np.integer):
                assert(np.issubdtype(base.dtype, np.inexact))
                dtype = base.dtype
            else:
                assert(np.issubdtype(x.dtype, np.inexact))
                dtype = x.dtype
        else:
            # Use the lowest precision of the chosen type class
            if np.can_cast(x.dtype, base.dtype):
                dtype = x.dtype
            else:
                dtype = base.dtype
        if rtol is None:
            rtol = config.get_rel_tolerance(dtype)
        elif not isinstance(rtol, (np.floating, float)):
            rtol = config.get_rel_tolerance(rtol)
        if atol is None:
            atol = config.get_abs_tolerance(dtype)
        elif not isinstance(atol, (np.floating, float)):
            atol = config.get_abs_tolerance(dtype)
        # Because we've subtracted the multiple to have something
        # close to zero, the relative error is going to be too
        # small. So instead we scale it by the value and add it
        # ourselves the absolute tolerance.
        atol += rtol * x/base
        return isclose(0, shim.round(x/base) - x/base, rtol=rtol, atol=atol)
            # Tolerance for isclose(a,b) is atol + rtol*abs(b),
            # so the '0' above must be first argument

def upcast(x, to_dtype=np.float64, from_dtype=None, cast_integers=False,
           same_kind=True, disable_rounding=False):
    """
    Upcast `x` to `to_dtype`, rounding out numerical errors due to the lower
    precision of the type of `x`.
    The rounding precision is determined by calling `get_abs_tolerance(x)`.
    Note that the rounding behaviour will raise an error if `x` is a Theano
    object.
    The rounding behaviour can be disable by setting `disable_rounding=False`.
    If `from_dtype` is None, use `x.dtype`.
    By default, integers are not upcast; to change this behaviour, pass
    `cast_integers=True`.
    When `same_kind` is `True`, only casts e.g. between 'float32' and 'float64'
    are permitted; others raise `TypeError`.

    Parameters
    ----------
    ...
    """
    # NOTE I'm not a use fan of working with string representations of dtypes,
    # but that's what Theano uses. Rewriting using  np.issubdtype & co. would
    # probably be better.
    x = shim.asarray(x)
    if np.issubdtype(x.dtype, np.integer) and not cast_integers:
        return x
    if from_dtype is None:
        from_dtype = x.dtype
    if np.can_cast(to_dtype, from_dtype):
        return x
    newx = shim.cast(x, to_dtype, same_kind=same_kind)
    if disable_rounding:
        return newx
    else:
        if shim.is_theano_object(x):
            raise ValueError("Disable rounding when using `upcast()` on "
                             "Theano variables.")
        decimals = -np.rint(np.log10(
                            config.get_abs_tolerance(from_dtype))).astype('int')
        return np.round(x, decimals=decimals)

def static_vars(**kwargs):
    """
    Declare static variables in a function with a decorator.
    Note that this only works with functions, not methods.
    """
    # Sourced from https://stackoverflow.com/a/279586
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# HistoryBase, PopulationHistoryBase and KernelBase are mostly provided for
# type-checking (otherwise one needs to import e.g. histories.py, which can
# lead to circular imports)

class HistoryBase(pydantic.BaseModel, abc.ABC):
    pass

class PopulationHistoryBase(HistoryBase):
    pass

class KernelBase(pydantic.BaseModel):
    pass

class _OpTupleCache:
    """OpCache is essentially a collection of _OpTupleCache's, one for each
    source-other pair (hence 'tuple')."""
    # TODO Move more of the functionality into this class

    @staticmethod
    def get_hash_val(other, op):
        return hash(hash(other) + hash(op))

    def __init__(self, other, op):
        self.index_list = {}
        self.data = None
        self._hash_val = self.get_hash_val(other, op)

    def __hash__(self):
        return self._hash_val

class OpCache:

    def __init__(self, source, op):
        """
        Parameters
        ----------
        source: object
            One of two members in an operation, and the one
            to which we are attaching this cache.
        op: callable
            Binary operation we want to cache. Return value
            should be a 1d array of same length as source
        op_shape: DEPRECATED int tuple
            Shape of the operation's output.
        """
        self.cache = {}
        self.old_cache = {}
        self.source = source
        self.op = op
        #self.op_shape = op_shape

    def clear(self):
        # Delete the whole cache. This can be necessary if e.g. a history
        # is changed
        self.cache = {}
        self.old_cache = {}

    def theano_reset(self):
        for key in self.old_cache:
            self.old_cache[key] = None

    def sanitize(self, arg):
        """Return a hashable value base on `arg`.
        No-op if `arg` is hashable.
        """
        if isinstance(arg, slice):
            assert( not any(shim.is_theano_object(x)
                            for x in [arg.start, arg.stop, arg.step]) )
            return (arg.start, arg.stop, arg.step)
        else:
            assert( not shim.is_theano_object(x) )
            return arg

    # def _get_value(self, x):
    #     def gv(x): # get value
    #         return x if not shim.isshared(x) else x.get_value()
    #     if isinstance(x, slice):
    #         return slice(gv(x.start), gv(x.stop), gv(x.step))
    #     else:
    #         return gv(x.start)

    def get(self, other, arg):
        """Will raise KeyError if operation not cached."""
        cache_idx = self.cache[hash(other)].index_list[self.sanitize(arg)]
        return self.cache[hash(other)].data[cache_idx]

    def ensureget(self, other, args):
        """Will compute and cache the operation if it isn't already."""
        # TODO: Make the cached data a plain Numpy array, since we never
        #       use Theano shared arrays.

        ###############################################
        # Don't use caching when compiling a Theano function
        # – Theano does its own optimization
        # if ( (hasattr(self, 'use_theano') and self.use_theano)
        #      or (hasattr(other, 'use_theano') and other.use_theano)
        #      or any(shim.is_theano_variable(arg) for arg in args) ):
        # FIXME: At present practically nothing will be cached. For kernels,
        #        instead of checking 'locked' state, should check if parameter set corresponds
        if ( shim.config.library != 'numpy'
             or not getattr(self.source, 'locked', True)
             or not getattr(other, 'locked', True) ):
            return shim.stack( [ self.op(other, arg) for arg in args ] )

        ################################################

        # Replace shared variables by their
        args = [arg for arg in args]

        # Create a set of keys for the cache dictionary
        arg_keys = [self.sanitize(arg) for arg in args]

        # Store a list of references to the cached operations we need
        # Storing None indicates that the operation with that argument
        # needs to be calculated
        data_keys = [None]*len(arg_keys)
        if hash(other) in self.cache:
            # Retrieve the indices of the already cached operations
            for i, key in enumerate(arg_keys):
                try:
                    data_keys[i] = self.cache[hash(other)].index_list[key]
                except KeyError:
                    pass
        else:
            # Check to see if this operation is in the disk cache
            # FIXME: I'm not sure on-disk caches will work if they contain
            #        theano objects (which can happened if the data is a
            #        shared variable). Currently we just deactivate the cache
            #        since we don't have a strong rationale to use it.
            # FIXME: There needs to be some mechanism to prevent the disk cache
            #        from blowing up. When doing a multi-dim sweep it makes sense
            #        to cache previous operations, but in other cases it doesn't
            #        (e.g. millions of iterations during training). Currently it's
            #        just deactivate to avoid issues.
            try:
                self.cache[hash(other)] = diskcache.load(
                    _OpTupleCache.get_hash_val(other, self.op))
            except KeyError:
                # It's not: create a new empty cache
                self.cache[hash(other)] = _OpTupleCache(other, self.op)
                self.old_cache[hash(other)] = None
            else:
                # We were able to load the op cache from file
                # Now retrieve the indices of the already cached operations
                for i, key in enumerate(arg_keys):
                    try:
                        data_keys[i] = self.cache[hash(other)].index_list[key]
                    except KeyError:
                        pass

        if None in data_keys:
            # There are operations with new arguments we need to compute
            new_data = shim.stack( [
                #shim.convolve(self[:], dis_kernel[slc], mode='valid')
                ###########################################
                # CUSTOMIZATION: Here is the call to the custom operation we are caching
                self.op(other, arg)
                ###########################################
                for arg, cache_idx in zip(args, data_keys)
                if cache_idx is None ] )
            if shim.is_theano_object(new_data):
                # It only makes since for a cache to store real numbers (i.e. Numpy variables)
                # So we will try to evaluate it – if it only depends on shared variables
                # (as it should), then this will work.
                # FIXME: The theano graph involved here should not involve any updates,
                #        but is there any way to ensure that ?
                try:
                    new_new_data = new_data.eval()
                except shim.getT().gof.fg.MissingInputError as e:
                    raise  # DEBUG
                    logger.warning("Unable to precompute a cached op between {} and {}. "
                                   "Typically this is because there are inputs in the graph; "
                                   "you likely want to replace those inputs by shared variables.\n\n"
                                   "The error raised by theano was {}."
                                   .format(self.source.name, other.name, str(e)))
                    # Abort caching. This is not fatal, but the user should fix the code
                    # to avoid ending up here.
                    return new_data
                else:
                    new_data = new_new_data

            if self.cache[hash(other)].data is None:
                # First time we cache data – create a new structure
                self.cache[hash(other)].data = shim.ShimmedTensorShared(new_data)
                    # TODO: Just replace by a normal Numpy array
                for i, akey in enumerate(arg_keys):
                    data_keys[i] = i
                    self.cache[hash(other)].index_list[akey] = i
            else:
                # Update the existing cache

                # Add the indices they will have in the newly augmented cache
                k = self.cache[hash(other)].data.get_value().shape[0]
                for i, (dkey, akey) in enumerate(zip(data_keys, arg_keys)):
                    if dkey is None:
                        data_keys[i] = k
                        self.cache[hash(other)]['index list'][akey] = k
                        k += 1

                # Create the new data cache
                if shim.config.library != 'numpy':
                    assert(self.old_cache is None)
                    # Keep the old cache in memory, otherwise updates mechanism will break
                    self.old_cache[hash(other)] = self.cache[hash(other)].data
                    self.cache[hash(other)].data = shim.concatenate(
                                (self.old_cache[hash(other)], new_data) )
                    # This is a shared variable, so add to the updates list
                    shim.add_update(self.old_cache[hash(other)].data,
                                    self.cache[hash(other)].data)
                else:
                    # Since we aren't using Theano, we don't need to create old_cache
                    # and can allow reuse of cache memory
                    self.cache[hash(other)].data.set_value(
                        shim.concatenate(
                            (self.cache[hash(other)].data.get_value(), new_data), axis = 0 )
                    )
        return self.cache[hash(other)].data[data_keys]
