# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Author: Alexandre René
"""
import os
import sys
import logging
from enum import IntEnum
import numpy as np
from collections import namedtuple, deque

import theano_shim as shim
from ._globals import *
from . import config
#import sinn.config as config
import sinn.diskcache as diskcache

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
logger = logging.getLogger('sinn')

###########################
# Utility functions

def clip_probabilities(prob_array,
                       min_prob = np.sqrt(config.abs_tolerance),
                       max_prob = 1-np.sqrt(config.abs_tolerance)):
    # For float64 variables, we should clip at least 1e-8 away from the bounds
    if not config.use_theano():
        if np.any(prob_array > 1) or np.any(prob_array < 1):
            log_queue(logger.warning, "Some probabilities were clipped.")
        elif np.any(prob_array > max_prob) or np.any(prob_array < min_prob):
            log_queue(logger.debug,
                      "Some valid probabilities were clipped for being "
                      "too close to 0 or 1.")

    return shim.clip(prob_array, min_prob, max_prob)
        # Clipping a little bit within the interval [0,1] avoids problems
        # with likelihoods (which tend to blow up when p = 0 or 1)

def isclose(a, b, rtol=None, atol=None, equal_nan=False):
    """Wrapper around numpy.isclose, which uses the sinn.config tolerances."""
    if shim.is_theano_object(a, b):
        logger.warning("Called `sinn.isclose` on a Theano object. This always returns True.")
        return True
    else:
        if rtol is None:
            rtol = config.rel_tolerance  # floatX precision
        if atol is None:
            atol = config.abs_tolerance  # floatX precision
        return np.isclose(a, b, config.rel_tolerance, config.abs_tolerance, equal_nan)

def ismultiple(x, base, rtol=None, atol=None):
    """Returns True if `x` is a multiple of `base`, up to the given numerical precision."""
    if shim.is_theano_object(x, base):
        logger.warning("Called `sinn.ismultiple` on a Theano object. This always returns True.")
        return True
    else:
        return isclose(0, shim.round(x/base) - x/base, rtol, atol)
            # Tolerance for isclose(a,b) is atol + rtol*abs(b),
            # so the '0' above must be first argument

def add_sibling_input(sibling, new_input):
    # TODO Move to Graph class
    for key, val in inputs.items():
        if sibling in val:
            inputs[key].add(new_input)

def get_updates():
    return shim.get_updates()

def theano_reset():
    global inputs
    for hist in inputs:
        # HACK sinn.inputs happens to have each history as
        # a key, but this is not what it's meant for
        if not hist.locked:
            hist.theano_reset()
    #inputs = {}
    shim.reset_updates()

class Node:
    def __init__(self, name):
        self.name = name

class DependencyGraph(dict):

    def __init__(self, name):
        self.name = name
        super().__init__()

    def add(self, obj):
        if obj not in self:
            if obj.name == 'JᕽAᐩI':
                pass
            self[obj] = set()

    def union(self, other):
        for obj in other:
            self.add(obj)
        if isinstance(other, DependencyGraph):
            self.match_template(other)
        return self

    def find_by_name(self, name, nofail=False):
        res = None
        for node in self:
            if node.name == name:
                assert(res is None)
                res = node
        if res is None and not nofail:
            raise ValueError("Node {} was not found.".format(name))
        return res

    def match_template(self, template):
        """Take a template graph, and reproduce the edges therein.
        Nodes are identified by their name.
        All names in `template` must already exist, but `self` can contain
        additional names.
        """
        for tmplnode, tmpledges in template.items():
            node = self.find_by_name(tmplnode.name)
            edge_list = [self.find_by_name(connected_node.name)
                         for connected_node in tmpledges]
            self[node] = self[node].union(edge_list)

    def strip(self):
        """Return a new graph reproducing the connections but stripping
        out everything else. Useful if you want to share the connectivity,
        but your objects contain large data."""
        nwgraph = Graph()

        node_set = set()
        for node, edges in self.items():
            node_set.add(node)
            node_set.union(edges)

        nwnode_set = set()
        for node in node_set:
            nwnode = Node(node.name)
            nwnode_set.add(nwnode)
            nwgraph[nwnode] = set()

        nwgraph.match_template(self)

        return nwgraph

inputs = DependencyGraph('sinn.inputs')
    # The inputs dictionary is keyed by histories. If 'hist' is a History instance,
    # inputs[hist] is a set containing all histories which appear in hist's
    # update function.
    # Whenever a history's __getitem__ method is called, it adds itself
    # to this dictionary

# HistoryBase and KernelBase are mostly provided for type-checking
# (otherwise one needs to import e.g. histories.py, which can
#  lead to circular imports)

class HistoryBase:

    def __init__(self, t0, tn):
        self.t0 = config.cast_floatX(t0)
        self.tn = config.cast_floatX(tn)
        self._tarr = None # Implement in child classes

    def get_time(self, t):
        raise NotImplementedError

    def get_t_idx(self, t):
        raise NotImplementedError

class KernelBase:
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
        if config.use_theano():
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
                #shim.lib.convolve(self[:], dis_kernel[slc], mode='valid')
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
                self.cache[hash(other)].data = shim.ShimmedShared(new_data)
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
                if config.use_theano():
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

#################################
#
# Functions and class for working with parameterized objects
#
##################################

# TODO: Subclass? namedtuple to have the casting done on instantiation (rather than when intialization the class it's attached to)
# This would also allow defining __eq__ to deal with numpy arrays

#Parameter = namedtuple('Parameter', ['name', 'cast_function', 'default'])
#Parameter.__new__.__defaults__ = [None]
def define_parameters(param_dict):
    """Call this function at the top of each model class, to define
    its `Parameters` attribute.
    """
    keys = param_dict.keys()
    Parameters = namedtuple('Parameters', keys)
    # Set default values for the parameters. A default value of `None`
    # indicates that the parameter is mandatory.
    # Parameters with no default value have their value set to `None`
    Parameters.__new__.__defaults__ = tuple([ param_dict[key][1]
                                              if isinstance(param_dict[key], tuple)
                                                and len(param_dict[key]) >= 2
                                              else None
                                              for key in keys ])
        # http://stackoverflow.com/a/18348004
    return Parameters

def params_are_equal(params1, params2):
    def get_value(param):
        if shim.isshared(param):
            return param.get_value()
        else:
            return param

    if set(params1._fields) != set(params2._fields):
        logger.warning("Comparing parameter sets for equality that don't even have the same fields.")
        return False
    for field in params1._fields:
        attr1 = getattr(params1, field)
        attr2 = getattr(params2, field)
        if type(attr1) != type(attr2):
            logger.error("The attributes {} in two different parameter sets are not even of the same type: {}({}), {}({}). This is almost certainly an error."
                      .format(field, str(attr1), str(type(attr1)),
                              str(attr2), str(type(attr2))))
            return False
        if np.any(get_value(attr1) != get_value(attr2)):
            return False
    return True

# def make_shared_tensor_params(params):
#     TParameters = namedtuple('TParameters', params._fields)
#     param_lst = []
#     for val, name in zip(params, params._fields):
#         # TODO: Check if val is already a theano tensor and adjust accordingly
#         try:
#             if val.dtype.kind in sp.typecodes['Float']:
#                 param_lst.append(theano.shared(sp.array(val, dtype=config.floatX)))
#             else:
#                 param_lst.append(theano.shared(val))
#         except ValueError:
#             # Can't convert val to numpy array – it's probably a theano tensor
#             # FIXME: if a scalar is not of type theano.config.floatX, this will actually
#             #        create a ElemWise.cast{} code, wrt which we can't differentiate
#             # FIXME: does it even make sense to produce a shared variable from another Theano variable ?
#             if val.dtype.kind in sp.typecodes['Float']:
#                 param_lst.append(T.cast(theano.shared(val), dtype=config.floatX))
#             else:
#                 param_lst.append(theano.shared(val))
#         param_lst[-1].name = name

#     return TParameters(*param_lst)

# def make_cst_tensor_params(param_names, params):
#     """
#     Construct a Parameters set of Theano constants from a
#     Parameters set of NumPy/Python objects.
#     Code seems obsolete, or at least in dire need of updating.
#     """
#     TParameters = namedtuple('TParameters', param_names)
#     global name_counter
#     id_nums = range(name_counter, name_counter + len(param_names))
#     name_counter += len(param_names)
#     return TParameters(*(T.constant(getattr(params,name), str(id_num) + '_' + name, dtype=config.floatX)
#                          for name, id_num in zip(param_names, id_nums)))

def get_parameter_subset(model, src_params):
    """
    Create a Parameters object with the same instances as src_params
    Use case: we need a handle on a kernel's parameters, e.g. because
    the parameters are shared with another kernel or some higher level
    function.

    Parameters
    ----------
    model: class instance derived from Model
        The model class for which we want a Parameter collection.
    src_params: namedtuple
        The pre-existing Parameter collection we want to reuse.
    """
    # TODO: use src_params._asdict() ?
    paramdict = {}
    for name in src_params._fields:
        if name in model.Parameters._fields:
            paramdict[name] = getattr(src_params, name)
    return model.Parameters(**paramdict)

def set_parameters(target, source):
    assert(hasattr(target, '_fields'))
    if hasattr(source, '_fields'):
        # We have a Parameter object
        assert( set(target._fields) == set(source._fields) )
        for field in target._fields:
            val = getattr(source, field)
            if shim.isshared(val):
                val = val.get_value()
            getattr(target, field).set_value( val )
    else:
        assert(isinstance(source, dict))
        assert( set(target._fields) == set(source.keys()) )
        for field in target._fields:
            val = source[field]
            if shim.isshared(val):
                val = val.get_value()
            getattr(target, field).set_value( val )

def convert_parameters_to_theano(params):
    param_dict = params._asdict()
    for key, val in param_dict.items():
        if isinstance(val, shim.ShimmedShared):
            param_dict[key] = shim.shared(val.get_value(), name=val.name)
    return param_dict


class ParameterMixin:

    Parameter_info = {}
        # Overload this in derived classes
        # Entries to Parameter dict: 'key': (dtype, default, ensure_2d)
        # If the ensure_2d flag is True, parameter will guaranteed to be a matrix with at least 2 dimensions
        # Default is ensure_2d = False.
    Parameters = define_parameters(Parameter_info)
        # Overload this in derived classes

    def __init__(self, *args, params, **kwargs):

        # try:
        #     params = kwargs.pop('params')
        # except KeyError:
        #     raise TypeError("Unsufficient arguments: ParameterMixin "
        #                     "requires a `params` argument.")
        self.set_parameters(params)
        super().__init__(*args, **kwargs)

    def cast_parameters(self, params):
        """
        Take parameters and cast them to the defined shape and type.

        Parameters
        ----------
        params: namedtuple

        Returns
        -------
        namedtuple
        """
        assert(self.parameters_are_valid(params))

        # Cast the parameters to ensure they're of prescribed type
        param_dict = {}
        for key in self.Parameters._fields:
            val = getattr(params, key)
            if shim.isshared(val) or shim.is_theano_object(val):
                # HACK We just assume that val has already been properly casted. We do this
                #      to keep the reference to the original variable
                param_dict[key] = val
            else:
                if isinstance(self.Parameter_info[key], tuple):
                    # TODO: Find a way to cast to dtype without making and then dereferencing an array
                    param_dict[key] = None
                    temp_val = np.asarray(val, dtype=self.Parameter_info[key][0])
                    # Check if we should ensure parameter is 2d.
                    try:
                        if self.Parameter_info[key][2]:
                            # Also wrap scalars in a 2D matrix so they play nice with algorithms
                            if temp_val.ndim < 2:
                                param_dict[key] = shim.shared( shim.add_axes(np.asarray(temp_val), 2 - temp_val.ndim),
                                                               name = key )
                    except KeyError:
                        pass
                    if param_dict[key] is None:
                        # `ensure_2d` is either False or unset
                        param_dict[key] = shim.shared(temp_val, name = key)
                else:
                    param_dict[key] = shim.shared(np.asarray(val, dtype=self.Parameter_info[key]),
                                                  name=key)
        return self.Parameters(**param_dict)

    def set_parameters(self, params):
        self.params = self.cast_parameters(params)

    def parameters_are_valid(self, params):
        """Returns `true` if all of the model's parameters can be set from `params`"""
        return set(self.Parameters._fields).issubset(set(params._fields))

    def get_parameter_subset(self, params):
        """
        Return the subset of parameters from params that relate to this model.

        Returns
        -------
        A Parameter namedtuple
        """
        return get_parameter_subset(self, params)

