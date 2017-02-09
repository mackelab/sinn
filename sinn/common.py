# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Author: Alexandre René
"""
import numpy as np
from collections import namedtuple

import sinn.config as config
import sinn.theano_shim as shim
floatX = config.floatX
lib = shim.lib


class HistoryBase:

    def __init__(self, t0, tn):
        self.t0 = config.cast_floatX(t0)
        self.tn = config.cast_floatX(tn)
        self._tarr = None # Implement in child classes

    def get_time(self, t):
        raise NotImplementedError

    def get_t_idx(self, t):
        raise NotImplementedError

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

    def theano_reset(self):
        for key in self.old_cache:
            self.old_cache[key] = None

    def sanitize(self, arg):
        """Return a hashable value base on `arg`.
        No-op if `arg` is hashable.
        """
        if isinstance(arg, slice):
            return (arg.start, arg.stop)
        else:
            return arg

    def get(self, other, arg):
        """Will raise KeyError if operation not cached."""
        cache_idx = self.cache[id(other)]['index list'][self.sanitize(arg)]
        return self.cache[id(other)]['data'][cache_idx]

    def ensureget(self, other, args):
        """Will compute and cache the operation if it isn't already."""
        # Create a set of keys for the cache dictionary
        arg_keys = [self.sanitize(arg) for arg in args]

        # Store a list of references to the cached operations we need
        # Storing None indicates that the operation with that argument
        # needs to be calculated
        data_keys = [None]*len(arg_keys)
        if id(other) in self.cache:
            # Retrieve the indices of the already cached convolutions
            for i, key in enumerate(arg_keys):
                try:
                    data_keys[i] = self.cache[id(other)]['index list'][key]
                except KeyError:
                    pass
        else:
            # Create a new empty cache

            # Which of the two members of the convolution – self or other –
            # is a History determines the shape of the result
            if isinstance(self.source, HistoryBase):
                hist = self.source
            else:
                assert( isinstance(other, HistoryBase) )
                hist = other
            #entry_shape = (len(hist),) + self.op_shape
            self.cache[id(other)] = {
                'data':       None,#shim.shared(np.zeros((0,) + entry_shape)),
                'index list': {}
            }
            self.old_cache[id(other)] = None


        if None in data_keys:
            # There are new convolutions we need to compute
            new_data = lib.stack( [
                #lib.convolve(self[:], dis_kernel[slc], mode='valid')
                ###########################################
                # CUSTOMIZATION: Here is the call to the custom convolution function
                #                      self._convolve_op_batch(kernel, slc)
                self.op(other, arg)
                ###########################################
                for arg, cache_idx in zip(args, data_keys)
                if cache_idx is None ] )
            if self.cache[id(other)]['data'] is None:
                # First time we cache data – create a new structure
                self.cache[id(other)]['data'] = shim.shared(new_data)
                for i, akey in enumerate(arg_keys):
                    data_keys[i] = i
                    self.cache[id(other)]['index list'][akey] = i
            else:
                # Update the existing cache

                # Add the indices they will have in the newly augmented cache
                k = self.cache[id(other)]['data'].get_value().shape[0]
                for i, (dkey, akey) in enumerate(zip(data_keys, arg_keys)):
                    if dkey is None:
                        data_keys[i] = k
                        self.cache[id(other)]['index list'][akey] = k
                        k += 1

                # Create the new data cache
                if config.use_theano:
                    assert(self.old_cache is None)
                    # Keep the old cache in memory, otherwise updates mechanism will break
                    self.old_cache[id(other)] = self.cache[id(other)]['data']
                    self.cache[id(other)]['data'] = lib.concatenate(
                                (self.old_cache[id(other)], new_data) )
                    # This is a shared variable, so add to the updates list
                    shim.theano_updates[self.old_cache[id(other)]['data']] = \
                                self.cache[id(other)]['data']
                else:
                    # Since we aren't using Theano, we don't need to create old_cache
                    # and can allow reuse of cache memory
                    self.cache[id(other)]['data'].set_value(
                        lib.concatenate(
                            (self.cache[id(other)]['data'].get_value(), new_data), axis = 0 )
                    )
        return self.cache[id(other)]['data'][data_keys]

#################################
#
# Functions and class for working with parameterized objects
#
##################################

#Parameter = namedtuple('Parameter', ['name', 'cast_function', 'default'])
#Parameter.__new__.__defaults__ = [None]
def define_parameters(param_dict):
    """Call this function at the top of each model class, to define
    its `Parameters` attribute.
    """
    keys = param_dict.keys()
    Parameters = namedtuple('Parameters', keys)
    # Set default values for the parameters. A default value of`None`
    # indicates that the parameter is mandatory.
    Parameters.__new__.__defaults__ = tuple([param_dict[key][1] for key in keys
                                             if isinstance(param_dict[key], tuple)
                                             and len(param_dict[key]) == 2])
        # http://stackoverflow.com/a/18348004
    return Parameters

def make_shared_tensor_params(params):
    TParameters = namedtuple('TParameters', params._fields)
    param_lst = []
    for val, name in zip(params, params._fields):
        # TODO: Check if val is already a theano tensor and adjust accordingly
        try:
            if val.dtype.kind in sp.typecodes['Float']:
                param_lst.append(theano.shared(sp.array(val, dtype=floatX)))
            else:
                param_lst.append(theano.shared(val))
        except ValueError:
            # Can't convert val to numpy array – it's probably a theano tensor
            # FIXME: if a scalar is not of type theano.config.floatX, this will actually
            #        create a ElemWise.cast{} code, wrt which we can't differentiate
            # FIXME: does it even make sense to produce a shared variable from another Theano variable ?
            if val.dtype.kind in sp.typecodes['Float']:
                param_lst.append(T.cast(theano.shared(val), dtype=floatX))
            else:
                param_lst.append(theano.shared(val))
        param_lst[-1].name = name

    return TParameters(*param_lst)

def make_cst_tensor_params(param_names, params):
    """
    Construct a Parameters set of Theano constants from a
    Parameters set of NumPy/Python objects.
    Code seems obsolete, or at least in dire need of updating.
    """
    TParameters = namedtuple('TParameters', param_names)
    global name_counter
    id_nums = range(name_counter, name_counter + len(param_names))
    name_counter += len(param_names)
    return TParameters(*(T.constant(getattr(params,name), str(id_num) + '_' + name, dtype=theano.config.floatX)
                         for name, id_num in zip(param_names, id_nums)))

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
    return class_instance.Parameters(**paramdict)


class ParameterMixin:

    Parameter_info = {}
        # Overload this in derived classes
        # Entries to Parameter dict: 'key': (dtype, default)
    Parameters = define_parameters(Parameter_info)
        # Overload this in derived classes

    def __init__(self, *args, **kwargs):

        try:
            params = kwargs.pop('params')
        except KeyError:
            raise TypeError("Unsufficient arguments: ParameterMixin "
                            "requires a `params` argument.")
        assert(self.parameters_are_valid(params))

        # Cast the parameters to ensure they're of prescribed type
        param_dict = {}
        for key in self.Parameters._fields:
            if isinstance(self.Parameter_info[key], tuple):
                param_dict[key] = np.asarray(getattr(params,key), dtype=self.Parameter_info[key][0])
            else:
                param_dict[key] = np.asarray(getattr(params, key), dtype=self.Parameter_info[key])
            # Wrap scalars in a 2D matrix so they play nice with algorithms
            if shim.get_ndims(param_dict[key]) < 2:
                param_dict[key] = shim.add_axes(param_dict[key], 2 - shim.get_ndims(param_dict[key]))

        self.params = self.Parameters(**param_dict)

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

