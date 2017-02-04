# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
from scipy.integrate import quad
from collections import namedtuple

import sinn.config as config
import sinn.theano_shim as shim
floatX = config.floatX
lib = shim.lib


#Parameter = namedtuple('Parameter', ['cast_function', 'default'])
#Parameter.__new__.__defaults__ = [None]
def define_parameters(param_dict):
    """Call this function at the top of each model class, to define
    its `Parameters` attribute.
    """
    keys = Parameter_dict.keys()
    Parameters = namedtuple('Parameters', keys)
    # Set default values for the parameters. A default value of`None`
    # indicates that the parameter is mandatory.
    Parameters.__new__.__defaults__ = [param_dict[key][1] for key in keys
                                       if isinstance(param_dict[key], tuple)
                                          and len(param_dict[key]) == 2]
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

class Model:
    """Abstract model class.

    A model implementations should derive from this class.
    It must minimally provide:
    - A `Parameter_info` dictionary of the form:
        ```
        Parameter_info = OrderedDict{ 'param_name': Parameter([cast function], [default value]),
                                      ... }
        ```
    - A class-level (outside any method) call
        `Parameters = com.define_parameters(Parameter_info)`

    If an `eval` method also provided, the default initializer can also attach it to
    a history object. It should have the signature
    `def eval(self, t)`
    where `t` is a time.

    Models are typically initialized with a reference to a history object,
    which is appropriate for storing the output of `eval`.

    Implementations may also provide class methods to aid inference:
    - likelihood: (params) -> float
    - likelihood_gradient: (params) -> vector
    If not provided, `likelihood_gradient` will be calculated by appyling theano's
    grad method to `likelihood`. (TODO)
    As class methods, these don't require an instance – they can be called on the class directly.
    """

    Parameter_info = {}                                           # Overload this in derived classes
        # Entries to Parameter dict: 'key': Parameter(fn, default)
    Parameters = namedtuple('Parameter', Parameter_info.keys())   # Overload this in derived classes

    def __init__(self, params, history=None):
        """
        Parameters
        ----------
        params: self.Parameters instance

        history: History instance
            If provided, and if this model has an `eval` method, then that method is
            attached to `history` as its update function.
        """
        # Cast the parameters to ensure they're prescribed type
        param_dict = {}
        for key in self.Parameters._fields:
            if isinstance(self.Parameter_info[key], tuple):
                param_dict[key] = self.Parameter_info[key][0](getattr(params, key))
            else:
                param_dict[key] = self.Parameter_info[key](getattr(params, key))
        self.params = self.Parameters(**param_dict)

        # Try to attach default updater
        if history is not None and hasattr(self, 'eval'):
            history.set_update_function(self.eval)

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

    #TODO: Provide default gradient (through theano.grad) if likelihood is provided

