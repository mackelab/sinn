"""
A simple convenient exchangeable interface, so we don't need
conditionals just to select between e.g. T.sum and np.sum.
More complicated calls can still make use of the config.use_theano flag

This module's `lib` attribute will be attached to either theano.tensor
or numpy, such that calls can be made as `theano_shim.lib.sum`.

This module also provides interchangeable interfaces to common operations,
such as type casting and checking, assertions and rounding.

Pointers for writing theano switches
------------------------------------
- Type checking
    + isinstance(x, theano.tensor.TensorVariable) will be True when
      x is a theano variable, but False for wrappers around Python
      objects such as shared variables.
    + isinstance(x, theano.gof.Variable) is more inclusive, returning
      True for shared variables as well.
"""

# TODO?: Move functions to another module, to minimise possible clashes with the * imports ?

import numpy as np
import sinn.config as config

#######################
# Import the appropriate numerical library into this namespace,
# so we can make calls like `lib.exp`

if config.use_theano:
    import theano
    import theano.tensor as T
    import theano.tensor as lib
    import theano.ifelse
    import theano.tensor.shared_randomstreams  # CPU only
    #from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU & GPU

    inf = 1e12
else:
    import numpy as lib
    inf = np.inf

#######################
# Assert equivalent
def check(stmt):
    """Check is a library-aware wrapper for assert.
    If stmt is a Theano variable, the behaviour depends on whether
    theano.config.compute_test_value:
        - If it is 'off', `check` is a no-op
        - Otherwise, use the test values to evaluate the assert
    """
    if not config.use_theano or not isinstance(stmt, theano.gof.Variable):
        assert(stmt)
    else:
        if theano.config.compute_test_value == 'off':
            return None
        else:
            assert(stmt.tag.test_value)

######################
# Retrieving test values
def get_test_value(var):
    if config.use_theano and isinstance(var, theano.gof.Variable):
        try:
            retval = var.tag.test_value
        except AttributeError:
            raise AttributeError("You've attempted to execute a function that "
                                 "requires a test_value for the variable {} to "
                                 "be set, and this value is not set.".format(var))
    else:
        retval = var
    return retval

######################
# Type checking
def istype(obj, type_str):
    """
    Parameters
    ----------
    obj: object
        The object of which we want to check the type.
    type_str: string or iterable
        If `obj` is of this type, the function returns True,
        otherwise it returns False. Valid values of `type_str`
        are those expected for a dtype. Examples are:
        - 'int', 'int32', etc.
        - 'float', 'float32', etc.
        `type_str` can also be an iterable of aforementioned
        strings. Function will return True if `obj` is of any
        of the specifed types

    Returns
    -------
    bool
    """
    # Wrap type_str if it was not passed as an iterable
    if isinstance(type_str, str):
        type_str = [type_str]
    # Check type
    if not config.use_theano or not isinstance(obj, theano.gof.Variable):
        return any(ts in str(np.asarray(obj).dtype) for ts in type_str)
            # We cast to string to be consistent with Theano, which uses
            # strings for it's dtypes
    else:
        return any(ts in obj.dtype for ts in type_str)

#######################
# Set functions to cast to an integer variable
# These will be a Theano type, if Theano is used
if config.use_theano:
    def cast_varint16(x):
        return T.cast(x, 'int16')
    def cast_varint32(x):
        return T.cast(x, 'int32')
    def cast_varint64(x):
        return T.cast(x, 'int64')

else:
    cast_varint16 = np.int16
    cast_varint32 = np.int32
    cast_varint64 = np.int64

#####################
# Set rounding function
def round(x):
    try:
        res = x.round()  # Theano variables have a round method
    except AttributeError:
        res = round(x)
    return res

#####################
# Convenience function for choosing largest of two arguments
def max_of_2(x, y):
    # numpy and Theano's max functions would require first
    # constructing an array from x and y, which is wasteful
    # for such a simple binary selection (also probably
    # makes the Theano graph more difficult to compute).
    return ifelse(x > y, x, y)

#####################
# Set random functions

class ShimmedRandomStreams:
    def __init__(self, seed=None):
        np.random.seed(seed)

    def normal(size=None, avg=0.0, std=1.0, ndim=None, dtype=None):
        return np.random(loc=avg, scale=std, size=size).astype(dtype)

if config.use_theano:
    RandomStreams = theano.tensor.shared_randomstreams.RandomStreams

else:
    RandomStreams = ShimmedRandomStreams



################################################
# Define Theano placeins, which execute
# equivalent Python code if Theano is not used.
# Many Python versions take useless arguments,
# to match the signature of the Theano version.
################################################

######################
# Interchangeable ifelse function
def ifelse(condition, then_branch, else_branch, name=None):
    if (config.use_theano and isinstance(condition, theano.gof.Variable)):
        # Theano function
        return theano.ifelse.ifelse(condition, then_branch,
                                    else_branch, name)
    else:
        # Python function
        if condition:
            return then_branch
        else:
            return else_branch

######################
# Shared variable constructor

class ShimmedShared:

    def __init__(self, value, name=None, strict=False, allow_downcast=None, **kwargs):
        self.name = name
        self._value = value

    def __getitem__(self, key):
        return self._value[key]

    def get_value(self, borrow=False, return_internal_type=False):
        return self._value

    def set_value(self, new_value, borrow=False):
        self._value = new_value

def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    if config.use_theano:
        return theano.shared(value, name, strict, allow_downcast, **kwargs)
    else:
        return ShimmedShared(value, name, strict, allow_downcast, **kwargs)


######################
# Interchangeable set_subtensor
def set_subtensor(x, y, inplace=False, tolerate_aliasing=False):
    if config.use_theano and isinstance(x, theano.gof.Variable):
        return T.set_subtensor(x, y, inplace, tolerate_aliasing)
    else:
        assert(x.base is not None)
            # Ensure that x is a view of another ndarray
        x[:] = y
        return x.base

def inc_subtensor(x, y, inplace=False, tolerate_aliasing=False):
    if config.use_theano and isinstance(x, theano.gof.Variable):
        return T.inc_subtensor(x, y, inplace, tolerate_aliasing)
    else:
        assert(x.base is not None)
            # Ensure that x is a view of another ndarray
        x[:] += y
        return x.base
