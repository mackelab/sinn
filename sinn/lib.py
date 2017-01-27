"""
A simple convenient exchangeable interface, so we don't need
conditionals just to select between e.g. T.sum and np.sum.
More complicated calls can still make use of the config.use_theano flag

Essentially this does a * import from either numpy or theano.tensor,
and adds a few functions and attributes to make the interface uniform
"""

# TODO?: Move functions to another module, to minimise possible clashes with the * imports ?

import numpy as np
import config

#######################
# Import the appropriate numerical library into this namespace,
# so we can make calls like `lib.exp`

if config.use_theano:
    import theano
    import theano.tensor as T
    import theano.ifelse
    from theano.tensor import *
    from theano.tensor.shared_randomstreams import RandomStreams  # CPU only
    #from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  # CPU & GPU

    inf = 1e12
else:
    from numpy import *
    # inf already imported

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
# Type checking
def istype(obj, type_str):
    """
    Parameters
    ----------
    obj: object
        The object of which we want to check the type.
    type_str: string
        If `obj` is of this type, the function returns True,
        otherwise it returns False. Valid values of `type_str`
        are those expected for a dtype. Examples are:
        - 'int', 'int32', etc.
        - 'float', 'float32', etc.
    """
    if not config.use_theano or not isinstance(x, theano.gof.Variable):
        return 'int' in str(np.asarray(x).dtype)
    else:
        return type_str in x.dtype

#######################
# Set functions to cast to an integer variable
# These will be a Theano type, if Theano is used
if config.use_theano
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
# Set random functions
if config.use_theano:
    def seed(seed=None):
        global rndstream
        rndstream = RandomStreams(seed=314)
else:
    seed = np.random.seed

################################################
# Define Theano placeins, which execute
# equivalent Python code if Theano is not used.
# Many Python versions take useless arguments,
# to match the signature of the Theano version.
################################################

######################
# Interchangeable ifelse function
if config.use_theano:
    ifelse = theano.ifelse.ifelse
else:
    def ifelse(condition, then_branch, else_branch, name=None):
        if condition:
            return then_branch
        else:
            return else_branch

######################
# Interchangeable set_subtensor
if config.use_theano:
    pass # already imported
else:
    def set_subtensor(x, y, inplace=False, tolerate_aliasing=False):
        return x = y
