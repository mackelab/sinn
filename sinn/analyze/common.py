import numpy as np
from collections import namedtuple

# ==================================
# Data type

ParameterAxis = namedtuple('ParameterAxis', ['name', 'stops', 'idx', 'scale', 'linearize_fn', 'inverse_linearize_fn'])
  #TODO Could probably just use 'scale', and have a lookup for the linearize functions

def get_index(axis, idx):
    if idx is None or isinstance(idx, int):
        return idx
    else:
        assert( isinstance(idx, float) )
        return np.searchsorted(axis.stops, idx)

def get_index_slice(axis, slc):
    """
    Return a slice of indices. Integer components of slc are left untouched, whereas
    float components are assumed to refer to stop values and are converted to an index.
    `slc` may also be a scalar (int or float), in which case a scalar index is returned.

    Parameters
    ----------
    axis:   ParameterAxis instance
    slc:  slice or scalar
    """
    # TODO: Something useful with slc.step

    if isinstance(slc, slice):
        return slice(get_index(axis, slc.start), get_index(axis, slc.stop), slc.step)
    else:
        assert( isinstance(slc, (int, float)) )
        return get_index(axis, slc)

# ==================================
# Axis scales

AxisScale = namedtuple('AxisScale', ['linearize_fn', 'inverse_linearize_fn'])

# functions
def noop(x):
    return x
def pow10(x):
    return 10**x

# The actual scales object
scales = {
    'linear' : AxisScale(noop, noop),
    'log'    : AxisScale(np.log10, pow10)
}
