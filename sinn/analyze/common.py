import numpy as np
from collections import namedtuple

# ==================================
# Data type

ParameterAxis = namedtuple('ParameterAxis', ['name', 'stops', 'idx', 'scale', 'linearize_fn', 'inverse_linearize_fn'])
  #TODO Could probably just use 'scale', and have a lookup for the linearize functions

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
