from collections import namedtuple

# ==================================
# Data type

ParameterAxis = namedtuple('ParameterAxis', ['name', 'stops', 'idx', 'scale', 'linearize_fn', 'inverse_linearize_fn'])
  #TODO Could probably just use 'scale', and have a lookup for the linearize functions

