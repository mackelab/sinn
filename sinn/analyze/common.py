from collections import namedtuple

# ==================================
# Data type

ParameterAxis = namedtuple('ParameterAxis', ['name', 'stops', 'idx', 'linearize_fn', 'inverse_linearize_fn'])

