"""
Module defining global objects that must have exactly one instance,
or singleton classes whose ID must be consistent.

Based on the NumPy module of the same name.
"""
from mackelab_toolbox.utils import sentinel

# # Doesn't really work: sunder attributes still ignored by import *
# __ALL__ = [
#     '_NoValue', '_ComputedValue', 'ComputedValueType'
# ]


# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading sinn._globals is not allowed')
_is_loaded = True


_NoValue = sentinel('<no value>')
"""Special keyword value.
Used as the default value assigned to a keyword in
order to check if it has been given a user defined value.
"""

# _ComputedValue    = sentinel('<computed value>')
# ComputedValueType = type(_ComputedValue)
# """Special default value for Pydantic objects.
# Used to hide a parent class attribute. If B subclasses A, and the other
# parameters of B suffice to compute the value of the parameter θ of A , then
# one does not want to pass θ as a parameter to B (risk of inconsisent state)
# or include it in B's schema.
# For this to work, the default value of θ must be set to `_ComputedValue`
# (type annotation is ignored), and `keep_untouched=(ComputedValueType,)` added to
# the model config.
# https://pydantic-docs.helpmanual.io/usage/model_config/
# """
