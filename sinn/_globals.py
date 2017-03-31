"""
Module defining global objects that must have exactly one instance,
or singleton classes whose ID must be consistent.

Based on the NumPy module of the same name.
"""

__ALL__ = [
    '_NoValue',
]


# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if '_is_loaded' in globals():
    raise RuntimeError('Reloading numpy._globals is not allowed')
_is_loaded = True


class _NoValue:
    """Special keyword value.
    This class may be used as the default value assigned to a keyword in
    order to check if it has been given a user defined value.
    """
pass

