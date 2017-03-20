"""
Module defining global singleton classes

Based on the NumPy module of the same name.
"""

__ALL__ = [
    '_NoValue',
    'inputs'
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

inputs = {}
    # The inputs dictionary is keyed by histories. If 'hist' is a History instance,
    # inputs[hist] is a set containing all histories which appear in hist's
    # update function.
    # Whenever a history's __getitem__ method is called, it adds itself
    # to this dictionary
