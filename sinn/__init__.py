from ._globals import _NoValue

import sinn.config as config
from .common import *
from .histories import TimeAxis, History
from .models import Model, ModelParams

# Set __version__ – See https://packaging.python.org/en/latest/guides/single-sourcing-package-version/ technique #5
from importlib import metadata
__version__ = metadata.version('sinn')
config.compat_version = __version__
    # If some behaviour changes with a particular version,
    # 'compat_version' can be set to recover the old behaviour.
