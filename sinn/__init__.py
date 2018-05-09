# Set __version__
from ._globals import _NoValue

import sinn.config as config
from .common import *
from .process_handling import *

import pkg_resources
__version__ = pkg_resources.require("sinn")[0].version
config.compat_version = __version__
    # If some behaviour changes with a particular version,
    # 'compate_version' can be set to recover the old behaviour.
