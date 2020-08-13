# -*- coding: utf-8 -*-
"""
Functions for maintaining an on-disk cache of objects.
The use-case here is a relatively long-running program, which may
create many objects that are long to compute. So we want to cache
the computations and save the objects for the program's whole lifetime, but
there are many of them so we don't want to leave them in memory.

This is, in effect, a very thin wrapper around python's 'shelve' module, with
the following changes:

- A single cache file shared throughout the executable. This allows setup
  code to set the cache file location.
- Manages opening/closing the cache file, so simple save()/load() calls suffice.
  (Rather than having to wrap in a `with open(cachefile...` context.)
- Logging when cache is used.
- Will not corrupt or destroey an existing cache file, e.g. if parallel
  processes try to use the same location.

Created Wed Feb 22 2017

Copyright 2017,2020 Alexandre René
"""

import os
import atexit
from warnings import warn
import logging
logger = logging.getLogger(__file__)

from pathlib import Path
import shelve
import mackelab_toolbox.iotools as io

# Do not set _disk_cache_file directly during runtime – use `diskcache.set_file`.
# This value is used by diskcache to remember where the file was
_disk_cache_path = None

def set_file(pathname):
    global _disk_cache_path
    # Remove previous cache file from disk
    unset_file()
    if pathname != None:
        try:
            f, shelf_pathname = io.get_free_file(pathname, max_files=20)
        except IOError:
            warn(f"Unable to create '{pathname}' for disk cache.")
            shelf_pathname = None
        else:
            f.close()
            if Path(shelf_pathname).absolute() != Path(pathname).absolute():
                warn(f"Unable to create the disk cache at '{pathname}' – "
                     f"using '{shelf_pathname}' instead. This may indicate a "
                     "stale cache, or clashing processes.")
            # Just opening and closing the file is enough to create it,
            # which will prevent other processes from stealing our cache file.
            with shelve.open(shelf_pathname, flag='n'):
                # Now overwrite the file with an empty database, so it's in the
                # expected format
                pass
        _disk_cache_path = shelf_pathname
    else:
        _disk_cache_path = None

def unset_file():
    """Remove the cache file from disk."""
    global _disk_cache_path
    for path in _find_db_paths(_disk_cache_path):
        os.remove(path)
    _disk_cache_path = None
atexit.register(unset_file)

def load(key):
    """Retrieve an object from the on-disk cache.
    `key` corresponds to the object's hash.

    Raises
    ------
    KeyError:
        - If cache file is not set.
        - If `key` is not found in the cache.
    FileNotFoundError:
        - If the cache file cannot be opened
    """
    key = str(key) # shelve module wants strings
    if _disk_cache_path is None:
        # There is no disk cache
        raise KeyError
    try:
        with shelve.open(_disk_cache_path) as db:
            logger.info(f"Loading {str(key)} from disk cache.")
            return db[key]
    except FileNotFoundError:
        logger.warning("Unable to open the disk cache file '{}'"
                       .format(_disk_cache_path))


def save(key, obj):
    """Save an object to the on disk cache.
    """
    if _disk_cache_path is None:
        # There is no disk cache
        return
    logger.info("Saving {} to disk cache.".format(str(obj)))
    try:
        with shelve.open(_disk_cache_path) as db:
            db[key] = obj
    except FileNotFoundError:
        warn(f"Unable to open the disk cache file '{_disk_cache_path}'.")

### Internal stuff ###

def _find_db_paths(pathname):
    """
    At least two storage patterns depending the dbm implementation:

    - Single file, at location 'pathname'
    - Multiple files, at locatons 'pathname.{dat|bak|dir}'

    This returns a list with all existing paths matching either pattern
    (for the second pattern, any extra extension, not just those listed, is
    matched).
    """
    if pathname is None:
        return []
    dirname = Path(pathname).parent
    filename = Path(pathname).name
    return [dirname.joinpath(fname) for fname in os.listdir(dirname)
            if fname == filename or Path(fname).stem == filename]

# Set the file on first import. It can be changed later by calling set_file
set_file(_disk_cache_path)
