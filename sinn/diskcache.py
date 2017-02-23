# -*- coding: utf-8 -*-
"""
Functions for maintaining an on-disk cache of objects.
The use-case here is a relatively long-running program, which may
create many objects that are long to compute. So we want to cache
the computations and save the objects for the program's whole lifetime, but
there are many of them so we don't want to leave them in memory.

Importantly, the cache is explicitly not maintained between program
executions. A blank cache file is created whenever this module is first imported.

Objects that are cached to disk must implement a __hash__ method that does not depend on the particular instance. Objects are retrieved with the result of their __hash__, so if it is instance dependent, that cache entry will never be retrieved.
Typically __hash__ should depend on the set of parameters that define the object.

Created Wed Feb 22 2017

author: Alexandre René
"""

import logging
logger = logging.getLogger('sinn.diskcache')

import sinn
import sinn.shelve as shelve
import sinn.iotools as io


def set_file(filename):
    if filename != "":
        try:
            f, filename = io._get_free_file(filename)
        except IOError:
            logger.warning("Unable to create '{}' for disk cache."
                        .format(filename))
            filename = ""
        else:
            f.close()
            # Just opening and closing the file is enough to create it,
            # which will prevent other processes from stealing our cache file.
            with shelve.open(filename, flag='n'):
                # Now overwrite the file with an empty database, so it's in the
                # expected format
                pass
    sinn.config.disk_cache_file = filename

# Set the file on first import. It can be changed later by calling set_file
set_file(sinn.config.disk_cache_file)

def load(key):
    """Retrieve an object from the on-disk cache.
    `key` corresponds to the object's hash.
    """
    key = str(key) # shelve module wants strings
    if sinn.config.disk_cache_file == "":
        # There is no disk cache
        raise KeyError
    try:
        with shelve.open(sinn.config.disk_cache_file) as db:
            if key in db:
                logger.info("Loading {} from disk cache.".format(str(key)))
                return db[key]
            else:
                raise KeyError
    except FileNotFoundError:
        logger.warning("Unable to open the disk cache file '{}'"
                       .format(sinn.config.disk_cache_file))


def save(obj):
    """Save an object to the on disk cache.
    Object should be hashable, and this hash is used to retrieve it later
    (so it should be possible to determine the hash without the object).
    """
    if cache_filename == "":
        # There is no disk cache
        return
    logger.info("Saving {} to disk cache.".format(str(obj)))
    try:
        with shelve.open(cache_filename) as db:
            db[str(hash(obj))] = obj
    except FileNotFoundError:
        logger.warning("Unable to open the disk cache file '{}'."
                       .format(cache_filename))


