# -*- coding: utf-8 -*-
"""

`save` will take any python object and dump to a file. It can then be reloaded
with `load`.

Raw format
----------

The problem with just saving a custom data object is that this isn't future-proof.
Changes to the objects can easily make old data unreadable, which is a Bad Thing.
To ensure future readiblity, all data objects should provide a `raw` method to save
at least the important stuff (mostly, the data). When saving an object, if it provides
a `raw` method, a second file will also be saved with the suffix '_raw'.

The output of a `raw` method should be a dictionary of the form:
`{attribute name : attribute value}`.
Each value must be a NumPy type

Data objects should also provide a `from_raw` method, to reconstruct such an object
from raw data.

--------------
Created on Mon Jul 18 2016

author: Alexandre René
"""

# TODO: Add .dat extension in save_data if there is none.
#       Try other reasonable combinations (.dat, .txt, no ext) in load_data if original filename is not found
# TODO: Function that goes through a list of property names.
#       If data contains the property, assign it, otherwise assign a default value.
#       Return a namedtuple

import os
import io
import logging
import numbers
import numpy as np
import dill
logger = logging.getLogger('sinn.iotools')

from mackelab.iotools import get_free_file

extensions = ['sin', 'sir', 'dat', 'txt']

##########################
# Public API

def save(file, data, format='raw', overwrite=False, only_raw=None):
    """Save `data`. By default, only the 'raw' representation is saved, if present.
    Not only is the raw format more future-proof, but it can be an order of
    magnitude more compact.
    If the raw save is unsuccessful (possibly because 'data' does not provide a
    'raw' method), than save falls back to saving a plain (dill) pickle of 'data'.

    Parameters
    ----------
    file: str
        Path name or file handle
        TODO: Currently only path names are supported. File handles raise NotImplementedError.
    data: Python object
        Data to save
    format: str
        The format in which to save the data. Possible values are:
          - 'raw' (default) Only the raw format. If it is unavailable, a warning is issued
            and the plain format is used.
          - 'plain' A plain (dill) pickle.
          - 'raw+plain' Both the raw and plain formats are saved. This might be used if e.g.
            the raw format discards information one would rather recover if possible, but
            more often then not this option just wastes more disk space.
    overwrite: bool
        If True, allow overwriting previously saved files. Default is false, in which case
        a number is appended to the filename to make it unique.
    only_raw: bool (Deprecated)
        Use `format` instead.
    Saving a dill pickle (in addition to the raw can be forced by passing `only_raw=False`.
    """
    save_formats = ['raw', 'plain', 'raw+plain']

    # Convert deprecated option
    if only_raw == True:
        format = 'raw'
    elif only_raw == False:
        format = 'raw+plain'

    # Check argument - format
    if format not in save_formats:
        format_names = ["'" + f + "'" for f in save_formats]
        if (format_names) > 1:
            format_desc = ", ".join(format_names[:-1]) + " and " + format_names[-1]
        logger.warning("Unrecognized save format {}.".format(format)
                       + "Recognized formats are " + format_desc)
        logger.warning("Setting the format to 'raw+plain'.")
            # We don't want to throw away the result of a long calculation because of a
            # flag error, so instead we will try to save into every format and let the user
            # sort out the files later.
        format = 'raw+plain'

    # Check argument - file
    if isinstance(file, io.IOBase):
        # TODO: Implement
        raise NotImplementedError
    else:
        assert(isinstance(file, str))
        filename = file

    # Save data in raw format
    if 'raw' in format:
        try:
            # First try to save a more future-proof raw datafile
            saveraw(filename, data, overwrite)
        except AttributeError:
            # TODO: Use custom error type
            logger.warning("Unable to save to raw format.")
            if format == 'raw':
                # Warn the user that we will use another format
                logger.warning("Will try a plain (dill) pickle dump.")
                format = 'raw+plain'

    # Save data in plain format
    if 'plain' in format:
        #os.makedirs(_get_savedir(), exist_ok=True)
        dirname = os.path.dirname(filename)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)
        try:
            relname, ext = os.path.splitext(_get_savedir() + filename)
            ext = ext if ext != "" or ext == ".sir" else ".sin"
            relpath = relname + ext
            if not overwrite:
                f, realrelpath = get_free_file(relpath)
            else:
                realrelpath = relpath
                f = open(realrelpath, 'wb')
        except IOError:
            logger.error("Could not create the filename")
            realrelpath = None

        else:
            dill.dump(data, f)
            f.close()

        return realrelpath

def saveraw(filename, data, overwrite=False):
    """Same as `save`, but only saves the raw data."""
    #os.makedirs(_get_savedir(), exist_ok=True)
    dirname = os.path.dirname(filename)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    relpath = _get_savedir() + filename

    if hasattr(data, 'raw'):
        relfilename, relext = os.path.splitext(relpath)
        try:
            if not overwrite:
                f, rawrelpath = get_free_file(relfilename + ".sir")
            else:
                rawrelpath = relfilename + ".sir"
                f = open(rawrelpath, 'wb')
        except IOError:
            logger.error("Could not create the filename")
        else:
            np.savez(f, **data.raw())
            f.close()
    else:
        # TODO: use custom error type
        raise AttributeError("{} has no 'raw' method.".format(str(data)))

def load(filename, basedir=None):
    #basename, ext = _parse_filename(filename)
    #filename = basename if ext is None else basename + '.' + ext
    with open(_get_savedir(basedir) + filename, 'rb') as f:
        try:
            return dill.load(f)
        except EOFError:
            logger.warning("File {} is corrupted or empty. A new "
                           "one is being computed, but you should "
                           "delete this one.".format(filename))
            raise FileNotFoundError

def loadraw(filename, basedir=None, return_path=False):
    fn, ext = os.path.splitext(filename)
    path = _get_savedir(basedir) + filename
    rawpath = _get_savedir(basedir) + fn + '.sir'

    # Try the raw path first
    if os.path.exists(rawpath):
        savepath = rawpath
        data = np.load(rawpath)
    else:
        savepath = path
        data = np.load(path)

    # Return
    if return_path:
        return data, savepath
    else:
        return data

def paramstr(x):
    """Sanitize a parameter in a way that's adequate for filenames.
    """
    if isinstance(x, numbers.Number):
        s = str(x).replace('.', '-')
    else:
        raise ValueError("Unsupported parameter type '{}'."
                         .format(type(x)))
    return s

###########################
# Internal functions

_savedir = ""
_max_files = 100 # Maximum number we will append to a file to make it unique. If this number is exceeded, an error is raised

def _get_savedir(savedir=None):
    # TODO: Replace with os.path.normpath

    if savedir is None:
        #return ""
        savedir = _savedir
    while len(savedir) > 0:
        if savedir[-1] == '/':
            savedir = savedir[:-1]
        else:
            break

    if len(savedir) > 0:
        savedir += '/'
    return savedir

# def _parse_filename(filename):
#     """Replace . with /. Return the extension separately, if present."""
#     flst = filename.split('.')
#     if flst[-1] in extensions:
#         filename = '/'.join(flst[:-1])
#         ext = flst[-1]
#     else:
#         filename = '/'.join(flst)
#         ext = None
#     return filename, ext

# def _get_free_file(relpath):
#     # Should avoid race conditions, since multiple processes may run in parallel

#     # Get a full path
#     # TODO: is cwd always what we want here ?
#     if relpath[0] == '/':
#         #relpath is already a full path name
#         pathname = relpath
#     else:
#         #Make a full path from relpath
#         pathname = os.getcwd() + '/' + relpath

#     # Make sure the directory exists
#     os.makedirs(os.path.dirname(pathname), exist_ok=True)

#     try:
#         f = open(pathname, mode='xb')
#         return f, pathname
#     except IOError:
#         name, ext = os.path.splitext(pathname)
#         for i in range(2, _max_files+2):
#             appendedname = name + "_" + str(i) + ext
#             try:
#                 f = open(appendedname, mode='xb')
#                 return f, appendedname
#             except IOError:
#                 continue

#         raise IOError
