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

import mackelab as ml
import mackelab.iotools

extensions = ['sin', 'sir', 'dat', 'txt']
known_types = {}

##########################
# Public API
# DEPRECATED: Use mackelab.iotools

def save(file, data, format='npr', overwrite=False):
    ml.iotools.save(file, data, format, overwrite)

def saveraw(file, data, overwrite=False):
    ml.iotools.save(file, data, 'npr', overwrite)

def load(filename, types=None, load_function=None, input_format=None):
    global known_types
    if types is not None:
        types = known_types.copy().update(types)
    else:
        types = known_types
    return ml.iotools.load(filename, types, load_function, input_format)

def loadraw(filename, basedir=None, return_path=False):
    fn, ext = os.path.splitext(filename)
    if basedir is None:
        basedir = ""
    path = basedir + filename
    rawpath = basedir + fn + '.sir'  # old extension

    # Try the raw path first
    if os.path.exists(rawpath):
        savepath = rawpath
        data = ml.iotools.load(rawpath, input_format='npr')
    else:
        savepath = path
        data = ml.iotools.load(path, input_format='npr')

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

