# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:25:35 2016

@author: rene
"""

# TODO: Add .dat extension in save_data if there is none.
#       Try other reasonable combinations (.dat, .txt, no ext) in load_data if original filename is not found
# TODO: Function that goes through a list of property names.
#       If data contains the property, assign it, otherwise assign a default value.
#       Return a namedtuple

import dill
import os
import os.path

##########################
# Public API

def save(filename, data):
    try:
        relpath = _get_savedir() + filename
        f, realrelpath = _get_free_file(relpath)

    except IOError:
        print("Could not create the filename")
        realrelpath = None

    else:
        dill.dump(data, f)
        f.close()

    return realrelpath


def load(filename):
    with open(_get_savedir() + filename, 'rb') as f:
        return dill.load(f)


###########################
# Internal functions


_savedir = "data"
_max_files = 100 # Maximum number we will append to a file to make it unique. If this number is exceeded, an error is raised

def _get_savedir():
    global _savedir

    while len(_savedir) > 0:
        if _savedir[-1] == '/':
            _savedir = _savedir[:-1]
        else:
            break

    return _savedir + '/'

def _get_free_file(relpath):
    # Should avoid race conditions, since multiple processes may run in parallel

    # Get a full path
    # TODO: is cwd always what we want here ?
    if relpath[0] == '/':
        #relpath is already a full path name
        pathname = relpath
    else:
        #Make a full path from relpath
        pathname = os.getcwd() + '/' + relpath

    # Make sure the directory exists
    os.makedirs(os.path.dirname(pathname), exist_ok=True)

    try:
        f = open(pathname, mode='xb')
        return f, pathname
    except IOError:
        name, ext = os.path.splitext(pathname)
        for i in range(2, _max_files+2):
            appendedname = name + "_" + str(i) + ext
            try:
                f = open(appendedname, mode='xb')
                return f, appendedname
            except IOError:
                continue

        raise IOError
