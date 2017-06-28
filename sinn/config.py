import logging
import numpy as np

import theano_shim as shim

#logLevel = logging.DEBUG # CRITICAL, INFO, WARNING, DEBUG, ERROR, FATAL
logger = logging.getLogger('sinn.config')

integration_precision = 1
truncation_ratio = 0.001

debug_level = 2
   # 0 - Turn off all optional tests and assertions
   # 1 - Basic tests, which add minimal execution time
   # 2 - Include tests using eval on Theano variables. This will slow execution
   # 3 - Asserts are added to the Theano graph. This may prevent certain Theano optimizations
   # NOTE: The debug_level is new, and only a few functions yet use it.

disk_cache_file = ""

# List of optional librairies we want to load.
# Some code may e.g. choose to use theano objects if it is available
# They may be added later with
# `config.librairies.add('packagename')`
# `config.reload`   (TODO)

librairies = set()

floatX = 'float32'
cast_floatX = float
rel_tolerance = 1e-5
abs_tolerance = 1e-8
    # This just creates the floatX variables. They are actually initialized below.

logging_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_librairies(library_list):
    global librairies, floatX, cast_floatX

    library_set = set(library_list)
    if 'theano' in library_set.difference(librairies):
        try:
            shim.load(load_theano=True, reraise=True)
        except ImportError as e:
            logger.error("Unable to load Theano.")
            logger.error(str(e))
            library_set.remove('theano')

    librairies.union(set(library_list))
    set_floatX()

def unload_librairies(library_list):
    global librairies

    if 'theano' in set(library_list):
        shim.load(load_theano=False)

    librairies.difference_update(library_list)
    set_floatX()

def load_theano(flag=True):
    """
    Call this to activate Theano in Theano-aware modules.
    Best done at the top of a script, right after the imports.
    """
    if flag:
        load_librairies(['theano'])
    else:
        unload_librairies(['theano'])

def use_theano():
    """Flag method: returns True if Theano is used."""
    return shim.config.use_theano


######################
# Set numerical tolerance
# This determines how close two numbers have to be to be consider equal

precision_dict = {
    '32': {'abs': 1e-4,
           'rel': 1e-4},
    '64': {'abs': 1e-10,
           'rel': 1e-8}}

def get_tolerance(var, tol_type):
    """
    Parameters
    ----------
    var: variable
        Variable for which we want to know the numerical tolerance.
    tol_type:
        Tolerance type. One of 'abs' or 'rel'.

    Returns
    -------
    float
    """
    var_type = shim.asarray(var).dtype
    if var_type in [np.float32, 'float32']:
        return precision_dict['32'][tol_type]
    elif var_type in [np.float64, 'float64']:
        return precision_dict['64'][tol_type]
    else:
        raise ValueError("Unknown dtype '{}'.".format(var_type))

def get_abs_tolerance(var):
    return get_tolerance(var, 'abs')

def get_rel_tolerance(var):
    return get_tolerance(var, 'rel')


#######################
# Set functions to cast to numerical float

# TODO: Rewrite these functions so they always check the value of floatX
#       That way we can change the cast precision by just changing floatX
# TODO: Move floatX to shim


def set_floatX(floatX_str = None):
    """
    Set the floatX attribute to be equal to theano.config.floatX
    If floatX_str is specified, then instead both the sinn and theano floatX
    are set to that value.
    """
    global floatX, cast_floatX, rel_tolerance, abs_tolerance

    if floatX_str is None:
        if 'theano' in librairies:
            floatX = shim.gettheano().config.floatX
            assert(floatX in ['float64', 'float32'])
            # if floatX == 'float32':
            #     cast_floatX = np.float32
            # elif floatX == 'float64':
            #     cast_floatX = np.float64
            # else:
            #     raise ValueError("The theano float type is set to '{}', which is unrecognized.".format(theano.config.floatX))
        else:
            if float(0.09) * 1e10 == 9e8:
                # Evaluates to true on a 64-bit float, but not a 32-bit.
                floatX = 'float64'
            else:
                floatX = 'float32'
    else:
        assert(floatX_str in ['float64', 'float32'])
        if 'theano' in librairies:
            shim.gettheano().config.floatX = floatX_str
        floatX = floatX_str

    cast_floatX = lambda x: shim.cast(x, floatX)


    # Direct access to the floatX tolerance:
    rel_tolerance = get_rel_tolerance(cast_floatX(1))
    abs_tolerance = get_abs_tolerance(cast_floatX(1))

set_floatX()



