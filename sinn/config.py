import numpy as np

integration_precision = 1
truncation_ratio = 0.001

# List of optional librairies we want to load.
# Some code may e.g. choose to use theano objects if it is available
# They may be added later with
# `config.librairies.add('packagename')`
# `config.reload`   (TODO)

librairies = set()
#librairies = ['theano']


if 'theano' in librairies:
    try:
        import theano
        #librairies.append('theano')
    except ImportError:
        print("The theano library was not found.")
        librairies.remove('theano')
        use_theano = False
    finally:
        use_theano = True
else:
    use_theano = False

#######################
# Set functions to cast to numerical float

# TODO: Rewrite these functions so they always check the value of floatX
#       That way we can change the cast precision by just changing floatX

if use_theano:
    floatX = theano.config.floatX
    if floatX == 'float32':
        cast_floatX = np.float32
    elif floatX == 'float64':
        floatX = np.float64
    else:
        raise ValueError("The theano float type is set to '{}', which is unrecognized.".format(theano.config.floatX))
else:
    cast_floatX = float
    if cast_floatX(0.09) * 1e10 == 9e8:
        # Evaluates to true on a 64-bit float, but not a 32-bit.
        floatX = 'float64'
    else:
        floatX = 'float32'


######################
# Set numerical tolerance
# This determines how close two numbers have to be to be consider equal

precision_dict = {
    '32': {'abs': 1e-4,
           'rel': 1e-4},
    '64': {'abs': 1e-12,
           'rel': 1e-12}}

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
    var_type = np.asarray(var).dtype
    if var_type == np.float32:
        return precision_dict['32'][tol_type]
    elif var_type == np.float64:
        return precision_dict['64'][tol_type]
    else:
        raise ValueError("Unknown dtype '{}'.".format(var_type))

def get_abs_tolerance(var):
    return get_tolerance(var, 'abs')

def get_rel_tolerance(var):
    return get_tolerance(var, 'rel')

# Direct access to the floatX tolerance:
rel_tolerance = get_rel_tolerance(cast_floatX(1))
abs_tolerance = get_abs_tolerance(cast_floatX(1))

