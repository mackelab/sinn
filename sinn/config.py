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
        import theano.ifelse
    except ImportError:
        print("The theano library was not found.")
        librairies.remove('theano')
        use_theano = False
    finally:
        use_theano = True

#######################
# Set functions to cast to numerical float
if use_theano
    if theano.config.floatX == 'float32':
        floatX = np.float32
    elif theano.config.floatX == 'float64':
        floatX = np.float64
    else:
        raise ValueError("The theano float type is set to '{}', which is unrecognized.".format(theano.config.floatX))
else:
    floatX = float


######################
# Set numerical tolerance

# Check if we have single or double precision
# Tolerance sets how close two numbers have to be to be consider equal
if floatX(0.09) * 1e10 == 9e8:
    # Double precision
    rel_tolerance = 1e-12
    abs_tolerance = 1e-12
else:
    # Float32 has ~10 significant digits -> leaves 6
    rel_precision = 1e-4
    abs_tolerance = 1e-4


# TODO: reload function. Should reload lib.py as well
