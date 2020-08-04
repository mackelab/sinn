import builtins
import inspect
from itertools import chain

from types import FunctionType
from typing import Callable, Optional

import sinn.config

# These are added to the namespace when deserializing a function
import math
import numpy as np
import theano_shim as shim
default_namespace = {'__builtins__': __builtins__,
                     'shim': shim, 'np': np, 'math': math}

def split_decorators(s):
    s = s.strip()
    decorator_lines = []
    while s[0] == "@":
        line, s = s.split("\n", 1)
        decorator_lines.append(line)
        s = s.lstrip()
        # print(line); print(decorator_lines); print(s)
    return decorator_lines, s

def serialize(f):
    """
    WIP. Encode a function into a string.
    Accepts only definitions of the form::

        def func_name():
            do_something

    or::

        @decorator
        def func_name():
            do_something

    This excludes, e.g. lambdas and dynamic definitions like ``decorator(func_name)``.
    However there can be multiple decorators.

    Upon decoding, the string is executed in place with :func:`exec`, and the
    user is responsible for ensuring any names referred to within the function
    body are available in the decoder's scope.
    """
    if isinstance(f, FunctionType):
        s = inspect.getsource(f)
        decorator_lines, s = split_decorators(s)
        if not s.startswith("def "):
            raise ValueError(
                f"Cannot serialize the following function:\n{s}\n"
                "It should be a standard function defined in a file; lambda "
                "expressions are not accepted.")
        return "\n".join(chain(decorator_lines, [s]))
    else:
        raise TypeError

def deserialize(s: str,
                globals: Optional[dict]=None, locals: Optional[dict]=None):
    """
    WIP. Decode a function from a string.
    Accepts only strings of the form::

        def func_name():
            do_something

    or::

        @decorator
        def func_name():
            do_something

    This excludes e.g. lambdas and dynamic definitions like ``decorator(func_name)``.
    However there can be multiple decorators using the '@' syntax.

    The string is executed in place with :func:`exec`, and the arguments
    `globals` and `locals` can be used to pass defined names.
    The two optional arguments are passed on to `exec`; the deserialized
    function is injected into `locals` if passed, otherwised into `global`.

    .. note:: The `locals` namespace will not be available within the function.
       So while it may be used to define decorators, generally `globals` is
       the namespace to use.

    .. note:: A few namespaces are added automatically to globals; by default,
       these are ``__builtins__``, ``shim``, ``np`` and ``math``. This can
       be changed by modifying the module variable
       `~sinn.function_serialization.default_namespace`.

    .. note:: Both `locals` and `globals` will be mutated by the call (in
       particular, the namespaces mentioned above are added to `globals` if not
       already present). If this is not desired, consider making a shallow copy
       of the dicts before passing to `deserialize`.
    """
    msg = ("Cannot decode serialized function. It should be a string as "
           f"returned by inspect.getsource().\nReceived value:\n{s}")
    if not sinn.config.trust_all_inputs:
        raise RuntimeError("Deserialization of functions saved as source code "
                           "requires executing them with `exec`, and is only "
                           "attempted if `sinn.config.trust_all_inputs` is "
                           "set to `True`.")
    if globals is None and locals is not None:
        # `exec` only takes positional arguments, and this combination is not possible
        raise ValueError("[deserialize]: Passing `locals` argument requires "
                         "also passing `globals`.")
    if isinstance(s, str):
        decorator_lines, s = split_decorators(s)
        if not s[:4] == "def ":
            raise ValueError(msg)
        fname = s[4:s.index('(')].strip() # Remove 'def', anything after the first '(', and then any extra whitespace
        s = "\n".join(chain(decorator_lines, [s]))
        if globals is None:
            globals = default_namespace.copy()
            exec(s, globals)
            f = globals[fname]
        elif locals is None:
            globals.update(default_namespace)
            exec(s, globals)
            f = globals[fname]
        else:
            globals.update(default_namespace)
            exec(s, globals, locals)  # Adds the function to `locals` dict
            f = locals[fname]
        return f
    else:
        raise ValueError(msg)

json_encoders = {FunctionType: serialize}
