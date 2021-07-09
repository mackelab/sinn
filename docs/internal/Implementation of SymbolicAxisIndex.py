# %% [markdown]
# # Implementation of `SymbolicAxisIndex`
#
# `SymbolicAxisIndex` subclasses `GraphExpression` (which is used as a mixin class) to allow using `AxisIndex` objects as Theano `Variables` in expression graphs. To work well, these objects need to work in all the ways a Theano Variable would be expected to work. This includes `clone`, `copy`, `eval` and support for pickle.
#
# This file serves to document how `SymbolicAxisIndex` types behave, and can be used to verify that they still follow expected behaviour.

# %%
import mackelab_toolbox as mtb
import mackelab_toolbox.typing
from mackelab_toolbox.cgshim import shim
shim.load('theano')
mtb.typing.freeze_types()

# %%
import numpy as np
import sinn
from sinn.axis import RangeAxis

# %%
import theano

# %%
import pickle

# %%
axis = RangeAxis(label="A", step=.1, min=0, max=4)

# %%
k_data = shim.shared(np.array(1, dtype='int8'), name="k data")

# %% [markdown]
# The easy case: a plain `GraphExpression`, without the extra `AxisIndex` machinery.

# %%
k = shim.graph.GraphExpression(k_data.type)

# %%
type(k)

# %%
k.eval({k:1})

# %%
(k+1).eval({k:1})

# %%
k2 = k.copy()
k2.eval({k:1}) == k.eval({k:1})

# %%
(k+1).clone()

# %%
shim.debugprint(k)

# %%
shim.debugprint(k2)

# %% [markdown]
# The target case: an `AxisIndex`, using `GraphExpression` as mixin.

# %%
k = axis.Index(k_data)

# %%
type(k)

# %%
k.eval()

# %% [markdown]
# Expressions to test that custom operations a) work and b) don't break Theano's graph optimizer.  
# (One thing that tended to happen in earlier versions was that the `__eq__` would prevent the optimization loop from terminating, by always returning False.)

# %%
(k+1-1).eval()

# %%
shim.eval( shim.sqrt( ((k+1) / (k+1))**2 ) )

# %%
ksymb = shim.tensor(k_data)

# %%
shim.graph.compile([ksymb], [shim.graph.clone(k+1-1, {k: ksymb})])

# %%
k2 = k.copy()
k2.eval() == k.eval()

# %%
(k+1-1).clone()

# %%
shim.debugprint(k)

# %%
shim.debugprint(k+1)

# %% [markdown]
# ---
# **Pickling**
#
# Being able to pickle objects is generally useful (in particular, a lot of parallelization libraries rely on it). But most important for us that Theano uses pickling to manage its compile cache: if a computation graph can't be pickled, it will never be cached.
#
# To avoid duplicate `Axis` instances, they are stored in a registry.
# If an `Axis` object is pickled and unpickled in the same process, it will reuse the instance from the registry instead of creating another (thus ensuring that to each history is associated one, and only one, `Axis`).
# > *NOTE*: There can still be one duplicate if the pickled object is unpickled in another process, which also creates its own Axis. This could probably be avoided if it becomes a problem.

# %%
import sinn.axis
sinn.axis._instance_registry

# %%
type(axis)

# %%
axis2 = pickle.loads(pickle.dumps(axis))

# %%
axis is axis2

# %%
axis.Index is axis2.Index

# %% [markdown]
# Pickling an `Axis`

# %%
pickle.dumps(k.axis)

# %% [markdown]
# Pickling an `AxisIndex`. This is more tricky, because the `AxisIndex` are created dynamically (one for each `Axis` instance). So we patched `ElemwiseWrapper.__getstate__` to recognize when it wraps an operation on an `AxisIndex`; when that is the case, the `AxisIndex` is replaced by the `Axis` instance (which *is* pickleable) and data identify which specific AxisIndex (Symbolic/Numeric, Absolute/Delta) to use. The `ElemwiseWrapper.__setstate__` is similarly patched to reverse this process.

# %%
pickle.dumps(k)

# %%
k2 = pickle.loads(pickle.dumps(k))

# %%
k2.axis is k.axis

# %%
(k2+1).eval() == (k+1).eval()

# %% [markdown]
# ---
# **Mechanism of the `_instance_registry` for pickling**
#
# The code below illustrates how the `_instance_registry` works to avoid duplicate instances. It is the sandbox implementation that served as a template for the pickling implementation in `Axis`.

# %%
import pickle

# %%
_instance_registry = {}


# %%
class Foo:
    a = 3
    _pickle_key = None
    
    def __new__(cls, pickle_key=None):
        print("__new__")
        global _instance_registry
        if pickle_key in _instance_registry:
            obj = _instance_registry[pickle_key]
            obj._skip_init = True
        else:
            obj = super().__new__(cls)
            if pickle_key is None:
                pickle_key = id(obj)
            _instance_registry[pickle_key] = obj
        return obj
    
    def __init__(self, pickle_key=None, **kwargs):
        print("__init__")
        if getattr(self, '_skip_init', False):
            del self._skip_init
            return
        if pickle_key is None:
            pickle_key = id(self)
        self._pickle_key = pickle_key
    
    def __getnewargs__(self):
        print("__getnewargs__")
        return (self._pickle_key,)
    
    def __getstate__(self):
        print("__getstate__")
        return (self.a,)
    
    def __setstate__(self, state):
        print("__setstate__")
        if getattr(self, '_skip_init', False):
            del self._skip_init
            return
        
        a, = state
        self.a = a    


# %%
foo1 = Foo()

# %%
_instance_registry

# %%
s = pickle.dumps(foo1)
s

# %% [markdown]
# `pickle_load` detects that an object is already in the registry and loads it

# %%
foo2 = pickle.loads(s)

# %%
foo2 is foo1

# %% [markdown]
# `__setstate__` is skipped when class is loaded from registry

# %%
foo1.a = 5

# %%
foo3 = pickle.loads(s)

# %%
foo3.a

# %% [markdown]
# Can load in a new session. To test, reload notebook and execute the cells defining `pickle`, `_instance_registry` and `Foo`.

# %%
s = b'\x80\x04\x95%\x00\x00\x00\x00\x00\x00\x00\x8c\x08__main__\x94\x8c\x03Foo\x94\x93\x94\x8a\x06\xa0\x89f|?\x7f\x85\x94\x81\x94K\x03\x85\x94b.'

# %%
foo4 = pickle.loads(s)

# %% [markdown]
# ---

# %%
