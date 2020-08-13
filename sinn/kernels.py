# -*- coding: utf-8 -*-
"""
Created on Thu Feb 2 2017

Author: Alexandre René
"""
import numpy as np
from collections import OrderedDict
import logging
import functools
import abc
logger = logging.getLogger(__file__)

import pydantic
from pydantic import validator, root_validator, Field
# from pydantic.fields import ModelField
# from pydantic.dataclasses import dataclass
import dataclasses  # For lighter structures that don't need validation
from typing import Optional, Tuple, List, Type, Any, Union
from numbers import Number
import mackelab_toolbox as mtb
from mackelab_toolbox.typing import NPValue, Slice, FloatX, Tensor
from mackelab_toolbox.parameters import ParameterSet

import theano_shim as shim
import sinn.common as com
from sinn.common import TensorWrapper, TensorDims
import sinn.config as config
import sinn.mixins as mixins
import sinn.diskcache as diskcache
from sinn.convolution import ConvolveMixin
from sinn.histories import History, Series

#TODO: Discretized kernels as a mixin class

# Don't hide CachedKernel within the decorator's scope, otherwise it can't be pickled
def checkcache(init):
    @functools.wraps(init)
    def cachedinit(self, name, *args, **kwargs):
        try:
            retrieved_kernel = diskcache.load(mtb.utils.stablehexdigest(self.json()))
        except KeyError:
            init(self, name, *args, **kwargs)
        else:
            self.__dict__update(retrieved_kernel.__dict__)
            self.name = name
    return cachedinit

# No cache decorator for generic kernels: f might be changed after the initialization,
# and any kernel with f=None might hit the same cache.
class Kernel(ConvolveMixin, com.KernelBase, abc.ABC):
    """
    .. warning:: Currently the `shape` is not used as below, but should simply be
    the shape of the result after application of the kernel.

    Generic Kernel class. All kernels should derive from this.
    Kernels associated to histories with shape (M,) should have shape (M,M),
    with indexing following kernel[to idx][from idx]. (See below for caveats.)

    Derived kernels must implement `_eval`. They should NOT implement `eval`,
    nor `__init__` (`__init__` is reserved by dataclasses.dataclass; use
    `__post_init__` instead.) They should probably also implement
    `__post_init__` and `_convolve_single_t`.

    Optimization note:
    If you only need the diagonal of the kernel (because it only depends
    on the 'from' population), then it doesn't need to be MxM. Typical
    options then for the 'shape':
    MxM:
        Usual shape. If the output is size M, it will be repeated (with tile)
        to produce an MxM matrix. Avoid this if you can use broadcasting instead.
    1xM or Mx1:
        Leave the number of dimensions unchanged, but flatten into one column / rom.
        This will typically work best with broadcasting.
        1xM : independent of 'to' pop.
        Mx1 : independent of 'from' pop.
    M:
        Throw away the row dimension and treat the result as a 1D array. Should be
        equivalent to defining a diagonal array.
    Refer to the `shape_output` method to see exactly how the output is reshaped.

    Parameters
    ----------
    name: str
        A unique identifier. May be printed to identify this kernel in output.
    shape: tuple of ints
    t0: float
        The time corresponding to f(0). Kernel is zero before this time.
    memory_time: float
        Time after which we can truncate the kernel.
        _Must not be a symbolic (e.g. Theano) variable._
        Instead of `memory_time`, one can instead specify τ:=`decay_const`. In
        this case this case a suitable memory time is deduced by assuming an
        exponential kernel of shape exp( -τ(t-t0) ) and using the value of
        `sinn.config.truncation_ratio` to compute the point at which the
        fraction remaining mass in the kernel is less than `truncation_ratio`.
    decay_const: float
        See `memory_time`.
    dtype: Numpy dtype
    """
    name       : str
    shape      : Tuple[NPValue[np.int16], ...]
    t0         : FloatX   = shim.cast_floatX(0.)
    decay_const: Optional[Tensor[FloatX]]
    memory_time: Optional[FloatX]
    dtype      : Type = np.dtype(shim.config.floatX)
    ndim       : NPValue[np.int8] = None

    # ----------
    # Pydantic config and validators

    class Config:
        # Allow assigning other attributes during initialization.
        extra = 'allow'
        json_encoders = {slice: Slice.json_encoder}

    @validator('ndim', pre=True, always=True)
    def set_ndim(cls, v, values):
        shape = values.get('shape', None)
        if shape:
            ndim = np.int8(len(shape))
        if v is not None and v != ndim:
            name = values.get('name', "")
            raise ValueError(f"While instantiating kernel {name}: `ndim` argument"
                             "unnecessary: it is calculated from `shape`.")
        return ndim

    @validator('memory_time', pre=True, always=True)
    def set_memory_time(cls, memory_time, values):
        """
        If memory_time is not provided, compute a default value based on
        `sinn.config.truncation_ratio`.
        """
        # Truncating after memory_time should not discard more than a fraction
        # config.truncation_ratio of the total area under the kernel.
        # (Divide ∫_t^∞ by ∫_0^∞ to get this formula.)
        decay_const, t0 = (values.get(x, None) for x in ('decay_const', 't0'))
        if memory_time is None and decay_const is not None and t0 is not None:
            assert(0 < config.truncation_ratio < 1)
            # We want a numerical value, so we use the test value associated to the variables
            decay_const_val = np.max(shim.get_test_value(decay_const))
            memory_time = t0 - decay_const_val * np.log(config.truncation_ratio)
        return memory_time

    # ---------------
    # Initialize internal vars and sanity test on instance

    def __init__(self, **kwargs):
        """
        Initializes the `_evalndim` internal variable, which is used when
        evaluating on time arrays.
        Also performs a sanity check that `self.eval` is computable.
        """
        super().__init__(**kwargs)
        self._evalndim = self.eval(0).ndim
            # even with a Theano function, this returns a Python scalar
        self._slice_cache = KernelSliceCache(self)
        # Sanity test on the eval method's shape
        try:
            eval_at_0 = shim.get_test_value(self.eval(0), nofail=True)
                # get_test_value returns None if eval(0) is a Theano var with no test value
            if eval_at_0 is not None:
                self.shape_output(eval_at_0, ())
        except (AssertionError, ValueError):
            raise ValueError("The parameters to the kernel's evaluation "
                             "function seem to have incompatible shapes. "
                             "The kernel's output has shape {}, but "
                             "you've set it to be reshaped to {}."
                             .format(self.eval(0).shape, self.shape))

    # ---------
    # Properties

    @property
    @abc.abstractmethod
    def mass(self):
        """Integral of the kernel. Derived classes should implement this method
        by integrating the expression analytically.
        """
        return NotImplemented

    @property
    def params(self):
        return ParameterSet(self.dict())

    @property
    def contravariant_axes(self):
        """Returns the "dot product" axes (those we expect to sum over).
        At present this is hard coded ( sum over last dimension if ndim == 2),
        but in the future we could allow this to be modified.
        """
        # Always use negative indices: these are used for summations on axes
        # which may have additional prepended dimensions for time, batches, ...
        if self.ndim == 2:
            return (-1,)
        else:
            return ()

    @property
    def dim_map(self):
        """
        A tuple, of length `self.ndim`, which lists for each output dimension
        the input dimension it corresponds to.
        Like `contravariant_axes`, currently hardcoded (always only 1 input dim),
        but in the future we could allow this to be modified.
        """
        return (1,)*self.ndim

    # --------
    # Methods to implement in derived classes

    @abc.abstractmethod
    def _eval(self, t, from_idx=slice(None, None)):
        """Derived classes must implement _eval. This is the functional
        form of the kernel."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Return a kernel which is equivalent to calling `self.eval(t)[idx]`,
        without the unnecessary calculation of the components excluded by `idx`.
        Not all kernels may provide this method.
        """
        raise NotImplementedError(f"{type(self)} does not implement indexing "
                                  "the kernel instance.")

    def _discretize_kernel(self, time_array, dt, idx_shift,
                           shape, name, symbolic, iterative):
        """
        Return an empty history to store the discretized kernel.
        Does not need to be subclassed if the default of returning
        DiscretizedKernelSeries is satisfactory.
        """
        return DiscretizedKernelSeries(
            original_kernel=self,
            idx_shift=idx_shift,
            time_array=time_array,
            dt=dt,
            shape=shape,
            name=name,
            symbolic=symbolic,
            iterative=iterative)

    @abc.abstractmethod
    def _convolve_single_t(self, hist, t, kernel_slice):
        """
        Knowing nothing of the kernel, we punt to the history's `convolve`,
        which will discretize the kernel and treat it as an arbitrary function.

        If derived classes don't want to implement `_convolve_single_t`, they
        should define the method as
            def _convolve_single_t(self, hist, t, kernel_slice):
                super()._convolve_single_t(hist, t, kernel_slice)
        """
        return hist._convolve_single_t(self, t, kernel_slice)

    # --------
    # Public facing methods

    # Public wrapper around _eval
    # TODO: Use __call__ instead ?
    def eval(self, t, from_idx=slice(None,None)):
        """
        Returns 0 for t < t0 or t >= t0 + memory_time.
        The asymmetric bounds ensure that if e.g. memory_time = 4Δt,
        than exactly 4 time bins will have a non-zero value.
        """
        if not shim.isscalar(t):
            tshape = t.shape
            #t = shim.add_axes(t, self.params[0].ndim-1, 'right')
                # FIXME: This way of fixing t dimensions is not robust
            if shim.isscalar(from_idx):
                t = shim.add_axes(t, self._evalndim - 1, 'right')
            else:
                t = shim.add_axes(t, self._evalndim, 'right')
            final_shape, ndim = self.get_final_shape(tshape)
            res = self.shape_output(self._eval(t, from_idx), tshape)
            return shim.switch(shim.and_(shim.ge(t, self.t0), shim.lt(t, self.t0+self.memory_time)),
                               shim.cast(res, self.dtype),
                               shim.zeros_like(res, dtype=self.dtype))
                               #shim.zeros(final_shape, ndim=ndim))
        else:
            tshape = ()
            final_shape, ndim = self.get_final_shape(tshape)
            res = self.shape_output(self._eval(t, from_idx), tshape)
            return shim.ifelse(shim.and_(shim.ge(t, self.t0), shim.lt(t, self.t0+self.memory_time)),
                               shim.cast(res, self.dtype),
                               shim.zeros_like(res, dtype=self.dtype))
                               #shim.zeros(final_shape, ndim=ndim))

    def theano_reset(self):
        """Make state clean for building a new Theano graph.
        This clears any discretizations of this kernel, since those
        may depend on different parameters."""
        logger.info("Resetting kernel {} for Theano".format(self.name))
        #attr_to_del = []
        for attr in dir(self):
            if attr[:9] == "discrete_":
                logger.info("Clearing and resetting stale kernel " + attr)
                getattr(self, attr).clear()
                getattr(self, attr).theano_reset()
        #        attr_to_del.append(attr)
        #for attr in attr_to_del:
        #    logger.info("Removing stale kernel " + attr)
        #    delattr(self, attr)

    def is_theano(self):
        return any(shim.is_theano_object(p) for p in self.params)

    def get_final_shape(self, tshape):
        if not shim.is_theano_object(tshape):
            assert(isinstance(tshape, tuple))
        # tshape is expected to be a tuple, but could be a Theano object
        # in that case it always has at least dimension 1 (but its shape may be 0)

        #tshapearr = shim.asarray(tshape, dtype='int8')
        if hasattr(tshape, 'ndim'):
            shapedims = tshape.ndim
        else:
            shapedims = len(tshape)
        if shapedims == 0:
            # tshape is in fact a scalar - no time dimension
            final_shape = self.shape
            ndim = len(self.shape)
        else:
            # add a time dimension
            assert(shapedims == 1)
            ndim = 1 + len(self.shape)
            final_shape = shim.concatenate( (tshape, self.shape) )
        return final_shape, ndim

    def get_empty_convolve_result(self):
        """
        Return an empty data structure, of the expected shape and format for
        a convolution result. Sometimes useful when one needs to construct
        results iterative (see BlockKernel). If the expected result should be
        in e.g. a sparse array, this method allows a subclass to specify that.
        """
        if shim.is_symbolic(self.dict().values()):
            return shim.tensor(np.empty(self.shape))
        else:
            return np.empty(self.shape)


    def shape_output(self, output, tshape):
        final_shape, ndim = self.get_final_shape(tshape)
        return shim.reshape(output, final_shape, ndim=ndim)

    def compute_discretized_kernels(self):
        """
        Compute all attached discretized kernels and lock them. This allows
        later operations to treat them as immutable.
        """
        for kernel in self._discretized_kernels:
            kernel._compute_up_to('end')
            kernel.lock()

    def discretize(self, refhist, compute=False):
        """
        Discretize a kernel, aligning it with the provided reference history.
        Discretized kernels are cached, so calling this function again with
        the same `refhist` argument will return the previously cached result.
        It is thus not an issue to call this function multiple times with
        `compute=True`.

        Parameters
        ----------
        refhist: History
            Reference history. The discretized kernel will have time stops
            aligned with those of `refhist`.
        eval: bool (default False)
            If True, also compute and lock the resulting kernel.

        Returns
        -------
        History
            Usually a Series, but different kernels may return different History
            subclasses.
        """

        dis_attr_name = "discrete_" + str(id(refhist))  # Unique id for discretized kernel

        if hasattr(self, dis_attr_name):
            # TODO: Check that this history (refhist) hasn't changed
            return getattr(self, dis_attr_name)

        else:
            #TODO: Add compability check of the kernel's shape with this history.
            #shim.check(self.shape == refhist.shape*2)
            #    # Ensure the kernel is square and of the right shape for this history

            # The kernel may start at a position other than zero, resulting in a shift
            # of the index corresponding to 't' in the convolution
            # TODO: if kernel.t0 is not divisible by dt, do an appropriate average
            idx_shift = int(round((self.t0 / refhist.dt64)))
                # We don't use shim.round because time indices must be Python numbers
            t0 = shim.cast(idx_shift * refhist.dt64, shim.config.floatX)
                # Ensure the discretized kernel's t0 is a multiple of dt

            memory_idx_len = int(self.memory_time // refhist.dt64) - idx_shift
                # It is essential here to use the same forumla as pad_time
            if memory_idx_len > 100:
                memory_idx_len -= 1
                # We substract one because for long enough kernels one bin more
                # or less makes no difference,
                # and doing so ensures that padding with `memory_time` always
                # is sufficient (no dumb numerical precision errors adds a bin)
                # NOTE: self.memory_time includes the time between 0 and t0,
                # and so we need to substract idx_shift to keep only the time after t0.
            #full_idx_len = memory_idx_len + idx_shift
            #    # `memory_time` is the amount of time before t0
            dis_name = ("dis_" + self.name + " (" + refhist.name + ")")
            # if shim.is_theano_variable(self.eval(0)):
            #     symbolic=True
            # else:
            #     symbolic=False
            symbolic=False

            if memory_idx_len <= 0:
                # The memory time of the kernel is shorter than the time step
                logger.warning("Kernel {} has length 0. Any parameters "
                               "associated to this kernel will be ignored and "
                               "all convolutions will be zero."
                               .format(self.name))
                memory_idx_len = 0
            tarr = t0 + np.arange(memory_idx_len)*refhist.dt64

            ################################################
            # To change discretization format, override this method
            dis_kernel = self._discretize_kernel(
                time_array = tarr,
                dt         = refhist.dt64,
                idx_shift  = idx_shift,
                shape      = self.shape,
                name       = dis_name,
                symbolic   = symbolic,
                iterative  = False)
                # iterative = False because kernels are non-iterative by
                # definition: they only depend on their parameters
            ################################################

            # Set the update function for the discretized kernel
            if config.integration_precision == 1:
                _kernel_func = self.eval
            elif config.integration_precision == 2:
                # TODO: Avoid recalculating eval at the same places by writing
                #       a _compute_up_to function and passing that to the series
                _kernel_func = lambda t: (self.eval(t) + self.eval(t+refhist.dt)) / 2
            else:
                # TODO: higher order integration with trapeze or simpson's rule
                raise NotImplementedError

            ## Kernel functions can only be defined to take times, so we wrap
            ## the function
            def kernel_func(t):
                t = dis_kernel.get_time(t)
                return _kernel_func(t)

            dis_kernel.set_update_function(kernel_func, inputs=[])
                # By definition, kernels don't depend on other histories,
                # hence inputs=[]

            # Attach the discretization to the kernel instance
            setattr(self, dis_attr_name, dis_kernel)

            return dis_kernel

    # ====================================
    # Caching interface

    # @classmethod
    # def _serialize(cls, params, shape, memory_time, t0, **kwargs):
    #     """Quick 'n dirty serializer"""
    #     # **kwargs are not treated, so they should be left to their
    #     # default value of None
    #     for key, val in kwargs.items():
    #         assert(val is None)
    #     return str(cls) + str(params) + str(shape) + str(memory_time) + str(t0)
    #
    # def __hash__(self):
    #     return hash(self._serialize(self.params, self.shape, self.memory_time, self.t0))

    @property
    def _discretized_kernels(self):
        """Returns an iterable of stored discretized forms of this kernel."""
        for attr in dir(self):
            if attr.startswith("discrete_"):
                yield getattr(self, attr)


    def update_params(self, **kwargs):
        # For Theano parameters when updating as part of a larger model,
        # this won't do anything: they all derive from model parameters, so
        # changes immediately propagate down. Still we leave
        # this here in case this method is called on its own.
        cant_update = Kernel.__fields__
            # Base parameters like t0 and memory_time time affect the
            # discretization and can't be updated
        invalid = [p for p in cant_update if p in kwargs]
        if len(invalid) > 0:
            raise ValueError(f"Kernel parameters {invalid} cannot be updated "
                             "in place.")
        # Reset all attached discretized kernels, since those are no longer valid
        logger.info("Updating kernel " + self.name + "...")
        kernel_to_del = []
        for kernel in self._discretized_kernels:
            if not kernel.use_theano:
                # Theano kernels are not precomputed, so they also
                # don't need to be deleted
                kernel.clear()
            else:
                assert kernel.cur_tidx < kernel.t0idx
        # Update non-Theano parameters
        for p, val in kwargs.items():
            selfp = getattr(self, p, None)
            if selfp is None: continue  # p not relevant for this kernel
            if shim.is_graph_object(selfp):
                if not shim.is_graph_object(val):
                    raise NotImplementedError
                assert selfp is val  # Theano param's value may have changed,
                                     # but it must remain the same object
            else:
                if shim.is_graph_object(val):
                    raise NotImplementedError
                setattr(self, p, val)

        # logger.info("Reinitializing kernel {} with new parameters {}."
        #             .format(self.name, str(new_params)))
        # self.initialize(self.name, new_params, self.shape, self.memory_time, self.t0)

    # ------------
    # Operations

    def __add__(self, other):
        if isinstance(other, Kernel):
            return CompositeKernel([self, other])

@dataclasses.dataclass
class KernelSliceCache:
    """Dict-like interface which knows how to hash index slices.
    >>> κ = ExpKernel([*])
    >>> cache = KernelSliceCache()  # Normally one uses the one attached to κ
    >>> cache[κ, (0, 1)]            # Fails
    >>> cache[κ, (0, 1)] = κ[0,1]
    >>> cache[κ, (0, slice(None)] = κ[0,slice(None)]
    >>> cache[κ, (0, 1)]            # Works
    >>> cache[κ, (0, slice(None))]  # Works
    """
    kernel: Kernel
    _store: dict = dataclasses.field(default_factory=lambda:{})

    def gethash(idx):
        desc = ( ('slice', e.start, e.stop, e.step)
                 if isinstance(e, slice) else e
                 for e in idx )
        return hash( tuple(itertools.chain([self.kernel], desc)) )

    def __getitem__(self, idx):
        return self._store[self.gethash(idx)]

    def __setitem__(self, idx, value):
        self._store[self.gethash(idx)] = value

###############
# Special kernel types
# (usually return types; they should not be instantiated directly)

class CompositeKernel(Kernel):
    """Linear composition of sub-kernels by addition.
    This is what results when kernels are added.
    """
    kernels    : List[Kernel]
    # ---------
    # Overridden / auto parameters
    name       : Optional[str]
    shape      : Optional[Tuple[NPValue[np.int16], ...]]
    t0         : Optional[FloatX]
    decay_const: Optional[Tensor[FloatX]]
    memory_time: Optional[FloatX]
    dtype      : Optional[Type]
    ndim       : Optional[NPValue[np.int8]]

    # Validators
    @root_validator(pre=True)
    def set_params(cls, values):
        kernels = values.get('kernels', None)
        if kernels is None:
            return values
        values['name'] = '+'.join((κ.name for κ in kernels))
        values['shape'] = kernels[0].shape
        if not all(κ.shape == kernels['shape'] for κ in kernels):
            shapes = ', '.join((str(κ.shape for κ in kernels)))
            raise ValueError("Kernels shapes are inconsistent. They are: "
                             f"{shapes}.")
        values['t0'] = kernels[0].t0
        if not all(κ.t0 == kernels[0].t0 for κ in kernels):
            t0s = ', '.join((str(κ.t0 for κ in kernels)))
            raise ValueError("Kernels t0 values are inconsistent. They are: "
                             f"{t0s}.")
        values['memory_time'] = max(κ.memory_time for κ in kernels)
        values['dtype'] = np.result_type(κ.dtype for κ in kernels)
        values['ndim'] = kernels[0].ndim
        if not all(κ.ndim == kernels[0].ndim for κ in kernels):
            ndims = ', '.join((str(κ.ndim for κ in kernels)))
            raise ValueError("Kernels have different numbers of dimensions: "
                             f"{ndims}.")
        return values

    # -----------
    # Overridden methods

    def eval(self, *args, **kwargs):
        shim.sum([κ.eval(*args, **kwargs) for κ in self.kernels])

    def __getitem__(self, idx):
        return CompositeKernel([κ[idx] for κ in self.kernels])

    def _convolve_single_t(self, *args, **kwargs):
        shim.sum([κ._convolve_single_t(*args, **kwargs)
                  for κ in self.kernels])

class DiscretizedKernel(pydantic.BaseModel):
    original_kernel: Kernel
    idx_shift      : int

    class Config:
        extra = 'allow'

    def __new__(cls, *args, **kwargs):
        if cls is DiscretizedKernel:
            raise TypeError("Base DiscretizedKernel class cannot be "
                            "instantiated. Use one of the derived classes.")
        return super().__new__(cls)

    @property
    def contravariant_axes(self):
        return self.original_kernel.contravariant_axes

class DiscretizedKernelSeries(Series, DiscretizedKernel):
    pass

#####################$
# Implemented kernels

# See https://pydantic-docs.helpmanual.io/usage/validators/#reuse-validators
def expand_to_2D(v):
    if not hasattr(v, 'ndim'):
        v = np.array(v)
    if v.ndim == 0:
        return v.reshape((1,1))
    if v.ndim == 1:
        raise TypeError("A 1-D kernel parameter cannot be broadcast "
                        "unambiguously. Add a dimension to make it either a "
                        "row or column vector.")
    return v
def normalized_shape(name):
    return validator(name, pre=True, allow_reuse=True)(expand_to_2D)

class ExpKernel(Kernel):
    """
    An exponential kernel, of the form κ(s) = c exp(-(s-t0)/τ).
    NOTE: The way things are coded now, t_offset is considered fixed. I.e.,
          one should not try to use this in a routine seeking to optimize t_offset.

    Parameters
    ----------
    name: str
        A unique identifier. May be printed to identify this kernel in output.
    params: ExpKernel.Parameters  (in **kwargs)
        - height: float, ndarray, Theano var
          Constant multiplying the exponential. c, in the expression above.
        - decay_const: float, ndarray, Theano var
          Characteristic time of the exponential. τ, in the expression above.
    shape: tuple
        (Optional) Overrides the default shape, which is computing by
        broadcasting all elements.
    memory_time: float
        (Optional) Time after which we can truncate the kernel. If left
        unspecified, calculated automatically.
        Must *not* be a Theano variable.
    t0: float or ndarray
        Time at which the kernel 'starts', i.e. κ(t0) = c,
        and κ(t) = 0 for t< t0.
        Must *not* be a Theano variable.
    """
    # Second argument enforces the kernel expectation of 2D parameters
    memory_time: Optional[FloatX]
    height     : Tensor[FloatX, 2]
    decay_const: Tensor[FloatX, 2]
    t_offset   : Tensor[FloatX, 2]
    shape      : Optional[Tuple[NPValue[np.int16], ...]]

    # Internal variables
    # _memory_blind_time: FloatX
    # _last_t    = None  # Keep track of the last convolution time
    # _last_conv = None  # Keep track of the last convolution result
    # _last_hist = None  # Keep track of the history object used for the last convolution

    # -----------
    # Validators

    _normalize_height      = normalized_shape('height')
    _normalize_decay_const = normalized_shape('decay_const')
    _normalized_toffset    = normalized_shape('t_offset')

    @validator('shape', pre=True, always=True)
    def set_shape(cls, shape):
        """
        If `shape` is not provided, deduce it by broadcasting all parameters
        against each other.
        """
        if shape is None:
            params = [values.get(x, None)
                      for x in ('height', 'decay_const', 't_offset')]
            if all(params):
                shape = np.broadcast(shim.graph.eval(p) for p in params).shape
                # Impose a shape of at least (1,)
                if shape == ():
                    shape = (1,)
        return shape

    # ---------
    # Initialization of other internal variables

    def __init__(self, **kwargs):
        """
        Initialize internal variables.
        """
        super().__init__(**kwargs)

        """`_memory_blind_time` corresponds to the zero time point and the start
        of the kernel."""
        # When we convolve, this time window before t_offset is ignored
        # (because the kernel is zero there), and therefore not included in
        # self._last_conv. So we need to extend the kernel slice by
        # t_offset-t0 when we reuse the cache data; this is the amount of time
        # window to which the cache is "blind".
        t_offset = self.t_offset
        if shim.isshared(t_offset):
            # If t_offset is a shared variable, grab its value.
            t_offset = self.t_offset.get_value()
        elif shim.graph.is_computable([t_offset]):
            # We can evaluate the parameter (it's likely a symbolic manipulation of a
            # shared variable). This takes a few seconds, but returns a pure Python value
            t_offset = shim.eval(t_offset)
        else:
            # There's nothing we can do: t_offset must remain symbolic
            pass
        self._memory_blind_time = shim.max(t_offset) - self.t0

        """`_last_t`, `_last_conv`, `_last_hist` implement the optimization for
        exponential, whereby the last convolution result is stored, so that if
        the next call is at the following time point, we only need multiply the
        `_last_conv` by an exponential factor."""
        self._last_t    = None  # Keep track of the last convolution time
        self._last_conv = None  # Keep track of the last convolution result
        self._last_hist = None  # Keep track of the history object used for the last convolution

    # -----------
    # Properties

    @property
    def mass(self):
        return 1/self.decay_const

    # -----------
    # Overridden methods

    def _eval(self, t, from_idx=slice(None,None)):
        return shim.switch(shim.lt(t, self.t_offset[...,from_idx]),
                           shim.cast(0, self.dtype, same_kind=False),
                           self.height[...,from_idx]
                             * shim.cast(
                                shim.exp(-(t-self.t_offset[...,from_idx])
                                           / self.decay_const[...,from_idx]),
                                self.dtype) )
            # We can use indexing because we've ensured parameters are 2D


    def __getitem__(self, idx):
        try:
            kslc = self._slice_cache[idx]
        except KeyError:
            newname  = self.name + "_{idx}"
            origshape = self.shape
            newshape = tuple(e.stop-e.start for e in idx if isinstance(e, slice))
            h = shim.broadcast_to(self.height,      origshape)[idx]
            τ = shim.broadcast_to(self.decay_const, origshape)[idx]
            Δ = shim.broadcast_to(self.t_offset,    origshape)[idx]
            kslc = type(self)(
                name=newname, shape=newshape, t0=self.t0,
                memory_time = self.memory_time, dtype=self.dtype,
                height=h, decay_const=τ, t_offset = Δ)
            self._slice_cache[idx] = kslc
        return kslc


    def _convolve_single_t(self, hist, t, kernel_slice):

        #TODO: store multiple caches, one per history
        #TODO: do something with kernel_slice
        #TODO: Allow Theano to make use of the exp kernel

        # We are careful here to avoid converting t to time if not required,
        # so that kernel slicing can work on indices

        if (kernel_slice != slice(None, None)
            or shim.is_theano_object(t)):
            # HACK Our caching does not deal with Theano times, so in that
            # case we bypass that as well.
            # FIXME Ideally we would allow Theano to use the optimized exp kernel as well,
            # when we need to do an iterative computation
            return hist._convolve_single_t(self, t, kernel_slice)
                # Exit before updating _last_t and _last_conv
        elif shim.asarray(t).dtype != shim.asarray(self._last_t).dtype:
            # This condition catches the case where e.g. _last_t is
            # an index but t is a time (then t > _last_t is a bad test).
            result = hist._convolve_single_t(self, t, kernel_slice)
            self._last_conv = result
        elif self._last_conv is not None and self._last_hist is hist:
            if t > self._last_t:
                Δt = t - self._last_t
                # Compute the amount left from the cache
                reduction_factor =  self.shape_output(shim.exp(-hist.time_interval(Δt)/self.decay_const), ())
                if shim.issparse(self._last_conv):
                    reduction_factor = shim.broadcast_to(self._last_conv.shape)
                # _last_conv may be sparse, so it must be first in multiplication
                reduced_cache = self._last_conv * reduction_factor
                # if hasattr(hist, 'pop_rmul'):
                #     # FIXME The convolution needs to keep separate contributions from the different pops
                #     reduced_cache = hist.pop_rmul(reduction_factor, self._last_conv)
                # else:
                #     reduced_cache = reduction_factor * self._last_conv

                # Add the convolution over the new time interval which is not cached
                # Index_interval will look at the dtype to determine the
                # numerical rounding tolerance, so we must make sure we don't
                # upcast a float32 to float64 (e.g. if blind_time is
                # float64 but Δt is float32 )
                t0 = self._memory_blind_time
                tn = shim.cast(self._memory_blind_time + Δt,
                               min(self._memory_blind_time.dtype, Δt.dtype))
                result = ( reduced_cache
                           + hist._convolve_single_t(self, t,
                                           slice(hist.index_interval(t0),
                                                 hist.index_interval(tn))) )
                self._last_conv = result
                    # We only cache the convolution up to the point at which every
                    # population "remembers" it.
                result += hist._convolve_single_t(self, t,
                                        slice(0, hist.index_interval(self._memory_blind_time)))
                                              # 0 idx corresponds to self.t0

            elif t == self._last_t:
                result = self._last_conv
            else:
                result = hist._convolve_single_t(self, t, kernel_slice)
                self._last_conv = result
        else:
            result = hist._convolve_single_t(self, t, kernel_slice)
            self._last_conv = result

        self._last_t = t
        #self._last_conv = result
        self._last_hist = hist

        return TensorWrapper(result,
                             TensorDims(contraction=self.contravariant_axes))

    def _convolve_op_batch(self, hist, kernel_slice):
        # For batch convolutions, we punt to the history
        return hist._convolve_op_batch(self, slice(None, None), kernel_slice)

class BoxKernel(Kernel):
    """
    Kernel equal to `height` on the half-open interval [start, stop) and
    equal to zero elsewhere.

    Parameters
    ----------
    height: array-like (float)
    start : array-like (float)
    stop  : array-like (float)
    """
    height: Tensor[FloatX]
    start : Tensor[FloatX]
    stop  : Tensor[FloatX]
    # ---- Make computed params of base Kernel class optional
    memory_time: FloatX = None

    # ----------
    # Validators and initializers

    @root_validator(pre=True)
    def set_memory_time(cls, values):
        start, stop = (values.get(x, None) for x in ('start', 'stop'))
        if values.get('memory_time', None) is not None:
            raise ValueError("`memory_time` parameter unnecessary for "
                             "BoxKernel.")
        if None not in (start, stop):
            values['memory_time'] = np.asarray(shim.get_test_value(stop) -
                                               shim.get_test_value(start)).max()
        return values

    # ----------
    # Overridden functions

    def _eval(self, t, **kwargs):
        if shim.isscalar(t):
            return shim.switch(shim.and_(self.start<=t, t<self.stop),
                               self.height,
                               shim.zeros(self.shape, dtype=self.dtype))
        else:
            # t has already been shaped to align with the function output in Kernel.eval
            return shim.switch(
                shim.and_ (self.start<=t, t<self.stop),
                shim.ones (t.shape,    dtype=shim.config.floatX) * self.height,
                shim.zeros(self.shape, dtype=self.dtype))

    def __getitem__(self, idx):
        try:
            kslc = self._slice_cache[idx]
        except KeyError:
            newname  = self.name + "_{idx}"
            origshape = self.shape
            newshape = tuple(e.stop-e.start for e in idx if isinstance(e, slice))
            h  = shim.broadcast_to(self.height,  origshape)[idx]
            t1 = shim.broadcast_to(self.start,   origshape)[idx]
            t2 = shim.broadcast_to(self.stop,    origshape)[idx]
            kslc = type(self)(
                name=newname, shape=newshape, t0=self.t0,
                memory_time = self.memory_time, dtype=self.dtype,
                height=h, start=t1, stop=t2)
            self._slice_cache[idx] = kslc
        return kslc

    def _convolve_single_t(hist, t, kernel_slice):
        # TODO: Compute index_interval, start, stop only once
        # raise NotImplementedError("Untested method. Do so and remove this "
        #                           "exception.")
        tidx  = hist.get_tidx(t)
        # Defaults if no kernel_slice.
        # Indices are relative to hist.t0idx and will be shifted to tidx
        kernel_start = hist.get_tidx(self.start)
        kernel_stop  = hist.get_tidx(self.stop)
        if kernel_slice.start is not None:
            kernel_start = shim.max(kernel_start,
                                    hist.get_tidx(kernel_slice.start))
        if kernel_slice.stop  is not None:
            kernel_start = shim.min(kernel_stop,
                                    hist.get_tidx(kernel_slice.stop))
        # Shift to tidx. Remember indices are relative to hist.toidx
        Δstart = kernel_stop  - hist.get_tidx(0.)
        Δstop  = kernel_start - hist.get_tidx(0.)
        hist_slice = slice(tidx - Δstart, tidx - Δstop)
        return hist[hist_slice].sum(axis=0)*hist.dt

##########
# Wrapped kernels

class KernelWrapper(Kernel):
    """
    Defines a set of standard accessor properties to make it easier to extend
    a kernel using composition (i.e. without subclassing it).
    """
    wrapped_kernel: Kernel
    # --- Attributes which can be determined from wrapped_kernel ---
    name       : Optional[str]
    shape      : Optional[Tuple[NPValue[np.int16], ...]]
    t0         : Optional[FloatX]
    decay_const: Optional[Tensor[FloatX]]
    memory_time: Optional[FloatX]
    dtype      : Optional[Type]
    ndim       : Optional[NPValue[np.int8]]

    def __new__(cls, *args, **kwargs):
        if cls is KernelWrapper:
            raise TypeError("Base KernelWrapper class cannot be "
                            "instantiated. Use one of the derived classes.")
        return object.__new__(cls, *args, **kwargs)

    # ----------
    # Validators

    @root_validator(pre=True)
    def set_attributes(cls, values):
        wk = values.get('wrapped_kernel', None)
        if wk is None:
            return values
        name = values.get('name', None)
        if name is None:
            values['name'] = f"{str(cls)}({wk.name})"
        donotreplace = ('wrapped_kernel', 'name')
        for k, v in values.items():
            if k not in donotreplace  and v is None:
                values[k] = getattr(wk, k)
        return values

    # ----------
    # Properties
    @property
    def contravariant_axes(self):
        return self.wrapped_kernel.contravariant_axes
    @property
    def dim_map(self):
        return self.wrapped_kernel.dim_map

class FactorizedKernel(KernelWrapper):
    """
    Kernel where the time component is factorized into a low-dim 'inner
    kernel'.
    Potentially more efficient than a FactorizedKernel, but not as general
    (unit-to-unit mappings cannot be expressed).

    .. note:: FactorizedKernel applies its projection matrices with an
    elementwise product + broadcasting, whereas CompressedKernel uses a
    dot product.

    Parameters
    ----------
    wrapped_kernel: Kernel
    outproj: array-like
        Short-fat array
        `inproj` and `outproj` are such that the full kernel is
        outproj @ inner_kernel @ inproj .
    """
    # Rename 'wrapped_kernel' to something more intuitive for the public API
    wrapped_kernel : Kernel=Field(..., alias='inner_kernel')
    outproj: Tensor[NPValue[shim.config.floatX]]
        # TODO: Allow any dtype for proj

    # ----------
    # Validators

    @root_validator
    def check_shape_consistency(cls, values):
        κ, oP = (values.get(x, None)
            for x in ('wrapped_kernel', 'outproj'))
        if None not in (κ, oP) and not shim.is_symbolic(κ, oP):
            if oP.shape[-1] != κ.shape[-1]:
                raise ValueError("Shapes for outproj and wrapped_kernel "
                                 "don't match. They are respectively "
                                 f"{oP.shape}, {κ.shape}.")

    # -------
    # Overridden methods

    def _eval(self, t, **kwargs):
        logger.warning("Calling `eval()` on a FactorizedKernel defeats the "
                       "optimization.")
        oP = self.outproj
        κ  = self.wrapped_kernel
        return oP * κ(t, **kwargs)

    def _convolve_single_t(self, hist, t, kernel_slice):
        oP = self.outproj
        κ = self.wrapped_kernel
        return TensorWrapper(
            oP * κ._convolve_single_t(hist, t, kernel_slice),
            TensorDims(contraction=self.contravariant_axes))

    def _convolve_op_batch(self, other, kernel_slice):
        oP = self.outproj
        κ = self.wrapped_kernel
        return TensorWrapper(
            oP * κ.convolve_op_batch(other, kernel_slice),
            TensorDims(contraction=self.contravariant_axes))

class CompressedKernel(KernelWrapper):
    """
    Kernel where the time component is factorized into a low-dim 'inner
    kernel'.
    Potentially more efficient than a FactorizedKernel, but not as general
    (unit-to-unit mappings cannot be expressed).

    .. note:: CompressedKernel applies its projection matrices with a dot
    product, whereas FactorizedKernel uses an elementwise product + broadcasting.

    Parameters
    ----------
    inner_kernel: Kernel
    inproj: array-like
        Tall-skinny array
    outproj: array-like
        Short-fat array
        `inproj` and `outproj` are such that the full kernel is
        outproj @ inner_kernel @ inproj .
    """
    # Rename 'wrapped_kernel' to something more intuitive for the public API
    wrapped_kernel : Kernel=Field(..., alias='inner_kernel')
    inproj : Tensor[NPValue[shim.config.floatX]]
    outproj: Tensor[NPValue[shim.config.floatX]]
        # TODO: Allow any dtype for proj

    # ----------
    # Validators

    @root_validator
    def check_shape_consistency(cls, values):
        κ, iP, oP = (values.get(x, None)
            for x in ('wrapped_kernel', 'inproj', 'outproj'))
        if None not in (κ, iP, oP) and not shim.is_symbolic(κ, iP, oP):
            if oP.shape[-1] != κ.shape[-2] or κ.shape[-1] != iP.shape[-2]:
                raise ValueError("Shapes for outproj, wrapped_kernel and inproj "
                                 "don't match. They are respectively "
                                 f"{oP.shape}, {κ.shape}, {iP.shape}.")

    # -------
    # Overridden methods

    def _eval(self, t, **kwargs):
        logger.warning("Calling `eval()` on a CompressedKernel defeats the "
                       "optimization.")
        iP = self.inproj
        oP = self.outproj
        κ = self.wrapped_kernel
        return oP.dot(κ(t, **kwargs).dot(iP.T))

    def _convolve_single_t(self, hist, t, kernel_slice):
        iP = self.inproj
        oP = self.outproj
        κ = self.wrapped_kernel
        if shim.isspsparse(hist._data):
            # scipy.sparse interface requires we use its dot method
            dot = lambda x, _iP: (x.T.dot(_iP)).T
        else:
            dot = lambda x, _iP: _iP.dot(x)
        Phist = hist._apply_op(dot, iP)
        return TensorWrapper(
            oP.dot(κ._convolve_single_t(Phist, t, kernel_slice)),
            TensorDims(contraction=self.contravariant_axes))

    def _convolve_op_batch(self, hist, kernel_slice):
        iP = self.inproj
        oP = self.outproj
        κ = self.inner_kernel
        if shim.isspsparse(hist._data):
            # scipy.sparse interface requires we use its dot method
            dot = lambda x, _iP: (x.T.dot(_iP)).T
        else:
            dot = lambda x, _iP: _iP.dot(x)
        Phist = hist._apply_op(dot, iP)
        return TensorWrapper(
            oP.dot(κ._convolve_op_batch(Phist, t, kernel_slice)),
            TensorDims(contraction=self.contravariant_axes))

class BlockKernel(KernelWrapper):
    """
    A kernel for which each element applies to a block of the data.
    """
    # Rename 'wrapped_kernel' to something more intuitive for the public API
    wrapped_kernel : Kernel=Field(..., alias='inner_kernel')
    block_slices   : Union[List[Slice], Tuple[List[Slice]]]

    @property
    def inner_kernel(self):
        return self.wrapped_kernel

    # -----------
    # Validators

    @validator('wrapped_kernel')
    def check_ndim(cls, v):
        if v.ndim != 2:
            raise ValueError("BlockKernel currently only supports 2D kernels.")
        return v

    @validator('block_slices')
    def check_slices(cls, slclst):
        """
        Check that list lengths are consistent with the kernel's shape
        Also wrap with 1-element tuple if slices are passed as bare list.
        """
        wrapped_kernel = values.get('wrapped_kernel', None)
        if isinstance(slices, list):
            slices = (slices,)
        if wrapped_kernel is not None:
            slclens = tuple(len(slclst) for slclst in slices)
            if slclens != wrapped_kernel.shape:
                raise ValueError("The number of block slices does not match "
                                 "the kernel's shape.\n"
                                 f"Block slices per dimension: {slclens}\n"
                                 f"Kernel shape: {inner_kernel.shape}")
        return slices

    # -------
    # Overridden methods

    def _eval(self, *args, **kwargs):
        return self.inner_kernel._eval(*args, **kwargs)

    def __getitem__(self, idx):
        raise NotImplementedError("No reason this function wouldn't be "
                                 "possible, but I don't have a use for it yet.")

    def _convolve_single_t(self, hist, t, kernel_slice):
        result = self.get_empty_convolve_result()
        for inslc, outslc, kernslc in self.block_iterator(
              kernel_dims=-1, include_time_slice=False):
            result[outslc] = self.wrapped_kernel[kernslc]._convolve_single_t(
                hist, t, kernel_slice)
        return TensorWrapper(result,
            TensorDims(contraction=self.contravariant_axes))

    # --------
    # Class-specific methods

    def block_iterator(self, kernel_dims=0, include_time_slice=True):
        """
        Parameters
        ----------
        kernel_dims: int (default: 0)
            Number of dimensions sliced kernel components should have.
            (These are obtained by replacing the first dimension(s) of the
            kernel index with length-1 slices.)
            This is intended for use with sparse arrays, which expect 2D
            arguments.
            Note that if the time slice is included, this will add yet another
            kernel dimension. So for a 2D kernel with `include_time_slice=True`,
            use `kernel_dims=1`.
            Use `-1` to indicate to keep all dimensions.
        include_time_slice: bool (default: True)
            If True, prepend the kernel slice with an ellipsis, to index time.
            Especially with a discretized kernel this is usually desired.

        Examples
        --------
        kernslc = (:, i1:i2)          # include_time_slice prepends ':'
        outslc = (..., i1:i2, i2:i3)  # all in/outslc prepend '...'
        """
        kern_idx_iter = itertools.product(*(range(l) for l in self.wrapped_kernel.shape))
            # Iterator for kernel components
        slices_iter = itertools.product(*self.block_slices)
            # Iterator for data slices corresponding to kernel components
        if self.ndim < kernel_dims:
            raise ValueError(f"Can't produce kernel output with {kernel_dims} "
                             f"dimensions: kernel has only {self.ndim} "
                             "dimensions.")
        if kernel_dims == -1:
            assert self.ndim == len(self.wrapped_kernel.shape)
            kernel_dims = self.ndim
        for kidx, dataslc in zip(kern_idx_iter, slices_iter):
            # To obtain an nD object, some of the dimensions have to be slices
            # rather than a plain index (so [i:i+1] instead of [i])
            kidx = tuple(slice(kidx[i],kidx[i]+1) for i in range(kernel_dims)) \
                    + kidx[kernel_dims:]
            if include_time_slice:
                kidx = (slice(None),) + kidx
            # Compute the index for the data structure the result would be
            # inserted into.
            outslc = (Ellipsis,) + tuple(dataslc[indim]
                                         for indim in self.dim_map)
            inslc  = (Ellipsis,) + dataslc

            yield inslc, outslc, kidx
