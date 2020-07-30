# -*- coding: utf-8 -*-
"""
Created Sun 7 Dec 2019
Based on fsgif_model.py (May 24 2017)

author: Alexandre René
"""

import numpy as np
import scipy as sp
from scipy.optimize import root
from collections import namedtuple, OrderedDict, Iterable
import logging
import copy
import operator

import theano_shim as shim
import mackelab_toolbox.typing
mackelab_toolbox.typing.freeze_types()

from typing import Type
from pydantic import validator, root_validator
from mackelab_toolbox.typing import NPType, FloatX

import mackelab_toolbox.utils as utils
import sinn
import sinn.config as config
from sinn.histories import TimeAxis, Series, PopulationHistory
import sinn.kernels as kernels
import sinn.models as models
import sinn.popterm
initializer = models.initializer

from typing import ClassVar

logger = logging.getLogger("fsgif_model")


homo = False  # HACK

# HACK
shim.cf.inf = 1e12
    # Actual infinity doesn't play nice in kernels, because inf*0 is undefined

# Debug flag(s)
debugprint = False
    # If true, all intermediate values are printed in the symbolic graph

# class Kernel_ε(kernels.ExpKernel):
#     """
#     This class just renames the generic ExpKernel to match the parameter
#     names for this kernel, and ensures the kernel is normalized.
#     """
#     τ_s         : Tensor[FloatX, 2]
#     Δ           : Tensor[FloatX, 2]
#     height      : Tensor[FloatX, 2]=None  # Indicate that these parameters are computed
#     decay_const : Tensor[FloatX, 2]=None  # in __post_init__ and are thus not required
#     t_offset    : Tensor[FloatX, 2]=None  # when constructing kernel
#
#     _computed_params = ('height', 'decay_const', 't_offset')
#
#     @root_validator(cls, values, pre=True)
#     def set_values(values):
#         τ_s, Δ = (values.get(x, None) for x in ('τ_s', 'Δ'))
#         if any(values.get(x, None) is not None
#                for x in cls._computed_params):
#            raise ValueError(f"Parameters {cls._computed_params} are computed "
#                             "automatically and should not be provided.")
#         if None not in (τ_s, Δ):
#             values['height']      = 1/τ_s
#             values['decay_const'] = τ_s
#             values['t_offset']    = Δ
#         return values


# The θ kernel is separated in two: θ1 is the constant equal to ∞ over (0, t_ref)
# θ2 is the exponentially decaying adaptation kernel
# class Kernel_θ1(kernels.BoxKernel):
#     Npops  : int
#     PopTerm: Type
#     # ---- Make computed params of base Kernel class optional
#     height : Tensor[FloatX]=None
#     start  : Tensor[FloatX]=0
#
#     @validator('start'):
#     def check_start(cls, start):
#         if start != 0:
#             raise ValueError("Do not set `start`. It is fixed to 0.")
#
#     @root_validator(pre=True)
#     def set_height_shape(cls, values):
#         Npops = values.get('N', None)
#         if Npops is not None:
#             values['height'] = PopTerm((shim.cf.inf,)*Npops)
#             values['shape']  = (Npops,Npops)



    # def get_kernel_params(model_params):
    #     if homo:
    #         Npops = len(model_params.N.get_value())
    #         height = (shim.cf.inf,)*Npops
    #         stop = model_params.t_ref
    #     else:
    #         height = (shim.cf.inf,)*model_params.N.get_value().sum()
    #         stop = model_params.t_ref
    #         # if isinstance(model_params.t_ref, sinn.popterm.PopTerm):
    #         #     # TODO: This is only required because some operations
    #         #     #       aren't yet supported by PopTerm, so we expand
    #         #     #       manually. Once that is fixed, we should remove this.
    #         #     stop = stop.expand_blocks(['Macro', 'Micro'])
    #     return Kernel_θ1.Parameters(
    #         height = height,
    #         start  = 0,
    #         stop   = stop
    #     )

    # def __init__(self, name, params=None, shape=None, **kwargs):
    #     kern_params = self.get_kernel_params(params)
    #     memory_time = shim.asarray(shim.get_test_value(kern_params.stop)
    #                                - kern_params.start).max()
    #         # FIXME: At present, if we don't set memory_time now, tn is not set
    #         #        properly
    #     super().__init__(name, params, shape,
    #                      t0 = 0,
    #                      memory_time = memory_time,  # FIXME: memory_time should be optional
    #                      **kwargs)
    #
    # def _eval_f(self, t, from_idx=slice(None,None)):
    #     if shim.isscalar(t):
    #         return self.params.height
    #     else:
    #         # t has already been shaped to align with the function output in Kernel.eval
    #         return shim.ones(t.shape, dtype=shim.config.floatX) * self.params.height

# class Kernel_θ2(models.ModelKernelMixin, kernels.ExpKernel):
#     @staticmethod
#     def get_kernel_params(model_params):
#         # if model_params.t_ref.ndim == 1:
#         #     t_offset = model_params.t_ref[np.newaxis,:]
#         # else:
#         #     t_offset = model_params.t_ref
#         t_offset = model_params.t_ref
#         return kernels.ExpKernel.Parameters(
#             height      = model_params.J_θ / model_params.τ_θ,
#             decay_const = model_params.τ_θ,
#             t_offset    = t_offset
#         )

# class ExpandedExpKernel(kernel.ExpKernel):
#     # UGLY HACK: Copied function from ExpKernel and added 'expand'
#     def _eval(self, t, from_idx=slice(None,None)):
#         if homo:
#             return super()._eval(t, from_idx)
#         else:
#             expand = lambda x: x.expand_blocks(['Macro', 'Micro']) if isinstance(x, sinn.popterm.PopTerm) else x
#             return shim.switch(shim.lt(t, expand(self.t_offset[...,from_idx])),
#                             0,
#                             expand(self.height[...,from_idx]
#                                 * shim.exp(-(t-self.t_offset[...,from_idx])
#                                         / self.decay_const[...,from_idx])) )


class GIF(models.Model):
    # ===================================
    # Class parameters
    class Config:
        pass
        # rng = shim.typing.RandomStream

    time :TimeAxis
    rng  :typing.AnyRNG

    # ====================================
    # Model parameters
    class Parameters(BaseModel):
        # NOTE: int32 * float32 = float64,  but  int16 * float32 = float32
        N      : Array['int16', 1]               # Maximum value: 2^16 == 65536
            # Must not be symbolic
        R      : Tensor[FloatX]               # membrane resistance (
        u_rest : Tensor[FloatX, 1]
        p      : Optional[Tensor[FloatX]]     # Connection probability between populations
        w      : Tensor[FloatX, 2]            # matrix of *population* connectivity strengths
        Γ      : Optional[Tensor['int8', 2]]  # binary connectivity between *neurons*
            # NOTE: `Γ` is typically obtained by calling `make_connectivity` with `N` and `p`.
        τ_m    : Tensor[FloatX, 1]            # membrane time constant (s)
        t_ref  : Tensor[FloatX, 1]            # absolute refractory period (s)
        u_th   : Tensor[FloatX, 1]            # non-adapting threshold (mV)
        u_r    : Tensor[FloatX, 1]            # reset potential (mV)
        c      : Tensor[FloatX, 1]            # escape rate at threshold (Hz)
        Δu     : Tensor[FloatX, 1]            # noise level (mV)
        Δ      : Tensor[FloatX, 2]            # transmission delay (s) (kernel ε)
        τ_s    : Tensor[FloatX, 2]            # synaptic time constant (mV) (kernel ε)
        # Adaptation parameters (θ-kernel dependent)
        J_θ    : Tensor[FloatX, 1]                # Integral of adaptation (mV s)
        τ_θ    : Tensor[FloatX, 1]

        @validator('Γ')
        def generate_connectivity_matrix(cls, Γ, values):
            p, N = (values.get(x, None) for x in ('p', 'N'))
            if (p is None and Γ is None) or (p is not None and Γ is not None):
                raise ValueError("Exactly one of `p` or `Γ` must be provided.")
            if Γ is None and N is not None:
                Γrows = []
                for Nα, prow in zip(N, p):
                    Γrows.append( np.concatenate(
                        [ np.random.binomial(1, pαβ, size=(Nα, Nβ))
                          for Nβ, pαβ in zip(N, prow) ],
                        axis = 1) )
                Γ = np.concatenate( Γrows, axis=0 )
            return Γ

    params :Parameters

    # ====================================
    # Histories

    class State:
        """
        Annotation-only class. Names must match histories; type is `Any`.
        In the future we may allow the type to be more specific.
        """
        # TODO? Have a placeholder so we can do Series.return_type ?
        s     :Any
        u     :Any
        t_hat :Any

    # Public
    s         : Spiketrain = None
    I_ext     : History
    # Private
    # ("Private" is just a convention: these histories can still be accessed,
    # and the auto-created history can be replaced by passing a keyword arg)
    RI_syn    : Series = None
    λ         : Series = AutoHist(name = 'λ', template='RI_syn')
    varθ      : Series = AutoHist(name = 'ϑ', template='RI_syn')
    u         : Series = AutoHist(name = 'u', template='RI_syn')
    t_hat     : Series = AutoHist(name = 't_hat', template='RI_syn')
        # Surrogate variable: time since last spike

    # ===================================
    # Kernels
    ε : Kernel = None
    θ : Kernel = None

    # ===================================
    # Internal variables

    min_memory_time : 'time.unit' = 0.  # When fitting time constants, helps
                                        # to ensure memory_time is long enough
    memory_time : 'time.unit' = None
    K           : 'time.AxisIndex' = None

    # ======================================
    # Initializers

    # Initializers - kernels
    @initializer('ε')
    def create_ε_kernel(cls, ε, N, τ_s, Δ, Γ):
        Npops = len(N)  # N cannot be symbolic
        Nneurons = sum(N)
        inner_shape = np.broadcast_to(shim.eval(τ_s), shim.eval(Δ))
        assert all(s == inner_shape[0] for s in inner_shape[1:])  # square shape
        if inner_shape[0] == Nneurons:
            # We have a heterogeneous kernel:
            assert homo is False  # TODO: remove `homo` flag entirely
            ε = ExpKernel(
                name='ε_inner', height=1/τ_s, decay_const=τ_s, t_offset=Δ,
                shape=(Nneurons,Nneurons)
                )
        else:
            # We have a homogeneous kernel:
            assert inner_shape[0] == Npops
            assert homo is True  # TODO: remove `homo` flag entirely
            ε = FactorizedKernel(
                name         = 'ε',
                outproj      = Γ,
                inner_kernel = BlockKernel(
                    inner_kernel = ExpKernel(
                        name='ε_inner', height=1/τ_s, decay_const=τ_s, t_offset=Δ,
                        shape=(Npops,Npops)
                        )
                    )
                )
        return ε

    @initializer('θ')
    def create_θ_kernel(cls, θ, N, J_θ, τ_θ, s):
        """
        The θ kernel is separated in two: θ1 is the constant equal to ∞ over (0, t_ref)
        θ2 is the exponentially decaying adaptation kernel
        """
        # FIXME: s
        Npops = len(N)
        Nneurons = sum(N)
        shape = np.broadcast_to(shim.eval(τ_θ), shim.eval(J_θ))
        assert len(shape) == 1  # Adaptation kernel is always 1D
        if shape[1] == Nneurons:
            assert homo is False  # TODO: remove `homo` flag entirely
        else:
            assert shape[0] == Npops:
            assert homo is True  # TODO: remove `homo` flag entirely
        θ1 = BoxKernel(name='θ1', height=s.PopTerm((shim.cf.inf,)*Npops),
                       shape=shape)
        θ2 = ExpKernel(name='θ2', height=J_θ/τ_θ, decay_const=τ_θ,
                       t_offset=t_ref, shape=shape)
        return θ1 + θ2

    #Initializers - histories

    @initializer('s')
    def create_s(cls, s, time, N):
        return Spiketrain(name='s', time=time, pop_sizes=N)

    @initializer('RI_syn')
    def create_RI(cls, RI, time, N):
        return Series(name='RI_syn', time=time, shape=(N.sum(),)
                      dytpe=shim.config.floatX)

    # Validators

    @root_validator
    def check_history_shapes(cls, values):
        s, I_ext = values.get('s', None)
        if s is not None and I_ext is not None:
            super().check_same_shape(s, I_ext)
        return values

    @root_validator
    def consistent_shapes(cls, values):
        # FIXME: This needs to allow some parameters to be micro, some meso
        # Raises ValueError if shapes don't match
        # np.broadcast(values.get(x, 1) for x in
        #     ('N', 'R', 'u_rest', 'p', 'w', 'τ_m', 't_ref', 'u_th', 'u_r',
        #      'c', 'Δu', 'Δ', 'τ_s', 'J_θ', 'τ_θ')
        return values

    @root_validator
    def convert_to_params_popterms(cls, values):
        s = values.get('s', None)
        params = values.get('params', None)
        if s is not None and params is not None:
            for k, v in params.items():
                if k != 'N':
                    params[k] = s.PopTerm(v)
        return values

    # def __init__(self, ):
    #     """
    #     Parameters
    #     ----------
    #     Γ: ndarray | tensor
    #         If not given, computed from `p`.
    #     set_weights: bool, ndarray
    #         (Optional) Set to True to indicate that network connectivity should be set using the
    #         `w` and `Γ` parameters. If the spike history is already filled, set to False to
    #         avoid overwriting the connectivity. If an ndarray, that array will be used directly
    #         to set connectivity, ignoring model parameters. Default is True.
    #     """

        # self.s = spike_history
        # self.I_ext = input_history
        # self.rndstream = random_stream
        # # This runs consistency tests on the parameters
        # # models.Model.same_shape(self.s, self.I)
        # models.Model.same_dt(self.s, self.I_ext)
        # models.Model.output_rng(self.s, self.rndstream)
        #
        # super().__init__(params,
        #                  t0=self.s.t0, tn=self.s.tn, dt=self.s.dt,
        #                  public_histories=(spike_history, input_history),
        #                  reference_history=self.s)
        # NOTE: Do not use `params` beyond here. Always use self.params.
        # N = self.params.N.get_value()
        # assert(N.ndim == 1)
        # N     = self.N
        # Npops = len(N)
        # self.Npops = Npops
        # τ_s   = self.τ_s
        # Δ     = self.Δ
        # J_θ   = self.J_θ
        # τ_θ   = self.τ_θ
        # t_ref = self.t_ref




        # self.params = self.params._replace(
        #     **{name: self.s.PopTerm(getattr(self.params, name))
        #     for name in params._fields
        #     if name != 'N'})

        # if values.name in ['t_ref', 'J_θ', 'τ_θ']:
        # Set the connection weights
        # if isinstance(set_weights, np.ndarray):
        #     self.s.set_connectivity(set_weights)
        # elif set_weights:
        #     # TODO: If parameters were less hacky, w would already be properly
        #     #       cast as an array
        #     w = (self.s.PopTerm(self.w) * self.Γ).expand.values
        #         # w includes both w and Γ from Eq. 20
        #     self.s.set_connectivity(w)

        # # Model variables
        # self.RI_syn = Series(name  = 'RI_syn',
        #                      time  = self.s.time,
        #                      shape = (N.sum(),),
        #                      dtype = shim.config.floatX)
        # self.λ      = Series(name='λ', template=self.RI_syn)
        # self.varθ   = Series(name='ϑ', template=self.RI_syn)
        # self.u      = Series(name='u', template=self.RI_syn)
        # # Surrogate variables
        # self.t_hat  = Series('t_hat', template=self.RI_syn)
        #     # time since last spike

        # self.statehists = [ getattr(self, varname) for varname in self.State._fields ]
        # Kernels
        # shape2d = (self.Npops, self.Npops)
        # self.ε = Kernel_ε('ε', self.params, shape=shape2d)
        # # if values.name in ['t_ref', 'J_θ', 'τ_θ']:
        # if homo:
        #     self.θ1 = Kernel_θ1('θ1', self.params, shape=(self.Npops,))
        #     self.θ2 = Kernel_θ2('θ2', self.params, shape=(self.Npops,))
        # else:
        #     self.θ1 = Kernel_θ1('θ1', self.params, shape=(sum(N),))
        #     self.θ2 = Kernel_θ2('θ2', self.params, shape=(sum(N),))

        # self.add_history(self.s)
        # self.add_history(self.I_ext)
        # self.add_history(self.λ)
        # self.add_history(self.varθ)
        # self.add_history(self.u)
        # self.add_history(self.RI_syn)
        # self.add_history(self.t_hat)
        #
        # self.s.set_update_function(self.s_fn, inputs=[self.λ])
        # self.λ.set_update_function(self.λ_fn, inputs=[self.u, self.varθ])
        # self.varθ.set_update_function(self.varθ_fn, inputs=[self.s])
        # self.u.set_update_function(self.u_fn, inputs=[self.u, self.t_hat, self.I_ext, self.RI_syn])
        # self.RI_syn.set_update_function(self.RI_syn_fn, inputs=[self.s])
        # self.t_hat.set_update_function(self.t_hat_fn, inputs=[self.t_hat, self.s])

        # Pad to allow convolution
        # FIXME Check with mesoGIF to see if memory_time could be better / more consistently treated
        # if memory_time is None:
        #     memory_time = 0
        # self.memory_time = shim.cast(max(memory_time,
        #                                  max( kernel.memory_time
        #                                       for kernel in [self.ε, self.θ1, self.θ2] ) ),
        #                              dtype=shim.config.floatX)
        # self.K = np.rint( self.memory_time / self.dt ).astype(int)
        # self.s.pad(self.memory_time)
        # self.s.memory_time = self.memory_time
        # Pad because these are ODEs (need initial condition)
        #self.u.pad(1)
        #self.t_hat.pad(1)

        # Expand the parameters to treat them as neural parameters
        # Original population parameters are kept as a copy
        # self.pop_params = copy.copy(self.params)
        # ExpandedParams = namedtuple('ExpandedParams', ['u_rest', 't_ref', 'u_r'])
        # self.expanded_params = ExpandedParams(
        #     u_rest = self.expand_param(self.params.u_rest, self.params.N),
        #     t_ref = self.expand_param(self.params.t_ref, self.params.N),
        #     u_r = self.expand_param(self.params.u_r, self.params.N)
        # )

        #if self.s._num_tidx.get_value() < self.s.t0idx:

        # TODO: Do something with the state variables if we don't initialize them
        #       (e.g. if we try to calculate t_hat, we will try to get t_hat[t-1],
        #       which is unset)

    def initialize(self, initializer='silent'):
        time = self.time
        # Set memory time
        kernels = (self.ε, self.θ)
        self.memory_time = shim.cast(max(self.min_memory_time,
                                         max(κ.memory_time for κ in kernels)),
                                     dtype=shim.config.floatX) * time.unit
        assert not shim.is_symbolic(self.memory_time)
        self.K = self.time.index_interval(self.memory_time)
        # Set padding
        self.s.pad(K)
        # Initalize state vars
        logger.info("Initializing model state variables...")
        self.initialize_state(initializer)
        logger.info("Done.")

    def initialize_state(self, initializer='silent'):
        """
        Parameters
        ----------
        initializer: 'silent' (default) | 'stationary'
        """
        # FIXME
        if not homo and initializer == 'stationary':
            raise NotImplementedError("Stationary initialization doesn't work with heterogeneous "
                                      "populations yet. Reason: "
                                      "`τmT = self.params.τ_m.flatten()[:, np.newaxis]` line")

        if initializer == 'stationary':
            θ_dis, θtilde_dis = mesoGIF.discretize_θkernel(
                [self.θ1, self.θ2], self._refhist, self.params)
            init_A = meso.get_stationary_activity(
                self, self.K, θ_dis, θtilde_dis)
            init_state = self.get_stationary_state(init_A)

        elif initializer == 'silent':
            init_A = np.zeros((len(self.s.pop_slices),))
            init_state = self.get_silent_latent_state()
        else:
            raise ValueError("Initializer string must be one of 'stationary', 'silent'")

        for varname in self.State.__fields__:
            hist = getattr(self, varname)
            if shim.eval(hist._num_tidx) < hist.t0idx:
                initval = getattr(init_state, varname)
                hist.pad(1)
                idx = hist.t0idx - 1; assert(idx >= 0)
                hist[idx] = initval

        # TODO: Combine the following into the loop above
        nbins = self.s.t0idx
        if self.s._num_tidx.get_value() < nbins:
            self.s[:nbins] = init_state.s
        #data = self.s._data
        #data[:nbins,:] = init_state.s
        #self.s._data.set_value(data, borrow=True)

    def get_silent_latent_state(self):
        # TODO: include spikes in model state, so we don't need this custom 'Stateplus'
        Stateplus = namedtuple('Stateplus', self.State.__fields__ + ('s',))
        state = Stateplus(
            u = self.params.u_rest.expand.values,
            t_hat = shim.ones(self.t_hat.shape) * self.memory_time,
            s = np.zeros((self.s.t0idx, self.s.shape[0]))
            )
        return state

    def get_stationary_state(self, Astar):
        # TODO: include spikes in model state, so we don't need this custom 'Stateplus'
        Stateplus = namedtuple('Stateplus', self.State.__fields__ + ('s',))
        # Initialize the spikes
        # We treat that as a Bernouilli process, with firing rate
        # given by Astar; this means ISI statistics will be off as
        # we ignore refractory effects, but the overall rate will be
        # correct.
        p = self.s.PopTerm(Astar).expand.values * self.dt
        nbins = self.s.t0idx
        nneurons = self.s.shape[0]
        s = np.random.binomial(1, p, (nbins, nneurons))
        # argmax returns first occurrence; by flipping s, we get the
        # index (from the end) of the last spike, i.e. number of bins - 1
        t_hat = (s[::-1].argmax(axis=0) + 1) * self.dt
        # u is initialized by integrating the ODE from the last spike
        # See documentation for details (TODO: not yet in docs)
        #τm_exp = self.expand_param(self.params.τ_m, self.params.N)
        τmT = self.params.τ_m.flatten()[:, np.newaxis]
        η1 = τmT * self.params.p * self.params.N * self.params.w
            # As in mesoGIF.get_η_csts
        u = np.where(t_hat <= self.params.t_ref.expand.values,
                     self.params.u_r.expand.values,
                     ((1 - np.exp(-t_hat/self.params.τ_m)) * self.s.PopTerm( (self.params.u_rest + η1.dot(Astar)) )
                       + self.params.u_r * np.exp(-t_hat/self.params.τ_m)).expand.values
                     )
        state = Stateplus(
            u = u,
            t_hat = t_hat,
            s = s
        )
        return state

    # @staticmethod
    # def make_connectivity(N, p):
    #     """
    #     Construct a binary neuron connectivity matrix, for use as this class'
    #     Γ parameter.
    #
    #     Parameters
    #     ----------
    #     N: array of ints
    #         Number of neurons in each population
    #     p: 2d array of floats between 0 and 1
    #         Connection probability. If connection probabilities are symmetric,
    #         this array should also be symmetric.
    #
    #     Returns
    #     -------
    #     2d binary matrix
    #     """
    #     # TODO: Use array to pre-allocate memory
    #     # TODO: Does it make sense to return a sparse matrix ? For low connectivity
    #     #       yes, but with the assumed all-to-all connectivity, we can't really
    #     #       have p < 0.2.
    #     Γrows = []
    #     for Nα, prow in zip(N, p):
    #         Γrows.append( np.concatenate([ np.random.binomial(1, pαβ, size=(Nα, Nβ))
    #                                     for Nβ, pαβ in zip(N, prow) ],
    #                                     axis = 1) )
    #
    #     Γ = np.concatenate( Γrows, axis=0 )
    #     return Γ

    # @staticmethod
    # def expand_param(param, N):
    #     """
    #     Expand a population parameter such that it can be multiplied directly
    #     with the spiketrain.
    #
    #     Parameters
    #     ----------
    #     param: ndarray
    #         Parameter to expand
    #
    #     N: tuple or ndarray
    #         Number of neurons in each population
    #     """
    #
    #     Npops = len(N)
    #     if param.ndim == 1:
    #         return shim.concatenate( [ param[i]*np.ones((N[i],))
    #                                    for i in range(Npops) ] )
    #
    #     elif param.ndim == 2:
    #         return shim.concatenate(
    #             [ shim.concatenate( [ param[i, j]* np.ones((N[i], N[j]))
    #                                   for j in range(Npops) ],
    #                                 axis = 1 )
    #               for i in range(Npops) ],
    #             axis = 0 )
    #     else:
    #         raise ValueError("Parameter {} has {} dimensions; can only expand "
    #                          "dimensions of 1d and 2d parameters."
    #                          .format(param.name, param.ndim))

    def loglikelihood(self, start, batch_size):
        # >>>>>>>>>>>>>> WARNING: Untested, incomplete <<<<<<<<<<<<<

        #######################
        # Some hacks to get around current limitations

        self.remove_other_histories()

        # End hacks
        #####################

        startidx = self.get_t_idx(start)
        batch_size = self.index_interval(batch_size)
        stopidx = startidx + batch_size
        N = self.params.N

        def logLstep(tidx, cum_logL):
            p = sinn.clip_probabilities(self.λ[tidx]*self.s.dt)
            s = shim.cast(self.s[tidx+self.s.t0idx], self.s.dtype)

            # L = s*n - (1-s)*(1-p)
            cum_logL += ( s*p - (1-p) + s*(1-p) ).sum()

            return [cum_logL], shim.get_updates()

        if shim.is_theano_object([self.s, self.params]):

            logger.info("Producing the likelihood graph.")

            if batch_size == 1:
                # No need for scan
                logL, upds = logLstep(start, 0)

            else:
                # FIXME np.float64 -> shim.floatX or sinn.floatX
                logL, upds = shim.gettheano().scan(logLstep,
                                                sequences = shim.getT().arange(startidx, stopidx),
                                                outputs_info = np.asarray(0, dtype=shim.config.floatX))
                self.apply_updates(upds)
                    # Applying updates is essential to remove the temporary iteration variable
                    # scan introduces from the shim updates dictionary

            logger.info("Likelihood graph complete")

            return logL[-1], upds
        else:
            # TODO: Remove this branch once shim.scan is implemented
            logL = 0
            for t in np.arange(startidx, stopidx):
                logL = logLstep(t, logL)[0][0]
            upds = shim.get_updates()

            return logL, upds

    # FIXME: Before replacing with Model's `advance`, need to remove HACKs
    def advance(self, stop):

        if stop == 'end':
            stopidx = self.tnidx
        else:
            stopidx = self.get_t_idx(stop)

        # Make sure we don't go beyond given data
        for h in self.history_set:
            # HACK: Should exclude kernels
            if h.name in ['θ_dis', 'θtilde_dis']:
                continue
            if h.locked:
                tn = h.get_time(h._num_tidx.get_value())
                if tn < self._refhist.get_time(stopidx):
                    logger.warning("Locked history '{}' is only provided "
                                   "up to t={}. Output will be truncated."
                                   .format(h.name, tn))
                    stopidx = self.nbar.get_t_idx(tn)

        if not shim.config.use_theano:
            self._refhist._compute_up_to(stopidx - self.t0idx + self._refhist.t0idx)
            for hist in self.statehists:
                hist._compute_up_to(stopidx - self.t0idx + hist.t0idx)

        else:
            curtidx = min( hist._num_tidx.get_value() - hist.t0idx + self.t0idx
                           for hist in self.statehists )
            assert(curtidx >= -1)

            if curtidx+1 < stopidx:
                self._advance(stopidx)
                for hist in self.statehists:
                    hist.theano_reset()

    def _advance(stopidx):
        raise NotImplementedError

    def f(self, u):
        """Link function. Maps difference between membrane potential & threshold
        to firing rate."""
        return self.params.c * shim.exp(u/self.params.Δu.flatten())


    @update_function('s', inputs=['λ', 'rng'])
    def s_fn(self, t):
        """Spike generation"""
        t_λ = self.s.get_t_for(t, self.λ)
        return ( self.rng.binomial( size = self.s.shape,
                                    n = 1,
                                    p = sinn.clip_probabilities(self.λ[t_λ]*self.s.dt) )
                 .nonzero()[0].astype(self.s.idx_dtype) )
            # nonzero returns a tuple, with oner element per axis

    @update_function('RI_syn', inputs=['s'])
    def RI_syn_fn(self, t):
        """Incoming synaptic current times membrane resistance. Eq. (20)."""
        t_s = self.RI_syn.get_t_for(t, self.s)
        return ( self.params.τ_m * self.s.convolve(self.ε, t_s) )
            # s includes the connection weights w, and convolution also includes
            # the sums over j and β in Eq. 20.
            # Need to fix spiketrain convolution before we can use the exponential
            # optimization. (see fixme comments in histories.spiketrain._convolve_single_t
            # and kernels.ExpKernel._convolve_single_t). Weights will then no longer
            # be included in the convolution

    @update_function('λ', inputs=['varθ'])
    def λ_fn(self, t):
        """Hazard rate. Eq. (23)."""
        # TODO: Use self.f here (requires overloading of ops to remove pop_rmul & co.)
        t_u = self.λ.get_t_for(t, self.u)
        t_varθ = self.λ.get_t_for(t, self.varθ)
        return (self.params.c * shim.exp(  (self.u[t_u] - self.varθ[t_varθ]) / self.params.Δu ) ).expand_axis(-1, 'Micro').values

    @update_function('varθ', inputs=['s'])
    def varθ_fn(self, t):
        """Dynamic threshold. Eq. (22)."""
        t_s = self.RI_syn.get_t_for(t, self.s)
        return (self.params.u_th + self.s.convolve(self.θ1, t_s)
                + self.s.convolve(self.θ2, t_s)).expand_axis(-1, 'Micro').values
            # Need to fix spiketrain convolution before we can use the exponential
            # optimization. (see fixme comments in histories.spiketrain._convolve_single_t
            # and kernels.ExpKernel._convolve_single_t)

    @update_function('u', inputs=['u', 't_hat', 'I_ext', 'RI_syn'])
    def u_fn(self, t):
        """Membrane potential. Eq. (21)."""
        if not shim.isscalar(t):
            tidx_u_m1 = self.u.array_to_slice(t - self.u.dt)
            t_that = self.t_hat.array_to_slice(self.u.get_t_for(t, self.t_hat))
            t_Iext = self.I_ext.array_to_slice(self.u.get_t_for(t, self.I_ext))
            t_RIsyn = self.RI_syn.array_to_slice(self.u.get_t_for(t, self.RI_syn))
        else:
            tidx_u_m1 = self.u.get_t_idx(t) - 1
            t_that = self.u.get_t_for(t, self.t_hat)
            t_Iext = self.u.get_t_for(t, self.I_ext)
            t_RIsyn = self.u.get_t_for(t, self.RI_syn)
            # Take care using this on another history than u – it may be padded
        # Euler approximation for the integral of the differential equation
        # 'm1' stands for 'minus 1', so it's the previous time bin
        red_factor = shim.exp(-self.u.dt/self.params.τ_m)
        return shim.switch( shim.ge(self.t_hat[t_that], self.params.t_ref.expand.values),
                            self.u[tidx_u_m1] * red_factor
                            + ( (self.params.u_rest + self.params.R * self.I_ext[t_Iext])
                                + self.RI_syn[t_RIsyn] )
                              * (1 - red_factor),
                            self.params.u_r.expand.values )

    @update_function('t_hat', ['t_hat', 's'])
    def t_hat_fn(self, t):
        """Update time since last spike"""
        if shim.isscalar(t):
            s_tidx_m1 = self.t_hat.get_tidx_for(t, self.s) - 1
            t_tidx_m1 = self.t_hat.get_tidx(t) - 1
            cond_tslice = 0
        else:
            s_t = self.t_hat.get_t_for(t, self.s)
            s_tidx_m1 = self.s.array_to_slice(s_t, lag=-1)
            t_tidx_m1 = self.t_hat.array_to_slice(t, lag=-1)
            cond_tslice = slice(None)
        # If the last bin was a spike, set the time to dt (time bin length)
        # Otherwise, add dt to the time
        return shim.switch( (self.s[s_tidx_m1] == 0)[cond_tslice],
                            self.t_hat[t_tidx_m1] + self.t_hat.dt,
                            self.t_hat.dt )

class mesoGIF:

    @staticmethod
    def discretize_θkernel(θ, reference_hist, params):
        """
        Parameters
        ----------
        θ: kernel, or iterable of kernels
            The kernel to discretize. If an iterable, its elements are summed.
        reference_hist: History
            The kernel will be discretized to be compatible with this history.
            (E.g. it will use the same time step.)
        params: namedtuple-like
            Must have the following attributes: Δu, N
        """
        ## Create discretized kernels
        # TODO: Once kernels can be combined, can just
        #       use A's discretize_kernel method
        # FIXME: Check with p.52, InitPopulations – pretty sure the indexing isn't quite right
        if not isinstance(θ, Iterable):
            θ = [θ]
        memory_time = max(kernel.memory_time for kernel in θ)
        dt = reference_hist.dt

        shape = (reference_hist.npops,) if homo else reference_hist.shape
        θ_dis = Series(reference_hist, 'θ_dis',
                       time_array = np.arange(dt, memory_time+dt, dt),
                       #t0 = dt,
                       #tn = memory_time+reference_hist.dt,
                       shape = shape,
                       iterative = False,
                       dtype=shim.config.floatX)
            # Starts at dt because memory buffer does not include current time

        # TODO: Use a history's `discretize_kernel` (prob. refhist)
        # TODO: Kernels should batch-update by default, so in theory we could
        # use update functions / arithmetic expressions instead of `set`ting
        # directly

        # θ_dis.set_update_function(
        #     lambda t: np.sum( (kernel.eval(t) for kernel in θ),
        #                       dtype=shim.config.floatX ).astype(shim.config.floatX) )
        θ_data = shim.sum([kernel.eval(θ_dis._tarr) for kernel in θ], axis=0,
                          dtype=shim.config.floatX).astype(shim.config.floatX)
            # If t is float64, specifying dtype inside sum() isn't always enough, so we astype as well
        # HACK Currently we only support updating by one history timestep
        #      at a time (Theano), so for kernels (which are fully computed
        #      at any time step), we index the underlying data tensor
        θ_dis.set(θ_data)
        # HACK θ_dis updates should not be part of the loglikelihood's computational graph
        #      but 'already there'
        if shim.is_theano_object(θ_dis._data):
            if θ_dis._num_data in shim.config.symbolic_updates:
                del shim.config.symbolic_updates[θ_dis._num_data]
            if θ_dis._num_tidx in shim.config.symbolic_updates:
                del shim.config.symbolic_updates[θ_dis._num_tidx]

        # TODO: Use operations
        θtilde_dis = Series(θ_dis, 'θtilde_dis', iterative=False)
        # HACK Proper way to ensure this would be to specify no. of bins (instead of tn) to history constructor
        # if len(θ_dis) != len(θtilde_dis):
        #     θtilde_dis._tarr = copy.copy(θ_dis._tarr)
        #     θtilde_dis._num_data.set_value(shim.zeros_like(θ_dis._num_data.get_value()))
        #     θtilde_dis._data = θtilde_dis._num_data
        #     θtilde_dis.tn = θtilde_dis._tarr[-1]
        #     θtilde_dis._unpadded_length = len(θtilde_dis._tarr)
        # HACK θ_dis._data should be θ_dis; then this can be made a lambda function
        # def θtilde_upd_fn(t):
        #     tidx = θ_dis.get_t_idx(t)
        #     return params.Δu * (1 - shim.exp(-θ_dis._data[tidx]/params.Δu) ) / params.N
        # θtilde_dis.set_update_function(θtilde_upd_fn)
        # self.θtilde_dis.set_update_function(
        #     lambda t: self.params.Δu * (1 - shim.exp(-self.θ_dis._data[t]/self.params.Δu) ) / self.params.N )
        θtilde_data = params.Δu * (1 - shim.exp(-θ_data/params.Δu) ) / params.N
            # Division by N follows definition in pseudocode; text puts division
            # in the expression for varθ.

        # HACK Currently we only support updating by one histories timestep
        #      at a time (Theano), so for kernels (which are fully computed
        #      at any time step), we index the underlying data tensor
        θtilde_dis.set(θtilde_data)
        # HACK θ_dis updates should not be part of the loglikelihood's computational graph
        #      but 'already there'
        if shim.is_theano_object(θtilde_dis._data):
            if θtilde_dis._num_data in shim.config.symbolic_updates:
                del shim.config.symbolic_updates[θtilde_dis._num_data]
            if θtilde_dis._num_tidx in shim.config.symbolic_updates:
                del shim.config.symbolic_updates[θtilde_dis._num_tidx]

        return θ_dis, θtilde_dis

    @staticmethod
    def get_stationary_activity(params, K, θ, θtilde):
        """
        Determine the stationary activity for these parameters by solving a
        self-consistency equation. For details see the notebook
        'docs/Initial_condition.ipynb'

        We make this a static method to allow external calls. In particular,
        this allows us to use this function in the initialization of GIF.

        TODO: Use get_η_csts rather than accessing parameters directly.

        Parameters  (not up to date)
        ----------
        params: Parameters instance
            Must be compatible with mesoGIF.Parameters
        dt: float
            Time step. Typically [mean field model].dt
        K: int
            Size of the memory vector. Typically [mean field model].K
        θ, θtilde: Series
            Discretized kernels θ and θtilde.
        """

        # To determine stationary activity we need actual Numpy values, not Theano variables
        pdict = params if isinstance(params, dict) else params.dict()
        params = GIF.Parameters.parse_obj(
            { name: shim.eval(value) for name, value in pdict.items() )
            # Create a new parameter object, of same type as the model parameters,
            # but with values rather than Theano variables
        if shim.is_theano_variable(θ._data):
            θarr = θ._data.eval()
        else:
            θarr = θ._data
        if shim.is_theano_variable(θtilde._data):
            θtildearr = θtilde._data.eval()
        else:
            θtildearr = θtilde._data

        dt = model.dt
        # TODO: Find something less ugly than the following. Maybe using np.vectorize ?
        class F:
            def __init__(self, model):
                self.model = model
                self.f = self.model.f
                if shim.is_theano_variable(self.f(0)):
                    # Compile a theano function for f
                    u_var = shim.getT().dscalar('u')
                    u_var.tag.test_value = 0
                        # Give a test value so we can run with compute_test_value == 'raise'
                    self.f = shim.gettheano().function([u_var], self.model.f(u_var))
            def __getitem__(self, α):
                def _f(u):
                    if isinstance(u, Iterable):
                        return np.array([self.f(ui)[α] for ui in u])
                    else:
                        return self.f(u)[α]
                return _f
        f = F(model)

        # Define the equation we need to solve
        k_refs = np.rint(params.t_ref / dt).astype('int')
        if (k_refs <= 0).any():
            raise ValueError("The timestep (currently {}) cannot be greater than the "
                             "shortest refractory period ({})."
                             .format(dt, params.t_ref.min()))
        jarrs = [np.arange(k0, K) for k0 in k_refs]
        memory_time = K * dt
        def rhs(A):
            a = lambda α: ( np.exp(-(jarrs[α]-k_refs[α]+1)*dt/params.τ_m[α]) * (params.u_r[α] - params.u_rest[α])
                + params.u_rest[α] - params.u_th[α] - θarr[k_refs[α]-1:K-1,α] )

            b = lambda α: ( (1 - np.exp(-(jarrs[α]-k_refs[α]+1)*dt/params.τ_m[α]))[:,np.newaxis]
                * params.τ_m[α] * params.p[α] * params.N * params.w[α] )

            # TODO: remove params.N factor once it's removed in model
            θtilde_dis = lambda α: θtildearr[k_refs[α]-1:K-1,α] * params.N[α] # starts at j+1, ends at K incl.
            c = lambda α: params.J_θ[0,α] * np.exp(-memory_time/params.τ_θ[0,α]) + dt * np.cumsum(θtilde_dis(α)[::-1])[::-1]

            ap = lambda α: params.u_rest[α] - params.u_th[α]

            bp = lambda α: (1 - np.exp(-dt/params.τ_m[α])) * params.τ_m[α] * params.p[α] * params.N * params.w[α]

            cp = lambda α: params.J_θ[0,α] * np.exp(-memory_time/params.τ_θ[0,α])

            return ( (k_refs + 1).astype(float)
                    + np.array( [ np.exp(- f[α](a(α) + (b(α) * A).sum(axis=-1) - c(α)*A[α])[:-1].cumsum()*dt).sum()
                                for α in range(len(params.N))])
                    + np.array( [ ( np.exp(- f[α](a(α) + (b(α) * A).sum(axis=-1) - c(α)*A[α]).sum()*dt)
                                / (1 - np.exp(-f[α](ap(α) + (bp(α)*A).sum(axis=-1) - cp(α)*A[α])*dt)) )
                                for α in range(len(params.N)) ] ).flatten()
                ) * A * dt - 1

        # Solve the equation for A*
        Aguess = np.ones(len(params.N)) * 10
        rhs(Aguess)
        res = root(rhs, Aguess)

        if not res.success:
            raise RuntimeError("Root-finding algorithm was unable to find a stationary activity.")
        else:
            return res.x

    @staticmethod
    def get_η_csts(model, K, θ, θtilde):
        """
        Returns the tensor constants which, along with the stationary activity,
        allow calculating the stationary value of each state variable. See the notebook
        'docs/Initial_condition.ipynb' for their definitions.

        Parameters (not up to date)
        ----------
        params: Parameters instance
            Must be compatible with mesoGIF.Parameters
        dt: float
            Time step. Typically [mean field model].dt
        K: int
            Size of the memory vector. Typically [mean field model].K
        θ, θtilde: Series
            Discretized kernels θ and θtilde.
        """

        params = model.params
        dt = model.dt

        # To determine stationary activity we need actual Numpy values, not Theano variables
        params = type(model.params)(
            **{ name: getattr(model.params, name).get_value()
                for name in model.params._fields } )
            # Create a new parameter object, of same type as the model parameters,
            # but with values rather than Theano variables
        if shim.is_theano_variable(θ._data):
            θarr = θ._data.eval()
        else:
            θarr = θ._data
        if shim.is_theano_variable(θtilde._data):
            θtildearr = θtilde._data.eval()
        else:
            θtildearr = θtilde._data

        # There are a number of factors K-1 below because the memory vector
        # doesn't include the first (current) bin
        Npop = len(params.N)
        τm = params.τ_m.flatten()
        τmT = τm[:,np.newaxis]  # transposed τ_m
        k_refs = np.rint(params.t_ref / dt).astype('int')
        jarrs = [np.arange(k0, K+1) for k0 in k_refs]
        memory_time = K*dt
        η = []
        η.append(τmT * params.p * params.N * params.w)  # η1
        η.append( (1 - np.exp(-dt/τmT)) * η[0] )        # η2
        η3 = np.empty((K, Npop))
        for α in range(Npop):
            red_factor = np.exp(-(jarrs[α]-k_refs[α]+1)*dt/params.τ_m[α])
            η3[:k_refs[α]-1, α] = params.u_r[α]
            η3[k_refs[α]-1:, α] = red_factor * (params.u_r[α] - params.u_rest[α]) + params.u_rest[α]
        η.append(η3)
        η4 = np.zeros((K, Npop))
        for α in range(Npop):
            η4[k_refs[α]-1:, α] = (1 - np.exp(- (jarrs[α] - k_refs[α] + 1)*dt / τm[α])) / (1 - np.exp(- dt / τm[α]))
        η.append(η4)
        η.append( params.u_th + θarr[:K] )   # η5
        # TODO: remove params.N factor once it's removed in model
        η.append( params.J_θ * np.exp(-memory_time/params.τ_θ.flatten())
                  + dt * params.N*np.cumsum(θtildearr[K-1::-1], axis=0)[::-1] )   # η6
        η.append( params.J_θ * np.exp(-memory_time/params.τ_θ.flatten()) )  # η7
        η.append( params.u_rest - params.u_th )  # η8
        η.append( η[2] - η[4] )  # η9
        η.append( η[3][..., np.newaxis] * η[1] )  # η10

        return η


models.register_model(GIF)
