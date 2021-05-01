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

from typing import Type, ClassVar
from pydantic import validator, root_validator
from mackelab_toolbox.typing import NPValue, FloatX

import mackelab_toolbox.utils as utils
import sinn
import sinn.config as config
from sinn.histories import Series, PopulationHistory
import sinn.kernels as kernels
import sinn.models as models
import sinn.popterm


logger = logging.getLogger("fsgif_model")


homo = False  # HACK

# HACK
shim.cf.inf = 1e12
    # Actual infinity doesn't play nice in kernels, because inf*0 is undefined

# Debug flag(s)
debugprint = False
    # If true, all intermediate values are printed in the symbolic graph

class Kernel_ε(kernels.ExpKernel):
    """
    This class just renames the generic ExpKernel to match the parameter
    names for this kernel, and ensures the kernel is normalized.
    """
    τ_s         : Tensor[FloatX, 2]
    Δ           : Tensor[FloatX, 2]
    height      : Tensor[FloatX, 2]=None  # Indicate that these parameters are computed
    decay_const : Tensor[FloatX, 2]=None  # in __post_init__ and are thus not required
    t_offset    : Tensor[FloatX, 2]=None  # when constructing kernel

    _computed_params = ('height', 'decay_const', 't_offset')

    @root_validator(cls, values, pre=True)
    def set_values(values):
        τ_s, Δ = (values.get(x, None) for x in ('τ_s', 'Δ'))
        if any(values.get(x, None) is not None
               for x in cls._computed_params):
           raise ValueError(f"Parameters {cls._computed_params} are computed "
                            "automatically and should not be provided.")
        if None not in (τ_s, Δ):
            values['height']      = 1/τ_s
            values['decay_const'] = τ_s
            values['t_offset']    = Δ
        return values


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

class Kernel_θ2(models.ModelKernelMixin, kernels.ExpKernel):
    @staticmethod
    def get_kernel_params(model_params):
        # if model_params.t_ref.ndim == 1:
        #     t_offset = model_params.t_ref[np.newaxis,:]
        # else:
        #     t_offset = model_params.t_ref
        t_offset = model_params.t_ref
        return kernels.ExpKernel.Parameters(
            height      = model_params.J_θ / model_params.τ_θ,
            decay_const = model_params.τ_θ,
            t_offset    = t_offset
        )

    # UGLY HACK: Copied function from ExpKernel and added 'expand'
    def _eval_f(self, t, from_idx=slice(None,None)):
        if homo:
            return super()._eval_f(t, from_idx)
        else:
            expand = lambda x: x.expand_blocks(['Macro', 'Micro']) if isinstance(x, sinn.popterm.PopTerm) else x
            return shim.switch(shim.lt(t, expand(self.params.t_offset[...,from_idx])),
                            0,
                            expand(self.params.height[...,from_idx]
                                * shim.exp(-(t-self.params.t_offset[...,from_idx])
                                        / self.params.decay_const[...,from_idx])) )


class GIF(models.Model):
    # ===================================
    # Class parameters
    requires_rng        :ClassVar[bool] = True
    default_initializer :ClassVar[Literal['silent', 'stationary'] = 'silent'

    class State(BaseModel):
        u     :Series
        t_hat :Series

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
    params :Parameters
    # ===================================
    # Dynamic variables
    s         : PopulationHistory
    I_ext     : History
    rndstream : shim.typing.RandomStream
    # ===================================

    # Validators
    @validator('rndstream')
    def check_rng(cls, rndstream, values):
        s = values.get('s', None)
        if s is not None:
            super().output_rng(s, rndstream)
        return rndstream

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

    # Initializers
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

    @root_validator
    def convert_to_params_popterms(cls, values):
        s = values.get('s', None)
        if s is not None:
            for k, v in values.items():
                if k != 'N':
                    values[k] = s.PopTerm(v)
        return values

    def __init__(self, ):
        """
        Parameters
        ----------
        Γ: ndarray | tensor
            If not given, computed from `p`.
        set_weights: bool, ndarray
            (Optional) Set to True to indicate that network connectivity should be set using the
            `w` and `Γ` parameters. If the spike history is already filled, set to False to
            avoid overwriting the connectivity. If an ndarray, that array will be used directly
            to set connectivity, ignoring model parameters. Default is True.
        """
        # FIXME
        if not homo and initializer == 'stationary':
            raise NotImplementedError("Stationary initialization doesn't work with heterogeneous "
                                      "populations yet. Reason: "
                                      "`τmT = self.params.τ_m.flatten()[:, np.newaxis]` line")

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
        N     = self.N
        Npops = len(N)
        self.Npops = Npops
        τ_s   = self.τ_s
        Δ     = self.Δ
        J_θ   = self.J_θ
        τ_θ   = self.τ_θ
        t_ref = self.t_ref


        ε = FactorizedKernel(
            name         = 'ε',
            outproj      = self.Γ,
            inner_kernel = BlockKernel(
                inner_kernel = ExpKernel(
                    name='ε_inner', height=1/τ_s, decay_const=τ_s, t_offset=Δ,
                    shape=(Npops,Npops)
                    )
                )
            )
        # The θ kernel is separated in two: θ1 is the constant equal to ∞ over (0, t_ref)
        # θ2 is the exponentially decaying adaptation kernel
        θ1 = BoxKernel(name='θ1', height=s.PopTerm((shim.cf.inf,)*Npops),
                       shape=(Npops,))
        θ2 = ExpKernel(name='θ2', height=J_θ/τ_θ, decay_const=τ_θ,
                       t_offset=t_ref, shape=(Npops,))
        self.θ = θ1 + θ2


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

        # Model variables
        self.RI_syn = Series(self.s, 'RI_syn',
                             shape = (N.sum(), ),
                             dtype = shim.config.floatX)
        self.λ      = Series(self.RI_syn, 'λ', dtype=shim.config.floatX)
        self.varθ   = Series(self.RI_syn, 'ϑ')
        self.u      = Series(self.RI_syn, 'u')
        # Surrogate variables
        self.t_hat  = Series(self.RI_syn, 't_hat')
            # time since last spike

        # self.statehists = [ getattr(self, varname) for varname in self.State._fields ]
        # Kernels
        shape2d = (self.Npops, self.Npops)
        self.ε = Kernel_ε('ε', self.params, shape=shape2d)
        # if values.name in ['t_ref', 'J_θ', 'τ_θ']:
        if homo:
            self.θ1 = Kernel_θ1('θ1', self.params, shape=(self.Npops,))
            self.θ2 = Kernel_θ2('θ2', self.params, shape=(self.Npops,))
        else:
            self.θ1 = Kernel_θ1('θ1', self.params, shape=(sum(N),))
            self.θ2 = Kernel_θ2('θ2', self.params, shape=(sum(N),))

        self.add_history(self.s)
        self.add_history(self.I_ext)
        self.add_history(self.λ)
        self.add_history(self.varθ)
        self.add_history(self.u)
        self.add_history(self.RI_syn)
        self.add_history(self.t_hat)

        self.s.set_update_function(self.s_fn, inputs=[self.λ])
        self.λ.set_update_function(self.λ_fn, inputs=[self.u, self.varθ])
        self.varθ.set_update_function(self.varθ_fn, inputs=[self.s])
        self.u.set_update_function(self.u_fn, inputs=[self.u, self.t_hat, self.I_ext, self.RI_syn])
        self.RI_syn.set_update_function(self.RI_syn_fn, inputs=[self.s])
        self.t_hat.set_update_function(self.t_hat_fn, inputs=[self.t_hat, self.s])

        # Pad to allow convolution
        # FIXME Check with mesoGIF to see if memory_time could be better / more consistently treated
        if memory_time is None:
            memory_time = 0
        self.memory_time = shim.cast(max(memory_time,
                                         max( kernel.memory_time
                                              for kernel in [self.ε, self.θ1, self.θ2] ) ),
                                     dtype=shim.config.floatX)
        self.K = np.rint( self.memory_time / self.dt ).astype(int)
        self.s.pad(self.memory_time)
        self.s.memory_time = self.memory_time
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
        logger.info("Initializing model state variables...")
        self.init_state_vars(initializer)
        logger.info("Done.")

        # TODO: Do something with the state variables if we don't initialize them
        #       (e.g. if we try to calculate t_hat, we will try to get t_hat[t-1],
        #       which is unset)

    def init_state_vars(self, initializer=None):
        if initializer is None:
            initializer = self.default_initializer
        else:
            # Update default initializer
            self.default_initializer = initializer

        if initializer == 'stationary':
            θ_dis, θtilde_dis = mesoGIF.discretize_θkernel(
                [self.θ1, self.θ2], self._refhist, self.params)
            init_A = mesoGIF.get_stationary_activity(
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
        Stateplus = namedtuple('Stateplus', self.State._fields + ('s',))
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

    @staticmethod
    def expand_param(param, N):
        """
        Expand a population parameter such that it can be multiplied directly
        with the spiketrain.

        Parameters
        ----------
        param: ndarray
            Parameter to expand

        N: tuple or ndarray
            Number of neurons in each population
        """

        Npops = len(N)
        if param.ndim == 1:
            return shim.concatenate( [ param[i]*np.ones((N[i],))
                                       for i in range(Npops) ] )

        elif param.ndim == 2:
            return shim.concatenate(
                [ shim.concatenate( [ param[i, j]* np.ones((N[i], N[j]))
                                      for j in range(Npops) ],
                                    axis = 1 )
                  for i in range(Npops) ],
                axis = 0 )
        else:
            raise ValueError("Parameter {} has {} dimensions; can only expand "
                             "dimensions of 1d and 2d parameters."
                             .format(param.name, param.ndim))

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

    def varθ_fn(self, t):
        """Dynamic threshold. Eq. (22)."""
        t_s = self.RI_syn.get_t_for(t, self.s)
        return (self.params.u_th + self.s.convolve(self.θ1, t_s)
                + self.s.convolve(self.θ2, t_s)).expand_axis(-1, 'Micro').values
            # Need to fix spiketrain convolution before we can use the exponential
            # optimization. (see fixme comments in histories.spiketrain._convolve_single_t
            # and kernels.ExpKernel._convolve_single_t)

    def λ_fn(self, t):
        """Hazard rate. Eq. (23)."""
        # TODO: Use self.f here (requires overloading of ops to remove pop_rmul & co.)
        t_u = self.λ.get_t_for(t, self.u)
        t_varθ = self.λ.get_t_for(t, self.varθ)
        return (self.params.c * shim.exp(  (self.u[t_u] - self.varθ[t_varθ]) / self.params.Δu ) ).expand_axis(-1, 'Micro').values

    def s_fn(self, t):
        """Spike generation"""
        t_λ = self.s.get_t_for(t, self.λ)
        return ( self.rndstream.binomial( size = self.s.shape,
                                          n = 1,
                                          p = sinn.clip_probabilities(self.λ[t_λ]*self.s.dt) )
                 .nonzero()[0].astype(self.s.idx_dtype) )
            # nonzero returns a tuple, with oner element per axis

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


class mesoGIF(models.Model):
    requires_rng = True
    Parameter_info = GIF.Parameter_info.copy()
    del Parameter_info['Γ']   # Remove the connectivity matrix
    Parameters = sinn.define_parameters(Parameter_info)

    default_initializer = 'stationary'

    # 'State' is an irreducible set of variables which uniquely define the model's state.
    # It is subdivided into latent and observed variables.
    LatentState = namedtuple('LatentState',
                       ['h', 'u',
                        'λ', 'λfree',
                        'g', 'm', 'v', 'x', 'y', 'z'])
    ObservedState = namedtuple('ObservedState',
                               ['A'])
    State = namedtuple('State', ObservedState._fields + LatentState._fields)
        # TODO: Some way of specifying how much memory is needed for each variable
        #       Or get this directly from the update functions, by some kind of introspection ?
    # HACK For propagating gradients without scan
    #      Order must be consistent with return value of symbolic_update
    #statevars = [ 'λfree', 'λ', 'g', 'h', 'u', 'v', 'm', 'x', 'y', 'z' ]

    def __init__(self, params, activity_history, input_history,
                 initializer=None, random_stream=None, memory_time=None):

        self.A = activity_history
        self.I_ext = input_history
        self.rndstream = random_stream
        if not isinstance(self.A, Series):
            raise ValueError("Activity history must be an instance of sinn.Series.")
        if not isinstance(self.I_ext, Series):
            raise ValueError("External input history must be an instance of sinn.Series.")
        # This runs consistency tests on the parameters
        models.Model.same_shape(self.A, self.I_ext)
        models.Model.same_dt(self.A, self.I_ext)
        models.Model.output_rng(self.A, self.rndstream)

        # Create the reference history
        # TODO: Will this still work when `nbar` is made a temporary ?
        # We use nbar because it is either the latest in the dependency cycle (if A is given)
        # or second latest (if A is not given)
        self.nbar = Series(self.A, 'nbar', symbolic=False)

        # Run the base class initialization
        super().__init__(params,
                         t0=self.nbar.t0, tn=self.nbar.tn, dt=self.nbar.dt,
                         public_histories=(self.A, self.I_ext),
                         reference_history=self.nbar)
        # NOTE: Do not use `params` beyond here. Always use self.params.

        N = self.params.N.get_value()
        assert(N.ndim == 1)
        self.Npops = len(N)

        # TODO: Move to History.index_interval
        self.Δ_idx = shim.concatenate(
            [ shim.concatenate (
                [ shim.asarray(self.A.index_interval(self.params.Δ.get_value()[α,β])).reshape(1,1)
                  for β in range(self.Npops) ],
                axis = 1)
              for α in range(self.Npops) ],
            axis = 0)
        # A will be indexed by 0 - Δ_idx
        self.A.pad(self.Δ_idx.max() + 1) # +1 HACK; not sure if required

        # Hyperparameters ?
        self.Nθ = 1  # Number of exponentials in threshold kernel
                     # Code currently does not allow for Nθ > 1

        # Kernels
        # shape2d = (self.Npops, self.Npops)
        # self.ε = Kernel_ε('ε', self.params, shape=shape2d)  # Not used: Exponential is hard-coded
        self.θ1 = Kernel_θ1('θ1', self.params, shape=(self.Npops,))
        self.θ2 = Kernel_θ2('θ2', self.params, shape=(self.Npops,))
            # Temporary; created to compute its memory time
        self.memory_time, self.K = self.get_memory_time(self.θ2); del self.θ2
        # HACK
        #self.memory_time = 0.02; self.K = self.A.index_interval(0.02)# - 1
            # DEBUG
        self.θ2 = Kernel_θ2('θ2', self.params, shape=(self.Npops,),
                            memory_time=self.memory_time)

        # Histories
        self.n = Series(self.A, 'n', dtype=self.params.N.dtype)
        self.h = Series(self.A, 'h')
        # self.h_tot = Series(self.A, 'h_tot', symbolic=False)
        self.h_tot = Series(self.A, 'h_tot')
        self.u = Series(self.A, 'u', shape=(self.K, self.Npops))
            # self.u[t][0] is the array of membrane potentials at time t, at lag Δt, of each population
            # TODO: Remove +1: P_λ_fn doesn't need it anymore
        self.varθ = Series(self.u, 'varθ')
        self.λ = Series(self.u, 'λ')

        # Temporary variables
        #self.nbar = Series(self.n, 'nbar', use_theano=False)
        # self.A_Δ = Series(self.A, 'A_Δ', shape=(self.Npops, self.Npops), symbolic=False)
        self.A_Δ = Series(self.A, 'A_Δ', shape=(self.Npops, self.Npops))
        #self.g = Series(self.A, 'g', shape=(self.Npops, self.Nθ,))
        self.g = Series(self.A, 'g', shape=(self.Npops,))  # HACK: Nθ = 1    # auxiliary variable(s) for the threshold of free neurons. (avoids convolution)

        # Free neurons
        self.x = Series(self.A, 'x')                        # number of free neurons
        self.y = Series(self.A, 'y', shape=(self.Npops, self.Npops))
            # auxiliary variable for the membrane potential of free neurons (avoids convolution)
        self.z = Series(self.x, 'z')
            # variance function integrated over free neurons
        self.varθfree = Series(self.A, 'varθfree', shape=(self.Npops,))  # HACK: Nθ = 1
        #self.λtilde = Series(self.u, 'λtilde')
            # In pseudocode, same symbol as λtildefree
        #self.λtildefree = Series(self.A, 'λtildefree')
        self.λfree = Series(self.A, 'λfree')
            #TODO: Either just take λtilde in the past, or make λtilde & λfree variables
        # self.Pfree = Series(self.λfree, 'Pfree', symbolic=False)
        self.Pfree = Series(self.λfree, 'Pfree')

        # Refractory neurons
        self.m = Series(self.u, 'm', shape=(self.K, self.Npops))           # Expected no. neurons for each last-spike bin
            # One more than v, because we need the extra spill-over bin to compute how many neurons become 'free' (Actually, same as v)
        # self.P_λ = Series(self.m, 'P_λ', symbolic=False)
        self.P_λ = Series(self.m, 'P_λ')
        self.v = Series(self.m, 'v', shape=(self.K, self.Npops))
        # self.P_Λ = Series(self.Pfree, 'P_Λ', symbolic=False)
        # self.X = Series(self.A, 'X', symbolic=False)
        # self.Y = Series(self.X, 'Y', symbolic=False)
        # self.Z = Series(self.X, 'Z', symbolic=False)
        # self.W = Series(self.X, 'W', symbolic=False)
        self.P_Λ = Series(self.Pfree, 'P_Λ')
        self.X = Series(self.A, 'X')
        self.Y = Series(self.X, 'Y')
        self.Z = Series(self.X, 'Z')
        self.W = Series(self.X, 'W')

        self.init_kernels()

        # Initialize the variables
        self.initialize(initializer)

        self.set_refractory_mask()
            # FIXME: Make dependence on t_ref symbolic - at present can't update t_ref

        # Set to which history the logL corresponds to
        self.observed_var = self.n._data

        #####################################################
        # Create the loglikelihood function
        # FIXME: Doesn't work with Theano histories because they only support updating tidx+1
        #        Need to create a Variable(History) type, which doesn't
        #        trigger '_compute_up_to'.
        # TODO: Use op and write as `self.nbar / self.params.N`
        #phist = Series(self.nbar, 'p')
        #phist.set_update_function(lambda t: self.nbar[t] / self.params.N)
        #phist.add_inputs([self.nbar])
        #phist = self.nbar / self.params.N
        #self.loglikelihood = self.make_binomial_loglikelihood(
        #    self.n, self.params.N, phist, approx='low p')
        #####################################################

        self.add_history(self.A)
        self.add_history(self.I_ext)
        self.add_history(self.θ_dis)
        self.add_history(self.θtilde_dis)
        self.add_history(self.n)
        self.add_history(self.h)
        self.add_history(self.h_tot)
        self.add_history(self.u)
        self.add_history(self.varθ)
        self.add_history(self.λ)
        self.add_history(self.A_Δ)
        self.add_history(self.g)
        self.add_history(self.x)
        self.add_history(self.y)
        self.add_history(self.z)
        self.add_history(self.varθfree)
        #self.add_history(self.λtildefree)
        self.add_history(self.λfree)
        #self.add_history(self.λtilde)
        self.add_history(self.Pfree)
        self.add_history(self.v)
        self.add_history(self.m)
        self.add_history(self.P_λ)
        self.add_history(self.P_Λ)
        self.add_history(self.X)
        self.add_history(self.Y)
        self.add_history(self.Z)
        self.add_history(self.W)
        self.add_history(self.nbar)

        self.A.set_update_function(self.A_fn)
        self.n.set_update_function(self.n_fn)
        self.h.set_update_function(self.h_fn)
        self.h_tot.set_update_function(self.h_tot_fn)
        self.u.set_update_function(self.u_fn)
        self.varθ.set_update_function(self.varθ_fn)
        self.λ.set_update_function(self.λ_fn)
        self.A_Δ.set_update_function(self.A_Δ_fn)
        self.g.set_update_function(self.g_fn)
        self.x.set_update_function(self.x_fn)
        self.y.set_update_function(self.y_fn)
        self.z.set_update_function(self.z_fn)
        self.varθfree.set_update_function(self.varθfree_fn)
        #self.λtildefree.set_update_function(self.λtildefree_fn)
        self.λfree.set_update_function(self.λfree_fn)
        #self.λtilde.set_update_function(self.λtilde_fn)
        self.Pfree.set_update_function(self.Pfree_fn)
        self.v.set_update_function(self.v_fn)
        self.m.set_update_function(self.m_fn)
        self.P_λ.set_update_function(self.P_λ_fn)
        self.P_Λ.set_update_function(self.P_Λ_fn)
        self.X.set_update_function(self.X_fn)
        self.Y.set_update_function(self.Y_fn)
        self.Z.set_update_function(self.Z_fn)
        self.W.set_update_function(self.W_fn)
        self.nbar.set_update_function(self.nbar_fn)


        # FIXME: At present, sinn dependencies don't support lagged
        #        inputs (all inputs are assumed to need the same time point t),
        #        while some of the dependencies below are on previous time points
        self.A.add_inputs([self.n])
        self.n.add_inputs([self.nbar])
        self.h.add_inputs([self.h, self.h_tot])
        #self.h.add_inputs([self.h_tot])
        self.h_tot.add_inputs([self.I_ext, self.A_Δ, self.y])
        self.u.add_inputs([self.u, self.h_tot])
        #self.u.add_inputs([self.h_tot])
        self.varθ.add_inputs([self.varθfree, self.n, self.θtilde_dis])
        #self.varθ.add_inputs([self.varθfree, self.θtilde_dis])
        self.λ.add_inputs([self.u, self.varθ])
        self.A_Δ.add_inputs([self.A])
        self.g.add_inputs([self.g, self.n])
        #self.g.add_inputs([])
        self.x.add_inputs([self.Pfree, self.x, self.m])
        #self.x.add_inputs([])
        self.y.add_inputs([self.y, self.A_Δ])
        #self.y.add_inputs([self.A_Δ])
        self.z.add_inputs([self.Pfree, self.z, self.x, self.v])
        #self.z.add_inputs([self.Pfree])
        self.varθfree.add_inputs([self.g])
        self.λfree.add_inputs([self.h, self.varθfree])
        self.Pfree.add_inputs([self.λfree, self.λfree])
        self.v.add_inputs([self.v, self.m, self.P_λ])
        #self.v.add_inputs([self.m, self.P_λ])
        self.m.add_inputs([self.n, self.m, self.P_λ])
        #self.m.add_inputs([])
        self.P_λ.add_inputs([self.λ])
        self.P_Λ.add_inputs([self.z, self.Z, self.Y, self.Pfree])
        #self.P_Λ.add_inputs([self.Z, self.Y, self.Pfree])
        self.X.add_inputs([self.m])
        self.Y.add_inputs([self.P_λ, self.v])
        #self.Y.add_inputs([self.P_λ])
        self.Z.add_inputs([self.v])
        #self.Z.add_inputs([])
        self.W.add_inputs([self.P_λ, self.m])
        self.nbar.add_inputs([self.W, self.Pfree, self.x, self.P_Λ, self.X])

        #if self.A._num_tidx.get_value() >= self.A.t0idx + len(self.A) - 1:
        if self.A.locked:
            self.given_A()


        # Used to fill the data by iterating
        # FIXME: This is only required because of our abuse of shared variable updates
        # if shim.config.use_theano:
        #     logger.info("Compiling advance function.")
        #     tidx = shim.getT().iscalar()
        #     self.remove_other_histories()  # HACK
        #     self.clear_unlocked_histories()
        #     self.theano_reset()
        #     self.nbar[tidx + self.nbar.t0idx]  # Fills updates
        #     self._advance_fn = shim.gettheano().function([tidx], [], updates=shim.get_updates())
        #     self.theano_reset()
        #     logger.info("Done.")

    @property
    def statehists(self):
        # HACK For propagating gradients without scan
        #      Order must be consistent with return value of symbolic_update
        # TODO: Use State rather than LatentState, so we don't need to add the A manually
        return utils.FixedGenerator(
            (getattr(self, varname) for varname in self.LatentState._fields),
            len(self.LatentState._fields) )

    def given_A(self):
        """Run this function when A is given data. It reverses the dependency
        n -> A to A -> n and fills the n array
        WARNING: We've hidden the dependency on params.N here.
        """

        assert(self.A.locked)
        #assert(self.A._num_tidx.get_value() >= self.A.t0idx + len(self.A) - 1)
        if self.A.cur_tidx < self.A.t0idx + len(self.A):
            logger.warning("Activity was only computed up to {}."
                           .format(self.A.tn))
        self.n.clear_inputs()
        # TODO: use op
        #self.n.set_update_function(lambda t: self.A[t] * self.params.N * self.A.dt)
        self.n.pad(*self.A.padding)
        self.A.pad(*self.n.padding)  # Increase whichever has less padding
        N = shim.cast(self.params.N, self.A.dtype, same_kind=False)
            # Do cast first to keep multiplication on same type
            # (otherwise, float32 * int32 => float64)
        ndata = (self.A._data * N * self.A.dt64).eval()
        assert(sinn.ismultiple(ndata, 1, rtol=self.A.dtype, atol=self.A.dtype).all()) # Make sure ndata is all integers
            # `rtol`, `atol` ensure we use float32 tolerance if A is float32
        self.n.symbolic = False
        self.n._iterative = False
        self.n.add_input(self.A)  # TODO: Useful ?
        self.n.set(ndata.round().astype(self.params.N.dtype))
            # TODO: Don't remove dependence on self.param.N
        self.n.lock()

        # HACK Everything below
        self.A_Δ.symbolic = False
            # `symbolic` should have 'None' value, indicating to deduce from
            # from inputs (which where would be self.A)
            # Or it could just return the state of `locked` ?
        self.A_Δ._iterative = False
        # Can't use `set()` because self.A may be unfilled
        tidx = self.A.get_tidx_for(self.A.cur_tidx, self.A_Δ)
        self.A_Δ._compute_up_to(tidx)
        # self.A_Δ._num_data.set_value(self.A_Δ._data.eval())
        # self.A_Δ._data = self.A_Δ._num_data
        # self.A_Δ._num_tidx.set_value(self.A_Δ._sym_tidx.eval())
        # self.A_Δ._sym_tidx = self.A_Δ._num_tidx
        self.A_Δ.lock()

    def get_memory_time(self, kernel, max_time=10):
        """
        Based on GetHistoryLength (p. 52). We set a global memory_time, rather
        than a population specific one; this is much easier to vectorize.

        Parameters
        ----------
        max_time: float
            Maximum allowable memory time, in seconds.
        """
        # def evalT(x):
        #     return (x.get_value() if shim.isshared(x)
        #             else x.eval() if shim.is_theano_variable(x)
        #             else x)

        if shim.is_theano_object(kernel.eval(0)):
            t = shim.getT().dscalar('t')
            t.tag.test_value = 0  # Don't fail if compute_test_value == 'raise'
            kernelfn = shim.gettheano().function([t], kernel.eval(t))
        else:
            kernelfn = lambda t: kernel.eval(t)

        T = shim.cast(max_time // self.A.dt64 * self.A.dt64, 'float64')  # make sure T is a multiple of dt
        #while (evalT(kernel.eval(T)) < 0.1 * self.Δ_idx).all() and T > self.A.dt:
        while (kernelfn(T) < 0.1 * self.Δ_idx).all() and T > self.A.dt64:
            T -= self.A.dt64

        T = max(T, 5*self.params.τ_m.get_value().max(), self.A.dt)
        K = self.index_interval(T, allow_rounding=True)
        return shim.cast(T,shim.config.floatX), K

    def init_kernels(self):
        if not hasattr(self, 'θ_dis'):
            assert(not hasattr(self, 'θtilde_dis'))
            self.θ_dis, self.θtilde_dis = self.discretize_θkernel(
                [self.θ1, self.θ2], self.A, self.params)
            self.θtilde_dis.add_inputs([self.θ_dis])

        else:
            assert(hasattr(self, 'θtilde_dis'))

        # Pad the the series involved in adaptation
        max_mem = self.u.shape[0]
            # The longest memories are of the size of u
        self.n.pad(max_mem)
        #self.θtilde_dis.pad(max_mem)
        self.varθ.pad(max_mem)
        self.varθfree.pad(max_mem)

        # >>>>> Extreme HACK, remove ASAP <<<<<
        self.θ_dis.locked = True
        self.θtilde_dis.locked = True
        # <<<<<

    def initialize(self, initializer=None, t=None):
        # TODO: Rename to 'initialize_state()' ?
        """
        Parameters
        ----------
        initializer: str
            One of
              - 'stationary': (Default) Stationary state under no input conditions.
              - 'silent': The last firing time of each neuron is set to -∞. Very artificial
                condition, that may require a long burnin time to remove the transient.
            If unspecified, uses the value of `self.default_initializer`.

        t: int | float
            Time at which we want to start the model. It will be intialized at the
            the time bin just before this point.

        TODO: Call this every time the model is updated
        """
        self.clear_unlocked_histories()

        # # If t is not None, convert to float so that it's consistent across histories
        # if t is not None:
        #     t = self.get_time(t)

        if initializer is None:
            initializer = self.default_initializer
        else:
            # Update default initializer
            self.default_initializer = initializer
        # TODO: Change latent -> RV to match pymc3 ?
        # Compute initial state
        if initializer == 'stationary':
            observed_state = self.get_stationary_activity(self, self.K, self.θ_dis, self.θtilde_dis)
            latent_state = self.get_stationary_state(observed_state)
                # TODO: Rename to 'get_stationary_latents'
        elif initializer == 'silent':
            observed_state = np.zeros(self.A.shape)
            latent_state = self.get_silent_latent_state()
        else:
            raise ValueError("Initializer string must be one of 'stationary', 'silent'")

        # Set the variables to the initial state
        if self.A._num_tidx.get_value() < self.A.t0idx:
            self.init_observed_vars(observed_state, t)
        self.init_latent_vars(latent_state, t)

    def init_observed_vars(self, init_A, t=None):
        """
        Originally based on InitPopulations (p. 52)

        Parameters
        ----------
        initializer: str
            One of
              - 'stationary': (Default) Stationary state under no input conditions.
              - 'silent': The last firing time of each neuron is set to -∞. Very artificial
                condition, that may require a long burnin time to remove the transient.

        t: int | float
            Time at which we want to start the model. It will be intialized at the
            the time bin just before this point.

        TODO: Call this every time the model is updated
        """
        # TODO: Use generic LatentState._fields; use 'get_stationary_observed'

        # Note that A is initialized w/ :tidx, so up to tidx-1
        if t is None:
            Atidx = self.A.t0idx
        else:
            tidx = self.get_t_idx(t)
            Atidx = tidx - self.t0idx + self.A.t0idx
        assert(Atidx >= 1)

        data = self.A._data.get_value(borrow=True)
        data[:Atidx,:] = init_A
        self.A._data.set_value(data, borrow=True)
        self.A._sym_tidx.set_value(Atidx - 1)
        assert self.A._num_tidx is self.A._sym_tidx

    def init_latent_vars(self, init_state, t=None):
        """
        Parameters
        ----------
        initializer: str
            One of
              - 'stationary': (Default) Stationary state under no input conditions.
              - 'silent': The last firing time of each neuron is set to -∞. Very artificial
                condition, that may require a long burnin time to remove the transient.

        t: int | float
            Time at which we want to start the model. It will be intialized at the
            the time bin just before this point.

        TODO: Call this every time the model is updated
        """

        # FIXME: Initialize series' to 0

        if t is None:
            tidx = self.t0idx; assert(tidx >= 0)
        else:
            tidx = self.get_t_idx(t)

        for varname in self.ObservedState._fields:
            hist = getattr(self, varname)
            if hist._num_tidx.get_value() < hist.t0idx - 1:
                raise RuntimeError("You must initialize the observed "
                                   "histories before the latents.")

        # Set initial values (make sure this is done after all padding is added)

        # ndata = self.n._data.get_value(borrow=True)
        # ndata[0] = self.params.N
        # self.n._data.set_value(ndata, borrow=True)
        # mdata = self.m._data.get_value(borrow=True)
        # mdata[0, -1, :] = self.params.N
        # self.m._data.set_value(mdata, borrow=True)

        for varname in self.LatentState._fields:
            hist = getattr(self, varname)
            initval = getattr(init_state, varname)
            hist.pad(1)  # Ensure we have at least one bin for the initial value
                # TODO: Allow longer padding
            histtidx = tidx - self.t0idx + hist.t0idx - 1; assert(histtidx >= 0)
            data = hist._data.get_value(borrow=True)
            data[histtidx,:] = initval
            hist._data.set_value(data, borrow=True)
            hist._sym_tidx.set_value(histtidx)
            assert hist._num_tidx is hist._sym_tidx

        # # Make all neurons free neurons
        # idx = self.x.t0idx - 1; assert(idx >= 0)
        # data = self.x._data.get_value(borrow=True)
        # data[idx,:] = self.params.N.get_value()
        # self.x._data.set_value(data, borrow=True)

        # # Set refractory membrane potential to u_rest
        # idx = self.u.t0idx - 1; assert(idx >= 0)
        # data = self.u._data.get_value(borrow=True)
        # data[idx,:] = self.params.u_rest.get_value()
        # self.u._data.set_value(data, borrow=True)

        # # Set free membrane potential to u_rest
        # idx = self.h.t0idx - 1; assert(idx >= 0)
        # data = self.h._data.get_value(borrow=True)
        # data[idx,:] = self.params.u_rest.get_value()
        # self.h._data.set_value(data, borrow=True)

        #self.g_l.set_value( np.zeros((self.Npops, self.Nθ)) )
        #self.y.set_value( np.zeros((self.Npops, self.Npops)) )

    def set_refractory_mask(self):
        # =============================
        # Set the refractory mask

        # TODO: Use a switch here, so ref_mask can have a symbolic dependency on t_ref
        # Create the refractory mask
        # This mask is zero for time bins within the refractory period,
        # such that it can be multiplied element-wise with arrays of length K
        self.ref_mask = np.ones(self.u.shape, dtype=np.int8)
        for l in range(self.ref_mask.shape[0]):
            # Loop over lags. l=0 corresponds to t-Δt, l=1 to t-2Δt, etc.
            for α in range(self.ref_mask.shape[1]):
                # Loop over populations α
                if (l+1)*self.dt < self.params.t_ref.get_value()[α]:
                    self.ref_mask[l, α] = 0
                else:
                    break

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
    def get_stationary_activity(model, K, θ, θtilde):
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
        params = type(model.params)(
            **{ name: shim.eval(getattr(model.params, name))
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

    def get_silent_latent_state(self):
        # K = self.varθ.shape[0]
        K = self.K
        state = self.LatentState(
            h = self.params.u_rest.get_value(),
            #h_tot = np.zeros(self.h_tot.shape),
            u = self.params.u_rest.get_value(),
            #varθ = self.params.params.u_th + self.θ_dis.get_data_trace()[:K],
            #varθfree = self.params.u_th,
            λ = np.zeros(self.λ.shape),         # Clamp the rates to zero
            λfree = np.zeros(self.λfree.shape), # idem
            g = np.zeros(self.g.shape),
            m = np.zeros(self.m.shape),
            v = np.zeros(self.v.shape),
            x = self.params.N.get_value(),
            y = np.zeros(self.y.shape),
            z = np.zeros(self.z.shape)          # Clamped to zero
            )
        return state

    def get_stationary_state(self, Astar):

        # HACK: force f to return plain Numpy object
        f = self.f
        if shim.is_theano_variable(f(0)):
            # Compile a theano function for f
            u_var = shim.getT().dvector('u')
            u_var.tag.test_value = self.params.u_rest.get_value()
            f = shim.gettheano().function([u_var], self.f(u_var))

        η = self.get_η_csts(self, self.K,
                            self.θ_dis, self.θtilde_dis)
        state = self.LatentState(
            h = self.params.u_rest.get_value() + η[0].dot(Astar),
            u = η[2] + η[9].dot(Astar),
            λ = np.stack(
                  [ f(u) for u in η[8] + (η[9]*Astar).sum(axis=-1) - η[5]*Astar ] ),
            λfree = f(η[7] + η[0].dot(Astar) - η[6].flatten()*Astar),
            g = Astar,
            # The following quantities are set below
            m = 0,
            v = 0,
            x = 0,
            y = Astar,
            z = 0
            )

        λprev = np.concatenate(
            ( np.zeros((1,) + state.λ.shape[1:]),
              state.λ[:-1] )  )
        P = 1 - np.exp(-(state.λ+λprev)/2 *self.dt)
        Pfree = 1 - np.exp(-state.λfree*self.dt)

        m = np.empty((self.K, self.Npops))
        v = np.empty((self.K, self.Npops))
        m[0] = Astar * self.params.N.get_value() * self.dt
        v[0, ...] = 0
        for i in range(1, self.K):
            m[i] = (1 - P[i])*m[i-1]
            v[i] = (1 - P[i])**2 * v[i-1] + P[i] * m[i-1]
        x = m[-1] / Pfree
        z = (x + v[-1]/Pfree) / (2 - Pfree)
        state = state._replace(
            m = m,
            v = v,
            x = x,
            z = z
            )

        return state

    # FIXME: Before replacing with Model's `advance`, need to remove HACKs
    def advance(self, stop):

        if stop == 'end':
            stopidx = self.tnidx
        else:
            stopidx = self.get_t_idx(stop)

        # Make sure we don't go beyond given data
        for hist in self.history_set:
            # HACK: Should exclude kernels
            if hist.name in ['θ_dis', 'θtilde_dis']:
                continue
            if hist.locked:
                tnidx = hist._num_tidx.get_value()
                if tnidx < stopidx - self.t0idx + hist.t0idx:
                    logger.warning("Locked history '{}' is only provided "
                                   "up to t={}. Output will be truncated."
                                   .format(hist.name, hist.get_time(tnidx)))
                    stopidx = tnidx - hist.t0idx + self.t0idx


        # TODO: Check that histories, rather than shim are symbolic ?
        #       Theano could have been loaded afterwards.
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
                # newvals = self._advance_fn(curtidx, stopidx)
                # # HACK: We change the history directly to avoid dealing with updates
                # for hist, newval in zip(self.statehists, newvals):
                #     valslice = slice(curtidx-self.t0idx+1+hist.t0idx, stopidx+hist.t0idx-self.t0idx)

                #     data = hist._num_data.get_value(borrow=True)
                #     data[valslice] = newval
                #     hist._num_data.set_value(data, borrow=True)
                #     hist._data = hist._num_data

                #     hist._num_tidx.set_value( valslice.stop - 1 )
                #     hist._sym_tidx = hist._num_tidx


    def remove_other_histories(self):
        """HACK: Remove histories from sinn.inputs that are not in this model."""
        histnames = [h.name for h in self.history_set]
        dellist = []
        for h in sinn.inputs:
            if h.name not in histnames:
                dellist.append(h)
        for h in dellist:
            del sinn.inputs[h]

    def loglikelihood(self, start, batch_size, data=None, avg=False,
                      flags=()):
        """
        Parameters
        ----------
        ...
        avg: bool
            True: Return the average log likelihood per time point in the batch.
            False (default): Return the loglikelihood of the batch.
        flags: iterable of str
            Flags change the output format. Except when debugging, this should
            be left to its default value, as other functions may expect its
            default return format. Possible values:
            - 'last_logL' (default) | 'all_logL': Return only the last
              cumulative log likelihood value, or all of them.
            - 'states' (default) | 'no_states': Return the sequence of state
              values, or omit it from the input.
            - 'updates' (default) | 'no_updates': Include the update dictionary
              returned by `scan()`.
            - 'optimize' (default) | 'mirror theano': Allow Numpy optimizations
              that break the mirroring of Numpy and Theano code.
            NOTE: At present flags only affect the output when not computing

        Returns
        -------
        log likelihood: symbolic scalar
            Symbolic expression for the likelihood
        state variable updates: list
            List of symbolic expressions, one per state variable.
            Each expression is a symbolic array of length `batch_size`, giving
            for each variable all the time slices between `start` and `start+batch_size`.
        symbolic update dictionary: dict
            Update dictionary returned by the internal call to `scan()`.

        TODO: I'm not sure the state variable updates are useful; if really needed,
              that information should be in the update dictionary.
        """
        # Set the output flags
        # TODO: Use an enum
        # Each row corresponds to an option. First element of row is default.
        flag_values = [['last_logL', 'all_logL'],
                       ['states', 'no_states'],
                       ['updates', 'no_updates'],
                       ['optimize', 'mirror theano']]
        if isinstance(flags, str):
            flags = (flags,)
        outflags = set()
        for option in flag_values:
            # Prepend with the default
            values = [option[0]] + [f for f in flags if f in option]
            outflags.add(values[-1])  # Will select default if no option was given

        ####################
        # Some hacks to get around current limitations

        self.remove_other_histories()

        # End hacks
        #####################

        batch_size = self.index_interval(batch_size)
        startidx = self.get_t_idx(start)
        stopidx = startidx + batch_size
        N = self.params.N
        if data is None:
            n_full = self.n
            t0idx = self.n.t0idx
        else:
            n_full = data.astype(self.params.N.dtype)
            t0idx = 0 # No offset if we provide data

        # Windowed test
        #windowlen = 5
        #stopidx -= windowlen

        def logLstep(tidx, *args):
            # Log likelihood at one time step
            # For batch sizes > 1, log likelihoods are automatically averaged
            # across time steps, so don't do so here
            if shim.is_theano_object(tidx):
                # statevar_updates, input_vars, output_vars = self.symbolic_update(tidx, args[2:])
                state_outputs, updates = self.symbolic_update(tidx, *args[2:])
                    # FIXME: make this args[1:] once n is in state variables
                #nbar = output_vars[self.nbar]
                # state_outputs = [shim.print(var) for var in state_outputs] # DEBUG
                nbar = self.symbolic_nbar(args[2:], state_outputs)
            else:
                nbar = self.nbar[tidx-self.t0idx+self.nbar.t0idx]
                state_outputs = []
                updates = shim.get_updates()
            if debugprint: nbar = shim.print(nbar, "nbar (log L)")
            p = sinn.clip_probabilities(nbar / self.params.N)
            n = n_full[tidx-self.t0idx+t0idx]
            if debugprint: p = shim.print(p, "p (log L)")
            if debugprint: n = shim.print(n, "n (log L)")

            assert(args[0].dtype == shim.config.floatX)
            cum_logL = args[0] + ( -shim.gammaln(n+1) - shim.gammaln(N-n+1)
                                   + n*shim.log(p)
                                   + (N-n)*shim.log(1-p)
                                  ).sum(dtype=shim.config.floatX).astype(shim.config.floatX)
                                  # Sometimes just setting dtype in sum is not enough

            return [cum_logL] + [n] + state_outputs, {}
            # FIXME: Remove [n] once it is included in state vars
            # return [cum_logL], shim.get_updates()

        if shim.is_theano_object([self.nbar._data, self.params, self.n._data]):
            logger.info("Producing the likelihood graph.")

            # Create the outputs_info list
            # First element is the loglikelihood, subsequent are aligned with input_vars
            outputs_info = [shim.cast(0, shim.config.floatX, same_kind=False)]

            # FIXME: Remove once 'n' is in state variables
            outputs_info.append( self.n._data[startidx - self.t0idx + self.n.t0idx - 1] )

            for hist in self.statehists:
                res = sinn.upcast(
                    hist._data[startidx - self.t0idx + hist.t0idx - 1],
                    disable_rounding=True)
                    # upcast is a no-op if it is not needed
                outputs_info.append(res)
                # HACK !!
                # if hist.name == 'v':
                #     outputs_info[-1] = shim.getT().unbroadcast(outputs_info[-1], 1)
                # elif hist.name == 'z':
                #     outputs_info[-1] = shim.getT().unbroadcast(outputs_info[-1], 0)
            if batch_size == 1:
                # No need for scan
                outputs, upds = logLstep(start, *outputs_info)
                logL = outputs[0]
                # outputs[0] = [outputs[0]]

            else:
                outputs, upds = shim.gettheano().scan(logLstep,
                                                      sequences = shim.getT().arange(startidx, stopidx),
                                                      outputs_info = outputs_info)
                                                      #outputs_info = np.float64(0))
                # HACK Since we are still using shared variables for data
                #for hist, new_data in outputs[1:]:
                #    hist.update(slice(startidx+hist.t0idx, stopidx+hist.t0idx),
                #                new_data)

                # Normalize the loglikelihood so that it is consistent when
                # we change batch size
                # outputs[0] = outputs[0] / shim.getT().arange(1, 1+batch_size)
                logL = outputs[0][-1]
                if avg:
                    logL = logL / batch_size

                self.apply_updates(upds)
                    # Applying updates is essential to remove the iteration variable
                    # scan introduces from the shim updates dictionary

            logger.info("Likelihood graph complete")

            return logL, outputs[1:], upds
                # logL = outputs[0]; outputs[1:] => statevars
        else:
            if 'mirror theano' in outflags:
                # TODO: Remove this branch once shim.scan is implemented
                logL = np.zeros(stopidx - startidx, dtype=shim.config.floatX)
                logL[0] = logLstep(startidx, np.array(0, dtype=shim.config.floatX))[0][0]
                for t in np.arange(startidx+1, stopidx):
                    logL[t-startidx] = logLstep(t, logL[t-startidx-1])[0][0]
            else:
                nbar = self.nbar[startidx:stopidx]
                p = sinn.clip_probabilities(nbar / self.params.N)
                n = n_full[startidx-self.t0idx+t0idx:stopidx-self.t0idx+t0idx]
                # To match the output of 'mirror theano', we first sum across
                # populations to have one logL per time point, then cumsum
                sumaxes = tuple(range(1, n.ndim)) # all axes except first
                logL = shim.cumsum( (-shim.gammaln(n+1) - shim.gammaln(N-n+1)
                                     + n*shim.log(p)
                                     + (N-n)*shim.log(1-p)
                                    ).sum(axis=sumaxes,
                                          dtype=shim.config.floatX),
                                    dtype=shim.config.floatX
                                  ).astype(shim.config.floatX)

            upds = shim.get_updates()

            if avg:
                logL /= np.arange(1,batch_size+1)

            retval = [logL] if 'all_logL' in outflags else [logL[-1]]
            if 'states' in outflags:
                retval.append([ self.n[startidx-self.t0idx+self.n.t0idx : stopidx-self.t0idx+self.n.t0idx] ]
                              + [ hist[startidx-self.t0idx+hist.t0idx : stopidx-self.t0idx+hist.t0idx]
                                  for hist in self.statehists])
            if 'updates' in outflags:
                retval.append( upds )
            return retval[0] if len(retval) == 1 else retval

    def f(self, u):
        """Link function. Maps difference between membrane potential & threshold
        to firing rate."""
        return self.params.c * shim.exp(u/self.params.Δu.flatten())

    def A_fn(self, t):
        """p. 52"""
        t_n = self.A.get_t_for(t, self.n)
        return self.n[t_n] / (self.params.N * self.A.dt)

    def h_fn(self, t):
        """p.53, also Eq. 92 p. 48"""
        tidx_h = self.h.get_t_idx(t)
        t_htot = self.h.get_t_for(t, self.h_tot)
        #tidx_h_tot = self.h_tot.get_t_idx(t)
        red_factor = shim.exp(-self.h.dt/self.params.τ_m.flatten() )
        return ( (self.h[tidx_h-1] - self.params.u_rest) * red_factor
                 + self.h_tot[t_htot] )

    def A_Δ_fn(self, t):
        """p.52, line 9"""
        tidx_A = self.A_Δ.get_t_for(t, self.A)
        # a = lambda α: [ self.A[tidx_A - self.Δ_idx[α, β]][β:β+1, np.newaxis]
        #                 for β in range(self.Npops) ]
        # b = lambda α: shim.concatenate( a(α), axis=1)
        # c = [ b(α) for α in range(self.Npops) ]
        # d = shim.concatenate( c, axis = 0)

        return shim.concatenate(
            [ shim.concatenate(
                [ self.A[tidx_A - self.Δ_idx[α, β]][..., β:β+1, np.newaxis]  # make scalar 2d
                  for β in range(self.Npops) ],
                axis = -1)
              for α in range(self.Npops) ],
            axis=-2)

    def h_tot_fn(self, t):
        """p.52, line 10, or Eq. 94, p. 48
        Note that the pseudocode on p. 52 includes the u_rest term, whereas in Eq. 94
        this term is instead included in the equation for h (Eq. 92). We follow the pseudocode here.
        DEBUGGING NOTE: To compare with an equivalent quantity of the spiking model,
        compare the mean-field's `h_tot - u_rest` to the spiking's
        `(RI_syn + RI_ext)*(1-e^(Δt/τ_m))`. Use the mean field's Δt (this is
        input intgrated over a time step, so the steps have to match.)
        """
        t_AΔ = self.h_tot.get_t_for(t, self.A_Δ)
        t_y = self.h_tot.get_t_for(t, self.y)
        t_Iext = self.h_tot.get_t_for(t, self.I_ext)
        # FIXME: Check again that indices are OK (i.e. should they be ±1 ?)
        τ_m = self.params.τ_m.flatten()[:,np.newaxis]
           # We have τ_sβ, but τ_mα. This effectively transposes τ_m
        red_factor_τm = shim.exp(-self.h_tot.dt/self.params.τ_m)
        red_factor_τmT = shim.exp(-self.h_tot.dt/τ_m)
        red_factor_τs = shim.exp(-self.h_tot.dt/self.params.τ_s)
        return ( self.params.u_rest + self.params.R*self.I_ext[t_Iext] * (1 - red_factor_τm)
                 + ( τ_m * (self.params.p * self.params.w) * self.params.N
                       * (self.A_Δ[t_AΔ]
                          + ( ( self.params.τ_s * red_factor_τs * ( self.y[t_y] - self.A_Δ[t_AΔ] )
                                - red_factor_τmT * (self.params.τ_s * self.y[t_y] - τ_m * self.A_Δ[t_AΔ]) )
                              / (self.params.τ_s - τ_m) ) )
                   ).sum(axis=-1) )

    def y_fn(self, t):
        """p.52, line 11"""
        tidx_y = self.y.get_t_idx(t)
        t_AΔ = self.y.get_t_for(t, self.A_Δ)
        red_factor = shim.exp(-self.y.dt/self.params.τ_s)
        return self.A_Δ[t_AΔ] + (self.y[tidx_y-1] - self.A_Δ[t_AΔ]) * red_factor

    # TODO: g and varθ: replace flatten by sum along axis=1

    def g_fn(self, t):
        """p. 53, line 5, also p. 45, Eq. 77b"""
        tidx_g = self.g.get_t_idx(t)
        tidx_n = self.g.get_tidx_for(t, self.n)
        # TODO: cache the reduction factor
        # FIXME: Not sure if this should be tidx_n-self.K-1
        red_factor = shim.exp(- self.g.dt/self.params.τ_θ)  # exponential reduction factor
        return ( self.g[tidx_g-1] * red_factor
                 + (1 - red_factor) * self.n[tidx_n-self.K] / (self.params.N * self.g.dt)
                ).flatten()

    def varθfree_fn(self, t):
        """p. 53, line 6 and p. 45 Eq. 77a"""
        #tidx_varθ = self.varθ.get_t_idx(t)
        # TODO: cache reduction factor
        t_g = self.varθfree.get_t_for(t, self.g)
        red_factor = (self.params.J_θ * shim.exp(-self.memory_time/self.params.τ_θ)).flatten()
        return self.params.u_th + red_factor * self.g[t_g]
            # TODO: sum over exponentials (l) of the threshold kernel

    def λfree_fn(self, t):
        """p. 53, line 8"""
        t_h = self.λfree.get_t_for(t, self.h)
        t_varθfree = self.λfree.get_t_for(t, self.varθfree)
        # FIXME: 0 or -1 ?
        return self.f(self.h[t_h] - self.varθfree[t_varθfree][0])

    def Pfree_fn(self, t):
        """p. 53, line 9"""
        tidx_λ = self.Pfree.get_tidx_for(t, self.λfree)
        #self.λfree._compute_up_to(tidx_λ)
            # HACK: force Theano to compute up to tidx_λ first
            #       This is required because of the hack in History._compute_up_to
            #       which assumes only one update per history is required
        return 1 - shim.exp(-0.5 * (self.λfree[tidx_λ-1] + self.λfree[tidx_λ]) * self.Pfree.dt )

    def X_fn(self, t):
        """p. 53, line 12"""
        #tidx_m = self.m.get_t_idx(t)
        t_m = self.X.get_t_for(t, self.m)
        return (self.m[t_m]).sum(axis=-2)
            # axis 0 is for lags, axis 1 for populations
            # FIXME: includes the absolute ref. lags

    def varθ_fn(self, t):
        """p.53, line 11, 15 and 16, and Eq. 110 (p.50)"""
        # Follows pseudocode definitions: (110)'s division by N is already
        # included in θtilde_dis
        # FIXME: does not correctly include cancellation from line 11
        t_varθfree = self.varθ.get_t_for(t, self.varθfree)
        tidx_n = self.varθ.get_tidx_for(t, self.n)
        # K = self.u.shape[0]
        K = self.K
        # HACK: use of ._data to avoid indexing θtilde (see comment where it is created)
        # TODO: sum should be shifted by one index (and first elem 0) instead of subtracting n[t_k]
        varθref = ( shim.cumsum(self.n[tidx_n-K:tidx_n]*self.θtilde_dis._data[:K][...,::-1,:],
                                axis=-2)
                    - self.n[tidx_n-K:tidx_n]*self.θtilde_dis._data[:K][...,::-1,:])[...,::-1,:]

        # FIXME: Use indexing that is robust to θtilde_dis' t0idx
        # FIXME: Check that this is really what line 15 says
        return self.θ_dis._data[:K] + self.varθfree[t_varθfree] + varθref

    def u_fn(self, t):
        """p.53, line 17 and 35"""
        tidx_u = self.u.get_tidx(t)
        t_htot = self.u.get_t_for(t, self.h_tot)
        red_factor = shim.exp(-self.u.dt/self.params.τ_m).flatten()[np.newaxis, ...]
        # TODO: Fix for array t
        return shim.concatenate(
            ( self.params.u_r[..., np.newaxis, :],
              ((self.u[tidx_u-1][:-1] - self.params.u_rest[np.newaxis, ...]) * red_factor + self.h_tot[t_htot][np.newaxis,...]) ),
            axis=-2)

    #def λtilde_fn(self, t):
    def λ_fn(self, t):
        """p.53, line 18"""
        t_u = self.λ.get_t_for(t, self.u)
        t_varθ = self.λ.get_t_for(t, self.varθ)
        return self.f(self.u[t_u] - self.varθ[t_varθ]) * self.ref_mask

    def P_λ_fn(self, t):
        """p.53, line 19"""
        tidx_λ = self.P_λ.get_tidx_for(t, self.λ)
        #self.λ._compute_up_to(tidx_λ)  # HACK: see Pfree_fn
        if shim.isscalar(t):
            slice_shape = (1,) + self.λ.shape[1:]
            λprev = shim.concatenate( ( shim.zeros(slice_shape),
                                        self.λ[tidx_λ-1][:-1] ),
                                    axis=0)
        else:
            assert(t.ndim == 1)
            slice_shape = (t.shape[0],) + (1,) + self.λ.shape[1:]
            λprev = shim.concatenate( ( shim.zeros(slice_shape, dtype=self.λ.dtype),
                                        self.λ[tidx_λ-1][:,:-1] ),
                                    axis=1)
        P_λ = 0.5 * (self.λ[tidx_λ][:] + λprev) * self.P_λ.dt
        return shim.switch(P_λ <= 0.01,
                           P_λ,
                           1 - shim.exp(-P_λ))

    # def λ_fn(self, t):
    #     """p.53, line 21 and 36"""
    #     return self.λtilde[t]
    #         # FIXME: check that λ[t] = 0 (line 36)

    def Y_fn(self, t):
        """p.53, line 22"""
        tidx_v = self.Y.get_tidx_for(t, self.v)
        t_Pλ = self.Y.get_t_for(t, self.P_λ)
        return (self.P_λ[t_Pλ] * self.v[tidx_v - 1]).sum(axis=-2)
            # FIXME: includes abs. refractory lags

    def Z_fn(self, t):
        """p.53, line 23"""
        #tidx_Z = self.Z.get_t_idx(t)
        tidx_v = self.Z.get_tidx_for(t, self.v)
        return self.v[tidx_v-1].sum(axis=-2)
            # FIXME: includes abs. refractory lags

    def W_fn(self, t):
        """p.53, line 24"""
        t_Pλ = self.W.get_t_for(t, self.P_λ)
        t_m = self.W.get_t_for(t, self.m)
        ref_mask = self.ref_mask[:self.m.shape[0],:]
            # ref_mask is slightly too large, so we truncate it
        return (self.P_λ[t_Pλ] * self.m[t_m] * ref_mask).sum(axis=-2)
            # FIXME: includes abs. refractory lags

    def v_fn(self, t):
        """p.53, line 25 and 34"""
        tidx_v = self.v.get_tidx(t)
        tidx_m = self.v.get_tidx_for(t, self.m)
        t_Pλ = self.v.get_tidx_for(t, self.P_λ)
        if shim.isscalar(t):
            slice_shape = (1,) + self.v.shape[1:]
            return shim.concatenate(
                ( shim.zeros(slice_shape, dtype=shim.config.floatX),
                  (1 - self.P_λ[t_Pλ][1:])**2 * self.v[tidx_v-1][:-1] + self.P_λ[t_Pλ][1:] * self.m[tidx_m-1][:-1]
                ),
                axis=0)
        else:
            assert(t.ndim == 1)
            slice_shape = t.shape + (1,) + self.v.shape[1:]
            return shim.concatenate(
                ( shim.zeros(slice_shape, dtype=shim.config.floatX),
                  (1 - self.P_λ[t_Pλ][:,1:])**2 * self.v[tidx_v-1][:,:-1] + self.P_λ[t_Pλ][:,1:] * self.m[tidx_m-1][:,:-1]
                ),
                axis=1)

    def m_fn(self, t):
        """p.53, line 26 and 33"""
        tidx_m = self.m.get_tidx(t)
        t_Pλ = self.m.get_t_for(t, self.P_λ)
        tidx_n = self.m.get_tidx_for(t, self.n)
        # TODO: update m_0 with n(t)
        # TODO: fix shape if t is array
        return shim.concatenate(
            ( self.n[tidx_n-1][np.newaxis,:],
              ((1 - self.P_λ._data[t_Pλ][1:]) * self.m[tidx_m-1][:-1]) ),
            axis=-2 )
            # HACK: Index P_λ data directly to avoid triggering its computational update before v_fn

    def P_Λ_fn(self, t):
        """p.53, line 28"""
        tidx_z = self.P_Λ.get_tidx_for(t, self.z)
        t_Z = self.P_Λ.get_t_for(t, self.Z)
        t_Y = self.P_Λ.get_t_for(t, self.Y)
        t_Pfree = self.P_Λ.get_t_for(t, self.Pfree)
        z = self.z[tidx_z-1] # Hack: Don't trigger computation of z 'up to' t-1
        Z = self.Z[t_Z]
        return shim.switch( Z + z > 0,
                            ( (self.Y[t_Y] + self.Pfree[t_Pfree]*z)
                              / (shim.abs(Z + z) + sinn.config.abs_tolerance) ),
                            0 )

    def nbar_fn(self, t):
        """p.53, line 29"""
        t_W = self.nbar.get_t_for(t, self.W)
        t_Pfree = self.nbar.get_t_for(t, self.Pfree)
        t_x = self.nbar.get_t_for(t, self.x)
        t_PΛ = self.nbar.get_t_for(t, self.P_Λ)
        t_X = self.nbar.get_t_for(t, self.X)
        return ( self.W[t_W] + self.Pfree[t_Pfree] * self.x[t_x]
                 + self.P_Λ[t_PΛ] * (self.params.N - self.X[t_X] - self.x[t_x]) )

    def n_fn(self, t):
        """p.53, lines 30 and 33"""
        t_nbar = self.n.get_t_for(t, self.nbar)
        if shim.isscalar(t):
            size = self.n.shape
            n = self.params.N
        else:
            size = t.shape + self.n.shape
            n = self.params.N[np.newaxis, ...]
        return self.rndstream.binomial( size = size,
                                        n = n,
                                        p = sinn.clip_probabilities(self.nbar[t_nbar]/self.params.N) ).astype(self.params.N.dtype)
            # If N.dtype < int64, casting as params.N.dtype allows to reduce the memory footprint
            # (and n may never be larger than N)

    def z_fn(self, t):
        """p.53, line 31"""
        tidx_x = self.z.get_tidx_for(t, self.x)
        #tidx_v = self.v.get_t_idx(t)
        tidx_z = self.z.get_tidx(t)
        t_Pfree = self.z.get_t_for(t, self.Pfree)
        t_v = self.z.get_t_for(t, self.v)
        return ( (1 - self.Pfree[t_Pfree])**2 * self.z[tidx_z-1]
                 + self.Pfree[t_Pfree]*self.x[tidx_x-1]
                 + self.v[t_v][0] )

    def x_fn(self, t):
        """p.53, line 32"""
        tidx_x = self.x.get_tidx(t)
        tidx_m = self.x.get_tidx_for(t, self.m)
        tidx_P = self.x.get_tidx_for(t, self.Pfree)
        # TODO: ensure that m can be used as single time buffer, perhaps
        #       by merging the second line with m_fn update ?
        return ( (1 - self.Pfree[tidx_P]) * self.x[tidx_x-1]
                 + self.m._data[tidx_m][-1] )
            # HACK: Index P_λ, m _data directly to avoid triggering it's computational udate before v_fn


    def symbolic_update(self, tidx, *statevars):
        """
        Temorary fix to get symbolic updates. Eventually sinn should
        be able to do this itself.
        """

        curstate = self.LatentState(*statevars)

        λfree0 = curstate.λfree
        λ0 = curstate.λ
        #Pfree0 = statevars[2]
        #P_λ0 = statevars[3]
        g0 = curstate.g
        h0 = curstate.h
        u0 = curstate.u
        v0 = curstate.v
        m0 = curstate.m
        x0 = curstate.x
        y0 = curstate.y
        z0 = curstate.z

        # convert model time to difference from t0
        tidx = tidx - self.t0idx
        if debugprint: tidx = shim.print(tidx, 'tidx')

        # shared constants
        tidx_n = tidx + self.n.t0idx

        # yt
        red_factor = shim.exp(-self.y.dt/self.params.τ_s)
        yt = self.A_Δ[tidx+self.A_Δ.t0idx] + (y0 - self.A_Δ[tidx+self.A_Δ.t0idx]) * red_factor
        yt.name = 'yt'
        if debugprint: yt = shim.print(yt)

        # htot
        τ_mα = self.params.τ_m.flatten()[:,np.newaxis]
        red_factor_τm = shim.exp(-self.h_tot.dt/self.params.τ_m)
        red_factor_τmT = shim.exp(-self.h_tot.dt/τ_mα)
        red_factor_τs = shim.exp(-self.h_tot.dt/self.params.τ_s)
        # τ_mα = shim.print(τ_mα, 'τ_mα')
        # red_factor_τm = shim.print(red_factor_τm, 'red_factor_τm')
        # red_factor_τmT = shim.print(red_factor_τmT, 'red_factor_τmT')
        # red_factor_τs = shim.print(red_factor_τs, 'red_factor_τs')
        h_tot = ( self.params.u_rest + self.params.R*self.I_ext[tidx+self.I_ext.t0idx] * (1 - red_factor_τm)
                 + ( τ_mα * (self.params.p * self.params.w) * self.params.N
                       * (self.A_Δ[tidx+self.A_Δ.t0idx]
                          + ( ( self.params.τ_s * red_factor_τs * ( yt - self.A_Δ[tidx+self.A_Δ.t0idx] )
                                - red_factor_τmT * (self.params.τ_s * yt - τ_mα * self.A_Δ[tidx+self.A_Δ.t0idx]) )
                              / (self.params.τ_s - τ_mα) ) )
                   ).sum(axis=-1, dtype=shim.config.floatX) )
        h_tot.name = 'h_tot'
        if debugprint: h_tot = shim.print(h_tot)


        # ht
        red_factor = shim.exp(-self.h.dt/self.params.τ_m.flatten() )
        ht = ( (h0 - self.params.u_rest) * red_factor + h_tot )
        ht.name = 'ht'
        if debugprint: ht = shim.print(ht)

        # ut
        red_factor = shim.exp(-self.u.dt/self.params.τ_m).flatten()[np.newaxis, ...]
        ut = shim.concatenate(
            ( self.params.u_r[..., np.newaxis, :],
              ((u0[:-1] - self.params.u_rest[np.newaxis, ...]) * red_factor + h_tot[np.newaxis,...]) ),
            axis=-2)
        ut.name = 'ut'
        if debugprint: ut = shim.print(ut)

        # gt
        red_factor = shim.exp(- self.g.dt/self.params.τ_θ)
        gt = ( g0 * red_factor
                 + (1 - red_factor) * self.n._data[tidx_n-self.K]
                   / (self.params.N * self.g.dt)
                ).flatten()
        gt.name = 'gt'
        if debugprint: gt = shim.print(gt)

        # varθfree
        red_factor = (self.params.J_θ * shim.exp(-self.memory_time/self.params.τ_θ)).flatten()
        varθfree =  self.params.u_th + red_factor * gt
        varθfree.name = 'varθfree'
        if debugprint: varθfree = shim.print(varθfree)

        # varθ
        # K = self.u.shape[0]
        K = self.K
        varθref = ( shim.cumsum(self.n._data[tidx_n-K:tidx_n] * self.θtilde_dis._data[:K][...,::-1,:],
                                axis=-2)
                              - self.n._data[tidx_n-K:tidx_n] * self.θtilde_dis._data[:K][...,::-1,:])[...,::-1,:]
        varθ = self.θ_dis._data[:K] + varθfree + varθref
        varθ.name = 'varθ'
        if debugprint: varθ = shim.print(varθ)

        # λt
        λt = self.f(ut - varθ) * self.ref_mask
        λt.name = 'λt'
        if debugprint: λt = shim.print(λt)

        # λfree
        λfreet = self.f(ht - varθfree[0])
        λfreet.name = 'λfreet'
        if debugprint: λfreet = shim.print(λfreet)

        # Pfreet
        Pfreet = 1 - shim.exp(-0.5 * (λfree0 + λfreet) * self.λfree.dt )
        Pfreet.name = 'Pfreet'
        if debugprint: Pfreet = shim.print(Pfreet)

        # P_λt
        λprev = shim.concatenate(
            ( shim.zeros((1,) + self.λ.shape[1:], dtype=shim.config.floatX),
              λ0[:-1] ) )
        P_λ_tmp = 0.5 * (λt + λprev) * self.P_λ.dt
        P_λt = shim.switch(P_λ_tmp <= 0.01,
                           P_λ_tmp,
                           1 - shim.exp(-P_λ_tmp))
        P_λt.name = 'P_λt'
        if debugprint: P_λt = shim.print(P_λt)

        # mt
        mt = shim.concatenate(
            ( self.n._data[tidx_n-1][np.newaxis,:], ((1 - P_λt[1:]) * m0[:-1]) ),
            axis=-2 )
        mt.name = 'mt'
        if debugprint: mt = shim.print(mt)

        # xt
        xt = ( (1 - Pfreet) * x0 + mt[-1] )
        xt.name = 'xt'
        if debugprint: xt = shim.print(xt)

        # vt
        vt = shim.concatenate(
            ( shim.zeros( (1,) + self.v.shape[1:] , dtype=shim.config.floatX),
              (1 - P_λt[1:])**2 * v0[:-1] + P_λt[1:] * m0[:-1] ),
            axis=-2)
        vt.name = 'vt'
        if debugprint: vt = shim.print(vt)

        # zt
        zt = ( (1 - Pfreet)**2 * z0  +  Pfreet*x0  + vt[0] )
        zt.name = 'zt'
        if debugprint: zt = shim.print(zt)

        newstate = self.LatentState(
            h = ht,
            u = ut,
            λ = λt,
            λfree = λfreet,
            g = gt,
            m = mt,
            v = vt,
            x = xt,
            y = yt,
            z = zt
            )

        #state_outputs = OrderedDict( (getattr(curstate, key),
        #                             for key in self.LatentState._fields )
        state_outputs = list(newstate)
        updates = {}

        # # TODO: use the key string itself
        # input_vars = OrderedDict( (getattr(self, key), getattr(curstate, key))
        #                           for key in self.LatentState._fields )

        # Output variables contain updates to the state variables, as well as
        # whatever other quantities we want to compute
#        output_vars = OrderedDict( (getattr(self, key), getattr(newstate, key))
#                                   for key in self.LatentState._fields )
        return state_outputs, updates

    def symbolic_nbar(self, curstate_list, newstate_list):
        """
        This is the part of the logL calculation "downstream" from the
        calculation of the new state.
        It takes as inputs two lists of symbolic variables, for the current
        and new state. Both are in the same order as the 'outputs_info' given
        to `scan`.
        """

        curstate = self.LatentState(*curstate_list)
        newstate = self.LatentState(*newstate_list)
        λ0 = curstate.λ
        λfree0 = curstate.λfree
        v0 = curstate.v
        z0 = curstate.z
        λt = newstate.λ
        λfreet = newstate.λfree
        mt = newstate.m
        xt = newstate.x

        λ0.name = 'λ0'
        λfree0.name = 'λfree0'
        v0.name = 'v0'
        z0.name = 'z0'
        λt.name = 'λt'
        λfreet.name = 'λfreet'
        mt.name = 'mt'
        xt.name = 'xt'

        # if debugprint: xt = shim.print(xt)
        # z0 = shim.print(z0)


        # Pfreet
        # TODO: Find way not to repeat this and P_λt from `symbolic_update()`
        Pfreet = 1 - shim.exp(-0.5 * (λfree0 + λfreet) * self.λfree.dt )
        Pfreet.name = 'Pfreet'
        if debugprint: Pfreet = shim.print(Pfreet)

        # P_λt
        λprev = shim.concatenate(
            ( shim.zeros((1,) + self.λ.shape[1:], dtype=shim.config.floatX),
              λ0[:-1] ) )
        λprev.name = 'λprev'
        P_λ_tmp = 0.5 * (λt + λprev) * self.P_λ.dt
        P_λt = shim.switch(P_λ_tmp <= 0.01,
                           P_λ_tmp,
                           1 - shim.exp(-P_λ_tmp))
        P_λt.name = 'P_λt (symb nbar)'
        if debugprint: P_λt = shim.print(P_λt)

        # W
        Wref_mask = self.ref_mask[:self.m.shape[0],:]
        W = (P_λt * mt * Wref_mask).sum(axis=-2)
        W.name = 'W'
        if debugprint: W = shim.print(W)

        # X
        X = mt.sum(axis=-2)
        X.name = 'X'
        if debugprint: X = shim.print(X)

        # Y
        Y = (P_λt * v0).sum(axis=-2)
        Y.name = 'Y'
        if debugprint: Y = shim.print(Y)

        # Z
        Z = v0.sum(axis=-2)
        Z.name = 'Z'
        if debugprint: Z = shim.print(Z)

        # P_Λ
        P_Λ = shim.switch( Z + z0 > 0,
                           ( (Y + Pfreet*z0)
                             / (shim.abs(Z + z0) + sinn.config.abs_tolerance) ),
                           0 )
        P_Λ.name = 'P_Λ'
        if debugprint: P_Λ = shim.print(P_Λ)

        # nbar
        # nbar = ( W + Pfreet * xt + P_Λ * (self.params.N - X - xt) )
        c = self.params.N - X - xt
        if debugprint: c = shim.print(c, "N - X - x")
        nbar = ( W + Pfreet * xt + P_Λ * c )

        return nbar

models.register_model(GIF)
models.register_model(mesoGIF)
