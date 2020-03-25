# -*- coding: utf-8 -*-
"""
Created Sun 7 Dec 2019
Based on fsgif_model.py (May 24 2017)

This is an adaptation of `mesogif_model.py` to use a `Series` object for the
spike history.

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

class Kernel_ε(models.ModelKernelMixin, kernels.ExpKernel):
    @staticmethod
    def get_kernel_params(model_params):
        return kernels.ExpKernel.Parameters(
            height      = 1/model_params.τ_s,
            decay_const = model_params.τ_s,
            t_offset    = model_params.Δ)

# The θ kernel is separated in two: θ1 is the constant equal to ∞ over (0, t_ref)
# θ2 is the exponentially decaying adaptation kernel
class Kernel_θ1(models.ModelKernelMixin, kernels.Kernel):
    Parameter_info = OrderedDict( ( ( 'height', 'floatX' ),
                                    ( 'start',  'floatX' ),
                                    ( 'stop',   'floatX' ) ) )
    Parameters = kernels.com.define_parameters(Parameter_info)
    @staticmethod
    def get_kernel_params(model_params):
        if homo:
            Npops = len(model_params.N.get_value())
            height = (shim.cf.inf,)*Npops
            stop = model_params.t_ref
        else:
            height = (shim.cf.inf,)*model_params.N.get_value().sum()
            stop = model_params.t_ref
            # if isinstance(model_params.t_ref, sinn.popterm.PopTerm):
            #     # TODO: This is only required because some operations
            #     #       aren't yet supported by PopTerm, so we expand
            #     #       manually. Once that is fixed, we should remove this.
            #     stop = stop.expand_blocks(['Macro', 'Micro'])
        return Kernel_θ1.Parameters(
            height = height,
            start  = 0,
            stop   = stop
        )

    def __init__(self, name, params=None, shape=None, **kwargs):
        kern_params = self.get_kernel_params(params)
        memory_time = shim.asarray(shim.get_test_value(kern_params.stop)
                                   - kern_params.start).max()
            # FIXME: At present, if we don't set memory_time now, tn is not set
            #        properly
        super().__init__(name, params, shape,
                         t0 = 0,
                         memory_time = memory_time,  # FIXME: memory_time should be optional
                         **kwargs)

    def _eval_f(self, t, from_idx=slice(None,None)):
        if shim.isscalar(t):
            return self.params.height
        else:
            # t has already been shaped to align with the function output in Kernel.eval
            return shim.ones(t.shape, dtype=shim.config.floatX) * self.params.height

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


    # # UGLY HACK: Copied function from ExpKernel and added 'expand'
    # def _eval_f(self, t, from_idx=slice(None,None)):
    #     if homo:
    #         return super()._eval_f(t, from_idx)
    #     else:
    #         return shim.switch(shim.lt(t, expand(self.params.t_offset[...,from_idx])),
    #                         0,
    #                         expand(self.params.height[...,from_idx]
    #                             * shim.exp(-(t-self.params.t_offset[...,from_idx])
    #                                     / self.params.decay_const[...,from_idx])) )

class GIF(models.Model):

    requires_rng = True
    # Entries to Parameter_info: ( 'parameter name',
    #                              (dtype, default value, shape_flag) )
    # If the shape_flag is True, the parameter will be reshaped into a 2d
    # matrix, if it isn't already. This is ensures the parameter is
    # consistent with kernel methods which assume inputs which are at least 2d
    # The last two options can be omitted; default flag is 'False'
    # Typically if a parameter will be used inside a kernel, shape_flag should be True.
    # NOTE: int32 * float32 = float64,  but  int16 * float32 = float32
    Parameter_info = OrderedDict( (( 'N',      'int16' ),     # Maximum value: 2^16 == 65536
                                   ( 'R',      'floatX' ),    # membrane resistance (
                                   ( 'u_rest', ('floatX', None, False) ),
                                   ( 'p',      'floatX' ),   # Connection probability between populations
                                   ( 'w',      'floatX' ),         # matrix of *population* connectivity strengths
                                   ( 'Γ',      'int8' ),               # binary connectivity between *neurons*
                                   ( 'τ_m',    'floatX'  ), # membrane time constant (s)
                                   ( 't_ref',  'floatX'  ), # absolute refractory period (s)
                                   ( 'u_th',   'floatX'  ),    # non-adapting threshold (mV)
                                   ( 'u_r',    'floatX'  ),    # reset potential (mV)
                                   ( 'c',      'floatX'  ),   # escape rate at threshold (Hz)
                                   ( 'Δu',     'floatX'  ),    # noise level (mV)
                                   ( 'Δ',      ('floatX', None, True)), # transmission delay (s) (kernel ε)
                                   ( 'τ_s',    ('floatX', None, True)), # synaptic time constant (mV) (kernel ε)
                                   # Adaptation parameters (θ-kernel dependent)
                                   ( 'J_θ',    ('floatX', None, False)), # Integral of adaptation (mV s)
                                   ( 'τ_θ',    ('floatX', None, False))
                                   ) )
        # NOTE: `Γ` is typically obtained by calling `make_connectivity` with `N` and `p`.
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['u', 't_hat', 's'])

    default_initializer = 'stationary'

    def __init__(self, params, spike_history, input_history,
                 initializer=None, set_weights=True, random_stream=None, memory_time=None):
        """
        Parameters
        ----------
        set_weights: bool, ndarray
            (Optional) Set to True to indicate that network connectivity should be set using the
            `w` and `Γ` parameters. If the spike history is already filled, set to False to
            avoid overwriting the connectivity. If an ndarray, that array will be used directly
            to set connectivity, ignoring model parameters. Default is True.
        """
        # FIXME
        if homo and initializer == 'stationary':
            raise NotImplementedError("Stationary initialization doesn't work with heterogeneous "
                                      "populations yet. Reason: "
                                      "`τmT = self.params.τ_m.flatten()[:, np.newaxis]` line")

        self.s = spike_history
        self.I_ext = input_history
        self.rndstream = random_stream
        # if not isinstance(self.s, PopulationHistory):
        #     raise ValueError("Spike history must be an instance of sinn.PopulationHistory.")
        if not isinstance(self.I_ext, Series):
            raise ValueError("External input history must be an instance of sinn.Series.")
        # This runs consistency tests on the parameters
        # models.Model.same_shape(self.s, self.I)
        models.Model.same_dt(self.s, self.I_ext)
        models.Model.output_rng(self.s, self.rndstream)

        super().__init__(params,
                         t0=self.s.t0, tn=self.s.tn, dt=self.s.dt,
                         public_histories=(spike_history, input_history),
                         reference_history=self.s)
        # NOTE: Do not use `params` beyond here. Always use self.params.
        N = self.params.N.get_value()
        assert(N.ndim == 1)
        self.Npops = len(N)

        self.original_params = self.params  # Store unexpanded params
        self.params = self.params._replace(
            **{name: self.expand_param(getattr(self.params, name), N)
            for name in params._fields
            if name != 'N'})

        # Set the connection weights
        if isinstance(set_weights, np.ndarray):
            self.s.set_connectivity(set_weights)
        elif set_weights:
            # TODO: If parameters were less hacky, w would already be properly
            #       cast as an array
            w = self.w * self.Γ
            self.s.set_connectivity(w)

        # Model variables
        self.RI_syn = Series(self.s, 'RI_syn',
                             shape = (N.sum(), ),
                             dtype = shim.config.floatX)
        self.λ = Series(self.RI_syn, 'λ', dtype=shim.config.floatX)
        self.varθ = Series(self.RI_syn, 'ϑ')
        self.u = Series(self.RI_syn, 'u')
        # Surrogate variables
        self.t_hat = Series(self.RI_syn, 't_hat')
            # time since last spike

        # self.statehists = [ getattr(self, varname) for varname in self.State._fields ]
        # Kernels
        # HACK: Because PopTerm doesn't support shared arrays
        if shim.is_theano_object(self.τ_s, self.Δ):
            shape2d = (sum(N), sum(N))
        else:
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

        #if self.s._original_tidx.get_value() < self.s.t0idx:
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
            θ_dis, θtilde_dis = self.discretize_θkernel(
                [self.θ1, self.θ2], self._refhist, self.params)
            init_A = self.get_stationary_activity(
                self, self.K, θ_dis, θtilde_dis)
            init_state = self.get_stationary_state(init_A)

        elif initializer == 'silent':
            init_A = np.zeros((len(self.s.pop_slices),))
            init_state = self.get_silent_latent_state()
        else:
            raise ValueError("Initializer string must be one of 'stationary', 'silent'")

        for varname in self.State._fields:
            hist = getattr(self, varname)
            if hist._original_tidx.get_value() < hist.t0idx:
                initval = getattr(init_state, varname)
                hist.pad(1)
                idx = hist.t0idx - 1; assert(idx >= 0)
                hist[idx] = initval

        # TODO: Combine the following into the loop above
        nbins = self.s.t0idx
        if self.s._original_tidx.get_value() < nbins:
            self.s[:nbins] = init_state.s
        #data = self.s._data
        #data[:nbins,:] = init_state.s
        #self.s._data.set_value(data, borrow=True)

    def get_silent_latent_state(self):
        state = self.State(
            u = self.params.u_rest,
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
        p = self.s.PopTerm(Astar) * self.dt
        nbins = self.s.t0idx
        nneurons = self.s.shape[0]
        s = np.random.binomial(1, p, (nbins, nneurons))
        # argmax returns first occurrence; by flipping s, we get the
        # index (from the end) of the last spike, i.e. number of bins - 1
        t_hat = (s[::-1].argmax(axis=0) + 1) * self.dt
        # u is initialized by integrating the ODE from the last spike
        # See documentation for details (TODO: not yet in docs)
        τmT = self.params.τ_m.flatten()[:, np.newaxis]
        η1 = τmT * self.params.p * self.params.N * self.params.w
            # As in mesoGIF.get_η_csts
        u = np.where(t_hat <= self.params.t_ref,
                     self.params.u_r,
                     ((1 - np.exp(-t_hat/self.params.τ_m)) * self.s.PopTerm( (self.params.u_rest + η1.dot(Astar)) )
                       + self.params.u_r * np.exp(-t_hat/self.params.τ_m))
                     )
        state = Stateplus(
            u = u,
            t_hat = t_hat,
            s = s
        )
        return state

    @staticmethod
    def make_connectivity(N, p):
        """
        Construct a binary neuron connectivity matrix, for use as this class'
        Γ parameter.

        Parameters
        ----------
        N: array of ints
            Number of neurons in each population
        p: 2d array of floats between 0 and 1
            Connection probability. If connection probabilities are symmetric,
            this array should also be symmetric.

        Returns
        -------
        2d binary matrix
        """
        # TODO: Use array to pre-allocate memory
        # TODO: Does it make sense to return a sparse matrix ? For low connectivity
        #       yes, but with the assumed all-to-all connectivity, we can't really
        #       have p < 0.2.
        Γrows = []
        for Nα, prow in zip(N, p):
            Γrows.append( np.concatenate([ np.random.binomial(1, pαβ, size=(Nα, Nβ))
                                        for Nβ, pαβ in zip(N, prow) ],
                                        axis = 1) )

        Γ = np.concatenate( Γrows, axis=0 )
        return Γ

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
        block_types = []
        for s in shim.get_test_value(param).shape:
            block_types.append('Macro' if s == 1
                               else 'Meso' if s == len(N)
                               else 'Micro' if s == sum(N)
                               else None)
        assert None not in block_types
        return sinn.popterm.expand_array(N, param, tuple(block_types))

        # Npops = len(N)
        # if param.ndim == 1:
        #     return shim.concatenate( [ param[i]*np.ones((N[i],))
        #                                for i in range(Npops) ] )
        #
        # elif param.ndim == 2:
        #     return shim.concatenate(
        #         [ shim.concatenate( [ param[i, j]* np.ones((N[i], N[j]))
        #                               for j in range(Npops) ],
        #                             axis = 1 )
        #           for i in range(Npops) ],
        #         axis = 0 )
        # else:
        #     raise ValueError("Parameter {} has {} dimensions; can only expand "
        #                      "dimensions of 1d and 2d parameters."
        #                      .format(param.name, param.ndim))

    @models.batch_function_scan('λ', 's')
    def logp(self, λ, s):
        # Function receives a slice of λ and s corresponding to the batch
        p = sinn.clip_probabilities(λ*self.s.dt)
        return ( s*p - (1-p) + s*(1-p) ).sum()  # sum over batch and neurons

    # def loglikelihood(self, start, batch_size, data=None, avg=False,
    #                   flags=()):
    #     # >>>>>>>>>>>>>> WARNING: Untested, incomplete <<<<<<<<<<<<<
    #
    #     #######################
    #     # Some hacks to get around current limitations
    #
    #     self.remove_other_histories()
    #
    #     # End hacks
    #     #####################
    #
    #     batch_size = self.index_interval(batch_size)
    #     startidx = self.get_t_idx(start)
    #     stopidx = startidx + batch_size
    #     N = self.params.N
    #     if data is None:
    #         n_full = self.n
    #         t0idx = self.n.t0idx
    #     else:
    #         n_full = data.astype(self.params.N.dtype)
    #         t0idx = 0 # No offset if we provide data
    #
    #     def logLstep(tidx, cum_logL):
    #         p = sinn.clip_probabilities(self.λ[tidx]*self.s.dt)
    #         s = shim.cast(self.s[tidx+self.s.t0idx], self.s.dtype)
    #
    #         # L = s*n - (1-s)*(1-p)
    #         cum_logL += ( s*p - (1-p) + s*(1-p) ).sum()
    #
    #         return [cum_logL], shim.get_updates()
    #
    #     if shim.is_theano_object([self.s, self.params]):
    #
    #         logger.info("Producing the likelihood graph.")
    #
    #         if batch_size == 1:
    #             # No need for scan
    #             logL, upds = logLstep(start, 0)
    #
    #         else:
    #             # FIXME np.float64 -> shim.floatX or sinn.floatX
    #             logL, upds = shim.gettheano().scan(logLstep,
    #                                             sequences = shim.getT().arange(startidx, stopidx),
    #                                             outputs_info = np.asarray(0, dtype=shim.config.floatX))
    #             self.apply_updates(upds)
    #                 # Applying updates is essential to remove the temporary iteration variable
    #                 # scan introduces from the shim updates dictionary
    #
    #         logger.info("Likelihood graph complete")
    #
    #         return logL[-1], upds
    #     else:
    #         # TODO: Remove this branch once shim.scan is implemented
    #         logL = 0
    #         for t in np.arange(startidx, stopidx):
    #             logL = logLstep(t, logL)[0][0]
    #         upds = shim.get_updates()
    #
    #         return logL, upds

    # FIXME: Before replacing with Model's `advance`, need to remove HACKs
    def advance(self, stop):

        if stop == 'end':
            stoptidx = self.tnidx
        else:
            stoptidx = self.get_t_idx(stop)

        # Make sure we don't go beyond given data
        for h in self.history_set:
            # HACK: Should exclude kernels
            if h.name in ['θ_dis', 'θtilde_dis']:
                continue
            if h.locked:
                tn = h.get_time(h._original_tidx.get_value())
                if tn < self._refhist.get_time(stoptidx):
                    logger.warning("Locked history '{}' is only provided "
                                   "up to t={}. Output will be truncated."
                                   .format(h.name, tn))
                    stoptidx = self.nbar.get_t_idx(tn)

        if not shim.config.use_theano:
            self._refhist[stoptidx - self.t0idx + self._refhist.t0idx]
            # We want to compute the whole model up to stoptidx, not just what is required for refhist
            for hist in self.statehists:
                hist.compute_up_to(stoptidx - self.t0idx + hist.t0idx)

        else:
            if not shim.graph.is_computable([hist._cur_tidx
                                       for hist in self.statehists]):
                raise TypeError("Advancing models is only implemented for "
                                "histories with a computable current time "
                                "index (i.e. the value of `hist._cur_tidx` "
                                "must only depend on symbolic constants and "
                                "shared vars).")
            curtidx = min( shim.graph.eval(hist._cur_tidx, max_cost=50)
                           - hist.t0idx + self.t0idx
                           for hist in self.statehists )
            assert(curtidx >= -1)

            if curtidx < stoptidx:
                self._advance(curtidx, stoptidx+1)
                # _advance applies the updates, so should get rid of them
                self.theano_reset()
    integrate = advance

    def f(self, u):
        """Link function. Maps difference between membrane potential & threshold
        to firing rate."""
        return self.params.c * shim.exp(u/self.params.Δu.flatten())


    def RI_syn_fn(self, t):
        """Incoming synaptic current times membrane resistance. Eq. (20)."""
        t_s = self.RI_syn.get_tidx_for(t, self.s)
        return ( self.params.τ_m * self.s.convolve(self.ε, t_s).sum(axis=-1) )


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
        return shim.switch( shim.ge(self.t_hat[t_that], self.params.t_ref),
                            self.u[tidx_u_m1] * red_factor
                            + ( (self.params.u_rest + self.params.R * self.I_ext[t_Iext])
                                + self.RI_syn[t_RIsyn] )
                              * (1 - red_factor),
                            self.params.u_r )

    def varθ_fn(self, t):
        """Dynamic threshold. Eq. (22)."""
        t_s = self.RI_syn.get_t_for(t, self.s)
        return (self.params.u_th + self.s.convolve(self.θ1, t_s)
                + self.s.convolve(self.θ2, t_s))
            # Need to fix spiketrain convolution before we can use the exponential
            # optimization. (see fixme comments in histories.spiketrain._convolve_single_t
            # and kernels.ExpKernel._convolve_single_t)

    def λ_fn(self, t):
        """Hazard rate. Eq. (23)."""
        # TODO: Use self.f here (requires overloading of ops to remove pop_rmul & co.)
        t_u = self.λ.get_t_for(t, self.u)
        t_varθ = self.λ.get_t_for(t, self.varθ)
        return (self.params.c * shim.exp(  (self.u[t_u] - self.varθ[t_varθ]) / self.params.Δu ) )

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
            cond = shim.eq(self.s[s_tidx_m1], 0)
            # If s is a sparse array, indexing doesn't reduce no. of dimensions
            # cond = cond[0]
        else:
            s_t = self.t_hat.get_t_for(t, self.s)
            s_tidx_m1 = self.s.array_to_slice(s_t, lag=-1)
            t_tidx_m1 = self.t_hat.array_to_slice(t, lag=-1)
            cond_tslice = slice(None)
            cond = shim.eq(self.s[s_tidx_m1], 0)
        # If the last bin was a spike, set the time to dt (time bin length)
        # Otherwise, add dt to the time
        return shim.switch( cond,
                            self.t_hat[t_tidx_m1] + self.t_hat.dt,
                            self.t_hat.dt )

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
            if θ_dis._original_data in shim.config.theano_updates:
                del shim.config.theano_updates[θ_dis._original_data]
            if θ_dis._original_tidx in shim.config.theano_updates:
                del shim.config.theano_updates[θ_dis._original_tidx]

        # TODO: Use operations
        θtilde_dis = Series(θ_dis, 'θtilde_dis', iterative=False)
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
            if θtilde_dis._original_data in shim.config.theano_updates:
                del shim.config.theano_updates[θtilde_dis._original_data]
            if θtilde_dis._original_tidx in shim.config.theano_updates:
                del shim.config.theano_updates[θtilde_dis._original_tidx]

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
        model: An instance of either GIF or mesoGIF
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

    def symbolic_update(self, tidx, u0, t_hat0):
        # Argument order is set by self.State
        # Only include unlocked histories

        tidx = tidx - self.t0idx
        t_s = self.get_tidx_for(tidx, self.s)
        t_Iext = self.get_tidx_for(tidx, self.I_ext)

        assert self.s.locked
        RI_syn = self.τ_m * self.s.convolve(self.ε, t_s).sum(axis=1)
            # This expression also appears nonstate_symbolic_update, but
            # Theano should recognize this and merge the expressions

        cond = shim.eq(self.s[t_s-1], 0)
        t_hat = shim.switch( cond,
                             t_hat0 + self.t_hat.dt,
                             self.t_hat.dt )

        red_factor = shim.exp(-self.u.dt/self.params.τ_m)
        u = shim.switch( shim.ge(t_hat, self.t_ref),
                         u0 * red_factor
                         + ( (self.u_rest + self.R * self.I_ext[t_Iext])
                             + RI_syn )
                           * (1 - red_factor),
                         self.u_r )

        # Same order as function signature
        state_outputs = [u, t_hat]
        updates = {}
        return state_outputs, updates

    def nonstate_symbolic_update(self, tidx, hists, curstate, curnonstate, newstate):
        """
        Arguments that the calling function will pass:
        hists: list(self.unlocked_nonstatehistories)
        curstate: symbolic variables corresponding to state histories
        curnonstate: symbolic variables corresponding to histories in hists
        newstate: `state_output` from `symbolic_update()`
        """
        tidx = tidx - self.t0idx

        u0, t_hat0 = curstate
        u, t_hat = newstate
        # Use hists to figure out the order of arguments
        # Ugly but functional
        RIsyn_idx = hists.index(self.RI_syn) if self.RI_syn in hists else None
        varθ_idx = hists.index(self.varθ) if self.varθ in hists else None
        λ_idx = hists.index(self.λ) if self.λ in hists else None

        # No need to unpack curnonstate because we don't use it
        # λ0 = curnonstate[λ_idx] ...

        assert self.s.locked
        t_s = self.get_tidx_for(tidx, self.s)
        RI_syn = self.τ_m * self.s.convolve(self.ε, t_s).sum(axis=1)
        varθ = (self.u_th + self.s.convolve(self.θ1, t_s)
                + self.s.convolve(self.θ2, t_s))
        λ = self.params.c * shim.exp(  (u - varθ)/self.Δu )

        nonstate_outputs = [None]*len(hists)
        # Uglier but still functional
        if RIsyn_idx is not None: nonstate_outputs[RIsyn_idx] = RI_syn
        if varθ_idx is not None: nonstate_outputs[varθ_idx] = varθ
        if λ_idx is not None: nonstate_outputs[λ_idx] = λ
        assert None not in nonstate_outputs
        updates = {}
        return nonstate_outputs, updates







models.register_model(GIF)
