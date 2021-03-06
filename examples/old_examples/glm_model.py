# -*- coding: utf-8 -*-
"""
Created Fri Mar 10 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict
import logging

import theano_shim as shim
import sinn
import sinn.config as config
import sinn.histories as histories
import sinn.kernels as kernels
import sinn.models as models

logger = logging.getLogger("sinn.models.glm")

Model = models.Model
ModelKernelMixin = models.ModelKernelMixin
Kernel = kernels.Kernel

# This seems to be a bug in dill, that K can't be a member of the model class
class ExpK(ModelKernelMixin, kernels.ExpKernel):
    @staticmethod
    def get_kernel_params(model_params):
        return kernels.ExpKernel.Parameters(
            height = 1,
            decay_const = model_params.τ,
            t_offset = 0)

class GLM_exp_kernel(Model):

    # Entries to Parameter_info: ( 'parameter name',
    #                              (dtype, default value, shape_flag) )
    # If the shape_flag is True, the parameter will be reshaped into a 2d
    # matrix, if it isn't already. This is ensures the parameter is
    # consistent with kernel methods which assume inputs which are at least 2d
    # The last two options can be omitted; default flag is 'False'
    # Typically if a parameter will be used inside a kernel, shape_flag should be True.
    Parameter_info = OrderedDict( ( ( 'N', 'int32' ),
                                    ( 'c', config.floatX ),
                                    ( 'J', (config.floatX, None, True) ),
                                    ( 'τ', (config.floatX, None, True) )) )
    Parameters = sinn.define_parameters(Parameter_info)

    def __init__(self, params, activity_history, input_history,
                 random_stream=None, memory_time=None):
        self.A = activity_history
        self.I = input_history
        self.rndstream = random_stream
        # This runs consistency tests on the parameters
        Model.same_shape(self.A, self.I)
        Model.same_dt(self.A, self.I)
        Model.output_rng(self.A, self.rndstream)

        super().__init__(params)
        # NOTE: Do not use `params` beyond here. Always use self.params.

        self.ρ = histories.Series(self.A, "ρ") # track hazard function

        # Compute Js*A+I before convolving, so we only convolve once
        # (We shamelessly abuse of unicode support for legibility)
        self.JᕽAᐩI = histories.Series(self.A, "JᕽAᐩI",
                                      shape = self.A.shape,
                                      f = self.JᕽAᐩI_fn)

        self.add_history(self.A)
        self.add_history(self.I)
        self.add_history(self.ρ)
        self.add_history(self.JᕽAᐩI)

        self.A.add_inputs([self.ρ])
        self.ρ.add_inputs([self.JᕽAᐩI])
        self.JᕽAᐩI.add_inputs([self.A, self.I])

        self.ρ.set_update_function(self.ρ_fn)
        self.A.set_update_function(self.A_fn)

        κshape = self.params.N.get_value().shape
        self.κ = ExpK('κ', self.params, κshape, memory_time=memory_time)
        self.add_kernel(self.κ)

        self.JᕽAᐩI.pad(self.κ.memory_time)

        # HACK Should only add dependencies from histories in history_inputs
        self.history_inputs.union(sinn.inputs)

    def JᕽAᐩI_fn(self, t):
        assert(len(self.A.shape) == 1) # Too lazy to make the more generic case
        if shim.isscalar(t):
            return shim.dot(self.params.J, self.A[t]) + self.I[t]
                                      # NxN  dot  N   +  N
        else:
            # Distribute the dot product along the time axis
            J = shim.add_axes(self.params.J, 1, 'before')
            if J.ndim == 3:
                return shim.sum(J * shim.add_axes(self.A[t], 1, pos='before last'), axis=-1) + self.I[t]
            else:
                return J*self.A[t] + self.I[t]

    def ρ_fn(self, t):
        if not shim.isscalar(t):
            tslice = self.JᕽAᐩI.time_array_to_slice(t)
            # assert(t.ndim == 1)  # No support for multiple time slices at the moment
            # tslice = shim.ifelse( shim.eq(t.shape[0], 1),
            #                       t[0], # t is just a singe time, wrapped in an arra
            #                       self.JᕽAᐩI.time_array_to_slice(t) )
            #                           # t is an array of times, but convolve wants a slice
        else:
            tslice = t
        return self.params.c * shim.exp(self.κ.convolve(self.JᕽAᐩI, tslice))

    def A_fn(self, t):
        p_arr = sinn.clip_probabilities(self.ρ[t] * self.A.dt)
        return self.rndstream.binomial( size = self.A.shape,
                                        n = self.params.N,
                                        p = p_arr ) / self.params.N / self.A.dt

    def A_range_fn(self, t_array):
        shim.check( t_array[0] < t_array[-1] )
            # We don't want to check that the entire array is ordered; this is a compromise
        if not shim.is_theano_object(self.A._sym_tidx):
            shim.check( t_array[0] == self.A._sym_tidx.get_value() + 1 )

        if not shim.is_theano_object(self.A._data):
            def loop(t):
                res = self.A_fn(t)
                # If you have any shared variables, update them here
                return res

            A_lst = [loop(t) for t in t_array]
            return A_lst

        else:
            res, upds = theano.scan(self.A_onestep,
                                    sequences = t_array,
                                    non_sequences = [self.JᕽAᐩI],
                                    name = 'A scan')


    def update_params(self, new_params):
        if np.all(new_params.J == self.params.J):
            self.JᕽAᐩI.locked = True
        else:
            self.JᕽAᐩI.locked = False
        super().update_params(new_params)

    def get_t_idx(self, t):
        """
        Returns the time index corresponding to t such that it is compatible
        with loglikelihood.
        """
        return self.A.get_t_idx(t)
    def index_interval(self, Δt):
        return self.A.index_interval(Δt)

    def loglikelihood(self, start=None, stop=None):

        hist_type_msg = ("To compute the loglikelihood, you need to use a NumPy "
                         "history for the {}, or compile the history beforehand.")
        if self.A.use_theano:
            if self.A.compiled_history is None:
                raise RuntimeError(hist_type_msg.format("activity"))
            else:
                Ahist = self.A.compiled_history
        else:
            Ahist = self.A

        ρhist = self.ρ
        if not ρhist.use_theano:
            ρhist.zero('all')  # This is not necessary (ρhist is already cleared)
                               # but it makes the _data object match what Theano
                               # graph receives as input.

        # We deliberately use times here (instead of indices) for start/
        # stop so that they remain consistent across different histories
        if start is None:
            start = Ahist.t0
        else:
            start = Ahist.get_time(start)
        if stop is None:
            stop = Ahist.tn
        else:
            stop = Ahist.get_time(stop)

        A_arr = Ahist[start:stop]

        # Binomial mean: Np = Nρdt.  E(A) = E(B)/N/dt = ρ
        ρ_arr = ρhist[start:stop]

        #------------------
        # True log-likelihood
        # Number of spikes
        k_arr = (A_arr * Ahist.dt * self.params.N).astype('int16')
        # Spiking probabilities
        p_arr = sinn.clip_probabilities(ρ_arr * Ahist.dt)

        # loglikelihood: -log k! - log (N-k)! + k log p + (N-k) log (1-p) + cst
        # We use the Stirling approximation for the second log
        l = shim.sum( -shim.log(shim.factorial(k_arr, exact=True))
                      -(self.params.N-k_arr)*shim.log(self.params.N - k_arr)
                      + self.params.N-k_arr
                      + k_arr*shim.log(p_arr)
                      + (self.params.N-k_arr)*shim.log(1-p_arr) )
            # with exact=True, factorial is computed only once for whole array

        return l

    def get_loglikelihood(self, *args, **kwargs):

        # Sanity check – it's easy to forget to clear histories in an interactive session
        uncleared_histories = []
        # HACK Shouldn't need to combine sinn.inputs
        # TODO Make separate function, so that it can be called within loglikelihood instead
        for hist in self.history_inputs.union(sinn.inputs):
            if ( not hist.locked and ( ( hist.use_theano and hist.compiled_history is not None
                                         and hist.compiled_history._sym_tidx.get_value() >= hist.t0idx )
                                       or (not hist.use_theano and hist._sym_tidx.get_value() >= hist.t0idx) ) ):
                uncleared_histories.append(hist)
        if len(uncleared_histories) > 0:
            raise RuntimeError("You are trying to produce a cost function graph, but have "
                               "uncleared histories. Either lock them (with their .lock() "
                               "method) or clear them (with their individual .clear() method "
                               "or the model's .clear_unlocked_histories() method). The latter "
                               "will delete data.\nUncleared histories: "
                               + str([hist.name for hist in uncleared_histories]))

        if sinn.config.use_theano():
            # TODO Precompile function
            def likelihood_f(model):
                if 'loglikelihood' not in self.compiled:
                    import theano
                    self.theano_reset()
                        # Make clean slate (in particular, clear the list of inputs)
                    logL = model.loglikelihood(*args, **kwargs)
                        # Calling logL sets the sinn.inputs, which we need
                        # before calling get_input_list
                    # DEBUG
                    # with open("logL_graph", 'w') as f:
                    #     theano.printing.debugprint(logL, file=f)
                    input_list, input_vals = self.get_input_list()
                    self.compiled['loglikelihood'] = {
                        'function': theano.function(input_list, logL,
                                                    on_unused_input='warn'),
                        'inputs'  : input_vals }
                    self.theano_reset()

                return self.compiled['loglikelihood']['function'](
                    *self.compiled['loglikelihood']['inputs'] )
                    # * is there to expand the list of inputs
        else:
            def likelihood_f(model):
                return model.loglikelihood(*args, **kwargs)
        return likelihood_f

    def get_input_list(self):
        # TODO: move to Models
        # TODO: get rid of entirely and use theano.gof.graph.inputs
        input_list = []
        input_vals = []
        # HACK Shouldn't need to combine sinn.inputs
        for hist in self.history_inputs.union(sinn.inputs):
            if shim.is_theano_variable(hist._num_data):
                shape = hist._tarr.shape + hist.shape
                if hist._num_data is not None:
                    # The graph triggered an update of the variable. The input to the
                    # function remains the original variable.
                    input_list.append(hist._num_data)
                else:
                    input_list.append(hist._data)
                input_vals.append(np.zeros(shape, dtype=sinn.config.floatX))
        return input_list, input_vals
