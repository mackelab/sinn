# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 2017

Author: Alexandre René
"""

import numpy as np
import scipy as sp
#from scipy.integrate import quad
#from collections import namedtuple
import logging
logger = logging.getLogger("sinn.models.common")
from collections import OrderedDict, Sequence
from inspect import isclass

import theano_shim as shim
import sinn.config as config
import sinn.common as com
import sinn.histories
import sinn.kernels
import sinn.diskcache as diskcache

_models = {}
registered_models = _models.keys()
    # I don't really like this, but it works. Ideally it would be some kind
    # of read-only property of the module.

def register_model(model, modelname=None):
    """
    Register a subclass of Model.
    Typically this is called from the module which implements the subclass.
    If `modelname` is unspecified, `model`'s class name is used.
    """
    global _models
    assert(isclass(model))
    if modelname is None:
        modelname = model.__name__
    assert(isinstance(modelname, str))
    _models[modelname] = model

def is_registered(modelname):
    """Returns True if a model is registered with this name."""
    global _models
    return modelname in _models

def get_model(modelname, *args, **kwargs):
    """Retrieves the model associated with the model name. Same arguments as dict.get()."""
    global _models
    return _models.get(modelname, *args, **kwargs)

class Model(com.ParameterMixin):
    """Abstract model class.

    A model implementations should derive from this class.
    It must minimally provide:
    - A `Parameter_info` dictionary of the form:
        (See sinn.common.Parameterize)
        ```
        Parameter_info = OrderedDict{ 'param_name': Parameter([cast function], [default value]),
                                      ... }
        ```
    - A class-level (outside any method) call
        `Parameters = com.define_parameters(Parameter_info)`

    Implementations may also provide class methods to aid inference:
    - likelihood: (params) -> float
    - likelihood_gradient: (params) -> vector
    If not provided, `likelihood_gradient` will be calculated by appyling theano's
    grad method to `likelihood`. (TODO)
    As class methods, these don't require an instance – they can be called on the class directly.
    """

    def __init__(self, params, public_histories, reference_history=None):
        # History is optional because more complex models have multiple histories.
        # They should keep track of them themselves.
        # ParameterMixin requires params as a keyword parameter
        """
        Parameters
        ----------
        params: self.Parameters instance

        public_histories: Ordered iterable
            List of histories that were explicitly passed to the model
            initializer. This allows to recover the histories that were passed
            (as usually they contain the desired output) without needing to
            know their internal name.

        reference_history: History instance
            If provided, a reference is kept to this history: Model evaluation
            may require querying some of its attributes, such as time step (dt).
            These attributes are allowed not to be constant.
            If no reference is given, the first history in `public_histories`
            is used.
        """
        # Format checks
        if not hasattr(self, 'requires_rng'):
            raise SyntaxError("Models require a `requires_rng` bool attribute.")

        super().__init__(params=params)
        self.kernel_list = []
        self.history_set = set()
        self.history_inputs = sinn.DependencyGraph('model.history_inputs')
        self.compiled = {}  # DEPRECATED

        if not isinstance(public_histories, (Sequence, OrderedDict)):
            raise TypeError("`public_histories` (type: {}) must be an ordered "
                            "container type. Recognized types: Sequence, "
                            "OrderedDict.".format(type(public_histories)))
        self.public_histories = public_histories

        if reference_history is not None:
            self._refhist = reference_history
            #self.add_history(reference_history)  # TODO: Possible to remove ?
        elif len(public_histories) > 0:
            self._refhist = self.public_histories[0]
        else:
            self._refhist = None

    def set_reference_history(self, reference_history):
        if self._refhist is None:
            raise RuntimeError("Reference history for this model is already set.")
        self._refhist = reference_history

    # TODO: Put the `if self._refhist is not None:` bit in a function decorator
    @property
    def t0(self):
        return self._refhist.t0

    @property
    def tn(self):
        return self._refhist.tn

    @property
    def t0idx(self):
        return 0
        #return self._refhist.t0idx

    @property
    def tnidx(self):
        return self._refhist.tnidx - self._refhist.t0idx + self.t0idx

    @property
    def tidx_dtype(self):
        return self._refhist.tidx_dtype

    @property
    def dt(self):
        if self._refhist is not None:
            return self._refhist.dt
        else:
            raise AttributeError("The reference history for this model was not set.")

    @property
    def cur_tidx(self):
        if self._refhist is not None:
            return self._refhist._cur_tidx - self._refhist.t0idx + self.t0idx
        else:
            raise AttributeError("The reference history for this model was not set.")

    def get_tidx(self, t, allow_rounding=False):
        """
        Returns the time index corresponding to t, with 0 corresponding to t0.
        """
        if self._refhist is not None:
            if shim.istype(t, 'int'):
                return t
            else:
                return self._refhist.get_tidx(t, allow_rounding) - self._refhist.t0idx + self.t0idx
        else:
            raise AttributeError("The reference history for this model was not set.")
    get_t_idx = get_tidx

    def index_interval(self, Δt, allow_rounding=False):
        if self._refhist is not None:
            return self._refhist.index_interval(Δt, allow_rounding)
        else:
            raise AttributeError("The reference history for this model was not set.")

    def get_time(self, t):
        if self._refhist is not None:
            if shim.istype(t, 'float'):
                return t
            else:
                assert(shim.istype(t, 'int'))
                tidx = t - self.t0idx + self._refhist.t0idx
                return self._refhist.get_time(tidx)
        else:
            raise AttributeError("The reference history for this model was not set.")

    # Simple consistency check functions
    @staticmethod
    def same_shape(*args):
        assert(all(arg1.shape == arg2.shape for arg1, arg2 in zip(args[:-1], args[1:])))
    @staticmethod
    def same_dt(*args):
        assert(all(sinn.isclose(arg1.dt, arg2.dt) for arg1, arg2 in zip(args[:-1], args[1:])))
    @staticmethod
    def output_rng(outputs, rngs):
        """
        Parameters
        ----------
        outputs: History
            Can also be a list of Histories
        rngs: random stream, or list of random streams
            The random stream(s) required to generate the histories in
            `outputs`
        """
        if isinstance(outputs, sinn.histories.History):
            outputs = [outputs]
        else:
            assert(all(isinstance(output, sinn.histories.History) for output in outputs))
        try:
            len(rngs)
        except TypeError:
            rngs = [rngs]

        if any( not shim.isshared(outhist._data) for outhist in outputs ):
            # Bypass test for Theano data
            return

        if ( any( outhist._cur_tidx.get_value() < len(outhist) - 1 for outhist in outputs )
             and any( rng is None for rng in rngs)
             and not all( outhist.locked for outhist in outputs ) ) :
            raise ValueError("Cannot generate {} without the required random number generator(s).".format(str([outhist.name for outhist in outputs])))
        elif ( all( outhist._cur_tidx.get_value() >= len(outhist) - 1 for outhist in outputs )
             and any( rng is not None for rng in rngs) ) :
            logger.warning("Your random number generator(s) will be unused, "
                           "since your data is already generated.")

    def cache(self, obj):
        """
        Call this function on all Kernel and History objects that should be
        saved to the disk cache.
        This function is cheap to call: the object is only written out when
        it's removed from program memory.
        """

        if isinstance(obj, sinn.kernels.Kernel):
            logger.warning("Deprecated. Use add_kernel instead.")
            self.kernel_list.append(obj)
        else:
            assert(isinstance(obj, sinn.histories.History))
            logger.warning("Histories aren't written to disk. Use add_history instead")
            self.history_inputs.add(obj)

    def add_history(self, hist):
        assert(isinstance(hist, sinn.histories.History))
        self.history_set.add(hist)
        if hist not in self.history_inputs:
            self.history_inputs.add(hist)
    def add_kernel(self, kernel):
        assert(isinstance(kernel, sinn.kernels.Kernel))
        if kernel not in self.kernel_list:
            self.kernel_list.append(kernel)

    def theano_reset(self):
        """Put model back into a clean state, to allow building a new Theano graph."""
        for hist in self.history_inputs:
            if not hist.locked:
                hist.theano_reset()
        for kernel in self.kernel_list:
            kernel.theano_reset()

        sinn.theano_reset() # theano_reset on histories will be called twice,
                            # but there's not much harm

    def update_params(self, new_params):
        """
        Update model parameters. Clears all histories except those whose `locked`
        attribute is True, as well as any kernel which depends on these parameters.

        TODO: Make `new_params` a dict and just update parameters in the dict.

        Parameters
        ----------
        new_params: same type as model.params | dict
            # TODO: Allow also **kwargs
        """
        def gettype(param):
            return type(param.get_value()) if shim.isshared(param) else type(param)
        if isinstance(new_params, self.Parameters):
            assert(all( gettype(param) == gettype(new_param)
                        for param, new_param in zip(self.params, new_params) ))
        elif isinstance(new_params, dict):
            assert(all( gettype(val) == gettype(getattr(self.params, name))
                        for name, val in new_params.items() ))
        else:
            raise NotImplementedError
        logger.monitor("Model params are now {}. Updating kernels...".format(self.params))

        # HACK Make sure sinn.inputs and models.history_inputs coincide
        sinn.inputs.union(self.history_inputs)
        self.history_inputs.union(sinn.inputs)

        # Determine the kernels for which parameters have changed
        kernels_to_update = []
        if isinstance(new_params, self.Parameters):
            for kernel in self.kernel_list:
                if not sinn.params_are_equal(
                        kernel.get_parameter_subset(new_params), kernel.params):
                    # Grab the subset of the new parameters relevant to this kernel,
                    # and compare to the kernel's current parameters. If any of
                    # them differ, add the kernel to the list of kernels to update.
                    kernels_to_update.append(kernel)
        else:
            assert(isinstance(new_params, dict))
            for kernel in self.kernel_list:
                if any(param_name in kernel.Parameters._fields
                       for param_name in new_params):
                    kernels_to_update.append(kernel)

        # Now update parameters. This must be done after the check above,
        # because Theano parameters automatically propagate to the kernels.
        sinn.set_parameters(self.params, new_params)

        # Loop over the list of kernels whose parameters have changed to do
        # two things:
        # - Remove any cached binary op that involves this kernel.
        #   (And write it to disk for later retrievel if these parameters
        #    are reused.)
        # - Update the kernel itself to the new parameters.
        for obj in list(self.history_inputs) + self.kernel_list:
            if obj not in kernels_to_update:
                for op in obj.cached_ops:
                    for kernel in kernels_to_update:
                        if hash(kernel) in op.cache:
                            diskcache.save(op.cache[hash(kernel)])
                            # TODO subclass op[other] and define __hash__
                            logger.monitor("Removing cache for binary op {} ({},{}) from heap."
                                        .format(str(op), obj.name, kernel.name))
                            del op.cache[hash(kernel)]

        for kernel in kernels_to_update:
            diskcache.save(kernel)
            logger.monitor("Updating kernel {}.".format(kernel.name))
            kernel.update_params(self.params)

        self.clear_unlocked_histories()

    def clear_unlocked_histories(self):
        """Clear all histories that have not been explicitly locked."""
        #for hist in self.history_inputs.union(sinn.inputs):
        for hist in self.history_inputs:
            # HACK: Removal of sinn.inputs is a more drastic version attempt
            #       at correcting the same problem as fsgif.remove_other_histories
            if not hist.locked:
                self.clear_history(hist)

    def clear_other_histories(self):
        """
        Clear unlocked histories that are not explicitly part of this model
        (but may be inputs).
        """
        # Implemented as a wrapper around clear_unlocked_histories:
        # first lock of this model's histories, clear histories, and then
        # revert to the original locked/unlocked status
        old_status = {hist: hist.locked for hist in self.history_set}
        for hist in self.history_set:
            if not hist.locked:
                hist.lock(warn=False)
        self.clear_unlocked_histories()
        for hist, status in old_status.items():
            if status == False:
                hist.unlock()

    def clear_history(self, history):
        # Clear the history, and remove any cached operations related to it
        # In contrast to `update_params`, we don't write these operations to
        # disk, because histories are data structures: there's no way of knowing
        # if they're equivalent to some already computed case other than comparing
        # the entire data.
        logger.monitor("Clearing history " + history.name)
        history.clear()
        #if history in self.history_inputs.union(sinn.inputs):
        if history in self.history_inputs:
            # HACK: Removal of sinn.inputs is a more drastic version attempt
            #       at correcting the same problem as fsgif.remove_other_histories
            for obj in list(self.history_inputs) + self.kernel_list:
                for op in obj.cached_ops:
                    if hash(history) in op.cache:
                        del op.cache[hash(history)]
        else:
            for obj in list(self.history_inputs) + self.kernel_list:
                for op in obj.cached_ops:
                    if hash(history) in op.cache:
                        logger.error("Uncached history {} is member of cached "
                                     "op {}. This may indicate a memory leak."
                                     .format(history.name, str(op)))

    def remove_other_histories(self):
        """HACK: Remove histories from sinn.inputs that are not in this model.
        Can remove this once we store dependencies in histories rather than in
        sinn.inputs."""
        histnames = [h.name for h in self.history_set]
        dellist = []
        for h in sinn.inputs:
            if h.name not in histnames:
                dellist.append(h)
        for h in dellist:
            del sinn.inputs[h]

    def apply_updates(self, update_dict):
        """
        Theano functions which produce updates (like scan) naturally will not
        update the history data structures. This method applies those updates
        by replacing the internal _data and _cur_tidx attributes of the history
        with the symbolic expression of the updates, allowing histories to be
        used in subsequent calculations.
        """
        # Update the history data
        for history in self.history_set:
            if history._original_tidx in update_dict:
                assert(history._original_data in update_dict)
                    # If you are changing tidx, then surely you must change _data as well
                history._cur_tidx = update_dict[history._original_tidx]
                history._data = update_dict[history._original_data]
            elif history._original_data in update_dict:
                history._data = update_dict[history._original_data]

        # Update the shim update dictionary
        shim.config.theano_updates.update(update_dict)

    def get_loglikelihood(self, *args, **kwargs):

        # Sanity check – it's easy to forget to clear histories in an interactive session
        uncleared_histories = []
        # HACK Shouldn't need to combine sinn.inputs
        # TODO Make separate function, so that it can be called within loglikelihood instead
        for hist in self.history_inputs.union(sinn.inputs):
            if ( not hist.locked and ( ( hist.use_theano and hist.compiled_history is not None
                                         and hist.compiled_history._cur_tidx.get_value() >= hist.t0idx )
                                       or (not hist.use_theano and hist._cur_tidx.get_value() >= hist.t0idx) ) ):
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

    def make_binomial_loglikelihood(self, n, N, p, approx=None):
        """
        Parameters
        ----------
        n: History
            Number of successful samples
        N: array of ints:
            Total number of samples.
            Must have n.shape == N.shape
        p: History
            Probability of success; first dimension is time.
            Must have n.shape == p.shape and len(n) == len(p).
        approx: str
            (Optional) If specified, one of:
            - 'low p': The probability is always very low; the Stirling approximation
              is used for the contribution from (1-p) to ensure numerical stability
            - 'high p': The probability is always very high; the Stirling approximation
              is used for the contribution from (p) to ensure numerical stability
            - 'Stirling': Use the Stirling approximation for both the contribution from
              (p) and (1-p) to ensure numerical stability.
            - 'low n' or `None`: Don't use any approximation. Make sure `n` is really low
              (n < 20) to avoid numerical issues.
        """

        def loglikelihood(start=None, stop=None):

            hist_type_msg = ("To compute the loglikelihood, you need to use a NumPy "
                             "history for the {}, or compile the history beforehand.")
            if n.use_theano:
                if n.compiled_history is None:
                    raise RuntimeError(hist_type_msg.format("events"))
                else:
                    nhist = n.compiled_history
            else:
                nhist = n

            phist = p
            # We deliberately use times here (instead of indices) for start/
            # stop so that they remain consistent across different histories
            if start is None:
                start = nhist.t0
            else:
                start = nhist.get_time(start)
            if stop is None:
                stop = nhist.tn
            else:
                stop = nhist.get_time(stop)

            n_arr_floats = nhist[start:stop]
            p_arr = phist[start:stop]

            # FIXME: This would break the Theano graph, no ?
            if shim.isshared(n_arr_floats):
                n_arr_floats = n_arr_floats.get_value()
            if shim.isshared(p_arr):
                p_arr = p_arr.get_value()

            p_arr = sinn.clip_probabilities(p_arr)

            if not shim.is_theano_object(n_arr_floats):
                assert(sinn.ismultiple(n_arr_floats, 1).all())
            n_arr = shim.cast(n_arr_floats, 'int32')

            #loglikelihood: -log n! - log (N-n)! + n log p + (N-n) log (1-p) + cst

            if approx == 'low p':
                # We use the Stirling approximation for the second log
                l = shim.sum( -shim.log(shim.factorial(n_arr, exact=False))
                              -(N-n_arr)*shim.log(N - n_arr) + N-n_arr + n_arr*shim.log(p_arr)
                              + (N-n_arr)*shim.log(1-p_arr) )
                    # with exact=True, factorial is computed only once for whole array
                    # but n_arr must not contain any elements greater than 20, as
                    # 21! > int64 (NumPy is then forced to cast to 'object', which
                    # does not play nice with numerical ops)
            else:
                raise NotImplementedError

            return l

        return loglikelihood

    # ==============================================
    # Model advancing code
    #
    # This code isn't 100% generic yet;
    # look for TODO tags for model-specific hacks
    #
    # Function overview:
    # - advance(self, stop): User-facing function
    # - _advance(self): Returns a function; use as `self._advance(stop)`:
    #   `self._advance` is a property which memoizes the compiled function.
    # - compile_advance_function(self): Function called by `_advance` the first
    #   time to do the compilation. Could conceivably also be used by a user.
    #   Returns a compiled function.
    # - advance_updates(self, stopidx): Function used by
    #   `compile_advance_function` to retrieve the set of symbolic updates.
    # ==============================================
    def advance(self, stop):
        """
        Allows advancing (or progagating forward, or integrating) a symbolic
        model.
        For a non-symbolic model the usual recursion is used – it's the
        same as calling `hist[stop]` on each history in the model.
        For a symbolic model, the function constructs the symbolic update
        function, compiles it, and than evaluates it with `stop` as argument.
        The update function is compiled only once, so subsequent calls to
        `advance` are much faster and benefit from the acceleration of running
        on compiled code.

        Parameters
        ----------
        stop: int, float
            Compute history up to this point (inclusive).
        """

        # TODO: Rename stopidx -> endidx
        if stop == 'end':
            stopidx = self.tnidx
        else:
            stopidx = self.get_tidx(stop)

        # Make sure we don't go beyond given data
        for hist in self.history_set:
            if hist.locked:
                tnidx = hist._original_tidx.get_value()
                if tnidx < stopidx - self.t0idx + hist.t0idx:
                    logger.warning("Locked history '{}' is only provided "
                                   "up to t={}. Output will be truncated."
                                   .format(hist.name, hist.get_time(tnidx)))
                    stopidx = tnidx - hist.t0idx + self.t0idx

        if not shim.config.use_theano:
            self._refhist[stopidx - self.t0idx + self._refhist.t0idx]
            # We want to compute the whole model up to stopidx, not just what is required for refhist
            for hist in self.statehists:
                hist.compute_up_to(stopidx - self.t0idx + hist.t0idx)

        else:
            curtidx = min( hist._original_tidx.get_value() - hist.t0idx + self.t0idx
                           for hist in self.statehists )
            assert(curtidx >= -1)

            if curtidx < stopidx:
                self._advance(stopidx+1)
                for hist in self.statehists:
                    hist.theano_reset()

    @property
    def _advance(self):
        """
        Attribute which caches the compilation of the advance function.
        """
        if not hasattr(self, '_advance_fn'):
            self._advance_fn = self.compile_advance_function()
        return self._advance_fn

    def compile_advance_function(self):
        stopidx_var = shim.getT().scalar('stopidx (model)',
                                         dtype=self.tidx_dtype)
        stopidx_var.tag.test_value = 2
            # Allow model to work with compute_test_value != 'ignore'
        logger.info("Compiling advance function.")
        updates = self.advance_updates(stopidx_var)
        fn = shim.graph.compile([stopidx_var], [], updates = updates)
        logger.info("Done.")
        self.theano_reset()
        return fn

    def advance_updates(self, stopidx):
        """

        Parameters
        ----------
        stopidx: symbolic (int)
            We want to compute the model up to this point.

        Returns
        -------
        Update dictionary:
            Compiling a function and providing this dictionary as 'updates' will return a function
            which fills in the histories up to `stopidx`.
        """
        self.remove_other_histories()  # HACK
        # self.clear_unlocked_histories()
        self.theano_reset()
        if not all(np.can_cast(stopidx.dtype, hist.tidx_dtype)
                   for hist in self.statehists):
            raise TypeError("`stopidx` cannot be safely cast to a time index. "
                            "This can happen if e.g. a history uses `int32` for "
                            "its time indices while `stopidx` is `int64`.")

        if len(self.statehists) == 0:
            raise NotImplementedError
        elif len(self.statehists) == 1:
            hist = self.statehists[0]
            startidx = hist._original_tidx - hist.t0idx + self.t0idx
        else:
            startidx = shim.smallest( *( hist._original_tidx - hist.t0idx + self.t0idx
                                        for hist in self.statehists ) )
        try:
            assert( shim.get_test_value(startidx) >= -1 )
                # Iteration starts at startidx + 1, and will break for indices < 0
        except AttributeError:
            # Unable to find test value; just skip check
            pass

        # TODO: Make `symbolic_update` a generic function that constructs
        # symbolic updates from the history update functions
        def onestep(tidx, *args):
            state_outputs, updates = self.symbolic_update(tidx, *args)
            for i in range(len(state_outputs)):
                state_outputs[i] = shim.cast(state_outputs[i],
                                             self.statehists[i].dtype)
            return state_outputs, updates
            #return list(state_outputs.values()), updates

        outputs_info = []
        for hist in self.statehists:
            # TODO: Generalize
            maxlag = hist.t0idx
            # maxlag = hist.index_interval(self.params.Δ.get_value())
            # HACK/FIXME: We should query history for its lags
            if maxlag > 1:
                lags = [-maxlag, -1]
            else:
                lags = [-1]
            tidx = startidx - self.t0idx + hist.t0idx
            assert(maxlag <= hist.t0idx)
                # FIXME Maybe not necessary if built into lag history
            if len(lags) == 1:
                assert(maxlag == 1)
                outputs_info.append( sinn.upcast(hist._data[tidx],
                                                 to_dtype=hist.dtype,
                                                 same_kind=True,
                                                 disable_rounding=True))
            else:
                outputs_info.append(
                    {'initial': sinn.upcast(hist._data[tidx+1-maxlag:tidx+1],
                                            to_dtype=hist.dtype,
                                            same_kind=True,
                                            disable_rounding=True),
                     'taps': lags})


        outputs, upds = shim.gettheano().scan(onestep,
                                              sequences = shim.arange(startidx+1, stopidx),
                                              outputs_info = outputs_info)
        # Ensure that all updates are of the right type
        # Theano can add updates for variables that don't have a dtype, e.g.
        # a RandomStateType variable, which is why we include the hasattr guard
        upds = OrderedDict([(orig_var,
                             (sinn.upcast(upd, to_dtype=orig_var.dtype,
                                          same_kind=True, disable_rounding=True))
                              if hasattr(orig_var, 'dtype') else upd)
                            for orig_var, upd in upds.items()])
        self.apply_updates(upds)
            # Applying updates ensures we remove the iteration variable
            # scan introduces from the shim updates dictionary

        # Update the state variables
        if not isinstance(outputs, list):
            # Scan does not wrap `outputs` in a list if it is a single variable
            outputs = [outputs]
        for hist, output in zip(self.statehists, outputs):
            valslice = slice(startidx - self.t0idx + hist.t0idx + 1,
                             stopidx  - self.t0idx + hist.t0idx)
            hist.update(valslice, output)

        hist_upds = shim.get_updates()
        # Ensure that all updates are of the right type
        # Theano can add updates for variables that don't have a dtype, e.g.
        # a RandomStateType variable, which is why we include the hasattr guard
        hist_upds = OrderedDict([(orig_var,
                                  (sinn.upcast(upd, to_dtype=orig_var.dtype,
                                              same_kind=True, disable_rounding=True))
                                   if hasattr(orig_var, 'dtype') else upd)
                                 for orig_var, upd in hist_upds.items()])
        return hist_upds


def Surrogate(model):
    """
    Execute `Surrogate(MyModel)` to get a class which can serve as a viable
    stand-in for `MyModel`.

    The surrogate model completely hides the model's __init__    method,
    avoiding the instantiation of potentially large data. If there is
    some initialization you do need, you can add attributes after
    instantiation, or subclass Surrogate(MyModel).

    Parameters
    ----------
    model: class
        Class the surrogate should mirror
    """
    if not issubclass(model, Model):
        raise ValueError("Can only create surrogates for subclasses of sinn.models.Model.")

    class SurrogateModel(model):

        def __init__(self, params, t0=0, dt=None):
            """
            Parameters
            ----------
            params:
                Same parameters as would be passed to create the class
            t0: float
                The time corresponding to time index 0. By default this is 0.
            dt: float
                The time step. Required in order to use `get_tidx` and `index_interval`;
                can be omitted if these functions are not used.
            """
            # Since we don't call super().__init__, we need to reproduce
            # the content of ParameterMixin.__init__
            self.set_parameters(params)
            # Set the attributes required for the few provided methods
            self._t0 = t0
            self._dt = dt   # dt is a read-only property of Model: can't just set the value

        @property
        def t0(self):
            return self._t0
        @property
        def dt(self):
            return self._dt

        def get_tidx(self, t, allow_rounding=False):
            if self.dt is None:
                raise AttributeError("You must provide a timestep 'dt' to the surrogate class "
                                    "in order to call 'get_tidx'.")
            if shim.istype(t, 'int'):
                return t
            else:
                try:
                    shim.check( (t * sinn.config.get_rel_tolerance(t) < self.dt).all() )
                except AssertionError:
                    raise ValueError("You've tried to convert a time (float) into an index "
                                    "(int), but the value is too large to ensure the absence "
                                    "of numerical errors. Try using a higher precision type.")
                t_idx = (t - self.t0) / self.dt
                r_t_idx = shim.round(t_idx)
                if (not shim.is_theano_object(r_t_idx) and not allow_rounding
                    and (abs(t_idx - r_t_idx) > config.get_abs_tolerance(t) / self.dt).all() ):
                    logger.error("t: {}, t0: {}, t-t0: {}, t_idx: {}, dt: {}"
                                .format(t, self._tarr[0], t - self._tarr[0], t_idx, self.dt) )
                    raise ValueError("Tried to obtain the time index of t=" +
                                    str(t) + ", but it does not seem to exist.")
                return shim.cast(r_t_idx, dtype = self._cur_tidx.dtype)

        def index_interval(Δt, allow_rounding=False):
            if self.dt is None:
                raise AttributeError("You must provide a timestep 'dt' to the surrogate class "
                                    "in order to call 'index_interval'.")
            if not shim.is_theano_object(Δt) and abs(Δt) < self.dt - config.abs_tolerance:
                if Δt == 0:
                    return 0
                else:
                    raise ValueError("You've asked for the index interval corresponding to "
                                    "Δt = {}, which is smaller than this history's step size "
                                    "({}).".format(Δt, self.dt))
            if shim.istype(Δt, 'int'):
                return Δt
            else:
                try:
                    shim.check( Δt * config.get_rel_tolerance(Δt) < self.dt )
                except AssertionError:
                    raise ValueError("You've tried to convert a time (float) into an index "
                                    "(int), but the value is too large to ensure the absence "
                                    "of numerical errors. Try using a higher precision type.")
                quotient = Δt / self.dt
                rquotient = shim.round(quotient)
                if not allow_rounding:
                    try:
                        shim.check( shim.abs(quotient - rquotient) < config.get_abs_tolerance(Δt) / self.dt )
                    except AssertionError:
                        logger.error("Δt: {}, dt: {}".format(Δt, self.dt) )
                        raise ValueError("Tried to convert t=" + str(Δt) + " to an index interval "
                                        "but its not a multiple of dt.")
                return shim.cast_int16( rquotient )

        def clear_unlocked_histories(self):
            pass

        def theano_reset(self):
            pass

        def advance(self, t):
            pass

        def loglikelihood(self, start, batch_size, data=None):
            pass

    return SurrogateModel

class ModelKernelMixin:
    """
    Kernels within models should include this mixin.
    Adds interoperability with model parameters
    """
    def __init__(self, name, params, shape=None, f=None, memory_time=None, t0=0, **kwargs):
        super().__init__(name,
                         params = self.get_kernel_params(params),
                         shape  = shape,
                         f      = f,
                         memory_time=memory_time,
                         t0     = t0,
                         **kwargs)

    def update_params(self, params):
        super().update_params(self.get_parameter_subset(params))

    def get_parameter_subset(self, params):
        """Given a set of model parameters, return the set which applies
        to this kernel. These will in general not be a strict subset of
        `model_params`, but derived from them using `get_kernel_params`.
        As a special case, if each of the kernel's parameters can
        be found in `params`, then it is assumed that they have already
        been converted, and `get_kernel_params` is not called again.
        """
        if all( field in params._fields for field in self.params._fields ):
            # params already converted for kernel
            return sinn.get_parameter_subset(self, params)
        else:
            # These are model parameters. Convert them for the kernel
            return self.cast_parameters(self.get_kernel_params(params))

    @staticmethod
    def get_kernel_params(model_params):
        raise NotImplementedError("Each of your model's kernels must "
                                  "implement the method `get_kernel_params`.")
