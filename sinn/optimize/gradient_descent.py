from enum import Enum
from collections import OrderedDict, deque
import time
import logging

import numpy as np
import scipy as sp
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    logger.warning("Unable to import matplotlib. Plotting functions "
                   "will not work.")


import theano
import theano.tensor as T

import theano_shim as shim
import sinn
import sinn.analyze as anlz
import sinn.analyze.heatmap
import sinn.iotools

logger = logging.getLogger('sinn.optimize.gradient_descent')

#######################################
#
# Status constants
#
#######################################

class ConvergeStatus(Enum):
    CONVERGED = 0
    NOTCONVERGED = 1
    ABORT = 2

# DEBUG ?
debug_flags = {}

######################################
#
# General use functions
#
######################################

# Gradient descent involves some form of iteration, for which we typically
# want to provide feedback. Often (e.g. in a Jupyter Notebook), we will
# want to refresh the feedback output, to avoid drowning the interface in
# status messages
try:
    __IPYTHON__
except NameError:
    # We are not within an IPython environment
    def clear_output():
        return
else:
    # We are within an IPython environment – use its clear_output function
    from IPython.display import clear_output

def maxdiff(it, n=10):
    """Return the maximum difference between the last element of it(terable)
    and the n elements before it."""
    return abs( np.array([ it[i]
                           for i in range(len(it) - n-1, len(it)) ])
                - it[-1] ).max()
def meandiff(it, n=10, end=-1):
    """Return the absolute difference between the means of it[-n:] and it[-2n:-n]."""
    assert(end < 0 or end >= n)
    return abs( np.mean([ it[end-i] for i in range(0, n) ])
                - np.mean([ it[end-i] for i in range(n, 2*n)]) )

def difftomean(it, n=10, end=-1):
    """Return the absolute difference between it[end] and the mean of it[end-n : end]."""
    return abs( np.mean([ it[end-i] for i in range(1, n) ]) - it[end] ).max()

def meanabsdev(it, n=10, end=-1):
    """Return the mean absolute deviation of it[-n-1:]"""
    arr = np.array( [it[end-i] for i in range(0, n)] )
    return abs( arr - arr.mean(axis=0) ).mean(axis=0)

def converged(it, r=0.01, p=0.99, n=10, m=10, abs_th=0.001):
    """
    r: resolution. Identify a difference in means of r*sigma with probability p
    p: certainty of each hypothesis test
    n: length of each test
    m: number of tests
    abs_th: changes below this threshold are treated as constant, no matter the
            standard deviation (specifically, this amount is added to the std
            to compute the reference deviation)
    """
    if len(it) < 2*n+m:
        return False
    else:
        # Get the threshold that ensures p statistical power
        arr = np.array( [it[-i] for i in range(1, n+1)] )
        std = arr.std(ddof=1, axis=0) + abs_th
        if np.any( std > 1):
            # Ensure that a large variance make the test accidentally succeed
            return False
        a = sp.stats.norm.ppf(p, loc=r*std, scale=1)
        s = np.sqrt(n) / std  # rescaling to get normalized Student-t
        # Check if the m last iterations were below the threshold
        return all( (meandiff(it, n=n, end=-i) * s < a).all()
                    for i in range(1, m+1) )

def get_indices(param):
    """Return an iterator for the indices of `param`."""
    if len(param.get_value().shape) == 1:
        for i in range(param.get_value().shape[0]):
            yield (i,)
    elif len(param.get_value().shape) == 2:
        for i in range(param.get_value().shape[0]):
            for j in range(param.get_value().shape[1]):
                yield (i, j)
    else:
        raise NotImplementedError

##########################################
#
# Stochastic gradient descent
#
##########################################


class SGD:

    def __init__(self, cost, model, optimizer,
                 burnin=None, datalen=None, mbatch_size=None,
                 sgd_file=None):
        """
        Important that burnin + datalen not exceed the amount of data
        burnin, datalen, mbatch_size should only be omitted when loading from a
        previously saved SGD instance.

        Parameters
        ----------
        cost: Theano graph
            Theano symbolic variable/graph for the cost function we want to optimize.
            Must derive from parameters and histories found in `model`
        model: sinn Model
            Provides a handle to the variables appearing in the cost graph.
        optimizer: str | class/factory function
            String specifying the optimizer to use. Possible values:
              - 'adam'
            If a class/factory function, should have the signature
              `optimizer(cost, fitparams, **kwargs)`
            where `fitparams` has the same form as the homonymous parameter to `compile`.
        burnin: float, int
            Burnin time, in the time units of the model
        datalen: float, int
            Amount of data on which to evaluate the cost, in the model's time units.
            Cost will be evaluated up to the time point 'burnin + datalen'.
        mbatch_size: int, float
            Size of minibatches, either in the model's time units (float) or time bins (int).
        sgd_file: File, where an SGD instance was saved. This can be used to continue a fit;
            in this case, `burnin`, `datalen` and `mbatch_size` are ignored.
        """

        self.cost_fn = cost
        self.model = model
        self.optimizer = optimizer

        self.tidx_var = theano.tensor.lscalar('tidx')
        self.mbatch_var = theano.tensor.lscalar('batch_size')

        self.substitutions = {}
        self._verified_transforms = {}
            # For security reasons, users should visually check that transforms defined
            # with strings are not malicious. We save a flag for each transform, indicating
            # that it has been verified.

        if sgd_file is None:
            self.burnin = burnin
            self.burnin_idx = model.get_t_idx(burnin)
            self.datalen = datalen
            self.data_idxlen = model.index_interval(datalen)
            self.mbatch_size = model.index_interval(mbatch_size)

            #tidx.tag.test_value = 978
            #shim.gettheano().config.compute_test_value = 'warn'
            self.fitparams = None
            self.trueparams = None

            self.initialize()
                # Create the state variables other functions may expect;
                # initialize will be called again after 'fitparams' is set.

        else:
            # TODO: Allow loading both raw (.sir) and sinn pickles (.sin)
            # TODO: Make from_raw a class method, to be consistent with History ?
            #       -> Would require saving the cost function as well
            self.fitparams = OrderedDict()
            self.param_evol = {}
            self.from_raw(sinn.iotools.loadraw(sgd_file))

    def raw(self, **kwargs):
        raw = {}

        def add_attr(attr, retrieve_fn=None):
            # Doing it this way allows to use keywords to avoid errors triggered by getattr(attr)
            if retrieve_fn is None:
                retrieve_fn = lambda attrname: getattr(self, attrname)
            raw[attr] = kwargs.pop(attr) if attr in kwargs else retrieve_fn(attr)

        add_attr('burnin')
        add_attr('burnin_idx')
        add_attr('datalen')
        add_attr('data_idxlen')
        add_attr('mbatch_size')

        add_attr('curtidx')
        add_attr('step_i')
        add_attr('circ_step_i')
        add_attr('output_width')
        add_attr('cum_cost')
        add_attr('cum_step_time')
        add_attr('tot_time')

        add_attr('step_cost')
        add_attr('cost_evol')

        if self.trueparams is not None:
            raw['true_param_names'] = np.array([p.name for p in self.trueparams])
            for p, val in self.trueparams.items():
                raw['true_param_val_' + p.name] = val

        raw['fit_param_names'] = np.array([p.name for p in self.fitparams])
        for p, val in self.fitparams.items():
            raw['mask_' + p.name] = val
        for p, val in self.param_evol.items():
            assert(p.name in raw['fit_param_names'])
            raw['evol_' + p.name] = np.array(val)

        raw['substituted_param_names'] = np.array([p.name for p in self.substitutions])
        for keyvar, subinfo in self.substitutions.items():
            raw['subs_' + keyvar.name] = np.array([subinfo[0].name, subinfo[1], subinfo[2]])

        raw.update(kwargs)

        return raw

    def from_raw(self, raw, trust_transforms=False):
        """
        Don't forget to call `verify_transforms` after this.
        """

        self.burnin = raw['burnin']
        self.burnin_idx = raw['burnin_idx']
        self.datalen = raw['datalen']
        self.data_idxlen = raw['data_idxlen']
        self.mbatch_size = raw['mbatch_size']

        self.curtidx = raw['curtidx']
        self.step_i = raw['step_i']
        self.circ_step_i = raw['circ_step_i']
        self.output_width = raw['output_width']
        self.cum_cost = raw['cum_cost']
        self.cum_step_time = raw['cum_step_time']
        self.tot_time = raw['tot_time']

        self.step_cost = deque(raw['step_cost'])
        self.cost_evol = deque(raw['cost_evol'])

        for name in raw['substituted_param_names']:
            p = None
            for p in self.model.params:
                if p.name == name:
                    break
            # Copied from self.transform
            # New transformed variable value will be set once the transform string is verified
            newvar = theano.shared(p.get_value(),
                                    broadcastable = p.broadcastable,
                                    name = raw['subs_'+name][0])
            inverse_transform = raw['subs_'+name][1]
            transform = raw['subs_'+name][2]
            self.substitutions[p] = (newvar, inverse_transform, transform)

        if 'true_param_names' in raw:
            self.trueparams = {}
            for name in raw['true_param_names']:
                p = None
                for q in self.model.params:
                    if q.name == name:
                        p = q
                        break
                # fitparam might also be transformed from a base model parameter
                for q, subinfo in self.substitutions.items():
                    if subinfo[0].name == name:
                        p = subinfo[0]
                        break
                assert(p is not None)
                self.trueparams[p] = raw['true_param_val_' + name]
        else:
            self.trueparams = None

        fit_param_names = raw['fit_param_names']
        for name in fit_param_names:
            p = None
            for q in self.model.params:
                if q.name == name:
                    p = q
                    break
            # fitparam might also be transformed from a base model parameter
            for q, subinfo in self.substitutions.items():
                if subinfo[0].name == name:
                    p = subinfo[0]
                    break
            assert(p is not None)
            self.fitparams[p] = raw['mask_' + name]
            self.param_evol[p] = raw['evol_' + name]


    def verify_transforms(self, trust_automatically=False):
        """
        Should be called immediately after loading from raw.

        Parameters
        ----------
        trust_automatically: bool
            Bypass the verification. Required to avoid user interaction, but
            since it beats the purpose of this function, to be used with care.
        """
        if len(self.substitutions) == 0 or all(_verified_transforms.values()):
            # Nothing to do; don't bother the user
            return

        if trust_automatically:
            trusted = True
        else:
            trusted = False
            print("Here are the transforms currently used:")
            for p, val in self.substitutions.items():
                print("{} -> {} – (to) '{}', (from) '{}'"
                      .format(p.name, val[0].name, val[2], val[1]))
            res = input("Press y to confirm that these transforms are not malicious.")
            if res[0].lower() == 'y':
                trusted = True

        if trusted:
            for p, subinfo in self.substitutions.items():
                self._verified_transforms[p] = True
                # Set the transformed value now that the transform has been verified
                subinfo[0].set_value(self._make_transform(p, subinfo[2])(p.get_value()))

    def _make_transform(self, variable, transform_desc):
        assert variable in self._verified_transforms
        if not self._verified_transforms[variable]:
            raise RuntimeError("Because they are interpreted with `eval`, you "
                               "must verify transforms before using them.")

        comps = transform_desc.split('->')

        try:
            if len(comps) == 1:
                # Specified just a callable, like 'log10'
                return eval('lambda x: ' + comps[0] + '(x)')
            elif len(comps) == 2:
                return eval('lambda ' + comps[0] + ': ' + comps[1])
            else:
                raise SyntaxError

        except SyntaxError:
            raise ValueError("Invalid transform description: \n '{}'"
                             .format(transform_desc))

    def transform(self, variable, newname, transform, inverse_transform):
        """
        Transform a variable, for example replacing it by its logarithm.
        If called multiple times with the same variable, only the last
        transform is saved.
        Transformations must be invertible.
        LIMITATION: This only really works for element-wise transformations,
        where `transform(x)[0,0]` only depends on `x[0,0]`. For more complex
        transformations, the `fitparams` argument to `compile` needs to be
        defined by directly accessing `substitutions[variable][0]`.

        Parameters
        ----------
        variable:  theano variable
            Must be part of the Theano graph for the log likelihood.
        newname: string
            Name to assign the new variable.
        transform: str
            TODO: Update docs to string defs for transforms
            Applied to `variable`, returns the new variable we want to fit.
            E.g. if we want to fit the log of `variable`, than `transform`
            could be specified as `lambda x: shim.log10(x)`.
            Make sure to use `shim` functions, rather directly `numpy` or `theano`
            to ensure expected behaviour.
        inverse_transform: str
            Given the new variable, returns the old one. Continuing with the log
            example, this would be `lambda x: 10**x`.
        """
        # # Check that variable is part of the fit parameters
        # if variable not in self.fitparams:
        #     raise ValueError("Variable '{}' is not part of the fit parameters."
        #                      .format(variable))

        assert(newname != variable.name)
            # TODO: More extensive test that name doesn't already exist

        # Check that variable is a shared variable
        if not shim.isshared(variable):
            raise ValueError("Only shared variables can be transformed.")

        # Check that variable is part of the computational graph
        self.model.theano_reset()
        self.model.clear_unlocked_histories()
        #logL = self.model.loglikelihood(self.tidx_var, self.tidx_var + self.mbatch_size)
        cost, statevar_upds, shared_upds = self.cost_fn(self.tidx_var, self.mbatch_var)
        self.model.clear_unlocked_histories()
        self.model.theano_reset()
        if variable not in theano.gof.graph.inputs([cost]):
            raise ValueError("'{}' is not part of the Theano graph for the cost."
                             .format(variable))

        # Since these transform descriptions are given by the user, they are assumed safe
        self._verified_transforms[variable] = True
        _transform = self._make_transform(variable, transform)
        _inverse_transform = self._make_transform(variable, inverse_transform)

        # Check that the transforms are callable and each other's inverse
        # Choosing a test value is error prone, since different variables will have
        # different domains – 0.5 is about as safe a value as we will find
        testx = 0.5
        try:
            _transform(testx)
        except TypeError as e:
            if "is not callable" in str(e):
                # FIXME This error might be confusing now that we save strings instead of callables
                raise ValueError("'transform' argument (current '{}' must be "
                                 "callable.".format(transform))
            else:
                # Some other TypeError
                raise
        try:
            _inverse_transform(testx)
        except TypeError as e:
            if "is not callable" in str(e):
                # FIXME See above
                raise ValueError("'inverse_transform' argument (current '{}' must be "
                                 "callable.".format(inverse_transform))
            else:
                # Some other TypeError
                raise
        if not sinn.isclose(testx, _inverse_transform(_transform(testx))):
            raise ValueError("The given inverse transform does not actually invert "
                             "`transform({})`.".format(testx))

        # Create the new transformed variable
        newvar = theano.shared(_transform(variable.get_value()),
                               broadcastable = variable.broadcastable,
                               name = newname)

        # Save the transform
        # We save the strings rather than callables, as that allows us to save them
        # along with the optimizer when we save to file
        self.substitutions[variable] = (newvar, inverse_transform, transform)
            # From this point on the inverse_transform is more likely to be used,
            # but in some cases we still need the original transform

        # If the ground truth of `variable` was set, compute that of `newvar`
        self._augment_ground_truth_with_transforms()

    def get_substituted_cost_graph(self, fitparams, **kwargs):

        # Store the learning rate since it's also used in the convergence test
        if 'lr' in kwargs:
            self._compiled_lr = kwargs['lr']
        else:
            self._compiled_lr = None

        # Create the Theano `replace` parameter from self.substitutions
        if len(self.substitutions) > 0:
            replace = { var: self._make_transform(var, subs[1])(subs[0])
                        for var, subs in self.substitutions.items() }
                # We use the inverse transform here, because transform(var)
                # is the variable we want in the graph
                # (e.g. to replace τ by log τ, we need a new variable `logτ`
                #  and then we would replace in the graph by `10**logτ`.
        else:
            replace = None

        # Ensure fitparams is a dictionary of param : mask pairs.
        if isinstance(fitparams, dict):
            fitparamsarg = fitparams
        else:
            fitparamsarg = dict(
                [ (param[0], param[1]) if isinstance(param, tuple) else (param, True)
                  for param in fitparams ]
                )

        # Create self.fitparams
        # If any of the fitparams appear in self.substitutions, change those
        # This assumes that the substitution transforms are element wise
        if self.fitparams is not None:
            logger.warning("Replacing 'fitparams'. Previous fitparams are lost.")
        self.fitparams = OrderedDict()
        for param, mask in fitparamsarg.items():
            if param in self.substitutions:
                self.fitparams[self.substitutions[param][0]] = mask
            else:
                self.fitparams[param] = mask


        self.model.theano_reset()
        self.model.clear_unlocked_histories()

        logger.info("Producing the cost function theano graph")
        cost, statevar_upds, shared_upds = self.cost_fn(self.tidx_var, self.mbatch_var)
        logger.info("Cost function graph complete.")
        if replace is not None:
            logger.info("Performing variable substitutions in Theano graph.")
            cost = theano.clone(cost, replace=replace)
            logger.info("Substitutions complete.")

        return cost, statevar_upds, shared_upds

    def compile(self, fitparams, **kwargs):
        # Compile step function

        cost, statevar_upds, shared_upds = self.get_substituted_cost_graph(fitparams, **kwargs)
            # Sets self.fitparams

        logger.info("Compiling the minibatch cost function.")
        # DEBUG (because on mini batches?)
        self.cost = theano.function([self.tidx_var, self.mbatch_var], cost)#, updates=cost_updates)
        logger.info("Done compilation.")

        # Function for stepping the model forward, e.g. for burnin
        #logger.info("Compiling the minibatch advancing function.")
        #self.cost = theano.function([self.tidx_var], updates=cost_updates)
        #logger.info("Done compilation.")

        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                logger.info("Calculating Adam optimizer updates.")
                optimizer_updates = Adam(-cost, self.fitparams, **kwargs)
            else:
                raise ValueError("Unrecognized optimizer '{}'.".format(self.optimizer))

        else:
            # Treat optimizer as a factory class or function
            try:
                logger.info("Calculating custom optimizer updates.")
                optimizer_updates = self.optimizer(-cost, self.fitparams, **kwargs)
            except TypeError as e:
                if 'is not callable' not in str(e):
                    # Some other TypeError was triggered; reraise
                    raise
                else:
                    raise ValueError("'optimizer' parameter should be either a string or a "
                                     "callable which returns an optimizer (such as a class "
                                     "name or a factory function).\nThe original error was:\n"
                                     + str(e))

        logger.info("Done calculating optimizer updates.")

        assert(len(shim.get_updates()) == 0)
        #shim.add_updates(optimizer_updates)

        logger.info("Compiling the optimization step function.")
        self._step = theano.function([self.tidx_var, self.mbatch_var], [], updates=optimizer_updates)#shim.get_updates())
        logger.info("Done compilation.")

        # # Compile likelihood function
        # self.model.clear_unlocked_histories()
        # self.model.theano_reset()
        # #cost = self.cost_fn(self.burnin, self.burnin + self.datalen)
        # cost, cost_updates = self.cost_fn(self.tidx_var, self.tidx_var + self.mbatch_size)
        # if len(replace) > 0:
        #     cost = theano.clone(cost, replace)

        # self.cost = theano.function([], cost)

        self.model.clear_unlocked_histories()
        self.model.theano_reset()  # clean the graph after compilation

        self.initialize()

    def set_ground_truth(self, trueparams):
        """
        If the true parameters are specified, they will be indicated in plots.

        Parameters
        ----------
        trueparams: Iterable of shared variables
        """
        self.trueparams = { param: param.get_value() for param in trueparams }
        self._augment_ground_truth_with_transforms()

    def _augment_ground_truth_with_transforms(self):
        """Add to the dictionary of ground truth parameters the results of transformed variables."""
        if self.trueparams is None:
            return
        for param, transformedparaminfo in self.substitutions.items():
            if param in self.trueparams:
                transformedparam = transformedparaminfo[0]
                transform = self._make_transform(param, transformedparaminfo[2])
                self.trueparams[transformedparam] = transform(self.trueparams[param])

    def get_param(self, name):
        for param in self.fitparams:
            if param.name == name:
                return param
        for param in self.substitutions:
            if param.name == name:
                return param

        raise KeyError("No parameter has the name '{}'".format(name))

    def set_param_values(self, new_params, mask=None):
        """
        Update parameter values.

        Parameters
        ----------
        new_params: dictionary
            Dictionary where keys are model parameters, and
            values are the new values they should take. It is possible to
            specify only a subset of parameters to update.
        mask: dictionary
            (Optional) Only makes sense to specify this option along with
            `new_params`. If given, only variable components corresponding
            to where the mask is True are updated.
        """
        for param, val in new_params.items():
            # Get the corresponding parameter mask, if specified
            parammask = None
            if mask is not None:
                if param in mask:
                    parammask = mask[param]
                else:
                    # Also check the list of substituted parameters
                    for subp, subinfo in self.substitutions.items():
                        if subp is param and subinfo[0] in mask:
                            parammask = mask[subinfo[0]]
                        elif subinfo[0] is param and subp in mask:
                            parammask = mask[subp]

            if shim.isshared(val):
                val = val.get_value()
            val = np.array(val, dtype=param.dtype)

            if parammask is not None:
                val = np.array([ newval if m else oldval
                                for newval, oldval, m in zip(val.flat,
                                                             param.get_value().flat,
                                                             parammask.flat) ])

            val = val.reshape(param.get_value().shape)
            param.set_value(val)

            # TODO: move to separate method, that can also be called e.g. after gradient descent
            # See if this parameter appears in the substitutions
            for subp, subinfo in self.substitutions.items():
                if param is subp:
                    # This parameter was substituted by another; update the new parameter
                    subinfo[0].set_value(self._make_transform(param, subinfo[2])(val))
                elif param is subinfo[0]:
                    # This parameter substitutes another; update the original
                    subp.set_value(self._make_transform(param, subinfo[1])(val))

    def initialize(self, new_params=None, mask=None):
        """
        Clear the likelihood and parameter histories.

        Parameters
        ----------
        new_params: dictionary
            (Optional) Dictionary where keys are model parameters, and
            values are the new values they should take. It is possible to
            specify only a subset of parameters to update.
        mask: dictionary
            (Optional) Only makes sense to specify this option along with
            `new_params`. If given, only variable components corresponding
            to where the mask is True are updated.
        """
        if new_params is not None:
            self.set_param_values(new_params, mask)

        if self.fitparams is not None:
            self.param_evol = {param: deque([param.get_value()])
                               for param in self.fitparams}
        else:
            self.param_evol = {}

        self.cost_evol = deque([])

        self.step_i = 0
        self.circ_step_i = 0
        self.output_width = 5 # Default width for printing number of steps
        self.step_cost = []
        self.curtidx = self.burnin_idx
        self.cum_step_time = 0
        self.cum_cost = 0
        self.tot_time = 0

    def converged(it, r, p=0.99, n=10, m=10, abs_th=0.001):
        """
        r: resolution. Identify a difference in means of r*sigma with probability p. A smaller r makes the test harder to satisfy; the learning rate is a good starting point for this value.
        p: certainty of each hypothesis test
        n: length of each test
        m: number of tests
        abs_th: changes below this threshold are treated as constant, no matter the
                standard deviation (specifically, this amount is added to the std
                to compute the reference deviation)
        """
        if len(it) < 2*n+m:
            return False
        else:
            # Get the threshold that ensures p statistical power
            arr = np.array( [it[-i] for i in range(1, n+1)] )
            std = arr.std(ddof=1, axis=0) + abs_th
            if np.any( std > 1):
                # Ensure that a large variance does not make the test accidentally succeed
                return False
            a = sp.stats.norm.ppf(p, loc=r*std, scale=1)
            s = np.sqrt(n) / std  # rescaling to get normalized Student-t
            # Check if the m last iterations were below the threshold
            return all( (meandiff(it, end=-i) * s < a).all()
                        for i in range(1, m+1) )

    def step(self, conv_res, cost_calc='cum', **kwargs):
        """
        Parameters
        ----------
        conv_res: float
            Convergence resolution. Smaller number means a more stringent test for convergence.
        cost_calc: str
            Determines how the cost is computed. Some values allow additional
            keywords to define behaviour. Default is 'cum'; possible values are:
              - 'cum': (cumulative) The cost of each step is computed, and added to the
                previously computed values. This is effectively the cost seen by the algorithm;
                it is cheap to compute, but gives only an approximation of the true cost,
                as it aggregates the result of many different values.
                After having used up all the data, the sum is saved and the process
                repeats for the next round.
              - 'full': At every step, the full likelihood over the whole data is computed.
                This is slower, but gives a better indication of whether the algorithm is
                going in the right direction.
                *Additional keyword*
                  + 'cost_period': int. Compute the cost only after this many steps

        """
        # Clear notebook of previous iteration's output
        clear_output(wait=True)
            #`wait` indicates to wait until something is printed, which avoids flicker

        if self.curtidx > self.burnin_idx + self.data_idxlen - self.mbatch_size:
            # We've run through the dataset
            # Reset time index to beginning
            self.curtidx = self.burnin_idx
            self.model.clear_unlocked_histories()
            if cost_calc == 'cum':
                self.cost_evol.append(self.cum_cost)
            #self.model.clear_unlocked_histories()
            self.cum_cost = 0
            self.circ_step_i = 0

            # HACK Fill the data corresponding to the burnin time
            #      (hack b/c ugly + repeated in iterate() + does not check indexing)
            logger.info("Iteration {:>{}} – Moving current index forward to the end of the burnin period."
                        .format(self.step_i, self.output_width))
            self.model.advance(self.burnin_idx)
            #for i in range(0, self.burnin_idx, self.mbatch_size):
            #    self._step(i)
            logger.info("Done.")

        else:
            # TODO: Check what it is that is cleared here, and why we need to do it
            self.model.clear_other_histories()

        t1 = time.perf_counter()
        self.model.advance(self.curtidx)
            # FIXME?: Currently `advance` is needed because grad updates don't change the data
        self._step(self.curtidx, self.mbatch_size)
        self.cum_step_time += time.perf_counter() - t1
        for param in self.fitparams:
            self.param_evol[param].append(param.get_value())

        #if cost_calc == 'cum':
        if True:
            if self.circ_step_i >= len(self.step_cost):
                # At first, extend step_cost as points are added
                assert(self.circ_step_i == len(self.step_cost))
                self.step_cost.append(self.cost(self.curtidx, self.mbatch_size))
            else:
                # After having looped through the data, reuse memory for the cumulative cost
                self.step_cost[self.circ_step_i] = self.cost(self.curtidx, self.mbatch_size)
            self.cum_cost += self.step_cost[self.circ_step_i]

        if ( cost_calc == 'full'
             and self.step_i % kwargs.get('cost_period', 1) == 0 ):
            self.cost_evol.append(self.cost(self.burnin_idx, self.data_idxlen))

        # Increment step counter
        self.step_i += 1
        self.circ_step_i += 1
        self.curtidx += self.mbatch_size

        # TODO: Use a circular iterator for step_cost, so that a) we don't need circ_step_i
        #       and b) we can test over intervals that straddle a reset of curtidx
        # Check to see if there have been meaningful changes in the last 10 iterations
        if ( converged(self.cost_evol, r=conv_res, n=4, m=3) and
             converged(self.step_cost[:self.circ_step_i], r=conv_res, n=100)
             and all( converged(self.param_evol[p], r=conv_res) for p in self.fitparams) ):
            logger.info("Converged. log L = {:.2f}".format(float(self.cost_evol[-1])))
            return ConvergeStatus.CONVERGED

        #Print progress  # TODO: move to top of step
        logger.info("Iteration {:>{}} – <log L> = {:.2f}"
                    .format(self.step_i,
                            self.output_width,
                            float(sum(self.step_cost[:self.circ_step_i])/(self.curtidx + self.mbatch_size -self.burnin_idx))))
        if cost_calc == 'full':
            logger.info(" "*(13+self.output_width) + "Last evaluated log L: {}".format(self.cost_evol[-1]))

        return ConvergeStatus.NOTCONVERGED

    def iterate(self, Nmax=int(5e3), lr=None, **kwargs):
        """
        Parameters
        ----------

        **kwargs: Additional keyword arguments are passed to `step`.
        """

        Nmax = int(Nmax)
        self.output_width = int(np.log10(Nmax))

        if lr is not None:
            # Override the learning rate set during compilation
            pass
        else:
            lr = self._compiled_lr
        assert(lr is not None)

        # HACK Fill the data corresponding to the burnin time
        #      (hack b/c ugly + repeated in step() + does not check indexing)
        # Possible fix: create another "skip burnin" function, with scan ?
        logger.info("Moving current index forward to the end of the burnin period.")
        #for i in range(0, self.burnin_idx, self.mbatch_size):
        #    self._step(i)
        self.model.advance(self.burnin_idx)
        logger.info("Done.")

        t1 = time.perf_counter()
        try:
            for i in range(Nmax):
                status = self.step(conv_res=lr, **kwargs)
                if status in [ConvergeStatus.CONVERGED, ConvergeStatus.ABORT]:
                    break
        except KeyboardInterrupt:
            print("Gradient descent was interrupted.")
        finally:
            self.tot_time = time.perf_counter() - t1

        if status != ConvergeStatus.CONVERGED:
            print("Did not converge.")

        if i > 0:
            logger.info("Cost/likelihood evaluation : {:.1f}s / {:.1f}s ({:.1f}% total "
                        "execution time)"
                        .format(self.cum_step_time,
                                self.tot_time,
                                self.cum_step_time / (self.tot_time) * 100))
            logger.info("Time per iteration: {:.3f}ms".format((self.tot_time)/self.step_i*1000))

        #with open("sgd-evol", 'wb') as f:
        #    pickle.dump((L_evol, param_evol), f)

    def get_evol(self):
        """
        Return a dictionary storing the evolution of the cost and parameters.

        Parameters
        ----------

        Returns
        -------
        dictionary:
             The log likelihood evolution is associated the string key 'logL'.
             Parameter evolutions are keyed by the parameter names.
             Each evolution is stored as an ndarray, with the first dimension
             corresponding to epochs.
        """
        evol = { param.name: np.array([val for val in self.param_evol[param]])
                 for param in self.fitparams }
        evol['logL'] = np.array([val for val in self.cost_evol])
        return evol

    def stats(self):
        return {'number iterations': self.step_i,
                'total time (s)' : self.tot_time,
                'average step time (ms)': self.cum_step_time / self.step_i * 1000,
                'average time per iteration (ms)': self.tot_time / self.step_i * 1000,
                'time spent stepping (%)': self.cum_step_time / (self.tot_time) * 100}

    def plot_cost_evol(self, evol=None):
        if evol is None:
            evol = self.get_evol()
        plt.title("Maximization of likelihood")
        plt.plot(evol['logL'])
        plt.xlabel("epoch")
        plt.ylabel("$\log L$")

    def plot_param_evol(self, ncols=3, evol=None):

        nrows = int(np.ceil(len(self.fitparams) / ncols))
        if evol is None:
            evol = self.get_evol()

        # if self.trueparams is None:
        #     trueparams = [None] * len(self.fitparams)
        # else:
        #     trueparams = self.trueparams
        # for i, (name, param, trueparam) in enumerate( zip(self.fitparams._fields,
        #                                                   self.fitparams,
        #                                                   trueparams),
        #                                               start=1):

        for i, param in enumerate(self.fitparams, start=1):
            plt.subplot(nrows,ncols,i)
            plt.plot(evol[param.name].reshape(len(evol[param.name]), -1))
                # Flatten the parameter values
            plt.title(param.name)
            plt.xlabel("epoch")
            plt.legend(["${}_{{{}}}$".format(param.name, ', '.join(str(i) for i in idx))
                        for idx in get_indices(param)])
                        #for i in range(param.get_value().shape[0])
                        #for j in range(param.get_value().shape[0])])
            plt.gca().set_prop_cycle(None)
            if self.trueparams is not None:
                if param in self.trueparams:
                    plt.plot( [ self.trueparams[param].flatten()
                                for i in range(len(evol[param.name])) ],
                              linestyle='dashed' )
                else:
                    logger.warning("Although ground truth parameters have been set, "
                                   "the value of '{}' was not.".format(param.name))

    def plot_param_evol_overlay(self, basedata, evol=None, **kwargs):
        """
        Parameters
        ----------
        basedata: Heatmap or […]
            The sinn data object on top of which to draw the overlay. Currently
            only heatmaps are supported.

        evol: dictionary, as returned from `get_evol()`
            The evolution of parameters, as returned from this instance's `get_evol`
            method. If not specified, `get_evol` is called to retrieve the latest
            parameter evolution.

        **kwargs:
            Additional keyword arguments are passed to `plt.plot`
        """

        if evol is None:
            evol = self.get_evol()

        if isinstance(basedata, anlz.heatmap.HeatMap):
            if len(basedata.axes) == 1:
                raise NotImplementedError("Overlaying not implemented on 1D heatmaps.\n"
                                          "What are you hoping to do with a 1D heatmap "
                                          "anyway ?")

            elif len(basedata.axes) == 2:
                # Construct lists of parameter evolution, one parameter (i.e. coord)
                # at a time
                plotcoords = []
                #fitparamnames = [p.name for p in self.fitparams]
                #subsparamnames = [p.name for p in self.substitutions]
                for ax in basedata.axes:
                    found = False
                    for param in self.fitparams:
                        if param.name == ax.name:
                            found = True
                            # Get the parameter evolution
                            if shim.isscalar(param):
                                plotcoords.append(evol[param.name])
                            else:
                                idx = list(ax.idx)
                                # Indexing for the heat map might neglect 1-element dimensions
                                if len(idx) < param.ndim:
                                    shape = param.get_value().shape
                                    axdim_i = 0
                                    for i, s in enumerate(shape):
                                        if s == 1 and len(idx) < param.ndim:
                                            idx.insert(i, 0)
                                    assert(len(idx) == param.ndim)

                                idx.insert(0, slice(None)) # First axis is time: grab all of those
                                idx = tuple(idx)           # Indexing won't work with a list

                                plotcoords.append(evol[param.name][idx])

                    if not found:
                        for param in self.substitutions:
                            # As above, but we also invert the variable transformation
                            if param.name == ax.name:
                                found = True
                                transformedparam = self._make_transform(param, self.substitutions[param][0])
                                inversetransform = self._make_transform(param, self.substitutions[param][1])
                                if shim.isscalar(param):
                                    plotcoords.append(inversetransform( evol[transformedparam.name] ))
                                else:
                                    idx = list(ax.idx)
                                    # Indexing for the heat map might neglect 1-element dimensions
                                    if len(idx) < param.ndim:
                                        shape = param.get_value().shape
                                        axdim_i = 0
                                        for i, s in enumerate(shape):
                                            if s == 1 and len(idx) < param.ndim:
                                                idx.insert(i, 0)
                                        assert(len(idx) == param.ndim)

                                    idx.insert(0, slice(None))  # First axis is time: grab all of those
                                    idx = tuple(idx)            # Indexing won't work with a list

                                    plotcoords.append(inversetransform( evol[transformedparam.name][idx] ))

                    if not found:
                        raise ValueError("The base data has a parameter '{}', which "
                                         "does not match any of the fit parameters."
                                         .format(ax.name))

                assert( len(plotcoords) == 2 )
                plt.plot(plotcoords[0], plotcoords[1], **kwargs)

            else:
                raise NotImplementedError("Overlaying currently implemented "
                                          "only on 2D heatmaps")

        else:
            raise ValueError( "Overlays are not currently supported for base data of "
                              "type {}.".format( str(type(basedata)) ) )

#######################################
#
# Optimizers
#
#######################################


"""
The MIT License (MIT)
Copyright (c) 2015 Alec Radford
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, grad_fn=None):
    """
    Adam optimizer. Returns a set of gradient descent updates.
    This is ported from the GitHub Gist by Alec Radford
    https://gist.github.com/Newmu/acb738767acb4788bac3 (MIT License).

    Parameters
    ----------
    cost: theano variable
        We want to minimize this cost.

    params: list
        List of Theano shared variables. Any element may be specified instead
        as a tuple pair, whose first element is the shared variable, and the
        second is a boolean mask array. If given, the mask array should be of
        the same shape as the shared variable – False entries indicate that
        we are not fitting for this parameter component, and so its gradient
        is to be set to zero.

    […]

    grad_fn: function
        If specified, use this instead of `T.grad` to compute the cost's gradient.
        Should have the same signature (i.e. `grad_fn(cost, params)`) and return
        a result of the same shape as `T.grad`.

    Returns
    -------
    Theano update dictionary for the parameters in `params`
    """
    tmpparams = []
    param_masks = []
    # Standardize the form of params
    if isinstance(params, dict):
        # Convert dictionary to a list of (param, mask_descriptor) tuples
        params = list(params.items())
    # Extract the gradient mask for each parameter
    for p in params:
        if isinstance(p, tuple):
            assert(len(p) == 2)
            tmpparams.append(p[0])
            if isinstance(p[1], bool):
                param_masks.append(np.ones(p[0].get_value().shape, dtype=int)
                                  * p[1])
            else:
                if p[1].shape != p[0].get_value().shape:
                    raise ValueError("Provided mask (shape {}) for parameter {} "
                                     "(shape {}) has a different shape."
                                     .format(p[1].shape, p[0].name, p[0].get_value().shape))
                param_masks.append(p[1])
        else:
            tmpparams.append(p)
            param_masks.append(None)
    params = tmpparams

    updates = OrderedDict()

    if grad_fn is None:
        grads = T.grad(cost, params)
    else:
        grads = grad_fn(cost, params)

    # DEBUG ?
    if 'print grads' in debug_flags:
        for i, p in enumerate(params):
            if p.name in debug_flags['print grads']:
                grads[i] = shim.print(grads[i], 'gradient ' + p.name)
    # Mask out the gradient for parameters we aren't fitting
    for i, m in enumerate(param_masks):
        if m is not None:
            grads[i] = grads[i]*m
                # m is an array of ones and zeros
    i = theano.shared(sinn.config.cast_floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        if hasattr(p, 'broadcastable'):
            m = theano.shared(p.get_value() * 0., broadcastable=p.broadcastable)
            v = theano.shared(p.get_value() * 0., broadcastable=p.broadcastable)
        else:
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates[m] = m_t
        updates[v] = v_t
        updates[p] = p_t
    updates[i] = i_t
    return updates



class NPAdam:
    """
    A pure NumPy version of the Adam optimizer

    params: list
        List of Theano shared variables. Any element may be specified instead
        as a tuple pair, whose first element is the shared variable, and the
        second is a boolean mask array. If given, the mask array should be of
        the same shape as the shared variable – False entries indicate that
        we are not fitting for this parameter component, and so its gradient
        is to be set to zero.

    […]

    Returns
    -------
    Theano update dictionary for the parameters in `params`
    """
    def __init__(self, grad_fn, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        self.param_masks = []
        # Standardize the form of params
        if isinstance(params, dict):
            # Convert dictionary to a list of (param, mask_descriptor) tuples
            params = list(params.items())
        # Extract the gradient mask for each parameter
        for p in params:
            if isinstance(p, tuple):
                assert(len(p) == 2)
                if isinstance(p[1], bool):
                    self.param_masks.append(np.ones(p[0].get_value().shape, dtype=int)
                                    * p[1])
                else:
                    if p[1].shape != p[0].get_value().shape:
                        raise ValueError("Provided mask (shape {}) for parameter {} "
                                        "(shape {}) has a different shape."
                                        .format(p[1].shape, p[0].name, p[0].get_value().shape))
                    self.param_masks.append(p[1])
            else:
                self.param_masks.append(None)

        self.grad_fn = grad_fn

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e

        self.i = 0
        self.m = deque([])
        self.v = deque([])
        for p in zip(params):
            self.m.append( np.zeros(p.shape) )
            self.v.append( np.zeros(p.shape) )

    def masked_grad_fn(self, params):
        grads = self.grad_fn(params)
        # Mask out the gradient for parameters we aren't fitting
        for i, m in enumerate(self.param_masks):
            if m is not None:
                grads[i] = grads[i]*m
                # m is an array of ones and zeros
        return grads

    def __call__(self, params):

        grads = self.masked_grad_fn(params)

        self.i += 1
        fix1 = 1. - (1. - self.b1)**self.i
        fix2 = 1. - (1. - self.b2)**self.i
        lr_t = lr * (np.sqrt(fix2) / fix1)

        p_t = []
        for p, g in zip(params, grads):
            self.m[i] = (b1 * g) + ((1. - b1) * self.m[i])
            self.v[i] = (b2 * g**2) + ((1. - b2) * self.v[i])
            g_t = self.m[i] / (np.sqrt(self.v[i]) + self.e)
            p_t = p - (lr_t * g_t)
        return updates
