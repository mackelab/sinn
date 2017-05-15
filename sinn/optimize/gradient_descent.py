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
        return all( (meandiff(it, end=-i) * s < a).all()
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

    def __init__(self, model, optimizer,
                 burnin, datalen, mbatch_size):
        """
        Important that burnin + datalen not exceed the amount of data
        """

        self.model = model
        self.optimizer = optimizer
        self.burnin = burnin
        self.burnin_idx = model.get_t_idx(burnin)
        self.datalen = datalen
        self.data_idxlen = model.index_interval(datalen)
        self.mbatch_size = model.index_interval(mbatch_size)

        self.step_i = 0
        self.output_width = 5 # Default width for printing number of steps
        self.tidx = theano.tensor.lscalar('tidx')
        #tidx.tag.test_value = 978
        #shim.gettheano().config.compute_test_value = 'warn'
        self.substitutions = {}
        self.trueparams = None

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
        transform: callable
            Applied to `variable`, returns the new variable we want to fit.
            E.g. if we want to fit the log of `variable`, than `transform`
            could be specified as `lambda x: shim.log10(x)`.
            Make sure to use `shim` functions, rather directly `numpy` or `theano`
            to ensure expected behaviour.
        inverse_transform: callable
            Given the new variable, returns the old one. Continuing with the log
            example, this would be `lambda x: 10**x`.
        """
        # # Check that variable is part of the fit parameters
        # if variable not in self.fitparams:
        #     raise ValueError("Variable '{}' is not part of the fit parameters."
        #                      .format(variable))

        # Check that variable is a shared variable
        if not shim.isshared(variable):
            raise ValueError("Only shared variables can be transformed.")

        # Check that variable is part of the computational graph
        self.model.theano_reset()
        self.model.clear_unlocked_histories()
        logL = self.model.loglikelihood(self.tidx, self.tidx + self.mbatch_size)
        self.model.clear_unlocked_histories()
        self.model.theano_reset()
        if variable not in theano.gof.graph.inputs([logL]):
            raise ValueError("'{}' is not part of the Theano graph for the log "
                             "likelihood".format(variable))

        # Check that the transforms are callable and each other's inverse
        # Choosing a test value is error prone, since different variables will have
        # different domains – 0.5 is about as safe a value as we will find
        testx = 0.5
        try:
            transform(testx)
        except TypeError as e:
            if "is not callable" in str(e):
                raise ValueError("'transform' argument (current '{}' must be "
                                 "callable.".format(transform))
            else:
                # Some other TypeError
                raise
        try:
            inverse_transform(testx)
        except TypeError as e:
            if "is not callable" in str(e):
                raise ValueError("'inverse_transform' argument (current '{}' must be "
                                 "callable.".format(inverse_transform))
            else:
                # Some other TypeError
                raise
        if not sinn.isclose(testx, inverse_transform(transform(testx))):
            raise ValueError("The given inverse transform does not actually invert "
                             "`transform({})`.".format(testx))

        # Create the new transformed variable
        newvar = theano.shared(transform(variable.get_value()),
                               broadcastable = variable.broadcastable,
                               name = newname)

        # Save the transform
        self.substitutions[variable] = (newvar, inverse_transform, transform)
            # From this point on the inverse_transform is more likely to be used,
            # but in some cases we still need the original transform

        # If the ground truth of `variable` was set, compute that of `newvar`
        self._augment_ground_truth_with_transforms()

    def compile(self, fitparams, **kwds):

        # Create the Theano `replace` parameter from self.substitutions
        if len(self.substitutions) > 0:
            replace = { var: subs[1](subs[0])
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
            fitparamsarg = {
                [ (param[0], param[1]) if isinstance(param, tuple) else (param, True)
                for param in fitparams ]
                }

        # Creat self.fitparams
        # If any of the fitparams appear in self.substitutions, change those
        # This assumes that the substitution transforms are element wise
        self.fitparams = OrderedDict()
        for param, mask in fitparamsarg.items():
            if param in self.substitutions:
                self.fitparams[self.substitutions[param][0]] = mask
            else:
                self.fitparams[param] = mask

        # Compile step function
        self.model.theano_reset()
        self.model.clear_unlocked_histories()
        logL = self.model.loglikelihood(self.tidx, self.tidx + self.mbatch_size)
        if replace is not None:
            logL = theano.clone(logL, replace=replace)

        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                updates = Adam(-logL, self.fitparams, **kwds)
            else:
                raise ValueError("Unrecognized optimizer '{}'.".format(self.optimizer))

        else:
            # Treat optimizer as a class or factory function
            try:
                updates = self.optimizer(-logL, self.fitparams, **kwds)
            except TypeError as e:
                if 'is not callable' not in str(e):
                    # Some other TypeError was triggered; reraise
                    raise
                else:
                    raise ValueError("'optimizer' parameter should be either a string or a "
                                     "callable which returns an optimizer (such as a class "
                                     "name or a factory function).\nThe original error was:\n"
                                     + str(e))

        updates.update(sinn.get_updates())

        self._step = theano.function([self.tidx], [], updates=updates)

        # Compile likelihood function
        self.model.clear_unlocked_histories()
        self.model.theano_reset()
        logL = self.model.loglikelihood(self.burnin, self.burnin + self.datalen)
        if len(replace) > 0:
            logL = theano.clone(logL, replace)

        self.logL = theano.function([], -logL)

        self.model.theano_reset()  # clean the graph after compilation

        self.initialize()

    def set_ground_truth(self, trueparams):
        """
        If the true parameters are specified, they will be indicated in plots.

        Parameters
        ----------
        trueparams: Iterable of shared variables
        """
        # try:
        #     self.fitparams
        # except AttributeError:
        #     pass
        # else:
        #     assert(len(trueparams) == len(self.fitparams))
        self.trueparams = { param: param.get_value() for param in trueparams }
        self._augment_ground_truth_with_transforms()

    def _augment_ground_truth_with_transforms(self):
        """Add to the dictionary of ground truth parameters the results of transformed variables."""
        if self.trueparams is None:
            return
        for param, transformedparaminfo in self.substitutions.items():
            if param in self.trueparams:
                transformedparam = transformedparaminfo[0]
                transform = transformedparaminfo[2]
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
                    subinfo[0].set_value(subinfo[2](val))
                elif param is subinfo[0]:
                    # This parameter substitutes another; update the original
                    subp.set_value(subinfo[1](val))

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

        self.param_evol = {param: deque([param.get_value()])
                           for param in self.fitparams}
        self.logL_evol = deque([np.inf])

        self.step_i = 0
        self.curtidx = self.burnin_idx
        self.cum_step_time = 0
        self.tot_time = 0

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
            return all( (meandiff(it, end=-i) * s < a).all()
                        for i in range(1, m+1) )

    def step(self):
        if self.curtidx > self.burnin_idx + self.datalen - self.mbatch_size:
            # We've run through the dataset
            # Reset time index to beginning
            self.curtidx = self.burnin_idx
            self.model.clear_unlocked_histories()
            self.logL_evol.append(self.logL())
            self.model.clear_unlocked_histories()
        else:
            # TODO: Check what it is that is cleared here, and why we need to do it
            self.model.clear_other_histories()

        t1 = time.perf_counter()
        self._step(self.curtidx)
        self.cum_step_time += time.perf_counter() - t1
        for param in self.fitparams:
            self.param_evol[param].append(param.get_value())

        # Increment step counter
        self.step_i += 1

        # Check to see if there have been meaningful changes in the last 10 iterations
        if ( converged(self.logL_evol)
             and all( converged(self.param_evol[p]) for p in self.fitparams) ):
            logger.info("Converged. log L = {:.2f}".format(float(self.logL_evol[-1])))
            return ConvergeStatus.CONVERGED

        #Print progress
        clear_output(wait=True)
            #`wait` indicates to wait until something is printed, which avoids flicker
        logger.info("Iteration {:>{}} – log L = {:.2f}"
                    .format(self.step_i,
                            self.output_width,
                            float(self.logL_evol[-1])))
        self.curtidx += self.mbatch_size

        return ConvergeStatus.NOTCONVERGED

    def iterate(self, Nmax=int(5e3)):

        Nmax = int(Nmax)
        self.output_width = int(np.log10(Nmax))

        t1 = time.perf_counter()
        for i in range(Nmax):
            status = self.step()
            if status in [ConvergeStatus.CONVERGED, ConvergeStatus.ABORT]:
                break
        self.tot_time = time.perf_counter() - t1

        if status != ConvergeStatus.CONVERGED:
            print("Did not converge.")

        if i > 0:
            logger.info("Likelihood evaluation : {:.1f}s / {:.1f}s ({:.1f}% total "
                        "execution time)"
                        .format(self.cum_step_time,
                                self.tot_time,
                                self.cum_step_time / (self.tot_time) * 100))
            logger.info("Time per iteration: {}ms".format((self.tot_time)/self.step_i*1000))

        #with open("sgd-evol", 'wb') as f:
        #    pickle.dump((L_evol, param_evol), f)

    def get_evol(self):
        """
        Return a dictionary storing the evolution of the likelihood and parameters.

        Parameters
        ----------

        Returns
        -------
        dictionary:
             The log likelihood evolution is associated the string key 'logL'.
             Parameter evolutions are keyed by the parameters themselves.
             Each evolution is stored as an ndarray, with the first dimension
             corresponding to epochs.
        """
        evol = {param: np.array([val for val in self.param_evol[param]])
                for param in self.fitparams}
        evol['logL'] = np.array([val for val in self.logL_evol])
        return evol

    def stats(self):
        return {'number iterations': self.step_i,
                'total time (s)' : self.tot_time,
                'average step time (ms)': self.cum_step_time / self.step_i * 1000,
                'average time per iteration (ms)': self.tot_time / self.step_i * 1000,
                'time spent stepping (%)': self.cum_step_time / (self.tot_time) * 100}

    def plot_logL_evol(self):
        plt.title("Minimization of likelihood")
        plt.plot(self.get_evol()['logL'])
        plt.xlabel("epoch")
        plt.ylabel("log L")

    def plot_param_evol(self, ncols=3):

        nrows = int(np.ceil(len(self.fitparams) / ncols))
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
            plt.plot(evol[param].reshape(len(evol[param]), -1))
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
                                for i in range(len(evol[param])) ],
                              linestyle='dashed' )
                else:
                    logger.warning("Although ground truth parameters have been set, "
                                   "the value of '{}' was not.".format(param.name))

    def plot_param_evol_overlay(self, basedata, evol=None):
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
                                plotcoords.append(evol[param])
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

                                plotcoords.append(evol[param][idx])

                    if not found:
                        for param in self.substitutions:
                            # As above, but we also invert the variable transformation
                            if param.name == ax.name:
                                found = True
                                transformedparam = self.substitutions[param][0]
                                inversetransform = self.substitutions[param][1]
                                if shim.isscalar(param):
                                    plotcoords.append(inversetransform( evol[transformedparam] ))
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

                                    plotcoords.append(inversetransform( evol[transformedparam][idx] ))

                    if not found:
                        raise ValueError("The base data has a parameter '{}', which "
                                         "does not match any of the fit parameters."
                                         .format(ax.name))

                assert( len(plotcoords) == 2 )
                plt.plot(plotcoords[0], plotcoords[1])

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

def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
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
                assert(p[1].shape == p[0].get_value().shape)
                param_masks.append(p[1])
        else:
            tmpparams.append(p)
            param_masks.append(None)
    params = tmpparams

    updates = OrderedDict()
    grads = T.grad(cost, params)
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
