# -*- coding: utf-8 -*-
"""
Created Fri Feb 24 2017

author: Alexandre René
"""
import logging
import os
from collections import namedtuple
import numpy as np
logger = logging.getLogger('sinn.sweep')

import sinn
import sinn.iotools as io
from sinn.analyze.heatmap import HeatMap

AxisStops = namedtuple('AxisStops', ['stops', 'linearize_fn', 'inverse_linearize_fn'])

def linspace(low, high, fineness):
    """Simple wrapper around numpy.linspace with a more easily tunable
    `fineness` parameter rather than `num`. Doubling the fineness
    doubles the number of points, and specifying the same fineness across
    different *space functions will lead to consistent results.
    For `linspace`, fineness is the number of stops per increase of 1.
    """
    # Note: lambdas don't play nice with pickling (and thence ipyparallel)
    def noop(x):
        return x
    return AxisStops(np.linspace(low, high, int((high-low)*fineness),
                                  dtype=sinn.config.floatX),
                     noop, noop)


def logspace(low, high, fineness):
    """Simple wrapper around numpy.logspace with a more easily tunable
    `fineness` parameter rather than `num`. Doubling the fineness
    doubles the number of points, and specifying the same fineness across
    different *space functions will in most cases lead to consistent results.
    For `logspace`, fineness is 1/5th the number of stops per decade.
    *Note* In contrast to numpy.logspcae, `low` and `high` here are the actual
    values of the bounds, not the corresponding exponents to 10.
    """
    # Note: lambdas don't play nice with pickling (and thence ipyparallel)
    def pow10(x):
        return x**10
    return AxisStops(np.logspace(np.log10(low), np.log10(high),
                                 num=int((np.log10(high)-np.log10(low)))
                                 * 5*fineness/np.log10(10),
                                     #5*fineness per decade
                                 base=10,
                                 dtype=sinn.config.floatX),
                     np.log10, pow10)


class ParameterSweep:

    def __init__(self, model):
        self.model = model
        self.params_to_sweep = []
        self.imports = []
        self.function = None
        self.shape = ()
        self.workdir = os.getcwd()

    def add_import(self, pkgname):
        if isinstance(pkgname, str):
            # Single package name
            self.imports.append(pkgname)
        else:
            # List of package names
            for name in pkgname:
                assert(isinstance(name, str))
                self.imports.append(name)

    def set_directory(self, dirname):
        assert(isinstance(dirname, str))
        self.workdir = os.abspath(dirname)

    def add_param(self, name, axis_stops, idx=None):
        self.params_to_sweep.append(HeatMap.ParameterAxis(name, axis_stops.stops, idx, axis_stops.linearize_fn, axis_stops.inverse_linearize_fn))
        self.shape += (len(axis_stops.stops),)

    def set_function(self, function, label):
        self.function = function
        self.function_label = label

    def param_list(self):
        # Flatten all parameter combinations into one long list
        # The last parameter added is the inner loop
        inner_len = np.cumprod((1,) + self.shape[:0:-1])[::-1]
        return ( [ p.stops[i // il % len(p.stops)]
                   for p, il in zip(self.params_to_sweep, inner_len)]
                 for i in range(np.prod(self.shape)) )

    def do_sweep(self, output_filename, ippclient=None):

        # variables that will be defined in the engine processes
        model = self.model
        params_to_sweep = self.params_to_sweep

        # Create a logger interface
        # Rather than connect the engines to a logging instance, we send their
        # messages to stdout, catch it in the controlling process (this module)
        # and write out to the latter's log file
        # TODO Distinguish info, error, warning… messages
        # TODO Direct output to logfiles directly from engines

        if ippclient == None:
            # Not using ipyparallel, so just use logger directly
            def loginfo(msg):
                logger.info(msg)
        else:
            def loginfo(msg):
                print(msg)

        def f(param_tuples):

            # First update the model with the new parameters
            loginfo("Updating model with new parameters {}".format(param_tuples))
            # `model` is a global variable in the process module.
            # It must be initialized before calls to f()
            new_params = model.params
            for param, val in zip(params_to_sweep, param_tuples):
                if param.idx is None:
                    param_val = val
                else:
                    try:
                        param_val = getattr(new_params, param.name)
                    except AttributeError:
                        raise ValueError(
                            "You are trying to sweep over the parameter "
                            "{}, which is not a parameter of model {}."
                            .format(param.name, str(model)))
                    try:
                        param_val[param.idx] = val
                    except IndexError:
                        # The parameter might have an extra dimension
                        # for broadcasting – try without it
                        # TODO Currently will break for params with dim>2
                        if param_val.shape[0] == 1:
                            param_val[0, param.idx] = val
                        elif param_val.shape[-1] == 1:
                            param_val[param.idx, 0] = val
                        else:
                            # Nope, param.idx really is incompatible
                            raise
                new_params._replace(**{param.name: param_val})
            model.update_params(new_params)
            if hasattr(model, 'initialize'):
                model.initialize()

            # Now execute and return the function of interest
            return self.function(model)

        if ippclient is not None:
            # Set up the environment on each client
            ippclient[:].execute("import os")
            ippclient[:].execute("os.chdir('" + self.workdir + "')")
            for pkgname in self.imports:
                ippclient[:].execute("import " + pkgname)

            ippclient[:].scatter('idnum', ippclient.ids, flatten=True, block=True)
            # Push the model to each engine's <globals> namespace
            ippclient[:].push({'model': self.model,
                               'params_to_sweep': self.params_to_sweep})

            # # Set up logger
            # from sinn.common import _logging_formatter
            # ippclient[:].execute("import logging")
            # ippclient[:].execute("logger = logging.getLogger('sinn.analyze.sweep-engine' + str(id))")
            # ippclient[:].execute("logger.setLevel(" + str(sinn.config.logLevel) + ")")
            # ippclient[:].execute("_fh = logging.handlers.RotatingFileHandler('sweep_' + '" + str(os.getpid()) + "' + '_engine' + str(idnum) + '.log', mode='w', maxBytes=5e7, backupCount=5)")
            # ippclient[:].execute("setLevel(logging.DEBUG)")
            # ippclient[:].push({'_logging_formatter': _logging_formatter})
            # ippclient[:].execute("_fh.setFormatter(_logging_formatter)")
            # ippclient[:].execute("logger.addHandler(_fh)")

            res_arr_raw = ippclient[:].map_async(f, self.param_list())
            monitor_async_result(res_arr_raw)
            res_arr = np.array(res_arr_raw.get()).reshape(self.shape)
        else:
            os.chdir(self.workdir)

            res_arr = np.fromiter(map(f, self.param_list()), sinn.config.floatX).reshape(self.shape)

        res = HeatMap(self.function_label, res_arr, self.params_to_sweep)

        io.save(output_filename, res)

def monitor_async_result(async_result):
    # Based on http://stackoverflow.com/a/40975521
    # initialize a stdout0 array for comparison
    stdout0 = async_result.stdout
    if any(msg != '' for msg in stdout0):
        print(stdout0)

    while not async_result.ready():
        # check if stdout changed for any engine
        if async_result.stdout != stdout0:
            for i in range(0,len(async_result.stdout)):
                if async_result.stdout[i] != stdout0[i]:
                    # print only new stdout's without previous message and remove '\n' at the end
                    logger.info('(Engine ' + str(i) + ') ' + async_result.stdout[i][len(stdout0[i]):-1])

                    # set stdout0 to last output for new comparison
                    stdout0[i] =  async_result.stdout[i]
        else:
            continue
