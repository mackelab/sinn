# -*- coding: utf-8 -*-
"""
Created Fri Feb 24 2017

author: Alexandre René
"""
import logging
import os
from collections import namedtuple, OrderedDict, Iterable
from copy import copy
import numpy as np
logger = logging.getLogger('sinn.sweep')

import theano_shim as shim
import sinn
import sinn.iotools as io
from .heatmap import HeatMap

AxisStops = namedtuple('AxisStops', ['stops', 'scale', 'linearize_fn', 'inverse_linearize_fn'])


__ALL__ = ['linspace',
           'logspace',
           'ParameterSweep'
           ]

def linspace(low, high, fineness):
    """Simple wrapper around numpy.linspace with a more easily tunable
    `fineness` parameter rather than `num`. Doubling the fineness
    doubles the number of points, and specifying the same fineness across
    different *space functions will lead to consistent results.
    For `linspace`, fineness is the number of stops per increase of 1.
    """
    # Note: lambdas don't play nice with pickling (and thence ipyparallel)
    # def noop(x):
    #     return x
    return AxisStops(np.linspace(low, high, int((high-low)*fineness),
                                  dtype=sinn.config.floatX),
                     'linear',
                     'x -> x', 'x -> x')


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
    # def pow10(x):
    #     return 10**x
    return AxisStops(np.logspace(np.log10(low), np.log10(high),
                                 num=int((np.log10(high)-np.log10(low))
                                         * 5*fineness/np.log10(10)),
                                     #5*fineness per decade
                                 base=10,
                                 dtype=sinn.config.floatX),
                     'log',
                     'x -> np.log10(x)', 'x -> 10**x')


class ParameterSweep:
    """
    When parallelizing, the following packages are imported by default in each process:
    - sinn
    - theano_shim as shim
    - numpy as np
    Other packages can be added to the list by calling `add_import`.
    """

    def __init__(self, model):
        self._model = model
        self.params_to_sweep = []
        self.imports = ['sinn', 'sinn.config', ('theano_shim', 'shim'), ('numpy', 'np')]
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
        if name not in self._model.Parameters._fields:
            raise ValueError("ParameterSweep: {} is not a model parameter."
                             .format(name))
        self.params_to_sweep.append(HeatMap.ParameterAxis(
            name, label_idx=idx, stops=axis_stops.stops, format='centers',
            transform_fn = axis_stops.linearize_fn,
            inverse_transform_fn = axis_stops.inverse_linearize_fn))
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

    def do_sweep(self, output_filename, ippclient=None, debug=False):
        """
        Use 'output_filename = None' to indicate not to save the output.
        """

        # variables that will be defined in the engine processes
        model = self._model
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

        def sweep_f(param_tuples):

            loginfo("Evaluating sweep at {}".format(str(param_tuples)))

            # First update the model with the new parameters
            # `model` is a global variable in the process module.
            # It must be initialized before calls to sweep_f()
            new_params_dict = OrderedDict( (key, val.get_value())
                                           for key, val in model.params._asdict().items())
            for param, val in zip(params_to_sweep, param_tuples):
                if param.idx is None:
                    param_val = val
                else:
                    idx = param.idx
                    if isinstance(idx, Iterable):
                        # Indexing rules for arrays and lists are different than for tuples
                        # We want to treat them all as tuples
                        idx = tuple(idx)

                    try:
                        param_val = copy(new_params_dict[param.name])
                        # Copying ensures that we have a new reference,
                        # ensuring that the model parameters are unchanged.
                        # Otherwise the kernel may think it hasn't changed,
                        # and would then not update itself.
                    except AttributeError:
                        raise ValueError(
                            "You are trying to sweep over the parameter "
                            "{}, which is not a parameter of model {}."
                            .format(param.name, str(model)))
                    try:
                        param_val[idx] = val
                    except IndexError:
                        # The parameter might have an extra dimension
                        # for broadcasting – try without it
                        # TODO Currently will break for params with dim>2
                        if param_val.shape[0] == 1:
                            param_val[0, idx] = val
                        elif param_val.shape[-1] == 1:
                            param_val[idx, 0] = val
                        else:
                            # Nope, idx really is incompatible
                            raise
                new_params_dict[param.name] = param_val
                #new_params = new_params._replace(**{param.name: param_val})
            model.update_params(model.Parameters(**new_params_dict))
                # update_params also resets all histories
            if hasattr(model, 'initialize'):
                model.initialize()

            # Now execute and return the function of interest
            loginfo("Evaluating likelihood function.")
            return self.function(model)

        if ippclient is not None:
            # Set up the environment on each client
            ippclient[:].block = True
            ippclient[:].execute("import os")
            ippclient[:].execute("os.chdir('" + self.workdir + "')")
            for pkgname in self.imports:
                if isinstance(pkgname, tuple):
                    ippclient[:].execute("import " + pkgname[0])
                    ippclient[:].execute(pkgname[1] + " = " + pkgname[0])
                else:
                    ippclient[:].execute("import " + pkgname)
            if sinn.config.use_theano():
                ippclient[:].execute("sinn.config.load_theano()")
                ippclient[:].execute("shim.theano.config.exception_verbosity = '{}'"
                                     .format(shim.gettheano().config.exception_verbosity))
                ippclient[:].execute("shim.theano.config.optimizer = '{}'"
                                     .format(shim.gettheano().config.optimizer))

            ippclient[:].scatter('idnum', ippclient.ids, flatten=True, block=True)

            ippclient[:].execute("sinn.inputs.clear()")
                # Clear out leftover inputs from a previous calculation

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
            ippclient[:].block = False
            res_arr_raw = ippclient[:].map_async(sweep_f, self.param_list())
            monitor_async_result(res_arr_raw)
            if not debug:
                res_arr = np.array(res_arr_raw.get()).reshape(self.shape)
            else:
                raise NotImplementedError
        else:
            os.chdir(self.workdir)

            if not debug:
                res_arr = np.fromiter(map(sweep_f, self.param_list()), sinn.config.floatX).reshape(self.shape)
            else:
                # For a debug run, no point in repeating the computation for different parameters
                res = sweep_f(next(self.param_list()))

        if not debug:
            res = HeatMap(self.function_label, 'density', res_arr, self.params_to_sweep)
            if output_filename is not None:
                io.save(output_filename, res)
        else:
            # Let the caller decide how to save the debug data
            pass

        return res

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
                    #TODO output may contain \n; split and output multiple lines, each with engine no.

                    # set stdout0 to last output for new comparison
                    stdout0[i] =  async_result.stdout[i]
        else:
            continue
