"""Collection of stock non-iterative histories, i.e. histories that do not depend on others
and can be batch-computed. This makes them suitable as external inputs.
Since these essentially behave as functions, we refer to them as 'history functions'.

To define a new history function, make a new class which derives from
IterativeSeries and define two functions:
  - init_params(self, [random_stream], param1, param2...) -> shape
    Takes the function parameters and stores them in the instance. This is to allow them
    being called in the definition of the update function. The way parameters are stored
    is arbitrary.
    This function is also the place to perform sanity checks on the parameters.
  - update_function(self, t)
    A History update_function, which will be past to the history's `set_update_function`
    method. It has access to whatever parameter storage was created in `set_params`.

History functions requiring a random number generator should also set the class attribute
`requires_rng` to True. A random stream object will then be expected to be passed as first
argument to the initializer, which can then be stored for use by `update_function`.
"""

import numpy as np
import logging
logger = logging.getLogger("sinn.models.inputs")

import theano_shim as shim
import sinn
from sinn.histories import Series

class SeriesFunction(Series):
    requires_rng = False

    def __init__(self, ref_hist=sinn._NoValue, name=None, t0=sinn._NoValue, tn=sinn._NoValue, dt=sinn._NoValue,
                 **kwargs):

        shape = self.init_params(**kwargs)
        if shape is None:
            raise ValueError("A history function's `init_params` method must return shape.")
        super().__init__(hist=ref_hist, name=name, t0=t0, tn=tn, dt=dt, shape=shape, iterative=False)
        self.set_update_function(self.update_function)

class Sin(SeriesFunction):
    # TODO: Allow Theano parameters

    def init_params( self, baseline=0, amplitude=None, frequency=None, period=None, phase=0, unit='Hz'):
        """
        If 'unit' is 'Hz', frequency gives the number of cycles per unit time,
        while phase is a shift in numbers of cycles. I.e. both frequency and phase
        are multiplied by 2π before being given to sin().
        """

        if amplitude is None:
            raise ValueError("'amplitude' is a required parameter.")

        if frequency is None and period is None:
            raise ValueError("Either frequency or period must be specified to create a sin input")

        self.baseline = baseline

        if unit.lower() == 'radians':
            self.amplitude = amplitude
            self.phase = phase
            self.frequency = frequency if frequency is not None else 2*np.pi/period
            if period is not None and frequency != 2*np.pi/period:
                raise ValueError("When creating a sin input, don't specify both frequency and period.")
        elif unit.lower() == 'hz':
            self.amplitude = amplitude
            self.phase = phase * 2 * np.pi
            self.frequency = frequency * 2*np.pi if frequency is not None else 2*np.pi/period
            if period is not None and frequency != 2*np.pi/period:
                raise ValueError("When creating a sin input, don't specify both frequency and period.")
        else:
            raise ValueError("When creating a sin input, 'unit' must be either "
                             "'radians' or 'Hz', not '{}'".format(unit))

        try:
            shape = np.broadcast(self.baseline, np.broadcast(
                self.amplitude, np.broadcast(self.frequency, self.phase))).shape
        except ValueError:
            raise ValueError("The parameters 'amplitude', 'frequency' and 'phase' have incompatible "
                             "dimensions: {}, {}, {}."
                             .format(self.amplitude.shape, self.frequency.shape, self.phase.shape))
        return shape

    def update_function(self, t):
        if not shim.isscalar(t):
            ndim = len(self.shape)
            if ndim == 0:
                ndim = 1
            t = shim.add_axes(t, ndim, 'after')
        return self.baseline + self.amplitude * shim.sin(self.frequency*t + self.phase)


class GaussianWhiteNoise(SeriesFunction):
    """
    Values will limited to the range ±clip_limit;
    """
    requires_rng = True

    def init_params(self, random_stream, shape, std=1.0, clip_limit=87):
        # exp(88) is the largest value scipy can store in a 32-bit float
        self.rndstream = random_stream
        self.std = shim.cast_floatX(std)
        self.clip_limit = np.int8(clip_limit)
        try:
            np.broadcast(std, np.ones(shape))
        except ValueError:
            raise ValueError("The shape of parameter 'std' is incompatible "
                             "with the requested output shape: "
                             "std: {}, shape: {}.".format(std, shape))

        return shape

    def update_function(self, t):
        # 'shape' and 'clip_limit' are not parameters we want to treat
        # with Theano, and since all parameters are shared, we use
        # `get_value()` to get a pure NumPy value.
        if shim.isscalar(t):
            outshape = self.shape
        else:
            assert(t.ndim==1)
            outshape = t.shape + self.shape
        return shim.clip(self.rndstream.normal(avg  = 0,
                                               std  = self.std/np.sqrt(self.dt),
                                               size = np.int32(outshape)),
                                                      # Theano expects array
                         -self.clip_limit,
                         self.clip_limit)

