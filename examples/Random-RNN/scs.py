import numpy as np
from numbers import Number
import mackelab_toolbox as mtb
from typing import Any
from mackelab_toolbox.typing import Array
from mackelab_toolbox.cgshim import shim

import sinn
from sinn.histories import TimeAxis, Series, AutoHist
from sinn.models import BaseModel, Model, validator, initializer, updatefunction

class SCS(Model):
    """
    The Sompolinsky-Crisanti-Sommers (SCS) model.

    In applications, you may want to add a read-in / read-out vectors.

    The SCS model is defined as follows:

    .. math::

       \dot{h}_i = -h_i + \sum_{j=1}^N J_{ij} \phi(h_j)

       \phi(h_j) = \tanh(g h_j)

    .. rubric:: Integration scheme

       Updates to `h` are computed with first-order Euler.

    Parameters
    ----------
    params: model Parameters (See `Model Parameters`)



    .. rubric:: Initializer
       Sets the value of `h` at time ``0-dt``.

       - If a numerical value (float or ndarray): Initialize `h` to with value.
       - Accepted string values:
         + 'zero' (default): All `h` values are zero.

    """

    time :TimeAxis

    class Parameters(BaseModel):
        """
        Parameters
        ----------
        N: int
           Number of neurons in the network
        J: Array (NxN)
           Connectivity matrix
        g: float
           Gain within the ``tanh`` nonlinearity.
        """
        N :int
        J :Array[float,2]  # A 2D array of floats
        g :float

        @validator('J')
        def J_is_NxN(cls, J, values):
            """Ensure the shape of J is consistent with N."""
            N = values.get('N', None);
            if N is None: return
            assert shim.eval(J.shape) == (N,N)
            return J

    params :Parameters
    h      :Series = None  # Set by initializer

    class State:
        h  :Any

    @initializer('h')
    def init_h(cls, v, time, N):
        return Series(name='h', time=time, shape=(N,), dtype=np.float64)

    def initialize(self, initializer='zero'):
        self.h.pad(1)
        if isinstance(initializer, (Number, np.ndarray)):
            self.h[-1] = initializer

    def φ(self, h):
        return shim.tanh(self.g*h)

    @updatefunction('h', inputs=['h'])
    def upd_h(self, tidx):
        dt = self.time.dt
        h = self.h; J = self.J; φ = self.φ
        return (1-dt)*h[tidx-1] + dt * shim.dot(J, φ(h[tidx-1]) )
