from collections import Callable, deque, namedtuple
import numpy as np
import mackelab as ml
from odictliteral import odict

import theano_shim as shim

import sinn
import sinn.models as models
from sinn.models import Model
from sinn.histories import Series
from sinn.history_functions import GaussianWhiteNoise

class DummyQuad(Model):
    """
    A minimal model for debugging.  Takes the square of the input and adds
    Gaussian noise
    Input: $I(t)$,  $ξ(t)$ : Gaussian white noise
    $X(t) = X(t-dt) + α I(t)^2 dt + σ ξ(t)$
    """
    requires_rng = False
    Parameters_info = odict['α':'floatX', 'σ':'floatX']
    Parameters = sinn.define_parameters(Parameters_info)
    State = namedtuple('State', ['X'])

    def __init__(self, params, X, I, rng=None):
        # Sanity checks
        assert(isinstance(X, Series))
        assert(isinstance(I, Series))
        sinn.models.Model.same_dt(X, I)
        sinn.models.Model.output_rng(X, rng)

        # Call parent initializer
        super().__init__(params,
                         public_histories=(X, I),
                         reference_history=X)

        # Make X, I into class attributes, so we can access them in methods
        self.X = X
        self.I = I
        self.rng = rng

        # Create the internal histories
        # `X` is passed as a template: defines default time stops & shape
        self.Xbar = Series(X, name='Xbar', shape=X.shape)
        self.ξ = GaussianWhiteNoise(X, name='ξ', random_stream=self.rng,
                                    shape=X.shape, std=1.)

        # Attache the histories to the model
        self.add_history(self.X)
        self.add_history(self.Xbar)

        # Attach the update functions to the internal histories
        self.X.set_update_function(self.X_fn)
        self.Xbar.set_update_function(self.Xbar_fn)

        # Tell histories which other histories they depend on
        self.X.add_inputs([self.Xbar, self.ξ])
        self.Xbar.add_inputs([self.X])

        # Pad the state histories to make space for the initial state
        self.X.pad(1)

    # ---------- Update equations --------- #

    def Xbar_fn(self, t):
        ti = self.Xbar.get_tidx_for(t, self.I); I = self.I[ti]
        tX = self.Xbar.get_tidx_for(t, self.X); X0 = self.X[tX-1]
        dt = self.Xbar.dt
        return X0 + self.α*I*dt

    def X_fn(self, t):
        tXb = self.X.get_tidx_for(t, self.Xbar); Xbar = self.Xbar[tXb]
        tξ  = self.X.get_tidx_for(t, self.ξ)   ; ξ = self.ξ[tξ]
        return Xbar + self.σ*ξ

    # --------- Log likelihood ----------- #

    @models.batch_function_scan('Xbar', 'X')
    def logp(self, Xbar, X):
        return shim.sum((X-Xbar)**2)

sinn.models.register_model(DummyQuad)
