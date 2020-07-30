import numpy as np
from typing import Any

import mackelab_toolbox as mtb
import mackelab_toolbox.typing
from mackelab_toolbox.cgshim import shim
import sinn
from sinn.histories import (
    TimeAxis, HistoryUpdateFunction, Series, Spiketrain, AutoHist, NotComputed)
from sinn.models import Model, BaseModel, ModelParams, initializer, updatefunction

class TestModelNoState(Model):
    """
    A simple test model: 1D O-U process driving independent spiking neurons.
    """
    time :TimeAxis
    class Parameters(ModelParams):
    # TODO: Also test
    # class Parameters(BaseModel):
    # class Parameters(BaseParameters):
        τ :float
        σ :float
        N :int

    rng    :mtb.typing.AnyRNG
    spikes :Spiketrain = None
    λ :Series = AutoHist(name='λ', shape=(1,), dtype=np.float64, iterative=True)

    @initializer('spikes')
    def create_spikes(cls, s, time, N):
        return Spiketrain(name='s', time=time, pop_sizes=(N,), iterative=True)

    @updatefunction('spikes', inputs=['λ', 'rng'])
    def upd_s(self, tidx):
        dt = self.time.dt.magnitude
        size = self.spikes.shape
        if shim.isscalar(tidx):
            tidx = [tidx]
        # if not shim.isscalar(tidx):
        #     size = (len(tidx),) + size
        retval = [shim.nonzero(
                      self.rng.binomial( size = self.λ.shape,
                                         n = 1,
                                         p = sinn.clip_probabilities(self.λ(ti)*dt)
                                         )
                      )[0]
                  for ti in tidx]
        return retval

    @updatefunction('λ', inputs=['λ'])
    def upd_λ(self, tidx):
        dt = self.time.dt.magnitude
        return (self.λ(tidx-1) * np.exp(-dt/self.τ)
                + self.σ * np.sqrt(dt) * self.rng.normal())

    def initialize(self, initializer=None):
        self.λ.pad(1)

class TestModel(TestModelNoState):
    class State:
        λ :Any
