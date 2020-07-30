# -*- coding: utf-8 -*-
"""
Created on Sun May 31 2020

Author: Alexandre René

Manifest:
    test_model()

"""

import numpy as np
import theano_shim as shim
import pint
ureg = pint.UnitRegistry()
import mackelab_toolbox as mtb
import mackelab_toolbox.typing

import pytest

# Change current directory to sinn/tests
if __name__ == "__main__":
    import sys, os
    from pathlib import Path
    import sinn
    os.chdir(Path(sinn.__file__).parent.parent/'tests')

def _test_model(cgshim):
    # All sinn imports need to be guarded inside functions, to avoid
    # shim state being messed up
    shim.load(cgshim)
    mtb.typing.freeze_types()
    import sinn

    with pytest.warns(UserWarning):
        # Warns that Model has no 'State' class
        from model_test_classes import TestModelNoState
    # NOTE: TestModel subclasses TestModelNoState, so we are also testing
    #       model subclassing
    from model_test_classes import TestModel
    from sinn.histories import TimeAxis


    TimeAxis.time_unit = ureg.s
    TimeAxis.time_step = np.float64(2**-6)

    model = TestModel(params = TestModel.Parameters(τ=1, σ=100, N=7),
                      time = TimeAxis(min=0, max=10),
                      rng = shim.config.RandomStreams())

    # Compute series history a few time steps
    model.λ(2*model.dt)

    # Compute series history to a particular time step, using model time axis
    model.λ(model.time.Index(3))

    # Evaluate symbolic updates
    model.λ.eval()
    assert model.λ.cur_tidx == 3

    # Evaluate spikes using already computed λ
    model.spikes(3)

    # Evaluate more spikes, triggering some λ computations for tidx>5
    model.spikes(4)

    # Evaluate all the histories in the model
    model.eval()
    assert model.spikes.get_trace().shape == (5,7)
    assert len(model.λ.get_trace()) == 5

    # model.spikes.(model.time.tnidx)
    model.advance(model.time.tnidx)


def test_model_theano():
    return _test_model('theano')
def test_model_numpy():
    return _test_model('numpy')

if __name__ == "__main__":
    # test_model_numpy()
    test_model_theano()
