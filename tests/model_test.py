# -*- coding: utf-8 -*-
"""
Created on Sun May 31 2020

Author: Alexandre René

Manifest:
    test_model()

TODO:
    - Test models with mixins
        + Esp. that kernel_identifiers, hist_identifiers and update functions
          are extracted correctly, even when some of the mixin methods replace
          those of the original model.
"""

import numpy as np
import theano_shim as shim
import pint
ureg = pint.UnitRegistry()
import mackelab_toolbox as mtb
import mackelab_toolbox.typing
import mackelab_toolbox.cgshim
# Deactivate the graph caches for tests
import mackelab_toolbox.theano
import mackelab_toolbox.serialize
mackelab_toolbox.theano.GraphCache.activated = False
mackelab_toolbox.theano.CompiledGraphCache.activated = False
mtb.typing.PintUnit.ureg = ureg

import pytest
from conftest import clean_theano_dir
import scipy.sparse

# Change current directory to sinn/tests
def cd_to_test_dir():
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
    cd_to_test_dir()
    mtb.serialize.config.trust_all_inputs = True

    with pytest.warns(UserWarning):
        # Warns that Model has no 'State' class
        from model_test_classes import TestModelNoState
    # NOTE: TestModel subclasses TestModelNoState, so we are also testing
    #       model subclassing
    from model_test_classes import TestModel
    from sinn.histories import TimeAxis

    TimeAxis.time_unit = ureg.s
    TimeAxis.time_step = np.float64(2**-6)

    model = TestModel(params = TestModel.Parameters(λ0=10, τ=1, σ=100, N=7),
                      time = TimeAxis(min=0, max=10),
                      rng = shim.config.RandomStream())

    # Compute series history a few time steps
    model.λ(2*model.dt)

    # Compute series history to a particular time step, using model time axis
    model.λ(model.time.Index(3))

    # Evaluate symbolic updates
    model.λ.eval()
    assert model.λ.cur_tidx == 3

    # Evaluate spikes using already computed λ
    model.spikes(3)

    # Evaluate more spikes, triggering some λ computations for tidx>3
    model.spikes(4)

    # Evaluate all the histories in the model
    model.eval()
    assert model.spikes.get_data_trace().shape == (5,7)
    assert len(model.λ.get_data_trace()) == 5

    end = model.time.Index(64)
    # end = model.time.tnidx
    model.integrate(end, histories=(model.spikes,))

    ## Serialization ##

    # FIXME: Support & test serialization of the CSMProperties attribute of Theano sparse array
    if cgshim == 'theano':
        return

    # Deserializes without error
    # with pytest.warns(scipy.sparse.SparseEfficiencyWarning):
    #     # We don't want to deal with the efficiency warning atm.
    model2 = TestModel.parse_raw(model.json())

    # Deserialized model has the same data
    model_compare(model, model2)

    # Deserialized model is distinct (no shared vars with original)
    assert model.λ.update_function.namespace is not model2.λ.update_function.namespace
    assert model.spikes.update_function.namespace is not model2.spikes.update_function.namespace

    model2.clear()
    assert model.cur_tidx == end
    assert model2.cur_tidx == -1

    # Update functions of the deserialized work as expected

    # TODO: Reset RNG and confirm that result is identical to original
    assert len(model2.spikes._num_data[0].get_value()) == 0
        # Ensure that clearing model2 removed all data, so that when we test
        # with 'sum' below, we aren't just reusing deserialized values.
    model2.integrate(30, histories=model2.spikes)
    assert model2.spikes.data.sum() > 0

def model_compare(model1, model2):
    cd_to_test_dir()
    from history_test import hist_compare
    from sinn.histories import History
    model1.cur_tidx != model2.cur_tidx
    model1.cur_tidx.plain == model2.cur_tidx.plain
    model1.time == model2.time
    model1.time.label == model2.time.label
    model1.time.pad_left != model2.time.pad_left
    model1.time.pad_right != model2.time.pad_right
    model1.time.pad_left.plain == model2.time.pad_left.plain
    model1.time.pad_right.plain == model2.time.pad_right.plain
    set([h.name for h in model1.history_set]) == set([h.name for h in model2.history_set])
    set([h.name for h in model1.statehists]) == set([h.name for h in model2.statehists])
    for attr in (attr for attr,v in model1.__dict__.items() if isinstance(v, History)):
        hist_compare(getattr(model1, attr), getattr(model2, attr))

def test_model_theano(clean_theano_dir):
    return _test_model('theano')
def test_model_numpy():
    return _test_model('numpy')

if __name__ == "__main__":
    # test_model_numpy()
    test_model_theano()
