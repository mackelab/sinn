# -*- coding: utf-8 -*-
"""
Created on Sun May 03 2020

Author: Alexandre René

Manifest:
    test_history_series_indexing()
    test_history_spiketrain_indexing()
    test_history_series_updates()
    test_history_spiketrain_updates()

"""

import numpy as np
import scipy.sparse
from types import SimpleNamespace
from pydantic import ValidationError

import theano_shim as shim
import mackelab_toolbox as mtb
import mackelab_toolbox.serialize
from mackelab_toolbox.transform import Transform
import mackelab_toolbox.typing
import mackelab_toolbox.cgshim
import pint
ureg = pint.UnitRegistry()

# Series.__fields__

import pytest

# import sys
# sys.path.append('/home/alex/Recherche/macke_lab/code/sinn/tests')
# import axis_test

# ===================
# Indexing tests
# These test access by axis index, padding, cur_tidx & tnidx properties

def _test_history_series_indexing(cgshim):

    shim.load(cgshim)
    mtb.typing.freeze_types()
    from sinn.histories import (
        TimeAxis, HistoryUpdateFunction, Series, Spiketrain, NotComputed)

    TimeAxis.time_unit = ureg.s
    TimeAxis.time_step = np.float64(2**-8)   #~0.0039. Powers of 2 are more numerically stable
    taxis = TimeAxis(min=0., max=1.)

    # We actually don't need to call `taxis.copy()`, because Pydantic copies all inputs
    x1 = Series(name='x1', time=taxis, shape=(3,), dtype=np.float64)
    x2 = Series(name='x2', time=taxis, shape=(2,), dtype=np.float64)
    assert x1.cur_tidx == -1
    assert x1.tnidx == len(taxis)-1 == len(x1)-1

    x1[:] = np.random.uniform(0, 1, size=(x1.time.padded_length,3))
    assert x1.tnidx == x1.cur_tidx
    assert np.all(shim.eval(x1[x1.tnidx]) == x1.data[x1.tnidx])

    assert x1.cur_tidx == taxis.unpadded_length - 1

    with pytest.warns(UserWarning):
        # Padding non-empty history invalidates data
        x1.pad(4)
    assert x1.cur_tidx == -5

    # Padding should not affect detached taxis, or axis from another History
    assert x1.pad_left == 4
    assert x1.time.padded_length == x1.time.unpadded_length + 4
    assert x2.pad_left == x2.pad_right == 0
    assert taxis.pad_left == taxis.pad_right == 0
    assert taxis.padded_length == taxis.unpadded_length

    x1[:] = np.random.uniform(0, 1, size=(x1.time.padded_length,3))

    assert x1.cur_tidx == taxis.unpadded_length - 1

    x1.pad(4)  # No warning because no padding was added
    with pytest.warns(UserWarning):
        # Adding padding triggers warning again
        x1.pad(5)

    x1[:] = np.random.uniform(0, 1, size=(x1.time.padded_length,3))

    first10 = x1._num_data.get_value()[:10]  # 5 padding + 5 data
    # `.data` omits padding
    assert np.all(first10[5:] == x1.data[:5])
    # negative indexing in axis space works
    assert np.all(shim.eval(x1[-1]) == first10[4])       # With single index
    assert np.all(shim.eval(x1[-1:2]) == first10[4:7])   # With slice index

    # Pydantic methods
    x1copy = x1.copy()
    assert x1copy._sym_data is x1copy._num_data
    assert x1copy._sym_tidx is x1copy._num_tidx

def _test_history_spiketrain_indexing(cgshim):
    shim.load(cgshim)
    mtb.typing.freeze_types()
    from sinn.histories import (
        TimeAxis, HistoryUpdateFunction, Series, Spiketrain, NotComputed)

    TimeAxis.time_unit = ureg.s
    TimeAxis.time_step = np.float64(2**-8)   #~0.15. Powers of 2 are more numerically stable
    taxis = TimeAxis(min=0., max=1.)

    # We actually don't need to call `taxis.copy()`, because Pydantic copies all inputs
    x1 = Spiketrain(name='x1', time=taxis, pop_sizes=(3,2,4))
    x2 = Spiketrain(name='x2', time=taxis, pop_sizes=(1,2,1))
    assert x1.cur_tidx == -1
    assert x1.tnidx == len(taxis)-1 == len(x1)-1
    # sizes of data, indices and indptr are 0, 0, (no. time bins)
    assert x1._num_data[0].shape == (0,)
    assert x1._num_data[1].shape == (0,)
    assert x1._num_data[2].shape == (x1.time.padded_length + 1,)

    def gen_spike_data(num_timepoints, shape, p=0.03):
        return [np.where(np.random.binomial(n=1, p=p, size=shape))[0]
                for i in range(num_timepoints)]
    x1[:] = gen_spike_data(x1.time.padded_length, 9)
    assert x1.tnidx == x1.cur_tidx
    assert np.all(shim.eval(x1[x1.tnidx]) == x1.data[x1.tnidx])

    assert x1.cur_tidx == taxis.unpadded_length - 1

    with pytest.warns(UserWarning):
        # Padding non-empty history invalidates data
        x1.pad(4)
    assert x1.cur_tidx == -5

    # Padding should not affect detached taxis, or axis from another History
    assert x1.pad_left == 4
    assert x1.time.padded_length == x1.time.unpadded_length + 4
    assert x2.pad_left == x2.pad_right == 0
    assert taxis.pad_left == taxis.pad_right == 0
    assert taxis.padded_length == taxis.unpadded_length
    assert x1._num_data[2].get_value().shape[0] == x1.time.padded_length + 1
        # Effectively the same test as the one below, but bypasses the sparse
        # array creation. Scipy itself checks that the shape is consistent
        # when it creates the sparse array, raising its own error.
    assert x1._get_num_csr().shape[0] == x1.time.padded_length

    # with pytest.warns(scipy.sparse.SparseEfficiencyWarning): # Silence expected warning
    x1[:] = gen_spike_data(x1.time.padded_length, 9)
    assert x1.cur_tidx == taxis.unpadded_length - 1

    x1.pad(4)  # No warning because no padding was added
    with pytest.warns(UserWarning):
        # Adding padding triggers warning again
        x1.pad(5)

    # with pytest.warns(scipy.sparse.SparseEfficiencyWarning): # Silence expected warning
    x1[:] = gen_spike_data(x1.time.padded_length, 9)

    first10 = x1._get_num_csr()[:10]  # 5 padding + 5 data
    # `.data` omits padding
    assert not (first10[5:] != x1.data[:5]).data.any()
    # negative indexing in axis space works
    assert np.all(x1[-1] == first10[4])       # With single index
    assert np.all(x1[-1:2] == first10[4:7])   # With slice index

    # Pydantic methods
    x1copy = x1.copy()
    assert x1copy._sym_data is x1copy._num_data
    assert x1copy._sym_tidx is x1copy._num_tidx

# test_history_spiketrain_indexing()

# =======================
# Update tests
# These test the definition of update functions, including with cross-dependencies

def _test_history_series_updates(cgshim):
    shim.load(cgshim)
    mtb.typing.freeze_types()
    from sinn.histories import (
        TimeAxis, HistoryUpdateFunction, Series, Spiketrain, NotComputed)

    Ti = 4  # Time point up to which to compute histories

    TimeAxis.time_unit = ureg.s
    TimeAxis.time_step = np.float64(2**-8)   #~0.15. Powers of 2 are more numerically stable
    taxis = TimeAxis(min=0., max=1.)

    x1 = Series(name='x1', time=taxis, shape=(3,), dtype=np.float64, iterative=False)
    x2 = Series(name='x2', time=taxis, shape=(3,), dtype=np.float64)

    hists = SimpleNamespace(x1=x1, x2=x2)
    # HistoryUpdateFunction.namespace = hists

    A = np.array([-0.1, 1.8, 0.5])
    x1_upd = HistoryUpdateFunction(
        # tidx is guaranteed to be a scalar – to operate on arrays of time
        # indices, we need to set range_update_function.
        func = lambda tidx: A*shim.sin(x1.get_time(tidx).magnitude),
        inputs = [],
        namespace = hists
        )
    x1_range_upd = HistoryUpdateFunction(
        # tidx is guaranteed to be a list of scalars
        func = lambda tidx: A*shim.sin(shim.stack(x1.get_time(tidx).magnitude)[:,np.newaxis]),
        inputs = [],
        namespace = hists
        )
    x2_upd = HistoryUpdateFunction(
        func = Transform("tidx -> x2[tidx-1]*shim.exp(-x2.dt.magnitude) "
                         " + x1(tidx.plain)*x2.dt.magnitude"),
        inputs = ['x1', 'x2'],
        namespace = hists
        )

    x1.update_function = x1_upd
    x2.update_function = x2_upd
    # x1.range_update_function = x1_range_upd  # Not implemented

    assert x1[Ti] == NotComputed.NotYetComputed
    t = Ti*TimeAxis.time_step
    assert np.all(shim.eval(x1(Ti), max_cost=200) == A*np.sin(t))  # Test evaluation
    assert np.all(shim.eval(x1[Ti], max_cost=200) == A*np.sin(t))  # Retrieval at Ti works

    with pytest.raises(IndexError):
        x2[-1]

    assert x2[Ti] == NotComputed.NotYetComputed
    with pytest.raises(IndexError):
        x2(Ti)
    x2.pad(1)
    x2[-1] = A
    x2(Ti)

    # Apply symbolic updates, if present
    updates = shim.get_updates()
    if x1._sym_tidx is not x1._num_tidx:
        # Symbolic history
        assert cgshim != 'numpy'
        assert x1.symbolic
        assert x2._sym_tidx is not x2._num_tidx
        x1.eval()
        x2.eval()
        # assert x1._sym_data is not x1._num_data
        # assert x1._num_tidx in updates.keys()
        # assert x1._num_data in updates.keys()
        # assert x1._sym_tidx is updates[x1._num_tidx]
        # assert x1._sym_data in updates.values()
        # x1._num_tidx.set_value(shim.eval(x1._sym_tidx, max_cost=None))
        # x1._num_data.set_value(shim.eval(x1._sym_data, max_cost=None))
        # object.__setattr__(x1, '_sym_tidx', x1._num_tidx)
        # object.__setattr__(x1, '_sym_data', x1._num_data)
        # del updates[x1._num_tidx]
        # del updates[x1._num_data]
    else:
        # Numeric history
        assert cgshim == 'numpy'
        assert not x1.symbolic
        assert not x2.symbolic
        assert x1._sym_data is x1._num_data
        assert x2._sym_data is x2._num_data
            # Numeric histories never detach _sym from _num
        assert len(updates) == 0

    assert np.all(x2.get_trace() == shim.eval(x2[x2.t0idx:x2.cur_tidx+1]))
    # TODO: solve ODE and compare with analytical soln
    assert np.all(x2.get_trace() == x2._num_data.get_value()[1:Ti+1+1])

    # Copies
    x2copy = x2.copy()
    assert x2copy.cur_tidx.plain == x2.cur_tidx.plain == Ti
    assert x2copy.update_function is None  # Update functions not copied
    x2copy[Ti]  # Can retrieve already computed data
    with pytest.raises(RuntimeError):
        x2copy(Ti)  # But using calling syntax fails b/c update function is not set
    assert x2copy[Ti+1] is NotComputed.NotYetComputed  # TODO: A different return value if not computable ?
    with pytest.raises(RuntimeError):
        x2copy(Ti+1)  # Trying to compute new values raises RuntimeError,
                    # because update_function is not set

def _test_history_spiketrain_updates(cgshim):
    shim.load(cgshim)
    shim.load('theano')
    mtb.typing.freeze_types()
    from sinn.histories import (
        TimeAxis, HistoryUpdateFunction, Series, Spiketrain, NotComputed)

    Ti = 4  # Time point up to which to compute histories

    TimeAxis.time_unit = ureg.s
    TimeAxis.time_step = np.float64(2**-8)   #~0.15. Powers of 2 are more numerically stable
    taxis = TimeAxis(min=0., max=1.)

    x1 = Spiketrain(name='x1', time=taxis, pop_sizes=(1,2,3), iterative=False)
    x2 = Spiketrain(name='x2', time=taxis, pop_sizes=(1,2,3))

    hists = SimpleNamespace(x1=x1, x2=x2)
    # HistoryUpdateFunction.namespace = hists

    rng = shim.config.RandomStreams()

    # def x1_fn(tidx):
    #     res = [shim.nonzero(rng.binomial(n=1, p=0.1, size=x1.shape))[0]
    #             for ti in shim.atleast_1d(tidx)]
    #     return res

    x1_upd = HistoryUpdateFunction(
        func = lambda tidx:
            [shim.nonzero(rng.binomial(n=1, p=0.9, size=x1.shape))[0]
             for ti in shim.atleast_1d(tidx)],
        # func = x1_fn,
        inputs = [],
        namespace = hists
        )
    x2_upd = HistoryUpdateFunction(
        func = Transform("tidx -> [shim.nonzero(rng.binomial("
                         "     n=1, p=0.2*shim.exp( (10*x1(ti) + x2[ti-10:ti].sum(axis=0))/15 ),"
                         f"    size={x2.shape}))[0]"
                         "   for ti in shim.atleast_1d(tidx)]"),
        inputs = ['x1', 'x2'],
        namespace = hists
        )
    x2_upd.func.simple.names['rng'] = rng

    x1.update_function = x1_upd
    x2.update_function = x2_upd

    assert x1[Ti] == NotComputed.NotYetComputed
    t = Ti*TimeAxis.time_step

    rng.seed(0)  # Not enough to be 100% predictable
    # assert np.all(x1(8) == np.array([0,0,1,0,0,0]))  # Test evaluation
    x12 = x1(Ti-2)
    assert np.all( shim.eval(x12.shape, max_cost=None) == x1.shape )
    assert shim.eval(x1[:Ti-1].sum(), max_cost=150) > 0   # Probability of all 0 negligible
    assert np.all(shim.eval(shim.eq(x1[Ti-2], x12),
                            max_cost=150))  # Retrieval at Ti-2 now works

    with pytest.raises(IndexError):
        x2[-1]

    # Apply the x1 updates so that we can test x2 padding
    # (padding fails if there are any pending updates)
    x1.eval()

    assert x2[Ti] == NotComputed.NotYetComputed
    with pytest.raises(IndexError):
        x2(Ti)
    x2.pad(10)
    x2[:0] = [np.array([], dtype=x2.dtype) for i in range(10)]
    x2(Ti)

    x2.eval()

    assert x2.get_trace().sum() > 0
        # This has essentially probability 1, and ensures that the tests below
        # don't succeed due to the updates not happening at all.
    assert np.all(x2.get_trace() == shim.eval(x2[x2.t0idx:x2.cur_tidx+1]))
    assert not (x2.get_trace() != x2._get_num_csr()[10:10+Ti+1]).data.any()

    # Copies
    x2copy = x2.copy()
    assert x2copy.cur_tidx.plain == x2.cur_tidx.plain == Ti
    assert x2copy.update_function is None  # Update functions not copied
    x2copy[Ti]  # Can retrieve already computed data
    with pytest.raises(RuntimeError):
        x2copy(Ti)  # But using calling syntax fails b/c update function is not set
    assert x2copy[Ti+1] is NotComputed.NotYetComputed  # TODO: A different return value if not computable ?
    with pytest.raises(RuntimeError):
        x2copy(Ti+1)  # Trying to compute new values raises RuntimeError,
                    # because update_function is not set

def _test_history_serialization(cgshim):

    shim.load(cgshim)
    mtb.typing.PintUnit.ureg = ureg

    mtb.typing.freeze_types()
    import sinn.config
    from sinn.histories import (
        TimeAxis, HistoryUpdateFunction, Series, Spiketrain, NotComputed)

    mtb.serialize.config.trust_all_inputs = True

    TimeAxis.time_unit = ureg.s
    TimeAxis.time_step = np.float64(2**-8)   #~0.0039. Powers of 2 are more numerically stable
    taxis = TimeAxis(min=0., max=1.)

    TimeAxis.parse_raw(taxis.json());

    # We actually don't need to call `taxis.copy()`, because Pydantic copies all inputs
    series1 = Series(name='series1', time=taxis, shape=(3,), dtype=np.float64)

    ## Test serialization without any update function ##
    series2 = Series.parse_raw(series1.json())

    ##Empty histories
    # TODO: Test with non default values for symbolic, iterative, etc.
    hist_compare(series1, series2)

    # Series
    A = np.array([-0.1, 1.8, 0.5])
    hists = SimpleNamespace(x1=series1, A=A)
    def x1_upd_noinputs_func(tidx):
        return A*shim.sin(series1.get_time(tidx).magnitude)
    def x1_upd_withinputs_func(self, tidx):
        A = self.A; x1 = self.x1
        return x1[tidx-1] * A*shim.sin(x1.get_time(tidx).magnitude)
    with pytest.warns(UserWarning):
        # Raises a warning because x1_upd_noinputs does not have 'self' in signature
        x1_upd_noinputs = HistoryUpdateFunction(
            func=x1_upd_noinputs_func, inputs=[], namespace=None)
    x1_upd_withinputs = HistoryUpdateFunction(
        func=x1_upd_withinputs_func, inputs=['x1'], namespace=hists)
    x1_λupd = HistoryUpdateFunction(
        func = lambda self, tidx: A*shim.sin(self.series1.get_time(tidx).magnitude),
        inputs = [], namespace = hists)

    series1.update_function = x1_upd_noinputs
    series2.update_function = x1_upd_withinputs
    series3 = series2.copy()
    series3.update_function = x1_λupd

    series1.pad(1)
    series1[-1] = 0
    series1(3)
    series1.eval()
    series2(3)
    series2.eval()

    # Direct deserialization accidentally works for functions with no inputs:
    # the namespace is not exported, and thus unavailable for name resolution...
    series4 = Series.parse_raw(series1.json())
    # ...as can be seen when trying to deserialize a history with input
    with pytest.raises(ValidationError):
        Series.parse_raw(series2.json())
    # We can work around this by injecting the namespace back into the
    # deserialized dict, before initializing the History
    # Note: .json() does two things: 1) call load_str_bytes
    #                                2) pass the result as keywords to __init__
    # This is what Model does to deserialize histories, injecting itself as
    # the namespace.
    from pydantic.parse import load_str_bytes
    obj = load_str_bytes(series2.json())
    obj['update_function']['namespace'] = hists
    obj['update_function'] = HistoryUpdateFunction(**obj['update_function'])
    series5 = Series(**obj)
    # Lambda functions can't be serialized
    with pytest.raises(ValueError):
        series3.json()

    ## Test deserialization of existing data
    ## Update functions are deserialized, but deserialized versions unused
    assert series1.cur_tidx > 0
    assert series2.cur_tidx > 0
    hist_compare(series1, series4)
    hist_compare(series2, series5)

    ## "Accidentally" deserialized update function can't be evaluated, because
    ## the namespace was not specified and it can't find 'A'
    series1(10)
    series1.eval()
    with pytest.raises(NameError):
        series4(10)

    ## Properly deserialized function works as expected, and evaluates
    ## identically to the original
    series2(10)
    series2.eval()
    series5(10)
    series5.eval()
    hist_compare(series2, series5)

    ### Spiketrains ###
    # FIXME: Support & test serialization of the CSMProperties attribute of Theano sparse array
    if cgshim == 'theano':
        return
    spikes1 = Spiketrain(name='spikes1', time=taxis, pop_sizes=(3,2,4))

    spikes2 = Spiketrain.parse_raw(spikes1.json())

    hist_compare(spikes1, spikes2)

    ω = 1e5*np.random.rand(*spikes1.shape)
    spikehists = SimpleNamespace(s1=spikes1, ω=ω)
    def do_nothing(f): return f  # Mock decorator
    @do_nothing
    def sin_spikes(self, tidx):  # Some pseudo-chaotic function
        t = self.s1.get_time(tidx).magnitude
        return np.where((np.sin(self.ω*t)>0))[0].astype('uint8')

    spikes1.update_function = HistoryUpdateFunction(
        func=sin_spikes, inputs=['s1'], namespace=spikehists)

    spikes1(30)

    # We serialize/deserialize as above
    # Note that we need to add the decorator to '_deserialization_locals'
    # for it to be available during deserialization.
    obj = load_str_bytes(spikes1.json())
    obj['update_function']['namespace'] = spikehists
    with pytest.raises(NameError):
        HistoryUpdateFunction(**obj['update_function'])
    HistoryUpdateFunction._deserialization_locals.update(do_nothing=do_nothing)
    obj['update_function'] = HistoryUpdateFunction(**obj['update_function'])
    import scipy.sparse
    with pytest.warns(scipy.sparse.SparseEfficiencyWarning):
        # We don't want to deal with the efficiency warning atm.
        spikes3 = Spiketrain(**obj)


    # Deserialized spike history matches original
    hist_compare(spikes1, spikes3)

    # Deserialized spike update function also matches original
    spikes1(60)
    spikes3(60)
    hist_compare(spikes1, spikes3)

def hist_compare(hist1, hist2):
    assert hist1.name == hist2.name
    assert hist1.shape == hist2.shape
    assert hist1.dtype == hist2.dtype
    assert hist1.iterative == hist2.iterative
    assert hist1.symbolic == hist2.symbolic
    assert hist1.locked == hist2.locked
    assert hist1.cur_tidx != hist2.cur_tidx
    assert hist1.cur_tidx.plain == hist2.cur_tidx.plain
    assert hist1._num_data is hist1._sym_data
    assert hist2._num_data is hist2._sym_data
    assert hist1._num_tidx is hist1._sym_tidx
    assert hist2._num_tidx is hist2._sym_tidx
    assert hist1._num_tidx.get_value() is not hist2._num_tidx.get_value()
    if shim.issparse(hist1.data):
        assert shim.issparse(hist2.data)
        assert (hist1.data != hist2.data).size == 0
        assert all(d1.get_value() is not d2.get_value()
                   for d1, d2 in zip(hist1._num_data, hist2._num_data))
    else:
        # Don't use method .all(): returns array or scalar depending on whether shapes match
        assert np.all(hist1.data == hist2.data)
        assert hist1._num_data.get_value().shape == hist2._num_data.get_value().shape
        assert np.all(hist1._num_data.get_value() == hist2._num_data.get_value())
        assert hist1._num_data.get_value() is not hist2._num_data.get_value()
    assert hist1.time.unit._REGISTRY is hist2.time.unit._REGISTRY
    assert hist1.time is not hist2.time
    assert hist1.time == hist2.time
    assert hist1.time.label == hist2.time.label
    assert np.all(hist1.time.stops_array == hist2.time.stops_array)
    assert hist1.time.pad_left  != hist2.time.pad_left
    assert hist1.time.pad_right != hist2.time.pad_right
    assert hist1.time.pad_left.plain  == hist2.time.pad_left.plain
    assert hist1.time.pad_right.plain == hist2.time.pad_right.plain

def test_numhistory_series_indexing():
    return _test_history_series_indexing('numpy')
def test_symhistory_series_indexing():
    return _test_history_series_indexing('theano')
def test_numhistory_spiketrain_indexing():
    return _test_history_spiketrain_indexing('numpy')
def test_symhistory_series_updates():
    return _test_history_spiketrain_indexing('theano')
def test_numhistory_series_updates():
    return _test_history_series_updates('numpy')
def test_symhistory_series_updates():
    return _test_history_series_updates('theano')
def test_numhistory_spiketrain_updates():
    return _test_history_spiketrain_updates('numpy')
def test_symhistory_spiketrain_updates():
    return _test_history_spiketrain_updates('theano')
def test_numhistory_serialization():
    return _test_history_serialization('numpy')
def test_symhistory_serialization():
    return _test_history_serialization('theano')
