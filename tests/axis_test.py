import numpy as np
# Import anything to do with types first
from mackelab_toolbox.transform import Transform, Bijection
import mackelab_toolbox as mtb
import mackelab_toolbox.typing
import pint
ureg = pint.UnitRegistry()
mtb.typing.PintValue.ureg = ureg
mtb.typing.PintUnit.ureg = ureg
import quantities as Q

import theano_shim  as shim
import mackelab_toolbox.cgshim as cgshim

import pytest
from conftest import clean_theano_dir

if __name__ == "__main__":  # For Hydrogen sessions
    cglib = 'numpy'

def _test_map_axis(cglib):

    shim.load(cglib)
    mtb.typing.freeze_types()
    from sinn.axis import MapAxis, RangeAxis, ArrayAxis
    import sinn.axis

    map1 = Transform("i -> np.sign(i)*(0.1*i)**2")
    map2 = Transform("i -> 0.1*i")
    B1 = Bijection(map='m1 -> np.sqrt(m1)', inverse_map='y1 -> y1**2')
    B2 = Bijection(map='m2 -> np.log10(m2)', inverse_map='logx -> 10**(logx)')
    axis1 = MapAxis(label='m1', index_map=map1, index_range=range(0, 50), unit=Q.m)
    axis2 = MapAxis(transform=B2, index_map=map2, index_range=axis1.stops.index_range, unit=ureg.s)
    axis3 = MapAxis(label='m1', index_map=map1, index_range=range(0, 50))

    assert axis3.unit is sinn.axis.unitless
    assert 87 * axis3.unit == 87

    assert axis1.unit_check(1*Q.m)
    assert axis2.unit_check(1*ureg.s)
    assert axis3.unit_check(1.)
    assert not axis1.unit_check(1.)
    assert not axis1.unit_check(1*Q.s)
    assert not axis1.unit_check(1*ureg.s)
    assert not axis2.unit_check(1.)
    assert not axis2.unit_check(1*ureg.m)
    assert not axis2.unit_check(1*Q.s)
    assert not axis3.unit_check(1*ureg.s)
    assert not axis3.unit_check(1*Q.m)

    assert np.isclose(axis2[np.arange(4,20,4).astype(axis1.Index.Numeric.nptype)],
                      np.array([0.4, 0.8, 1.2, 1.6])*ureg.s).all()
    # Same test with axis1 – it's done differently because Quantities isn't
    # as compatible with NumPy
    x1 = axis1[np.arange(4,20,4).astype(axis1.Index.Numeric.nptype)]
    x2 = np.array([0.16, 0.64, 1.44, 2.56]) * Q.m
    assert np.isclose(x1.magnitude, x2.magnitude).all()
    assert x1.units == x2.units

    assert axis1.Index.is_compatible(axis1.Index(5))
    assert axis1.Index.is_compatible(axis2.Index(5)) is False
    assert axis1.Index.is_compatible(5)
    assert axis1.Index.is_compatible(axis2.Index.Delta(5)) is False
        # Fails because axes don't have a step (so Δs can't be compared)
        # For RangeAxis below, last assert is True

    assert np.all(np.array(axis1.stops) == (0.1*np.arange(50))**2)
    assert np.all(axis1.stops_array == (0.1*np.arange(50))**2)

    f = Transform("x -> np.sqrt(x)")
    assert np.all(np.array(axis1.stops.transform(f)) == 0.1*np.arange(50))

    with pytest.raises(ValueError):
        axis1.index(0.2*Q.m)


    # Check that dict and json export don't throw errors
    # TODO: Check that returned val is as expected
    axis1.dict()
    axis2.dict()
    axis1.json()
    axis2.json()

    assert axis1.index(0.25*Q.m) == 5
    assert axis2.index(0.2*ureg.s) == 2

    x = 0.2*ureg.s
    assert axis2.unit_convert(x).magnitude == 0.2
    assert axis1.unit_convert(0.2*Q.m).magnitude == 0.2
    assert axis3.unit_convert(3.) == 3.

    assert axis2.index(400*ureg.ms) == 4
    assert np.all(axis2.index(np.arange(22,39)*0.1*ureg.s) == np.arange(22,39))
    axidx = axis2.index(np.s_[1.1*ureg.s:3.0*ureg.s])
    axis1slc = slice(axis1.Index.make_index(11), axis1.Index.make_index(30))
    axis2slc = slice(axis2.Index.make_index(11), axis2.Index.make_index(30))
    assert axidx == slice(11, 30)
    assert axidx != axis1slc
    assert axidx == axis2slc
    with pytest.raises(ValueError):
        # start/stop don't correspond to index values
        assert axis2.index(np.s_[1.04*ureg.s:2.93*ureg.s]) == np.arange(11, 30)

    with pytest.raises(TypeError):
        axis1.index(0.25*Q.s)

    with pytest.raises(TypeError):
        # Wrong unit type
        axis1.index(0.25*ureg.m)

    with pytest.raises(AttributeError):
        # Axis 1 does not define a transformation
        axis1.transformed

    # FIXME: MapAxis(**axis1.dict()) doesn't work for the same reason we had
    #        to overload MapAxis.parse_obj
    axis4 = MapAxis.parse_obj(axis2.dict())
    axis5 = MapAxis.parse_raw(axis2.transformed.json())

    assert isinstance(axis4.transformed, MapAxis)
    assert axis4.transformed == axis5

    # axis2.stops.index_range
    # axis2.edges.stops.index_range
    # axis2.edges.stops
    # axis2.edges.centers

    # Wouldn't work with axis1 because non-regular
    assert (axis2.edges.centers.stops == axis2.stops)  # Check invertibility
    assert (axis2.edges.centers == axis2)   # Check axis.__eq__


    # Copying
    axis1copy = axis1.copy()
    assert axis1copy is not axis1
    assert axis1copy.stops is not axis1.stops
    assert axis1copy == axis1

    assert axis1.stops.Index.axis is axis1   # Ensure Index.axis wasn't remapped by copy
    assert axis1copy.stops.Index.axis is axis1copy  # Copied Index points to new axis
    assert hasattr(axis1copy.stops.index_map, 'astexpr')

    # AxisIndex type hierarchy is preserved on copy
    # Todo: Move to Hierarchy test
    assert axis1copy.Index is not axis1.Index
    assert axis1copy.Index.Numeric is not axis1.Index.Numeric
    assert axis1copy.Index.Delta.Numeric is not axis1.Index.Delta.Numeric
    assert axis1copy.Index.Numeric.Delta is not axis1.Index.Numeric.Delta
    assert axis1copy.Index.Symbolic is not axis1.Index.Symbolic
    assert axis1copy.Index.Delta.Symbolic is not axis1.Index.Delta.Symbolic
    assert axis1copy.Index.Symbolic.Delta is not axis1.Index.Symbolic.Delta
    assert axis1copy.Index.Delta.Numeric is axis1copy.Index.Numeric.Delta
    assert axis1copy.Index.Delta.Symbolic is axis1copy.Index.Symbolic.Delta

    assert axis1copy.Index.step == axis1.Index.step
    assert axis1copy.Index.Delta.step == axis1.Index.Delta.step

    # Index attributes are correctly recast to new index type
    type(axis1.pad_left) is axis1.Index.Numeric.Delta
    type(axis1copy.pad_left) is axis1copy.Index.Numeric.Delta
    type(axis1.pad_right) is axis1.Index.Numeric.Delta
    type(axis1copy.pad_right) is axis1copy.Index.Numeric.Delta

    # Padding
    # assert axis1.index(0.25*Q.m) == 5
    # assert axis2.index(0.2*ureg.s) == 2
    assert (axis1.stops.pad(2) == (2,0))
    assert (axis1.stops.pad(2) == (0,0))  # Padding isn't duplicated
    assert axis1.index(0*Q.m) == 0
    assert axis1.data_index(0*Q.m) == 2

    assert axis1[5] == (0.5)**2*Q.m

    assert axis1.stops.pad_left == 2
    assert axis1copy.stops.pad_left == 0

    assert axis1.stops.padded_length == 52
    assert axis1.stops.unpadded_length == 50
    assert len(axis1) == axis1.padded_length

    i = axis1.Index(4)
    Δ = axis1.Index(7)-axis1.Index(2)
    assert isinstance(i, axis1.Index)
    assert isinstance(i, axis1.Index.Delta)
    assert not isinstance(Δ, axis1.Index)
    assert isinstance(Δ, axis1.Index.Delta)


    # FIXME: should raise an error
    # with pytest.raises(TypeError): # Can't pad with absolute index
    #     axis1.pad(axis1.Index(5))
    axis1.pad(axis1.Index(7)-axis1.Index(2)) # *Can* pad with relative (delta) index
    assert axis1.pad_left == 5
    axis1.pad(axis1.Index.Delta(8))
    assert axis1.pad_left == 8
    with pytest.raises(TypeError): # Wrong index type
        axis1.pad(axis2.Index(5))
    with pytest.raises(TypeError): # Can't pad with floats if no step
        axis2.pad(1*ureg.s)

    # Data index and plain preserve identity
    if cgshim == 'numpy':
        kplain = shim.shared(4)  # Also test downcasting for plain data
    else:
        kplain = shim.shared(axis1.Index.nptype(4))  # Can't downcast symbolics
    k = axis1.Index(kplain)
    assert k.plain is k.plain
    # FIXME: This should also work
    # assert k.plain is kplain
    assert k.data_index is k.data_index

    # Clean up state of symbolic updates
    shim.reset_updates()

def _test_regular_axis(cglib):

    shim.load(cglib)
    mtb.typing.freeze_types()
    from sinn.axis import MapAxis, RangeAxis, ArrayAxis
    import sinn.axis

    # These ranges test that when rounding max, the floor is correctly taken
    B1 = Bijection(map='m1 -> np.sqrt(m1)', inverse_map='y1 -> y1**2')
    Δt = np.float64(0.1)
    axis1 = RangeAxis(label='r1', min=0., max=10., step=Δt, unit=ureg.s)
    axis2 = RangeAxis(label='r2', transform=B1, min=0., max=9.99,  step=Δt, unit=ureg.s)
    axis3 = RangeAxis(label='r3', min=0., max=10.01, step=Δt, unit=ureg.s)

    assert axis1.unit_check(1*ureg.s)
    assert not axis1.unit_check(1.)
    assert not axis1.unit_check(1*Q.s)

    assert axis1.Index.is_compatible(axis1.Index(5))
    assert axis1.Index.is_compatible(axis2.Index(5)) is False
    assert axis1.Index.is_compatible(5)
    assert axis1.Index.is_compatible(axis2.Index.Delta(5))
    # After reporting compatibility, check that we can actually perform arithmetic
    # TODO: Make a new arithmetic checking section ?Q
    i1 = axis1.Index(10); i2 = axis2.Index.Delta(5)
    assert i1 - i2 == 5
    assert i1 + i2 == 15

    assert len(axis1) == len(axis2) + 1 == len(axis3) == 101
    assert axis1.index(1*ureg.s) == axis1.index(1.*ureg.s) == 10

    assert axis1.step.magnitude == axis1.stops.step

    f = Transform("x -> np.sqrt(x)")
    assert np.all(np.isclose(np.array(axis1.stops.transform(f)),
                             np.sqrt(0.1*np.arange(101))))

    with pytest.raises(ValueError):
        axis1.index(0.244*ureg.s)

    assert axis2.index(0.2*ureg.s) == 2
    assert axis2.index(400*ureg.ms) == 4
    assert np.all(axis2.index(np.arange(22,39)*0.1*ureg.s) == np.arange(22,39))
    axidx = axis2.index(np.s_[1.1*ureg.s:3.0*ureg.s])
    axis1slc = slice(axis1.Index.make_index(11), axis1.Index.make_index(30))
    axis2slc = slice(axis2.Index.make_index(11), axis2.Index.make_index(30))
    assert axidx == slice(11, 30)
    assert axidx != axis1slc
    assert axidx == axis2slc
    with pytest.raises(ValueError):
        # start/stop don't correspond to index values
        assert axis2.index(np.s_[1.04*ureg.s:2.93*ureg.s]) == np.arange(11, 30)


    assert np.all(axis1[4:20:4] == np.arange(4,20,4)*0.1 * ureg.s)
    assert np.all(np.equal(axis1[np.arange(4,20,4).astype(axis1.Index.Numeric.nptype)],
                           np.arange(4,20,4)*0.1 * ureg.s))

    with pytest.raises(TypeError):
        axis1.index(0.25*ureg.mV)

    # FIXME: MapAxis(**axis1.dict()) doesn't work for the same reason we had
    #        to overload MapAxis.parse_obj
    axis4 = RangeAxis.parse_obj(axis2.desc)
    axis5 = MapAxis.parse_obj(axis2.transformed_desc)

    assert isinstance(axis4.transformed, MapAxis)
    assert axis4.transformed == axis5

    assert isinstance(axis1.edges, ArrayAxis)

    axis1.edges.centers.stops_array == axis1.stops_array

    assert axis1.edges.centers.stops.isclose(axis1.stops)  # Check invertibility
    assert axis2 == axis4   # Check axis.__eq__
    # assert axis1.edges.centers == axis1   # Fails b/c numerical errors

    # Copying
    axis1copy = axis1.copy()
    assert axis1copy is not axis1
    assert axis1copy.stops is not axis1.stops
    assert axis1copy == axis1

    assert axis1.stops.Index.axis is axis1   # Index.axis wasn't remapped by copy
    assert axis1copy.stops.Index.axis is axis1copy  # Copied index points to new axis
    #  no astexpr for RangeMapping
    assert type(axis1.stops.i0) is axis1.stops.Index.Numeric
    assert isinstance(axis1.stops.i0, axis1.stops.Index)
    assert type(axis1copy.stops.i0) is axis1copy.stops.Index.Numeric  # i0 type was correctly updated
    assert isinstance(axis1copy.stops.i0, axis1copy.stops.Index)
    assert not isinstance(axis1copy.stops.i0, axis1.stops.Index)

    assert axis1copy.Index.step == axis1.Index.step
    assert axis1copy.Index.Delta.step == axis1.Index.Delta.step


    # Padding
    # assert axis1.index(0.25*Q.m) == 5
    # assert axis2.index(0.2*ureg.s) == 2
    assert (axis1.stops.pad(2) == (2,0))
    assert (axis1.stops.pad(2) == (0,0))  # Padding isn't duplicated
    assert axis1.index(0*ureg.s) == 0
    assert axis1.data_index(0*ureg.s) == 2

    assert axis1[5] == 0.5*ureg.s

    assert axis1.stops.pad_left == 2
    assert axis1copy.stops.pad_left == 0

    assert axis1.stops.padded_length == 103
    assert axis1.stops.unpadded_length == 101
    assert len(axis1) == axis1.padded_length

    # FIXME: should raise an error
    # with pytest.raises(TypeError): # Can't pad with absolute index
    #     axis1.pad(axis1.Index(5))
    axis1.pad(axis1.Index(7)-axis1.Index(2)) # *Can* pad with relative (delta) index
    assert axis1.pad_left == 5
    axis1.pad(axis1.Index.Delta(8))
    assert axis1.pad_left == 8
    with pytest.raises(TypeError): # Wrong index type
        axis1.pad(axis2.Index(5))

    # In contrast to MapAxis, can pad with real quantities
    axis2.pad(1*ureg.s)
    assert axis2.pad_left == 10

    # Data index and plain preserve identity
    if cgshim == 'numpy':
        kplain = shim.shared(4)  # Also test downcasting for plain data
    else:
        kplain = shim.shared(axis1.Index.nptype(4))  # Can't downcast symbolics
    k = axis1.Index(kplain)
    assert k.plain is k.plain
    # FIXME: This should also work
    # assert k.plain is kplain
    assert k.data_index is k.data_index

    # Clean up state of symbolic updates
    shim.reset_updates()

def _test_symbolic_indexing(cglib):

    shim.load(cglib)
    mtb.typing.freeze_types()
    from sinn.axis import MapAxis, RangeAxis, ArrayAxis
    import sinn.axis

    map1 = Transform("i -> np.sign(i)*(0.1*i)**2")
    # B1 = Bijection(map='m1 -> np.sqrt(m1)', inverse_map='y1 -> y1**2')
    mapaxis1 = MapAxis(label='m1', index_map=map1, index_range=range(0, 50),
                       unit=ureg.m)
    Δt = np.float64(2**-3)  # No rounding errors with powers of two
    rangeaxis1 = RangeAxis(label='r1', min=0., max=10., step=Δt, unit=ureg.s)
    # rangeaxis2 = RangeAxis(label='r2', transform=B1, min=0., max=10., step=Δt,
    #                        unit=ureg.s)

    assert mapaxis1.Index is mapaxis1.stops.Index
    assert issubclass(mapaxis1.stops.Index, mapaxis1.stops.Index.Delta)

    NumIdxMap = sinn.axis.NumericAbstractAxisIndex._created_indexes[id(mapaxis1)]
    SymbIdxMap = sinn.axis.SymbolicAbstractAxisIndex._created_indexes[id(mapaxis1)]
    SymbIdxDeltaMap = sinn.axis.SymbolicAbstractAxisIndexDelta._created_indexes[id(mapaxis1)]
    NumIdxRange = sinn.axis.NumericAbstractAxisIndex._created_indexes[id(rangeaxis1)]
    SymbIdxRange = sinn.axis.SymbolicAbstractAxisIndex._created_indexes[id(rangeaxis1)]
    SymbIdxDeltaRange = sinn.axis.SymbolicAbstractAxisIndexDelta._created_indexes[id(rangeaxis1)]
    assert SymbIdxMap.Delta is SymbIdxDeltaMap
    assert SymbIdxDeltaMap.Delta is SymbIdxDeltaMap
    assert SymbIdxDeltaMap.Absolute is SymbIdxMap

    assert issubclass(SymbIdxMap, mapaxis1.Index)
    assert issubclass(SymbIdxMap, SymbIdxMap.Delta)
    assert issubclass(SymbIdxMap, sinn.axis.AbstractAxisIndex)
    assert issubclass(SymbIdxMap, sinn.axis.AbstractAxisIndexDelta)
    assert issubclass(SymbIdxMap, sinn.axis.SymbolicAbstractAxisIndex)
    assert issubclass(SymbIdxMap, sinn.axis.SymbolicAbstractAxisIndexDelta)
    assert not issubclass(SymbIdxMap, sinn.axis.NumericAbstractAxisIndex)
    assert not issubclass(SymbIdxMap, sinn.axis.NumericAbstractAxisIndexDelta)
    assert issubclass(SymbIdxMap.Delta, mapaxis1.Index.Delta)
    assert not issubclass(SymbIdxMap.Delta, SymbIdxMap)
    assert not issubclass(SymbIdxMap.Delta, sinn.axis.AbstractAxisIndex)
    assert issubclass(SymbIdxMap.Delta, sinn.axis.AbstractAxisIndexDelta)
    assert not issubclass(SymbIdxMap.Delta, sinn.axis.SymbolicAbstractAxisIndex)
    assert issubclass(SymbIdxMap.Delta, sinn.axis.SymbolicAbstractAxisIndexDelta)
    assert not issubclass(SymbIdxMap.Delta, sinn.axis.NumericAbstractAxisIndex)
    assert not issubclass(SymbIdxMap.Delta, sinn.axis.NumericAbstractAxisIndexDelta)
    assert not issubclass(SymbIdxMap, SymbIdxRange.Delta)
    assert not issubclass(SymbIdxMap.Delta, rangeaxis1.Index.Delta)

    if cglib == 'theano':
        IdxMap  = SymbIdxMap
        IdxRange = SymbIdxRange
    elif cglib == 'numpy':
        IdxMap  = NumIdxMap
        IdxRange = NumIdxRange
    i = shim.shared(np.int8(14), name='i')
    assert shim.is_graph_object(i) is (cglib == 'theano')
        # Make sure we actually loaded theano when expected

    # Casting to Index
    midx = mapaxis1.Index(i)
    midx2 = mapaxis1.Index(i+4)
    Δmidx = midx2-midx
    assert type(midx) is IdxMap
    assert type(Δmidx) is IdxMap.Delta
    # No need to check all `isinstance` combos: once `type() is`, the
    # subclass relationships follow from the tests above
    ridx = rangeaxis1.Index(i)
    ridx2 = rangeaxis1.Index(i+4)
    Δridx = ridx2-ridx
    assert type(ridx) is IdxRange
    assert type(Δridx) is IdxRange.Delta

    # Operations on indices
    midx + 1
    with pytest.raises(ValueError):
        midx + midx2
    assert isinstance(midx - midx2, type(midx).Delta)
    ridx + 1
    with pytest.raises(ValueError):
        ridx + ridx2
    assert isinstance(ridx - ridx2, type(ridx).Delta)

    # Clean up state of symbolic updates
    shim.reset_updates()

def _test_numeric_indexing(cglib):

    shim.load(cglib)
    mtb.typing.freeze_types()
    from sinn.axis import MapAxis, RangeAxis, ArrayAxis
    import sinn.axis

    map1 = Transform("i -> np.sign(i)*(0.1*i)**2")
    # B1 = Bijection(map='m1 -> np.sqrt(m1)', inverse_map='y1 -> y1**2')
    mapaxis1 = MapAxis(label='m1', index_map=map1, index_range=range(0, 50),
                       unit=ureg.m)
    Δt = np.float64(2**-3)  # No rounding errors with powers of two
    rangeaxis1 = RangeAxis(label='r1', min=0., max=10., step=Δt, unit=ureg.s)
    # rangeaxis2 = RangeAxis(label='r2', transform=B1, min=0., max=10., step=Δt,
    #                        unit=ureg.s)

    assert mapaxis1.Index is mapaxis1.stops.Index
    assert issubclass(mapaxis1.stops.Index, mapaxis1.stops.Index.Delta)

    NumIdxMap = sinn.axis.NumericAbstractAxisIndex._created_indexes[id(mapaxis1)]
    SymbIdxMap = sinn.axis.SymbolicAbstractAxisIndex._created_indexes[id(mapaxis1)]
    NumIdxDeltaMap = sinn.axis.NumericAbstractAxisIndexDelta._created_indexes[id(mapaxis1)]
    NumIdxRange = sinn.axis.NumericAbstractAxisIndex._created_indexes[id(rangeaxis1)]
    SymbIdxRange = sinn.axis.SymbolicAbstractAxisIndex._created_indexes[id(rangeaxis1)]
    NumIdxDeltaRange = sinn.axis.NumericAbstractAxisIndexDelta._created_indexes[id(rangeaxis1)]
    assert NumIdxMap.Delta is NumIdxDeltaMap
    assert NumIdxDeltaMap.Delta is NumIdxDeltaMap
    assert NumIdxDeltaMap.Absolute is NumIdxMap

    assert issubclass(NumIdxMap, mapaxis1.Index)
    assert issubclass(NumIdxMap, NumIdxMap.Delta)
    assert issubclass(NumIdxMap, sinn.axis.AbstractAxisIndex)
    assert issubclass(NumIdxMap, sinn.axis.AbstractAxisIndexDelta)
    assert issubclass(NumIdxMap, sinn.axis.NumericAbstractAxisIndex)
    assert issubclass(NumIdxMap, sinn.axis.NumericAbstractAxisIndexDelta)
    assert not issubclass(NumIdxMap, sinn.axis.SymbolicAbstractAxisIndex)
    assert not issubclass(NumIdxMap, sinn.axis.SymbolicAbstractAxisIndexDelta)
    assert issubclass(NumIdxMap.Delta, mapaxis1.Index.Delta)
    assert not issubclass(NumIdxMap.Delta, NumIdxMap)
    assert not issubclass(NumIdxMap.Delta, sinn.axis.AbstractAxisIndex)
    assert issubclass(NumIdxMap.Delta, sinn.axis.AbstractAxisIndexDelta)
    assert not issubclass(NumIdxMap.Delta, sinn.axis.NumericAbstractAxisIndex)
    assert issubclass(NumIdxMap.Delta, sinn.axis.NumericAbstractAxisIndexDelta)
    assert not issubclass(NumIdxMap.Delta, sinn.axis.SymbolicAbstractAxisIndex)
    assert not issubclass(NumIdxMap.Delta, sinn.axis.SymbolicAbstractAxisIndexDelta)
    assert not issubclass(NumIdxMap, NumIdxRange.Delta)
    assert not issubclass(NumIdxMap.Delta, rangeaxis1.Index.Delta)

    # Todo: duplicate w/ symbolic_indexing
    if cglib == 'theano':
        IdxMap  = SymbIdxMap
        IdxRange = SymbIdxRange
    elif cglib == 'numpy':
        IdxMap  = NumIdxMap
        IdxRange = NumIdxRange
    i = shim.shared(np.int8(14), name='i')
    assert shim.is_graph_object(i) is (cglib == 'theano')
        # Make sure we actually loaded theano when expected

    # Casting to Index
    midx = mapaxis1.Index(i)
    midx2 = mapaxis1.Index(i+4)
    Δmidx = midx2-midx
    assert type(midx) is IdxMap
    assert type(Δmidx) is IdxMap.Delta
    # No need to check all `isinstance` combos: once `type() is`, the
    # subclass relationships follow from the tests above
    ridx = rangeaxis1.Index(i)
    ridx2 = rangeaxis1.Index(i+4)
    Δridx = ridx2-ridx
    assert type(ridx) is IdxRange
    assert type(Δridx) is IdxRange.Delta

    # Casting arrays currently does not return an index type
    arrmidx = mapaxis1.Index(np.arange(4))
    assert not isinstance(arrmidx, mapaxis1.Index)
    assert arrmidx.dtype is np.dtype('int8')

    # Operations on indices
    midx + 1
    with pytest.raises(ValueError):
        midx + midx2
    assert isinstance(midx - midx2, type(midx).Delta)
    ridx + 1
    with pytest.raises(ValueError):
        ridx + ridx2
    assert isinstance(ridx - ridx2, type(ridx).Delta)

    # Clean up state of symbolic updates
    shim.reset_updates()

def test_num_map_axis():
    _test_map_axis('numpy')
def test_theano_map_axis(clean_theano_dir):
    _test_map_axis('theano')
def test_num_regular_axis():
    _test_regular_axis('numpy')
def test_theano_regular_axis(clean_theano_dir):
    _test_regular_axis('theano')
def test_num_symbolic_indexing():
    _test_symbolic_indexing('numpy')
def test_theano_symbolic_indexing(clean_theano_dir):
    _test_symbolic_indexing('theano')
def test_num_numeric_indexing():
    _test_numeric_indexing('numpy')
def test_theano_numeric_indexing(clean_theano_dir):
    _test_numeric_indexing('theano')

if __name__ == "__main__":
    # test_map_axis()
    # test_regular_axis()
    # test_num_numeric_indexing()
    test_theano_symbolic_indexing()
