from mackelab_toolbox.transform import Bijection
from sinn.axis import MapAxis, RegularAxis, ArrayAxis
import pint
ureg = pint.UnitRegistry()
import quantities as Q

def map_axis_test():
    def map1(index): return (0.1*index)**2
    def map2(index): return 0.1*index
    B1 = Bijection(map='m1 -> np.sqrt(m1)', inverse_map='y1 -> y1**2')
    B2 = Bijection(map='m2 -> np.log10(m1)', inverse_map='logx -> 10**(logx)')
    axis1 = MapAxis('m1', map1, index_range=range(0, 50), unit=Q.m)
    axis2 = MapAxis(B, map2, axis1.stops.index_range, unit=ureg.s)

    assert np.array(axis1.stops) == (0.1*np.arange(50))**2
    def f(x): return np.sqrt(x)
    assert np.array(axis1.stops.transform(f)) == 0.1*np.arange(50)

    try: axis1.index(0.2*Q.m)
    except ValueError: pass
    else: assert False

    assert axis1.index(0.25*Q.m) == 5
    assert axis2.index(0.2*ureg.s) == 2
    assert axis2.index(400*ureg.ms) == 4
    assert axis2.index(np.arange(22,39)*0.1*ureg.s) == np.arange(22,39)
    assert axis2.index(np.s_[1.04*ureg.s:2.93*ureg.s]) == np.arange(11, 30)

    try: axis1.index(0.25*Q.s)
    except TypeError: pass
    else: assert False

    try: axis1.index(0.25*ureg.m)
    except TypeError: pass
    else: assert False


    axis4 = MapAxis(**axis1.desc)
    axis5 = MapAxis(**axis1.transformed_desc)

    assert isinstance(axis4.transformed, MapAxis)
    assert axis4.transformed == axis5

    assert axis1.edges.centers.stops == axis.stops  # Check invertibility
    assert axis1.edges.centers == axis1   # Check axis.__eq__

def regular_axis_test():
    axis1 = RegularAxis('r1', 0.1, 0, 10, unit=ureg.s)
    axis2 = RegularAxis('r2', 0.1, 0, 9.99, unit=ureg.s)
    axis3 = RegularAxis('r3', 0.1, 0, 10.01, unit=ureg.s)

    assert len(axis1) == len(axis2) + 1 == len(axis3) == 101
    assert axis1.index(1) == axis1.index(1.) == 10

    def f(x): return np.sqrt(x)
    assert np.array(axis1.stops.transform(f)) == 0.1*np.arange(50)

    try: axis1.index(0.244*ureg.s)
    except ValueError: pass
    else: assert False

    assert axis2.index(0.2*ureg.s) == 2
    assert axis2.index(400*ureg.ms) == 4
    assert axis2.index(np.arange(22,39)*0.1*ureg.s) == np.arange(22,39)
    assert axis2.index(np.s_[1.04*ureg.s:2.93*ureg.s]) == np.arange(11, 30)

    try: axis1.index(0.25*ureg.mV)
    except TypeError: pass
    else: assert False

    axis4 = MapAxis(**axis1.desc)
    axis5 = MapAxis(**axis1.transformed_desc)

    assert isinstance(axis4.transformed, MapAxis)
    assert axis4.transformed == axis5

    assert isinstance(axis1.edges, ArrayAxis)

    assert axis1.edges.centers.stops == axis.stops  # Check invertibility
    assert axis1.edges.centers == axis1   # Check axis.__eq__

def __main__():
    map_axis_test()
    regular_axis_test()
