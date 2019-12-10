import numpy as np
import copy
from collections import Iterable
import itertools
from numbers import Number

import theano_shim as shim
from sinn.common import PopulationHistoryBase

# TODO: Implement PopElTerm as a metaclass which takes pop_sizes and returns a class
#       with a fixed pop_size. See class Spiketrain.__init__ for why we want this.
# TODO: Have the metaclass also determine to use PopTermMeso or PopTermMicro based on value shape
# TODO: Completely rewrite by subclassing ndarray and use ndim, etc. Rather than nested
#       PopTerms, use a "broadcast" shape like (Meso, Meso, Micro).
#       Should be able to expand individual axes to different levels (micro, meso, macro)
# TODO: Could this be implemented with numpy.block ?
# TODO: Should have an expansion level like "Plain" or "ndarray" which doesn't correspond
#       to any block level, even though it may have a compatible size.
#       E.g. If we have two populations, arrays of 2 and 3 PopTerms should have the
#       same type. At the moment, because be assign type purely on shape, they would
#       have respectively PopTermMeso and PopTermPart types.
#       Importantly, a Plain dimension would not be expanded.
# TODO: Make this a generic BroadcastableBlockArray (meta)class and move to
#       mackelab.utils. Leave here only the specialization for PopulationHistory

class ShapeError(ValueError):
    pass

class PopTerm(np.ndarray):
    """
    Subclass of numpy.ndarray which introduces an extra broadcasting
    block level (the "population"). Normal array operations (+,-,*, etc.)
    are carried out by transparently broadcasting along blocks when
    required. General support for all numpy methods is not yet implemented:
    currently each is added by hand as it is needed (currently 'max', 'min',
    'dot' and 'flatten' are supported).

    Block broadcasting:
    Suppose we have a (600,) PopTerm array representing 2 populations of size
    500 and 100. This can multiplied by a (2,) array (PopTerm or plain
    ndarray), and broadcasting will happen as expected, the elements of the
    second array being matched to populations of the first.
    An error is raised when attempting to multiply by an array whose shape
    does not match any block size, or by a PopTerm with different block sizes.

    Current implementation defines three levels: 'Micro', 'Meso' and 'Macro'.
    A 'Micro' dimension has the largest number of elements, a 'Meso'
    dimension as many elements as there are populations (or blocks) and
    a 'Macro' dimension has exactly 1 element, which applies to all populations.

    An eventual goal is to rename 'populations' simply 'blocks', and allow
    arbitrarily deep nested block structure. This class would then be
    renamed to 'BlockBroadcastableArray', or 'BBArray'.

    The current implementation makes use of np.tile more than it should, which
    may incur a performance hit on large arrays.
    """

    # BlockTypes: Defined at bottow of module
    # BlockNames: Defined at bottow of module
    # BlockLevels: Defined at bottow of module
    # shim_class = ShimmedPopTerm

    def __new__(cls, pop_sizes, values, block_types):
        expected_size = ( 1 if issubclass(cls, PopTermMacro)
                          else len(pop_sizes) if issubclass(cls, PopTermMeso)
                          else sum(pop_sizes) if issubclass(cls, PopTermMicro)
                          else None )
        if shim.isshared(values):
            raise TypeError("Use a `ShimmedPopTerm` to create popterm from "
                            "a shared variable.")
        assert(len(block_types) == np.asarray(values).ndim)
        assert(issubclass(cls, cls.BlockTypes[block_types[0]]))
        if not issubclass(cls, PopTermPart):
            assert(expected_size is not None)
            if len(values) != expected_size:
                raise ValueError("Provided values (length {}) not commensurate with the "
                                "expected number of elements ({})."
                                .format(len(values), expected_size))

        # TODO: assert: all pop_values elements are of same type
        # TODO: Remove the ugly nesting: allow N-D PopTerm
        if not isinstance(values, Iterable) or not isinstance(values[0], Iterable):
            obj = np.asarray(values)
        else:
            NestedType = cls.BlockTypes[block_types[1]]
            try:
                obj = np.asarray( [ NestedType(pop_sizes,
                                                 value,
                                                 block_types[1:])
                                    for value in values ] )
            except ValueError:
                raise ValueError("Tried to make a nested PopTerm, but the 2D dimension of "
                    "values ({}) is incompatible with population sizes ({})."
                    .format(len(values[0]), pop_sizes))

        obj = obj.view(cls)
        obj.pop_sizes = pop_sizes
        obj.pop_slices = PopTerm._get_pop_slices(pop_sizes)
        obj.block_types = tuple(block_types)
        return obj

    def __array_finalize__(self, obj):
        # explicitly called from constructor => obj is None
        # view casting => obj is any subclass of ndarray
        # new-from-template => instance this subclass
        if obj is None or not isinstance(obj, PopTerm):
            return

        if obj is not None and not hasattr(obj, 'pop_sizes'):
            logger.warning("Constructing a view from a plain ndarray. "
                           "Population sizes are not set.")
        if obj is not None and hasattr(obj, 'pop_sizes'):
            if not hasattr(obj, 'pop_sizes'):
                raise NotImplementedError("Constructing arrays from arrays which "
                                          "are not subclasses of PopTerm"
                                          "is not currently implemented.")
            if (hasattr(self, 'pop_sizes') and hasattr(obj, 'pop_size')
                and np.any(self.pop_sizes != obj.pop_sizes)):
                raise ValueError("Trying to take an array view based on a PopTerm "
                                 "with different population sizes ({} vs {})."
                                 .format(self.pop_sizes, obj.pop_sizes))
            if not hasattr(self, 'pop_sizes'):
                self.pop_sizes = obj.pop_sizes

            self.pop_slices = PopTerm._get_pop_slices(self.pop_sizes)

            if hasattr(obj, 'pop_slices'):
                assert(all( selfslc == objslc
                             for selfslc, objslc in zip(self.pop_slices, obj.pop_slices) ))

            self.block_types = obj.block_types

    def __getitem__(self, key):
        res = self.view(np.ndarray)[key]
        if res.ndim == 0:
            return res
        else:
            if key is None:
                block_types = ('Macro',) + self.block_types
            elif isinstance(key, Iterable) and not isinstance(key, (str, bytes)):
                # Expand the Ellipsis, if one is present
                if Ellipsis in key:
                    ellidx = key.index(Ellipsis)
                    nkeyints = len( [k for k in key if isinstance(k, int)] )
                    nkeyslices = len( [k for k in key if isinstance(k, slice)] )
                    nkeynones = len( [k for k in key if k is None] )
                    expected_size = nkeyslices + nkeynones - nkeyints
                        # The expected size of the result without the Ellipsis
                    dim_diff = res.ndim - expected_size
                        # The difference in no. of dimmensions is due to the Ellpisis
                        # We replace those dimensions by ':' (i.e. slice(None))
                    assert(dim_diff >= 0)
                    key = key[:ellidx] + (slice(None),)*dim_diff + key[ellidx+1:]
                #assert(len(key) == res.ndim)
                # Adapt the block_types to the result's new shape
                block_types = self.block_types
                for i, dimkey in enumerate(key):
                    if dimkey is None:
                        # np.newaxis: add an axis to the block types
                        block_types = block_types[:i] + ('Macro',) + block_types[i:]
                    elif isinstance(dimkey, int):
                        # collapse this dimension in the block types
                        block_types = block_types[:i] + block_types[i+1:]
                    elif isinstance(dimkey, slice) and dimkey != slice(None):
                        # Slice: convert this axis to 'Plain', since it's partial
                        block_types = block_types[:i] + ('Plain',) + block_types[i+1:]
                    elif dimkey is Ellipsis:
                        raise NotImplementedError
                    else:
                        ValueError("Unrecognized array key '{}'.".format(key))
                assert(len(block_types) == res.ndim)
            else:
                block_types = self.block_types

            cls = self.BlockTypes[block_types[0]]
            res = res.view(cls)
            res.block_types = block_types

        res.pop_sizes = self.pop_sizes
        res.pop_slices = self.pop_slices
        return res

    def __str__(self):
        return str(self.view(np.ndarray)) + " " + str(self.block_types)
    def __repr__(self):
        return repr(self.view(np.ndarray)) + " " + repr(self.block_types)

    @staticmethod
    def _get_pop_slices(pop_sizes):
        pop_slices = []
        i = 0
        for pop_size in pop_sizes:
            pop_slices.append(slice(i, i+pop_size))
            i += pop_size
        return pop_slices

    #def __len__(self):
        #return len(self._data)

    @property
    def _data(self):
        # DEPRECATED: Added just to ease transition to deriving from ndarray
        return self.view(np.ndarray)

    @_data.setter
    def _data(self, value):
        assert(isinstance(value, np.ndarray))
        self[:] = value

    @property
    def nested(self):
        return super().ndim > 1
        #return self._data.dtype == np.object

    @property
    def ndim(self):
        if self.nested:
            return self[0].ndim + 1
        else:
            return super().ndim

    # TODO: Implement with dictionary interface instead, so we can iterate
    # over block sizes with `items()`
    def block_size(self, block_type):
        if block_type == 'Micro':
            return sum(self.pop_sizes)
        elif block_type == 'Meso':
            return len(self.pop_sizes)
        elif block_type == 'Macro':
            return 1
        elif block_type == 'Plain':
            return None
        else:
            raise ValueError("Unrecognized block type '{}'"
                             .format(block_type))

    @staticmethod
    def expand_block_types(popterm1, popterm2, shape1=None, shape2=None):
        bts1 = popterm1.block_types
        bts2 = popterm2.block_types
        if shape1 is None: shape1 = popterm1.shape
        if shape2 is None: shape2 = popterm2.shape

        # Check that shapes are compatible
        # Iterating from the end ensures that we discard the extra
        # initial dimensions in popterm1 or popterm2
        for bt1, size1, bt2, size2 in zip(bts1[::-1],
                                          shape1[::-1],
                                          bts2[::-1],
                                          shape2[::-1]):
            if ( ( (bt1 == 'Plain' or bt2 == 'Plain')
                   and (size1 != 1 and size2 != 1) )
                 and (size1 != size2) ) :
                raise ShapeError("Plain array shapes are incompatible.\n "
                                 + "  {} ({})\n".format(shape1,
                                                        popterm1.block_types)
                                 + "  {} ({})\n".format(shape2,
                                                        popterm2.block_types))

        # If one term has more dimensions, split those off:
        # they don't need to be expanded
        Δi = len(bts1) - len(bts2)
        if Δi > 0:
            bts  = bts1[:Δi]
            bts1 = bts1[Δi:]
            shape1 = shape1[Δi:]
        elif Δi < 0:
            Δi = abs(Δi)
            bts  = bts2[:Δi]
            bts2 = bts2[Δi:]
            shape2 = shape2[Δi:]
        else:
            bts = ()

        # For the remaining terms, take whichever is 'largest' at each position
        return bts + tuple(bt1 if size1 > size2 else bt2
                           for bt1, size1, bt2, size2
                           in zip(bts1, shape1, bts2, shape2))

    # @staticmethod
    # def _largest_block(bt1, bt2):
    #     for bt in ['Micro', 'Meso', 'Macro', 'Plain']:
    #         # Ordered from smallest to largest scale: return the first hit
    #         if bt1 == bt or bt2 == bt:
    #             return bt
    #     raise ValueError("Unrecognized block type(s) '{}', '{}'"
    #                      .format(bt1, bt2))

    def expand_blocks(self, block_types):
        if isinstance(block_types, str):
            # Special case: allow to expand all axes to same type
            block_types = (block_types,) * self.ndim
        if len(block_types) < self.ndim:
            raise ValueError("Specified expansion for only {} dimensions, but "
                             "we have {}."
                             .format(len(block_types), self.ndim))
        elif len(block_types) > self.ndim:
            block_types = block_types[-self.ndim:]
        res = self
        for dim, target in zip(range(self.ndim-1, -1, -1), block_types[::-1]):
            # For the default row-major order, I think it's better to expand
            # inner axes first. Not sure though.
            if target is not None and target not in ('Plain', 'None'):
                res = res.expand_axis(dim, target)

        return res

    def expand_axis(self, axis, target_type):
        """
        Parameters
        ----------
        axis: int
        target_type: string
            Indicates block_type
        """
        if axis < 0:
            axis = self.ndim + axis
        if self.block_types[axis] == 'Plain':
            # Plain axes are not expandable; they just use plain broadcasting
            return self
        if target_type == 'Plain':
            raise ValueError("'Plain' type has no defined size.")
        if axis > self.ndim:
            raise ValueError("Asked to expand axis {}, but this array has "
                             "only {} dimensions.".format(axis, self.ndim))
        if self.BlockLevels[self.block_types[axis]] < self.BlockLevels[target_type]:
            raise ShapeError("Cannot expand smaller scale '{}' to larger scale '{}' axis."
                             .format(self.block_types[axis], target_type))
        if self.block_types[axis] == target_type:
            return self
        else:
            # Shorthands used by all expansions
            block_types = (self.block_types[:axis] + (target_type,)
                           + self.block_types[axis+1:])
            cls = self.BlockTypes[block_types[0]]
            Δi = (self.ndim - 1 - axis)
            shape_pad = (1,) * Δi

            # Perform expansion based on our own and the target's block type
            # It's only possible to expand from larger to smaller scale
            if target_type == 'Macro':
                assert(False) # Any 'Macro' expansion should already be dealt with
            elif target_type == 'Meso':
                assert(self.block_types[axis] == 'Macro')
                    # All other possib ilities are either already dealt with or illegal
                assert(self.shape[axis] == 1)
                    # By definition of Macro, there is only one element on axis 'axis'
                slc = (Ellipsis, slice(0,1)) + (slice(None),) * Δi
                return cls(self.pop_sizes,
                           shim.tile( self[slc],
                                      (self.block_size('Meso'),) + shape_pad ),
                           block_types)
            elif target_type == 'Micro':
                if self.block_types[axis] == 'Macro':
                    slc = (Ellipsis, slice(0,1)) + (slice(None),) * Δi
                    return cls(self.pop_sizes,
                               shim.tile( self[slc],
                                          (self.block_size('Micro'),) + shape_pad ),
                               block_types)
                elif self.block_types[axis] == 'Meso':
                    slcs = [(Ellipsis, slice(i,i+1)) + (slice(None),) * Δi
                            for i in range(self.block_size('Meso'))]
                    return cls(self.pop_sizes,
                               shim.concatenate(
                                   [ np.tile( self[slc],
                                              (self.pop_sizes[i],) + shape_pad )
                                     for i, slc in enumerate(slcs) ],
                                   axis=axis ),
                               block_types)

                else:
                    assert(False)
            else:
                raise ValueError("Unrecognized expansion target type '{}'."
                                 .format(target_type))


    @property
    def expand(self):
        return self.expand_blocks('Micro')
    @property
    def expand_meso(self):
        return self.expand_blocks('Meso')

    @property
    def values(self):
        if self.nested:
            # We have nested PopTerms
            assert( all( isinstance(el, PopTerm) for el in self ) )
            return np.array([el.values for el in self])
        else:
            return self._data

    def get_value(self):
        # Fake a Shared variable
        return self.values

    # Ufunc override as per https://docs.scipy.org/doc/numpy/neps/ufunc-overrides.html#recommendations-for-implementing-binary-operations
    @staticmethod
    def _disables_array_ufunc(obj):
        try:
            return obj.__array_ufunc__ is None
        except AttributeError:
            return False

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if len(inputs) > 2:
            return NotImplemented
        else:
            # FIXME: This whole 'selfidx' business is because we need to
            #        remove 'self' from the arguments to '_apply_op'; fix
            #        '_apply_op', and we can make this less hacky.
            selfidx = [i for i, e in enumerate(inputs) if e is self]
            if len(selfidx) == 0:
                return NotImplemented
            selfidx = selfidx[0] # We just need any operand that is 'self'
            inputs = inputs[:selfidx] + inputs[selfidx+1:]
            #assert( all( i is not self for i in inputs ) )
                # assert(self not in inputs) calls __eq__, which leads to recursion error
            # if 'maximum' in str(ufunc):
            #     pass
            if len(inputs) == 0:
                op = lambda self: getattr(ufunc, method)(self)
            elif selfidx == 0:
                op = lambda self, b: getattr(ufunc, method)(self, b)
            else:
                assert(selfidx == 1 and len(inputs) == 1)
                op = lambda self, b: getattr(ufunc, method)(b, self)
            return self._apply_op( op, *inputs )

    def infer_block_types(self, shape, allow_plain=True):
        """
        Returns inferred block types (Micro, Meso, Macro, Plain)
        from the provided shape.
        'Plain' Type can be disallowed by specifying 'allow_plain=False'
        """
        block_types = tuple(
            'Macro' if size == self.block_size('Macro') else
            'Meso'  if size == self.block_size('Meso')  else
            'Micro' if size == self.block_size('Micro') else
            'Plain'
            for size in shape )
        if not allow_plain and 'Plain' in block_types:
            error_str = ("\n\nAt least one dimension is not compatible "
                         "with the population sizes;\n"
                         "Shape: {};\n".format(shape))
            # TODO: Use dictionary interface to `block_size` when implemented
            #       to avoid hard-coding block names
            block_size_str = ''.join(["  {}: {}\n".format(blockname, self.block_size(blockname))
                                      for blockname in ['Macro', 'Meso', 'Micro']])
            raise ShapeError(error_str + "Block sizes:\n" + block_size_str)
        return block_types

    def _apply_op(self, op, b=None):
        #FIXME: We assume op returns something of same (possibly broadcasted) shape as the inputs
        #       This is not always true, e.g. for .max()
        if b is None:
            return type(self)(self.pop_sizes, op(self._data), self.block_types)

        elif isinstance(b, PopTerm):
            if np.any(b.pop_sizes != self.pop_sizes):
                raise TypeError("Population terms must have the same population sizes.")
            block_types = self.expand_block_types(self, b)
            a1 = self.expand_blocks(block_types).view(np.ndarray)
            a2 =    b.expand_blocks(block_types).view(np.ndarray)
            cls = self.BlockTypes[block_types[0]]
            return cls(self.pop_sizes, op(a1, a2), block_types)

        elif isinstance(b, PopulationHistoryBase):
            # TODO: Implement operations on PopulationHistory (and raise NotImplementedError on Spiketimes if needed)
            #       Then we can remove NotImplementedError here
            raise NotImplementedError
            # if not b.pop_sizes == self.pop_sizes:
            #     raise TypeError("Population terms must have the same population sizes.")
            # return op(self._data, b)

        elif isinstance(b, Number):
            return type(self)(self.pop_sizes, op(self._data, b),
                              self.block_types)

        elif isinstance(b, np.ndarray):
            if b.ndim == 0:
                return type(self)(self.pop_sizes, op(self._data, b),
                                  self.block_types)
            else:
                # Infer block types from the array shape:
                # Dimensions beyond 'self' always remain 'Plain';
                # for the dimensions that align with self, assign the block
                # level that matches the shape, if it exists, other assign 'Plain'
                startdim = b.ndim - self.ndim
                block_types = self.infer_block_types(b.shape[:startdim])
                    # 'block_types' is used only to cast 'b': it does not need to
                    # include self's block type
                block_types += self.infer_block_types(b.shape[startdim:])
                cls = self.BlockTypes[block_types[0]]
                return op(self, cls(self.pop_sizes, b, block_types))

        else:
            raise TypeError("Unrecognized operand type '{}'.".format(type(b)))

    # Array methods
    # FIXME: Make max, min as ufuncs
    def max(self, *args, **kwargs):
        return np.max(self._data, *args, **kwargs)
    def min(self, *args, **kwargs):
        return np.min(self._data, *args, **kwargs)
    def any(self, *args, **kwargs):
        return np.array(self).any()
    def all(self, *args, **kwargs):
        return np.array(self).all()
    def dot(self, *args, **kwargs):
        # TODO: Check compatibility, do expansions
        res = self.view(np.ndarray).dot(*args, **kwargs)
        block_types = self.block_types[:-1]
            # Dot product removes last dimension of left operand
        if len(block_types) == 0:
            # Don't cast scalars as PopTerm
            return res
        else:
            cls = self.BlockTypes[block_types[0]]
            return cls(self.pop_sizes, res, block_types)
    def flatten(self, *args, **kwargs):
        # FIXME Only make affected axes plain ?
        res = self.view(np.ndarray).flatten(*args, **kwargs)
        return self.BlockTypes['Plain'](self.pop_sizes, res, ('Plain',)*res.ndim)


    # Operations
    def __abs__(self):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.abs(self)
    def __add__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.add(self, other)
    def __radd__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.add(other, self)
    def __sub__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.subtract(self, other)
    def __rsub__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.subtract(other, self)
    def __mul__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.multiply(self, other)
    def __rmul__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.multiply(other, self)
    def __matmul__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.matmul(self, other)
    def __rmatmul__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.matmul(other, self)
     # Using operator.matmul rather than @ prevents import fails on Python <3.5
    def __truediv__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.divide(self, other)
    def __rtruediv__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.divide(other, self)
    def __pow__(self, other, modulo=None):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.power(self, other)
    def __floordiv__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.floor_divide
    def __rfloordiv__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.floor_divide(other, self)
    def __mod__(self, other):
        if self._disables_array_ufunc(other):
            return NotImplemented
        return np.mod(self, other)

class PopTermPart(PopTerm):
    @property
    def expand(self):
        raise NotImplementedError
    @property
    def expand_meso(self):
        raise NotImplementedError

class PopTermMicro(PopTerm):
    """
    A variable which properly broadcasts when multiplied/added to a population History
    (e.g. SpikeTrain)
    One entry per population member
    """
    pass

class PopTermMeso(PopTerm):
    """
    A variable which properly broadcasts when multiplied/added to a population History
    (e.g. SpikeTrain)
    Checks that the population sizes match before applying operations.
    One entry per population.
    """
    pass

class PopTermMacro(PopTerm):
    """
    A variable which properly broadcasts when multiplied/added to a population History
    (e.g. SpikeTrain)
    Checks that the population sizes match before applying operations.
    Represents a single entry, common to all populations
    """
    pass

class ShimmedPopTerm(object):
    """
    WIP This is a first attempt to get basic PopTerm functionality with
    shared variables.
    """
    # The strategy is to multiply the inputs by an array of ones, broadcasting
    # each dimension to the Micro size. The possibly large-dimensional array
    # this defines should never actually be allocated, thanks to graph
    # optimizations. Moreover, the symbolic values are untouched and should
    # propagate correctly.
    #
    # **NOTE** Currently this is just a function return a Theano expression, and
    # in particular it **will not** create a `ShimmedPopTerm` instance.
    # So `isinstance()` checks on `PopTerm` will fail, as will operations
    # like +,* which rely on recognizing a PopTerm for their broadcasting
    # """
    def __new__(cls, pop_sizes, values, block_types):
        # TODO: Reduce duplication with PopTerm.__new__
        expected_size = ( 1 if issubclass(cls, ShimmedPopTermMacro)
                          else len(pop_sizes) if issubclass(cls, ShimmedPopTermMeso)
                          else sum(pop_sizes) if issubclass(cls, ShimmedPopTermMicro)
                          else None )
        assert shim.isshared(values)
        assert issubclass(cls, cls.BlockTypes[block_types[0]])
        if not issubclass(cls, PopTermPart):
            assert(expected_size is not None)
            if len(values) != expected_size:
                raise ValueError("Provided values (length {}) not commensurate with the "
                                "expected number of elements ({})."
                                .format(len(values), expected_size))
        return expand_array(pop_sizes, values, block_types)

def expand_array(pop_sizes, values, block_types):
    """
    Arrays are always expanded to the large 'Micro' size;
    `block_types` corresponds to the input sizes.
    """
    sharedvar = values; values = sharedvar.get_value(borrow=True)
    assert len(block_types) == shim.asarray(values).ndim

    c = sharedvar
    ndim = values.ndim
    microsize = sum(pop_sizes)
    for i, btype in enumerate(block_types):
        if btype == 'Macro':
            s = [1]*ndim; s[i] = microsize
            c *= shim.ones(s)
        elif btype == 'Meso':
            s = [1]*ndim
            k = [slice(None)]*ndim
            subarrs = []
            for j, popsize in enumerate(pop_sizes):
                s[i] = int(popsize)  # It seems arrays of numpy ints don't work
                k[i] = slice(j,j+1)
                subarrs.append(shim.tile(c[tuple(k)], s))
            c = shim.concatenate(subarrs, axis=i)
        else:
            assert btype == 'Micro'
            # Nothing to do: already expanded

    # HACK ? A PopTerm is expected to have an 'expand' property
    c.expand = c
    c.expand_meso = NotImplemented
    c.values = c
    return c

        # obj.pop_sizes = pop_sizes
        # obj.pop_slices = PopTerm._get_pop_slices(pop_sizes)
        # obj.block_types = tuple(block_types)
        # return obj

class ShimmedPopTermMicro(ShimmedPopTerm):
    pass

class ShimmedPopTermMeso(ShimmedPopTerm):
    pass

class ShimmedPopTermMacro(ShimmedPopTerm):
    pass

class ShimmedPopTermPart(ShimmedPopTerm):
    pass

PopTerm.BlockTypes = {'Micro': PopTermMicro,
                      'Meso': PopTermMeso,
                      'Macro': PopTermMacro,
                      'Plain': PopTermPart}
PopTerm.BlockNames = {PopTermMicro: 'Micro',
                      PopTermMeso: 'Meso',
                      PopTermMacro: 'Macro',
                      PopTermPart: 'Plain'}
PopTerm.BlockLevels = {'Micro': 0,
                       'Meso' : 1,
                       'Macro': 2,
                       'Plain': None}

ShimmedPopTerm.BlockTypes = {'Micro': ShimmedPopTermMicro,
                             'Meso': ShimmedPopTermMeso,
                             'Macro': ShimmedPopTermMacro,
                             'Plain': ShimmedPopTermPart}
ShimmedPopTerm.BlockNames = {ShimmedPopTermMicro: 'Micro',
                             ShimmedPopTermMeso: 'Meso',
                             ShimmedPopTermMacro: 'Macro',
                             ShimmedPopTermPart: 'Plain'}

PopTerm.shim_class = ShimmedPopTerm
PopTermMicro.shim_class = ShimmedPopTermMicro
PopTermMeso.shim_class = ShimmedPopTermMeso
PopTermMacro.shim_class = ShimmedPopTermMacro
