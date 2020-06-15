import numpy as np
import copy
import builtins
from collections.abc import Iterable
import itertools
from numbers import Number
import operator

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

def parse_signature(argnames, args, kwargs):
    """Emulate signature parsing.
    Used in functions that must provide multiple signatures, and so only
    define *args, **kwargs.

    Parameters
    ----------
    argnames: iterable of strings
        The variable names which would be in a signature.
    args: tuple
    kwargs: dict

    Returns
    -------
    generator
        Values are ordered as in `argnames`.
    """
    # Parse signature
    i = 0
    for argname in argnames:
        if argname not in kwargs:
            kwargs[argname] = args[i]
            i += 1
    if i != len(args):  # Not all arguments were parsed
        raise TypeError("superfluous arguments in signature of "
                        "`{cls.__qualname__}.__new__`.")
    return (kwargs[k] for k in argnames)


# Defining the same type multiple times would lead to weird errors when checking
# with `isinstance` and `issubclass`
# We overload the `type` function to ensure we only define each type once
# TODO: Ensure 'data_type' attribute is always defined
def type(*args):
    # Don't catch any errors: let builtins.type raise the expected ones
    if len(args) <= 1:
        return builtins.type(*args)
    else:
        name = args[0]
        ct = type.created_types
        if name in ct:
            return ct[name]
        else:
            T = builtins.type(*args)
            ct[name] = T
            return T
type.created_types = {}

# Ufunc override as per https://docs.scipy.org/doc/numpy/neps/ufunc-overrides.html#recommendations-for-implementing-binary-operations
def _disables_array_ufunc(obj):
    try:
        return obj.__array_ufunc__ is None
    except AttributeError:
        return False

# Get consistent type name for PopTerm blocks
# TODO: remove superfluous prefix
def get_popterm_name(data_or_type, block_name=None):
    if isinstance(data_or_type, builtins.type):
        if issubclass(data_or_type, PopTerm):
            T = data_or_type.data_type
            assert T is not None
            if issubclass(data_or_type, SymbolicPopTerm):
                prefix = "Symbolic"
            else:
                prefix = "Numpy"
        else:
            T = data_or_type
            if issubclass(T, shim.config.ShimmedAndGraphTypes):
                prefix = "Symbolic"
            else:
                prefix = "Numpy"
    else:
        if isinstance(data_or_type, PopTerm):
            T = data_or_type.data_type
            assert T is not None
        else:
            T = type(data_or_type)
        if shim.is_shimmed_or_symbolic(data_or_type):
            prefix = "Symbolic"
        else:
            prefix = "Numpy"
    if block_name is None:
        block_name = ""
    return f"{prefix}PopTerm{block_name}({T.__qualname__})"

def _get_baseT(data_type, block=False):
    """
    block: bool
        True: Return a type derived from PopTermBlock
    """
    assert isinstance(data_type, builtins.type)
    if block:
        baseTname = get_popterm_name(data_type, "Block")
    else:
        baseTname = get_popterm_name(data_type)
    if baseTname in type.created_types:
        return type.created_types[baseTname]
    if block:
        Tbases = (PopTermBlock, _get_baseT(data_type, False))
    else:
        Tbases = ()
        if issubclass(data_type, shim.config.ShimmedAndGraphTypes):
            # If present, SymbolicPopTerm must come before NumpyPopTerm
            Tbases += (SymbolicPopTerm,)
        if issubclass(data_type, np.ndarray):
            Tbases += (NumpyPopTerm,)
        assert set(Tbases) & set((SymbolicPopTerm, NumpyPopTerm))
        Tbases += (data_type,)
    T = type(baseTname, Tbases, {'data_type': data_type})  # type() takes care of adding to `created_types`
    # Resolve MRO and update __bases__
    assert T.mro()[0] is T
    T.__bases__ = tuple(T.mro()[1:])
    # Construct the block types for this new PopTerm type
    if not block:  # Only the base type needs to define new BlockTypes
        _construct_block_types(T)
    return T

def _construct_block_types(PopTermT):
    assert PopTermT is not PopTerm
    if not hasattr(PopTermT, 'BlockTypes'):
        assert not hasattr(PopTermT, 'BlockNames')
    else:
        assert PopTermT.BlockTypes is PopTerm.BlockTypes
            # PopTermT.BlockTypes is not really defined: it is retrieving the
            # the value of its parent
    PopTermT.BlockTypes = {}
    PopTermT.BlockNames = {}
    # Note: PopTermT might already be a Block type, or some other subclass
    # of the base PopTerm type
    dataT = PopTermT.data_type
    # Retrieve the base block type
    baseblockT = _get_baseT(dataT, block=True)
    # baseT = _get_baseT(dataT)
    # if issubclass(dataT, (shim.ShimmedTensorShared, shim.config.SymbolicSharedType)):
    #     # SymbolicPopTerm must be before NumpyPopTerm in mro
    #     if SymbolicPopTerm not in bases: bases += (SymbolicPopTerm,)
    # if issubclass(dataT, np.ndarray):
    #     if NumpyPopTerm not in bases: bases += (NumpyPopTerm,)
    for block_name, blockT in PopTerm.BlockTypes.items():
        Tname = get_popterm_name(dataT, block_name)
        abcblockT  = PopTerm.BlockTypes[block_name]
            # The types in PopTerm.BlockTypes aren't really abstract base classes,
            # but they probably should be
            # Really what we want to do here is `abcblockT.register(T)`
        T = type(Tname, (baseblockT,abcblockT), {'data_type': dataT})
        assert T.mro()[0] is T
        # Update __bases__ with mro
        T.__bases__ = tuple(T.mro()[1:])
        PopTermT.BlockTypes[block_name] = T
        PopTermT.BlockNames[T] = block_name

def _normalize_data(data):
    """
    Cast data to a smaller set of data types.
    Allows supporting things like lists and tuples to initialize arrays.
    """
    # Used datatypes
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, shim.cf.ShimmedAndGraphTypes):
        return data
    # elif isinstance(data, shim.cf.GraphTypes):
    #     return data
    # Castable datatypes
    elif isinstance(data, (list, tuple)):
        return np.asarray(data)
    # Unsupported datatypes
    else:
        raise TypeError(f"PopTerm does not support '{type(data)}' data.")

def infer_shape(block_types, pop_sizes):
    """
    Return the shape of a data array with population sizes given by `pop_sizes`
    an aggregated according to the types `block_types`.
    """
    assert all(bt in ('Macro', 'Meso', 'Micro') for bt in block_types)
    return tuple( 1 if bt == 'Macro'
                  else len(pop_sizes) if bt == 'Meso'
                  else sum(pop_sizes)
                  for bt in block_types )

# class PopTerm(abc.ABC):
class PopTerm:
    def __new__(cls, pop_sizes, values, block_types):
        """

        Implementation notes
        --------------------
        Subclasses of PopTerm need not, but should, implement a `view` method
        akin to the one in NumPy's ndarray.
        Implementations might only support a subset of ndarray's uses, e.g. only
        support viewing to 1 or 2 types.
        """

        if not isinstance(values, cls):
            values = _normalize_data(values)
            # We determine which subclass to return based on the type of `values`
            # We perform no further construction: The __new__() method will call
            # PopTerm.__new__() recursively, and it is there that we finish
            # construction
            if shim.is_shimmed_or_symbolic(values):
                # if   cls is PopTerm: _cls = SymbolicPopTerm
                # elif issubclass(cls, SymbolicPopTerm): _cls = cls
                # else: raise TypeError("Attempted to construct non-shared PopTerm "
                #                       "for shared data.")
                return SymbolicPopTerm.__new__(cls, pop_sizes, values, block_types)
            else:
                # if   cls is PopTerm: _cls = NumpyPopTerm
                # elif issubclass(cls, NumpyPopTerm): _cls = cls
                # else: raise TypeError("Use NumpyPopTerm for non-shared data types. "
                #                       "(Or use `PopTerm`, and let the type be "
                #                       "determined automatically)
                assert isinstance(values, np.ndarray)
                return NumpyPopTerm.__new__(cls, pop_sizes, values, block_types)
        else:
            # We arrived here from within the initializer of a subclass
            # => `values` is the object we want to return
            assert isinstance(values, cls)
                # FIXME: Redundant
            assert getattr(values, 'data_type', None) is not None
            values.data_type   # Ensure that it doesn't raise 'NotImplementedError'
            obj = values
            obj.pop_sizes = pop_sizes
            obj.pop_slices = PopTerm._get_pop_slices(pop_sizes)
            obj.block_types = tuple(block_types)

            # # If this is the first time we create an instance of this class,
            # # we need to create the `BlockTypes` and `BlockNames` dictionaries
            # # FIXME: Redundant with PopTermBlock
            # # 'Tblock'    | 'T'           => old | new block type (types)
            # baseT = _get_baseT(type(values))
            # if not hasattr(baseT, 'BlockTypes') or baseT.BlockTypes == PopTerm.BlockTypes:
            #     # Having BlockTypes equal to those of PopTerm means they haven't been set
            #     # assert not hasattr(cls, 'BlockNames')
            #     _construct_block_types(cls)

            # for block_name, Tblock in obj.BlockTypes.items():
            #     Tbases = ( (cls,) if issubclass(cls, Tblock)
            #                else (Tblock,) if issubclass(Tblock, cls)
            #                else (Tblock, cls) )  # Temporary Tbases; we resolve mro later
            #        # Duplicate types in __bases__ are an error
            #     Tname = get_popterm_name(values, block_name)
            #     T = type(Tname, Tbases, {})
            #     # Resolve MRO and update __bases__
            #     assert T.mro()[0] is T
            #     # Remove any block types which are not parents to Tblock
            #     # This can happen if e.g. cls is already a BlockType
            #     bases = []
            #     for _T in T.mro()[1:]:  # Index [0] is T itself
            #         if (not issubclass(_T, PopTermBlock)
            #             or issubclass(_T, Tblock)
            #             or issubclass(Tblock, _T)):   # Not sure both directions are needed
            #             bases.append(_T)
            #     T.__bases__ = tuple(bases)
            #     obj.BlockTypes[block_name] = T
            #     del obj.BlockNames[Tblock]       # Remove old block type
            #     obj.BlockNames[T] = block_name   # Add new block type
        return obj

    def __init__(self, pop_sizes, values, block_types):
        # TODO: move anything that can be moved here
        # At the moment, this served only to catch calls to __init__ which
        # follow __new__, and may otherwise proceed to the underlying data type
        pass

    @classmethod
    def redirect_to_datatype__new__(cls, expected_argnames, args, kwargs):
        """
        Returns the result of datatype.__new__ if `args`, `kwargs` are
        inconsistent with `expected_sig`.
        Returns None if arguments are consistent with `expected_sig`.

        This function probably only makes sense to call within a __new__ method.

        Parameters
        ----------
        expected_argnames: iterable of strings
            The variable names of the normal (non-redirected) signature.
        args: tuple
        kwargs: dict

        Returns
        -------
        None | A variable of the type wrapped by this PopTerm (`cls.data_type`)
            None: Arguments are consistent with the expected signature. You may
                continue.
            (not None): You should abort construction and return this instead.
        """
        # # We will test against the expected variable names for a shared var
        # shared_argnames = ['name', 'type', 'strict','allow_downcast', 'container']
        #     # We exclude 'value' from this list because it is similar to 'values'
        #     # and might hide a typo
        # # Remove names that are in the expected signature
        # for name in expected_argnames:
        #     if name in shared_argnames:
        #         shared_argnames.remove(name)

        if ( len(args) + len(kwargs) > len(expected_argnames)
             or any(kw not in expected_argnames for kw in kwargs) ):
            # __new__ is being called to construct an instance of the underlying
            # data type.

            # __init__ would not be called automatically anyway, so instead of
            # redirecting to __new__ we initialize the class as usual
            # Since shared types don't copy the data by design, this should be fine
            return cls.data_type(*args, **kwargs)
        else:
            return None

    @classmethod
    def _validate_initializer(cls, pop_sizes, values, block_types):
        # Every subclass calls this at the beginning of its __new__ method
        expected_size = ( 1 if issubclass(cls, PopTermMacro)
                          else len(pop_sizes) if issubclass(cls, PopTermMeso)
                          else sum(pop_sizes) if issubclass(cls, PopTermMicro)
                          else None )
        assert issubclass(cls, cls.BlockTypes[block_types[0]])
        if not issubclass(cls, PopTermPart):
            assert(expected_size is not None)
            wrong_size = False
            if shim.isshared(values):
                wrong_size = (len(values.get_value()) != expected_size)
            elif not shim.is_symbolic(values):
                wrong_size = (len(values) != expected_size)
            elif shim.is_symbolic(values):
                test_value = getattr(values.tag, 'test_value', None)
                if test_value is not None:
                    wrong_size = (len(test_value) != expected_size)
                else:
                    # No way to get a test_value (and we don't want to use an
                    # expensive `eval()`), so we must trust the user.
                    pass
            if wrong_size:
                raise ValueError("Provided values (length {}) not commensurate "
                                 "with the expected number of elements ({})."
                                .format(len(values), expected_size))

    def _get_block_types_after_indexing(self, key, result_ndim):
        """Return the list of block types obtained after a slice operation."""
        if result_ndim == 0:
            return ()
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
                    dim_diff = result_ndim - expected_size
                        # The difference in no. of dimmensions is due to the Ellpisis
                        # We replace those dimensions by ':' (i.e. slice(None))
                    assert(dim_diff >= 0)
                    key = key[:ellidx] + (slice(None),)*dim_diff + key[ellidx+1:]
                #assert(len(key) == result_ndim)
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
                assert(len(block_types) == result_ndim)
            else:
                block_types = self.block_types

        return block_types

    # __getitem__  must be defined in a subclass, which is why we make
    # it an abstract method
    # However, subclass implementations may still call super(), which needs
    # to be delegated up the MRO, which is why we still need to implement that
    # delegation
    # Another solution would be to not implement the methods at all (so as not
    # to trap the super() delegation) and rely on documentation to tell users
    # to implement these methods in the subclass.
    # @abc.abstractmethod
    def __getitem__(self, key):
        return super().__getitem__(self, key)

    # @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError
    # @abc.abstractmethod
    def __repr(self):
        raise NotImplementedError

    @property
    # @abc.abstractmethod
    def data_type(self):
        return None  # None will trip an assertion in __new__ if data_type is not defined in derived class

    @property
    # @abc.abstractmethod
    def nested(self):
        raise NotImplementedError
    @property
    def ndim(self):
        return super().ndim

    # Abstract array methods
    # @abc.abstractmethod
    # def dot(self, *args, **kwargs):
    #     raise NotImplementedError
    # @abc.abstractmethod
    # def flatten(self):
    #     raise NotImplementedError
    def dot(self, *args, **kwargs):
        # TODO: Check compatibility, do expansions
        res = self.data_type.dot(self, *args, **kwargs)
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
        res = self.data_type.flatten(self, *args, **kwargs)
        return self.BlockTypes['Plain'](self.pop_sizes, res, ('Plain',)*res.ndim)

    # def get_value(self):
    #     # Fake a Shared variable
    #     return self.values
    #
    @staticmethod
    def _get_pop_slices(pop_sizes):
        pop_slices = []
        i = 0
        for pop_size in pop_sizes:
            pop_slices.append(slice(i, i+pop_size))
            i += pop_size
        return pop_slices

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
        if shape1 is None: shape1 = infer_shape(bts1, popterm1.pop_sizes)
        if shape2 is None: shape2 = infer_shape(bts2, popterm2.pop_sizes)

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
                                   [ shim.tile( self[slc],
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

    ############
    # Templates for array operations
    # Subclasses need to implement these themselves

    # # Projection operations – results dim < input dim
    # # FIXME: Make max, min as ufuncs
    # def max(self, *args, **kwargs):
    #     return self.data_type.max(*args, **kwargs)
    # def min(self, *args, **kwargs):
    #     return self.data_type.min(*args, **kwargs)
    # def any(self, *args, **kwargs):
    #     return self.data_type.any(self, *args, **kwargs)
    # def all(self, *args, **kwargs):
    #     return self.data_type.all(self, *args, **kwargs)
    #
    # # Shape-preserving operations
    # def __abs__(self):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__abs__(self)
    # def __add__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__add__(self, other)
    # def __radd__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__radd__(self, other)
    # def __sub__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__sub__(self, other)
    # def __rsub__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__rsub__(self, other)
    # def __mul__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__mul__(self, other)
    # def __rmul__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__rmul__(self, other)
    # def __matmul__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.__matmul__(self, other)
    # def __rmatmul__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__rmatmul__(self, other)
    # def __truediv__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__truediv__(self, other)
    # def __rtruediv__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__rtruediv__(self, other)
    # def __pow__(self, other, modulo=None):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__pow__(self, other)
    # def __floordiv__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__floordiv__(self, other)
    # def __rfloordiv__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__rfloordiv__(self, other)
    # def __mod__(self, other):
    #     if _disables_array_ufunc(other):
    #         return NotImplemented
    #     return self.data_type.__mod__(self, other)

class NumpyPopTerm(PopTerm, np.ndarray):
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
    # shim_class = SymbolicPopTerm
    data_type = np.ndarray

    def __new__(cls, pop_sizes, values, block_types):
        values = _normalize_data(values)
        cls._validate_initializer(pop_sizes, values, block_types)
        assert len(block_types) == np.asarray(values).ndim

        # TODO: assert: all pop_values elements are of same type
        # TODO: Remove the ugly nesting: allow N-D PopTerm
        if not isinstance(values, Iterable) or not isinstance(values[0], Iterable):
            obj = np.asarray(values)
        else:
            if len(block_types) > 2:
                raise NotImplementedError("Nested PopTerms only implemented "
                                          "for 2D data.")
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

        # obj.pop_sizes = pop_sizes
        # obj.pop_slices = PopTerm._get_pop_slices(pop_sizes)
        # obj.block_types = tuple(block_types)
        return PopTerm.__new__(cls, pop_sizes, obj, block_types)

    # ====================================================
    # ndarray subclassing

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
            if len(inputs) == 0:
                op = lambda self: getattr(ufunc, method)(self)
            elif selfidx == 0:
                op = lambda self, b: getattr(ufunc, method)(self, b)
            else:
                assert(selfidx == 1 and len(inputs) == 1)
                op = lambda self, b: getattr(ufunc, method)(b, self)
            return self._apply_op( op, *inputs )

    # ====================================================

    def __getitem__(self, key):
        res = self.data_type.__getitem__(self, key)

        if res.ndim == 0:
            return res

        block_types = self._get_block_types_after_indexing(key, res.ndim)
        cls = self.BlockTypes[block_types[0]]
        res = res.view(cls)

        # TODO: Reuse the initialization in PopTerm.__new__ ?
        res.block_types = block_types
        res.pop_sizes = self.pop_sizes
        res.pop_slices = self.pop_slices
        return res

    def __str__(self):
        return str(self.view(np.ndarray)) + " " + str(self.block_types)
    def __repr__(self):
        return repr(self.view(np.ndarray)) + " " + repr(self.block_types)

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

    # Shape-preserving operations
    def __abs__(self):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__abs__(self)
    def __add__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__add__(self, other)
    def __radd__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__radd__(self, other)
    def __sub__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__sub__(self, other)
    def __rsub__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__rsub__(self, other)
    def __mul__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__mul__(self, other)
    def __rmul__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__rmul__(self, other)
    def __matmul__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.__matmul__(self, other)
    def __rmatmul__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__rmatmul__(self, other)
    def __truediv__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__truediv__(self, other)
    def __rtruediv__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__rtruediv__(self, other)
    def __pow__(self, other, modulo=None):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__pow__(self, other)
    def __floordiv__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__floordiv__(self, other)
    def __rfloordiv__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__rfloordiv__(self, other)
    def __mod__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self.data_type.__mod__(self, other)


    # Projection operations – results dim < input dim
    # We need to bypass the ufunc calls because our ufunc implementation
    # assume the result has the same shape as the input
    # FIXME: Make these ufuncs
    def max(self, *args, **kwargs):
        return self.view(np.ndarray).max(*args, **kwargs)
    def min(self, *args, **kwargs):
        return self.view(np.ndarray).min(*args, **kwargs)
    def any(self, *args, **kwargs):
        return self.view(np.ndarray).any()
    def all(self, *args, **kwargs):
        return self.view(np.ndarray).all()

    # @property
    # def ndim(self):
    #     return super().ndim
    #
    # @staticmethod
    # def _largest_block(bt1, bt2):
    #     for bt in ['Micro', 'Meso', 'Macro', 'Plain']:
    #         # Ordered from smallest to largest scale: return the first hit
    #         if bt1 == bt or bt2 == bt:
    #             return bt
    #     raise ValueError("Unrecognized block type(s) '{}', '{}'"
    #                      .format(bt1, bt2))

    # @property
    # def values(self):
    #     return self.view(np.ndarray)
        # if self.nested:
        #     # We have nested PopTerms
        #     assert( all( isinstance(el, PopTerm) for el in self ) )
        #     return np.array([el.values for el in self])
        # else:
        #     return self._data


class PopTermBlock(PopTerm):
    def __new__(cls, *args, block_type=None, **kwargs):
        # `block_type` explicit here because it is passed by PopTermMicro and co.
        # and we want to catch it.
        """
        pop_sizes:   tuple|list of ints
        values:      array-like
        block_types: tuple|list of str
        block_type:  "outer" block type
        """
        # Expected signature
        sig = ('pop_sizes', 'values', 'block_types')
        # Detect if this is a SharedVar signature
        symbolic = cls.redirect_to_datatype__new__(sig, args, kwargs)
        if symbolic is not None:
            # Abort constructing the PopTerm and return a plain ShareVariable
            return symbolic
        # Parse expected signature
        pop_sizes, values, block_types = parse_signature(sig, args, kwargs)
        if block_type is None:
            raise TypeError("Required argument `block_type` is missing.")
        assert isinstance(block_type, str)

        values = _normalize_data(values)
        # Tblock = PopTerm.BlockTypes[block_type]
        # Tbases = ( (cls,) if issubclass(cls, Tblock)
        #            else (Tblock,) if issubclass(Tblock, cls)
        #            else (Tblock, cls) )  # Temporary Tbases; we resolve mro later
        #     # Duplicate types in __bases__ are an error
        # if shim.isshared(values):
        #     #Tvalues = type(shim.shared(0))
        #     Tbases += (SymbolicPopTerm,)
        # else:
        #     Tbases += (NumpyPopTerm,)
        baseT      = _get_baseT(type(values))
        baseblockT = _get_baseT(type(values), block=True)
        Tname = get_popterm_name(values, block_type)
        T = type(Tname, (baseblockT,), {'data_type': type(values)})
        # Resolve MRO and update __bases__
        assert T.mro()[0] is T
        T.__bases__ = tuple(T.mro()[1:])
        # Find earliest PopTerm type which is not a Block
        # Tbase = None
        # for _T in T.mro():
        #     if issubclass(_T, PopTerm) and not issubclass(_T, PopTermBlock):
        #         Tbase = _T
        #         break
        # assert Tbase is not None
        # Tbase = _get_baseT(type(values))
        return baseT.__new__(T, pop_sizes, values, block_types)

class PopTermPart(PopTermBlock):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, block_type='Plain', **kwargs)
    @property
    def expand(self):
        raise NotImplementedError
    @property
    def expand_meso(self):
        raise NotImplementedError

class PopTermMicro(PopTermBlock):
    """
    A variable which properly broadcasts when multiplied/added to a population History
    (e.g. Spiketrain)
    One entry per population member
    """
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, block_type='Micro', **kwargs)

class PopTermMeso(PopTermBlock):
    """
    A variable which properly broadcasts when multiplied/added to a population History
    (e.g. Spiketrain)
    Checks that the population sizes match before applying operations.
    One entry per population.
    """
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, block_type='Meso', **kwargs)

class PopTermMacro(PopTermBlock):
    """
    A variable which properly broadcasts when multiplied/added to a population History
    (e.g. Spiketrain)
    Checks that the population sizes match before applying operations.
    Represents a single entry, common to all populations
    """
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, block_type='Macro', **kwargs)

class SymbolicPopTerm(PopTerm):
    """
    This class extends PopTerm functionality to symbolic variables.
    Still in beta; expect some issues.
    """

    data_type = None  # Sentinel value marking unset; set in __new__

    def __new__(cls, *args, **kwargs):
        """
        Signature: cls, pop_sizes, values, block_types

        (the variadic arguments are used to make the constructor interoperable
        with the constructor of the underlying symbolic type)
        """
        # Expected signature
        sig = ('pop_sizes', 'values', 'block_types')
        # Detect if this is a signature for the underlying type
        symbolic = cls.redirect_to_datatype__new__(sig, args, kwargs)
        if symbolic is not None:
            # Abort constructing the PopTerm and return a plain symbolic type
            return symbolic
        # Parse expected signature
        pop_sizes, values, block_types = parse_signature(sig, args, kwargs)

        values = _normalize_data(values)
        cls._validate_initializer(pop_sizes, values, block_types)
        assert shim.is_shimmed_or_symbolic(values)
        # We create the type dynamically because `values` might be a
        # shimmed shared var, or an actual shared var
        Tvalues = type(values)  # The type we are wrapping with a PopTerm
        # Determine the types we need to derive from
        # => make sure none appears twice
        Tparents = (SymbolicPopTerm,)
        if issubclass(Tvalues, np.ndarray):
            if NumpyPopTerm not in Tparents: Tparents += (NumpyPopTerm,)
        if Tvalues not in Tparents: Tparents += (Tvalues,)
        if cls is SymbolicPopTerm:
            # Class needs to be specialized to the data type
            Tname = f"SymbolicPopTerm({Tvalues.__qualname__})"
            assert SymbolicPopTerm not in Tvalues.mro()
            T = type(Tname, Tparents, {'data_type': Tvalues})
        else:
            # Another __new__ method has presumably already specialized `cls`
            assert all(issubclass(cls, _T) for _T in Tparents)
            T = cls
        if NumpyPopTerm in Tparents:
            return NumpyPopTerm.__new__(T, pop_sizes, values, block_types)
        elif shim.is_graph_object(values):
            # HACK!!!! I really don't like monkey patching the class like this
            # Creating a new object with symbolic types is difficult, because
            # just copying the __dict__ invalidates the objects it points to
            # (I think). For relevant cloning functions, see in particular
            # theano.gof.graph.clone()  (this is the function called by `eval`)
            # => A much nicer solution would be to use `clone`, but specify
            # a subtype to which to clone to (cloning to same type does us no good)
            values.__class__ = T
            return PopTerm.__new__(T, pop_sizes, values, block_types)
        else:
            # Generic solution:  We instantiate an empty object, then update
            # its dict with that of `values`
            # Certain types (like ndarray) don't let you create a new instance
            # with object.__new__, which is why we need to special case them
            try:
                obj = object.__new__(T)
            except TypeError as e:
                raise TypeError("PopTerm does not support data of type "
                                f"'{cls}'.\nOriginal error message: {e}.")
            obj.__dict__.update(values.__dict__)
                # This makes a shallow copy to the new object.
                # So data isn't copied uselessly, but the original is protected
                # from changes to the new object
                # => Unless we change a mutable attribute, but then we probably
                # want the change to be propagated
            return PopTerm.__new__(T, pop_sizes, obj, block_types)

    def __getitem__(self, key):
        values = self.data_type.__getitem__(self, key)
        if values.ndim == 0:
            return values
        block_types = self._get_block_types_after_indexing(key, values.ndim)
        cls = self.BlockTypes[block_types[0]]
        res = cls(self.pop_sizes, values, block_types)
        return res

    def __str__(self):
        return self.data_type.__str__(self) + " " + str(self.block_types)
    def __repr__(self):
        return self.data_type.__repr__(self) + " " + repr(self.block_types)

    @property
    def nested(self):
        return self.ndim > 1

    def view(self, type):
        """Only supports type=ndarray.

        Parameters
        ----------
        type: type compatible with the data
            `ndarray`: equivalent to calling `self.getvalues()`.
            At present only `ndarray` is supported

        Returns
        -------
        A view of the data in the format specified by `type`.
        """
        if shim.is_pure_symbolic(self):
            raise TypeError("`view` is not supported for true symbolic PopTerms, "
                            "although it is supported for shared types.")
        if type is np.ndarray:
            if isinstance(self, np.ndarray):
                # A ShimmedTensorShared variable implements `get_value` with `view`
                # so using get_value() here would lead to infinite recursion
                return self.data_type.view(self, np.ndarray)
            else:
                return self.get_value()
        else:
            raise ValueError("SymbolicPopTerm can only produce a view to the "
                             "following type: ndarray.")

    def _apply_op(self, op, a=None, b=None):
        # a can (and often is) `self`
        # HACK: a is not really optional. Only so because sometimes NumpyPopTerm
        #       ends up calling here (in which case we redirect to NumpyPopTerm)
        #FIXME: We assume op returns something of same (possibly broadcasted) shape as the inputs
        #       This is not always true, e.g. for .max()
        #FIXME: Copied verbatim from NumpyPopTerm._apply_op with these changes:
        #  - isinstance(b, PopTerm):
        #        remove `view(np.ndarray)`
        #  - add `isinstance(b, shim.SymbolicType)`
        #  - remove `_data`
        # F

        # HACK: ShimmedTensorShared types need use the NumpyPopTerm, because they
        #       subclass ndarray.
        #       This should be fixed by a saner inheritance structure, and/or
        #       by merging the _apply_op methods
        if isinstance(self, np.ndarray):
            return NumpyPopTerm._apply_op(self, op, a)
        assert a is not None

        if b is None:
            return type(a)(self.pop_sizes, op(a), self.block_types)
        else:
            assert a is self or b is self

        if isinstance(b, PopTerm) and isinstance(a, PopTerm):
            if np.any(a.pop_sizes != b.pop_sizes):
                raise TypeError("Population terms must have the same population sizes.")
            block_types = self.expand_block_types(a, b)
            a1 =    a.expand_blocks(block_types)
            b1 =    b.expand_blocks(block_types)
            if isinstance(a1, np.ndarray): a1 = a1.view(np.ndarray)
            if isinstance(b1, np.ndarray): b1 = b1.view(np.ndarray)
            cls = self.BlockTypes[block_types[0]]
            return cls(a.pop_sizes, op(a1, b1), block_types)

        elif isinstance(b, PopulationHistoryBase):
            # TODO: Implement operations on PopulationHistory (and raise NotImplementedError on Spiketimes if needed)
            #       Then we can remove NotImplementedError here
            raise NotImplementedError
            # if not b.pop_sizes == self.pop_sizes:
            #     raise TypeError("Population terms must have the same population sizes.")
            # return op(self._data, b)

        elif isinstance(b, Number):
            return type(a)(self.pop_sizes, op(a, b),
                           self.block_types)
        elif isinstance(a, Number):
            return type(b)(self.pop_sizes, op(a, b),
                           self.block_types)

        elif shim.is_shimmed_or_symbolic(b):
            if b.ndim == 0:
                return type(a)(self.pop_sizes, op(a, b),
                               self.block_types)
            else:
                raise TypeError(
                    "Only 0-dim symbolic types can be multiplied with a PopTerm."
                    "To multiply a higher dim type, cast it as a PopTerm.")
        elif shim.is_shimmed_or_symbolic(a):
            if a.ndim == 0:
                return type(b)(self.pop_sizes, op(a, b),
                               self.block_types)
            else:
                raise TypeError(
                    "Only 0-dim symbolic types can be multiplied with a PopTerm."
                    "To multiply a higher dim type, cast it as a PopTerm.")

        elif isinstance(b, np.ndarray):
            # Use views as np.ndarray for the operations to avoid tripping
            # theano's assertions
            if b.ndim == 0:
                return type(self)(self.pop_sizes, op(a, b.view(np.ndarray)),
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
                return op(self,
                          cls(self.pop_sizes, b, block_types).view(np.ndarray))
        elif isinstance(a, np.ndarray):
            if a.ndim == 0:
                return type(self)(self.pop_sizes, op(a.view(np.ndarray), b),
                                  self.block_types)
            else:
                startdim = a.ndim - self.ndim
                block_types = self.infer_block_types(a.shape[:startdim])
                block_types += self.infer_block_types(a.shape[startdim:])
                cls = self.BlockTypes[block_types[0]]
                return op(cls(self.pop_sizes, a, block_types).view(np.ndarray),
                          b)

        else:
            raise TypeError("Unrecognized operand type '{}'.".format(type(b)))

    # Shape-preserving operations
    def __abs__(self):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__rmul__, self)
    def __add__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__add__, self, other)
    def __radd__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__radd__, self, other)
    def __sub__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__sub__, self, other)
    def __rsub__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__rsub__, self, other)
    def __mul__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__mul__, self, other)
    def __rmul__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__rmul__, self, other)
    def __matmul__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.__matmul__, self, other)
    def __rmatmul__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__rmatmul__, self, other)
    def __truediv__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__truediv__, self, other)
    def __rtruediv__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__rtruediv__, self, other)
    def __pow__(self, other, modulo=None):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__pow__, self, other)
    def __floordiv__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__floordiv__, self, other)
    def __rfloordiv__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__rfloordiv__, self, other)
    def __mod__(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return self._apply_op(self.data_type.__mod__, self, other)


    # Projection operations – results dim < input dim
    # We need to bypass the ufunc calls because our ufunc implementation
    # assume the result has the same shape as the input
    # FIXME: Make these ufuncs
    def max(self, *args, **kwargs):
        return self.view(np.ndarray).max(*args, **kwargs)
    def min(self, *args, **kwargs):
        return self.view(np.ndarray).min(*args, **kwargs)
    def any(self, *args, **kwargs):
        return self.view(np.ndarray).any()
    def all(self, *args, **kwargs):
        return self.view(np.ndarray).all()


def expand_array(pop_sizes, values, block_types):
    """
    Arrays are always expanded to the large 'Micro' size;
    `block_types` corresponds to the input sizes.
    """
    if shim.isshared(values):
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

#
# class NumpyPopTermPart (PopTermPart , NumpyPopTerm): pass
# class NumpyPopTermMicro(PopTermMicro, NumpyPopTerm): pass
# class NumpyPopTermMeso (PopTermMeso , NumpyPopTerm): pass
# class NumpyPopTermMacro(PopTermMacro, NumpyPopTerm): pass
# class SymbolicPopTermPart (PopTermPart , SymbolicPopTerm): pass
# class SymbolicPopTermMicro(PopTermMicro, SymbolicPopTerm): pass
# class SymbolicPopTermMeso (PopTermMeso , SymbolicPopTerm): pass
# class SymbolicPopTermMacro(PopTermMacro, SymbolicPopTerm): pass

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

# SymbolicPopTerm.BlockTypes = {'Micro': SymbolicPopTermMicro,
#                              'Meso': SymbolicPopTermMeso,
#                              'Macro': SymbolicPopTermMacro,
#                              'Plain': SymbolicPopTermPart}
# SymbolicPopTerm.BlockNames = {SymbolicPopTermMicro: 'Micro',
#                              SymbolicPopTermMeso: 'Meso',
#                              SymbolicPopTermMacro: 'Macro',
#                              SymbolicPopTermPart: 'Plain'}
#
# PopTerm.shim_class = SymbolicPopTerm
# PopTermMicro.shim_class = SymbolicPopTermMicro
# PopTermMeso.shim_class = SymbolicPopTermMeso
# PopTermMacro.shim_class = SymbolicPopTermMacro
