# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Author: Alexandre René
"""

from pydantic import BaseModel

class CachedOperation(BaseModel):
    """All op mixins which contain an OpCache object should inherit
    from this class."""
    # __slots__ = ('cached_ops',)
        # Setting __slots__ causes TypeError: multiple bases have instance lay-out conflict
    cached_ops : list = []

    # def __init__(self, *a, **kw):
    #     object.__setattr__(self, 'cached_ops', [])
    #     super().__init__(*a, **kw)
    def copy(self, *a, **kw):
        excl = set(kw.pop('exclude', set()))
        excl.add('cached_ops')
        m = super().copy(*a, exclude=excl, **kw)
        m.cached_ops = []
        # object.__setattr__(m, 'cached_ops', [])
        return m
    @classmethod
    def parse_obj(cls, *a, **kw):
        m = super().parse_obj(*a, **kw)
        m.cached_ops = []
        # object.__setattr__(m, 'cached_ops', [])
        return m
    def dict(cls, *a, **kw):
        excl = set(kw.pop('exclude', set()))
        excl.add('cached_ops')
        return super().dict(*a, exclude=excl, **kw)
    def json(cls, *a, **kw):
        excl = set(kw.pop('exclude', set()))
        excl.add('cached_ops')
        return super().json(*a, exclude=excl, **kw)

    def clear(self):
        # All cached binary ops are now invalid, so delete them
        # (`clear` can be called in the middle of `copy`, when cached_ops isn't set)
        for op in getattr(self, 'cached_ops', []):
            op.clear()
        try:
            super().clear()
        except AttributeError:
            pass
