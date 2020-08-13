# -*- coding: utf-8 -*-
"""
Created on Sat Feb 4 2017

Copyright 2017-2020 Alexandre René
"""

from pydantic import BaseModel
from sinn.utils.pydantic import add_exclude_mask

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
        excl = kw.pop('exclude', None)
        excl = add_exclude_mask(excl, {'cached_ops'})
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
    def dict(self, *a, exclude=None, **kw):
        exclude = add_exclude_mask(exclude, {'cached_ops'})
        return super().dict(*a, exclude=exclude, **kw)
    def json(self, *a, exclude=None, **kw):
        exclude = add_exclude_mask(exclude, {'cached_ops'})
        return super().json(*a, exclude=exclude, **kw)

    def clear(self, *a, **kw):
        # All cached binary ops are now invalid, so delete them
        # (`clear` can be called in the middle of `copy`, when cached_ops isn't set)
        for op in getattr(self, 'cached_ops', []):
            op.clear()
        try:
            super().clear(*a, **kw)
        except AttributeError:
            pass
