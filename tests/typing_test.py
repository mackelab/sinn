"""
Created on Mon Nov 2 2020

Author: Alexandre René

Manifest:
    test_IndexableNamespace

"""

from pydantic import BaseModel
import mackelab_toolbox as mtb
import mackelab_toolbox.cgshim
import mackelab_toolbox.typing

def test_IndexableNamespace():

    mtb.typing.freeze_types()
    from sinn.common import IndexableNamespace

    class Foo(BaseModel):
        bar: IndexableNamespace
        class Config:
            json_encoders={IndexableNamespace: IndexableNamespace.json_encoder}

    foo = Foo(bar=IndexableNamespace(a=[], b=3))
    foo2 = Foo.parse_raw(foo.json())

    assert foo == foo2
    assert foo.bar is not foo2.bar
    assert foo.bar.a is not foo2.bar.a
    assert foo.bar.b is foo2.bar.b  # Python reuses literals
