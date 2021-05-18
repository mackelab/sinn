from pydantic import BaseModel, validator
from pydantic.typing import AnyCallable
from typing import Callable as CallableT
from inspect import signature
from parameters import ParameterSet

## initializer decorator
def initializer(
    *fields, uninitialized=None, pre=True, always=False, **dec_kwargs
) -> CallableT[[AnyCallable], classmethod]:
    """
    Specialized validator for writing more complex default initializers with
    less boilerplate. Does two things:

    - Changes the default for `pre` to ``True``.
    - Always sets `always=True` (the `always` parameter is still accepted,
      but with a slightly different meaning; see note below).
    - Allows model parameters to be specified as keyword arguments in the
      validator signature. This works with both model-level parameters, and
      the parameters defined in the `Parameters` subclass.

    .. Note:: The point of an initializer is to replace a default value, so
       it doesn't make sense to set `always=False`. However, by default an
       initializer will *not* execute if a value is already provided.
       (The logic being that if a value is provided, it doesn't need to be
       initialized.) Thus, in analogy with `~pydantic.validator`, the `always`
       keyword is provided to specify that an initializer should be run even if
       a value for that parameter is provided.

    Example
    -------

    The following

    >>> class Model(BaseModel):
    >>>   a: float
    >>>   t: float = None
    >>>   @initializer('t'):
    >>>   def set_t(t, a):
    >>>     return a/4

    is equivalent to

    >>> class Model(BaseModel):
    >>>   a: float
    >>>   t: float = None
    >>>   @validator('t', pre=True, always=True):
    >>>   def set_t(t, values):
    >>>     if t is not None:
    >>>       return t
    >>>     a = values.get('a', None)
    >>>     if a is None:
    >>>       raise AssertionError(
    >>>         "'a' cannot be found within the model parameters. This may be "
    >>>         "because it is defined after 't' in the list of parameters, "
    >>>         "or because its own validation failed.")
    >>>     return a/4

    Parameters
    ----------
    *fields
    pre (default: True)
    each_item
    check_fields
    allow_reuse: As in `pydantic.validator`, although some arguments may not
        be so relevant.

    always: bool
        - `True`: Always run the initializer. This is the same as setting
          `always=True` with a Pydantic `~pydantic.validator`.
        - `False` (default): Only run the initializer when the value is **not**
          provided. Note that this is the opposite effect to setting
          `always=False` with a Pydantic `~pydantic.validator`.

    uninitialized: Any (default: None)
        The initializer is only executed when the parameter is equal to this
        value.
    """

    val_fn = validator(*fields, pre=pre, always=True, **dec_kwargs)

    # Refer to pydantic.class_validators.make_generic_validator
    def dec(f: AnyCallable) -> classmethod:
        sig = signature(f)
        args = list(sig.parameters.keys())
        # 'value' is the first argument != from 'self', 'cls'
        # It is positional, and the only required argument
        if args[0] in ('self', 'cls'):
            req_val_args = args[:2]
            opt_val_args = set(args[2:])  # Remove cls and value
        else:
            req_val_args = args[:1]
            opt_val_args = set(args[1:])  # Remove value
        # opt_validator_args will store the list of arguments recognized
        # by pydantic.validator. Everything else is assumed to match an earlier
        # parameter.
        param_args = set()
        for arg in opt_val_args:
            if arg not in ('values', 'config', 'field', '**kwargs'):
                param_args.add(arg)
        for arg in param_args:
            opt_val_args.remove(arg)
        def new_f(cls, v, values, field, config):
            if not always and v is not uninitialized:
                return v
            param_kwargs = {}
            params = values.get('params', None)
            if not isinstance(params, BaseModel):
                params = None  # We must not be within a sinn Model => 'params' does not have special meaning
            for p in param_args:
                if p in values:  # Try module-level param first
                    pval = values.get(p)
                elif params is not None and hasattr(params, p):
                    pval = getattr(params, p)
                else:
                    raise AssertionError(
                      f"'{p}' cannot be found within the model parameters. "
                      "This may be because it is "
                      f"defined after '{field.name}' in the list of parameters, "
                      "or because its own validation failed.")
                param_kwargs[p] = pval

            # Now assemble the expected standard arguments
            if len(req_val_args) == 2:
                val_args = (cls, v)
            else:
                val_args = (v,)
            val_kwargs = {}
            if 'values' in opt_val_args: val_kwargs['values'] = values
            if 'field' in opt_val_args: val_kwargs['field'] = field
            if 'config' in opt_val_args: val_kwargs['config'] = config

            return f(*val_args, **val_kwargs, **param_kwargs)

        # Can't use @wraps because we changed the signature
        new_f.__name__ = f.__name__
        # Having a different qualname is required to avoid overwriting validators
        # (Pydantic identifies them by name, and otherwise they all have `new_f`)
        new_f.__qualname__ = f.__qualname__
        new_f.__doc__ = f.__doc__

        return val_fn(new_f)

    return dec

def add_exclude_mask(exclude, mask):
    """
    Merge `exclude` and `mask` into a single set/dict, in the format
    expected by BaseModel's `json`, `dict` and `copy` methods.
    This is used to specialize these methods within particular BaseModels,
    in order to ensure certain attributes are always excluded from export.

    :param exclude: set | dict | None
    :param mask: set | dict | ParameterSet
        Hierarchies can be indicated with either nested dicts or by separating
        the levels in the key names with a period.
    :returns: set | dict
        Returns dict if `exclude` or `mask` are a dict, otherwise returns
        set.
    """
    if exclude is not None and not isinstance(exclude, (dict, set)):
        raise TypeError(f"Argument 'exclude' should be either a set or dict, "
                        f"but received '{exclude}'")
    if not isinstance(mask, (dict, set)):
        raise TypeError(f"Argument 'mask' should be either a set or dict, "
                        f"but received '{mask}'")
    if isinstance(exclude, dict) or isinstance(mask, dict):
        if exclude is None:
            exclude = {}
        elif isinstance(exclude, set):
            exclude = {attr: ... for attr in exclude}

        if isinstance(mask, set):
            exclude.update({attr: ... for attr in mask})
        else:
            assert isinstance(mask, dict)
            # Use ParameterSet to resolve dotted hierarchies
            for attr, excl in ParameterSet(mask).items():
                if excl is ...:
                    exclude[attr] = ...
                elif exclude.get(attr, None) is not ...:
                    # If it matches ..., there is nothing to do:
                    # everything under 'attr' is already excluded
                    exclude[attr] = add_exclude_mask(
                        exclude.get(attr, None), excl)
    else:
        if exclude is None:
            exclude = set()
        exclude = exclude | mask

    return exclude
