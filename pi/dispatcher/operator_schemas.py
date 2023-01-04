import inspect
import numbers
import typing
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    NamedTuple,
)

from ..types_ import Device, Number, Size, dtype

__all__ = [
    "ArgsKwargsPair",
    "create_type_hint",
    "type_matches",
]

import pi


class ArgsKwargsPair(NamedTuple):
    """
    Simple named tuple for wrapping args/kwargs pairs.
    """

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


_manual_overrides: Dict[Callable, List[inspect.Signature]] = {}

_type_eval_globals = {
    "Device": Device,
    "Size": Size,
    "pi_dtype": dtype,
    "Number": Number
    # "Layout": torch.layout,
    # "number": numbers.Number,
    # "Future": torch.jit.Future,
    # "AnyEnumType": enum.Enum,
    # "QScheme": torch.qscheme,
    # "__torch__": _FakeGlobalNamespace(),
    # "NoneType": type(None),
    # "t": typing.TypeVar("t"),
}
for k in dir(typing):
    _type_eval_globals[k] = getattr(typing, k)


def type_str_to_python_type(type_str) -> Any:
    from pi import Tensor
    return eval(type_str, _type_eval_globals | {"Tensor": Tensor})


def create_type_hint(x):
    try:
        if isinstance(x, list) or isinstance(x, tuple):
            # todo(chilli): Figure out the right way for mypy to handle this
            if isinstance(x, list):

                def ret_type(x):
                    return List[x]  # type: ignore[valid-type]

            else:

                def ret_type(x):
                    return Tuple[x, ...]

            if len(x) == 0:
                return ret_type(Any)
            base_type = x[0]
            for t in x:
                if issubclass(t, base_type):
                    continue
                elif issubclass(base_type, t):
                    base_type = t
                else:
                    return ret_type(Any)
            return ret_type(base_type)
    except Exception as e:
        # We tried to create a type hint for list but failed.
        warnings.warn(
            f"We were not able to successfully create type hint from the type {x} because of {e}"
        )
        pass
    return x


def type_matches(signature_type: Any, argument_type: Any):
    if isinstance(signature_type, str):
        signature_type = type_str_to_python_type(signature_type)
    assert not isinstance(signature_type, str), f"signature_type should be a real type {signature_type=}"
    # both sig type and arg type should be produced by type hints
    sig_origin_type = getattr(signature_type, "__origin__", signature_type)

    # hack because type annotations in ._tensor are actually strings (from __future__)
    # TODO(max): figure out how to annotate class methods correctly (in the dispatcher)
    if signature_type is argument_type:
        return True

    # Union types in signature. Given type needs to match one of the
    # contained types in the Union
    if sig_origin_type is typing.Union and signature_type != argument_type:
        sig_contained = signature_type.__args__
        return any(type_matches(c, argument_type) for c in sig_contained)

    if signature_type is List[int] and argument_type is int:
        # int can be promoted to List[int]
        return True

    if getattr(signature_type, "__origin__", None) in {list, List}:
        sig_el_type = signature_type.__args__[0]
        if not inspect.isclass(sig_el_type):
            warnings.warn(
                f"Does not support nested parametric types, got {signature_type}. Please file a bug."
            )
            return False
        if getattr(argument_type, "__origin__", None) in {list, List}:
            return issubclass(argument_type.__args__[0], sig_el_type)

        def is_homogeneous_tuple(t):
            if not getattr(t, "__origin__", None) in {tuple, Tuple}:
                return False
            contained = t.__args__
            if t.__args__ == ((),):  # Tuple[()].__args__ == ((),) for some reason
                return True
            return all((c is Ellipsis) or issubclass(c, sig_el_type) for c in contained)

        # Tuple[T] is accepted for List[T] parameters
        return is_homogeneous_tuple(argument_type)

    # Dtype is an int in schemas
    if signature_type is int and argument_type is pi.dtype:
        return True

    if signature_type is numbers.Number and argument_type in {int, float}:
        return True
    if inspect.isclass(argument_type) and inspect.isclass(signature_type):
        return issubclass(argument_type, signature_type)

    return False


def _args_kwargs_to_normalized_args_kwargs(
    sig: inspect.Signature,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    normalize_to_only_use_kwargs: bool,
) -> Optional[ArgsKwargsPair]:
    # Don't currently support positional-only
    # or varargs (*args, **kwargs) signatures
    supported_parameter_types = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
        return None

    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    new_kwargs: Dict[str, Any] = {}
    new_args: List[Any] = []
    for i, param in enumerate(sig.parameters):
        if not normalize_to_only_use_kwargs and i < len(args):
            new_args.append(bound_args.arguments[param])
        else:
            new_kwargs[param] = bound_args.arguments[param]

    return ArgsKwargsPair(tuple(new_args), new_kwargs)


def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(
        f"'{type(self).__name__}' object does not support mutation."
    )


def _create_immutable_container(base, mutable_functions):
    container = type("immutable_" + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container


immutable_list = _create_immutable_container(
    list,
    [
        "__delitem__",
        "__iadd__",
        "__imul__",
        "__setitem__",
        "append",
        "clear",
        "extend",
        "insert",
        "pop",
        "remove",
    ],
)
immutable_list.__reduce__ = lambda self: (immutable_list, (tuple(iter(self)),))


immutable_dict = _create_immutable_container(
    dict, ["__delitem__", "__setitem__", "clear", "pop", "popitem", "update"]
)
immutable_dict.__reduce__ = lambda self: (immutable_dict, (iter(self.items()),))


def map_aggregate(a, fn: Callable):
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if isinstance(a, tuple):
        t = tuple(map_aggregate(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(a, "_fields") else type(a)(*t)
    elif isinstance(a, list):
        return immutable_list(map_aggregate(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return immutable_dict((k, map_aggregate(v, fn)) for k, v in a.items())
    elif isinstance(a, slice):
        return slice(
            map_aggregate(a.start, fn),
            map_aggregate(a.stop, fn),
            map_aggregate(a.step, fn),
        )
    else:
        return fn(a)
