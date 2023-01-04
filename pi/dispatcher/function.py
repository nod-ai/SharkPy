import functools
from textwrap import indent
from typing import Tuple, Any, Optional, Dict, cast


from .operator_schemas import type_matches, create_type_hint, map_aggregate


class AmbiguousLookupError(LookupError):
    """A signature cannot be resolved due to ambiguity."""


class NotFoundLookupError(LookupError):
    """A signature cannot be resolved because no applicable method can be found."""


class Function:
    def __init__(self, f):
        self._f = f

        self._cache = {}
        self._overloads = {}

        # self.__name__ = "Dispatcher" + f.__name__
        # self.__qualname__ = "Dispatcher" + f.__qualname__
        # self.__module__ = "Dispatcher" + f.__module__
        self.__name__ = f.__name__
        self.__qualname__ = f.__qualname__
        self.__module__ = f.__module__

    def register(
        self,
        signature,
        f,
        precedence=0,
    ):
        assert (
            signature not in self._overloads
        ), f"{signature} already registered {self._overloads[signature]}"
        self._overloads[signature] = (f, precedence)

    def resolve_signature(
        self,
        args: Tuple[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        args_types: Optional[Tuple[Any]] = None,
        kwargs_types: Optional[Dict[str, Any]] = None,
    ):
        candidates = []
        for candidate_signature in self._overloads:
            try:
                candidate_signature.bind(*args, **kwargs)
                candidates.append(candidate_signature)
            except TypeError as e:
                continue

        if len(candidates) == 0:
            raise NotFoundLookupError(
                f'For function "{self._f.__name__}", '
                f"signature {args}, {kwargs} could not be resolved."
            )
        elif len(candidates) == 1:
            return candidates[0]
        else:
            args_types = args_types if args_types else cast(Tuple[Any], ())
            kwargs_types = kwargs_types if kwargs_types else {}
            for candidate_signature in self._overloads.keys():
                sig_matches = True
                try:
                    bound_types = candidate_signature.bind(*args_types, **kwargs_types)
                    for arg_name, arg_type in bound_types.arguments.items():
                        param = candidate_signature.parameters[arg_name]
                        sig_matches = sig_matches and type_matches(
                            param.annotation, arg_type
                        )
                except TypeError as e:
                    sig_matches = False
                if sig_matches:
                    return candidate_signature

        args = indent("\n".join(map(str, args)), "\t\t")
        kwargs = indent("\n".join(map(str, kwargs.items())), "\t\t")
        args_types = indent("\n".join(map(str, args_types)), "\t\t")
        kwargs_types = indent("\n".join(map(str, kwargs_types.items())), "\t\t")
        candidates = indent("\n".join(map(str, candidates)), "\t\t")
        raise AmbiguousLookupError(
            f"Tried to normalize arguments to {self._f.__name__} but failed for\n args:\n{args}\n kwargs:\n{kwargs}\n args_types:\n{args_types}\n kwargs_types:\n{kwargs_types}\n tried candidates:\n{candidates}\n"
        )

    def resolve_overload(self, signature):
        f, _ = self._overloads[signature]
        return f

    def __call__(self, *args, **kwargs):
        args_types = map_aggregate(args, type)
        assert isinstance(args_types, tuple)
        args_types = tuple([create_type_hint(i) for i in args_types])
        kwargs_types = {k: type(v) for k, v in kwargs.items()}
        signature = self.resolve_signature(args, kwargs, args_types, kwargs_types)
        f = self.resolve_overload(signature)
        return f(*args, **kwargs)

    def __repr__(self):
        return f"<function {self._f} with " f"{len(self._overloads)} overload(s)>"


class ClassFunction:
    _pending = []

    def __init__(self, construct_function):
        self.function = construct_function

    def __get__(self, instance, owner):
        return functools.partial(self.function, instance)
