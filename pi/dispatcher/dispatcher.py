import inspect
import logging
from typing import Dict, Callable, List

from .function import Function, ClassFunction


__all__ = ["Dispatcher", "dispatch"]


log = logging.getLogger(__name__)

_manual_overrides: Dict[Callable, List[inspect.Signature]] = {}


def is_in_class(f):
    parts = f.__qualname__.split(".")
    return len(parts) >= 2 and parts[-2] != "<locals>"


class Dispatcher:
    def __init__(self):
        self._functions = {}
        self._classes = {}

    def __call__(self, overload=None, precedence=0):
        def construct_function():
            sig = inspect.signature(overload)
            # if (
            #     owner is not None
            #     and f"{owner.__module__}.{owner.__qualname__}" == get_class(overload)
            #     and "self" in sig.parameters
            # ):
            #     parameters = OrderedDict(sig.parameters)
            #     parameters["self"] = parameters["self"].replace(annotation=owner)
            #     sig = sig.replace(parameters=tuple(parameters.values()))
            return self._add_overload(
                overload,
                sig,
                precedence=precedence,
            )

        if is_in_class(overload):
            return ClassFunction(construct_function())
        return construct_function()

    def _get_function(self, overload) -> Function:
        name = overload.__name__
        if is_in_class(overload):
            if name not in self._classes:
                self._classes[name] = Function(overload)
            return self._classes[name]
        else:
            if name not in self._functions:
                self._functions[name] = Function(overload)
            return self._functions[name]

    def _add_overload(self, overload, signature, precedence):
        f = self._get_function(overload)
        f.register(signature, overload, precedence)
        return f

    def clear_cache(self):
        for f in self._functions.values():
            f.clear_cache()


dispatch = Dispatcher()
