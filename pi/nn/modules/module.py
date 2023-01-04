import inspect
import itertools
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Callable, Union

from ..._tensor import Tensor
from ...types_ import dtype as pi_dtype
from ..parameter import (
    Parameter,
    UninitializedParameter,
    UninitializedBuffer,
    is_uninitialized,
)
from ...utils import hooks
from ...utils.hooks import RemovableHandle


class Module:
    _parameters: Dict[str, Optional[Union[Parameter, UninitializedParameter]]]
    _buffers: Dict[str, Optional[Union[Tensor, UninitializedBuffer]]]
    _modules: Dict[str, Optional["Module"]]
    _forward_pre_hooks: OrderedDict[str, Callable]
    _forward_post_hooks: OrderedDict[str, Callable]
    _forward: Callable

    def __init__(self):
        _set = super().__setattr__
        _get = super().__getattribute__

        _set("_parameters", {})
        _set("_buffers", {})
        _set("_modules", {})
        _set("_forward_pre_hooks", OrderedDict())
        _set("_forward_post_hooks", OrderedDict())
        _set(
            "_initialize_hook",
            _get("register_forward_pre_hook")(_get("_infer_parameters")),
        )

        if "forward" in dir(self):
            orig_forward = _get("forward")
            # super attr is Module.__call__ d'oh
            call = self.__call__
            _set("_forward", orig_forward)
            _set("forward", call)
            # TODO(max): checks here
            if hasattr(orig_forward, "__placeholders__"):
                # setattr(call, "__annotations__", orig_forward.__annotations__)
                call.__dict__["__placeholders__"] = orig_forward.__placeholders__

        super(Module, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, *args, **kwargs):
        for hook_id, hook in self._forward_pre_hooks.items():
            result = hook(self, *args, **kwargs)  # type: ignore[misc]
            if result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    args, kwargs = result
                else:
                    raise RuntimeError(
                        "forward pre-hook must return None or a tuple "
                        f"of (new_args, new_kwargs), but got {result}."
                    )
        result = self._forward(*args, **kwargs)
        for hook_id, hook in self._forward_post_hooks.items():
            result = hook(self, result, *args, **kwargs)

        return result

    def _register_hook(
        self,
        hook: Callable[..., None],
        hook_dict: OrderedDict[str, Callable],
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        handle = hooks.RemovableHandle(hook_dict)
        hook_name = hook.__func__.__name__ if inspect.ismethod(hook) else hook.__name__
        hook_id = f"{hook_name}_{handle.id}"
        hook_dict[hook_id] = hook
        if prepend:
            hook_dict.move_to_end(hook_id, last=False)  # type: ignore[attr-defined]
        return handle

    def register_forward_pre_hook(
        self,
        hook: Callable[..., None],
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        return self._register_hook(hook, self._forward_pre_hooks, prepend=prepend)

    def register_forward_post_hook(
        self,
        hook: Callable[..., None],
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        return self._register_hook(hook, self._forward_post_hooks, prepend=prepend)

    def __getattribute__(self, item):
        return super(Module, self).__getattribute__(item)

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if value.__class__.__name__ == UninitializedParameter.__name__:
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
            )
            assert isinstance(
                value, UninitializedParameter
            ), f"class comparison failed {type(value)} {UninitializedParameter}"
            self.register_(name, value)
        elif name in self._parameters:
            assert value is None or (
                isinstance(self._parameters[name], UninitializedParameter)
                and isinstance(value, Tensor)
            ), f"{name}:{type(value).__module__}.{type(value).__name__} cannot override parameter {name}:{type(self._parameters[name]).__module__}.{type(self._parameters[name]).__name__}"
            self.register_(name, value)
        else:
            if isinstance(value, Module):
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                )
                self.register_module(name, value)
            elif name in self._modules:
                assert value is None, f"{type(value)} cannot override module {name}"
                self.register_module(name, value)
            else:
                if name in self._buffers:
                    assert value is None, f"{type(value)} cannot override buffer {name}"
                    self.register_buffer(name, value)
                else:
                    super().__setattr__(name, value)

    def __getattr__(
        self, name: str
    ) -> Union[Tensor, "Module", UninitializedParameter, UninitializedBuffer]:
        _parameters = self.__dict__["_parameters"]
        if name in _parameters:
            return _parameters[name]
        _buffers = self.__dict__["_buffers"]
        if name in _buffers:
            return _buffers[name]
        modules = self.__dict__["_modules"]
        if name in modules:
            return modules[name]
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")

    def register_buffer(
        self,
        name: str,
        tensor: Optional[Union[Tensor, UninitializedBuffer]],
        persistent: bool = True,
    ) -> None:
        self._buffers[name] = tensor

    def register_(
        self, name: str, param: Optional[Union[Parameter, UninitializedParameter]]
    ) -> None:
        self._parameters[name] = param

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        self._modules[name] = module

    def initialize_parameters(self, *_args, **_kwargs):
        parameters = self.__dict__["_parameters"]
        for name, param in sorted(parameters.items()):
            if param.__class__.__name__ == UninitializedParameter.__name__:
                assert isinstance(
                    param, UninitializedParameter
                ), f"class comparison failed {type(param)} {UninitializedParameter}"
                parameters[name] = param()

    def has_uninitialized_params(self):
        params = self._parameters.values()
        buffers = self._buffers.values()
        for param in itertools.chain(params, buffers):
            if is_uninitialized(param):
                return param
        return None

    def not_uninitialized(self):
        params = self._parameters.values()
        buffers = self._buffers.values()

        for param in itertools.chain(params, buffers):
            if not is_uninitialized(param):
                return param
        return None

    def _infer_parameters(self, _self, *args, **kwargs):
        self.initialize_parameters(*args, **kwargs)
        if uninitialized_param := self.has_uninitialized_params():
            raise RuntimeError(
                f"module {self.__class__.__name__} has not been fully initialized; {uninitialized_param}"
            )
            self._initialize_hook.remove()
            delattr(self, "_initialize_hook")

    def to(self, dtype: pi_dtype):
        if initialized_param := self.not_uninitialized():
            raise RuntimeError(
                f"module {self.__class__.__name__} has already been initialized; {initialized_param}"
            )

        for name, param in self._parameters.items():
            assert is_uninitialized(param), f"{param} already initialized"
            self._parameters[name] = UninitializedParameter(*param.size, dtype=dtype)

        for name, buffer in self._buffers.items():
            assert is_uninitialized(buffer), f"{buffer} already initialized"
            self._buffers[name] = UninitializedBuffer(*buffer.size, dtype=dtype)

        return self
