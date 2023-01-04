from __future__ import annotations
import builtins
import warnings
from numbers import Number
from typing import (
    Tuple,
    Optional,
    Any,
    List,
    Dict,
    Callable,
    Union,
    Sequence,
    Generator,
    Literal,
)
import numpy as np
import pi
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.dialects._ods_common import get_op_result_or_value
from torch_mlir.ir import DenseElementsAttr
from torch_mlir.ir import Value as MLIRValue
from .types_ import (
    dtype as pi_dtype,
    Size,
    device,
    layout,
    Device,
    memory_format,
    contiguous_format,
    is_a_torch_tensor
)
from .dispatcher import dispatch
from pi._mlir import _Torch_Tensor


class ComplexReturnType:
    def __init__(self, name):
        self.name = name


class Tensor(_Torch_Tensor):
    requires_grad: bool
    shape: Size
    data: Tensor
    names: List[str]
    device: device
    dtype: pi_dtype
    layout: layout
    real: Tensor
    imag: Tensor
    T: Tensor
    H: Tensor
    mT: Tensor
    mH: Tensor
    ndim: int
    output_nr: int
    _version: int
    _base: Optional[Tensor]
    _cdata: int
    grad_fn: Any
    _grad_fn: Any
    _grad: Optional[Tensor]
    grad: Optional[Tensor]
    _backward_hooks: Optional[Dict[int, Callable[[Tensor], Optional[Tensor]]]]

    @classmethod
    def _is_pi_tensor(self):
        return True

    @property
    def __class__(self):
        return MLIRValue

    def __init__(self, tensor: MLIRValue):
        tensor = get_op_result_or_value(tensor)
        super(Tensor, self).__init__(tensor._CAPIPtr)

    def __abs__(self: Tensor) -> Tensor:
        return pi.abs(self)

    def __add__(self: Tensor, other: Any) -> Tensor:
        return pi.add(self, other)

    @dispatch
    def __and__(self: Tensor, other: Tensor) -> Tensor:
        return pi.__and__(self, other)

    @dispatch
    def __and__(self: Tensor, other: Number) -> Tensor:
        return pi.__and__(self, other)

    @dispatch
    def __and__(self: Tensor, other: Any) -> Tensor:
        return pi.__and__(self, other)

    def __bool__(self: Tensor) -> builtins.bool:
        raise NotImplementedError("__bool__")

    def __complex__(self: Tensor) -> builtins.complex:
        raise NotImplementedError("__complex__")

    def __div__(self: Tensor, other: Tensor) -> Tensor:
        return pi.div(self, other)

    def __eq__(self: Tensor, other: Any) -> Tensor:
        return pi.eq(self, other)

    def __float__(self: Tensor) -> builtins.float:
        raise NotImplementedError("__float__")

    def __floordiv__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__floordiv__")

    def __ge__(self: Tensor, other: Any) -> Tensor:
        return pi.ge(self, other)

    def __getitem__(
        self: Tensor, indices: Union[None, int, slice, Tensor, List, Tuple]
    ) -> Tensor:
        if not isinstance(indices, Sequence):
            indices = [indices]

        t = self
        for i, ind in enumerate(indices):
            if isinstance(ind, int):
                t = pi.slice(t, ind)
            elif isinstance(ind, slice):
                t = pi.slice(t, dim=i, start=ind.start, end=ind.stop, step=ind.step or 1)

        return t


    def __gt__(self: Tensor, other: Any) -> Tensor:
        return pi.gt(self, other)

    def __iadd__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__iadd__")

    @dispatch
    def __iand__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__iand__")

    @dispatch
    def __iand__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__iand__")

    @dispatch
    def __iand__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__iand__")

    def __idiv__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__idiv__")

    def __ifloordiv__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__ifloordiv__")

    @dispatch
    def __ilshift__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__ilshift__")

    @dispatch
    def __ilshift__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__ilshift__")

    @dispatch
    def __ilshift__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__ilshift__")

    def __imod__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__imod__")

    def __imul__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__imul__")

    def __index__(self: Tensor) -> builtins.int:
        raise NotImplementedError("__index__")

    def __int__(self: Tensor) -> builtins.int:
        raise NotImplementedError("__int__")

    def __invert__(self: Tensor) -> Tensor:
        raise NotImplementedError("__invert__")

    @dispatch
    def __ior__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__ior__")

    @dispatch
    def __ior__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__ior__")

    @dispatch
    def __ior__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__ior__")

    @dispatch
    def __irshift__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__irshift__")

    @dispatch
    def __irshift__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__irshift__")

    @dispatch
    def __irshift__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__irshift__")

    def __isub__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__isub__")

    @dispatch
    def __ixor__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__ixor__")

    @dispatch
    def __ixor__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__ixor__")

    @dispatch
    def __ixor__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__ixor__")

    def __le__(self: Tensor, other: Any) -> Tensor:
        return pi.le(self, other)

    def __long__(self: Tensor) -> builtins.int:
        raise NotImplementedError("__long__")

    @dispatch
    def __lshift__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__lshift__")

    @dispatch
    def __lshift__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__lshift__")

    @dispatch
    def __lshift__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__lshift__")

    def __lt__(self: Tensor, other: Any) -> Tensor:
        return pi.lt(self, other)

    def __matmul__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__matmul__")

    def __mod__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__mod__")

    def __mul__(self: Tensor, other: Tensor) -> Tensor:
        return pi.mul(self, other)

    def __ne__(self: Tensor, other: Any) -> Tensor:
        return pi.ne(self, other)

    def __neg__(self: Tensor) -> Tensor:
        return pi.neg(self)

    def __nonzero__(self: Tensor) -> builtins.bool:
        raise NotImplementedError("__nonzero__")

    @dispatch
    def __or__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__or__")

    @dispatch
    def __or__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__or__")

    @dispatch
    def __or__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__or__")

    def __pow__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__pow__")

    def __radd__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__radd__")

    def __rand__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__rand__")

    def __rfloordiv__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__rfloordiv__")

    def __rmul__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__rmul__")

    def __ror__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__ror__")

    def __rpow__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__rpow__")

    @dispatch
    def __rshift__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__rshift__")

    @dispatch
    def __rshift__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__rshift__")

    @dispatch
    def __rshift__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__rshift__")

    def __rsub__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__rsub__")

    def __rtruediv__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__rtruediv__")

    def __rxor__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__rxor__")

    def __setitem__(
        self: Tensor,
        indices: Union[None, int, slice, Tensor, List, Tuple],
        val: Union[Tensor, Number],
    ) -> None:
        raise NotImplementedError("__setitem__")

    def __sub__(self: Tensor, other: Tensor) -> Tensor:
        return pi.sub(self, other)

    def __truediv__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__truediv__")

    @dispatch
    def __xor__(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("__xor__")

    @dispatch
    def __xor__(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("__xor__")

    @dispatch
    def __xor__(self: Tensor, other: Any) -> Tensor:
        raise NotImplementedError("__xor__")

    def __repr__(self):
        return "Tensor" + str(self)

    def _addmm_activation(
        self: Tensor,
        mat1: Tensor,
        mat2: Tensor,
        *,
        beta: Number = 1,
        alpha: Number = 1,
        use_gelu: bool = False,
    ) -> Tensor:
        raise NotImplementedError("_addmm_activation")

    def _autocast_to_full_precision(
        self: Tensor, cuda_enabled: bool, cpu_enabled: bool
    ) -> Tensor:
        raise NotImplementedError("_autocast_to_full_precision")

    def _autocast_to_reduced_precision(
        self: Tensor,
        cuda_enabled: bool,
        cpu_enabled: bool,
        cuda_dtype: pi_dtype,
        cpu_dtype: pi_dtype,
    ) -> Tensor:
        raise NotImplementedError("_autocast_to_reduced_precision")

    def _coalesced_(self: Tensor, coalesced: bool) -> Tensor:
        raise NotImplementedError("_coalesced_")

    def _conj(self: Tensor) -> Tensor:
        raise NotImplementedError("_conj")

    def _conj_physical(self: Tensor) -> Tensor:
        raise NotImplementedError("_conj_physical")

    def _dimI(self: Tensor) -> int:
        raise NotImplementedError("_dimI")

    def _dimV(self: Tensor) -> int:
        raise NotImplementedError("_dimV")

    def _indices(self: Tensor) -> Tensor:
        raise NotImplementedError("_indices")

    def _is_view(self: Tensor) -> bool:
        raise NotImplementedError("_is_view")

    def _is_zerotensor(self: Tensor) -> bool:
        raise NotImplementedError("_is_zerotensor")

    def _make_subclass(
        cls,
        data: Tensor,
        require_grad: bool = False,
        dispatch_strides: bool = False,
        dispatch_device: bool = False,
        device_for_backend_keys: Optional[device] = None,
    ) -> Tensor:
        raise NotImplementedError("_make_subclass")

    def _neg_view(self: Tensor) -> Tensor:
        raise NotImplementedError("_neg_view")

    def _nested_tensor_size(self: Tensor) -> Tensor:
        raise NotImplementedError("_nested_tensor_size")

    def _nnz(self: Tensor) -> int:
        raise NotImplementedError("_nnz")

    def _to_dense(self: Tensor, dtype: Optional[pi_dtype] = None) -> Tensor:
        raise NotImplementedError("_to_dense")

    def _values(self: Tensor) -> Tensor:
        raise NotImplementedError("_values")

    def abs(self: Tensor) -> Tensor:
        return pi.abs(self)

    def abs_(self: Tensor) -> Tensor:
        return pi.abs_(self)

    def absolute(self: Tensor) -> Tensor:
        raise NotImplementedError("absolute")

    def absolute_(self: Tensor) -> Tensor:
        raise NotImplementedError("absolute_")

    def acos(self: Tensor) -> Tensor:
        raise NotImplementedError("acos")

    def acos_(self: Tensor) -> Tensor:
        raise NotImplementedError("acos_")

    def acosh(self: Tensor) -> Tensor:
        raise NotImplementedError("acosh")

    def acosh_(self: Tensor) -> Tensor:
        raise NotImplementedError("acosh_")

    def add(
        self: Tensor,
        other: Union[Tensor, Number],
        *,
        alpha: Optional[Number] = 1,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        if out is not None:
            raise NotImplementedError("add.out variant")
        return pi.add(self, other, alpha)

    def add_(
        self: Tensor, other: Union[Tensor, Number], *, alpha: Optional[Number] = 1
    ) -> Tensor:
        return pi.add_(self, other, alpha)

    def addbmm(
        self: Tensor,
        batch1: Tensor,
        batch2: Tensor,
        *,
        beta: Number = 1,
        alpha: Number = 1,
    ) -> Tensor:
        raise NotImplementedError("addbmm")

    def addbmm_(
        self: Tensor,
        batch1: Tensor,
        batch2: Tensor,
        *,
        beta: Number = 1,
        alpha: Number = 1,
    ) -> Tensor:
        raise NotImplementedError("addbmm_")

    def addcdiv(
        self: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: Number = 1
    ) -> Tensor:
        return pi.addcdiv(self, tensor1, tensor2, value)

    def addcdiv_(
        self: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: Number = 1
    ) -> Tensor:
        return pi.addcdiv_(self, tensor1, tensor2, value)

    def addcmul(
        self: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: Number = 1
    ) -> Tensor:
        return pi.addcmul(self, tensor1, tensor2, value)

    def addcmul_(
        self: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: Number = 1
    ) -> Tensor:
        return pi.addcmul_(self, tensor1, tensor2, value)

    def addmm(
        self: Tensor, mat1: Tensor, mat2: Tensor, *, beta: Number = 1, alpha: Number = 1
    ) -> Tensor:
        return pi.addmm(self, mat1, mat2, beta, alpha)

    def addmm_(
        self: Tensor, mat1: Tensor, mat2: Tensor, *, beta: Number = 1, alpha: Number = 1
    ) -> Tensor:
        raise NotImplementedError("addmm_")

    def addmv(
        self: Tensor, mat: Tensor, vec: Tensor, *, beta: Number = 1, alpha: Number = 1
    ) -> Tensor:
        raise NotImplementedError("addmv")

    def addmv_(
        self: Tensor, mat: Tensor, vec: Tensor, *, beta: Number = 1, alpha: Number = 1
    ) -> Tensor:
        raise NotImplementedError("addmv_")

    def addr(
        self: Tensor, vec1: Tensor, vec2: Tensor, *, beta: Number = 1, alpha: Number = 1
    ) -> Tensor:
        raise NotImplementedError("addr")

    def addr_(
        self: Tensor, vec1: Tensor, vec2: Tensor, *, beta: Number = 1, alpha: Number = 1
    ) -> Tensor:
        raise NotImplementedError("addr_")

    def adjoint(self: Tensor) -> Tensor:
        raise NotImplementedError("adjoint")

    def align_as(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("align_as")

    @dispatch
    def align_to(
        self: Tensor, order: Sequence[Union[str, ellipsis, None]], ellipsis_idx: int
    ) -> Tensor:
        raise NotImplementedError("align_to")

    @dispatch
    def align_to(self: Tensor, names: Sequence[Union[str, ellipsis, None]]) -> Tensor:
        raise NotImplementedError("align_to")

    @dispatch
    def all(self: Tensor) -> Tensor:
        return pi.all(self)

    @dispatch
    def all(self: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return pi.all(self, dim, keepdim)

    @dispatch
    def all(
        self: Tensor, dim: Union[str, ellipsis, None], keepdim: bool = False
    ) -> Tensor:
        raise NotImplementedError("all")

    def allclose(
        self: Tensor,
        other: Tensor,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        raise NotImplementedError("allclose")

    def amax(self: Tensor, dim: Union[int, Size] = (), keepdim: bool = False) -> Tensor:
        return pi.amax(self, dim, keepdim)

    def amin(self: Tensor, dim: Union[int, Size] = (), keepdim: bool = False) -> Tensor:
        raise NotImplementedError("amin")

    def aminmax(
        self: Tensor, *, dim: Optional[int] = None, keepdim: bool = False
    ) -> ComplexReturnType("aminmax"):
        raise NotImplementedError("aminmax")

    def angle(self: Tensor) -> Tensor:
        raise NotImplementedError("angle")

    @dispatch
    def any(self: Tensor) -> Tensor:
        return pi.any(self)

    @dispatch
    def any(self: Tensor, dim: int, keepdim: bool = False) -> Tensor:
        return pi.any(self, dim, keepdim)

    @dispatch
    def any(
        self: Tensor, dim: Union[str, ellipsis, None], keepdim: bool = False
    ) -> Tensor:
        raise NotImplementedError("any")

    def apply_(self: Tensor, callable: Callable) -> Tensor:
        raise NotImplementedError("apply_")

    def arccos(self: Tensor) -> Tensor:
        raise NotImplementedError("arccos")

    def arccos_(self: Tensor) -> Tensor:
        raise NotImplementedError("arccos_")

    def arccosh(self: Tensor) -> Tensor:
        raise NotImplementedError("arccosh")

    def arccosh_(self: Tensor) -> Tensor:
        raise NotImplementedError("arccosh_")

    def arcsin(self: Tensor) -> Tensor:
        raise NotImplementedError("arcsin")

    def arcsin_(self: Tensor) -> Tensor:
        raise NotImplementedError("arcsin_")

    def arcsinh(self: Tensor) -> Tensor:
        raise NotImplementedError("arcsinh")

    def arcsinh_(self: Tensor) -> Tensor:
        raise NotImplementedError("arcsinh_")

    def arctan(self: Tensor) -> Tensor:
        raise NotImplementedError("arctan")

    def arctan2(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("arctan2")

    def arctan2_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("arctan2_")

    def arctan_(self: Tensor) -> Tensor:
        raise NotImplementedError("arctan_")

    def arctanh(self: Tensor) -> Tensor:
        raise NotImplementedError("arctanh")

    def arctanh_(self: Tensor) -> Tensor:
        raise NotImplementedError("arctanh_")

    def argmax(
        self: Tensor, dim: Optional[int] = None, keepdim: bool = False
    ) -> Tensor:
        return pi.argmax(self, dim, keepdim)

    def argmin(
        self: Tensor, dim: Optional[int] = None, keepdim: bool = False
    ) -> Tensor:
        raise NotImplementedError("argmin")

    @dispatch
    def argsort(
        self: Tensor, *, stable: bool, dim: int = -1, descending: bool = False
    ) -> Tensor:
        raise NotImplementedError("argsort")

    @dispatch
    def argsort(self: Tensor, dim: int = -1, descending: bool = False) -> Tensor:
        raise NotImplementedError("argsort")

    @dispatch
    def argsort(
        self: Tensor, dim: Union[str, ellipsis, None], descending: bool = False
    ) -> Tensor:
        raise NotImplementedError("argsort")

    def argwhere(self: Tensor) -> Tensor:
        raise NotImplementedError("argwhere")

    def as_strided(
        self: Tensor,
        size: List[int],
        stride: List[int],
        storage_offset: Optional[int] = None,
    ) -> Tensor:
        raise NotImplementedError("as_strided")

    def as_strided_(
        self: Tensor,
        size: List[int],
        stride: List[int],
        storage_offset: Optional[int] = None,
    ) -> Tensor:
        raise NotImplementedError("as_strided_")

    def as_strided_scatter(
        self: Tensor,
        src: Tensor,
        size: List[int],
        stride: List[int],
        storage_offset: Optional[int] = None,
    ) -> Tensor:
        return pi.as_strided_scatter(self, src, size, stride, storage_offset)

    def asin(self: Tensor) -> Tensor:
        raise NotImplementedError("asin")

    def asin_(self: Tensor) -> Tensor:
        raise NotImplementedError("asin_")

    def asinh(self: Tensor) -> Tensor:
        raise NotImplementedError("asinh")

    def asinh_(self: Tensor) -> Tensor:
        raise NotImplementedError("asinh_")

    def atan(self: Tensor) -> Tensor:
        raise NotImplementedError("atan")

    def atan2(self: Tensor, other: Tensor) -> Tensor:
        return pi.atan2(self, other)

    def atan2_(self: Tensor, other: Tensor) -> Tensor:
        return pi.atan2_(self, other)

    def atan_(self: Tensor) -> Tensor:
        raise NotImplementedError("atan_")

    def atanh(self: Tensor) -> Tensor:
        raise NotImplementedError("atanh")

    def atanh_(self: Tensor) -> Tensor:
        raise NotImplementedError("atanh_")

    def baddbmm(
        self: Tensor,
        batch1: Tensor,
        batch2: Tensor,
        *,
        beta: Number = 1,
        alpha: Number = 1,
    ) -> Tensor:
        return pi.baddbmm(self, batch1, batch2, beta, alpha)

    def baddbmm_(
        self: Tensor,
        batch1: Tensor,
        batch2: Tensor,
        *,
        beta: Number = 1,
        alpha: Number = 1,
    ) -> Tensor:
        return pi.baddbmm_(self, batch1, batch2, beta, alpha)

    @dispatch
    def bernoulli(self: Tensor, *, generator: Optional[Generator] = None) -> Tensor:
        raise NotImplementedError("bernoulli")

    @dispatch
    def bernoulli(
        self: Tensor, p: float, *, generator: Optional[Generator] = None
    ) -> Tensor:
        return pi.bernoulli(self, p, generator)

    @dispatch
    def bernoulli_(
        self: Tensor, p: Tensor, *, generator: Optional[Generator] = None
    ) -> Tensor:
        return pi.bernoulli_(self, p, generator)

    @dispatch
    def bernoulli_(
        self: Tensor, p: float = 0.5, *, generator: Optional[Generator] = None
    ) -> Tensor:
        return pi.bernoulli_(self, p, generator)

    def bfloat16(self: Tensor) -> Tensor:
        raise NotImplementedError("bfloat16")

    def bincount(
        self: Tensor, weights: Optional[Tensor] = None, minlength: int = 0
    ) -> Tensor:
        return pi.bincount(self, weights, minlength)

    @dispatch
    def bitwise_and(self: Tensor, other: Tensor) -> Tensor:
        return pi.bitwise_and(self, other)

    @dispatch
    def bitwise_and(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_and")

    @dispatch
    def bitwise_and_(self: Tensor, other: Tensor) -> Tensor:
        return pi.bitwise_and_(self, other)

    @dispatch
    def bitwise_and_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_and_")

    @dispatch
    def bitwise_left_shift(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("bitwise_left_shift")

    @dispatch
    def bitwise_left_shift(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_left_shift")

    @dispatch
    def bitwise_left_shift_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("bitwise_left_shift_")

    @dispatch
    def bitwise_left_shift_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_left_shift_")

    def bitwise_not(self: Tensor) -> Tensor:
        return pi.bitwise_not(self)

    def bitwise_not_(self: Tensor) -> Tensor:
        return pi.bitwise_not_(self)

    @dispatch
    def bitwise_or(self: Tensor, other: Tensor) -> Tensor:
        return pi.bitwise_or(self, other)

    @dispatch
    def bitwise_or(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_or")

    @dispatch
    def bitwise_or_(self: Tensor, other: Tensor) -> Tensor:
        return pi.bitwise_or_(self, other)

    @dispatch
    def bitwise_or_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_or_")

    @dispatch
    def bitwise_right_shift(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("bitwise_right_shift")

    @dispatch
    def bitwise_right_shift(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_right_shift")

    @dispatch
    def bitwise_right_shift_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("bitwise_right_shift_")

    @dispatch
    def bitwise_right_shift_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_right_shift_")

    @dispatch
    def bitwise_xor(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("bitwise_xor")

    @dispatch
    def bitwise_xor(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_xor")

    @dispatch
    def bitwise_xor_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("bitwise_xor_")

    @dispatch
    def bitwise_xor_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("bitwise_xor_")

    def bmm(self: Tensor, mat2: Tensor) -> Tensor:
        return pi.bmm(self, mat2)

    def bool(self: Tensor) -> Tensor:
        raise NotImplementedError("bool")

    @dispatch
    def broadcast_to(self: Tensor, size: List[int]) -> Tensor:
        return pi.broadcast_to(self, size)

    @dispatch
    def broadcast_to(self: Tensor, *size: int) -> Tensor:
        return pi.broadcast_to(self, *size)

    def byte(self: Tensor) -> Tensor:
        raise NotImplementedError("byte")

    def cauchy_(
        self: Tensor,
        median: float = 0,
        sigma: float = 1,
        *,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        raise NotImplementedError("cauchy_")

    def ccol_indices(self: Tensor) -> Tensor:
        raise NotImplementedError("ccol_indices")

    def ceil(self: Tensor) -> Tensor:
        return pi.ceil(self)

    def ceil_(self: Tensor) -> Tensor:
        return pi.ceil_(self)

    def chalf(self: Tensor, *, memory_format: Optional[memory_format] = None) -> Tensor:
        raise NotImplementedError("chalf")

    def char(self: Tensor) -> Tensor:
        raise NotImplementedError("char")

    def cholesky(self: Tensor, upper: bool = False) -> Tensor:
        raise NotImplementedError("cholesky")

    def cholesky_inverse(self: Tensor, upper: bool = False) -> Tensor:
        raise NotImplementedError("cholesky_inverse")

    def cholesky_solve(self: Tensor, input2: Tensor, upper: bool = False) -> Tensor:
        raise NotImplementedError("cholesky_solve")

    def chunk(self: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
        raise NotImplementedError("chunk")

    @dispatch
    def clamp(
        self: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
    ) -> Tensor:
        return pi.clamp(self, min, max)

    @dispatch
    def clamp(
        self: Tensor, min: Optional[Number] = None, max: Optional[Number] = None
    ) -> Tensor:
        return pi.clamp(self, min, max)

    @dispatch
    def clamp_(
        self: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
    ) -> Tensor:
        return pi.clamp_(self, min, max)

    @dispatch
    def clamp_(
        self: Tensor, min: Optional[Number] = None, max: Optional[Number] = None
    ) -> Tensor:
        return pi.clamp_(self, min, max)

    @dispatch
    def clamp_max(self: Tensor, max: Tensor) -> Tensor:
        raise NotImplementedError("clamp_max")

    @dispatch
    def clamp_max(self: Tensor, max: Number) -> Tensor:
        return pi.clamp_max(self, max)

    @dispatch
    def clamp_max_(self: Tensor, max: Tensor) -> Tensor:
        raise NotImplementedError("clamp_max_")

    @dispatch
    def clamp_max_(self: Tensor, max: Number) -> Tensor:
        return pi.clamp_max_(self, max)

    @dispatch
    def clamp_min(self: Tensor, min: Tensor) -> Tensor:
        raise NotImplementedError("clamp_min")

    @dispatch
    def clamp_min(self: Tensor, min: Number) -> Tensor:
        return pi.clamp_min(self, min)

    @dispatch
    def clamp_min_(self: Tensor, min: Tensor) -> Tensor:
        raise NotImplementedError("clamp_min_")

    @dispatch
    def clamp_min_(self: Tensor, min: Number) -> Tensor:
        return pi.clamp_min_(self, min)

    @dispatch
    def clip(
        self: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError("clip")

    @dispatch
    def clip(
        self: Tensor, min: Optional[Number] = None, max: Optional[Number] = None
    ) -> Tensor:
        raise NotImplementedError("clip")

    @dispatch
    def clip_(
        self: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError("clip_")

    @dispatch
    def clip_(
        self: Tensor, min: Optional[Number] = None, max: Optional[Number] = None
    ) -> Tensor:
        raise NotImplementedError("clip_")

    def clone(self: Tensor, *, memory_format: Optional[memory_format] = None) -> Tensor:
        return pi.clone(self, memory_format)

    def coalesce(self: Tensor) -> Tensor:
        raise NotImplementedError("coalesce")

    def col_indices(self: Tensor) -> Tensor:
        raise NotImplementedError("col_indices")

    def conj(self: Tensor) -> Tensor:
        raise NotImplementedError("conj")

    def conj_physical(self: Tensor) -> Tensor:
        raise NotImplementedError("conj_physical")

    def conj_physical_(self: Tensor) -> Tensor:
        raise NotImplementedError("conj_physical_")

    def contiguous(self: Tensor, memory_format=contiguous_format) -> Tensor:
        return pi.contiguous(self, memory_format)

    def copy_(self: Tensor, src: Tensor, non_blocking: bool = False) -> Tensor:
        return pi.copy_(self, src, non_blocking)

    @dispatch
    def copysign(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("copysign")

    @dispatch
    def copysign(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("copysign")

    @dispatch
    def copysign_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("copysign_")

    @dispatch
    def copysign_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("copysign_")

    def corrcoef(self: Tensor) -> Tensor:
        raise NotImplementedError("corrcoef")

    def cos(self: Tensor) -> Tensor:
        return pi.cos(self)

    def cos_(self: Tensor) -> Tensor:
        return pi.cos_(self)

    def cosh(self: Tensor) -> Tensor:
        raise NotImplementedError("cosh")

    def cosh_(self: Tensor) -> Tensor:
        raise NotImplementedError("cosh_")

    @dispatch
    def count_nonzero(self: Tensor, dim: Optional[int] = None) -> Tensor:
        raise NotImplementedError("count_nonzero")

    @dispatch
    def count_nonzero(self: Tensor, dim: Size) -> Tensor:
        raise NotImplementedError("count_nonzero")

    @dispatch
    def count_nonzero(self: Tensor, *dim: int) -> Tensor:
        raise NotImplementedError("count_nonzero")

    def cov(
        self: Tensor,
        *,
        correction: int = 1,
        fweights: Optional[Tensor] = None,
        aweights: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError("cov")

    def cpu(self: Tensor) -> Tensor:
        raise NotImplementedError("cpu")

    def cross(self: Tensor, other: Tensor, dim: Optional[int] = None) -> Tensor:
        raise NotImplementedError("cross")

    def crow_indices(self: Tensor) -> Tensor:
        raise NotImplementedError("crow_indices")

    def cuda(
        self: Tensor,
        device: Optional[Union[device, int, str]] = None,
        non_blocking: bool = False,
    ) -> Tensor:
        raise NotImplementedError("cuda")

    @dispatch
    def cummax(self: Tensor, dim: int) -> ComplexReturnType("cummax"):
        raise NotImplementedError("cummax")

    @dispatch
    def cummax(
        self: Tensor, dim: Union[str, ellipsis, None]
    ) -> ComplexReturnType("cummax"):
        raise NotImplementedError("cummax")

    @dispatch
    def cummin(self: Tensor, dim: int) -> ComplexReturnType("cummin"):
        raise NotImplementedError("cummin")

    @dispatch
    def cummin(
        self: Tensor, dim: Union[str, ellipsis, None]
    ) -> ComplexReturnType("cummin"):
        raise NotImplementedError("cummin")

    @dispatch
    def cumprod(self: Tensor, dim: int, *, dtype: Optional[pi_dtype] = None) -> Tensor:
        raise NotImplementedError("cumprod")

    @dispatch
    def cumprod(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("cumprod")

    @dispatch
    def cumprod_(self: Tensor, dim: int, *, dtype: Optional[pi_dtype] = None) -> Tensor:
        raise NotImplementedError("cumprod_")

    @dispatch
    def cumprod_(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("cumprod_")

    @dispatch
    def cumsum(self: Tensor, dim: int, *, dtype: Optional[pi_dtype] = None) -> Tensor:
        return pi.cumsum(self, dim, dtype)

    @dispatch
    def cumsum(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("cumsum")

    @dispatch
    def cumsum_(self: Tensor, dim: int, *, dtype: Optional[pi_dtype] = None) -> Tensor:
        raise NotImplementedError("cumsum_")

    @dispatch
    def cumsum_(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("cumsum_")

    def data_ptr(self: Tensor) -> int:
        raise NotImplementedError("data_ptr")

    def deg2rad(self: Tensor) -> Tensor:
        raise NotImplementedError("deg2rad")

    def deg2rad_(self: Tensor) -> Tensor:
        raise NotImplementedError("deg2rad_")

    def dense_dim(self: Tensor) -> int:
        raise NotImplementedError("dense_dim")

    def dequantize(self: Tensor) -> Tensor:
        raise NotImplementedError("dequantize")

    def det(self: Tensor) -> Tensor:
        raise NotImplementedError("det")

    def detach(self: Tensor) -> Tensor:
        raise NotImplementedError("detach")

    def detach_(self: Tensor) -> Tensor:
        raise NotImplementedError("detach_")

    def diag(self: Tensor, diagonal: int = 0) -> Tensor:
        raise NotImplementedError("diag")

    def diag_embed(
        self: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1
    ) -> Tensor:
        raise NotImplementedError("diag_embed")

    def diagflat(self: Tensor, offset: int = 0) -> Tensor:
        raise NotImplementedError("diagflat")

    @dispatch
    def diagonal(
        self: Tensor,
        *,
        outdim: Union[str, ellipsis, None],
        dim1: Union[str, ellipsis, None],
        dim2: Union[str, ellipsis, None],
        offset: int = 0,
    ) -> Tensor:
        raise NotImplementedError("diagonal")

    @dispatch
    def diagonal(self: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
        raise NotImplementedError("diagonal")

    def diagonal_scatter(
        self: Tensor, src: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1
    ) -> Tensor:
        return pi.diagonal_scatter(self, src, offset, dim1, dim2)

    def diff(
        self: Tensor,
        n: int = 1,
        dim: int = -1,
        prepend: Optional[Tensor] = None,
        append: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError("diff")

    def digamma(self: Tensor) -> Tensor:
        raise NotImplementedError("digamma")

    def digamma_(self: Tensor) -> Tensor:
        raise NotImplementedError("digamma_")

    def dim(self: Tensor) -> int:
        return pi.dim(self)

    def dist(self: Tensor, other: Tensor, p: Number = 2) -> Tensor:
        raise NotImplementedError("dist")

    def div(
        self: Tensor,
        other: Union[Tensor, Number],
        *,
        rounding_mode: Optional[str] = None,
    ) -> Tensor:
        return pi.div(self, other, rounding_mode)

    def div_(
        self: Tensor,
        other: Union[Tensor, Number],
        *,
        rounding_mode: Optional[str] = None,
    ) -> Tensor:
        return pi.div_(self, other, rounding_mode)

    @dispatch
    def divide(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("divide")

    @dispatch
    def divide(self: Tensor, other: Tensor, *, rounding_mode: Optional[str]) -> Tensor:
        raise NotImplementedError("divide")

    @dispatch
    def divide(self: Tensor, other: Number, *, rounding_mode: Optional[str]) -> Tensor:
        raise NotImplementedError("divide")

    @dispatch
    def divide(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("divide")

    @dispatch
    def divide_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("divide_")

    @dispatch
    def divide_(self: Tensor, other: Tensor, *, rounding_mode: Optional[str]) -> Tensor:
        raise NotImplementedError("divide_")

    @dispatch
    def divide_(self: Tensor, other: Number, *, rounding_mode: Optional[str]) -> Tensor:
        raise NotImplementedError("divide_")

    @dispatch
    def divide_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("divide_")

    def dot(self: Tensor, tensor: Tensor) -> Tensor:
        raise NotImplementedError("dot")

    def double(self: Tensor) -> Tensor:
        return torch_cast(self, pi_dtype.float64)

    @dispatch
    def dsplit(self: Tensor, sections: int) -> List[Tensor]:
        raise NotImplementedError("dsplit")

    @dispatch
    def dsplit(self: Tensor, indices: Size) -> List[Tensor]:
        raise NotImplementedError("dsplit")

    @dispatch
    def dsplit(self: Tensor, *indices: int) -> List[Tensor]:
        raise NotImplementedError("dsplit")

    def element_size(self: Tensor) -> int:
        raise NotImplementedError("element_size")

    @dispatch
    def eq(self: Tensor, other: Tensor) -> Tensor:
        return pi.eq(self, other)

    @dispatch
    def eq(self: Tensor, other: Number) -> Tensor:
        return pi.eq(self, other)

    @dispatch
    def eq_(self: Tensor, other: Tensor) -> Tensor:
        return pi.eq_(self, other)

    @dispatch
    def eq_(self: Tensor, other: Number) -> Tensor:
        return pi.eq_(self, other)

    def equal(self: Tensor, other: Tensor) -> bool:
        raise NotImplementedError("equal")

    def erf(self: Tensor) -> Tensor:
        return pi.erf(self)

    def erf_(self: Tensor) -> Tensor:
        return pi.erf_(self)

    def erfc(self: Tensor) -> Tensor:
        raise NotImplementedError("erfc")

    def erfc_(self: Tensor) -> Tensor:
        raise NotImplementedError("erfc_")

    def erfinv(self: Tensor) -> Tensor:
        raise NotImplementedError("erfinv")

    def erfinv_(self: Tensor) -> Tensor:
        raise NotImplementedError("erfinv_")

    def exp(self: Tensor) -> Tensor:
        return pi.exp(self)

    def exp2(self: Tensor) -> Tensor:
        raise NotImplementedError("exp2")

    def exp2_(self: Tensor) -> Tensor:
        raise NotImplementedError("exp2_")

    def exp_(self: Tensor) -> Tensor:
        return pi.exp_(self)

    @dispatch
    def expand(self: Tensor, size: List[int], *, implicit: bool = False) -> Tensor:
        return pi.expand(self, size, implicit)

    @dispatch
    def expand(self: Tensor, *size: int, implicit: bool = False) -> Tensor:
        raise NotImplementedError("expand")

    def expand_as(self: Tensor, other: Tensor) -> Tensor:
        return pi.expand_as(self, other)

    def expm1(self: Tensor) -> Tensor:
        return pi.expm1(self)

    def expm1_(self: Tensor) -> Tensor:
        return pi.expm1_(self)

    def exponential_(
        self: Tensor, lambd: float = 1, *, generator: Optional[Generator] = None
    ) -> Tensor:
        raise NotImplementedError("exponential_")

    @dispatch
    def fill_(self: Tensor, value: Tensor) -> Tensor:
        return pi.fill_(self, value)

    @dispatch
    def fill_(self: Tensor, value: Number) -> Tensor:
        return pi.fill_(self, value)

    def fill_diagonal_(self: Tensor, fill_value: Number, wrap: bool = False) -> Tensor:
        raise NotImplementedError("fill_diagonal_")

    def fix(self: Tensor) -> Tensor:
        raise NotImplementedError("fix")

    def fix_(self: Tensor) -> Tensor:
        raise NotImplementedError("fix_")

    @dispatch
    def flatten(self: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        return pi.flatten(self, start_dim, end_dim)

    @dispatch
    def flatten(
        self: Tensor, start_dim: int, end_dim: int, out_dim: Union[str, ellipsis, None]
    ) -> Tensor:
        raise NotImplementedError("flatten")

    @dispatch
    def flatten(
        self: Tensor,
        start_dim: Union[str, ellipsis, None],
        end_dim: Union[str, ellipsis, None],
        out_dim: Union[str, ellipsis, None],
    ) -> Tensor:
        raise NotImplementedError("flatten")

    @dispatch
    def flatten(
        self: Tensor,
        dims: Sequence[Union[str, ellipsis, None]],
        out_dim: Union[str, ellipsis, None],
    ) -> Tensor:
        raise NotImplementedError("flatten")

    @dispatch
    def flip(self: Tensor, dims: Size) -> Tensor:
        return pi.flip(self, dims)

    @dispatch
    def flip(self: Tensor, *dims: int) -> Tensor:
        return pi.flip(self, *dims)

    def fliplr(self: Tensor) -> Tensor:
        raise NotImplementedError("fliplr")

    def flipud(self: Tensor) -> Tensor:
        raise NotImplementedError("flipud")

    def float(self: Tensor) -> Tensor:
        raise NotImplementedError("float")

    @dispatch
    def float_power(self: Tensor, exponent: Tensor) -> Tensor:
        raise NotImplementedError("float_power")

    @dispatch
    def float_power(self: Tensor, exponent: Number) -> Tensor:
        raise NotImplementedError("float_power")

    @dispatch
    def float_power_(self: Tensor, exponent: Tensor) -> Tensor:
        raise NotImplementedError("float_power_")

    @dispatch
    def float_power_(self: Tensor, exponent: Number) -> Tensor:
        raise NotImplementedError("float_power_")

    def floor(self: Tensor) -> Tensor:
        return pi.floor(self)

    def floor_(self: Tensor) -> Tensor:
        return pi.floor_(self)

    def floor_divide(
        self: Tensor, other: Union[Tensor, Number], *, out: Optional[Tensor] = None
    ) -> Tensor:
        if out is not None:
            raise NotImplementedError("floor_divide.out variant")
        return pi.floor_divide(self, other)

    def floor_divide_(self: Tensor, other: Union[Tensor, Number]) -> Tensor:
        raise NotImplementedError("floor_divide_")

    def fmax(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("fmax")

    def fmin(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("fmin")

    @dispatch
    def fmod(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("fmod")

    @dispatch
    def fmod(self: Tensor, other: Number) -> Tensor:
        return pi.fmod(self, other)

    @dispatch
    def fmod_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("fmod_")

    @dispatch
    def fmod_(self: Tensor, other: Number) -> Tensor:
        return pi.fmod_(self, other)

    def frac(self: Tensor) -> Tensor:
        raise NotImplementedError("frac")

    def frac_(self: Tensor) -> Tensor:
        raise NotImplementedError("frac_")

    def frexp(self: Tensor) -> ComplexReturnType("frexp"):
        raise NotImplementedError("frexp")

    @dispatch
    def gather(
        self: Tensor, dim: int, index: Tensor, *, sparse_grad: bool = False
    ) -> Tensor:
        return pi.gather(self, dim, index, sparse_grad)

    @dispatch
    def gather(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        index: Tensor,
        *,
        sparse_grad: bool = False,
    ) -> Tensor:
        raise NotImplementedError("gather")

    def gcd(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("gcd")

    def gcd_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("gcd_")

    @dispatch
    def ge(self: Tensor, other: Tensor) -> Tensor:
        return pi.ge(self, other)

    @dispatch
    def ge(self: Tensor, other: Number) -> Tensor:
        return pi.ge(self, other)

    @dispatch
    def ge_(self: Tensor, other: Tensor) -> Tensor:
        return pi.ge_(self, other)

    @dispatch
    def ge_(self: Tensor, other: Number) -> Tensor:
        return pi.ge_(self, other)

    def geometric_(
        self: Tensor, p: float, *, generator: Optional[Generator] = None
    ) -> Tensor:
        raise NotImplementedError("geometric_")

    def geqrf(self: Tensor) -> ComplexReturnType("geqrf"):
        raise NotImplementedError("geqrf")

    def ger(self: Tensor, vec2: Tensor) -> Tensor:
        raise NotImplementedError("ger")

    def get_device(self: Tensor) -> int:
        raise NotImplementedError("get_device")

    @dispatch
    def greater(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("greater")

    @dispatch
    def greater(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("greater")

    @dispatch
    def greater_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("greater_")

    @dispatch
    def greater_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("greater_")

    @dispatch
    def greater_equal(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("greater_equal")

    @dispatch
    def greater_equal(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("greater_equal")

    @dispatch
    def greater_equal_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("greater_equal_")

    @dispatch
    def greater_equal_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("greater_equal_")

    @dispatch
    def gt(self: Tensor, other: Tensor) -> Tensor:
        return pi.gt(self, other)

    @dispatch
    def gt(self: Tensor, other: Number) -> Tensor:
        return pi.gt(self, other)

    @dispatch
    def gt_(self: Tensor, other: Tensor) -> Tensor:
        return pi.gt_(self, other)

    @dispatch
    def gt_(self: Tensor, other: Number) -> Tensor:
        return pi.gt_(self, other)

    def half(self: Tensor) -> Tensor:
        raise NotImplementedError("half")

    def hardshrink(self: Tensor, lambd: Number = 0.5) -> Tensor:
        raise NotImplementedError("hardshrink")

    def has_names(self: Tensor) -> bool:
        raise NotImplementedError("has_names")

    def heaviside(self: Tensor, values: Tensor) -> Tensor:
        raise NotImplementedError("heaviside")

    def heaviside_(self: Tensor, values: Tensor) -> Tensor:
        raise NotImplementedError("heaviside_")

    def histc(
        self: Tensor, bins: int = 100, min: Number = 0, max: Number = 0
    ) -> Tensor:
        raise NotImplementedError("histc")

    @dispatch
    def histogram(
        self: Tensor,
        bins: Tensor,
        *,
        weight: Optional[Tensor] = None,
        density: bool = False,
    ) -> ComplexReturnType("histogram"):
        raise NotImplementedError("histogram")

    @dispatch
    def histogram(
        self: Tensor,
        bins: int = 100,
        *,
        range: Optional[Sequence[float]] = None,
        weight: Optional[Tensor] = None,
        density: bool = False,
    ) -> ComplexReturnType("histogram"):
        raise NotImplementedError("histogram")

    @dispatch
    def hsplit(self: Tensor, sections: int) -> List[Tensor]:
        raise NotImplementedError("hsplit")

    @dispatch
    def hsplit(self: Tensor, indices: Size) -> List[Tensor]:
        raise NotImplementedError("hsplit")

    @dispatch
    def hsplit(self: Tensor, *indices: int) -> List[Tensor]:
        raise NotImplementedError("hsplit")

    def hypot(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("hypot")

    def hypot_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("hypot_")

    def i0(self: Tensor) -> Tensor:
        raise NotImplementedError("i0")

    def i0_(self: Tensor) -> Tensor:
        raise NotImplementedError("i0_")

    def igamma(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("igamma")

    def igamma_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("igamma_")

    def igammac(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("igammac")

    def igammac_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("igammac_")

    @dispatch
    def index_add(
        self: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Number = 1
    ) -> Tensor:
        raise NotImplementedError("index_add")

    @dispatch
    def index_add(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        index: Tensor,
        source: Tensor,
        *,
        alpha: Number = 1,
    ) -> Tensor:
        raise NotImplementedError("index_add")

    def index_add_(
        self: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Number = 1
    ) -> Tensor:
        raise NotImplementedError("index_add_")

    @dispatch
    def index_copy(self: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
        raise NotImplementedError("index_copy")

    @dispatch
    def index_copy(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, source: Tensor
    ) -> Tensor:
        raise NotImplementedError("index_copy")

    @dispatch
    def index_copy_(self: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
        raise NotImplementedError("index_copy_")

    @dispatch
    def index_copy_(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, source: Tensor
    ) -> Tensor:
        raise NotImplementedError("index_copy_")

    @dispatch
    def index_fill(self: Tensor, dim: int, index: Tensor, value: Tensor) -> Tensor:
        raise NotImplementedError("index_fill")

    @dispatch
    def index_fill(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, value: Tensor
    ) -> Tensor:
        raise NotImplementedError("index_fill")

    @dispatch
    def index_fill(self: Tensor, dim: int, index: Tensor, value: Number) -> Tensor:
        raise NotImplementedError("index_fill")

    @dispatch
    def index_fill(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, value: Number
    ) -> Tensor:
        raise NotImplementedError("index_fill")

    @dispatch
    def index_fill_(self: Tensor, dim: int, index: Tensor, value: Tensor) -> Tensor:
        raise NotImplementedError("index_fill_")

    @dispatch
    def index_fill_(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, value: Tensor
    ) -> Tensor:
        raise NotImplementedError("index_fill_")

    @dispatch
    def index_fill_(self: Tensor, dim: int, index: Tensor, value: Number) -> Tensor:
        raise NotImplementedError("index_fill_")

    @dispatch
    def index_fill_(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, value: Number
    ) -> Tensor:
        raise NotImplementedError("index_fill_")

    def index_put(
        self: Tensor,
        indices: Optional[Union[Tuple[Tensor, ...], List[Tensor]]],
        values: Tensor,
        accumulate: bool = False,
    ) -> Tensor:
        return pi.index_put(self, indices, values, accumulate)

    def index_put_(
        self: Tensor,
        indices: Optional[Union[Tuple[Tensor, ...], List[Tensor]]],
        values: Tensor,
        accumulate: bool = False,
    ) -> Tensor:
        return pi.index_put_(self, indices, values, accumulate)

    def index_reduce(
        self: Tensor,
        dim: int,
        index: Tensor,
        source: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
    ) -> Tensor:
        raise NotImplementedError("index_reduce")

    def index_reduce_(
        self: Tensor,
        dim: int,
        index: Tensor,
        source: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
    ) -> Tensor:
        raise NotImplementedError("index_reduce_")

    @dispatch
    def index_select(self: Tensor, dim: int, index: Tensor) -> Tensor:
        return pi.index_select(self, dim, index)

    @dispatch
    def index_select(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor
    ) -> Tensor:
        raise NotImplementedError("index_select")

    def indices(self: Tensor) -> Tensor:
        raise NotImplementedError("indices")

    def inner(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("inner")

    def int(self: Tensor) -> Tensor:
        raise NotImplementedError("int")

    def int_repr(self: Tensor) -> Tensor:
        raise NotImplementedError("int_repr")

    def inverse(self: Tensor) -> Tensor:
        raise NotImplementedError("inverse")

    def is_coalesced(self: Tensor) -> bool:
        raise NotImplementedError("is_coalesced")

    def is_complex(self: Tensor) -> bool:
        raise NotImplementedError("is_complex")

    def is_conj(self: Tensor) -> bool:
        raise NotImplementedError("is_conj")

    def is_contiguous(self: Tensor, memory_format=contiguous_format) -> bool:
        raise NotImplementedError("is_contiguous")

    is_cuda: bool

    def is_distributed(self: Tensor) -> bool:
        raise NotImplementedError("is_distributed")

    def is_floating_point(self: Tensor) -> bool:
        return pi.is_floating_point(self)

    def is_inference(self: Tensor) -> bool:
        raise NotImplementedError("is_inference")

    is_ipu: bool
    is_leaf: bool
    is_meta: bool
    is_mkldnn: bool
    is_mps: bool

    def is_neg(self: Tensor) -> bool:
        raise NotImplementedError("is_neg")

    is_nested: bool

    def is_nonzero(self: Tensor) -> bool:
        raise NotImplementedError("is_nonzero")

    is_ort: bool

    def is_pinned(
        self: Tensor, device: Optional[Union[device, str, None]] = None
    ) -> bool:
        raise NotImplementedError("is_pinned")

    is_quantized: bool

    def is_same_size(self: Tensor, other: Tensor) -> bool:
        raise NotImplementedError("is_same_size")

    def is_set_to(self: Tensor, tensor: Tensor) -> bool:
        raise NotImplementedError("is_set_to")

    def is_signed(self: Tensor) -> bool:
        raise NotImplementedError("is_signed")

    is_sparse: bool
    is_sparse_csr: bool
    is_vulkan: bool

    def isclose(
        self: Tensor,
        other: Tensor,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> Tensor:
        raise NotImplementedError("isclose")

    def isfinite(self: Tensor) -> Tensor:
        raise NotImplementedError("isfinite")

    def isinf(self: Tensor) -> Tensor:
        raise NotImplementedError("isinf")

    def isnan(self: Tensor) -> Tensor:
        raise NotImplementedError("isnan")

    def isneginf(self: Tensor) -> Tensor:
        raise NotImplementedError("isneginf")

    def isposinf(self: Tensor) -> Tensor:
        raise NotImplementedError("isposinf")

    def isreal(self: Tensor) -> Tensor:
        raise NotImplementedError("isreal")

    def istft(
        self: Tensor,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[Tensor] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: Optional[bool] = None,
        length: Optional[int] = None,
        return_complex: bool = False,
    ) -> Tensor:
        raise NotImplementedError("istft")

    def item(self: Tensor) -> Number:
        return pi.item(self)

    def kron(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("kron")

    @dispatch
    def kthvalue(
        self: Tensor, k: int, dim: int = -1, keepdim: bool = False
    ) -> ComplexReturnType("kthvalue"):
        raise NotImplementedError("kthvalue")

    @dispatch
    def kthvalue(
        self: Tensor, k: int, dim: Union[str, ellipsis, None], keepdim: bool = False
    ) -> ComplexReturnType("kthvalue"):
        raise NotImplementedError("kthvalue")

    def lcm(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("lcm")

    def lcm_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("lcm_")

    def ldexp(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("ldexp")

    def ldexp_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("ldexp_")

    @dispatch
    def le(self: Tensor, other: Tensor) -> Tensor:
        return pi.le(self, other)

    @dispatch
    def le(self: Tensor, other: Number) -> Tensor:
        return pi.le(self, other)

    @dispatch
    def le_(self: Tensor, other: Tensor) -> Tensor:
        return pi.le_(self, other)

    @dispatch
    def le_(self: Tensor, other: Number) -> Tensor:
        return pi.le_(self, other)

    @dispatch
    def lerp(self: Tensor, end: Tensor, weight: Tensor) -> Tensor:
        return pi.lerp(self, end, weight)

    @dispatch
    def lerp(self: Tensor, end: Tensor, weight: Number) -> Tensor:
        raise NotImplementedError("lerp")

    @dispatch
    def lerp_(self: Tensor, end: Tensor, weight: Tensor) -> Tensor:
        return pi.lerp_(self, end, weight)

    @dispatch
    def lerp_(self: Tensor, end: Tensor, weight: Number) -> Tensor:
        raise NotImplementedError("lerp_")

    @dispatch
    def less(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("less")

    @dispatch
    def less(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("less")

    @dispatch
    def less_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("less_")

    @dispatch
    def less_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("less_")

    @dispatch
    def less_equal(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("less_equal")

    @dispatch
    def less_equal(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("less_equal")

    @dispatch
    def less_equal_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("less_equal_")

    @dispatch
    def less_equal_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("less_equal_")

    def lgamma(self: Tensor) -> Tensor:
        raise NotImplementedError("lgamma")

    def lgamma_(self: Tensor) -> Tensor:
        raise NotImplementedError("lgamma_")

    def log(self: Tensor) -> Tensor:
        return pi.log(self)

    def log10(self: Tensor) -> Tensor:
        raise NotImplementedError("log10")

    def log10_(self: Tensor) -> Tensor:
        raise NotImplementedError("log10_")

    def log1p(self: Tensor) -> Tensor:
        return pi.log1p(self)

    def log1p_(self: Tensor) -> Tensor:
        return pi.log1p_(self)

    def log2(self: Tensor) -> Tensor:
        return pi.log2(self)

    def log2_(self: Tensor) -> Tensor:
        return pi.log2_(self)

    def log_(self: Tensor) -> Tensor:
        return pi.log_(self)

    def log_normal_(
        self: Tensor,
        mean: float = 1,
        std: float = 2,
        *,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        raise NotImplementedError("log_normal_")

    @dispatch
    def log_softmax(self: Tensor, dim: int, dtype: Optional[pi_dtype] = None) -> Tensor:
        return pi.log_softmax(self, dim, dtype)

    @dispatch
    def log_softmax(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("log_softmax")

    def logaddexp(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("logaddexp")

    def logaddexp2(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("logaddexp2")

    @dispatch
    def logcumsumexp(self: Tensor, dim: int) -> Tensor:
        raise NotImplementedError("logcumsumexp")

    @dispatch
    def logcumsumexp(self: Tensor, dim: Union[str, ellipsis, None]) -> Tensor:
        raise NotImplementedError("logcumsumexp")

    def logdet(self: Tensor) -> Tensor:
        raise NotImplementedError("logdet")

    def logical_and(self: Tensor, other: Tensor) -> Tensor:
        return pi.logical_and(self, other)

    def logical_and_(self: Tensor, other: Tensor) -> Tensor:
        return pi.logical_and_(self, other)

    def logical_not(self: Tensor) -> Tensor:
        return pi.logical_not(self)

    def logical_not_(self: Tensor) -> Tensor:
        return pi.logical_not_(self)

    def logical_or(self: Tensor, other: Tensor) -> Tensor:
        return pi.logical_or(self, other)

    def logical_or_(self: Tensor, other: Tensor) -> Tensor:
        return pi.logical_or_(self, other)

    def logical_xor(self: Tensor, other: Tensor) -> Tensor:
        return pi.logical_xor(self, other)

    def logical_xor_(self: Tensor, other: Tensor) -> Tensor:
        return pi.logical_xor_(self, other)

    def logit(self: Tensor, eps: Optional[float] = None) -> Tensor:
        raise NotImplementedError("logit")

    def logit_(self: Tensor, eps: Optional[float] = None) -> Tensor:
        raise NotImplementedError("logit_")

    @dispatch
    def logsumexp(self: Tensor, dim: Union[int, Size], keepdim: bool = False) -> Tensor:
        return pi.logsumexp(self, dim, keepdim)

    @dispatch
    def logsumexp(
        self: Tensor, dim: Sequence[Union[str, ellipsis, None]], keepdim: bool = False
    ) -> Tensor:
        raise NotImplementedError("logsumexp")

    def long(self: Tensor) -> Tensor:
        raise NotImplementedError("long")

    @dispatch
    def lt(self: Tensor, other: Tensor) -> Tensor:
        return pi.lt(self, other)

    @dispatch
    def lt(self: Tensor, other: Number) -> Tensor:
        return pi.lt(self, other)

    @dispatch
    def lt_(self: Tensor, other: Tensor) -> Tensor:
        return pi.lt_(self, other)

    @dispatch
    def lt_(self: Tensor, other: Number) -> Tensor:
        return pi.lt_(self, other)

    def lu_solve(self: Tensor, LU_data: Tensor, LU_pivots: Tensor) -> Tensor:
        raise NotImplementedError("lu_solve")

    def map2_(self: Tensor, x: Tensor, y: Tensor, callable: Callable) -> Tensor:
        raise NotImplementedError("map2_")

    def map_(self: Tensor, tensor: Tensor, callable: Callable) -> Tensor:
        raise NotImplementedError("map_")

    @dispatch
    def masked_fill(self: Tensor, mask: Tensor, value: Tensor) -> Tensor:
        return pi.masked_fill(self, mask, value)

    @dispatch
    def masked_fill(self: Tensor, mask: Tensor, value: Number) -> Tensor:
        return pi.masked_fill(self, mask, value)

    @dispatch
    def masked_fill_(self: Tensor, mask: Tensor, value: Tensor) -> Tensor:
        return pi.masked_fill_(self, mask, value)

    @dispatch
    def masked_fill_(self: Tensor, mask: Tensor, value: Number) -> Tensor:
        return pi.masked_fill_(self, mask, value)

    def masked_scatter(self: Tensor, mask: Tensor, source: Tensor) -> Tensor:
        raise NotImplementedError("masked_scatter")

    def masked_scatter_(self: Tensor, mask: Tensor, source: Tensor) -> Tensor:
        raise NotImplementedError("masked_scatter_")

    def masked_select(self: Tensor, mask: Tensor) -> Tensor:
        return pi.masked_select(self, mask)

    def matmul(self: Tensor, other: Tensor) -> Tensor:
        return pi.matmul(self, other)

    def matrix_exp(self: Tensor) -> Tensor:
        raise NotImplementedError("matrix_exp")

    def matrix_power(self: Tensor, n: int) -> Tensor:
        raise NotImplementedError("matrix_power")

    @dispatch
    def max(self: Tensor) -> Tensor:
        return pi.max(self)

    @dispatch
    def max(self: Tensor, other: Tensor) -> Tensor:
        return pi.max(self, other)

    @dispatch
    def max(self: Tensor, dim: int, keepdim: bool = False) -> ComplexReturnType("max"):
        return pi.max(self, dim, keepdim)

    @dispatch
    def max(
        self: Tensor, dim: Union[str, ellipsis, None], keepdim: bool = False
    ) -> ComplexReturnType("max"):
        raise NotImplementedError("max")

    def maximum(self: Tensor, other: Tensor) -> Tensor:
        return pi.maximum(self, other)

    @dispatch
    def mean(self: Tensor, *, dtype: Optional[pi_dtype] = None) -> Tensor:
        return pi.mean(self, dtype)

    @dispatch
    def mean(
        self: Tensor,
        dim: Optional[Union[int, Size]],
        keepdim: bool = False,
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        return pi.mean(self, dim, keepdim, dtype)

    @dispatch
    def mean(
        self: Tensor,
        dim: Sequence[Union[str, ellipsis, None]],
        keepdim: bool = False,
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("mean")

    @dispatch
    def median(self: Tensor) -> Tensor:
        raise NotImplementedError("median")

    @dispatch
    def median(
        self: Tensor, dim: int, keepdim: bool = False
    ) -> ComplexReturnType("median"):
        raise NotImplementedError("median")

    @dispatch
    def median(
        self: Tensor, dim: Union[str, ellipsis, None], keepdim: bool = False
    ) -> ComplexReturnType("median"):
        raise NotImplementedError("median")

    @dispatch
    def min(self: Tensor) -> Tensor:
        raise NotImplementedError("min")

    @dispatch
    def min(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("min")

    @dispatch
    def min(self: Tensor, dim: int, keepdim: bool = False) -> ComplexReturnType("min"):
        raise NotImplementedError("min")

    @dispatch
    def min(
        self: Tensor, dim: Union[str, ellipsis, None], keepdim: bool = False
    ) -> ComplexReturnType("min"):
        raise NotImplementedError("min")

    def minimum(self: Tensor, other: Tensor) -> Tensor:
        return pi.minimum(self, other)

    def mm(self: Tensor, mat2: Tensor) -> Tensor:
        return pi.mm(self, mat2)

    @dispatch
    def mode(
        self: Tensor, dim: int = -1, keepdim: bool = False
    ) -> ComplexReturnType("mode"):
        raise NotImplementedError("mode")

    @dispatch
    def mode(
        self: Tensor, dim: Union[str, ellipsis, None], keepdim: bool = False
    ) -> ComplexReturnType("mode"):
        raise NotImplementedError("mode")

    @dispatch
    def moveaxis(self: Tensor, source: int, destination: int) -> Tensor:
        raise NotImplementedError("moveaxis")

    @dispatch
    def moveaxis(self: Tensor, source: Size, destination: Size) -> Tensor:
        raise NotImplementedError("moveaxis")

    @dispatch
    def movedim(self: Tensor, source: int, destination: int) -> Tensor:
        raise NotImplementedError("movedim")

    @dispatch
    def movedim(self: Tensor, source: Size, destination: Size) -> Tensor:
        raise NotImplementedError("movedim")

    def msort(self: Tensor) -> Tensor:
        raise NotImplementedError("msort")

    def mul(
        self: Tensor, other: Union[Tensor, Number], *, out: Optional[Tensor] = None
    ) -> Tensor:
        if out is not None:
            raise NotImplementedError("mul.out variant")
        return pi.mul(self, other)

    def mul_(self: Tensor, other: Union[Tensor, Number]) -> Tensor:
        return pi.mul_(self, other)

    def multinomial(
        self: Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        raise NotImplementedError("multinomial")

    @dispatch
    def multiply(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("multiply")

    @dispatch
    def multiply(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("multiply")

    @dispatch
    def multiply_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("multiply_")

    @dispatch
    def multiply_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("multiply_")

    def mv(self: Tensor, vec: Tensor) -> Tensor:
        return pi.mv(self, vec)

    def mvlgamma(self: Tensor, p: int) -> Tensor:
        raise NotImplementedError("mvlgamma")

    def mvlgamma_(self: Tensor, p: int) -> Tensor:
        raise NotImplementedError("mvlgamma_")

    def nan_to_num(
        self: Tensor,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
    ) -> Tensor:
        raise NotImplementedError("nan_to_num")

    def nan_to_num_(
        self: Tensor,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
    ) -> Tensor:
        raise NotImplementedError("nan_to_num_")

    def nanmean(
        self: Tensor,
        dim: Optional[Union[int, Size]] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("nanmean")

    @dispatch
    def nanmedian(self: Tensor) -> Tensor:
        raise NotImplementedError("nanmedian")

    @dispatch
    def nanmedian(
        self: Tensor, dim: int, keepdim: bool = False
    ) -> ComplexReturnType("nanmedian"):
        raise NotImplementedError("nanmedian")

    @dispatch
    def nanmedian(
        self: Tensor, dim: Union[str, ellipsis, None], keepdim: bool = False
    ) -> ComplexReturnType("nanmedian"):
        raise NotImplementedError("nanmedian")

    @dispatch
    def nanquantile(
        self: Tensor,
        q: Tensor,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        interpolation: str = "linear",
    ) -> Tensor:
        raise NotImplementedError("nanquantile")

    @dispatch
    def nanquantile(
        self: Tensor,
        q: float,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        interpolation: str = "linear",
    ) -> Tensor:
        raise NotImplementedError("nanquantile")

    def nansum(
        self: Tensor,
        dim: Optional[Union[int, Size]] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("nansum")

    @dispatch
    def narrow(self: Tensor, dim: int, start: Tensor, length: int) -> Tensor:
        raise NotImplementedError("narrow")

    @dispatch
    def narrow(self: Tensor, dim: int, start: int, length: int) -> Tensor:
        return pi.narrow(self, dim, start, length)

    def narrow_copy(self: Tensor, dim: int, start: int, length: int) -> Tensor:
        raise NotImplementedError("narrow_copy")

    def ndimension(self: Tensor) -> int:
        raise NotImplementedError("ndimension")

    @dispatch
    def ne(self: Tensor, other: Tensor) -> Tensor:
        return pi.ne(self, other)

    @dispatch
    def ne(self: Tensor, other: Number) -> Tensor:
        return pi.ne(self, other)

    @dispatch
    def ne_(self: Tensor, other: Tensor) -> Tensor:
        return pi.ne_(self, other)

    @dispatch
    def ne_(self: Tensor, other: Number) -> Tensor:
        return pi.ne_(self, other)

    def neg(self: Tensor) -> Tensor:
        return pi.neg(self)

    def neg_(self: Tensor) -> Tensor:
        return pi.neg_(self)

    def negative(self: Tensor) -> Tensor:
        raise NotImplementedError("negative")

    def negative_(self: Tensor) -> Tensor:
        raise NotImplementedError("negative_")

    def nelement(self: Tensor) -> int:
        raise NotImplementedError("nelement")

    @dispatch
    def new(self: Tensor, *args: Any, device: Device = None) -> Tensor:
        raise NotImplementedError("new")

    @dispatch
    def new(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("new")

    @dispatch
    def new(self: Tensor, size: Size, *, device: Device = None) -> Tensor:
        raise NotImplementedError("new")

    @dispatch
    def new_empty(
        self: Tensor,
        size: List[int],
        *,
        dtype: Optional[pi_dtype] = None,
        layout: Optional[layout] = None,
        device: Optional[Union[device, str, None]] = None,
        pin_memory: Optional[bool] = False,
        requires_grad: Optional[bool] = False,
    ) -> Tensor:
        return pi.new_empty(
            self, size, dtype, layout, device, pin_memory, requires_grad
        )

    @dispatch
    def new_empty(
        self: Tensor,
        *size: int,
        dtype: Optional[pi_dtype] = None,
        layout: Optional[layout] = None,
        device: Optional[Union[device, str, None]] = None,
        pin_memory: Optional[bool] = False,
        requires_grad: Optional[bool] = False,
    ) -> Tensor:
        raise NotImplementedError("new_empty")

    def new_empty_strided(
        self: Tensor,
        size: List[int],
        stride: List[int],
        *,
        dtype: Optional[pi_dtype] = None,
        layout: Optional[layout] = None,
        device: Optional[Union[device, str, None]] = None,
        pin_memory: Optional[bool] = False,
        requires_grad: Optional[bool] = False,
    ) -> Tensor:
        raise NotImplementedError("new_empty_strided")

    def new_full(
        self: Tensor,
        size: List[int],
        fill_value: Number,
        *,
        dtype: Optional[pi_dtype] = None,
        layout: Optional[layout] = None,
        device: Optional[Union[device, str, None]] = None,
        pin_memory: Optional[bool] = False,
        requires_grad: Optional[bool] = False,
    ) -> Tensor:
        raise NotImplementedError("new_full")

    @dispatch
    def new_ones(
        self: Tensor,
        size: Size,
        dtype: Optional[pi_dtype] = None,
        device: Device = None,
        requires_grad: bool = False,
    ) -> Tensor:
        return pi.new_ones(self, size, dtype, device, requires_grad)

    @dispatch
    def new_ones(
        self: Tensor,
        size: List[int],
        *,
        dtype: Optional[pi_dtype] = None,
        layout: Optional[layout] = None,
        device: Optional[Union[device, str, None]] = None,
        pin_memory: Optional[bool] = False,
        requires_grad: Optional[bool] = False,
    ) -> Tensor:
        return pi.new_ones(self, size, dtype, layout, device, pin_memory, requires_grad)

    @dispatch
    def new_ones(
        self: Tensor,
        *size: int,
        dtype: Optional[pi_dtype] = None,
        layout: Optional[layout] = None,
        device: Optional[Union[device, str, None]] = None,
        pin_memory: Optional[bool] = False,
        requires_grad: Optional[bool] = False,
    ) -> Tensor:
        raise NotImplementedError("new_ones")

    def new_tensor(
        self: Tensor,
        data: Any,
        dtype: Optional[pi_dtype] = None,
        device: Device = None,
        requires_grad: bool = False,
    ) -> Tensor:
        raise NotImplementedError("new_tensor")

    @dispatch
    def new_zeros(
        self: Tensor,
        size: List[int],
        *,
        dtype: Optional[pi_dtype] = None,
        layout: Optional[layout] = None,
        device: Optional[Union[device, str, None]] = None,
        pin_memory: Optional[bool] = False,
        requires_grad: Optional[bool] = False,
    ) -> Tensor:
        return pi.new_zeros(
            self, size, dtype, layout, device, pin_memory, requires_grad
        )

    @dispatch
    def new_zeros(
        self: Tensor,
        *size: int,
        dtype: Optional[pi_dtype] = None,
        layout: Optional[layout] = None,
        device: Optional[Union[device, str, None]] = None,
        pin_memory: Optional[bool] = False,
        requires_grad: Optional[bool] = False,
    ) -> Tensor:
        raise NotImplementedError("new_zeros")

    def nextafter(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("nextafter")

    def nextafter_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("nextafter_")

    @dispatch
    def nonzero(self: Tensor, *, as_tuple: Literal[False] = False) -> Tensor:
        raise NotImplementedError("nonzero")

    @dispatch
    def nonzero(self: Tensor, *, as_tuple: Literal[True]) -> Tuple[Tensor, ...]:
        raise NotImplementedError("nonzero")

    def normal_(
        self: Tensor,
        mean: float = 0,
        std: float = 1,
        *,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        raise NotImplementedError("normal_")

    @dispatch
    def not_equal(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("not_equal")

    @dispatch
    def not_equal(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("not_equal")

    @dispatch
    def not_equal_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("not_equal_")

    @dispatch
    def not_equal_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("not_equal_")

    def numel(self: Tensor) -> int:
        return pi.numel(self)

    def numpy(self: Tensor, *, force: bool = False) -> Any:
        raise NotImplementedError("numpy")

    def orgqr(self: Tensor, input2: Tensor) -> Tensor:
        raise NotImplementedError("orgqr")

    def ormqr(
        self: Tensor,
        input2: Tensor,
        input3: Tensor,
        left: bool = True,
        transpose: bool = False,
    ) -> Tensor:
        raise NotImplementedError("ormqr")

    def outer(self: Tensor, vec2: Tensor) -> Tensor:
        raise NotImplementedError("outer")

    @dispatch
    def permute(self: Tensor, dims: Size) -> Tensor:
        return pi.permute(self, dims)

    @dispatch
    def permute(self: Tensor, *dims: int) -> Tensor:
        return pi.permute(self, dims)

    def pin_memory(
        self: Tensor, device: Optional[Union[device, str, None]] = None
    ) -> Tensor:
        raise NotImplementedError("pin_memory")

    def pinverse(self: Tensor, rcond: float = 1e-15) -> Tensor:
        raise NotImplementedError("pinverse")

    def polygamma(self: Tensor, n: int) -> Tensor:
        raise NotImplementedError("polygamma")

    def polygamma_(self: Tensor, n: int) -> Tensor:
        raise NotImplementedError("polygamma_")

    def positive(self: Tensor) -> Tensor:
        raise NotImplementedError("positive")

    @dispatch
    def pow(self: Tensor, exponent: Tensor) -> Tensor:
        return pi.pow(self, exponent)

    @dispatch
    def pow(self: Tensor, exponent: Number) -> Tensor:
        return pi.pow(self, exponent)

    @dispatch
    def pow_(self: Tensor, exponent: Tensor) -> Tensor:
        raise NotImplementedError("pow_")

    @dispatch
    def pow_(self: Tensor, exponent: Number) -> Tensor:
        raise NotImplementedError("pow_")

    def prelu(self: Tensor, weight: Tensor) -> Tensor:
        return pi.prelu(self, weight)

    @dispatch
    def prod(self: Tensor, *, dtype: Optional[pi_dtype] = None) -> Tensor:
        raise NotImplementedError("prod")

    @dispatch
    def prod(
        self: Tensor,
        dim: int,
        keepdim: bool = False,
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("prod")

    @dispatch
    def prod(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        keepdim: bool = False,
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("prod")

    def put(
        self: Tensor, index: Tensor, source: Tensor, accumulate: bool = False
    ) -> Tensor:
        raise NotImplementedError("put")

    def put_(
        self: Tensor, index: Tensor, source: Tensor, accumulate: bool = False
    ) -> Tensor:
        raise NotImplementedError("put_")

    def q_per_channel_axis(self: Tensor) -> int:
        raise NotImplementedError("q_per_channel_axis")

    def q_per_channel_scales(self: Tensor) -> Tensor:
        raise NotImplementedError("q_per_channel_scales")

    def q_per_channel_zero_points(self: Tensor) -> Tensor:
        raise NotImplementedError("q_per_channel_zero_points")

    def q_scale(self: Tensor) -> float:
        raise NotImplementedError("q_scale")

    def q_zero_point(self: Tensor) -> int:
        raise NotImplementedError("q_zero_point")

    def qr(self: Tensor, some: bool = True) -> ComplexReturnType("qr"):
        raise NotImplementedError("qr")

    @dispatch
    def quantile(
        self: Tensor,
        q: Tensor,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        interpolation: str = "linear",
    ) -> Tensor:
        raise NotImplementedError("quantile")

    @dispatch
    def quantile(
        self: Tensor,
        q: float,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        interpolation: str = "linear",
    ) -> Tensor:
        raise NotImplementedError("quantile")

    def rad2deg(self: Tensor) -> Tensor:
        raise NotImplementedError("rad2deg")

    def rad2deg_(self: Tensor) -> Tensor:
        raise NotImplementedError("rad2deg_")

    @dispatch
    def random_(self: Tensor, *, generator: Optional[Generator] = None) -> Tensor:
        raise NotImplementedError("random_")

    @dispatch
    def random_(
        self: Tensor,
        from_: int,
        to: Optional[int],
        *,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        raise NotImplementedError("random_")

    @dispatch
    def random_(
        self: Tensor, to: int, *, generator: Optional[Generator] = None
    ) -> Tensor:
        raise NotImplementedError("random_")

    def ravel(self: Tensor) -> Tensor:
        raise NotImplementedError("ravel")

    def reciprocal(self: Tensor) -> Tensor:
        return pi.reciprocal(self)

    def reciprocal_(self: Tensor) -> Tensor:
        return pi.reciprocal_(self)

    def refine_names(
        self: Tensor, names: Sequence[Union[str, ellipsis, None]]
    ) -> Tensor:
        raise NotImplementedError("refine_names")

    def relu(self: Tensor) -> Tensor:
        return pi.relu(self)

    def relu_(self: Tensor) -> Tensor:
        return pi.relu_(self)

    @dispatch
    def remainder(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("remainder")

    @dispatch
    def remainder(self: Tensor, other: Number) -> Tensor:
        return pi.remainder(self, other)

    @dispatch
    def remainder_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("remainder_")

    @dispatch
    def remainder_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("remainder_")

    def rename(
        self: Tensor, names: Optional[Sequence[Union[str, ellipsis, None]]]
    ) -> Tensor:
        raise NotImplementedError("rename")

    def rename_(
        self: Tensor, names: Optional[Sequence[Union[str, ellipsis, None]]]
    ) -> Tensor:
        raise NotImplementedError("rename_")

    def renorm(self: Tensor, p: Number, dim: int, maxnorm: Number) -> Tensor:
        raise NotImplementedError("renorm")

    def renorm_(self: Tensor, p: Number, dim: int, maxnorm: Number) -> Tensor:
        raise NotImplementedError("renorm_")

    @dispatch
    def repeat(self: Tensor, repeats: List[int]) -> Tensor:
        return pi.repeat(self, repeats)

    @dispatch
    def repeat(self: Tensor, *repeats: int) -> Tensor:
        return pi.repeat(self, *repeats)

    @dispatch
    def repeat_interleave(
        self: Tensor,
        repeats: Tensor,
        dim: Optional[int] = None,
        *,
        output_size: Optional[int] = None,
    ) -> Tensor:
        raise NotImplementedError("repeat_interleave")

    @dispatch
    def repeat_interleave(
        self: Tensor,
        repeats: int,
        dim: Optional[int] = None,
        *,
        output_size: Optional[int] = None,
    ) -> Tensor:
        raise NotImplementedError("repeat_interleave")

    def requires_grad_(self: Tensor, mode: bool = True) -> Tensor:
        raise NotImplementedError("requires_grad_")

    @dispatch
    def reshape(self: Tensor, shape: List[int]) -> Tensor:
        return pi.reshape(self, shape)

    @dispatch
    def reshape(self: Tensor, *shape: int) -> Tensor:
        return pi.reshape(self, shape)

    def reshape_as(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("reshape_as")

    @dispatch
    def resize_(
        self: Tensor, size: List[int], *, memory_format: Optional[memory_format] = None
    ) -> Tensor:
        return pi.resize_(self, size, memory_format)

    @dispatch
    def resize_(
        self: Tensor, *size: int, memory_format: Optional[memory_format] = None
    ) -> Tensor:
        raise NotImplementedError("resize_")

    def resize_as_(
        self: Tensor,
        the_template: Tensor,
        *,
        memory_format: Optional[memory_format] = None,
    ) -> Tensor:
        raise NotImplementedError("resize_as_")

    def resize_as_sparse_(self: Tensor, the_template: Tensor) -> Tensor:
        raise NotImplementedError("resize_as_sparse_")

    def resolve_conj(self: Tensor) -> Tensor:
        raise NotImplementedError("resolve_conj")

    def resolve_neg(self: Tensor) -> Tensor:
        raise NotImplementedError("resolve_neg")

    def retain_grad(self: Tensor) -> None:
        raise NotImplementedError("retain_grad")

    def roll(
        self: Tensor, shifts: Union[int, Size], dims: Union[int, Size] = ()
    ) -> Tensor:
        return pi.roll(self, shifts, dims)

    def rot90(self: Tensor, k: int = 1, dims: Size = (0, 1)) -> Tensor:
        raise NotImplementedError("rot90")

    @dispatch
    def round(self: Tensor) -> Tensor:
        return pi.round(self)

    @dispatch
    def round(self: Tensor, *, decimals: int) -> Tensor:
        raise NotImplementedError("round")

    @dispatch
    def round_(self: Tensor) -> Tensor:
        return pi.round_(self)

    @dispatch
    def round_(self: Tensor, *, decimals: int) -> Tensor:
        raise NotImplementedError("round_")

    def row_indices(self: Tensor) -> Tensor:
        raise NotImplementedError("row_indices")

    def rsqrt(self: Tensor) -> Tensor:
        return pi.rsqrt(self)

    def rsqrt_(self: Tensor) -> Tensor:
        return pi.rsqrt_(self)

    @dispatch
    def scatter(self: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
        raise NotImplementedError("scatter")

    @dispatch
    def scatter(
        self: Tensor, dim: int, index: Tensor, src: Tensor, *, reduce: str
    ) -> Tensor:
        raise NotImplementedError("scatter")

    @dispatch
    def scatter(
        self: Tensor, dim: int, index: Tensor, value: Number, *, reduce: str
    ) -> Tensor:
        raise NotImplementedError("scatter")

    @dispatch
    def scatter(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, src: Tensor
    ) -> Tensor:
        raise NotImplementedError("scatter")

    @dispatch
    def scatter(self: Tensor, dim: int, index: Tensor, value: Number) -> Tensor:
        raise NotImplementedError("scatter")

    @dispatch
    def scatter(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, value: Number
    ) -> Tensor:
        raise NotImplementedError("scatter")

    @dispatch
    def scatter_(self: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
        raise NotImplementedError("scatter_")

    @dispatch
    def scatter_(
        self: Tensor, dim: int, index: Tensor, src: Tensor, *, reduce: str
    ) -> Tensor:
        raise NotImplementedError("scatter_")

    @dispatch
    def scatter_(
        self: Tensor, dim: int, index: Tensor, value: Number, *, reduce: str
    ) -> Tensor:
        raise NotImplementedError("scatter_")

    @dispatch
    def scatter_(self: Tensor, dim: int, index: Tensor, value: Number) -> Tensor:
        raise NotImplementedError("scatter_")

    @dispatch
    def scatter_add(self: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
        return pi.scatter_add(self, dim, index, src)

    @dispatch
    def scatter_add(
        self: Tensor, dim: Union[str, ellipsis, None], index: Tensor, src: Tensor
    ) -> Tensor:
        raise NotImplementedError("scatter_add")

    def scatter_add_(self: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
        raise NotImplementedError("scatter_add_")

    def scatter_reduce(
        self: Tensor,
        dim: int,
        index: Tensor,
        src: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
    ) -> Tensor:
        raise NotImplementedError("scatter_reduce")

    def scatter_reduce_(
        self: Tensor,
        dim: int,
        index: Tensor,
        src: Tensor,
        reduce: str,
        *,
        include_self: bool = True,
    ) -> Tensor:
        raise NotImplementedError("scatter_reduce_")

    @dispatch
    def select(self: Tensor, dim: int, index: int) -> Tensor:
        return pi.select(self, dim, index)

    @dispatch
    def select(self: Tensor, dim: Union[str, ellipsis, None], index: int) -> Tensor:
        raise NotImplementedError("select")

    def select_scatter(self: Tensor, src: Tensor, dim: int, index: int) -> Tensor:
        return pi.select_scatter(self, src, dim, index)

    def sgn(self: Tensor) -> Tensor:
        raise NotImplementedError("sgn")

    def sgn_(self: Tensor) -> Tensor:
        raise NotImplementedError("sgn_")

    def short(self: Tensor) -> Tensor:
        raise NotImplementedError("short")

    def sigmoid(self: Tensor) -> Tensor:
        return pi.sigmoid(self)

    def sigmoid_(self: Tensor) -> Tensor:
        return pi.sigmoid_(self)

    def sign(self: Tensor) -> Tensor:
        raise NotImplementedError("sign")

    def sign_(self: Tensor) -> Tensor:
        raise NotImplementedError("sign_")

    def signbit(self: Tensor) -> Tensor:
        raise NotImplementedError("signbit")

    def sin(self: Tensor) -> Tensor:
        return pi.sin(self)

    def sin_(self: Tensor) -> Tensor:
        return pi.sin_(self)

    def sinc(self: Tensor) -> Tensor:
        raise NotImplementedError("sinc")

    def sinc_(self: Tensor) -> Tensor:
        raise NotImplementedError("sinc_")

    def sinh(self: Tensor) -> Tensor:
        raise NotImplementedError("sinh")

    def sinh_(self: Tensor) -> Tensor:
        raise NotImplementedError("sinh_")

    @dispatch
    def size(self: Tensor) -> Size:
        return pi.size(self)

    @dispatch
    def size(self: Tensor, dim: int) -> int:
        return pi.size(self, dim)

    def slice_scatter(
        self: Tensor,
        src: Tensor,
        dim: int = 0,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ) -> Tensor:
        return pi.slice_scatter(self, src, dim, start, end, step)

    def slogdet(self: Tensor) -> ComplexReturnType("slogdet"):
        raise NotImplementedError("slogdet")

    def smm(self: Tensor, mat2: Tensor) -> Tensor:
        raise NotImplementedError("smm")

    @dispatch
    def softmax(self: Tensor, dim: int, dtype: Optional[pi_dtype] = None) -> Tensor:
        return pi.softmax(self, dim, dtype)

    @dispatch
    def softmax(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("softmax")

    @dispatch
    def sort(
        self: Tensor, *, stable: Optional[bool], dim: int = -1, descending: bool = False
    ) -> ComplexReturnType("sort"):
        raise NotImplementedError("sort")

    @dispatch
    def sort(
        self: Tensor, dim: int = -1, descending: bool = False
    ) -> ComplexReturnType("sort"):
        raise NotImplementedError("sort")

    @dispatch
    def sort(
        self: Tensor,
        *,
        stable: Optional[bool],
        dim: Union[str, ellipsis, None],
        descending: bool = False,
    ) -> ComplexReturnType("sort"):
        raise NotImplementedError("sort")

    @dispatch
    def sort(
        self: Tensor, dim: Union[str, ellipsis, None], descending: bool = False
    ) -> ComplexReturnType("sort"):
        raise NotImplementedError("sort")

    def sparse_dim(self: Tensor) -> int:
        raise NotImplementedError("sparse_dim")

    def sparse_mask(self: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError("sparse_mask")

    def sparse_resize_(
        self: Tensor, size: Size, sparse_dim: int, dense_dim: int
    ) -> Tensor:
        raise NotImplementedError("sparse_resize_")

    def sparse_resize_and_clear_(
        self: Tensor, size: Size, sparse_dim: int, dense_dim: int
    ) -> Tensor:
        raise NotImplementedError("sparse_resize_and_clear_")

    @dispatch
    def split(self: Tensor, split_size: int, dim: int = 0) -> Sequence[Tensor]:
        raise NotImplementedError("split")

    @dispatch
    def split(
        self: Tensor, split_size: Tuple[int, ...], dim: int = 0
    ) -> Sequence[Tensor]:
        raise NotImplementedError("split")

    def split_with_sizes(
        self: Tensor, split_sizes: List[int], dim: int = 0
    ) -> List[Tensor]:
        raise NotImplementedError("split_with_sizes")

    def sqrt(self: Tensor) -> Tensor:
        return pi.sqrt(self)

    def sqrt_(self: Tensor) -> Tensor:
        return pi.sqrt_(self)

    def square(self: Tensor) -> Tensor:
        return pi.square(self)

    def square_(self: Tensor) -> Tensor:
        return pi.square_(self)

    @dispatch
    def squeeze(self: Tensor) -> Tensor:
        return pi.squeeze(self)

    @dispatch
    def squeeze(self: Tensor, dim: int) -> Tensor:
        return pi.squeeze(self, dim)

    @dispatch
    def squeeze(self: Tensor, dim: Union[str, ellipsis, None]) -> Tensor:
        raise NotImplementedError("squeeze")

    @dispatch
    def squeeze_(self: Tensor) -> Tensor:
        raise NotImplementedError("squeeze_")

    @dispatch
    def squeeze_(self: Tensor, dim: int) -> Tensor:
        raise NotImplementedError("squeeze_")

    @dispatch
    def squeeze_(self: Tensor, dim: Union[str, ellipsis, None]) -> Tensor:
        raise NotImplementedError("squeeze_")

    def sspaddmm(
        self: Tensor, mat1: Tensor, mat2: Tensor, *, beta: Number = 1, alpha: Number = 1
    ) -> Tensor:
        raise NotImplementedError("sspaddmm")

    @dispatch
    def std(
        self: Tensor,
        dim: Optional[Union[int, Size]],
        unbiased: bool = True,
        keepdim: bool = False,
    ) -> Tensor:
        return pi.std(self, dim, unbiased, keepdim)

    @dispatch
    def std(
        self: Tensor,
        dim: Optional[Union[int, Size]] = None,
        *,
        correction: Optional[int] = None,
        keepdim: bool = False,
    ) -> Tensor:
        return pi.std(self, dim, correction, keepdim)

    @dispatch
    def std(self: Tensor, unbiased: bool = True) -> Tensor:
        return pi.std(self, unbiased)

    @dispatch
    def std(
        self: Tensor,
        dim: Sequence[Union[str, ellipsis, None]],
        unbiased: bool = True,
        keepdim: bool = False,
    ) -> Tensor:
        raise NotImplementedError("std")

    @dispatch
    def std(
        self: Tensor,
        dim: Sequence[Union[str, ellipsis, None]],
        *,
        correction: Optional[int] = None,
        keepdim: bool = False,
    ) -> Tensor:
        raise NotImplementedError("std")

    def storage_offset(self: Tensor) -> int:
        raise NotImplementedError("storage_offset")

    @dispatch
    def stride(self: Tensor) -> Tuple[int, ...]:
        raise NotImplementedError("stride")

    @dispatch
    def stride(self: Tensor, _int) -> int:
        raise NotImplementedError("stride")

    def sub(
        self: Tensor,
        other: Union[Tensor, Number],
        *,
        alpha: Optional[Number] = 1,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        if out is not None:
            raise NotImplementedError("sub.out variant")
        return pi.sub(self, other, alpha)

    def sub_(
        self: Tensor, other: Union[Tensor, Number], *, alpha: Optional[Number] = 1
    ) -> Tensor:
        return pi.sub_(self, other, alpha)

    @dispatch
    def subtract(self: Tensor, other: Tensor, *, alpha: Number = 1) -> Tensor:
        raise NotImplementedError("subtract")

    @dispatch
    def subtract(self: Tensor, other: Number, alpha: Number = 1) -> Tensor:
        raise NotImplementedError("subtract")

    @dispatch
    def subtract_(self: Tensor, other: Tensor, *, alpha: Number = 1) -> Tensor:
        raise NotImplementedError("subtract_")

    @dispatch
    def subtract_(self: Tensor, other: Number, alpha: Number = 1) -> Tensor:
        raise NotImplementedError("subtract_")

    @dispatch
    def sum(self: Tensor, *, dtype: Optional[pi_dtype] = None) -> Tensor:
        return pi.sum(self, dtype)

    @dispatch
    def sum(
        self: Tensor,
        dim: Optional[Union[int, Size]],
        keepdim: bool = False,
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        return pi.sum(self, dim, keepdim, dtype)

    @dispatch
    def sum(
        self: Tensor,
        dim: Sequence[Union[str, ellipsis, None]],
        keepdim: bool = False,
        *,
        dtype: Optional[pi_dtype] = None,
    ) -> Tensor:
        raise NotImplementedError("sum")

    @dispatch
    def sum_to_size(self: Tensor, size: Size) -> Tensor:
        raise NotImplementedError("sum_to_size")

    @dispatch
    def sum_to_size(self: Tensor, *size: int) -> Tensor:
        raise NotImplementedError("sum_to_size")

    def svd(
        self: Tensor, some: bool = True, compute_uv: bool = True
    ) -> ComplexReturnType("svd"):
        raise NotImplementedError("svd")

    def swapaxes(self: Tensor, axis0: int, axis1: int) -> Tensor:
        raise NotImplementedError("swapaxes")

    def swapaxes_(self: Tensor, axis0: int, axis1: int) -> Tensor:
        raise NotImplementedError("swapaxes_")

    def swapdims(self: Tensor, dim0: int, dim1: int) -> Tensor:
        raise NotImplementedError("swapdims")

    def swapdims_(self: Tensor, dim0: int, dim1: int) -> Tensor:
        raise NotImplementedError("swapdims_")

    def symeig(
        self: Tensor, eigenvectors: bool = False, upper: bool = True
    ) -> ComplexReturnType("symeig"):
        raise NotImplementedError("symeig")

    def t(self: Tensor) -> Tensor:
        return pi.t(self)

    def t_(self: Tensor) -> Tensor:
        raise NotImplementedError("t_")

    def take(self: Tensor, index: Tensor) -> Tensor:
        raise NotImplementedError("take")

    def take_along_dim(
        self: Tensor, indices: Tensor, dim: Optional[int] = None
    ) -> Tensor:
        raise NotImplementedError("take_along_dim")

    def tan(self: Tensor) -> Tensor:
        raise NotImplementedError("tan")

    def tan_(self: Tensor) -> Tensor:
        raise NotImplementedError("tan_")

    def tanh(self: Tensor) -> Tensor:
        return pi.tanh(self)

    def tanh_(self: Tensor) -> Tensor:
        return pi.tanh_(self)

    @dispatch
    def tensor_split(self: Tensor, indices: List[int], dim: int = 0) -> List[Tensor]:
        raise NotImplementedError("tensor_split")

    @dispatch
    def tensor_split(
        self: Tensor, tensor_indices_or_sections: Tensor, dim: int = 0
    ) -> List[Tensor]:
        raise NotImplementedError("tensor_split")

    @dispatch
    def tensor_split(self: Tensor, sections: int, dim: int = 0) -> List[Tensor]:
        raise NotImplementedError("tensor_split")

    @dispatch
    def tile(self: Tensor, dims: Size) -> Tensor:
        raise NotImplementedError("tile")

    @dispatch
    def tile(self: Tensor, *dims: int) -> Tensor:
        raise NotImplementedError("tile")

    @dispatch
    def to(
        self: Tensor, dtype: pi_dtype, non_blocking: bool = False, copy: bool = False
    ) -> Tensor:
        return pi.to(self, dtype, non_blocking, copy)

    @dispatch
    def to(
        self: Tensor,
        device: Optional[Union[device, str]] = None,
        dtype: Optional[pi_dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Tensor:
        return pi.to(self, device, dtype, non_blocking, copy)

    @dispatch
    def to(
        self: Tensor, other: Tensor, non_blocking: bool = False, copy: bool = False
    ) -> Tensor:
        return pi.to(self, other, non_blocking, copy)

    def to_dense(self: Tensor, dtype: Optional[pi_dtype] = None) -> Tensor:
        raise NotImplementedError("to_dense")

    def to_mkldnn(self: Tensor, dtype: Optional[pi_dtype] = None) -> Tensor:
        raise NotImplementedError("to_mkldnn")

    def to_padded_tensor(
        self: Tensor, padding: float, output_size: Optional[List[int]] = None
    ) -> Tensor:
        raise NotImplementedError("to_padded_tensor")

    @dispatch
    def to_sparse(
        self: Tensor,
        *,
        layout: Optional[layout] = None,
        blocksize: Optional[Union[int, Size]] = None,
    ) -> Tensor:
        raise NotImplementedError("to_sparse")

    @dispatch
    def to_sparse(self: Tensor, sparse_dim: int) -> Tensor:
        raise NotImplementedError("to_sparse")

    @dispatch
    def to_sparse_bsc(self: Tensor, blocksize: Union[int, Size]) -> Tensor:
        raise NotImplementedError("to_sparse_bsc")

    @dispatch
    def to_sparse_bsc(self: Tensor, *blocksize: int) -> Tensor:
        raise NotImplementedError("to_sparse_bsc")

    @dispatch
    def to_sparse_bsr(self: Tensor, blocksize: Union[int, Size]) -> Tensor:
        raise NotImplementedError("to_sparse_bsr")

    @dispatch
    def to_sparse_bsr(self: Tensor, *blocksize: int) -> Tensor:
        raise NotImplementedError("to_sparse_bsr")

    def to_sparse_csc(self: Tensor) -> Tensor:
        raise NotImplementedError("to_sparse_csc")

    def to_sparse_csr(self: Tensor) -> Tensor:
        raise NotImplementedError("to_sparse_csr")

    def tolist(self: Tensor) -> List:
        raise NotImplementedError("tolist")

    def topk(
        self: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> ComplexReturnType("topk"):
        return pi.topk(self, k, dim, largest, sorted)

    def trace(self: Tensor) -> Tensor:
        raise NotImplementedError("trace")

    @dispatch
    def transpose(self: Tensor, dim0: int, dim1: int) -> Tensor:
        return pi.transpose(self, dim0, dim1)

    @dispatch
    def transpose(
        self: Tensor, dim0: Union[str, ellipsis, None], dim1: Union[str, ellipsis, None]
    ) -> Tensor:
        raise NotImplementedError("transpose")

    def transpose_(self: Tensor, dim0: int, dim1: int) -> Tensor:
        raise NotImplementedError("transpose_")

    def triangular_solve(
        self: Tensor,
        A: Tensor,
        upper: bool = True,
        transpose: bool = False,
        unitriangular: bool = False,
    ) -> ComplexReturnType("triangular_solve"):
        raise NotImplementedError("triangular_solve")

    def tril(self: Tensor, diagonal: int = 0) -> Tensor:
        raise NotImplementedError("tril")

    def tril_(self: Tensor, diagonal: int = 0) -> Tensor:
        raise NotImplementedError("tril_")

    def triu(self: Tensor, diagonal: int = 0) -> Tensor:
        return pi.triu(self, diagonal)

    def triu_(self: Tensor, diagonal: int = 0) -> Tensor:
        return pi.triu_(self, diagonal)

    def true_divide(
        self: Tensor, other: Union[Tensor, Number], *, out: Optional[Tensor] = None
    ) -> Tensor:
        if out is not None:
            raise NotImplementedError("true_divide.out variant")
        raise NotImplementedError("true_divide")

    def true_divide_(self: Tensor, other: Union[Tensor, Number]) -> Tensor:
        raise NotImplementedError("true_divide_")

    def trunc(self: Tensor) -> Tensor:
        raise NotImplementedError("trunc")

    def trunc_(self: Tensor) -> Tensor:
        raise NotImplementedError("trunc_")

    @dispatch
    def unbind(self: Tensor, dim: int = 0) -> List[Tensor]:
        raise NotImplementedError("unbind")

    @dispatch
    def unbind(self: Tensor, dim: Union[str, ellipsis, None]) -> List[Tensor]:
        raise NotImplementedError("unbind")

    @dispatch
    def unflatten(
        self: Tensor,
        dim: Union[str, ellipsis, None],
        sizes: Size,
        names: Sequence[Union[str, ellipsis, None]],
    ) -> Tensor:
        raise NotImplementedError("unflatten")

    @dispatch
    def unflatten(self: Tensor, dim: int, sizes: Size) -> Tensor:
        raise NotImplementedError("unflatten")

    def unfold(self: Tensor, dimension: int, size: int, step: int) -> Tensor:
        raise NotImplementedError("unfold")

    def uniform_(
        self: Tensor,
        from_: float = 0,
        to: float = 1,
        *,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        return pi.uniform_(self, from_, to, generator)

    def unsafe_chunk(self: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
        raise NotImplementedError("unsafe_chunk")

    def unsafe_split(self: Tensor, split_size: int, dim: int = 0) -> List[Tensor]:
        raise NotImplementedError("unsafe_split")

    def unsafe_split_with_sizes(
        self: Tensor, split_sizes: List[int], dim: int = 0
    ) -> List[Tensor]:
        raise NotImplementedError("unsafe_split_with_sizes")

    def unsqueeze(self: Tensor, dim: int) -> Tensor:
        return pi.unsqueeze(self, dim)

    def unsqueeze_(self: Tensor, dim: int) -> Tensor:
        return pi.unsqueeze_(self, dim)

    def values(self: Tensor) -> Tensor:
        raise NotImplementedError("values")

    @dispatch
    def var(
        self: Tensor,
        dim: Optional[Union[int, Size]],
        unbiased: bool = True,
        keepdim: bool = False,
    ) -> Tensor:
        return pi.var(self, dim, unbiased, keepdim)

    @dispatch
    def var(
        self: Tensor,
        dim: Optional[Union[int, Size]] = None,
        *,
        correction: Optional[int] = None,
        keepdim: bool = False,
    ) -> Tensor:
        return pi.var(self, dim, correction, keepdim)

    @dispatch
    def var(self: Tensor, unbiased: bool = True) -> Tensor:
        return pi.var(self, unbiased)

    @dispatch
    def var(
        self: Tensor,
        dim: Sequence[Union[str, ellipsis, None]],
        unbiased: bool = True,
        keepdim: bool = False,
    ) -> Tensor:
        raise NotImplementedError("var")

    @dispatch
    def var(
        self: Tensor,
        dim: Sequence[Union[str, ellipsis, None]],
        *,
        correction: Optional[int] = None,
        keepdim: bool = False,
    ) -> Tensor:
        raise NotImplementedError("var")

    def vdot(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("vdot")

    @dispatch
    def view(self: Tensor, dtype: pi_dtype) -> Tensor:
        raise NotImplementedError("view")

    @dispatch
    def view(self: Tensor, size: List[int]) -> Tensor:
        return pi.view(self, size)

    @dispatch
    def view(self: Tensor, *size: int) -> Tensor:
        return pi.view(self, size)

    def view_as(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("view_as")

    @dispatch
    def vsplit(self: Tensor, sections: int) -> List[Tensor]:
        raise NotImplementedError("vsplit")

    @dispatch
    def vsplit(self: Tensor, indices: Size) -> List[Tensor]:
        raise NotImplementedError("vsplit")

    @dispatch
    def vsplit(self: Tensor, *indices: int) -> List[Tensor]:
        raise NotImplementedError("vsplit")

    def where(self: Tensor, condition: Tensor, other: Tensor) -> Tensor:
        return pi.where(self, condition, other)

    @dispatch
    def xlogy(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("xlogy")

    @dispatch
    def xlogy(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("xlogy")

    @dispatch
    def xlogy_(self: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError("xlogy_")

    @dispatch
    def xlogy_(self: Tensor, other: Number) -> Tensor:
        raise NotImplementedError("xlogy_")

    def zero_(self: Tensor) -> Tensor:
        return pi.zero_(self)


def from_numpy(arr: np.ndarray):
    from pi import DEBUG

    if DEBUG:
        arr = np.ones_like(arr, dtype=np.float32)
    attr = DenseElementsAttr.get(arr)
    vt = Tensor(torch_dialect.ValueTensorLiteralOp(attr))
    return vt


def empty(shape: Tuple[int, ...], dtype: "pi.dtype" = None, **kwargs) -> Tensor:
    if np.prod(shape) == 0:
        return Tensor(None)
    elif dtype is not None:
        dtype = dtype.to_np_type()
    return from_numpy(np.empty(shape, dtype))


def randint(low: int, high: int, size: Tuple[int, ...]) -> Tensor:
    return from_numpy(np.random.randint(low, high, size))


def randn(*size: Tuple[int, ...]) -> Tensor:
    if size == ((),):
        return from_numpy(np.random.randn())
    return from_numpy(np.random.randn(*size))


def uniform(low: float, high: float, size: Tuple[int, ...]) -> Tensor:
    return from_numpy(np.random.uniform(low, high, size))


def rand(*size: Tuple[int, ...], **kwargs) -> Tensor:
    if size == ((),):
        return from_numpy(np.random.rand())
    if isinstance(size[0], tuple):
        assert len(size) == 1, f"malformed size tuple {size}"
        size = size[0]
    return from_numpy(np.random.rand(*size))


def ones(*size: Tuple[int, ...], **kwargs) -> Tensor:
    dtype = kwargs.get("dtype", None)
    if dtype is not None:
        dtype = dtype.to_np_type()
    if isinstance(size[0], tuple):
        assert len(size) == 1, f"malformed size tuple {size}"
        size = size[0]
    return from_numpy(np.ones(size, dtype=dtype))


def zeros(*size: Tuple[int, ...], **kwargs) -> Tensor:
    dtype = kwargs.get("dtype", None)
    if dtype is not None:
        dtype = dtype.to_np_type()
    return from_numpy(np.zeros(size, dtype))


def tensor(data: Any, dtype: Optional["pi.dtype"] = None) -> Tensor:
    if dtype is not None:
        dtype = dtype.to_np_type()
    return from_numpy(np.array(data, dtype=dtype))


def LongTensor(data: Any) -> Tensor:
    return from_numpy(np.array(data, dtype=pi_dtype.int64.to_np_type()))


def clone(x: Tensor, **kwargs):
    warnings.warn(f"not actually cloning")
    return x


def torch_cast(x: Tensor, dtype: pi_dtype):
    assert is_a_torch_tensor(x), f"x should be a Tensor but is {type(x)}"
    return Tensor(torch_dialect.AtenToDtypeOp(x, dtype.value, False, False, None))


def ScalarImplicit(x: Tensor):
    assert is_a_torch_tensor(x), f"x should be a Tensor but is {type(x)}"
    return Tensor(torch_dialect.AtenScalarImplicitOp(x))


__all__ = [
    "from_numpy",
    "empty",
    "randint",
    "randn",
    "rand",
    "uniform",
    "ones",
    "tensor",
    "Tensor",
    "LongTensor",
    "zeros",
    "ScalarImplicit",
]
