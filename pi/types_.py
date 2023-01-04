import builtins
import re
import weakref
from enum import Enum
from typing import Union, List, Tuple, Any, NewType

import numpy as np
from torch_mlir import ir
from torch_mlir.dialects._ods_common import (
    get_op_result_or_value,
)
from torch_mlir.ir import (
    Type as MLIRType,
    Value as MLIRValue,
)

# !torch.vtensor<[1,2,3],f32>
reg = re.compile(r"!torch.vtensor<\[(.*)\],(.*)>")


def parse_sizes_from_tensor_type_str(t: ir.OpView) -> List[int]:
    # TODO(max): pull straight from the ranked type
    t = get_op_result_or_value(t)
    sizes, dtype = reg.findall(str(t.type))[0]
    sizes = [s if s != "?" else "-1" for s in sizes.split(",")]
    return list(map(int, sizes)), dtype


def get_type(t: Union[MLIRType, MLIRValue]):
    if not isinstance(t, MLIRType):
        assert isinstance(
            t, MLIRValue
        ), f"unknown type {type(t).__module__}.{type(t).__name__})"
        t = t.type
    return t


def is_a_torch_tensor(t):
    try:
        t = get_op_result_or_value(t)
        type_str = str(t.type)
        return "torch.tensor" in type_str or "torch.vtensor" in type_str
    except:
        return False


# IntegerType.get_signless(32) -> i32
# IntegerType.get_signed(32) -> si32
# IntegerType.get_unsigned(32) -> ui32


class dtype(Enum):
    """
    |-------------------|--------------------|
    | Torch Type        | MLIR Type          |
    |-------------------|--------------------|
    | torch.bfloat16    | bf16               |
    | torch.bool        | i1                 |
    | torch.complex*    | complex<*>         |
    | torch.float16     | f16                |
    | torch.float32     | f32                |
    | torch.float64     | f64                |
    | torch.int16       | si16               |
    | torch.int32       | si32               |
    | torch.int64       | si64               |
    | torch.int8        | si8                |
    | torch.qint8       | !torch.qint8       |
    | torch.quint8      | !torch.quint8      |
    | torch.uint8       | ui8                |
    |-------------------|--------------------|
    """

    uint8 = 0
    int8 = 1
    int16 = 2
    int32 = 3
    int64 = 4
    float16 = 5
    float32 = 6
    float64 = 7
    # complex_half 8
    complex32 = 9
    complex64 = 10
    bool = 11
    qint8 = 12
    quint8 = 13
    # qint32 14
    bfloat16 = 15

    # qint4x2 16
    # qint2x4 17

    def to_mlir_type(self):
        match self:
            case dtype.bfloat16:
                return ir.BF16Type.get()
            case dtype.bool:
                return ir.IntegerType.get_signless(1)
            case dtype.complex32:
                return ir.ComplexType.get(ir.F32Type.get())
            case dtype.complex64:
                return ir.ComplexType.get(ir.F64Type.get())
            case dtype.float16:
                return ir.F16Type.get()
            case dtype.float32:
                return ir.F32Type.get()
            case dtype.float64:
                return ir.F64Type.get()
            case dtype.int8:
                return ir.IntegerType.get_signed(8)
            case dtype.int16:
                return ir.IntegerType.get_signed(16)
            case dtype.int32:
                return ir.IntegerType.get_signed(32)
            case dtype.int64:
                return ir.IntegerType.get_signed(64)
            case dtype.uint8:
                return ir.IntegerType.get_unsigned(8)
            case _:
                raise NotImplementedError("Something's wrong with the internet")

    def to_np_type(self):
        match self:
            case dtype.bfloat16 | dtype.float16:
                return np.half
            case dtype.bool:
                return np.bool_
            case dtype.complex32 | dtype.complex64:
                return np.complex_
            case dtype.float32:
                return np.float32
            case dtype.float64:
                return np.float64
            case dtype.int8:
                return np.int8
            case dtype.int16:
                return np.int16
            case dtype.int32:
                return np.int32
            case dtype.int64:
                return np.int64
            case dtype.uint8:
                return np.uint8
            case _:
                raise NotImplementedError("Something's wrong with the internet")

    @staticmethod
    def from_mlir_type(t: str):
        match t:
            case "bf16":
                return dtype.bfloat16
            case "i1":
                return dtype.bool
            case "complex32":
                return dtype.complex32
            case "complex64":
                return dtype.complex64
            case "f16":
                return dtype.float16
            case "f32":
                return dtype.float32
            case "f64":
                return dtype.float64
            case "si8":
                return dtype.int8
            case "si16":
                return dtype.int16
            case "si32":
                return dtype.int32
            case "si64":
                return dtype.int64
            case "ui8":
                return dtype.uint8
            case _:
                raise NotImplementedError(f"Something's wrong with the internet {t}")


bfloat16 = dtype.bfloat16
bool = dtype.bool
complex32 = dtype.complex32
complex64 = dtype.complex64
half = float16 = dtype.float16
float = float32 = dtype.float32
double = float64 = dtype.float64
int8 = dtype.int8
int16 = dtype.int16
int32 = dtype.int32
long = int64 = dtype.int64
qint8 = dtype.qint8
quint8 = dtype.quint8
uint8 = dtype.uint8

Size = size = Union[List[int], Tuple[int, ...]]

Number = Union[builtins.int, builtins.float, builtins.bool]
Generator = Any
device = Device = NewType("Device", str)

class BroadcastingListCls(object):
    def __getitem__(self, types):
        return


BroadcastingList1 = BroadcastingList2 = BroadcastingList3 = BroadcastingListCls()

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched: "weakref.WeakKeyDictionary[Callable, Dict[str, Callable]]" = (
    weakref.WeakKeyDictionary()
)  # noqa: T484


def boolean_dispatch(
    arg_name, arg_index, default, if_true, if_false, module_name, func_name
):
    def fn(*args, **kwargs):
        dispatch_flag = False
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name,
    }
    return fn


class layout(Enum):
    strided = 1
    sparse_coo = 2
    sparse_csr = 3
    sparse_csc = 4
    sparse_bsr = 5
    sparse_bsc = 6
    _mkldnn = 7


strided: layout = layout.strided
sparse_coo: layout = layout.sparse_coo
sparse_csr: layout = layout.sparse_csr
sparse_csc: layout = layout.sparse_csc
sparse_bsr: layout = layout.sparse_bsr
sparse_bsc: layout = layout.sparse_bsc
_mkldnn: layout = layout._mkldnn


class memory_format(Enum):
    contiguous_format = 0
    preserve_format = 1
    channels_last = 2
    channels_last_3d = 3


contiguous_format = memory_format.contiguous_format
preserve_format = memory_format.preserve_format
channels_last = memory_format.channels_last
channels_last_3d = memory_format.channels_last_3d
