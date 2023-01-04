import sys
import os

old_flags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
from pi import _mlir

from torch_mlir import ir

torch_mlir_pybind11_module_local_id = next(d for d in dir(ir.Value) if "pybind" in d)

assert _mlir.get_pybind11_module_local_id() == torch_mlir_pybind11_module_local_id

# ctx = ir.Context()
# with ctx:
#     loc = ir.Location.unknown()
#     with loc:
#         module = ir.Module.create()
#         with ir.InsertionPoint.at_block_begin(module.body) as ip:
#             v = arith.ConstantOp.create_index(10, ip=ip)
#             assert _mlir._load_foreign(v)
