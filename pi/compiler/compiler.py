import os

from torch_mlir import ir

# noinspection PyUnresolvedReferences
from torch_mlir.dialects import arith, linalg, math, memref, torch as torch_dialect

# noinspection PyUnresolvedReferences
from pi.dialects import affine_

from pi.compiler.tracing.trace import trace


def mlir_trace(script_path):
    assert os.path.isabs(script_path), f"script path must be absolute {script_path}"
    top_mlir_context = ir.Context()
    mlir_location = ir.Location.unknown(context=top_mlir_context)
    with top_mlir_context, mlir_location:
        mlir_module = ir.Module.create(loc=mlir_location)
        torch_dialect.register_dialect(top_mlir_context)

    mlir_module = trace(
        script_path,
        top_mlir_context,
        mlir_location,
        mlir_module,
    )
    return mlir_module
