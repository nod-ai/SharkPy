# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import contextlib
from io import StringIO
import os
import sys
import tempfile

from torch_mlir.passmanager import PassManager
from torch_mlir.ir import StringAttr


def get_module_name_for_debug_dump(module):
    """Gets a name suitable for a debug dump.

    The name is not guaranteed to be unique.
    """
    if not "torch.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["torch.debug_module_name"]).value


class TorchMlirCompilerError(Exception):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return self.value


def run_pipeline_with_repro_report(
    module,
    pipeline: str,
    description: str,
    enable_ir_printing=False,
    print_pipeline=False,
):
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True
        )
        # Lower module in place to make it ready for compiler backends.
        with module.context:
            pm = PassManager.parse(pipeline)
            if print_pipeline:
                print(pm)
            if enable_ir_printing:
                pm.enable_ir_printing()
            pm.run(module)
    except Exception as e:
        print(e, file=sys.stderr)
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        # Put something descriptive here even if description is empty.
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:
            {sys.stderr.getvalue()}

            For Torch-MLIR developers, the error can be reproduced with:
            $ torch-mlir-opt -pass-pipeline='{pipeline}' {filename}
            Add '{debug_options}' to get the IR dump for debugging purpose.
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise TorchMlirCompilerError(trimmed_message) from None
    finally:
        sys.stderr = original_stderr


def lower_pi_to_linalg(module, enable_ir_printing=False):
    run_pipeline_with_repro_report(
        module,
        "builtin.module("
        + ",".join(
            [
                # "builtin.module(torchscript-module-to-torch-backend-pipeline)",
                # "torchscript-module-to-torch-backend-pipeline",
                "symbol-dce",
                "inline{default-pipeline= max-iterations=4}",
                "torch-adjust-calling-conventions",
                "torch-lower-to-backend-contract{decompose=true max-iterations=10}",
                "torch-backend-to-linalg-on-tensors-backend-pipeline",
            ]
        )
        + ")",
        "Lowering Torch MLIR -> Linalg",
        enable_ir_printing,
    )
    return module


@contextlib.contextmanager
def mlir_cm(enable_multithreading=False):
    from torch_mlir.ir import Context, Location, Module, InsertionPoint
    from torch_mlir.dialects import torch as torch_dialect

    with Context() as ctx, Location.unknown():
        ctx.enable_multithreading(enable_multithreading)
        torch_dialect.register_dialect(ctx, True)
        module = Module.create()
        with InsertionPoint(module.body):
            yield module
