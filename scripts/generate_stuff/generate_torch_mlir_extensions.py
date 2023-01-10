import argparse
import warnings
from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import List, Callable

from torch_mlir.dialects.torch.importer.jit_ir.build_tools.registry import (
    JitOperator,
    Registry,
    _get_default_value,
    _rename_python_keyword_parameter_name,
)
from torch_mlir.dialects.torch.importer.jit_ir.build_tools.utils import TextEmitter

# from scripts.generate_stuff.generate_pytorch_wrappers import (
#     tensor_method_signatures_dict,
#     function_signatures_dict,
# )


ALL_OPS = []
DEBUG = False
BLACKLIST = {"dtype", "PrimUncheckedCastOp", "device"}
SUBS = {"linalg_vector_norm": "vector_norm"}
UNIQUE_OPS = []

subs = {
    "Scalar": "Number",
    "t2": "Tensor",
    "t1": "Tensor",
    "t": "Tensor",
    "Dict(str, t)": "Dict[str, Tensor]",
    "device": "Device",
}

TORCH_TYPE_TO_ODS_TYPE = {
    "Any": "Torch_AnyType",
    "Device": "Torch_DeviceType",
    "Device?": "Torch_DeviceType",
    "Dict": "Torch_DictType",
    "Dict(str, t)": "Torch_DictType",
    "Generator": "Torch_GeneratorType",
    "Generator?": "Torch_GeneratorType",
    "Scalar": "TorchScalarType",
    "Scalar?": "TorchScalarType",
    "Tensor": "Torch_ValueTensorType",
    "Tensor?": "Torch_ValueTensorType",
    "Tensor?[]": "TorchListOfValueTensorType",
    "Tensor[]": "TorchListOfValueTensorType",
    "__torch__.torch.classes.quantized.LinearPackedParamsBase": "Torch_LinearParamsType",
    "bool": "Torch_BoolType",
    "bool?": "Torch_BoolType",
    "bool[]": "TorchListOfTorchBoolType",
    "float": "Torch_FloatType",
    "float?": "Torch_FloatType",
    "float[]": "TorchListOfTorchFloatType",
    "float[]?": "TorchListOfTorchFloatType",
    "int": "Torch_IntType",
    "int?": "Torch_IntType",
    "int[]": "TorchListOfTorchIntType",
    "int[]?": "TorchListOfTorchIntType",
    "str": "Torch_StringType",
    "str?": "Torch_StringType",
    "str[]": "TorchListOfTorchStringType",
    "t": "Torch_ValueTensorType",
    "t1": "Torch_ValueTensorType",
    "t2": "Torch_ValueTensorType",
    "t[]": "TorchListOfValueTensorType",
}

EXISTING = {
    "ConstantFloatOp",
    "ConstantIntOp",
    "ConstantStrOp",
    "ConstantBoolOp",
    "PrimListConstructOp",
    "PrimUncheckedCastOp",
    "PrimTupleConstructOp",
    "AtenScalarImplicitOp",
}

SCALAR_TYPES = {
    "int",
    "bool",
    "str",
    "float",
    "Any",
    "Device",
    "Scalar",
}


def _get_function_signature(
    self,
    function_kind: str,
    parameter_decl_builder: Callable[["SIG_ATTR_TYPE"], str],
    ret_decl_builder: Callable[["SIG_ATTR_TYPE"], str],
) -> str:
    def_name = self.unqualified_name
    parameter_decls = list(map(parameter_decl_builder, self.arguments))
    ret_decls = list(map(ret_decl_builder, self.returns))
    parameters = ", ".join(parameter_decls)
    result = ", ".join(ret_decls)
    if len(ret_decls) >= 2:
        result = f"Tuple[{result}]"

    if len(ret_decls) == 0:
        result = "None"

    # TODO: leave off return annot because plum tries to promote
    # return f"def {def_name}({parameters}) -> {result}:"
    def_name = SUBS.get(def_name, def_name)
    return f"def {def_name}({parameters}):"


# JitOperator._get_function_signature = _get_function_signature


# pyt_type is reversed
def convert_pytorch_type_to_typehints(pyt_type: str):
    """notice that pytorch types can be read off right to left"""

    if pyt_type.endswith("?"):
        nested, interior = convert_pytorch_type_to_typehints(pyt_type[:-1])
        return f"Optional[{nested}]", interior
    elif pyt_type.endswith("[]"):
        nested, interior = convert_pytorch_type_to_typehints(pyt_type[:-2])
        return f"List[{nested}]", interior
    else:
        interior = subs.get(pyt_type, pyt_type)

        return interior, interior


def py_reserved_keywords(k):
    subs = {
        "from": "from_",
        "self": "self_",
        "list": "list_",
    }
    return subs.get(k, k)


def get_wrapper_function_signature(operator):
    def parameter_decl_builder(arg: "SIG_ATTR_TYPE") -> str:
        pytype = convert_pytorch_type_to_typehints(arg["type"])[0]
        default = _get_default_value(arg)
        if arg["name"] == "out":
            default = " = None"
            pytype = f"Optional[{pytype}]"
        if "dtype" in arg["name"]:
            if arg["type"][-1] == "?":
                default = " = None"
                pytype = f"Optional[pi_dtype]"
            else:
                default = ""
                pytype = f"pi_dtype"
        if "layout" in arg["name"]:
            if arg["type"][-1] == "?":
                default = " = None"
                pytype = f"Optional[pi_layout]"
            else:
                default = ""
                pytype = f"pi_layout"
        if "memory_format" in arg["name"]:
            if arg["type"][-1] == "?":
                default = " = None"
                pytype = f"Optional[pi_memory_format]"
            else:
                default = ""
                pytype = f"pi_memory_format"
        parameter_name = py_reserved_keywords(
            _rename_python_keyword_parameter_name(arg["name"])
        )
        return f"{parameter_name}: {pytype}{default}"

    def ret_decl_builder(arg: "SIG_ATTR_TYPE") -> str:
        ret = convert_pytorch_type_to_typehints(arg["type"])[0]
        if not ret:
            ret = "None"
        return ret

    return _get_function_signature(
        operator, "", parameter_decl_builder, ret_decl_builder
    )


def convert_pytorch_type_to_torch_dialect_op(arg_name, pyt_type, p_td, emitter_td):
    _, interior = convert_pytorch_type_to_typehints(pyt_type)

    if pyt_type.endswith("?"):
        p_td(f"if {arg_name} is not None:")
        with emitter_td.indent():
            convert_pytorch_type_to_torch_dialect_op(
                arg_name, pyt_type[:-1], p_td, emitter_td
            )
        p_td(f"else:")
        with emitter_td.indent():
            p_td(f"{arg_name} = torch_dialect.ConstantNoneOp()")
        p_td("")
    elif pyt_type.endswith("[]"):
        if interior == "Tensor":
            pass  # p_td(f"{arg_name} = get_op_results_or_values({arg_name})")
        else:
            op = convert_pytorch_type_to_torch_dialect_op(
                None, pyt_type[:-2], p_td, emitter_td
            )
            p_td(
                f"{arg_name} = [{op}(a) if not is_mlir_value(a) else a for a in {arg_name}]"
            )
        p_td(f"{arg_name} = torch_dialect.PrimListConstructOp({arg_name})")
    else:
        if interior in {"int", "bool", "float", "Number", "str", "Device"}:
            op = f"torch_dialect.Constant{interior.capitalize()}Op"
            if arg_name is not None:
                p_td(f"{arg_name} = {op}({arg_name})")
            else:
                return op
        else:
            if arg_name is not None:
                p_td(
                    f"assert is_mlir_value({arg_name}), f'`{arg_name}` should be a Value but is {{type({arg_name})}}'"
                )


def emit_torch_wrappers(
    operators,
    all,
    stubs_emitter_td: TextEmitter,
):
    counts = Counter(all)
    stub_td = lambda *args: stubs_emitter_td.print(*args)
    for operator in operators:
        args = {
            py_reserved_keywords(arg["name"]): convert_pytorch_type_to_typehints(
                arg["type"]
            )[0]
            for arg in operator.arguments
        }
        op_name, cpp_class_name = operator.get_mlir_names()
        if operator.overload_name:
            stub_td(f"# overload {operator.overload_name}")
        if counts[operator.unqualified_name] > 1:
            stub_td("@dispatch")
        stub_td(get_wrapper_function_signature(operator))
        with stubs_emitter_td.indent():
            for arg in operator.arguments:
                arg_name = py_reserved_keywords(arg["name"])
                if "dtype" in arg_name:
                    stub_td(f"if {arg_name} is not None:")
                    with stubs_emitter_td.indent():
                        stub_td(
                            f"assert isinstance({arg_name}, pi_dtype), f'expected pi_dtype, got {{type({arg_name})}}'"
                        )
                        stub_td(f"{arg_name} = {arg_name}.value")
                if "layout" in arg_name:
                    stub_td(f"if {arg_name} is not None:")
                    with stubs_emitter_td.indent():
                        stub_td(
                            f"assert isinstance({arg_name}, pi_layout), f'expected pi_layout, got {{type({arg_name})}}'"
                        )
                        stub_td(f"{arg_name} = {arg_name}.value")
                if "memory_format" in arg_name:
                    stub_td(f"if {arg_name} is not None:")
                    with stubs_emitter_td.indent():
                        stub_td(
                            f"assert isinstance({arg_name}, pi_memory_format), f'expected pi_memory_format, got {{type({arg_name})}}'"
                        )
                        stub_td(f"{arg_name} = {arg_name}.value")
                if arg["type"] == "Tensor":
                    stub_td(
                        f"assert is_a_torch_tensor({arg_name}), f'`{arg_name}` should be a Tensor but is {{type({arg_name})}}'"
                    )
                elif arg["type"] == "Tensor?":
                    stub_td(f"if {arg_name} is not None:")
                    with stubs_emitter_td.indent():
                        stub_td(
                            f"assert is_a_torch_tensor({arg_name}), f'`{arg_name}` should be a Tensor but is {{type({arg_name})}}'"
                        )
                elif arg["type"] in {"Tensor[]", "Tensor?[]"}:
                    stub_td(
                        f"assert builtins.all(is_a_torch_tensor(t) or t is None for t in {arg_name})"
                    )
                elif arg["type"] == "int[]":
                    stub_td(f"if not isinstance({arg_name}, (tuple, builtins.list)):")
                    with stubs_emitter_td.indent():
                        stub_td(f"{arg_name} = [{arg_name}]")
                elif arg["type"] == "int[]?":
                    stub_td(
                        f"if {arg_name} is not None and not isinstance({arg_name}, (tuple, builtins.list)):"
                    )
                    with stubs_emitter_td.indent():
                        stub_td(f"{arg_name} = [{arg_name}]")

            if DEBUG:
                stub_td(f"print('running {get_wrapper_function_signature(operator)}')")
                stub_td(f"return")
            else:
                call_str = f'torch_dialect.{cpp_class_name}({", ".join([f"{k}" for k, _v in args.items()])})'
                if len(operator.returns) == 0:
                    ret = call_str
                elif len(operator.returns) == 1:
                    ret_type = operator.returns[0]["type"]
                    ret_ods_type = TORCH_TYPE_TO_ODS_TYPE[ret_type].replace(
                        "Scalar", "_Number"
                    )
                    if ret_ods_type == "Torch_ValueTensorType":
                        ret = f"return Tensor({call_str}.result)"
                    else:
                        ret = f"return _{ret_ods_type.replace('Type', '')}({call_str}.result)"
                else:
                    stub_td(f"op_results = get_op_results_or_values({call_str})")
                    ret = f"return tuple([Tensor(o) if is_a_torch_tensor(o) else o for o in op_results])"
                stub_td(f"{ret}")
            stub_td("\n")


def generate_torch_wrappers(torch_ops_ext_dir: Path):
    torch_wrappers_fp = torch_ops_ext_dir / "_torch_wrappers.py"
    with open(torch_wrappers_fp, "w") as stubs_td:
        stubs_emitter_td = TextEmitter(stubs_td)
        stubs_emitter_td._INDENT = "    "
        stubs_emitter_td.print(
            dedent(
                f"""\
            import builtins
            from numbers import Number
            from typing import List, Optional, Any, Dict
            
            from ._tensor import Tensor, ScalarImplicit
            from .types_ import is_a_torch_tensor, Device, Generator, dtype as pi_dtype, layout as pi_layout, memory_format as pi_memory_format
            from pi._mlir import (
                _Torch_Any,
                _Torch_Bool,
                _Torch_Device,
                _Torch_Float,
                _Torch_Int,
                _Torch_Number,
                _Torch_String,
                _Torch_ValueTensor,
                _TorchListOfValueTensor,
                _TorchListOfTorchString,
                _TorchListOfTorchInt,
                is_dtype,
            )
            from .dispatcher import dispatch
            from torch_mlir.dialects import torch as torch_dialect
            from torch_mlir.dialects._ods_common import (
                get_op_results_or_values,
            )
            
            """
            )
        )

        BLACKLIST = {"dtype", "PrimUncheckedCastOp", "device"}
        SUBS = {"linalg_vector_norm": "vector_norm"}
        TORCH_WRAPPERS = sorted(UNIQUE_OPS, key=lambda t: t.unqualified_name)
        TORCH_WRAPPERS = [
            t for t in TORCH_WRAPPERS if t.unqualified_name not in BLACKLIST
        ]
        ALL = [t for t in ALL_OPS if t not in BLACKLIST]
        emit_torch_wrappers(TORCH_WRAPPERS, ALL, stubs_emitter_td)
        stubs_emitter_td.print("\n\n")
        all = [
            f'"{SUBS.get(t.unqualified_name, t.unqualified_name)}"'
            for t in TORCH_WRAPPERS
        ]
        stubs_emitter_td.print(
            f"__all__ = ['ScalarImplicit', {', '.join(sorted(set(all)))}]"
        )


def raw_emit_op(
    operator: JitOperator,
    emitter_td: TextEmitter,
    *,
    traits: List[str],
    has_folder: bool,
    has_canonicalizer: bool,
):
    p_td = lambda *args: emitter_td.print(*args)
    op_name, cpp_class_name = operator.get_mlir_names()
    if cpp_class_name in {"QuantizedLinearOp"} | EXISTING:
        return
    if operator.unqualified_name == "torch.quantized.linear":
        return

    # Generate unique result names for ops with nameless results
    multiple_results = len(operator.returns) > 1

    if operator.is_vararg:
        warnings.warn(f"{cpp_class_name} is vararg; skipping")
        return
    else:
        args = {
            py_reserved_keywords(arg["name"]): convert_pytorch_type_to_typehints(
                arg["type"]
            )[0]
            for arg in operator.arguments
        }
        for k, v in args.items():
            args[k] = v.replace("Tensor", "Value")

    def generic_result_name(i):
        return "result" + (str(i) if multiple_results else "")

    if operator.is_varret:
        warnings.warn(f"{cpp_class_name} is vararg; skipping")
        return

    UNIQUE_OPS.append(operator)
    ALL_OPS.append(operator.unqualified_name)

    # print(f"{has_folder=}, {has_canonicalizer=}")

    p_td(f"class {cpp_class_name}:")

    with emitter_td.indent():
        # generate __init__ args
        args_str = ", ".join([f"{k}: {v}" for k, v in args.items()])
        if args_str:
            args_str = f" {args_str},"
        p_td(f"def __init__(self,{args_str} *, loc=None, ip=None):")

        with emitter_td.indent():
            # check if we need torch_Constant* ops
            if any(
                [
                    convert_pytorch_type_to_typehints(arg["type"])[1] != "Tensor"
                    or "?" in arg["type"]
                    or "[]" in arg["type"]
                    for arg in operator.arguments
                ]
            ):
                p_td(f"from torch_mlir.dialects import torch as torch_dialect\n\n")

            arg_names = []
            for arg in operator.arguments:
                arg_name = py_reserved_keywords(arg["name"])
                arg_pytorch_type = arg["type"]
                arg_pytype = arg["pytype"]
                # we check for python None
                ods_type = TORCH_TYPE_TO_ODS_TYPE[arg_pytorch_type]
                # if "dtype" in arg_name:
                #     p_td(f"if not isinstance({arg_name}, MLIRType):")
                #     with emitter_td.indent():
                #         convert_pytorch_type_to_torch_dialect_op(
                #             arg_name, arg_pytorch_type, p_td, emitter_td
                #         )
                #     p_td(f"else:")
                #     with emitter_td.indent():
                #         p_td(
                #             f"""assert is_dtype({arg_name}), f'`{arg_name}` should be a dtype but is {{type({arg_name})}}'"""
                #         )
                #
                #         p_td("\n")
                # else:
                p_td(f"if not is_mlir_value({arg_name}):")
                with emitter_td.indent():
                    convert_pytorch_type_to_torch_dialect_op(
                        arg_name, arg_pytorch_type, p_td, emitter_td
                    )
                p_td(f"else:")
                with emitter_td.indent():
                    p_td(f"{arg_name} = get_op_result_or_value({arg_name})")
                    p_td(
                        f"""assert is_a_{ods_type}({arg_name}.type), f'`{arg_name}` should be a {ods_type} but is {{type({arg_name})}}'"""
                    )

                    p_td("\n")

                arg_names.append(arg_name)

            ret_type_names = []
            for e, ret in enumerate(operator.returns):
                ret_name = py_reserved_keywords(ret["name"])
                ret_pytorch_type = ret["type"]
                # we want to check for Scalar types but ScalarType isn't a real type (it's NumberType)
                ods_type = TORCH_TYPE_TO_ODS_TYPE[ret_pytorch_type].replace(
                    "Scalar", "_Number"
                )
                if "List" in ods_type or "Tensor" in ods_type:
                    name = f"{ret_name or generic_result_name(e)}"
                    p_td(f"""{name}_type = _{ods_type}()""")
                    ret_type_names.append(f"{name}_type")

            if ret_type_names:
                ret_type_names = f"{', '.join(ret_type_names)}, "
            else:
                ret_type_names = ""

            if arg_names:
                arg_names = f"{', '.join(arg_names)}, "
            else:
                arg_names = ""

            p_td(
                f"super({cpp_class_name}, self).__init__({ret_type_names}{arg_names}loc=loc, ip=ip)"
            )
            p_td("\n")
        p_td("\n")


import torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen

torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen.raw_emit_op = (
    raw_emit_op
)

from torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen import emit_ops


def generate_exts(torch_ops_ext_dir: Path):
    torch_ops_ext_fp = torch_ops_ext_dir / "_torch_ops_ext.py"
    registry = Registry.load()
    with open(torch_ops_ext_fp, "w") as f_td:
        emitter_td = TextEmitter(f_td)
        emitter_td._INDENT = "    "
        emitter_td.print(
            dedent(
                f"""\
        try:
            from torch_mlir.ir import *
            from torch_mlir.ir import Type as MLIRType
            from torch_mlir.dialects._ods_common import (
                get_default_loc_context,
                get_op_result_or_value,
                get_op_results_or_values,
            )
            from ._torch_ops_ext_custom import *
            from pi._mlir import (
                is_a_TorchListOfTorchBoolType,
                is_a_TorchListOfTorchIntType,
                is_a_TorchListOfTorchStringType,
                is_a_TorchListOfValueTensorType,
                is_a_TorchScalarType,
                is_a_Torch_AnyType,
                is_a_Torch_BoolType,
                is_a_Torch_DeviceType,
                is_a_Torch_DictType,
                is_a_Torch_FloatType,
                is_a_Torch_GeneratorType,
                is_a_Torch_IntType,
                is_a_Torch_StringType,
                is_a_Torch_ValueTensorType,
                _TorchListOfTorchBoolType,
                _TorchListOfTorchFloatType,
                _TorchListOfTorchIntType,
                _TorchListOfTorchStringType,
                _TorchListOfValueTensorType,
                _Torch_AnyType,
                _Torch_BoolType,
                _Torch_DeviceType,
                _Torch_FloatType,
                _Torch_IntType,
                _Torch_NumberType,
                _Torch_StringType,
                _Torch_ValueTensorType,
                is_dtype,
            )
            
        except ImportError as e:
            raise RuntimeError("Error loading imports from extension module") from e

        from numbers import Number
        from typing import List, Optional, Any, Generator, Dict
        Device = str


        """
            )
        )
        emit_ops(emitter_td, registry)


def main(args):
    generate_exts(args.ext_dir)
    generate_torch_wrappers(args.wrapper_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PI wrappers")
    parser.add_argument(
        "--ext_dir",
        help="directory where torch-mlir python binding extensions should be placed",
        default="../../pi/dialects",
        type=Path,
    )
    parser.add_argument(
        "--wrapper_dir",
        help="directory where PyTorch wrappers should be placed",
        default="../../pi",
        type=Path,
    )
    args = parser.parse_args()
    main(args)
