import operator
import sys
from collections import namedtuple

import ast
import ctypes
import inspect
import os

from pi.compiler.annotations import PI_EXPORT_ATTR_NAME

# noinspection PyUnresolvedReferences
import pi.compiler.tracing.handlers
import pyccolo as pyc
import traceback
import types
from contextlib import contextmanager
from pathlib import Path
from pyccolo import TraceEvent, AstRewriter, fast, register_raw_handler
from pyccolo.emit_event import _TRACER_STACK
from runpy import run_module
from typing import Optional, Union, Tuple, List

import pi
from torch_mlir import ir

# this needs to be all of the dialects that will be used in the user scripts
# (in order to register ops)
# noinspection PyUnresolvedReferences
from torch_mlir.dialects import (
    func as func_dialect,
    scf,
    torch as torch_dialect,
    tensor,
    arith,
)
from torch_mlir.dialects._ods_common import get_op_result_or_value
from torch_mlir.ir import Type as MLIRType, IntegerType, F64Type

from pi.dialects import affine_, value_


# from uncompyle6 import main


def print_tb():
    for line in traceback.format_stack():
        print(line.strip())


MLIRStackFrame = namedtuple(
    "MLIRStackFrame",
    [
        "block",
        "block_args",
        "context",
        "insertion_point",
        "location",
        "scope_name",
    ],
)


def infer_mlir_type(py_val) -> MLIRType:
    if isinstance(py_val, int):
        # return IntegerType.get_signed(64)
        return IntegerType.get_signless(64)
    elif isinstance(py_val, float):
        return F64Type.get()
    else:
        raise Exception(f"unsupported val type {type(py_val)} {py_val}")


def make_constant(
    py_cst: Union[int, float, bool],
    index_type: bool = False,
    ip: ir.InsertionPoint = None,
):
    if index_type:
        constant = arith.ConstantOp.create_index(py_cst, ip=ip)
    else:
        constant = arith.ConstantOp(infer_mlir_type(py_cst), py_cst, ip=ip)

    return constant


class BreakFor(ast.NodeTransformer):
    def visit_For(self, node: ast.For) -> ast.For:
        for n in node.body:
            self.visit(n)
        if not isinstance(node.body[-1], ast.Break):
            break_ = ast.Break(
                lineno=node.body[-1].lineno + 1, col_offset=node.body[-1].col_offset
            )
            node.body.append(break_)
        return node


class ExplicitReturn(ast.NodeTransformer):
    def visit_Return(self, node: ast.Return) -> ast.Return:
        if node.value is None:
            node.value = ast.Constant(
                value=None,
                lineno=node.lineno,
                col_offset=node.col_offset + len("return "),
            )
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        for n in node.body:
            self.visit(n)
        if not isinstance(node.body[-1], ast.Return):
            return_ = ast.Return(
                value=None if node.name == "__init__" else ast.Constant(value=None),
                lineno=node.body[-1].lineno + 1,
                col_offset=node.col_offset + len("return "),
            )
            node.body.append(return_)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        for n in node.body:
            self.visit(n)
        return node


class BreakIf(ast.NodeTransformer):
    def visit_If(self, node: ast.If):
        # self.generic_visit(node)

        if not node.orelse:
            return node

        tests = []
        this_if = node
        while isinstance(this_if.orelse[0], ast.If):
            tests.append(this_if.test)
            this_if = this_if.orelse[0]
        tests.append(this_if.test)
        test_all = (
            ast.parse(f'not ({" and ".join([ast.unparse(t) for t in tests])})')
            .body[0]
            .value
        )
        this_if.orelse = [ast.If(test=test_all, body=this_if.orelse, orelse=[])]
        return node


class MLIRRewriter(AstRewriter):
    def __init__(self, tracers, **kwargs):
        self.mlir_tracer = next(t for t in tracers if isinstance(t, MLIRTracer))
        super(MLIRRewriter, self).__init__(tracers, **kwargs)

    def visit(self, node: ast.AST):
        mod_node = BreakFor().visit(node)
        mod_node = ExplicitReturn().visit(mod_node)
        # mod_node = BreakIf().visit(mod_node)
        with open("debug_mod_node_before_rewrite.py", "w") as f:
            f.write(ast.unparse(mod_node))
        mod_node = super(MLIRRewriter, self).visit(mod_node)
        with open("debug_mod_node_after_rewrite.py", "w") as f:
            f.write(ast.unparse(mod_node))
        return mod_node


def update_frame_locals(frame, updates: dict):
    # TODO(max): jot this down somewhere
    frame.f_locals.update(updates)
    # https://stackoverflow.com/a/46434344
    # passing c_int(0) will only update
    # passing c_int(1) will remove and update variables as well
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))


class MLIRTracer(pyc.BaseTracer):
    def __init__(
        self,
        script_path: str,
        mlir_context: ir.Context,
        mlir_location: ir.Location,
        mlir_module: ir.Module,
    ):
        super().__init__()
        self.script_path = script_path
        self.mlir_context = mlir_context
        self.mlir_location = mlir_location
        self.mlir_module = mlir_module
        self.local_defs = {}

        f64_type = ir.F64Type.get(context=self.mlir_context)
        self.type_mapping = {
            "float": f64_type,
            # "float_memref": ir.MemRefType.get((-1,), f64_type, loc=self.mlir_location),
            "int": ir.IntegerType.get_signed(64, context=self.mlir_context),
            "i32": ir.IntegerType.get_signless(32, context=self.mlir_context),
            "uint": ir.IntegerType.get_unsigned(64, context=self.mlir_context),
            "bool": ir.IntegerType.get_signless(1, context=self.mlir_context),
        }
        self.pyfunc_to_mlir_func_op = {}

        self.scf_fors = False
        self._mlir_block_stack: List[MLIRStackFrame] = []
        self.enter_mlir_block_scope(
            self.mlir_module.body,
            mlir_context=self.mlir_context or self.mlir_context,
            scope_name="module",
        )
        # dirty dirty hack
        self.if_bodies_executed = set()
        self.binops_executed = {}
        self.fn_to_node = {}

    def enter_mlir_block_scope(
        self,
        block: ir.Block,
        *,
        block_args: Tuple[str] = None,
        mlir_context: Optional[ir.Context] = None,
        insertion_point: Optional[ir.InsertionPoint] = None,
        mlir_location: Optional[ir.Location] = None,
        # for debugging
        scope_name: Optional[str] = None,
    ):
        if mlir_context is None:
            # one past the last op but still inside the block
            mlir_context = ir.Context()
        if insertion_point is None:
            # one past the last op but still inside the block
            insertion_point = ir.InsertionPoint(block)
        if mlir_location is None:
            mlir_location = self.mlir_location
        if block_args is None:
            block_args = ()
        if scope_name is None:
            scope_name = "UNKNOWN_SCOPE"

        mlir_context.__enter__()
        insertion_point.__enter__()
        mlir_location.__enter__()
        self._mlir_block_stack.append(
            MLIRStackFrame(
                block,
                block_args,
                mlir_context,
                insertion_point,
                mlir_location,
                scope_name,
            )
        )
        return block, mlir_location

    def exit_mlir_block_scope(self, scope_name=None):
        if scope_name is None:
            scope_name = "UNKNOWN_SCOPE"
        mlir_stack_frame = self._mlir_block_stack.pop()
        assert (
            mlir_stack_frame.scope_name == scope_name
        ), f"enter and exit in two different scopes {mlir_stack_frame.scope_name} {scope_name}"
        mlir_stack_frame.location.__exit__(None, None, None)
        mlir_stack_frame.insertion_point.__exit__(None, None, None)
        mlir_stack_frame.context.__exit__(None, None, None)
        return mlir_stack_frame.block

    @contextmanager
    def mlir_block_scope(
        self,
        block: ir.Block,
        *,
        block_args: Tuple[str] = None,
        insertion_point: Optional[ir.InsertionPoint] = None,
        mlir_location: Optional[ir.Location] = None,
        scope_name: Optional[str] = None,
    ):
        if scope_name is None:
            scope_name = "UNKNOWN_SCOPE"
        yield self.enter_mlir_block_scope(
            block,
            block_args=block_args,
            insertion_point=insertion_point,
            mlir_location=mlir_location,
            scope_name=scope_name,
        )
        self.exit_mlir_block_scope(scope_name=scope_name)

    def mlir_peek_block_scope(self):
        assert len(self._mlir_block_stack), "no block scope yet"
        return self._mlir_block_stack[-1]

    def get_or_make_mlir_constant(
        self,
        py_cst: Union[int, float, bool],
        name: Optional[str] = None,
        index_type: bool = False,
    ):
        assert isinstance(py_cst, (int, float, bool))
        if name is None:
            name = str(py_cst)
        if (name, index_type) not in self.local_defs:
            self.local_defs[name, index_type] = make_constant(
                py_cst,
                index_type,
                # TODO(max): check some stuff here (i.e. whether you're inside a func)
                ip=ir.InsertionPoint.at_block_begin(self.func_op_entry_block),
            )
        return self.local_defs[name, index_type]

    def should_propagate_handler_exception(
        self, evt: TraceEvent, exc: Exception
    ) -> bool:
        return True

    def should_instrument_file(self, filename: str) -> bool:
        return filename == self.script_path

    # handlers

    @TraceEvent.before_class_body
    def handle_before_class_body(
        self,
        old_ret,
        node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **_,
    ):
        return False

    @pyc.before_function_body
    def handle_before_function_body(
        self,
        old_ret,
        node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **_,
    ):
        # func = types.FunctionType(
        #     frame.f_code,
        #     frame.f_globals,
        #     frame.f_code.co_name,
        #     (), # defaults
        #     closure = tuple(types.CellType(None) for _ in range(len(frame.f_code.co_freevars)))
        # )

        if frame.f_code.co_name == "__init__":
            return False
        full_arg_spec = inspect.getfullargspec(
            frame.f_back.f_locals[frame.f_code.co_name]
        )
        arg_types = []
        for a in node.args.args:
            if isinstance(a.annotation, ast.Name):
                arg_types.append(self.type_mapping[a.annotation.id])
            elif isinstance(a.annotation, ast.Constant):
                arg_types.append(self.type_mapping[a.annotation.value])
            else:
                raise Exception(f"unknown annotation type {a.annotation}")
        arg_types = tuple(arg_types)
        func_op = func_dialect.FuncOp(
            name=frame.f_code.co_name,
            type=(
                arg_types,
                (),
            ),
            visibility="private",
            loc=self.mlir_location,
        )
        self.pyfunc_to_mlir_func_op[frame.f_code.co_name] = func_op
        # TODO(max): don't do this hacky stuff
        self.func_op_entry_block = func_op.add_entry_block()

        update_frame_locals(
            frame, {full_arg_spec.args[i]: a for i, a in enumerate(func_op.arguments)}
        )

        self.enter_mlir_block_scope(
            # TODO(max): don't do this hacky stuff
            self.func_op_entry_block,
            scope_name=frame.f_code.co_name,
        )

    @pyc.after_return
    def handle_after_return(
        self,
        old_ret,
        node_id_or_node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **_,
    ):
        assert frame.f_code.co_name in self.pyfunc_to_mlir_func_op

        func_op = self.pyfunc_to_mlir_func_op[frame.f_code.co_name]
        func_type = func_op.type
        if old_ret is not None:
            mlir_return_val = get_op_result_or_value(old_ret)
            canonical_func_type = ir.FunctionType.get(
                inputs=func_type.inputs, results=[mlir_return_val.type]
            )
            func_op.attributes["function_type"] = ir.TypeAttr.get(canonical_func_type)
            if isinstance(mlir_return_val, pi.Tensor):
                mlir_return_val = mlir_return_val.value
            func_dialect.ReturnOp((mlir_return_val,))
        else:
            func_dialect.ReturnOp(())
        self.exit_mlir_block_scope(scope_name=frame.f_code.co_name)

        return old_ret

    @pyc.exit_module
    def handle_exit_module(
        self,
        old_ret,
        node_id_or_node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        for loc_name, loc in frame.f_locals.items():
            if inspect.isfunction(loc) and getattr(loc, PI_EXPORT_ATTR_NAME, False):
                print(loc)
        self.exit_mlir_block_scope(scope_name="module")

    ast_rewriter_cls = MLIRRewriter

    def make_ast_rewriter(self, **kwargs) -> AstRewriter:
        return self.ast_rewriter_cls(_TRACER_STACK, **kwargs)

    @pyc.before_for_loop_body(reentrant=True)
    def handle_before_for_loop_body(
        self,
        old_ret,
        node: ast.For,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        range_iter = eval(ast.unparse(node.iter), frame.f_globals, frame.f_locals)
        if not isinstance(range_iter, range):
            raise RuntimeError("Only `range` iterator currently supported")
        for arg in [range_iter.start, range_iter.stop, range_iter.step]:
            assert isinstance(arg, int), f"symbolic range not supported yet {arg}"

        for arg in [range_iter.start, range_iter.stop, range_iter.step]:
            assert isinstance(arg, int), f"symbolic range not supported yet {arg}"

        start, stop, step = range_iter.start, range_iter.stop, range_iter.step
        if self.scf_fors:
            start, stop, step = [
                self.get_or_make_mlir_constant(c, index_type=self.scf_fors)
                for c in [start, stop, step]
            ]
            loop = scf.ForOp(start, stop, step, [], loc=self.mlir_location)
        else:
            loop = affine_.AffineForOp(start, stop, step, loc=self.mlir_location)

        assert isinstance(
            node.target, ast.Name
        ), f"structured for induction vars not supported {node.target}"
        induction_var_name = node.target.id
        update_frame_locals(frame, {induction_var_name: loop.induction_variable})
        self.enter_mlir_block_scope(loop.body)

    #
    @pyc.after_for_loop_iter(reentrant=True)
    def handle_after_for_loop_iter(
        self,
        old_ret,
        node_id_or_node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        # TODO(max): handle yielded vals
        yielded_vals = []
        if self.scf_fors:
            scf.YieldOp(yielded_vals, loc=self.mlir_location)
        else:
            affine_.AffineYieldOp(yielded_vals, loc=self.mlir_location)

        self.exit_mlir_block_scope()

    @pyc.before_compare(reentrant=True)
    @pyc.before_binop(reentrant=True)
    def handle_before_binop(
        self,
        old_ret,
        node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        def eval_op(x, y):
            hash = id(x), id(y)
            if hash not in self.binops_executed:
                x, y = map(
                    lambda v: self.get_or_make_mlir_constant(v)
                    if isinstance(v, (float, int, bool))
                    else v,
                    (x, y),
                )
                if isinstance(node, ast.Compare):
                    op = node.ops[0].__class__.__name__.lower()
                else:
                    # ...god damn it
                    op = node.op.__class__.__name__.lower().replace("mult", "mul")
                if isinstance(x, pi.Tensor):
                    assert isinstance(y, pi.Tensor)
                    self.binops_executed[hash] = getattr(operator, op)(x, y)
                else:
                    self.binops_executed[hash] = getattr(value_, op)(x, y)
            return self.binops_executed[hash]

        return eval_op

    # @pyc.before_subscript_slice
    # @pyc.before_subscript_del
    @pyc.before_subscript_load(reentrant=True)
    @pyc.before_subscript_store(reentrant=True)
    def handle_subscr(
        self,
        old_ret,
        node,
        frame: types.FrameType,
        event,
        guard_for_spec,
        **kwargs,
    ):
        this = self

        class _dummy:
            def __getitem__(self, indices):
                target = frame.f_locals[node.value.id]
                indices = tuple(
                    map(
                        lambda v: this.get_or_make_mlir_constant(v, index_type=True)
                        if isinstance(v, (float, int, bool))
                        else v,
                        indices,
                    )
                )
                return target[indices]

            def __setitem__(self, indices, value):
                target = frame.f_locals[node.value.id]
                # TODO(max): handle value
                indices = tuple(
                    map(
                        lambda v: this.get_or_make_mlir_constant(v, index_type=True)
                        if isinstance(v, (float, int, bool))
                        else v,
                        indices,
                    )
                )
                target[indices] = value

        return _dummy()

    @pyc.after_if_test(reentrant=True)
    def handle_if(self, cond, node: ast.If, frame, event, guard_for_spec, **kwargs):
        # TODO(max): handle affine if
        has_orelse = len(node.orelse)
        body_copy = fast.copy_ast(ast.Module(body=node.body, type_ignores=[]))
        body_copy_str = ast.unparse(body_copy)
        if has_orelse:
            orelse_body_copy = fast.copy_ast(
                ast.Module(body=node.orelse, type_ignores=[])
            )
            orelse_body_copy_str = ast.unparse(orelse_body_copy)

        # dirty dirty hack but oh well
        if body_copy_str not in self.if_bodies_executed:
            self.if_bodies_executed.add(body_copy_str)
            if_ = scf.IfOp(
                cond.result, [], hasElse=len(node.orelse), loc=self.mlir_location
            )
            with self.mlir_block_scope(if_.then_block):
                pyc.exec(body_copy, frame.f_globals, frame.f_locals)
                yielded_vals = []
                scf.YieldOp(yielded_vals, loc=self.mlir_location)

            if has_orelse and orelse_body_copy_str not in self.if_bodies_executed:
                self.if_bodies_executed.add(orelse_body_copy_str)
                with self.mlir_block_scope(if_.else_block):
                    pyc.exec(orelse_body_copy, frame.f_globals, frame.f_locals)
                    yielded_vals = []
                    scf.YieldOp(yielded_vals, loc=self.mlir_location)

        # need this so the bodies don't actually executed
        return False


def get_script_as_module(script: str) -> str:
    # ref: https://nvbn.github.io/2016/08/17/ast-import/
    script_path = Path(script)
    script_dir = script_path.parent.as_posix()
    module_name = os.path.splitext(script_path.name)[0]
    sys.path.insert(0, script_dir)
    return module_name


def trace(script_path, mlir_context, mlir_location, mlir_module) -> ir.Module:
    module_to_run = get_script_as_module(script_path)
    with MLIRTracer(
        script_path,
        mlir_context,
        mlir_location,
        mlir_module,
    ):
        run_module(module_to_run)

    return mlir_module


# TODO(max): jot this down somewhere

# def dis_code(code):
#     return dis(
#         data=code.co_code,
#         python_version=(3, 10),
#         co_varnames=code.co_varnames,
#         co_names=code.co_names,
#         co_consts=code.co_consts,
#         co_cellvars=code.co_cellvars,
#         co_freevars=code.co_freevars,
#         co_lnotab=code.co_lnotab,
#         co_firstlineno=code.co_firstlineno,
#     )
# @pyc.opcode
# def handle_opcode(
#     self,
#     old_ret,
#     _node_id_or_node,
#     frame: types.FrameType,
#     event,
#     _guard_for_spec,
#     **kwargs,
# ):
#     # print_tb()
#     discodes = dis_code(frame.f_code)
#     byte_index = frame.f_lasti
#     instr = frame.f_code.co_code[byte_index]
#     dis_inst = discodes[byte_index]
#     arg = frame.f_code.co_code[byte_index + 1]
#     print(f"{instr=} {dis_inst} {byte_index=} {arg=}")
#     # Bytecode.from_code(frame.f_code)
#     # print(f"{old_ret=} {event.to_ast()=} {kwargs=}")
#     exit(0)

# @pyc.before_function_body
# def handle_before_function_body(self, old_ret, node_id_or_node, frame: types.FrameType, event, guard_for_spec, **kwargs):
#     # deparsed = main.decompile(None, frame.f_code, out=sys.stdout)
#     # val = frame_stack_read(frame, -1)
#     # cyber_frame = Frame(frame)
#     # bcodes = Bytecode.from_code(frame.f_code)
#     # discodes = dis_code(frame.f_code)
#     # byte_index = frame.f_lasti
#     # instr = frame.f_code.co_code[byte_index]
#     # dis_inst = discodes[byte_index]
#     # arg = frame.f_code.co_code[byte_index + 1]
#     # print(f"{instr=} {dis_inst} {byte_index=} {arg=}")
#
#
#     print(kwargs)
#     print(f"handle_before_function_body")
