import re

skip = [
    "run_on_actual_value",
    "__subclasscheck__",
    "__instancecheck__",
    "_is_pi_tensor",
    "__class__",
    "type",
    "value",
    "__init__",
]

# other = [
#     "bitwise_and",
#     "bitwise_and_",
#     "bitwise_or",
#     "bitwise_or_",
#     "logical_and",
#     "logical_and_",
#     # "logical_not",
#     # "logical_not_",
#     "logical_or",
#     "logical_or_",
#     "logical_or",
#     "logical_or_",
#     "logical_xor",
#     "logical_xor_",
# ]

reg = re.compile(r"\s{4}def (\w*)\(")


cannot_find = set(l.strip() for l in open("skip_funs.txt").readlines())
skip_sigs = set(l.strip() for l in open("skip_sigs.txt").readlines())


# fn_match = re.comp(r"(?P<function>\w+)\s?\((?P<arg>(?P<args>\w+(,\s?)?)+)\)", s)
# fn_dict = fn_match.groupdict()
# del fn_dict['args']
# fn_dict['arg'] = [arg.strip() for arg in fn_dict['arg'].split(',')]

import ast


class c_re(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name in skip:
            return node
        else:
            kwonlyargs = {k.arg for k in node.args.kwonlyargs}
            arg_names = [a.arg for a in node.args.args if a not in kwonlyargs]
            if node.args.vararg:
                arg_names.append(f"*{node.args.vararg.arg}")
            if node.args.kwonlyargs:
                for kwarg in node.args.kwonlyargs:
                    arg_names.append(f"{kwarg.arg}")
            # if node.name in other:
            #     arg_names.append("other")

            if node.args.args[0].arg == "self":
                node.args.args[0].annotation = ast.Name("Tensor")
            body = []
            if "out" in arg_names:
                body.append(
                    ast.parse(f"if out is not None: raise NotImplementedError('{node.name}.out variant')")
                )
                arg_names.remove("out")
            ret = f"return pi.{node.name}({', '.join(arg_names)})"
            sig = ast.unparse(node).replace("@overload", "").strip().split(" ", 1)[1].split("->")[0].strip()

            if node.name in cannot_find or sig in skip_sigs or "Union[str, ellipsis, None]" in ast.unparse(node):
                print(sig)
                body.append(
                    # ast.parse(f'"""{ret}"""'),
                    ast.parse(f"raise NotImplementedError('{node.name}')")
                )
            else:
                body.append(
                    ast.parse(f"{ret}")
                )

            node.body = body
                
        return node



class re(ast.NodeTransformer):

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if node.name == "Tensor":
            node = c_re().visit(node)
        return node

new_lines = []
with open("tensorbase.pyi") as f:
    tree = ast.parse(f.read())

rewr = re()

rewr.visit(tree)


    # lines = f.readlines()
    # for l in lines:
    #     if "NotImplementedError" not in l:
    #         new_lines.append(l.rstrip())
    #     if match := reg.findall(l):
    #         assert len(match) == 1, (match, len((match)))
    #         match = match[0]
    #         if match in skip: continue
    #         args = l.split("(")[1].strip()[:-2].split(",")
    #         for i, a in enumerate(args):
    #             a = a.split("=")[0]
    #             args[i] = a.strip()
    #         args = ', '.join(args)
    #         print(args)
    #         new_lines.append(f"        return pi.{match}({args})")


with open("_tensor.py", "w") as f:
    f.write(ast.unparse(tree))
