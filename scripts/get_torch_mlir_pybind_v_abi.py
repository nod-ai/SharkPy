"""
the python bindings in mlir are py::module_local

https://github.com/llvm/llvm-project/blob/817f64e7ce545a2d0ec4484b4066cb73ddb31fdd/mlir/lib/Bindings/Python/IRCore.cpp#L243

that means they can't be extended in python (or something like that). despite this pybind does stash relevant type info
in an attribute of the binding classes

https://github.com/pybind/pybind11/blob/a34596bfe1947b4a6b0bcc4218e1f72d0c2e9b4c/include/pybind11/pybind11.h#L1353

the name of the attribute is PYBIND11_MODULE_LOCAL_ID, which is a macro that expans to `<some stuff>__gxx_abi_version`.
so in order to subclass the mlir python binding classes we have to line exactly with the ABI of torch-mlir (as well
pybind version, stdlib, and compiler). on clang it doesn't matter (they hardcode __gxx_abi_version == 02 https://lists.llvm.org/pipermail/cfe-dev/2015-June/043561.html)
but on gcc you can indeed pass -fabi-version=X. this script pulls that info out from torch_mlir
"""

from torch_mlir import ir

pybind11_module_local_id = next(d for d in dir(ir.Value) if "pybind" in d)
assert pybind11_module_local_id.startswith("__pybind11_module_local_")
pybind11_module_local_id = pybind11_module_local_id.replace(
    "__pybind11_module_local_", ""
)
assert pybind11_module_local_id.endswith("__")
pybind11_module_local_id = pybind11_module_local_id[:-2]
pybind_internal_v, compiler, stdlib, cxxabi = pybind11_module_local_id.split("_")
assert cxxabi.startswith("cxxabi10")
cxxabi = cxxabi.replace("cxxabi10", "")
# print(pybind_internal_v, compiler, stdlib, int(cxxabi))
print(int(cxxabi))
