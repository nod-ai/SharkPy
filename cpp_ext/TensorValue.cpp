//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "TensorValue.h"
#include "TorchTypesCAPI.h"

#include "mlir/CAPI/Support.h"

using namespace mlir;
using namespace mlir::python;

Torch_Tensor Torch_Tensor::createFromCapsule_(const py::capsule &capsule) {
  MlirValue value = {capsule.get_pointer()};
  if (mlirValueIsNull(value))
    throw py::error_already_set();
  MlirOperation owner;
  if (mlirValueIsAOpResult(value))
    owner = mlirOpResultGetOwner(value);
  if (mlirValueIsABlockArgument(value))
    owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(value));
  if (mlirOperationIsNull(owner))
    throw py::error_already_set();

  MlirContext ctx = mlirOperationGetContext(owner);
  auto *unownedContextWrapper = new PyMlirContext(ctx);
  auto pyCtxRef = py::reinterpret_steal<py::object>(mlirPythonContextToCapsule(ctx));
  assert(pyCtxRef && "cast to py::object failed");
  auto ctxRef = PyMlirContextRef(unownedContextWrapper, std::move(pyCtxRef));

  auto pyOpRef = py::reinterpret_steal<py::object>(mlirPythonOperationToCapsule(owner));
  auto *unownedOperation =
      new PyOperation(std::move(ctxRef), owner);
  unownedOperation->handle = pyOpRef;
  auto ownerRef = PyOperationRef(unownedOperation, std::move(pyOpRef));

  return {ownerRef, value};
}

PYBIND11_NOINLINE bool try_load_foreign_module_local(py::handle src) {
  constexpr auto *local_key = PYBIND11_MODULE_LOCAL_ID;
  const auto pytype = py::type::handle_of(src);
  if (!hasattr(pytype, local_key)) {
    std::cerr << "wrong local key\n";
    return false;
  }

  py::detail::type_info *foreign_typeinfo = py::reinterpret_borrow<py::capsule>(getattr(pytype, local_key));
  assert(foreign_typeinfo != nullptr);
  if (foreign_typeinfo->module_local_load == &pybind11::detail::type_caster_generic::local_load) {
    std::cerr << "wrong module loader\n";
    return false;
  }

  //  auto caster = pybind11::detail::type_caster_generic(foreign_typeinfo);
  //  if (caster.load(src, false)) {
  //    return caster.value;
  //  } else {
  //    std::cerr << "caster.load failed";
  //    return false;
  //  }

  if (auto *result = foreign_typeinfo->module_local_load(src.ptr(), foreign_typeinfo)) {
    return true;
  }
  std::cerr << "load failed\n";
  return false;
}

void bindValues(py::module &m) {
  py::object op_result_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("OpResult");
  py::object value_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("Value");

  m.def("_load_foreign", [](const py::object &mlirvalue) {
    py::handle value_handle = mlirvalue;
    auto loaded = try_load_foreign_module_local(value_handle);
    return loaded;
  });

  py::class_<Torch_Tensor>(m, "_Torch_Tensor", value_)
      .def(py::init<>([](const py::capsule &capsule) {
        return Torch_Tensor::createFromCapsule_(capsule);
      }));
}