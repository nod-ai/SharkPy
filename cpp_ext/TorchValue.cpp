//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "TorchValue.h"
#include "TorchTypesCAPI.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Support.h"

using namespace mlir;
using namespace mlir::python;

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
  py::object value_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("Value");

  m.def("_load_foreign", [](const py::object &mlirvalue) {
    py::handle value_handle = mlirvalue;
    auto loaded = try_load_foreign_module_local(value_handle);
    return loaded;
  });

#define DEFINE_VALUE(TTT)                                                                           \
  py::class_<Torch_##TTT>(m, "_Torch_" #TTT, value_)                                                \
      .def(py::init<>([](const py::handle apiObject) {                                              \
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);                         \
        return Torch_##TTT::createFromCapsule_(capsule);                                            \
      }))                                                                                           \
      .def_property_readonly("type", [](PyValue &self) {                                            \
        return Torch_##TTT##Type(self.parentOperation->getContext(), mlirValueGetType(self.get())); \
      })                                                                                            \
      .def("__repr__", [](PyValue &self) {                                                          \
        PyPrintAccumulator printAccum;                                                              \
        printAccum.parts.append(#TTT "(");                                                          \
        mlirValuePrint(self.get(), printAccum.getCallback(), printAccum.getUserData());             \
        printAccum.parts.append(")");                                                               \
        return printAccum.join();                                                                   \
      })                                                                                            \
      .def("__str__", [](const py::handle apiObject) {                                              \
        return py::repr(apiObject);                                                                 \
      });

  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_VALUE)
  TORCH_MLIR_FORALL_CONTAINER_TYPES(DEFINE_VALUE)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_VALUE)
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_VALUE)
#undef DEFINE_VALUE

#define DEFINE_VALUE(TTT)                                                                   \
  py::class_<TorchListOfTorch##TTT>(m, "_TorchListOfTorch" #TTT, value_)                    \
      .def(py::init<>([](const py::handle apiObject) {                                      \
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);                 \
        return TorchListOfTorch##TTT::createFromCapsule_(capsule);                          \
      }))                                                                                   \
      .def_property_readonly("type", [](PyValue &self) {                                    \
        auto elType = torchMlirTorchListTypeGetContainedType(mlirValueGetType(self.get())); \
        auto listType = torchMlirTorchListTypeGet(elType);                                  \
        return Torch_ListType::createFromMlirType_(listType);                               \
      })                                                                                    \
      .def("__repr__", [](PyValue &self) {                                                  \
        PyPrintAccumulator printAccum;                                                      \
        printAccum.parts.append(#TTT "(");                                                  \
        mlirValuePrint(self.get(), printAccum.getCallback(), printAccum.getUserData());     \
        printAccum.parts.append(")");                                                       \
        return printAccum.join();                                                           \
      })                                                                                    \
      .def("__str__", [](const py::handle apiObject) {                                      \
        return py::repr(apiObject);                                                         \
      });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_VALUE)
  TORCH_MLIR_FORALL_CONTAINER_TYPES(DEFINE_VALUE)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_VALUE)
#undef DEFINE_VALUE

#define DEFINE_VALUE(TTT)                                                                   \
  py::class_<TorchListOf##TTT>(m, "_TorchListOf" #TTT, value_)                              \
      .def(py::init<>([](const py::handle apiObject) {                                      \
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);                 \
        return TorchListOf##TTT::createFromCapsule_(capsule);                               \
      }))                                                                                   \
      .def_property_readonly("type", [](PyValue &self) {                                    \
        auto elType = torchMlirTorchListTypeGetContainedType(mlirValueGetType(self.get())); \
        auto listType = torchMlirTorchListTypeGet(elType);                                  \
        return Torch_ListType::createFromMlirType_(listType);                               \
      })                                                                                    \
      .def("__repr__", [](PyValue &self) {                                                  \
        PyPrintAccumulator printAccum;                                                      \
        printAccum.parts.append(#TTT "(");                                                  \
        mlirValuePrint(self.get(), printAccum.getCallback(), printAccum.getUserData());     \
        printAccum.parts.append(")");                                                       \
        return printAccum.join();                                                           \
      })                                                                                    \
      .def("__str__", [](const py::handle apiObject) {                                      \
        return py::repr(apiObject);                                                         \
      });

  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_VALUE)
#undef DEFINE_VALUE
}