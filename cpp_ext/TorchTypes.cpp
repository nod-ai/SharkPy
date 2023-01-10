//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#include "IRModule.h"

#include "TorchTypes.h"
#include "TorchTypesCAPI.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace mlir;
using namespace mlir::python;

#include <iostream>

bool hasEnding(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

#define IMPL_STRUCT(TTT)                                                                                                               \
  Torch_##TTT##Type Torch_##TTT##Type::createFromHandles(const py::handle optionalSizesHandle, const py::handle optionalDtypeHandle) { \
    auto src = py::module::import("torch_mlir.ir")                                                                                     \
                   .attr("Context")                                                                                                    \
                   .attr("current");                                                                                                   \
    py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                                                                              \
    MlirContext mlirContext = {ctxCapsule.get_pointer()};                                                                              \
                                                                                                                                       \
    int64_t numSizes = -1;                                                                                                             \
    std::vector<int64_t> optionalSizes;                                                                                                \
    if (!optionalSizesHandle.is(py::none())) {                                                                                         \
      optionalSizes = py::cast<std::vector<int64_t>>(optionalSizesHandle);                                                             \
      numSizes = optionalSizes.size();                                                                                                 \
    }                                                                                                                                  \
                                                                                                                                       \
    MlirType optionalDtype;                                                                                                            \
    if (!optionalDtypeHandle.is(py::none())) {                                                                                         \
      py::capsule optionalDtypeCapsule = mlirApiObjectToCapsule(optionalDtypeHandle);                                                  \
      optionalDtype = {optionalDtypeCapsule.get_pointer()};                                                                            \
    } else {                                                                                                                           \
      optionalDtype = {nullptr};                                                                                                       \
    }                                                                                                                                  \
                                                                                                                                       \
    auto mlirTensorType = torchMlirTorch##TTT##TypeGet(mlirContext, numSizes, optionalSizes.data(), optionalDtype);                    \
    return createFromMlirType_(mlirTensorType);                                                                                        \
  };
TORCH_MLIR_FORALL_TENSOR_TYPES(IMPL_STRUCT)
#undef IMPL_STRUCT

void bindTypes(py::module &m) {
  py::object type_ =
      (py::object) py::module_::import("torch_mlir.ir").attr("Type");

  py::object location =
      (py::object) py::module_::import("torch_mlir.ir").attr("Location");

#define DEFINE_PYBIND(TTT) py::class_<Torch_##TTT##Type>(m, "_Torch_" #TTT "Type", type_)                 \
                               .def(py::init<>([]() {                                                     \
                                 auto src = py::module::import("torch_mlir.ir")                           \
                                                .attr("Context")                                          \
                                                .attr("current");                                         \
                                 py::capsule ctxCapsule = mlirApiObjectToCapsule(src);                    \
                                 MlirContext mlirContext = {ctxCapsule.get_pointer()};                    \
                                 auto torchType = torchMlirTorch##TTT##TypeGet(mlirContext);              \
                                 return Torch_##TTT##Type::createFromMlirType_(torchType);                \
                               }))                                                                        \
                               .def(py::init<>([](const py::handle apiObject) {                           \
                                 auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);      \
                                 return Torch_##TTT##Type::createFromCapsule_(capsule);                   \
                               }))                                                                        \
                               .def("__repr__", [](PyType &self) {                                        \
                                 PyPrintAccumulator printAccum;                                           \
                                 printAccum.parts.append(#TTT "(");                                       \
                                 mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData()); \
                                 printAccum.parts.append(")");                                            \
                                 return printAccum.join();                                                \
                               })                                                                         \
                               .def("__str__", [](const py::handle apiObject) {                           \
                                 return py::repr(apiObject);                                              \
                               });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                                             \
  py::class_<Torch_##TTT##Type>(m, "_Torch_" #TTT "Type", type_)                                       \
      .def(py::init<>([](const py::handle optionalSizesHandle, const py::handle optionalDtypeHandle) { \
             return Torch_##TTT##Type::createFromHandles(optionalSizesHandle, optionalDtypeHandle);    \
           }),                                                                                         \
           py::arg("sizes") = py::none(), py::arg("dtype") = py::none())                               \
      .def(py::init<>([](const py::handle apiObject) {                           \
        auto capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);      \
        return Torch_##TTT##Type::createFromCapsule_(capsule); }))                                             \
      .def_property_readonly("sizes", [](const py::handle &self) {                                     \
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(self);                          \
        MlirType rawType = {capsule.get_pointer()};                                                    \
        auto rank = torchMlirTorch##TTT##TypeGetRank(rawType);                                         \
        const int64_t *sizes_ptr = torchMlirTorch##TTT##TypeGetSizes(rawType);                         \
        std::vector<int64_t> sizes(sizes_ptr, sizes_ptr + rank);                                       \
        return sizes;                                                                                  \
      })                                                                                               \
      .def_property_readonly("dtype", [](const py::handle &self) {                                     \
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(self);                          \
        MlirType rawType = {capsule.get_pointer()};                                                    \
                                                                                                       \
        return torchMlirTorch##TTT##TypeGetDtype(rawType);                                       \
      })                                                                                               \
      .def("__repr__", [](PyType &self) {                                                              \
        PyPrintAccumulator printAccum;                                                                 \
        mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());                       \
        printAccum.parts.append(")");                                                                  \
        return printAccum.join();                                                                      \
      });

  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT)                                                                                            \
  m.def("_TorchListOf" #TTT "Type", ([](const py::handle optionalSizesHandle, const py::handle optionalDtypeHandle) { \
          auto tensorType = Torch_##TTT##Type::createFromHandles(optionalSizesHandle, optionalDtypeHandle);           \
          auto listType = torchMlirTorchListTypeGet(tensorType);                                                      \
          return Torch_ListType::createFromMlirType_(listType);                                                       \
        }),                                                                                                           \
        py::arg("sizes") = py::none(), py::arg("dtype") = py::none());

  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND
}

void bindTypeHelpers(py::module &m) {

#define DEFINE_PYBIND(TTT)                                       \
  m.def("_TorchListOfTorch" #TTT "Type", []() {                  \
    auto src = py::module::import("torch_mlir.ir")               \
                   .attr("Context")                              \
                   .attr("current");                             \
    py::capsule ctxCapsule = mlirApiObjectToCapsule(src);        \
    MlirContext mlirContext = {ctxCapsule.get_pointer()};        \
    MlirType elType = torchMlirTorch##TTT##TypeGet(mlirContext); \
    auto listType = torchMlirTorchListTypeGet(elType);           \
    return Torch_ListType::createFromMlirType_(listType);        \
  });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT) m.def(                                                \
    "is_a_Torch_" #TTT "Type", [](const py::handle apiObject) {                  \
      py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject); \
      MlirType rawType = {capsule.get_pointer()};                                \
      return torchMlirTypeIsATorch##TTT(rawType);                                \
    });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_CONTAINER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

#define DEFINE_PYBIND(TTT) m.def(                                                                                                         \
    ("is_a_TorchListOf" + std::string(hasEnding(#TTT, "Tensor") ? "" : "Torch") + #TTT "Type").c_str(), [](const py::handle typeHandle) { \
      py::capsule typeCapsule = pybind11::detail::mlirApiObjectToCapsule(typeHandle);                                                     \
      MlirType type = {typeCapsule.get_pointer()};                                                                                        \
      if (torchMlirTypeIsATorchList(type)) {                                                                                              \
        auto elType = torchMlirTorchListTypeGetContainedType(type);                                                                       \
        return torchMlirTypeIsATorch##TTT(elType);                                                                                        \
      }                                                                                                                                   \
      return false;                                                                                                                       \
    });
  TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_PYBIND)
  TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_PYBIND)
#undef DEFINE_PYBIND

  m.def(
      "is_a_TorchScalarType", [](const py::handle apiObject) {
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);
        MlirType rawType = {capsule.get_pointer()};
        auto isScalar = (
#define OR(TTT) torchMlirTypeIsATorch##TTT(rawType) ||
            TORCH_MLIR_FORALL_NUMBER_TYPES(OR) false
#undef OR
        );
        return isScalar;
      });
  m.def(
      "is_a_TorchType", [](const py::handle apiObject) {
        py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);
        MlirType rawType = {capsule.get_pointer()};
        auto isScalar = (
#define OR(TTT) torchMlirTypeIsATorch##TTT(rawType) ||
            TORCH_MLIR_FORALL_NUMBER_TYPES(OR)
                TORCH_MLIR_FORALL_CONTAINER_TYPES(OR)
                    TORCH_MLIR_FORALL_TENSOR_TYPES(OR)
                        TORCH_MLIR_FORALL_OTHER_TYPES(OR) false
#undef OR
        );
        return isScalar;
      });

  m.def("is_dtype", [](const py::handle apiObject) {
    py::capsule capsule = pybind11::detail::mlirApiObjectToCapsule(apiObject);
    MlirType type = {capsule.get_pointer()};
    return mlirTypeIsAInteger(type) || mlirTypeIsABF16(type) || mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type);
  });
}
