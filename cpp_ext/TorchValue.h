//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "IRModule.h"
#include "TorchTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::python;

template<typename DerivedTy>
struct TorchValue : PyValue {
  TorchValue(PyOperationRef operationRef, MlirValue value)
      : PyValue(std::move(operationRef), value) {}

  static DerivedTy createFromCapsule_(const py::capsule &capsule) {
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
};

#define DEFINE_VALUE(TTT)                                     \
  struct Torch_##TTT : TorchValue<Torch_##TTT> {              \
    Torch_##TTT(PyOperationRef operationRef, MlirValue value) \
        : TorchValue(std::move(operationRef), value) {}       \
  };
TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_VALUE)
TORCH_MLIR_FORALL_CONTAINER_TYPES(DEFINE_VALUE)
TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_VALUE)
TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_VALUE)
#undef DEFINE_VALUE

#define DEFINE_VALUE(TTT)                                               \
  struct TorchListOfTorch##TTT : TorchValue<TorchListOfTorch##TTT> {    \
    TorchListOfTorch##TTT(PyOperationRef operationRef, MlirValue value) \
        : TorchValue(std::move(operationRef), value) {}                 \
  };
TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_VALUE)
TORCH_MLIR_FORALL_CONTAINER_TYPES(DEFINE_VALUE)
TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_VALUE)
#undef DEFINE_VALUE

#define DEFINE_VALUE(TTT)                                          \
  struct TorchListOf##TTT : TorchValue<TorchListOf##TTT> {         \
    TorchListOf##TTT(PyOperationRef operationRef, MlirValue value) \
        : TorchValue(std::move(operationRef), value) {}            \
  };
TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_VALUE)
#undef DEFINE_VALUE

void bindValues(py::module &m);