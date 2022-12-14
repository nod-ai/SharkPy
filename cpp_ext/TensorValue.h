//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "IRModule.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::python;


struct Torch_Tensor : PyValue {
  Torch_Tensor(PyOperationRef operationRef, MlirValue value)
      : PyValue(std::move(operationRef), value) {}

  static Torch_Tensor createFromCapsule_(const py::capsule& capsule);
};

void bindValues(py::module &m);