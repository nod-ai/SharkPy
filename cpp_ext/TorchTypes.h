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
#include "TorchTypesCAPI.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::python;

// Note: TorchScript does not consider !torch.bool to be a Scalar.
#define TORCH_MLIR_FORALL_NUMBER_TYPES(_) \
  _(Float)                                \
  _(Int)                                  \
  _(Number)                               \
  _(QInt8)                                \
  _(QUInt8)

#define TORCH_MLIR_FORALL_TENSOR_TYPES(_) \
  _(NonValueTensor)                       \
  _(ValueTensor)

#define TORCH_MLIR_FORALL_CONTAINER_TYPES(_) \
  _(Dict)                                    \
  _(List)                                    \
  _(Tuple)

#define TORCH_MLIR_FORALL_OTHER_TYPES(_) \
  _(Any)                                 \
  _(Bool)                                \
  _(Device)                              \
  _(Generator)                           \
  _(LinearParams)                        \
  _(None)                                \
  _(String)
//  _(Optional)                            \
//  _(NnModule)                            \
//  _(Union)

#define DEFINE_STRUCT(TTT)                                                                                                      \
  struct Torch_##TTT##Type : public PyConcreteType<Torch_##TTT##Type> {                                                         \
    Torch_##TTT##Type(PyMlirContextRef contextRef, MlirType t) : PyConcreteType<Torch_##TTT##Type>(std::move(contextRef), t) {} \
  };
TORCH_MLIR_FORALL_NUMBER_TYPES(DEFINE_STRUCT)
TORCH_MLIR_FORALL_CONTAINER_TYPES(DEFINE_STRUCT)
TORCH_MLIR_FORALL_OTHER_TYPES(DEFINE_STRUCT)
#undef DEFINE_STRUCT

#define DEFINE_STRUCT(TTT)                                                                                                      \
  struct Torch_##TTT##Type : public PyConcreteType<Torch_##TTT##Type> {                                                         \
    Torch_##TTT##Type(PyMlirContextRef contextRef, MlirType t) : PyConcreteType<Torch_##TTT##Type>(std::move(contextRef), t) {} \
    static Torch_##TTT##Type createFromHandles(const py::handle optionalSizesHandle, const py::handle optionalDtypeHandle);                              \
  };
TORCH_MLIR_FORALL_TENSOR_TYPES(DEFINE_STRUCT)
#undef DEFINE_STRUCT

void bindTypes(py::module &m);
void bindTypeHelpers(py::module &typeHandle);
