#include "IRModule.h"
#include "TensorValue.h"
#include "TorchTypes.h"
#include "TorchTypesCAPI.h"
#include "dylib.hpp"
#include "mlir-c/Bindings/Python/Interop.h"

#include <iostream>

namespace py = pybind11;
using namespace mlir::python;

// no clue why but without this i get a missing symbol error
namespace llvm {
int DisableABIBreakingChecks = 1;
int EnableABIBreakingChecks = 0;
}// namespace llvm

PYBIND11_MODULE(_mlir, m) {
  dylib lib2("TorchMLIRAggregateCAPI");

  if (!lib2.has_symbol("mlirValueIsAOpResult")) {
    throw std::runtime_error("symbol 'mlirValueIsAOpResult' not found in 'TorchMLIRAggregateCAPI' lib");
  }

  m.def("get_pybind11_module_local_id", []() { return PYBIND11_MODULE_LOCAL_ID; });
//  m.def("bind_all", [&m]() {
//  });
  bindValues(m);
  bindTypes(m);
  bindTypeHelpers(m);
}
