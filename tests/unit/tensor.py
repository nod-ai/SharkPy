import pi

from torch_mlir.dialects import torch
from torch_mlir.ir import IntegerType
import numpy as np

from pi.mlir_utils import mlir_cm

if __name__ == "__main__":
    with mlir_cm() as module:
        z = pi.empty((1, 2, 3))
        t = torch.ConstantIntOp(1).result
        print(t.type)
        i = IntegerType.get_unsigned(32)
        print(i)
        print(module)
        vt = pi.from_numpy(np.random.rand(10, 10))
        z = vt + vt
        # pi.ops.aten.ScalarImplicit(pi.rand())
        z = pi.rand(1, 1)
        print(module)
        # sizes = parse_sizes_from_tensor_type_str(z)
        # print(sizes)
