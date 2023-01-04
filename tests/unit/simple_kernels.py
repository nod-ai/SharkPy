from torch_mlir.dialects import memref, linalg, torch, arith, tensor
from torch_mlir.ir import Attribute, DenseFPElementsAttr, DenseElementsAttr
import numpy as np


def saxpy(a: float, b: float):
    A = memref.AllocaOp((10, 30))
    B = memref.AllocaOp((30, 20))
    C = memref.AllocaOp((10, 20))
    for i in range(10):
        for j in range(30):
            for k in range(20):
                C[i, k] += A[i, j] * B[j, k] * a + b
    return C


saxpy(1, 2)


def conditionals(a: float, b: float, c: float):
    A = memref.AllocaOp((1, 3))
    for i in range(10):
        if a > 3:
            A[1, 1] = a * i
        elif a > 4:
            A[1, 2] = b * i
        # else:
        #     A[1, 3] = c

    return A


conditionals(10, 2, 3)


def linalg_ops(min: float, max: float, seed: "i32"):
    A = memref.AllocaOp((10, 30))
    linalg.fill_rng_2d(min, max, seed, outs=A)
    B = memref.AllocaOp((30, 20))
    C = memref.AllocaOp((10, 20))
    linalg.matmul(A, B, outs=C)
    K = memref.AllocaOp((3, 3))
    output = memref.AllocaOp((7, 17))
    for i in range(10):
        linalg.conv_2d(C, K, outs=output)

    return output


linalg_ops(0, 1, 42)


def torch_ops():
    z = torch.ConstantFloatOp(value=256.0)
    attr = DenseFPElementsAttr(Attribute.parse("dense<0.0> : tensor<3x5xf32>"))
    a = torch.ValueTensorLiteralOp(attr)
    b = torch.ValueTensorLiteralOp(attr)
    c = torch.AtenAddTensorOp(a.result, b.result, 1)
    return c


torch_ops()
