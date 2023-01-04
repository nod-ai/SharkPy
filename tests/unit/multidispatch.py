import pi
from pi.mlir_utils import mlir_cm
import torch

Tensor = pi.Tensor


conv2d = pi.conv2d


# def std(self_: Tensor, unbiased: bool = True):
# def std(self_: Tensor, dim: Optional[List[int]], unbiased: bool = True, keepdim: bool = False):
# def std(self_: Tensor, dim: Optional[List[int]] = None, correction: Optional[int] = None, keepdim: bool = False):
with mlir_cm() as module:
    # z = Tensor(None)
    # pi.std(z, True)
    # pi.std(z, unbiased=True)
    # pi.std(z)
    # pi.__contains__([1, 2, 3], 4)
    z = pi.empty((1,2,3))

    pi.abs(z)

    w = pi.empty((4, 5, 6))
    z.bitwise_and(w)

    print(module)
