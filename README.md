- [PI](#PI)
- [Installing](#installing)
- [Minimal example](#minimal-example)
- [Moderately interesting example](#moderately-interesting-example)
- [Torch-MLIR](#torch-mlir)

<p align="center">
    <img width="598" alt="image" src="https://user-images.githubusercontent.com/5657668/205545845-544fe701-79d5-43c1-beec-09763f22cc85.png">
</p>

# PI

Early days of a Python frontend for MLIR.

# Installing

Just 

```shell
pip install - requirements.txt 
pip install . --no-build-isolation
```

and you're good to go.

# Minimal example

[simple_kernels.py](./tests/unit/simple_kernels.py) (in [tests](./tests/unit)) looks like this

```python
from pi.dialects import memref, linalg

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
```

Running [test_compiler.py](./tests/test_compiler.py) produces

```mlir
module {
  func.func private @saxpy(%arg0: f64, %arg1: f64) -> memref<10x20xf64> {
    %0 = memref.alloca() : memref<10x30xf64>
    %1 = memref.alloca() : memref<30x20xf64>
    %2 = memref.alloca() : memref<10x20xf64>
    affine.for %arg2 = 0 to 10 {
      affine.for %arg3 = 0 to 30 {
        affine.for %arg4 = 0 to 20 {
          %3 = memref.load %2[%arg2, %arg4] : memref<10x20xf64>
          %4 = memref.load %0[%arg2, %arg3] : memref<10x30xf64>
          %5 = memref.load %1[%arg3, %arg4] : memref<30x20xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = arith.mulf %6, %arg0 : f64
          %8 = arith.addf %7, %arg1 : f64
          %9 = arith.addf %3, %8 : f64
          memref.store %9, %2[%arg2, %arg4] : memref<10x20xf64>
        }
      }
    }
    return %2 : memref<10x20xf64>
  }
  func.func private @conditionals(%arg0: f64, %arg1: f64, %arg2: f64) -> memref<1x3xf64> {
    %c2 = arith.constant 2 : index
    %c4_i64 = arith.constant 4 : i64
    %c1 = arith.constant 1 : index
    %c3_i64 = arith.constant 3 : i64
    %0 = memref.alloca() : memref<1x3xf64>
    affine.for %arg3 = 0 to 10 {
      %1 = arith.uitofp %c3_i64 : i64 to f64
      %2 = arith.cmpf ogt, %arg0, %1 : f64
      scf.if %2 {
        %3 = arith.index_cast %arg3 : index to i64
        %4 = arith.uitofp %3 : i64 to f64
        %5 = arith.mulf %arg0, %4 : f64
        memref.store %5, %0[%c1, %c1] : memref<1x3xf64>
      } else {
        %3 = arith.uitofp %c4_i64 : i64 to f64
        %4 = arith.cmpf ogt, %arg0, %3 : f64
        scf.if %4 {
          %5 = arith.index_cast %arg3 : index to i64
          %6 = arith.uitofp %5 : i64 to f64
          %7 = arith.mulf %arg1, %6 : f64
          memref.store %7, %0[%c1, %c2] : memref<1x3xf64>
        }
      }
    }
    return %0 : memref<1x3xf64>
  }
  func.func private @linalg_ops(%arg0: f64, %arg1: f64, %arg2: i32) -> memref<7x17xf64> {
    %0 = memref.alloca() : memref<10x30xf64>
    linalg.fill_rng_2d ins(%arg0, %arg1, %arg2 : f64, f64, i32) outs(%0 : memref<10x30xf64>)
    %1 = memref.alloca() : memref<30x20xf64>
    %2 = memref.alloca() : memref<10x20xf64>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%0, %1 : memref<10x30xf64>, memref<30x20xf64>) outs(%2 : memref<10x20xf64>)
    %3 = memref.alloca() : memref<3x3xf64>
    %4 = memref.alloca() : memref<7x17xf64>
    affine.for %arg3 = 0 to 10 {
      linalg.conv_2d ins(%2, %3 : memref<10x20xf64>, memref<3x3xf64>) outs(%4 : memref<7x17xf64>)
    }
    return %4 : memref<7x17xf64>
  }
}
```

<span style="font-size:4em;">🎉</span>

# Moderately interesting example

[simple_kernels.py](./tests/unit/simple_kernels.py) has a [SAXPY](https://www.ibm.com/docs/en/essl/6.2?topic=vss-saxpy-daxpy-caxpy-zaxpy-multiply-vector-by-scalar-add-vector-store-in-vector) implementation:

```python
def saxpy(a: float, b: float):
    A = memref.AllocaOp((10, 30))
    B = memref.AllocaOp((30, 20))
    C = memref.AllocaOp((10, 20))
    for i in range(10):
        for j in range(30):
            for k in range(20):
                C[i, k] += A[i, j] * B[j, k] * a + b
    return C
```

[test_tiling.py](./tests/test_tiling.py) operates on this kernel, lowering it to `affine.for` loops **and** running a few passes them, including tiling. 
In particular, transforming from this

```mlir
func.func private @saxpy(%arg0: f64, %arg1: f64) -> memref<10x20xf64> {
  %0 = memref.alloca() : memref<10x30xf64>
  %1 = memref.alloca() : memref<30x20xf64>
  %2 = memref.alloca() : memref<10x20xf64>
  affine.for %arg2 = 0 to 10 {
    affine.for %arg3 = 0 to 30 {
      affine.for %arg4 = 0 to 20 {
        %3 = memref.load %2[%arg2, %arg4] : memref<10x20xf64>
        %4 = memref.load %0[%arg2, %arg3] : memref<10x30xf64>
        %5 = memref.load %1[%arg3, %arg4] : memref<30x20xf64>
        %6 = arith.mulf %4, %5 : f64
        %7 = arith.mulf %6, %arg0 : f64
        %8 = arith.addf %7, %arg1 : f64
        %9 = arith.addf %3, %8 : f64
        memref.store %9, %2[%arg2, %arg4] : memref<10x20xf64>
      }
    }
  }
  return %2 : memref<10x20xf64>
}
```

to this

```mlir
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 2)>
func.func private @saxpy(%arg0: f64, %arg1: f64) -> memref<10x20xf64> {
  %0 = memref.alloca() : memref<10x30xf64>
  %1 = memref.alloca() : memref<30x20xf64>
  %2 = memref.alloca() : memref<10x20xf64>
  affine.for %arg2 = 0 to 10 step 2 {
    affine.for %arg3 = 0 to 30 step 2 {
      affine.for %arg4 = 0 to 20 step 2 {
        affine.for %arg5 = #map0(%arg2) to #map1(%arg2) {
          affine.for %arg6 = #map0(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map0(%arg4) to #map1(%arg4) {
              %3 = memref.load %2[%arg5, %arg7] : memref<10x20xf64>
              %4 = memref.load %0[%arg5, %arg6] : memref<10x30xf64>
              %5 = memref.load %1[%arg6, %arg7] : memref<30x20xf64>
              %6 = arith.mulf %4, %5 : f64
              %7 = arith.mulf %6, %arg0 : f64
              %8 = arith.addf %7, %arg1 : f64
              %9 = arith.addf %3, %8 : f64
              memref.store %9, %2[%arg5, %arg7] : memref<10x20xf64>
            }
          }
        }
      }
    }
  }
  return %2 : memref<10x20xf64>
}
```

# Torch-MLIR

Preliminary support for the `torch-mlir` dialect is available:

```python
class MyConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        y = self.conv(x)
        z = y + y
        w = z * z
        return w
```

lowers to

```mlir
module {
  func.func private @simple_conv2d() -> !torch.vtensor {
    %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<1x3x32x32xf32>) : !torch.vtensor<[1,3,32,32],f32>
    %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.vtensor<[1],f32>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<1x3x3x3xf32>) : !torch.vtensor<[1,3,3,3],f32>
    %int1 = torch.constant.int 1
    %int1_0 = torch.constant.int 1
    %3 = torch.prim.ListConstruct %int1, %int1_0 : (!torch.int, !torch.int) -> !torch.list<int>
    %int0 = torch.constant.int 0
    %int0_1 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0, %int0_1 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_2 = torch.constant.int 1
    %int1_3 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int1_2, %int1_3 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_4 = torch.constant.int 1
    %6 = torch.aten.conv2d %0, %2, %1, %3, %4, %5, %int1_4 : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[1,3,3,3],f32>, !torch.vtensor<[1],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor
    %7 = "torch.constant.number"() {value = 1 : i64} : () -> !torch.number
    %8 = torch.aten.add.Tensor %6, %6, %7 : !torch.vtensor, !torch.vtensor, !torch.number -> !torch.vtensor
    %9 = torch.aten.mul.Tensor %8, %8 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
    return %9 : !torch.vtensor
  }
}
```

This is very rough right now; to get a rough idea of the current status check the [latest tests](https://github.com/nod-ai/PI/actions?query=workflow%3ATest++).

Currently, we're passing 561 out of 770 of Torch-MLIR's test-suite (`torch-mlir==20230104.708`).

# Build Wheel

```shell
pip install - requirements.txt 
pip wheel . --no-build-isolation --wheel-dir wheelhouse
```