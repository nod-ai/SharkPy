import warnings
from typing import List, Optional, Tuple, Union, Callable

import math

import pi
from . import _reduction as _Reduction
from .modules.utils import _list_with_default, _pair, _triple, _single
from ..types_ import (
    dtype,
    BroadcastingList2,
    boolean_dispatch,
    BroadcastingList3,
    BroadcastingList1,
)
from .. import _VF

Tensor = pi.Tensor

# conv1d = pi.conv1d

conv2d = pi.conv2d

# conv3d = pi.conv3d

conv_transpose1d = pi.conv_transpose1d

# conv_transpose2d = pi.conv_transpose2d

# conv_transpose3d = pi.conv_transpose3d

# conv_tbc = pi.conv_tbc

# Pooling
# avg_pool1d = pi.avg_pool1d

avg_pool2d = pi._C._nn.avg_pool2d


# avg_pool3d = pi._C._nn.avg_pool3d


def fractional_max_pool2d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    output_size: Optional[BroadcastingList2[int]] = None,
    output_ratio: Optional[BroadcastingList2[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool2d requires specifying either "
            "an output_size or an output_ratio"
        )
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _pair(output_ratio)
        output_size = [
            int(input.size(-2) * _output_ratio[0]),
            int(input.size(-1) * _output_ratio[1]),
        ]

    if _random_samples is None:
        n_batch = 1 if input.dim() == 3 else input.size(0)
        _random_samples = pi.rand(
            n_batch, input.size(-3), 2, dtype=input.dtype, device=input.device
        )
    return pi._C._nn.fractional_max_pool2d(
        input, kernel_size, output_size, _random_samples
    )


def _fractional_max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    output_size: Optional[BroadcastingList2[int]] = None,
    output_ratio: Optional[BroadcastingList2[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tensor:
    return fractional_max_pool2d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool2d_with_indices,
    if_false=_fractional_max_pool2d,
    module_name=__name__,
    func_name="fractional_max_pool2d",
)


def fractional_max_pool3d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = None,
    output_ratio: Optional[BroadcastingList3[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if output_size is None and output_ratio is None:
        raise ValueError(
            "fractional_max_pool3d requires specifying either "
            "an output_size or an output_ratio"
        )
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _triple(output_ratio)
        output_size = [
            int(input.size(-3) * _output_ratio[0]),
            int(input.size(-2) * _output_ratio[1]),
            int(input.size(-1) * _output_ratio[2]),
        ]

    if _random_samples is None:
        n_batch = 1 if input.dim() == 4 else input.size(0)
        _random_samples = pi.rand(
            n_batch, input.size(-4), 3, dtype=input.dtype, device=input.device
        )
    return pi._C._nn.fractional_max_pool3d(
        input, kernel_size, output_size, _random_samples
    )


def _fractional_max_pool3d(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    output_size: Optional[BroadcastingList3[int]] = None,
    output_ratio: Optional[BroadcastingList3[float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[Tensor] = None,
) -> Tensor:
    return fractional_max_pool3d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool3d_with_indices,
    if_false=_fractional_max_pool3d,
    module_name=__name__,
    func_name="fractional_max_pool3d",
)


def max_pool1d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    dilation: BroadcastingList1[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor]:
    if stride is None:
        stride = pi.jit.annotate(List[int], [])
    return pi.max_pool1d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


def _max_pool1d(
    input: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    dilation: BroadcastingList1[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    if stride is None:
        stride = pi.jit.annotate(List[int], [])
    return pi.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool1d_with_indices,
    if_false=_max_pool1d,
    module_name=__name__,
    func_name="max_pool1d",
)


def max_pool2d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor]:
    return pi._C._nn.max_pool2d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


def _max_pool2d(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    dilation: BroadcastingList2[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    return pi.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool2d_with_indices,
    if_false=_max_pool2d,
    module_name=__name__,
    func_name="max_pool2d",
)


def max_pool3d_with_indices(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    dilation: BroadcastingList3[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor]:
    return pi._C._nn.max_pool3d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )


def _max_pool3d(
    input: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    dilation: BroadcastingList3[int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor:
    if stride is None:
        stride = pi.jit.annotate(List[int], [])
    return pi.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool3d_with_indices,
    if_false=_max_pool3d,
    module_name=__name__,
    func_name="max_pool3d",
)


def _unpool_output_size(
    input: Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    output_size: Optional[List[int]],
) -> List[int]:
    input_size = input.size()
    default_size = pi.jit.annotate(List[int], [])
    for d in range(len(kernel_size)):
        default_size.append(
            (input_size[-len(kernel_size) + d] - 1) * stride[d]
            + kernel_size[d]
            - 2 * padding[d]
        )
    if output_size is None:
        ret = default_size
    else:
        if len(output_size) == len(kernel_size) + 2:
            output_size = output_size[2:]
        if len(output_size) != len(kernel_size):
            raise ValueError(
                "output_size should be a sequence containing "
                "{} or {} elements, but it has a length of '{}'".format(
                    len(kernel_size), len(kernel_size) + 2, len(output_size)
                )
            )
        for d in range(len(kernel_size)):
            min_size = default_size[d] - stride[d]
            max_size = default_size[d] + stride[d]
            if not (min_size < output_size[d] < max_size):
                raise ValueError(
                    'invalid output_size "{}" (dim {} must be between {} and {})'.format(
                        output_size, d, min_size, max_size
                    )
                )

        ret = output_size
    return ret


def max_unpool1d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList1[int],
    stride: Optional[BroadcastingList1[int]] = None,
    padding: BroadcastingList1[int] = 0,
    output_size: Optional[BroadcastingList1[int]] = None,
) -> Tensor:
    kernel_size = _single(kernel_size)
    if stride is not None:
        _stride = _single(stride)
    else:
        _stride = kernel_size
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    if isinstance(output_size, list):
        output_size = output_size + [1]
    else:
        output_size = output_size + (1,)
    return pi._C._nn.max_unpool2d(
        input.unsqueeze(-1), indices.unsqueeze(-1), output_size
    ).squeeze(-1)


def max_unpool2d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    output_size: Optional[BroadcastingList2[int]] = None,
) -> Tensor:
    kernel_size = _pair(kernel_size)
    if stride is not None:
        _stride = _pair(stride)
    else:
        _stride = kernel_size
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    return pi._C._nn.max_unpool2d(input, indices, output_size)


def max_unpool3d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    padding: BroadcastingList3[int] = 0,
    output_size: Optional[BroadcastingList3[int]] = None,
) -> Tensor:
    kernel_size = _triple(kernel_size)
    if stride is not None:
        _stride = _triple(stride)
    else:
        _stride = kernel_size
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    return pi._C._nn.max_unpool3d(input, indices, output_size, _stride, padding)


def lp_pool2d(
    input: Tensor,
    norm_type: Union[int, float],
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    kw, kh = _pair(kernel_size)
    if stride is not None:
        out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool2d(
            input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode
        )

    return (pi.sign(out) * relu(pi.abs(out))).mul(kw * kh).pow(1.0 / norm_type)


def lp_pool1d(
    input: Tensor,
    norm_type: Union[int, float],
    kernel_size: int,
    stride: Optional[BroadcastingList1[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    if stride is not None:
        out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool1d(
            input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode
        )

    return (pi.sign(out) * relu(pi.abs(out))).mul(kernel_size).pow(1.0 / norm_type)


def adaptive_max_pool1d_with_indices(
    input: Tensor, output_size: BroadcastingList1[int], return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    return pi.adaptive_max_pool1d(input, output_size)


def _adaptive_max_pool1d(
    input: Tensor, output_size: BroadcastingList1[int], return_indices: bool = False
) -> Tensor:
    return adaptive_max_pool1d_with_indices(input, output_size)[0]


adaptive_max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool1d_with_indices,
    if_false=_adaptive_max_pool1d,
    module_name=__name__,
    func_name="adaptive_max_pool1d",
)


def adaptive_max_pool2d_with_indices(
    input: Tensor, output_size: BroadcastingList2[int], return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    output_size = _list_with_default(output_size, input.size())
    return pi._C._nn.adaptive_max_pool2d(input, output_size)


def _adaptive_max_pool2d(
    input: Tensor, output_size: BroadcastingList2[int], return_indices: bool = False
) -> Tensor:
    return adaptive_max_pool2d_with_indices(input, output_size)[0]


adaptive_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool2d_with_indices,
    if_false=_adaptive_max_pool2d,
    module_name=__name__,
    func_name="adaptive_max_pool2d",
)


def adaptive_max_pool3d_with_indices(
    input: Tensor, output_size: BroadcastingList3[int], return_indices: bool = False
) -> Tuple[Tensor, Tensor]:
    output_size = _list_with_default(output_size, input.size())
    return pi._C._nn.adaptive_max_pool3d(input, output_size)


def _adaptive_max_pool3d(
    input: Tensor, output_size: BroadcastingList3[int], return_indices: bool = False
) -> Tensor:
    return adaptive_max_pool3d_with_indices(input, output_size)[0]


adaptive_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool3d_with_indices,
    if_false=_adaptive_max_pool3d,
    module_name=__name__,
    func_name="adaptive_max_pool3d",
)

# adaptive_avg_pool1d = pi.adaptive_avg_pool1d


def adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    _output_size = _list_with_default(output_size, input.size())
    return pi._C._nn.adaptive_avg_pool2d(input, _output_size)


def adaptive_avg_pool3d(input: Tensor, output_size: BroadcastingList3[int]) -> Tensor:
    _output_size = _list_with_default(output_size, input.size())
    return pi._C._nn.adaptive_avg_pool3d(input, _output_size)


# Activation functions
def dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, " "but got {}".format(p)
        )
    return (
        _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
    )


def alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, " "but got {}".format(p)
        )
    return (
        _VF.alpha_dropout_(input, p, training)
        if inplace
        else _VF.alpha_dropout(input, p, training)
    )


def dropout1d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, " "but got {}".format(p)
        )
    inp_dim = input.dim()
    if inp_dim not in (2, 3):
        raise RuntimeError(
            f"dropout1d: Expected 2D or 3D input, but received a {inp_dim}D input. "
            "Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
            "spatial dimension, a channel dimension, and an optional batch dimension "
            "(i.e. 2D or 3D inputs)."
        )

    is_batched = inp_dim == 3
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)

    return result


def dropout2d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, " "but got {}".format(p)
        )
    inp_dim = input.dim()
    if inp_dim not in (3, 4):
        warn_msg = (
            f"dropout2d: Received a {inp_dim}-D input to dropout2d, which is deprecated "
            "and will result in an error in a future release. To retain the behavior "
            "and silence this warning, please use dropout instead. Note that dropout2d "
            "exists to provide channel-wise dropout on inputs with 2 spatial dimensions, "
            "a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs)."
        )
        warnings.warn(warn_msg)

    # TODO: Properly support no-batch-dim inputs. For now, these are NOT supported; passing
    # a 3D input will perform dropout1d behavior instead. This was done historically and the
    # behavior is maintained here for now.
    # See https://github.com/pytorch/pytorch/issues/77081
    if inp_dim == 3:
        warnings.warn(
            "dropout2d: Received a 3D input to dropout2d and assuming that channel-wise "
            "1D dropout behavior is desired - input is interpreted as shape (N, C, L), where C "
            "is the channel dim. This behavior will change in a future release to interpret the "
            "input as one without a batch dimension, i.e. shape (C, H, W). To maintain the 1D "
            "channel-wise dropout behavior, please switch to using dropout1d instead."
        )

    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    return result


def dropout3d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, " "but got {}".format(p)
        )
    inp_dim = input.dim()
    if inp_dim not in (4, 5):
        warn_msg = (
            f"dropout3d: Received a {inp_dim}-D input to dropout3d, which is deprecated "
            "and will result in an error in a future release. To retain the behavior "
            "and silence this warning, please use dropout instead. Note that dropout3d "
            "exists to provide channel-wise dropout on inputs with 3 spatial dimensions, "
            "a channel dimension, and an optional batch dimension (i.e. 4D or 5D inputs)."
        )
        warnings.warn(warn_msg)

    is_batched = inp_dim == 5
    if not is_batched:
        input = input.unsqueeze_(0) if inplace else input.unsqueeze(0)

    result = (
        _VF.feature_dropout_(input, p, training)
        if inplace
        else _VF.feature_dropout(input, p, training)
    )

    if not is_batched:
        result = result.squeeze_(0) if inplace else result.squeeze(0)
    return result


def feature_alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, " "but got {}".format(p)
        )
    return (
        _VF.feature_alpha_dropout_(input, p, training)
        if inplace
        else _VF.feature_alpha_dropout(input, p, training)
    )


def _threshold(
    input: Tensor, threshold: float, value: float, inplace: bool = False
) -> Tensor:
    if inplace:
        result = _VF.threshold_(input, threshold, value)
    else:
        result = _VF.threshold(input, threshold, value)
    return result


threshold = _threshold

threshold_ = _VF.threshold_


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        result = pi.relu_(input)
    else:
        result = pi.relu(input)
    return result


relu_ = pi.relu_


def glu(input: Tensor, dim: int = -1) -> Tensor:
    if input.dim() == 0:
        raise RuntimeError(
            "glu does not support scalars because halving size must be even"
        )
    return pi._C._nn.glu(input, dim)


def hardtanh(
    input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False
) -> Tensor:
    if inplace:
        result = pi._C._nn.hardtanh_(input, min_val, max_val)
    else:
        result = pi._C._nn.hardtanh(input, min_val, max_val)
    return result


hardtanh_ = pi._C._nn.hardtanh_


def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        result = pi._C._nn.relu6_(input)
    else:
        result = pi._C._nn.relu6(input)
    return result


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    if inplace:
        result = pi._C._nn.elu_(input, alpha)
    else:
        result = pi._C._nn.elu(input, alpha)
    return result


# elu_ = pi._C._nn.elu_


def selu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        result = pi.selu_(input)
    else:
        result = pi.selu(input)
    return result


# selu_ = pi.selu_


def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    if inplace:
        result = pi.celu_(input, alpha)
    else:
        result = pi.celu(input, alpha)
    return result


# celu_ = pi.celu_


def leaky_relu(
    input: Tensor, negative_slope: float = 0.01, inplace: bool = False
) -> Tensor:
    if inplace:
        result = pi._C._nn.leaky_relu_(input, negative_slope)
    else:
        result = pi._C._nn.leaky_relu(input, negative_slope)
    return result


leaky_relu_ = pi._C._nn.leaky_relu_


# prelu = pi.prelu


def rrelu(
    input: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    if inplace:
        result = pi.rrelu_(input, lower, upper, training)
    else:
        result = pi.rrelu(input, lower, upper, training)
    return result


# rrelu_ = pi.rrelu_

# logsigmoid = pi._C._nn.log_sigmoid

gelu = pi._C._nn.gelu

# hardshrink = pi.hardshrink


def tanhshrink(input):
    return input - input.tanh()


def softsign(input):
    return input / (input.abs() + 1)


softplus = pi._C._nn.softplus


def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        "Implicit dimension choice for {} has been deprecated. "
        "Change the call to include dim=X as an argument.".format(name),
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def softmin(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[dtype] = None,
) -> Tensor:
    if dim is None:
        dim = _get_softmax_dim("softmin", input.dim(), _stacklevel)
    if dtype is None:
        ret = (-input).softmax(dim)
    else:
        ret = (-input).softmax(dim, dtype=dtype)
    return ret


def softmax(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[dtype] = None,
) -> Tensor:
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> Tensor:
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -pi.empty_like(logits, memory_format=pi.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = pi.zeros_like(
            logits, memory_format=pi.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def log_softmax(
    input: Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[dtype] = None,
) -> Tensor:
    if dim is None:
        dim = _get_softmax_dim("log_softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.log_softmax(dim)
    else:
        ret = input.log_softmax(dim, dtype=dtype)
    return ret


# softshrink = pi._C._nn.softshrink


def tanh(input):
    return input.tanh()


def sigmoid(input):
    return input.sigmoid()


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return pi._C._nn.hardsigmoid_(input)
    return pi._C._nn.hardsigmoid(input)


linear = pi._C._nn.linear


# bilinear = pi.bilinear


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return pi._C._nn.silu_(input)
    return pi._C._nn.silu(input)


def mish(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return pi._C._nn.mish_(input)
    return pi._C._nn.mish(input)


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return pi._C._nn.hardswish_(input)
    return pi._C._nn.hardswish(input)


def _no_grad_embedding_renorm_(
    weight: Tensor, input: Tensor, max_norm: float, norm_type: float
) -> Tuple[Tensor, Tensor]:
    pi.embedding_renorm_(weight.detach(), input, max_norm, norm_type)


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        raise NotImplementedError
        # Note [embedding_renorm contiguous]
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        # input = input.contiguous()
        # Note [embedding_renorm set_grad_enabled]
        # XXX: equivalent to
        # with pi.no_grad():
        #   pi.embedding_renorm_
        # remove once script supports set_grad_enabled
        # _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    return pi.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Optional[Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tensor:
    # Check for backward compatibility.
    # Used to be embedding_bag(weight, input, ...)
    # Now is     embedding_bag(input, weight, ...)
    if weight.dtype == pi.long and input.is_floating_point():
        warnings.warn(
            "Argument order of nn.functional.embedding_bag was changed. "
            "Usage `embedding_bag(weight, input, ...)` is deprecated, "
            "and should now be `embedding_bag(input, weight, ...)`."
        )
        weight, input = input, weight

    if per_sample_weights is not None and input.size() != per_sample_weights.size():
        raise ValueError(
            "embedding_bag: If per_sample_weights ({}) is not None, "
            "then it must have the same shape as the input ({})".format(
                per_sample_weights.shape, input.shape
            )
        )

    if input.dim() == 2:
        if offsets is not None:
            type_str = "<unknown>"
            type_str = str(type(offsets))
            raise ValueError(
                "if input is 2D, then offsets has to be None"
                ", as input is treated is a mini-batch of"
                " fixed length sequences. However, found "
                "offsets of type {}".format(type_str)
            )
        offsets = pi.arange(
            0, input.numel(), input.size(1), dtype=input.dtype, device=input.device
        )

        input = input.reshape(-1)
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.reshape(-1)
    elif input.dim() == 1:
        if offsets is None:
            raise ValueError("offsets has to be a 1D Tensor but got None")
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")
    else:
        raise ValueError(
            "input has to be 1D or 2D Tensor,"
            " but got Tensor of dimension {}".format(input.dim())
        )
    if mode == "sum":
        mode_enum = 0
    elif mode == "mean":
        mode_enum = 1
    elif mode == "max":
        mode_enum = 2

        if scale_grad_by_freq:
            raise ValueError(
                "max mode does not support scaling the gradient by the frequency"
            )

        if sparse:
            raise ValueError("max mode does not support sparse weights")

    else:
        raise ValueError("mode has to be one of sum, mean or max")

    if max_norm is not None:
        # XXX: equivalent to
        # with pi.no_grad():
        #   pi.nembedding_renorm_
        # remove once script supports set_grad_enabled
        raise NotImplementedError
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)

    if per_sample_weights is not None and mode != "sum":
        raise NotImplementedError(
            "embedding_bag: per_sample_weights was not None. "
            "per_sample_weights is only supported for mode='sum' "
            "(got mode='{}'). Please open a feature request on GitHub.".format(mode)
        )

    ret, _, _, _ = pi.embedding_bag(
        weight,
        input,
        offsets,
        scale_grad_by_freq,
        mode_enum,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )
    return ret


def _verify_batch_size(size: List[int]) -> None:
    # XXX: JIT script does not support the reduce from functools, and mul op is a
    # builtin, which cannot be used as a value to a func yet, so rewrite this size
    # check to a simple equivalent for loop
    #
    # TODO: make use of reduce like below when JIT is ready with the missing features:
    # from operator import mul
    # from functools import reduce
    #
    #   if reduce(mul, size[2:], size[0]) == 1
    size_prods = size[0]
    for i in range(len(size) - 2):
        size_prods *= size[i + 2]
    if size_prods == 1:
        raise ValueError(
            "Expected more than 1 value per channel when training, got input size {}".format(
                size
            )
        )


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    if training:
        _verify_batch_size(input.size())

    return pi.batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        False,
    )


def _verify_spatial_size(size: List[int]) -> None:
    # Verify that there is > 1 spatial element for instance norm calculation.
    size_prods = 1
    for i in range(2, len(size)):
        size_prods *= size[i]
    if size_prods == 1:
        raise ValueError(
            "Expected more than 1 spatial element when training, got input size {}".format(
                size
            )
        )


def instance_norm(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    if use_input_stats:
        _verify_spatial_size(input.size())
    return pi.instance_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        use_input_stats,
        momentum,
        eps,
        pi.backends.cudnn.enabled,
    )


def layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    return pi.layer_norm(input, normalized_shape, weight, bias, eps, False)


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    if input.dim() < 2:
        raise RuntimeError(
            f"Expected at least 2 dimensions for input tensor but received {input.dim()}"
        )
    _verify_batch_size(
        [input.size(0) * input.size(1) // num_groups, num_groups]
        + list(input.size()[2:])
    )
    return pi.group_norm(
        input, num_groups, weight, bias, eps, pi.backends.cudnn.enabled
    )


def local_response_norm(
    input: Tensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
) -> Tensor:
    dim = input.dim()
    if dim < 3:
        raise ValueError(
            "Expected 3D or higher dimensionality \
                         input (got {} dimensions)".format(
                dim
            )
        )

    if input.numel() == 0:
        return input

    div = input.mul(input).unsqueeze(1)
    if dim == 3:
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        raise NotImplementedError
        # sizes = input.size()
        # div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        # div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        # div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1)
        # div = div.view(sizes)
    div = div.mul(alpha).add(k).pow(beta)
    return input / div


# loss


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    return pi.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        _Reduction.get_enum(reduction),
        zero_infinity,
    )


def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return pi._C._nn.nll_loss_nd(
        input, target, weight, _Reduction.get_enum(reduction), ignore_index
    )


def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    size_average: Optional[bool] = None,
    eps: float = 1e-8,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        ret = input
        raise ValueError(reduction + " is not a valid value for reduction")

    ret = pi.poisson_nll_loss(
        input, target, log_input, full, eps, _Reduction.get_enum(reduction)
    )
    return ret


def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if var.size() != input.size():

        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == var.size():
            var = pi.unsqueeze(var, -1)

        # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif (
            input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1
        ):  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if pi.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (pi.log(var) + (input - target) ** 2 / var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def kl_div(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        if reduction == "mean":
            warnings.warn(
                "reduction: 'mean' divides the total loss by both the batch size and the support size."
                "'batchmean' divides only by the batch size, and aligns with the KL div math definition."
                "'mean' will be changed to behave the same as 'batchmean' in the next major release."
            )

        # special case for batchmean
        if reduction == "batchmean":
            reduction_enum = _Reduction.get_enum("sum")
        else:
            reduction_enum = _Reduction.get_enum(reduction)

    reduced = pi.kl_div(input, target, reduction_enum, log_target=log_target)

    if reduction == "batchmean" and input.dim() != 0:
        reduced = reduced / input.size()[0]

    return reduced


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return pi._C._nn.cross_entropy_loss(
        input,
        target,
        weight,
        _Reduction.get_enum(reduction),
        ignore_index,
        label_smoothing,
    )


def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if target.size() != input.size():
        raise ValueError(
            "Using a target size ({}) that is different to the input size ({}) is deprecated. "
            "Please ensure they have the same size.".format(target.size(), input.size())
        )

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)

    return pi._C._nn.binary_cross_entropy(input, target, weight, reduction_enum)


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)

    if not (target.size() == input.size()):
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                target.size(), input.size()
            )
        )

    return pi.binary_cross_entropy_with_logits(
        input, target, weight, pos_weight, reduction_enum
    )


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(
                target.size(), input.size()
            ),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = pi.broadcast_tensors(input, target)
    return pi._C._nn.smooth_l1_loss(
        expanded_input, expanded_target, _Reduction.get_enum(reduction), beta
    )


def huber_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    delta: float = 1.0,
) -> Tensor:
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(
                target.size(), input.size()
            ),
            stacklevel=2,
        )

    expanded_input, expanded_target = pi.broadcast_tensors(input, target)
    return pi._C._nn.huber_loss(
        expanded_input, expanded_target, _Reduction.get_enum(reduction), delta
    )


def l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(
                target.size(), input.size()
            ),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = pi.broadcast_tensors(input, target)
    return pi._C._nn.l1_loss(
        expanded_input, expanded_target, _Reduction.get_enum(reduction)
    )


def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(
                target.size(), input.size()
            ),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = pi.broadcast_tensors(input, target)
    return pi._C._nn.mse_loss(
        expanded_input, expanded_target, _Reduction.get_enum(reduction)
    )


def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if input1.dim() != input2.dim() or input1.dim() != target.dim():
        raise RuntimeError(
            (
                "margin_ranking_loss : All input tensors should have same dimension but got sizes: "
                "input1: {}, input2: {}, target: {} ".format(
                    input1.size(), input2.size(), target.size()
                )
            )
        )
    return pi.margin_ranking_loss(input1, input2, target, margin, reduction_enum)


def hinge_embedding_loss(
    input: Tensor,
    target: Tensor,
    margin: float = 1.0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return pi.hinge_embedding_loss(input, target, margin, reduction_enum)


def multilabel_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return pi._C._nn.multilabel_margin_loss(input, target, reduction_enum)


def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return pi._C._nn.soft_margin_loss(input, target, reduction_enum)


def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))

    if weight is not None:
        loss = loss * weight

    class_dim = input.dim() - 1
    C = input.size(class_dim)
    loss = loss.sum(dim=class_dim) / C  # only return N loss values

    if reduction == "none":
        ret = loss
    elif reduction == "mean":
        ret = loss.mean()
    elif reduction == "sum":
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return pi.cosine_embedding_loss(input1, input2, target, margin, reduction_enum)


def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if p != 1 and p != 2:
        raise ValueError("only p == 1 and p == 2 supported")
    if weight is not None:
        if weight.dim() != 1:
            raise ValueError("weight must be one-dimensional")

    return pi._C._nn.multi_margin_loss(input, target, p, margin, weight, reduction_enum)


# pixel_shuffle = pi.pixel_shuffle

# pixel_unshuffle = pi.pixel_unshuffle

# channel_shuffle = pi.channel_shuffle

# native_channel_shuffle = pi.native_channel_shuffle


def upsample(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):  # noqa: F811

    warnings.warn(
        "nn.functional.upsample is deprecated. Use nn.functional.interpolate instead."
    )
    return interpolate(input, size, scale_factor, mode, align_corners)


def interpolate(
    input: Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:  # noqa: F811

    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "Input and output must have the same number of spatial dimensions, but got "
                    f"input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "output size in (o1, o2, ...,oK) format."
                )
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "Input and scale_factor must have the same number of spatial dimensions, but "
                    f"got input with spatial dimensions of {list(input.shape[2:])} and "
                    f"scale_factor of shape {scale_factor}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "scale_factor in (s1, s2, ...,sK) format."
                )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if (
        recompute_scale_factor is not None
        and recompute_scale_factor
        and size is not None
    ):
        raise ValueError(
            "recompute_scale_factor is not meaningful with an explicit size."
        )

    # "area" mode always requires an explicit size rather than scale factor.
    # Re-use the recompute_scale_factor code path.
    if mode == "area" and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        if not pi.jit.is_scripting() and pi._C._get_tracing_state():
            # make scale_factor a tensor in tracing so constant doesn't get baked in
            output_size = [
                (
                    pi.floor(
                        (
                            input.size(i + 2).float()
                            * pi.tensor(scale_factors[i], dtype=pi.float32)
                        ).float()
                    )
                )
                for i in range(dim)
            ]
        else:
            assert scale_factors is not None
            output_size = [
                int(math.floor(float(input.size(i + 2)) * scale_factors[i]))
                for i in range(dim)
            ]
        scale_factors = None

    if antialias and not (mode in ("bilinear", "bicubic") and input.ndim == 4):
        raise ValueError(
            "Anti-alias option is only supported for bilinear and bicubic modes"
        )

    if input.dim() == 3 and mode == "nearest":
        return pi._C._nn.upsample_nearest1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest":
        return pi._C._nn.upsample_nearest2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest":
        return pi._C._nn.upsample_nearest3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "nearest-exact":
        return pi._C._nn._upsample_nearest_exact1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest-exact":
        return pi._C._nn._upsample_nearest_exact2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest-exact":
        return pi._C._nn._upsample_nearest_exact3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool1d(input, output_size)
    if input.dim() == 4 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool2d(input, output_size)
    if input.dim() == 5 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool3d(input, output_size)

    if input.dim() == 3 and mode == "linear":
        assert align_corners is not None
        return pi._C._nn.upsample_linear1d(
            input, output_size, align_corners, scale_factors
        )
    if input.dim() == 4 and mode == "bilinear":
        assert align_corners is not None
        if antialias:
            return pi._C._nn._upsample_bilinear2d_aa(
                input, output_size, align_corners, scale_factors
            )
        return pi._C._nn.upsample_bilinear2d(
            input, output_size, align_corners, scale_factors
        )
    if input.dim() == 5 and mode == "trilinear":
        assert align_corners is not None
        return pi._C._nn.upsample_trilinear3d(
            input, output_size, align_corners, scale_factors
        )
    if input.dim() == 4 and mode == "bicubic":
        assert align_corners is not None
        if antialias:
            return pi._C._nn._upsample_bicubic2d_aa(
                input, output_size, align_corners, scale_factors
            )
        return pi._C._nn.upsample_bicubic2d(
            input, output_size, align_corners, scale_factors
        )

    if input.dim() == 3 and mode == "bilinear":
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    if input.dim() == 3 and mode == "trilinear":
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    if input.dim() == 4 and mode == "linear":
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    if input.dim() == 4 and mode == "trilinear":
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    if input.dim() == 5 and mode == "linear":
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    if input.dim() == 5 and mode == "bilinear":
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

    raise NotImplementedError(
        "Input Error: Only 3D, 4D and 5D input Tensors supported"
        " (got {}D) for the modes: nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact"
        " (got {})".format(input.dim(), mode)
    )


def upsample_nearest(input, size=None, scale_factor=None):  # noqa: F811

    # DeprecationWarning is ignored by default
    warnings.warn(
        "nn.functional.upsample_nearest is deprecated. Use nn.functional.interpolate instead."
    )
    return interpolate(input, size, scale_factor, mode="nearest")


def upsample_bilinear(input, size=None, scale_factor=None):  # noqa: F811

    # DeprecationWarning is ignored by default
    warnings.warn(
        "nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead."
    )
    return interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)


GRID_SAMPLE_INTERPOLATION_MODES = {
    "bilinear": 0,
    "nearest": 1,
    "bicubic": 2,
}

GRID_SAMPLE_PADDING_MODES = {
    "zeros": 0,
    "border": 1,
    "reflection": 2,
}


def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> Tensor:
    if mode != "bilinear" and mode != "nearest" and mode != "bicubic":
        raise ValueError(
            "nn.functional.grid_sample(): expected mode to be "
            "'bilinear', 'nearest' or 'bicubic', but got: '{}'".format(mode)
        )
    if (
        padding_mode != "zeros"
        and padding_mode != "border"
        and padding_mode != "reflection"
    ):
        raise ValueError(
            "nn.functional.grid_sample(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            "but got: '{}'".format(padding_mode)
        )

    if mode == "bilinear":
        mode_enum = 0
    elif mode == "nearest":
        mode_enum = 1
    else:  # mode == 'bicubic'
        mode_enum = 2

    if padding_mode == "zeros":
        padding_mode_enum = 0
    elif padding_mode == "border":
        padding_mode_enum = 1
    else:  # padding_mode == 'reflection'
        padding_mode_enum = 2

    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False

    return pi.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)


def affine_grid(
    theta: Tensor, size: List[int], align_corners: Optional[bool] = None
) -> Tensor:
    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False

    # enforce floating point dtype on theta
    if not theta.is_floating_point():
        raise ValueError(
            "Expected theta to have floating point type, but got {}".format(theta.dtype)
        )
    # check that shapes and sizes match
    if len(size) == 4:
        if theta.dim() != 3 or theta.shape[-2] != 2 or theta.shape[-1] != 3:
            raise ValueError(
                "Expected a batch of 2D affine matrices of shape Nx2x3 "
                "for size {}. Got {}.".format(size, theta.shape)
            )
        spatial_size = size[-2:]  # spatial dimension sizes
    elif len(size) == 5:
        if theta.dim() != 3 or theta.shape[-2] != 3 or theta.shape[-1] != 4:
            raise ValueError(
                "Expected a batch of 3D affine matrices of shape Nx3x4 "
                "for size {}. Got {}.".format(size, theta.shape)
            )
        spatial_size = size[-3:]  # spatial dimension sizes
    else:
        raise NotImplementedError(
            "affine_grid only supports 4D and 5D sizes, "
            "for 2D and 3D affine transforms, respectively. "
            "Got size {}.".format(size)
        )
    # check for empty span
    if align_corners and min(spatial_size) == 1:
        warnings.warn(
            "Since version 1.3.0, affine_grid behavior has changed "
            "for unit-size grids when align_corners=True. "
            "This is not an intended use case of affine_grid. "
            "See the documentation of affine_grid for details."
        )
    elif min(size) <= 0:
        raise ValueError("Expected non-zero, positive output size. Got {}".format(size))

    return pi.affine_grid_generator(theta, size, align_corners)


pad = pi._C._nn.pad


# distance


# pairwise_distance = pi.pairwise_distance

# pdist = pi.pdist

# cosine_similarity = pi.cosine_similarity

# one_hot = pi._C._nn.one_hot


def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    return pi.triplet_margin_loss(
        anchor, positive, negative, margin, p, eps, swap, reduction_enum
    )


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:
    # Check validity of reduction mode
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

    # Check dimensions
    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    if not (a_dim == p_dim and p_dim == n_dim):
        raise RuntimeError(
            (
                f"The anchor, positive, and negative tensors are expected to have "
                f"the same number of dimensions, but got: anchor {a_dim}D, "
                f"positive {p_dim}D, and negative {n_dim}D inputs"
            )
        )

    # Calculate loss
    if distance_function is None:
        distance_function = pi.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.  If True, and if the positive example is closer to the
    # negative example than the anchor is, swaps the positive example and the
    # anchor in the loss computation.
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = pi.minimum(dist_neg, dist_swap)
    loss = pi.clamp_min(margin + dist_pos - dist_neg, 0)

    # Apply reduction
    if reduction == "sum":
        return pi.sum(loss)
    elif reduction == "mean":
        return pi.mean(loss)
    else:  # reduction == "none"
        return loss


def normalize(
    input: Tensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is None:
        denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
        return input / denom
    else:
        denom = input.norm(p, dim, keepdim=True).clamp_min_(eps).expand_as(input)
        return pi.div(input, denom, out=out)


def assert_int_or_pair(arg: List[int], arg_name: str, message: str) -> None:
    assert isinstance(arg, int) or len(arg) == 2, message.format(arg_name)


def unfold(
    input: Tensor,
    kernel_size: BroadcastingList2[int],
    dilation: BroadcastingList2[int] = 1,
    padding: BroadcastingList2[int] = 0,
    stride: BroadcastingList2[int] = 1,
) -> Tensor:
    return pi._C._nn.im2col(
        input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride)
    )


def fold(
    input: Tensor,
    output_size: BroadcastingList2[int],
    kernel_size: BroadcastingList2[int],
    dilation: BroadcastingList2[int] = 1,
    padding: BroadcastingList2[int] = 0,
    stride: BroadcastingList2[int] = 1,
) -> Tensor:
    return pi._C._nn.col2im(
        input,
        _pair(output_size),
        _pair(kernel_size),
        _pair(dilation),
        _pair(padding),
        _pair(stride),
    )


# multihead attention


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (
        Eq,
        Eq,
    ), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (
        Eq,
        Ek,
    ), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (
        Eq,
        Ev,
    ), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (
        Eq,
    ), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (
        Eq,
    ), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (
        Eq,
    ), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# _scaled_dot_product_attention = pi._C._nn._scaled_dot_product_attention


def _mha_shape_check(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    num_heads: int,
):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, (
            "For batched (3-D) `query`, expected `key` and `value` to be 3-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, (
            "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, (
                "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert (
                    attn_mask.shape == expected_shape
                ), f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}"
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor"
        )

    return is_batched


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    tens_ops = (
        query,
        key,
        value,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        out_proj_weight,
        out_proj_bias,
    )
    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != pi.bool and not pi.is_floating_point(key_padding_mask):
            raise AssertionError(
                "only bool and floating types of key_padding_mask are supported"
            )
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, pi.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == pi.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(pi.bool)
        else:
            assert (
                attn_mask.is_floating_point() or attn_mask.dtype == pi.bool
            ), f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = pi.cat([k, bias_k.repeat(1, bsz, 1)])
        v = pi.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert (
            static_k.size(2) == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_v.size(0) == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert (
            static_v.size(2) == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = pi.cat(
            [k, pi.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = pi.cat(
            [v, pi.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == pi.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == pi.bool:
        new_attn_mask = pi.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if attn_mask is not None:
        if attn_mask.size(0) == 1:
            attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)

    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p, need_weights, is_causal
    )
    attn_output = (
        attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
    )

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
