import warnings
from typing import Optional, Tuple, Union

import pi
from pi import Tensor

from .module import Module
from .. import functional as F
from ..init import xavier_uniform_, constant_, xavier_normal_
from ..parameter import UninitializedParameter
from .linear import NonDynamicallyQuantizableLinear

__all__ = [
    "Threshold",
    "ReLU",
    "RReLU",
    "Hardtanh",
    "ReLU6",
    "Sigmoid",
    "Hardsigmoid",
    "Tanh",
    "SiLU",
    "Mish",
    "Hardswish",
    "ELU",
    "CELU",
    "SELU",
    "GLU",
    "GELU",
    "Hardshrink",
    "LeakyReLU",
    "LogSigmoid",
    "Softplus",
    "Softshrink",
    "MultiheadAttention",
    "PReLU",
    "Softsign",
    "Tanhshrink",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
]


class Threshold(Module):

    __constants__ = ["threshold", "value", "inplace"]

    threshold: float
    value: float
    inplace: bool

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def forward(self, input: Tensor) -> Tensor:
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def extra_repr(self):
        inplace_str = ", inplace=True" if self.inplace else ""
        return "threshold={}, value={}{}".format(
            self.threshold, self.value, inplace_str
        )


class ReLU(Module):

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class RReLU(Module):

    __constants__ = ["lower", "upper", "inplace"]

    lower: float
    upper: float
    inplace: bool

    def __init__(
        self, lower: float = 1.0 / 8, upper: float = 1.0 / 3, inplace: bool = False
    ):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self):
        inplace_str = ", inplace=True" if self.inplace else ""
        return "lower={}, upper={}{}".format(self.lower, self.upper, inplace_str)


class Hardtanh(Module):

    __constants__ = ["min_val", "max_val", "inplace"]

    min_val: float
    max_val: float
    inplace: bool

    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        super(Hardtanh, self).__init__()
        if min_value is not None:
            warnings.warn(
                "keyword argument min_value is deprecated and rename to min_val"
            )
            min_val = min_value
        if max_value is not None:
            warnings.warn(
                "keyword argument max_value is deprecated and rename to max_val"
            )
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        assert self.max_val > self.min_val

    def forward(self, input: Tensor) -> Tensor:
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "min_val={}, max_val={}{}".format(
            self.min_val, self.max_val, inplace_str
        )


class ReLU6(Hardtanh):
    def __init__(self, inplace: bool = False):
        super(ReLU6, self).__init__(0.0, 6.0, inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return pi.sigmoid(input)


class Hardsigmoid(Module):

    __constants__ = ["inplace"]

    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(Hardsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardsigmoid(input, self.inplace)


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return pi.tanh(input)


class SiLU(Module):

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Mish(Module):

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.mish(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Hardswish(Module):

    __constants__ = ["inplace"]

    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(Hardswish, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardswish(input, self.inplace)


class ELU(Module):

    __constants__ = ["alpha", "inplace"]
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "alpha={}{}".format(self.alpha, inplace_str)


class CELU(Module):

    __constants__ = ["alpha", "inplace"]
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super(CELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.celu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "alpha={}{}".format(self.alpha, inplace_str)


class SELU(Module):

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(SELU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.selu(input, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class GLU(Module):

    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.glu(input, self.dim)

    def extra_repr(self) -> str:
        return "dim={}".format(self.dim)


class GELU(Module):

    __constants__ = ["approximate"]
    approximate: str

    def __init__(self, approximate: str = "none") -> None:
        super(GELU, self).__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return "approximate={}".format(repr(self.approximate))


class Hardshrink(Module):

    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.hardshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return "{}".format(self.lambd)


class LeakyReLU(Module):

    __constants__ = ["inplace", "negative_slope"]
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "negative_slope={}{}".format(self.negative_slope, inplace_str)


class LogSigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.logsigmoid(input)


class Softplus(Module):
    __constants__ = ["beta", "threshold"]
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return "beta={}, threshold={}".format(self.beta, self.threshold)


class Softshrink(Module):
    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super(Softshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.softshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return str(self.lambd)


class MultiheadAttention(Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[Union[UninitializedParameter, pi.Tensor]]
    bias_v: Optional[Union[UninitializedParameter, pi.Tensor]]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = UninitializedParameter(
                (embed_dim, embed_dim), **factory_kwargs
            )
            self.k_proj_weight = UninitializedParameter(
                (embed_dim, self.kdim), **factory_kwargs
            )
            self.v_proj_weight = UninitializedParameter(
                (embed_dim, self.vdim), **factory_kwargs
            )
            # self.register_("in_proj_weight", None)
        else:
            self.in_proj_weight = UninitializedParameter(
                (3 * embed_dim, embed_dim), **factory_kwargs
            )
            # self.register_("q_proj_weight", None)
            # self.register_("k_proj_weight", None)
            # self.register_("v_proj_weight", None)

        if bias:
            self.in_proj_bias = UninitializedParameter(3 * embed_dim, **factory_kwargs)
        else:
            self.register_("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = UninitializedParameter((1, 1, embed_dim), **factory_kwargs)
            self.bias_v = UninitializedParameter((1, 1, embed_dim), **factory_kwargs)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != pi.bool and not pi.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        why_not_fast_path = ""
        if not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif (
            self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype
        ):
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif pi.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if pi.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all(
                [
                    (x is None or x.is_cuda or "cpu" in str(x.device))
                    for x in tensor_args
                ]
            ):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif pi.is_grad_enabled() and any(
                [x is not None and x.requires_grad for x in tensor_args]
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                return pi._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type,
                )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[int]]:

        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None
        if attn_mask is not None:
            mask_type = 0
            merged_mask = attn_mask
        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask
        if (attn_mask is not None) and (key_padding_mask is not None):
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2
            key_padding_mask_expanded = key_padding_mask.view(
                batch_size, 1, 1, seq_len
            ).expand(-1, self.num_heads, -1, -1)
            attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                batch_size, self.num_heads, -1, -1
            )
            merged_mask = attn_mask_expanded.logical_or(key_padding_mask_expanded)
        return merged_mask, mask_type


class PReLU(Module):

    __constants__ = ["num_parameters"]
    num_parameters: int

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super(PReLU, self).__init__()
        self.weight = UninitializedParameter(num_parameters, **factory_kwargs).fill_(
            init
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, self.weight)

    def extra_repr(self) -> str:
        return "num_parameters={}".format(self.num_parameters)


class Softsign(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.softsign(input)


class Tanhshrink(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.tanhshrink(input)


class Softmin(Module):

    __constants__ = ["dim"]
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(Softmin, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.softmin(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return "dim={dim}".format(dim=self.dim)


class Softmax(Module):
    __constants__ = ["dim"]
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return "dim={dim}".format(dim=self.dim)


class Softmax2d(Module):
    def forward(self, input: Tensor) -> Tensor:
        assert (
            input.dim() == 4 or input.dim() == 3
        ), "Softmax2d requires a 3D or 4D tensor as input"
        return F.softmax(input, -3, _stacklevel=5)


class LogSoftmax(Module):

    __constants__ = ["dim"]
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return "dim={dim}".format(dim=self.dim)
