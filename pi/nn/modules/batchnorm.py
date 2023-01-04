from typing import Optional, Any

import pi
from pi import Tensor
from .module import Module
from .. import functional as F
from ..parameter import UninitializedParameter, UninitializedBuffer
from .. import init


__all__ = [
    "BatchNorm1d",
    # "LazyBatchNorm1d",
    "BatchNorm2d",
    # "LazyBatchNorm2d",
    "BatchNorm3d",
    # "LazyBatchNorm3d",
    "SyncBatchNorm",
]


class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = UninitializedParameter(num_features, **factory_kwargs)
            self.bias = UninitializedParameter(num_features, **factory_kwargs)
        else:
            self.register_("weight", None)
            self.register_("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean",
                UninitializedBuffer(pi.zeros, (num_features,), **factory_kwargs),
            )
            self.register_buffer(
                "running_var",
                UninitializedBuffer(pi.ones, (num_features,), **factory_kwargs),
            )
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer(
                "num_batches_tracked",
                UninitializedBuffer(
                    0,
                    dtype=pi.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = pi.tensor(0, dtype=pi.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


# class _LazyNormBase(LazyModuleMixin, _NormBase):
#
#     weight: UninitializedParameter  # type: ignore[assignment]
#     bias: UninitializedParameter  # type: ignore[assignment]
#
#     def __init__(
#         self,
#         eps=1e-5,
#         momentum=0.1,
#         affine=True,
#         track_running_stats=True,
#         device=None,
#         dtype=None,
#     ) -> None:
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super(_LazyNormBase, self).__init__(
#             # affine and track_running_stats are hardcoded to False to
#             # avoid creating tensors that will soon be overwritten.
#             0,
#             eps,
#             momentum,
#             False,
#             False,
#             **factory_kwargs,
#         )
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.weight = Uninitialized(**factory_kwargs)
#             self.bias = Uninitialized(**factory_kwargs)
#         if self.track_running_stats:
#             self.running_mean = UninitializedBuffer(**factory_kwargs)
#             self.running_var = UninitializedBuffer(**factory_kwargs)
#             self.num_batches_tracked = pi.tensor(
#                 0,
#                 dtype=pi.long,
#                 **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
#             )
#
#     def reset_parameters(self) -> None:
#         if not self.has_uninitialized_params() and self.num_features != 0:
#             super().reset_parameters()
#
#     def initialize_parameters(self, input) -> None:  # type: ignore[override]
#         if self.has_uninitialized_params():
#             self.num_features = input.shape[1]
#             if self.affine:
#                 assert isinstance(self.weight, UninitializedParameter)
#                 assert isinstance(self.bias, UninitializedParameter)
#                 self.weight.materialize((self.num_features,))
#                 self.bias.materialize((self.num_features,))
#             if self.track_running_stats:
#                 self.running_mean.materialize(
#                     (self.num_features,)
#                 )  # type:ignore[union-attr]
#                 self.running_var.materialize(
#                     (self.num_features,)
#                 )  # type:ignore[union-attr]
#             self.reset_parameters()


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


# class LazyBatchNorm1d(_LazyNormBase, _BatchNorm):
#
#     cls_to_become = BatchNorm1d  # type: ignore[assignment]
#
#     def _check_input_dim(self, input):
#         if input.dim() != 2 and input.dim() != 3:
#             raise ValueError(
#                 "expected 2D or 3D input (got {}D input)".format(input.dim())
#             )


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


# class LazyBatchNorm2d(_LazyNormBase, _BatchNorm):
#
#     cls_to_become = BatchNorm2d  # type: ignore[assignment]
#
#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class BatchNorm3d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))


# class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
#
#     cls_to_become = BatchNorm3d  # type: ignore[assignment]
#
#     def _check_input_dim(self, input):
#         if input.dim() != 5:
#             raise ValueError("expected 5D input (got {}D input)".format(input.dim()))


class SyncBatchNorm(_BatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Optional[Any] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SyncBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.process_group = process_group

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError(
                "expected at least 2D input (got {}D input)".format(input.dim())
            )

    def _check_non_zero_input_channels(self, input):
        if input.size(1) == 0:
            raise ValueError(
                "SyncBatchNorm number of input channels should be non-zero"
            )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (
            bn_training
            and self.training
            and pi.distributed.is_available()
            and pi.distributed.is_initialized()
        )
        if need_sync:
            # currently only GPU input is supported
            if not input.is_cuda:
                raise ValueError("SyncBatchNorm expected input tensor to be on GPU")

            process_group = pi.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = pi.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            raise NotImplementedError

            # assert bn_training
            # return sync_batch_norm.apply(
            #     input,
            #     self.weight,
            #     self.bias,
            #     running_mean,
            #     running_var,
            #     self.eps,
            #     exponential_average_factor,
            #     process_group,
            #     world_size,
            # )

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):

        module_output = module
        if isinstance(module, pi.nn.modules.batchnorm._BatchNorm):
            module_output = pi.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                module_output.weight = module.weight
                module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output
