try:
    from torch_mlir.ir import *
    from torch_mlir.ir import Type as MLIRType
    from torch_mlir.dialects._ods_common import (
        get_default_loc_context,
        get_op_result_or_value,
        get_op_results_or_values,
    )
    from ._torch_ops_ext_custom import *
    from pi._mlir import (
        is_a_TorchListOfTorchBoolType,
        is_a_TorchListOfTorchIntType,
        is_a_TorchListOfTorchStringType,
        is_a_TorchListOfValueTensorType,
        is_a_TorchScalarType,
        is_a_Torch_AnyType,
        is_a_Torch_BoolType,
        is_a_Torch_DeviceType,
        is_a_Torch_DictType,
        is_a_Torch_FloatType,
        is_a_Torch_GeneratorType,
        is_a_Torch_IntType,
        is_a_Torch_StringType,
        is_a_Torch_ValueTensorType,
        _TorchListOfTorchBoolType,
        _TorchListOfTorchFloatType,
        _TorchListOfTorchIntType,
        _TorchListOfTorchStringType,
        _TorchListOfValueTensorType,
        _Torch_AnyType,
        _Torch_BoolType,
        _Torch_DeviceType,
        _Torch_FloatType,
        _Torch_IntType,
        _Torch_NumberType,
        _Torch_StringType,
        _Torch_ValueTensorType,
        is_dtype,
    )

except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from numbers import Number
from typing import List, Optional, Any, Generator, Dict
Device = str


class AtenTanhOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTanhOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenTanh_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTanh_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardtanhOp:
    def __init__(self, self_: Value, min_val: Number, max_val: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(min_val):
            min_val = torch_dialect.ConstantNumberOp(min_val)
        else:
            min_val = get_op_result_or_value(min_val)
            assert is_a_TorchScalarType(min_val.type), f'`min_val` should be a TorchScalarType but is {type(min_val)}'
            
        if not is_mlir_value(max_val):
            max_val = torch_dialect.ConstantNumberOp(max_val)
        else:
            max_val = get_op_result_or_value(max_val)
            assert is_a_TorchScalarType(max_val.type), f'`max_val` should be a TorchScalarType but is {type(max_val)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenHardtanhOp, self).__init__(result_type, self_, min_val, max_val, loc=loc, ip=ip)
        
    
class AtenHardtanh_Op:
    def __init__(self, self_: Value, min_val: Number, max_val: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(min_val):
            min_val = torch_dialect.ConstantNumberOp(min_val)
        else:
            min_val = get_op_result_or_value(min_val)
            assert is_a_TorchScalarType(min_val.type), f'`min_val` should be a TorchScalarType but is {type(min_val)}'
            
        if not is_mlir_value(max_val):
            max_val = torch_dialect.ConstantNumberOp(max_val)
        else:
            max_val = get_op_result_or_value(max_val)
            assert is_a_TorchScalarType(max_val.type), f'`max_val` should be a TorchScalarType but is {type(max_val)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenHardtanh_Op, self).__init__(result_type, self_, min_val, max_val, loc=loc, ip=ip)
        
    
class AtenReluOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenReluOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRelu_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRelu_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRelu6Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRelu6Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRelu6_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRelu6_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLeakyReluOp:
    def __init__(self, self_: Value, negative_slope: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(negative_slope):
            negative_slope = torch_dialect.ConstantNumberOp(negative_slope)
        else:
            negative_slope = get_op_result_or_value(negative_slope)
            assert is_a_TorchScalarType(negative_slope.type), f'`negative_slope` should be a TorchScalarType but is {type(negative_slope)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLeakyReluOp, self).__init__(result_type, self_, negative_slope, loc=loc, ip=ip)
        
    
class AtenLeakyRelu_Op:
    def __init__(self, self_: Value, negative_slope: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(negative_slope):
            negative_slope = torch_dialect.ConstantNumberOp(negative_slope)
        else:
            negative_slope = get_op_result_or_value(negative_slope)
            assert is_a_TorchScalarType(negative_slope.type), f'`negative_slope` should be a TorchScalarType but is {type(negative_slope)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLeakyRelu_Op, self).__init__(result_type, self_, negative_slope, loc=loc, ip=ip)
        
    
class AtenLogOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLog_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLog_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSigmoidOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSigmoidOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSigmoid_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSigmoid_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardsigmoidOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenHardsigmoidOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardsigmoid_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenHardsigmoid_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardswishOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenHardswishOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardswish_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenHardswish_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenErfOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenErfOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenErf_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenErf_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSiluOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSiluOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSilu_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSilu_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSinOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSinOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSin_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSin_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenExpOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenExpOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenExp_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenExp_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenExpm1Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenExpm1Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenExpm1_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenExpm1_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenCosOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCosOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenCos_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCos_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAtan2Op:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAtan2Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenAtan2_Op:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAtan2_Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNegOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNegOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenNeg_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNeg_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFloorOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFloorOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFloor_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFloor_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenCeilOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCeilOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenCeil_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCeil_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBitwiseNotOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBitwiseNotOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBitwiseNot_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBitwiseNot_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenDivTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDivTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenDiv_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDiv_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalOrOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogicalOrOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalOr_Op:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogicalOr_Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalAndOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogicalAndOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalAnd_Op:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogicalAnd_Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalXorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogicalXorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalXor_Op:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogicalXor_Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalNotOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogicalNotOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLogicalNot_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogicalNot_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLerpTensorOp:
    def __init__(self, self_: Value, end: Value, weight: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(end):
            assert is_mlir_value(end), f'`end` should be a Value but is {type(end)}'
        else:
            end = get_op_result_or_value(end)
            assert is_a_Torch_ValueTensorType(end.type), f'`end` should be a Torch_ValueTensorType but is {type(end)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLerpTensorOp, self).__init__(result_type, self_, end, weight, loc=loc, ip=ip)
        
    
class AtenLerp_TensorOp:
    def __init__(self, self_: Value, end: Value, weight: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(end):
            assert is_mlir_value(end), f'`end` should be a Value but is {type(end)}'
        else:
            end = get_op_result_or_value(end)
            assert is_a_Torch_ValueTensorType(end.type), f'`end` should be a Torch_ValueTensorType but is {type(end)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLerp_TensorOp, self).__init__(result_type, self_, end, weight, loc=loc, ip=ip)
        
    
class AtenEqTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenEqTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenEq_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenEq_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGtTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGtTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGt_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGt_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGeTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGeTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGe_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGe_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLtTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLtTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLt_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLt_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLeTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLeTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLe_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLe_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNeTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNeTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNe_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNe_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenDivScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDivScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenDiv_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDiv_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNeScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNeScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNe_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNe_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenEqScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenEqScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenEq_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenEq_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGtScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGtScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGt_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGt_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGeScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGeScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGe_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGe_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLtScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLtScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLt_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLt_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLeScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLeScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLe_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLe_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenFmodScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFmodScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenFmod_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFmod_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMaskedFillScalarOp:
    def __init__(self, self_: Value, mask: Value, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'`mask` should be a Value but is {type(mask)}'
        else:
            mask = get_op_result_or_value(mask)
            assert is_a_Torch_ValueTensorType(mask.type), f'`mask` should be a Torch_ValueTensorType but is {type(mask)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaskedFillScalarOp, self).__init__(result_type, self_, mask, value, loc=loc, ip=ip)
        
    
class AtenMaskedFill_ScalarOp:
    def __init__(self, self_: Value, mask: Value, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'`mask` should be a Value but is {type(mask)}'
        else:
            mask = get_op_result_or_value(mask)
            assert is_a_Torch_ValueTensorType(mask.type), f'`mask` should be a Torch_ValueTensorType but is {type(mask)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaskedFill_ScalarOp, self).__init__(result_type, self_, mask, value, loc=loc, ip=ip)
        
    
class AtenMaskedFillTensorOp:
    def __init__(self, self_: Value, mask: Value, value: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'`mask` should be a Value but is {type(mask)}'
        else:
            mask = get_op_result_or_value(mask)
            assert is_a_Torch_ValueTensorType(mask.type), f'`mask` should be a Torch_ValueTensorType but is {type(mask)}'
            
        if not is_mlir_value(value):
            assert is_mlir_value(value), f'`value` should be a Value but is {type(value)}'
        else:
            value = get_op_result_or_value(value)
            assert is_a_Torch_ValueTensorType(value.type), f'`value` should be a Torch_ValueTensorType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaskedFillTensorOp, self).__init__(result_type, self_, mask, value, loc=loc, ip=ip)
        
    
class AtenMaskedFill_TensorOp:
    def __init__(self, self_: Value, mask: Value, value: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'`mask` should be a Value but is {type(mask)}'
        else:
            mask = get_op_result_or_value(mask)
            assert is_a_Torch_ValueTensorType(mask.type), f'`mask` should be a Torch_ValueTensorType but is {type(mask)}'
            
        if not is_mlir_value(value):
            assert is_mlir_value(value), f'`value` should be a Value but is {type(value)}'
        else:
            value = get_op_result_or_value(value)
            assert is_a_Torch_ValueTensorType(value.type), f'`value` should be a Torch_ValueTensorType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaskedFill_TensorOp, self).__init__(result_type, self_, mask, value, loc=loc, ip=ip)
        
    
class AtenClampOp:
    def __init__(self, self_: Value, min: Optional[Number], max: Optional[Number], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(min):
            if min is not None:
                min = torch_dialect.ConstantNumberOp(min)
            else:
                min = torch_dialect.ConstantNoneOp()
        else:
            min = get_op_result_or_value(min)
            assert is_a_TorchScalarType(min.type), f'`min` should be a TorchScalarType but is {type(min)}'
            
        if not is_mlir_value(max):
            if max is not None:
                max = torch_dialect.ConstantNumberOp(max)
            else:
                max = torch_dialect.ConstantNoneOp()
        else:
            max = get_op_result_or_value(max)
            assert is_a_TorchScalarType(max.type), f'`max` should be a TorchScalarType but is {type(max)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenClampOp, self).__init__(result_type, self_, min, max, loc=loc, ip=ip)
        
    
class AtenClamp_Op:
    def __init__(self, self_: Value, min: Optional[Number], max: Optional[Number], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(min):
            if min is not None:
                min = torch_dialect.ConstantNumberOp(min)
            else:
                min = torch_dialect.ConstantNoneOp()
        else:
            min = get_op_result_or_value(min)
            assert is_a_TorchScalarType(min.type), f'`min` should be a TorchScalarType but is {type(min)}'
            
        if not is_mlir_value(max):
            if max is not None:
                max = torch_dialect.ConstantNumberOp(max)
            else:
                max = torch_dialect.ConstantNoneOp()
        else:
            max = get_op_result_or_value(max)
            assert is_a_TorchScalarType(max.type), f'`max` should be a TorchScalarType but is {type(max)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenClamp_Op, self).__init__(result_type, self_, min, max, loc=loc, ip=ip)
        
    
class AtenClampMinOp:
    def __init__(self, self_: Value, min: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(min):
            min = torch_dialect.ConstantNumberOp(min)
        else:
            min = get_op_result_or_value(min)
            assert is_a_TorchScalarType(min.type), f'`min` should be a TorchScalarType but is {type(min)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenClampMinOp, self).__init__(result_type, self_, min, loc=loc, ip=ip)
        
    
class AtenClampMin_Op:
    def __init__(self, self_: Value, min: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(min):
            min = torch_dialect.ConstantNumberOp(min)
        else:
            min = get_op_result_or_value(min)
            assert is_a_TorchScalarType(min.type), f'`min` should be a TorchScalarType but is {type(min)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenClampMin_Op, self).__init__(result_type, self_, min, loc=loc, ip=ip)
        
    
class AtenClampMaxOp:
    def __init__(self, self_: Value, max: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(max):
            max = torch_dialect.ConstantNumberOp(max)
        else:
            max = get_op_result_or_value(max)
            assert is_a_TorchScalarType(max.type), f'`max` should be a TorchScalarType but is {type(max)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenClampMaxOp, self).__init__(result_type, self_, max, loc=loc, ip=ip)
        
    
class AtenClampMax_Op:
    def __init__(self, self_: Value, max: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(max):
            max = torch_dialect.ConstantNumberOp(max)
        else:
            max = get_op_result_or_value(max)
            assert is_a_TorchScalarType(max.type), f'`max` should be a TorchScalarType but is {type(max)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenClampMax_Op, self).__init__(result_type, self_, max, loc=loc, ip=ip)
        
    
class AtenLog2Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLog2Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLog2_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLog2_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSqrtOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSqrtOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSqrt_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSqrt_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLog1pOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLog1pOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLog1p_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLog1p_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRsqrtOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRsqrtOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRsqrt_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRsqrt_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAbsOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAbsOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAbs_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAbs_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenReciprocalOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenReciprocalOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenReciprocal_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenReciprocal_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBitwiseAndTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBitwiseAndTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenBitwiseAnd_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBitwiseAnd_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenBitwiseOrTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBitwiseOrTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenBitwiseOr_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBitwiseOr_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenThresholdOp:
    def __init__(self, self_: Value, threshold: Number, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(threshold):
            threshold = torch_dialect.ConstantNumberOp(threshold)
        else:
            threshold = get_op_result_or_value(threshold)
            assert is_a_TorchScalarType(threshold.type), f'`threshold` should be a TorchScalarType but is {type(threshold)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenThresholdOp, self).__init__(result_type, self_, threshold, value, loc=loc, ip=ip)
        
    
class AtenThreshold_Op:
    def __init__(self, self_: Value, threshold: Number, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(threshold):
            threshold = torch_dialect.ConstantNumberOp(threshold)
        else:
            threshold = get_op_result_or_value(threshold)
            assert is_a_TorchScalarType(threshold.type), f'`threshold` should be a TorchScalarType but is {type(threshold)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenThreshold_Op, self).__init__(result_type, self_, threshold, value, loc=loc, ip=ip)
        
    
class AtenSquareOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSquareOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSquare_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSquare_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenUnsqueezeOp:
    def __init__(self, self_: Value, dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenUnsqueezeOp, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenUnsqueeze_Op:
    def __init__(self, self_: Value, dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenUnsqueeze_Op, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenZeroOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenZeroOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenZero_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenZero_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFillScalarOp:
    def __init__(self, self_: Value, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFillScalarOp, self).__init__(result_type, self_, value, loc=loc, ip=ip)
        
    
class AtenFill_ScalarOp:
    def __init__(self, self_: Value, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFill_ScalarOp, self).__init__(result_type, self_, value, loc=loc, ip=ip)
        
    
class AtenFillTensorOp:
    def __init__(self, self_: Value, value: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(value):
            assert is_mlir_value(value), f'`value` should be a Value but is {type(value)}'
        else:
            value = get_op_result_or_value(value)
            assert is_a_Torch_ValueTensorType(value.type), f'`value` should be a Torch_ValueTensorType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFillTensorOp, self).__init__(result_type, self_, value, loc=loc, ip=ip)
        
    
class AtenFill_TensorOp:
    def __init__(self, self_: Value, value: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(value):
            assert is_mlir_value(value), f'`value` should be a Value but is {type(value)}'
        else:
            value = get_op_result_or_value(value)
            assert is_a_Torch_ValueTensorType(value.type), f'`value` should be a Torch_ValueTensorType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFill_TensorOp, self).__init__(result_type, self_, value, loc=loc, ip=ip)
        
    
class AtenDivTensorModeOp:
    def __init__(self, self_: Value, other: Value, rounding_mode: Optional[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        if not is_mlir_value(rounding_mode):
            if rounding_mode is not None:
                rounding_mode = torch_dialect.ConstantStrOp(rounding_mode)
            else:
                rounding_mode = torch_dialect.ConstantNoneOp()
        else:
            rounding_mode = get_op_result_or_value(rounding_mode)
            assert is_a_Torch_StringType(rounding_mode.type), f'`rounding_mode` should be a Torch_StringType but is {type(rounding_mode)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDivTensorModeOp, self).__init__(result_type, self_, other, rounding_mode, loc=loc, ip=ip)
        
    
class AtenDiv_TensorModeOp:
    def __init__(self, self_: Value, other: Value, rounding_mode: Optional[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        if not is_mlir_value(rounding_mode):
            if rounding_mode is not None:
                rounding_mode = torch_dialect.ConstantStrOp(rounding_mode)
            else:
                rounding_mode = torch_dialect.ConstantNoneOp()
        else:
            rounding_mode = get_op_result_or_value(rounding_mode)
            assert is_a_Torch_StringType(rounding_mode.type), f'`rounding_mode` should be a Torch_StringType but is {type(rounding_mode)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDiv_TensorModeOp, self).__init__(result_type, self_, other, rounding_mode, loc=loc, ip=ip)
        
    
class AtenMulTensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMulTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMul_TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMul_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenAddTensorOp:
    def __init__(self, self_: Value, other: Value, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAddTensorOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenAdd_TensorOp:
    def __init__(self, self_: Value, other: Value, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAdd_TensorOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenSubTensorOp:
    def __init__(self, self_: Value, other: Value, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSubTensorOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenSub_TensorOp:
    def __init__(self, self_: Value, other: Value, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSub_TensorOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenAddScalarOp:
    def __init__(self, self_: Value, other: Number, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAddScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenAdd_ScalarOp:
    def __init__(self, self_: Value, other: Number, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAdd_ScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenSubScalarOp:
    def __init__(self, self_: Value, other: Number, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSubScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenSub_ScalarOp:
    def __init__(self, self_: Value, other: Number, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSub_ScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenMulScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMulScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMul_ScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMul_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenAddcmulOp:
    def __init__(self, self_: Value, tensor1: Value, tensor2: Value, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(tensor1):
            assert is_mlir_value(tensor1), f'`tensor1` should be a Value but is {type(tensor1)}'
        else:
            tensor1 = get_op_result_or_value(tensor1)
            assert is_a_Torch_ValueTensorType(tensor1.type), f'`tensor1` should be a Torch_ValueTensorType but is {type(tensor1)}'
            
        if not is_mlir_value(tensor2):
            assert is_mlir_value(tensor2), f'`tensor2` should be a Value but is {type(tensor2)}'
        else:
            tensor2 = get_op_result_or_value(tensor2)
            assert is_a_Torch_ValueTensorType(tensor2.type), f'`tensor2` should be a Torch_ValueTensorType but is {type(tensor2)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAddcmulOp, self).__init__(result_type, self_, tensor1, tensor2, value, loc=loc, ip=ip)
        
    
class AtenAddcmul_Op:
    def __init__(self, self_: Value, tensor1: Value, tensor2: Value, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(tensor1):
            assert is_mlir_value(tensor1), f'`tensor1` should be a Value but is {type(tensor1)}'
        else:
            tensor1 = get_op_result_or_value(tensor1)
            assert is_a_Torch_ValueTensorType(tensor1.type), f'`tensor1` should be a Torch_ValueTensorType but is {type(tensor1)}'
            
        if not is_mlir_value(tensor2):
            assert is_mlir_value(tensor2), f'`tensor2` should be a Value but is {type(tensor2)}'
        else:
            tensor2 = get_op_result_or_value(tensor2)
            assert is_a_Torch_ValueTensorType(tensor2.type), f'`tensor2` should be a Torch_ValueTensorType but is {type(tensor2)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAddcmul_Op, self).__init__(result_type, self_, tensor1, tensor2, value, loc=loc, ip=ip)
        
    
class AtenAddcdivOp:
    def __init__(self, self_: Value, tensor1: Value, tensor2: Value, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(tensor1):
            assert is_mlir_value(tensor1), f'`tensor1` should be a Value but is {type(tensor1)}'
        else:
            tensor1 = get_op_result_or_value(tensor1)
            assert is_a_Torch_ValueTensorType(tensor1.type), f'`tensor1` should be a Torch_ValueTensorType but is {type(tensor1)}'
            
        if not is_mlir_value(tensor2):
            assert is_mlir_value(tensor2), f'`tensor2` should be a Value but is {type(tensor2)}'
        else:
            tensor2 = get_op_result_or_value(tensor2)
            assert is_a_Torch_ValueTensorType(tensor2.type), f'`tensor2` should be a Torch_ValueTensorType but is {type(tensor2)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAddcdivOp, self).__init__(result_type, self_, tensor1, tensor2, value, loc=loc, ip=ip)
        
    
class AtenAddcdiv_Op:
    def __init__(self, self_: Value, tensor1: Value, tensor2: Value, value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(tensor1):
            assert is_mlir_value(tensor1), f'`tensor1` should be a Value but is {type(tensor1)}'
        else:
            tensor1 = get_op_result_or_value(tensor1)
            assert is_a_Torch_ValueTensorType(tensor1.type), f'`tensor1` should be a Torch_ValueTensorType but is {type(tensor1)}'
            
        if not is_mlir_value(tensor2):
            assert is_mlir_value(tensor2), f'`tensor2` should be a Value but is {type(tensor2)}'
        else:
            tensor2 = get_op_result_or_value(tensor2)
            assert is_a_Torch_ValueTensorType(tensor2.type), f'`tensor2` should be a Torch_ValueTensorType but is {type(tensor2)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAddcdiv_Op, self).__init__(result_type, self_, tensor1, tensor2, value, loc=loc, ip=ip)
        
    
class AtenMaximumOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaximumOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMinimumOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMinimumOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMishOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMishOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRsubScalarOp:
    def __init__(self, self_: Value, other: Number, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRsubScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenGeluOp:
    def __init__(self, self_: Value, approximate: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(approximate):
            approximate = torch_dialect.ConstantStrOp(approximate)
        else:
            approximate = get_op_result_or_value(approximate)
            assert is_a_Torch_StringType(approximate.type), f'`approximate` should be a Torch_StringType but is {type(approximate)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGeluOp, self).__init__(result_type, self_, approximate, loc=loc, ip=ip)
        
    
class AtenPowTensorScalarOp:
    def __init__(self, self_: Value, exponent: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(exponent):
            exponent = torch_dialect.ConstantNumberOp(exponent)
        else:
            exponent = get_op_result_or_value(exponent)
            assert is_a_TorchScalarType(exponent.type), f'`exponent` should be a TorchScalarType but is {type(exponent)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenPowTensorScalarOp, self).__init__(result_type, self_, exponent, loc=loc, ip=ip)
        
    
class AtenPowTensorTensorOp:
    def __init__(self, self_: Value, exponent: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(exponent):
            assert is_mlir_value(exponent), f'`exponent` should be a Value but is {type(exponent)}'
        else:
            exponent = get_op_result_or_value(exponent)
            assert is_a_Torch_ValueTensorType(exponent.type), f'`exponent` should be a Torch_ValueTensorType but is {type(exponent)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenPowTensorTensorOp, self).__init__(result_type, self_, exponent, loc=loc, ip=ip)
        
    
class AtenThresholdBackwardOp:
    def __init__(self, grad_output: Value, self_: Value, threshold: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(threshold):
            threshold = torch_dialect.ConstantNumberOp(threshold)
        else:
            threshold = get_op_result_or_value(threshold)
            assert is_a_TorchScalarType(threshold.type), f'`threshold` should be a TorchScalarType but is {type(threshold)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenThresholdBackwardOp, self).__init__(result_type, grad_output, self_, threshold, loc=loc, ip=ip)
        
    
class AtenFloorDivideOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFloorDivideOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenSoftplusOp:
    def __init__(self, self_: Value, beta: Number, threshold: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(beta):
            beta = torch_dialect.ConstantNumberOp(beta)
        else:
            beta = get_op_result_or_value(beta)
            assert is_a_TorchScalarType(beta.type), f'`beta` should be a TorchScalarType but is {type(beta)}'
            
        if not is_mlir_value(threshold):
            threshold = torch_dialect.ConstantNumberOp(threshold)
        else:
            threshold = get_op_result_or_value(threshold)
            assert is_a_TorchScalarType(threshold.type), f'`threshold` should be a TorchScalarType but is {type(threshold)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSoftplusOp, self).__init__(result_type, self_, beta, threshold, loc=loc, ip=ip)
        
    
class AtenPreluOp:
    def __init__(self, self_: Value, weight: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenPreluOp, self).__init__(result_type, self_, weight, loc=loc, ip=ip)
        
    
class AtenUniformOp:
    def __init__(self, self_: Value, from_: float, to: float, generator: Optional[Generator], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(from_):
            from_ = torch_dialect.ConstantFloatOp(from_)
        else:
            from_ = get_op_result_or_value(from_)
            assert is_a_Torch_FloatType(from_.type), f'`from_` should be a Torch_FloatType but is {type(from_)}'
            
        if not is_mlir_value(to):
            to = torch_dialect.ConstantFloatOp(to)
        else:
            to = get_op_result_or_value(to)
            assert is_a_Torch_FloatType(to.type), f'`to` should be a Torch_FloatType but is {type(to)}'
            
        if not is_mlir_value(generator):
            if generator is not None:
                assert is_mlir_value(generator), f'`generator` should be a Value but is {type(generator)}'
            else:
                generator = torch_dialect.ConstantNoneOp()
        else:
            generator = get_op_result_or_value(generator)
            assert is_a_Torch_GeneratorType(generator.type), f'`generator` should be a Torch_GeneratorType but is {type(generator)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenUniformOp, self).__init__(result_type, self_, from_, to, generator, loc=loc, ip=ip)
        
    
class AtenUniform_Op:
    def __init__(self, self_: Value, from_: float, to: float, generator: Optional[Generator], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(from_):
            from_ = torch_dialect.ConstantFloatOp(from_)
        else:
            from_ = get_op_result_or_value(from_)
            assert is_a_Torch_FloatType(from_.type), f'`from_` should be a Torch_FloatType but is {type(from_)}'
            
        if not is_mlir_value(to):
            to = torch_dialect.ConstantFloatOp(to)
        else:
            to = get_op_result_or_value(to)
            assert is_a_Torch_FloatType(to.type), f'`to` should be a Torch_FloatType but is {type(to)}'
            
        if not is_mlir_value(generator):
            if generator is not None:
                assert is_mlir_value(generator), f'`generator` should be a Value but is {type(generator)}'
            else:
                generator = torch_dialect.ConstantNoneOp()
        else:
            generator = get_op_result_or_value(generator)
            assert is_a_Torch_GeneratorType(generator.type), f'`generator` should be a Torch_GeneratorType but is {type(generator)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenUniform_Op, self).__init__(result_type, self_, from_, to, generator, loc=loc, ip=ip)
        
    
class AtenRandLikeOp:
    def __init__(self, self_: Value, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRandLikeOp, self).__init__(result_type, self_, dtype, layout, device, pin_memory, memory_format, loc=loc, ip=ip)
        
    
class AtenBernoulliOp:
    def __init__(self, self_: Value, generator: Optional[Generator], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(generator):
            if generator is not None:
                assert is_mlir_value(generator), f'`generator` should be a Value but is {type(generator)}'
            else:
                generator = torch_dialect.ConstantNoneOp()
        else:
            generator = get_op_result_or_value(generator)
            assert is_a_Torch_GeneratorType(generator.type), f'`generator` should be a Torch_GeneratorType but is {type(generator)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBernoulliOp, self).__init__(result_type, self_, generator, loc=loc, ip=ip)
        
    
class AtenBernoulli_FloatOp:
    def __init__(self, self_: Value, p: float, generator: Optional[Generator], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(p):
            p = torch_dialect.ConstantFloatOp(p)
        else:
            p = get_op_result_or_value(p)
            assert is_a_Torch_FloatType(p.type), f'`p` should be a Torch_FloatType but is {type(p)}'
            
        if not is_mlir_value(generator):
            if generator is not None:
                assert is_mlir_value(generator), f'`generator` should be a Value but is {type(generator)}'
            else:
                generator = torch_dialect.ConstantNoneOp()
        else:
            generator = get_op_result_or_value(generator)
            assert is_a_Torch_GeneratorType(generator.type), f'`generator` should be a Torch_GeneratorType but is {type(generator)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBernoulli_FloatOp, self).__init__(result_type, self_, p, generator, loc=loc, ip=ip)
        
    
class AtenRandintLowOp:
    def __init__(self, low: int, high: int, size: List[int], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(low):
            low = torch_dialect.ConstantIntOp(low)
        else:
            low = get_op_result_or_value(low)
            assert is_a_Torch_IntType(low.type), f'`low` should be a Torch_IntType but is {type(low)}'
            
        if not is_mlir_value(high):
            high = torch_dialect.ConstantIntOp(high)
        else:
            high = get_op_result_or_value(high)
            assert is_a_Torch_IntType(high.type), f'`high` should be a Torch_IntType but is {type(high)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRandintLowOp, self).__init__(result_type, low, high, size, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenBernoulliTensorOp:
    def __init__(self, self_: Value, p: Value, generator: Optional[Generator], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(p):
            assert is_mlir_value(p), f'`p` should be a Value but is {type(p)}'
        else:
            p = get_op_result_or_value(p)
            assert is_a_Torch_ValueTensorType(p.type), f'`p` should be a Torch_ValueTensorType but is {type(p)}'
            
        if not is_mlir_value(generator):
            if generator is not None:
                assert is_mlir_value(generator), f'`generator` should be a Value but is {type(generator)}'
            else:
                generator = torch_dialect.ConstantNoneOp()
        else:
            generator = get_op_result_or_value(generator)
            assert is_a_Torch_GeneratorType(generator.type), f'`generator` should be a Torch_GeneratorType but is {type(generator)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBernoulliTensorOp, self).__init__(result_type, self_, p, generator, loc=loc, ip=ip)
        
    
class AtenBernoulli_TensorOp:
    def __init__(self, self_: Value, p: Value, generator: Optional[Generator], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(p):
            assert is_mlir_value(p), f'`p` should be a Value but is {type(p)}'
        else:
            p = get_op_result_or_value(p)
            assert is_a_Torch_ValueTensorType(p.type), f'`p` should be a Torch_ValueTensorType but is {type(p)}'
            
        if not is_mlir_value(generator):
            if generator is not None:
                assert is_mlir_value(generator), f'`generator` should be a Value but is {type(generator)}'
            else:
                generator = torch_dialect.ConstantNoneOp()
        else:
            generator = get_op_result_or_value(generator)
            assert is_a_Torch_GeneratorType(generator.type), f'`generator` should be a Torch_GeneratorType but is {type(generator)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBernoulli_TensorOp, self).__init__(result_type, self_, p, generator, loc=loc, ip=ip)
        
    
class AtenRandnOp:
    def __init__(self, size: List[int], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRandnOp, self).__init__(result_type, size, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenRandnGeneratorOp:
    def __init__(self, size: List[int], generator: Optional[Generator], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(generator):
            if generator is not None:
                assert is_mlir_value(generator), f'`generator` should be a Value but is {type(generator)}'
            else:
                generator = torch_dialect.ConstantNoneOp()
        else:
            generator = get_op_result_or_value(generator)
            assert is_a_Torch_GeneratorType(generator.type), f'`generator` should be a Torch_GeneratorType but is {type(generator)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRandnGeneratorOp, self).__init__(result_type, size, generator, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenTriuOp:
    def __init__(self, self_: Value, diagonal: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(diagonal):
            diagonal = torch_dialect.ConstantIntOp(diagonal)
        else:
            diagonal = get_op_result_or_value(diagonal)
            assert is_a_Torch_IntType(diagonal.type), f'`diagonal` should be a Torch_IntType but is {type(diagonal)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTriuOp, self).__init__(result_type, self_, diagonal, loc=loc, ip=ip)
        
    
class AtenTriu_Op:
    def __init__(self, self_: Value, diagonal: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(diagonal):
            diagonal = torch_dialect.ConstantIntOp(diagonal)
        else:
            diagonal = get_op_result_or_value(diagonal)
            assert is_a_Torch_IntType(diagonal.type), f'`diagonal` should be a Torch_IntType but is {type(diagonal)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTriu_Op, self).__init__(result_type, self_, diagonal, loc=loc, ip=ip)
        
    
class AtenRoundOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRoundOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRound_Op:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRound_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenIndexPutOp:
    def __init__(self, self_: Value, indices: List[Optional[Value]], values: Value, accumulate: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_TorchListOfValueTensorType(indices.type), f'`indices` should be a TorchListOfValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(values):
            assert is_mlir_value(values), f'`values` should be a Value but is {type(values)}'
        else:
            values = get_op_result_or_value(values)
            assert is_a_Torch_ValueTensorType(values.type), f'`values` should be a Torch_ValueTensorType but is {type(values)}'
            
        if not is_mlir_value(accumulate):
            accumulate = torch_dialect.ConstantBoolOp(accumulate)
        else:
            accumulate = get_op_result_or_value(accumulate)
            assert is_a_Torch_BoolType(accumulate.type), f'`accumulate` should be a Torch_BoolType but is {type(accumulate)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenIndexPutOp, self).__init__(result_type, self_, indices, values, accumulate, loc=loc, ip=ip)
        
    
class AtenIndexPut_Op:
    def __init__(self, self_: Value, indices: List[Optional[Value]], values: Value, accumulate: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_TorchListOfValueTensorType(indices.type), f'`indices` should be a TorchListOfValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(values):
            assert is_mlir_value(values), f'`values` should be a Value but is {type(values)}'
        else:
            values = get_op_result_or_value(values)
            assert is_a_Torch_ValueTensorType(values.type), f'`values` should be a Torch_ValueTensorType but is {type(values)}'
            
        if not is_mlir_value(accumulate):
            accumulate = torch_dialect.ConstantBoolOp(accumulate)
        else:
            accumulate = get_op_result_or_value(accumulate)
            assert is_a_Torch_BoolType(accumulate.type), f'`accumulate` should be a Torch_BoolType but is {type(accumulate)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenIndexPut_Op, self).__init__(result_type, self_, indices, values, accumulate, loc=loc, ip=ip)
        
    
class AtenIndexPutHackedTwinOp:
    def __init__(self, self_: Value, indices: List[Value], values: Value, accumulate: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_TorchListOfValueTensorType(indices.type), f'`indices` should be a TorchListOfValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(values):
            assert is_mlir_value(values), f'`values` should be a Value but is {type(values)}'
        else:
            values = get_op_result_or_value(values)
            assert is_a_Torch_ValueTensorType(values.type), f'`values` should be a Torch_ValueTensorType but is {type(values)}'
            
        if not is_mlir_value(accumulate):
            accumulate = torch_dialect.ConstantBoolOp(accumulate)
        else:
            accumulate = get_op_result_or_value(accumulate)
            assert is_a_Torch_BoolType(accumulate.type), f'`accumulate` should be a Torch_BoolType but is {type(accumulate)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenIndexPutHackedTwinOp, self).__init__(result_type, self_, indices, values, accumulate, loc=loc, ip=ip)
        
    
class AtenIndexPut_HackedTwinOp:
    def __init__(self, self_: Value, indices: List[Value], values: Value, accumulate: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_TorchListOfValueTensorType(indices.type), f'`indices` should be a TorchListOfValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(values):
            assert is_mlir_value(values), f'`values` should be a Value but is {type(values)}'
        else:
            values = get_op_result_or_value(values)
            assert is_a_Torch_ValueTensorType(values.type), f'`values` should be a Torch_ValueTensorType but is {type(values)}'
            
        if not is_mlir_value(accumulate):
            accumulate = torch_dialect.ConstantBoolOp(accumulate)
        else:
            accumulate = get_op_result_or_value(accumulate)
            assert is_a_Torch_BoolType(accumulate.type), f'`accumulate` should be a Torch_BoolType but is {type(accumulate)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenIndexPut_HackedTwinOp, self).__init__(result_type, self_, indices, values, accumulate, loc=loc, ip=ip)
        
    
class AtenLinearOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLinearOp, self).__init__(result_type, input, weight, bias, loc=loc, ip=ip)
        
    
class AtenMmOp:
    def __init__(self, self_: Value, mat2: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(mat2):
            assert is_mlir_value(mat2), f'`mat2` should be a Value but is {type(mat2)}'
        else:
            mat2 = get_op_result_or_value(mat2)
            assert is_a_Torch_ValueTensorType(mat2.type), f'`mat2` should be a Torch_ValueTensorType but is {type(mat2)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMmOp, self).__init__(result_type, self_, mat2, loc=loc, ip=ip)
        
    
class AtenAddmmOp:
    def __init__(self, self_: Value, mat1: Value, mat2: Value, beta: Number, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(mat1):
            assert is_mlir_value(mat1), f'`mat1` should be a Value but is {type(mat1)}'
        else:
            mat1 = get_op_result_or_value(mat1)
            assert is_a_Torch_ValueTensorType(mat1.type), f'`mat1` should be a Torch_ValueTensorType but is {type(mat1)}'
            
        if not is_mlir_value(mat2):
            assert is_mlir_value(mat2), f'`mat2` should be a Value but is {type(mat2)}'
        else:
            mat2 = get_op_result_or_value(mat2)
            assert is_a_Torch_ValueTensorType(mat2.type), f'`mat2` should be a Torch_ValueTensorType but is {type(mat2)}'
            
        if not is_mlir_value(beta):
            beta = torch_dialect.ConstantNumberOp(beta)
        else:
            beta = get_op_result_or_value(beta)
            assert is_a_TorchScalarType(beta.type), f'`beta` should be a TorchScalarType but is {type(beta)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAddmmOp, self).__init__(result_type, self_, mat1, mat2, beta, alpha, loc=loc, ip=ip)
        
    
class AtenMatmulOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMatmulOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMvOp:
    def __init__(self, self_: Value, vec: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(vec):
            assert is_mlir_value(vec), f'`vec` should be a Value but is {type(vec)}'
        else:
            vec = get_op_result_or_value(vec)
            assert is_a_Torch_ValueTensorType(vec.type), f'`vec` should be a Torch_ValueTensorType but is {type(vec)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMvOp, self).__init__(result_type, self_, vec, loc=loc, ip=ip)
        
    
class AtenConv2dOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], stride: List[int], padding: List[int], dilation: List[int], groups: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenConv2dOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, groups, loc=loc, ip=ip)
        
    
class AtenConvTranspose1dOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], stride: List[int], padding: List[int], output_padding: List[int], groups: int, dilation: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenConvTranspose1dOp, self).__init__(result_type, input, weight, bias, stride, padding, output_padding, groups, dilation, loc=loc, ip=ip)
        
    
class AtenConvTranspose2dInputOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], stride: List[int], padding: List[int], output_padding: List[int], groups: int, dilation: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenConvTranspose2dInputOp, self).__init__(result_type, input, weight, bias, stride, padding, output_padding, groups, dilation, loc=loc, ip=ip)
        
    
class AtenConvTranspose3dInputOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], stride: List[int], padding: List[int], output_padding: List[int], groups: int, dilation: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenConvTranspose3dInputOp, self).__init__(result_type, input, weight, bias, stride, padding, output_padding, groups, dilation, loc=loc, ip=ip)
        
    
class AtenConvolutionOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert is_a_Torch_BoolType(transposed.type), f'`transposed` should be a Torch_BoolType but is {type(transposed)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenConvolutionOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, loc=loc, ip=ip)
        
    
class AtenConvolutionOverrideableOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert is_a_Torch_BoolType(transposed.type), f'`transposed` should be a Torch_BoolType but is {type(transposed)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenConvolutionOverrideableOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, loc=loc, ip=ip)
        
    
class Aten_ConvolutionOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, allow_tf32: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert is_a_Torch_BoolType(transposed.type), f'`transposed` should be a Torch_BoolType but is {type(transposed)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        if not is_mlir_value(benchmark):
            benchmark = torch_dialect.ConstantBoolOp(benchmark)
        else:
            benchmark = get_op_result_or_value(benchmark)
            assert is_a_Torch_BoolType(benchmark.type), f'`benchmark` should be a Torch_BoolType but is {type(benchmark)}'
            
        if not is_mlir_value(deterministic):
            deterministic = torch_dialect.ConstantBoolOp(deterministic)
        else:
            deterministic = get_op_result_or_value(deterministic)
            assert is_a_Torch_BoolType(deterministic.type), f'`deterministic` should be a Torch_BoolType but is {type(deterministic)}'
            
        if not is_mlir_value(cudnn_enabled):
            cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled)
        else:
            cudnn_enabled = get_op_result_or_value(cudnn_enabled)
            assert is_a_Torch_BoolType(cudnn_enabled.type), f'`cudnn_enabled` should be a Torch_BoolType but is {type(cudnn_enabled)}'
            
        if not is_mlir_value(allow_tf32):
            allow_tf32 = torch_dialect.ConstantBoolOp(allow_tf32)
        else:
            allow_tf32 = get_op_result_or_value(allow_tf32)
            assert is_a_Torch_BoolType(allow_tf32.type), f'`allow_tf32` should be a Torch_BoolType but is {type(allow_tf32)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_ConvolutionOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, loc=loc, ip=ip)
        
    
class Aten_ConvolutionDeprecatedOp:
    def __init__(self, input: Value, weight: Value, bias: Optional[Value], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert is_a_Torch_BoolType(transposed.type), f'`transposed` should be a Torch_BoolType but is {type(transposed)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        if not is_mlir_value(benchmark):
            benchmark = torch_dialect.ConstantBoolOp(benchmark)
        else:
            benchmark = get_op_result_or_value(benchmark)
            assert is_a_Torch_BoolType(benchmark.type), f'`benchmark` should be a Torch_BoolType but is {type(benchmark)}'
            
        if not is_mlir_value(deterministic):
            deterministic = torch_dialect.ConstantBoolOp(deterministic)
        else:
            deterministic = get_op_result_or_value(deterministic)
            assert is_a_Torch_BoolType(deterministic.type), f'`deterministic` should be a Torch_BoolType but is {type(deterministic)}'
            
        if not is_mlir_value(cudnn_enabled):
            cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled)
        else:
            cudnn_enabled = get_op_result_or_value(cudnn_enabled)
            assert is_a_Torch_BoolType(cudnn_enabled.type), f'`cudnn_enabled` should be a Torch_BoolType but is {type(cudnn_enabled)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_ConvolutionDeprecatedOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, loc=loc, ip=ip)
        
    
class AtenRollOp:
    def __init__(self, self_: Value, shifts: List[int], dims: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(shifts):
            shifts = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in shifts]
            shifts = torch_dialect.PrimListConstructOp(shifts)
        else:
            shifts = get_op_result_or_value(shifts)
            assert is_a_TorchListOfTorchIntType(shifts.type), f'`shifts` should be a TorchListOfTorchIntType but is {type(shifts)}'
            
        if not is_mlir_value(dims):
            dims = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dims]
            dims = torch_dialect.PrimListConstructOp(dims)
        else:
            dims = get_op_result_or_value(dims)
            assert is_a_TorchListOfTorchIntType(dims.type), f'`dims` should be a TorchListOfTorchIntType but is {type(dims)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRollOp, self).__init__(result_type, self_, shifts, dims, loc=loc, ip=ip)
        
    
class AtenConvolutionBackwardOp:
    def __init__(self, grad_output: Value, input: Value, weight: Value, bias_sizes: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias_sizes):
            if bias_sizes is not None:
                bias_sizes = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in bias_sizes]
                bias_sizes = torch_dialect.PrimListConstructOp(bias_sizes)
            else:
                bias_sizes = torch_dialect.ConstantNoneOp()
        else:
            bias_sizes = get_op_result_or_value(bias_sizes)
            assert is_a_TorchListOfTorchIntType(bias_sizes.type), f'`bias_sizes` should be a TorchListOfTorchIntType but is {type(bias_sizes)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert is_a_Torch_BoolType(transposed.type), f'`transposed` should be a Torch_BoolType but is {type(transposed)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        if not is_mlir_value(output_mask):
            output_mask = [torch_dialect.ConstantBoolOp(a) if not is_mlir_value(a) else a for a in output_mask]
            output_mask = torch_dialect.PrimListConstructOp(output_mask)
        else:
            output_mask = get_op_result_or_value(output_mask)
            assert is_a_TorchListOfTorchBoolType(output_mask.type), f'`output_mask` should be a TorchListOfTorchBoolType but is {type(output_mask)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        result2_type = _Torch_ValueTensorType()
        super(AtenConvolutionBackwardOp, self).__init__(result0_type, result1_type, result2_type, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask, loc=loc, ip=ip)
        
    
class AtenConvolutionBackwardOverrideableOp:
    def __init__(self, grad_output: Value, input: Value, weight: Value, stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert is_a_Torch_BoolType(transposed.type), f'`transposed` should be a Torch_BoolType but is {type(transposed)}'
            
        if not is_mlir_value(output_padding):
            output_padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_padding]
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert is_a_TorchListOfTorchIntType(output_padding.type), f'`output_padding` should be a TorchListOfTorchIntType but is {type(output_padding)}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert is_a_Torch_IntType(groups.type), f'`groups` should be a Torch_IntType but is {type(groups)}'
            
        if not is_mlir_value(output_mask):
            output_mask = [torch_dialect.ConstantBoolOp(a) if not is_mlir_value(a) else a for a in output_mask]
            output_mask = torch_dialect.PrimListConstructOp(output_mask)
        else:
            output_mask = get_op_result_or_value(output_mask)
            assert is_a_TorchListOfTorchBoolType(output_mask.type), f'`output_mask` should be a TorchListOfTorchBoolType but is {type(output_mask)}'
            
        grad_input_type = _Torch_ValueTensorType()
        grad_weight_type = _Torch_ValueTensorType()
        grad_bias_type = _Torch_ValueTensorType()
        super(AtenConvolutionBackwardOverrideableOp, self).__init__(grad_input_type, grad_weight_type, grad_bias_type, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask, loc=loc, ip=ip)
        
    
class AtenFlipOp:
    def __init__(self, self_: Value, dims: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dims):
            dims = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dims]
            dims = torch_dialect.PrimListConstructOp(dims)
        else:
            dims = get_op_result_or_value(dims)
            assert is_a_TorchListOfTorchIntType(dims.type), f'`dims` should be a TorchListOfTorchIntType but is {type(dims)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFlipOp, self).__init__(result_type, self_, dims, loc=loc, ip=ip)
        
    
class AtenNativeBatchNormOp:
    def __init__(self, input: Value, weight: Optional[Value], bias: Optional[Value], running_mean: Optional[Value], running_var: Optional[Value], training: bool, momentum: float, eps: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(running_mean):
            if running_mean is not None:
                assert is_mlir_value(running_mean), f'`running_mean` should be a Value but is {type(running_mean)}'
            else:
                running_mean = torch_dialect.ConstantNoneOp()
        else:
            running_mean = get_op_result_or_value(running_mean)
            assert is_a_Torch_ValueTensorType(running_mean.type), f'`running_mean` should be a Torch_ValueTensorType but is {type(running_mean)}'
            
        if not is_mlir_value(running_var):
            if running_var is not None:
                assert is_mlir_value(running_var), f'`running_var` should be a Value but is {type(running_var)}'
            else:
                running_var = torch_dialect.ConstantNoneOp()
        else:
            running_var = get_op_result_or_value(running_var)
            assert is_a_Torch_ValueTensorType(running_var.type), f'`running_var` should be a Torch_ValueTensorType but is {type(running_var)}'
            
        if not is_mlir_value(training):
            training = torch_dialect.ConstantBoolOp(training)
        else:
            training = get_op_result_or_value(training)
            assert is_a_Torch_BoolType(training.type), f'`training` should be a Torch_BoolType but is {type(training)}'
            
        if not is_mlir_value(momentum):
            momentum = torch_dialect.ConstantFloatOp(momentum)
        else:
            momentum = get_op_result_or_value(momentum)
            assert is_a_Torch_FloatType(momentum.type), f'`momentum` should be a Torch_FloatType but is {type(momentum)}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert is_a_Torch_FloatType(eps.type), f'`eps` should be a Torch_FloatType but is {type(eps)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        result2_type = _Torch_ValueTensorType()
        super(AtenNativeBatchNormOp, self).__init__(result0_type, result1_type, result2_type, input, weight, bias, running_mean, running_var, training, momentum, eps, loc=loc, ip=ip)
        
    
class AtenBatchNormOp:
    def __init__(self, input: Value, weight: Optional[Value], bias: Optional[Value], running_mean: Optional[Value], running_var: Optional[Value], training: bool, momentum: float, eps: float, cudnn_enabled: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(running_mean):
            if running_mean is not None:
                assert is_mlir_value(running_mean), f'`running_mean` should be a Value but is {type(running_mean)}'
            else:
                running_mean = torch_dialect.ConstantNoneOp()
        else:
            running_mean = get_op_result_or_value(running_mean)
            assert is_a_Torch_ValueTensorType(running_mean.type), f'`running_mean` should be a Torch_ValueTensorType but is {type(running_mean)}'
            
        if not is_mlir_value(running_var):
            if running_var is not None:
                assert is_mlir_value(running_var), f'`running_var` should be a Value but is {type(running_var)}'
            else:
                running_var = torch_dialect.ConstantNoneOp()
        else:
            running_var = get_op_result_or_value(running_var)
            assert is_a_Torch_ValueTensorType(running_var.type), f'`running_var` should be a Torch_ValueTensorType but is {type(running_var)}'
            
        if not is_mlir_value(training):
            training = torch_dialect.ConstantBoolOp(training)
        else:
            training = get_op_result_or_value(training)
            assert is_a_Torch_BoolType(training.type), f'`training` should be a Torch_BoolType but is {type(training)}'
            
        if not is_mlir_value(momentum):
            momentum = torch_dialect.ConstantFloatOp(momentum)
        else:
            momentum = get_op_result_or_value(momentum)
            assert is_a_Torch_FloatType(momentum.type), f'`momentum` should be a Torch_FloatType but is {type(momentum)}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert is_a_Torch_FloatType(eps.type), f'`eps` should be a Torch_FloatType but is {type(eps)}'
            
        if not is_mlir_value(cudnn_enabled):
            cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled)
        else:
            cudnn_enabled = get_op_result_or_value(cudnn_enabled)
            assert is_a_Torch_BoolType(cudnn_enabled.type), f'`cudnn_enabled` should be a Torch_BoolType but is {type(cudnn_enabled)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBatchNormOp, self).__init__(result_type, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled, loc=loc, ip=ip)
        
    
class AtenLayerNormOp:
    def __init__(self, input: Value, normalized_shape: List[int], weight: Optional[Value], bias: Optional[Value], eps: float, cudnn_enable: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(normalized_shape):
            normalized_shape = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in normalized_shape]
            normalized_shape = torch_dialect.PrimListConstructOp(normalized_shape)
        else:
            normalized_shape = get_op_result_or_value(normalized_shape)
            assert is_a_TorchListOfTorchIntType(normalized_shape.type), f'`normalized_shape` should be a TorchListOfTorchIntType but is {type(normalized_shape)}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert is_a_Torch_FloatType(eps.type), f'`eps` should be a Torch_FloatType but is {type(eps)}'
            
        if not is_mlir_value(cudnn_enable):
            cudnn_enable = torch_dialect.ConstantBoolOp(cudnn_enable)
        else:
            cudnn_enable = get_op_result_or_value(cudnn_enable)
            assert is_a_Torch_BoolType(cudnn_enable.type), f'`cudnn_enable` should be a Torch_BoolType but is {type(cudnn_enable)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLayerNormOp, self).__init__(result_type, input, normalized_shape, weight, bias, eps, cudnn_enable, loc=loc, ip=ip)
        
    
class AtenNativeLayerNormOp:
    def __init__(self, input: Value, normalized_shape: List[int], weight: Optional[Value], bias: Optional[Value], eps: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(normalized_shape):
            normalized_shape = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in normalized_shape]
            normalized_shape = torch_dialect.PrimListConstructOp(normalized_shape)
        else:
            normalized_shape = get_op_result_or_value(normalized_shape)
            assert is_a_TorchListOfTorchIntType(normalized_shape.type), f'`normalized_shape` should be a TorchListOfTorchIntType but is {type(normalized_shape)}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert is_a_Torch_FloatType(eps.type), f'`eps` should be a Torch_FloatType but is {type(eps)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        result2_type = _Torch_ValueTensorType()
        super(AtenNativeLayerNormOp, self).__init__(result0_type, result1_type, result2_type, input, normalized_shape, weight, bias, eps, loc=loc, ip=ip)
        
    
class AtenMaxPool2dOp:
    def __init__(self, self_: Value, kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(kernel_size):
            kernel_size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in kernel_size]
            kernel_size = torch_dialect.PrimListConstructOp(kernel_size)
        else:
            kernel_size = get_op_result_or_value(kernel_size)
            assert is_a_TorchListOfTorchIntType(kernel_size.type), f'`kernel_size` should be a TorchListOfTorchIntType but is {type(kernel_size)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(ceil_mode):
            ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode)
        else:
            ceil_mode = get_op_result_or_value(ceil_mode)
            assert is_a_Torch_BoolType(ceil_mode.type), f'`ceil_mode` should be a Torch_BoolType but is {type(ceil_mode)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaxPool2dOp, self).__init__(result_type, self_, kernel_size, stride, padding, dilation, ceil_mode, loc=loc, ip=ip)
        
    
class AtenMaxPool2dWithIndicesOp:
    def __init__(self, self_: Value, kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(kernel_size):
            kernel_size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in kernel_size]
            kernel_size = torch_dialect.PrimListConstructOp(kernel_size)
        else:
            kernel_size = get_op_result_or_value(kernel_size)
            assert is_a_TorchListOfTorchIntType(kernel_size.type), f'`kernel_size` should be a TorchListOfTorchIntType but is {type(kernel_size)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(ceil_mode):
            ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode)
        else:
            ceil_mode = get_op_result_or_value(ceil_mode)
            assert is_a_Torch_BoolType(ceil_mode.type), f'`ceil_mode` should be a Torch_BoolType but is {type(ceil_mode)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        super(AtenMaxPool2dWithIndicesOp, self).__init__(result0_type, result1_type, self_, kernel_size, stride, padding, dilation, ceil_mode, loc=loc, ip=ip)
        
    
class AtenMaxPool2dWithIndicesBackwardOp:
    def __init__(self, grad_output: Value, self_: Value, kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, indices: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(kernel_size):
            kernel_size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in kernel_size]
            kernel_size = torch_dialect.PrimListConstructOp(kernel_size)
        else:
            kernel_size = get_op_result_or_value(kernel_size)
            assert is_a_TorchListOfTorchIntType(kernel_size.type), f'`kernel_size` should be a TorchListOfTorchIntType but is {type(kernel_size)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(dilation):
            dilation = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dilation]
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert is_a_TorchListOfTorchIntType(dilation.type), f'`dilation` should be a TorchListOfTorchIntType but is {type(dilation)}'
            
        if not is_mlir_value(ceil_mode):
            ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode)
        else:
            ceil_mode = get_op_result_or_value(ceil_mode)
            assert is_a_Torch_BoolType(ceil_mode.type), f'`ceil_mode` should be a Torch_BoolType but is {type(ceil_mode)}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'`indices` should be a Value but is {type(indices)}'
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_Torch_ValueTensorType(indices.type), f'`indices` should be a Torch_ValueTensorType but is {type(indices)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaxPool2dWithIndicesBackwardOp, self).__init__(result_type, grad_output, self_, kernel_size, stride, padding, dilation, ceil_mode, indices, loc=loc, ip=ip)
        
    
class AtenAvgPool2dOp:
    def __init__(self, self_: Value, kernel_size: List[int], stride: List[int], padding: List[int], ceil_mode: bool, count_include_pad: bool, divisor_override: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(kernel_size):
            kernel_size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in kernel_size]
            kernel_size = torch_dialect.PrimListConstructOp(kernel_size)
        else:
            kernel_size = get_op_result_or_value(kernel_size)
            assert is_a_TorchListOfTorchIntType(kernel_size.type), f'`kernel_size` should be a TorchListOfTorchIntType but is {type(kernel_size)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(padding):
            padding = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in padding]
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert is_a_TorchListOfTorchIntType(padding.type), f'`padding` should be a TorchListOfTorchIntType but is {type(padding)}'
            
        if not is_mlir_value(ceil_mode):
            ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode)
        else:
            ceil_mode = get_op_result_or_value(ceil_mode)
            assert is_a_Torch_BoolType(ceil_mode.type), f'`ceil_mode` should be a Torch_BoolType but is {type(ceil_mode)}'
            
        if not is_mlir_value(count_include_pad):
            count_include_pad = torch_dialect.ConstantBoolOp(count_include_pad)
        else:
            count_include_pad = get_op_result_or_value(count_include_pad)
            assert is_a_Torch_BoolType(count_include_pad.type), f'`count_include_pad` should be a Torch_BoolType but is {type(count_include_pad)}'
            
        if not is_mlir_value(divisor_override):
            if divisor_override is not None:
                divisor_override = torch_dialect.ConstantIntOp(divisor_override)
            else:
                divisor_override = torch_dialect.ConstantNoneOp()
        else:
            divisor_override = get_op_result_or_value(divisor_override)
            assert is_a_Torch_IntType(divisor_override.type), f'`divisor_override` should be a Torch_IntType but is {type(divisor_override)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAvgPool2dOp, self).__init__(result_type, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, loc=loc, ip=ip)
        
    
class AtenSoftmaxIntOp:
    def __init__(self, self_: Value, dim: int, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSoftmaxIntOp, self).__init__(result_type, self_, dim, dtype, loc=loc, ip=ip)
        
    
class AtenLogSoftmaxIntOp:
    def __init__(self, self_: Value, dim: int, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogSoftmaxIntOp, self).__init__(result_type, self_, dim, dtype, loc=loc, ip=ip)
        
    
class Aten_LogSoftmaxOp:
    def __init__(self, self_: Value, dim: int, half_to_float: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(half_to_float):
            half_to_float = torch_dialect.ConstantBoolOp(half_to_float)
        else:
            half_to_float = get_op_result_or_value(half_to_float)
            assert is_a_Torch_BoolType(half_to_float.type), f'`half_to_float` should be a Torch_BoolType but is {type(half_to_float)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_LogSoftmaxOp, self).__init__(result_type, self_, dim, half_to_float, loc=loc, ip=ip)
        
    
class AtenAdaptiveAvgPool2dOp:
    def __init__(self, self_: Value, output_size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(output_size):
            output_size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_size]
            output_size = torch_dialect.PrimListConstructOp(output_size)
        else:
            output_size = get_op_result_or_value(output_size)
            assert is_a_TorchListOfTorchIntType(output_size.type), f'`output_size` should be a TorchListOfTorchIntType but is {type(output_size)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAdaptiveAvgPool2dOp, self).__init__(result_type, self_, output_size, loc=loc, ip=ip)
        
    
class AtenTopkOp:
    def __init__(self, self_: Value, k: int, dim: int, largest: bool, sorted: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(k):
            k = torch_dialect.ConstantIntOp(k)
        else:
            k = get_op_result_or_value(k)
            assert is_a_Torch_IntType(k.type), f'`k` should be a Torch_IntType but is {type(k)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(largest):
            largest = torch_dialect.ConstantBoolOp(largest)
        else:
            largest = get_op_result_or_value(largest)
            assert is_a_Torch_BoolType(largest.type), f'`largest` should be a Torch_BoolType but is {type(largest)}'
            
        if not is_mlir_value(sorted):
            sorted = torch_dialect.ConstantBoolOp(sorted)
        else:
            sorted = get_op_result_or_value(sorted)
            assert is_a_Torch_BoolType(sorted.type), f'`sorted` should be a Torch_BoolType but is {type(sorted)}'
            
        values_type = _Torch_ValueTensorType()
        indices_type = _Torch_ValueTensorType()
        super(AtenTopkOp, self).__init__(values_type, indices_type, self_, k, dim, largest, sorted, loc=loc, ip=ip)
        
    
class AtenTransposeIntOp:
    def __init__(self, self_: Value, dim0: int, dim1: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim0):
            dim0 = torch_dialect.ConstantIntOp(dim0)
        else:
            dim0 = get_op_result_or_value(dim0)
            assert is_a_Torch_IntType(dim0.type), f'`dim0` should be a Torch_IntType but is {type(dim0)}'
            
        if not is_mlir_value(dim1):
            dim1 = torch_dialect.ConstantIntOp(dim1)
        else:
            dim1 = get_op_result_or_value(dim1)
            assert is_a_Torch_IntType(dim1.type), f'`dim1` should be a Torch_IntType but is {type(dim1)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTransposeIntOp, self).__init__(result_type, self_, dim0, dim1, loc=loc, ip=ip)
        
    
class AtenPermuteOp:
    def __init__(self, self_: Value, dims: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dims):
            dims = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dims]
            dims = torch_dialect.PrimListConstructOp(dims)
        else:
            dims = get_op_result_or_value(dims)
            assert is_a_TorchListOfTorchIntType(dims.type), f'`dims` should be a TorchListOfTorchIntType but is {type(dims)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenPermuteOp, self).__init__(result_type, self_, dims, loc=loc, ip=ip)
        
    
class AtenBmmOp:
    def __init__(self, self_: Value, mat2: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(mat2):
            assert is_mlir_value(mat2), f'`mat2` should be a Value but is {type(mat2)}'
        else:
            mat2 = get_op_result_or_value(mat2)
            assert is_a_Torch_ValueTensorType(mat2.type), f'`mat2` should be a Torch_ValueTensorType but is {type(mat2)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBmmOp, self).__init__(result_type, self_, mat2, loc=loc, ip=ip)
        
    
class AtenCumsumOp:
    def __init__(self, self_: Value, dim: int, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCumsumOp, self).__init__(result_type, self_, dim, dtype, loc=loc, ip=ip)
        
    
class AtenFloorDivideScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFloorDivideScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogsumexpOp:
    def __init__(self, self_: Value, dim: List[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
            dim = torch_dialect.PrimListConstructOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLogsumexpOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenMeanDimOp:
    def __init__(self, self_: Value, dim: Optional[List[int]], keepdim: bool, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
                dim = torch_dialect.PrimListConstructOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMeanDimOp, self).__init__(result_type, self_, dim, keepdim, dtype, loc=loc, ip=ip)
        
    
class Aten__And__TensorOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten__And__TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class Aten_SoftmaxOp:
    def __init__(self, self_: Value, dim: int, half_to_float: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(half_to_float):
            half_to_float = torch_dialect.ConstantBoolOp(half_to_float)
        else:
            half_to_float = get_op_result_or_value(half_to_float)
            assert is_a_Torch_BoolType(half_to_float.type), f'`half_to_float` should be a Torch_BoolType but is {type(half_to_float)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_SoftmaxOp, self).__init__(result_type, self_, dim, half_to_float, loc=loc, ip=ip)
        
    
class AtenMeanOp:
    def __init__(self, self_: Value, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMeanOp, self).__init__(result_type, self_, dtype, loc=loc, ip=ip)
        
    
class AtenStdOp:
    def __init__(self, self_: Value, unbiased: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(unbiased):
            unbiased = torch_dialect.ConstantBoolOp(unbiased)
        else:
            unbiased = get_op_result_or_value(unbiased)
            assert is_a_Torch_BoolType(unbiased.type), f'`unbiased` should be a Torch_BoolType but is {type(unbiased)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenStdOp, self).__init__(result_type, self_, unbiased, loc=loc, ip=ip)
        
    
class AtenStdDimOp:
    def __init__(self, self_: Value, dim: Optional[List[int]], unbiased: bool, keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
                dim = torch_dialect.PrimListConstructOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(unbiased):
            unbiased = torch_dialect.ConstantBoolOp(unbiased)
        else:
            unbiased = get_op_result_or_value(unbiased)
            assert is_a_Torch_BoolType(unbiased.type), f'`unbiased` should be a Torch_BoolType but is {type(unbiased)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenStdDimOp, self).__init__(result_type, self_, dim, unbiased, keepdim, loc=loc, ip=ip)
        
    
class AtenStdCorrectionOp:
    def __init__(self, self_: Value, dim: Optional[List[int]], correction: Optional[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
                dim = torch_dialect.PrimListConstructOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(correction):
            if correction is not None:
                correction = torch_dialect.ConstantIntOp(correction)
            else:
                correction = torch_dialect.ConstantNoneOp()
        else:
            correction = get_op_result_or_value(correction)
            assert is_a_Torch_IntType(correction.type), f'`correction` should be a Torch_IntType but is {type(correction)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenStdCorrectionOp, self).__init__(result_type, self_, dim, correction, keepdim, loc=loc, ip=ip)
        
    
class AtenVarOp:
    def __init__(self, self_: Value, unbiased: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(unbiased):
            unbiased = torch_dialect.ConstantBoolOp(unbiased)
        else:
            unbiased = get_op_result_or_value(unbiased)
            assert is_a_Torch_BoolType(unbiased.type), f'`unbiased` should be a Torch_BoolType but is {type(unbiased)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenVarOp, self).__init__(result_type, self_, unbiased, loc=loc, ip=ip)
        
    
class AtenVarDimOp:
    def __init__(self, self_: Value, dim: Optional[List[int]], unbiased: bool, keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
                dim = torch_dialect.PrimListConstructOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(unbiased):
            unbiased = torch_dialect.ConstantBoolOp(unbiased)
        else:
            unbiased = get_op_result_or_value(unbiased)
            assert is_a_Torch_BoolType(unbiased.type), f'`unbiased` should be a Torch_BoolType but is {type(unbiased)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenVarDimOp, self).__init__(result_type, self_, dim, unbiased, keepdim, loc=loc, ip=ip)
        
    
class AtenVarCorrectionOp:
    def __init__(self, self_: Value, dim: Optional[List[int]], correction: Optional[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
                dim = torch_dialect.PrimListConstructOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(correction):
            if correction is not None:
                correction = torch_dialect.ConstantIntOp(correction)
            else:
                correction = torch_dialect.ConstantNoneOp()
        else:
            correction = get_op_result_or_value(correction)
            assert is_a_Torch_IntType(correction.type), f'`correction` should be a Torch_IntType but is {type(correction)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenVarCorrectionOp, self).__init__(result_type, self_, dim, correction, keepdim, loc=loc, ip=ip)
        
    
class AtenVarMeanCorrectionOp:
    def __init__(self, self_: Value, dim: Optional[List[int]], correction: Optional[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
                dim = torch_dialect.PrimListConstructOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(correction):
            if correction is not None:
                correction = torch_dialect.ConstantIntOp(correction)
            else:
                correction = torch_dialect.ConstantNoneOp()
        else:
            correction = get_op_result_or_value(correction)
            assert is_a_Torch_IntType(correction.type), f'`correction` should be a Torch_IntType but is {type(correction)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        super(AtenVarMeanCorrectionOp, self).__init__(result0_type, result1_type, self_, dim, correction, keepdim, loc=loc, ip=ip)
        
    
class AtenVarMeanOp:
    def __init__(self, self_: Value, unbiased: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(unbiased):
            unbiased = torch_dialect.ConstantBoolOp(unbiased)
        else:
            unbiased = get_op_result_or_value(unbiased)
            assert is_a_Torch_BoolType(unbiased.type), f'`unbiased` should be a Torch_BoolType but is {type(unbiased)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        super(AtenVarMeanOp, self).__init__(result0_type, result1_type, self_, unbiased, loc=loc, ip=ip)
        
    
class AtenNllLossForwardOp:
    def __init__(self, self_: Value, target: Value, weight: Optional[Value], reduction: int, ignore_index: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(target):
            assert is_mlir_value(target), f'`target` should be a Value but is {type(target)}'
        else:
            target = get_op_result_or_value(target)
            assert is_a_Torch_ValueTensorType(target.type), f'`target` should be a Torch_ValueTensorType but is {type(target)}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(reduction):
            reduction = torch_dialect.ConstantIntOp(reduction)
        else:
            reduction = get_op_result_or_value(reduction)
            assert is_a_Torch_IntType(reduction.type), f'`reduction` should be a Torch_IntType but is {type(reduction)}'
            
        if not is_mlir_value(ignore_index):
            ignore_index = torch_dialect.ConstantIntOp(ignore_index)
        else:
            ignore_index = get_op_result_or_value(ignore_index)
            assert is_a_Torch_IntType(ignore_index.type), f'`ignore_index` should be a Torch_IntType but is {type(ignore_index)}'
            
        output_type = _Torch_ValueTensorType()
        total_weight_type = _Torch_ValueTensorType()
        super(AtenNllLossForwardOp, self).__init__(output_type, total_weight_type, self_, target, weight, reduction, ignore_index, loc=loc, ip=ip)
        
    
class AtenNllLossBackwardOp:
    def __init__(self, grad_output: Value, self_: Value, target: Value, weight: Optional[Value], reduction: int, ignore_index: int, total_weight: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(target):
            assert is_mlir_value(target), f'`target` should be a Value but is {type(target)}'
        else:
            target = get_op_result_or_value(target)
            assert is_a_Torch_ValueTensorType(target.type), f'`target` should be a Torch_ValueTensorType but is {type(target)}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(reduction):
            reduction = torch_dialect.ConstantIntOp(reduction)
        else:
            reduction = get_op_result_or_value(reduction)
            assert is_a_Torch_IntType(reduction.type), f'`reduction` should be a Torch_IntType but is {type(reduction)}'
            
        if not is_mlir_value(ignore_index):
            ignore_index = torch_dialect.ConstantIntOp(ignore_index)
        else:
            ignore_index = get_op_result_or_value(ignore_index)
            assert is_a_Torch_IntType(ignore_index.type), f'`ignore_index` should be a Torch_IntType but is {type(ignore_index)}'
            
        if not is_mlir_value(total_weight):
            assert is_mlir_value(total_weight), f'`total_weight` should be a Value but is {type(total_weight)}'
        else:
            total_weight = get_op_result_or_value(total_weight)
            assert is_a_Torch_ValueTensorType(total_weight.type), f'`total_weight` should be a Torch_ValueTensorType but is {type(total_weight)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNllLossBackwardOp, self).__init__(result_type, grad_output, self_, target, weight, reduction, ignore_index, total_weight, loc=loc, ip=ip)
        
    
class AtenBincountOp:
    def __init__(self, self_: Value, weights: Optional[Value], minlength: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(weights):
            if weights is not None:
                assert is_mlir_value(weights), f'`weights` should be a Value but is {type(weights)}'
            else:
                weights = torch_dialect.ConstantNoneOp()
        else:
            weights = get_op_result_or_value(weights)
            assert is_a_Torch_ValueTensorType(weights.type), f'`weights` should be a Torch_ValueTensorType but is {type(weights)}'
            
        if not is_mlir_value(minlength):
            minlength = torch_dialect.ConstantIntOp(minlength)
        else:
            minlength = get_op_result_or_value(minlength)
            assert is_a_Torch_IntType(minlength.type), f'`minlength` should be a Torch_IntType but is {type(minlength)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBincountOp, self).__init__(result_type, self_, weights, minlength, loc=loc, ip=ip)
        
    
class AtenLinalgVectorNormOp:
    def __init__(self, self_: Value, ord: Number, dim: Optional[List[int]], keepdim: bool, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(ord):
            ord = torch_dialect.ConstantNumberOp(ord)
        else:
            ord = get_op_result_or_value(ord)
            assert is_a_TorchScalarType(ord.type), f'`ord` should be a TorchScalarType but is {type(ord)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
                dim = torch_dialect.PrimListConstructOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLinalgVectorNormOp, self).__init__(result_type, self_, ord, dim, keepdim, dtype, loc=loc, ip=ip)
        
    
class AtenFrobeniusNormDimOp:
    def __init__(self, self_: Value, dim: List[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
            dim = torch_dialect.PrimListConstructOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFrobeniusNormDimOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenMseLossOp:
    def __init__(self, self_: Value, target: Value, reduction: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(target):
            assert is_mlir_value(target), f'`target` should be a Value but is {type(target)}'
        else:
            target = get_op_result_or_value(target)
            assert is_a_Torch_ValueTensorType(target.type), f'`target` should be a Torch_ValueTensorType but is {type(target)}'
            
        if not is_mlir_value(reduction):
            reduction = torch_dialect.ConstantIntOp(reduction)
        else:
            reduction = get_op_result_or_value(reduction)
            assert is_a_Torch_IntType(reduction.type), f'`reduction` should be a Torch_IntType but is {type(reduction)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMseLossOp, self).__init__(result_type, self_, target, reduction, loc=loc, ip=ip)
        
    
class AtenUpsampleNearest2dBackwardOp:
    def __init__(self, grad_output: Value, output_size: List[int], input_size: List[int], scales_h: Optional[float], scales_w: Optional[float], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(output_size):
            output_size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_size]
            output_size = torch_dialect.PrimListConstructOp(output_size)
        else:
            output_size = get_op_result_or_value(output_size)
            assert is_a_TorchListOfTorchIntType(output_size.type), f'`output_size` should be a TorchListOfTorchIntType but is {type(output_size)}'
            
        if not is_mlir_value(input_size):
            input_size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in input_size]
            input_size = torch_dialect.PrimListConstructOp(input_size)
        else:
            input_size = get_op_result_or_value(input_size)
            assert is_a_TorchListOfTorchIntType(input_size.type), f'`input_size` should be a TorchListOfTorchIntType but is {type(input_size)}'
            
        if not is_mlir_value(scales_h):
            if scales_h is not None:
                scales_h = torch_dialect.ConstantFloatOp(scales_h)
            else:
                scales_h = torch_dialect.ConstantNoneOp()
        else:
            scales_h = get_op_result_or_value(scales_h)
            assert is_a_Torch_FloatType(scales_h.type), f'`scales_h` should be a Torch_FloatType but is {type(scales_h)}'
            
        if not is_mlir_value(scales_w):
            if scales_w is not None:
                scales_w = torch_dialect.ConstantFloatOp(scales_w)
            else:
                scales_w = torch_dialect.ConstantNoneOp()
        else:
            scales_w = get_op_result_or_value(scales_w)
            assert is_a_Torch_FloatType(scales_w.type), f'`scales_w` should be a Torch_FloatType but is {type(scales_w)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenUpsampleNearest2dBackwardOp, self).__init__(result_type, grad_output, output_size, input_size, scales_h, scales_w, loc=loc, ip=ip)
        
    
class AtenConstantPadNdOp:
    def __init__(self, self_: Value, pad: List[int], value: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(pad):
            pad = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in pad]
            pad = torch_dialect.PrimListConstructOp(pad)
        else:
            pad = get_op_result_or_value(pad)
            assert is_a_TorchListOfTorchIntType(pad.type), f'`pad` should be a TorchListOfTorchIntType but is {type(pad)}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert is_a_TorchScalarType(value.type), f'`value` should be a TorchScalarType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenConstantPadNdOp, self).__init__(result_type, self_, pad, value, loc=loc, ip=ip)
        
    
class AtenPadOp:
    def __init__(self, self_: Value, pad: List[int], mode: str, value: Optional[float], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(pad):
            pad = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in pad]
            pad = torch_dialect.PrimListConstructOp(pad)
        else:
            pad = get_op_result_or_value(pad)
            assert is_a_TorchListOfTorchIntType(pad.type), f'`pad` should be a TorchListOfTorchIntType but is {type(pad)}'
            
        if not is_mlir_value(mode):
            mode = torch_dialect.ConstantStrOp(mode)
        else:
            mode = get_op_result_or_value(mode)
            assert is_a_Torch_StringType(mode.type), f'`mode` should be a Torch_StringType but is {type(mode)}'
            
        if not is_mlir_value(value):
            if value is not None:
                value = torch_dialect.ConstantFloatOp(value)
            else:
                value = torch_dialect.ConstantNoneOp()
        else:
            value = get_op_result_or_value(value)
            assert is_a_Torch_FloatType(value.type), f'`value` should be a Torch_FloatType but is {type(value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenPadOp, self).__init__(result_type, self_, pad, mode, value, loc=loc, ip=ip)
        
    
class AtenSqueezeDimOp:
    def __init__(self, self_: Value, dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSqueezeDimOp, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenSqueezeOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSqueezeOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFlattenUsingIntsOp:
    def __init__(self, self_: Value, start_dim: int, end_dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(start_dim):
            start_dim = torch_dialect.ConstantIntOp(start_dim)
        else:
            start_dim = get_op_result_or_value(start_dim)
            assert is_a_Torch_IntType(start_dim.type), f'`start_dim` should be a Torch_IntType but is {type(start_dim)}'
            
        if not is_mlir_value(end_dim):
            end_dim = torch_dialect.ConstantIntOp(end_dim)
        else:
            end_dim = get_op_result_or_value(end_dim)
            assert is_a_Torch_IntType(end_dim.type), f'`end_dim` should be a Torch_IntType but is {type(end_dim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFlattenUsingIntsOp, self).__init__(result_type, self_, start_dim, end_dim, loc=loc, ip=ip)
        
    
class AtenDimOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        super(AtenDimOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenSizeOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _TorchListOfTorchIntType()
        super(AtenSizeOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBoolTensorOp:
    def __init__(self, a: Value, *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        super(AtenBoolTensorOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenIsFloatingPointOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        super(AtenIsFloatingPointOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenOnesOp:
    def __init__(self, size: List[int], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenOnesOp, self).__init__(result_type, size, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenNewOnesOp:
    def __init__(self, self_: Value, size: List[int], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNewOnesOp, self).__init__(result_type, self_, size, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenZerosOp:
    def __init__(self, size: List[int], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenZerosOp, self).__init__(result_type, size, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenNewZerosOp:
    def __init__(self, self_: Value, size: List[int], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNewZerosOp, self).__init__(result_type, self_, size, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenTensorOp:
    def __init__(self, data: List[Value], dtype: Optional[int], device: Optional[Device], requires_grad: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(data):
            data = torch_dialect.PrimListConstructOp(data)
        else:
            data = get_op_result_or_value(data)
            assert is_a_TorchListOfValueTensorType(data.type), f'`data` should be a TorchListOfValueTensorType but is {type(data)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(requires_grad):
            requires_grad = torch_dialect.ConstantBoolOp(requires_grad)
        else:
            requires_grad = get_op_result_or_value(requires_grad)
            assert is_a_Torch_BoolType(requires_grad.type), f'`requires_grad` should be a Torch_BoolType but is {type(requires_grad)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTensorOp, self).__init__(result_type, data, dtype, device, requires_grad, loc=loc, ip=ip)
        
    
class AtenTensorBoolOp:
    def __init__(self, t: bool, dtype: Optional[int], device: Optional[Device], requires_grad: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(t):
            t = torch_dialect.ConstantBoolOp(t)
        else:
            t = get_op_result_or_value(t)
            assert is_a_Torch_BoolType(t.type), f'`t` should be a Torch_BoolType but is {type(t)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(requires_grad):
            requires_grad = torch_dialect.ConstantBoolOp(requires_grad)
        else:
            requires_grad = get_op_result_or_value(requires_grad)
            assert is_a_Torch_BoolType(requires_grad.type), f'`requires_grad` should be a Torch_BoolType but is {type(requires_grad)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTensorBoolOp, self).__init__(result_type, t, dtype, device, requires_grad, loc=loc, ip=ip)
        
    
class AtenTensorIntOp:
    def __init__(self, t: int, dtype: Optional[int], device: Optional[Device], requires_grad: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(t):
            t = torch_dialect.ConstantIntOp(t)
        else:
            t = get_op_result_or_value(t)
            assert is_a_Torch_IntType(t.type), f'`t` should be a Torch_IntType but is {type(t)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(requires_grad):
            requires_grad = torch_dialect.ConstantBoolOp(requires_grad)
        else:
            requires_grad = get_op_result_or_value(requires_grad)
            assert is_a_Torch_BoolType(requires_grad.type), f'`requires_grad` should be a Torch_BoolType but is {type(requires_grad)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTensorIntOp, self).__init__(result_type, t, dtype, device, requires_grad, loc=loc, ip=ip)
        
    
class Aten_ShapeAsTensorOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_ShapeAsTensorOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAllOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAllOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAllBoolOp:
    def __init__(self, self_: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = [torch_dialect.ConstantBoolOp(a) if not is_mlir_value(a) else a for a in self_]
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchListOfTorchBoolType(self_.type), f'`self_` should be a TorchListOfTorchBoolType but is {type(self_)}'
            
        super(AtenAllBoolOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenAnyOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAnyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAnyDimOp:
    def __init__(self, self_: Value, dim: int, keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAnyDimOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenArangeOp:
    def __init__(self, end: Number, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(end):
            end = torch_dialect.ConstantNumberOp(end)
        else:
            end = get_op_result_or_value(end)
            assert is_a_TorchScalarType(end.type), f'`end` should be a TorchScalarType but is {type(end)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenArangeOp, self).__init__(result_type, end, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenArangeStartOp:
    def __init__(self, start: Number, end: Number, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(start):
            start = torch_dialect.ConstantNumberOp(start)
        else:
            start = get_op_result_or_value(start)
            assert is_a_TorchScalarType(start.type), f'`start` should be a TorchScalarType but is {type(start)}'
            
        if not is_mlir_value(end):
            end = torch_dialect.ConstantNumberOp(end)
        else:
            end = get_op_result_or_value(end)
            assert is_a_TorchScalarType(end.type), f'`end` should be a TorchScalarType but is {type(end)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenArangeStartOp, self).__init__(result_type, start, end, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenArangeStartStepOp:
    def __init__(self, start: Number, end: Number, step: Number, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(start):
            start = torch_dialect.ConstantNumberOp(start)
        else:
            start = get_op_result_or_value(start)
            assert is_a_TorchScalarType(start.type), f'`start` should be a TorchScalarType but is {type(start)}'
            
        if not is_mlir_value(end):
            end = torch_dialect.ConstantNumberOp(end)
        else:
            end = get_op_result_or_value(end)
            assert is_a_TorchScalarType(end.type), f'`end` should be a TorchScalarType but is {type(end)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantNumberOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_TorchScalarType(step.type), f'`step` should be a TorchScalarType but is {type(step)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenArangeStartStepOp, self).__init__(result_type, start, end, step, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenArangeStartOutOp:
    def __init__(self, start: Number, end: Number, step: Number, out: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(start):
            start = torch_dialect.ConstantNumberOp(start)
        else:
            start = get_op_result_or_value(start)
            assert is_a_TorchScalarType(start.type), f'`start` should be a TorchScalarType but is {type(start)}'
            
        if not is_mlir_value(end):
            end = torch_dialect.ConstantNumberOp(end)
        else:
            end = get_op_result_or_value(end)
            assert is_a_TorchScalarType(end.type), f'`end` should be a TorchScalarType but is {type(end)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantNumberOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_TorchScalarType(step.type), f'`step` should be a TorchScalarType but is {type(step)}'
            
        if not is_mlir_value(out):
            assert is_mlir_value(out), f'`out` should be a Value but is {type(out)}'
        else:
            out = get_op_result_or_value(out)
            assert is_a_Torch_ValueTensorType(out.type), f'`out` should be a Torch_ValueTensorType but is {type(out)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenArangeStartOutOp, self).__init__(result_type, start, end, step, out, loc=loc, ip=ip)
        
    
class AtenArgmaxOp:
    def __init__(self, self_: Value, dim: Optional[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = torch_dialect.ConstantIntOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenArgmaxOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenBucketizeTensorOp:
    def __init__(self, self_: Value, boundaries: Value, out_int32: bool, right: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(boundaries):
            assert is_mlir_value(boundaries), f'`boundaries` should be a Value but is {type(boundaries)}'
        else:
            boundaries = get_op_result_or_value(boundaries)
            assert is_a_Torch_ValueTensorType(boundaries.type), f'`boundaries` should be a Torch_ValueTensorType but is {type(boundaries)}'
            
        if not is_mlir_value(out_int32):
            out_int32 = torch_dialect.ConstantBoolOp(out_int32)
        else:
            out_int32 = get_op_result_or_value(out_int32)
            assert is_a_Torch_BoolType(out_int32.type), f'`out_int32` should be a Torch_BoolType but is {type(out_int32)}'
            
        if not is_mlir_value(right):
            right = torch_dialect.ConstantBoolOp(right)
        else:
            right = get_op_result_or_value(right)
            assert is_a_Torch_BoolType(right.type), f'`right` should be a Torch_BoolType but is {type(right)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBucketizeTensorOp, self).__init__(result_type, self_, boundaries, out_int32, right, loc=loc, ip=ip)
        
    
class AtenCloneOp:
    def __init__(self, self_: Value, memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCloneOp, self).__init__(result_type, self_, memory_format, loc=loc, ip=ip)
        
    
class AtenLiftFreshCopyOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLiftFreshCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenContiguousOp:
    def __init__(self, self_: Value, memory_format: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(memory_format):
            memory_format = torch_dialect.ConstantIntOp(memory_format)
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenContiguousOp, self).__init__(result_type, self_, memory_format, loc=loc, ip=ip)
        
    
class AtenCopyOp:
    def __init__(self, self_: Value, src: Value, non_blocking: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'`src` should be a Value but is {type(src)}'
        else:
            src = get_op_result_or_value(src)
            assert is_a_Torch_ValueTensorType(src.type), f'`src` should be a Torch_ValueTensorType but is {type(src)}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert is_a_Torch_BoolType(non_blocking.type), f'`non_blocking` should be a Torch_BoolType but is {type(non_blocking)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCopyOp, self).__init__(result_type, self_, src, non_blocking, loc=loc, ip=ip)
        
    
class AtenCopy_Op:
    def __init__(self, self_: Value, src: Value, non_blocking: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'`src` should be a Value but is {type(src)}'
        else:
            src = get_op_result_or_value(src)
            assert is_a_Torch_ValueTensorType(src.type), f'`src` should be a Torch_ValueTensorType but is {type(src)}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert is_a_Torch_BoolType(non_blocking.type), f'`non_blocking` should be a Torch_BoolType but is {type(non_blocking)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCopy_Op, self).__init__(result_type, self_, src, non_blocking, loc=loc, ip=ip)
        
    
class Aten_ToCopyOp:
    def __init__(self, self_: Value, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], non_blocking: bool, memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert is_a_Torch_BoolType(non_blocking.type), f'`non_blocking` should be a Torch_BoolType but is {type(non_blocking)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_ToCopyOp, self).__init__(result_type, self_, dtype, layout, device, pin_memory, non_blocking, memory_format, loc=loc, ip=ip)
        
    
class AtenDetachOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDetachOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenEmbeddingOp:
    def __init__(self, weight: Value, indices: Value, padding_idx: int, scale_grad_by_freq: bool, sparse: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'`indices` should be a Value but is {type(indices)}'
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_Torch_ValueTensorType(indices.type), f'`indices` should be a Torch_ValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(padding_idx):
            padding_idx = torch_dialect.ConstantIntOp(padding_idx)
        else:
            padding_idx = get_op_result_or_value(padding_idx)
            assert is_a_Torch_IntType(padding_idx.type), f'`padding_idx` should be a Torch_IntType but is {type(padding_idx)}'
            
        if not is_mlir_value(scale_grad_by_freq):
            scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        else:
            scale_grad_by_freq = get_op_result_or_value(scale_grad_by_freq)
            assert is_a_Torch_BoolType(scale_grad_by_freq.type), f'`scale_grad_by_freq` should be a Torch_BoolType but is {type(scale_grad_by_freq)}'
            
        if not is_mlir_value(sparse):
            sparse = torch_dialect.ConstantBoolOp(sparse)
        else:
            sparse = get_op_result_or_value(sparse)
            assert is_a_Torch_BoolType(sparse.type), f'`sparse` should be a Torch_BoolType but is {type(sparse)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenEmbeddingOp, self).__init__(result_type, weight, indices, padding_idx, scale_grad_by_freq, sparse, loc=loc, ip=ip)
        
    
class AtenEmbeddingBagPaddingIdxOp:
    def __init__(self, weight: Value, indices: Value, offsets: Value, scale_grad_by_freq: bool, mode: int, sparse: bool, per_sample_weights: Optional[Value], include_last_offset: bool, padding_idx: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'`indices` should be a Value but is {type(indices)}'
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_Torch_ValueTensorType(indices.type), f'`indices` should be a Torch_ValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(offsets):
            assert is_mlir_value(offsets), f'`offsets` should be a Value but is {type(offsets)}'
        else:
            offsets = get_op_result_or_value(offsets)
            assert is_a_Torch_ValueTensorType(offsets.type), f'`offsets` should be a Torch_ValueTensorType but is {type(offsets)}'
            
        if not is_mlir_value(scale_grad_by_freq):
            scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        else:
            scale_grad_by_freq = get_op_result_or_value(scale_grad_by_freq)
            assert is_a_Torch_BoolType(scale_grad_by_freq.type), f'`scale_grad_by_freq` should be a Torch_BoolType but is {type(scale_grad_by_freq)}'
            
        if not is_mlir_value(mode):
            mode = torch_dialect.ConstantIntOp(mode)
        else:
            mode = get_op_result_or_value(mode)
            assert is_a_Torch_IntType(mode.type), f'`mode` should be a Torch_IntType but is {type(mode)}'
            
        if not is_mlir_value(sparse):
            sparse = torch_dialect.ConstantBoolOp(sparse)
        else:
            sparse = get_op_result_or_value(sparse)
            assert is_a_Torch_BoolType(sparse.type), f'`sparse` should be a Torch_BoolType but is {type(sparse)}'
            
        if not is_mlir_value(per_sample_weights):
            if per_sample_weights is not None:
                assert is_mlir_value(per_sample_weights), f'`per_sample_weights` should be a Value but is {type(per_sample_weights)}'
            else:
                per_sample_weights = torch_dialect.ConstantNoneOp()
        else:
            per_sample_weights = get_op_result_or_value(per_sample_weights)
            assert is_a_Torch_ValueTensorType(per_sample_weights.type), f'`per_sample_weights` should be a Torch_ValueTensorType but is {type(per_sample_weights)}'
            
        if not is_mlir_value(include_last_offset):
            include_last_offset = torch_dialect.ConstantBoolOp(include_last_offset)
        else:
            include_last_offset = get_op_result_or_value(include_last_offset)
            assert is_a_Torch_BoolType(include_last_offset.type), f'`include_last_offset` should be a Torch_BoolType but is {type(include_last_offset)}'
            
        if not is_mlir_value(padding_idx):
            if padding_idx is not None:
                padding_idx = torch_dialect.ConstantIntOp(padding_idx)
            else:
                padding_idx = torch_dialect.ConstantNoneOp()
        else:
            padding_idx = get_op_result_or_value(padding_idx)
            assert is_a_Torch_IntType(padding_idx.type), f'`padding_idx` should be a Torch_IntType but is {type(padding_idx)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        result2_type = _Torch_ValueTensorType()
        result3_type = _Torch_ValueTensorType()
        super(AtenEmbeddingBagPaddingIdxOp, self).__init__(result0_type, result1_type, result2_type, result3_type, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, loc=loc, ip=ip)
        
    
class Aten_EmbeddingBagOp:
    def __init__(self, weight: Value, indices: Value, offsets: Value, scale_grad_by_freq: bool, mode: int, sparse: bool, per_sample_weights: Optional[Value], include_last_offset: bool, padding_idx: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'`indices` should be a Value but is {type(indices)}'
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_Torch_ValueTensorType(indices.type), f'`indices` should be a Torch_ValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(offsets):
            assert is_mlir_value(offsets), f'`offsets` should be a Value but is {type(offsets)}'
        else:
            offsets = get_op_result_or_value(offsets)
            assert is_a_Torch_ValueTensorType(offsets.type), f'`offsets` should be a Torch_ValueTensorType but is {type(offsets)}'
            
        if not is_mlir_value(scale_grad_by_freq):
            scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        else:
            scale_grad_by_freq = get_op_result_or_value(scale_grad_by_freq)
            assert is_a_Torch_BoolType(scale_grad_by_freq.type), f'`scale_grad_by_freq` should be a Torch_BoolType but is {type(scale_grad_by_freq)}'
            
        if not is_mlir_value(mode):
            mode = torch_dialect.ConstantIntOp(mode)
        else:
            mode = get_op_result_or_value(mode)
            assert is_a_Torch_IntType(mode.type), f'`mode` should be a Torch_IntType but is {type(mode)}'
            
        if not is_mlir_value(sparse):
            sparse = torch_dialect.ConstantBoolOp(sparse)
        else:
            sparse = get_op_result_or_value(sparse)
            assert is_a_Torch_BoolType(sparse.type), f'`sparse` should be a Torch_BoolType but is {type(sparse)}'
            
        if not is_mlir_value(per_sample_weights):
            if per_sample_weights is not None:
                assert is_mlir_value(per_sample_weights), f'`per_sample_weights` should be a Value but is {type(per_sample_weights)}'
            else:
                per_sample_weights = torch_dialect.ConstantNoneOp()
        else:
            per_sample_weights = get_op_result_or_value(per_sample_weights)
            assert is_a_Torch_ValueTensorType(per_sample_weights.type), f'`per_sample_weights` should be a Torch_ValueTensorType but is {type(per_sample_weights)}'
            
        if not is_mlir_value(include_last_offset):
            include_last_offset = torch_dialect.ConstantBoolOp(include_last_offset)
        else:
            include_last_offset = get_op_result_or_value(include_last_offset)
            assert is_a_Torch_BoolType(include_last_offset.type), f'`include_last_offset` should be a Torch_BoolType but is {type(include_last_offset)}'
            
        if not is_mlir_value(padding_idx):
            padding_idx = torch_dialect.ConstantIntOp(padding_idx)
        else:
            padding_idx = get_op_result_or_value(padding_idx)
            assert is_a_Torch_IntType(padding_idx.type), f'`padding_idx` should be a Torch_IntType but is {type(padding_idx)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        result2_type = _Torch_ValueTensorType()
        result3_type = _Torch_ValueTensorType()
        super(Aten_EmbeddingBagOp, self).__init__(result0_type, result1_type, result2_type, result3_type, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, loc=loc, ip=ip)
        
    
class AtenEmptyLikeOp:
    def __init__(self, self_: Value, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenEmptyLikeOp, self).__init__(result_type, self_, dtype, layout, device, pin_memory, memory_format, loc=loc, ip=ip)
        
    
class AtenNewEmptyOp:
    def __init__(self, self_: Value, size: List[int], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNewEmptyOp, self).__init__(result_type, self_, size, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenZerosLikeOp:
    def __init__(self, self_: Value, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenZerosLikeOp, self).__init__(result_type, self_, dtype, layout, device, pin_memory, memory_format, loc=loc, ip=ip)
        
    
class AtenOnesLikeOp:
    def __init__(self, self_: Value, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenOnesLikeOp, self).__init__(result_type, self_, dtype, layout, device, pin_memory, memory_format, loc=loc, ip=ip)
        
    
class AtenEmptyMemoryFormatOp:
    def __init__(self, size: List[int], dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenEmptyMemoryFormatOp, self).__init__(result_type, size, dtype, layout, device, pin_memory, memory_format, loc=loc, ip=ip)
        
    
class AtenExpandOp:
    def __init__(self, self_: Value, size: List[int], implicit: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(implicit):
            implicit = torch_dialect.ConstantBoolOp(implicit)
        else:
            implicit = get_op_result_or_value(implicit)
            assert is_a_Torch_BoolType(implicit.type), f'`implicit` should be a Torch_BoolType but is {type(implicit)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenExpandOp, self).__init__(result_type, self_, size, implicit, loc=loc, ip=ip)
        
    
class AtenExpandAsOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenExpandAsOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenBroadcastToOp:
    def __init__(self, self_: Value, size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBroadcastToOp, self).__init__(result_type, self_, size, loc=loc, ip=ip)
        
    
class AtenIndexTensorOp:
    def __init__(self, self_: Value, indices: List[Optional[Value]], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_TorchListOfValueTensorType(indices.type), f'`indices` should be a TorchListOfValueTensorType but is {type(indices)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenIndexTensorOp, self).__init__(result_type, self_, indices, loc=loc, ip=ip)
        
    
class AtenIndexTensorHackedTwinOp:
    def __init__(self, self_: Value, indices: List[Value], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_TorchListOfValueTensorType(indices.type), f'`indices` should be a TorchListOfValueTensorType but is {type(indices)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenIndexTensorHackedTwinOp, self).__init__(result_type, self_, indices, loc=loc, ip=ip)
        
    
class AtenIndexSelectOp:
    def __init__(self, self_: Value, dim: int, index: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(index):
            assert is_mlir_value(index), f'`index` should be a Value but is {type(index)}'
        else:
            index = get_op_result_or_value(index)
            assert is_a_Torch_ValueTensorType(index.type), f'`index` should be a Torch_ValueTensorType but is {type(index)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenIndexSelectOp, self).__init__(result_type, self_, dim, index, loc=loc, ip=ip)
        
    
class Aten_IndexPutImplOp:
    def __init__(self, self_: Value, indices: List[Optional[Value]], values: Value, accumulate: bool, unsafe: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_TorchListOfValueTensorType(indices.type), f'`indices` should be a TorchListOfValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(values):
            assert is_mlir_value(values), f'`values` should be a Value but is {type(values)}'
        else:
            values = get_op_result_or_value(values)
            assert is_a_Torch_ValueTensorType(values.type), f'`values` should be a Torch_ValueTensorType but is {type(values)}'
            
        if not is_mlir_value(accumulate):
            accumulate = torch_dialect.ConstantBoolOp(accumulate)
        else:
            accumulate = get_op_result_or_value(accumulate)
            assert is_a_Torch_BoolType(accumulate.type), f'`accumulate` should be a Torch_BoolType but is {type(accumulate)}'
            
        if not is_mlir_value(unsafe):
            unsafe = torch_dialect.ConstantBoolOp(unsafe)
        else:
            unsafe = get_op_result_or_value(unsafe)
            assert is_a_Torch_BoolType(unsafe.type), f'`unsafe` should be a Torch_BoolType but is {type(unsafe)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_IndexPutImplOp, self).__init__(result_type, self_, indices, values, accumulate, unsafe, loc=loc, ip=ip)
        
    
class Aten_IndexPutImpl_Op:
    def __init__(self, self_: Value, indices: List[Optional[Value]], values: Value, accumulate: bool, unsafe: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_TorchListOfValueTensorType(indices.type), f'`indices` should be a TorchListOfValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(values):
            assert is_mlir_value(values), f'`values` should be a Value but is {type(values)}'
        else:
            values = get_op_result_or_value(values)
            assert is_a_Torch_ValueTensorType(values.type), f'`values` should be a Torch_ValueTensorType but is {type(values)}'
            
        if not is_mlir_value(accumulate):
            accumulate = torch_dialect.ConstantBoolOp(accumulate)
        else:
            accumulate = get_op_result_or_value(accumulate)
            assert is_a_Torch_BoolType(accumulate.type), f'`accumulate` should be a Torch_BoolType but is {type(accumulate)}'
            
        if not is_mlir_value(unsafe):
            unsafe = torch_dialect.ConstantBoolOp(unsafe)
        else:
            unsafe = get_op_result_or_value(unsafe)
            assert is_a_Torch_BoolType(unsafe.type), f'`unsafe` should be a Torch_BoolType but is {type(unsafe)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_IndexPutImpl_Op, self).__init__(result_type, self_, indices, values, accumulate, unsafe, loc=loc, ip=ip)
        
    
class AtenItemOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        super(AtenItemOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenMaskedSelectOp:
    def __init__(self, self_: Value, mask: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'`mask` should be a Value but is {type(mask)}'
        else:
            mask = get_op_result_or_value(mask)
            assert is_a_Torch_ValueTensorType(mask.type), f'`mask` should be a Torch_ValueTensorType but is {type(mask)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaskedSelectOp, self).__init__(result_type, self_, mask, loc=loc, ip=ip)
        
    
class AtenNumelOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        super(AtenNumelOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenRepeatOp:
    def __init__(self, self_: Value, repeats: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(repeats):
            repeats = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in repeats]
            repeats = torch_dialect.PrimListConstructOp(repeats)
        else:
            repeats = get_op_result_or_value(repeats)
            assert is_a_TorchListOfTorchIntType(repeats.type), f'`repeats` should be a TorchListOfTorchIntType but is {type(repeats)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRepeatOp, self).__init__(result_type, self_, repeats, loc=loc, ip=ip)
        
    
class AtenReshapeOp:
    def __init__(self, self_: Value, shape: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(shape):
            shape = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in shape]
            shape = torch_dialect.PrimListConstructOp(shape)
        else:
            shape = get_op_result_or_value(shape)
            assert is_a_TorchListOfTorchIntType(shape.type), f'`shape` should be a TorchListOfTorchIntType but is {type(shape)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenReshapeOp, self).__init__(result_type, self_, shape, loc=loc, ip=ip)
        
    
class Aten_ReshapeAliasOp:
    def __init__(self, self_: Value, size: List[int], stride: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_ReshapeAliasOp, self).__init__(result_type, self_, size, stride, loc=loc, ip=ip)
        
    
class AtenResize_Op:
    def __init__(self, self_: Value, size: List[int], memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenResize_Op, self).__init__(result_type, self_, size, memory_format, loc=loc, ip=ip)
        
    
class AtenSelectIntOp:
    def __init__(self, self_: Value, dim: int, index: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(index):
            index = torch_dialect.ConstantIntOp(index)
        else:
            index = get_op_result_or_value(index)
            assert is_a_Torch_IntType(index.type), f'`index` should be a Torch_IntType but is {type(index)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSelectIntOp, self).__init__(result_type, self_, dim, index, loc=loc, ip=ip)
        
    
class AtenSizeIntOp:
    def __init__(self, self_: Value, dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        super(AtenSizeIntOp, self).__init__(self_, dim, loc=loc, ip=ip)
        
    
class AtenStackOp:
    def __init__(self, tensors: List[Value], dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(tensors):
            tensors = torch_dialect.PrimListConstructOp(tensors)
        else:
            tensors = get_op_result_or_value(tensors)
            assert is_a_TorchListOfValueTensorType(tensors.type), f'`tensors` should be a TorchListOfValueTensorType but is {type(tensors)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenStackOp, self).__init__(result_type, tensors, dim, loc=loc, ip=ip)
        
    
class AtenSumOp:
    def __init__(self, self_: Value, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSumOp, self).__init__(result_type, self_, dtype, loc=loc, ip=ip)
        
    
class AtenSumDimIntListOp:
    def __init__(self, self_: Value, dim: Optional[List[int]], keepdim: bool, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
                dim = torch_dialect.PrimListConstructOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSumDimIntListOp, self).__init__(result_type, self_, dim, keepdim, dtype, loc=loc, ip=ip)
        
    
class AtenMaxOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenMaxOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenMaxDimOp:
    def __init__(self, self_: Value, dim: int, keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        values_type = _Torch_ValueTensorType()
        indices_type = _Torch_ValueTensorType()
        super(AtenMaxDimOp, self).__init__(values_type, indices_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenAmaxOp:
    def __init__(self, self_: Value, dim: List[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dim]
            dim = torch_dialect.PrimListConstructOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_TorchListOfTorchIntType(dim.type), f'`dim` should be a TorchListOfTorchIntType but is {type(dim)}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert is_a_Torch_BoolType(keepdim.type), f'`keepdim` should be a Torch_BoolType but is {type(keepdim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAmaxOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenToDtypeOp:
    def __init__(self, self_: Value, dtype: int, non_blocking: bool, copy: bool, memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            dtype = torch_dialect.ConstantIntOp(dtype)
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert is_a_Torch_BoolType(non_blocking.type), f'`non_blocking` should be a Torch_BoolType but is {type(non_blocking)}'
            
        if not is_mlir_value(copy):
            copy = torch_dialect.ConstantBoolOp(copy)
        else:
            copy = get_op_result_or_value(copy)
            assert is_a_Torch_BoolType(copy.type), f'`copy` should be a Torch_BoolType but is {type(copy)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenToDtypeOp, self).__init__(result_type, self_, dtype, non_blocking, copy, memory_format, loc=loc, ip=ip)
        
    
class AtenToDtypeLayoutOp:
    def __init__(self, self_: Value, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], non_blocking: bool, copy: bool, memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert is_a_Torch_BoolType(non_blocking.type), f'`non_blocking` should be a Torch_BoolType but is {type(non_blocking)}'
            
        if not is_mlir_value(copy):
            copy = torch_dialect.ConstantBoolOp(copy)
        else:
            copy = get_op_result_or_value(copy)
            assert is_a_Torch_BoolType(copy.type), f'`copy` should be a Torch_BoolType but is {type(copy)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenToDtypeLayoutOp, self).__init__(result_type, self_, dtype, layout, device, pin_memory, non_blocking, copy, memory_format, loc=loc, ip=ip)
        
    
class AtenToOtherOp:
    def __init__(self, self_: Value, other: Value, non_blocking: bool, copy: bool, memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert is_a_Torch_BoolType(non_blocking.type), f'`non_blocking` should be a Torch_BoolType but is {type(non_blocking)}'
            
        if not is_mlir_value(copy):
            copy = torch_dialect.ConstantBoolOp(copy)
        else:
            copy = get_op_result_or_value(copy)
            assert is_a_Torch_BoolType(copy.type), f'`copy` should be a Torch_BoolType but is {type(copy)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenToOtherOp, self).__init__(result_type, self_, other, non_blocking, copy, memory_format, loc=loc, ip=ip)
        
    
class AtenToPrimDeviceOp:
    def __init__(self, self_: Value, device: Optional[Device], dtype: Optional[int], non_blocking: bool, copy: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert is_a_Torch_BoolType(non_blocking.type), f'`non_blocking` should be a Torch_BoolType but is {type(non_blocking)}'
            
        if not is_mlir_value(copy):
            copy = torch_dialect.ConstantBoolOp(copy)
        else:
            copy = get_op_result_or_value(copy)
            assert is_a_Torch_BoolType(copy.type), f'`copy` should be a Torch_BoolType but is {type(copy)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenToPrimDeviceOp, self).__init__(result_type, self_, device, dtype, non_blocking, copy, loc=loc, ip=ip)
        
    
class AtenToDeviceOp:
    def __init__(self, self_: Value, device: Device, dtype: int, non_blocking: bool, copy: bool, memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(device):
            device = torch_dialect.ConstantDeviceOp(device)
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(dtype):
            dtype = torch_dialect.ConstantIntOp(dtype)
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert is_a_Torch_BoolType(non_blocking.type), f'`non_blocking` should be a Torch_BoolType but is {type(non_blocking)}'
            
        if not is_mlir_value(copy):
            copy = torch_dialect.ConstantBoolOp(copy)
        else:
            copy = get_op_result_or_value(copy)
            assert is_a_Torch_BoolType(copy.type), f'`copy` should be a Torch_BoolType but is {type(copy)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenToDeviceOp, self).__init__(result_type, self_, device, dtype, non_blocking, copy, memory_format, loc=loc, ip=ip)
        
    
class AtenTypeAsOp:
    def __init__(self, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTypeAsOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenViewOp:
    def __init__(self, self_: Value, size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenViewOp, self).__init__(result_type, self_, size, loc=loc, ip=ip)
        
    
class Aten_UnsafeViewOp:
    def __init__(self, self_: Value, size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_UnsafeViewOp, self).__init__(result_type, self_, size, loc=loc, ip=ip)
        
    
class AtenWhereSelfOp:
    def __init__(self, condition: Value, self_: Value, other: Value, *, loc=None, ip=None):
        if not is_mlir_value(condition):
            assert is_mlir_value(condition), f'`condition` should be a Value but is {type(condition)}'
        else:
            condition = get_op_result_or_value(condition)
            assert is_a_Torch_ValueTensorType(condition.type), f'`condition` should be a Torch_ValueTensorType but is {type(condition)}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenWhereSelfOp, self).__init__(result_type, condition, self_, other, loc=loc, ip=ip)
        
    
class AtenWhereScalarOp:
    def __init__(self, condition: Value, self_: Number, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(condition):
            assert is_mlir_value(condition), f'`condition` should be a Value but is {type(condition)}'
        else:
            condition = get_op_result_or_value(condition)
            assert is_a_Torch_ValueTensorType(condition.type), f'`condition` should be a Torch_ValueTensorType but is {type(condition)}'
            
        if not is_mlir_value(self_):
            self_ = torch_dialect.ConstantNumberOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchScalarType(self_.type), f'`self_` should be a TorchScalarType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenWhereScalarOp, self).__init__(result_type, condition, self_, other, loc=loc, ip=ip)
        
    
class AtenWhereScalarOtherOp:
    def __init__(self, condition: Value, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(condition):
            assert is_mlir_value(condition), f'`condition` should be a Value but is {type(condition)}'
        else:
            condition = get_op_result_or_value(condition)
            assert is_a_Torch_ValueTensorType(condition.type), f'`condition` should be a Torch_ValueTensorType but is {type(condition)}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenWhereScalarOtherOp, self).__init__(result_type, condition, self_, other, loc=loc, ip=ip)
        
    
class AtenWhereScalarSelfOp:
    def __init__(self, condition: Value, self_: Number, other: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(condition):
            assert is_mlir_value(condition), f'`condition` should be a Value but is {type(condition)}'
        else:
            condition = get_op_result_or_value(condition)
            assert is_a_Torch_ValueTensorType(condition.type), f'`condition` should be a Torch_ValueTensorType but is {type(condition)}'
            
        if not is_mlir_value(self_):
            self_ = torch_dialect.ConstantNumberOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchScalarType(self_.type), f'`self_` should be a TorchScalarType but is {type(self_)}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'`other` should be a Value but is {type(other)}'
        else:
            other = get_op_result_or_value(other)
            assert is_a_Torch_ValueTensorType(other.type), f'`other` should be a Torch_ValueTensorType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenWhereScalarSelfOp, self).__init__(result_type, condition, self_, other, loc=loc, ip=ip)
        
    
class AtenSliceTensorOp:
    def __init__(self, self_: Value, dim: int, start: Optional[int], end: Optional[int], step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(start):
            if start is not None:
                start = torch_dialect.ConstantIntOp(start)
            else:
                start = torch_dialect.ConstantNoneOp()
        else:
            start = get_op_result_or_value(start)
            assert is_a_Torch_IntType(start.type), f'`start` should be a Torch_IntType but is {type(start)}'
            
        if not is_mlir_value(end):
            if end is not None:
                end = torch_dialect.ConstantIntOp(end)
            else:
                end = torch_dialect.ConstantNoneOp()
        else:
            end = get_op_result_or_value(end)
            assert is_a_Torch_IntType(end.type), f'`end` should be a Torch_IntType but is {type(end)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_Torch_IntType(step.type), f'`step` should be a Torch_IntType but is {type(step)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSliceTensorOp, self).__init__(result_type, self_, dim, start, end, step, loc=loc, ip=ip)
        
    
class AtenLenTensorOp:
    def __init__(self, t: Value, *, loc=None, ip=None):
        if not is_mlir_value(t):
            assert is_mlir_value(t), f'`t` should be a Value but is {type(t)}'
        else:
            t = get_op_result_or_value(t)
            assert is_a_Torch_ValueTensorType(t.type), f'`t` should be a Torch_ValueTensorType but is {type(t)}'
            
        super(AtenLenTensorOp, self).__init__(t, loc=loc, ip=ip)
        
    
class AtenCpuOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCpuOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenGatherOp:
    def __init__(self, self_: Value, dim: int, index: Value, sparse_grad: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(index):
            assert is_mlir_value(index), f'`index` should be a Value but is {type(index)}'
        else:
            index = get_op_result_or_value(index)
            assert is_a_Torch_ValueTensorType(index.type), f'`index` should be a Torch_ValueTensorType but is {type(index)}'
            
        if not is_mlir_value(sparse_grad):
            sparse_grad = torch_dialect.ConstantBoolOp(sparse_grad)
        else:
            sparse_grad = get_op_result_or_value(sparse_grad)
            assert is_a_Torch_BoolType(sparse_grad.type), f'`sparse_grad` should be a Torch_BoolType but is {type(sparse_grad)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGatherOp, self).__init__(result_type, self_, dim, index, sparse_grad, loc=loc, ip=ip)
        
    
class AtenScatterAddOp:
    def __init__(self, self_: Value, dim: int, index: Value, src: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(index):
            assert is_mlir_value(index), f'`index` should be a Value but is {type(index)}'
        else:
            index = get_op_result_or_value(index)
            assert is_a_Torch_ValueTensorType(index.type), f'`index` should be a Torch_ValueTensorType but is {type(index)}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'`src` should be a Value but is {type(src)}'
        else:
            src = get_op_result_or_value(src)
            assert is_a_Torch_ValueTensorType(src.type), f'`src` should be a Torch_ValueTensorType but is {type(src)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenScatterAddOp, self).__init__(result_type, self_, dim, index, src, loc=loc, ip=ip)
        
    
class AtenIntImplicitOp:
    def __init__(self, a: Value, *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        super(AtenIntImplicitOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenFloatImplicitOp:
    def __init__(self, a: Value, *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        super(AtenFloatImplicitOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenTensorFloatOp:
    def __init__(self, t: float, dtype: Optional[int], device: Optional[Device], requires_grad: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(t):
            t = torch_dialect.ConstantFloatOp(t)
        else:
            t = get_op_result_or_value(t)
            assert is_a_Torch_FloatType(t.type), f'`t` should be a Torch_FloatType but is {type(t)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(requires_grad):
            requires_grad = torch_dialect.ConstantBoolOp(requires_grad)
        else:
            requires_grad = get_op_result_or_value(requires_grad)
            assert is_a_Torch_BoolType(requires_grad.type), f'`requires_grad` should be a Torch_BoolType but is {type(requires_grad)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTensorFloatOp, self).__init__(result_type, t, dtype, device, requires_grad, loc=loc, ip=ip)
        
    
class AtenIntTensorOp:
    def __init__(self, a: Value, *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        super(AtenIntTensorOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenFloatTensorOp:
    def __init__(self, a: Value, *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        super(AtenFloatTensorOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenDropoutOp:
    def __init__(self, input: Value, p: float, train: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(p):
            p = torch_dialect.ConstantFloatOp(p)
        else:
            p = get_op_result_or_value(p)
            assert is_a_Torch_FloatType(p.type), f'`p` should be a Torch_FloatType but is {type(p)}'
            
        if not is_mlir_value(train):
            train = torch_dialect.ConstantBoolOp(train)
        else:
            train = get_op_result_or_value(train)
            assert is_a_Torch_BoolType(train.type), f'`train` should be a Torch_BoolType but is {type(train)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDropoutOp, self).__init__(result_type, input, p, train, loc=loc, ip=ip)
        
    
class AtenDropout_Op:
    def __init__(self, self_: Value, p: float, train: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(p):
            p = torch_dialect.ConstantFloatOp(p)
        else:
            p = get_op_result_or_value(p)
            assert is_a_Torch_FloatType(p.type), f'`p` should be a Torch_FloatType but is {type(p)}'
            
        if not is_mlir_value(train):
            train = torch_dialect.ConstantBoolOp(train)
        else:
            train = get_op_result_or_value(train)
            assert is_a_Torch_BoolType(train.type), f'`train` should be a Torch_BoolType but is {type(train)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDropout_Op, self).__init__(result_type, self_, p, train, loc=loc, ip=ip)
        
    
class AtenNativeDropoutOp:
    def __init__(self, input: Value, p: float, train: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(p):
            p = torch_dialect.ConstantFloatOp(p)
        else:
            p = get_op_result_or_value(p)
            assert is_a_Torch_FloatType(p.type), f'`p` should be a Torch_FloatType but is {type(p)}'
            
        if not is_mlir_value(train):
            if train is not None:
                train = torch_dialect.ConstantBoolOp(train)
            else:
                train = torch_dialect.ConstantNoneOp()
        else:
            train = get_op_result_or_value(train)
            assert is_a_Torch_BoolType(train.type), f'`train` should be a Torch_BoolType but is {type(train)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        super(AtenNativeDropoutOp, self).__init__(result0_type, result1_type, input, p, train, loc=loc, ip=ip)
        
    
class AtenTOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenNumpyTOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNumpyTOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFullOp:
    def __init__(self, size: List[int], fill_value: Number, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(fill_value):
            fill_value = torch_dialect.ConstantNumberOp(fill_value)
        else:
            fill_value = get_op_result_or_value(fill_value)
            assert is_a_TorchScalarType(fill_value.type), f'`fill_value` should be a TorchScalarType but is {type(fill_value)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFullOp, self).__init__(result_type, size, fill_value, dtype, layout, device, pin_memory, loc=loc, ip=ip)
        
    
class AtenFullLikeOp:
    def __init__(self, self_: Value, fill_value: Number, dtype: Optional[int], layout: Optional[int], device: Optional[Device], pin_memory: Optional[bool], memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(fill_value):
            fill_value = torch_dialect.ConstantNumberOp(fill_value)
        else:
            fill_value = get_op_result_or_value(fill_value)
            assert is_a_TorchScalarType(fill_value.type), f'`fill_value` should be a TorchScalarType but is {type(fill_value)}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        if not is_mlir_value(layout):
            if layout is not None:
                layout = torch_dialect.ConstantIntOp(layout)
            else:
                layout = torch_dialect.ConstantNoneOp()
        else:
            layout = get_op_result_or_value(layout)
            assert is_a_Torch_IntType(layout.type), f'`layout` should be a Torch_IntType but is {type(layout)}'
            
        if not is_mlir_value(device):
            if device is not None:
                device = torch_dialect.ConstantDeviceOp(device)
            else:
                device = torch_dialect.ConstantNoneOp()
        else:
            device = get_op_result_or_value(device)
            assert is_a_Torch_DeviceType(device.type), f'`device` should be a Torch_DeviceType but is {type(device)}'
            
        if not is_mlir_value(pin_memory):
            if pin_memory is not None:
                pin_memory = torch_dialect.ConstantBoolOp(pin_memory)
            else:
                pin_memory = torch_dialect.ConstantNoneOp()
        else:
            pin_memory = get_op_result_or_value(pin_memory)
            assert is_a_Torch_BoolType(pin_memory.type), f'`pin_memory` should be a Torch_BoolType but is {type(pin_memory)}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert is_a_Torch_IntType(memory_format.type), f'`memory_format` should be a Torch_IntType but is {type(memory_format)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFullLikeOp, self).__init__(result_type, self_, fill_value, dtype, layout, device, pin_memory, memory_format, loc=loc, ip=ip)
        
    
class AtenBaddbmmOp:
    def __init__(self, self_: Value, batch1: Value, batch2: Value, beta: Number, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(batch1):
            assert is_mlir_value(batch1), f'`batch1` should be a Value but is {type(batch1)}'
        else:
            batch1 = get_op_result_or_value(batch1)
            assert is_a_Torch_ValueTensorType(batch1.type), f'`batch1` should be a Torch_ValueTensorType but is {type(batch1)}'
            
        if not is_mlir_value(batch2):
            assert is_mlir_value(batch2), f'`batch2` should be a Value but is {type(batch2)}'
        else:
            batch2 = get_op_result_or_value(batch2)
            assert is_a_Torch_ValueTensorType(batch2.type), f'`batch2` should be a Torch_ValueTensorType but is {type(batch2)}'
            
        if not is_mlir_value(beta):
            beta = torch_dialect.ConstantNumberOp(beta)
        else:
            beta = get_op_result_or_value(beta)
            assert is_a_TorchScalarType(beta.type), f'`beta` should be a TorchScalarType but is {type(beta)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBaddbmmOp, self).__init__(result_type, self_, batch1, batch2, beta, alpha, loc=loc, ip=ip)
        
    
class AtenBaddbmm_Op:
    def __init__(self, self_: Value, batch1: Value, batch2: Value, beta: Number, alpha: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(batch1):
            assert is_mlir_value(batch1), f'`batch1` should be a Value but is {type(batch1)}'
        else:
            batch1 = get_op_result_or_value(batch1)
            assert is_a_Torch_ValueTensorType(batch1.type), f'`batch1` should be a Torch_ValueTensorType but is {type(batch1)}'
            
        if not is_mlir_value(batch2):
            assert is_mlir_value(batch2), f'`batch2` should be a Value but is {type(batch2)}'
        else:
            batch2 = get_op_result_or_value(batch2)
            assert is_a_Torch_ValueTensorType(batch2.type), f'`batch2` should be a Torch_ValueTensorType but is {type(batch2)}'
            
        if not is_mlir_value(beta):
            beta = torch_dialect.ConstantNumberOp(beta)
        else:
            beta = get_op_result_or_value(beta)
            assert is_a_TorchScalarType(beta.type), f'`beta` should be a TorchScalarType but is {type(beta)}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert is_a_TorchScalarType(alpha.type), f'`alpha` should be a TorchScalarType but is {type(alpha)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenBaddbmm_Op, self).__init__(result_type, self_, batch1, batch2, beta, alpha, loc=loc, ip=ip)
        
    
class AtenFftFftOp:
    def __init__(self, self_: Value, n: Optional[int], dim: int, norm: Optional[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(n):
            if n is not None:
                n = torch_dialect.ConstantIntOp(n)
            else:
                n = torch_dialect.ConstantNoneOp()
        else:
            n = get_op_result_or_value(n)
            assert is_a_Torch_IntType(n.type), f'`n` should be a Torch_IntType but is {type(n)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(norm):
            if norm is not None:
                norm = torch_dialect.ConstantStrOp(norm)
            else:
                norm = torch_dialect.ConstantNoneOp()
        else:
            norm = get_op_result_or_value(norm)
            assert is_a_Torch_StringType(norm.type), f'`norm` should be a Torch_StringType but is {type(norm)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenFftFftOp, self).__init__(result_type, self_, n, dim, norm, loc=loc, ip=ip)
        
    
class AtenAliasCopyOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAliasCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAsStridedCopyOp:
    def __init__(self, self_: Value, size: List[int], stride: List[int], storage_offset: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(storage_offset):
            if storage_offset is not None:
                storage_offset = torch_dialect.ConstantIntOp(storage_offset)
            else:
                storage_offset = torch_dialect.ConstantNoneOp()
        else:
            storage_offset = get_op_result_or_value(storage_offset)
            assert is_a_Torch_IntType(storage_offset.type), f'`storage_offset` should be a Torch_IntType but is {type(storage_offset)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAsStridedCopyOp, self).__init__(result_type, self_, size, stride, storage_offset, loc=loc, ip=ip)
        
    
class AtenDiagonalCopyOp:
    def __init__(self, self_: Value, offset: int, dim1: int, dim2: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(offset):
            offset = torch_dialect.ConstantIntOp(offset)
        else:
            offset = get_op_result_or_value(offset)
            assert is_a_Torch_IntType(offset.type), f'`offset` should be a Torch_IntType but is {type(offset)}'
            
        if not is_mlir_value(dim1):
            dim1 = torch_dialect.ConstantIntOp(dim1)
        else:
            dim1 = get_op_result_or_value(dim1)
            assert is_a_Torch_IntType(dim1.type), f'`dim1` should be a Torch_IntType but is {type(dim1)}'
            
        if not is_mlir_value(dim2):
            dim2 = torch_dialect.ConstantIntOp(dim2)
        else:
            dim2 = get_op_result_or_value(dim2)
            assert is_a_Torch_IntType(dim2.type), f'`dim2` should be a Torch_IntType but is {type(dim2)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDiagonalCopyOp, self).__init__(result_type, self_, offset, dim1, dim2, loc=loc, ip=ip)
        
    
class AtenExpandCopyOp:
    def __init__(self, self_: Value, size: List[int], implicit: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(implicit):
            implicit = torch_dialect.ConstantBoolOp(implicit)
        else:
            implicit = get_op_result_or_value(implicit)
            assert is_a_Torch_BoolType(implicit.type), f'`implicit` should be a Torch_BoolType but is {type(implicit)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenExpandCopyOp, self).__init__(result_type, self_, size, implicit, loc=loc, ip=ip)
        
    
class AtenPermuteCopyOp:
    def __init__(self, self_: Value, dims: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dims):
            dims = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in dims]
            dims = torch_dialect.PrimListConstructOp(dims)
        else:
            dims = get_op_result_or_value(dims)
            assert is_a_TorchListOfTorchIntType(dims.type), f'`dims` should be a TorchListOfTorchIntType but is {type(dims)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenPermuteCopyOp, self).__init__(result_type, self_, dims, loc=loc, ip=ip)
        
    
class Aten_ReshapeAliasCopyOp:
    def __init__(self, self_: Value, size: List[int], stride: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_ReshapeAliasCopyOp, self).__init__(result_type, self_, size, stride, loc=loc, ip=ip)
        
    
class AtenSelectCopyIntOp:
    def __init__(self, self_: Value, dim: int, index: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(index):
            index = torch_dialect.ConstantIntOp(index)
        else:
            index = get_op_result_or_value(index)
            assert is_a_Torch_IntType(index.type), f'`index` should be a Torch_IntType but is {type(index)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSelectCopyIntOp, self).__init__(result_type, self_, dim, index, loc=loc, ip=ip)
        
    
class AtenDetachCopyOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDetachCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSliceCopyTensorOp:
    def __init__(self, self_: Value, dim: int, start: Optional[int], end: Optional[int], step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(start):
            if start is not None:
                start = torch_dialect.ConstantIntOp(start)
            else:
                start = torch_dialect.ConstantNoneOp()
        else:
            start = get_op_result_or_value(start)
            assert is_a_Torch_IntType(start.type), f'`start` should be a Torch_IntType but is {type(start)}'
            
        if not is_mlir_value(end):
            if end is not None:
                end = torch_dialect.ConstantIntOp(end)
            else:
                end = torch_dialect.ConstantNoneOp()
        else:
            end = get_op_result_or_value(end)
            assert is_a_Torch_IntType(end.type), f'`end` should be a Torch_IntType but is {type(end)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_Torch_IntType(step.type), f'`step` should be a Torch_IntType but is {type(step)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSliceCopyTensorOp, self).__init__(result_type, self_, dim, start, end, step, loc=loc, ip=ip)
        
    
class AtenSqueezeCopyOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSqueezeCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSqueezeCopyDimOp:
    def __init__(self, self_: Value, dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSqueezeCopyDimOp, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenTCopyOp:
    def __init__(self, self_: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenTransposeCopyIntOp:
    def __init__(self, self_: Value, dim0: int, dim1: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim0):
            dim0 = torch_dialect.ConstantIntOp(dim0)
        else:
            dim0 = get_op_result_or_value(dim0)
            assert is_a_Torch_IntType(dim0.type), f'`dim0` should be a Torch_IntType but is {type(dim0)}'
            
        if not is_mlir_value(dim1):
            dim1 = torch_dialect.ConstantIntOp(dim1)
        else:
            dim1 = get_op_result_or_value(dim1)
            assert is_a_Torch_IntType(dim1.type), f'`dim1` should be a Torch_IntType but is {type(dim1)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTransposeCopyIntOp, self).__init__(result_type, self_, dim0, dim1, loc=loc, ip=ip)
        
    
class AtenUnsqueezeCopyOp:
    def __init__(self, self_: Value, dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenUnsqueezeCopyOp, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenViewCopyOp:
    def __init__(self, self_: Value, size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenViewCopyOp, self).__init__(result_type, self_, size, loc=loc, ip=ip)
        
    
class AtenViewCopyDtypeOp:
    def __init__(self, self_: Value, dtype: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dtype):
            dtype = torch_dialect.ConstantIntOp(dtype)
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenViewCopyDtypeOp, self).__init__(result_type, self_, dtype, loc=loc, ip=ip)
        
    
class AtenUnfoldCopyOp:
    def __init__(self, self_: Value, dimension: int, size: int, step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dimension):
            dimension = torch_dialect.ConstantIntOp(dimension)
        else:
            dimension = get_op_result_or_value(dimension)
            assert is_a_Torch_IntType(dimension.type), f'`dimension` should be a Torch_IntType but is {type(dimension)}'
            
        if not is_mlir_value(size):
            size = torch_dialect.ConstantIntOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_Torch_IntType(size.type), f'`size` should be a Torch_IntType but is {type(size)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_Torch_IntType(step.type), f'`step` should be a Torch_IntType but is {type(step)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenUnfoldCopyOp, self).__init__(result_type, self_, dimension, size, step, loc=loc, ip=ip)
        
    
class AtenSelectScatterOp:
    def __init__(self, self_: Value, src: Value, dim: int, index: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'`src` should be a Value but is {type(src)}'
        else:
            src = get_op_result_or_value(src)
            assert is_a_Torch_ValueTensorType(src.type), f'`src` should be a Torch_ValueTensorType but is {type(src)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(index):
            index = torch_dialect.ConstantIntOp(index)
        else:
            index = get_op_result_or_value(index)
            assert is_a_Torch_IntType(index.type), f'`index` should be a Torch_IntType but is {type(index)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSelectScatterOp, self).__init__(result_type, self_, src, dim, index, loc=loc, ip=ip)
        
    
class AtenSliceScatterOp:
    def __init__(self, self_: Value, src: Value, dim: int, start: Optional[int], end: Optional[int], step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'`src` should be a Value but is {type(src)}'
        else:
            src = get_op_result_or_value(src)
            assert is_a_Torch_ValueTensorType(src.type), f'`src` should be a Torch_ValueTensorType but is {type(src)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(start):
            if start is not None:
                start = torch_dialect.ConstantIntOp(start)
            else:
                start = torch_dialect.ConstantNoneOp()
        else:
            start = get_op_result_or_value(start)
            assert is_a_Torch_IntType(start.type), f'`start` should be a Torch_IntType but is {type(start)}'
            
        if not is_mlir_value(end):
            if end is not None:
                end = torch_dialect.ConstantIntOp(end)
            else:
                end = torch_dialect.ConstantNoneOp()
        else:
            end = get_op_result_or_value(end)
            assert is_a_Torch_IntType(end.type), f'`end` should be a Torch_IntType but is {type(end)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_Torch_IntType(step.type), f'`step` should be a Torch_IntType but is {type(step)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenSliceScatterOp, self).__init__(result_type, self_, src, dim, start, end, step, loc=loc, ip=ip)
        
    
class AtenDiagonalScatterOp:
    def __init__(self, self_: Value, src: Value, offset: int, dim1: int, dim2: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'`src` should be a Value but is {type(src)}'
        else:
            src = get_op_result_or_value(src)
            assert is_a_Torch_ValueTensorType(src.type), f'`src` should be a Torch_ValueTensorType but is {type(src)}'
            
        if not is_mlir_value(offset):
            offset = torch_dialect.ConstantIntOp(offset)
        else:
            offset = get_op_result_or_value(offset)
            assert is_a_Torch_IntType(offset.type), f'`offset` should be a Torch_IntType but is {type(offset)}'
            
        if not is_mlir_value(dim1):
            dim1 = torch_dialect.ConstantIntOp(dim1)
        else:
            dim1 = get_op_result_or_value(dim1)
            assert is_a_Torch_IntType(dim1.type), f'`dim1` should be a Torch_IntType but is {type(dim1)}'
            
        if not is_mlir_value(dim2):
            dim2 = torch_dialect.ConstantIntOp(dim2)
        else:
            dim2 = get_op_result_or_value(dim2)
            assert is_a_Torch_IntType(dim2.type), f'`dim2` should be a Torch_IntType but is {type(dim2)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenDiagonalScatterOp, self).__init__(result_type, self_, src, offset, dim1, dim2, loc=loc, ip=ip)
        
    
class AtenAsStridedScatterOp:
    def __init__(self, self_: Value, src: Value, size: List[int], stride: List[int], storage_offset: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'`src` should be a Value but is {type(src)}'
        else:
            src = get_op_result_or_value(src)
            assert is_a_Torch_ValueTensorType(src.type), f'`src` should be a Torch_ValueTensorType but is {type(src)}'
            
        if not is_mlir_value(size):
            size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in size]
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert is_a_TorchListOfTorchIntType(size.type), f'`size` should be a TorchListOfTorchIntType but is {type(size)}'
            
        if not is_mlir_value(stride):
            stride = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in stride]
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert is_a_TorchListOfTorchIntType(stride.type), f'`stride` should be a TorchListOfTorchIntType but is {type(stride)}'
            
        if not is_mlir_value(storage_offset):
            if storage_offset is not None:
                storage_offset = torch_dialect.ConstantIntOp(storage_offset)
            else:
                storage_offset = torch_dialect.ConstantNoneOp()
        else:
            storage_offset = get_op_result_or_value(storage_offset)
            assert is_a_Torch_IntType(storage_offset.type), f'`storage_offset` should be a Torch_IntType but is {type(storage_offset)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenAsStridedScatterOp, self).__init__(result_type, self_, src, size, stride, storage_offset, loc=loc, ip=ip)
        
    
class AtenUpsampleNearest2dOp:
    def __init__(self, self_: Value, output_size: List[int], scales_h: Optional[float], scales_w: Optional[float], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(output_size):
            output_size = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in output_size]
            output_size = torch_dialect.PrimListConstructOp(output_size)
        else:
            output_size = get_op_result_or_value(output_size)
            assert is_a_TorchListOfTorchIntType(output_size.type), f'`output_size` should be a TorchListOfTorchIntType but is {type(output_size)}'
            
        if not is_mlir_value(scales_h):
            if scales_h is not None:
                scales_h = torch_dialect.ConstantFloatOp(scales_h)
            else:
                scales_h = torch_dialect.ConstantNoneOp()
        else:
            scales_h = get_op_result_or_value(scales_h)
            assert is_a_Torch_FloatType(scales_h.type), f'`scales_h` should be a Torch_FloatType but is {type(scales_h)}'
            
        if not is_mlir_value(scales_w):
            if scales_w is not None:
                scales_w = torch_dialect.ConstantFloatOp(scales_w)
            else:
                scales_w = torch_dialect.ConstantNoneOp()
        else:
            scales_w = get_op_result_or_value(scales_w)
            assert is_a_Torch_FloatType(scales_w.type), f'`scales_w` should be a Torch_FloatType but is {type(scales_w)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenUpsampleNearest2dOp, self).__init__(result_type, self_, output_size, scales_h, scales_w, loc=loc, ip=ip)
        
    
class Aten__Contains__StrOp:
    def __init__(self, dict: Dict[str, Value], key: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(dict):
            assert is_mlir_value(dict), f'`dict` should be a Value but is {type(dict)}'
        else:
            dict = get_op_result_or_value(dict)
            assert is_a_Torch_DictType(dict.type), f'`dict` should be a Torch_DictType but is {type(dict)}'
            
        if not is_mlir_value(key):
            key = torch_dialect.ConstantStrOp(key)
        else:
            key = get_op_result_or_value(key)
            assert is_a_Torch_StringType(key.type), f'`key` should be a Torch_StringType but is {type(key)}'
            
        super(Aten__Contains__StrOp, self).__init__(dict, key, loc=loc, ip=ip)
        
    
class Aten__Contains__IntListOp:
    def __init__(self, l: List[int], item: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            l = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in l]
            l = torch_dialect.PrimListConstructOp(l)
        else:
            l = get_op_result_or_value(l)
            assert is_a_TorchListOfTorchIntType(l.type), f'`l` should be a TorchListOfTorchIntType but is {type(l)}'
            
        if not is_mlir_value(item):
            item = torch_dialect.ConstantIntOp(item)
        else:
            item = get_op_result_or_value(item)
            assert is_a_Torch_IntType(item.type), f'`item` should be a Torch_IntType but is {type(item)}'
            
        super(Aten__Contains__IntListOp, self).__init__(l, item, loc=loc, ip=ip)
        
    
class Aten__Getitem__DictStrOp:
    def __init__(self, self_: Dict[str, Value], key: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_DictType(self_.type), f'`self_` should be a Torch_DictType but is {type(self_)}'
            
        if not is_mlir_value(key):
            key = torch_dialect.ConstantStrOp(key)
        else:
            key = get_op_result_or_value(key)
            assert is_a_Torch_StringType(key.type), f'`key` should be a Torch_StringType but is {type(key)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten__Getitem__DictStrOp, self).__init__(result_type, self_, key, loc=loc, ip=ip)
        
    
class Aten_SetItemStrOp:
    def __init__(self, l: Dict[str, Value], idx: str, v: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            assert is_mlir_value(l), f'`l` should be a Value but is {type(l)}'
        else:
            l = get_op_result_or_value(l)
            assert is_a_Torch_DictType(l.type), f'`l` should be a Torch_DictType but is {type(l)}'
            
        if not is_mlir_value(idx):
            idx = torch_dialect.ConstantStrOp(idx)
        else:
            idx = get_op_result_or_value(idx)
            assert is_a_Torch_StringType(idx.type), f'`idx` should be a Torch_StringType but is {type(idx)}'
            
        if not is_mlir_value(v):
            assert is_mlir_value(v), f'`v` should be a Value but is {type(v)}'
        else:
            v = get_op_result_or_value(v)
            assert is_a_Torch_ValueTensorType(v.type), f'`v` should be a Torch_ValueTensorType but is {type(v)}'
            
        super(Aten_SetItemStrOp, self).__init__(l, idx, v, loc=loc, ip=ip)
        
    
class AtenKeysStrOp:
    def __init__(self, self_: Dict[str, Value], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_DictType(self_.type), f'`self_` should be a Torch_DictType but is {type(self_)}'
            
        result_type = _TorchListOfTorchStringType()
        super(AtenKeysStrOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenGetDefaultStrOp:
    def __init__(self, self_: Dict[str, Value], key: str, default_value: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_DictType(self_.type), f'`self_` should be a Torch_DictType but is {type(self_)}'
            
        if not is_mlir_value(key):
            key = torch_dialect.ConstantStrOp(key)
        else:
            key = get_op_result_or_value(key)
            assert is_a_Torch_StringType(key.type), f'`key` should be a Torch_StringType but is {type(key)}'
            
        if not is_mlir_value(default_value):
            assert is_mlir_value(default_value), f'`default_value` should be a Value but is {type(default_value)}'
        else:
            default_value = get_op_result_or_value(default_value)
            assert is_a_Torch_ValueTensorType(default_value.type), f'`default_value` should be a Torch_ValueTensorType but is {type(default_value)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGetDefaultStrOp, self).__init__(result_type, self_, key, default_value, loc=loc, ip=ip)
        
    
class AtenDeleteDictStrOp:
    def __init__(self, self_: Dict[str, Value], key: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_DictType(self_.type), f'`self_` should be a Torch_DictType but is {type(self_)}'
            
        if not is_mlir_value(key):
            key = torch_dialect.ConstantStrOp(key)
        else:
            key = get_op_result_or_value(key)
            assert is_a_Torch_StringType(key.type), f'`key` should be a Torch_StringType but is {type(key)}'
            
        super(AtenDeleteDictStrOp, self).__init__(self_, key, loc=loc, ip=ip)
        
    
class AtenCatOp:
    def __init__(self, tensors: List[Value], dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(tensors):
            tensors = torch_dialect.PrimListConstructOp(tensors)
        else:
            tensors = get_op_result_or_value(tensors)
            assert is_a_TorchListOfValueTensorType(tensors.type), f'`tensors` should be a TorchListOfValueTensorType but is {type(tensors)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenCatOp, self).__init__(result_type, tensors, dim, loc=loc, ip=ip)
        
    
class AtenAppendTOp:
    def __init__(self, self_: List[Value], el: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchListOfValueTensorType(self_.type), f'`self_` should be a TorchListOfValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(el):
            assert is_mlir_value(el), f'`el` should be a Value but is {type(el)}'
        else:
            el = get_op_result_or_value(el)
            assert is_a_Torch_ValueTensorType(el.type), f'`el` should be a Torch_ValueTensorType but is {type(el)}'
            
        result_type = _TorchListOfValueTensorType()
        super(AtenAppendTOp, self).__init__(result_type, self_, el, loc=loc, ip=ip)
        
    
class AtenAddTOp:
    def __init__(self, a: List[Value], b: List[Value], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.PrimListConstructOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchListOfValueTensorType(a.type), f'`a` should be a TorchListOfValueTensorType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.PrimListConstructOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_TorchListOfValueTensorType(b.type), f'`b` should be a TorchListOfValueTensorType but is {type(b)}'
            
        result_type = _TorchListOfValueTensorType()
        super(AtenAddTOp, self).__init__(result_type, a, b, loc=loc, ip=ip)
        
    
class AtenEqIntListOp:
    def __init__(self, a: List[int], b: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in a]
            a = torch_dialect.PrimListConstructOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchListOfTorchIntType(a.type), f'`a` should be a TorchListOfTorchIntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in b]
            b = torch_dialect.PrimListConstructOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_TorchListOfTorchIntType(b.type), f'`b` should be a TorchListOfTorchIntType but is {type(b)}'
            
        super(AtenEqIntListOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenListTOp:
    def __init__(self, l: List[Value], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            l = torch_dialect.PrimListConstructOp(l)
        else:
            l = get_op_result_or_value(l)
            assert is_a_TorchListOfValueTensorType(l.type), f'`l` should be a TorchListOfValueTensorType but is {type(l)}'
            
        result_type = _TorchListOfValueTensorType()
        super(AtenListTOp, self).__init__(result_type, l, loc=loc, ip=ip)
        
    
class AtenSliceTOp:
    def __init__(self, l: List[Value], start: Optional[int], end: Optional[int], step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            l = torch_dialect.PrimListConstructOp(l)
        else:
            l = get_op_result_or_value(l)
            assert is_a_TorchListOfValueTensorType(l.type), f'`l` should be a TorchListOfValueTensorType but is {type(l)}'
            
        if not is_mlir_value(start):
            if start is not None:
                start = torch_dialect.ConstantIntOp(start)
            else:
                start = torch_dialect.ConstantNoneOp()
        else:
            start = get_op_result_or_value(start)
            assert is_a_Torch_IntType(start.type), f'`start` should be a Torch_IntType but is {type(start)}'
            
        if not is_mlir_value(end):
            if end is not None:
                end = torch_dialect.ConstantIntOp(end)
            else:
                end = torch_dialect.ConstantNoneOp()
        else:
            end = get_op_result_or_value(end)
            assert is_a_Torch_IntType(end.type), f'`end` should be a Torch_IntType but is {type(end)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_Torch_IntType(step.type), f'`step` should be a Torch_IntType but is {type(step)}'
            
        result_type = _TorchListOfValueTensorType()
        super(AtenSliceTOp, self).__init__(result_type, l, start, end, step, loc=loc, ip=ip)
        
    
class AtenInsertTOp:
    def __init__(self, self_: List[Value], idx: int, el: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchListOfValueTensorType(self_.type), f'`self_` should be a TorchListOfValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(idx):
            idx = torch_dialect.ConstantIntOp(idx)
        else:
            idx = get_op_result_or_value(idx)
            assert is_a_Torch_IntType(idx.type), f'`idx` should be a Torch_IntType but is {type(idx)}'
            
        if not is_mlir_value(el):
            assert is_mlir_value(el), f'`el` should be a Value but is {type(el)}'
        else:
            el = get_op_result_or_value(el)
            assert is_a_Torch_ValueTensorType(el.type), f'`el` should be a Torch_ValueTensorType but is {type(el)}'
            
        super(AtenInsertTOp, self).__init__(self_, idx, el, loc=loc, ip=ip)
        
    
class AtenNeIntListOp:
    def __init__(self, a: List[int], b: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in a]
            a = torch_dialect.PrimListConstructOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchListOfTorchIntType(a.type), f'`a` should be a TorchListOfTorchIntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in b]
            b = torch_dialect.PrimListConstructOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_TorchListOfTorchIntType(b.type), f'`b` should be a TorchListOfTorchIntType but is {type(b)}'
            
        super(AtenNeIntListOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenAnyBoolOp:
    def __init__(self, self_: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = [torch_dialect.ConstantBoolOp(a) if not is_mlir_value(a) else a for a in self_]
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchListOfTorchBoolType(self_.type), f'`self_` should be a TorchListOfTorchBoolType but is {type(self_)}'
            
        super(AtenAnyBoolOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenSortIntOp:
    def __init__(self, self_: List[int], reverse: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in self_]
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchListOfTorchIntType(self_.type), f'`self_` should be a TorchListOfTorchIntType but is {type(self_)}'
            
        if not is_mlir_value(reverse):
            reverse = torch_dialect.ConstantBoolOp(reverse)
        else:
            reverse = get_op_result_or_value(reverse)
            assert is_a_Torch_BoolType(reverse.type), f'`reverse` should be a Torch_BoolType but is {type(reverse)}'
            
        super(AtenSortIntOp, self).__init__(self_, reverse, loc=loc, ip=ip)
        
    
class AtenAddStrOp:
    def __init__(self, a: str, b: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantStrOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_StringType(a.type), f'`a` should be a Torch_StringType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantStrOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_StringType(b.type), f'`b` should be a Torch_StringType but is {type(b)}'
            
        super(AtenAddStrOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenEqStrOp:
    def __init__(self, a: str, b: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantStrOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_StringType(a.type), f'`a` should be a Torch_StringType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantStrOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_StringType(b.type), f'`b` should be a Torch_StringType but is {type(b)}'
            
        super(AtenEqStrOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLenStrOp:
    def __init__(self, s: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(s):
            s = torch_dialect.ConstantStrOp(s)
        else:
            s = get_op_result_or_value(s)
            assert is_a_Torch_StringType(s.type), f'`s` should be a Torch_StringType but is {type(s)}'
            
        super(AtenLenStrOp, self).__init__(s, loc=loc, ip=ip)
        
    
class AtenStrOp:
    def __init__(self, elem: Value, *, loc=None, ip=None):
        if not is_mlir_value(elem):
            assert is_mlir_value(elem), f'`elem` should be a Value but is {type(elem)}'
        else:
            elem = get_op_result_or_value(elem)
            assert is_a_Torch_ValueTensorType(elem.type), f'`elem` should be a Torch_ValueTensorType but is {type(elem)}'
            
        super(AtenStrOp, self).__init__(elem, loc=loc, ip=ip)
        
    
class AtenJoinOp:
    def __init__(self, self_: str, values: List[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = torch_dialect.ConstantStrOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_StringType(self_.type), f'`self_` should be a Torch_StringType but is {type(self_)}'
            
        if not is_mlir_value(values):
            values = [torch_dialect.ConstantStrOp(a) if not is_mlir_value(a) else a for a in values]
            values = torch_dialect.PrimListConstructOp(values)
        else:
            values = get_op_result_or_value(values)
            assert is_a_TorchListOfTorchStringType(values.type), f'`values` should be a TorchListOfTorchStringType but is {type(values)}'
            
        super(AtenJoinOp, self).__init__(self_, values, loc=loc, ip=ip)
        
    
class AtenFloatScalarOp:
    def __init__(self, a: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchScalarType(a.type), f'`a` should be a TorchScalarType but is {type(a)}'
            
        super(AtenFloatScalarOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenFloatStrOp:
    def __init__(self, a: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantStrOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_StringType(a.type), f'`a` should be a Torch_StringType but is {type(a)}'
            
        super(AtenFloatStrOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenIntFloatOp:
    def __init__(self, a: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        super(AtenIntFloatOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenIntScalarOp:
    def __init__(self, a: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchScalarType(a.type), f'`a` should be a TorchScalarType but is {type(a)}'
            
        super(AtenIntScalarOp, self).__init__(a, loc=loc, ip=ip)
        
    
class Aten__RangeLengthOp:
    def __init__(self, lo: int, hi: int, step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(lo):
            lo = torch_dialect.ConstantIntOp(lo)
        else:
            lo = get_op_result_or_value(lo)
            assert is_a_Torch_IntType(lo.type), f'`lo` should be a Torch_IntType but is {type(lo)}'
            
        if not is_mlir_value(hi):
            hi = torch_dialect.ConstantIntOp(hi)
        else:
            hi = get_op_result_or_value(hi)
            assert is_a_Torch_IntType(hi.type), f'`hi` should be a Torch_IntType but is {type(hi)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_Torch_IntType(step.type), f'`step` should be a Torch_IntType but is {type(step)}'
            
        super(Aten__RangeLengthOp, self).__init__(lo, hi, step, loc=loc, ip=ip)
        
    
class Aten__DeriveIndexOp:
    def __init__(self, index: int, start: int, step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(index):
            index = torch_dialect.ConstantIntOp(index)
        else:
            index = get_op_result_or_value(index)
            assert is_a_Torch_IntType(index.type), f'`index` should be a Torch_IntType but is {type(index)}'
            
        if not is_mlir_value(start):
            start = torch_dialect.ConstantIntOp(start)
        else:
            start = get_op_result_or_value(start)
            assert is_a_Torch_IntType(start.type), f'`start` should be a Torch_IntType but is {type(start)}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert is_a_Torch_IntType(step.type), f'`step` should be a Torch_IntType but is {type(step)}'
            
        super(Aten__DeriveIndexOp, self).__init__(index, start, step, loc=loc, ip=ip)
        
    
class AtenGtIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenGtIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGeIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenGeIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLtIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenLtIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLeIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenLeIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNeIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenNeIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenEqIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenEqIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenFloordivIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenFloordivIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenRemainderIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenRemainderIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenRemainderScalarOp:
    def __init__(self, self_: Value, other: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert is_a_TorchScalarType(other.type), f'`other` should be a TorchScalarType but is {type(other)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenRemainderScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenAddIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenAddIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenSubIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenSubIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenMulIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenMulIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenDivIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenDivIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNegIntOp:
    def __init__(self, a: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        super(AtenNegIntOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenLogIntOp:
    def __init__(self, a: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        super(AtenLogIntOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenAddFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenAddFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenSubFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_FloatType(b.type), f'`b` should be a Torch_FloatType but is {type(b)}'
            
        super(AtenSubFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenMulFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_FloatType(b.type), f'`b` should be a Torch_FloatType but is {type(b)}'
            
        super(AtenMulFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenDivFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_FloatType(b.type), f'`b` should be a Torch_FloatType but is {type(b)}'
            
        super(AtenDivFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNegFloatOp:
    def __init__(self, a: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        super(AtenNegFloatOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenEqFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_FloatType(b.type), f'`b` should be a Torch_FloatType but is {type(b)}'
            
        super(AtenEqFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGtFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_FloatType(b.type), f'`b` should be a Torch_FloatType but is {type(b)}'
            
        super(AtenGtFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGeFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_FloatType(b.type), f'`b` should be a Torch_FloatType but is {type(b)}'
            
        super(AtenGeFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLtFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_FloatType(b.type), f'`b` should be a Torch_FloatType but is {type(b)}'
            
        super(AtenLtFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLtFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenLtFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGeFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenGeFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNeFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenNeFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGtFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(AtenGtFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class Aten__And__BoolOp:
    def __init__(self, a: bool, b: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantBoolOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_BoolType(a.type), f'`a` should be a Torch_BoolType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantBoolOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_BoolType(b.type), f'`b` should be a Torch_BoolType but is {type(b)}'
            
        super(Aten__And__BoolOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNeBoolOp:
    def __init__(self, a: bool, b: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantBoolOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_BoolType(a.type), f'`a` should be a Torch_BoolType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantBoolOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_BoolType(b.type), f'`b` should be a Torch_BoolType but is {type(b)}'
            
        super(AtenNeBoolOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class Aten__Is__Op:
    def __init__(self, self_: Value, obj: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(obj):
            assert is_mlir_value(obj), f'`obj` should be a Value but is {type(obj)}'
        else:
            obj = get_op_result_or_value(obj)
            assert is_a_Torch_ValueTensorType(obj.type), f'`obj` should be a Torch_ValueTensorType but is {type(obj)}'
            
        super(Aten__Is__Op, self).__init__(self_, obj, loc=loc, ip=ip)
        
    
class Aten__Isnot__Op:
    def __init__(self, self_: Value, obj: Value, *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(obj):
            assert is_mlir_value(obj), f'`obj` should be a Value but is {type(obj)}'
        else:
            obj = get_op_result_or_value(obj)
            assert is_a_Torch_ValueTensorType(obj.type), f'`obj` should be a Torch_ValueTensorType but is {type(obj)}'
            
        super(Aten__Isnot__Op, self).__init__(self_, obj, loc=loc, ip=ip)
        
    
class Aten__Not__Op:
    def __init__(self, self_: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = torch_dialect.ConstantBoolOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_BoolType(self_.type), f'`self_` should be a Torch_BoolType but is {type(self_)}'
            
        super(Aten__Not__Op, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenLenTOp:
    def __init__(self, a: List[Value], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.PrimListConstructOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchListOfValueTensorType(a.type), f'`a` should be a TorchListOfValueTensorType but is {type(a)}'
            
        super(AtenLenTOp, self).__init__(a, loc=loc, ip=ip)
        
    
class Aten__Getitem__TOp:
    def __init__(self, list_: List[Value], idx: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(list_):
            list_ = torch_dialect.PrimListConstructOp(list_)
        else:
            list_ = get_op_result_or_value(list_)
            assert is_a_TorchListOfValueTensorType(list_.type), f'`list_` should be a TorchListOfValueTensorType but is {type(list_)}'
            
        if not is_mlir_value(idx):
            idx = torch_dialect.ConstantIntOp(idx)
        else:
            idx = get_op_result_or_value(idx)
            assert is_a_Torch_IntType(idx.type), f'`idx` should be a Torch_IntType but is {type(idx)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten__Getitem__TOp, self).__init__(result_type, list_, idx, loc=loc, ip=ip)
        
    
class Aten_SetItemTOp:
    def __init__(self, l: List[Value], idx: int, el: Value, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            l = torch_dialect.PrimListConstructOp(l)
        else:
            l = get_op_result_or_value(l)
            assert is_a_TorchListOfValueTensorType(l.type), f'`l` should be a TorchListOfValueTensorType but is {type(l)}'
            
        if not is_mlir_value(idx):
            idx = torch_dialect.ConstantIntOp(idx)
        else:
            idx = get_op_result_or_value(idx)
            assert is_a_Torch_IntType(idx.type), f'`idx` should be a Torch_IntType but is {type(idx)}'
            
        if not is_mlir_value(el):
            assert is_mlir_value(el), f'`el` should be a Value but is {type(el)}'
        else:
            el = get_op_result_or_value(el)
            assert is_a_Torch_ValueTensorType(el.type), f'`el` should be a Torch_ValueTensorType but is {type(el)}'
            
        result_type = _TorchListOfValueTensorType()
        super(Aten_SetItemTOp, self).__init__(result_type, l, idx, el, loc=loc, ip=ip)
        
    
class AtenDivOp:
    def __init__(self, a: Number, b: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchScalarType(a.type), f'`a` should be a TorchScalarType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantNumberOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_TorchScalarType(b.type), f'`b` should be a TorchScalarType but is {type(b)}'
            
        super(AtenDivOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenAddOp:
    def __init__(self, a: Number, b: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchScalarType(a.type), f'`a` should be a TorchScalarType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantNumberOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_TorchScalarType(b.type), f'`b` should be a TorchScalarType but is {type(b)}'
            
        super(AtenAddOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenSubOp:
    def __init__(self, a: Number, b: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchScalarType(a.type), f'`a` should be a TorchScalarType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantNumberOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_TorchScalarType(b.type), f'`b` should be a TorchScalarType but is {type(b)}'
            
        super(AtenSubOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenCeilScalarOp:
    def __init__(self, a: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchScalarType(a.type), f'`a` should be a TorchScalarType but is {type(a)}'
            
        super(AtenCeilScalarOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenSqrtIntOp:
    def __init__(self, a: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        super(AtenSqrtIntOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenBoolFloatOp:
    def __init__(self, a: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        super(AtenBoolFloatOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenBoolIntOp:
    def __init__(self, a: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        super(AtenBoolIntOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenEqDeviceOp:
    def __init__(self, a: Device, b: Device, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantDeviceOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_DeviceType(a.type), f'`a` should be a Torch_DeviceType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantDeviceOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_DeviceType(b.type), f'`b` should be a Torch_DeviceType but is {type(b)}'
            
        super(AtenEqDeviceOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenCeilFloatOp:
    def __init__(self, a: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_FloatType(a.type), f'`a` should be a Torch_FloatType but is {type(a)}'
            
        super(AtenCeilFloatOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenNarrowOp:
    def __init__(self, self_: Value, dim: int, start: int, length: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(start):
            start = torch_dialect.ConstantIntOp(start)
        else:
            start = get_op_result_or_value(start)
            assert is_a_Torch_IntType(start.type), f'`start` should be a Torch_IntType but is {type(start)}'
            
        if not is_mlir_value(length):
            length = torch_dialect.ConstantIntOp(length)
        else:
            length = get_op_result_or_value(length)
            assert is_a_Torch_IntType(length.type), f'`length` should be a Torch_IntType but is {type(length)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNarrowOp, self).__init__(result_type, self_, dim, start, length, loc=loc, ip=ip)
        
    
class Aten_SoftmaxBackwardDataOp:
    def __init__(self, grad_output: Value, output: Value, dim: int, input_dtype: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(output):
            assert is_mlir_value(output), f'`output` should be a Value but is {type(output)}'
        else:
            output = get_op_result_or_value(output)
            assert is_a_Torch_ValueTensorType(output.type), f'`output` should be a Torch_ValueTensorType but is {type(output)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(input_dtype):
            input_dtype = torch_dialect.ConstantIntOp(input_dtype)
        else:
            input_dtype = get_op_result_or_value(input_dtype)
            assert is_a_Torch_IntType(input_dtype.type), f'`input_dtype` should be a Torch_IntType but is {type(input_dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_SoftmaxBackwardDataOp, self).__init__(result_type, grad_output, output, dim, input_dtype, loc=loc, ip=ip)
        
    
class AtenTanhBackwardOp:
    def __init__(self, grad_output: Value, output: Value, *, loc=None, ip=None):
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(output):
            assert is_mlir_value(output), f'`output` should be a Value but is {type(output)}'
        else:
            output = get_op_result_or_value(output)
            assert is_a_Torch_ValueTensorType(output.type), f'`output` should be a Torch_ValueTensorType but is {type(output)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenTanhBackwardOp, self).__init__(result_type, grad_output, output, loc=loc, ip=ip)
        
    
class AtenGeluBackwardOp:
    def __init__(self, grad_output: Value, self_: Value, approximate: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(approximate):
            approximate = torch_dialect.ConstantStrOp(approximate)
        else:
            approximate = get_op_result_or_value(approximate)
            assert is_a_Torch_StringType(approximate.type), f'`approximate` should be a Torch_StringType but is {type(approximate)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenGeluBackwardOp, self).__init__(result_type, grad_output, self_, approximate, loc=loc, ip=ip)
        
    
class Aten_LogSoftmaxBackwardDataOp:
    def __init__(self, grad_output: Value, output: Value, dim: int, input_dtype: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(output):
            assert is_mlir_value(output), f'`output` should be a Value but is {type(output)}'
        else:
            output = get_op_result_or_value(output)
            assert is_a_Torch_ValueTensorType(output.type), f'`output` should be a Torch_ValueTensorType but is {type(output)}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert is_a_Torch_IntType(dim.type), f'`dim` should be a Torch_IntType but is {type(dim)}'
            
        if not is_mlir_value(input_dtype):
            input_dtype = torch_dialect.ConstantIntOp(input_dtype)
        else:
            input_dtype = get_op_result_or_value(input_dtype)
            assert is_a_Torch_IntType(input_dtype.type), f'`input_dtype` should be a Torch_IntType but is {type(input_dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(Aten_LogSoftmaxBackwardDataOp, self).__init__(result_type, grad_output, output, dim, input_dtype, loc=loc, ip=ip)
        
    
class AtenNativeLayerNormBackwardOp:
    def __init__(self, grad_out: Value, input: Value, normalized_shape: List[int], mean: Value, rstd: Value, weight: Optional[Value], bias: Optional[Value], output_mask: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_out):
            assert is_mlir_value(grad_out), f'`grad_out` should be a Value but is {type(grad_out)}'
        else:
            grad_out = get_op_result_or_value(grad_out)
            assert is_a_Torch_ValueTensorType(grad_out.type), f'`grad_out` should be a Torch_ValueTensorType but is {type(grad_out)}'
            
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(normalized_shape):
            normalized_shape = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in normalized_shape]
            normalized_shape = torch_dialect.PrimListConstructOp(normalized_shape)
        else:
            normalized_shape = get_op_result_or_value(normalized_shape)
            assert is_a_TorchListOfTorchIntType(normalized_shape.type), f'`normalized_shape` should be a TorchListOfTorchIntType but is {type(normalized_shape)}'
            
        if not is_mlir_value(mean):
            assert is_mlir_value(mean), f'`mean` should be a Value but is {type(mean)}'
        else:
            mean = get_op_result_or_value(mean)
            assert is_a_Torch_ValueTensorType(mean.type), f'`mean` should be a Torch_ValueTensorType but is {type(mean)}'
            
        if not is_mlir_value(rstd):
            assert is_mlir_value(rstd), f'`rstd` should be a Value but is {type(rstd)}'
        else:
            rstd = get_op_result_or_value(rstd)
            assert is_a_Torch_ValueTensorType(rstd.type), f'`rstd` should be a Torch_ValueTensorType but is {type(rstd)}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'`bias` should be a Value but is {type(bias)}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert is_a_Torch_ValueTensorType(bias.type), f'`bias` should be a Torch_ValueTensorType but is {type(bias)}'
            
        if not is_mlir_value(output_mask):
            output_mask = [torch_dialect.ConstantBoolOp(a) if not is_mlir_value(a) else a for a in output_mask]
            output_mask = torch_dialect.PrimListConstructOp(output_mask)
        else:
            output_mask = get_op_result_or_value(output_mask)
            assert is_a_TorchListOfTorchBoolType(output_mask.type), f'`output_mask` should be a TorchListOfTorchBoolType but is {type(output_mask)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        result2_type = _Torch_ValueTensorType()
        super(AtenNativeLayerNormBackwardOp, self).__init__(result0_type, result1_type, result2_type, grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask, loc=loc, ip=ip)
        
    
class AtenEmbeddingDenseBackwardOp:
    def __init__(self, grad_output: Value, indices: Value, num_weights: int, padding_idx: int, scale_grad_by_freq: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'`indices` should be a Value but is {type(indices)}'
        else:
            indices = get_op_result_or_value(indices)
            assert is_a_Torch_ValueTensorType(indices.type), f'`indices` should be a Torch_ValueTensorType but is {type(indices)}'
            
        if not is_mlir_value(num_weights):
            num_weights = torch_dialect.ConstantIntOp(num_weights)
        else:
            num_weights = get_op_result_or_value(num_weights)
            assert is_a_Torch_IntType(num_weights.type), f'`num_weights` should be a Torch_IntType but is {type(num_weights)}'
            
        if not is_mlir_value(padding_idx):
            padding_idx = torch_dialect.ConstantIntOp(padding_idx)
        else:
            padding_idx = get_op_result_or_value(padding_idx)
            assert is_a_Torch_IntType(padding_idx.type), f'`padding_idx` should be a Torch_IntType but is {type(padding_idx)}'
            
        if not is_mlir_value(scale_grad_by_freq):
            scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        else:
            scale_grad_by_freq = get_op_result_or_value(scale_grad_by_freq)
            assert is_a_Torch_BoolType(scale_grad_by_freq.type), f'`scale_grad_by_freq` should be a Torch_BoolType but is {type(scale_grad_by_freq)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenEmbeddingDenseBackwardOp, self).__init__(result_type, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, loc=loc, ip=ip)
        
    
class AtenNativeBatchNormBackwardOp:
    def __init__(self, grad_out: Value, input: Value, weight: Optional[Value], running_mean: Optional[Value], running_var: Optional[Value], save_mean: Optional[Value], save_invstd: Optional[Value], train: bool, eps: float, output_mask: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_out):
            assert is_mlir_value(grad_out), f'`grad_out` should be a Value but is {type(grad_out)}'
        else:
            grad_out = get_op_result_or_value(grad_out)
            assert is_a_Torch_ValueTensorType(grad_out.type), f'`grad_out` should be a Torch_ValueTensorType but is {type(grad_out)}'
            
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'`input` should be a Value but is {type(input)}'
        else:
            input = get_op_result_or_value(input)
            assert is_a_Torch_ValueTensorType(input.type), f'`input` should be a Torch_ValueTensorType but is {type(input)}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'`weight` should be a Value but is {type(weight)}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert is_a_Torch_ValueTensorType(weight.type), f'`weight` should be a Torch_ValueTensorType but is {type(weight)}'
            
        if not is_mlir_value(running_mean):
            if running_mean is not None:
                assert is_mlir_value(running_mean), f'`running_mean` should be a Value but is {type(running_mean)}'
            else:
                running_mean = torch_dialect.ConstantNoneOp()
        else:
            running_mean = get_op_result_or_value(running_mean)
            assert is_a_Torch_ValueTensorType(running_mean.type), f'`running_mean` should be a Torch_ValueTensorType but is {type(running_mean)}'
            
        if not is_mlir_value(running_var):
            if running_var is not None:
                assert is_mlir_value(running_var), f'`running_var` should be a Value but is {type(running_var)}'
            else:
                running_var = torch_dialect.ConstantNoneOp()
        else:
            running_var = get_op_result_or_value(running_var)
            assert is_a_Torch_ValueTensorType(running_var.type), f'`running_var` should be a Torch_ValueTensorType but is {type(running_var)}'
            
        if not is_mlir_value(save_mean):
            if save_mean is not None:
                assert is_mlir_value(save_mean), f'`save_mean` should be a Value but is {type(save_mean)}'
            else:
                save_mean = torch_dialect.ConstantNoneOp()
        else:
            save_mean = get_op_result_or_value(save_mean)
            assert is_a_Torch_ValueTensorType(save_mean.type), f'`save_mean` should be a Torch_ValueTensorType but is {type(save_mean)}'
            
        if not is_mlir_value(save_invstd):
            if save_invstd is not None:
                assert is_mlir_value(save_invstd), f'`save_invstd` should be a Value but is {type(save_invstd)}'
            else:
                save_invstd = torch_dialect.ConstantNoneOp()
        else:
            save_invstd = get_op_result_or_value(save_invstd)
            assert is_a_Torch_ValueTensorType(save_invstd.type), f'`save_invstd` should be a Torch_ValueTensorType but is {type(save_invstd)}'
            
        if not is_mlir_value(train):
            train = torch_dialect.ConstantBoolOp(train)
        else:
            train = get_op_result_or_value(train)
            assert is_a_Torch_BoolType(train.type), f'`train` should be a Torch_BoolType but is {type(train)}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert is_a_Torch_FloatType(eps.type), f'`eps` should be a Torch_FloatType but is {type(eps)}'
            
        if not is_mlir_value(output_mask):
            output_mask = [torch_dialect.ConstantBoolOp(a) if not is_mlir_value(a) else a for a in output_mask]
            output_mask = torch_dialect.PrimListConstructOp(output_mask)
        else:
            output_mask = get_op_result_or_value(output_mask)
            assert is_a_TorchListOfTorchBoolType(output_mask.type), f'`output_mask` should be a TorchListOfTorchBoolType but is {type(output_mask)}'
            
        result0_type = _Torch_ValueTensorType()
        result1_type = _Torch_ValueTensorType()
        result2_type = _Torch_ValueTensorType()
        super(AtenNativeBatchNormBackwardOp, self).__init__(result0_type, result1_type, result2_type, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, loc=loc, ip=ip)
        
    
class AtenNativeDropoutBackwardOp:
    def __init__(self, grad_output: Value, mask: Value, scale: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'`mask` should be a Value but is {type(mask)}'
        else:
            mask = get_op_result_or_value(mask)
            assert is_a_Torch_ValueTensorType(mask.type), f'`mask` should be a Torch_ValueTensorType but is {type(mask)}'
            
        if not is_mlir_value(scale):
            scale = torch_dialect.ConstantFloatOp(scale)
        else:
            scale = get_op_result_or_value(scale)
            assert is_a_Torch_FloatType(scale.type), f'`scale` should be a Torch_FloatType but is {type(scale)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenNativeDropoutBackwardOp, self).__init__(result_type, grad_output, mask, scale, loc=loc, ip=ip)
        
    
class AtenLeakyReluBackwardOp:
    def __init__(self, grad_output: Value, self_: Value, negative_slope: Number, self_is_result: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'`grad_output` should be a Value but is {type(grad_output)}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert is_a_Torch_ValueTensorType(grad_output.type), f'`grad_output` should be a Torch_ValueTensorType but is {type(grad_output)}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'`self_` should be a Value but is {type(self_)}'
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_Torch_ValueTensorType(self_.type), f'`self_` should be a Torch_ValueTensorType but is {type(self_)}'
            
        if not is_mlir_value(negative_slope):
            negative_slope = torch_dialect.ConstantNumberOp(negative_slope)
        else:
            negative_slope = get_op_result_or_value(negative_slope)
            assert is_a_TorchScalarType(negative_slope.type), f'`negative_slope` should be a TorchScalarType but is {type(negative_slope)}'
            
        if not is_mlir_value(self_is_result):
            self_is_result = torch_dialect.ConstantBoolOp(self_is_result)
        else:
            self_is_result = get_op_result_or_value(self_is_result)
            assert is_a_Torch_BoolType(self_is_result.type), f'`self_is_result` should be a Torch_BoolType but is {type(self_is_result)}'
            
        result_type = _Torch_ValueTensorType()
        super(AtenLeakyReluBackwardOp, self).__init__(result_type, grad_output, self_, negative_slope, self_is_result, loc=loc, ip=ip)
        
    
class PrimLayoutOp:
    def __init__(self, a: Value, *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        super(PrimLayoutOp, self).__init__(a, loc=loc, ip=ip)
        
    
class PrimTupleIndexOp:
    def __init__(self, tup: Any, i: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(tup):
            assert is_mlir_value(tup), f'`tup` should be a Value but is {type(tup)}'
        else:
            tup = get_op_result_or_value(tup)
            assert is_a_Torch_AnyType(tup.type), f'`tup` should be a Torch_AnyType but is {type(tup)}'
            
        if not is_mlir_value(i):
            i = torch_dialect.ConstantIntOp(i)
        else:
            i = get_op_result_or_value(i)
            assert is_a_Torch_IntType(i.type), f'`i` should be a Torch_IntType but is {type(i)}'
            
        super(PrimTupleIndexOp, self).__init__(tup, i, loc=loc, ip=ip)
        
    
class PrimDeviceOp:
    def __init__(self, a: Value, *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        super(PrimDeviceOp, self).__init__(a, loc=loc, ip=ip)
        
    
class PrimDtypeOp:
    def __init__(self, a: Value, *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        super(PrimDtypeOp, self).__init__(a, loc=loc, ip=ip)
        
    
class PrimNumToTensorScalarOp:
    def __init__(self, a: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchScalarType(a.type), f'`a` should be a TorchScalarType but is {type(a)}'
            
        result_type = _Torch_ValueTensorType()
        super(PrimNumToTensorScalarOp, self).__init__(result_type, a, loc=loc, ip=ip)
        
    
class PrimMinSelfIntOp:
    def __init__(self, self_: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in self_]
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchListOfTorchIntType(self_.type), f'`self_` should be a TorchListOfTorchIntType but is {type(self_)}'
            
        super(PrimMinSelfIntOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class PrimMinIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(PrimMinIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class PrimMaxSelfIntOp:
    def __init__(self, self_: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = [torch_dialect.ConstantIntOp(a) if not is_mlir_value(a) else a for a in self_]
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert is_a_TorchListOfTorchIntType(self_.type), f'`self_` should be a TorchListOfTorchIntType but is {type(self_)}'
            
        super(PrimMaxSelfIntOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class PrimMaxIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_IntType(a.type), f'`a` should be a Torch_IntType but is {type(a)}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert is_a_Torch_IntType(b.type), f'`b` should be a Torch_IntType but is {type(b)}'
            
        super(PrimMaxIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class PrimRaiseExceptionOp:
    def __init__(self, msg: str, cls: Optional[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(msg):
            msg = torch_dialect.ConstantStrOp(msg)
        else:
            msg = get_op_result_or_value(msg)
            assert is_a_Torch_StringType(msg.type), f'`msg` should be a Torch_StringType but is {type(msg)}'
            
        if not is_mlir_value(cls):
            if cls is not None:
                cls = torch_dialect.ConstantStrOp(cls)
            else:
                cls = torch_dialect.ConstantNoneOp()
        else:
            cls = get_op_result_or_value(cls)
            assert is_a_Torch_StringType(cls.type), f'`cls` should be a Torch_StringType but is {type(cls)}'
            
        super(PrimRaiseExceptionOp, self).__init__(msg, cls, loc=loc, ip=ip)
        
    
class PrimUninitializedOp:
    def __init__(self, *, loc=None, ip=None):
        super(PrimUninitializedOp, self).__init__(loc=loc, ip=ip)
        
    
class PrimAbsScalarOp:
    def __init__(self, a: Number, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert is_a_TorchScalarType(a.type), f'`a` should be a TorchScalarType but is {type(a)}'
            
        super(PrimAbsScalarOp, self).__init__(a, loc=loc, ip=ip)
        
    
class PrimsConvertElementTypeOp:
    def __init__(self, a: Value, dtype: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'`a` should be a Value but is {type(a)}'
        else:
            a = get_op_result_or_value(a)
            assert is_a_Torch_ValueTensorType(a.type), f'`a` should be a Torch_ValueTensorType but is {type(a)}'
            
        if not is_mlir_value(dtype):
            dtype = torch_dialect.ConstantIntOp(dtype)
        else:
            dtype = get_op_result_or_value(dtype)
            assert is_a_Torch_IntType(dtype.type), f'`dtype` should be a Torch_IntType but is {type(dtype)}'
            
        result_type = _Torch_ValueTensorType()
        super(PrimsConvertElementTypeOp, self).__init__(result_type, a, dtype, loc=loc, ip=ip)
        
    
