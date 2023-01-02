PI_XFAIL_SET = {
    # view op
    "ElementwiseFlattenBroadcastModule_basic",
    "FlattenDynamicModule_basic",
    "ElementwiseExpm1Module_basic",
    "FlattenRank0Module_basic",

    # unknown
    "UniformModule_basic",
    "BernoulliFloatModule_basic",

    # torchvision
    "IouOfModule_basic",
    "ResNet18Module",
    "ResNet18Module_basic",
    "ResNet18StaticModule_basic",
    "MobilenetV3Module_basic",

    # tuple returns
    "BernoulliTensorModule_basic",
    "AtenEmbeddingBagSumExample_basic",
    "VarMeanUnbiasedModule_basic",
    "VarMeanBiasedModule_basic",
    "TestMultipleTensorReturn_basic",
    "TestMultipleTensorAndPrimitiveTypesReturn_basic",
    "MaxPool2dWithIndicesModule_basic",
    "MaxPool2dWithIndicesCeilModeTrueModule_basic",
    "MaxPool2dWithIndicesAllOnesModule_basic",
    "ElementwiseClampModule_basic",
    "ElementwiseClampMaxModule_basic",
    "ElementwiseClampMinModule_basic",
    "NativeLayerNormModule_basic",
    "NativeLayerNormDynamicModule_basic",
    "NativeBatchNormNoneWeightModule_basic",
    "NativeBatchNorm3DModule_basic",
    "NativeBatchNorm2DModule_basic",
    "NativeBatchNorm1DModule_basic",
    "Aten_EmbeddingBagExample_basic",
    "DropoutTrainModule_basic",
    "ReturnThreeTensorFloat32_basic",
    "ReturnTwoTensorF32I64_basic",

    # python cast type return
    "AddIntModule_basic",
    "SubIntModule_basic",
    "SubFloatModule_basic",
    "MulIntModule_basic",
    "DivIntModule_basic",
    "DivFloatModule_basic",
    "CeilFloatModule_basic",
    "SqrtIntModule_basic",
    "SqrtIntConstantModule_basic",
    "BoolFloatFalseModule_basic",
    "BoolFloatTrueModule_basic",
    "BoolFloatConstantModule_basic",
    "BoolIntFalseModule_basic",
    "BoolIntTrueModule_basic",
    "BoolIntConstantModule_basic",
    "AtenIntTensorByteDtypeModule_basic",
    "AtenIntTensorCharDtypeModule_basic",

    # type/api overload
    "ToDtypeLayoutStridedModule_basic",
    "SqueezeModule_broadcast",
    "SqueezeDimModule_unitDim",
    "SqueezeDimModule_static",
    "SqueezeDimModule_negDim",
    "SqueezeDimModule_identity",
    "SqueezeDimModule_dynamic",
    "ArangeZeroElementOutputModule_basic",
    "ArangeStartStepIntModule_basic",
    "ArangeStartStepFloatModule_basic",
    "ArangeStartNegativeStepIntModule_basic",
    "ArangeStartNegativeStepFloatModule_basic",
    "ArangeStartIntModule_basic",
    "ArangeStartFloatModule_basic",
    "ArangeNegativeStartIntModule_basic",
    "ArangeNegativeStartFloatModule_basic",
    "ArangeFloatModule_basic",
    "ArangeIntModule_basic",
    "ArangeFalsePinMemoryModule_basic",
    "ArangeDtypeIntModule_basic",
    "ArangeDtypeFloatModule_basic",
    "MaxPool2dWithIndicesStaticModule_basic",
    "MaxPool2dWithIndicesNonDefaultStrideModule_basic",
    "MaxPool2dWithIndicesNonDefaultParamsModule_basic",
    "MaxPool2dWithIndicesNonDefaultPaddingModule_basic",
    "MaxPool2dWithIndicesNonDefaultDilationModule_basic",
    "MaxPool2dWithIndicesFullSizeKernelModule_basic",
    "MaxPool2dWithIndicesAllNegativeValuesModule_basic",
    "MaxPool2dModule_basic",
    "MaxPool2dCeilModeTrueModule_basic",
    "TensorToFloat_basic",
    "TensorToFloatZeroRank_basic",
    "RandnDtypeDeviceModule_basic",
    "BernoulliModule_basic",
    "HBC_basic",
    "AnyBoolTrueModule_basic",
    "AnyBoolFalseModule_basic",
    "AllBoolTrueModule_basic",
    "ReduceSumDimIntListKeepDimNegativeDimStaticModule_basic",
    "ReduceSumDimIntListKeepDimIntModule_basic",
    "ReduceSumDimIntListKeepDimFloatModule_basic",
    "ReduceSumDimIntListIntModule_basic",
    "ReduceSumDimIntListFloatModule_basic",
    "ReduceSumDimIntListEmptyDimModule_basic",
    "ReduceSumDimIntListElementTypeBoolModule_basic",
    "ReduceSumDimIntListDtypeIntModule_basic",
    "ReduceSumDimIntListDtypeFloatModule_basic",
    "ReduceMaxNegativeDim_basic",
    "ReduceMaxKeepDim_basic",
    "ReduceMaxKeepDimReturnBoth_basic",
    "ReduceMaxAlongDim_basic",
    "ReduceMaxAlongDimNegative_basic",
    "ReduceAmaxSingleDim_basic",
    "BroadcastToSameRankStaticModule_basic",
    "BroadcastZeroRankInputStaticModule_basic",
    "ScalarImplicitFloatModule_basic",
    "ScalarImplicitIntModule_basic",
    "SortIntListReverse_basic",
    "SortIntList_basic",
    "TransposeIntModule_basic",
    "TransposeIntNegDimsModule_basic",

    # error: operand types should have the same type as the list contained type
    "IndexTensorHackedTwinModule3dInput_basic",
    "IndexTensorHackedTwinModule_basic",
    "IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims_basic",
    "IndexTensorModule3dInput_basic",
    "IndexTensorModule_basic",
    "IndexTensorMultiInputContiguousCenter_basic",
    "IndexTensorMultiInputContiguousOneDimDynamic_basic",
    "IndexTensorMultiInputNonContiguousDynamic_basic",
    "IndexTensorMultiInputNonContiguousMultipleStaticDims_basic",
    "IndexTensorMultiInputNonContiguousOneDimDynamic_basic",
    "IndexTensorMultiInputNonContiguous_basic",
    "IndexTensorMultiInputOneDim_basic",
    "IndexTensorMultiInputThreeIndexers_basic",
    "IndexTensorMultiInput_basic",
    "IndexTensorSelectDimModule_basic",

    # segfault (lol)

    "IndexPutImpl3DIntNonAccumulateModule_basic",
    "IndexPutImpl3DIntAccumulateModule_basic",
    "IndexPutImpl3DFloatNonAccumulateModule_basic",
    "IndexPutImpl3DFloatAccumulateModule_basic",
    "IndexPutImpl2DIntNonAccumulateModule_basic",
    "IndexPutImpl2DIntAccumulateModule_basic",
    "IndexPutImpl2DFloatNonAccumulateModule_basic",
    "IndexPutImpl2DFloatAccumulateModule_basic",
    "IndexPutImpl1DIntNonAccumulateModule_basic",
    "IndexPutImpl1DIntAccumulateModule_basic",
    "IndexPutImpl1DFloatNonAccumulateModule_basic",
    "IndexPutImpl1DFloatAccumulateModule_basic",
    "IndexPut1DFloatNonAccumulateModule_basic",
    "ZeroInt64Module_basic",
    "ZeroInt32Module_basic",
    "ZeroFloat32Module_basic",
    "CopyModule_basic",
    "CopyWithDifferentDTypesAndSizesModule_basic",
    "CopyWithDifferentDTypesModule_basic",
    "CopyWithDifferentSizesModule_basic",

    # eager/lazy materialization
    "LayerNormNormalizeOverAllDimsModule_basic",
    "LayerNormModule_basic",
    "LayerNormLastDimModule_basic",
    "BatchNorm3DModule_basic",
    "BatchNorm2DModule_basic",
    "BatchNorm1DWith2DInputModule_basic",
    "BatchNorm1DModule_basic",
    "TensorIntModule_basic",
    "TensorLiteralModule_basic",
    "TensorOpaqueLiteralModule_basic",

    # backends
    "ConvolutionBackwardModule2DPadded_basic",
    "ConvolutionBackwardModule2D_basic",

    # failed to legalize operation
    "NumpyTRank0Module_basic",
    "TModuleRank0_basic",
    "TModuleRank1_basic",
    "TModuleRank2_basic",

    # error: found an op that was marked as backend illegal
    "AtenToDeviceModule_basic",
}
