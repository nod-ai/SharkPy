from pathlib import Path

from pi.mlir_utils import run_pipeline_with_repro_report

from pi.compiler.compiler import mlir_trace

mlir_module = mlir_trace(str(Path(__file__).parent / "simple_kernels.py"))

tile_size = 2

pipeline = "builtin.module(" + ",".join(
    list(
        filter(
            None,
            [
                f"func.func(cse)",
                f"func.func(loop-invariant-code-motion)",
                f"func.func(sccp)",
                f"func.func(affine-loop-tile{{tile-size={tile_size}}})"
                if tile_size > 0
                else None,
            ],
        )
    )
) + ")"

print(pipeline)
run_pipeline_with_repro_report(
    mlir_module,
    pipeline,
    description=""
)

print(mlir_module)
