import difflib
import functools
import inspect
import logging
import operator
import traceback
from multiprocessing import cpu_count
from multiprocess.pool import Pool

# noinspection PyUnresolvedReferences
import pi

# noinspection PyUnresolvedReferences
import torch_mlir
import torch_mlir_e2e_test
import torch_mlir_e2e_test.registry

from xfail import PI_XFAIL_SET

from torch_mlir_e2e_test.test_suite import COMMON_TORCH_MLIR_LOWERING_XFAILS
from torch_mlir_e2e_test.test_suite import (
    register_all_tests as torch_mlir_register_all_tests,
)

from pi.dialects import (
    remove_modules,
    RewriteOverload,
    patch_meta_path,
)
from pi.mlir_utils import lower_pi_to_linalg
from pi.testing.util import (
    PIConfig,
    TorchDialectConfig,
    set_weights,
    lower_torch_mlir_to_linalg,
)
from pi.testing.registry import GLOBAL_TEST_REGISTRY as PI_GLOBAL_TEST_REGISTRY
from pi.dispatcher.function import NotFoundLookupError, AmbiguousLookupError


FORMAT = "%(asctime)s, %(levelname)-8s [%(filename)s:%(module)s:%(funcName)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(FORMAT)

# logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

# Create handlers
# c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file.log")
# c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)


# def sig_handler(signum, frame):
#     print(f"segfault {signum}")
#
#
# signal.signal(signal.SIGSEGV, sig_handler)


def run_torch_mlir_tests():
    torch_mlir_register_all_tests()
    import torch_mlir_e2e_test.registry
    import torch_mlir_e2e_test.framework

    tests = sorted(
        torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY, key=lambda t: t.unique_name
    )
    tests = list(
        filter(
            lambda t: t.unique_name
            not in PI_XFAIL_SET | COMMON_TORCH_MLIR_LOWERING_XFAILS,
            tests,
        )
    )
    num_processes = min(int(cpu_count() * 1.1), len(tests))

    torch_dialect_config = TorchDialectConfig()
    pool = Pool(num_processes)

    def compile_and_run_test(test):
        mod = test.program_factory()
        set_weights(mod)
        mod.eval()
        torch_mlir_module = torch_dialect_config.compile(mod)
        torch_mlir_module_str = str(torch_mlir_module)
        return test.unique_name, str(lower_torch_mlir_to_linalg(torch_mlir_module)), torch_mlir_module_str

    handles = pool.map_async(compile_and_run_test, tests)
    # handles = map(compile_and_run_test, tests)
    torch_mlir_linalg_module_strs = {}
    for name, linalg_module, torch_module in handles.get():
        # for name, s in handles:
        torch_mlir_linalg_module_strs[name] = linalg_module, torch_module

    return torch_mlir_linalg_module_strs


def run_pi_tests(torch_mlir_linalg_module_strs):
    torch_mlir_register_all_tests()
    tests = sorted(PI_GLOBAL_TEST_REGISTRY, key=lambda t: t.unique_name)
    assert tests, "failed to load tests"

    pi.nn.Module.train = lambda *args, **kwargs: None

    pi_config = PIConfig()
    torch_dialect_config = TorchDialectConfig()
    (
        PASS,
        NotImplementedErrorFAIL,
        NotFoundLookupErrorFAIL,
        AmbiguousLookupErrorFAIL,
        lower_to_linalg_FAIL,
        irFAIL,
        TOTAL,
        SKIP,
    ) = (0, 0, 0, 0, 0, 0, 0, 0)
    for test in tests:
        TOTAL += 1
        if test.unique_name in PI_XFAIL_SET | COMMON_TORCH_MLIR_LOWERING_XFAILS:
            print(f"skipping {test.unique_name}")
            SKIP += 1
            continue
        print(f"running {test.unique_name}")

        test_module = test.program_factory()
        torch_linalg_module, torch_dialect_module = torch_mlir_linalg_module_strs[test.unique_name]

        try:
            pi_mlir_module = pi_config.compile(test, test_module)
        except NotImplementedError as e:
            print(traceback.format_exc(-2))
            print(f"{e}")
            print(f"FAIL pi compile NotImplementedError")
            NotImplementedErrorFAIL += 1
            print()
            continue
        except NotFoundLookupError as e:
            print(traceback.format_exc())
            print(f"{e}")
            print(f"FAIL dispatcher error")
            NotFoundLookupErrorFAIL += 1
            print()
            continue
        except AmbiguousLookupError as e:
            print(traceback.format_exc())
            print(f"{e}")
            print(f"FAIL dispatcher error")
            AmbiguousLookupErrorFAIL += 1
            print("\ntorch_mlir module\n")
            print(torch_dialect_module)
            print(torch_linalg_module)
            print()
            continue
        except Exception as e:
            print("\ntorch_mlir module\n")
            print(torch_dialect_module)
            print(torch_linalg_module)
            # torch_script_compiled = torchscript_config.compile(mod)
            # frozen = torch.jit.freeze(torch_script_compiled)
            # torch_mlir_module = torch_dialect_config.compile(mod)
            # print("frozen.graph\n", frozen.graph)
            # print("torch_mlir_module\n", torch_mlir_module)
            print(f"FAIL pi compile Exception")
            raise e

        pi_torch_dialect_module_str = str(pi_mlir_module)
        try:
            pi_linalg_module_str = str(lower_pi_to_linalg(pi_mlir_module, enable_ir_printing=False))
        except Exception as e:
            print(traceback.format_exc())
            print("\npi_mlir_module\n")
            print(pi_torch_dialect_module_str)
            print(pi_mlir_module)
            print("\ntorch_mlir module\n")
            print(torch_dialect_module)
            print(torch_linalg_module)
            print(f"FAIL lower_pi_to_linalg Exception")
            lower_to_linalg_FAIL += 1
            print()
            continue

        diff = list(
            difflib.unified_diff(
                str(pi_linalg_module_str).splitlines(),
                str(torch_linalg_module).splitlines(),
                lineterm="",
            )
        )

        if len(diff):
            print(f"\n{''.join('*' * 10)}\ndiff\n{''.join('*' * 10)}\n")
            print("\n".join(diff))
            print()
            # print("torch_mlir_linalg_module_str:\n", torch_mlir_linalg_module_str)
            # print("pi_mlir_linalg_module_str:\n", pi_mlir_linalg_module_str)
            print(f"FAIL IR diff")
            irFAIL += 1

            # torch_script_compiled = torchscript_config.compile(mod)
            # frozen = torch.jit.freeze(torch_script_compiled)
            # torch_mlir_module = torch_dialect_config.compile(mod)
            # print("frozen.graph\n", frozen.graph)
            # print("torch_mlir_module\n", torch_mlir_module)
        else:
            print("PASS")
            PASS += 1
        print()

    assert (
        functools.reduce(
            operator.add,
            (
                PASS,
                NotImplementedErrorFAIL,
                NotFoundLookupErrorFAIL,
                AmbiguousLookupErrorFAIL,
                lower_to_linalg_FAIL,
                irFAIL,
                SKIP,
            ),
            0,
        )
        == TOTAL
    )
    assert NotImplementedErrorFAIL == 0, "missing torch_wrappers impl; you probably need to run generate_torch_mlir_extensions.py"
    assert NotFoundLookupErrorFAIL == 0, "pytorch api changed; good luck"
    print(
        f"\n{''.join('*' * 10)}\n\n{PASS=}\n{NotImplementedErrorFAIL=}\n{NotFoundLookupErrorFAIL=}\n{AmbiguousLookupErrorFAIL=}\n{lower_to_linalg_FAIL=}\n{irFAIL=}\n{SKIP=}\nout of {TOTAL=}\n\n{''.join('*' * 10)}\n"
    )


def main():
    torch_mlir_linalg_module_strs = run_torch_mlir_tests()
    overloads = [
        RewriteOverload(
            f"torch_mlir_e2e_test.test_suite.{k}",
            {
                "torch.": "pi.",
                "from torch import nn": "from pi import nn",
                "torch_mlir_e2e_test.annotations": "pi.compiler.annotations",
                "torch_mlir_e2e_test": "pi.testing",
                "import torchvision.models as models": "",
                "import torch": "import pi",
                "import functorch": "",
            },
        )
        for k, v in torch_mlir_e2e_test.test_suite.__dict__.items()
        if inspect.ismodule(v)
    ]
    torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY = []
    torch_mlir_e2e_test.registry._SEEN_UNIQUE_NAMES = set()
    remove_modules(lambda mod: mod.startswith("torch_mlir_e2e_test"))
    remove_modules(lambda mod: mod == "torch" or mod.startswith("torch."))

    overloads = {o.name: o for o in overloads}
    with patch_meta_path(overloads):
        run_pi_tests(torch_mlir_linalg_module_strs)


if __name__ == "__main__":
    main()
