import logging
import sys
from importlib.machinery import ModuleSpec

FORMAT = "%(asctime)s, %(levelname)-8s [%(filename)s:%(module)s:%(funcName)s:%(lineno)d] %(message)s"
formatter = logging.Formatter(FORMAT)
from typing import Optional


logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)

# Create handlers
# c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file.log")
# c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)


from pathlib import Path
import importlib
import pi
from pi.dialects import ImportOverload, patch_meta_path, remove_modules

pi_package_root_path = Path(pi.__file__).parent


def sdfsf():
    import torch

    assert (
        torch.__spec__.origin == f"{pi_package_root_path}/__init__.py"
    ), f"monkey patch failed {torch.__spec__.origin}"

    hash(pi.Tensor)
    hash(torch.Tensor)

    assert pi.Tensor == torch.Tensor, f"monkey patch failed {pi.Tensor} {torch.Tensor}"


def import_overload():
    import torch

    assert (
        torch.__spec__.origin != f"{pi_package_root_path}/__init__.py"
    ), f"already monkey patched: {torch.__spec__.origin}"

    sys.modules["torch"] = pi

    def test():
        import torch

        assert (
            torch.__spec__.origin == f"{pi_package_root_path}/__init__.py"
        ), f"monkey patch failed {torch.__spec__.origin}"

        hash(pi.Tensor)
        hash(torch.Tensor)

        assert (
            pi.Tensor == torch.Tensor
        ), f"monkey patch failed {pi.Tensor} {torch.Tensor}"

    test()


if __name__ == "__main__":
    import_overload()
