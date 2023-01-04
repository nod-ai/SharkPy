import sys

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from importlib.machinery import SourceFileLoader, ModuleSpec
from importlib.util import find_spec, spec_from_loader, decode_source
from pathlib import Path
from types import ModuleType
from typing import Generator, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass(order=True, frozen=True)
class ImportOverload:
    name: str
    origin: Path
    is_package: bool
    submodule_search_locations: List[Path] = None

    def __post_init__(self):
        if self.is_package and self.submodule_search_locations is None:
            assert (
                self.origin.name == "__init__.py"
            ), f"default search path for {self.name} isn't a package: {self.origin}"
            object.__setattr__(self, "submodule_search_locations", [self.origin.parent])


@dataclass(order=True, frozen=True)
class RewriteOverload:
    name: str
    replacement_rules: dict[str, str]


_base_overloads = [
    ImportOverload(
        "torch_mlir.dialects._arith_ops_ext",
        Path(__file__).parent / "_arith_ops_ext.py",
        False,
    ),
    ImportOverload(
        "torch_mlir.dialects._memref_ops_ext",
        Path(__file__).parent / "_memref_ops_ext.py",
        False,
    ),
    ImportOverload(
        "torch_mlir.dialects._torch_ops_ext_custom",
        Path(__file__).parent / "_torch_ops_ext_custom.py",
        False,
    ),
    ImportOverload(
        "torch_mlir.dialects._torch_ops_ext",
        Path(__file__).parent / "_torch_ops_ext.py",
        False,
    ),
]

BASE_OVERLOADS: Dict[str, ImportOverload] = {i.name: i for i in _base_overloads}


class RewriterLoader(SourceFileLoader):
    # TraceLoader(tracers_to_use, spec.loader.name, spec.loader.path)
    def __init__(self, rewrite_rules: dict[str, str], name, path) -> None:
        super().__init__(name, path)
        self.rewrite_rules = rewrite_rules

    def get_data(self, path) -> bytes:
        source = self.get_rewritten_source(path)
        return bytes(source, encoding="utf-8")

    def get_code(self, fullname):
        source_path = self.get_filename(fullname)
        source_bytes = self.get_data(source_path)
        return self.source_to_code(source_bytes, source_path)

    def get_rewritten_source(self, source_path) -> str:
        source_bytes = super().get_data(source_path)
        source = decode_source(source_bytes)
        for k, v in self.rewrite_rules.items():
            source = source.replace(k, v)
        return source

    def source_to_code(self, data: bytes | str, path: str = ...):
        return super(RewriterLoader, self).source_to_code(data, path)

    def exec_module(self, module: ModuleType) -> None:
        super().exec_module(module)


# this is based on the birdseye finder (which uses import hooks based on MacroPy's):
# https://github.com/alexmojaki/birdseye/blob/9974af715b1801f9dd99fef93ff133d0ab5223af/birdseye/import_hook.py
class Overloader(MetaPathFinder):
    def __init__(self, overloads) -> None:
        self.tracers = None
        self._thread = threading.current_thread()
        self.overloads: Dict[str, ImportOverload] = overloads

    @contextmanager
    def _clear_preceding_finders(self) -> Generator[None, None, None]:
        """
        Clear all preceding finders from sys.meta_path, and restore them afterwards.
        """
        orig_finders = sys.meta_path
        try:
            sys.meta_path = sys.meta_path[sys.meta_path.index(self) + 1 :]  # noqa: E203
            yield
        finally:
            sys.meta_path = orig_finders

    def _find_plain_spec(self, fullname, path, target):
        """Try to find the original module using all the
        remaining meta_path finders."""
        spec = None
        self_seen = False
        for finder in sys.meta_path:
            if finder is self:
                self_seen = True
                continue
            elif not self_seen or "pytest" in finder.__module__:
                # when testing with pytest, it installs a finder that for
                # some yet unknown reasons makes birdseye
                # fail. For now it will just avoid using it and pass to
                # the next one
                continue
            if hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, target=target)
            elif hasattr(finder, "load_module"):
                spec = spec_from_loader(fullname, finder)

            if spec is not None and spec.origin != "builtin":
                return spec

    def find_spec(self, fullname, path=None, target=None):
        logger.debug(f"finding spec for {fullname=} {path=} {target=}")

        if threading.current_thread() is not self._thread:
            return None
        if target is None:
            with self._clear_preceding_finders():
                spec = find_spec(fullname, path)
        else:
            spec = self._find_plain_spec(fullname, path, target)

        if fullname not in self.overloads:
            if spec is None or not (
                hasattr(spec.loader, "get_source") and callable(spec.loader.get_source)
            ):  # noqa: E128
                if fullname != "org":
                    # stdlib pickle.py at line 94 contains a ``from
                    # org.python.core for Jython which is always failing,
                    # of course
                    logger.debug("Failed finding spec for %s", fullname)
                return None

            if not isinstance(spec.loader, SourceFileLoader):
                return None
            return spec

        logger.debug("patching spec for %s", fullname)

        overload = self.overloads[fullname]
        if isinstance(overload, ImportOverload):
            new_path = str(overload.origin)
            source_file_loader = SourceFileLoader(fullname, new_path)
            spec = ModuleSpec(
                name=fullname,
                loader=source_file_loader,
                origin=new_path,
                is_package=overload.is_package,
            )
            if overload.is_package:
                spec.submodule_search_locations = [
                    str(p) for p in overload.submodule_search_locations
                ]
            spec.has_location = True
        elif isinstance(overload, RewriteOverload):
            assert spec is not None
            spec.loader = RewriterLoader(
                overload.replacement_rules, spec.loader.name, spec.loader.path
            )

        return spec


def patch_meta_path_non_context(overloads=None) -> Callable:
    if overloads is None:
        overloads = BASE_OVERLOADS
    orig_meta_path_entry = None

    def cleanup_callback():
        if orig_meta_path_entry is None:
            del sys.meta_path[0]
        else:
            sys.meta_path[0] = orig_meta_path_entry

    if len(sys.meta_path) > 0 and isinstance(sys.meta_path[0], Overloader):
        orig_meta_path_entry = sys.meta_path[0]
        sys.meta_path[0] = Overloader(overloads)
    else:
        sys.meta_path.insert(0, Overloader(overloads))

    return cleanup_callback


@contextmanager
def patch_meta_path(overloads=None) -> Generator[None, None, None]:
    cleanup_callback = None
    try:
        cleanup_callback = patch_meta_path_non_context(overloads)
        yield
    except:
        raise
    finally:
        if cleanup_callback is not None:
            cleanup_callback()


def remove_modules(pred: Callable):
    to_delete = []
    for mod in sys.modules:
        if pred(mod):
            logger.debug(f"removing from sys.modules {mod}")
            to_delete.append(mod)
    for mod in to_delete:
        del sys.modules[mod]
