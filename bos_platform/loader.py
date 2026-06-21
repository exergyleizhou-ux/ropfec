"""
Load BOS platform implementations.

Default: in-tree stubs under bos_platform/.
Optional: set BOS_PLATFORM_PATH to a checkout whose layout is either:
  - <path>/bos_platform/*.py   (package directory)
  - <path> itself if it contains signal_control.py, kalman.py, ...
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class BosPlatformBundle:
    SignalAPI: Any
    Kalman: Any
    TemporalWorkflow: Any
    OPA: Any
    DigitalTwin: Any
    source: str


def _resolve_bos_platform_dir(root: str) -> Optional[str]:
    root = os.path.abspath(root)
    nested = os.path.join(root, "bos_platform")
    if os.path.isdir(nested) and os.path.isfile(os.path.join(nested, "signal_control.py")):
        return nested
    if os.path.isfile(os.path.join(root, "signal_control.py")):
        return root
    return None


def _ensure_pkg(pkg_name: str, base_dir: str) -> types.ModuleType:
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [base_dir]  # type: ignore[attr-defined]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    return pkg


def _load_package_module(base_dir: str, pkg_name: str, module_name: str) -> Any:
    path = os.path.join(base_dir, f"{module_name}.py")
    if not os.path.isfile(path):
        raise ImportError(f"missing {path}")
    _ensure_pkg(pkg_name, base_dir)
    full = f"{pkg_name}.{module_name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, path, submodule_search_locations=[base_dir]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_class_from_file(base_dir: str, module_name: str, class_name: str) -> Any:
    pkg_name = "_bos_platform_ext_flat"
    mod = _load_package_module(base_dir, pkg_name, module_name)
    return getattr(mod, class_name)


def _try_import_external(root: str) -> Optional[BosPlatformBundle]:
    base = _resolve_bos_platform_dir(root)
    if base is None:
        return None
    bmac_home = os.environ.get("BMAC_HOME", "").strip()
    if bmac_home and bmac_home not in sys.path:
        sys.path.insert(0, bmac_home)
    pkg_name = "_bos_platform_ext"
    # Load private modules first (adapter uses relative imports)
    preload: List[str] = ["_config", "_delegate", "_http"]
    try:
        for name in preload:
            p = os.path.join(base, f"{name}.py")
            if os.path.isfile(p):
                _load_package_module(base, pkg_name, name)
        return BosPlatformBundle(
            SignalAPI=getattr(_load_package_module(base, pkg_name, "signal_control"), "SignalAPI"),
            Kalman=getattr(_load_package_module(base, pkg_name, "kalman"), "Kalman"),
            TemporalWorkflow=getattr(_load_package_module(base, pkg_name, "temporal"), "TemporalWorkflow"),
            OPA=getattr(_load_package_module(base, pkg_name, "opa"), "OPA"),
            DigitalTwin=getattr(_load_package_module(base, pkg_name, "digital_twin"), "DigitalTwin"),
            source=f"external:{os.path.abspath(root)}",
        )
    except Exception:
        return None


def load_bos_platform(prefer_external: bool = True) -> BosPlatformBundle:
    env_root = os.environ.get("BOS_PLATFORM_PATH", "").strip()
    if prefer_external and env_root:
        ext = _try_import_external(env_root)
        if ext is not None:
            return ext

    from .signal_control import SignalAPI
    from .kalman import Kalman
    from .temporal import TemporalWorkflow
    from .opa import OPA
    from .digital_twin import DigitalTwin

    return BosPlatformBundle(
        SignalAPI=SignalAPI,
        Kalman=Kalman,
        TemporalWorkflow=TemporalWorkflow,
        OPA=OPA,
        DigitalTwin=DigitalTwin,
        source="in_tree_stubs",
    )


def apply_bundle_to_module(bundle: BosPlatformBundle, module: Any) -> None:
    """Expose bundle classes on a module namespace (used by bos_platform.__init__)."""
    module.SignalAPI = bundle.SignalAPI
    module.Kalman = bundle.Kalman
    module.TemporalWorkflow = bundle.TemporalWorkflow
    module.OPA = bundle.OPA
    module.DigitalTwin = bundle.DigitalTwin
    module.BosPlatformBundle = BosPlatformBundle
    module.load_bos_platform = load_bos_platform
    module.__bos_platform_source__ = bundle.source
