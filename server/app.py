"""Compatibility ASGI module expected by certain OpenEnv validators."""

from __future__ import annotations

from pathlib import Path
import importlib.util


_ROOT = Path(__file__).resolve().parents[1]
_SERVER_FILE = _ROOT / "server.py"
_SPEC = importlib.util.spec_from_file_location("cbic_server_module", _SERVER_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load server module from {_SERVER_FILE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

app = _MODULE.app


def main() -> None:
    """Compatibility entrypoint forwarding to the root server module."""
    _MODULE.main()


if __name__ == "__main__":
    main()
