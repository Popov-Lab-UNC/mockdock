from __future__ import annotations

import pytest

from mockdock.utils import resolve_backend


def test_resolve_backend_auto_resolves_to_autodock_gpu():
    out = resolve_backend("auto")
    assert out == "autodock_gpu"


def test_resolve_backend_autodock_gpu_resolves_to_autodock_gpu():
    out = resolve_backend("autodock_gpu")
    assert out == "autodock_gpu"


def test_resolve_backend_vina_raises_error():
    with pytest.raises(ValueError, match="Unsupported docking backend 'vina'"):
        resolve_backend("vina")


def test_resolve_backend_unknown_raises_error():
    with pytest.raises(ValueError, match="Unsupported docking backend 'unknown'"):
        resolve_backend("unknown")
