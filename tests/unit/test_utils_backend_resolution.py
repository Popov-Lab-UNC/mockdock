from __future__ import annotations

from pathlib import Path

from mockdock.utils import resolve_backend


def test_resolve_backend_auto_uses_adgpu_when_available(monkeypatch):
    monkeypatch.setattr("mockdock.utils.shutil.which", lambda *_: "/usr/bin/adgpu")
    out = resolve_backend("auto", n_gpus=1, adgpu_executable="adgpu")
    assert out == "autodock_gpu"


def test_resolve_backend_auto_falls_back_to_vina_without_gpu(monkeypatch):
    monkeypatch.setattr("mockdock.utils.shutil.which", lambda *_: "/usr/bin/adgpu")
    out = resolve_backend("auto", n_gpus=0, adgpu_executable="adgpu")
    assert out == "vina"


def test_resolve_backend_autodock_gpu_falls_back_when_missing(monkeypatch):
    monkeypatch.setattr("mockdock.utils.shutil.which", lambda *_: None)
    monkeypatch.setattr(Path, "exists", lambda *_: False)
    out = resolve_backend("autodock_gpu", n_gpus=1, adgpu_executable="adgpu")
    assert out == "vina"
