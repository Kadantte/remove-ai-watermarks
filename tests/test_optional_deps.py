"""Tests for the shared optional-dependency availability guard."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util

import pytest

from remove_ai_watermarks import optional_deps
from remove_ai_watermarks._internal.watermark_profiles import PIXELS_MODULES, VISIBLE_EXTRA


def _fake_find_spec(specs: dict[str, importlib.machinery.ModuleSpec | None]):
    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        return specs[name]

    return find_spec


def _real_spec(name: str) -> importlib.machinery.ModuleSpec:
    spec = importlib.util.find_spec(name)
    assert spec is not None
    assert spec.loader is not None
    return spec


class TestModuleAvailable:
    def test_installed_module_is_available(self):
        assert optional_deps.module_available("json") is True

    def test_missing_module_is_not_available(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({"ghost": None}))
        assert optional_deps.module_available("ghost") is False

    def test_namespace_package_remnant_is_not_available(self, monkeypatch):
        # A leftover data dir in site-packages (e.g. trustmark/models/ surviving
        # an uninstall) resolves to a namespace-package spec with loader=None;
        # the guard must not report it as installed.
        ns_spec = importlib.machinery.ModuleSpec("trustmark", loader=None, is_package=True)
        assert ns_spec.loader is None
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({"trustmark": ns_spec}))
        assert optional_deps.module_available("trustmark") is False

    def test_any_namespace_member_fails_the_conjunction(self, monkeypatch):
        ns_spec = importlib.machinery.ModuleSpec("spandrel", loader=None, is_package=True)
        specs = {"spandrel": ns_spec, "torch": _real_spec("json")}
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(specs))
        assert optional_deps.module_available("torch", "spandrel") is False

    def test_all_real_members_are_available(self):
        assert optional_deps.module_available("json", "logging") is True


class TestPixelsAvailable:
    @pytest.mark.parametrize("broken_name", PIXELS_MODULES)
    def test_import_failure_inside_present_package_is_unavailable(self, broken_name, monkeypatch):
        monkeypatch.setattr(optional_deps, "module_available", lambda *_names: True)
        imported: list[str] = []

        def import_module(name: str):
            imported.append(name)
            if name == broken_name:
                raise ImportError("numpy.core.multiarray failed to import")
            return object()

        monkeypatch.setattr(importlib, "import_module", import_module)

        assert optional_deps.pixels_available() is False
        assert broken_name in imported

    def test_cli_guard_converts_an_internal_import_failure_to_the_install_hint(self, capsys, monkeypatch):
        from remove_ai_watermarks.cli import _pixels_required

        monkeypatch.setattr(optional_deps, "module_available", lambda *_names: True)

        def import_module(_name: str):
            raise ImportError("numpy.core.multiarray failed to import")

        monkeypatch.setattr(importlib, "import_module", import_module)

        @_pixels_required
        def broken_command() -> None:
            raise ImportError("numpy.core.multiarray failed to import")

        with pytest.raises(SystemExit) as caught:
            broken_command()

        assert caught.value.code == 1
        assert VISIBLE_EXTRA in capsys.readouterr().out


class TestGuardsUseSharedHelper:
    def test_trustmark_is_available_rejects_namespace_remnant(self, monkeypatch):
        from remove_ai_watermarks import trustmark_detector

        ns_spec = importlib.machinery.ModuleSpec("trustmark", loader=None, is_package=True)
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({"trustmark": ns_spec}))
        assert trustmark_detector.is_available() is False
