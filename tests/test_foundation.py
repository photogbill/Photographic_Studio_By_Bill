"""Phase 0 rules: app-folder paths, logging, privacy scrubbing, CLI wiring."""
from __future__ import annotations

import os

from studio import log as slog
from studio import paths


def test_default_folders_live_inside_the_app_folder():
    for name in ("LOGS", "CACHE", "SETTINGS", "OUTPUTS", "CALIBRATION_LOCAL"):
        p = getattr(paths, name)
        assert paths.APP_ROOT in p.parents
    assert paths.APP_ROOT in paths.temp_dir().parents


def test_scrub_removes_private_fields():
    d = {"model": "R5", "SerialNumber": "123", "gps": {"lat": 1}, "nested": [{"InternalSerial": "x", "iso": 100}]}
    s = slog.scrub(d)
    assert s["model"] == "R5" and s["SerialNumber"] == "<removed>" and s["gps"] == "<removed>"
    assert s["nested"][0]["InternalSerial"] == "<removed>" and s["nested"][0]["iso"] == 100


def test_stage_logging_writes_a_line():
    lg = slog.get("test")
    with slog.stage("unit-test-stage", lg, n=1):
        pass
    logfile = paths.get("LOGS") / "tests.log"
    assert logfile.exists()
    assert "unit-test-stage" in logfile.read_text(encoding="utf-8")


def test_diagnostic_report_has_versions_and_no_serials():
    text = slog.diagnostic_report(tail_lines=20)
    assert "Photographic Studio By Bill" in text and "numpy" in text
    assert "SerialNumber" not in text


def test_cli_version_and_parser(capsys):
    from studio import cli
    assert cli.main(["version"]) == 0
    assert "Photographic Studio By Bill" in capsys.readouterr().out
    a = cli.build_parser().parse_args(["develop", "x.cr3", "--format", "png", "--space", "srgb"])
    assert a.format == "png" and a.space == "srgb"


def test_cli_collect_filters_non_raws(tmp_path):
    from studio import cli
    (tmp_path / "a.CR3").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.cr2").write_bytes(b"x")
    assert [os.path.basename(p) for p in cli.collect([str(tmp_path)])] == ["a.CR3"]
    assert sorted(os.path.basename(p) for p in cli.collect([str(tmp_path)], recursive=True)) == ["a.CR3", "c.cr2"]


def test_engine_package_does_not_import_gui():
    """Checked in a fresh interpreter: test runners and plugins may load Qt themselves."""
    import subprocess
    import sys
    from studio.paths import APP_ROOT
    code = ("import sys; import studio.engine, studio.engine.api, studio.engine.pipeline.pipeline; "
            "sys.exit(1 if any(m.startswith(('PySide6', 'PyQt')) for m in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=str(APP_ROOT)).returncode == 0
