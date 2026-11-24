"""Without the optional astronomy packages the Studio still starts and says plainly why a FITS file will not open."""
from __future__ import annotations

import pytest

from conftest import ROOT


def test_fits_without_astropy_is_said_plainly(monkeypatch):
    from studio.engine.io import fits as FX
    from studio.engine.io import scanfile as S
    monkeypatch.setattr(FX, "available", lambda: False)
    d = ROOT / "outputs" / "pytest_fits"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "not_readable_here.fits"
    p.write_bytes(b"SIMPLE  =                    T" + b" " * 2850)
    assert S.is_scan(str(p))
    with pytest.raises(S.UnsupportedScan, match="astropy"):
        S.open_scan(str(p))
    from studio.engine.io.export import is_own_export
    assert not is_own_export(str(p))


def test_export_dialog_offers_fits_only_when_it_can(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication
    from studio.app import main as M
    from studio.engine.io import fits as FX
    app = QApplication.instance() or QApplication([])
    st = M.Settings()
    st.d = {"export_dir": str(tmp_path)}
    st.set = lambda k, v: st.d.__setitem__(k, v)
    monkeypatch.setattr(FX, "available", lambda: False)
    dlg = M.ExportDialog(["a.CR3"], st)
    assert dlg.fmt.count() == 5 and "FITS" in dlg.fmt.itemText(4)
    assert not dlg.fmt.model().item(4).isEnabled()
    monkeypatch.setattr(FX, "available", lambda: True)
    dlg2 = M.ExportDialog(["a.CR3"], st)
    assert dlg2.fmt.model().item(4).isEnabled()
    dlg2.fmt.setCurrentIndex(4)
    assert dlg2.spec().format == "fits"
    del app
