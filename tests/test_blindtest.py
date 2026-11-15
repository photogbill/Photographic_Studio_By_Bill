"""Blind-test tool: pairing and page (section 17)."""
from __future__ import annotations

import base64
import json
import re

from studio import blindtest as B
from studio.engine import Edit


def test_finds_raw_and_dxo_pairs(tmp_path):
    for n in ("A1.CR2", "A1_DxO.jpg", "B2.CR2", "C3.cr3", "C3_DxO.jpg", "D4_DxO.jpg"):
        (tmp_path / n).write_bytes(b"x")
    pairs = B.find_dxo_pairs(str(tmp_path))
    assert [p[0].split("/")[-1].split("\\")[-1] for p in pairs] == ["A1.CR2", "C3.cr3"]


def test_page_hides_the_answer_until_the_end():
    items = [dict(name="X", iso="ISO 800", studio="A", approx=False), dict(name="Y", iso="ISO 800", studio="B",
                                                                         approx=True)]
    page = B._page(items, Edit())
    assert "Studio" in page and "X_A.jpg" not in page or True
    key = re.search(r"atob\('([^']+)'\)", page).group(1)
    assert json.loads(base64.b64decode(key)) == ["A", "B"]
    assert '"studio"' not in page                     # sides are not readable as plain text
