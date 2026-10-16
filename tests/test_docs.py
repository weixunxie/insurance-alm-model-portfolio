"""Every number quoted in README.md, the IC summary and methodology.md must come from outputs/metrics.json.

Run scripts/run_all.py first: it writes metrics.json and renders the docs from docs/templates/.
"""

import json

import pytest

from alm.config import OUTPUT_DIR, REPO_ROOT
from alm.report import hand_typed_numbers, render

DOCS = {"README.md.tmpl": REPO_ROOT / "README.md",
        "investment_committee_summary.md.tmpl": REPO_ROOT / "docs" / "investment_committee_summary.md",
        "methodology.md.tmpl": REPO_ROOT / "docs" / "methodology.md",
        "interview_notes.md.tmpl": REPO_ROOT / "docs" / "interview_notes.md"}
TEMPLATES = REPO_ROOT / "docs" / "templates"


@pytest.fixture(scope="module")
def metrics():
    path = OUTPUT_DIR / "metrics.json"
    if not path.exists():
        pytest.fail("outputs/metrics.json missing: run scripts/run_all.py first")
    return json.loads(path.read_text())


@pytest.mark.parametrize("template", list(DOCS))
def test_doc_matches_render_from_metrics(template, metrics):
    expected = render((TEMPLATES / template).read_text(), metrics)
    assert DOCS[template].read_text() == expected, f"{DOCS[template].name} is stale: rerun scripts/run_all.py"


@pytest.mark.parametrize("template", list(DOCS))
def test_no_hand_typed_numbers(template):
    assert hand_typed_numbers((TEMPLATES / template).read_text()) == []
