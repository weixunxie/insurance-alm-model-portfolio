"""Methodology: docs/methodology.md and docs/assumptions.md rendered as written, plus repository links."""

from dash import dcc, html

from dashboard.components import page_header, section
from dashboard.data import DOCS

REPO_URL = "https://github.com/weixunxie/insurance-alm-model-portfolio"
DOC_FILES = [("methodology.md", "Methodology notes"), ("assumptions.md", "Assumptions register")]


def read_doc(name: str) -> str:
    """Markdown as written; only HTML comments (the generated-file banner) are dropped."""
    lines = (DOCS / name).read_text().splitlines()
    return "\n".join(ln for ln in lines if not ln.strip().startswith("<!--"))


def layout():
    return html.Div([
        page_header("Methodology",
                    "The two documents below are rendered from the repository without edits. Every number in them "
                    "is generated from outputs/metrics.json by scripts/run_all.py."),
        html.P(["Source: ", html.A("GitHub repository", href=REPO_URL), " · ",
                html.A("README", href=f"{REPO_URL}#readme"), " · ",
                html.A("docs/methodology.md", href=f"{REPO_URL}/blob/main/docs/methodology.md"), " · ",
                html.A("docs/assumptions.md", href=f"{REPO_URL}/blob/main/docs/assumptions.md")],
               className="provenance"),
        *[section(title, html.Div(className="panel markdown", children=dcc.Markdown(read_doc(name))),
                  note=f"docs/{name}") for name, title in DOC_FILES],
    ])
