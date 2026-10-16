"""Render README / IC summary from templates so every quoted number comes from outputs/metrics.json.

Placeholder syntax inside a template:  {{dotted.path.into.metrics|format_spec}}
e.g. {{phase5.portfolios.C.net_spread_income|.3f}} or {{inputs.limits.hy_max|.0f}}.
The format spec is a Python format spec ('.3f', '+.2f', '.1%', ...). Without one, str() is used.
Lines containing <!-- static --> are exempt from the "no hand-typed numbers" test
(used only for worked toy examples that are test fixtures, not model outputs).
"""

import re
from pathlib import Path

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*(?:\|\s*([^}]*?))?\s*\}\}")
STATIC_MARK = "<!-- static -->"


def lookup(metrics: dict, path: str):
    node = metrics
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"metrics has no '{path}' (failed at '{key}')")
        node = node[key]
    return node


def render(template: str, metrics: dict) -> str:
    def sub(m):
        value = lookup(metrics, m.group(1))
        fmt = m.group(2)
        return format(value, fmt) if fmt else str(value)
    return PLACEHOLDER.sub(sub, template)


def render_file(template_path: Path, out_path: Path, metrics: dict) -> None:
    out_path.write_text(render(template_path.read_text(), metrics))


def hand_typed_numbers(template: str) -> list[str]:
    """Decimal numbers, percentages and dollar figures written outside placeholders (should be none)."""
    found = []
    for line in template.splitlines():
        if STATIC_MARK in line:
            continue
        text = PLACEHOLDER.sub("", line)
        found += re.findall(r"\d+\.\d+|\d+(?:\.\d+)?\s?%|\$\s?\d[\d,.]*", text)
    return found
