"""Locate and load configuration files from the repo-level config/ folder."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
OUTPUT_DIR = REPO_ROOT / "outputs"


def load_yaml(name: str) -> dict:
    """Load config/<name> (e.g. 'curve.yaml') as a dict."""
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)
