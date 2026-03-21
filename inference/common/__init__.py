"""Shared helpers for inference CLI scripts."""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(INFERENCE_DIR)

for path in (INFERENCE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
