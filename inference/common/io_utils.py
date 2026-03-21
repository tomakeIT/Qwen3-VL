from __future__ import annotations

import json
from typing import Any, List

import yaml

from utils.utils import dict_to_namespace


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_rows(path: str) -> List[Any]:
    rows: List[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json_or_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def load_config_namespace(path: str):
    return dict_to_namespace(load_json_or_yaml(path))
