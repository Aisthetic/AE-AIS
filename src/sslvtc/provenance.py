"""Provenance: dump effective config + git SHA + seed alongside every result."""
from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path


def _cfg_to_dict(cfg) -> dict:
    def _convert(obj):
        if dataclasses.is_dataclass(obj):
            return {f.name: _convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj
    return _convert(cfg)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def save_provenance(cfg, out_dir: str | Path, tag: str) -> Path:
    """Write {tag}_provenance.json with config + git SHA. Returns path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "git_sha": _git_sha(),
        "config": _cfg_to_dict(cfg),
    }
    p = out / f"{tag}_provenance.json"
    p.write_text(json.dumps(payload, indent=2))
    return p
