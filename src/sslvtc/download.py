"""Download MarineCadastre (US Coast Guard NavCenter) daily AIS zips.

URL pattern (confirmed via sibling GenAIS project):
  https://coast.noaa.gov/htdata/CMSP/AISHandler/{year}/AIS_{year}_{MM}_{DD}.zip
Each zip holds one CSV for one day, US/CA/MX coastal coverage.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import requests
from tqdm import tqdm

from .config import PipelineConfig


def _days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _days_for_month(dl, month: int) -> list[int]:
    n = _days_in_month(dl.year, month)
    if dl.days is not None:
        return [d for d in dl.days if 1 <= d <= n]
    all_days = list(range(1, n + 1))
    if dl.max_days_per_month and dl.max_days_per_month < n:
        # evenly spaced sample across the month
        import numpy as np
        idx = np.linspace(0, n - 1, num=dl.max_days_per_month).round().astype(int)
        return [all_days[i] for i in sorted(set(idx))]
    return all_days


def daily_urls(cfg: PipelineConfig) -> list[tuple[str, str]]:
    """Return (url, filename) for the configured months/days."""
    dl = cfg.download
    out: list[tuple[str, str]] = []
    for month in dl.months:
        for day in _days_for_month(dl, month):
            url = dl.url_template.format(year=dl.year, month=month, day=day)
            out.append((url, Path(url).name))
    return out


def download_all(cfg: PipelineConfig, *, overwrite: bool = False) -> list[Path]:
    """Download all configured daily zips into paths.raw. Skips existing files."""
    raw_dir = Path(cfg.paths.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for url, name in tqdm(daily_urls(cfg), desc="download", unit="file"):
        dest = raw_dir / name
        if dest.exists() and not overwrite:
            saved.append(dest)
            continue
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            tqdm.write(f"skip {name}: {exc}")
            continue
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
        saved.append(dest)
    return saved
