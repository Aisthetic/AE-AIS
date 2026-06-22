"""Multi-seed experiment runner with GPU dispatcher.

run_repeated(): run one (split, fraction, method) config over N seeds, returns DataFrame.
gpu_dispatch(): round-robin assign jobs across available GPUs, run 3 concurrently.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .config import PipelineConfig


def _config_to_dict(cfg: PipelineConfig) -> dict:
    """Recursively convert frozen dataclass to plain dict."""
    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(v) for v in obj]
        return obj
    return _to_dict(cfg)


def run_repeated(
    cfg: PipelineConfig,
    *,
    n_seeds: int = 3,
    seeds: list[int] | None = None,
    supervised_only: bool = False,
    split_column: str = "split",
    progress: bool = False,
) -> pd.DataFrame:
    """Run training for each seed; return per-seed rows + mean±std summary.

    Uses the current process (no subprocess) — for subprocess-based GPU dispatch,
    use gpu_dispatch().
    """
    from .train import train

    seeds = seeds or list(range(42, 42 + n_seeds))
    rows = []
    for seed in seeds:
        c = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=seed))
        res = train(c, supervised_only=supervised_only, progress=progress, split_column=split_column)
        rows.append({
            "seed": seed,
            "split_protocol": split_column,
            "labeled_fraction": cfg.train.labeled_fraction,
            "beta": cfg.train.beta,
            "supervised_only": supervised_only,
            "test_acc": round(res["test_acc"] * 100, 4),
            "test_f1": round(res["test_f1"] * 100, 4),
            "test_balanced_acc": round(res["test_balanced_acc"] * 100, 4),
            "best_val_f1": round(res["best_val_f1"] * 100, 4),
        })

    df = pd.DataFrame(rows)
    # Append mean±std summary rows
    for col in ("test_acc", "test_f1", "test_balanced_acc"):
        mean = df[col].mean()
        std = df[col].std(ddof=1) if len(df) > 1 else 0.0
        df.loc[len(df)] = {
            "seed": "mean±std",
            "split_protocol": split_column,
            "labeled_fraction": cfg.train.labeled_fraction,
            "beta": cfg.train.beta,
            "supervised_only": supervised_only,
            col: f"{mean:.2f}±{std:.2f}",
            **{c: "" for c in ("test_acc", "test_f1", "test_balanced_acc", "best_val_f1") if c != col},
        }
    return df


# ---------------------------------------------------------------------------
# GPU dispatcher — subprocess-based for true parallelism
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = """\
import json, sys, dataclasses
from src.sslvtc.config import load_config, _build, PipelineConfig
from src.sslvtc.train import train

payload = json.loads(sys.argv[1])
cfg = _build(PipelineConfig, payload["cfg"])
res = train(
    cfg,
    supervised_only=payload["supervised_only"],
    split_column=payload["split_column"],
    progress=False,
    tag=payload["tag"],
)
out = {
    "seed": cfg.train.seed,
    "labeled_fraction": cfg.train.labeled_fraction,
    "beta": cfg.train.beta,
    "supervised_only": payload["supervised_only"],
    "split_protocol": payload["split_column"],
    "test_acc": res["test_acc"],
    "test_f1": res["test_f1"],
    "test_balanced_acc": res["test_balanced_acc"],
    "best_val_f1": res["best_val_f1"],
}
print(json.dumps(out))
"""


def gpu_dispatch(
    cfg: PipelineConfig,
    jobs: list[dict],
    *,
    n_concurrent: int | None = None,
    gpu_ids: list[int] | None = None,
    results_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run a list of job specs in parallel across GPUs.

    Each job dict may contain: seed, labeled_fraction, beta, supervised_only, split_column.
    Jobs are dispatched round-robin across gpu_ids; up to n_concurrent run at once.

    Returns a DataFrame with one row per job plus a mean±std summary.

    Example::
        jobs = [
            {"seed": 42, "labeled_fraction": 0.20, "supervised_only": False},
            {"seed": 43, "labeled_fraction": 0.20, "supervised_only": False},
            {"seed": 44, "labeled_fraction": 0.20, "supervised_only": False},
        ]
        df = gpu_dispatch(cfg, jobs)
    """
    import concurrent.futures

    n_gpus = torch_device_count()
    if gpu_ids is None:
        gpu_ids = list(range(max(n_gpus, 1)))
    if n_concurrent is None:
        n_concurrent = len(gpu_ids)

    results_dir = Path(results_dir or cfg.paths.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write worker script to a temp file
    script_file = results_dir / "_worker.py"
    script_file.write_text(_WORKER_SCRIPT)

    cfg_dict = _config_to_dict(cfg)

    def _run_job(job: dict, gpu_id: int) -> dict:
        seed = job.get("seed", cfg.train.seed)
        frac = job.get("labeled_fraction", cfg.train.labeled_fraction)
        beta = job.get("beta", cfg.train.beta)
        sup = job.get("supervised_only", False)
        split_col = job.get("split_column", "split")

        job_cfg = cfg_dict.copy()
        job_cfg["train"] = {**cfg_dict["train"], "seed": seed, "labeled_fraction": frac, "beta": beta}
        tag = f"gpu{gpu_id}_seed{seed}_f{int(frac*100)}_{'sup' if sup else 'ssl'}_{split_col}"

        payload = json.dumps({
            "cfg": job_cfg,
            "supervised_only": sup,
            "split_column": split_col,
            "tag": tag,
        })

        env = os.environ.copy()
        if torch_device_count() > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[gpu_id % len(gpu_ids)])

        proc = subprocess.run(
            [sys.executable, str(script_file), payload],
            capture_output=True, text=True, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"job failed (gpu={gpu_id}, seed={seed}):\n{proc.stderr[-2000:]}")
        # Last line of stdout is the JSON result
        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f"no JSON output from job (gpu={gpu_id}, seed={seed})\nstdout:\n{proc.stdout[-1000:]}")

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = {
            pool.submit(_run_job, job, i % len(gpu_ids)): job
            for i, job in enumerate(jobs)
        }
        for fut in concurrent.futures.as_completed(futures):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    # Numeric summary per (labeled_fraction, supervised_only, split_protocol)
    for col in ("test_acc", "test_f1", "test_balanced_acc"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def torch_device_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def make_jobs(
    seeds: list[int],
    fractions: list[float],
    supervised_only_vals: list[bool],
    split_columns: list[str],
    beta_grid: dict[float, float] | None = None,
) -> list[dict]:
    """Cartesian product of (seed × fraction × supervised_only × split_column)."""
    from itertools import product
    jobs = []
    for seed, frac, sup, split_col in product(seeds, fractions, supervised_only_vals, split_columns):
        beta = (beta_grid or {}).get(frac, 10.0)
        jobs.append({
            "seed": seed,
            "labeled_fraction": frac,
            "beta": beta,
            "supervised_only": sup,
            "split_column": split_col,
        })
    return jobs
