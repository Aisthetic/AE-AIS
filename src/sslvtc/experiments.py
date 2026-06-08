"""Experiment harness reproducing the paper's tables (trends, on the bbox subset).

Each function returns a DataFrame and writes CSV/plots into paths.results. Run
after `extract` has produced processed/index.parquet.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from .baselines import run_classical_baselines, run_mlp
from .config import EncodingConfig, PipelineConfig
from .plotting import plot_accuracy_curves, plot_confusion
from .train import train, train_supervised_classifier

# Paper Table 5: beta grid per labeled fraction.
BETA_GRID = {0.05: 10.0, 0.20: 10.0, 0.40: 100.0, 0.60: 1000.0}


def _with(cfg: PipelineConfig, **train_overrides) -> PipelineConfig:
    return dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **train_overrides))


def _with_encoding(cfg: PipelineConfig, **enc_overrides) -> PipelineConfig:
    return dataclasses.replace(cfg, encoding=dataclasses.replace(cfg.encoding, **enc_overrides))


def _save(cfg: PipelineConfig, df: pd.DataFrame, name: str) -> Path:
    out = Path(cfg.paths.results)
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    df.to_csv(p, index=False)
    return p


def table2_method_comparison(cfg: PipelineConfig) -> pd.DataFrame:
    """Table 2: classical (raw) + MLP/CNN with raw vs seven-hot, fully supervised."""
    c = _with(cfg, labeled_fraction=1.0)
    classical = run_classical_baselines(cfg.paths.processed, cfg.encoding, mode="raw")
    rows = [
        {"method": "SVM (raw)", "accuracy": round(classical["SVM"] * 100, 2)},
        {"method": "DecisionTree (raw)", "accuracy": round(classical["DecisionTree"] * 100, 2)},
        {"method": "KNN (raw)", "accuracy": round(classical["KNN"] * 100, 2)},
        {"method": "MLP (raw)", "accuracy": round(classical["MLP"] * 100, 2)},
        {"method": "CNN (raw)",
         "accuracy": round(train_supervised_classifier(c, mode="raw") * 100, 2)},
        {"method": "MLP + seven-hot",
         "accuracy": round(run_mlp(cfg.paths.processed, cfg.encoding, "sevenhot") * 100, 2)},
        {"method": "CNN + seven-hot (SSL-VTC classifier)",
         "accuracy": round(train_supervised_classifier(c, mode="sevenhot") * 100, 2)},
    ]
    df = pd.DataFrame(rows)
    _save(cfg, df, "table2_method_comparison.csv")
    return df


def table3_static_ablation(cfg: PipelineConfig) -> pd.DataFrame:
    """Table 3: drop static attrs (WO-LEN/WID/DRA/LWD) vs full. Supervised CNN classifier."""
    c = _with(cfg, labeled_fraction=1.0)
    variants = {
        "WO-LWD": dict(use_len=False, use_wid=False, use_dra=False),
        "WO-LEN": dict(use_len=False),
        "WO-WID": dict(use_wid=False),
        "WO-DRA": dict(use_dra=False),
        "Full": dict(),
    }
    rows = []
    for name, enc in variants.items():
        cc = _with_encoding(c, **enc)
        acc = train_supervised_classifier(cc, mode="sevenhot")
        rows.append({"variant": name, "accuracy": round(acc * 100, 2)})
    df = pd.DataFrame(rows)
    _save(cfg, df, "table3_static_ablation.csv")
    return df


def table4_ssl_vs_baseline(cfg: PipelineConfig, fractions=(0.05, 0.20, 0.40, 0.60)) -> pd.DataFrame:
    """Table 4: SSL-VTC vs labeled-only baseline across labeled fractions + Fig 7 curves."""
    rows = []
    for frac in fractions:
        beta = BETA_GRID.get(frac, cfg.train.beta)
        c = _with(cfg, labeled_fraction=frac, beta=beta)
        ssl = train(c, supervised_only=False, progress=False)
        base = train(c, supervised_only=True, progress=False)
        rows.append({
            "labeled_fraction": frac, "beta": beta,
            "baseline_acc": round(base["test_acc"] * 100, 2),
            "sslvtc_acc": round(ssl["test_acc"] * 100, 2),
        })
        if frac in (0.20, 0.40):
            pct = int(frac * 100)
            plot_accuracy_curves(
                {"SSL-VTC": ssl["history"]["test_acc"], "baseline": base["history"]["test_acc"]},
                Path(cfg.paths.results) / f"fig7_curves_{pct}pct.png",
                title=f"{pct}% labeled",
            )
            plot_confusion(ssl["confusion_matrix"],
                           Path(cfg.paths.results) / f"confusion_sslvtc_{pct}pct.png",
                           title=f"SSL-VTC {pct}% labeled")
            plot_confusion(base["confusion_matrix"],
                           Path(cfg.paths.results) / f"confusion_baseline_{pct}pct.png",
                           title=f"baseline {pct}% labeled")
    df = pd.DataFrame(rows)
    _save(cfg, df, "table4_ssl_vs_baseline.csv")
    return df


def table6_beta_sensitivity(cfg: PipelineConfig, betas=(1.0, 10.0, 100.0, 1000.0)) -> pd.DataFrame:
    """Table 6: beta sweep at 40% labeled."""
    rows = []
    for beta in betas:
        c = _with(cfg, labeled_fraction=0.40, beta=beta)
        res = train(c, supervised_only=False, progress=False)
        rows.append({"beta": beta, "accuracy": round(res["test_acc"] * 100, 2)})
    df = pd.DataFrame(rows)
    _save(cfg, df, "table6_beta.csv")
    return df


def table7_missing_static(cfg: PipelineConfig, avail=(0.05, 0.20, 0.60)) -> pd.DataFrame:
    """Table 7: missing-static handling at 20% labeled.

    `avail` = fraction of trajectories whose static info is kept available; the
    rest have their static fields withheld (set NaN) and reconstructed via either
    zero-fill or mean-fill. Compares both fills x baseline/SSL-VTC.
    """
    rows = []
    c = _with(cfg, labeled_fraction=0.20)
    for frac in avail:
        for fill in ("zero", "mean"):
            ssl = train(c, supervised_only=False, mode="sevenhot", missing_static_fill=fill,
                        static_available_fraction=frac, progress=False)
            base = train(c, supervised_only=True, mode="sevenhot", missing_static_fill=fill,
                         static_available_fraction=frac, progress=False)
            rows.append({
                "available_static": frac, "fill": fill,
                "baseline_acc": round(base["test_acc"] * 100, 2),
                "sslvtc_acc": round(ssl["test_acc"] * 100, 2),
            })
    df = pd.DataFrame(rows)
    _save(cfg, df, "table7_missing_static.csv")
    return df
