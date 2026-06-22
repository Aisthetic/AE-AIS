"""Experiment harness reproducing the paper's tables + leakage study.

Each function returns a DataFrame and writes CSV/plots into paths.results. Run
after `extract` has produced processed/index.parquet.
"""
from __future__ import annotations

import dataclasses
import json
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


def table3_static_ablation(
    cfg: PipelineConfig,
    split_column: str = "split",
) -> pd.DataFrame:
    """Table 3: drop static attrs (WO-LEN/WID/DRA/LWD) vs full. Supervised CNN classifier.

    Pass split_column='split_vd' to run under the vessel-disjoint protocol.
    """
    c = _with(cfg, labeled_fraction=1.0)
    variants = {
        "WO-LWD": dict(use_len=False, use_wid=False, use_dra=False),
        "WO-LEN": dict(use_len=False),
        "WO-WID": dict(use_wid=False),
        "WO-DRA": dict(use_dra=False),
        "Full": dict(),
    }
    suffix = "_vd" if split_column == "split_vd" else ""
    csv_name = f"table3_static_ablation{suffix}.csv"
    # fishing is class_idx for "fishing"; resolve from the project label map.
    from . import CLASS_TO_IDX
    fishing_idx = CLASS_TO_IDX.get("fishing")
    rows = []
    for name, enc in variants.items():
        cc = _with_encoding(c, **enc)
        m = train_supervised_classifier(cc, mode="sevenhot", split_column=split_column, return_metrics=True)
        recalls = m.get("per_class_recall", [])
        row = {
            "variant": name,
            "split_protocol": split_column,
            "accuracy": round(m["accuracy"] * 100, 2),
            "macro_f1": round(m["macro_f1"] * 100, 2),
            "balanced_acc": round(m["balanced_accuracy"] * 100, 2),
        }
        if fishing_idx is not None and fishing_idx < len(recalls):
            row["fishing_recall"] = round(recalls[fishing_idx] * 100, 2)
        rows.append(row)
        _save(cfg, pd.DataFrame(rows), csv_name)  # incremental: persist after each variant
    df = pd.DataFrame(rows)
    _save(cfg, df, csv_name)
    return df


def table3_leakage_comparison(cfg: PipelineConfig) -> pd.DataFrame:
    """Run static ablation under both temporal and vessel-disjoint splits; return combined DataFrame."""
    df_t = table3_static_ablation(cfg, split_column="split")
    df_vd = table3_static_ablation(cfg, split_column="split_vd")
    df = pd.concat([df_t, df_vd], ignore_index=True)
    _save(cfg, df, "table3_leakage_comparison.csv")
    return df


def table4_ssl_vs_baseline(
    cfg: PipelineConfig,
    fractions=(0.05, 0.20, 0.40, 0.60),
    split_column: str = "split",
) -> pd.DataFrame:
    """Table 4: SSL-VTC vs labeled-only baseline across labeled fractions + Fig 7 curves."""
    suffix = "_vd" if split_column == "split_vd" else ""
    csv_name = f"table4_ssl_vs_baseline{suffix}.csv"
    rows = []
    for frac in fractions:
        beta = BETA_GRID.get(frac, cfg.train.beta)
        c = _with(cfg, labeled_fraction=frac, beta=beta)
        pct = int(frac * 100)
        ssl = train(c, supervised_only=False, progress=False, split_column=split_column,
                    tag=f"sslvtc_f{pct}{suffix}")
        base = train(c, supervised_only=True, progress=False, split_column=split_column,
                     tag=f"baseline_f{pct}{suffix}")
        rows.append({
            "labeled_fraction": frac,
            "beta": beta,
            "split_protocol": split_column,
            "baseline_acc": round(base["test_acc"] * 100, 2),
            "baseline_f1": round(base["test_f1"] * 100, 2),
            "sslvtc_acc": round(ssl["test_acc"] * 100, 2),
            "sslvtc_f1": round(ssl["test_f1"] * 100, 2),
        })
        _save(cfg, pd.DataFrame(rows), csv_name)  # incremental: persist after each fraction
        if frac in (0.20, 0.40):
            pct = int(frac * 100)
            sfx = "_vd" if split_column == "split_vd" else ""
            plot_accuracy_curves(
                {"SSL-VTC": ssl["history"]["test_acc"], "baseline": base["history"]["test_acc"]},
                Path(cfg.paths.results) / f"fig7_curves_{pct}pct{sfx}.png",
                title=f"{pct}% labeled [{split_column}]",
            )
            plot_confusion(ssl["confusion_matrix"],
                           Path(cfg.paths.results) / f"confusion_sslvtc_{pct}pct{sfx}.png",
                           title=f"SSL-VTC {pct}% labeled [{split_column}]")
            plot_confusion(base["confusion_matrix"],
                           Path(cfg.paths.results) / f"confusion_baseline_{pct}pct{sfx}.png",
                           title=f"baseline {pct}% labeled [{split_column}]")
    df = pd.DataFrame(rows)
    _save(cfg, df, csv_name)
    return df


def leakage_inflation_table(cfg: PipelineConfig, fractions=(0.05, 0.20, 0.40, 0.60)) -> pd.DataFrame:
    """Phase 1.4: Δacc and Δf1 (temporal − vessel_disjoint) per method/fraction."""
    df_t = table4_ssl_vs_baseline(cfg, fractions=fractions, split_column="split")
    df_vd = table4_ssl_vs_baseline(cfg, fractions=fractions, split_column="split_vd")
    merged = df_t.merge(df_vd, on=["labeled_fraction", "beta"], suffixes=("_t", "_vd"))
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            "labeled_fraction": row["labeled_fraction"],
            "baseline_delta_acc": round(row["baseline_acc_t"] - row["baseline_acc_vd"], 2),
            "baseline_delta_f1": round(row["baseline_f1_t"] - row["baseline_f1_vd"], 2),
            "sslvtc_delta_acc": round(row["sslvtc_acc_t"] - row["sslvtc_acc_vd"], 2),
            "sslvtc_delta_f1": round(row["sslvtc_f1_t"] - row["sslvtc_f1_vd"], 2),
        })
    df = pd.DataFrame(rows)
    _save(cfg, df, "leakage_inflation.csv")
    return df


def table6_beta_sensitivity(cfg: PipelineConfig, betas=(1.0, 10.0, 100.0, 1000.0)) -> pd.DataFrame:
    """Table 6: beta sweep at 40% labeled."""
    rows = []
    for beta in betas:
        c = _with(cfg, labeled_fraction=0.40, beta=beta)
        res = train(c, supervised_only=False, progress=False)
        rows.append({
            "beta": beta,
            "accuracy": round(res["test_acc"] * 100, 2),
            "macro_f1": round(res["test_f1"] * 100, 2),
        })
    df = pd.DataFrame(rows)
    _save(cfg, df, "table6_beta.csv")
    return df


def table7_missing_static(cfg: PipelineConfig, avail=(0.05, 0.20, 0.60)) -> pd.DataFrame:
    """Table 7: missing-static handling at 20% labeled."""
    rows = []
    c = _with(cfg, labeled_fraction=0.20)
    for frac in avail:
        for fill in ("zero", "mean"):
            ssl = train(c, supervised_only=False, mode="sevenhot", missing_static_fill=fill,
                        static_available_fraction=frac, progress=False)
            base = train(c, supervised_only=True, mode="sevenhot", missing_static_fill=fill,
                         static_available_fraction=frac, progress=False)
            rows.append({
                "available_static": frac,
                "fill": fill,
                "baseline_acc": round(base["test_acc"] * 100, 2),
                "baseline_f1": round(base["test_f1"] * 100, 2),
                "sslvtc_acc": round(ssl["test_acc"] * 100, 2),
                "sslvtc_f1": round(ssl["test_f1"] * 100, 2),
            })
    df = pd.DataFrame(rows)
    _save(cfg, df, "table7_missing_static.csv")
    return df
