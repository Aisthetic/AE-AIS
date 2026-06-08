"""Command-line entry point: download | ingest | extract | train | eval | baselines."""
from __future__ import annotations

import argparse
import json

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sslvtc", description="SSL-VTC pipeline")
    parser.add_argument("--config", "-c", default=None, help="path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download MarineCadastre AIS zips")
    sub.add_parser("ingest", help="clean+filter+label raw files into parquet")
    sub.add_parser("extract", help="run 5-step extraction -> tensors + index")

    p_train = sub.add_parser("train", help="train SSL-VTC")
    p_train.add_argument("--supervised-only", action="store_true", help="labeled-only baseline")

    sub.add_parser("eval", help="evaluate saved/just-trained model on test split")
    sub.add_parser("baselines", help="run classical Table-2 baselines")

    p_exp = sub.add_parser("experiment", help="run a paper table")
    p_exp.add_argument("which", choices=["table2", "table3", "table4", "table6", "table7", "all"])

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "download":
        from .download import download_all
        saved = download_all(cfg)
        print(f"downloaded/kept {len(saved)} files in {cfg.paths.raw}")

    elif args.command == "ingest":
        from .ingest import ingest_all
        out = ingest_all(cfg)
        print(f"ingested -> {out}")

    elif args.command == "extract":
        from .extract import extract_all
        out = extract_all(cfg)
        print(f"extracted -> {out}")

    elif args.command == "train":
        from .train import save_result, train
        res = train(cfg, supervised_only=args.supervised_only)
        tag = "baseline" if args.supervised_only else "sslvtc"
        save_result(res, cfg.paths.results, tag)
        print(json.dumps({
            "best_val_acc": res["best_val_acc"],
            "test_acc": res["test_acc"],
            "alpha": res["alpha"],
            "n_labeled": res["n_labeled"],
            "n_unlabeled": res["n_unlabeled"],
            "input_shape": res["input_shape"],
        }, indent=2))

    elif args.command == "eval":
        from .train import train
        res = train(cfg)
        print(f"test_acc={res['test_acc']:.4f}\nconfusion=\n{res['confusion_matrix']}")

    elif args.command == "baselines":
        from .baselines import run_classical_baselines
        print(json.dumps(run_classical_baselines(cfg.paths.processed, cfg.encoding, mode="raw"), indent=2))

    elif args.command == "experiment":
        from . import experiments as exp
        fns = {
            "table2": exp.table2_method_comparison,
            "table3": exp.table3_static_ablation,
            "table4": exp.table4_ssl_vs_baseline,
            "table6": exp.table6_beta_sensitivity,
            "table7": exp.table7_missing_static,
        }
        which = list(fns) if args.which == "all" else [args.which]
        for name in which:
            print(f"\n=== {name} ===")
            print(fns[name](cfg).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
