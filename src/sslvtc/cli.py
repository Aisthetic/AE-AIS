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
    p_train.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    p_train.add_argument("--tag", default=None, help="checkpoint/result tag")
    p_train.add_argument("--split-column", default="split", help="index.parquet column for splits")

    p_split = sub.add_parser("vessel-split", help="compute vessel-disjoint split on index.parquet")
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--train-frac", type=float, default=0.70)
    p_split.add_argument("--val-frac", type=float, default=0.15)
    p_split.add_argument("--report", action="store_true", help="print overlap stats")

    sub.add_parser("eval", help="evaluate saved/just-trained model on test split")
    sub.add_parser("baselines", help="run classical Table-2 baselines")

    p_sweep = sub.add_parser("run-sweep", help="multi-seed GPU-dispatched sweep")
    p_sweep.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p_sweep.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.20, 0.40, 0.60])
    p_sweep.add_argument("--splits", nargs="+", default=["split", "split_vd"],
                         help="split columns to sweep over")
    p_sweep.add_argument("--supervised-only", action="store_true")
    p_sweep.add_argument("--n-concurrent", type=int, default=None)
    p_sweep.add_argument("--out", default=None, help="output CSV path")

    p_exp = sub.add_parser("experiment", help="run a paper table")
    p_exp.add_argument("which", choices=[
        "table2", "table3", "table3-vd", "table3-leakage",
        "table4", "table4-vd", "leakage-inflation",
        "table6", "table7", "all",
    ])

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
        tag = args.tag or ("baseline" if args.supervised_only else "sslvtc")
        res = train(
            cfg,
            supervised_only=args.supervised_only,
            resume=args.resume,
            tag=tag,
            split_column=args.split_column,
        )
        save_result(res, cfg.paths.results, tag, cfg=cfg)
        print(json.dumps({
            "best_val_acc": res["best_val_acc"],
            "best_val_f1": res["best_val_f1"],
            "test_acc": res["test_acc"],
            "test_f1": res["test_f1"],
            "test_balanced_acc": res["test_balanced_acc"],
            "test_per_class_recall": res["test_per_class_recall"],
            "alpha": res["alpha"],
            "n_labeled": res["n_labeled"],
            "n_unlabeled": res["n_unlabeled"],
            "input_shape": list(res["input_shape"]),
        }, indent=2))

    elif args.command == "vessel-split":
        from .splits import make_vessel_disjoint_split, report_overlap
        idx = make_vessel_disjoint_split(
            cfg.paths.processed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        print(f"vessel-disjoint split written to index.parquet (split_vd column)")
        print(idx["split_vd"].value_counts().to_string())
        if args.report:
            stats = report_overlap(cfg.paths.processed)
            print(json.dumps(stats, indent=2))

    elif args.command == "eval":
        from .train import train
        res = train(cfg)
        print(f"test_acc={res['test_acc']:.4f}\nconfusion=\n{res['confusion_matrix']}")

    elif args.command == "baselines":
        from .baselines import run_classical_baselines
        print(json.dumps(run_classical_baselines(cfg.paths.processed, cfg.encoding, mode="raw"), indent=2))

    elif args.command == "run-sweep":
        from .runner import gpu_dispatch, make_jobs
        from . import experiments as exp
        jobs = make_jobs(
            seeds=args.seeds,
            fractions=args.fractions,
            supervised_only_vals=[True, False] if not args.supervised_only else [True],
            split_columns=args.splits,
            beta_grid=exp.BETA_GRID,
        )
        print(f"dispatching {len(jobs)} jobs across {len(args.seeds)} seeds × "
              f"{len(args.fractions)} fractions × {len(args.splits)} splits...")
        df = gpu_dispatch(cfg, jobs, n_concurrent=args.n_concurrent)
        out_path = args.out or str(Path(cfg.paths.results) / "sweep_results.csv")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nresults -> {out_path}")
        print(df.to_string(index=False))

    elif args.command == "experiment":
        from . import experiments as exp
        fns = {
            "table2": exp.table2_method_comparison,
            "table3": exp.table3_static_ablation,
            "table3-vd": lambda c: exp.table3_static_ablation(c, split_column="split_vd"),
            "table3-leakage": exp.table3_leakage_comparison,
            "table4": exp.table4_ssl_vs_baseline,
            "table4-vd": lambda c: exp.table4_ssl_vs_baseline(c, split_column="split_vd"),
            "leakage-inflation": exp.leakage_inflation_table,
            "table6": exp.table6_beta_sensitivity,
            "table7": exp.table7_missing_static,
        }
        core = ["table2", "table3", "table3-leakage", "table4", "leakage-inflation", "table6", "table7"]
        which = core if args.which == "all" else [args.which]
        for name in which:
            print(f"\n=== {name} ===")
            print(fns[name](cfg).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
