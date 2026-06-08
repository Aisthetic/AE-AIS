# SSL-VTC

Reimplementation of **SSL-VTC** (Duan, Ma, Miao, Zhang 2022, *Ocean & Coastal Management* 218:106015) —
a VAE-based semi-supervised model that classifies vessel trajectories into 4 ship types
(fishing, passenger, cargo, tanker) from MarineCadastre AIS data, training on labeled +
unlabeled trajectories jointly.

Method spec: [SSL-VTC_implementation.md](SSL-VTC_implementation.md).

## Environment

Requires **Python ≥3.10** (the system Python 3.9 is too old for the torch stack;
Python 3.14 is too new for torch wheels). This repo was built against Python 3.12.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Device is auto-selected: Apple **MPS** → CUDA → CPU (`train.device: auto` in config).

## Pipeline

Driven by `configs/default.yaml` (bounded bbox subset; edit `bbox` to change region/scale).

```bash
.venv/bin/sslvtc -c configs/default.yaml download   # MarineCadastre AIS zips (Jan–Jun 2019)
.venv/bin/sslvtc -c configs/default.yaml ingest     # clean + filter + label -> parquet
.venv/bin/sslvtc -c configs/default.yaml extract    # 5-step extraction -> seven-hot tensors + index
.venv/bin/sslvtc -c configs/default.yaml train      # train SSL-VTC (semi-supervised)
.venv/bin/sslvtc -c configs/default.yaml train --supervised-only   # labeled-only baseline
.venv/bin/sslvtc -c configs/default.yaml baselines  # classical baselines (raw)
```

### Experiments (paper tables → `results/`)

```bash
.venv/bin/sslvtc -c configs/default.yaml experiment table2   # method comparison (raw vs seven-hot; SVM/DT/KNN/MLP/CNN)
.venv/bin/sslvtc -c configs/default.yaml experiment table3   # static-info ablation (WO-LEN/WID/DRA/LWD)
.venv/bin/sslvtc -c configs/default.yaml experiment table4   # SSL vs baseline across label fractions (+ Fig 7 curves + confusion matrices)
.venv/bin/sslvtc -c configs/default.yaml experiment table6   # beta sensitivity at 40% labeled
.venv/bin/sslvtc -c configs/default.yaml experiment table7   # missing-static handling (zero vs mean fill)
.venv/bin/sslvtc -c configs/default.yaml experiment all      # run every table
```

Each writes a CSV (e.g. `results/table4_ssl_vs_baseline.csv`); Table 4 also emits
`fig7_curves_{20,40}pct.png` and `confusion_{sslvtc,baseline}_{20,40}pct.png`.

## Full run on a cluster (real Jan–Jun 2019 data)

Target config: [configs/gulf2019.yaml](configs/gulf2019.yaml) — US Gulf of Mexico
bbox, full Jan–Jun 2019, paper temporal split (train Jan–Apr / val May / test Jun).
Good 4-class balance (fishing/cargo/tanker/passenger).

```bash
scripts/run_full.sh configs/gulf2019.yaml      # download → ingest → extract → experiment all
# or submit the SLURM template:
sbatch scripts/slurm_sslvtc.sh                  # edit partition/gres/account first
```

- MarineCadastre daily files are **~290–335 MB zipped each** (URL host:
  `coast.noaa.gov/htdata/CMSP/AISDataHandler`). Full 6 months = **181 files (~52 GB)**.
- For a lighter run set `download.max_days_per_month: 4` (→ 24 files) or
  `download.days: [...]` in the config.
- On the cluster set `train.device: cuda` (the SLURM script does this for you).
- Split stages: data prep (download/ingest/extract) is CPU-heavy and can run as a
  separate non-GPU job; only `experiment` needs the GPU.

### Optional: real-data smoke from a sibling project

`scripts/load_genais.py` loads real NY-Harbor June 2022 AIS from the sibling
GenAIS project into `clean_{split}.parquet` (split by day-of-month). Useful to
exercise the pipeline on real messages, but that region is ferry-dominated
(~64% passenger, ~no fishing) so the 4-class accuracy is not meaningful — use the
Gulf config for real numbers.

## Smoke test (no download)

```bash
.venv/bin/python scripts/make_synthetic.py data/smoke/raw
.venv/bin/sslvtc -c configs/smoke.yaml ingest
.venv/bin/sslvtc -c configs/smoke.yaml extract
.venv/bin/sslvtc -c configs/smoke.yaml train
.venv/bin/python -m pytest tests/ -q
```

## Layout

```
src/sslvtc/
  config.py     frozen-dataclass config + YAML loader
  device.py     mps/cuda/cpu selection
  download.py   MarineCadastre daily-zip downloader
  ingest.py     CSV/zip -> filtered, labeled parquet (column aliasing, bbox, shiptype map)
  extract.py    paper Section 3 — 5-step extraction; stores raw normalized [T,7] tensors
  encoding.py   normalization + seven-hot / raw encoding (applied at load time)
  dataset.py    TrajectoryDataset (encode transform, missing-static withhold) + labeled/unlabeled split
  models.py     ConvBlock, Classifier, Encoder, Decoder, SSLVTC
  loss.py       Kingma M2 loss (L1 + L2 + alpha*L_clf)
  train.py      Algorithm 1 joint loop + evaluate
  baselines.py  SVM/DT/KNN/MLP classical baselines
  experiments.py  Tables 2/3/4/6/7 harness
  plotting.py   Fig 7 accuracy curves + confusion-matrix plots
  cli.py        entry point
```

## Notes / known deviations from the paper

- **Bounded bbox subset**, not the full ~277 GB US/CA/MX corpus. Trends reproduce; absolute accuracy will differ.
- **Seven-hot bin counts** are not given in the paper; defaults follow a Nguyen/TrAISformer-style scheme and are tunable in `encoding.bins`.
- **Conv flatten size** is computed dynamically (paper hard-codes 250). Decoder reconstructs to exact `(T, D)` via a final bilinear resize for robustness across chosen `T_fixed`/`D`.
