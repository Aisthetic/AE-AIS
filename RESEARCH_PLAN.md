# SSL-VTC++ — Research & Engineering Plan

**Goal:** turn the current SSL-VTC reimplementation into a publishable paper. The contribution is **not** "beat 88.96% by a point." It is: *expose vessel-identity leakage in AIS trajectory-classification benchmarks, propose a leakage-controlled evaluation, and a method that wins under the honest protocol.*

This document is the implementation spec. It assumes the reader knows the codebase layout in [README.md](README.md) and the method spec in [SSL-VTC_implementation.md](SSL-VTC_implementation.md).

---

## 0. Current state (facts, not vibes)

**Reproduction works.** Pipeline `download → ingest → extract → train` runs end-to-end on real Gulf-of-Mexico 2019 AIS.

- Data on the 10T volume: `/mnt/storage_1_10T/zezzahed/AIS_Data/gulf2019/{raw,interim,processed,results}`.
- Config: [configs/gulf2019_storage.yaml](configs/gulf2019_storage.yaml) (absolute paths to the volume; `device: cuda`, `batch_size: 512`, `num_workers: 16`).
- Extraction output: **69,420 trajectories** — train 41,465 / val 14,115 / test 13,840.
- Best training run so far: **val 88.0% / test 87.9% @ epoch 20**, 20% labeled, β=10. Paper target 88.96%. Reproduction is sound.

**Parallelization already done:**
- [src/sslvtc/ingest.py](src/sslvtc/ingest.py): `ingest_all(cfg, workers)` — `ProcessPoolExecutor` over files. 181 files in ~2.5 min on 32 cores.
- [src/sslvtc/extract.py](src/sslvtc/extract.py): `extract_split(df, cfg, n_workers)` chunks MMSIs across workers; `extract_all(cfg, workers)` parallel tensor save. Full extract ~15 s.
- [src/sslvtc/train.py](src/sslvtc/train.py): GPU training, `pin_memory`, `_TrainStep` wrapper. **Note:** `nn.DataParallel` is incompatible with the M2 sub-module routing (encoder/decoder/classifier are called individually inside the loss, not via one `forward`) — it triggers a CUDA scatter assert. Single-GPU only. This is fine: the model is tiny (~17 GB on the RTX 6000). **Use the 3 GPUs for parallel *experiments* (seed × fraction × split), not data-parallel one model.**

**Known fragilities (must fix before any results are trustworthy):**
1. **No checkpointing.** `save_result` ([train.py:194](src/sslvtc/train.py#L194)) runs only after all 50 epochs. We already lost a 32-epoch run to a crash. **Non-negotiable fix.**
2. **Noisy plateau.** Converges ~epoch 13, then oscillates 86–88%. No LR schedule, no EMA, no early stopping.
3. **Single seed.** Every number is one draw. No variance → not publishable.
4. **Accuracy-only metric.** Heavy class imbalance; accuracy hides minority behavior. Need macro-F1, per-class recall, balanced accuracy.

---

## 1. The core finding: vessel-identity leakage

Measured directly on `processed/index.parquet` (columns: `traj_id, split, label, label_idx, mmsi, path`):

| Quantity | Value |
|---|---|
| test vessels (MMSI) also present in train | **70.5%** (1465/2077) |
| val vessels also present in train | 78.5% (1694/2158) |
| **test trajectories whose vessel appears in train** | **77.3%** |

Static features — `LEN`, `WID`, `DRA` — are **constant per MMSI**. They are a perfect vessel fingerprint. The paper's Table 3 reports static info gives **+29.1%** accuracy over kinematic-only. Under a split where 77% of test vessels were seen in training, that gain is **partly memorization of vessel identity, not generalization of vessel behavior**.

The temporal split (train Jan–Apr / val May / test Jun) controls *time* leakage but does nothing about *identity* leakage. **This is the paper.**

**Hypothesis to confirm in Phase 1:** static-info gain (WO-LWD → Full) is large under the temporal split and **collapses** under a vessel-disjoint split. If true, that single figure carries the paper.

> **[RESULT — 2026-06-16] Hypothesis REFUTED. We are in the pivot branch.**
> Decision gate run on paper-faithful (static-complete) fullus2019 data, supervised CNN ablation under both splits:
>
> | metric | WO-LWD → Full (temporal, 80.9% leak) | WO-LWD → Full (vessel-disjoint, 0% leak) |
> |---|---|---|
> | accuracy | 72.86 → 86.04 (**+13.2**) | 72.47 → 86.24 (**+13.8**) |
> | macro-F1 | 70.72 → 85.18 (+14.5) | 73.40 → 87.09 (+13.7) |
> | fishing recall | 59.95 → 81.95 (+22.0) | 69.06 → **89.98** (+20.9) |
>
> The static gain does **not** collapse — it is identical with/without identity leakage. Full-model accuracy is the same across splits (86.0 vs 86.2), and fishing recall (the most identity-dependent class) is **higher** under the honest vessel-disjoint split, not lower. **There is no leakage inflation.** Static features (LEN/WID/DRA) encode class-typical info that transfers to unseen vessels, not an identity fingerprint.
>
> **Implication:** the paper's "static info helps" is sound and generalizes. The contribution pivots to (a) the static-completeness **selection bias** (§1b, solid) and (b) this **leakage audit** as a rigorous negative control — we measure 80.9% identity overlap and demonstrate it does not inflate the headline result.
>
> **[UPDATE — faithful model, caveat resolved]** The ~86 ceiling was the undisclosed static **bin resolution** (bin sweep: 10/20/10→86.0, 20/40/20→88.6, 30/60/30→90.5, 50/100/50→**91.79** ≈ paper 92.22; scripts/sweep_static_bins.py). Re-ran the decisive gate on the faithful fine-bin model (Full=91.58 ≈ paper), scripts/gate_faithful.py:
>
> | split | Full acc | macro-F1 | fishing recall | static gain (Full−WO-LWD) |
> |---|---|---|---|---|
> | temporal (80.9% leak) | 91.58 | 91.08 | 90.14 | **+18.72** |
> | vessel-disjoint (0% leak) | 89.49 | 90.21 | 88.02 | **+17.02** |
>
> **Δ static gain = 1.70 → NO COLLAPSE even on a paper-strength model with sharp static encoding.** ~91% of the static benefit transfers to unseen vessels. Sharper bins introduce a *small* identity component (Δ1.7, ~9% of the gain) absent in the coarse model — a richer, honest nuance, not a collapse. The "too-weak-model" caveat is now closed: the no-collapse finding holds at paper accuracy.
>
> **FINAL PAPER DIRECTION:** re-evaluation / reproducibility paper with three findings — (1) static-completeness **selection bias** (§1b); (2) **leakage audit**: 80.9% identity overlap measured, proven non-inflating via vessel-disjoint eval on a faithful model; (3) **reproducibility**: headline accuracy swings 86→92 with an undisclosed static-bin hyperparameter. SSL (M2) gives only marginal gains here (table4: ≤0.7pt) and collapses at 5% labels (loss imbalance), so it is not the headline.

### 1b. Second finding: static-completeness selection bias

Reproducing the paper's class balance on full nationwide MarineCadastre 2019 AIS surfaced a second, independent methodological problem. The paper's Table 1 is **cargo-dominant** (cargo 53.8%, fishing 2.0%). The raw nationwide data is **roughly class-balanced** (cargo ≈ fishing ≈ passenger ≈ 28% of messages each). The paper's distribution is *only* recoverable by **requiring complete static info (LEN+WID+DRA) per vessel** — which we confirmed reproduces it almost exactly (cargo 52.3%, tanker 25.2%, passenger 18.9%, fishing 3.6%; see [DATA_ADAPTATION_PLAN.md](DATA_ADAPTATION_PLAN.md)).

Static-reporting rates by class (train, message level):

| class | reports Draft | reports all of LEN/WID/DRA |
|---|---|---|
| cargo | 90.0% | 90.0% |
| tanker | 82.9% | 82.9% |
| passenger | 26.4% | 26.4% |
| **fishing** | **8.8%** | **8.8%** |

Draft is the discriminator: fishing and small Class-B passenger craft rarely transmit it. So the paper's benchmark is **silently pre-filtered to vessels that report static info** — overwhelmingly cargo/tanker. This makes the headline "static info gives +29.1%" partly **circular**: it is measured on a population *selected for having* static info, and it discards the very vessel classes (fishing, small passenger) where behavior — not dimensions — must carry the classification. The bias compounds the identity leakage in §1: not only are test vessels seen in training, the cohort itself excludes the hard, static-poor cases.

**Implication for the paper.** Two protocol problems, not one: (a) vessel-identity leakage, (b) static-completeness selection bias. The honest evaluation controls both — vessel-disjoint split *and* report results on the full (static-incomplete-inclusive) population with the missing-static machinery, not just the static-complete cohort.

Class balance of the Gulf extraction (differs from paper, worth noting — Gulf is fishing-heavy):

| split | cargo | fishing | passenger | tanker |
|---|---|---|---|---|
| train | 9137 | 10642 | 9212 | 12474 |
| val | 2426 | 5233 | 2914 | 3542 |
| test | 2217 | 5465 | 2837 | 3321 |

---

## Phase 0 — Make results trustworthy (do first, ~1 day)

Nothing downstream is credible until this is done. Each item is small.

### 0.1 Checkpointing + resume
**File:** [src/sslvtc/train.py](src/sslvtc/train.py), `train()`.
- After every epoch, save `{epoch, model_state, opt_state, best_val, history, rng_state}` to `paths.results/checkpoints/{tag}_last.pt`.
- On new best val, also write `{tag}_best.pt`.
- Add `resume: bool` (CLI `--resume`) that loads `_last.pt` and continues from `epoch+1`.
- Persist `history` to `paths.results/{tag}_history.json` each epoch so a crash never loses the curve.
**Acceptance:** kill mid-run, `--resume`, training continues from the right epoch with identical trajectory.

### 0.2 LR schedule + EMA + early stopping
**File:** [src/sslvtc/train.py](src/sslvtc/train.py).
- Cosine annealing (`torch.optim.lr_scheduler.CosineAnnealingLR`, `T_max = epochs`), step per epoch.
- Weight EMA (decay 0.999): keep a shadow copy, update post-`opt.step()`, **evaluate with EMA weights**, save EMA weights as the model. Add `train.ema_decay` to config (None disables).
- Early stopping: stop if val macro-F1 hasn't improved in `train.patience` epochs (default 15). Add `train.patience`.
**Config:** extend `TrainConfig` in [src/sslvtc/config.py](src/sslvtc/config.py): `ema_decay: float | None = 0.999`, `patience: int = 15`, `lr_schedule: str = "cosine"`.
**Acceptance:** val curve stops oscillating ±1%; best epoch is stable across 2 seeds.

### 0.3 Metrics suite
**New file:** `src/sslvtc/metrics.py`.
- `classification_metrics(y_true, y_pred, n_classes) -> dict` returning: `accuracy`, `balanced_accuracy`, `macro_f1`, `per_class_recall` (list), `per_class_precision`, `confusion_matrix`.
- Rewire `evaluate()` ([train.py:24](src/sslvtc/train.py#L24)) to return the full dict, not just `(acc, cm)`. Update callers in `experiments.py` and `cli.py`.
- **Model selection switches to macro-F1**, not accuracy (imbalance-aware).
**Acceptance:** `eval` prints all metrics + confusion matrix; per-class fishing recall is visible.

### 0.4 Multi-seed harness
**File:** new `src/sslvtc/runner.py` (or extend `experiments.py`).
- `run_repeated(cfg, n_seeds, **train_kwargs) -> DataFrame` with seeds e.g. `[42, 43, 44]`, returns per-seed rows + a mean±std summary row.
- A small **GPU dispatcher**: round-robin assign each `(seed, fraction, split)` job to `CUDA_VISIBLE_DEVICES ∈ {0,1,2}`, run 3 concurrently (subprocess per job, since one model = one GPU). This is the right use of the 3 GPUs.
**Acceptance:** one command produces a CSV with mean±std over 3 seeds for a given setting.

### 0.5 Determinism
- Seed `torch`, `numpy`, `random`; set `torch.use_deterministic_algorithms(True)` where feasible, `cudnn.benchmark=False`. Log the effective config (dataclass → JSON) next to every result for provenance.

---

## Phase 1 — The leakage study (spine of the paper)

### 1.1 Vessel-disjoint split
**File:** [src/sslvtc/extract.py](src/sslvtc/extract.py) (or a post-hoc resplit util `src/sslvtc/splits.py` operating on `index.parquet` — preferred, no re-extraction).
- Add a `split_mode` concept: `"temporal"` (current) and `"vessel_disjoint"`.
- Vessel-disjoint: partition **unique MMSIs** into train/val/test (e.g., 70/15/15) **stratified by vessel class**, so no MMSI crosses splits. A vessel's class is its trajectories' (single) label.
- Write a second index `index_vessel.parquet` (or add a `split_vd` column) so both protocols read from the same tensors. `TrajectoryDataset` ([dataset.py:49](src/sslvtc/dataset.py#L49)) gains a `split_column` arg (default `"split"`).
- **Report the overlap stats** (the 77.3% number) as a table — it is a result.
**Acceptance:** `index_vessel.parquet` has 0% MMSI overlap across splits; class proportions within ±3% across splits.

### 1.2 Re-run static ablation under both splits
- Generalize `table3_static_ablation` ([experiments.py:59](src/sslvtc/experiments.py#L59)) to take `split_mode` and run under both.
- Variants: WO-LWD, WO-LEN, WO-WID, WO-DRA, Full (already wired via `EncodingConfig.use_len/use_wid/use_dra`).
- **Headline figure:** grouped bar — accuracy AND macro-F1 for each variant, temporal vs vessel-disjoint. Expected: static gain large under temporal, small under vessel-disjoint.

### 1.3 Re-run Table 4 (SSL vs baseline) under both splits, all metrics, 3 seeds
- Fractions 5/20/40/60% (β grid already in `experiments.py` `BETA_GRID`).
- Expected story: under vessel-disjoint, absolute numbers drop; **the SSL gain may grow** (kinematic generalization is where SSL helps), strengthening the "SSL earns its keep when you can't memorize identity" argument.

### 1.4 Quantify inflation
- One summary table: Δaccuracy and Δmacro-F1 (temporal − vessel-disjoint) per method/fraction = "how much the standard protocol overstates performance."

**Decision gate:** if 1.2 shows the static-gain collapse, the paper is real and the method (Phase 2) follows. If it does *not* collapse (vessels genuinely behave class-typically regardless of identity), pivot the story to "static info is legitimately predictive and transferable" — still a result, just a different framing. Either way Phase 1 is decisive and cheap. **Run Phase 1 before committing to Phase 2.**

---

## Phase 2 — Method that wins under the honest protocol

Two independent upgrades; ablate each.

### 2.1 Time-aware encoder (replace seven-hot CNN)
**Motivation:** seven-hot bin counts are unspecified by the paper and arbitrary in our config ([encoding.bins](configs/gulf2019_storage.yaml)); the "trajectory-as-image" CNN ([models.py ConvBlock](src/sslvtc/models.py#L16)) ignores temporal order and irregular Δt sampling.
**Plan:**
- New backbone in `src/sslvtc/models.py`: a small **Transformer encoder** (or temporal CNN) over the per-message feature sequence, with **explicit Δt / time encoding** (AIS is irregularly sampled; current pipeline discards timestamps at Step 4 — keep a per-message Δt channel through extraction).
- Keep the seven-hot path as a baseline; add a `model.backbone: {sevenhot_cnn, temporal_transformer}` switch.
- This also removes the bin-count hyperparameter sensitivity (report robustness as a result).
**Risk:** moderate novelty alone ("transformer on AIS" exists — TrAISformer, GeoTrackNet). Sell it as *part of the leakage-robust recipe*, not the headline.

### 2.2 Modern SSL: self-supervised pretrain + consistency
**Motivation:** M2 (Kingma 2014) is dated; FixMatch/contrastive/masked-AE dominate label-scarce regimes (our 5% setting).
**Plan:**
- **Pretrain** on the full unlabeled pool (all 41k train trajectories, labels hidden): masked-AIS reconstruction (mask message spans, reconstruct) and/or contrastive (SimCLR-style) with **AIS-specific augmentations**: segment crop, time-warp, Gaussian jitter on kinematics, **static-feature dropout**.
- **Fine-tune** the classifier on the labeled fraction; optionally add FixMatch-style consistency (weak/strong aug agreement) on unlabeled.
- Compare against the M2 baseline (already implemented in [loss.py](src/sslvtc/loss.py)) at every fraction/split.

### 2.3 Static-feature debiasing (the novel bit)
**Motivation:** we want to *use* length/width/draft (they are legitimately informative) **without memorizing identity**. This directly attacks the leakage we expose.
**Options (ablate):**
- **Static-feature dropout** during training (randomly zero/mean-fill static channels) — the existing `missing_static_fill` + `static_available_fraction` machinery in [dataset.py](src/sslvtc/dataset.py#L55) already supports this; repurpose it as a regularizer.
- **Identity-invariance penalty:** adversarial head predicting MMSI (or a vessel-bucket) from the representation, trained to be *uninformative* (gradient reversal). Representation keeps class info, drops identity.
- **Class-balanced loss** (focal / class-balanced re-weighting) so minority fishing recall improves — pairs naturally with the macro-F1 metric switch.
**Acceptance:** under vessel-disjoint, our method recovers a meaningful fraction of the static-info gain that the baseline loses — *that* is the contribution.

---

## Phase 3 — Evidence & paper artifacts

**Experimental matrix** (every cell = 3 seeds, mean±std):

| Axis | Values |
|---|---|
| split | temporal, vessel-disjoint |
| labeled fraction | 5%, 20%, 40%, 60% |
| method | supervised-only, M2 (SSL-VTC), ours (2.1+2.2+2.3) |
| metrics | accuracy, balanced acc, macro-F1, per-class recall |

**Figures/tables for the paper:**
1. Leakage table (77.3% overlap) + protocol diagram.
2. Static-ablation: temporal vs vessel-disjoint (the money figure).
3. Inflation table (Δ temporal − vessel-disjoint).
4. Main results: ours vs M2 vs supervised, both splits, emphasis on 5%.
5. Per-class confusion matrices (fishing recall before/after debiasing).
6. Ablation of 2.1/2.2/2.3 contributions.
7. Bin-count robustness (seven-hot vs learned encoder).

**Reproducibility:** dump effective config JSON + git SHA + seed with every result CSV. Keep `scripts/run_full.sh` updated to the new pipeline; add `scripts/run_paper.sh` that regenerates all tables.

---

## Related work to position against (cite)

- Duan, Ma, Miao, Zhang 2022 (SSL-VTC, the baseline we extend/critique).
- Kingma et al. 2014 (M2 semi-supervised VAE — the dated SSL we replace).
- Nguyen et al. 2018/2021 (TrAISformer; seven-hot/four-hot AIS encoding origin).
- GeoTrackNet (Nguyen) — AIS anomaly/representation.
- FixMatch (Sohn 2020), MixMatch, SimCLR, masked autoencoders — modern SSL we adopt.
- Leakage/evaluation-validity literature (e.g., dataset-bias, group-disjoint splits) to ground the methodological critique.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Static gain does **not** collapse under vessel-disjoint | Pivot to "static info is transferable" framing; still a result (Phase 1 decides cheaply). |
| Vessel-disjoint numbers too low to be interesting | They *are* the honest numbers — that's the point; emphasize the SSL gain and debiasing recovery. |
| "Transformer on AIS is not novel" | Don't lead with the backbone; lead with leakage + debiasing. |
| Single-region (Gulf) limits generality | Add a second region (the NY-Harbor config exists, [configs/nyharbor.yaml](configs/nyharbor.yaml)) as an external-validity check, even if smaller. |
| Compute | 3 GPUs, parallel jobs via the dispatcher (0.4); model is tiny, full matrix is feasible in days not weeks. |

---

## Immediate next actions (in order)

1. **Phase 0.1** checkpointing + resume (stop losing runs).
2. **Phase 0.3** metrics suite (macro-F1, per-class recall).
3. **Phase 1.1** vessel-disjoint split + overlap table.
4. **Phase 1.2** static ablation both splits → decision gate.
5. Branch into Phase 2 only after the gate.

First commit should land 0.1 + 0.3 + 1.1 together; that unblocks the decisive Phase-1.2 experiment.
