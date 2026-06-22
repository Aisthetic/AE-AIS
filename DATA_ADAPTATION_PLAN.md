# Data Adaptation Plan — Match the SSL-VTC Paper Benchmark

**Goal:** replace the Gulf-of-Mexico subset with the **full US/Canada/Mexico coastal
coverage** used by Duan et al. 2022, so that (a) "beat 88.96%" is measured on the
*same benchmark* the paper reports, and (b) the vessel-identity-leakage critique
lands on the exact dataset the paper used — not a regional proxy.

This is the prerequisite for re-running [RESEARCH_PLAN.md](RESEARCH_PLAN.md) on
paper-comparable data.

---

## 0. Key discovery (changes everything)

**The raw daily zips are already nationwide.** Download fetches whole-country daily
AIS files; the *bbox is applied at ingest* ([ingest.py:101-110](src/sslvtc/ingest.py#L101-L110)),
not at download. "Gulf" is purely an ingest-time filter.

Verified facts (this machine, this volume):

| Fact | Value |
|---|---|
| Raw files present | **181** daily zips (all 6 months) |
| Raw size | 48 GB (zips) |
| One day decompressed | ~785 MB CSV |
| Full 6-month decompressed | **~142 GB CSV** |
| RAM | 250 GB total / ~203 GB free |
| Cores | 32 |
| Disk free on 10T volume | **1.2 TB** (89% used — tight) |

**Implication:** no re-download. This is a **re-ingest (no bbox) → re-extract** job.
The only real engineering is **memory-hardening** the ingest and extract stages,
which currently hold an entire split in RAM and would OOM at nationwide scale.

Paper class distribution to expect after the change (Table 1, train split) — heavy
imbalance, fishing scarce:

| | fishing | passenger | cargo | tanker |
|---|---|---|---|---|
| paper train | 1,436 | 13,678 | 37,842 | 17,335 |

This is the **opposite** of our balanced Gulf extraction (fishing-heavy). It makes
the macro-F1 / per-class-recall switch from Phase 0.3 **essential**, not optional.

---

## 1. New config — `configs/fullus2019.yaml`

Copy [configs/gulf2019_storage.yaml](configs/gulf2019_storage.yaml) and change two things:

```yaml
# Remove the geographic filter → nationwide coverage (paper-equivalent).
bbox:
  lat_min: null
  lat_max: null
  lon_min: null
  lon_max: null

paths:
  raw:       /mnt/storage_1_10T/zezzahed/AIS_Data/gulf2019/raw   # REUSE — do not re-download
  interim:   /mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/interim
  processed: /mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/processed
  results:   /mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/results
```

Everything else (extraction thresholds, encoding bins, model, train) stays
**identical to the paper** — those are data-agnostic and already paper-faithful.

**Reuse the existing `raw/` path** — the zips are nationwide already. Do not copy or
re-download 48 GB.

---

## 2. Harden `ingest_all` for scale (memory) — REQUIRED

**Problem:** [ingest.py:168-184](src/sslvtc/ingest.py#L168-L184) collects every
per-file DataFrame for a split into a Python list in the **parent process**, then
`pd.concat`s the whole split and writes one parquet. Nationwide train (4 months,
~120 days) filtered to 4 classes is tens of GB held in RAM at once — borderline at
203 GB free, and it duplicates across the concat. Not safe.

**Fix:** write **partitioned parquet** — each worker writes its own shard, parent
never holds a whole split.

- In `_ingest_file_task`, after cleaning, write directly to
  `interim/{split}/part_{YYYY_MM_DD}.parquet` and return only the path + row count
  (not the DataFrame).
- Drop the in-memory `buckets` accumulation entirely.
- `ingest_all` returns the interim dir; splits become **directories of parquet
  shards**, not single files.

**Acceptance:** peak parent RAM < ~30 GB during ingest; `interim/train/` etc. each
contain one parquet per day.

---

## 3. Harden `extract_all` / `extract_split` for scale (memory) — REQUIRED

**Problem:** [extract.py:128](src/sslvtc/extract.py#L128) does
`pd.read_parquet(path)` on the *whole* split, then
[extract.py:101](src/sslvtc/extract.py#L101) builds 32 partial copies via
`df[df["MMSI"].isin(chunk)].copy()`. At nationwide scale that is the full split
plus ~one extra full copy spread across workers — OOM.

**Fix:** stream by partition + shard by MMSI without whole-split copies.

- Read the partitioned interim as a `pyarrow.dataset` (Step 2 output), or read
  shards day-by-day.
- **Group all messages by MMSI first** (a vessel's messages may span multiple daily
  shards — this matters: Step 1 trajectory division is per-MMSI-per-day, so messages
  for one MMSI must be co-located before `_divide`). Two options:
  1. **Repartition by MMSI hash** once (write `interim/{split}_bymmsi/bucket_{k}.parquet`,
     k in 0..N), then each worker reads one bucket end-to-end. Clean, scales, costs
     one extra pass + transient disk.
  2. If RAM allows (it may — 203 GB), read the whole split with only the needed
     columns (`MMSI, BaseDateTime, LAT, LON, SOG, COG, Length, Width, Draft,
     label, label_idx`) as pyarrow, then `np.array_split` MMSI groups by **reference**
     (pass index ranges, not `.copy()`).

  **Recommended: option 1** (MMSI-hash repartition) — robust regardless of split size,
  and the same buckets can be reused if extraction is re-run with different thresholds.

**Acceptance:** peak RAM < ~50 GB; full extract completes; `processed/index.parquet`
written with `traj_id, split, label, label_idx, mmsi, path` (+ the `DT` 8th tensor
column already added in Phase 2.1).

---

## 4. Run the adapted pipeline

```bash
# No download — raw is already nationwide.
python -m src.sslvtc.cli -c configs/fullus2019.yaml ingest
python -m src.sslvtc.cli -c configs/fullus2019.yaml extract
```

**Disk budget check (do before extract):**
- interim partitioned parquet (snappy): est. ~40–60 GB for 142 GB of source CSV.
- tensors: ~95k trajectories × 160 × 8 × 4 B ≈ **0.5 GB** (negligible).
- Fits in 1.2 TB free with margin. If disk gets tight, delete `interim/` after a
  successful extract (tensors + index are all training needs).

---

## 5. Sanity-check against the paper (decision point)

Before re-running the research plan, confirm the data now resembles the paper:

1. **Class distribution table** — should flip to cargo-heavy, fishing-scarce,
   matching Table 1 in shape (counts won't be identical; MarineCadastre 2019 +
   our thresholds differ slightly, but the *ordering* cargo > tanker > passenger >
   fishing should hold).
2. **Reproduce paper Table 2 / 3 / 4** on the new data with the existing
   `experiment` commands (temporal split, `split` column). Targets:
   - CNN + seven-hot ≈ 92.22%
   - SSL-VTC 40% labeled ≈ 88.96%
   - WO-LWD ≈ 71.43%
   If we land within a few points, the benchmark is faithfully reproduced and
   "beating 88.96%" becomes a meaningful claim.

---

## 6. Re-run RESEARCH_PLAN on the new data

Once `fullus2019/processed/index.parquet` exists and Step 5 passes, everything in
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) runs unchanged, just pointed at the new config:

```bash
python -m src.sslvtc.cli -c configs/fullus2019.yaml vessel-split --report
python -m src.sslvtc.cli -c configs/fullus2019.yaml experiment table3-leakage
python -m src.sslvtc.cli -c configs/fullus2019.yaml experiment leakage-inflation
# ... then Phase 2 methods under both splits
```

The leakage measurement (Phase 1.1) is **re-run on the paper's actual benchmark** —
the 77.3% identity-overlap number becomes a statement about *the paper's dataset*,
not a regional proxy. That is the strongest possible version of the critique.

---

## What stays unchanged

- All model / loss / extraction-threshold / encoding code — data-agnostic.
- All Phase 0 / 1 / 2 work already implemented (checkpointing, metrics, vessel split,
  transformer, debiasing) — operates on whatever `index.parquet` it's given.
- The Gulf results stay on disk as a **second region** — useful as the external-validity
  check the research plan's risk table already calls for.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Ingest/extract OOM at nationwide scale | Steps 2 & 3 (partitioned writes, MMSI-hash sharding) — **do not skip** |
| Disk fills (1.2 TB free, 89% used) | parquet snappy; delete `interim/` after extract; tensors are tiny |
| More vessels missing LEN/WID/DRA nationwide | existing `missing_static_fill` + withhold machinery handles it; report missing-rate as a data stat |
| Class imbalance worse than Gulf | macro-F1 / balanced-acc / per-class recall (Phase 0.3) already the default selection metric |
| Reproduced numbers drift from paper | expected within a few pts (data vintage + threshold differences); document the gap, don't chase it |
| MMSI spanning daily shards breaks trajectory division | Step 3 co-locates by MMSI **before** `_divide` — explicitly required |
