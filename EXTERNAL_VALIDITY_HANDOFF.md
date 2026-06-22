# Handoff — External Validity on European AIS (Danish DMA)

**For:** another agent (or a local Claude Code session) running on a machine that *can* reach
`web.ais.dk`. The cluster where the rest of this project runs **cannot** reach that host
(`web.ais.dk` → connection refused / http 000, while NOAA / dma.dk / zenodo all return 200), so
the download must happen from a different network and the files copied onto the data volume.

---

## 0. Why this task exists (context)

This repo is a **re-evaluation / reproducibility study** of SSL-VTC (Duan et al., 2022), a model
that classifies AIS vessel trajectories into 4 types (fishing / passenger / cargo / tanker).
Full write-up: [paper/PAPER_FULL.md](paper/PAPER_FULL.md). Plain-language: [paper/PAPER_IDEA_SIMPLE.md](paper/PAPER_IDEA_SIMPLE.md).

**Finding 1 (the one this task validates)** — a *static-completeness selection bias*: the
benchmark only keeps vessels that broadcast all static fields (length/width/**draft**). Draft is
**Missing Not At Random (MNAR)** — small boats (fishing, small passenger) rarely report it. On
US nationwide 2019 data (210.8 M messages):

| class | reports draft |
|---|---|
| cargo | 89.7% |
| tanker | 84.7% |
| passenger | 23.6% |
| **fishing** | **12.1%** |

The filter therefore deletes ~92% of fishing trajectories, and the paper's model collapses
(macro-F1 91→23) on the excluded vessels.

**The gap:** all our evidence is US MarineCadastre 2019. A reviewer will ask *"is this a
one-dataset fluke?"* gulf2019 doesn't answer it (it's a bbox crop of the *same* US data).
A **European dataset from a different authority** is the strong external-validity check.

**The goal of THIS task:** confirm the **MNAR draft-missingness mechanism replicates in European
waters**. If fishing/small boats under-report draft in Danish AIS too, the selection bias is a
general property of AIS — not US-specific. **Stats only, no model training needed.** Cheap and
decisive.

---

## 1. The dataset: Danish Maritime Authority (DMA) open AIS

- Source: `https://web.ais.dk/aisdata/` — open, raw, daily AIS CSVs for Danish/Baltic/North-Sea
  waters. Files named like `aisdk-YYYY-MM-DD.zip` (each unzips to one large CSV).
- It carries the fields we need, including **draught** (draft) and **ship type**.
- A different region, fleet mix, and *regulator* than US MarineCadastre — genuine independence.

**You only need a few days** (e.g. 3–5 daily files), not the whole archive — we are measuring a
*rate* (draft-reporting fraction by ship type), not training.

---

## 2. Steps

### 2.1 Download (on the machine that can reach web.ais.dk)
Grab a handful of daily files, e.g.:
```bash
mkdir -p danishais_raw
cd danishais_raw
for d in 2019-05-01 2019-05-02 2019-05-03; do
  curl -O "https://web.ais.dk/aisdata/aisdk-${d}.zip"
done
```
(Any recent dates are fine — match 2019 if you want the same year as the US data.)

### 2.2 Inspect the schema (column names differ from MarineCadastre)
DMA CSV columns are typically: `Timestamp, Type of mobile, MMSI, Latitude, Longitude,
Navigational status, ROT, SOG, COG, Heading, IMO, Callsign, Name, Ship type, Cargo type,
Width, Length, Type of position fixing device, Draught, Destination, ETA, ...`

Key mappings to our canonical names (see `src/sslvtc/ingest.py::COLUMN_ALIASES`):
- `Latitude → LAT`, `Longitude → LON`, `SOG → SOG`, `COG → COG`
- `Length → Length`, `Width → Width`, **`Draught → Draft`** (alias already present)
- `Ship type` → vessel class. **Note:** DMA `Ship type` is usually a *text category*
  (`Fishing`, `Cargo`, `Tanker`, `Passenger`, `Pleasure`, `Undefined`, ...), NOT the ITU numeric
  code MarineCadastre uses. You must map text→class (see 2.3).
- `Timestamp` format is `DD/MM/YYYY HH:MM:SS` (different from MarineCadastre) — parse accordingly.

### 2.3 The minimal analysis (this is the whole deliverable)
You do **not** need the full ingest/extract/train pipeline. The MNAR check is a few lines of
pandas over the raw CSVs. Compute, per ship type, the fraction of **vessels** (group by MMSI)
that **ever report a non-null Draught**:

```python
import pandas as pd, glob, zipfile
TEXT2CLS = {  # DMA 'Ship type' text -> our 4 classes (case-insensitive contains)
    "fishing": "fishing", "passenger": "passenger", "cargo": "cargo", "tanker": "tanker",
}
rows = []
for zp in glob.glob("danishais_raw/*.zip"):
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        for chunk in pd.read_csv(z.open(name), chunksize=500_000,
                                 usecols=["MMSI","Ship type","Draught"]):
            chunk["cls"] = chunk["Ship type"].str.lower().map(
                lambda s: next((v for k,v in TEXT2CLS.items() if isinstance(s,str) and k in s), None))
            chunk = chunk.dropna(subset=["cls"])
            rows.append(chunk[["MMSI","cls","Draught"]])
df = pd.concat(rows, ignore_index=True)
ves = df.groupby(["MMSI","cls"]).agg(has_draft=("Draught", lambda s: s.notna().any())).reset_index()
print((ves.groupby("cls")["has_draft"].mean()*100).round(1))
```

**Success criterion:** fishing (and ideally small passenger) report draft at a **much lower rate
than cargo/tanker** — same pattern as the US table above. If so, the MNAR selection bias is
confirmed in independent European data → external validity established.

Save the output table to `paper/external_validity_danish.csv` and add a short paragraph + the
table to [paper/PAPER_FULL.md](paper/PAPER_FULL.md) §8 (replace the "single region/year"
limitation with the confirmed cross-region result), plus a bar figure mirroring Fig 1.2 via
`paper/make_figures.py`.

### 2.4 (Optional, stronger) Full replication
If you want the *consequence* too (not just the mechanism): drop the DMA CSVs into a new config
mirroring `configs/fullus2019.yaml` (set `bbox: null`, point `paths.raw` at the DMA files), add a
`Draught→Draft` already-handled alias, extend `vessel_type_to_label` in `src/sslvtc/ingest.py`
to accept DMA text categories, then run `ingest → extract` and repeat the §4 consequence
experiment. This is heavier and not required for the external-validity claim.

---

## 3. What to hand back
1. `paper/external_validity_danish.csv` — draft-reporting rate by class (Danish AIS).
2. One paragraph + table + figure added to `paper/PAPER_FULL.md` (§8 / new §4.7).
3. Note the dates/files used (for reproducibility).

---

## 4. Pointers (existing code to reuse, do not rewrite)
- `src/sslvtc/ingest.py` — `COLUMN_ALIASES` (has `draught→Draft`, `beam→Width`),
  `vessel_type_to_label` (extend for DMA text categories).
- `scripts/static_reporting_stats.py` — the US version of this exact MNAR computation; mirror it.
- `paper/make_figures.py` — `f1_draft()` is the figure to mirror for the Danish result.
- `paper/finding1_selection_bias.md` §2.2 — the framing this validates.
