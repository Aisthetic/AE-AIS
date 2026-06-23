# Paper Handoff — IARIA DATA ANALYTICS 2026

Handoff for an agent continuing the paper on the **cluster** (where the AIS data lives).
This continues work done off-cluster (laptop) where the trajectory data is **not** present.

## Target venue
- **The Fifteenth International Conference on Data Analytics (DATA ANALYTICS 2026)**, IARIA.
- CfP: https://www.iaria.org/conferences2026/CfPDATAANALYTICS26.html
- **Submission deadline: 2026-06-26.** Type: **Regular paper, 6 pages** (4 extra pages buyable for a fee).
- Relevant tracks the paper is framed for: *Trust in data analytics*; *Active learning (semi-supervised, faulty/partial labels)*; *Data preparation, feature selection*; *Storing, dropping and filtering data*; *Testing/debugging/monitoring of ML*.
- Tone: neutral data-quality / trust re-evaluation, **not** an attack. Credit SSL-VTC's core idea as sound.

## Where things are (all under repo root `AE-AIS/`)
- `overleaf/main.tex` — the paper (this is the deliverable). Uses `\documentclass[conference,flushend]{iaria}`.
- `overleaf/iaria.cls` — vendored IARIA class (IEEEtran-based, biblatex/biber, IEEE style). **Do not edit.**
- `overleaf/refs.bib` — 9 references, all cited.
- `overleaf/figures/` — figures referenced by `main.tex` (PNG).
- `overleaf/main.pdf` — last built PDF (committed for convenience).
- `paper/PAPER_FULL.md` — the full 9-section source draft with every result/number (mean over 3 seeds). **Source of truth for content**; `main.tex` is a 6-page condensation of it.
- `paper/finding{1,2,3}_*.md`, `paper/PAPER_IDEA_SIMPLE.md` — per-finding deep writeups.
- `paper/make_figures.py` — regenerates all result figures into `paper/figures/`.
- `paper/external_validity_danish.csv` — Danish MNAR stats.

## Current paper state
- Compiles clean: `exit 0`, 0 unresolved refs, **4 pages** (under the 6-page budget — room to expand).
- Sections: Intro, Background+Data, F1 Selection Bias (MNAR), F2 Identity Leakage (benign), F3 Reproducibility, External Validity (Danish), Corrected Protocol, Limitations+Conclusion.
- 5 figures wired: `f_traj_maps` (datasets, **PLACEHOLDER — see Task 1**), `f1_2_draft`, `f1_4_consequence`, `f2_2_nocollapse`, `f3_1_bins`; 2 tables (per-class boost, US/Danish draft).
- Author block is a **placeholder** (Jane Smith / University of …) — fill in before submission.
- No em dashes anywhere (`---`) by request; en-dash ranges (`--`) kept.

## Build (on cluster)
```bash
# needs TeX Live with: ieeetran biblatex biber ncctools floatrow sttools pbalance biblatex-ieee orcidlink enumitem preprint
cd overleaf
latexmk -pdf main.tex          # biber pass handled automatically
# page count: pdfinfo main.pdf | grep Pages
```
If a `.sty` is missing: `tlmgr install <pkg>` (or `sudo tlmgr ...`). LaTeX build artifacts are gitignored; commit only `main.tex`, `refs.bib`, `figures/`, and optionally `main.pdf`.

## Task 1 (BLOCKER — needs cluster data) — real trajectory figure
`overleaf/figures/f_traj_maps.png` is a yellow **placeholder**. The real figure plots sampled
vessel trajectories per class on each region. Data is cluster-only (`/mnt/storage_1_10T/zezzahed/AIS_Data/...`).

Run:
```bash
python scripts/plot_trajectories.py \
  --region "US 2019"     /mnt/storage_1_10T/zezzahed/AIS_Data/fullus2019/processed \
  --region "Danish 2019" /mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/processed \
  --per-class 40 --out paper/figures/f_traj_maps.png
cp paper/figures/f_traj_maps.png overleaf/figures/
cd overleaf && latexmk -pdf main.tex
```
- Script un-normalizes LAT/LON from `processed/normalization_stats.json` (`deg = norm*(max-min)+min`); reads `index.parquet` + `tensors/*.npy` (`[T,8]` = LAT,LON,SOG,COG,WID,LEN,DRA,DT).
- Coastlines drawn if `cartopy` is installed; else clean lon/lat panel with aspect correction. Installing cartopy gives nicer maps.
- Tune `--per-class` for density; add a bbox/extent clip if a region is too zoomed-out (not yet implemented — add if needed).
- Verify the rendered figure visually before committing (the off-cluster agent could not).

## Open tasks / suggestions
1. **Task 1 above** (real trajectory map) — highest priority, only doable on cluster.
2. Fill author block in `overleaf/main.tex` (names, affiliation, email, ORCID).
3. Paper is 4pp; 6pp allowed. Optional expansion if reviewers want more depth:
   - Restore the MNAR draft-rate table (currently prose) from `PAPER_FULL.md` §4.2.
   - Add `f1_1_distribution` (raw vs benchmark class mix) and/or expand the VAE-collapse result (§4.6) to its own paragraph + table.
   - Add the `f1_3_kept` survival figure.
   All result PNGs already exist in `paper/figures/` (regenerate via `python paper/make_figures.py`).
4. Final pass: check IARIA formatting (Letter 8.5x11, 2-column), figure captions, that every `\cite` resolves, page count ≤ 6 (or ≤ 10 if extra pages purchased).
5. Proofread against `PAPER_FULL.md` to ensure no number drifted during condensation.

## Conventions
- Single branch: **main only** (no feature branches). Commit + push to main.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Keep `iaria.cls` untouched. Keep numbers consistent with `PAPER_FULL.md` (3-seed means).
- No em dashes in the paper text.
