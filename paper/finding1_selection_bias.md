# Finding 1 — The Hidden "Casting Call": A Selection Bias in the Benchmark

**The short version:** The paper we are studying tries to guess what *type* a ship is
(fishing boat? cargo ship? tanker? passenger ferry?) from its movement data. It claims that
adding the ship's physical size (length, width, how deep it sits in the water) makes the guess
much better — and that claim genuinely holds up. The problem is *who the method gets tested
on*. The benchmark quietly keeps only ships that broadcast their size, and the ships that don't
are overwhelmingly **fishing boats and small passenger boats**. So it is built from a
non-representative slice of the ocean.

We show this undocumented filter (1) deletes **~92% of fishing trajectories**; (2) makes the
**paper's own model collapse** on the excluded boats (macro-F1 **91 → 23**; just **64 on the
real ocean**) — though retraining on the full population recovers it to 88; and (3) means the
size feature that drives the benchmark **barely helps the fishing target** (+4.7 recall, vs +22
for tankers), which is already well-classified from motion alone. The method isn't broken — but
the benchmark's headline reflects an easy, hand-picked slice, and that is never disclosed.

---

## 0. Background you need (plain language)

- **AIS** = a radio system ships use to broadcast where they are, how fast they're going, and
  some facts about themselves. Think of it like every ship constantly tweeting its status.
- **MMSI** = a ship's unique ID number (like a license plate). Every AIS message carries it.
- **Trajectory** = one ship's path over a stretch of time, built by stringing its messages
  together.
- **The task** = look at a trajectory and predict the ship's *type*: fishing, passenger,
  cargo, or tanker.
- **Kinematic features** = things about *motion*: latitude, longitude, speed, heading.
- **Static features** = things about the *ship itself* that don't change: **L**ength,
  **W**idth, **DRA** = **dra**ft (how deep the hull sits in the water).
- **The paper's headline** (Duan et al. 2022, "SSL-VTC"): adding static features raised
  accuracy from 71.4% to 92.2% — a big jump they call "+29.1%."

The puzzle that started this finding: their dataset has *almost no fishing boats* (2%), even
though it supposedly covers all of U.S./Canada/Mexico coastal waters — which are full of
fishing boats. Why?

## 1. What we did

We rebuilt their dataset from the exact same public source (MarineCadastre AIS, 2019, all six
months, the whole coastline). We followed their cleaning recipe step by step. Then we asked two
simple questions:

1. Before any filtering, what's the real mix of ship types in the raw data?
2. How often does each ship type actually broadcast its size (especially draft)?

All numbers below are computed over the **full** training data — **210.8 million AIS messages
from 12,852 distinct vessels** — not a sample.

## 2. What we found

### 2.1 The raw ocean is balanced; the benchmark is cargo-heavy

![Fig 1.1](figures/f1_1_distribution.png)

In the real, unfiltered data, fishing boats are **29.9%** of the traffic — *more* common than
cargo (27.2%). In the benchmark, fishing is only **2%**. Something removed almost all of them.

### 2.2 The culprit is the Draft field

Length and width are broadcast by almost everyone. **Draft is different:**

![Fig 1.2](figures/f1_2_draft.png)

- Big commercial ships (cargo, tanker): report draft ~85–90% of the time.
- Fishing boats: report draft only **12.1%** of the time.
- Small passenger boats: only **23.6%**.

Why? Big ships have professional-grade (Class A) transponders and legal reporting duties;
small boats use cheaper (Class B) units and often leave draft blank. So **"must have a draft
value" acts as a filter that mostly deletes small boats.**

### 2.3 Requiring complete size info recreates the paper's exact mix

When we keep only ships that report **all three** size fields, the type mix snaps to almost
exactly the paper's numbers:

| ship type | our result (require all size info) | paper's Table 1 |
|---|---|---|
| cargo | 52.3% | 53.8% |
| tanker | 25.2% | 24.7% |
| passenger | 18.9% | 19.5% |
| fishing | **3.6%** | **2.0%** |

Cargo and tanker match within ~1 point. This is strong evidence the paper applied this filter
(probably without realizing its side effect, since their whole method *needs* the size fields).

### 2.4 The filter throws away half the data — mostly fishing and small passenger boats

![Fig 1.3](figures/f1_3_kept.png)

Turning the filter on cut our dataset from **260,543 down to 133,492** trajectories (exactly
half). The losses are wildly uneven: **~92% of fishing trajectories** and **~66% of passenger
trajectories** are deleted (only 8.3% of fishing and 34.5% of passenger survive), while big
ships are mostly kept (cargo 91% kept, tanker 84%). This is measured directly from the
full inclusive extraction, not estimated.

## 3. Does the bias actually matter? (The consequence experiment)

Showing the filter *exists* isn't enough — we need to show it *changes the answer*. We trained
the same paper-strength model (fine size encoding, ~92% on the benchmark) two ways, then scored
each on three populations. All numbers are macro-F1 (score averaged equally over the four ship
types, so weak rare-class performance shows up), mean over 3 random seeds.

- **Model A — the paper's model:** trained only on ships with complete size (the benchmark).
- **Model B — the realistic model:** trained on *all* ships, missing size filled with the class
  average.
- **Populations:** *complete* (ships the benchmark keeps), *dropped* (ships it deletes — mostly
  fishing/small), *all* (the real ocean = complete + dropped).

![Fig 1.4](figures/f1_4_consequence.png)

| model | complete (kept) | dropped | all (real ocean) |
|---|---|---|---|
| **A — paper's** (trained on complete) | 91.2 | **23.4** | 64.3 |
| **B — realistic** (trained on all) | 87.9 | 78.4 | 88.1 |

The headline jumps out: **the paper's model collapses from 91 to 23 on the vessels the benchmark
deleted.** On those dropped ships, A's per-class recall for passenger and tanker falls to
**~0** — because A was trained to lean on real size, and dropped ships have only filled-in
(identical) size, so it can no longer tell the size-dependent classes apart. The benchmark never
measures this, because those ships aren't in its test set.

Retraining on the full population (Model B) **recovers the dropped ships to 78** and even scores
*higher* on the real ocean overall (88 vs A's 64). So the failure isn't fundamental — it's
caused by training and testing on the filtered slice. (Methodology note: A is evaluated with its
own training normalization, B with its own — each as it would actually be deployed; the dropped
collapse is identical with or without matched normalization, so it isn't an artifact.)

### 3b. Where does the size benefit actually go? (per-class)

This is the operational crux: the celebrated size boost lifts the *big commercial ships*, not
the fishing target. We compare the realistic model with size vs kinematic-only (motion only),
per class, 3 seeds:

![Fig 1.5](figures/f1_5_perclass.png)

| class | recall, motion only | recall, + size | **size boost** |
|---|---|---|---|
| **fishing** (the target) | 86.8 | 91.5 | **+4.7** |
| passenger | 82.6 | 89.5 | +6.9 |
| cargo | 74.4 | 88.9 | +14.5 |
| **tanker** | 61.0 | 83.1 | **+22.1** |

Size adds **+22 for tankers and +14 for cargo** — the large vessels that actually broadcast
size — but only **+4.7 for fishing**, the smallest of any class. And fishing is *already* at
86.8% from motion alone: fishing boats move distinctively (slow, looping, area-bound), so they
barely need size — which they mostly can't broadcast anyway (only 12% report draft).

**Honest bound:** +4.7 is real, not zero, so "size is irrelevant for fishing" would be
overstating it. The defensible claim: *the size benefit accrues to large vessels; the fishing
target gains little and is already well-classified from motion.* For the operational use case
(spotting illegal fishing), the size feature that makes the benchmark look great does almost
nothing — and the model that relies on it is the one that collapses (§3a) on exactly those
boats.

## 4. Why this is a problem

Imagine claiming "wearing glasses predicts being a good reader" — but you only tested people
who *already wear glasses*. That's circular. Here:

1. **The "size helps" claim is tested on a crowd pre-selected for having size info.** The very
   ships where size is missing — fishing and small boats — are excluded from the experiment
   meant to prove size is useful.
2. **The reported number answers an easier question than deployment.** §3 showed the paper's own
   model scores 91 on the kept slice but **64 on the real ocean** (collapsing to 23 on the
   dropped boats). The benchmark's headline reflects the easy, kept slice — not the full ocean a
   real coast guard must classify (where fishing and small craft matter most, e.g. illegal
   fishing).
3. **The ship mix is an accident of who reports draft, not of the real ocean.** Any per-class
   conclusion (e.g. "fishing is rare and hard") is partly an artifact of the filter, not a fact
   about maritime traffic.
4. **It stacks with the next problem (Finding 2):** not only is the crowd pre-selected, the same
   ships also show up in both the study session and the final exam.

## 5. Being fair to the paper (threats to validity)

- We can't *prove* they did this on purpose; we show it's the one filter that reproduces their
  numbers, and the match is very close.
- Data versions changed over the years, so our totals differ a bit (133k vs their 115k, ~15%),
  but the *shape* matches.
- The A-vs-B comparison changes both the training set and the test set at once, so the ~3-point
  gap bundles "harder test population" with "noisier training." This is why we also report the
  **kept-vs-dropped split within the same model B** — that isolates the population effect (the
  dropped ships are ~9 macro-F1 points harder, no confound).

## 6. What should be done instead

This isn't hypothetical — it's exactly the protocol we ran as **World B in §3**, so the fix is
already demonstrated:

- **Keep the small boats** — don't drop a ship just because a field is blank (World B).
- **Impute missing size honestly** (mean/zero/learned; our pipeline's `missing_static_fill`),
  rather than deleting the vessel.
- **Report per-type scores** (per-class recall + macro-F1) so fishing performance stays visible
  instead of being filtered out of sight.

The one thing only the benchmark's authors can fix is **disclosure**: any "must have complete
data" rule has to be stated, because it changes who the method is judged on. Our recommendation
is to evaluate on the inclusive population by default and report the static-complete subset only
as a secondary, clearly-labeled slice.

## 7. Where the evidence lives (for reproduction)

- `configs/fullus2019.yaml` → `extraction.require_complete_static: true` (static-complete cohort)
- `configs/fullus2019_inclusive.yaml` → `require_complete_static: false` (realistic cohort)
- `src/sslvtc/extract.py` → `_passes_filter` (the size-completeness gate)
- `scripts/static_reporting_stats.py` → §2 full-dataset stats (`static_reporting_full.csv`)
- `scripts/tag_static_complete.py` → per-trajectory `static_complete` flag on the inclusive index
- `scripts/consequence_experiment.py` → §3 A-vs-B experiment (`consequence_experiment.csv`, 3 seeds)
- `scripts/grid_perclass_experiment.py` → §3b per-class size boost + B grid (`grid_perclass.csv`, 3 seeds)
- `scripts/grid_A_cnorm.py` + `configs/fullus2019_inclusive_cnorm.yaml` → §3a clean A grid with
  matched normalization (`grid_A_cnorm.csv`, 3 seeds)
- `DATA_ADAPTATION_PLAN.md` → how we discovered and reproduced the filter
- Figures: `paper/make_figures.py`
