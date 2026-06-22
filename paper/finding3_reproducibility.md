# Finding 3 — The Headline Score Depends on a Setting Nobody Wrote Down

**The short version:** The paper reports 92.2% accuracy. We followed their recipe exactly and
got stuck at 86% — a 6-point gap. We hunted down the cause: it's a single setting the paper
never specified (how finely the ship-size numbers are chopped into buckets before feeding the
network). Turning that one knob, and *nothing else*, moves the score from 86% all the way to
92%. So the famous number isn't reproducible from the paper as written. Two smaller
reproducibility problems came along for the ride: the "semi-supervised" trick barely helps, and
training falls apart when labels are very scarce.

---

## 0. Background you need

- **Reproducibility** = if someone follows your written method, do they get your numbers? If a
  result needs an undocumented setting, it isn't reproducible.
- **Seven-hot encoding** = a way to feed numbers to a neural net. Instead of giving the raw
  number (say, length = 137m), you chop the range into **bins** (buckets) and switch on the one
  the value falls into. More bins = finer detail. *Example:* with 10 length-bins, 137m and 142m
  might land in the same bucket (the net can't tell them apart); with 100 bins, they're
  distinct. The paper uses this scheme but **never says how many bins** each field gets.
- **Supervised vs semi-supervised:** "supervised" learns only from labeled examples.
  "Semi-supervised" (SSL) also uses *unlabeled* data to help. The paper's selling point is that
  its SSL method beats a plain supervised baseline, especially when labels are scarce.

## 1. What we did

1. **Rule out the boring explanations.** We ran a controlled test (`diag_supervised.py`) logging
   accuracy every epoch, under two learning-rate styles, to check it wasn't the optimizer, the
   schedule, or how we pick the "best" checkpoint.
2. **Sweep the suspect.** We kept the motion-feature bins fixed and only varied the **size**
   bins (width/length/draft), from coarse to fine, measuring accuracy each time. A key clue
   pointed us here: the *no-size* model already matched the paper (~72% vs their 71%), so the
   entire gap had to live in how size is encoded.

## 2. What we found

### 2.1 It's not the optimizer or the checkpoint choice

| learning-rate style | best accuracy reached (any epoch) |
|---|---|
| cosine (decaying) | 85.97 |
| constant (the paper's) | 86.01 |

Both top out at ~86 and **never touch 92**, no matter which epoch you pick. So the ceiling is
baked into the *encoding*, not the training.

### 2.2 One knob — size-bin count — moves 86 → 92

![Fig 3.1](figures/f3_1_bins.png)

As we give the size fields more bins, accuracy climbs steadily and lands at **91.8%**,
essentially the paper's 92.2% (red dashed line). Nothing else changed. **The headline number is
a function of an unstated setting.**

*Why it works:* finer bins let the network see ship size more precisely, so it squeezes more
(genuinely useful — see Finding 2) information out of length/width/draft.

### 2.3 Bonus problem: the semi-supervised trick barely helps

When we reproduce the SSL-vs-baseline comparison on faithful data, the semi-supervised method
is almost a tie with the plain baseline:

![Fig 3.2](figures/f3_2_ssl.png)

| labels used | baseline | SSL-VTC | gain |
|---|---|---|---|
| 20% | 81.67 | 82.33 | +0.7 |
| 40% | 84.83 | 85.03 | +0.2 |
| 60% | 85.53 | 85.51 | −0.0 |

The paper reports a clear SSL advantage (e.g. at 40%: 86.1 → 89.0). On our reproduction the
semi-supervised part buys almost nothing — so SSL shouldn't be the headline.

### 2.4 Bonus problem: collapse when labels are very scarce (5%)

At the 5%-label setting — the one that *most* shows off semi-supervised learning — the model
gives up and just predicts "cargo" for everything (cargo is ~52% of the data, and the model's
accuracy sticks at ~51%). We confirmed this isn't a bug in our stopping rule (we added a warm-up
guard and it still happened). The real cause: with so few labels, the math objective is
dominated by a "rebuild the input" term that drowns out the "classify correctly" term, so the
classifier never learns. This 5% case is exactly where the paper claims its biggest win — and
it's the hardest to reproduce.

## 3. Why this matters

1. **The flagship 92.2% can't be reproduced from the text** — 6 points ride on an undocumented
   knob. For this family of AIS methods, the bin resolution also changes the network's input
   size, so it's not a minor detail.
2. **Sensitivity should be reported as a result,** not hidden. A fair paper would show the
   accuracy-vs-bin-count curve (Fig 3.1) — or sidestep the knob entirely with a learned encoder
   (our codebase already has a transformer backbone scaffolded for this).
3. **The semi-supervised contribution is shaky** on faithful data and breaks in the very
   low-label regime that motivates it.

## 4. Being fair (threats to validity)

- We reach 91.8, about 0.4 short of 92.2 — close enough to blame the bins, though a little of
  the residual could be data-version differences.
- We changed width/length/draft bins *together*. We didn't isolate which one matters most
  (length probably dominates, matching the size-ablation). A per-field sweep is easy follow-up.

## 5. Where the evidence lives

- `scripts/diag_supervised.py` (rules out optimizer/selection) → `/tmp/diag_supervised.json`
- `scripts/sweep_static_bins.py` → `sweep_static_bins.csv` (the 86→92 curve)
- `table4_ssl_vs_baseline.csv` (marginal SSL gain + 5% collapse)
- `src/sslvtc/dataset.py` (RAM-packing that made the sweep fast enough to run) and `train.py`
  (`min_epochs_before_stop` warm-up guard used to rule out early-stopping as the 5% cause)
- Figures: `paper/make_figures.py`
