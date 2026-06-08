# SSL-VTC Implementation Spec

Re-implementation guide for the paper:

> Duan, Ma, Miao, Zhang (2022). *A semi-supervised deep learning approach for vessel trajectory classification based on AIS data.* Ocean & Coastal Management 218, 106015.

Goal: classify vessel trajectories into **4 ship types** (fishing, passenger, cargo, tanker) from AIS data, using a **VAE-based semi-supervised model (SSL-VTC)** that trains on labeled + unlabeled trajectories together.

Stack used in paper: **PyTorch**, single NVIDIA TITAN V GPU.

---

## 1. Dataset

**Source:** U.S. Coast Guard / MarineCadastre AIS — https://marinecadastre.gov/ais/
- Coverage: coastal waters of Canada, US, Mexico.
- Period: **January–June 2019**.
- Raw size: train 168.3 GB, val 52.6 GB, test 56.0 GB (full raw AIS).

**Split (temporal, NOT random):**
| Split | Months 2019 |
|-------|-------------|
| Train | Jan–Apr |
| Validation | May |
| Test | June |

Rationale: train on past, predict future → predictive classification task. Val used for model selection; test for final report.

**Classes (4):** fishing, passenger, cargo, tanker. Derived from AIS ship-type code → map to these 4 categories (drop/ignore others).

**Processed trajectory counts (Table 1):**
| | Fishing | Passenger | Cargo | Tanker |
|---|---|---|---|---|
| Train | 1436 | 13678 | 37842 | 17335 |
| Val | 479 | 4349 | 11808 | 5291 |
| Test | 786 | 5030 | 12263 | 5223 |

Note: heavy class imbalance (cargo dominant, fishing rarest). Paper reports plain accuracy; no resampling mentioned.

**AIS fields used (per message):**
- Kinematic: `LAT`, `LON`, `SOG` (speed over ground), `COG` (course over ground)
- Static: `LEN` (length), `WID` (width), `DRA` (draft)
- Plus `MMSI`, `BaseDateTime` (used only for grouping, then dropped)

---

## 2. Trajectory Extraction (Section 3)

Run per the 5 steps below to convert raw AIS rows → fixed-length trajectory tensors.

**Step 1 — Trajectory Division**
- Group messages by `MMSI` (unique vessel id).
- Within a vessel, split by **calendar day**.
- Within a day, cut a new trajectory whenever the gap between adjacent message timestamps **> 2 hours**.

**Step 2 — Trajectory Filtering**
- Drop trajectory if its **time span < 6 hours**.
- Drop trajectory if it has **< 160 AIS messages**.

**Step 3 — Abnormal Trajectory Removal**
- Drop if **max SOG ≤ 1 knot/h** (essentially stationary).
- Drop if fraction of messages with **SOG > 2 knot/h is ≤ 30%** of total messages.

**Step 4 — Normalization + Seven-hot Encoding**
- Discard timestamp.
- Normalize the 7 attributes (`LAT, LON, SOG, COG, WID, LEN, DRA`).
- **Seven-hot encoding:** one-hot encode each of the 7 attributes independently (discretize each into bins → one bit = 1), then concatenate the 7 one-hot vectors → single binary vector per message.
  - Purpose: discretization helps the CNN learn spatio-temporal structure vs. raw floats.
  - ⚠️ **Paper does not state the bin count per attribute.** You must choose bin resolution per attribute (config it). See ref: Nguyen et al. 2018/2021 (the inspiration) for binning scheme. Treat as a tunable hyperparameter; pick bins so the concatenated per-message vector width is consistent and the conv stack flattens to 250 (see §4).

**Step 5 — Trajectory Sampling**
- Trajectories have variable length → **sample a fixed number of messages** from each so all trajectories share one length. (Paper doesn't give the exact count; min is 160 from Step 2, so fixed length ≤ 160 — use a config value, e.g. 160. Tune.)

**Output per trajectory:** a 2D tensor `[T_fixed, seven_hot_dim]` (treated as a single-channel image by the CNN).

---

## 3. Model: SSL-VTC (Section 4)

VAE-based semi-supervised classifier (the Kingma et al. 2014 "M2" deep generative model adapted to AIS). Three jointly-trained CNN modules:

- **Classifier** `q(y|x)` — discriminative learning.
- **Encoder** `q(z|x,y)` — latent variable extraction.
- **Decoder** `p(x|y,z)` — reconstruction (generative learning).

Latent var `z` per trajectory. Label `y` one of 4 classes.

### 3.1 Loss

```
L = L1 + L2 + α · L_clf            (Eq. 1)
α = β · (n2 / n1)                  (Eq. 2)
```
- `n1` = #labeled samples, `n2` = #unlabeled samples, `β` = hyperparameter.
- `L1` — neg. variational bound over **labeled** data (Eqs. 4–5).
- `L2` — neg. variational bound over **unlabeled** data (Eqs. 8–9).
- `L_clf` — cross-entropy on labeled data (Eq. 10).

**Labeled bound** (per sample, minimization form, Eq. 4):
```
J(x,y) = − E_{q(z|x,y)} [ log p(x|y,z) + log p(y) + log p(z) − log q(z|x,y) ]
L1 = Σ_{(x,y)∈D1} J(x,y)                                     (Eq. 5)
```
Components: reconstruction `log p(x|y,z)`, label prior `log p(y)`, latent KL term `log p(z) − log q(z|x,y)` (z → Gaussian prior).

**Unlabeled bound** (Eqs. 7–9): marginalize over all possible labels weighted by classifier posterior, plus entropy term:
```
U(x) = Σ_y q(y|x) · J(x,y)  +  H(q(y|x))      [note sign per Eq. 8]
L2 = Σ_{x∈D2} U(x)                                          (Eq. 9)
```
where `H` is the entropy of the classifier output (Cover & Thomas).

**Classifier loss** (Eq. 10):
```
L_clf = E_{x∈D1} [ − log q(y|x) ]   # cross-entropy
```

> Implementation note: this is the standard Kingma M2 semi-supervised VAE loss. Reuse a reference M2 implementation for the bound math; swap in the CNN modules below. Reparameterization trick for `z` (μ, σ).

### 3.2 Encoder `q(z|x,y)`

- **Trajectory branch:** input = seven-hot trajectory `[1, T, D]`. 5 conv layers, each + ReLU. Channels: `1 → 5 → 5 → 5 → 5`. Kernel sizes: `10, 10, 10, 5, 3`. Flatten → feature vector **size 250**.
- **Label branch:** one-hot label → 1 FC layer → vector of **50** nodes.
- Concatenate (250 + 50 = 300) → 2 FC layers → output **μ** and **σ**.
- Reparameterize: `z = μ + σ·ε`, `ε ~ N(0,I)`.
- Labeled input: pass its true label. **Unlabeled input: pair x with every possible label** (run once per class).

### 3.3 Decoder `p(x|y,z)`

- Concatenate `z` and one-hot label `y`.
- 2 FC layers → vector size **250** → reshape.
- 5 deconv (transposed conv) layers. Output channels: `5 → 5 → 5 → 5 → 1`. Kernel sizes: `3, 5, 10, 10, 10` (mirror of encoder).
- First 4 deconv layers + ReLU; **last deconv + Sigmoid** (outputs in [0,1], matching binary seven-hot target).

### 3.4 Classifier `q(y|x)`

- Input = seven-hot trajectory.
- 5 conv layers, each + ReLU. Channels `1 → 5 → 5 → 5 → 5`, kernels `10, 10, 10, 5, 3` (same conv block as encoder).
- Flatten → size **250** → 1 FC layer → **Softmax** over 4 classes.

> The classifier conv block and encoder conv block share the same architecture (not necessarily shared weights — paper treats them as separate modules).

### 3.5 Training (Algorithm 1)

```
for epoch in 1..E:
    batch_L = sample minibatch from D1 (labeled)
    compute L1 (Eq.5), L_clf (Eq.10) on batch_L
    batch_U = sample minibatch from D2 (unlabeled)
    compute L2 (Eq.9) on batch_U
    L = L1 + L2 + α·L_clf
    gradient step on classifier + encoder + decoder params (jointly)
```

**Hyperparameters (Section 5.2):**
- Optimizer: **Adam**
- Learning rate: **0.0001**
- Batch size: **100**
- Epochs: **50**
- Model selection: pick best epoch on **validation** set, report its **test accuracy**.

**β grid search (Table 5)** — depends on labeled fraction:
| Labeled % | 5% | 20% | 40% | 60% |
|---|---|---|---|---|
| β | 10 | 10 | 100 | 1000 |

β sensitivity (40% labeled, Table 6): acc rises sharply β 1→100, plateaus 100→1000.

---

## 4. Implementation Notes / Gotchas

1. **Flatten size 250 is fixed** by the paper. Your `T_fixed` (sampled length) and `D` (seven-hot width) plus the 5-conv stack (kernels 10,10,10,5,3, 5 channels, default stride/no-pad assumed) must produce flatten=250. Back-solve conv geometry to hit 250, or add pooling/stride to match. This constrains §2 Step 4/5 choices — tune `D` (bins) and `T_fixed` together.
2. **Conv padding/stride not specified.** Assume stride 1, no padding (default) unless 250 doesn't work out; then adjust. Document whatever you pick.
3. **Seven-hot bin resolution not specified.** Tunable. Start from Nguyen et al. 2018/2021 scheme. All 7 attributes normalized before binning.
4. **Unlabeled forward pass** runs the trajectory through encoder once per class label (4×) to compute the marginalized L2 — budget compute accordingly.
5. **Decoder target** is the binary seven-hot tensor → use binary cross-entropy reconstruction (sigmoid output), consistent with Bernoulli `p(x|y,z)`.
6. **z prior** = standard Gaussian; KL closed form from μ, σ.

---

## 5. Experiments to Reproduce (Section 5.3)

**5.3.1 — Classifier + seven-hot validation (Table 2), full-label supervised:**
| Method | Acc % |
|---|---|
| SVM | 78.50 |
| Decision Tree | 77.62 |
| KNN | 81.25 |
| MLP | 81.80 |
| CNN (raw values) | 85.08 |
| MLP + seven-hot | 89.61 |
| **CNN + seven-hot (ours)** | **92.22** |

**5.3.2 — Static-info ablation (Table 3):**
| Variant | Acc % |
|---|---|
| WO-LWD (no len/wid/draft) | 71.43 |
| WO-LEN | 80.54 |
| WO-WID | 88.48 |
| WO-DRA | 87.96 |
| Full | 92.22 |
→ length most important; static info gives +29.1% over kinematic-only.

**5.3.3 — Semi-supervised vs baseline (Table 4).** Labeled fraction ∈ {5,20,40,60}%, rest = unlabeled. Baseline = classifier trained on labeled only.
| Labeled % | 5 | 20 | 40 | 60 |
|---|---|---|---|---|
| Baseline (WO-LWD) | 56.36 | 65.02 | 67.15 | 69.88 |
| Baseline | 72.60 | 80.59 | 86.13 | 89.35 |
| SSL-VTC (WO-LWD) | 59.74 | 66.83 | 68.08 | 70.25 |
| **SSL-VTC** | **77.93** | **84.51** | **88.96** | **90.01** |
→ SSL gain largest when labels scarcest.

**5.3.4 — β sensitivity (Table 6, 40% labeled):** β=1→75.02, 10→84.44, 100→88.96, 1000→88.94.

**5.3.5 — Missing static info (Table 7, 20% labeled).** Static available for {5,20,60}% of data; fill rest with **zero** vs **mean**. Mean > zero in all cases:
| Avail static % | 5 | 20 | 60 |
|---|---|---|---|
| | Zero / Mean | Zero / Mean | Zero / Mean |
| Baseline | 66.02 / 68.63 | 68.64 / 70.12 | 80.04 / 80.76 |
| SSL-VTC | 67.39 / 70.37 | 76.08 / 77.68 | 81.59 / 82.66 |

Also produce: test-accuracy-vs-epoch curves (20% & 40% labeled), confusion matrices per method/fraction.

---

## 6. Suggested Build Order

1. AIS ingest + Steps 1–3 (division/filter/abnormal) → save intermediate trajectory CSVs per split.
2. Seven-hot encoder + sampling (Steps 4–5) → tensors; verify flatten=250 geometry.
3. CNN classifier alone → reproduce Table 2 / 92.22% (sanity gate before VAE).
4. Add encoder/decoder + M2 loss → SSL-VTC; reproduce Table 4.
5. Ablations (Tables 3, 6, 7), curves, confusion matrices.
