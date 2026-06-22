"""Generate all figures for the three finding documents. Outputs to paper/figures/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
C = {"cargo": "#1f77b4", "tanker": "#ff7f0e", "passenger": "#2ca02c", "fishing": "#d62728"}


# ---- Finding 1 ----
def f1_distribution():
    classes = ["cargo", "tanker", "passenger", "fishing"]
    raw = [27.2, 13.6, 29.4, 29.9]      # raw message-level share (full dataset, 210.8M msgs)
    paper = [53.8, 24.7, 19.5, 2.0]     # paper Table 1 (train)
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, raw, w, label="Raw nationwide AIS (what's really out there)", color="#888")
    ax.bar(x + w/2, paper, w, label="SSL-VTC benchmark (Table 1)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("share of data (%)")
    ax.set_title("Fig 1.1  The benchmark is cargo-heavy; reality is balanced\nFishing is 28.7% of real data but only 2% of the benchmark")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "f1_1_distribution.png"); plt.close(fig)


def f1_draft():
    classes = ["cargo", "tanker", "passenger", "fishing"]
    draft = [89.7, 84.7, 23.6, 12.1]  # full dataset (12,852 vessels)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(classes, draft, color=[C[c] for c in classes])
    ax.axhline(50, ls="--", color="gray", lw=1)
    for b, v in zip(bars, draft):
        ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v}%", ha="center", fontsize=10)
    ax.set_ylabel("% of vessels that report Draft")
    ax.set_ylim(0, 100)
    ax.set_title("Fig 1.2  Draft is the hidden filter\nFishing & small passenger boats almost never report it")
    fig.tight_layout(); fig.savefig(FIG / "f1_2_draft.png"); plt.close(fig)


def f1_kept():
    classes = ["cargo", "tanker", "passenger", "fishing"]
    # trajectory-level survival from the actual inclusive extraction (static_complete crosstab)
    kept = [91.4, 83.6, 34.5, 8.3]
    dropped = [100 - k for k in kept]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(classes, kept, label="kept", color="#2ca02c")
    ax.bar(classes, dropped, bottom=kept, label="silently dropped", color="#d62728", alpha=0.75)
    for i, c in enumerate(classes):
        ax.text(i, 102, f"drop {dropped[i]:.0f}%", ha="center", fontsize=9, color="#d62728")
    ax.set_ylabel("% of vessels")
    ax.set_ylim(0, 112)
    ax.set_title('Fig 1.3  Who survives the "must have static info" filter\n~91% of fishing vessels are thrown away')
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "f1_3_kept.png"); plt.close(fig)


def f1_consequence():
    # 2x3 grid: macro-F1 for the paper's model A vs realistic model B, across populations
    pops = ["complete\n(kept)", "dropped", "all (real ocean)"]
    A = [91.2, 23.4, 64.3]   # paper's model (matched-norm cnorm eval, 3 seeds)
    B = [87.9, 78.4, 88.1]   # realistic model (trained on all, 3 seeds)
    x = np.arange(len(pops)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ba = ax.bar(x - w/2, A, w, label="Model A — paper's (trained on complete only)", color="#1f77b4")
    bb = ax.bar(x + w/2, B, w, label="Model B — realistic (trained on all)", color="#2ca02c")
    for bars in (ba, bb):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.8, f"{b.get_height():.0f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(pops)
    ax.set_ylabel("macro-F1 (%)"); ax.set_ylim(0, 100)
    ax.annotate("paper's model\ncollapses on the\nboats it deleted", xy=(1-w/2, 23.4), xytext=(0.55, 45),
                fontsize=8, color="#1f77b4", arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    ax.set_title("Fig 1.4  The paper's model is near-useless on the excluded vessels\n"
                 "A collapses 91->23 on dropped boats; retraining on all (B) recovers it to 78")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "f1_4_consequence.png"); plt.close(fig)


def f1_perclass():
    # per-class size boost on the realistic cohort (with size vs kinematic-only), 3 seeds
    classes = ["fishing", "passenger", "cargo", "tanker"]
    nosize = [86.8, 82.6, 74.4, 61.0]
    withsize = [91.5, 89.5, 88.9, 83.1]
    boost = [round(w - n, 1) for w, n in zip(withsize, nosize)]
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w/2, nosize, w, label="kinematic only (motion)", color="#999")
    ax.bar(x + w/2, withsize, w, label="+ size (length/width/draft)", color="#1f77b4")
    for i in range(len(classes)):
        ax.text(x[i], max(nosize[i], withsize[i]) + 1.5, f"+{boost[i]}", ha="center",
                fontsize=11, fontweight="bold",
                color=("#d62728" if classes[i] == "fishing" else "#333"))
    ax.set_xticks(x); ax.set_xticklabels([c + ("\n(target)" if c == "fishing" else "") for c in classes])
    ax.set_ylabel("recall (%)"); ax.set_ylim(50, 100)
    ax.set_title("Fig 1.5  Size helps the big ships, barely the fishing target\n"
                 "Size boost: tanker +22, cargo +14 — but fishing only +4.7 (already high from motion)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "f1_5_perclass.png"); plt.close(fig)


# ---- Finding 2 ----
def f2_leakage():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["Temporal split\n(paper's method)", "Vessel-disjoint\n(our fix)"]
    vals = [80.9, 0.0]
    bars = ax.bar(labels, vals, color=["#d62728", "#2ca02c"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+2, f"{v}%", ha="center", fontsize=12)
    ax.set_ylabel("% of test trajectories whose\nvessel was also in training")
    ax.set_ylim(0, 100)
    ax.set_title("Fig 2.1  How much the test set 'leaks' training vessels")
    fig.tight_layout(); fig.savefig(FIG / "f2_1_leakage.png"); plt.close(fig)


def f2_nocollapse():
    # static gain (Full - WO-LWD) accuracy, coarse vs fine, temporal vs vd
    groups = ["Coarse model\n(86% acc)", "Faithful model\n(92% acc)"]
    temporal = [13.18, 18.72]
    vd = [13.77, 17.02]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, temporal, w, label="Temporal (80.9% leakage)", color="#d62728")
    ax.bar(x + w/2, vd, w, label="Vessel-disjoint (0% leakage)", color="#2ca02c")
    for i in range(2):
        ax.text(x[i]-w/2, temporal[i]+0.3, f"+{temporal[i]:.1f}", ha="center", fontsize=9)
        ax.text(x[i]+w/2, vd[i]+0.3, f"+{vd[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("accuracy boost from static features (pts)")
    ax.set_title("Fig 2.2  The static-feature boost barely changes when leakage is removed\n(if it were memorization, green would crash)")
    ax.legend(fontsize=9); ax.set_ylim(0, 22)
    fig.tight_layout(); fig.savefig(FIG / "f2_2_nocollapse.png"); plt.close(fig)


def f2_fishing():
    groups = ["Coarse model", "Faithful model"]
    temporal = [81.95, 90.14]
    vd = [89.98, 88.02]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, temporal, w, label="Temporal (leaky)", color="#d62728")
    ax.bar(x + w/2, vd, w, label="Vessel-disjoint (honest)", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("fishing recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Fig 2.3  Fishing recall (the 'hardest, most identity-like' class)\ndoes NOT fall on the honest split")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "f2_3_fishing.png"); plt.close(fig)


# ---- Finding 3 ----
def f3_bins():
    bins = [40, 80, 120, 200]
    acc = [86.01, 88.59, 90.52, 91.79]
    labels = ["10/20/10", "20/40/20", "30/60/30", "50/100/50"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(bins, acc, "o-", color="#1f77b4", lw=2, ms=8)
    for b, a, l in zip(bins, acc, labels):
        ax.annotate(f"{a}\n({l})", (b, a), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax.axhline(92.22, ls="--", color="#d62728", lw=1.5, label="paper's reported 92.22")
    ax.set_xlabel("total static bins (WID+LEN+DRA resolution)")
    ax.set_ylabel("Full-model accuracy (%)")
    ax.set_ylim(84, 94)
    ax.set_title("Fig 3.1  One undisclosed knob moves the headline 86 → 92\nFiner static bins alone recover the paper's number")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "f3_1_bins.png"); plt.close(fig)


def f3_ssl():
    fracs = ["20%", "40%", "60%"]
    base = [81.67, 84.83, 85.53]
    ssl = [82.33, 85.03, 85.51]
    x = np.arange(len(fracs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, base, w, label="labeled-only baseline", color="#888")
    ax.bar(x + w/2, ssl, w, label="SSL-VTC (semi-supervised)", color="#1f77b4")
    for i in range(3):
        ax.text(x[i], max(base[i], ssl[i])+0.4, f"+{ssl[i]-base[i]:+.1f}".replace("++","+"), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(fracs)
    ax.set_xlabel("fraction of training labels used")
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(78, 90)
    ax.set_title("Fig 3.2  On faithful data, semi-supervised barely beats the baseline\n(paper claimed a clear gain)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "f3_2_ssl.png"); plt.close(fig)


def f_ev_danish_draft():
    classes = ["cargo", "tanker", "passenger", "fishing"]
    us     = [89.7, 84.7, 23.6, 12.1]
    danish = [97.8, 98.9, 77.6, 47.5]
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, us,     w, label="US 2019 (12,852 vessels)",    color=[C[c] for c in classes], alpha=0.55)
    ax.bar(x + w/2, danish, w, label="Danish 2019 (1,464 vessels)", color=[C[c] for c in classes])
    for i, (u, d) in enumerate(zip(us, danish)):
        ax.text(x[i]-w/2, u+1.5, f"{u}%", ha="center", fontsize=9, alpha=0.75)
        ax.text(x[i]+w/2, d+1.5, f"{d}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("% of vessels that ever report Draught")
    ax.set_ylim(0, 110)
    ax.set_title("Fig EV.1  MNAR replicates in Danish/Baltic AIS\n"
                 "Fishing lowest in both regions — structural gap vs cargo/tanker preserved")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "f_ev_danish_draft.png"); plt.close(fig)


if __name__ == "__main__":
    for fn in [f1_distribution, f1_draft, f1_kept, f1_consequence, f1_perclass, f2_leakage,
               f2_nocollapse, f2_fishing, f3_bins, f3_ssl, f_ev_danish_draft]:
        fn(); print("wrote", fn.__name__)
    print("done ->", FIG)
