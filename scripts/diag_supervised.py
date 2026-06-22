"""Diagnostic: why does the supervised Full classifier plateau at ~86 vs paper 92?

Trains the CNN classifier (sevenhot, Full static, 100% labels) and logs
val_acc / val_f1 / test_acc EVERY epoch under two LR regimes. Then reports:
  - peak test_acc reachable at ANY epoch          -> is 92 even attainable?
  - test_acc @ best-val-accuracy epoch (paper sel) -> what paper-style selection gives
  - test_acc @ best-val-macroF1 epoch (our sel)    -> selection-metric cost
  - final-epoch test_acc                            -> does cosine LR drop it late?

Run:  CUDA_VISIBLE_DEVICES=1 python scripts/diag_supervised.py
"""
from __future__ import annotations
import json, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.sslvtc.config import load_config
from src.sslvtc.device import get_device
from src.sslvtc.models import Classifier
from src.sslvtc.dataset import TrajectoryDataset
from src.sslvtc.metrics import classification_metrics

CONFIG = "configs/fullus2019.yaml"
EPOCHS = 50


def run(lr_schedule: str) -> dict:
    cfg = load_config(CONFIG)
    device = get_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)

    enc = cfg.encoding  # Full: use_len/wid/dra all True (default)
    mk = lambda sp: TrajectoryDataset(cfg.paths.processed, sp, enc, mode="sevenhot")
    train_ds, val_ds, test_ds = mk("train"), mk("val"), mk("test")
    t, d = train_ds.shape()
    n_classes = cfg.model.n_classes
    bs, nw = cfg.train.batch_size, cfg.train.num_workers
    pin = device.type == "cuda"

    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
    vl = DataLoader(val_ds, batch_size=bs, num_workers=nw, pin_memory=pin)
    tel = DataLoader(test_ds, batch_size=bs, num_workers=nw, pin_memory=pin)

    clf = Classifier(cfg.model, t, d).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if lr_schedule == "cosine" else None
    loss_fn = torch.nn.CrossEntropyLoss()

    @torch.no_grad()
    def ev(loader):
        clf.eval(); yt, yp = [], []
        for x, y in loader:
            yp.append(clf(x.to(device)).argmax(1).cpu().numpy()); yt.append(y.numpy())
        m = classification_metrics(np.concatenate(yt), np.concatenate(yp), n_classes)
        return m["accuracy"], m["macro_f1"]

    hist = {"val_acc": [], "val_f1": [], "test_acc": [], "lr": []}
    for ep in range(EPOCHS):
        clf.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss_fn(clf(x), y).backward(); opt.step()
        if sched: sched.step()
        va, vf = ev(vl); ta, _ = ev(tel)
        hist["val_acc"].append(va); hist["val_f1"].append(vf); hist["test_acc"].append(ta)
        hist["lr"].append(opt.param_groups[0]["lr"])
        print(f"  [{lr_schedule}] ep {ep+1:2d}/{EPOCHS}  val_acc={va*100:.2f} val_f1={vf*100:.2f} test_acc={ta*100:.2f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)

    va, vf, ta = hist["val_acc"], hist["val_f1"], hist["test_acc"]
    i_acc, i_f1 = int(np.argmax(va)), int(np.argmax(vf))
    return {
        "schedule": lr_schedule,
        "peak_test_acc": round(max(ta)*100, 2),
        "test_at_bestValAcc": round(ta[i_acc]*100, 2),
        "test_at_bestValF1": round(ta[i_f1]*100, 2),
        "final_test_acc": round(ta[-1]*100, 2),
        "best_val_acc_epoch": i_acc + 1,
        "best_val_f1_epoch": i_f1 + 1,
        "history": hist,
    }


if __name__ == "__main__":
    out = {}
    for sch in ("cosine", "none"):
        print(f"\n===== LR schedule: {sch} =====", flush=True)
        t0 = time.time()
        out[sch] = run(sch)
        out[sch]["minutes"] = round((time.time()-t0)/60, 1)
    print("\n================ SUMMARY (paper Full=92.22) ================")
    for sch, r in out.items():
        print(f"[{sch:7s}] peak={r['peak_test_acc']}  @bestValAcc={r['test_at_bestValAcc']} (ep{r['best_val_acc_epoch']})  "
              f"@bestValF1={r['test_at_bestValF1']} (ep{r['best_val_f1_epoch']})  final={r['final_test_acc']}  ({r['minutes']}min)")
    json.dump(out, open("/tmp/diag_supervised.json", "w"), indent=2)
    print("\nfull curves -> /tmp/diag_supervised.json")
