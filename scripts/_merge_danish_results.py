"""Merge regime_A.csv + regime_B.csv -> consequence_danish.csv and print summary."""
import pandas as pd
from pathlib import Path

BASE = Path("/mnt/storage_1_10T/zezzahed/AIS_Data/danishais2019/results")
OUT  = BASE / "consequence_danish.csv"

a = pd.read_csv(BASE / "regime_A.csv")
b = pd.read_csv(BASE / "regime_B.csv")
df = pd.concat([a, b], ignore_index=True)
df.to_csv(OUT, index=False)

print(df.to_string(index=False))
print("\n==== mean over seeds ====")
print(df.groupby("regime")[["accuracy","macro_f1","fishing_recall"]].agg(["mean","std"]).round(2).to_string())
print(f"\n-> {OUT}")
