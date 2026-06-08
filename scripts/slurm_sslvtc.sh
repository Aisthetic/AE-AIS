#!/usr/bin/env bash
#SBATCH --job-name=sslvtc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=sslvtc_%j.out
# SLURM template for the full SSL-VTC run on a lab cluster.
# Adjust partition/gres/account to your site. Edit CONFIG as needed.
set -euo pipefail

CONFIG="${CONFIG:-configs/gulf2019.yaml}"

module load python/3.12 cuda 2>/dev/null || true   # adapt to your module system

# one-time env setup (skip if .venv already built on the cluster fs)
if [ ! -d .venv ]; then
  python3.12 -m venv .venv
  .venv/bin/pip install -e ".[dev]"
fi

# Force CUDA on the cluster (config device: auto will also pick cuda if present).
export PYTHON=.venv/bin/python
sed -i 's/device: auto/device: cuda/' "$CONFIG" 2>/dev/null || true

# Stage 1: data prep (CPU-heavy; can be a separate non-GPU job to save GPU hours)
.venv/bin/python -m sslvtc.cli -c "$CONFIG" download
.venv/bin/python -m sslvtc.cli -c "$CONFIG" ingest
.venv/bin/python -m sslvtc.cli -c "$CONFIG" extract

# Stage 2: experiments (GPU)
.venv/bin/python -m sslvtc.cli -c "$CONFIG" experiment all
