import json
import math
import os
import random
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Path helpers ──────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    """Convert a model name like 'bert-base-uncased' to 'bert_base_uncased'."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def safe_mkdir_run_dir(root: str) -> str:
    """
    Create a run directory at *root*.

    If *root* already exists and is non-empty, create root__run2, root__run3, …
    This prevents accidentally overwriting a previous experiment.
    """
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
        return root

    if len(os.listdir(root)) == 0:
        return root

    i = 2
    while True:
        candidate = f"{root}__run{i}"
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        i += 1


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_json(path: str, obj: Dict[str, Any]):
    """Write *obj* to *path* as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ── GPU power monitoring ──────────────────────────────────────────────────────

class NvidiaPowerMonitor:
    """
    Samples GPU power draw via nvidia-smi in a background thread.

    Usage::

        monitor = NvidiaPowerMonitor(interval_s=1.0)
        monitor.start()
        # ... run training ...
        monitor.stop()
        print(monitor.stats())
        monitor.save_csv("power_samples.csv")
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._stop      = threading.Event()
        self.samples: List[Tuple[float, float]] = []
        self._t0        = None
        self._thread    = None

    @staticmethod
    def _read_power_w() -> float:
        try:
            out = subprocess.check_output(
                "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
                shell=True,
            )
            return float(out.decode("utf-8").strip())
        except Exception:
            return float("nan")

    def start(self):
        self._stop.clear()
        self.samples = []
        self._t0     = time.time()

        def _run():
            while not self._stop.is_set():
                p = self._read_power_w()
                t = time.time() - self._t0
                self.samples.append((t, p))
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def stats(self) -> Dict[str, Any]:
        """Return average power and estimated energy consumption."""
        vals = [
            p for _, p in self.samples
            if not (isinstance(p, float) and math.isnan(p))
        ]
        if not vals:
            return {"available": False}

        avg_power_w = float(np.mean(vals))
        # Energy (kWh) = sum(W) * dt / 3,600,000
        energy_kwh  = float(np.sum(vals) * self.interval_s / 3_600_000.0)
        return {
            "available":   True,
            "avg_power_w": avg_power_w,
            "energy_kwh":  energy_kwh,
            "num_samples": len(vals),
            "interval_s":  self.interval_s,
        }

    def save_csv(self, path: str):
        """Save raw (t_sec, power_w) samples to a CSV file."""
        pd.DataFrame(self.samples, columns=["t_sec", "power_w"]).to_csv(path, index=False)
