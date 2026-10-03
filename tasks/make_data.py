"""Write the small synthetic CSV files used by the figure tasks (deterministic)."""

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "data"


def training_curves(rng: np.random.Generator) -> list[dict]:
    rows = []
    for method, ceiling, speed in [("baseline", 0.82, 0.10), ("ours", 0.88, 0.14), ("ours w/o prior", 0.85, 0.12)]:
        for seed in range(3):
            noise = rng.normal(0, 0.006, 50).cumsum() * 0.3
            for epoch in range(1, 51):
                acc = ceiling - (ceiling - 0.35) * np.exp(-speed * epoch) + noise[epoch - 1] + rng.normal(0, 0.004)
                rows.append({"method": method, "seed": seed, "epoch": epoch, "val_acc": round(float(acc), 4)})
    return rows


def ablation() -> list[dict]:
    table = {
        "full model": (98.1, 0.3, 94.7, 0.5),
        "no velocity term": (95.4, 0.6, 90.2, 0.8),
        "no multi-step": (96.8, 0.4, 92.9, 0.6),
        "single scale": (94.9, 0.7, 89.5, 0.9),
    }
    rows = []
    for variant, (m1, s1, m2, s2) in table.items():
        rows.append({"variant": variant, "dataset": "MVTec-AD", "auroc": m1, "std": s1})
        rows.append({"variant": variant, "dataset": "VisA", "auroc": m2, "std": s2})
    return rows


def write(name: str, rows: list[dict]) -> None:
    HERE.mkdir(exist_ok=True)
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    write("curves.csv", training_curves(np.random.default_rng(0)))
    write("ablation.csv", ablation())
    print(f"wrote CSV files to {HERE}")
