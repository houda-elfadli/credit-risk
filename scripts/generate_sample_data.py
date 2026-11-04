"""Genere un fichier data/raw/credit.csv fictif pour tester le projet."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500
data = pd.DataFrame(
    {
        "age": rng.integers(21, 70, n),
        "income": rng.normal(40000, 12000, n).round(0),
        "loan_amount": rng.normal(15000, 6000, n).round(0),
        "credit_history": rng.integers(0, 2, n),
    }
)
risk = (data["loan_amount"] / data["income"]) - 0.2 * data["credit_history"]
data["default"] = (risk + rng.normal(0, 0.1, n) > 0.4).astype(int)
data.loc[rng.choice(n, 10, replace=False), "income"] = np.nan

out = Path("data/raw/credit.csv")
out.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(out, index=False)
print(f"Fichier genere : {out}")
