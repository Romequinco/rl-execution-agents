"""Carga de los datos del taller (repo best-execution del profesor).

Los ficheros son pickles de pandas: diccionario {"train", "val", "test"} → diccionario
por día → DataFrame cada 5 s (13:00-17:25) con columnas date, ask1, bid1, vask1, vbid1,
last, cumvol. Se usa la carpeta ``new_version/`` porque los pickles de la raíz se
crearon con pandas 1.x y no cargan desde pandas 2.1 (README del taller).

Solo se cargan estos ficheros conocidos del taller: un pickle ejecuta código al
deserializarse, así que no debe usarse esta función con ficheros de otra procedencia.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs" / "raw" / "MIAX14" / "best-execution" / "data" / "new_version"
ASSETS = {"REP": "rep_data.pkl", "SAN": "san_data.pkl"}


def load(asset: str = "REP") -> dict:
    """Devuelve {"train": {día: DataFrame}, "val": {...}, "test": {...}}.

    Con la variable de entorno ``RL_DATA=synthetic`` devuelve datos sintéticos con el mismo
    esquema: solo para probar que los notebooks se ejecutan de principio a fin.
    """
    if os.environ.get("RL_DATA") == "synthetic":
        return {"train": synthetic_data(8, seed=1), "val": synthetic_data(3, seed=2), "test": synthetic_data(3, seed=3)}
    with open(DATA_DIR / ASSETS[asset], "rb") as f:
        return pickle.load(f)


def summary(data: dict) -> dict:
    """Número de días y filas por split, para la celda de exploración de datos."""
    return {
        split: {"dias": len(days), "filas_por_dia": sorted({len(df) for df in days.values()})[:3]}
        for split, days in data.items()
    }


def synthetic_data(n_days=3, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for d in range(n_days):
        day = pd.Timestamp("2019-01-02") + pd.Timedelta(days=d)
        idx = pd.date_range(day + pd.Timedelta(hours=13), day + pd.Timedelta(hours=17, minutes=25), freq="5s")
        mid = 15 + np.cumsum(rng.normal(0, 0.005, len(idx)))
        spread = 0.01
        vol = rng.poisson(800, len(idx)) * (rng.random(len(idx)) < 0.6)
        out[day.date()] = pd.DataFrame({
            "date": idx, "ask1": mid + spread / 2, "bid1": mid - spread / 2,
            "vask1": rng.integers(100, 5000, len(idx)), "vbid1": rng.integers(100, 5000, len(idx)),
            "last": mid, "cumvol": 2_000_000 + np.cumsum(vol),
        }, index=idx)
    return out
