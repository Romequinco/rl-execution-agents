"""Carga de los datos del taller (repo best-execution del profesor).

Los ficheros son pickles de pandas: diccionario {"train", "val", "test"} → diccionario
{datetime.date: DataFrame}. Cada DataFrame tiene una fila cada 5 s de 13:00 a 17:25
(~3.179 filas) con columnas ask1, bid1, vask1, vbid1, last, cumvol y date. REP: 60 días de
train (2018), 30 de val y 30 de test (2019).

Se usa la carpeta ``new_version/`` porque los pickles de la raíz se crearon con pandas 1.x
y no cargan desde pandas 2.1 (README del taller). Solo se cargan estos ficheros conocidos:
un pickle ejecuta código al deserializarse, así que esta función no debe usarse con
ficheros de otra procedencia.
"""
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs" / "raw" / "MIAX14" / "best-execution" / "data" / "new_version"
ASSETS = {"REP": "rep_data.pkl", "SAN": "san_data.pkl"}


def load(asset: str = "REP") -> dict:
    """Devuelve {"train": {día: DataFrame}, "val": {...}, "test": {...}}."""
    with open(DATA_DIR / ASSETS[asset], "rb") as f:
        return pickle.load(f)


def summary(data: dict) -> dict:
    """Número de días y filas por split, para la celda de exploración de datos."""
    return {
        split: {"dias": len(days), "filas_por_dia": sorted({len(df) for df in days.values()})[:3]}
        for split, days in data.items()
    }
