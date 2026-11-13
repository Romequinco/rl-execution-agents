"""Barrido de la idea B del ej. 4: plan al 90 %, penalización del pendiente k y γ.

Un factor cada vez sobre la configuración final del ej. 2 (TrackingCierre, c = 20, γ = 0,9):
- referencia del ej. 2: horizon = 1, k = 0;
- plan comprimido: horizon = 0,9 con k ∈ {0, 1, 10, 50};
- γ alto (G3) para un término terminal: γ = 0,995 con k ∈ {0, 10}.
Tres semillas; selección solo con train. Reproducir desde la raíz: python scripts/ej4_ideaB.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from common import ej4, protocol as pr  # noqa: E402

CONFIGS = [(1.0, 0.0, 0.9),
           (0.9, 0.0, 0.9), (0.9, 1.0, 0.9), (0.9, 10.0, 0.9), (0.9, 50.0, 0.9),
           (0.9, 0.0, 0.995), (0.9, 10.0, 0.995)]

if __name__ == "__main__":
    t0 = time.time()
    res = ej4.run_all_b(CONFIGS, seeds=(0, 1, 2), workers=4)
    pr.save_df(res, "ej4_ideaB")
    tab = res.groupby(["horizon", "k", "gamma"]).apply(ej4.summary)
    print(tab.round(3).to_string())
    print(f"{time.time() - t0:.0f} s")
