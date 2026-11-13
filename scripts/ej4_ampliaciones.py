"""Ampliaciones del ej. 4: idea A, 5 semillas, banda de finalización y pruebas de robustez.

1. Idea B con 5 semillas (se añaden las semillas 3 y 4 a la configuración elegida).
2. Banda de finalización (B-2): recompensa con pasillo 80-95 % en vez de plan fijo, 5 semillas.
3. Robustez del agente de B (5 semillas): órdenes fuera del rango de entrenamiento y empezar con retraso.
4. Idea A: estado con 4 variables de mercado y término local de precio de peso w ∈ {0; 0,01; 0,03},
   placebo (variables barajadas) con w ∈ {0,01; 0,03}, 3 semillas, selección en validación (270 episodios).
   Regla fijada antes de mirar: entre los w reales que completan el 100 % y quedan en banda ≥ 95 %,
   el de mayor ahorro medio frente al TWAP a 0,9·N. Test una sola vez con el elegido, su placebo y las referencias.
Reproducir desde la raíz: python scripts/ej4_ampliaciones.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import pandas as pd  # noqa: E402

from common import data as dt, ej4, protocol as pr  # noqa: E402

SEEDS5 = (0, 1, 2, 3, 4)
SEEDS_A = (0, 1, 2)
W_REAL, W_PLACEBO = (0.0, 0.01, 0.03), (0.01, 0.03)


def robustness(D):
    train = D["train"]
    R = ej4.make_reward(0.9, 1.0, 0.9)
    wide = ej4.make_grid_custom(train, Ns=(250, 300, 700, 800), rhos=(0.04, 0.05, 0.15, 0.2), n_days=3)
    inside = pr.make_grid(train, n_days=3, seed=0)
    rows = []
    for s in SEEDS5:
        a = ej4.load_agent_b(s, train, env_kw=ej4.WIDE)
        rows.append(ej4.evaluate_general(a, train, wide, R, 0.9, env_kw=ej4.WIDE).assign(prueba="fuera de rango", seed=s))
        for delay in (0.2, 0.4, 0.6):
            rows.append(ej4.evaluate_general(a, train, inside, R, 0.9, delay=delay)
                        .assign(prueba=f"retraso {delay:.0%}", seed=s))
    rows.append(ej4.evaluate_general(ej4.TWAPh(0.9), train, wide, R, 0.9, env_kw=ej4.WIDE).assign(prueba="fuera de rango", seed=-1))
    for delay in (0.2, 0.4, 0.6):
        rows.append(ej4.evaluate_general(ej4.TWAPh(0.9), train, inside, R, 0.9, delay=delay)
                    .assign(prueba=f"retraso {delay:.0%}", seed=-1))
    return pd.concat(rows, ignore_index=True)


def choose_and_test(D, A):
    norm = ej4.market_norm(D["train"])
    real = A[~A.placebo]
    tab = real.groupby("w").agg(completas=("completed", "mean"), en_banda=("in_band", "mean"),
                                ahorro=("slip_vs_ref_bps", "mean"))
    ok = tab[(tab.completas == 1.0) & (tab.en_banda >= 0.95)]
    w = float(ok.ahorro.idxmax()) if len(ok) else 0.0
    grid = pr.make_grid(D["test"], n_days=len(D["test"]), seed=0)
    R = ej4.TrackingPriceCierre(price_weight=w)
    rows = []
    for placebo in (False, True):
        for s in SEEDS_A:
            a = ej4.load_agent_a(w, placebo, s, D["test"], norm)
            rows.append(ej4.evaluate_general(a, D["test"], grid, R, 0.9, env_cls=ej4.MarketEnv,
                                             env_kw=dict(norm=norm, placebo=placebo))
                        .assign(politica="DDQN placebo" if placebo else "DDQN mercado", seed=s))
    for name, pol in (("regla spread", ej4.SpreadRule()), ("TWAP 0,9·N", ej4.TWAPh(0.9))):
        rows.append(ej4.evaluate_general(pol, D["test"], grid, R, 0.9, env_cls=ej4.MarketEnv, env_kw=dict(norm=norm))
                    .assign(politica=name, seed=-1))
    return w, tab, pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("A", (w, False, s)) for w in W_REAL for s in SEEDS_A]
    jobs += [("A", (w, True, s)) for w in W_PLACEBO for s in SEEDS_A]
    jobs += [("B", (0.9, 1.0, 0.9, s)) for s in (3, 4)]
    jobs += [("banda", (s,)) for s in SEEDS5]
    ej4.run_jobs(jobs, workers=4)
    print(f"entrenamientos: {time.time() - t0:.0f} s", flush=True)
    D = dt.load("REP")

    b5 = ej4.run_all_b([(0.9, 1.0, 0.9)], seeds=SEEDS5)
    pr.save_df(b5, "ej4_ideaB_5semillas")
    band = pd.concat([pd.read_csv(pr.RESULTS / ej4.RUNS_BAND / f"s{s}_eval.csv").assign(seed=s) for s in SEEDS5])
    pr.save_df(band, "ej4_banda")
    rob = robustness(D)
    pr.save_df(rob, "ej4_robustez")
    A = pd.concat([pd.read_csv(pr.RESULTS / ej4.RUNS_A / f"{ej4.name_a(w, p, s)}_eval.csv").assign(w=w, placebo=p, seed=s)
                   for w, p in [(w, False) for w in W_REAL] + [(w, True) for w in W_PLACEBO] for s in SEEDS_A])
    pr.save_df(A, "ej4_ideaA_val")
    w, tab, test = choose_and_test(D, A)
    pr.save_df(test, "ej4_ideaA_test")
    pr.save_json({"w_elegido": w, "regla": "100 % completas, banda ≥ 95 %, mayor ahorro medio en validación",
                  "validacion": tab.reset_index().to_dict("records")}, "ej4_ideaA")

    S = lambda df, by: df.groupby(by).apply(ej4.summary, include_groups=False)  # noqa: E731
    print("B 5 semillas\n", S(b5, "seed").round(3).to_string())
    print("Banda\n", S(band, "seed").round(3).to_string())
    print("Robustez\n", S(rob, ["prueba", "seed"]).round(3).to_string())
    print("A validación\n", A.groupby(["w", "placebo"]).agg(completas=("completed", "mean"), en_banda=("in_band", "mean"),
          t_fin=("t_fin", "mean"), ahorro=("slip_vs_ref_bps", "mean"), ahorro_std=("slip_vs_ref_bps", "std"),
          err=("max_err_h", "mean")).round(3).to_string())
    print("A test (w =", w, ")\n", test.groupby("politica").agg(completas=("completed", "mean"), en_banda=("in_band", "mean"),
          ahorro=("slip_vs_ref_bps", "mean"), ahorro_std=("slip_vs_ref_bps", "std")).round(3).to_string())
    print(f"total: {time.time() - t0:.0f} s")
