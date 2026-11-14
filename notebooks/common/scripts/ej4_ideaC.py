"""Idea C del ej. 4: parada por aprendizaje, checkpoints y reanudación (D-4.2).

Sobre la configuración elegida en la idea B y con el ε por mejora del ej. 3 (variante d):
1. Cinco semillas con parada: objetivo (error ≤ TARGET en 2 evaluaciones seguidas), estancamiento
   (PATIENCE evaluaciones sin mejorar 0,05) o tope de MAX_EP episodios. Se guardan *_best.pt y *_last.pt.
2. Mejor frente a último: evaluación completa de los dos checkpoints de cada semilla.
3. Demostración de recuperación: la semilla 0 se interrumpe en el episodio CUT (a mitad, antes de
   la parada); un agente nuevo (otra semilla) carga *_last.pt, el del episodio 50, y continúa; se
   comprueba que la curva y los pesos coinciden con el entrenamiento sin interrumpir.
4. Sin parada: las mismas semillas durante FIXED episodios, para comparar mejor y último cuando el
   entrenamiento sigue después de alcanzar el objetivo.
Reproducir desde la raíz: python notebooks/common/scripts/ej4_ideaC.py --horizon 0.9 --k 1 --gamma 0.9
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "notebooks"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import ej4, protocol as pr  # noqa: E402

TARGET, HITS, PATIENCE, MAX_EP, CUT, FIXED = 1.5, 2, 6, 600, 60, 300
SEEDS = (0, 1, 2, 3, 4)
CK = pr.MODELS / "ej4_ideaC"


def one(args):
    horizon, k, gamma, seed = args
    kw = dict(horizon=horizon, k=k, gamma=gamma, target=TARGET, target_hits=HITS, patience=PATIENCE,
              max_episodes=MAX_EP)
    t0 = time.time()
    tr = ej4.make_trainer(seed=seed, checkpoint=f"{CK}_s{seed}", **kw)
    ep, ev = tr.run()
    out = {"seed": seed, "stop": tr.stop_reason, "episodes": tr.episode, "seconds": time.time() - t0,
           "e_star": tr.e_star, "ep": ep.assign(seed=seed), "ev": ev.assign(seed=seed)}
    # mejor frente a último: evaluación completa (slippage, banda, pendiente) de cada checkpoint
    final = []
    for which in ("best", "last"):
        t = ej4.make_trainer(seed=seed, **kw)
        info = t.agent.load(f"{CK}_s{seed}_{which}.pt")
        res = ej4.evaluate(t.agent, t.data, t.grid, t.reward, horizon)
        final.append(res.assign(seed=seed, checkpoint=which, ckpt_episode=info["episode"]))
    # sin parada: FIXED episodios, mismo bucle y checkpoints propios
    nk = {**kw, "target": None, "patience": None, "max_episodes": FIXED}
    fx = ej4.make_trainer(seed=seed, checkpoint=f"{CK}_fijo_s{seed}", **nk)
    _, ev_fx = fx.run()
    out["ev_fixed"] = ev_fx.assign(seed=seed)
    for which in ("best", "last"):
        t = ej4.make_trainer(seed=seed, **nk)
        info = t.agent.load(f"{CK}_fijo_s{seed}_{which}.pt")
        res = ej4.evaluate(t.agent, t.data, t.grid, t.reward, horizon)
        final.append(res.assign(seed=seed, checkpoint=f"sin parada · {which}", ckpt_episode=info["episode"]))
    for which in ("best", "last"):
        Path(f"{CK}_fijo_s{seed}_{which}.pt").unlink(missing_ok=True)
    out["final"] = pd.concat(final)
    if seed == 0:  # demostración de interrupción y reanudación
        b = ej4.make_trainer(seed=0, checkpoint=f"{CK}_demo", **kw)
        b.run(CUT)
        assert b.stop_reason is None and b.episode == CUT, "la interrupción debe ocurrir antes de la parada"
        c = ej4.make_trainer(seed=7, checkpoint=f"{CK}_demo", **kw)
        ck_ep = c.load(f"{CK}_demo_last.pt")["episode"]
        _, ev_c = c.run()
        same_w = all(torch.equal(v, c.agent.primary_model.state_dict()[n])
                     for n, v in tr.agent.primary_model.state_dict().items())
        out["demo"] = {"cut_episode": CUT, "checkpoint_episode": int(ck_ep),
                       "resumed_seed": 7, "curves_equal": bool(ev.equals(ev_c)),
                       "weights_equal": bool(same_w), "stop_resumed": c.stop_reason, "episodes_resumed": c.episode}
        for f in (f"{CK}_demo_last.pt", f"{CK}_demo_best.pt"):
            Path(f).unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=0.9)
    ap.add_argument("--k", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    a = ap.parse_args()
    with ProcessPoolExecutor(4) as pool:
        outs = list(pool.map(one, [(a.horizon, a.k, a.gamma, s) for s in SEEDS]))
    pr.save_df(pd.concat([o["ep"] for o in outs]), "ej4_ideaC_train")
    pr.save_df(pd.concat([o["ev"] for o in outs]), "ej4_ideaC_evals")
    pr.save_df(pd.concat([o["final"] for o in outs]), "ej4_ideaC_final")
    pr.save_df(pd.concat([o["ev_fixed"] for o in outs]), "ej4_ideaC_evals_fijo")
    summary = {"config": vars(a), "fixed_episodes": FIXED, "target": TARGET, "target_hits": HITS, "patience": PATIENCE, "max_episodes": MAX_EP,
               "e_star": outs[0]["e_star"],
               "runs": [{k: o[k] for k in ("seed", "stop", "episodes", "seconds")} for o in outs],
               "demo": outs[0]["demo"]}
    pr.save_json(summary, "ej4_ideaC")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    fin = pd.concat([o["final"] for o in outs])
    print(fin.groupby(["seed", "checkpoint"]).apply(ej4.summary, include_groups=False).round(3).to_string())
