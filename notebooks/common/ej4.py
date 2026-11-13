"""Utilidades del ej. 4 (ideas B y C).

No modifica protocol.py ni rewards.py: sus hashes forman parte de las auditorías de los
ej. 1-3. Lo que el ej. 4 necesita de más (métricas frente al plan comprimido, la política
TWAP de referencia que acaba en h·N y la caché de entrenamientos) vive aquí.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as dt, exploration as ex, protocol as pr, rewards as rw
from .agents import DDQNAgent
from .env import BestExecutionEnv

# Configuración final del ej. 2 (D-2.2…D-2.7), la misma que usa el ej. 3 como escenario "final"
BASE = dict(clip=20.0, gamma=0.9, alpha=1e-3, buffer_size=10_000, batch_size=64, hidden_neurons=32,
            learn_after=8, update_target_after=1000, nepisodes=300)
BAND = (0.80, 0.95)          # banda de finalización aceptada para «≈ 90 % de la duración»
RUNS_B = "ej4_ideaB_runs"


def make_reward(horizon=0.9, k=0.0, gamma=0.9, clip=20.0):
    """Seguimiento con cierre (ej. 2) sobre el plan que acaba en horizon·N, más -k·pendiente/n al final."""
    return rw.TrackingCierre(gamma=gamma, clip=clip, horizon=horizon, pending_penalty=k)


class TWAPh:
    """TWAP determinista que acaba en h·N: compra si lo ejecutado no supera el plan comprimido.
    Con h = 1 es exactamente TWAPAgent. Sirve de referencia para la idea B."""
    epsilon, gamma = 0.0, 1.0

    def __init__(self, horizon=0.9, env=None):
        self.horizon, self.env = horizon, env

    def act(self, s):
        T, V = s[0], s[1]
        v_plan = 1.0 - min(1.0, (1.0 - T) / self.horizon)     # volumen restante que marca el plan
        return int(V >= v_plan - 1e-9)

    def set_trainable(self, train=False):
        pass


def metrics_h(env, horizon):
    """Métricas extra frente al plan comprimido y frente a la banda de finalización."""
    N, n = env.episode_bins, env.vol_care
    e = pr.trajectory(env)
    plan = np.minimum(n, np.arange(1, N + 1) * n / (horizon * N))
    err = np.abs(e - plan)
    t_fin = len(env.vol_hist) / N
    return {"max_err_h": float(err.max()), "mean_err_h": float(err.mean()),
            "in_band": bool(env.exec_vol >= n and BAND[0] <= t_fin <= BAND[1])}


def evaluate(agent, data, grid, reward, horizon=0.9):
    """protocol.evaluate (greedy, rejilla fija, slippage frente al TWAP de N pasos) más las
    métricas frente al plan de horizon·N. Repite los episodios con el mismo entorno de semilla 0."""
    base = pr.evaluate(agent, data, grid, reward=reward)
    env = BestExecutionEnv(data, reward=reward, seed=0)
    old_env, old_train = getattr(agent, "env", None), getattr(agent, "_trainable", False)
    agent.env = env
    agent.set_trainable(False)
    extra = []
    try:
        for cfg in grid:
            pr.run_episode(env, agent, cfg)
            extra.append(metrics_h(env, horizon))
    finally:
        agent.env = old_env
        agent.set_trainable(old_train)
    return pd.concat([base, pd.DataFrame(extra)], axis=1)


def summary(res: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "completas": res.completed.mean(), "pendientes": res.pending.mean(),
        "t_fin": res.t_fin.mean(), "t_fin_std": res.t_fin.std(), "t_fin_min": res.t_fin.min(),
        "t_fin_max": res.t_fin.max(), "en_banda": res.in_band.mean(),
        "max_err_h": res.max_err_h.mean(), "mean_err_h": res.mean_err_h.mean(),
        "slip_vs_twap_bps": res.slippage_vs_twap_bps.mean()})


# ---------------------------------------------------------------------- idea B: caché y ejecución
def run_name(horizon, k, gamma, seed):
    return f"h{horizon:g}__k{k:g}__g{gamma:g}__s{seed}"


def run_b(horizon, k, gamma, seed, asset="REP", nepisodes=None):
    """Un entrenamiento de la idea B con la config. final del ej. 2 cambiando solo horizon, k y γ.
    Se guarda en results/ej4_ideaB_runs/<nombre>_{train,eval}.csv y <nombre>.pt; si existe, no se repite."""
    import torch
    out = pr.RESULTS / RUNS_B
    name = run_name(horizon, k, gamma, seed)
    f_ev = out / f"{name}_eval.csv"
    if f_ev.exists():
        return name
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**BASE, "gamma": gamma, "nepisodes": nepisodes or BASE["nepisodes"]}
    train = dt.load(asset)["train"]
    grid = pr.make_grid(train, n_days=3, seed=0)
    pr.set_seeds(seed)
    reward = make_reward(horizon, k, gamma, cfg["clip"])
    env = BestExecutionEnv(train, reward=reward, seed=seed)
    agent = DDQNAgent(env, seed=seed, epsilon=1.0, gamma=gamma, alpha=cfg["alpha"], buffer_size=cfg["buffer_size"],
                      batch_size=cfg["batch_size"], hidden_neurons=cfg["hidden_neurons"])
    pr.fill_buffer(agent, env, 25)
    ep, _ = pr.train(agent, env, cfg["nepisodes"], ex.LinearDecay(1.0, 0.05, nepisodes=cfg["nepisodes"]),
                     cfg["learn_after"], cfg["update_target_after"], verbose=False)
    res = evaluate(agent, train, grid, reward, horizon)
    ep.to_csv(out / f"{name}_train.csv", index=False)
    agent.save(out / f"{name}.pt", horizon=horizon, k=k, gamma=gamma, seed=seed)
    res.to_csv(f_ev, index=False)
    return name


def _run_b(args):
    return run_b(*args)


def run_all_b(configs, seeds=(0, 1, 2), workers=4):
    """configs: lista de (horizon, k, gamma). Lanza en paralelo lo que falte y devuelve las evaluaciones."""
    from concurrent.futures import ProcessPoolExecutor
    todo = [(h, k, g, s) for h, k, g in configs for s in seeds
            if not (pr.RESULTS / RUNS_B / f"{run_name(h, k, g, s)}_eval.csv").exists()]
    if todo:
        with ProcessPoolExecutor(min(workers, len(todo))) as pool:
            list(pool.map(_run_b, todo))
    rows = []
    for h, k, g in configs:
        for s in seeds:
            ev = pd.read_csv(pr.RESULTS / RUNS_B / f"{run_name(h, k, g, s)}_eval.csv")
            rows.append(ev.assign(horizon=h, k=k, gamma=g, seed=s))
    return pd.concat(rows, ignore_index=True)


def load_train_b(configs, seeds=(0, 1, 2)):
    rows = []
    for h, k, g in configs:
        for s in seeds:
            tr = pd.read_csv(pr.RESULTS / RUNS_B / f"{run_name(h, k, g, s)}_train.csv")
            rows.append(tr.assign(horizon=h, k=k, gamma=g, seed=s))
    return pd.concat(rows, ignore_index=True)
