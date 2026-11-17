"""Reentrena el DDQN del Ej. 1 igual que `notebooks/execution_lab.ipynb` (≈2 min) y guarda en
`results/presentacion/cache/` lo que necesitan las figuras de la presentación. No toca el resto de `results/`.

Uso: python results/presentacion/ej1_cache.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
from common import data as dt, exploration as ex, protocol as pr, rewards as rw  # noqa: E402
from common.agents import DDQNAgent, TWAPAgent  # noqa: E402
from common.env import BestExecutionEnv  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"


def main():
    torch.set_num_threads(1)
    data = dt.load("REP"); train = data["train"]
    grid = pr.make_grid(train, n_days=3, seed=0)
    # Mismos valores que la celda de parámetros del notebook de entrega
    epsilon, min_epsilon, gamma, alpha = 1.0, 0.02, 0.99, 1e-3
    buffer_size, batch_size, hidden, nepisodes, seed = 50_000, 64, 64, 4000, 0
    decay = (epsilon - min_epsilon) / (0.5 * nepisodes)
    pr.set_seeds(seed)
    env = BestExecutionEnv(train, look_back=60, reward=rw.Default(), seed=seed, obs_mode="TV")
    agent = DDQNAgent(env, gamma=gamma, epsilon=epsilon, alpha=alpha, batch_size=batch_size,
                      buffer_size=buffer_size, hidden_neurons=hidden, seed=seed, force_first_buy=True)
    pr.fill_buffer(agent, env, n_episodes=25)
    ckpt = Path(tempfile.mkdtemp()) / "ej1_ddqn"
    ep, ev = pr.train(agent, env, nepisodes, ex.LinearDecay(epsilon, min_epsilon, decay),
                      learn_after=batch_size, update_target_after=10 * batch_size, log_every=500,
                      eval_data=data["val"], eval_grid=pr.make_grid(data["val"], n_days=3, seed=0),
                      eval_every=250, eval_reward=rw.Default(), checkpoint=ckpt,
                      checkpoint_metric="execution", verbose=False)
    best = agent.load(f"{ckpt}_best.pt"); agent.set_trainable(False)
    res = pr.evaluate(agent, train, grid, reward=rw.Default(), obs_mode="TV")
    twap = pr.evaluate(TWAPAgent(), train, grid, reward=rw.Default(), obs_mode="TV", with_twap=False)
    test = pr.evaluate(agent, data["test"], pr.make_grid(data["test"], n_days=3, seed=0),
                       reward=rw.Default(), obs_mode="TV")
    trajs = []
    for cfg in grid:
        e = BestExecutionEnv(train, reward=rw.Default(), seed=0, obs_mode="TV"); agent.env = e
        pr.run_episode(e, agent, cfg)
        trajs.append((pr.trajectory(e) / e.vol_care).tolist())
    # Episodio de ejemplo (el mismo que enseña el notebook): agente, TWAP y volumen del mercado
    e = BestExecutionEnv(train, reward=rw.Default(), seed=1, obs_mode="TV"); agent.env = e
    pr.run_episode(e, agent, grid[4])
    t = BestExecutionEnv(train, reward=rw.Default(), seed=1, obs_mode="TV")
    pr.run_episode(t, TWAPAgent(), grid[4])
    _, market, mid = e.stats_df()
    out = {
        "best_episode": int(best["episode"]),
        "kpi": {"completas": int(res.completed.sum()), "total": len(res),
                "completas_test": int(test.completed.sum()),
                "t_fin": float(res.t_fin.mean()), "t_fin_twap": float(twap.t_fin.mean()),
                "max_err": float(res.max_err.mean()), "max_err_twap": float(twap.max_err.mean()),
                "slip": float(res.slippage_bps.mean()), "slip_twap": float(twap.slippage_bps.mean())},
        "trajs": trajs,
        "ejemplo": {"N": int(e.episode_bins), "n": int(e.vol_care),
                    "agente": (pr.trajectory(e) / e.vol_care).tolist(),
                    "twap": (pr.trajectory(t) / t.vol_care).tolist(),
                    "vol_mercado": np.asarray(market["vol"], dtype=float).tolist(),
                    "mid": np.asarray(mid, dtype=float).tolist()},
    }
    CACHE.mkdir(exist_ok=True)
    (CACHE / "ej1.json").write_text(json.dumps(out), encoding="utf-8")
    ev.to_csv(CACHE / "ej1_validacion.csv", index=False)
    ep.to_csv(CACHE / "ej1_entrenamiento.csv", index=False)
    print(out["best_episode"], out["kpi"]); print(list(ev.columns)); print(ev.to_string())


if __name__ == "__main__":
    main()
