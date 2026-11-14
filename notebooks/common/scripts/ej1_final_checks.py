"""Pruebas finales acotadas del ej. 1; ejecutar desde la raíz del repositorio.

Mismo reward del LAB, presupuesto y semillas. Cada alternativa cambia un solo
hiperparámetro. La rejilla de train sirve para diagnóstico, no para generalización.
"""
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "notebooks"))
from common import data as dt, exploration as ex, protocol as pr, rewards as rw
from common.agents import DDQNAgent, TWAPAgent
from common.env import BestExecutionEnv


BASE = dict(gamma=0.99, alpha=1e-3, epsilon=1.0, batch_size=64,
            buffer_size=50_000, hidden_neurons=64, force_first_buy=True)
CASES = {"base": {}, "alpha_1e-4": {"alpha": 1e-4}, "gamma_0.999": {"gamma": 0.999}}
SEEDS = (0, 1, 2)
NEPISODES = 4000
MIN_EPSILON = 0.02
DECAY = (BASE["epsilon"] - MIN_EPSILON) / (0.5 * NEPISODES)
OUT = ROOT / "results" / "ej1_final_checks"


class FixedAction:
    def __init__(self, first_only=False):
        self.env = None
        self.first_only = first_only

    def act(self, s):
        return int(not self.first_only or self.env.state_pos == 0)

    def set_trainable(self, train=False):
        pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    data = dt.load("REP")["train"]
    grid = pr.make_grid(data, n_days=3, seed=0)
    fingerprint_paths = [ROOT / "notebooks" / "common" / f"{name}.py"
                         for name in ("env", "rewards", "agents", "protocol", "exploration", "data")]
    fingerprint_paths += [dt.DATA_DIR / dt.ASSETS["REP"], Path(__file__)]
    metadata = dict(base=BASE, cases=CASES, seeds=SEEDS, nepisodes=NEPISODES,
                    min_epsilon=MIN_EPSILON, epsilon_decay=DECAY, fill_episodes=25,
                    learn_after=64, update_target_after=640, obs_mode="TV",
                    reward="Default: (episode_vwap-price)/episode_vwap if buy; else 0; done=0",
                    grid=grid, torch_threads=1, python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__,
                    sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in fingerprint_paths})
    (OUT / "protocol.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    baselines = []
    for name, policy in [("TWAP", TWAPAgent()), ("BuyAll", FixedAction()),
                         ("FirstBuyOnly", FixedAction(first_only=True))]:
        ev = pr.evaluate(policy, data, grid, reward=rw.Default())
        ev.insert(0, "policy", name)
        baselines.append(ev)
    pd.concat(baselines).to_csv(OUT / "baselines.csv", index=False)

    summaries = []
    for case, changes in CASES.items():
        for seed in SEEDS:
            t0 = time.perf_counter()
            pr.set_seeds(seed)
            env = BestExecutionEnv(data, reward=rw.Default(), seed=seed, obs_mode="TV")
            agent = DDQNAgent(env, **(BASE | changes), seed=seed)
            fill_steps = pr.fill_buffer(agent, env, n_episodes=25)
            exploration = ex.LinearDecay(BASE["epsilon"], MIN_EPSILON, DECAY)
            # Dos tramos conservan exactamente la misma pauta de epsilon. El contador
            # de actualizaciones se reinicia en train: se usa un único entrenamiento
            # de 4000 episodios y se evalúa al final para no cambiar esa pauta.
            ep, _ = pr.train(agent, env, NEPISODES, exploration,
                             learn_after=64, update_target_after=640, verbose=False)
            assert np.isfinite(ep[["reward", "disc_reward", "mse"]].to_numpy()).all()
            ev = pr.evaluate(agent, data, grid, reward=rw.Default())
            assert len(ev) == 27 and np.isfinite(ev["slippage_bps"]).all()
            ep.to_csv(OUT / f"{case}_seed{seed}_train.csv", index=False)
            ev.to_csv(OUT / f"{case}_seed{seed}_eval.csv", index=False)
            row = dict(case=case, seed=seed, **(BASE | changes),
                       episodes=len(ep), fill_steps=fill_steps,
                       seconds=time.perf_counter() - t0,
                       **pr.summarize(ev).to_dict())
            summaries.append(row)
            pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
            print(f"{case} seed={seed}: complete={ev.completed.sum()}/27 "
                  f"pending={ev.pending.mean():.2f}, t_fin={ev.t_fin.mean():.3f}, "
                  f"slippage={ev.slippage_bps.mean():.3f} bps, "
                  f"{row['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
