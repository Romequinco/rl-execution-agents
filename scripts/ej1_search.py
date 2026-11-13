"""Busqueda ampliada del ej. 1, sin modificar reward, estado ni algoritmo.

19 configuraciones de un solo factor (2000 episodios, semilla 0); dos alternativas
finalistas y la base se confirman con 4000 episodios y semillas 0, 1, 2. La seleccion
usa exclusivamente train. Los procesos de entrenamiento usan un hilo de torch.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from common import data as dt, exploration as ex, protocol as pr, rewards as rw
from common.agents import DDQNAgent, TWAPAgent
from common.env import BestExecutionEnv

OUT = ROOT / "results" / "ej1_search"
BASE = dict(gamma=0.99, alpha=0.001, epsilon=1.0, batch_size=64,
            buffer_size=50000, hidden_neurons=64, force_first_buy=True,
            learn_after=64, update_target_after=640)
CASES = {
    "base": {},
    "gamma_0.90": {"gamma": 0.90},
    "gamma_0.95": {"gamma": 0.95},
    "gamma_0.98": {"gamma": 0.98},
    "gamma_0.995": {"gamma": 0.995},
    "gamma_0.999": {"gamma": 0.999},
    "alpha_1e-4": {"alpha": 0.0001},
    "alpha_3e-4": {"alpha": 0.0003},
    "alpha_3e-3": {"alpha": 0.003},
    "buffer_10000": {"buffer_size": 10000},
    "buffer_100000": {"buffer_size": 100000},
    "batch_32": {"batch_size": 32},
    "batch_128": {"batch_size": 128},
    "hidden_32": {"hidden_neurons": 32},
    "hidden_128": {"hidden_neurons": 128},
    "learn_16": {"learn_after": 16},
    "learn_128": {"learn_after": 128},
    "target_320": {"update_target_after": 320},
    "target_1280": {"update_target_after": 1280},
}
MIN_EPSILON = 0.02
DECAY = 0.00049  # misma pauta en ambas etapas; minimo al episodio 2000
_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = dt.load("REP")["train"]
    return _DATA


def grids(extended):
    primary = pr.make_grid(data(), n_days=3, seed=0)
    if not extended:
        return [("primary", primary)]
    return [("primary", primary),
            ("additional", pr.make_grid(data(), n_days=9, seed=17))]


def evaluate(agent, extended):
    frames = []
    for name, grid in grids(extended):
        ev = pr.evaluate(agent, data(), grid, reward=rw.Default(), with_twap=False)
        ev.insert(0, "grid", name)
        np.testing.assert_allclose(ev["return"], ev.n_buys * ev.slippage_bps / 1e4,
                                   atol=1e-12, rtol=1e-9)
        frames.append(ev)
    return pd.concat(frames, ignore_index=True)


def metrics(ev):
    complete = ev[ev.completed]
    return dict(n_eval=len(ev), completed=int(ev.completed.sum()),
                completion_rate=float(ev.completed.mean()),
                pending_fraction=float((ev.pending / ev.n).mean()),
                pending_mean=float(ev.pending.mean()),
                slippage_bps=float(complete.slippage_bps.mean()) if len(complete) else None,
                t_fin=float(ev.t_fin.mean()), max_err=float(ev.max_err.mean()))


def run_job(stage, case, seed, fingerprint):
    torch.set_num_threads(1)
    episodes = 2000 if stage == "screen" else 4000
    cfg = BASE | CASES[case]
    folder = OUT / stage / f"{case}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    signature = dict(stage=stage, case=case, seed=seed, episodes=episodes,
                     config=cfg, fingerprint=fingerprint)
    manifest = folder / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise RuntimeError(f"Cache incompatible: {folder}")
        assert all((folder / f).exists() for f in ("train.csv", "eval.csv", "model.pt"))
        return saved["summary"]
    started = time.perf_counter()
    pr.set_seeds(seed)
    env = BestExecutionEnv(data(), seed=seed, reward=rw.Default(), obs_mode="TV")
    agent_cfg = {k: v for k, v in cfg.items() if k not in ("learn_after", "update_target_after")}
    agent = DDQNAgent(env, **agent_cfg, seed=seed)
    fill_steps = pr.fill_buffer(agent, env, n_episodes=25)
    ep, _ = pr.train(agent, env, episodes, ex.LinearDecay(1.0, MIN_EPSILON, DECAY),
                     learn_after=cfg["learn_after"],
                     update_target_after=cfg["update_target_after"], verbose=False)
    assert len(ep) == episodes
    assert np.isfinite(ep[["reward", "disc_reward", "mse"]].to_numpy()).all()
    ev = evaluate(agent, stage == "confirm")
    summary = dict(stage=stage, case=case, seed=seed, episodes=episodes,
                   **cfg, fill_steps=fill_steps, seconds=time.perf_counter() - started,
                   **metrics(ev))
    ep.to_csv(folder / "train.csv", index=False)
    ev.to_csv(folder / "eval.csv", index=False)
    agent.save(folder / "model.pt", search_signature=signature)
    manifest.write_text(json.dumps(dict(signature=signature, summary=summary), indent=2),
                        encoding="utf-8")
    return summary


def run_stage(pool, stage, jobs, fingerprint):
    futures = [pool.submit(run_job, stage, case, seed, fingerprint) for case, seed in jobs]
    results = []
    for future in as_completed(futures):
        row = future.result()
        results.append(row)
        pd.DataFrame(results).to_csv(OUT / f"{stage}_summary.csv", index=False)
        print(f"{stage} {len(results)}/{len(jobs)}: {row['case']} seed={row['seed']} "
              f"complete={row['completed']}/{row['n_eval']} "
              f"pending={row['pending_mean']:.2f} slip={row['slippage_bps']} "
              f"({row['seconds']:.0f}s)", flush=True)
    return pd.DataFrame(results)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "notebooks" / "common" / f"{n}.py"
             for n in ("env", "rewards", "agents", "protocol", "exploration", "data")]
    paths += [Path(__file__), dt.DATA_DIR / dt.ASSETS["REP"]]
    protocol = dict(base=BASE, cases=CASES, screen_episodes=2000, screen_seeds=[0],
                    confirm_episodes=4000, confirm_seeds=[0, 1, 2], finalists=2,
                    fill_episodes=25, epsilon_min=MIN_EPSILON, epsilon_decay=DECAY,
                    primary_grid=grids(False), confirm_grids=grids(True),
                    reward="Default, terminal=0", obs_mode="TV", torch_threads=1,
                    workers=3, python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, pandas=pd.__version__,
                    screen_ranking="completion desc, pending_fraction asc, completed slippage desc; ties prefer base",
                    final_rule="All 324 orders completed; mean slippage gain >= 0.25 bps over base and gain > 0 in at least 2/3 paired seeds; otherwise retain base. Rank eligible alternatives by mean slippage.",
                    sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    serialized = json.dumps(protocol, indent=2, default=str)
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
    (OUT / "protocol.json").write_text(serialized, encoding="utf-8")
    with ProcessPoolExecutor(max_workers=3) as pool:
        screen = run_stage(pool, "screen", [(name, 0) for name in CASES], fingerprint)
        ranked = screen.sort_values(["completion_rate", "pending_fraction", "slippage_bps", "case"],
                                    ascending=[False, True, False, True], na_position="last")
        ranked.to_csv(OUT / "screen_ranking.csv", index=False)
        finalists = ranked.loc[ranked.case != "base", "case"].head(2).tolist()
        (OUT / "finalists.json").write_text(json.dumps(finalists, indent=2), encoding="utf-8")
        print(f"Finalistas: {finalists}; referencia: base", flush=True)
        confirm = run_stage(pool, "confirm", [(c, s) for c in ["base", *finalists]
                                                for s in (0, 1, 2)], fingerprint)
    baseline = confirm[confirm.case == "base"].set_index("seed")
    comparison = []
    for name, group in confirm.groupby("case", sort=False):
        paired = group.set_index("seed").slippage_bps - baseline.slippage_bps
        row = dict(case=name, completed=int(group.completed.sum()), n_eval=int(group.n_eval.sum()),
                   min_seed_completion=float(group.completion_rate.min()),
                   pending_fraction=float(group.pending_fraction.mean()),
                   slippage_bps=float(group.slippage_bps.mean()),
                   slippage_seed_std=float(group.slippage_bps.std()),
                   gain_over_base_bps=float(paired.mean()),
                   improved_seeds=int((paired > 1e-9).sum()),
                   t_fin=float(group.t_fin.mean()))
        row["eligible"] = (row["completed"] == row["n_eval"] and
                           row["gain_over_base_bps"] >= 0.25 and row["improved_seeds"] >= 2)
        comparison.append(row)
    comp = pd.DataFrame(comparison)
    comp.to_csv(OUT / "comparison.csv", index=False)
    eligible = comp[comp.eligible].sort_values("slippage_bps", ascending=False)
    selected = "base" if eligible.empty else eligible.iloc[0].case
    decision = dict(selected_case=selected, selected_seed=0, parameters=BASE | CASES[selected],
                    episodes=4000, comparison=comparison, protocol_sha256=fingerprint,
                    selection_split="train only", test_used_for_selection=False,
                    limitations="One-factor search, single-seed screening and three-seed confirmation; not a global optimum or statistical significance test.")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    torch.set_num_threads(1)
    twap = evaluate(TWAPAgent(), True)
    twap.to_csv(OUT / "twap_train.csv", index=False)
    print(comp.to_string(index=False), flush=True)
    print(f"SELECCION: {selected}", flush=True)


if __name__ == "__main__":
    main()
