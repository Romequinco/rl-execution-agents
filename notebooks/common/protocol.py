"""Protocolo común: rejilla fija de evaluación, métricas, llenado del buffer,
bucle de entrenamiento (el del notebook original, con ganchos) y guardado de resultados.

Métricas por episodio (docs/teoria/10_hoja_de_ruta_practica.md §4.3), todas sobre el
horizonte completo N aunque la orden se acabe antes:
- completed / pending: si se ejecutó toda la orden y cuántos títulos quedaron.
- t_fin: paso en que terminó, en tanto por uno de N.
- max_err / mean_err: |e_t − t·n/N| en títulos (el TWAP determinista da ≤ 1).
- n_buys, gap_mean, gap_std: compras y separación entre ellas (en pasos).
- slippage_bps: (VWAP mercado − VWAP algo) / VWAP mercado · 1e4; positivo = compra
  mejor que el mercado. slippage_vs_twap_bps: lo mismo frente al TWAP determinista.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .agents import TWAPAgent
from .env import BestExecutionEnv

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("RL_RESULTS", ROOT / "results"))  # RL_RESULTS: pruebas sin tocar results/
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------- rejilla
def make_grid(data: dict, Ns=(400, 500, 600), rhos=(0.075, 0.10, 0.125), n_days=2, seed=0, look_back=60):
    """Episodios fijos de evaluación: N × ρ × días, con hora de inicio al azar (semilla fija).
    Demuestra el requisito del ej. 2 (distintos días, horas, duraciones y tamaños)."""
    rng = np.random.default_rng(seed)
    days = list(data.keys())
    chosen = [days[i] for i in rng.choice(len(days), size=min(n_days, len(days)), replace=False)]
    grid = []
    for day in chosen:
        for N in Ns:
            start = int(rng.integers(look_back, len(data[day]) - N + 1))
            for rho in rhos:
                grid.append({"day": day, "start_pos": start, "n_bins": N, "vol_care": int(rho * N), "rho": rho})
    return grid


# ---------------------------------------------------------------------- episodios
def run_episode(env, agent, cfg=None):
    """Ejecuta un episodio completo con la política actual del agente (sin aprender)."""
    s = env.reset(**{k: cfg[k] for k in ("day", "start_pos", "n_bins", "vol_care")}) if cfg else env.reset()
    ret = 0.0
    while True:
        a = agent.act(s)
        s, r, done, _ = env.step(a)
        ret += r
        if done:
            return ret


def trajectory(env) -> np.ndarray:
    """e_t para t = 1..N (rellena con el valor final si la orden acabó antes)."""
    e = np.cumsum(env.vol_hist)
    if len(e) < env.episode_bins:
        e = np.append(e, np.full(env.episode_bins - len(e), e[-1] if len(e) else 0))
    return e


def episode_metrics(env, ret=np.nan) -> dict:
    N, n = env.episode_bins, env.vol_care
    e = trajectory(env)
    plan = np.minimum(n, np.arange(1, N + 1) * n / N)
    err = np.abs(e - plan)
    buys = np.flatnonzero(np.asarray(env.vol_hist) > 0)
    gaps = np.diff(buys)
    gap_mean, gap_std = (float(gaps.mean()), float(gaps.std())) if len(gaps) else (np.nan, np.nan)
    algo = env._compute_algo_vwap()
    return {
        "N": N, "n": n, "return": ret,
        "completed": bool(env.exec_vol >= n), "pending": int(n - env.exec_vol),
        "t_fin": len(env.vol_hist) / N,
        "max_err": float(err.max()), "mean_err": float(err.mean()),
        "n_buys": int(len(buys)), "gap_mean": gap_mean, "gap_std": gap_std,
        "algo_vwap": algo, "mkt_vwap": env.episode_vwap,
        "slippage_bps": (env.episode_vwap - algo) / env.episode_vwap * 1e4 if np.isfinite(algo) else np.nan,
    }


def evaluate(agent, data, grid, reward=None, obs_mode="TV", with_twap=True) -> pd.DataFrame:
    """Evaluación greedy sobre la rejilla. Usa un entorno propio y reasigna agent.env
    durante la evaluación: DDQNAgent.act consulta env.state_pos (errata E5)."""
    env = BestExecutionEnv(data, reward=reward, seed=0, obs_mode=obs_mode)
    twap_env = BestExecutionEnv(data, reward=reward, seed=0, obs_mode=obs_mode)
    twap = TWAPAgent(twap_env)
    old_env, old_train = getattr(agent, "env", None), getattr(agent, "_trainable", False)
    agent.env = env
    agent.set_trainable(False)
    rows = []
    try:
        for i, cfg in enumerate(grid):
            ret = run_episode(env, agent, cfg)
            m = {"ep": i, "day": str(cfg["day"]), "rho": cfg["rho"], **episode_metrics(env, ret)}
            if with_twap:
                run_episode(twap_env, twap, cfg)
                tw = episode_metrics(twap_env)
                m["slippage_vs_twap_bps"] = (tw["algo_vwap"] - m["algo_vwap"]) / tw["algo_vwap"] * 1e4 \
                    if np.isfinite(m["algo_vwap"]) else np.nan
            rows.append(m)
    finally:
        agent.env = old_env
        agent.set_trainable(old_train)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- entrenamiento
def fill_buffer(agent, env, n_episodes=25):
    """Llena parcialmente el buffer con episodios aleatorios (p = ρ) antes de aprender.
    El original llena ``buffer_size`` pasos enteros; en clase se recomendó 20-30 episodios."""
    old_eps, old_train = agent.epsilon, agent._trainable
    agent.epsilon, agent._trainable = 1.0, True
    steps = 0
    for _ in range(n_episodes):
        s = env.reset()
        while True:
            a = agent.act(s)
            s1, r, done, _ = env.step(a)
            agent.experience(s, a, r, s1, done)
            s, steps = s1, steps + 1
            if done:
                break
    agent.epsilon, agent._trainable = old_eps, old_train
    return steps


def _checkpoint_score(ev, metric):
    """Puntuación a maximizar: completitud, menos pendientes y precio, por ese orden.

    No se omiten órdenes sin compras: un slippage no finito es el peor desempate
    posible. El error TWAP sigue disponible para los otros ejercicios.
    """
    if metric == "max_err":
        return (-float(ev.max_err.mean()),)
    if metric == "execution":
        slippage = ev.slippage_bps.to_numpy(dtype=float)
        price = float(slippage.mean()) if np.isfinite(slippage).all() else -np.inf
        return (float(ev.completed.mean()), -float(ev.pending.mean()), price)
    raise ValueError(f"Unknown checkpoint metric: {metric}")


def train(agent, env, nepisodes, exploration, learn_after=None, update_target_after=None,
          eval_data=None, eval_grid=None, eval_every=25, eval_reward=None,
          checkpoint=None, stop_patience=None, stop_min_delta=0.05, stop_target=None,
          start_episode=0, log_every=25, verbose=True,
          checkpoint_metric="max_err", checkpoint_config=None):
    """Bucle del notebook original (LAB celda 18) con ganchos para los ejercicios 3 y 4.

    - exploration: LinearDecay o ImprovementEpsilon (on_episode_end / on_eval).
    - eval_*: cada ``eval_every`` episodios, evaluación greedy en la rejilla fija; la
      métrica de mejora es el max_err medio.
    - checkpoint: ruta base; guarda el mejor modelo (``*_best.pt``) y el último (``*_last.pt``).
      También evalúa el episodio final si no coincide con eval_every.
      ``checkpoint_metric``: max_err (defecto) o execution (más órdenes completas,
      menos pendientes y mayor slippage, en ese orden). No cambia el reward.
      ``checkpoint_config`` añade la configuración a ambos archivos.
      Exploración adaptativa y parada siguen utilizando max_err.
    - stop_patience: parada si la métrica no mejora ``stop_min_delta`` en ese número de
      evaluaciones; stop_target: parada si la métrica baja de ese valor.
    Devuelve (DataFrame por episodio, DataFrame por evaluación).
    """
    learn_after = learn_after or agent.batch_size
    update_target_after = update_target_after or 10 * learn_after
    if checkpoint_metric not in ("max_err", "execution"):
        raise ValueError(f"Unknown checkpoint metric: {checkpoint_metric}")
    if eval_grid is not None and (not eval_grid or eval_data is None or eval_every <= 0):
        raise ValueError("Evaluation needs data, a nonempty grid and eval_every > 0")
    if checkpoint and eval_grid is None:
        raise ValueError("Checkpoint selection needs an evaluation grid")
    agent.set_trainable(True)
    learn_counter, mse = 0, np.nan
    ep_rows, ev_rows = [], []
    best, since_best = np.inf, 0
    best_score = None
    t0 = time.time()
    for episode in range(start_episode, start_episode + nepisodes):
        s = env.reset()
        step, cum_reward, dis_cum_reward = 0, 0.0, 0.0
        while True:
            a = agent.act(s)
            s1, r, done, _ = env.step(a)
            agent.experience(s, a, r, s1, done)
            learn_counter += 1
            cum_reward += r
            dis_cum_reward += agent.gamma ** step * r
            s = s1
            step += 1
            if not learn_counter % learn_after:
                mse = agent.learn()
            if not learn_counter % update_target_after:
                agent.update_target()
            if done:
                exploration.on_episode_end(agent, episode)
                ep_rows.append({"episode": episode, "steps": step, "reward": cum_reward,
                                "disc_reward": dis_cum_reward, "mse": mse, "epsilon": agent.epsilon,
                                "completed": env.exec_vol >= env.vol_care, "N": env.episode_bins, "n": env.vol_care})
                break
        if verbose and log_every and not (episode + 1) % log_every:
            last = pd.DataFrame(ep_rows[-log_every:])
            print(f"Episode {episode + 1}: steps {last.steps.mean():.0f}, rew {last.reward.mean():.3f}, "
                  f"mse {mse:.4f}, eps {agent.epsilon:.2f}, {time.time() - t0:.0f}s")
        if eval_grid is not None and (not (episode + 1) % eval_every or
                                      (checkpoint and episode + 1 == start_episode + nepisodes)):
            ev = evaluate(agent, eval_data, eval_grid, reward=eval_reward, obs_mode=env.obs_mode, with_twap=False)
            metric = float(ev.max_err.mean())
            score = _checkpoint_score(ev, checkpoint_metric)
            # El criterio por defecto conserva stop_min_delta para los ejercicios existentes.
            is_best = (metric < best - stop_min_delta if checkpoint_metric == "max_err"
                       else best_score is None or score > best_score)
            if is_best:
                best_score = score
            exploration.on_eval(agent, metric)
            agent.set_trainable(True)
            ev_rows.append({"episode": episode + 1, "max_err": metric, "mean_err": ev.mean_err.mean(),
                            "completed": ev.completed.mean(), "pending": ev.pending.mean(),
                            "slippage_bps": ev.slippage_bps.mean(), "return": ev["return"].mean(),
                            "epsilon": agent.epsilon, "is_best": is_best})
            if checkpoint:
                metadata = dict(episode=episode + 1, metric=metric, checkpoint_metric=checkpoint_metric,
                                selection_score=score, validation=ev_rows[-1].copy())
                if checkpoint_config is not None:
                    metadata["config"] = checkpoint_config
                agent.save(f"{checkpoint}_last.pt", **metadata)
                if is_best:
                    agent.save(f"{checkpoint}_best.pt", **metadata)
            if verbose:
                print(f"Eval {episode + 1}: completas {int(ev.completed.sum())}/{len(ev)}, "
                      f"pendientes {ev.pending.mean():.2f}, slippage {ev.slippage_bps.mean():.3f} pb"
                      f"{' [mejor modelo]' if is_best else ''}")
            if metric < best - stop_min_delta:
                best, since_best = metric, 0
            else:
                since_best += 1
            if stop_target is not None and metric <= stop_target:
                if verbose:
                    print(f"Parada: métrica {metric:.2f} ≤ objetivo {stop_target} en el episodio {episode + 1}")
                break
            if stop_patience and since_best >= stop_patience:
                if verbose:
                    print(f"Parada: {stop_patience} evaluaciones sin mejorar en el episodio {episode + 1}")
                break
    return pd.DataFrame(ep_rows), pd.DataFrame(ev_rows)


# ---------------------------------------------------------------------- resultados
def save_df(df: pd.DataFrame, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_df(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv")


def save_json(obj: dict, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def summarize(ev: pd.DataFrame) -> pd.Series:
    """Media y dispersión de las métricas clave de una evaluación."""
    cols = ["completed", "pending", "t_fin", "max_err", "mean_err", "n_buys", "slippage_bps"]
    if "slippage_vs_twap_bps" in ev:
        cols.append("slippage_vs_twap_bps")
    out = {}
    for c in cols:
        out[f"{c}_mean"] = ev[c].mean()
        out[f"{c}_std"] = ev[c].std()
    return pd.Series(out)
