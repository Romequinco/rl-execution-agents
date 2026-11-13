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
    epsilon, gamma, _trainable = 0.0, 1.0, False

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


class Policy:
    """Política fija s -> acción con la interfaz de protocol.run_episode."""
    epsilon, gamma, _trainable = 0.0, 1.0, False

    def __init__(self, fn, env=None):
        self.fn, self.env = fn, env

    def act(self, s):
        return self.fn(s)

    def set_trainable(self, train=False):
        pass


def reference_policies(horizon=0.9):
    """Políticas fijas para calibrar los términos de la recompensa de la idea B."""
    tw = TWAPh(horizon)
    return {
        f"TWAP {horizon:g}·N (objetivo)": tw.act,
        "TWAP 1·N (ej. 2)": TWAPh(1.0).act,
        f"TWAP {horizon:g}·N que deja el último 5 %": lambda s: int(tw.act(s) and s[1] > 0.05 + 1e-9),
        f"TWAP 0,8·N (acaba pronto)": TWAPh(0.8).act,
        "todo al principio": lambda s: 1,
    }


def calibration(data, grid, gammas=(0.9, 0.995), ks=(1.0, 10.0, 50.0), horizon=0.9, clip=20.0):
    """Tamaño de cada término de la recompensa por episodio, con políticas fijas en la rejilla.

    seguimiento = suma de recompensas de paso + cierre (TrackingCierre con k = 0);
    terminal(k) = -k · pendiente / n, que se cobra una sola vez al final. Se da también su peso
    descontado visto desde el 90 % del episodio, γ^(0,1·N), que es donde se decide el final."""
    rows = []
    for pname, fn in reference_policies(horizon).items():
        for g in gammas:
            env = BestExecutionEnv(data, reward=make_reward(horizon, 0.0, g, clip), seed=0)
            pol = Policy(fn, env)
            for cfg in grid:
                s = env.reset(**{k: cfg[k] for k in ("day", "start_pos", "n_bins", "vol_care")})
                rs = []
                while True:
                    s, r, done, _ = env.step(pol.act(s))
                    rs.append(r)
                    if done:
                        break
                rs = np.array(rs)
                N, n = env.episode_bins, env.vol_care
                pend = (n - env.exec_vol) / n
                row = {"política": pname, "gamma": g, "seguimiento": rs.sum(),
                       "seguimiento_desc": float((g ** np.arange(len(rs)) * rs).sum()),
                       "pendiente_frac": pend, "t_fin": len(env.vol_hist) / N,
                       "peso_final_desde_0.9N": g ** (0.1 * N)}
                for k in ks:
                    row[f"terminal_k{k:g}"] = -k * pend
                rows.append(row)
    return pd.DataFrame(rows).groupby(["política", "gamma"], sort=False).mean()


# ---------------------------------------------------------------------- idea C: parada, checkpoints y reanudación
def greedy_metric(agent, data, grid, reward, horizon):
    """Error máx. medio frente al plan de horizon·N (y completitud) en la rejilla, sin explorar."""
    env = BestExecutionEnv(data, reward=reward, seed=0)
    old_env, old_train = agent.env, agent._trainable
    agent.env = env
    agent.set_trainable(False)
    rows = []
    try:
        for cfg in grid:
            pr.run_episode(env, agent, cfg)
            rows.append({**metrics_h(env, horizon), "completed": env.exec_vol >= env.vol_care,
                         "t_fin": len(env.vol_hist) / env.episode_bins})
    finally:
        agent.env, agent._trainable = old_env, old_train
    df = pd.DataFrame(rows)
    return float(df.max_err_h.mean()), df


class Trainer:
    """Bucle de protocol.train con el estado completo del entrenamiento en un solo objeto, para
    que un checkpoint permita reanudar exactamente donde se paró (idea C, D-4.2).

    Frente a protocol.train (que no se toca, por los hashes de los ej. 1-3) corrige:
    - best / since_best / racha de objetivo viven en el estado: al reanudar no se sobrescribe
      ``*_best.pt`` con un modelo peor ni se reinicia la paciencia;
    - learn_counter se conserva, así que las fases de learn y update_target no se desplazan;
    - el checkpoint guarda también buffer, generadores (agente, entorno, torch) y el plan de ε;
    - el objetivo exige ``target_hits`` evaluaciones seguidas, no una sola;
    - ``*_best.pt`` se guarda con cualquier mejora (min_delta solo cuenta para la paciencia).

    Reglas de parada, la primera que se cumpla: (a) objetivo ``metric ≤ target`` durante
    ``target_hits`` evaluaciones seguidas; (b) ``patience`` evaluaciones sin mejorar en ``min_delta``;
    (c) tope ``max_episodes``.
    """

    def __init__(self, agent, env, exploration, data, grid, reward, horizon, learn_after=8,
                 update_target_after=1000, eval_every=25, target=None, target_hits=2, patience=None,
                 min_delta=0.05, max_episodes=600, checkpoint=None):
        self.agent, self.env, self.exploration = agent, env, exploration
        self.data, self.grid, self.reward, self.horizon = data, grid, reward, horizon
        self.learn_after, self.update_target_after, self.eval_every = learn_after, update_target_after, eval_every
        self.target, self.target_hits, self.patience, self.min_delta = target, target_hits, patience, min_delta
        self.max_episodes, self.checkpoint = max_episodes, checkpoint
        self.episode, self.learn_counter, self.mse = 0, 0, np.nan
        self.best, self.best_kept, self.since_best, self.hits = np.inf, np.inf, 0, 0
        self.stop_reason = None
        self.ep_rows, self.ev_rows = [], []

    # -- estado completo ------------------------------------------------------------
    _STATE = ("episode", "learn_counter", "mse", "best", "best_kept", "since_best", "hits", "stop_reason",
              "ep_rows", "ev_rows", "exploration")

    def save(self, path, **extra):
        import copy
        import torch
        a = self.agent
        state = {k: copy.deepcopy(getattr(self, k)) for k in self._STATE}
        state.update(buffer=list(a.buffer), agent_rng=a.rng.bit_generator.state,
                     env_rng=self.env.rng.bit_generator.state, torch_rng=torch.get_rng_state())
        a.save(path, trainer=state, **extra)

    def load(self, path):
        import torch
        ckpt = self.agent.load(path)
        st = ckpt["trainer"]
        for k in self._STATE:
            setattr(self, k, st[k])
        self.agent.buffer.clear()
        self.agent.buffer.extend(st["buffer"])
        self.agent.rng.bit_generator.state = st["agent_rng"]
        self.env.rng.bit_generator.state = st["env_rng"]
        torch.set_rng_state(st["torch_rng"])
        return ckpt

    # -- bucle ----------------------------------------------------------------------
    def _evaluate(self):
        metric, df = greedy_metric(self.agent, self.data, self.grid, self.reward, self.horizon)
        self.exploration.on_eval(self.agent, metric)
        self.agent.set_trainable(True)
        improved = metric < self.best_kept                      # cualquier mejora -> *_best.pt
        if metric < self.best - self.min_delta:                 # mejora suficiente -> reinicia paciencia
            self.best, self.since_best = metric, 0
        else:
            self.since_best += 1
        self.hits = self.hits + 1 if self.target is not None and metric <= self.target else 0
        row = {"episode": self.episode, "max_err_h": metric, "completed": df.completed.mean(),
               "t_fin": df.t_fin.mean(), "epsilon": self.agent.epsilon, "saved_best": improved}
        if improved:
            self.best_kept = metric
        self.ev_rows.append(row)
        if self.target is not None and self.hits >= self.target_hits:
            self.stop_reason = f"objetivo: {self.target_hits} evaluaciones seguidas con error ≤ {self.target}"
        elif self.patience and self.since_best >= self.patience:
            self.stop_reason = f"estancamiento: {self.patience} evaluaciones sin mejorar {self.min_delta}"
        elif self.episode >= self.max_episodes:
            self.stop_reason = f"tope de {self.max_episodes} episodios"
        if self.checkpoint:
            if improved:
                self.save(f"{self.checkpoint}_best.pt", episode=self.episode, metric=metric)
            self.save(f"{self.checkpoint}_last.pt", episode=self.episode, metric=metric)

    def run(self, n_episodes=None):
        """Entrena hasta una regla de parada o, si se da, durante n_episodes más (simula una interrupción)."""
        a, env = self.agent, self.env
        a.set_trainable(True)
        stop_at = self.episode + n_episodes if n_episodes else np.inf
        while self.stop_reason is None and self.episode < stop_at:
            s = env.reset()
            step, cum = 0, 0.0
            while True:
                act = a.act(s)
                s1, r, done, _ = env.step(act)
                a.experience(s, act, r, s1, done)
                self.learn_counter += 1
                cum += r
                s, step = s1, step + 1
                if not self.learn_counter % self.learn_after:
                    self.mse = a.learn()
                if not self.learn_counter % self.update_target_after:
                    a.update_target()
                if done:
                    break
            self.exploration.on_episode_end(a, self.episode)
            self.ep_rows.append({"episode": self.episode, "steps": step, "reward": cum, "mse": self.mse,
                                 "epsilon": a.epsilon, "completed": env.exec_vol >= env.vol_care})
            self.episode += 1
            if not self.episode % self.eval_every:
                self._evaluate()
        return pd.DataFrame(self.ep_rows), pd.DataFrame(self.ev_rows)


def make_trainer(horizon=0.9, k=0.0, gamma=0.9, seed=0, asset="REP", checkpoint=None, **kw):
    """Agente + entorno + ε por mejora (variante d del ej. 3: E0 fijo = error de «no comprar», escala log)
    con la configuración final del ej. 2 y la recompensa de la idea B."""
    import torch
    torch.set_num_threads(1)
    train = dt.load(asset)["train"]
    grid = pr.make_grid(train, n_days=3, seed=0)
    reward = make_reward(horizon, k, gamma, BASE["clip"])
    pr.set_seeds(seed)
    env = BestExecutionEnv(train, reward=reward, seed=seed)
    agent = DDQNAgent(env, seed=seed, epsilon=1.0, gamma=gamma, alpha=BASE["alpha"], buffer_size=BASE["buffer_size"],
                      batch_size=BASE["batch_size"], hidden_neurons=BASE["hidden_neurons"])
    pr.fill_buffer(agent, env, 25)
    e_star = greedy_metric(TWAPh(horizon), train, grid, reward, horizon)[0]
    e0 = float(np.mean([c["vol_care"] for c in grid]))
    sched = ex.ImprovementEpsilon(target_error=e_star, eps_max=1.0, eps_min=0.05, margin=0.1, patience=2,
                                  base_error=e0, log_scale=True)
    sched.init(agent)
    tr = Trainer(agent, env, sched, train, grid, reward, horizon, learn_after=BASE["learn_after"],
                 update_target_after=BASE["update_target_after"], checkpoint=checkpoint, **kw)
    tr.e_star, tr.e0 = e_star, e0
    return tr


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
