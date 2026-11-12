"""Ayudas del ejercicio 3: figuras del Punto 1 y la política «olvidadiza».

Usa el estilo de ``plots.py`` (``save``, rejilla suave, títulos en español) y fija un color
por política para todo el notebook: TWAP azul, aleatoria gris, DDQN naranja y olvidadiza morado.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .agents import TWAPAgent
from .env import BestExecutionEnv
from .plots import ALGO, MKT, save
from .protocol import run_episode, trajectory

COLORS = {"TWAP": MKT, "aleatoria": "#8A8A8A", "DDQN": ALGO, "olvidadiza": "#805AD5",
          "no comprar": "#C53030", "sin entrenar": "#F6AD55"}
EPS = "#38A169"


class Forgetful(TWAPAgent):
    """Regla del TWAP que deja de comprar en el último ``cutoff`` del episodio (T ≤ cutoff).
    Va pegada al plan el 90 % del tiempo y acaba con ~n·cutoff títulos pendientes."""

    def __init__(self, env=None, cutoff=0.1):
        super().__init__(env)
        self.cutoff = cutoff

    def act(self, s) -> int:
        return int(s[1] >= s[0] - 1e-9 and s[0] > self.cutoff)


class NoBuy(TWAPAgent):
    """Política fija que no compra nunca: su error máximo es n (todo pendiente al final)."""

    def act(self, s) -> int:
        return 0


def metrics_table(results: dict) -> pd.DataFrame:
    """Media de las métricas de la rejilla por política (filas) en unidades legibles."""
    rows = {k: {"error máx. (títulos)": r.max_err.mean(), "error medio (títulos)": r.mean_err.mean(),
                "completadas (%)": 100 * r.completed.mean(), "pendientes (títulos)": r.pending.mean(),
                "t_fin (% de N)": 100 * r.t_fin.mean(), "slippage (pb)": r.slippage_bps.mean()}
            for k, r in results.items()}
    return pd.DataFrame(rows).T


def first_below(ev, thr=2.0):
    hit = ev.episode[ev.max_err < thr]
    return int(hit.min()) if len(hit) else None


def plot_train_vs_greedy(ep, ev, twap_err, eps_min_ep, thr=2.0, window=20, save_as=None):
    """Arriba: recompensa de entrenamiento por paso (media móvil) frente a ε.
    Abajo: error máximo de la política greedy en la rejilla, en escala log."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    rps = ep.reward / ep.steps
    a1.plot(ep.episode, rps, color=ALGO, alpha=0.2, lw=0.8)
    a1.plot(ep.episode, rps.rolling(window, min_periods=1).mean(), color=ALGO, lw=2,
            label=f"recompensa de entrenamiento por paso (media móvil {window})")
    a1.set_ylabel("recompensa media por paso"); a1.grid(alpha=0.3)
    b1 = a1.twinx()
    b1.fill_between(ep.episode, ep.epsilon, color=EPS, alpha=0.12, step="post")
    b1.plot(ep.episode, ep.epsilon, color=EPS, lw=1.2, label="ε (exploración)")
    b1.set_ylabel("ε", color=EPS); b1.set_ylim(0, 1.05); b1.tick_params(axis="y", colors=EPS)
    h = a1.get_legend_handles_labels()[0] + b1.get_legend_handles_labels()[0]
    a1.legend(h, [x.get_label() for x in h], loc="lower right", fontsize=9)
    a1.set_title("Lo que se ve durante el entrenamiento: la curva sube mientras baja ε")

    a2.plot(ev.episode, ev.max_err, "o-", color=ALGO, lw=2, label="DDQN greedy (27 episodios de la rejilla)")
    a2.axhline(twap_err, color=COLORS["TWAP"], ls="--", lw=1.3, label=f"TWAP determinista ({twap_err:.2f})".replace(".", ","))
    a2.axhline(thr, color="k", ls=":", lw=1, label=f"umbral {thr:g} títulos")
    a2.set_yscale("log"); a2.set_ylabel("error máx. de seguimiento (títulos, log)")
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    a2.yaxis.set_major_formatter(fmt); a2.yaxis.set_minor_formatter(fmt)
    a2.set_xlabel("episodio de entrenamiento"); a2.grid(alpha=0.3, which="both")
    a2.set_title("Lo que de verdad ha aprendido: evaluación greedy cada 25 episodios")
    ok = first_below(ev, thr)
    for ax in (a1, a2):
        ax.axvline(eps_min_ep, color=EPS, ls="--", lw=1)
        if ok:
            ax.axvline(ok, color=ALGO, ls="--", lw=1)
    ymax = a2.get_ylim()[1]
    a2.text(eps_min_ep - 3, ymax * 0.6, f"ε llega al mínimo\n(episodio {eps_min_ep})", color=EPS, fontsize=9, va="top", ha="right")
    if ok:
        a2.text(ok + 3, ymax * 0.6, f"greedy < {thr:g} títulos\n(episodio {ok})", color=ALGO, fontsize=9, va="top")
    else:
        b = ev.loc[ev.max_err.idxmin()]
        a2.annotate(f"mejor greedy: {b.max_err:.1f} títulos\n(episodio {int(b.episode)}), lejos del umbral".replace(".", ",", 1),
                    xy=(b.episode, b.max_err), xytext=(b.episode - 95, b.max_err * 0.45), color=ALGO, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=ALGO, lw=0.8))
    a2.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); save(fig, save_as)
    return fig


def _box(ax, data, names, log=False):
    rng = np.random.default_rng(0)
    for i, (k, v) in enumerate(zip(names, data)):
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=COLORS[k], alpha=0.15, edgecolor=COLORS[k]),
                   medianprops=dict(color=COLORS[k], lw=2), whiskerprops=dict(color=COLORS[k]),
                   capprops=dict(color=COLORS[k]))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    return rng


def plot_slippage_vs_error(tw, ra, save_as=None):
    """Los 27 episodios por política: slippage (pareado por episodio) y error máximo.
    Marcador hueco = la orden no se completó."""
    names = ["TWAP", "aleatoria"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    _box(a1, [tw.slippage_bps, ra.slippage_bps], names)
    wins = ra.slippage_bps.values > tw.slippage_bps.values
    for y0, y1, w in zip(tw.slippage_bps, ra.slippage_bps, wins):
        a1.plot([0, 1], [y0, y1], color="#38A169" if w else "#C9C9C9", lw=1.1 if w else 0.7, zorder=1)
    for i, (k, d) in enumerate(zip(names, (tw, ra))):
        for x, col in ((a1, "slippage_bps"), (a2, "max_err")):
            done = d.completed.values.astype(bool)
            j = i + np.random.default_rng(1).uniform(-0.06, 0.06, len(d)) * (x is a2)
            x.scatter(j[done], d[col][done], color=COLORS[k], s=22, zorder=3)
            x.scatter(j[~done], d[col][~done], facecolor="white", edgecolor=COLORS[k], s=22, zorder=3)
    a1.axhline(0, color="k", lw=0.8)
    a1.set_ylabel("slippage frente al VWAP de mercado (pb, + = mejor)")
    a1.set_title(f"Slippage: la aleatoria gana en {wins.sum()} de {len(wins)} episodios\n"
                 "(verde = episodio en que la aleatoria tiene mejor slippage)", fontsize=10)
    _box(a2, [tw.max_err, ra.max_err], names, log=True)
    a2.set_ylabel("error máx. de seguimiento (títulos, log)")
    a2.set_title(f"Error máximo: separa las dos políticas sin ambigüedad\n"
                 f"(hueco = orden no completada: {int((~ra.completed.astype(bool)).sum())} de {len(ra)} en la aleatoria)",
                 fontsize=10)
    fig.suptitle("Mismos 27 episodios de la rejilla, dos métricas", fontsize=12)
    fig.tight_layout(); save(fig, save_as)
    return fig, int(wins.sum())


def run_cfg(agent, data, cfg):
    env = BestExecutionEnv(data, seed=0); agent.env = env
    run_episode(env, agent, cfg)
    N, n = env.episode_bins, env.vol_care
    plan = np.minimum(n, np.arange(1, N + 1) * n / N)
    return trajectory(env), plan


def plot_max_vs_mean(agents: dict, data, cfg, table: pd.DataFrame, save_as=None):
    """(1) ejecución frente al plan en un episodio, (2) error en cada paso con su media y su
    máximo, (3) error máximo y medio en la rejilla por política."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    N, n = cfg["n_bins"], cfg["vol_care"]
    t = np.arange(1, N + 1)
    lines = []
    for k, ag in reversed(list(agents.items())):
        e, plan = run_cfg(ag, data, cfg)
        err = np.abs(e - plan)
        ax[0].step(t, e, where="post", color=COLORS[k], lw=1.6, label=k)
        ax[1].plot(t, err, color=COLORS[k], lw=0.9, alpha=0.75)
        ax[1].axhline(err.mean(), color=COLORS[k], ls="--", lw=1.4)
        ax[1].scatter([t[err.argmax()]], [err.max()], color=COLORS[k], s=40, zorder=4)
        lines.append((k, f"{k}: medio {err.mean():.2f} · máx. {err.max():.2f}".replace(".", ",")))
    for i, (k, txt) in enumerate(reversed(lines)):
        ax[1].text(0.03, 0.93 - 0.08 * i, txt, transform=ax[1].transAxes, color=COLORS[k], fontsize=9,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax[0].plot(t, plan, "k--", lw=1, label="plan TWAP (t·n/N)")
    ax[0].axvspan(0.9 * N, N, color="#805AD5", alpha=0.08, label="último 10 % (olvidadiza no compra)")
    ax[0].set_xlabel("paso t (5 s)"); ax[0].set_ylabel("títulos ejecutados e_t")
    ax[0].set_title(f"Ejecución frente al plan · {cfg['day']} · N={N}, n={n}", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=0.3)
    ax[1].axvspan(0.9 * N, N, color="#805AD5", alpha=0.08)
    ax[1].set_xlabel("paso t (5 s)"); ax[1].set_ylabel("|e_t − plan_t| (títulos)")
    ax[1].set_title("Error en cada paso (discontinua = su media)", fontsize=10); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(0, None)
    cols = ["error máx. (títulos)", "error medio (títulos)"]
    x = np.arange(len(cols)); w = 0.8 / len(table)
    for i, (k, row) in enumerate(table[cols].iterrows()):
        b = ax[2].bar(x + (i - (len(table) - 1) / 2) * w, row.values, w, color=COLORS[k], label=k)
        ax[2].bar_label(b, labels=[f"{v:.2f}".replace(".", ",") for v in row.values], fontsize=8)
    ax[2].set_xticks(x); ax[2].set_xticklabels(["error máximo", "error medio"])
    ax[2].set_ylabel("títulos (media de los 27 episodios)")
    ax[2].set_title("En la rejilla: el medio apenas distingue TWAP y olvidadiza", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis="y")
    fig.tight_layout(); save(fig, save_as)
    return fig


# ---------------------------------------------------------------------- Punto 2
def progress(E, E0, Es):
    """p = clip((E0 − E)/(E0 − E*), 0, 1); vale 1 si E0 ≤ E* (no hay camino que recorrer)."""
    E = np.asarray(E, dtype=float)
    return np.ones_like(E) if E0 <= Es else np.clip((E0 - E) / (E0 - Es), 0, 1)


def plot_ladder(ladder: dict, colors: dict, Es, save_as=None):
    """Escalera de referencias: error máximo medio en la rejilla, eje horizontal en log."""
    ladder = dict(sorted(ladder.items(), key=lambda kv: -kv[1]))
    names, vals = list(ladder), np.array(list(ladder.values()))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=[colors[k] for k in names], height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v * 1.06, yi, f"{v:.2f}".replace(".", ","), va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_ylim(-0.8, len(names) - 0.5)
    ax.set_xscale("log"); ax.set_xlim(0.5, vals.max() * 2.2)
    ax.axvline(Es, color=MKT, ls="--", lw=1)
    ax.axvspan(Es, vals.max(), color="#E2E8F0", alpha=0.35, zorder=0)
    ax.text(Es * 1.05, -0.55, "E* (objetivo)", color=MKT, fontsize=9, va="center")
    ax.set_xlabel("error máx. de seguimiento medio en la rejilla (títulos, log; menor es mejor)")
    ax.set_title("Escalera de referencias en los 27 episodios de la rejilla: de no hacer nada al TWAP determinista")
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout(); save(fig, save_as)
    return fig


def plot_progress_e0(ev, e0s: dict, colors: dict, Es, eps_max=1.0, eps_min=0.05, save_as=None):
    """Progreso p y la ε que saldría de cada E0 (ε = εmax − (εmax − εmin)·p_ref, p_ref = mejor p)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), sharex=True)
    for i, (k, e0) in enumerate(e0s.items()):
        ls = "--" if i == 1 else "-"
        p = progress(ev.max_err, e0, Es)
        eps = eps_max - (eps_max - eps_min) * np.maximum.accumulate(p)
        a1.plot(ev.episode, p, marker="o", ls=ls, color=colors[k], lw=1.8, ms=4, label=f"{k} (E0 = {e0:.2f})".replace(".", ","))
        a2.step(ev.episode, eps, where="post", ls=ls, color=colors[k], lw=1.8, label=k)
    a1.set_ylim(-0.05, 1.05); a1.set_ylabel("progreso p"); a1.set_title("Progreso hacia el TWAP según el E0 elegido")
    a1.legend(fontsize=9, loc="center right")
    a2.axhline(eps_min, color="k", ls=":", lw=1); a2.text(ev.episode.iloc[0], eps_min + 0.02, "εmin", fontsize=9)
    a2.set_ylim(0, 1.05); a2.set_ylabel("ε"); a2.legend(fontsize=9, loc="center right"); a2.set_title("ε que resultaría (sin la regla de empeorar, Punto 3)")
    for a in (a1, a2):
        a.set_xlabel("episodio de entrenamiento (evaluación cada 25)"); a.grid(alpha=0.3)
    fig.tight_layout(); save(fig, save_as)
    return fig


# ---------------------------------------------------------------------- Punto 3
class _Dummy:
    epsilon = 1.0


def replay(metrics, target, reset_to_max=False, **kw):
    """Reproduce offline ``ImprovementEpsilon`` sobre una serie de E ya medida. Devuelve su
    ``history`` (metric, progress, ref, epsilon). ``reset_to_max``: variante que, al confirmar
    que empeora, vuelve a ε = εmax (p_ref = 0) en lugar de bajar p_ref al progreso actual."""
    from .exploration import ImprovementEpsilon
    sch, ag = ImprovementEpsilon(target_error=target, **kw), _Dummy()
    for m in metrics:
        before = sch.best_progress
        sch.on_eval(ag, m)
        if reset_to_max and sch.best_progress < before:
            sch.best_progress, ag.epsilon = 0.0, sch.eps_max
            sch.history[-1].update(ref=0.0, epsilon=sch.eps_max)
    h = pd.DataFrame(sch.history)
    h["rise"] = h.epsilon.diff() > 1e-12
    return h


def _p_panel(ax, ep, p, notes=True):
    ax.plot(ep, p, "o-", color=ALGO, lw=1.5, ms=5, label="progreso p de cada evaluación")
    if notes:
        for e, txt, dy in ((200, "caída", -0.0), (225, "pico", 0.0)):
            i = int(np.flatnonzero(ep.values == e)[0])
            ax.annotate(f"{txt}: {p.iloc[i]:.2f}".replace(".", ","), xy=(e, p.iloc[i]),
                        xytext=(e - 60 if txt == "caída" else e - 75, p.iloc[i] + (0.05 if txt == "caída" else 0.08)),
                        color=ALGO, fontsize=9, arrowprops=dict(arrowstyle="->", color=ALGO, lw=0.8))
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("progreso p"); ax.grid(alpha=0.3)


def plot_last_vs_best(ep, p, eps_max=1.0, eps_min=0.05, title="Progreso medido", save_as=None):
    """ε atado al último progreso frente a ε atado al mejor progreso alcanzado."""
    last = eps_max - (eps_max - eps_min) * p
    best = eps_max - (eps_max - eps_min) * np.maximum.accumulate(p)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    _p_panel(a1, ep, p)
    a1.plot(ep, np.maximum.accumulate(p), color="k", lw=1, ls=":", drawstyle="steps-post", label="mejor p hasta ahora (p_ref)")
    a1.legend(fontsize=9, loc="lower right"); a1.set_title(title)
    a2.step(ep, last, where="post", color="#C53030", lw=1.8, label="ε con el último p (da bandazos)")
    a2.step(ep, best, where="post", color=EPS, lw=2.2, label="ε con el mejor p (estable, solo baja)")
    for e in (200, 225):
        i = int(np.flatnonzero(ep.values == e)[0])
        a2.annotate(f"{last.iloc[i]:.2f}".replace(".", ","), xy=(e, last.iloc[i]), xytext=(e + 4, last.iloc[i] + 0.04),
                    color="#C53030", fontsize=9)
    a2.axhline(eps_min, color="k", ls=":", lw=1)
    a2.set_ylim(0, 1.05); a2.set_ylabel("ε"); a2.set_xlabel("episodio de entrenamiento (evaluación cada 25)")
    a2.legend(fontsize=9, loc="upper right"); a2.grid(alpha=0.3)
    a2.set_title("ε resultante: ¿último progreso o mejor progreso?")
    fig.tight_layout(); save(fig, save_as)
    return fig


def plot_rise_rule(ep, variants: dict, colors: dict, save_as=None):
    """Arriba: p y p_ref de la regla elegida. Abajo: ε de cada variante; marca las subidas."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    h0 = next(iter(variants.values()))
    _p_panel(a1, ep, h0.progress, notes=False)
    a1.step(ep, h0.ref, where="post", color="k", lw=1.3, ls="--", label="p_ref de la regla elegida")
    a1.fill_between(ep, h0.ref - 0.1, h0.ref, step="post", color="k", alpha=0.07, label="margen 0,1 bajo p_ref")
    a1.legend(fontsize=9, loc="lower right"); a1.set_title("Cuándo se confirma que empeora: 2 evaluaciones seguidas por debajo del margen")
    for k, h in variants.items():
        a2.step(ep, h.epsilon, where="post", color=colors[k], lw=2.2 if k == next(iter(variants)) else 1.5,
                ls="-" if k == next(iter(variants)) else "--", label=k)
        for e, eps in zip(ep[h.rise.values], h.epsilon[h.rise.values]):
            a2.scatter([e], [eps], color=colors[k], s=60, zorder=4, edgecolor="k")
    a2.axhline(0.05, color="k", ls=":", lw=1)
    a2.set_ylim(0, 1.05); a2.set_ylabel("ε"); a2.set_xlabel("episodio de entrenamiento (evaluación cada 25)")
    a2.legend(fontsize=9, loc="upper center"); a2.grid(alpha=0.3)
    a2.set_title("ε por evaluación (punto = ε sube porque el agente ha empeorado)")
    fig.tight_layout(); save(fig, save_as)
    return fig


def sensitivity(metrics, target, margins=(0.05, 0.1, 0.2), patiences=(1, 2, 3)):
    rows = []
    for m in margins:
        for pt in patiences:
            h = replay(metrics, target, margin=m, patience=pt)
            rows.append({"margen": m, "paciencia": pt, "subidas de ε": int(h.rise.sum()),
                         "episodios de las subidas": ", ".join(str(25 * (i + 1)) for i in np.flatnonzero(h.rise)) or "—",
                         "ε medio": h.epsilon.mean(), "ε final": h.epsilon.iloc[-1]})
    return pd.DataFrame(rows)


def plot_sensitivity(tab, save_as=None):
    """Mapa margen × paciencia: color = subidas de ε; texto = subidas y ε final."""
    piv = tab.pivot(index="margen", columns="paciencia", values="subidas de ε")
    fin = tab.pivot(index="margen", columns="paciencia", values="ε final")
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.imshow(piv.values, cmap="Greens", vmin=0, vmax=max(3, piv.values.max()), aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]} subida(s)\nε final {fin.values[i, j]:.2f}".replace(".", ","),
                    ha="center", va="center", fontsize=9, color="white" if piv.values[i, j] >= 3 else "k")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{m:.2f}".replace(".", ",") for m in piv.index])
    ax.set_xlabel("paciencia (evaluaciones seguidas)"); ax.set_ylabel("margen")
    ax.set_title("Sensibilidad de la regla de subida (serie del escenario B)", fontsize=10)
    fig.tight_layout(); save(fig, save_as)
    return fig


# ---------------------------------------------------------------------- Punto 4
# Escenario A: configuración final del ej. 2 (D-2.2…D-2.7). Escenario B: la de partida del ej. 2
# (Tracking c = 5 sin cierre, buffer 50.000), inestable; sus ejecuciones están cacheadas y no se reentrenan.
SCENARIOS = {
    "final": dict(reward="cierre", clip=20.0, gamma=0.9, alpha=1e-3, buffer_size=10_000, batch_size=64,
                  hidden_neurons=32, learn_after=8, update_target_after=1000, nepisodes=300, frozen=False),
    "inestable": dict(reward="seguimiento", clip=5.0, gamma=0.9, alpha=1e-3, buffer_size=50_000, batch_size=64,
                      hidden_neurons=32, learn_after=8, update_target_after=1000, nepisodes=300, frozen=True),
}
SCHEDULES, SEEDS = ("lineal", "mejora"), (0, 1, 2)
ALL_SCHEDULES = ("lineal", "mejora", "mejora_c", "mejora_d")   # Punto 5: E0 fijo (c) y E0 fijo en escala log (d)
FROZEN_SCHEDULES = ("lineal", "mejora")                       # en escenarios congelados, solo estas vienen de la caché
RUNS_DIR = "ej3_p4_runs"


def make_reward(cfg):
    from . import rewards as rw
    return rw.TrackingCierre(gamma=cfg["gamma"], clip=cfg["clip"]) if cfg["reward"] == "cierre" else rw.Tracking(clip=cfg["clip"])


def _paths(scenario, schedule, seed):
    from . import protocol as pr
    base = pr.RESULTS / RUNS_DIR / f"{scenario}__{schedule}__s{seed}"
    return base.with_name(base.name + "_train.csv"), base.with_name(base.name + "_evals.csv")


def run_one(scenario, schedule, seed, target=0.9, asset="REP"):
    """Un entrenamiento completo, cacheado en results/ej3_p4_runs/<escenario>__<plan>__s<semilla>_*.csv.
    Solo cambia el plan de ε: misma semilla, mismo llenado del buffer, mismos episodios (generador
    propio del entorno, D-1.3), ε inicial = 1 en los dos (``init``), misma rejilla y evaluación cada
    25 episodios también para el lineal."""
    import torch
    from . import data as dt, exploration as ex, protocol as pr
    from .agents import DDQNAgent
    cfg = SCENARIOS[scenario]
    f_tr, f_ev = _paths(scenario, schedule, seed)
    if f_ev.exists():
        return f_ev.name
    if cfg["frozen"] and schedule in FROZEN_SCHEDULES:
        raise FileNotFoundError(f"{f_ev.name}: el escenario '{scenario}' solo se lee de la caché")
    torch.set_num_threads(1)
    f_ev.parent.mkdir(parents=True, exist_ok=True)
    train = dt.load(asset)["train"]
    grid = pr.make_grid(train, n_days=3, seed=0)
    pr.set_seeds(seed)
    env = BestExecutionEnv(train, reward=make_reward(cfg), seed=seed)
    agent = DDQNAgent(env, seed=seed, epsilon=1.0, gamma=cfg["gamma"], alpha=cfg["alpha"],
                      buffer_size=cfg["buffer_size"], batch_size=cfg["batch_size"], hidden_neurons=cfg["hidden_neurons"])
    pr.fill_buffer(agent, env, 25)
    e0 = float(np.mean([c["vol_care"] for c in grid]))       # error de «no comprar» = n medio de la rejilla
    extra = {"mejora": {}, "mejora_c": dict(base_error=e0), "mejora_d": dict(base_error=e0, log_scale=True)}
    sched = (ex.LinearDecay(1.0, 0.05, nepisodes=cfg["nepisodes"]) if schedule == "lineal"
             else ex.ImprovementEpsilon(target_error=target, eps_max=1.0, eps_min=0.05, margin=0.1, patience=2,
                                        **extra[schedule]))
    sched.init(agent)
    ep, ev = pr.train(agent, env, cfg["nepisodes"], sched, cfg["learn_after"], cfg["update_target_after"],
                      eval_data=train, eval_grid=grid, eval_every=25, eval_reward=make_reward(cfg),
                      log_every=0, verbose=False)
    ep[["episode", "steps", "reward", "epsilon", "completed", "N", "n"]].to_csv(f_tr, index=False)
    ev.to_csv(f_ev, index=False)
    return f_ev.name


def _run_args(args):
    return run_one(*args)


def run_all(scenarios=("final", "inestable"), target=0.9, workers=4, schedules=SCHEDULES):
    """Lanza en paralelo (procesos, 1 hilo de PyTorch cada uno) las ejecuciones que falten y devuelve
    (evals, train) de todos los escenarios con columnas escenario/plan/semilla."""
    from concurrent.futures import ProcessPoolExecutor
    todo = [(sc, s, seed, target) for sc in scenarios for s in schedules for seed in SEEDS
            if not _paths(sc, s, seed)[1].exists()]
    if todo:
        with ProcessPoolExecutor(min(workers, len(todo))) as pool:
            list(pool.map(_run_args, todo))
    evs, trs = [], []
    for sc in scenarios:
        for s in schedules:
            for seed in SEEDS:
                tr, ev = load_run(sc, s, seed)
                evs.append(ev.assign(config=sc, schedule=s, seed=seed)); trs.append(tr.assign(config=sc, schedule=s, seed=seed))
    return pd.concat(evs, ignore_index=True), pd.concat(trs, ignore_index=True)


def load_run(scenario, schedule, seed):
    f_tr, f_ev = _paths(scenario, schedule, seed)
    return pd.read_csv(f_tr), pd.read_csv(f_ev)


def run_metrics(ep, ev, thr=2.0):
    e, x = ev.max_err.values, ev.episode.values
    ok = x[e <= thr]
    return {"E0": e[0], "E final": e[-1], "E mejor": e.min(),
            "área E (media)": float(np.trapezoid(e, x) / (x[-1] - x[0])),
            "1.er ep. E ≤ 2": ok[0] if len(ok) else np.nan,
            "ε medio": ep.epsilon.mean(), "subidas de ε": int((np.diff(ev.epsilon.values) > 1e-12).sum())}


SCHED_COLORS = {"lineal": "#718096", "mejora": ALGO}


def plot_comparison(evals, trains, configs, titles=None, save_as=None):
    """Arriba: E (media ± banda min-máx de 3 semillas) por config; abajo: ε de cada semilla."""
    fig, axes = plt.subplots(2, len(configs), figsize=(7.5 * len(configs), 8.5), sharex="col",
                             gridspec_kw={"height_ratios": [1.5, 1]}, squeeze=False)
    for j, cfg in enumerate(configs):
        a1, a2 = axes[0, j], axes[1, j]
        for s, col in SCHED_COLORS.items():
            g = evals[(evals.config == cfg) & (evals.schedule == s)]
            m = g.groupby("episode").max_err
            a1.fill_between(m.mean().index, m.min(), m.max(), color=col, alpha=0.15)
            a1.plot(m.mean().index, m.mean(), "o-", color=col, lw=2.2, ms=4,
                    label=f"ε {'lineal' if s == 'lineal' else 'por mejora'} (media de 3 semillas; banda = mín.-máx.)")
            t = trains[(trains.config == cfg) & (trains.schedule == s)]
            for k, (seed, gs) in enumerate(t.groupby("seed")):
                a2.plot(gs.episode + 1, gs.epsilon, color=col, lw=1.4, alpha=0.85,
                        ls=["-", "--", ":"][k], label=f"{s} · semilla {seed}")
            ge = g.sort_values(["seed", "episode"])
            for seed, gg in ge.groupby("seed"):
                up = gg.epsilon.diff().values > 1e-12
                a2.scatter(gg.episode[up], gg.epsilon[up], color=col, edgecolor="k", s=55, zorder=4)
        a1.axhline(0.9, color=MKT, ls="--", lw=1.2, label="TWAP determinista (0,90)")
        a1.axhline(2, color="k", ls=":", lw=1, label="umbral 2 títulos")
        a1.set_yscale("log"); a1.set_ylabel("error máx. greedy en la rejilla (títulos, log)")
        a1.set_title(titles.get(cfg, cfg) if titles else cfg, fontsize=12); a1.grid(alpha=0.3, which="both")
        a2.axhline(0.05, color="k", ls=":", lw=1)
        a2.set_ylim(0, 1.05); a2.set_ylabel("ε (punto = ε sube)"); a2.set_xlabel("episodio de entrenamiento")
        a2.grid(alpha=0.3); a2.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.suptitle("ε lineal frente a ε por mejora · mismas semillas, episodios y rejilla", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.955), frameon=False)
    save(fig, save_as)
    return fig


# ---------------------------------------------------------------------- Punto 5
SCHED5 = {"lineal": ("ε lineal", "#718096"), "mejora": ("a) E0 = 1.ª evaluación", ALGO),
          "mejora_c": ("c) E0 fijo, escala lineal", "#C53030"), "mejora_d": ("d) E0 fijo, escala log", "#805AD5")}


def plot_scale(e0_seeds, e0_fixed, Es, marks=(7, 2), save_as=None):
    """Progreso p frente a E (eje log) para las opciones a (E0 de cada semilla), c y d."""
    E = np.geomspace(Es, e0_fixed, 400)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, e0 in enumerate(e0_seeds):
        ax.plot(E, progress(E, e0, Es), color=ALGO, lw=1.3, alpha=0.8, ls=["-", "--", ":"][i % 3],
                label=f"a) E0 = 1.ª evaluación, semilla {i} ({e0:.2f})".replace(".", ","))
    pc = progress(E, e0_fixed, Es)
    pd_ = np.clip(np.log(e0_fixed / E) / np.log(e0_fixed / Es), 0, 1)
    ax.plot(E, pc, color=SCHED5["mejora_c"][1], lw=2.2, label=f"c) E0 fijo = {e0_fixed:.2f}, escala lineal".replace(".", ","))
    ax.plot(E, pd_, color=SCHED5["mejora_d"][1], lw=2.2, label="d) E0 fijo, escala log")
    for m in marks:
        ax.axvline(m, color="k", ls=":", lw=1)
        for col, p in ((SCHED5["mejora_c"][1], progress(m, e0_fixed, Es)),
                       (SCHED5["mejora_d"][1], np.log(e0_fixed / m) / np.log(e0_fixed / Es))):
            ax.scatter([m], [p], color=col, s=40, zorder=4, edgecolor="k")
            ax.text(m * 1.06, p - 0.06, f"{float(p):.2f}".replace(".", ","), color=col, fontsize=9)
        ax.text(m, 1.04, f"E = {m}", ha="center", fontsize=9)
    ax.set_xscale("log"); ax.set_xlim(e0_fixed * 1.05, Es * 0.95)       # de izquierda a derecha: mejora
    ax.set_xticks([50, 20, 10, 7, 5, 2, Es])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}".replace(".", ",")))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("error máx. greedy E (títulos, log; hacia la derecha = mejor)"); ax.set_ylabel("progreso p")
    ax.set_ylim(-0.03, 1.1); ax.grid(alpha=0.3, which="both")
    ax.set_title("Progreso según cómo se fija E0 y la escala", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout(); save(fig, save_as)
    return fig


def plot_comparison4(evals, trains, configs, titles, save_as=None):
    """Arriba: E medio de 3 semillas por plan (eje log); abajo: ε medio por episodio."""
    fig, axes = plt.subplots(2, len(configs), figsize=(7.5 * len(configs), 8), sharex="col",
                             gridspec_kw={"height_ratios": [1.5, 1]}, squeeze=False)
    for j, cfg in enumerate(configs):
        a1, a2 = axes[0, j], axes[1, j]
        for s, (lab, col) in SCHED5.items():
            g = evals[(evals.config == cfg) & (evals.schedule == s)].groupby("episode").max_err.mean()
            a1.plot(g.index, g.values, "o-", color=col, lw=2.2 if s != "lineal" else 2.6, ms=4, label=lab)
            t = trains[(trains.config == cfg) & (trains.schedule == s)].groupby("episode").epsilon.mean()
            a2.plot(t.index + 1, t.values, color=col, lw=1.8, label=lab)
        a1.axhline(0.9, color=MKT, ls="--", lw=1.2, label="TWAP determinista (0,90)")
        a1.axhline(2, color="k", ls=":", lw=1, label="umbral 2 títulos")
        a1.set_yscale("log"); a1.set_ylabel("E medio de 3 semillas (títulos, log)")
        a1.set_yticks([0.9, 2, 5, 10, 20, 40])
        a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        a1.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        a1.set_title(titles.get(cfg, cfg), fontsize=12); a1.grid(alpha=0.3, which="both")
        a2.axhline(0.05, color="k", ls=":", lw=1)
        a2.set_ylim(0, 1.05); a2.set_ylabel("ε medio de 3 semillas"); a2.set_xlabel("episodio de entrenamiento")
        a2.grid(alpha=0.3)
    fig.suptitle("Cuatro planes de ε con las mismas semillas, episodios y rejilla", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.955), frameon=False)
    save(fig, save_as)
    return fig
