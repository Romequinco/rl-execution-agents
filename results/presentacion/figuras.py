"""Figuras de la presentación, en `results/presentacion/figs/`, con el estilo oscuro de las diapositivas
(fondo azul petróleo, Bahnschrift, ámbar para nuestro agente y coral para el del LAB).

Cada figura se crea con el tamaño físico en mm de su hueco en la diapositiva (16:9, 254 mm de ancho).
Lee `results/` y `results/presentacion/cache/` (lo genera `ej1_cache.py`); no entrena ni escribe en `results/`.

Uso: python results/presentacion/figuras.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
from common import data as dt, protocol as pr, rewards as rw  # noqa: E402
from common.agents import DDQNAgent  # noqa: E402
from common.env import BestExecutionEnv  # noqa: E402

DIR = Path(__file__).resolve().parent
FIGS, CACHE = DIR / "figs", DIR / "cache"
BG, FG, MUTED, OURS, LAB, OLD, BAND = "#10273A", "#EAF0F4", "#9FB3C2", "#F5B301", "#FF7A6B", "#5B768A", "#3FB68B"
MM = 1 / 25.4

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220, "font.family": "Bahnschrift", "font.size": 11,
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": FG,
    "axes.edgecolor": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": MUTED, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})
PCT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f} %")


def num(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", transparent=True)   # el fondo lo pone la diapositiva
    plt.close(fig)
    print("  ", name)


def new(w_mm, h_mm, *args, **kw):
    return plt.subplots(*args, figsize=(w_mm * MM, h_mm * MM), constrained_layout=True, **kw)


def logplain(ax, ticks, axis="y"):
    """Eje log con marcas escritas como números normales."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    a.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}".replace(".", ",") for t in ticks]))
    a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plan(ax):
    ax.plot([0, 1], [0, 1], color=FG, ls=(0, (5, 4)), lw=1.3)


def curvas(ax, trajs, color, lw=1.7, alpha=0.8):
    for t in trajs:
        t = np.asarray(t); ax.plot(np.arange(1, len(t) + 1) / len(t), t, color=color, lw=lw, alpha=alpha)


def ejes_orden(ax, ylabel=True, cuartos=True):
    """Ejes comunes de las figuras «parte comprada frente a tiempo»."""
    ticks = [0, .25, .5, .75, 1] if cuartos else [0, .5, 1]
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="tiempo transcurrido (% del plazo)")
    ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    if ylabel:
        ax.set_ylabel("parte de la orden ya comprada")


# ----------------------------------------------------------------------------- datos comunes
TRAIN = dt.load("REP")["train"]
GRID = pr.make_grid(TRAIN, n_days=3, seed=0)
E_TWAP = pr.load_df("00_twap_grid").max_err.mean()
EJ1 = json.loads((CACHE / "ej1.json").read_text(encoding="utf-8"))


def ej2_agent_envs():
    rew = rw.TrackingCierre(gamma=0.9, clip=20.0)
    agent = DDQNAgent(BestExecutionEnv(TRAIN, reward=rew, seed=0), gamma=0.9, alpha=1e-3,
                      buffer_size=10_000, hidden_neurons=32)
    agent.load(pr.MODELS / "ej2_ddqn.pt"); agent.set_trainable(False)
    envs = []
    for cfg in GRID:
        e = BestExecutionEnv(TRAIN, reward=rew, seed=0); agent.env = e
        pr.run_episode(e, agent, cfg); envs.append(e)
    return agent, envs


# ----------------------------------------------------------------------------- portada, Ej. 1 y Ej. 2
class _Guion:
    """Política que repite unas compras ya decididas (los pasos en que compró el agente del Ej. 1)."""

    def __init__(self, pasos):
        self.pasos, self.k = {int(x) for x in pasos}, 0

    def act(self, s):
        self.k += 1
        return int(self.k in self.pasos)


def _compras(tr):
    return np.flatnonzero(np.diff(np.r_[0, np.asarray(tr, dtype=float)]) > 1e-9) + 1


def _dif(slip):
    """Slippage en palabras: positivo es pagar menos que el precio medio del mercado."""
    return f"{num(abs(slip), 1)} pb más {'barato' if slip > 0 else 'caro'}"


def portada(envs):
    """Una orden: precio de Repsol, precio medio que paga cada agente y cuándo compra cada uno."""
    ex = EJ1["ejemplo"]; N, n = ex["N"], ex["n"]
    c_lab = _compras(ex["agente"])
    e_lab = BestExecutionEnv(TRAIN, reward=rw.Default(), seed=1, obs_mode="TV")
    pr.run_episode(e_lab, _Guion(c_lab), GRID[4])        # mismas compras que el agente del Ej. 1
    assert np.allclose(pr.trajectory(e_lab) / n, ex["agente"])
    e_our = envs[4]; assert e_our.episode_bins == N
    m_lab, m_our = pr.episode_metrics(e_lab), pr.episode_metrics(e_our)
    c_our = _compras(pr.trajectory(e_our))
    print("   portada: slippage LAB %.2f pb, nuestro %.2f pb" % (m_lab["slippage_bps"], m_our["slippage_bps"]))
    mid = np.asarray(ex["mid"], dtype=float)[:N]; x = np.arange(1, len(mid) + 1)
    fig, (a, b) = new(228, 80, 2, 1, sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    a.plot(x, mid, color=MUTED, lw=1.4)
    a.text(N, mid[-60:].max() + (mid.max() - mid.min()) * 0.04, "precio de Repsol", color=MUTED, ha="right", va="bottom")
    # precio medio acumulado hasta cada paso: el del mercado (VWAP) y el que lleva pagado cada agente
    my_lab, mkt, _ = e_lab.stats_df(); my_our, _, _ = e_our.stats_df()
    curvas_vwap = [(np.asarray(mkt["vwap"], dtype=float)[:N], FG, (0, (5, 4)), "precio medio del mercado (VWAP)", False),
                   (np.asarray(my_lab["vwap"], dtype=float)[:N], LAB, "-",
                    f"precio medio pagado con la recompensa del LAB: {_dif(m_lab['slippage_bps'])}", False),
                   (np.asarray(my_our["vwap"], dtype=float)[:N], OURS, "-",
                    f"precio medio pagado con la nuestra: {_dif(m_our['slippage_bps'])}", True)]
    # los rótulos van juntos abajo a la izquierda, donde no hay curvas, en el color de cada línea
    for i, (y, col, ls, txt, _) in enumerate(curvas_vwap):
        a.plot(x[:len(y)], y, color=col, ls=ls, lw=1.6)
        a.text(0.004, 0.34 - 0.14 * i, ("- -  " if col == FG else "") + txt, color=col, transform=a.transAxes, va="center",
               fontweight="normal" if col == FG else "bold")
    a.set_yticks([]); a.grid(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_visible(False)
    a.tick_params(axis="x", length=0)
    b.eventplot([c_our, c_lab], colors=[OURS, LAB], lineoffsets=[0, 1.7], linelengths=0.62, linewidths=1.6)
    b.text(c_lab.max() + 8, 1.7, "con la recompensa del LAB: lo compra todo al principio", color=LAB, va="center", fontweight="bold")
    b.text(0, 0.62, "con la nuestra: una compra cada diez pasos", color=OURS, va="center", fontweight="bold")
    b.set_yticks([]); b.grid(False); b.spines["left"].set_visible(False)
    b.set(xlim=(0, N), ylim=(-0.5, 2.25))
    b.set_xticks([0, 100, 200, 300, 400, 500])
    b.set_xlabel(f"cada raya es la compra de un título: una orden de {n} en {N} pasos de 5 segundos")
    save(fig, "00_portada")


def ej1():
    fig, ax = new(150, 88)
    plan(ax); curvas(ax, EJ1["trajs"], LAB); ejes_orden(ax)
    ax.text(0.25, 0.92, "el agente: toda la orden comprada\nal 21 % del plazo", color=LAB, va="top", fontweight="bold")
    ax.text(0.60, 0.50, "el plan de un TWAP:\ncomprar a ritmo constante", va="top")
    save(fig, "01_ej1")


def ej2(agent, e2):
    fig, ax = new(150, 88)
    plan(ax); curvas(ax, EJ1["trajs"], OLD, lw=1.1, alpha=0.7); curvas(ax, e2, OURS, lw=1.9); ejes_orden(ax)
    ax.text(0.25, 0.92, "antes, con la recompensa del LAB", color=MUTED, va="top")
    ax.text(0.56, 0.46, "ahora: las 27 órdenes\nvan sobre el plan", color=OURS, va="top", fontweight="bold")
    save(fig, "02_ej2")

    # anexo A: política aprendida
    fig, ax = new(80, 74)
    T, V = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    A = agent.getQs(np.c_[T.ravel(), V.ravel()], agent.primary_model).argmax(axis=1).reshape(T.shape)
    ax.imshow(A, origin="lower", extent=[0, 1, 0, 1], cmap=ListedColormap(["#1C3D55", OURS]), vmin=0, vmax=1, aspect="auto")
    plan(ax); ax.grid(False)
    ax.text(0.30, 0.76, "comprar", color=BG, fontweight="bold", ha="center", fontsize=12)
    ax.text(0.70, 0.22, "esperar", color=FG, fontweight="bold", ha="center", fontsize=12)
    ax.set(xlabel="tiempo que queda", ylabel="volumen que queda")
    ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    save(fig, "91_politica")


# ----------------------------------------------------------------------------- Ej. 3
def _g(df, sc, s, sd):
    return df[(df.config == sc) & (df.schedule == s) & (df.seed == sd)].sort_values("episode")


def ej3():
    ev, tr = pr.load_df("ej3_p5_evals"), pr.load_df("ej3_p5_train")
    evd, evl = _g(ev, "inestable", "mejora_d", 0), _g(ev, "inestable", "lineal", 0)
    trd, trl = _g(tr, "inestable", "mejora_d", 0), _g(tr, "inestable", "lineal", 0)
    fig, (a, b) = new(150, 88, 2, 1, sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a.plot(evl.episode, evl.max_err, "o-", color=MUTED, lw=1.8, ms=3.5)
    a.plot(evd.episode, evd.max_err, "o-", color=OURS, lw=2.4, ms=4)
    a.axhline(E_TWAP, color=FG, ls=(0, (5, 4)), lw=1.2)
    a.set(ylabel="error en el examen\n(títulos)", yscale="log", xlim=(0, 395), ylim=(0.7, 70))
    logplain(a, [1, 10, 50])
    a.text(308, evl.max_err.iloc[-1], "ε del LAB:\nbaja con los episodios", color=MUTED, va="center")
    a.text(308, evd.max_err.iloc[-1] * 1.7, "ε según la mejora", color=OURS, va="center", fontweight="bold")
    a.text(4, 0.98, "error de un TWAP", va="bottom", fontsize=10)
    b.plot(trl.episode + 1, trl.epsilon, color=MUTED, lw=1.8)
    b.plot(trd.episode + 1, trd.epsilon, color=OURS, lw=2.4)
    up = evd.epsilon.diff().values > 1e-12
    b.scatter(evd.episode[up], evd.epsilon[up], color=OURS, edgecolor=FG, s=50, zorder=4)
    for x0, y0 in zip(evd.episode[up], evd.epsilon[up]):
        b.text(x0 + 8, y0 + 0.22, "el examen empeora\ny ε vuelve a subir", color=OURS, va="center", fontsize=10)
    b.set(ylabel="ε: fracción de\nacciones al azar", xlabel="episodio de entrenamiento", ylim=(0, 1.08))
    b.set_yticks([0, 0.5, 1]); b.set_yticklabels(["0", "0,5", "1"]); b.set_xticks([0, 100, 200, 300])
    save(fig, "03_ej3")

    # anexo B: las 3 semillas en los dos escenarios
    fig, axes = new(150, 92, 1, 2, sharey=True)
    for ax, sc, tit in [(axes[0], "inestable", "Entrenamiento inestable"), (axes[1], "final", "Entrenamiento estable")]:
        for s, col in [("lineal", MUTED), ("mejora_d", OURS)]:
            m = ev[(ev.config == sc) & (ev.schedule == s)].groupby("episode").max_err
            ax.fill_between(m.mean().index, m.min(), m.max(), color=col, alpha=0.22, lw=0)
            ax.plot(m.mean().index, m.mean(), "o-", color=col, lw=2.1, ms=3)
        ax.axhline(E_TWAP, color=FG, ls=(0, (5, 4)), lw=1.1)
        ax.set(title=tit, yscale="log", xlabel="episodio de entrenamiento"); logplain(ax, [1, 10, 50])
        ax.set_xticks([0, 100, 200, 300])
    axes[0].set_ylabel("error en el examen (títulos)")
    axes[1].text(300, 30, "ε del LAB", color=MUTED, ha="right")
    axes[1].text(300, 16, "ε según la mejora", color=OURS, fontweight="bold", ha="right")
    save(fig, "92_semillas")


# ----------------------------------------------------------------------------- Ej. 4
def ej4():
    fig, (a1, a2, a3) = new(228, 67, 1, 3)
    B = pr.load_df("ej4_ideaB")
    bins = np.linspace(0.75, 1.0, 51)
    a1.axvspan(0.8, 0.95, color=BAND, alpha=0.2, lw=0)
    a1.hist(B[B.horizon == 1].t_fin, bins=bins, color=OLD)
    a1.hist(B[(B.horizon != 1) & (B.k == 1) & (B.gamma == 0.9)].t_fin, bins=bins, color=OURS)
    a1.set(title="Acabar al 90 % del plazo", xlabel="momento en que acaba la orden (% del plazo)",
           ylabel="número de órdenes", xlim=(0.75, 1.0))
    a1.xaxis.set_major_formatter(PCT); a1.set_xticks([0.8, 0.9, 1.0])
    top = a1.get_ylim()[1] * 1.4; a1.set_ylim(0, top)
    a1.text(0.805, top * 0.97, "banda pedida: 80-95 %", color=BAND, va="top", fontsize=10)
    a1.text(0.874, top * 0.62, "plan\nal 90 %", color=OURS, ha="right", va="top", fontweight="bold")
    a1.text(0.972, top * 0.62, "ejercicio 2", color=MUTED, ha="right", va="top")

    CE, CF = pr.load_df("ej4_ideaC_evals"), pr.load_df("ej4_ideaC_evals_fijo")
    CS = json.loads((pr.RESULTS / "ej4_ideaC.json").read_text(encoding="utf-8"))
    for _, g in CF.groupby("seed"):
        a2.plot(g.episode, g.max_err_h, "-", color=OLD, lw=1)
    for _, g in CE.groupby("seed"):
        a2.plot(g.episode, g.max_err_h, "-", color=OURS, lw=1.8)
        a2.plot(g.episode.iloc[-1], g.max_err_h.iloc[-1], "o", color=OURS, ms=6.5, mec=FG, mew=0.8, zorder=5)
    a2.axhline(CS["target"], color=FG, ls=(0, (5, 4)), lw=1.1)
    a2.text(305, CS["target"] * 1.12, "objetivo: 1,5 títulos", ha="right", fontsize=10)
    a2.annotate("aquí para", xy=(131, 1.02), xytext=(185, 7), color=OURS, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=OURS))
    a2.text(305, 0.78, "sin parar, hasta los 300", ha="right", va="top", fontsize=10, color=MUTED)
    a2.set(title="Parar a tiempo", yscale="log", xlabel="episodio de entrenamiento", ylabel="error (títulos)",
           xlim=(0, 310), ylim=(0.5, 22))
    logplain(a2, [1, 10]); a2.set_xticks([0, 100, 200, 300])

    AT = pr.load_df("ej4_ideaA_test")
    m = AT.groupby("politica").slip_vs_ref_bps.mean()
    pols = [("DDQN mercado", "agente", OURS), ("DDQN placebo", "datos\nbarajados", OLD),
            ("regla spread", "regla\nsimple", MUTED)]
    for i, (p, lab, col) in enumerate(pols):
        a3.bar(i, m[p], color=col, width=0.58)
        a3.text(i, max(m[p], 0) + 0.015, f"{m[p]:+.2f}".replace(".", ","), ha="center", fontweight="bold")
    a3.axhline(0, color=FG, lw=0.8)
    a3.set_xticks(range(3)); a3.set_xticklabels([p[1] for p in pols])
    a3.set(title="Mirar el mercado", ylabel="ahorro frente al TWAP (pb)", ylim=(-0.08, 0.45))
    a3.set_yticks([0, 0.2, 0.4]); a3.set_yticklabels(["0", "0,2", "0,4"])
    save(fig, "04_ej4")


# ----------------------------------------------------------------------------- conclusiones y anexo C
def resumen():
    rows = [("Recompensa del LAB", EJ1["kpi"]["max_err"], LAB, 1),
            ("Comprar al azar", pr.load_df("00_random_grid").max_err.mean(), OLD, 2),
            ("Nuestra recompensa", pr.load_df("ej2_eval_train").max_err.mean(), OURS, 2), ("Un TWAP", E_TWAP, FG, 2)]
    fig, ax = new(150, 90)
    for i, (lab, v, col, d) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.58)
        ax.text(v * 1.12, i, num(v, d), va="center", fontweight="bold", fontsize=13)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=12); ax.invert_yaxis()
    ax.set(xscale="log", xlim=(0.5, 120), xlabel="títulos de separación respecto al plan (escala logarítmica)")
    logplain(ax, [1, 10, 100], "x"); ax.grid(False); ax.tick_params(axis="y", length=0)
    save(fig, "05_resumen")


def pasillo():
    P, F = pr.load_df("ej4_banda"), pr.load_df("ej4_ideaB_5semillas")
    fig, axes = new(150, 92, 1, 2, sharey=True)
    rng = np.random.default_rng(0)
    for ax, df, tit, col in [(axes[0], F, "Plan fijo al 90 %", MUTED), (axes[1], P, "Pasillo 80-95 %", OURS)]:
        ax.axhspan(0.8, 0.95, color=BAND, alpha=0.2, lw=0)
        for s, g in df.groupby("seed"):
            ax.plot(s + 1 + rng.uniform(-0.16, 0.16, len(g)), g.t_fin, "o", color=col, ms=3.6, alpha=0.8, mew=0)
        ax.set(title=tit, xlabel="repetición del entrenamiento", ylim=(0.70, 1.0), xlim=(0.5, 5.5)); ax.set_xticks(range(1, 6))
        ax.yaxis.set_major_formatter(PCT); ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    axes[0].set_ylabel("momento en que acaba cada orden")
    axes[0].text(5.4, 0.945, "banda pedida", color=BAND, ha="right", va="top", fontsize=10)
    save(fig, "93_pasillo")


def main():
    agent, envs = ej2_agent_envs()
    e2 = [pr.trajectory(e) / e.vol_care for e in envs]
    portada(envs); ej1(); ej2(agent, e2); ej3(); ej4(); resumen(); pasillo()


if __name__ == "__main__":
    main()
