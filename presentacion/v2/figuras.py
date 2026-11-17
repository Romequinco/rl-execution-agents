"""Figuras de la presentación, en `presentacion/figs/`. Cada figura tiene el tamaño físico de su hueco
en la diapositiva (16:9, 254 mm de ancho), así que los tamaños de letra son los reales.

Lee `results/` y `presentacion/cache/` (lo genera `ej1_cache.py`); no entrena ni escribe en `results/`.

Uso: python presentacion/figuras.py
"""
import json
import locale
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
from common import data as dt, protocol as pr, rewards as rw  # noqa: E402
from common.agents import DDQNAgent  # noqa: E402
from common.env import BestExecutionEnv  # noqa: E402

DIR = Path(__file__).resolve().parent
FIGS, CACHE = DIR / "figs", DIR.parent / "cache"
ALGO, MKT, OK, KO, GREY, INK = "#D9822B", "#2B6CB0", "#2F9E6E", "#C53030", "#718096", "#102a43"
MM = 1 / 25.4

for loc in ("es_ES.UTF-8", "es_ES", "Spanish_Spain"):
    try:
        locale.setlocale(locale.LC_NUMERIC, loc); break
    except locale.Error:
        pass
plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220, "font.size": 9, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 8.2, "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
    "axes.formatter.use_locale": True, "axes.axisbelow": True,
})


def es(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", facecolor="white")
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


def plan(ax, **kw):
    ax.plot([0, 1], [0, 1], "k--", lw=1, **kw)


# ----------------------------------------------------------------------------- datos comunes
TRAIN = dt.load("REP")["train"]
GRID = pr.make_grid(TRAIN, n_days=3, seed=0)
E_TWAP = pr.load_df("00_twap_grid").max_err.mean()
E0 = float(np.mean([c["vol_care"] for c in GRID]))
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


def curvas(ax, trajs, color, **kw):
    for t in trajs:
        t = np.asarray(t); ax.plot(np.arange(1, len(t) + 1) / len(t), t, color=color, lw=0.9, alpha=0.55, **kw)


# ----------------------------------------------------------------------------- portada
def portada(e2_trajs):
    fig, axes = new(140, 81, 1, 2, sharey=True)
    for ax, trajs, col, tit, msg in [(axes[0], EJ1["trajs"], KO, "Recompensa del LAB", "compra todo\nal principio"),
                                     (axes[1], e2_trajs, ALGO, "Nuestra recompensa", "reparte como\nun TWAP")]:
        curvas(ax, trajs, col, zorder=3); plan(ax, label="plan TWAP")
        ax.set_title(tit, fontsize=12, color=col); ax.set_xlabel("tiempo transcurrido (t / N)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
        ax.text(0.95, 0.08, msg, ha="right", transform=ax.transAxes, fontsize=12, color=col, fontweight="bold")
    axes[0].set_ylabel("volumen comprado (e / n)")
    axes[1].legend(loc="upper left", fontsize=9)
    save(fig, "00_portada")


# ----------------------------------------------------------------------------- Ej. 1
def ej1():
    ev = pd.read_csv(CACHE / "ej1_validacion.csv")
    ex = EJ1["ejemplo"]; N = ex["N"]; x = np.arange(1, N + 1) / N
    fig, (a1, a2, a3) = new(230, 67, 1, 3, gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    vol = np.asarray(ex["vol_mercado"])[:N]; vol = np.clip(vol / np.quantile(vol, 0.99), 0, 1)
    a1.bar(np.arange(1, len(vol) + 1) / N, vol * 0.3, width=1 / N, color=MKT, alpha=0.35)
    a1.plot(x, ex["twap"], color=GREY, lw=2)
    a1.plot(x, ex["agente"], color=KO, lw=2.2)
    a1.set(title=f"Un episodio (N = {N}, n = {ex['n']})", xlabel="tiempo transcurrido (t / N)",
           ylabel="volumen comprado (e / n)", xlim=(0, 1), ylim=(0, 1.03))
    a1.text(0.25, 0.93, "DDQN (LAB)", color=KO, fontweight="bold", va="top", fontsize=9.5)
    a1.text(0.70, 0.60, "TWAP", color=GREY, fontweight="bold", va="top", fontsize=9.5)
    a1.text(0.99, 0.33, "volumen del mercado", color=MKT, ha="right", fontsize=8.2)
    curvas(a2, EJ1["trajs"], KO); plan(a2, label="plan TWAP"); a2.legend(loc="lower right")
    a2.set(title="Las 27 órdenes de la rejilla", xlabel="tiempo transcurrido (t / N)", xlim=(0, 1), ylim=(0, 1.03))
    k = EJ1["kpi"]
    a2.annotate(f"acaba al {100 * k['t_fin']:.0f} %\ndel plazo", xy=(k["t_fin"], 1.0), xytext=(0.27, 0.70),
                fontsize=9.5, color=KO, fontweight="bold", arrowprops=dict(arrowstyle="->", color=KO))
    comp = 100 * ev["completed"]
    cols = [OK if e == EJ1["best_episode"] else GREY for e in ev["episode"]]
    a3.bar(ev["episode"], comp, width=170, color=cols)
    cero = ev["episode"][comp == 0]
    a3.plot(cero, np.full(len(cero), 3), "x", color=KO, ms=6, mew=2)
    a3.text(120, 113, "✕ 0 %: no compra nada", color=KO, fontsize=8.2, va="center")
    a3.set(title="Validación cada 250 episodios", xlabel="episodio de entrenamiento",
           ylabel="% de órdenes completas", ylim=(0, 124))
    a3.annotate("modelo\nelegido", xy=(EJ1["best_episode"], 101), xytext=(EJ1["best_episode"], 112), ha="center",
                fontsize=9, color=OK, fontweight="bold", va="center")
    save(fig, "01_ej1")


# ----------------------------------------------------------------------------- Ej. 2
def ej2(agent, envs):
    ev = pr.load_df("ej2_evals")
    fig, (a1, a2, a3) = new(144, 64, 1, 3, gridspec_kw={"width_ratios": [1, 1, 1.12]})
    curvas(a1, [pr.trajectory(e) / e.vol_care for e in envs], ALGO); plan(a1)
    a1.set(title="27 trayectorias", xlabel="tiempo transcurrido (t / N)", ylabel="volumen comprado (e / n)",
           xlim=(0, 1), ylim=(0, 1.03))
    T, V = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    A = agent.getQs(np.c_[T.ravel(), V.ravel()], agent.primary_model).argmax(axis=1).reshape(T.shape)
    a2.imshow(A, origin="lower", extent=[0, 1, 0, 1], cmap="Oranges", vmin=0, vmax=1.4, aspect="auto")
    plan(a2); a2.grid(False)
    a2.text(0.30, 0.78, "comprar", color="white", fontweight="bold", ha="center", fontsize=9)
    a2.text(0.70, 0.20, "esperar", color=INK, fontweight="bold", ha="center", fontsize=9)
    a2.set(title="Política aprendida", xlabel="tiempo restante T", ylabel="volumen restante V")
    a3.plot(ev.episode, ev.max_err, "o-", color=ALGO, lw=2, ms=3.5)
    a3.axhline(E_TWAP, color=MKT, ls="--", lw=1.3)
    a3.set(title="Aprendizaje", xlabel="episodio", ylabel="error frente al plan (títulos)", yscale="log")
    logplain(a3, [1, 2, 5])
    a3.text(62, 3.2, "DDQN\nsin explorar", color=ALGO, fontweight="bold", va="center")
    a3.text(305, 1.0, "TWAP: 0,90", color=MKT, fontweight="bold", ha="right", va="bottom")
    save(fig, "02_ej2")


# ----------------------------------------------------------------------------- Ej. 3
def ej3():
    ev, tr = pr.load_df("ej3_p5_evals"), pr.load_df("ej3_p5_train")
    g = lambda df, sc, s, sd: df[(df.config == sc) & (df.schedule == s) & (df.seed == sd)].sort_values("episode")
    evd, evl = g(ev, "inestable", "mejora_d", 0), g(ev, "inestable", "lineal", 0)
    trd, trl = g(tr, "inestable", "mejora_d", 0), g(tr, "inestable", "lineal", 0)
    fig, axes = new(146, 93, 2, 2, sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    a, b = axes[0, 0], axes[1, 0]
    a.plot(evl.episode, evl.max_err, "o-", color=GREY, lw=1.6, ms=3, label="ε lineal (LAB)")
    a.plot(evd.episode, evd.max_err, "o-", color=ALGO, lw=2.2, ms=3.5, label="ε por mejora")
    a.axhline(E_TWAP, color=MKT, ls="--", lw=1.2)
    a.set(title="B · inestable, una ejecución", ylabel="E: error greedy (títulos)", yscale="log")
    a.legend(loc="lower left"); logplain(a, [1, 10])
    b.plot(trl.episode + 1, trl.epsilon, color=GREY, lw=1.6, ls="--", label="ε lineal")
    b.plot(trd.episode + 1, trd.epsilon, color=ALGO, lw=2.2, label="ε por mejora")
    up = evd.epsilon.diff().values > 1e-12
    b.scatter(evd.episode[up], evd.epsilon[up], color=ALGO, edgecolor="k", s=42, zorder=4, label="ε sube")
    b.set(ylabel="ε", xlabel="episodio de entrenamiento", ylim=(0, 1.08)); b.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.98), ncol=3, fontsize=8, columnspacing=0.9, handlelength=1.6, handletextpad=0.4, borderpad=0.2)
    for ax, sc, tit in [(axes[0, 1], "inestable", "B · inestable (3 semillas)"),
                        (axes[1, 1], "final", "A · estable (3 semillas)")]:
        for s, col, lab in [("lineal", GREY, "ε lineal (LAB)"), ("mejora_d", ALGO, "ε por mejora")]:
            m = ev[(ev.config == sc) & (ev.schedule == s)].groupby("episode").max_err
            ax.fill_between(m.mean().index, m.min(), m.max(), color=col, alpha=0.18)
            ax.plot(m.mean().index, m.mean(), "o-", color=col, lw=2, ms=3, label=lab)
        ax.axhline(E_TWAP, color=MKT, ls="--", lw=1.2, label="TWAP")
        ax.set(title=tit, yscale="log"); logplain(ax, [1, 10])
        ax.legend(loc="lower left" if sc == "inestable" else "upper right")
    axes[1, 1].set_xlabel("episodio de entrenamiento")
    save(fig, "03_ej3")


# ----------------------------------------------------------------------------- Ej. 4
def ej4():
    B, B5 = pr.load_df("ej4_ideaB"), pr.load_df("ej4_ideaB_5semillas")
    bins = np.linspace(0.75, 1.0, 51)
    fig, ax = new(69, 48)
    ax.axvspan(0.8, 0.95, color=OK, alpha=0.13, label="banda 80-95 %")
    ax.hist(B[B.horizon == 1].t_fin, bins=bins, color=GREY, alpha=0.75, label="plan al 100 % (ej. 2)")
    ax.hist(B[(B.horizon != 1) & (B.k == 1) & (B.gamma == 0.9)].t_fin, bins=bins, color=ALGO, alpha=0.85, label="plan al 90 %")
    ax.set(xlabel="fin de la orden (fracción del tiempo)", ylabel="órdenes"); ax.legend(loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    save(fig, "04_ej4_B")

    CE, CF = pr.load_df("ej4_ideaC_evals"), pr.load_df("ej4_ideaC_evals_fijo")
    CS = json.loads((pr.RESULTS / "ej4_ideaC.json").read_text(encoding="utf-8"))
    fig, ax = new(69, 48)
    cols = plt.cm.tab10.colors
    for s, gdf in CF.groupby("seed"):
        ax.plot(gdf.episode, gdf.max_err_h, ":", color=cols[s], lw=1, alpha=0.7)
    for s, gdf in CE.groupby("seed"):
        ax.plot(gdf.episode, gdf.max_err_h, "-", color=cols[s], lw=1.6)
        ax.plot(gdf.episode.iloc[-1], gdf.max_err_h.iloc[-1], "X", color=cols[s], ms=8, mec="k", mew=0.6)
    ax.axhline(CS["target"], color=OK, ls="--", lw=1.2)
    ax.text(298, CS["target"] * 1.12, "objetivo: 1,5 títulos", color=OK, ha="right", fontsize=8.2)
    ax.text(298, 17, "✕ para · punteado: sin parar", ha="right", va="top", fontsize=8.2, color=INK)
    ax.set(yscale="log", xlabel="episodio (5 semillas)", ylabel="error frente al plan"); logplain(ax, [1, 10])
    save(fig, "04_ej4_C")

    AT = pr.load_df("ej4_ideaA_test")
    dia = AT.groupby(["politica", "day"]).slip_vs_ref_bps.mean().reset_index()
    stats = dia.groupby("politica").slip_vs_ref_bps.agg(["mean", "sem", "count"])
    print(stats.round(3))
    pols = [("DDQN mercado", "agente", ALGO), ("DDQN placebo", "placebo", GREY),
            ("regla spread", "regla simple", MKT)]
    fig, ax = new(69, 48)
    for i, (p, lab, col) in enumerate(pols):
        m, se = stats.loc[p, "mean"], stats.loc[p, "sem"]
        ax.bar(i, m, yerr=se, color=col, width=0.62, capsize=3, error_kw=dict(lw=1))
        ax.text(i, m + se + 0.03 if m >= 0 else 0.03, f"{m:+.2f}".replace(".", ",").replace("-", "−"), ha="center", fontweight="bold", color=INK)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([p[1] for p in pols])
    ax.set(ylabel="ahorro vs TWAP (pb)", ylim=(-0.12, 0.72)); ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.95, "test · error estándar por días (30)", transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=GREY)
    save(fig, "04_ej4_A")
    return B5


# ----------------------------------------------------------------------------- conclusiones
def resumen(b5):
    rnd = pr.load_df("00_random_grid").max_err.mean()
    rows = [("Recompensa del LAB (ej. 1)", EJ1["kpi"]["max_err"], KO), ("Política aleatoria", rnd, GREY),
            ("Seguimiento + cierre (ej. 2)", pr.load_df("ej2_eval_train").max_err.mean(), ALGO),
            ("Plan al 90 % (ej. 4 · B)", b5.max_err_h.mean(), ALGO), ("TWAP determinista", E_TWAP, MKT)]
    fig, ax = new(104, 86)
    for i, (lab, v, col) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.62)
        ax.text(v * 1.12, i, es(v), va="center", fontweight="bold", color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set(xscale="log", xlim=(0.5, 110), xlabel="títulos (escala log)")
    logplain(ax, [1, 10, 100], "x"); ax.set_title("Error frente al plan TWAP", fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, "05_resumen")


def main():
    agent, envs = ej2_agent_envs()
    portada([pr.trajectory(e) / e.vol_care for e in envs])
    ej1(); ej2(agent, envs); ej3(); b5 = ej4(); resumen(b5)


if __name__ == "__main__":
    main()
