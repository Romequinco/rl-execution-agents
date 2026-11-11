"""Gráficos. ``plot_results`` deriva de src/report/report.py de best-execution (LGPL-2.1):
[MOD] crea una figura nueva en cada llamada (el original reutiliza ``plt.figure(1)`` y
superpone episodios), usa el mid como precio de mercado y admite título y guardado."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .protocol import FIGURES, trajectory

ALGO, MKT, MID = "#D9822B", "#2B6CB0", "#C53030"


def _save(fig, name):
    if name:
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / f"{name}.png", dpi=130, bbox_inches="tight")


def plot_results(env, title=None, mode="max", save=None):
    """VWAP de mercado frente al del algoritmo (arriba) y volumen acumulado (abajo)."""
    my_df, market_df, mpx = env.stats_df()
    mkt_vol = market_df["vol"]
    norm_mkt_vol = mkt_vol / (mkt_vol.max() if mode == "max" else mkt_vol.quantile(0.99))
    norm_mkt_vol = norm_mkt_vol.clip(upper=1)
    cum_mkt_vol = mkt_vol.cumsum() / mkt_vol.sum()
    cum_my_vol = my_df["vol"].cumsum() / env.vol_care
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax1.plot(market_df["vwap"], color=MKT, label="VWAP mercado")
    ax1.plot(my_df["vwap"], color=ALGO, label="VWAP algo")
    ax1.plot(mpx, color=MID, alpha=0.6, lw=0.8, label="mid")
    ax1.set_title(title or f"Market vs algo · N={env.episode_bins}, n={env.vol_care}")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(cum_mkt_vol, color=MKT, label="mercado: vol. acumulado")
    ax2.plot(cum_my_vol, color=ALGO, label="algo: vol. acumulado")
    width = (mkt_vol.index[1] - mkt_vol.index[0]) * 0.8 if len(mkt_vol) > 1 else 0.0001
    ax2.bar(mkt_vol.index, norm_mkt_vol * 0.3, width=width, color=MKT, alpha=0.4, label="vol. mercado")
    ax2.bar(my_df.index, (my_df["vol"] > 0) * 0.1, width=width, color=ALGO, alpha=0.8, label="compras algo")
    ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_trajectories(envs_or_trajs, labels=None, title="Volumen ejecutado frente al plan TWAP", save=None):
    """e_t / n frente a t / N para varios episodios; la diagonal es el TWAP perfecto."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="TWAP ideal")
    for i, item in enumerate(envs_or_trajs):
        if hasattr(item, "vol_hist"):
            e, N, n = trajectory(item), item.episode_bins, item.vol_care
        else:
            e, N, n = item
        ax.plot(np.arange(1, N + 1) / N, e / n, lw=1.2, alpha=0.8, label=labels[i] if labels else None)
    ax.set_xlabel("tiempo transcurrido (t / N)"); ax.set_ylabel("volumen ejecutado (e / n)")
    ax.set_title(title); ax.grid(alpha=0.3)
    if labels:
        ax.legend(fontsize=8)
    _save(fig, save)
    return fig


def plot_policy_map(agent, obs_mode="TV", res=101, title="Política greedy sobre (T, V)", save=None):
    """Acción greedy del agente en cada (tiempo restante, volumen restante). Un TWAP
    perfecto compra por encima de la diagonal V ≥ T."""
    T, V = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res))
    S = np.c_[T.ravel(), V.ravel()]
    if obs_mode == "TVd":
        S = np.c_[S, S[:, 1] - S[:, 0]]
    A = agent.getQs(S, agent.primary_model).argmax(axis=1).reshape(T.shape)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(A, origin="lower", extent=[0, 1, 0, 1], cmap="Oranges", vmin=0, vmax=1.4, aspect="auto")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="frontera TWAP (V = T)")
    ax.set_xlabel("tiempo restante T"); ax.set_ylabel("volumen restante V")
    ax.set_title(f"{title}\n(naranja = comprar)"); ax.legend(loc="lower right")
    _save(fig, save)
    return fig


def plot_training(ep: pd.DataFrame, ev: pd.DataFrame = None, window=20, title="Entrenamiento", save=None):
    """Recompensa por episodio (media móvil), ε y, si hay evaluaciones, la métrica greedy."""
    n = 3 if ev is not None and len(ev) else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.6))
    axes[0].plot(ep.episode, ep.reward, alpha=0.25, color=ALGO)
    axes[0].plot(ep.episode, ep.reward.rolling(window, min_periods=1).mean(), color=ALGO)
    axes[0].set_title("recompensa por episodio"); axes[0].grid(alpha=0.3)
    axes[1].plot(ep.episode, ep.epsilon, color=MKT); axes[1].set_title("ε"); axes[1].grid(alpha=0.3)
    if n == 3:
        axes[2].plot(ev.episode, ev.max_err, "o-", color=MID)
        axes[2].set_title("error máx. de seguimiento (eval. greedy, títulos)"); axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel("episodio")
    fig.suptitle(title); fig.tight_layout()
    _save(fig, save)
    return fig
