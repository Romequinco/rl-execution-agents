"""Entorno de ejecución de una orden CARE de compra.

Derivado de ``BestExecutionEnv`` del repositorio best-execution (Alberto Oteo García y
Jesús Sanz del Real, GNU LGPL-2.1): misma estructura, mismos nombres y las 13 funciones
numeradas del notebook ``execution_lab.ipynb`` completadas. Este fichero sigue bajo
LGPL-2.1, no bajo la licencia MIT del resto del repositorio.

Cambios propios respecto al original, marcados con ``# [MOD]``:
- Generador aleatorio propio (``seed``) para que los episodios sean reproducibles y la
  exploración del agente no altere la secuencia de episodios.
- ``reset`` admite forzar día, posición de inicio, duración y tamaño de la orden, para
  evaluar siempre sobre la misma rejilla de episodios.
- La recompensa se delega en un objeto de ``rewards.py`` (ejercicios 2 y 4).
- Las series del episodio se precalculan en arrays de numpy (consejo de clase: pandas es
  varias veces más lento en el bucle de pasos).
- ``stats_df`` usa el mismo precio de mercado (mid) que el entorno; el original grafica
  el VWAP de mercado con ``ask1`` (errata E3 de docs/teoria/08_entorno_best_execution.md).
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import rewards as rw


class BestExecutionEnv:

    def __init__(self, data: dict, look_back: int = 60, reward=None, seed: Optional[int] = None,
                 bins_range: Tuple[int, int] = (400, 600), pct_range: Tuple[float, float] = (0.075, 0.125),
                 obs_mode: str = "TV"):
        """
        data: {día: DataFrame} de un split (columnas ask1, bid1, cumvol...).
        look_back: bins previos al inicio que se guardan en ``episode_full`` para features.
        reward: objeto de rewards.py; por defecto la recompensa del notebook original.
        obs_mode: "TV" = [tiempo restante, volumen restante] (original); "TVd" añade d = V - T.
        """
        self.data = data
        self.days = list(data.keys())
        self.look_back = look_back
        self.rng = np.random.default_rng(seed)  # [MOD]
        self.reward = reward if reward is not None else rw.Default()  # [MOD]
        self.bins_range = bins_range
        self.pct_range = pct_range
        self.obs_mode = obs_mode

        self.episode_bins = None
        self.episode_full_len = None
        self.vol_care = None

        self.actions_fn = {
            0: self._do_nothing,
            1: self._agg_action,
        }
        self.n_actions = len(self.actions_fn)
        self.n_features = self._detect_num_feat()

    # ------------------------------------------------------------------ episodio
    def _detect_num_feat(self) -> int:
        """Número de variables del estado (se adapta a observation_builder)."""
        self.reset()
        return self.observation_builder().shape[0]

    def _reset_env_episode_params(self):
        """Reset de las variables internas del episodio."""
        self.episode_full_len = None
        self.episode = None
        self.episode_full = None
        self.episode_vwap = None
        self.market_ep_vol = None
        self.state_pos = 0
        self.algo_vwap = 0
        self.exec_vol = 0
        self.actions_hist = []
        self.algo_vwap_hist = []
        self.market_vwap_hist = []
        self.reward_hist = []
        self.price_hist = []
        self.vol_hist = []

    def _generate_episode_params(self, n_bins: Optional[int] = None, vol_care: Optional[int] = None):
        """(1) Duración uniforme en [400, 600] bins y orden CARE = U[0,075; 0,125] · bins."""
        lo, hi = self.bins_range
        self.episode_bins = int(n_bins) if n_bins is not None else int(self.rng.integers(lo, hi + 1))
        if vol_care is None:
            pct_bins = self.rng.uniform(*self.pct_range)
            vol_care = int(pct_bins * self.episode_bins)
        self.vol_care = int(vol_care)
        self.episode_full_len = self.episode_bins + self.look_back

        assert lo <= self.episode_bins <= hi
        assert int(self.episode_bins * self.pct_range[0]) <= self.vol_care <= int(self.episode_bins * self.pct_range[1])
        assert isinstance(self.vol_care, int)

    def _generate_episode(self, day=None, start_pos: Optional[int] = None,
                          n_bins: Optional[int] = None, vol_care: Optional[int] = None):
        """(2) Día y hora de inicio. El original sortea una hora y repite si no cabe el
        episodio; aquí se sortea directamente una posición válida (mismo resultado, sin
        bucle). [MOD] día y posición forzables."""
        self._generate_episode_params(n_bins, vol_care)
        selected_day = day if day is not None else self.days[self.rng.integers(len(self.days))]
        data_day = self.data[selected_day]
        first, last = self.look_back, len(data_day) - self.episode_bins
        if last < first:
            raise ValueError(f"El día {selected_day} es demasiado corto para {self.episode_bins} bins")
        hour_pos = int(start_pos) if start_pos is not None else int(self.rng.integers(first, last + 1))
        assert first <= hour_pos <= last, "posición de inicio fuera de rango"
        initial_position = hour_pos - self.look_back
        final_position = hour_pos + self.episode_bins
        self.selected_day = selected_day
        self.hour_pos = hour_pos
        self.episode_full = data_day.iloc[initial_position:final_position]
        self.episode = data_day.iloc[hour_pos:final_position]
        assert len(self.episode_full) == self.episode_full_len

    def reset(self, day=None, start_pos: Optional[int] = None, n_bins: Optional[int] = None,
              vol_care: Optional[int] = None) -> np.ndarray:
        """Nuevo episodio; devuelve la primera observación."""
        self._reset_env_episode_params()
        self._generate_episode(day, start_pos, n_bins, vol_care)
        self._compute_episode_market_feat()
        return self.observation_builder()

    # ------------------------------------------------------------------ estado
    def time_left(self) -> float:
        return (self.episode_bins - self.state_pos) / self.episode_bins

    def vol_left(self) -> float:
        return (self.vol_care - self.exec_vol) / self.vol_care

    def plan_deviation(self, horizon: float = 1.0) -> float:
        """Títulos ejecutados menos los que tocarían según un TWAP que acaba en
        ``horizon`` · duración (positivo = vamos por delante del plan)."""
        target = min(self.vol_care, self.state_pos * self.vol_care / (horizon * self.episode_bins))
        return self.exec_vol - target

    def observation_builder(self) -> np.ndarray:
        """(3) Tiempo restante y volumen restante en tanto por uno ([MOD] opcional d = V - T)."""
        time_left, vol_left = self.time_left(), self.vol_left()
        if self.obs_mode == "TVd":
            return np.array([time_left, vol_left, vol_left - time_left])
        return np.array([time_left, vol_left])

    # ------------------------------------------------------------------ mercado
    def _compute_episode_market_feat(self) -> Tuple[float, pd.Series]:
        """(4) VWAP y volumen de mercado del episodio. Sin ejecuciones de mercado, el precio
        de cada bin es el mid; el volumen negociado durante el bin t es el del bin t+1
        (diferencia de cumvol)."""
        mid = (self.episode["ask1"] + self.episode["bid1"]) / 2
        self.market_ep_vol = self.episode["cumvol"].diff()
        self.market_ep_vol.iloc[0] = 0
        cum_vol = self.market_ep_vol.sum()
        self.episode_vwap = float((mid.values[:-1] * self.market_ep_vol.values[1:]).sum() / cum_vol) \
            if cum_vol > 0 else float(mid.iloc[0])
        # [MOD] arrays precalculados para el bucle de pasos
        self._mid = mid.values.astype(float)
        self._ask = self.episode["ask1"].values.astype(float)
        self._bin_vol = np.append(self.market_ep_vol.values[1:], 0.0).astype(float)
        self._cum_pv = np.cumsum(self._mid * self._bin_vol)
        self._cum_v = np.cumsum(self._bin_vol)
        return self.episode_vwap, self.market_ep_vol

    def _compute_market_vwap(self) -> float:
        """(5) VWAP de mercado acumulado hasta el step actual incluido."""
        sum_vol = self._cum_v[self.state_pos]
        if sum_vol == 0:
            return float(self._mid[self.state_pos])
        return float(self._cum_pv[self.state_pos] / sum_vol)

    def _compute_algo_vwap(self) -> float:
        """(8) VWAP del algoritmo hasta el step actual (nan si aún no ha comprado)."""
        v_arr = np.asarray(self.vol_hist, dtype=float)
        if v_arr.sum() == 0:
            return np.nan
        p_arr = np.asarray(self.price_hist, dtype=float)
        return float((p_arr * v_arr).sum() / v_arr.sum())

    # ------------------------------------------------------------------ recompensa y fin
    def _compute_reward(self, price: float, vol: float) -> float:
        """(6) Recompensa del paso, delegada en el objeto ``reward`` ([MOD])."""
        return self.reward.step(self, price, vol)

    def _compute_stop_conditions(self) -> Tuple[bool, bool]:
        """(10) Tiempo agotado, orden completada."""
        is_bins_complete = self.state_pos >= self.episode_bins
        is_ord_complete = self.exec_vol >= self.vol_care
        return is_bins_complete, is_ord_complete

    def _compute_done_reward(self) -> float:
        """(12) Recompensa adicional al terminar (0 en el original)."""
        return self.reward.done(self)

    def _compute_done(self) -> bool:
        """(11) Termina si se cumple cualquiera de las condiciones de parada."""
        is_bins_complete, is_ord_complete = self._compute_stop_conditions()
        return is_bins_complete or is_ord_complete

    # ------------------------------------------------------------------ acciones
    def _record(self, price: float, vol: int) -> float:
        self.price_hist.append(price)
        self.vol_hist.append(vol)
        self.exec_vol += vol
        self.algo_vwap = self._compute_algo_vwap()
        self.algo_vwap_hist.append(self.algo_vwap)
        return self._compute_reward(price, vol)

    def _agg_action(self) -> float:
        """(7) Compra agresiva de 1 título al ask1 del step."""
        return self._record(float(self._ask[self.state_pos]), 1)

    def _do_nothing(self) -> float:
        """(9) No hacer nada: precio y volumen 0."""
        return self._record(0.0, 0)

    def step(self, action: int):
        """Evalúa la acción, calcula la recompensa y devuelve (obs, r, done, info)."""
        market_vwap = self._compute_market_vwap()
        act_fn = self.actions_fn.get(int(action))
        if act_fn is None:
            raise ValueError(f"Invalid action {action}. Valid actions {list(self.actions_fn)}")
        self.reward.before_step(self, int(action))
        reward = act_fn()
        self.actions_hist.append(int(action))
        self.market_vwap_hist.append(market_vwap)
        self.reward_hist.append(reward)
        self.state_pos += 1
        done = self._compute_done()
        if done:
            reward += self._compute_done_reward()
            return None, reward, done, {}
        return self.observation_builder(), reward, done, {}

    def action_sample(self, rng=None, p: Optional[float] = None) -> int:
        """(13) Acción aleatoria. Por defecto compra con probabilidad ρ = vol_care / bins,
        el ritmo de un TWAP: con p = 0,5 el agente acabaría la orden en ~2n pasos y casi
        nunca vería la parte final del episodio. ``rng`` permite usar el generador del
        agente para no consumir el del entorno ([MOD])."""
        rng = rng if rng is not None else self.rng
        p = self.vol_care / self.episode_bins if p is None else p
        return int(rng.random() < p)

    # ------------------------------------------------------------------ informe
    def stats_df(self):
        """Datos para plot_results: (algo, mercado, mid)."""
        idx = self.episode.index
        n = len(self.algo_vwap_hist)
        my_df = pd.DataFrame({"vwap": self.algo_vwap_hist, "vol": self.vol_hist}, index=idx[:n])
        my_df = my_df.reindex(idx)
        my_df["vol"] = my_df["vol"].fillna(0)
        my_df["vwap"] = my_df["vwap"].ffill()
        market_vwap = pd.Series(np.where(self._cum_v > 0, self._cum_pv / np.maximum(self._cum_v, 1e-12), self._mid), index=idx)
        market_df = pd.DataFrame({"vwap": market_vwap, "vol": self._bin_vol}, index=idx)
        mpx = pd.Series(self._mid, index=idx)
        return my_df, market_df, mpx
