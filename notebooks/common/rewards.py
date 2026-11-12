"""Sistemas de recompensa intercambiables para BestExecutionEnv.

Cada recompensa implementa tres ganchos que el entorno llama en este orden:
``before_step(env, action)`` antes de ejecutar la acción, ``step(env, price, vol)``
después de ejecutarla (con ``env.exec_vol`` ya actualizado y ``env.state_pos`` aún en
el paso actual) y ``done(env)`` una vez, al terminar el episodio.

Notación: N = env.episode_bins, n = env.vol_care, t = env.state_pos, e = env.exec_vol.
El plan TWAP que acaba en h·N pasos marca tras el paso t el objetivo
``min(n, (t + 1) · n / (h · N))`` títulos.
"""
import numpy as np


class Reward:
    name = "base"

    def before_step(self, env, action: int) -> None:
        pass

    def step(self, env, price: float, vol: float) -> float:
        raise NotImplementedError

    def done(self, env) -> float:
        return 0.0

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


def plan_target(env, horizon: float = 1.0, after_step: bool = True) -> float:
    """Títulos que debería llevar ejecutados un TWAP que acaba en horizon · N pasos."""
    t = env.state_pos + (1 if after_step else 0)
    return min(env.vol_care, t * env.vol_care / (horizon * env.episode_bins))


class Default(Reward):
    """Recompensa del notebook original (LAB celda 7, función 6): 0 si no compra y
    (VWAP del episodio - precio) / VWAP del episodio si compra. Es un slippage frente al
    VWAP *global* del episodio, que el agente no puede anticipar con [T, V]."""
    name = "default"

    def step(self, env, price, vol):
        if vol == 0:
            return 0.0
        return (env.episode_vwap - price) / env.episode_vwap


class TWAPRule(Reward):
    """Control (R1): +1 si la acción coincide con la regla del TWAP determinista
    («si voy igual o por detrás del plan, compro»), -1 si no. Le dice al agente qué
    hacer; sirve para comprobar que la red puede aprender la frontera."""
    name = "regla_twap"

    def before_step(self, env, action):
        self._should_buy = env.exec_vol <= plan_target(env, after_step=False) + 1e-9
        self._action = action

    def step(self, env, price, vol):
        return 1.0 if (self._action == 1) == self._should_buy else -1.0


class Tracking(Reward):
    """Seguimiento del plan (R2): penaliza la distancia, en títulos, entre lo ejecutado
    y el plan TWAP tras cada paso. No dice qué acción tomar, solo qué es «ir bien».

    r_t = -min(|e_{t+1} - plan_{t+1}|, clip) / clip  ∈ [-1, 0]

    horizon < 1 hace que el plan acabe antes (idea B del ej. 4, ~90 % de la duración).
    pending_penalty > 0 añade al final -pending_penalty · (títulos pendientes / n).
    """
    name = "seguimiento"

    def __init__(self, clip: float = 5.0, horizon: float = 1.0, pending_penalty: float = 0.0):
        self.clip = clip
        self.horizon = horizon
        self.pending_penalty = pending_penalty

    def step(self, env, price, vol):
        d = env.exec_vol - plan_target(env, self.horizon)
        return -min(abs(d), self.clip) / self.clip

    def done(self, env):
        pending = env.vol_care - env.exec_vol
        return -self.pending_penalty * pending / env.vol_care if pending > 0 else 0.0


class TrackingCierre(Tracking):
    """Seguimiento del plan sin atajo por terminar antes (ej. 2).

    Con ``Tracking`` el episodio acaba al completar la orden y, con él, la penalización:
    comprar todo de golpe cuesta como mucho n pasos de -1 (el error está acotado), y un
    TWAP perfecto paga ~0,05-0,1 por paso durante N ≈ 10·n pasos. Sin descontar, los dos
    retornos se parecen (-47,6 frente a -43,0 en la rejilla) y con γ alto el agente
    aprende a acabar pronto.

    Al terminar antes de N, ``done`` cobra los pasos que faltan como si el agente se
    quedara quieto con lo ejecutado, descontados con el γ del agente:

        r_cierre = Σ_{k=1}^{N-t} γ^k · r_seguimiento(e, t + k)

    (se suma a la recompensa del último paso, de ahí γ^k). Así el retorno descontado es el mismo que si el episodio durara siempre N pasos, que es
    como se mide el error de seguimiento (protocol.episode_metrics).
    """
    name = "seguimiento_cierre"

    def __init__(self, gamma: float = 0.9, clip: float = 5.0, horizon: float = 1.0, pending_penalty: float = 0.0):
        super().__init__(clip, horizon, pending_penalty)
        self.gamma = gamma

    def done(self, env):
        r = super().done(env)
        N, n, t = env.episode_bins, env.vol_care, env.state_pos
        if t >= N:
            return r
        k = np.arange(t + 1, N + 1)
        plan = np.minimum(n, k * n / (self.horizon * N))
        pen = -np.minimum(np.abs(env.exec_vol - plan), self.clip) / self.clip
        return r + float((self.gamma ** np.arange(1, len(k) + 1) * pen).sum())


class TrackingPrice(Tracking):
    """Idea A del ej. 4: seguimiento del plan más un término local de precio al comprar,
    (mid_t - precio) / mid_t en pb · price_weight. Local y factible: depende del spread
    del momento, que puede entrar en el estado."""
    name = "seguimiento_precio"

    def __init__(self, clip: float = 5.0, horizon: float = 1.0, pending_penalty: float = 0.0,
                 price_weight: float = 0.05):
        super().__init__(clip, horizon, pending_penalty)
        self.price_weight = price_weight

    def step(self, env, price, vol):
        r = super().step(env, price, vol)
        if vol > 0:
            mid = env._mid[env.state_pos]
            r += self.price_weight * (mid - price) / mid * 1e4
        return r


REWARDS = {cls.name: cls for cls in (Default, TWAPRule, Tracking, TrackingCierre, TrackingPrice)}


def make(name: str, **kwargs) -> Reward:
    return REWARDS[name](**kwargs)
