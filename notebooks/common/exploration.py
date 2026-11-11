"""Planes de exploración (ε) para DDQNAgent.

- ``LinearDecay``: el del notebook original, ε baja una cantidad fija por episodio.
  Depende solo del número de episodios: es lo que el ejercicio 3 pide sustituir.
- ``ImprovementEpsilon``: ε según la mejora medida (ejercicio 3).

Los dos exponen ``on_episode_end(agent, episode)`` y ``on_eval(agent, metric)``; el bucle
de entrenamiento llama al primero tras cada episodio y al segundo tras cada evaluación.
"""
import numpy as np


class LinearDecay:
    def __init__(self, epsilon=1.0, min_epsilon=0.05, epsilon_decay=None, nepisodes=None, frac=0.5):
        """Si no se da ``epsilon_decay``, se calcula para llegar a ``min_epsilon`` en
        ``frac · nepisodes`` episodios (como en los notebooks de rl-material)."""
        self.start = epsilon
        self.min_epsilon = min_epsilon
        if epsilon_decay is None:
            epsilon_decay = (epsilon - min_epsilon) / max(1, frac * nepisodes)
        self.epsilon_decay = epsilon_decay

    def init(self, agent):
        agent.epsilon = self.start

    def on_episode_end(self, agent, episode):
        agent.epsilon = max(agent.epsilon - self.epsilon_decay, self.min_epsilon)

    def on_eval(self, agent, metric):
        pass


class ImprovementEpsilon:
    """ε como función de la mejora medida en evaluación greedy.

    Cada evaluación da un error de seguimiento E (títulos; menor es mejor). La mejora es
    la fracción del camino recorrido entre la referencia sin aprender (E0, primera
    evaluación) y el objetivo (E*, el error del TWAP determinista):

        progreso = clip((E0 - E) / (E0 - E*), 0, 1)

    ε = eps_max - (eps_max - eps_min) · progreso_ref, donde progreso_ref es el mejor
    progreso alcanzado. Si durante ``patience`` evaluaciones seguidas el progreso cae más de
    ``margin`` por debajo de progreso_ref, se rebaja progreso_ref al valor actual y ε vuelve
    a subir: el agente ha empeorado y necesita explorar otra vez. Entre evaluaciones ε no
    cambia, sea cual sea el número de episodios.
    """

    def __init__(self, target_error, eps_max=1.0, eps_min=0.05, margin=0.1, patience=2):
        self.target_error = target_error
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.margin = margin
        self.patience = patience
        self.base_error = None
        self.best_progress = 0.0
        self._worse = 0
        self.history = []

    def init(self, agent):
        agent.epsilon = self.eps_max

    def on_episode_end(self, agent, episode):
        pass

    def progress(self, error):
        span = self.base_error - self.target_error
        if span <= 0:
            return 1.0
        return float(np.clip((self.base_error - error) / span, 0.0, 1.0))

    def on_eval(self, agent, metric):
        if self.base_error is None:
            self.base_error = max(metric, self.target_error)
        p = self.progress(metric)
        if p >= self.best_progress:
            self.best_progress, self._worse = p, 0
        elif p < self.best_progress - self.margin:
            self._worse += 1
            if self._worse >= self.patience:
                self.best_progress, self._worse = p, 0
        agent.epsilon = self.eps_max - (self.eps_max - self.eps_min) * self.best_progress
        self.history.append({"metric": metric, "progress": p, "ref": self.best_progress, "epsilon": agent.epsilon})
