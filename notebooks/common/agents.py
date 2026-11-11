"""Agentes: Double DQN en PyTorch y TWAP determinista.

``DDQNAgent`` porta a PyTorch el agente de ``src/agents/dqn.py`` del repositorio
best-execution (Oteo García y Sanz del Real, GNU LGPL-2.1) conservando su interfaz
(``act``, ``experience``, ``sample``, ``learn``, ``update_target``, ``set_trainable``),
sus hiperparámetros por defecto, el Combined Experience Replay y el objetivo Double DQN.
Por ser derivado, este fichero sigue bajo LGPL-2.1.

Cambios propios, marcados con ``# [MOD]``: PyTorch en CPU en lugar de TensorFlow 2.15
(que no instala en Python ≥ 3.12 y es mucho más lento por paso), generador aleatorio
propio del agente, ``learn`` vectorizado y ``save``/``load`` para los checkpoints del
ejercicio 4.
"""
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """Igual que DDQNAgent.network del original: 2 capas ocultas ReLU y salida lineal."""

    def __init__(self, n_features: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DDQNAgent:

    def __init__(self, env, gamma=0.99, epsilon=0.9, alpha=0.01, batch_size=64, buffer_size=50000,
                 hidden_neurons=32, trainable=True, seed=None, force_first_buy=True):
        self.env = env
        self.na = env.n_actions
        self.n_features = env.n_features
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.hidden_neurons = hidden_neurons
        self.force_first_buy = force_first_buy  # el original compra siempre en el paso 0
        self.rng = np.random.default_rng(seed)  # [MOD]
        if seed is not None:
            torch.manual_seed(seed)

        self.primary_model = QNetwork(self.n_features, self.na, hidden_neurons)
        self.optimizer = torch.optim.Adam(self.primary_model.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        # Double Deep Q-Learning: copia congelada que evalúa la acción elegida por la primaria
        self.target_model = QNetwork(self.n_features, self.na, hidden_neurons)
        self.update_target()
        self._trainable = trainable

    # ------------------------------------------------------------------ memoria
    def experience(self, s, a, r, s1, done=False):
        if done:
            s1 = None
        self.buffer.append((s, a, r, s1))

    def sample(self):
        """Combined Experience Replay: batch_size - 1 experiencias al azar más la última."""
        idx = self.rng.choice(len(self.buffer) - 1, size=self.batch_size - 1, replace=False)
        idx = np.append(idx, len(self.buffer) - 1)
        return [self.buffer[i] for i in idx]

    # ------------------------------------------------------------------ política
    @staticmethod
    def getQs(s, model) -> np.ndarray:
        with torch.no_grad():
            return model(torch.as_tensor(np.asarray(s), dtype=torch.float32)).numpy()

    def act(self, s) -> int:
        if self.force_first_buy and self.env.state_pos == 0:
            return 1
        if self._trainable and self.rng.random() < self.epsilon:
            return self.env.action_sample(rng=self.rng)
        return self._act_no_explore(s)

    def _act_no_explore(self, s) -> int:
        return int(np.argmax(self.getQs(np.asarray(s).reshape(1, -1), self.primary_model)))

    def set_trainable(self, train=False):
        self._trainable = train

    def update_target(self):
        self.target_model.load_state_dict(self.primary_model.state_dict())

    # ------------------------------------------------------------------ aprendizaje
    def learn(self) -> float:
        """Un paso de descenso por gradiente sobre un batch; devuelve el MSE."""
        if len(self.buffer) < self.batch_size:
            return float("nan")
        batch = self.sample()
        s = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        a = torch.as_tensor([b[1] for b in batch], dtype=torch.int64)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        not_done = torch.as_tensor([b[3] is not None for b in batch])
        s1 = torch.zeros_like(s)
        if not_done.any():
            s1[not_done] = torch.as_tensor(np.array([b[3] for b in batch if b[3] is not None]), dtype=torch.float32)

        with torch.no_grad():
            best_next = self.primary_model(s1).argmax(dim=1, keepdim=True)  # la primaria elige...
            q_next = self.target_model(s1).gather(1, best_next).squeeze(1)  # ...y la target evalúa
            target = r + self.gamma * q_next * not_done.float()

        q = self.primary_model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------ checkpoints [MOD]
    def save(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"primary": self.primary_model.state_dict(), "target": self.target_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon, **extra}, path)

    def load(self, path) -> dict:
        ckpt = torch.load(path, weights_only=False)
        self.primary_model.load_state_dict(ckpt["primary"])
        self.target_model.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        return ckpt


class TWAPAgent:
    """TWAP determinista (LAB celda 9) sobre el estado s = [T, V]: compra cuando el volumen
    restante es mayor o igual que el tiempo restante, es decir, cuando va igual o por
    detrás del plan («cuando estés igual o por debajo, lanza otra hija», T07)."""
    epsilon = 0.0
    gamma = 1.0

    def __init__(self, env=None):
        self.env = env

    def act(self, s) -> int:
        return int(s[1] >= s[0] - 1e-9)

    def set_trainable(self, train=False):
        pass
