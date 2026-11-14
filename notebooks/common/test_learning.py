"""Cálculos Double DQN con resultados conocidos, sin datos de mercado."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn

from .agents import DDQNAgent


class TestDoubleDQN(unittest.TestCase):
    def setUp(self):
        self.agent = DDQNAgent(SimpleNamespace(n_actions=2, n_features=2),
                               gamma=0.5, batch_size=2, seed=0)
        # Q_online([x,y])=[x,y]; Q_target=[5,20]. Los argmax discrepan.
        self.agent.primary_model = nn.Linear(2, 2)
        self.agent.target_model = nn.Linear(2, 2)
        with torch.no_grad():
            self.agent.primary_model.weight.copy_(torch.eye(2))
            self.agent.primary_model.bias.zero_()
            self.agent.target_model.weight.zero_()
            self.agent.target_model.bias.copy_(torch.tensor([5.0, 20.0]))
        self.agent.optimizer = torch.optim.SGD(self.agent.primary_model.parameters(), lr=0.1)
        self.agent.loss_fn = Mock(wraps=nn.MSELoss())

    def mixed_batch(self):
        self.agent.experience(np.array([1.0, 2.0]), 1, 3.0, np.array([4.0, 1.0]))
        # done=True elimina s1 aunque se haya entregado una observación siguiente.
        self.agent.experience(np.array([2.0, 1.0]), 0, -2.0, np.array([4.0, 1.0]), done=True)

    def test_online_selects_target_evaluates_and_terminal_has_no_future(self):
        self.mixed_batch()
        loss = self.agent.learn()
        q, target = self.agent.loss_fn.call_args.args
        # Online elige 0; target la valora en 5, no en su máximo 20.
        # y_0=3+0.5*5=5.5; y_1=-2 (terminal).
        torch.testing.assert_close(q.detach(), torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(target, torch.tensor([5.5, -2.0]))
        self.assertFalse(target.requires_grad)
        self.assertAlmostEqual(loss, ((2 - 5.5)**2 + (2 - (-2))**2) / 2)
        # Gradiente solo por las Q ejecutadas, sin propagar por el objetivo.
        torch.testing.assert_close(self.agent.primary_model.weight.grad,
                                   torch.tensor([[8.0, 4.0], [-3.5, -7.0]]))
        torch.testing.assert_close(self.agent.primary_model.bias.grad, torch.tensor([4.0, -3.5]))
        self.assertTrue(all(p.grad is None for p in self.agent.target_model.parameters()))

    def test_all_terminal_batch_targets_are_exact_rewards(self):
        for reward in (3.0, -2.0):
            self.agent.experience(np.array([1.0, 2.0]), 1, reward, None, done=True)
        loss = self.agent.learn()
        _, target = self.agent.loss_fn.call_args.args
        torch.testing.assert_close(target, torch.tensor([3.0, -2.0]))
        self.assertAlmostEqual(loss, ((2 - 3)**2 + (2 - (-2))**2) / 2)

    def test_target_changes_only_on_explicit_sync(self):
        self.mixed_batch()
        online = {k: v.clone() for k, v in self.agent.primary_model.state_dict().items()}
        target = {k: v.clone() for k, v in self.agent.target_model.state_dict().items()}
        self.agent.learn()
        self.assertTrue(any(not torch.equal(v, online[k])
                            for k, v in self.agent.primary_model.state_dict().items()))
        for k, v in self.agent.target_model.state_dict().items():
            torch.testing.assert_close(v, target[k], rtol=0, atol=0)
        self.agent.update_target()
        for k, v in self.agent.target_model.state_dict().items():
            torch.testing.assert_close(v, self.agent.primary_model.state_dict()[k], rtol=0, atol=0)
        # Copia de valores, sin compartir almacenamiento entre redes.
        with torch.no_grad():
            self.agent.primary_model.bias.add_(1)
        torch.testing.assert_close(self.agent.target_model.bias, self.agent.primary_model.bias - 1)


if __name__ == "__main__":
    unittest.main()
