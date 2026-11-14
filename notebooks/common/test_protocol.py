"""Validación periódica y selección de checkpoints con episodios controlados."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from . import exploration as ex, protocol as pr
from .agents import DDQNAgent


class TinyEnv:
    n_actions, n_features, obs_mode = 2, 2, "TV"
    episode_bins, vol_care = 1, 1

    def reset(self):
        self.state_pos = self.exec_vol = 0
        return np.ones(2)

    def step(self, action):
        self.state_pos += 1
        self.exec_vol = int(action)
        return None, 1.0, True, {}


def validation(completed=(True, True), pending=(0, 0), slippage=(1.0, 1.0)):
    return pd.DataFrame({"completed": completed, "pending": pending, "slippage_bps": slippage,
                         "max_err": [5.0, 5.0], "mean_err": [2.0, 2.0], "return": [0.1, 0.1]})


class TestValidation(unittest.TestCase):
    def test_completion_then_pending_then_price_select_checkpoint(self):
        score = lambda ev: pr._checkpoint_score(ev, "execution")
        complete = validation(slippage=(-10.0, -10.0))
        incomplete = validation((False, True), (1, 0), (100.0, 100.0))
        more_pending = validation((False, True), (2, 0), (200.0, 200.0))
        self.assertGreater(score(complete), score(incomplete))
        self.assertGreater(score(incomplete), score(more_pending))
        self.assertGreater(score(validation(slippage=(2.0, 2.0))), score(validation()))
        self.assertGreater(score(validation()), score(validation(slippage=(np.nan, 100.0))))

    def test_best_survives_regression_and_last_includes_final_episode(self):
        env = TinyEnv()
        agent = DDQNAgent(env, batch_size=1, hidden_neurons=2, seed=0)
        evaluations = [validation(slippage=(2.0, 2.0)),
                       validation((False, True), (1, 0), (100.0, 100.0)),
                       validation(slippage=(1.0, 1.0))]
        config = {"seed": 0, "selection_split": "val"}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(pr, "evaluate", side_effect=evaluations) as evaluate, \
                patch.object(agent, "update_target", wraps=agent.update_target) as sync:
            checkpoint = Path(tmp) / "model"
            _, ev = pr.train(agent, env, 5, ex.LinearDecay(1.0, 0.1, nepisodes=5),
                             learn_after=1, update_target_after=2,
                             eval_data={"val": "sentinel"}, eval_grid=[{"fixed": True}], eval_every=2,
                             checkpoint=checkpoint, checkpoint_metric="execution",
                             checkpoint_config=config, verbose=False)
            self.assertEqual(ev.episode.tolist(), [2, 4, 5])
            self.assertEqual(ev.is_best.tolist(), [True, False, False])
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(sync.call_count, 2)
            self.assertEqual(len(agent.buffer), 5)
            for call in evaluate.call_args_list:
                self.assertEqual(call.args[1], {"val": "sentinel"})
            best = torch.load(f"{checkpoint}_best.pt", weights_only=False)
            last = torch.load(f"{checkpoint}_last.pt", weights_only=False)
            self.assertEqual(best["episode"], 2)
            self.assertEqual(last["episode"], 5)
            self.assertEqual(best["selection_score"], (1.0, 0.0, 2.0))
            self.assertEqual(best["config"], config)
            self.assertEqual(last["config"], config)
            self.assertTrue(agent._trainable)


if __name__ == "__main__":
    unittest.main()
