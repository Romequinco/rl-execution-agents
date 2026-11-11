"""Invariantes del entorno, las recompensas y los agentes.

Usa datos sintéticos con el mismo esquema que rep_data.pkl (DataFrame cada 5 s de 13:00
a 17:25 con ask1, bid1, vask1, vbid1, last, cumvol), así que no necesita los pickles.
Ejecutar desde notebooks/:  python -m unittest common.test_env -v
"""
import tempfile
import unittest

import numpy as np
import pandas as pd

from . import exploration as ex
from . import protocol as pr
from . import rewards as rw
from .agents import DDQNAgent, TWAPAgent
from .data import synthetic_data
from .env import BestExecutionEnv


class TestEnv(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        self.env = BestExecutionEnv(self.data, seed=1)

    def test_params_in_range(self):
        for _ in range(50):
            self.env.reset()
            N, n = self.env.episode_bins, self.env.vol_care
            self.assertTrue(400 <= N <= 600)
            self.assertTrue(int(0.075 * N) <= n <= int(0.125 * N))
            self.assertEqual(len(self.env.episode), N)
            self.assertEqual(len(self.env.episode_full), N + 60)

    def test_first_observation(self):
        s = self.env.reset()
        np.testing.assert_allclose(s, [1.0, 1.0])
        self.assertEqual(self.env.n_features, 2)
        self.assertEqual(BestExecutionEnv(self.data, obs_mode="TVd").n_features, 3)

    def test_forced_reset_is_reproducible(self):
        day = list(self.data)[1]
        self.env.reset(day=day, start_pos=500, n_bins=450, vol_care=45)
        self.assertEqual(self.env.episode.index[0], self.data[day].index[500])
        self.assertEqual((self.env.episode_bins, self.env.vol_care), (450, 45))

    def test_episode_vwap_matches_manual(self):
        self.env.reset()
        ep = self.env.episode
        mid = ((ep.ask1 + ep.bid1) / 2).values
        v = np.diff(ep.cumvol.values)
        self.assertAlmostEqual(self.env.episode_vwap, (mid[:-1] * v).sum() / v.sum(), places=10)

    def test_market_vwap_last_step_equals_episode_vwap(self):
        self.env.reset()
        self.env.state_pos = self.env.episode_bins - 1
        self.assertAlmostEqual(self.env._compute_market_vwap(), self.env.episode_vwap, places=10)

    def test_buy_all_ends_when_order_complete(self):
        self.env.reset()
        steps, done = 0, False
        while not done:
            _, _, done, _ = self.env.step(1)
            steps += 1
        self.assertEqual(steps, self.env.vol_care)
        self.assertEqual(self.env.exec_vol, self.env.vol_care)

    def test_do_nothing_ends_at_horizon_with_pending(self):
        self.env.reset()
        steps, done, s1 = 0, False, None
        while not done:
            s1, _, done, _ = self.env.step(0)
            steps += 1
        self.assertIsNone(s1)
        self.assertEqual(steps, self.env.episode_bins)
        self.assertTrue(np.isnan(self.env._compute_algo_vwap()))

    def test_invalid_action(self):
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(2)

    def test_action_sample_rate(self):
        self.env.reset(n_bins=500, vol_care=50)
        rng = np.random.default_rng(0)
        rate = np.mean([self.env.action_sample(rng=rng) for _ in range(20000)])
        self.assertAlmostEqual(rate, 0.1, delta=0.01)


class TestTWAP(unittest.TestCase):
    def test_deterministic_twap_tracks_plan(self):
        data = synthetic_data()
        env = BestExecutionEnv(data, seed=2)
        agent = TWAPAgent(env)
        for cfg in pr.make_grid(data, n_days=2, seed=3):
            pr.run_episode(env, agent, cfg)
            m = pr.episode_metrics(env)
            self.assertTrue(m["completed"], cfg)
            self.assertLessEqual(m["max_err"], 1.0 + 1e-9, cfg)

    def test_twap_rule_reward_is_max_for_twap(self):
        data = synthetic_data()
        env = BestExecutionEnv(data, reward=rw.TWAPRule(), seed=4)
        ret = pr.run_episode(env, TWAPAgent(env))
        self.assertAlmostEqual(ret, len(env.vol_hist))

    def test_tracking_reward_bounds(self):
        data = synthetic_data()
        env = BestExecutionEnv(data, reward=rw.Tracking(clip=5, pending_penalty=0), seed=5)
        env.reset()
        rs = [env.step(0)[1] for _ in range(100)]
        self.assertTrue(all(-1.0 <= r <= 0.0 for r in rs))
        self.assertLess(rs[-1], rs[0])


class TestAgent(unittest.TestCase):
    def test_fill_learn_eval_checkpoint(self):
        data = synthetic_data()
        env = BestExecutionEnv(data, reward=rw.Tracking(), seed=6)
        agent = DDQNAgent(env, buffer_size=5000, seed=6)
        self.assertGreater(pr.fill_buffer(agent, env, n_episodes=3), 64)
        loss = agent.learn()
        self.assertTrue(np.isfinite(loss))
        grid = pr.make_grid(data, Ns=(400,), rhos=(0.1,), n_days=1)
        with tempfile.TemporaryDirectory() as tmp:
            ep, ev = pr.train(agent, env, nepisodes=4, exploration=ex.ImprovementEpsilon(target_error=1.0),
                              eval_data=data, eval_grid=grid, eval_every=2, checkpoint=f"{tmp}/m", verbose=False)
            self.assertEqual(len(ep), 4)
            self.assertEqual(len(ev), 2)
            other = DDQNAgent(env, buffer_size=10, seed=7)
            other.load(f"{tmp}/m_last.pt")
            s = np.array([[0.5, 0.6]])
            np.testing.assert_allclose(other.getQs(s, other.primary_model), agent.getQs(s, agent.primary_model))
        res = pr.evaluate(agent, data, grid, reward=rw.Tracking())
        self.assertIn("slippage_vs_twap_bps", res)
        self.assertTrue(set(res.columns) >= {"max_err", "completed", "t_fin"})

    def test_first_action_is_forced_buy(self):
        data = synthetic_data()
        env = BestExecutionEnv(data, seed=8)
        agent = DDQNAgent(env, epsilon=0.0, seed=8)
        s = env.reset()
        self.assertEqual(agent.act(s), 1)


class TestExploration(unittest.TestCase):
    def test_improvement_epsilon(self):
        class A:
            epsilon = 1.0
        a, sched = A(), ex.ImprovementEpsilon(target_error=1.0, eps_max=1.0, eps_min=0.0, margin=0.1, patience=2)
        sched.on_eval(a, 11.0)
        self.assertAlmostEqual(a.epsilon, 1.0)
        sched.on_eval(a, 6.0)
        self.assertAlmostEqual(a.epsilon, 0.5)
        sched.on_eval(a, 8.0)              # empeora una vez: ε se mantiene
        self.assertAlmostEqual(a.epsilon, 0.5)
        sched.on_eval(a, 8.0)              # segunda vez: ε vuelve a subir
        self.assertAlmostEqual(a.epsilon, 0.7)

    def test_linear_decay(self):
        class A:
            epsilon = 1.0
        a, sched = A(), ex.LinearDecay(1.0, 0.05, nepisodes=100, frac=0.5)
        for e in range(100):
            sched.on_episode_end(a, e)
        self.assertAlmostEqual(a.epsilon, 0.05)


if __name__ == "__main__":
    unittest.main()
