"""Invariantes del ej. 4. Ejecutar desde notebooks/: python -m unittest common.test_ej4 -v"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import ej4
from . import protocol as pr
from .agents import TWAPAgent
from .data import load


class TestIdeaB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load("REP")["train"]
        cls.grid = pr.make_grid(cls.train, n_days=1, seed=0)

    def test_twaph_1_es_el_twap(self):
        for s in np.random.default_rng(0).uniform(size=(500, 2)):
            self.assertEqual(ej4.TWAPh(1.0).act(s), TWAPAgent().act(s))

    def test_twaph_09_acaba_en_banda(self):
        res = ej4.evaluate(ej4.TWAPh(0.9), self.train, self.grid, ej4.make_reward(0.9), horizon=0.9)
        self.assertTrue(res.completed.all())
        self.assertTrue(res.in_band.all())
        self.assertLess(res.max_err_h.max(), 1.0)

    def test_terminal_solo_con_pendiente(self):
        rows = ej4.calibration(self.train, self.grid, gammas=(0.9,), ks=(10.0,))
        for name, row in rows.iterrows():
            self.assertAlmostEqual(row["terminal_k10"], -10 * row["pendiente_frac"])


class TestIdeaC(unittest.TestCase):
    def test_reanudacion_exacta(self):
        with tempfile.TemporaryDirectory() as d:
            ck = str(Path(d) / "c")
            a = ej4.make_trainer(seed=0, max_episodes=10, eval_every=5)
            _, ev_a = a.run()
            b = ej4.make_trainer(seed=0, max_episodes=10, eval_every=5, checkpoint=ck)
            b.run(5)
            c = ej4.make_trainer(seed=3, max_episodes=10, eval_every=5, checkpoint=ck)
            c.load(ck + "_last.pt")
            _, ev_c = c.run()
            self.assertTrue(ev_a.equals(ev_c))
            for k, v in a.agent.primary_model.state_dict().items():
                self.assertTrue(torch.equal(v, c.agent.primary_model.state_dict()[k]))

    def test_best_no_empeora_al_reanudar(self):
        with tempfile.TemporaryDirectory() as d:
            ck = str(Path(d) / "c")
            t = ej4.make_trainer(seed=0, max_episodes=15, eval_every=5, checkpoint=ck)
            _, ev = t.run()
            best = torch.load(ck + "_best.pt", weights_only=False)["metric"]
            self.assertAlmostEqual(best, ev.max_err_h.min())


if __name__ == "__main__":
    unittest.main()
