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


class TestAmpliaciones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load("REP")
        cls.norm = ej4.market_norm(cls.data["train"])

    def test_variables_de_mercado_causales(self):
        # Cambiar el futuro de un día no cambia las variables del pasado
        df = next(iter(self.data["train"].values())).copy()
        x1 = ej4._raw_features(df)
        df.iloc[500:, :4] = df.iloc[500:, :4] * 1.1
        x2 = ej4._raw_features(df)
        np.testing.assert_array_equal(x1[:500], x2[:500])

    def test_placebo_mismas_variables_otro_orden(self):
        kw = dict(norm=self.norm, reward=ej4.TrackingPriceCierre(), seed=0)
        cfg = pr.make_grid(self.data["train"], n_days=1, seed=0)[0]
        args = {k: cfg[k] for k in ("day", "start_pos", "n_bins", "vol_care")}
        real, plac = ej4.MarketEnv(self.data["train"], **kw), ej4.MarketEnv(self.data["train"], placebo=True, **kw)
        real.reset(**args), plac.reset(**args)
        self.assertFalse(np.array_equal(real._feat, plac._feat))
        np.testing.assert_allclose(np.sort(real._feat, axis=0), np.sort(plac._feat, axis=0))

    def test_pasillo_sin_castigo_dentro(self):
        b = ej4.BandTracking()
        n, N = 50, 500
        for t in (100, 250, 400):
            dentro = (t * n / (0.95 * N) + min(n, t * n / (0.80 * N))) / 2
            self.assertEqual(float(b._pen(dentro, t, n, N)), 0.0)
            self.assertLess(float(b._pen(0, t, n, N)), 0.0)


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
