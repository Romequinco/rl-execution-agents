"""Audita los resultados de ej1_search.py y genera un resumen en Markdown."""
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "ej1_search"


def main():
    # El fingerprint del experimento corresponde al JSON serializado con LF.
    # read_text normaliza los CRLF que write_text introduce en Windows.
    protocol_text = (OUT / "protocol.json").read_text(encoding="utf-8")
    protocol = json.loads(protocol_text)
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(protocol_text.encode()).hexdigest() == decision["protocol_sha256"]
    for name, digest in protocol["sha256"].items():
        # Los .py también se comparan con LF: con autocrlf=true Windows los deja en CRLF.
        raw = (ROOT / name).read_bytes()
        assert digest in {hashlib.sha256(raw).hexdigest(),
                          hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}, name
    manifests = list(OUT.glob("*/*/manifest.json"))
    assert len(manifests) == 28
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        signature = manifest["signature"]
        assert signature["fingerprint"] == decision["protocol_sha256"]
        episodes = 2000 if signature["stage"] == "screen" else 4000
        train = pd.read_csv(path.parent / "train.csv")
        ev = pd.read_csv(path.parent / "eval.csv")
        assert len(train) == episodes
        assert len(ev) == (27 if episodes == 2000 else 108)
        assert np.isfinite(train[["reward", "disc_reward", "mse"]].to_numpy()).all()
        np.testing.assert_allclose(ev["return"], ev.n_buys * ev.slippage_bps / 1e4,
                                   atol=1e-12, rtol=1e-9)
    current = torch.load(ROOT / "results/models/ej1_ddqn.pt", weights_only=False)
    chosen = decision["selected_case"]
    confirmed = torch.load(OUT / "confirm" / f"{chosen}_seed0" / "model.pt", weights_only=False)
    for key in ("primary", "target"):
        assert all(torch.equal(v, confirmed[key][k]) for k, v in current[key].items())
    config = json.loads((ROOT / "results/ej1_config.json").read_text(encoding="utf-8"))
    assert all(config[k] == v for k, v in decision["parameters"].items())
    notebook_path = ROOT / "notebooks/ej1_execution_lab.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    prior = json.loads((ROOT / "results/ej1_final_checks/decision.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(notebook_path.read_bytes()).hexdigest() == prior["notebook_sha256"]
    for case in ("base", "alpha_1e-4"):
        for seed in (0, 1, 2):
            folder = OUT / "confirm" / f"{case}_seed{seed}"
            old = ROOT / "results/ej1_final_checks"
            pd.testing.assert_frame_equal(pd.read_csv(folder / "train.csv"),
                                          pd.read_csv(old / f"{case}_seed{seed}_train.csv"))
    for seed in (0, 1, 2):
        ev = pd.read_csv(OUT / "confirm" / f"target_1280_seed{seed}" / "eval.csv")
        np.testing.assert_allclose(ev.t_fin, ev.n / ev.N, atol=1e-12)
    audit = dict(training_runs=28, training_episodes=74000, screen_configurations=19,
                 confirmation_configurations=3, confirmation_seeds=[0, 1, 2],
                 evaluations_per_confirmation=108, reward_identity_checked_all_runs=True,
                 source_hashes_verified=True, selected_checkpoint_matches_current=True,
                 notebook_unchanged_and_executed=True, previous_six_runs_reproduced=True)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    screen = pd.read_csv(OUT / "screen_ranking.csv")
    comp = pd.read_csv(OUT / "comparison.csv")
    lines = ["# Búsqueda ampliada de parámetros · Ejercicio 1", "", "Fecha: 13/11/2026.", "",
             "**Resultado: conservar la configuración actual.** El reward original del LAB, con recompensa terminal 0, permanece intacto.", "",
             "## Procedimiento", "",
             "- Criba: 19 configuraciones, 2000 episodios y semilla 0; cada alternativa cambia un solo parámetro.",
             "- Confirmación: dos alternativas finalistas y la base; entrenamientos nuevos de 4000 episodios con semillas 0, 1 y 2.",
             "- Epsilon de 1 a 0,02 en 2000 episodios; llenado inicial de 25 episodios; estado [T,V] y compra inicial forzada.",
             "- Evaluación greedy: 27 episodios en la criba; 108 por semilla en la confirmación (27 originales + 81 adicionales, con más días y ventanas de train).",
             "- Selección solo con train. Test no se ha utilizado para elegir parámetros en esta búsqueda.",
             "- Criterio prefijado: completar 324/324 órdenes, mejorar al menos 0,25 pb de media y mejorar en al menos dos semillas. El margen de 0,25 pb es práctico; no es un contraste estadístico.",
             "- Los desempates de la criba se ordenan por nombre. La base se incluye siempre en la confirmación.", "",
             "## Valores explorados", "", "| Parámetro | Valores |", "|---|---|"]
    for name in ("gamma", "alpha", "buffer_size", "batch_size", "hidden_neurons", "learn_after", "update_target_after"):
        values = ", ".join(f"{v:g}" for v in sorted(screen[name].unique()))
        lines.append(f"| `{name}` | {values} |")
    lines += ["", "## Confirmación", "", "| Configuración | Completas | Slippage medio (pb) | Mejora frente a base (pb) | Semillas que mejoran |",
              "|---|---:|---:|---:|---:|"]
    for row in comp.itertuples():
        full = row.completed == row.n_eval
        price = f"{row.slippage_bps:.6f}" if full else "No comparable: incompleta"
        gain = f"{row.gain_over_base_bps:.6f}" if full else "—"
        lines.append(f"| {row.case} | {row.completed}/{row.n_eval} | {price} | {gain} | {row.improved_seeds}/3 |")
    lines += ["", "Mayor slippage es mejor: (VWAP mercado − VWAP algoritmo) / VWAP mercado × 10 000. Esta tabla usa la rejilla ampliada; sus medias no se comparan directamente con las de los 27 episodios originales.", "",
              "`target_1280` coincide con BuyAll en las tres semillas de esta evaluación. La mejora de aproximadamente 0,004 pb procede de la pequeña diferencia de la semilla 0 de la base y no supera el criterio prefijado.", "",
              "`alpha=1e-4` completa las órdenes y mejora el precio en la criba de 2000 episodios/semilla 0. A los 4000 episodios esa semilla deja volumen pendiente en los 108 casos; las otras dos completan todo. Se descarta por esa inconsistencia.", "",
              "## Configuración conservada", "", "| Parámetro | Valor |", "|---|---|"]
    for key in ("gamma", "alpha", "buffer_size", "batch_size", "hidden_neurons", "learn_after", "update_target_after", "nepisodes", "seed"):
        lines.append(f"| `{key}` | {config[key]} |")
    lines += ["", "## Verificación y alcance", "",
              "Se auditan los 28 entrenamientos (74 000 episodios): presupuestos, resultados finitos, identidad del reward y hashes de código y datos. Las tres semillas de la base y de alpha bajo reproducen exactamente los entrenamientos anteriores. El modelo actual coincide con el confirmado de la base/semilla 0 y el notebook ejecutado sigue válido. Los 18 tests existentes han pasado durante esta búsqueda.", "",
              "Es una búsqueda de factores individuales con criba en una semilla y confirmación en tres. No se prueban todas las combinaciones ni se demuestra un óptimo global. La completitud observada está limitada a los episodios evaluados. El error de seguimiento TWAP no se utiliza para seleccionar parámetros del ejercicio 1.", "",
              "Cada ejecución conserva entrenamiento, evaluación y checkpoint. Reproducir desde la raíz:", "", "```powershell",
              "..\\.venv\\Scripts\\python.exe notebooks/common/scripts/ej1_search.py",
              "..\\.venv\\Scripts\\python.exe notebooks/common/scripts/ej1_search_audit.py", "```", ""]
    (OUT / "RESUMEN.md").write_text("\n".join(lines), encoding="utf-8")
    print("Auditoria completa: 28 entrenamientos; reward y fuentes intactos; modelo y notebook confirmados.")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
