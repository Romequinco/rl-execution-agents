<div align="center">

# RL Execution Agents

**Aprendizaje por refuerzo para ejecutar una orden CARE de compra sobre el libro de órdenes de Santander y Repsol**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-unittest-2F9E6E)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![Estado](https://img.shields.io/badge/estado-completo-2F9E6E)

Máster MIAX · Aprendizaje por Refuerzo · Taller 3 · Noviembre 2026<br>
Óscar Romero Quincoces · Fernando Dapena Tauste · Daniel García López

[Taller 1 (tabular)](https://github.com/Romequinco/rl-trading-agents) · [Taller 2 (deep)](https://github.com/Romequinco/deep-rl-trading-agents)

</div>

---

## El problema

Hay que comprar N títulos de Repsol en un plazo fijo. Cada 5 segundos, un agente decide si compra un título o espera.
Aprende con aprendizaje por refuerzo (Double DQN) sobre datos reales del libro de órdenes. La referencia es un
**TWAP**, que compra al mismo ritmo todo el rato.

| Ejercicio | Puntos | Qué se pide |
|---|---:|---|
| 1 | 3,5 | Entorno y entrenamiento de RL completos, de principio a fin |
| 2 | 3,5 | Recompensa que replique un TWAP y ajuste motivado de los hiperparámetros |
| 3 | 1,5 | Exploración (ε) que baje según la mejora del agente, no según el número de episodios |
| 4 | 1,5 | Mejora libre: libro de órdenes, volumen pendiente y uso del ~90 % del tiempo, early stopping y checkpoints |

## Notebooks

| Notebook | Qué hace |
|---|---|
| [`00_datos_y_entorno`](notebooks/00_datos_y_entorno.ipynb) | Los datos, el simulador y las políticas de referencia |
| [`ej1_execution_lab`](notebooks/ej1_execution_lab.ipynb) | Ej. 1: entrenamiento, validación cada 250 episodios y selección del mejor modelo |
| [`ej2_twap_recompensas`](notebooks/ej2_twap_recompensas.ipynb) | Ej. 2: una recompensa que hace que el agente copie al TWAP |
| [`ej3_exploracion`](notebooks/ej3_exploracion.ipynb) | Ej. 3: explorar menos a medida que el agente mejora |
| [`ej4_extras`](notebooks/ej4_extras.ipynb) | Ej. 4: ideas A, B y C del enunciado y tres extras |

El código está en [`notebooks/common/`](notebooks/common) y las decisiones, con sus pruebas, en
[`docs/DECISIONES.md`](docs/DECISIONES.md). Los scripts de `scripts/` lanzan los entrenamientos largos.

```bash
pip install -r requirements.txt
cd notebooks
python -m unittest common.test_env common.test_learning common.test_protocol common.test_ej4 -v
jupyter lab
```

Los notebooks guardan CSV, figuras y modelos en `results/`; algunos reutilizan resultados existentes.
El ejercicio 1 vuelve a entrenar al ejecutar su celda de entrenamiento. Guarda el progreso en
`ej1_validation.csv`, el mejor modelo en `models/ej1_ddqn_best.pt` y el último en `models/ej1_ddqn_last.pt`.
Selecciona con datos de validación, priorizando órdenes completas, menos pendientes y mejor slippage,
por ese orden. `models/ej1_ddqn.pt` corresponde al seleccionado; test solo se evalúa después de elegirlo.

## Estructura

| Carpeta | Contenido |
|---|---|
| `docs/raw/` | Material del taller: `MIAX14/` con el enunciado, `rl-material` y el repo `best-execution` con los datos. Las transcripciones de las clases del 6 y 7 de noviembre (`26 11 06.pdf`, `26 11 07.pdf`) y las diapositivas (`applying_RL_MIAX.pdf`, `casos_practicos.pdf`) se citan en `docs/teoria` pero no se publican: se quedan en local por privacidad y derechos |
| `docs/raw/MIAX14/best-execution/` | Repo del taller ([Oteo95/best-execution](https://github.com/Oteo95/best-execution), LGPL-2.1): `workshop/execution_lab.ipynb` y `data/*.pkl` |
| `docs/raw/referencias/` | Copia de los Talleres 1 y 2 tal como están en GitHub, como referencia de estructura y decisiones |

## Síntesis del material (`docs/teoria/`)

Todo lo visto en clase, con cada afirmación citada a su fuente (minuto de la transcripción, página de diapositiva o celda).
Empezad por [`notebook_partida.md`](docs/teoria/notebook_partida.md).

| Documento | Contenido |
|---|---|
| [`notebook_partida.md`](docs/teoria/notebook_partida.md) | Enunciado ejercicio a ejercicio, evaluación, logística y todo lo que dijo el profesor de la práctica |
| [`00_repaso_rl.md`](docs/teoria/00_repaso_rl.md) | Componentes de RL, Markov, cuándo usar RL, TD, Q-learning, SARSA, Expected SARSA, Double Q |
| [`01_aproximacion_funcional_y_dqn.md`](docs/teoria/01_aproximacion_funcional_y_dqn.md) | Aproximación lineal, DQN, replay (ER/PER/CER), target, Double, Dueling, papers de DeepMind |
| [`02_diseno_de_entornos.md`](docs/teoria/02_diseno_de_entornos.md) | API gym/gymnasium, steps, estado, acciones, episodios, eficiencia y curriculum |
| [`03_recompensas_y_descuento.md`](docs/teoria/03_recompensas_y_descuento.md) | Diseño y calibración de recompensas; cómo elegir γ según el horizonte |
| [`04_exploracion_memoria_y_estabilidad.md`](docs/teoria/04_exploracion_memoria_y_estabilidad.md) | ε por mejora, α, buffer, catastrophic forgetting, checkpoints, semillas |
| [`05_profit_seeking.md`](docs/teoria/05_profit_seeking.md) | RL para algoritmos de inversión |
| [`06_algoritmos_de_ejecucion.md`](docs/teoria/06_algoritmos_de_ejecucion.md) | Orden CARE, dilema del trader, TWAP/VWAP/POV, VWAP slippage |
| [`07_rl_para_ejecucion.md`](docs/teoria/07_rl_para_ejecucion.md) | Estado, acciones y recompensas de un agente de ejecución; resultados del TFM del profesor |
| [`08_entorno_best_execution.md`](docs/teoria/08_entorno_best_execution.md) | El código del taller función a función, los datos y sus erratas |
| [`09_material_rl_notebooks.md`](docs/teoria/09_material_rl_notebooks.md) | Los 18 notebooks de `rl-material` |
| [`10_hoja_de_ruta_practica.md`](docs/teoria/10_hoja_de_ruta_practica.md) | Propuesta para abordar cada ejercicio, con opciones y riesgos |
| [`11_glosario.md`](docs/teoria/11_glosario.md) | Términos de RL y de ejecución |
| [`COBERTURA.md`](docs/teoria/COBERTURA.md) | Dónde quedó cada tramo, página y celda del material |

## Resultados

Medidos sobre 27 órdenes fijas de Repsol, salvo que se indique otra cosa. El «error» es cuántos títulos se separa el
agente, como máximo, del plan que debe seguir; cuanto menor, mejor.

| Ejercicio | Resultado |
|---|---|
| Referencia | El TWAP completa todas las órdenes y acaba al 98 % del tiempo, con un error de 0,90 |
| 1 | El agente entrena y ejecuta de principio a fin. Con la recompensa original del taller compra casi todo al principio: acaba al 11 % del tiempo |
| 2 | Con la recompensa de seguir el plan, copia al TWAP: error de 0,90, igual que el TWAP, y todas las órdenes completas |
| 3 | Explorar según la mejora ayuda cuando el entrenamiento es inestable (error final de 0,96 frente a 5,65), pero no cuando ya era estable |
| 4 · B | Con un plan que termina al 90 %, compra todo y acaba al **89 % del tiempo**, igual en 5 repeticiones y en test |
| 4 · C | Para solo al cumplir el objetivo (100-150 episodios en vez de 300) con el mismo resultado; se puede cortar y seguir exactamente igual |
| 4 · A | Mirando el mercado ahorra **+0,15 pb** por orden frente al TWAP; un placebo con datos desordenados no ahorra nada. Una regla simple ahorra más (+0,34) pero es casi tres veces más irregular |
| 4 · extras | Un pasillo 80-95 % en vez de plan fijo da libertad pero menos estabilidad. El agente no falla con órdenes que nunca vio ni empezando tarde |

## Licencia

Código propio bajo [MIT](LICENSE). El material de `docs/raw/` pertenece a sus autores y conserva su licencia original.
