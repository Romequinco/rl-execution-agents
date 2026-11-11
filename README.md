<div align="center">

# RL Execution Agents

**Aprendizaje por refuerzo para ejecutar una orden CARE de compra sobre el libro de órdenes de Santander y Repsol**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![Estado](https://img.shields.io/badge/estado-en%20curso-D69E2E)

Máster MIAX · Aprendizaje por Refuerzo · Taller 3 · Noviembre 2026<br>
Óscar Romero Quincoces · Fernando Dapena Tauste · Daniel García López

[Taller 1 (tabular)](https://github.com/Romequinco/rl-trading-agents) · [Taller 2 (deep)](https://github.com/Romequinco/deep-rl-trading-agents)

</div>

---

## El problema

Un agente recibe una orden CARE de compra de N títulos y decide, cada 5 segundos, si comprar un título al ask o esperar.
El objetivo es completar la orden con el menor VWAP slippage frente al mercado. Los datos son snapshots del primer nivel del
libro de 13:00 a 17:25 (Santander y Repsol), y el punto de partida es el notebook `execution_lab.ipynb` del taller.

| Ejercicio | Puntos | Qué se pide |
|---|---:|---|
| 1 | 3,5 | Entorno y entrenamiento de RL completos, de principio a fin |
| 2 | 3,5 | Recompensa que replique un TWAP y ajuste motivado de los hiperparámetros |
| 3 | 1,5 | Exploración (ε) que baje según la mejora del agente, no según el número de episodios |
| 4 | 1,5 | Mejora libre: libro de órdenes, volumen pendiente y uso del ~90 % del tiempo, early stopping y checkpoints |

## Estructura

| Carpeta | Contenido |
|---|---|
| `docs/raw/` | Material del taller tal cual: transcripciones de las clases del 6 y 7 de noviembre (`26 11 06.pdf`, `26 11 07.pdf`) y `MIAX14/` (enunciado, teoría, `rl-material` y el repo `best-execution` con los datos) |
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

Los notebooks de la práctica y el registro de decisiones se añadirán en las siguientes sesiones.

## Licencia

Código propio bajo [MIT](LICENSE). El material de `docs/raw/` pertenece a sus autores y conserva su licencia original.
