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

La síntesis del material en `docs/teoria/`, el registro de decisiones y los notebooks de la práctica se añadirán en las siguientes
sesiones.

## Licencia

Código propio bajo [MIT](LICENSE). El material de `docs/raw/` pertenece a sus autores y conserva su licencia original.
