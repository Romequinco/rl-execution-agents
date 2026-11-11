# El código del taller: `BestExecutionEnv`, `DDQNAgent` y datos

> **Fuentes:**
> - **LAB** = `docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`, celdas 0-21 (leído completo). Es el notebook que hay que completar y entregar.
> - **LABSTEP** = `docs/raw/MIAX14/best-execution/workshop/execution_lab-step_by_step.ipynb`, celdas 0-34 (leído completo): las mismas funciones del entorno fuera de la clase, en orden lógico.
> - **BE** = repositorio del taller `docs/raw/MIAX14/best-execution/` (original `Oteo95/best-execution`, licencia LGPL-2.1): `README.md`, `setup.py`, `src/agents/dqn.py`, `src/report/report.py`, `images/twap_model/twap_reproduction_dqn.png`, `images/twap_model/data_exploration/{orderbook_example.PNG, executions_example.PNG}`, `dockerfiles/devcontainers/{docker-compose.yml, python/dockerfile, python/requirements.txt}`, `.devcontainer/devcontainer.json`, `.gitignore`.
> - **DATOS** = `docs/raw/MIAX14/best-execution/data/{rep_data.pkl, san_data.pkl}` y `data/new_version/{rep_data.pkl, san_data.pkl, orderbook.pkl, executions.pkl}`. **No se han cargado** (deserializar un pickle de terceros ejecuta código): todo lo que se dice de ellos sale de un escaneo estático de los ficheros (cadenas, protocolo, tamaño), del texto del notebook y de las dos imágenes de `images/twap_model/data_exploration/`.
> - **T07** = transcripción automática de la clase del 07/11/2026 (`docs/raw/26 11 07.pdf`), profesor Jesús Sanz. Tramos: ~00:03:51-00:08:03 (Docker), ~00:10:19-00:18:03 (eficiencia), ~04:29:59-04:30:38 (ideas del ej. 4: rango de consumo de la duración), ~04:31:40-05:05:40 (presentación del código y de los datos).
> - De apoyo: **ENUNCIADO** (`docs/raw/MIAX14/ejercicios_taller_rl.docx`) para contrastar; **APP** p.81 (CER) (`docs/raw/MIAX14/applying_RL_MIAX.pdf`); **CASOS** p.51 (comparar el algoritmo con el TWAP fuera de *train*) y p.54 (gráfico de resultados) (`docs/raw/MIAX14/casos_practicos.pdf`). Metadatos (`language_info`) de los dos notebooks de `workshop/`.
>
> **Convenciones:** "(LAB celda 7)" = celda del notebook contando desde 0. "(T07 ~04:45)" = hora aproximada en la transcripción; todo lo que viene de ella es paráfrasis y lo que no se entiende bien va marcado *(dudoso)*. "(BE dqn.py)" = fichero del repositorio. Las funciones del entorno se citan con el número que les da su docstring: (1) `_generate_episode_params` ... (13) `action_sample`. Notación: $N$ = `episode_bins` (pasos del episodio), $n$ = `vol_care` (títulos de la orden CARE), $t$ = `state_pos`, $e_t$ = `exec_vol`, $a_t, b_t$ = `ask1`, `bid1` de la fila $t$, $m_t=(a_t+b_t)/2$, $C_t$ = `cumvol`. **Nota:** = contenido propio (cálculos, inferencias, correcciones); las erratas del código se señalan con su corrección razonada y se recogen en §15. Aquí se describe **qué** pide cada hueco y **qué cálculo** se espera, no se escribe la solución completa.
>
> Escrito el 10/11/2026.

---

## 1. Qué hay en el repositorio y para qué sirve cada pieza

**Idea.** El taller es un simulador de libro de órdenes "naive" (`BestExecutionEnv`) que el alumno completa, más un agente Double DQN ya escrito (`DDQNAgent`) y una función de gráficos (`plot_results`). El README lo presenta como la base de un proyecto de unas 10 horas para que los alumnos vean el proceso completo de un problema real de RL, no solo la teoría sobre simuladores de juegos estacionarios; avisa de que el simulador asume situaciones poco realistas, por ejemplo **impacto de mercado nulo** (BE README.md). Con ese simulador, la tarea principal es diseñar las recompensas y los modelos de RL hasta obtener algoritmos con un rendimiento parecido al de los algoritmos tradicionales; la imagen de ejemplo es un DQN que sigue una política TWAP (BE README.md, *Motivation of the project*).

**Cabecera del notebook.** LAB celda 0 da como autores a Alberto Oteo García y Jesús Sanz del Real. LAB celda 1 resume el taller: completar el entorno básico `BestExecutionEnv` hasta obtener un simulador de mercado "relativamente naive" pero funcional, y aplicar después algoritmos clásicos y basados en IA. El problema es optimizar el *timing* de las órdenes hijas de una orden CARE para comprar (vender) a los precios más bajos (altos) de la sesión, y el *baseline* es el habitual del sector, el **VWAP slippage**: la desviación, en puntos básicos, entre el VWAP del mercado y el del algoritmo, con $\mathrm{VWAP}=\sum_{i=1}^{T}p_i v_i \big/ \sum_{i=1}^{T} v_i$ (LAB celda 1).

**Nota:** la celda 1 dice que se optimiza el *timing* "para minimizar el impacto de mercado", pero el simulador supone impacto 0 (BE README.md; T07 ~04:44:55): lo único optimizable es el *timing* frente a la trayectoria de precios, que no depende del agente. La celda tampoco fija el signo del *slippage*; hay que definirlo uno mismo (lo natural para una compra: positivo si el algoritmo compra más barato que el mercado, como la recompensa por defecto de §6.6).

| Pieza | Ruta (en `docs/raw/MIAX14/best-execution/`) | Qué es |
|---|---|---|
| Notebook a entregar | `workshop/execution_lab.ipynb` | Explicación + clase `BestExecutionEnv` con huecos `"-----"` y `# TODO` + `TWAP` + parámetros + bucles de buffer, entrenamiento y evaluación (LAB) |
| Notebook paso a paso | `workshop/execution_lab-step_by_step.ipynb` | Las 13 funciones sacadas de la clase como código suelto, en orden lógico, para probarlas una a una (LABSTEP) |
| Agente | `src/agents/dqn.py` | `DDQNAgent` en TensorFlow/Keras (§9) |
| Gráfico | `src/report/report.py` | `plot_results(env, mode="max")` (§12) |
| Datos | `data/*.pkl`, `data/new_version/*.pkl` | Libro de órdenes de REP y SAN cada 5 s; libro a 3 niveles y ejecuciones de REP (§3) |
| Imagen objetivo | `images/twap_model/twap_reproduction_dqn.png` | Ejemplo de DQN que reproduce un TWAP; el ENUNCIADO la cita como lo que se pretende conseguir en el ej. 2 |
| Exploración de datos | `images/twap_model/data_exploration/*.PNG` | Capturas de `orderbook.pkl` y `executions.pkl` (§3.4) |
| Instalación | `setup.py`, `README.md`, `.devcontainer/`, `dockerfiles/` | §2 |

Cómo lo presentó el profesor (T07 ~04:33-04:35): el fichero que se entrega es `execution_lab`, que tiene "una clase que es tu entorno" con huecos tipo "complétame" y algunos *tips*; después se fijan epsilon y demás parámetros, se entrena y "os tiene que sacar algo", que es lo que se ve en `images/` (un TWAP). Reconoce que la clase "es un poco más difícil de manejar" porque hay que acceder a ella, y por eso preparó el `step_by_step`, que saca las funciones de la clase "en el orden lógico que se tiene que seguir": se completan allí, se copian a la clase y se añade `self.` donde haga falta para que las demás funciones accedan a ellas (T07 ~04:34:29-04:35:21). Para él son "como 10 pasitos, no son muchos más" (T07 ~05:01:06).

El README resume además el contexto de negocio (órdenes DMA frente a CARE, troceo en órdenes hijas, algoritmos de *slicing* y *smart order routing*, y el dilema del trader de Kissell y Glantz, 2003: ejecutar rápido impacta, ejecutar despacio expone al riesgo de precio) (BE README.md). Eso se desarrolla en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md); el planteamiento RL del caso, en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

**Nota:** el README escribe "Robert Kissel"; el autor es Robert **Kissell** (*Optimal Trading Strategies*, 2003).

---

## 2. Instalación y entorno de trabajo

### 2.1 Instalación local (la que documenta el README actual)
El README da dos caminos (BE README.md):
- **Dev Container** (requiere Docker): abrir el repo en VS Code y *Dev Containers: Reopen in Container*.
- **Local sin Docker**: entorno virtual con **Python 3.9-3.11 (recomendado 3.11)** y `pip install -e .` desde la raíz. Comandos: `python3.11 -m venv .venv` (Windows: `py -3.11 -m venv .venv`), activar con `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`) y `pip install -e .`. Instala las dependencias fijadas en `setup.py` (ruedas precompiladas para Windows, macOS Intel, macOS 12+ en Apple Silicon y Linux, sin compilador) y hace importable `src/` desde cualquier notebook (`from src.agents.dqn import DDQNAgent`).

`setup.py` (paquete `best-execution` 1.0.0, autor Jesús Sanz del Real) fija, con el motivo en comentario (BE setup.py):

| Paquete | Versión | Motivo indicado |
|---|---|---|
| Python | `>=3.9.2,<3.12` | TF 2.15 solo tiene ruedas para 3.9-3.11; 3.9.0 y 3.9.1 los excluye `cryptography` |
| `tensorflow` | 2.15.1 | Última con Keras 2, la API para la que está escrito `dqn.py` (TF ≥ 2.16 usa Keras 3) |
| `numpy` | 1.26.4 | TF 2.15 exige numpy < 2 |
| `pandas` | 2.0.3 | Los `data/*.pkl` se serializaron con pandas 1.x y `pickle.load` falla con ellos desde pandas 2.1 |
| `matplotlib` | 3.9.4 | Última serie con Python 3.9 |
| `jupyterlab` | 4.5.11 | Última 4.x con Python 3.9 |
| `argon2-cffi-bindings`, `cryptography` | 25.1.0, 48.0.1 | Dependencias transitivas cuyas versiones nuevas no tienen ruedas para macOS Intel |

Si ni `pip install -e .` ni el Dev Container funcionan, LAB celda 3 y LABSTEP celda 1 traen comentadas dos líneas (`import sys; sys.path.append("..")`) para añadir la raíz del repo al *path* e importar `src`. Los notebooks deben abrirse con `workshop/` como directorio de trabajo, porque leen los datos con rutas relativas `../data/...` (LAB celda 4; LABSTEP celda 2).

### 2.2 Dev Container y Docker
- `.devcontainer/devcontainer.json`: contenedor "Python" que se levanta con `docker-compose`, servicio `best-exe-devcontainer-python`, carpeta de trabajo `/workspace`, puerto 8080 reenviado, usuario `vscode`, y ajustes de VS Code (pylint, flake8, pytest, docstrings formato numpy, fin de línea `\n`) con las extensiones GitLens, autoDocstring y Python.
- `dockerfiles/devcontainers/docker-compose.yml`: construye con contexto la raíz del repo, monta el repo en `/workspace`, deja el contenedor vivo con un bucle `sleep` y define `PYTHONPATH=/workspace/src:/workspace`.
- `dockerfiles/devcontainers/python/dockerfile`: `FROM jesussanz/best_execution:v1` (imagen publicada por el profesor). Debajo, **comentada**, la receta de respaldo: imagen `mcr.microsoft.com/vscode/devcontainers/python:3.8`, instalación de `requirements.txt`, y `xvfb`, `python3-opengl` y `swig` (pantalla virtual para renderizar entornos gráficos).
- `requirements.txt` (pila antigua): matplotlib 3.6.1, jupyterlab 4.1.3, flake8, pylint, numpy 1.19.2, pandas 1.4.0, tensorflow 2.5.0.

Lo que dijo el profesor (T07 ~00:03:51-00:08:03): en el README están las instrucciones del devcontainer; además hace falta tener Docker instalado (él lo usa con el *backend* de Linux/WSL, la configuración que propone Docker por defecto en Windows). Explicó el contenedor con la analogía de los contenedores de carga: aísla todo lo que necesita la aplicación, como una mini máquina virtual pero sin crear un ordenador completo. Un alumno añadió que garantiza una copia exacta en todos los equipos y que es un entregable natural para *cloud* o un servidor. El Dockerfile parte de una imagen ya publicada; el otro (el comentado) era el de respaldo "por si acaso fallase", que instalaba Python 3.8 ("ya ha pasado un tiempo"), los requisitos y la parte de la ventana virtual que enseñó el día anterior (ver [09_material_rl_notebooks.md](09_material_rl_notebooks.md)). Para un notebook sencillo no hace falta; para todo lo que salga del notebook, sí.

Al presentar la práctica matizó que **no hace falta el contenedor**: otros alumnos la han hecho sin él; basta con poder leer el pickle con datos de bolsa y ejecutar (T07 ~04:31:43-04:32:11). Lo que dice que hace falta se transcribe como "Copa y Tor ... y un lumpipa más" *(dudoso: puede ser "Python" o "PyTorch", y "numpy/pandas")*. Dijo también que no hay requisitos sobre qué algoritmo usar (T07 ~04:32:11, transcripción muy ruidosa). Si falla la lectura del pickle, "o el devcontainer o ..." *(dudoso: la transcripción dice "Ellistol X Header"; por contexto puede ser `pip install -e .` o el `sys.path` de la celda 3)* (T07 ~04:35:45-04:35:53).

**Nota (incoherencia de versiones):** la imagen del Dev Container y su receta de respaldo corresponden a la pila antigua (Python 3.8, TF 2.5, pandas 1.4), mientras que `setup.py` exige Python ≥ 3.9.2 con TF 2.15 y pandas 2.0.3. No se ha descargado la imagen para comprobar qué trae; si es Python 3.8, `pip install -e .` dentro del contenedor fallaría por `python_requires`. Lo refuerza que los metadatos de LAB y de LABSTEP registran un *kernel* con **Python 3.8.18** (`language_info.version`), la pila antigua; los dos notebooks se guardaron sin salidas. El ENUNCIADO permite no reutilizar paquetes ni versiones, "incluye el uso de PyTorch en vez de TensorFlow"; la elección (y Python 3.13) se discute en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

### 2.3 Eficiencia del código del entorno
El profesor usó el taller como ejemplo de su principio de que los entornos, además de estables y sin sesgos, deben ser muy eficientes, porque en RL se itera paso a paso creando observaciones nuevas, mientras que en supervisado el DataFrame ya está construido (T07 ~00:10:37-00:12:08). Del notebook de best-execution, "de ahí saldrá la práctica", dijo que trabaja con pandas "por facilitarlo" (T07 ~00:12:19-00:12:31); con un problema sencillo vale pandas, y si no hay que optimizar (T07 ~00:17:48). La demo `%timeit` (DataFrame frente a array frente a diccionario, unos 45 ns en el más rápido) y el ejemplo de reconstruir el libro de Santander (de 20-30 min a 2-3 s) están en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

**Nota:** en `BestExecutionEnv` cada paso hace accesos `.iloc` de pandas (precio en (7), *slices* en (5)), pero el coste dominante del bucle es probablemente la llamada a la red de Keras en cada `act` y en cada `learn`, no pandas (estimación no medida). Si el entrenamiento es lento, conviene medir antes de reescribir (p. ej. pasar `ask1`, `bid1` y `cumvol` a arrays de numpy en `reset`).

---

## 3. Los datos

### 3.1 Ficheros, versiones y particiones

| Fichero | Tamaño | Protocolo pickle | Contenido | Días | Particiones |
|---|---|---|---|---|---|
| `data/rep_data.pkl` | 25,9 MB | 5 (versión antigua) | Repsol, 1.er nivel, 5 s | 120 (60+30+30) | train: 60 primeros días hábiles de 2018; val: 30 primeros de 2019; test: días hábiles 31-60 de 2019 |
| `data/san_data.pkl` | 21,4 MB | 5 | Santander, mismo formato | misma partición según el notebook | idem |
| `data/new_version/rep_data.pkl`, `san_data.pkl` | 25,9 / 21,4 MB | 4 | Los mismos datos, serializados con otra versión de pandas | idem | idem |
| `data/new_version/orderbook.pkl` | 33,6 MB | 4 | Repsol, libro hasta 3 niveles, 5 s | 80 | train: 40 primeros días hábiles de 2019; val: 41-60; test: 61-80 |
| `data/new_version/executions.pkl` | 11,0 MB | 4 | Repsol, todas las ejecuciones de mercado | 80 | idem |

*Fuente: LAB celda 2 (descripción y particiones); DATOS (tamaños, protocolo, nombres de columnas).* En clase dio el test como los días "30 60" de 2019 (T07 ~04:42:53); el notebook precisa días hábiles 31-60 (imprecisión oral sin consecuencias).

- Todas las sesiones van de **13:00 a 17:25** (LAB celda 2). **Nota:** con fotos cada 5 s eso son 265 min × 12 = 3180 intervalos, es decir, **3181 filas** por día si la rejilla está completa (la captura de `orderbook.pkl` empieza en 13:00:00 y acaba en 17:25:00).
- En `rep_data.pkl` los nombres de columna aparecen 120 veces cada uno en el escaneo: un DataFrame por día, 120 días; en `orderbook.pkl` y `executions.pkl`, 80. En `san_data.pkl` cada nombre de columna aparece solo 3 veces (el pickle memoriza las cadenas), así que no sirve para contar días; el notebook no distingue entre los dos activos (DATOS).
- **Nota (inferencias del escaneo, no verificadas):** el tipo `M8` (datetime64) aparece 124 veces en REP (120 días + 4) y 123 en SAN, lo que apunta a unos 119-120 días también en SAN. SAN contiene un bloque `i8` (enteros) que REP no tiene, así que alguna columna puede venir como entero en un activo y como *float* en el otro: conviene hacer `astype(float)` al cargar. En `executions.pkl` no aparece el *offset* `Milli`: su índice es irregular (una fila por ejecución), no una rejilla de 5 s.
- **Versión antigua frente a nueva.** El profesor explicó que hay dos juegos con exactamente los mismos datos: los antiguos para quien use una versión de pandas muy antigua y los nuevos, que "deberían funcionar para todos" (T07 ~04:38:51-04:39:13). **Nota:** el escaneo es coherente con eso: los antiguos referencian `pandas.core.internals.blocks.new_block` (pandas 1.x), los nuevos `pandas._libs.internals._unpickle_block` (pandas 2.x). `setup.py` fija pandas 2.0.3 precisamente para poder abrir los antiguos, que es lo que hace LAB celda 4 (`../data/rep_data.pkl`); con pandas ≥ 2.1 (y con Python 3.12-3.13) habrá que leer `../data/new_version/...`. No comprobado cargándolos.
- **Qué fichero usar.** El profesor: basta con uno de los dos activos, y train/val/test "los puedes juntar, me da igual" (T07 ~04:39:15-04:39:25). El ENUNCIADO dice que la práctica puede hacerse "para cualquiera de los 3 ficheros de datos" *(no precisa cuáles; lo más probable es REP, SAN y `orderbook.pkl`, que tiene las mismas columnas de primer nivel)*.

### 3.2 Estructura: diccionario → día → DataFrame
`pickle.load` devuelve un diccionario con claves `"train"`, `"val"`, `"test"` (LAB celda 4, `dict_.keys()`); cada partición es a su vez un diccionario **por día** cuyo valor es un DataFrame con índice temporal cada 5 s (LAB celda 5; T07 ~04:39:41-04:39:58). Es este diccionario de días (`dict_["train"]`) lo que recibe el entorno como `data`.

Columnas de `rep_data`/`san_data` (DATOS; LAB celda 2; T07 ~04:39:58-04:40:15 y ~04:42:23-04:42:34):

| Columna | Significado |
|---|---|
| `ask1` | Mejor precio de venta (primera posición del lado vendedor) |
| `bid1` | Mejor precio de compra |
| `vask1`, `vbid1` | Volumen disponible en esas primeras posiciones |
| `last` | Último precio ejecutado |
| `cumvol` | Volumen negociado acumulado en la sesión hasta esa foto |
| `date` | Día (columna de tipo objeto) |

El índice es un `DatetimeIndex` llamado `time`; su frecuencia se guarda como un *offset* en milisegundos (`Milli`), presumiblemente 5000 ms = 5 s (DATOS) *(dudoso: el valor no se ve en el escaneo)*. **Nota:** las claves de día son probablemente objetos `datetime.date` (el escaneo referencia la clase `date` del módulo `datetime`) *(inferencia)*.

Como los datos empiezan a las 13:00, `cumvol` **no empieza en 0**: ya incluye el volumen de la mañana (T07 ~04:43:18-04:43:30). En la captura de `orderbook.pkl` del 25/04/2019, la primera fila tiene `cumvol` = 4.365.397 y la última (17:25:00) 6.150.106: unos 1,78 millones de títulos de Repsol en la tarde (BE images/.../orderbook_example.PNG).

### 3.3 Convención temporal de cada foto (deducida de las capturas)
**Nota (inferencia propia comprobada con las dos capturas del mismo día, 25/04/2019):** cada fila etiquetada con la hora $t$ contiene el estado **al final** del intervalo $[t, t+5\,\text{s})$, es decir, el último valor dentro de ese intervalo (la convención por defecto de un `resample('5s').last()`):
- `executions_example.PNG`: hay una ejecución de 922 títulos a 14,870 a las 13:00:14 y otra de 44 a 13:00:19.
- `orderbook_example.PNG`: `cumvol` pasa de 4.365.397 (fila 13:00:05) a 4.366.319 (fila 13:00:10), +922, y a 4.366.363 (fila 13:00:15), +44. La ejecución de las 13:00:14 ya está en la fila 13:00:10. Al final del día, la ejecución de 242 títulos a las 17:25:00,0007 aparece en la fila 17:25:00 (6.149.864 + 242 = 6.150.106).
- Además, la fila 13:00:05 muestra `ask1` = 14,870 con `vask1` = 922, y en la fila 13:00:10 el *ask* ya ha subido a 14,875: la compra de 922 barrió exactamente ese nivel.

Consecuencias:
- $v_t = C_t - C_{t-1}$ (el `diff` de la fila $t$) es el volumen negociado **durante** $[t, t+5\,\text{s})$, y el precio vigente al empezar ese intervalo es la foto de la fila $t-1$. Por eso la pista del notebook, "mid de cada bin por el volumen del bin siguiente" (LAB celda 7, función (4)), es la alineación correcta: $m_{t-1}\,v_t$ (§6.4).
- El agente, que decide tras ver la fila $t$, ejecuta al `ask1` de esa misma fila: es el precio al inicio del siguiente intervalo con latencia 0, la asunción que el profesor llamó "tonta" y sesgada (§6.7).

### 3.4 Los ficheros extra: `orderbook.pkl` y `executions.pkl`
- `orderbook.pkl`: mismas claves; columnas `ask1-3`, `bid1-3`, `vask1-3`, `vbid1-3`, `last`, `cumvol`, `date`, por 5 s (DATOS; LAB celda 2). Sirve para características de profundidad o desbalanceo del libro.
- `executions.pkl`: una fila por ejecución, índice `time` con precisión de nanosegundos y columnas `sec`, `date`, `vol`, `price`, `nivel`, `bidask` (DATOS; captura `executions_example.PNG`). Según el profesor, contiene todas las ejecuciones desde la una, a qué precio y el nivel de agresión (si llegaron a la primera, segunda o tercera posición) (T07 ~04:44:06).
  - **Nota (inferencias sobre la captura, *dudosas*):** `sec` es un entero creciente (1.345.826 ... 2.696.181) que no son segundos; parece un número de secuencia de mensaje o de orden, y dos filas pueden compartirlo (dos ejecuciones de la misma agresión a las 13:01:05). `bidask` vale `A` o `B`; cruzando con el libro, la compra de 922 que barre el *ask* aparece como `B` y una venta contra el *bid* (44 a 14,870) como `A`, lo que sugiere que indica el **lado del agresor** (`B` = comprador, `A` = vendedor). Con solo dos casos, no es concluyente.
- El profesor insistió en que **no son necesarios**: "es abrir y ver qué pinta tienen" por si alguien quiere hacer algo con ellos (T07 ~04:43:45-04:44:30).
- **Errata de clase:** al describir `orderbook.pkl` dijo que incluye "3 o 4 posiciones" (T07 ~04:43:45); el fichero tiene exactamente **3 niveles** (`ask1-3`, `bid1-3`, `vask1-3`, `vbid1-3`; DATOS, LAB celda 2).
- **Errata de clase:** dudó de si `orderbook` era de Santander y no recordaba el activo de `executions` (T07 ~04:43:45-04:43:58); el notebook dice que ambos son **Repsol** (LAB celda 2), y la captura lo confirma: precios en torno a 14,9 €. **Nota (conocimiento externo, no sale de las fuentes):** son precios propios de Repsol en abril de 2019; Santander cotizaba entonces en torno a 4 €. **Nota:** la captura es del 25/04/2019, que cae hacia el día hábil 80 de 2019, es decir, en el tramo de test (61-80).

### 3.5 Asunciones de la práctica (lo que dijo el profesor)
- **Episodios:** no hace falta el día entero; pueden durar "90 min, 2 h", lo que puede durar un TWAP normal (T07 ~04:44:39-04:44:45). Poco después dice que el valor por defecto (400-600 pasos) es "menos de una hora", que "puedes cambiarlo" y que se pueden hacer de 2 o 3 h (T07 ~04:46:51-04:47:02). **Nota (ambigüedad, no errata):** el código usa 400-600 pasos (33-50 min) y el ENUNCIADO pone el ejemplo de 400 pasos, pero el profesor permite alargar los episodios a 90 min-3 h. Si se cambia, hay que motivarlo y recalibrar gamma y el tamaño del *buffer* (ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md)).
- **Órdenes hijas de 1 título**, para asumir **impacto 0**, y **coste de transacción 0** (T07 ~04:44:55; ENUNCIADO, ej. 2).
- **Steps de 5 s:** 12 por minuto, **720 por hora** (T07 ~04:45:14-04:45:47).
- **Flexibilidad:** con hijas de 1 título, una orden de 720 títulos en 720 pasos obliga a ejecutar en todos los *bins* ("no tengo flexibilidad en absoluto"); con 360, uno sí y uno no (ratio 0,5) (T07 ~04:46:11-04:46:39). Para darle "un poquito de juego" quiere ejecutar más o menos en **1 de cada 10** pasos: multiplicar los 720 por un factor que la transcripción da como "0 9" daría una orden de **65 títulos**, es decir, 65 barritas en el gráfico final (T07 ~04:47:10-04:47:36) *(dudoso: por el resultado, 65 ≈ 720 × 0,09 = 64,8, lo más probable es que dijera "cero coma cero nueve" y que la transcripción lo recortara; con exactamente el 10 % saldrían 72)*. El código usa un porcentaje uniforme en [7,5 %, 12,5 %] (§6.1).
- **Errata de clase:** dice "serían una orden hija de 65 títulos" (T07 ~04:47:29) y, más adelante, "quiero un tamaño de esta orden hija" (T07 ~04:50:26); en los dos casos se refiere a la orden **CARE**: las hijas son de 1 título.
- En una ejecución de ejemplo salieron "114 / 214" pasos y una orden de 55 títulos (T07 ~04:48:09-04:48:16) *(dudoso: con $n=55$ y el ratio entre 0,075 y 0,125, $N$ tiene que estar entre 440 y 600, así que 114 o 214 no son posibles; quizá dijo 514)*. Justo antes dice que "aquí había unos ascensores y tal" *(dudoso: casi seguro "unos asserts", los de (1), que comprueban precisamente $N$ y $n$; §6.1)* (T07 ~04:48:09).

---

## 4. `BestExecutionEnv`: atributos uno a uno

El constructor es `BestExecutionEnv(data, look_back=60)` (LAB celda 7). El profesor los repasó en clase (T07 ~05:00:01-05:00:52).

| Atributo | Valor inicial | Lo fija | Significado (docstring y clase) |
|---|---|---|---|
| `data` | argumento | `__init__` | Diccionario {día: DataFrame} de una partición. **Errata:** el docstring dice "Dataframe" |
| `look_back` | 60 | `__init__` | Ventana de *bins* previos al inicio del episodio para calcular características "roladas" en $t=0$; es el rango máximo de las ventanas |
| `episode_bins` | `None` | (1) | $N$, número de pasos del episodio, entero en [400, 600] |
| `episode_full_len` | `None` | (1) | `episode_bins + look_back` |
| `vol_care` | `None` | (1) | $n$, títulos de la orden CARE ("50, 60 títulos, algo así", T07 ~05:00:10) |
| `actions_fn` | `{0: _do_nothing, 1: _agg_action}` | `__init__` | Diccionario acción → función que la evalúa y devuelve la recompensa: "comprar o mantener, se acabó" (T07 ~05:00:15) |
| `n_actions` | `len(actions_fn)` = 2 | `__init__` | Número de acciones |
| `n_features` | `_detect_num_feat()` = 2 | `__init__` | Tamaño de la observación; se detecta generando un episodio de prueba, para adaptarse solo si se cambia `observation_builder` |
| `episode` | `None` | (2) | DataFrame del episodio: $N$ filas desde la hora de inicio |
| `episode_full` | `None` | (2) | `episode` precedido de los `look_back` *bins* anteriores |
| `episode_vwap` | `None` | (4) | VWAP de mercado de **todo** el episodio (se precalcula) |
| `market_ep_vol` | `None` | (4) | Títulos negociados por el mercado en cada *bin* del episodio |
| `state_pos` | 0 | `step` | $t$, paso actual |
| `exec_vol` | 0 | (7)/(9) | $e_t$, títulos ejecutados por el algoritmo |
| `algo_vwap` | 0 | `_reset_env_episode_params` | Se inicializa en el reset pero ninguna función lo actualiza (el VWAP del algoritmo se guarda en `algo_vwap_hist`) |
| `actions_hist` | `[]` | nadie | Acciones tomadas. El profesor presentó los históricos de recompensas y acciones "por si queréis plotearlos después" (T07 ~05:00:46-05:00:52), pero **errata:** ninguna función lo rellena (E7, §15) |
| `algo_vwap_hist` | `[]` | (7)/(9) | VWAP del algoritmo tras cada paso (no está en el docstring) |
| `market_vwap_hist` | `[]` | `step` | VWAP de mercado acumulado en cada paso |
| `reward_hist` | `[]` | `step` | Recompensa de cada paso (sin la de final de episodio) |
| `price_hist` | `[]` | (7)/(9) | Precio de cada paso (0 si no se ejecuta) |
| `vol_hist` | `[]` | (7)/(9) | Títulos ejecutados en cada paso (1 o 0) |

El profesor explicó el `look_back` así: si el episodio empieza a las 14:00, `episode` es el DataFrame de 14:00 a 15:00, y `episode_full` es el episodio "más 60 *bins* anteriores", porque a veces se quiere información previa para calcular características con ventana (T07 ~04:59:13-04:59:54). Con 5 s por *bin*, 60 *bins* son **5 minutos**.

---

## 5. Orden de las llamadas

```
__init__(data, look_back)
 └─ _detect_num_feat()
     ├─ _reset_env_episode_params()
     ├─ _generate_episode()            (2)
     │   └─ _generate_episode_params() (1)
     └─ observation_builder()          (3)   → n_features = len(obs)

reset()
 ├─ _reset_env_episode_params()
 ├─ _generate_episode()                (2) → (1)
 ├─ _compute_episode_market_feat()     (4)
 └─ return observation_builder()       (3)

step(action)
 ├─ market_vwap = _compute_market_vwap()          (5)
 ├─ reward = actions_fn[action]()
 │    ├─ 1 → _agg_action()   (7) → _compute_algo_vwap() (8) → _compute_reward(price, 1) (6)
 │    └─ 0 → _do_nothing()   (9) → _compute_algo_vwap() (8) → _compute_reward(0, 0)     (6)
 ├─ market_vwap_hist.append(market_vwap); reward_hist.append(reward)
 ├─ state_pos += 1
 ├─ done = _compute_done()                         (11) → _compute_stop_conditions() (10)
 ├─ si done: reward += _compute_done_reward()      (12); return None, reward, True, {}
 └─ si no:   return np.array(observation_builder()), reward, False, {}

agente.act(s) → (exploración) env.action_sample()  (13)
```

*Fuente: LAB celda 7.* Lo que dijo el profesor de este orden (T07 ~04:48:26-04:52:04 y ~04:56:51-04:57:52): primero se inicializa la partida con `reset`, que genera la partida, pone las variables a 0 o vacías y precarga lo que se va a necesitar después; el bucle de entrenamiento es el típico (`act` con epsilon-greedy y luego `step`). Dentro de `reset`: se generan los parámetros del episodio (tamaño y duración), se construye el episodio (activo, día y hora de inicio, "un día a la una y otro a las dos") y se precalcula el precio medio de mercado de ese periodo, porque puede servir para la recompensa y, como el algoritmo no impacta en el mercado, se puede dejar calculado de antemano. `step` es "la función más importante": recibe la acción, la evalúa para tener la recompensa (y guardarla si se quiere), avanza una posición en el DataFrame, decide si ha terminado según las condiciones de parada y devuelve la nueva observación, la recompensa, si ha terminado y un diccionario de información.

**Nota:** `step` devuelve 4 valores (API clásica de gym), no los 5 de gymnasium (`terminated`, `truncated`); ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md). Aquí `done` mezcla "orden completada" y "tiempo agotado", pero como el tiempo restante forma parte del estado, el fin por tiempo es un estado terminal genuino del MDP y no un corte artificial: tratarlo como terminal (sin *bootstrap*) es correcto. **Nota:** el profesor dijo que `reset` elige el activo (SAN o REP) y una hora de inicio "a la una, a las dos"; el código recibe **un solo** diccionario de datos y elige cualquier *bin* como inicio (§6.2).

---

## 6. Las 13 funciones del entorno, en orden

En la clase el orden físico no coincide con la numeración (por ejemplo, (8) aparece antes que (6)); aquí se sigue la numeración, que es también la del LABSTEP (secciones 1 a 13, con 7-8 juntas). Para cada una: qué hace, qué pide el `TODO`, qué cálculo se espera y qué trampas tiene.

### 6.1 (1) `_generate_episode_params`: tamaño y duración de la orden
- **Qué hace:** define la orden del episodio (LAB celda 7; LABSTEP celda 6).
- **TODOs:** `episode_bins` = "Int aleatorio entre 400 y 600 como un objeto numpy"; `pct_bins` = "Float aleatorio entre 0.075 y 0.125" (uniforme); `vol_care` = "Int multiplicación pct_bins y episode_bins".
- **Cálculo esperado:** $N\sim\mathcal{U}\{400,\dots,600\}$, $\rho\sim\mathcal{U}[0{,}075;\,0{,}125]$, $n=\lfloor \rho N\rfloor$. Después, `episode_full_len = N + look_back`.
- **Asserts que validan el hueco:** $400\le N\le 600$, $\lfloor 0{,}075N\rfloor \le n \le \lfloor 0{,}125N\rfloor$ e `isinstance(vol_care, int)`.
- **Interpretación:** el docstring llama a $\rho$ "porcentaje de steps en el que hay que ejecutar una orden para cubrir la orden care": con hijas de 1 título, hay que ejecutar en uno de cada $1/\rho$ ≈ 8-13 pasos.
- **Cifras (Nota):** $n$ va de $\lfloor 400\cdot0{,}075\rfloor=30$ a $\lfloor 600\cdot0{,}125\rfloor=75$ títulos, unos 50 de media; el episodio dura de 33 min 20 s a 50 min.
- **Trampas (Nota):** el límite superior de `np.random.randint` es exclusivo (para incluir 600 hay que pedir hasta 601); el `assert isinstance(..., int)` exige un `int` de Python, no un `np.int64`, así que la conversión final a `int(...)` es necesaria. Además, `vol_care` hay que **truncarlo** (`int(...)`), no redondearlo: con `round` puede fallar `assert vol_care <= int(N*0.125)`. Por ejemplo, con $N=406$ el límite es $\lfloor 50{,}75\rfloor=50$, y $\rho=0{,}1247$ da $\rho N=50{,}63$, que redondea a 51. El profesor comentó que estos son los "params" de la orden y que lo que se complete en el LABSTEP es lo que luego va a la clase (T07 ~04:36:05-04:37:01).

### 6.2 (2) `_generate_episode`: día y hora de inicio
- **Qué hace:** llama a (1) y, en un bucle `while length_episode != episode_full_len`, elige día y hora hasta obtener un episodio completo.
- **TODOs:** `selected_day` ("Usa np.random.choice y los días data.keys"); `data_day` = el DataFrame de ese día; `init_time` = "una hora de inicio aleatoria"; `episode_full` = filas de `data_day` entre `initial_position` y `final_position`; `episode` = filas entre `hour_pos` y `final_position`.
- **Lógica ya escrita:** `hour_pos = data_day.index.get_loc(init_time)`, `initial_position = hour_pos - look_back`, `final_position = hour_pos + episode_bins`; si `initial_position < 0` se repite (no hay *look back* suficiente); la longitud de `episode_full` se compara con `episode_full_len`, de modo que si el episodio se sale del final del día (el *slice* por posición queda truncado) también se repite. Es un muestreo por rechazo.
- **Cálculo esperado:** selección por **posición** (`iloc`) de los rangos $[h-60,\,h+N)$ y $[h,\,h+N)$.
- **Cifras (Nota):** con 3181 filas por día, las posiciones de inicio válidas son $60\le h\le 3181-N$: el inicio más temprano es 13:05 y el más tardío 16:35:05 para $N=600$ o 16:51:45 para $N=400$. Con la hora de inicio uniforme sobre todas las filas, se acepta a la primera en torno al 82 % de las veces ($N=500$).
- **Trampas (Nota):** `np.random.choice(data.keys())` falla, porque `dict_keys` no es una secuencia; hay que pasar `list(data.keys())`. `init_time` tiene que ser una **etiqueta existente** de `data_day.index`, porque `get_loc` busca la etiqueta exacta: lo directo es `np.random.choice(data_day.index)` (uniforme sobre las filas, que es lo que supone el cálculo del 82 %); una hora aleatoria generada fuera de la rejilla de 5 s (o que caiga en un hueco del día) da `KeyError`. Elegir directamente una posición entera en el rango válido evita tanto el `KeyError` como el bucle de rechazo. Si se quieren mezclar SAN y REP (así describió el profesor el `reset`, "te selecciona tu activo, pues Santander o Repsol", T07 ~04:50:36, aunque el código no lo hace, E24, y él mismo dijo que basta con uno de los dos, T07 ~04:39:21), no basta con unir los dos diccionarios: las claves son días y coincidirían; hay que usar claves compuestas (activo, día).

### 6.3 (3) `observation_builder`: el estado
- **Qué hace:** construye el vector de observación. Por defecto, "primera característica: tiempo restante en porcentaje; segunda: volumen restante en porcentaje".
- **TODO:** `time_left`, `vol_left` ("Utiliza episode_bins, state_pos, exec_vol, vol_care"); el código ya hace `obs = np.array([time_left, vol_left])`.
- **Cálculo esperado:** $T_t = 1 - t/N$, $V_t = 1 - e_t/n$ (fracciones en [0,1]; "porcentaje" en el docstring). En el `reset` vale $[1, 1]$.
- **Lo que dijo el profesor:** la primera observación son tiempo y volumen restantes; para el ej. 2 (TWAP) bastan esas dos, sin características de mercado, porque "tienen que ir de la mano" para que al final salga la diagonal (T07 ~04:52:04-04:52:47). Qué más meter en el estado para el ej. 4 está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).
- **Trampa (Nota):** `_detect_num_feat` llama a (3) tras (2) pero **sin** (4); si se añaden al estado características que dependen de `episode_vwap` o `market_ep_vol`, el constructor fallará. Habría que añadir (4) en `_detect_num_feat`, o calcular esas características solo a partir de `episode_full`.

### 6.4 (4) `_compute_episode_market_feat`: VWAP y volumen de mercado del episodio
- **Qué hace:** precalcula en el `reset` el VWAP de mercado del episodio completo y el volumen por *bin*. Docstring: "Como no tenemos las ejecuciones de mercado, asumimos que el precio es el mid price de cada step".
- **TODOs:** `mid` con `ask1` y `bid1` de `episode` ("Opcional: utiliza un precio más realista para el mkt VWAP"); `market_ep_vol` (después el código fuerza `market_ep_vol.iloc[0] = 0`, por lo que tiene que ser una `Series` y su primer valor no existe: el `diff` de la primera fila); `cum_vol` = volumen acumulado del mercado en el episodio; `episode_vwap` con la pista "mid de cada bin por el volumen del bin siguiente (usa `.values` para que pandas no alinee por índice)".
- **Cálculo esperado:** con $v_t = C_t - C_{t-1}$ y $v_0 := 0$,

$$
\mathrm{VWAP}^{mkt}_{ep}=\frac{\sum_{t=0}^{N-2} m_t\,v_{t+1}}{\sum_{t=1}^{N-1} v_t}.
$$

  El `.values` es necesario porque, si se multiplican dos `Series` desplazadas, pandas las alinea por índice y deshace el desplazamiento.
- Es la fórmula del VWAP de LAB celda 1, $\sum p_i v_i / \sum v_i$, con $p$ = *mid*. Según §3.3, el emparejamiento $m_t\,v_{t+1}$ es el correcto: el volumen del intervalo siguiente se valora al precio vigente cuando empieza.
- **Notas:** (a) poner $v_0 = 0$ pierde el volumen del primer *bin*; se podría recuperar con la última fila de `episode_full` antes del episodio. (b) Alternativas de precio más realistas: `last`, la media de los *mids* de dos fotos consecutivas (lo que propuso el profesor, §6.7) o, para REP 2019, `executions.pkl`. (c) Precalcular esto solo es legítimo porque el agente no impacta en el mercado (T07 ~04:51:01-04:51:16).

### 6.5 (5) `_compute_market_vwap`: VWAP de mercado hasta el paso actual
- **Qué hace:** se llama al principio de cada `step` y su valor solo se guarda en `market_vwap_hist`; por defecto no entra en la recompensa.
- **TODOs:** `mid_p` y `mkt_p` = "un precio para el volumen ejecutado por el mercado en cada step" (pista: "fíjate en (4)"); `v` = volumen por *bin*; `p_arr`, `v_arr` = los valores hasta el paso actual incluido (el LABSTEP los da en comentario: `mkt_p.values[:state_pos + 1]`, `v.values[:state_pos + 1]`); `sum_vol`; si `sum_vol == 0`, "devuelve el último precio hasta el step"; si no, el VWAP acumulado.
- **Cálculo esperado:** $\mathrm{VWAP}^{mkt}_{\le t} = \sum_{k\le t} p^{mkt}_k v_k \big/ \sum_{k\le t} v_k$.
- **Trampa (Nota):** para ser coherente con (4), el precio del volumen $v_k$ debe ser $m_{k-1}$, es decir, `mkt_p` = *mid* desplazado un *bin*. Pero entonces `mkt_p[0]` es `NaN` y, aunque $v_0=0$, $\mathrm{NaN}\cdot 0=\mathrm{NaN}$ contamina la suma: hay que rellenar ese primer valor. La versión desplazada ($m_{k-1}v_k$) usa el mismo emparejamiento que (4) y que `stats_df`, que hace $a_t\,v_{t+1}$ con `diff().shift(-1)` (§12.2), salvo *ask*/*mid* y un desfase de una fila en el acumulado; usar $m_k v_k$ rompe la coherencia con ambos. Con la convención de §3.3 (inferencia, depende de ella), $v_t$ ya se ha negociado cuando el agente ve la fila $t$, así que incluirlo en el paso $t$ no es mirar al futuro; lo sería usar $v_{t+1}$ o `episode_vwap`, que incluye todo el episodio.
- **Primer paso (Nota):** si `v` se toma de `market_ep_vol`, en $t=0$ `sum_vol` vale siempre 0 (por `market_ep_vol.iloc[0] = 0`), y la función devuelve "el último precio hasta el step", es decir, el precio de la fila 0. Esa rama no es un caso raro, sino el primer paso de **todos** los episodios; y si `mkt_p` es el *mid* desplazado, ese "último precio" es justo el `NaN` del párrafo anterior.
- El profesor: calcular el VWAP de mercado parcial "para ver si lo estás haciendo mejor" se puede usar en la recompensa, pero **no es necesario**, porque para replicar un TWAP las recompensas no tienen que depender del rendimiento (T07 ~04:57:55-04:58:18).

### 6.6 (6) `_compute_reward(price, vol)`: la recompensa por defecto
- **Docstring:** "el reward es el ratio de la diferencia entre el episode_vwap y el precio de la acción tomada, dividido entre episode_vwap".
- **TODOs:** recompensa cuando `vol == 0` (libre); recompensa cuando `vol > 0` (pista: `episode_vwap` y `price`); opcional: usar `self` y eliminar los parámetros.
- **Cálculo esperado (por defecto):** $r_t = \dfrac{\mathrm{VWAP}^{mkt}_{ep}-p_t}{\mathrm{VWAP}^{mkt}_{ep}}$ si se ejecuta; un valor fijo (lo natural, 0) si no.
- Es **la función que hay que tocar en el ej. 2** (T07 ~04:58:27), y el profesor insistió en que el TWAP se haga "por recompensas", no con la regla (T07 ~04:54:30-04:54:36).

**Nota (lo que hace de verdad la recompensa por defecto):**
1. Con hijas de 1 título, el VWAP del algoritmo es la media simple de sus precios $\bar p$, así que la suma de recompensas de un episodio completo es
$$\sum_t r_t = n\,\frac{\mathrm{VWAP}^{mkt}_{ep}-\bar p}{\mathrm{VWAP}^{mkt}_{ep}},$$
$n$ veces el *VWAP slippage* (en fracción; ×10⁴ para pasarlo a pb), con signo positivo si se compra por debajo del mercado.
2. **Magnitud:** en Repsol a ~15 € el *tick* de 0,005 equivale a unos 3,3 pb; las recompensas son del orden de $10^{-4}$, y sus cuadrados ($10^{-8}$) hacen que la pérdida MSE se imprima como 0 con `np.round(mse)` (§11). Conviene reescalar (por ejemplo, a pb); ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).
3. **Sesgo de signo:** el agente compra al *ask* y el VWAP de referencia se calcula con el *mid*, así que en promedio cada ejecución recibe una recompensa negativa de aproximadamente medio *spread* (unos −1,7 pb con un *spread* de un *tick*). Si no ejecutar vale 0 y no hay penalización por volumen pendiente (`_compute_done_reward` devuelve 0), la política óptima con esta recompensa es **no comprar**, y el episodio termina por tiempo con la orden casi sin ejecutar.
4. **Factibilidad:** `episode_vwap` es el VWAP **final** del episodio, información futura que no está en el estado $[T, V]$, de modo que la recompensa no es predecible desde el estado (ver la discusión de recompensas factibles en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md)).

Por todo ello la recompensa por defecto no produce un TWAP. Las opciones para el ej. 2 (desviación del plan TWAP, penalización terminal por volumen pendiente, etc.) están en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

### 6.7 (7) `_agg_action`: comprar 1 título agresivamente
- **Docstring:** "Acción agresiva de compra de un título a precio de `episode['ask1']`. Devolvemos el reward asociado a esa acción."
- **TODOs en orden:** `price` = `ask1` en `state_pos`; guardar `price` en `price_hist`, sumar 1 a `exec_vol` y añadir 1 a `vol_hist` (el código ya fija `vol = 1`); `algo_vwap` con "la función apropiada", es decir (8); guardarlo en `algo_vwap_hist`; `reward` con (6).
- **Precio de ejecución (lo que explicó el profesor):** el notebook ejecuta con la información **del comienzo del *bin*** y asumiendo **latencia 0**; es un precio "tonto, que está sesgado" (T07 ~05:02:05-05:03:39). En esos 5 s el mercado ha podido negociar, por ejemplo, 72 (o 32) títulos *(dudoso: la transcripción da primero 32 y una línea después 72)*, y no se sabe a qué precio porque solo se tienen fotos. Mejor aproximar lo que pasa entre fotos: precios medios entre una foto y la siguiente, la media de esos medios o, si se asume una latencia (por ejemplo 2 s), una interpolación en función de ella. Justificó el precio medio así: entre una foto y la siguiente, en media unas operaciones se habrán ejecutado en un lado del libro y otras en el otro, así que el precio medio es una aproximación razonable (T07 ~05:04:18-05:04:44). Mencionó también que hay un fichero con el precio exacto "que no hace falta que lo utilicéis" *(dudoso; probablemente `executions.pkl`)* (T07 ~05:03:46-05:03:54). "Da exactamente igual" lo que se elija, pero ese tipo de decisiones son las asunciones de impacto que hay que razonar al programar el algoritmo cuando no se tiene información ejecución a ejecución (T07 ~05:03:46-05:05:30). Dice además que en muchos algoritmos se piden "dos precios", uno más o menos realista *(dudoso)* (T07 ~05:02:00-05:02:11).
- **Nota:** el argumento del precio medio vale para el flujo de **mercado**, que es mixto, pero no para el agente, que siempre compra al *ask*; de ahí el sesgo de ½ *spread* por compra de §6.6.
- **Nota:** el código ignora `vask1`; con hijas de 1 título siempre hay volumen en la primera posición, pero si se aumentara el tamaño de las hijas habría que comprobar la liquidez y recorrer niveles (`orderbook.pkl`).

### 6.8 (8) `_compute_algo_vwap`: VWAP del algoritmo
- **TODO:** `p_arr`, `v_arr` desde `price_hist` y `vol_hist`, y el cociente.
- **Cálculo esperado:** $\mathrm{VWAP}^{algo}_{\le t}=\sum_i p_i q_i \big/ \sum_i q_i$, con $q_i\in\{0,1\}$.
- **Trampas (Nota):** `price_hist` contiene ceros en los pasos sin ejecución, así que una media simple de `price_hist` sería incorrecta; hay que ponderar por `vol_hist`. Si todavía no se ha ejecutado nada, $\sum q_i=0$ y el cociente es 0/0 (NaN con aviso en numpy, `ZeroDivisionError` con escalares de Python). El agente lo evita forzando la compra en el paso 0 (§9.4), pero en la evaluación esa protección falla (§12, errata E5).

### 6.9 (9) `_do_nothing`: no comprar
- **TODO:** "Repite el proceso de `_agg_action`" con precio y volumen 0: añadir 0 a `price_hist` y a `vol_hist`, recalcular y guardar `algo_vwap`, y devolver `_compute_reward(0, 0)`.
- **Trampa (Nota):** es imprescindible que esta acción también añada entradas a `price_hist`, `vol_hist` y `algo_vwap_hist`. `stats_df` asocia `algo_vwap_hist` y `vol_hist` a las primeras filas de `episode.index`: si solo se registran los pasos con compra, la construcción falla por longitudes distintas o, si ambas listas tienen la misma longitud, todas las compras se dibujan amontonadas al principio del gráfico.

### 6.10 (10) `_compute_stop_conditions`
- **Docstring:** "Return: tiempo agotado, orden completada" (en ese orden).
- **Cálculo esperado:** `is_bins_complete` = el paso actual (ya incrementado en `step`) ha llegado a $N$; `is_ord_complete` = $e_t \ge n$.
- **Trampa (Nota):** si el tiempo agotado se escribe con `>` en vez de `>=`, en `state_pos` $=N$ el episodio no termina y el estado tiene $T=0$. Si la política compra en ese paso extra, `episode['ask1'].iloc[N]` lanza `IndexError`; si no compra, `_do_nothing` no lee precios y el *slice* de (5) no falla, así que el episodio dura $N+1$ pasos sin error visible (con un paso fuera del episodio). El fallo es intermitente y depende de la política; usar `>=`.

### 6.11 (11) `_compute_done`
- **TODO:** obtener las condiciones con (10), desempaquetarlas y devolver `done = True` si se cumple **cualquiera** de ellas. El profesor mencionó como condición de parada "si tu volumen ha terminado o no" (T07 ~04:57:28); la de tiempo agotado sale del *docstring* de (10) (LAB celda 7).

### 6.12 (12) `_compute_done_reward`
- **TODO:** "Free style"; por defecto devuelve 0. `step` la suma a la recompensa del último paso.
- **Nota:** es el lugar natural para una penalización por volumen pendiente o por terminar demasiado pronto (ej. 2 e idea B del ej. 4: evitar que quede volumen pendiente y que la orden consuma aproximadamente el 90 % de la duración). Esta recompensa terminal **no** entra en `reward_hist` (se añade después de registrar), pero sí en el `cum_reward` del bucle y en la transición que se guarda en el *buffer*. Calibración en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

### 6.13 (13) `action_sample`
- **TODO:** acción aleatoria con claves de `actions_fn`; en comentario, `np.random.choice([0, 1], p=[1-p, p])`. "Opcional: ¿Qué distribución de prob es mejor para la exploración?"
- **Nota (por qué importa la pregunta):** con $p=0{,}5$ y $n\approx 0{,}1N$, una política aleatoria completa la orden en unos $2n\approx 0{,}2N$ pasos (≈100 de 500): los episodios aleatorios terminan con $T\approx 0{,}8$, y el *buffer* inicial y la exploración casi no visitan estados de final de episodio. Con $p \approx n/N$ (disponible dentro del entorno como `vol_care / episode_bins`) la política aleatoria va, en media, al ritmo de un TWAP y recorre todo el episodio. Ver la discusión de ponderar acciones aleatorias en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

### 6.14 Funciones sin número: `_detect_num_feat`, `_reset_env_episode_params`, `reset`, `step`, `stats_df`
Ya están escritas. `_reset_env_episode_params` vuelve a poner a `None`, 0 o lista vacía todo lo del episodio (no toca `episode_bins` ni `vol_care`, que reescribe (1)). `stats_df` se describe en §12. Como `__init__` llama a `_detect_num_feat`, **el entorno no se puede ni instanciar hasta completar (1), (2) y (3)**.

**Nota (efectos del episodio de prueba del constructor):** `__init__` llama a `_detect_num_feat` **antes** de poner `episode`, `episode_full`, `state_pos`, etc. a `None`/0/vacío. Tras construir el entorno no hay episodio cargado: hay que llamar a `reset()` antes del primer `step()` (si no, (5) falla sobre `episode = None`). En cambio, `episode_bins`, `vol_care` y `episode_full_len` quedan con los valores del episodio de prueba. Ese episodio consume números aleatorios, lo que importa si se fijan semillas para reproducir resultados.

---

## 7. El agente TWAP (celdas 8-9)

LAB celda 8 define el TWAP como algoritmo que trocea la orden en hijas equidistantes en el tiempo durante el horario elegido por el cliente. Advierte de que su patrón es fácil de adivinar y menciona las defensas habituales (aleatorizar tamaño y tiempo entre envíos, limitar la participación a un porcentaje del volumen, no agredir más de una posición del libro). Para el taller se pide un TWAP **sin modificaciones**: hijas equidistantes, sin mecanismos contra la detección.

```python
class TWAP(DDQNAgent):
    def act(self, s): # [T, V]
        # TODO: Configura un TWAP determinista utilizando s
```

Regla que dio el profesor (T07 ~04:53:20-04:54:20): con los dos parámetros del estado hay que poder configurar un TWAP **determinista**: si vas por delante del plan de ejecución, no ejecutes; cuando estés por debajo o igual, lanza otra hija. Es "una regla de un *if*". Como los escenarios son finitos, incluso se podría aprender con Q-learning tabular: dos estados ("volumen restante por encima de $T$" o "igual o por debajo") por dos acciones (ejecutar o no), una matriz 2×2 (T07 ~04:54:50-04:55:24).

Justo antes de enunciarla dice "Esto es importante. Esto no es necesario" (T07 ~04:53:18-04:53:20) *(dudoso: no se sabe a qué señala en pantalla; puede referirse a la regla determinista, porque enseguida añade "quiero que lo hagáis también por recompensas", T07 ~04:54:30, o a otra parte del código)*. LAB celda 8, en cambio, es explícita: "se pide a los participantes implementar un TWAP sin la necesidad de realizar ninguna modificación", y el ej. 1 manda completar el notebook (ENUNCIADO).

**Nota:** con $T=1-t/N$ y $V=1-e/n$, "ir por delante" es $V<T$. La regla ejecuta en $t=0$ ($V=T=1$) y después cada vez que $V\ge T$, es decir, $e/n\le t/N$: la hija número $k+1$ ($k=0,\dots,n-1$) sale en $t=\lceil kN/n\rceil$ y la orden se completa hacia $t\approx N-N/n$, un intervalo antes del final; el episodio termina entonces por "orden completada". Al heredar de `DDQNAgent`, `TWAP(env)` construye igualmente las dos redes de Keras y el bucle de entrenamiento seguirá llamando a `learn` (inofensivo pero inútil); su `act` sustituye la compra forzada del paso 0, que la propia regla ya cumple. El TWAP determinista sirve de **referencia** para comparar el agente RL (ENUNCIADO, ej. 2; CASOS p.51 propone comparar el algoritmo contra el TWAP fuera de *train*).

**Nota (comparación en *floats*):** en los puntos de empate ($e/n=t/N$ exactamente), $1-e/n$ y $1-t/N$ pueden no salir idénticos en coma flotante y, si $V$ sale una milésima por debajo de $T$, la compra se retrasa un paso. Es más robusto comparar con tolerancia (`V >= T - 1e-9`) o reconstruir enteros ($e\cdot N\le t\cdot n$, con `env.exec_vol`, `env.state_pos`, `env.vol_care` y `env.episode_bins`).

---

## 8. Parametrización (celdas 10-12)

LAB celda 10 explica que el agente es el "cerebro" del sistema y puede tener o no inteligencia artificial (el TWAP depende solo de una regla); los algoritmos de RL requieren parametrizar su modelo. Celda 11 define cada parámetro:

| Parámetro | Definición del notebook | Valor en celda 12 | Defecto en `DDQNAgent` |
|---|---|---|---|
| `epsilon` | Exploración para métodos de valor, en [0,1] | TODO | 0,9 |
| `min_epsilon` | Mínimo, por si se quiere dejar aleatoriedad en entrenamientos largos en los que epsilon decaería a 0 | TODO | — |
| `epsilon_decay` | Cantidad que se resta a epsilon "cada determinado número de episodios" | TODO | — |
| `gamma` | Descuento; con tareas episódicas puede estar en [0,1] | TODO | 0,99 |
| `alpha` | *Step-size*; como el problema es no estacionario, estático o adaptativo "pero nunca un valor que decaiga a cero" | TODO | 0,01 |
| `buffer_size` | Memoria para obtener datos aproximadamente i.i.d. | TODO | 50.000 |
| `batch_size` | Tamaño de los paquetes que se envían al modelo al aprender | TODO | 64 |
| `hidden_neurons` | Neuronas de las capas ocultas (los agentes del taller se mantienen sencillos) | TODO | 32 |
| `nepisodes` | Sesiones de *trading* (episodios) de entrenamiento | TODO | — |
| `n_log` | Cada cuántos episodios se imprime un *log* | 25 | — |
| `learn_after` | Cada cuántos *steps* aprende la red | `= batch_size` | — |
| `update_target_after` | Cada cuántos *steps* se copian los pesos de la red *primary* a la *target* | `= 10 * learn_after` | — |

La celda 12 termina con `env = BestExecutionEnv(data, 60)` y la 14 con `agent = "-----"`, donde se instancia `DDQNAgent(env, gamma=..., epsilon=..., alpha=..., batch_size=..., buffer_size=..., hidden_neurons=...)` o el `TWAP`.

Lo que dijo el profesor: los parámetros de la red (alfa, etc.) "son menos importantes"; lo que le interesa son sobre todo los del notebook *(dudoso: "los que están en este no Food")* (T07 ~05:01:11-05:01:19). El ej. 1 no exige motivar la parametría; el ej. 2 sí (ENUNCIADO).

**Notas:** (a) En `DDQNAgent`, `alpha` es la tasa de aprendizaje de Adam, no el *step-size* tabular de la definición; la advertencia de no llevarlo a cero sigue siendo válida. (b) `gamma = 0.99` tiene un horizonte efectivo $1/(1-\gamma)=100$ pasos, muy por debajo de los 400-600 del episodio; el cálculo de gamma hacia atrás está en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md). (c) Con decaimiento lineal por episodio, para pasar de $\varepsilon_0$ a $\varepsilon_{min}$ en $K$ episodios, `epsilon_decay` $=(\varepsilon_0-\varepsilon_{min})/K$; este esquema es justo lo que el ej. 3 pide sustituir (ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md)).

---

## 9. `DDQNAgent` línea a línea (`src/agents/dqn.py`)

### 9.1 Constructor
`DDQNAgent(env, gamma=0.99, epsilon=0.9, alpha=0.01, batch_size=64, buffer_size=50000, hidden_neurons=32, trainable=True)`:
- Guarda `env` y lee de él `na = env.n_actions` y `n_features = env.n_features`.
- `minus_factor` y `div_factor` = `None`: **no se usan** en ningún sitio (parecen huecos previstos para normalizar el estado).
- `buffer = deque(maxlen=buffer_size)`: memoria FIFO; al llenarse, se descarta lo más antiguo.
- `primary_model = network()`, compilado con pérdida `"mse"` y `Adam(learning_rate=alpha)`.
- `target_model = network()` (sin compilar, solo predice) y `update_target()` para que empiece con los mismos pesos. Comentario del código: "Double Deep Q-Learning → add a frozen copy of the primary model".
- `_trainable = trainable`.

### 9.2 La red
`Input(n_features)` → `Dense(hidden, relu)` → `Dense(hidden, relu)` → `Dense(na, linear)`: un MLP de dos capas ocultas que devuelve un valor Q por acción. **Nota:** con 2 entradas, 32 neuronas y 2 acciones son $(2\cdot32+32)+(32\cdot32+32)+(32\cdot2+2)=96+1056+66=1218$ parámetros. Se importa `Concatenate`, pero no se usa.

### 9.3 Memoria: `experience` y `sample` (CER)
- `experience(s, a, r, s1, done=False)`: si `done`, `s1 = None`; guarda la tupla `(s, a, r, s1)`. El fin de episodio se codifica con `s1 is None`, sin guardar `done` aparte.
- `sample()` implementa **Combined Experience Replay**: toma `batch_size - 1` índices al azar sin reemplazo entre las transiciones anteriores a la última (`np.random.choice(len(buffer)-1, ...)`) y añade siempre la **última** transición (`len(buffer)-1`). APP p.81 lo resume como añadir la última experiencia al *batch*, lo que "ayuda a corregir el tamaño del buffer", y cita el paper *A Deeper Look at Experience Replay* (arXiv 1712.01275). **Nota (conocimiento externo):** sus autores son Zhang y Sutton (2017). Requiere que el *buffer* tenga al menos `batch_size` transiciones, garantizado por el llenado previo (§10); como restricción de parámetros, `np.random.choice(len(buffer)-1, batch_size-1, replace=False)` lanza `ValueError` si `buffer_size < batch_size`, porque el *buffer* nunca llega a tener transiciones suficientes.
- **Nota (eficiencia, §2.3):** `np.random.choice` sin reemplazo permuta todas las posiciones (~50.000) en cada `learn`, y acceder por índice a un `deque` cuesta O(n) hacia el centro. Con *buffers* grandes conviene un *array* circular de numpy o muestrear con reemplazo (`np.random.randint`). Teoría de ER/PER/CER en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md).

### 9.4 `act`: epsilon-greedy con compra forzada en el paso 0
```python
if self.env.state_pos == 0:
    return 1
```
- En el **primer paso de cada episodio** el agente compra siempre, entrene o no.
- Si `_trainable`: con probabilidad $1-\varepsilon$ actúa de forma *greedy* (`_act_no_explore`) y con probabilidad $\varepsilon$ llama a `env.action_sample()` (13). Si no es entrenable, siempre *greedy*.
- `_act_no_explore(s)`: `s.reshape(1, -1)`, Q con `primary_model` y `argmax`.
- `getQs(s, model) = model(s).numpy()`: llamada directa al modelo, sin `predict`.

**Nota (por qué la compra forzada):** garantiza que desde el paso 0 haya al menos una ejecución, de modo que el VWAP del algoritmo (8) nunca es 0/0, y reproduce la primera hija de un TWAP en $t=0$. El precio es que el agente nunca aprende ni decide nada en el primer paso, y que depende de `self.env`, lo que provoca la errata de la evaluación (§12).

### 9.5 `learn`: actualización Double DQN
1. `batch = sample()`; `s_list` con todos los estados; `s1_list` solo con los siguientes estados **no terminales**.
2. `qs = Q_primary(s_list)`; `qs1 = Q_primary(s1_list)` ("primary model selects the best next action"); `q_target = Q_target(s1_list)` ("...and target model evaluates it").
3. `targets = qs` (la misma matriz): para cada transición solo se cambia la columna de la acción tomada; las demás quedan igual que la predicción y no aportan gradiente.
4. Objetivo:
$$
y=\begin{cases} r & \text{si } s' \text{ es terminal}\\[2pt] r+\gamma\,Q_{\theta^-}\!\big(s',\ \arg\max_{a'}Q_{\theta}(s',a')\big) & \text{si no}\end{cases}
$$
   con el índice `k` recorriendo solo las filas no terminales de `qs1`/`q_target`.
5. `loss = primary_model.train_on_batch(s_list, targets)`, un paso de Adam sobre la MSE, y devuelve la pérdida.

Es Double DQN en sentido estricto (selección con la red online y evaluación con la *target*) más red objetivo fija, que se actualiza desde fuera cada `update_target_after` pasos con `update_target()` (copia de pesos). Teoría y erratas de las fórmulas en el Taller 2: [03_dqn.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md) (§8 memoria, §9 red objetivo) y [04_doble_dqn_y_duelo.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/04_doble_dqn_y_duelo.md) (§2); resumen de este taller en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md).

**Notas:** pérdida MSE sin Huber ni recorte de gradiente; si un *batch* no tuviera ninguna transición no terminal, `s1_list` quedaría vacío y la llamada al modelo fallaría (improbable con 64 transiciones). `getQ(s, a)` llama a `getQs(s)` sin el argumento `model` y lanzaría `TypeError`; no se usa en ningún sitio (código muerto).

### 9.6 Alternativas que aceptó el profesor
No es obligatorio el doble DQN: "podéis tirar este notebook por completo" mientras, con esa parametría y SAN o REP, se llegue al TWAP por recompensas y se enseñen las imágenes; el entorno lo pone cada uno como quiera (T07 ~04:55:40-04:56:16). Si el *deep learning* da problemas (necesita el *buffer* y es más inestable), vale Q-learning, refactorizando el código (T07 ~05:01:28-05:01:45). Bromeó con que quien meta funciones de impacto se gana la matrícula *(dudoso)* (T07 ~04:56:20). El ENUNCIADO lo confirma: no es necesario reutilizar el notebook, el algoritmo de Deep RL ni los paquetes (PyTorch permitido).

---

## 10. Llenado inicial del *buffer* (celda 16)

```python
# En este punto eps es 1 -> actuando random
s = env.reset()
for exps in range(buffer_size):
    a = agent.act(s); s1, r, done, _ = env.step(a)
    agent.experience(s, a, r, s1, done); s = s1
    if not exps % 10000: print(...)
    if done: s = env.reset()
```
LAB celda 15 lo justifica: antes de entrenar, el modelo necesita experiencias de las que aprender, así que se llena el *buffer* con una política estocástica para explorar el entorno. Se hacen exactamente `buffer_size` pasos, de modo que la memoria queda llena.

**Notas:** (a) El comentario supone `epsilon = 1`, pero el agente se crea con el `epsilon` elegido en la celda 12; si es menor que 1, parte del llenado lo decide una red sin entrenar. (b) Con `action_sample` a $p=0{,}5$, cada episodio aleatorio dura unos 100 pasos (§6.13): 50.000 pasos serían unos 500 episodios cortos, todos de la primera parte del horizonte. (c) Llenarlo con un TWAP o con mezcla de políticas es una opción discutida en clase, con sus riesgos: [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

---

## 11. Bucle de entrenamiento (celda 18)

Estructura (LAB celda 18):
1. `agent.set_trainable(True)`; contadores y listas `history_steps`, `history_rewards`, `history_disc_rewards`, `history_losses`.
2. Para cada `episode in range(nepisodes)`: `s = env.reset()` y, hasta `done`:
   - `a = agent.act(s)`; `s1, r, done, _ = env.step(a)`; `agent.experience(s, a, r, s1, done)`.
   - `learn_counter += 1`; `cum_reward += r`; `dis_cum_reward += gamma**step * r`; `step += 1`.
   - Cada `learn_after` pasos: `mse = agent.learn()`. Cada `update_target_after` pasos: `agent.update_target()`.
   - Al terminar: `epsilon = max(epsilon - epsilon_decay, min_epsilon)` (decaimiento **lineal por episodio**); se guardan recompensa, recompensa descontada, `mse` y pasos; cada `n_log` episodios se llama **otra vez** a `agent.learn()` y se imprime la media de pasos y recompensa de los últimos `n_log` episodios, la `mse` y epsilon.

El profesor lo describió como un número fijo de episodios ("la definición clásica que decía que no me gustaba"), 1.000 o 10.000, con unas listas para ir registrando, y un bucle "fácil": `act` con epsilon-greedy y después `step` (T07 ~04:49:16-04:50:00). Qué toca cada ejercicio (T07 ~04:58:18-04:58:53): **ej. 1**, que el notebook se ejecute hasta el final, sin recompensa específica; **ej. 2**, la función de recompensa; **ej. 3**, epsilon para que no sea "simplemente proporcional al número de episodios"; **ej. 4**, libre (por ejemplo, que termine en un rango de ejecución del 70-90 %; poco antes había dicho 80-95 %, T07 ~04:30:22). Detalle en [notebook_partida.md](notebook_partida.md).

**Notas (comprobadas en el código):**
- `mse` solo existe tras la primera llamada a `learn`. Si el primer episodio termina antes de `learn_after` pasos (por ejemplo, con un `batch_size` grande y episodios cortos), `history_losses.append(mse)` lanza `NameError`. Además, `history_losses` guarda la última pérdida calculada, que puede venir de un episodio anterior.
- El `agent.learn()` de los episodios de *log* es un paso de gradiente **extra**, no una medición: la pérdida impresa es la de ese *batch* concreto.
- `np.round(mse)` redondea a 0 decimales; con recompensas del orden de $10^{-4}$ siempre imprime 0 (§6.6). La recompensa media `rew` se imprime con `np.round(..., 2)`: con recompensas de ~$10^{-4}$ por compra y unas 50 compras por episodio sale 0,0 o −0,01, igual de poco informativa.
- `episode_losses = []` se inicializa en cada episodio pero nunca se rellena (código muerto); parece prevista para la media de pérdidas del episodio, que es la corrección natural de E4.
- Como `not episode % n_log` vale `True` con `episode = 0`, el *log* (y el `learn()` extra) se dispara también en el primer episodio.
- Con `learn_after = 64` se aprende una vez cada 64 pasos (unas 8 actualizaciones en un episodio de 500 pasos) y la *target* se copia cada 640 pasos. Cuántas actualizaciones por paso hacer es un hiperparámetro más ([04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md)).
- `history_rewards` incluye la recompensa terminal (12); `env.reward_hist` no.

---

## 12. Evaluación, `stats_df` y `plot_results` (celdas 20-21)

### 12.1 Celda 20
`agent.set_trainable(False)` (siempre *greedy*), `env = BestExecutionEnv(dict_["test"], 60)`, un episodio completo acumulando `gamma**step * r` (tras `step += 1`) y `plot_results(env)`. La celda 21 dibuja la media móvil de 20 episodios de `history_rewards`.

**Errata E5 (comprobada):** se crea un entorno nuevo, pero **no** se reasigna `agent.env`. `act` consulta `self.env.state_pos`, que es el `state_pos` del entorno de **entrenamiento** (con el valor en que acabó su último episodio), así que en test la compra forzada del paso 0 no se aplica. Si la red elige "no comprar" en el primer paso, `_compute_algo_vwap` divide 0/0 y los primeros valores de `algo_vwap_hist` son NaN. Corrección: `agent.env = env` antes de evaluar. Además, la variable `cum_reward` de esta celda es descontada (y con el exponente desplazado una unidad respecto al bucle de entrenamiento); no se usa después. Por último, evaluar en test es opcional: el ENUNCIADO no exige validación en el ej. 2 ("basta con enseñar algunas imágenes correctas en el conjunto de train").

### 12.2 `stats_df()` (en la clase del entorno)
- **Algoritmo:** DataFrame con `algo_vwap_hist` y `vol_hist`, indexado con las primeras `len(algo_vwap_hist)` horas del episodio y reindexado a todo el episodio: `vol` se rellena con 0 y `vwap` con el último valor (si la orden termina antes, la línea queda plana hasta el final).
- **Mercado:** `p = episode["ask1"]`, `v = cumvol.diff().shift(-1)` (volumen del *bin* siguiente, como la pista de (4)), y para el último *bin*, que no tiene siguiente, el último `diff` de `episode_full`; `market_vwap = cumsum(p·v) / cumsum(v)`.
- `mpx` = *mid* del episodio.

**Errata E3:** el VWAP de mercado del gráfico se calcula con `ask1`, mientras que el del entorno (4) y la recompensa usan el *mid*. Comparar las compras (al *ask*) con un VWAP al *ask* es más justo visualmente, pero el *slippage* que se ve en el gráfico y el que mide la recompensa difieren en medio *spread* (unos 1,7 pb en REP con un *tick* de *spread*). Además, el volumen del último *bin* se aproxima con el del penúltimo intervalo. Hay que elegir una sola definición y usarla en recompensa, gráfico e informe.

### 12.3 `plot_results(env, mode="max")` (`src/report/report.py`)
- Figura de 15×10 con dos paneles.
- **Arriba, "Market VS Algo":** VWAP de mercado (azul), VWAP del algoritmo (naranja) y precio medio (rojo).
- **Abajo:** volumen acumulado del mercado normalizado por su total (azul) y del algoritmo dividido por `env.vol_care` (naranja), más barras: las del mercado normalizadas por su máximo (`mode="max"`) o por su percentil 99 (cualquier otro `mode`), recortadas a 1; las del algoritmo, de altura fija 0,1 en cada paso con compra. Anchos de barra 0,00005 y 0,00007 días (≈4,3 s y ≈6 s).
- **Nota:** si la orden no se completa, la línea naranja acaba por debajo de 1: es la comprobación visual directa de volumen pendiente. Con `mode="max"`, un solo *bin* con un bloque grande aplasta todas las demás barras azules; `mode="q"` (cualquier valor distinto de `"max"`) lo evita.
- **Nota:** dibuja siempre en `plt.figure(1, figsize=(15, 10))` y no llama a `plt.show()` (BE report.py, líneas 18-44). Si se llama varias veces en una misma celda (por ejemplo, para enseñar varios episodios como pide el requisito del ej. 2, §16), todos se superponen en la misma figura. Hay que llamar a `plt.show()` (o `plt.close(1)`) entre llamadas, o adaptar la función.

### 12.4 La imagen objetivo (`images/twap_model/twap_reproduction_dqn.png`)
Descripción (BE): episodio de un día "06", de ~16:07 a ~16:42 (unos 35 min, ≈420 pasos, dentro de 400-600).
- **Arriba:** el *mid* (rojo) oscila entre ~15,01 y ~15,06, con una caída brusca hacia 16:35. El VWAP de mercado (azul) y el del algoritmo (naranja) suben casi juntos de ~15,025 a ~15,048; el azul baja a ~15,04 entre 16:35 y 16:38 y vuelve a ~15,048 al final.
- **Abajo:** la línea naranja es una escalera casi **diagonal** de 0 a 1 que llega a 1 justo al final del episodio. La azul avanza despacio hasta ~0,22 y salta de golpe a ~0,96 hacia 16:38: un único *bin* concentra en torno al 75 % del volumen de mercado del episodio. Las barras naranjas, de altura 0,1 y casi equiespaciadas, son unas 40-45 (recuento aproximado por píxeles), coherente con $n\approx 0{,}1N$; las azules son casi invisibles salvo la del bloque, que llega a 1 (el efecto de `mode="max"`).

Esto es lo que se espera del ej. 2: compras repartidas uniformemente y volumen acumulado en diagonal hasta completar la orden (ENUNCIADO: "No se pide una réplica exacta, pero sí que tengan un comportamiento similar"). El profesor: "65 barritas en ese gráfico que os va a salir al final" para una orden de 65 títulos (T07 ~04:47:36). **Nota:** CASOS p.54 (resultado del algoritmo "conservador" de su TFM, con el TWAP como referencia) usa la misma disposición (VWAP de mercado y del algoritmo, precio, volúmenes acumulados en % y barras de órdenes de mercado y del algoritmo escaladas) y añade una media móvil del precio; se comenta en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

---

## 13. El notebook `execution_lab-step_by_step`

- **Estructura (LABSTEP celdas 0-34):** carga `rep_data.pkl` (celda 2), `data = dict_['train']`, `look_back = 60` (celda 3); la celda 4 replica `__init__` como **variables globales** con funciones *stub* que devuelven `None` (incluido `_detect_num_feat`, por lo que `n_features` es `None`); después una sección por función: 1 `_generate_episode_params` (celda 6), 2 `_generate_episode` (8), 3 `observation_builder` (10), 4 `_compute_episode_market_feat` (12), 5 `_compute_market_vwap` (14), 6 `_compute_reward` (16-17), 7-8 `_compute_algo_vwap` y `_agg_action` (19-20), 9 `_do_nothing` (22), 10 `_compute_stop_conditions` (24), 11 `_compute_done` (26), 12 `_compute_done_reward` (28), 13 `action_sample` (30) y una prueba de `step` ("Comprobemos que todo va bien", celdas 32-34).
- **Cómo se usa:** cada celda tiene el cálculo **suelto** (con los mismos TODOs que la clase) seguido de un `print` del resultado y, debajo, una definición vacía de la función (`def ...: return None`). Se completa el cálculo, se comprueba con el `print` y luego se pasa a la función y a la clase con `self.` (T07 ~04:34:42-04:35:21).
- **Diferencias con LAB:** en (5) da en comentario la forma de `p_arr` y `v_arr` (`mkt_p.values[:state_pos + 1]`, `v.values[:state_pos + 1]`); en (6) fija antes `price = 0`, `vol = 0` para probar la rama sin ejecución (celda 16); el *stub* de `_agg_action` declara `global exec_vol` porque modifica esa variable global; la celda 34 reconstruye `actions_fn` porque el diccionario de la celda 4 apuntaba a las versiones vacías.
- **Lo que dijo el profesor:** sobre la variable `global`, "creo que era porque estos 3 en particular estaban con el punto 6" *(dudoso; en la versión del repo el único `global` está en el punto 7; una lectura posible es que "punto 6" sea la **celda** 6 del LABSTEP, la de (1), y "estos 3" sus tres variables reasignadas, `episode_bins`, `vol_care` y `episode_full_len`)* (T07 ~04:36:46-04:37:01). Mostró por encima la solución y el notebook vaciado, y comentó que un nombre "se llama distinto" *(dudoso: no se identifica cuál)* (T07 ~04:37:08-04:38:23).
- **Nota (trampa):** la prueba de `step` llama a las **funciones**, no a las variables calculadas en las celdas: si solo se completan los cálculos sueltos y no los `def`, `action_sample()` devuelve `None` y `step` lanza `ValueError("Invalid action None")`, o bien recompensa y `done` salen `None`. En particular, el *stub* de `_compute_done_reward` devuelve `None` (no 0, LABSTEP celda 28), así que si `done` es `True` la celda 34 falla con `TypeError` en `reward += _compute_done_reward()`. Todo se prueba sobre **un único** episodio generado una vez y con `state_pos = 0`; no hay `reset` ni bucle.
- **Nota (`global` al pasar el código a los `def`):** en el LABSTEP las funciones trabajan sobre variables globales. Las que **reasignan** una variable (no las que solo leen o hacen `append` sobre una lista) necesitan `global`: en (1) `episode_bins`, `vol_care` y `episode_full_len`; en (2) `episode` y `episode_full`; en (4) `episode_vwap` y `market_ep_vol`; en (7) `exec_vol`, el único que trae el repo (LABSTEP celdas 4, 6, 8, 12, 20). Sin `global` la función crea una variable local, la global se queda como estaba (`None`) y lo siguiente que la use falla. La prueba de `step` no llama a (1), (2) ni (4), sino que usa lo que dejaron las celdas sueltas, por lo que el error solo aflora si se llaman esas funciones. En la clase todo esto se convierte en `self.`.
- **Nota (estado contaminado):** ejecutar las celdas sueltas modifica el estado global. La celda 20 añade a `price_hist`, `vol_hist` y `algo_vwap_hist` y suma 1 a `exec_vol`, y la 22 vuelve a añadir; la prueba de `step` (celda 34) parte, por tanto, de históricos contaminados, y cada reejecución acumula más. La celda 19 (VWAP del algoritmo) va antes que `_agg_action`, con listas vacías: 0/0 da `nan` con aviso en numpy, o `ZeroDivisionError` con escalares. Conviene reejecutar la celda 4 (y las 6-12) antes de la prueba final.

---

## 14. Lo que explicó el profesor del código en T07, en orden

| Hora | Contenido |
|---|---|
| ~00:03:51-00:08:03 | Docker y devcontainer del repo; imagen publicada y Dockerfile de respaldo con Python 3.8 (§2.2) |
| ~00:10:19-00:18:03 | Eficiencia: pandas frente a numpy frente a diccionario; el notebook usa pandas "por facilitarlo" (§2.3) |
| ~04:31:40-04:36:00 | No hace falta contenedor; ficheros: `execution_lab` (a entregar, clase con huecos y *tips*, imagen de ejemplo en `images/`) y `step_by_step` (funciones en orden lógico) (§1, §13) |
| ~04:36:00-04:38:30 | Parámetros de la orden: pasos en [400, 600] y tamaño de la orden; `global`; nombre que no coincide (§6.1, §13) |
| ~04:38:30-04:45:00 | Pickles antiguos y nuevos; SAN/REP; estructura diccionario → día → DataFrame de 5 s; particiones; sesión de 13:00 a 17:25; ficheros extra; asunciones (§3) |
| ~04:45:00-04:48:25 | 720 pasos por hora; flexibilidad 1 de cada 10; 65 títulos = 65 barras (§3.5) |
| ~04:48:25-04:52:10 | `reset` y bucle de entrenamiento con número fijo de episodios (§5, §11) |
| ~04:52:10-04:56:34 | Observación $[T, V]$; TWAP determinista con un *if*; alternativa Q-learning 2×2; libertad para rehacer el notebook (§6.3, §7, §9.6) |
| ~04:56:34-05:01:00 | `step`; VWAP parcial opcional; qué toca cada ejercicio; `look_back` (§5, §6.5, §11) |
| ~05:01:00-05:05:40 | Atributos; los parámetros de la red importan menos; Q-learning si DQN da guerra; precio de ejecución entre fotos (§4, §8, §6.7) |

---

## 15. Erratas, sesgos y riesgos detectados

Todas comprobadas leyendo el código línea a línea (salvo las marcadas como de clase o no verificadas).

| # | Dónde | Problema | Consecuencia | Corrección / qué hacer |
|---|---|---|---|---|
| E1 | LAB celda 7, docstring de `__init__` | `data` se describe como "Dataframe" | Confusión al completar (2) | Es un diccionario {día: DataFrame} de una partición |
| E2 | (7) `_agg_action` | Ejecuta al `ask1` de la foto actual, con latencia 0, sin mirar `vask1` | Precio "tonto y sesgado" (T07 ~05:03:39) | Razonar la asunción: media de *mids* entre fotos, interpolación según latencia, `last` o `executions.pkl` (REP 2019) |
| E3 | `stats_df` frente a (4) | Gráfico con VWAP al `ask1`; entorno y recompensa con el *mid*; último volumen aproximado | El *slippage* del gráfico y el de la recompensa difieren en ~½ *spread* | Unificar la definición de precio de mercado |
| E4 | LAB celda 18 | `mse` indefinida si no se ha aprendido aún; valor obsoleto en `history_losses`; `learn()` extra en los episodios de *log*; `np.round(mse)` imprime 0; `episode_losses` declarada y sin usar | `NameError` o métricas engañosas | Inicializar `mse = np.nan`, rellenar `episode_losses` y registrar su media, no entrenar al loguear, imprimir con decimales o notación científica |
| E5 | LAB celda 20 | No se reasigna `agent.env` al entorno de test | Sin compra forzada en test; posible NaN en el VWAP del algoritmo | `agent.env = env` antes de evaluar |
| E6 | `dqn.py` | `getQ` llama a `getQs(s)` sin `model`; `Concatenate`, `minus_factor` y `div_factor` sin uso | Código muerto (fallaría si se usara) | Ignorar o borrar |
| E7 | Entorno | `actions_hist` nunca se rellena, aunque el profesor lo presentó como histórico "por si queréis plotearlos después" (T07 ~05:00:52) | No se pueden graficar acciones | Añadir `self.actions_hist.append(action)` en `step` |
| E8 | `step` | La recompensa terminal (12) no entra en `reward_hist` | Históricos del entorno y del bucle no cuadran | Tenerlo en cuenta al analizar |
| E9 | (6) por defecto | Recompensa con el VWAP final del episodio (información futura), sesgo de −½ *spread* por compra y sin penalización por volumen pendiente | El óptimo es no comprar; recompensas ~$10^{-4}$ | Rediseñar (ej. 2) y reescalar; ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md) |
| E10 | (13) con $p=0{,}5$ | Política aleatoria que completa la orden en ~20 % del tiempo | *Buffer* y exploración sesgados al principio del episodio | $p\approx n/N$ o mezcla de políticas |
| E11 | (4) | `market_ep_vol.iloc[0] = 0` | Se pierde el volumen del primer *bin* | Usar `episode_full` para el primer `diff` |
| E12 | (5) | Si el precio se desplaza un *bin* (coherente con (4)), el primer precio es NaN y NaN·0 = NaN | VWAP parcial NaN | Rellenar el primer precio |
| E13 | (2) | `np.random.choice(data.keys())` falla; bucle por rechazo | Error o ineficiencia | `list(data.keys())`; muestrear la posición en el rango válido |
| E14 | (1) | `randint` excluye el límite superior; el `assert` exige `int` de Python | 600 nunca sale; `AssertionError` con `np.int64` | Límite 601; `int(...)` |
| E15 | `_detect_num_feat` | No llama a (4) | Romper el constructor al añadir características de mercado | Añadir (4) o calcular las características desde `episode_full` |
| E16 | (9) | Si no registra en `price_hist`/`vol_hist`/`algo_vwap_hist` | `stats_df` falla o dibuja las compras al principio | Registrar en todos los pasos |
| E17 | (10) | `>` en lugar de `>=` para el tiempo | Episodio de $N+1$ pasos; `IndexError` solo si en el paso extra el agente compra (fallo intermitente) | Usar `>=` |
| E18 | `TWAP(DDQNAgent)` | Construye redes y aprende sin usarlas | Tiempo perdido | Inofensivo; se puede evitar con una clase propia |
| E19 | Dev Container frente a `setup.py` | Pila antigua (Python 3.8, TF 2.5, pandas 1.4) frente a Python ≥ 3.9.2, TF 2.15, pandas 2.0.3; los metadatos de LAB y LABSTEP registran un *kernel* Python 3.8.18, sin salidas guardadas | `pip install -e .` puede fallar dentro del contenedor *(no verificado)* | Elegir un único camino y documentar versiones |
| E20 | LAB celda 4 | Lee el pickle antiguo | Falla con pandas ≥ 2.1 | Usar `data/new_version/` |
| E21 | Clase (T07 ~04:44:45, ~04:46:51-04:46:59) | Ambigüedad, no errata: "episodios de 90 min, 2 h" y "puedes cambiarlo... de 2 o 3 h", frente a los 400-600 pasos del código (33-50 min) | Duración del episodio abierta | Se puede alargar, pero hay que motivarlo y recalibrar gamma y *buffer* (§3.5) |
| E22 | Clase (T07 ~04:43:45) | Duda si `orderbook` es de SAN; dice que tiene "3 o 4 posiciones" | — | Es Repsol (LAB celda 2; precios de ~14,9 €) y tiene exactamente 3 niveles (DATOS) |
| E23 | Clase (T07 ~04:47:21-04:47:29, ~04:50:26) | Llama "orden hija" a la orden CARE de 65 títulos; el factor sale transcrito como "0 9" | — | Es la orden CARE (las hijas son de 1 título); el factor es casi seguro 0,09 (720 × 0,09 ≈ 65) *(dudoso, por la transcripción)* |
| E24 | Clase (T07 ~04:50:36) | `reset` "elige activo" y hora "a la una, a las dos" | — | El código usa un solo activo y cualquier *bin* como inicio |
| E25 | BE README.md | "Kissel" | — | Kissell |
| E26 | LAB celda 11 | `alpha` descrito como *step-size* tabular | Confusión | En `DDQNAgent` es la tasa de aprendizaje de Adam |
| E27 | LABSTEP celdas 28 y 34 | La prueba de `step` usa los `def` vacíos si no se rellenan; el *stub* de `_compute_done_reward` devuelve `None`; las celdas sueltas contaminan los históricos; faltan `global` al reasignar | `ValueError`, `TypeError` o `None` | Rellenar también las funciones (con `global`) y reejecutar la celda 4 antes de probar (§13) |
| E28 | `.gitignore` de BE | Ignora `data/`, pero en este repo los pickles (~139 MB en total) están en `docs/raw` | Peso y posible problema de licencia de datos de mercado en un repo público *(no verificado)* | Decidir antes de publicar ([10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md)) |

---

## 16. Qué hay que tocar en el código para cada ejercicio (resumen)

- **Ej. 1:** completar (1)-(13), `TWAP.act` (LAB celda 8 pide implementarlo y el ej. 1 manda completar el notebook; la frase "esto no es necesario" de clase es ambigua, §7; en todo caso sirve de referencia), los parámetros de la celda 12, instanciar el agente (celda 14) y ejecutar hasta `plot_results`. Corregir E4 y E5 evita sorpresas.
- **Ej. 2:** (6) y (12) (recompensas), con estado $[T, V]$ suficiente; ajustar y motivar gamma, *buffer*, episodios y demás parámetros; posiblemente (13). Requisito del ENUNCIADO: el algoritmo debe funcionar para distintos días, momentos temporales, duraciones del episodio y tamaños de la orden CARE. (1) y (2) ya aleatorizan las cuatro cosas, así que no hay que fijarlas para entrenar; las evidencias deberían ser varios `plot_results` de episodios con $N$ y $n$ distintos, anotando en cada gráfico día, hora, $N$ y $n$ (y separando las figuras, §12.3).
- **Ej. 3:** la línea `agent.epsilon = max(...)` del bucle (celda 18) y un criterio de mejora (evaluación *greedy* periódica, etc.).
- **Ej. 4:** libre (estado con libro de órdenes, recompensas del 80-95 %, parada por aprendizaje y *checkpoints*...).

Requisitos exactos en [notebook_partida.md](notebook_partida.md); propuesta razonada en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md); términos en [11_glosario.md](11_glosario.md).
