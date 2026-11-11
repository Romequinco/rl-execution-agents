# Material de RL de clase (`rl-material`)

> **Fuentes:**
> - **RLM** = repositorio `docs/raw/MIAX14/rl-material/` (autor: Jesús Sanz del Real; licencia MIT, 2021). Leídos completos: los 17 notebooks de `notebooks/` (`cliff-walking`, `insights`, `profit-seeking`, `render-env-devcon` y los 13 de `notebooks/rl-models-basic/`), `README.md`, `setup.py`, `src/utils/plots.py`, `src/utils/bmeapi.py`, `src/rl/requirements.txt`, `.devcontainer/devcontainer.json`, `dockerfiles/devcontainers/{docker-compose.yml, python/dockerfile, python/requirements.txt}` y las 6 imágenes de `notebooks/static/`.
> - **T06** = transcripción automática de la clase del 06/11/2026 (`docs/raw/26 11 06.pdf`). Tramos: ~00:59:57-01:06:52 (presentación de `rl-material`, Docker, instalación, demo de Q-learning), ~01:06:52-01:19:37 (SARSA, *cliff walking*, Expected SARSA, Double Q-learning), ~01:20:00-01:26:22 (aproximación funcional y lineal), ~02:11:09-02:19:53 (estado, CartPole, discretización), ~02:50:34-02:55:15 (signo de la recompensa y γ en el acantilado), ~03:00:40-03:06:15 (decaimiento de ε "como en los notebooks"), ~03:26:17-03:30:18 y ~03:47:37-03:50:29 (River Swim y efecto de γ, ejecutados en `insights`), ~04:13:11-04:15:06 (`deque` y llenado inicial del buffer).
> - **T07** = transcripción automática de la clase del 07/11/2026 (`docs/raw/26 11 07.pdf`). Tramos: ~00:04:54-00:07:53 (Docker y el `dockerfile`), ~00:12:35-00:15:56 (demo `%timeit` de `insights`), ~00:18:03-00:23:02 (*cliff walking* y sesgo del ε decay), ~00:25:05-00:25:55 (combinar mejoras, "Rainbow"), ~00:40:00 (escollos: recompensas y exploración), ~00:42:13-00:44:08 (notebook `profit-seeking`, mostrado sin ejecutar), ~03:48:03-03:48:18 (buffer en episodios y *cliff walking*), ~04:49:20-04:49:45 (el bucle del LAB).
> - **APP** = diapositivas *Applying Reinforcement Learning* (`docs/raw/MIAX14/applying_RL_MIAX.pdf`): p.1-3, p.22-36 (con imágenes de p.35-36), p.51-53 (CartPole y estado), p.72 (River Swim), p.74 (α), p.75-76 (γ), p.79-80 (*Experience Replay*).
> - **CASOS** = diapositivas *Casos Prácticos* (`docs/raw/MIAX14/casos_practicos.pdf`): p.14-15 (con imagen de p.14).
> - **BE dqn.py** = `docs/raw/MIAX14/best-execution/src/agents/dqn.py`; **BE setup.py** = `docs/raw/MIAX14/best-execution/setup.py`; **LAB** = `docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`. Solo para comparar.
> - **T1 / T2** = Talleres 1 y 2 (`docs/raw/referencias/rl-trading-agents/` y `docs/raw/referencias/deep-rl-trading-agents/`), para enlazar la teoría ya desarrollada.
>
> **Convenciones:** "(RLM TD-q-learning, celda 7)" = celda 7 de `notebooks/rl-models-basic/TD-q-learning.ipynb`, numeradas desde 0 (`cliff-walking`, `insights`, `profit-seeking` y `render-env-devcon` están en `notebooks/`, el resto en `notebooks/rl-models-basic/`). "(T06 ~01:03:39)" = hora aproximada en la transcripción; todo lo que viene de ellas es paráfrasis de un texto automático muy ruidoso, con citas literales cortas entre comillas; lo que no se entiende bien va marcado *(dudoso)*. "(APP p.35)" = página del PDF. **Nota:** = contenido propio (cálculos, simulaciones, correcciones, conexiones). Los errores del código se señalan con su corrección razonada (resumen en §14). Ningún notebook tiene salidas guardadas, así que **no hay resultados impresos que citar**: lo que se dice de resultados viene de las diapositivas, de la clase o de cálculos propios marcados como tales.
>
> Escrito el 10/11/2026.

---

## 1. Qué es `rl-material` y qué papel tuvo en el taller

**Idea.** Es un repositorio de notebooks del profesor con implementaciones mínimas, de "pizarra", de los algoritmos de valor vistos en el máster (TD tabular, aproximación lineal, DQN y sus mejoras), más tres notebooks de apoyo (`cliff-walking`, `insights`, `profit-seeking`) y uno de utilidades de renderizado. No es material de la práctica: la práctica sale de `best-execution` ([08_entorno_best_execution.md](08_entorno_best_execution.md)). Pero el agente y el bucle de entrenamiento de la práctica son los mismos que los de estos notebooks (§13).

Lo que dijo el profesor del material:
- **Los dos más importantes** son `TD-q-learning` y `deep`, "que son la base del resto"; se pueden mirar por si surgen dudas de código, y el README explica cómo ejecutarlos (APP p.1). No hace falta estudiar las diapositivas antes de clase (APP p.1).
- Está en el Drive del curso, con "unos cuantos algoritmos" para que los usemos; **no iba a dedicar tiempo de clase** a él; dudas antes de clase, en los descansos o por correo (T06 ~01:00:23-01:00:54). También dijo que hay otros paquetes que "incluso estarán mejor" que el suyo (T06 ~01:06:38-01:06:44).
- Lo mantiene funcionando "de clase a clase" con **Docker** (T06 ~01:01:02); ver §2.
- **Aviso:** el material usa la librería antigua **`gym`**, que ya no se usa; lo actual es `gymnasium`, y no lo ha adaptado (T06 ~01:05:20-01:05:43). Ver [02_diseno_de_entornos.md §2.3](02_diseno_de_entornos.md).
- Hizo una **demo de Q-learning con renderizado** (el CartPole de `TD-q-learning`, preguntándose cuántos episodios le había puesto) para enseñar que el render solo funciona dentro del contenedor si se arranca Jupyter con el comando de pantalla virtual (T06 ~01:05:54-01:06:32).
- El **decaimiento de ε por episodios** "es lo que vais a ver que en los notebooks está definido así", igual que en muchos ejemplos, pero es arbitrario y no es lo que hay que hacer (T06 ~03:03:41-03:05:07). Ver §4.2 y [04_exploracion_memoria_y_estabilidad.md §2-3](04_exploracion_memoria_y_estabilidad.md).
- En clase se ejecutaron en directo: `TD-q-learning` (T06 ~01:05:54), `cliff-walking` (T06 ~01:11:25-01:12:45 y T07 ~00:19:56-00:22:58), `insights` (River Swim y γ en T06 ~03:27-03:50; `%timeit` en T07 ~00:12:35-00:15:56). `profit-seeking` solo se mostró y se comentó, sin ejecutarlo, porque la API de la que saca los datos ya no funciona (T07 ~00:42:35-00:44:00).

**Nota (número de notebooks).** El repositorio tiene **17** notebooks, no 18: 13 en `rl-models-basic/` (4 TD tabulares, 3 lineales, 6 deep) y 4 en `notebooks/` (`cliff-walking`, `insights`, `profit-seeking`, `render-env-devcon`). Todos se guardaron con kernel Python 3.8 (3.8.11, 3.8.17 o 3.8.18 según el notebook), es decir, se ejecutaron presumiblemente en la imagen Docker antigua (el profesor recordó que la hizo con Python 3.8, T07 ~00:07:36-00:07:44) y no en la instalación con Python 3.11 que propone el README, y todos están **sin salidas**.

### 1.1 Mapa del repositorio

| Ruta | Contenido |
|---|---|
| `README.md` | Las dos formas de instalar (Dev Container o `pip install -e .`), versiones de Python y resolución de problemas (§2) |
| `setup.py` | Paquete `rl-material` 1.0.0, Python `>=3.9,<3.12`, dependencias fijadas (§2.2) |
| `src/utils/plots.py` | `plot_training_results` (§3.1) |
| `src/utils/bmeapi.py` | Cliente `APIBMEHandler` de la API del Instituto BME, ya inactiva (§3.2) |
| `src/rl/requirements.txt` | Requisitos antiguos (TF 2.5, gym 0.23.1, box2d) de la imagen Docker original (§2.3) |
| `.devcontainer/`, `dockerfiles/` | Configuración del Dev Container (§2.3) |
| `notebooks/rl-models-basic/` | `TD-q-learning`, `TD-sarsa`, `TD-expected-sarsa`, `TD-double-q-learning`, `linear-approximation`, `linear-sarsa`, `linear-expected-sarsa`, `deep`, `deep-fixed`, `deep-double`, `deep-dueling`, `deep-expected-sarsa`, `deep-rainbow` |
| `notebooks/` | `cliff-walking`, `insights`, `profit-seeking`, `render-env-devcon` |
| `notebooks/static/` | 6 imágenes usadas en los notebooks (§12) |

*Fuente: APP p.1; T06 ~01:00:23-01:06:52, ~03:03:41-03:05:07; RLM (estructura del repositorio y metadatos de los notebooks).*

---

## 2. Instalación y entorno

### 2.1 Lo que contó el profesor (T06 ~01:01:02-01:06:32)

- Usa **Docker**: hay "un par de imágenes" publicadas y también los requisitos para construirlas desde cero. Funciona en el 90 % de los casos; siempre hay un 10 % de alumnos a los que algo les falla, y entonces o se mira caso por caso o se le pregunta a ChatGPT (T06 ~01:01:11-01:01:28).
- Hace falta la extensión **Dev Containers** de VS Code, que permite trabajar dentro de Docker desde la interfaz de VS Code, aislando paquetes y requisitos; el README lo explica (T06 ~01:01:36-01:02:12).
- **Alternativa 2:** crear un entorno y hacer `pip install -e .` en la terminal; así todo lo que hay en `src` se importa directamente, sin `sys.path.append` (T06 ~01:02:53-01:03:21). A él esta vía no le funciona en su portátil nuevo, que por seguridad no le deja instalar TensorFlow 2.15 *(dudoso: "Tenso. Flow: 2, Quince")* (T06 ~01:02:26-01:02:38).
- Al abrir la carpeta, VS Code ofrece reabrirla en el contenedor; si no aparece el mensaje, se pulsa el botón de abajo a la izquierda. **La primera vez tarda unos 10 minutos**, porque la imagen de TensorFlow pesa unos **2 GB**; después arranca enseguida (T06 ~01:03:25-01:04:00).
- Para **ver los renders** dentro del contenedor hay que lanzar Jupyter Lab con un comando que crea una pantalla virtual (T06 ~01:04:27-01:04:35); si no, el render no funciona (T06 ~01:06:23). Es el `xvfb-run` del README (§2.3).
- En T07 explicó qué es un contenedor: como los contenedores de carga de los barcos, aísla todo lo que necesita la aplicación (dependencias incluidas) y es como una mini máquina virtual, pero más ligera, porque no hace falta crear un ordenador completo, solo lo necesario; y pidió a "los informáticos" que lo explicaran mejor (T07 ~00:04:54-00:05:56). Un alumno añadió que evita los problemas de dependencias y de versiones al cambiar de máquina, que garantiza una copia exacta en todos los equipos y que la imagen es un entregable natural para una plataforma *cloud* o un servidor (T07 ~00:06:00-00:06:32). Se comentó que para un notebook sencillo ("un notebook tonto") no hace falta, pero para todo lo demás sí *(dudoso quién lo dice)* (T07 ~00:06:55-00:07:06).
- A continuación enseñó el `dockerfile`: parte de la imagen publicada e instala encima los *requirements*; la parte comentada es la receta de la que salió esa imagen, que deja "por si acaso fallase": instalaba Python, que entonces era la 3.8 ("ha pasado un tiempo desde entonces"), los requisitos y la ventana virtual para el render que había enseñado el día anterior (T07 ~00:07:11-00:07:53). Es coherente con que los notebooks estén guardados con Python 3.8 (§1, §2.3). Ver [notebook_partida.md](notebook_partida.md) para el entorno de la práctica.

### 2.2 Opción 2 del README: instalación local con `pip` (`setup.py`)

| Paso | Qué hacer |
|---|---|
| Python | **3.11** (recomendado; en Windows y macOS la 3.11.9 es la última con instalador oficial). 3.9 y 3.10 soportadas; **3.12 o superior no** (`pip install` se para con *"requires a different Python"*); 3.8 o inferior tampoco |
| Entorno virtual | `py -3.11 -m venv .venv` y `.venv\Scripts\Activate.ps1` (Windows; si PowerShell bloquea el script, `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`), `python3.11 -m venv .venv` (macOS/Linux) o `conda create -n rl-material python=3.11` |
| Instalar | `python -m pip install --upgrade pip` y `pip install -e .` desde la raíz; `-e` (editable) hace que los cambios en `src/` se vean reiniciando el kernel |
| Comprobar | `python -c "import gym, tensorflow as tf; env = gym.make('LunarLander-v2'); env.reset(); print('OK', gym.__version__, tf.__version__)"` |
| Ejecutar | `jupyter lab` desde la raíz, o VS Code con las extensiones Python y Jupyter y el kernel del entorno. Fuera de Docker no hace falta `xvfb` |

Dependencias fijadas en `setup.py`, con el motivo que da el propio fichero:

| Paquete | Versión | Motivo |
|---|---|---|
| `gym` | 0.23.1 | Los notebooks usan la API antigua: `reset()` devuelve la observación y `step()` 4 valores |
| `Box2D` | 2.3.10 | Física de LunarLander; no `gym[box2d]` porque `box2d-py` necesita SWIG y compilador de C++ |
| `pygame` | 2.6.1 | Render de CartPole y LunarLander en gym 0.23.1 |
| `tensorflow` | 2.15.1 | Último TF con Keras 2; Keras 3 (TF ≥ 2.16) rompe las redes *dueling* y *rainbow*. Solo hay *wheels* para Python 3.9-3.11 |
| `numpy` / `pandas` / `matplotlib` | 1.26.4 / 2.2.3 / 3.9.4 | — |
| `requests`, `jupyterlab>=4,<5` | — | API de BME y entorno de notebooks |

Problemas frecuentes que recoge el README: `SSL: CERTIFICATE_VERIFY_FAILED` en redes corporativas (pedir el certificado raíz del proxy a IT, guardarlo en la raíz, que está en `.gitignore`, y `pip config --site set global.cert <ruta>`); si el error SSL desaparece pero las descargas siguen denegadas, la empresa bloquea PyPI (pedir acceso a IT o instalar fuera de la red corporativa); en Windows, usar el instalador de python.org y no la versión de Microsoft Store; la 3.11 puede convivir con cualquier otra versión de Python, así que no hace falta desinstalar nada: basta crear el entorno con `py -3.11`, `python3.11` o conda; una **ventana de pygame extra** al renderizar con `mode='rgb_array'`, esperable en gym 0.23.1 (se cierra con `env.close()`; en Windows puede marcar "No responde"); en Linux/WSL sin pantalla, `SDL_VIDEODRIVER=dummy` antes de Jupyter (el modo `human` necesita pantalla real); en Windows, rutas demasiado largas al instalar TensorFlow (clonar en una ruta corta o activar *long paths*) y `DLL load failed` (instalar el *Visual C++ Redistributable*). **No hace falta GPU**: todo corre en CPU (en Windows nativo TensorFlow es solo CPU).

### 2.3 Opción 1: Dev Container (Docker)

- Requisitos: Docker, VS Code y la extensión Dev Containers. Se abre la carpeta raíz, botón *"Open a remote window"* (abajo a la izquierda) → *"Dev Containers: Reopen in Container"*.
- `.devcontainer/devcontainer.json`: servicio `rl-material-miax` del `docker-compose.yml`, carpeta de trabajo `/workspace`, puerto 8080 reenviado, usuario `vscode`, extensiones GitLens, autoDocstring y Python, y ajustes de *linting* (pylint, flake8) y pytest.
- `docker-compose.yml`: construye con `dockerfiles/devcontainers/python/dockerfile`, monta el repo en `/workspace`, deja el contenedor vivo con un bucle `sleep`, y fija `PYTHONPATH=/workspace/src:/workspace` (por eso dentro del contenedor `from src.utils...` funciona sin instalar nada).
- `dockerfile`: parte de la imagen publicada `jesussanz/rl_material:v2` y solo añade `matplotlib==3.6.1`, `jupyterlab==4.0.0`, pytest, pytest-cov, flake8 y pylint. Las líneas comentadas muestran cómo se construía la imagen base: `mcr.microsoft.com/vscode/devcontainers/python:3.8`, `xvfb`, `python3-opengl`, `swig` y `src/rl/requirements.txt` (`numpy==1.19.2`, `pandas==1.3.0`, `tensorflow==2.5.0`, `scipy==1.7.1`, `gym==0.23.1`, `pyglet==1.5.23`, `gym[box2d]`).
- Para renderizar entornos gym dentro del contenedor: `xvfb-run -a -s "-screen 0 1400x900x24" jupyter-lab`. Para pararlo: `docker stop devcontainers_rl-material-miax_1`.

**Nota (dos entornos que no son el mismo).** Los notebooks se guardaron con Python 3.8, presumiblemente en la imagen Docker construida con `src/rl/requirements.txt` (TF 2.5, según las líneas comentadas del `dockerfile`; el repositorio no dice qué versión de TF lleva la imagen publicada `jesussanz/rl_material:v2`), mientras que el README recomienda 3.11 con TF 2.15.1. El código debería ser compatible con ambos (API Keras 2) *(no verificado ejecutando)*, pero las dos vías no dan exactamente las mismas versiones.

**Nota (compatibilidad con la práctica).** `best-execution` fija `tensorflow==2.15.1`, `numpy==1.26.4` y Python `>=3.9.2,<3.12` (`rl-material`: `>=3.9,<3.12`), pero **`pandas==2.0.3`**, porque los `.pkl` de datos se guardaron con pandas 1.x y `pickle.load` falla con pandas ≥ 2.1 (BE setup.py). `rl-material` fija `pandas==2.2.3`. Por tanto, **no se puede instalar uno encima del otro sin romper algo**: o dos entornos virtuales, o un entorno con los pines de `best-execution` al que se añadan a mano `gym==0.23.1`, `Box2D==2.3.10` y `pygame==2.6.1` (los notebooks de `rl-material` apenas usan pandas: solo `insights` y `profit-seeking`). Y, como el README, ninguno de los dos funciona con Python 3.12 o 3.13: hay que crear el entorno con `py -3.11`. Ver [notebook_partida.md](notebook_partida.md) y [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

*Fuente: RLM README.md, setup.py, .devcontainer/devcontainer.json, dockerfiles/devcontainers/*, src/rl/requirements.txt, .gitignore; BE setup.py; T06 ~01:01:02-01:06:32; T07 ~00:04:54-00:07:53.*

---

## 3. Utilidades de `src/utils`

### 3.1 `plots.py`: `plot_training_results(rewards, losses, window=100, scale="log")`

- Suaviza `rewards` y `losses` con una **media móvil** de `window` episodios: `np.convolve(x, np.ones(window)/window, mode='valid')`.
- Dibuja la recompensa (azul `royalblue`, eje izquierdo "Reward") y la pérdida (color `lavender`, eje derecho "Loss", con `twinx`) frente a "Episodes".
- `scale` se aplica a **los dos** ejes. Los notebooks usan `"log"` en CartPole (recompensas positivas: pasos sobrevividos) y `"linear"` en LunarLander y *cliff walking*, donde hay recompensas negativas que un eje logarítmico no podría mostrar.

**Nota:** `lavender` es casi blanco y la curva de pérdida apenas se ve; y si hay menos episodios que `window`, `mode='valid'` no da una media móvil con sentido (NumPy intercambia los papeles de las dos series). Cosmético en los notebooks, que siempre tienen más episodios que la ventana.

### 3.2 `bmeapi.py`: `APIBMEHandler(market, algo_tag)`

Cliente HTTP (con `requests`) de una pasarela de la API del Instituto BME para la competición de algoritmos del máster (`competi = 'mia_X'`), con la clave de usuario vacía en el código. Métodos:

| Método | Qué hace |
|---|---|
| `get_ticker_master()` | Maestro de valores del mercado → DataFrame |
| `get_close_data()` | Cierres diarios de todos los valores del maestro → DataFrame (una columna por ticker) |
| `get_close_data_ticker(ticker)` | Serie de cierres de un ticker (también `'benchmark'`) |
| `get_user_algos()` | Algoritmos del participante |
| `post_alloc(str_date, alloc_list)` | Envía una asignación de cartera para una fecha |
| `get_allocs()` | Asignaciones enviadas, convertidas con `allocs_to_frame` (fecha × ticker) |
| `run_backtest()` | Lanza el backtest; imprime el estado y `result`; devuelve el diccionario `content` (con `result` y `trades`; el DataFrame de operaciones se construye pero no se devuelve) o `None` si la respuesta no es 200 (entonces imprime el texto de la respuesta) |
| `delete_allocs()` | Borra las asignaciones |

**Nota:** en todos los métodos la clave de usuario viaja como parámetro de la URL (`?key=...`), una práctica poco segura si se reutiliza el cliente con una clave real.

Solo lo usa `profit-seeking` (`get_close_data` y `get_close_data_ticker('benchmark')`). El profesor dijo que la API ya no está activa para los alumnos (T07 ~00:42:35-00:42:59), así que ese notebook **no se puede ejecutar** tal cual. Ver §10 y [05_profit_seeking.md §10](05_profit_seeking.md).

*Fuente: RLM src/utils/plots.py, src/utils/bmeapi.py; T07 ~00:42:35-00:42:59.*

---

## 4. El patrón común de los notebooks

**Idea.** Todos los notebooks de algoritmos siguen el mismo esqueleto, y entender uno es entenderlos todos. Las variantes cambian, casi siempre, **una o dos líneas** del cálculo del objetivo, que es justo el mensaje del profesor: muchos algoritmos con nombres distintos son "una copia uno del otro" con un parámetro cambiado (T06 ~01:16:19-01:16:53).

### 4.1 Esqueleto

1. **Clase del agente** con:
   - `act(s)`: ε-greedy, `if np.random.rand() > self.epsilon: argmax(Q) else: env.action_space.sample()`.
   - `act_no_explore(s)`: greedy, para ver el agente al final.
   - `learn(...)`: calcula el objetivo y actualiza. En los tabulares y los lineales **devuelve el error TD al cuadrado** `(target - qsa) ** 2`; en los deep devuelve la pérdida MSE de `train_on_batch`, media sobre el batch y las `na` salidas (§8.4). Los bucles lo registran como "loss" o "mse" (en `profit-seeking` el valor devuelto ni se guarda).
   - En los deep, además `network()`, `experience()` (guarda `(s, a, r, s1)` con `s1=None` si el episodio terminó) y `sample()`.
2. **Celda de parámetros y bucle de entrenamiento**: por episodio, `reset` → `act` → `step` → `learn` hasta `done`; al acabar el episodio se reduce ε, se guardan la recompensa y la pérdida y cada `n_log` episodios se imprime un resumen.
3. **`plot_training_results`** con la media móvil (en CartPole, LunarLander y *cliff walking*, salvo su bloque de Expected SARSA, que no tiene gráfico; `profit-seeking` usa `pandas` `rolling(100)` en la celda 14).
4. **Render** del agente greedy con `imshow` + `set_data` (la "opción 2" de `render-env-devcon`, §11) en los notebooks de CartPole y LunarLander (TD-q-learning añade además el render en ventana de la celda 11); `cliff-walking` imprime el render de texto paso a paso (celdas 11, 18 y 23) y `profit-seeking` no renderiza: dibuja el NAV del activo, el del benchmark y los tramos invertidos (celda 18).

Todos usan la **API antigua de gym** (`s = env.reset()`, `s1, r, done, info = env.step(a)`), sin distinguir terminación de truncado. Ver [02_diseno_de_entornos.md §2.4](02_diseno_de_entornos.md).

### 4.2 Exploración y step size

- **ε lineal por episodio con suelo:** `epsilon_decay = (1 - min_epsilon) / (nepisodes * f)` y, al final de cada episodio, `epsilon = max(epsilon - epsilon_decay, min_epsilon)`, con `f` = 0,9, 0,8, 0,75 o 0,5 según el notebook (tabla de §4.3). Es justo el decaimiento "didáctico" que el profesor criticó: depende del número de episodios fijado de antemano, puede bajar demasiado pronto o demasiado tarde, y no responde a si el agente mejora (T06 ~03:03:41-03:05:07). Lo correcto según la clase: evaluar la política periódicamente sin exploración y bajar ε según la mejora, con un tope de tiempo opcional (T06 ~03:04:52-03:05:16). Ver [04_exploracion_memoria_y_estabilidad.md §3](04_exploracion_memoria_y_estabilidad.md).
- **α fijo** en los tabulares y en los deep (learning rate de Adam), y **α lineal decreciente** en los lineales (§7). APP p.74 (*Step Size* α) explica la diferencia: un α muy grande puede dar mucha importancia a un hecho poco probable, uno menor recoge más casuística y uno muy pequeño puede quedar atascado en mínimos locales; la estrategia clásica es empezar alto e ir bajando conforme mejora el aprendizaje ("alpha decay"), como hacen los lineales, aunque "nos despreocupamos un poco" usando optimizadores como Adam o RMSProp, como hacen los deep. La diapositiva añade que la novedad también aplica a α: debería ser mayor en estados nuevos, proporcional al número de visitas. Ninguno de los notebooks lo hace. Ver [04_exploracion_memoria_y_estabilidad.md §5.4](04_exploracion_memoria_y_estabilidad.md).
- **Inicialización**: tablas con ruido gaussiano pequeño (`np.random.randn(na) * init_factor`, `init_factor=0.001`) y pesos lineales con `init_factor` 0,01 o 0,1. El profesor recordó que inicializar con ceros, con 1 o con −1 tiene impacto (T06 ~03:30:26-03:30:46).

### 4.3 Tabla resumen de los 17 notebooks

| Notebook | Entorno | Algoritmo | Episodios | γ | α | ε (1 → mín., en qué fracción) |
|---|---|---|---|---|---|---|
| `TD-q-learning` | CartPole-v1 discretizado | Q-learning tabular | 10.000 | 0,995 | 0,1 | → 0,05 en el 90 % |
| `TD-sarsa` | CartPole-v1 discretizado | SARSA | 10.000 | 0,995 | 0,1 | → 0,05 en el 90 % |
| `TD-expected-sarsa` | CartPole-v1 discretizado | Expected SARSA | 10.000 | 0,995 | 0,1 | → 0,05 en el 90 % |
| `TD-double-q-learning` | CartPole-v1 discretizado | Double Q-learning | 10.000 | 0,995 | 0,05 | → 0,05 en el 90 % |
| `cliff-walking` | CliffWalking-v0 | Q-learning, SARSA y Expected SARSA | 300 cada uno | 0,99 | 0,1 | → 0,05 en el 50 % |
| `linear-approximation` | LunarLander-v2 | Q-learning lineal | 2.000 | 0,99 | 0,001 → 8·10⁻⁷ en el 90 % | → 0,05 en el 75 % |
| `linear-sarsa` | LunarLander-v2 | SARSA lineal | 2.000 | 0,99 | 0,001 → 8·10⁻⁷ en el 90 % | → 0,05 en el 75 % |
| `linear-expected-sarsa` | LunarLander-v2 | Expected SARSA lineal | 2.000 | 0,99 | 0,01 → 8·10⁻⁷ en el 90 % | → 0,05 en el 50 % |
| `deep` | LunarLander-v2 | DQN (replay uniforme, sin red objetivo) | 100 | 0,995 | 10⁻⁴ (Adam) | → 0,05 en el 90 % |
| `deep-fixed` | LunarLander-v2 | DQN + red objetivo fija | 100 | 0,995 | 10⁻⁴ | → 0,05 en el 90 % |
| `deep-double` | LunarLander-v2 | Double DQN | 100 | 0,995 | 10⁻⁴ | → 0,05 en el 90 % |
| `deep-dueling` | LunarLander-v2 | Dueling DQN (en realidad DQN, §8.4) | 100 | 0,995 | 10⁻⁴ | → 0,05 en el 90 % |
| `deep-expected-sarsa` | LunarLander-v2 | DQN con objetivo esperado | 100 | 0,995 | 10⁻⁴ | → 0,05 en el 90 % |
| `deep-rainbow` | LunarLander-v2 | Dueling + Double + Expected + CER | 1.000 | 0,995 | 10⁻⁴ | → 0,05 en el 90 % |
| `profit-seeking` | `AlgoEnv` propio (Ibex, diario) | DDQN + CER | 3.000 | 0,995 | 10⁻³ | → 0,03 en el 80 % |
| `insights` | River Swim propio y celdas de cálculo | — (exploración por conteos, γ, `deque`, `%timeit`) | — | — | — | — |
| `render-env-devcon` | CartPole-v1 | — (acciones aleatorias para probar el render) | — | — | — | — |

*Fuente: RLM, celdas de clase y de parámetros de cada notebook; APP p.74; T06 ~01:16:19-01:16:53, ~03:03:41-03:05:16, ~03:30:26-03:30:46.*

---

## 5. Los cuatro notebooks TD tabulares (CartPole discretizado)

Teoría de Q-learning, SARSA, Expected SARSA y Double Q-learning, y lo que se dijo de ellos en el repaso: [00_repaso_rl.md §9-12](00_repaso_rl.md); más a fondo, [T2 02_td_sarsa_qlearning.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md) y [T1 02_td_sarsa_qlearning.md](../raw/referencias/rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md).

### 5.1 `TD-q-learning` (uno de los dos más importantes)

**Estructura** (12 celdas, 0-11):
- Celda 0: título "Q-LEARNING - SUTTON".
- Celda 2: clase `QLearning` mínima: `Qvalues = defaultdict(lambda: np.random.randn(na) * init_factor)`, `act` ε-greedy y `learn` con
$$
Q(s,a)\leftarrow(1-\alpha)\,Q(s,a)+\alpha\big(r+\gamma\max_{a'}Q(s',a')\big),\qquad \text{objetivo}=r\ \text{si}\ \texttt{done}.
$$
  Es literalmente la fórmula de APP p.22 en su forma "(1−α)·valor antiguo + α·target".
- Celdas 3-4: el **pseudocódigo** en una celda *raw* (no ejecutable): crear el entorno y el agente y, por episodio, `act → step → learn → s = s1` hasta `done`. Es la traducción del pseudocódigo de Sutton y Barto de APP p.25.
- Celda 6 ("Un poquito más de verbose"): la clase completa, con `encode` (discretización), `getQs`, `act_no_explore` y valores por defecto `gamma=0.99, epsilon=0.1, alpha=0.5, grid_step=20`.
- Celda 7: discretización, parámetros y entrenamiento (abajo).
- Celda 8: `plot_training_results(window=100, scale="log")`.
- Celda 9: render del agente greedy durante 500 pasos dentro del contenedor; celdas 10-11: "RENDER OUTSIDE THE DEVCONTAINER" con `env.render()` en modo ventana e impresión del último paso.

**Discretización** (celda 7). Se juegan 500 episodios con acciones aleatorias guardando los estados; el rango de cada una de las 4 variables (posición y velocidad del carro, ángulo y velocidad angular del palo) es la diferencia entre sus **percentiles 90 y 10** (la alternativa comentada usa máximo menos mínimo, con el comentario "grid 200" frente a "grid 50"), y
$$
\texttt{encode}(s)=\operatorname{round}\!\Big(\frac{s}{\text{rango}/50}\Big)\ \text{como tupla de enteros},
$$
es decir, celdas de anchura (p90 − p10)/50 por variable, sin límite en el número de celdas. La tupla es la clave del `defaultdict`.

**Parámetros** (celda 7): `grid_step=50`, `gamma=0.995`, `epsilon=1`, `alpha=0.1`, `nepisodes=10000`, `n_log=500`, `min_epsilon=0.05`, `epsilon_decay=(1-0.05)/(10000·0.9)≈1,06·10⁻⁴`. Se registran los **pasos sobrevividos** por episodio (en CartPole la recompensa es +1 por paso, así que coincide con el retorno sin descontar; máximo 500) y la media del error TD² del episodio. El log imprime `max_step`, que en realidad es la **media** de los últimos `n_log` episodios.

**Lo que se dijo en clase:**
- Demo en directo con render (T06 ~01:05:54-01:06:32), §1.
- **APP p.51-52** (Estados, CartPole) muestran el dibujo del carro con el palo y sus cuatro magnitudes rotuladas (*Angle*, *Angular velocity*, *Velocity* y *Position* respecto a un 0 marcado en el eje) y preguntan qué características pasar al algoritmo y, en p.52, "¿Y si quisiéramos hacer un Q-learning? ¿Es posible?". Los cuatro notebooks TD son la respuesta: sí, discretizando. APP p.53 lo recoge entre los requisitos del estado: "Discretos en aprendizaje tabular" y "Estandarizar, normalizar las variables".
- CartPole como ejemplo de diseño del estado: los alumnos propusieron posición, distancia y velocidad; las 4 características del entorno son las de arriba, y si se quita una cuesta mucho converger a la solución perfecta de 500 pasos, aunque sea un problema sencillo (T06 ~02:13:57-02:16:53). Más información también puede complicarlo por dimensionalidad (T06 ~02:16:57-02:17:19). Ver [02_diseno_de_entornos.md §5.3](02_diseno_de_entornos.md).
- Para usar Q-learning tabular con estados continuos hay que **discretizar**, y el nivel de discretización decide lo bien que aprende: muy fino → espacio inmenso que no converge; muy grueso ("alto/bajo") → mucho ruido; algo como **5 o 10 rangos**; el notebook, con Expected SARSA, llega a la solución perfecta discretizando (T06 ~02:17:42-02:18:59). Añadió que a veces se cree tener un problema continuo y es pereza de discretizar *(dudoso)*, y que no hay que saltar enseguida a soluciones complejas *(dudoso: "no paséis a soluciones que sean espectaculares")* (T06 ~02:19:08-02:19:26).

**Nota (discretización del código frente a la de clase).** Con `grid_step=50` el notebook usa unas **50 celdas solo dentro del rango central p10-p90** de cada variable (más las de las colas), no 5-10: unas $50^4\approx6\cdot10^6$ celdas posibles en el núcleo, aunque la tabla solo crece con los estados visitados. Ver [02_diseno_de_entornos.md §5.8](02_diseno_de_entornos.md).

**Nota (reproducir el ejemplo numérico de APP p.23-24 con este `learn`).** Con `Qvalues[s0] = [45, 40]`, `Qvalues[s3]` con máximo 100, `alpha=0.1`, `gamma=1`, `learn(s0, a=1, r=5, s1=s3)` calcula `target = 5 + 1·100 = 105` y `Q(s0,a1) = 0,9·40 + 0,1·105 = 46,5`, el valor de la diapositiva, y devuelve como "loss" $(105-40)^2=4.225$.

### 5.2 `TD-sarsa`

- Celda 0 (markdown): SARSA es TD **on-policy**: aprende según su política actual (p. ej. ε-greedy); Q-learning es off-policy porque aprende el máximo (greedy) aunque no se comporte así al entrenar. Los on-policy toman un **camino más seguro**: tienen en cuenta que su política tiene movimientos aleatorios y se apartan del acantilado. Imagen `safer-path-sutton` (§12).
- Celda 2: la clase (que conserva el nombre `QLearning`) con `learn(s, a, r, s1, a1, done)` y objetivo $r+\gamma Q(s',a')$, con el comentario "simplemente añadimos una entrada extra (a1)".
- Celda 3: mismo entorno, discretización y parámetros que §5.1 (`n_log=1000`); el bucle elige `a` antes del bucle, y en cada paso `a1 = agent.act(s1)` antes de `learn`, y luego `a = a1`. Celdas 4-5: gráfico y render.

Lo que dijo el profesor (T06 ~01:07:14-01:12:45): en Q-learning se le dan al agente 4 cosas (estado, acción, recompensa inmediata y estado siguiente); en SARSA, además, **la acción siguiente que realmente toma**, y se usa el valor de esa acción y no el de la que se cree óptima. Con una política 95 % greedy y 5 % aleatoria ambos se parecen mucho, salvo en ese 5 %. Por eso es on-policy: aprende y ejecuta con la política actual. En el código el cambio es mínimo (indexar por la nueva acción) y la política final es muy distinta. Le importa menos que aprendamos SARSA que entender que estos algoritmos surgen de **pequeñas modificaciones**.

### 5.3 `TD-expected-sarsa`

- Celda 0: como Q-learning, pero con el **valor esperado** del siguiente estado según la política actual, en vez del máximo; esta pequeña modificación lleva a un camino más seguro, como SARSA. Es computacionalmente más complejo que SARSA, pero elimina la aleatoriedad de elegir `a1`, así que suele rendir mejor con el mismo número de episodios. Imagen `expected-sarsa-sutton` (§12).
- Celda 2: `ExpectedSarsa.learn` con
$$
\mathbb E_\pi[Q(s',\cdot)]=(1-\varepsilon)\max_{a'}Q(s',a')+\varepsilon\cdot\operatorname{media}_{a'}Q(s',a'),
$$
  y el comentario "Question → Why not e.g., median?".
- Celdas 3-5: mismos parámetros que §5.1 (`n_log=1000`), gráfico y render.

Lo que dijo el profesor (T06 ~01:15:08-01:16:19): para más estabilidad todavía, en vez de la acción tomada se usa la media según la política actual: con una política 50 % greedy y 50 % aleatoria, el valor es 50 % el de la acción que maximiza más 50 % la media del resto *(dudoso: "la media del resto"; el código usa la media de todas)*. A la pregunta del notebook: posiblemente se podría usar la **mediana**; que lo probemos y "conseguiréis resultados similares", y cualquier otra idea, porque en RL la lista de algoritmos es inmensa (T06 ~01:16:08-01:16:37). Coincide con APP p.27-28.

**Nota (por qué la media de todas).** Con ε-greedy la parte aleatoria elige uniformemente entre **todas** las acciones, incluida la greedy, así que la fórmula del código es la expectativa exacta. Con la mediana dejaría de ser el valor esperado de la política, aunque sería un estimador más robusto. Ver [00_repaso_rl.md §11](00_repaso_rl.md).

**Nota (ε del objetivo esperado).** En todas las variantes Expected (`TD-expected-sarsa`, `linear-expected-sarsa`, `deep-expected-sarsa` y `deep-rainbow`, celda 2) el objetivo usa `self.epsilon`, la misma ε de exploración, que decae de 1 a 0,05 por episodios. Con ε ≈ 1 el objetivo es $r+\gamma\,\text{media}_{a'}Q(s',a')$, el valor de la política aleatoria; al final se parece al de Q-learning (95 % el máximo, 5 % la media). Es la expectativa *on-policy* correcta, pero el objetivo no es estacionario: cambia con el calendario de ε. Si se lleva al `DDQNAgent` de la práctica (§13, punto 7), la ε del objetivo y la de exploración quedan atadas, salvo que se separen a propósito.

### 5.4 `TD-double-q-learning`

- Celdas 0-2: "esta pequeña modificación pretende resolver el **sesgo de maximización** del Q-learning tradicional", con las imágenes `dql-diagram-sutton` y `dql-pseudo-code-sutton` (§12).
- Celda 4: `DoubleQLearning` con dos tablas `Q1` y `Q2`. `act_no_explore` usa $\arg\max_a(Q_1+Q_2)$. En `learn` se sortea con probabilidad 0,5 cuál se actualiza (`q_update`) y cuál evalúa (`q_target`):
$$
a^*=\arg\max_a Q_{\text{upd}}(s',a),\qquad Q_{\text{upd}}(s,a)\leftarrow(1-\alpha)Q_{\text{upd}}(s,a)+\alpha\big(r+\gamma\,Q_{\text{tgt}}(s',a^*)\big).
$$
  Es exactamente el pseudocódigo de Sutton y Barto de la imagen y de APP p.29-30.
- Celda 5: mismos parámetros que §5.1 salvo **`alpha=0.05`**; celdas 6-7: gráfico y render.

Lo que dijo el profesor (T06 ~01:17:03-01:18:46): el ejemplo de la imagen. Desde A, ir a la derecha da 0; ir a la izquierda lleva a B, desde donde se obtiene una recompensa normal de media −0,1 y desviación 1. Si la partida empieza mal (un −0,7) la creencia inicial ya está sesgada y cuesta salir de ella; si empieza bien (un 0,4) el agente se lo cree más. Un problema tan sencillo ya tiene un problema de inicialización. Double Q-learning usa dos diccionarios y pondera las creencias de uno con las del otro, reduciendo el sesgo. Muchas mejoras consisten en crear dos redes o una copia que recibe los pesos cada cierto tiempo, "todo en busca de esa estabilidad", y esa es la palabra con la que hay que quedarse (T06 ~01:18:40-01:19:25).

### 5.5 Notas comunes a los cuatro

- **Nota (error en el muestreo para la rejilla).** Las celdas que recogen los 500 episodios aleatorios escriben `s1, *_, done = env.step(a)` (RLM TD-q-learning celda 7; TD-sarsa celda 3; TD-expected-sarsa celda 3; TD-double-q-learning celda 5). Con la API de 4 valores `(obs, r, done, info)` eso asigna a `done` el **diccionario `info`**, no el indicador de fin (con la de 5 valores pasaría lo mismo). Consecuencia razonada para gym 0.23.1 *(no verificado ejecutando)*: `info` es `{}` (falso) hasta que el envoltorio `TimeLimit` añade `'TimeLimit.truncated'` al llegar a 500 pasos, de modo que cada episodio de muestreo dura 500 pasos y la mayoría de los estados guardados son posteriores a la caída del palo (gym avisa pero sigue integrando la física). Los percentiles 10-90 se calcularían entonces sobre estados que el agente casi nunca verá, con rangos más anchos de lo que se pretende y celdas más gruesas. Corrección: `s1, _, done, _ = env.step(a)`.
- **Nota (fin por tiempo tratado como terminal).** En CartPole-v1 de gym 0.23.1 el límite de 500 pasos llega como `done=True` y `learn` usa `target = r`, aunque el estado no sea terminal. Sesga a la baja el valor de los estados del final. Ver [02_diseno_de_entornos.md §2.4](02_diseno_de_entornos.md).
- **Nota (detalles menores).** En TD-q-learning la celda 11 no reasigna `s` tras `env.reset()`: empieza desde el último `s` de la celda 9. En TD-sarsa la clase se sigue llamando `QLearning`. Las imágenes se enlazan como `.png` en minúscula, pero los ficheros son `.PNG` (§12).

*Fuente: RLM TD-q-learning celdas 0-11; TD-sarsa celdas 0-5; TD-expected-sarsa celdas 0-5; TD-double-q-learning celdas 0-7; APP p.22-30, p.51-53; T06 ~01:05:54-01:06:32, ~01:07:14-01:19:37, ~02:13:57-02:19:26.*

---

## 6. `cliff-walking`: Q-learning, SARSA y Expected SARSA en el acantilado

**El entorno** (`CliffWalking-v0`; imagen `safer-path-sutton`, celda 1): una rejilla de 4 × 12 con salida S abajo a la izquierda, meta G abajo a la derecha y el acantilado entre ambas en la fila inferior. Cada paso cuesta −1 y caer al acantilado, −100 y vuelta a S. La figura dibuja el "Optimal path" (rojo, pegado al borde) y el "Safer path" (azul, por la fila de arriba del todo).

**Estructura** (25 celdas):
- Q-learning (celdas 3-11): la clase mínima (celda 4), el pseudocódigo *raw* (celda 6), la versión "verbose" con `getQs` y `act_no_explore` (celda 8), el entrenamiento (celda 9), el gráfico (celda 10, `window=10`, lineal) y el recorrido greedy impreso paso a paso con `env.render()` (celda 11).
- SARSA (celdas 12-18): `class SARSA(QLearning)` que solo redefine `learn` con `a1` (celda 13), su pseudocódigo con los cambios marcados con flechas `# <--` (celda 15), entrenamiento, gráfico y recorrido.
- Expected SARSA (celdas 19-23): `class ExpectedSARSA(QLearning)` con el objetivo esperado (celda 20); el pseudocódigo es el de Q-learning (celda 21); entrenamiento (celda 22) y recorrido (celda 23), **sin gráfico**. La celda 24 está vacía.

**Parámetros** (celdas 9, 16, 22): `nepisodes=300`, `print_every=50`, `gamma=0.99`, `alpha=0.1`, ε de 1 a 0,05 con `epsilon_decay=(1-0.05)/(300·0.5)=0,95/150≈0,0063`, de modo que llega al mínimo en el **episodio 150**. Se registra la recompensa **descontada** `total_reward += gamma**step * r`.

**Lo que se dijo en clase:**
- **Demo en T06** (~01:11:25-01:12:45): primero Q-learning ("parece que ha aprendido hacia el 150"), luego SARSA, que solo indexa además por la nueva acción, y el resultado es un camino distinto, por arriba *(dudoso: "la arena para arriba")*. Un cambio mínimo, una política final muy distinta. Lo que se busca con estas variantes es **estabilidad**: los entrenamientos son aleatorios porque dependen de los caminos recorridos y de la exploración; se dice que, con todo bien configurado, de 10 entrenamientos 3 pueden fallar por completo (T06 ~01:12:52-01:13:31). En el acantilado, el −100 hace difícil saber que el camino pegado al borde es el óptimo, y con SARSA el aprendizaje suele ser más suave y convergente (T06 ~01:13:58-01:14:31). Añadió *(dudoso)* que, sin un factor de descuento alto, el agente seguramente intentaría acercarse más al estado final, y que quizá SARSA, cambiando γ, volvería a pegarse al acantilado; enseguida pareció matizarlo ("tampoco lo creo") (T06 ~01:14:31-01:14:52). Es coherente con lo que dijo después: en este entorno el −1 por paso ya mete la urgencia (siguiente punto).
- **Recompensa y γ** (T06 ~02:50:34-02:55:15): en el acantilado no hay premio por llegar, sino −1 por cada paso sin llegar, y eso ya mete la urgencia, así que no hace falta un γ bajo. También se podría resolver con +50 al llegar y −1 por paso. Con un γ muy agresivo (0,2) y un premio final de 50, el premio pierde valor y el profesor admitió no tener claro el efecto *(dudoso)*. Ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).
- **El sesgo de evaluar con exploración** (T07 ~00:19:56-00:22:58): con este mismo notebook preguntó cuándo diríamos que ha aprendido mirando la curva de recompensas por episodio. Esa curva se calcula **con exploración**, y su forma sigue al ε decay: ε llega al 5 % hacia el episodio 150 de 300 (paso 0,95/150), y "qué casualidad", es más o menos donde la curva parece estabilizarse; la volatilidad que queda es el 5 % de exploración. Al cambiarlo en directo se vio que había aprendido hacia el **episodio 30** *(dudoso cómo lo mostró: "voy a hacer que va a acabar esto muy al principio"; puede ser que acortara la exploración o que evaluara sin ella)*. Con 20 episodios y una evaluación correcta, sin ese sesgo, ya se vería que la recompensa ha subido y que convendría bajar la exploración y terminar. Por eso no le gusta el ε decay temporal. Ver [04_exploracion_memoria_y_estabilidad.md §2-3](04_exploracion_memoria_y_estabilidad.md).
- **Tamaño del buffer en episodios** (T07 ~03:48:03-03:48:18): como regla general habló de unas 100 partidas, según la diversidad de casuísticas que se quiera cubrir, pero en el *cliff walking* no hacen falta las últimas 100: bastan las últimas 10, porque no hay tantas casuísticas. **Nota:** el notebook es tabular y no tiene buffer; es un ejemplo ilustrativo. Ver [04_exploracion_memoria_y_estabilidad.md §6.3](04_exploracion_memoria_y_estabilidad.md).

**Nota (valores de referencia para leer la curva).** Con γ = 0,99 y `total_reward += gamma**step * r` (con `step` desde 0), el camino óptimo (13 pasos) da $-\sum_{k=0}^{12}0{,}99^k\approx-12{,}25$, el camino intermedio por la segunda fila (15 pasos) ≈ −13,99 y el seguro por la fila superior (17 pasos) ≈ −15,71; cada caída al acantilado suma del orden de −100. Una media móvil entre −12 y −16 indica que el agente ha aprendido a llegar sin caer, que es lo que preguntaba el profesor. Además, `f=0.5` (ε mínimo en el episodio 150 de 300) es el patrón que dibujó en T06 como ejemplo de decaimiento arbitrario: "le bajas muy rápido" y a mitad se queda con un residual del 5 % (T06 ~03:05:36-03:05:59).

**Nota (lo que el notebook no tiene).** No hay evaluación greedy periódica: la curva de la celda 10 (y la de la celda 17 para SARSA) mezcla aprendizaje y exploración, que es justo el sesgo que señaló el profesor. Añadir cada N episodios un episodio con `act_no_explore` (sin `learn`) bastaría para verlo.

**Nota (por qué la diferencia puede ser pequeña).** La figura de Sutton y Barto es con ε = 0,1 constante. Aquí ε baja a 0,05 y el recorrido final se dibuja con la política greedy, así que la diferencia entre el camino de SARSA y el de Q-learning puede ser menor que en la figura o variar entre ejecuciones (sin semilla fijada).

**Nota (bucle sin límite).** Las celdas de recorrido greedy (11, 18 y 23) iteran con `for step in itertools.count()`, sin tope. Si la política greedy entra en un ciclo o apunta contra un borde y se queda en el sitio, el bucle no termina *(no verificado ejecutando; `CliffWalking-v0` en gym 0.23.1 se registra sin `max_episode_steps`, y caer al acantilado devuelve a S sin `done`)*. Es un riesgo real con pocos episodios o al cambiar parámetros, como se hizo en directo en T07. Corrección: `for step in range(100)`. En la práctica, cualquier evaluación greedy debe tener un tope de pasos.

*Fuente: RLM cliff-walking celdas 0-24; APP p.26-28; T06 ~01:11:25-01:14:52, ~02:50:34-02:55:15, ~03:05:36-03:05:59; T07 ~00:18:03-00:23:02, ~03:48:03-03:48:18.*

---

## 7. Los tres notebooks lineales (LunarLander)

Teoría (aproximación funcional, SGD, forma lineal) en [01_aproximacion_funcional_y_dqn.md §2-4](01_aproximacion_funcional_y_dqn.md).

**Entorno.** `LunarLander-v2`: 8 variables de estado continuas (posición, velocidades, ángulo, velocidad angular y dos indicadores de contacto de las patas) y 4 acciones discretas (no hacer nada, motor izquierdo, principal, derecho).

**Modelo** (celda 2 de los tres): $Q(s,\cdot)=\Theta\,s$ con `Qtheta = np.random.randn(na, n_features) * init_factor` (una fila de pesos por acción, sin término independiente) y actualización por semigradiente de la fila de la acción tomada:
$$
\theta_a\leftarrow\theta_a+\alpha\,(y-\theta_a^\top s)\,s.
$$

| Notebook | Objetivo $y$ (no terminal) | `alpha` inicial | ε: fracción de episodios hasta el mínimo | `init_factor` |
|---|---|---|---|---|
| `linear-approximation` | $r+\gamma\max_{a'}Q(s',a')$ | 0,001 | 75 % | 0,01 |
| `linear-sarsa` | $r+\gamma Q(s',a')$ con `a1 = agent.act(s1)` | 0,001 | 75 % | 0,01 |
| `linear-expected-sarsa` | $r+\gamma[(1-\varepsilon)\max+\varepsilon\,\text{media}]$, con la ε actual de exploración (§5.3) | 0,01 | 50 % | 0,1 |

En `linear-sarsa` la clase se sigue llamando `LinearApproximation` (con `learn(s, a, r, s1, a1, done)` y un constructor que ya no acepta `mean`/`std`), igual que `TD-sarsa` conserva `QLearning` (§5.5).

Comunes (celda 3): `nepisodes=2000`, `n_log=250`, `gamma=0.99`, ε de 1 a 0,05, **α lineal decreciente** hasta `min_alpha=8e-7` en el 90 % de los episodios (`alpha_decay=(alpha-8e-7)/1800`), registro de la recompensa acumulada **sin descontar** y del error TD² medio del episodio. El log imprime también α. Celda 4: gráfico lineal con ventana 100. Celda 5: render greedy con el título "Step, rew". Celda 6 (en `linear-approximation` y `linear-expected-sarsa`): muestra la matriz `agent.Qtheta`.

**Lo que se dijo en clase:**
- De tabular a aproximación funcional (T06 ~01:20:00-01:22:44): un diccionario con estados y valores finitos no sirve con estados infinitos (no se puede vivir cada uno muchas veces), no cabe en memoria y no generaliza; en un coche autónomo no te puedes jugar la vida con estados parecidos no vistos y valores muy distintos. Se ajustan pesos en vez de valores y se mantiene la filosofía (argmax). Las regresiones son rápidas y buena primera aproximación (T06 ~01:22:47-01:23:08).
- A la pregunta de si se puede hacer con **árboles**, contestó que entendería que sí, pero que nunca lo ha hecho y no sabe cómo se ajustarían los puntos de corte (él tiene la fórmula para ajustar los pesos del lineal); sugirió preguntarlo a ChatGPT o buscarlo y comentárselo (T06 ~01:23:29-01:24:13, ~01:25:51-01:26:01). Un alumno dijo que no habían visto la parte lineal y el profesor le remitió a este notebook para comentarlo al día siguiente (T06 ~01:24:17-01:24:21).
- Ejemplo del modelo lineal (T06 ~01:24:26-01:25:51): "es muy básico", una matriz de coeficientes *(la transcripción dice "2 matrices", dudoso)*, una fila por acción y una columna por característica (cita pesos como 0,2, 0,7 y 1,2 y un estado como 3, 0, 1,2 y 7). Si para la acción 0 el producto da 20 y el valor observado es 30, se suben los pesos de modo que la siguiente predicción dé unos 22: es como una regresión distinta por acción.
- **APP p.35** (imagen): "Primera aproximación funcional: aproximación lineal", con variantes lineal SARSA y lineal Expected SARSA, "buena aproximación" y "rápido", y tres gráficos del "Ejemplo de CartPole": **Lineal** sube de forma casi continua de ~40 a ~370 pasos en 20.000 episodios; **Lineal Sarsa**, de ~30 a ~280 en 20.000; **Lineal Expected Sarsa** empieza en ~100, tiene un pico de ~200 hacia el episodio 1.000, cae y sube con fuerza hasta la meseta de **500** (el máximo) hacia el episodio ~4.000-5.000, donde se mantiene hasta el 16.000.
- **APP p.36** (imagen): el gráfico de barras del paper de DQN en Atari, puntuación normalizada respecto a un humano (eje de 0 % a ~4.500 %, con una marca de "nivel humano o superior" frente a "por debajo") para varias decenas de juegos, DQN (azul) frente a "Best Linear Learner" (gris), de Video Pinball y Boxing (DQN muy por encima del nivel humano) a Montezuma's Revenge (~0 %). El mensaje: lo lineal "no lo hace tan mal"; los juegos están pensados para funcionar mejor con imágenes; en bolsa se pasan características y lo lineal puede ser buena aproximación; la dispersión de resultados es importante; y la pregunta abierta de qué entienden los autores por "best linear learner". Ver [01_aproximacion_funcional_y_dqn.md §4](01_aproximacion_funcional_y_dqn.md).

**Nota (diapositiva frente a notebooks).** Los gráficos de APP p.35 son de **CartPole** con 16.000-20.000 episodios; los notebooks lineales actuales usan **LunarLander-v2** con 2.000 episodios. Las curvas de la diapositiva salieron, presumiblemente, de una versión anterior del material (con CartPole) y no se reproducen ejecutando estos notebooks.

**Nota (limitaciones del modelo).** Sin término independiente y sin normalizar el estado (el constructor de `linear-approximation` acepta `mean` y `std` pero no los usa), $Q(s,\cdot)$ es proporcional a $s$ y no puede representar un valor base por acción; las variables de LunarLander tienen escalas distintas, lo que hace que el α efectivo varíe por variable. Es coherente con la insistencia del profesor en normalizar (T06 ~02:19:28-02:19:38). Añadir un 1 al vector de estado y estandarizar con los estados de episodios aleatorios es la mejora mínima.

**Nota (el render).** En la celda 5 el título acumula `gamma**step * r` empezando en $\gamma^1$, y la línea final `print(f"Done Rew: ...")` imprime solo **la última recompensa descontada**, no el total. En LunarLander la última recompensa es ±100 al posarse o estrellarse, así que su signo sí indica cómo acabó, pero no es el retorno del episodio.

*Fuente: RLM linear-approximation celdas 0-6; linear-sarsa celdas 0-5; linear-expected-sarsa celdas 0-6; APP p.31-36; T06 ~01:20:00-01:26:22 (árboles y ejemplo lineal, ~01:23:29-01:26:01), ~02:19:28-02:19:38.*

---

## 8. Los seis notebooks deep (LunarLander)

Teoría de DQN, replay, red objetivo, Double, Dueling, Rainbow y CER en [01_aproximacion_funcional_y_dqn.md §5-11](01_aproximacion_funcional_y_dqn.md); más a fondo, [T2 03_dqn.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md) y [T2 04_doble_dqn_y_duelo.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/04_doble_dqn_y_duelo.md). Buffer, batch y frecuencia de aprendizaje en [04_exploracion_memoria_y_estabilidad.md §6-7](04_exploracion_memoria_y_estabilidad.md).

### 8.1 `deep` (uno de los dos más importantes): el esqueleto

**Clase `DQNAgent`** (celda 2):
- `network()`: `Input(n_features) → Dense(hidden, relu) → Dense(hidden, relu) → Dense(na, linear)`, compilada con `loss="mse"` y `Adam(learning_rate=alpha)`.
- `buffer = deque(maxlen=buffer_size)`; `experience(s, a, r, s1, done)` guarda `(s, a, r, s1)` con `s1=None` si `done`.
- `sample()`: **replay uniforme**, `batch_size` índices sin reemplazo.
- `getQs(s)`: `self.model(s).numpy()`, con el comentario de que `model.predict(s)` es lento.
- `learn()`: muestrea un batch, predice `qs = Q(s_list)` y `qs1 = Q(s1_list)` (solo los no terminales, con un contador `k` para emparejarlos), construye `targets = qs` y sustituye solo la entrada de la acción tomada por $r$ (terminal) o $r+\gamma\max_{a'}Q(s',a')$, y hace **un paso** con `train_on_batch`. Devuelve la pérdida.

**Parámetros** (celda 3): `nepisodes=100`, `n_log=25`, ε de 1 a 0,05 en el 90 % de los episodios (paso ≈ 0,0106), `gamma=0.995`, `alpha=0.0001`, `buffer_size=10000`, `batch_size=64`, `hidden_neurons=120`, **`learn_after=batch_size`**.

**Bucle** (celda 3):
1. **Llenado inicial del buffer** con 10.000 transiciones jugadas con `agent.act` y ε = 1, es decir, aleatorias (imprime cada 5.000).
2. **Entrenamiento**: por paso, `act → step → experience`; un contador global `learn_counter` y **un `learn()` cada 64 pasos**; al terminar el episodio se baja ε y se registran recompensa acumulada, recompensa descontada, pasos y la última pérdida; cada 25 episodios se hace **un `learn()` extra** y se imprime.
3. Celda 4: gráfico lineal con ventana 10. Celda 5: render greedy, como en los lineales.

**Nota (llenado inicial frente a lo dicho en clase).** APP p.79 aconseja llenar el buffer antes de aprender: "aleatorias, como mínimo", o mejor con experiencias reales o "algo base", y "no hace falta llenarlo entero" (independencia frente a ruido). En clase lo explicó así: el buffer empieza vacío, y si se juegan 64 pasos y se aprende, el batch solo tiene esas 64 observaciones, muy correlacionadas entre sí; como mínimo hay que dejar pasar un tiempo antes del primer aprendizaje (T06 ~04:14:37-04:15:06). Los deep van más allá del mínimo: llenan el buffer entero (10.000 de 10.000) con acciones aleatorias, y `learn_after = batch_size` solo fija la frecuencia (un `learn()` cada 64 pasos), no una espera inicial. Tras unos 10.000 pasos de entrenamiento esas transiciones aleatorias ya han salido del `deque`. Ver [04_exploracion_memoria_y_estabilidad.md §6.6](04_exploracion_memoria_y_estabilidad.md).

**Nota (presupuesto de aprendizaje).** Con un paso de gradiente cada 64 pasos de entorno y 100 episodios de LunarLander (que duran de decenas a 1.000 pasos), el agente hace del orden de **cientos a un par de miles de actualizaciones** en total. Es un notebook para entender el código, no para resolver el entorno. `deep-rainbow` usa 1.000 episodios.

### 8.2 Diferencias de cada variante respecto a `deep`

| Notebook | Clase | Qué cambia | Parámetros añadidos |
|---|---|---|---|
| `deep-fixed` | `FixedDQNAgent` | `target_model = tf.keras.models.clone_model(model)` con `update_target()` (copia de pesos); el objetivo usa $\max_{a'}Q_{\theta^-}(s',a')$, calculado con `target_model.predict` | `update_target_after = 10 * learn_after` = 640 pasos |
| `deep-double` | `DDQNAgent` | `primary_model` y `target_model` (construida con `network()` y sincronizada); `getQs(s, model)` estático; la principal **elige** $a^*=\arg\max_a Q_\theta(s',a)$ y la objetivo **evalúa** $Q_{\theta^-}(s',a^*)$. Comentario: la objetivo no se entrena y "no hay intercambio de papeles", a diferencia de la Double Q tabular | 640 pasos; gráfico con ventana 20 |
| `deep-dueling` | `DuelingDQNAgent` | `network()` con dos cabezas: `adv = Dense(na)`, centrada restando su media, `v = Dense(1)`, `out = tf.subtract(v, adv)`; una "V2" comentada con `Lambda` (que usaría `K` sin importar y, además, con `K.mean(x[:,1:], keepdims=True)` sin `axis=1`, centraría con la media de todo el batch y no por muestra: tampoco serviría si se descomentara). **Pero** después ejecuta `out = Dense(na)(capa)` (§8.4). Sin red objetivo | — |
| `deep-expected-sarsa` | `ExpectedDQNAgent` | Objetivo $r+\gamma[(1-\varepsilon)\max_{a'}Q(s',a')+\varepsilon\,\text{media}_{a'}Q(s',a')]$ con la ε actual de exploración (objetivo no estacionario, §5.3). Sin red objetivo | — |
| `deep-rainbow` | `RainbowDQNAgent` | **Dueling** activo (salida estándar comentada), **red objetivo**, **CER** en `sample()` (`batch_size − 1` índices al azar entre todos menos el último, más el último añadido) y objetivo **"Expected DDQL"**: $(1-\varepsilon)\,Q_{\theta^-}(s',a^*)+\varepsilon\,\text{media}_{a'}Q_{\theta^-}(s',a')$ con $a^*$ elegida por la principal. Las versiones "Standard Replay" y "Standard DDQL" quedan comentadas | `nepisodes=1000`; 640 pasos |

Lo que dijo el profesor: no hay ningún modelo de RL que predomine; con acciones continuas se descarta la mitad; se puede preguntar a ChatGPT cuál conviene, pero hay que sentirse libre para probar variaciones y combinarlas: lo que se llama "Rainbow" es eso, por ejemplo un Dueling Double Deep Q-learning que junta la filosofía de uno y de otro (T07 ~00:25:05-00:25:55). Y, en general, que muchas mejoras consisten en duplicar la red o tener una copia que recibe los pesos cada cierto tiempo, en busca de estabilidad (T06 ~01:16:37-01:18:46).

### 8.3 Qué tiene y qué no tiene el "Rainbow" de clase

**Nota.** El Rainbow de Hessel et al. combina seis mejoras: Double, PER, Dueling, *multi-step*, distribucional y *noisy nets*. El de clase combina Double, Dueling, CER y el objetivo esperado de Expected SARSA, que no es una de las seis. Es un "Rainbow" en el sentido del profesor (combinar lo que convenga), no el del paper. Ver [01_aproximacion_funcional_y_dqn.md §11](01_aproximacion_funcional_y_dqn.md).

### 8.4 Erratas de los deep

- **`deep-dueling` no es Dueling.** La línea `# Standard out` → `out = Dense(self.na, activation="linear")(capa)` está sin comentar y **sobrescribe** la salida de duelo: el modelo es un DQN estándar y las cabezas `adv` y `v` quedan fuera del grafo (RLM deep-dueling, celda 2). Corrección: comentar esa línea, como en `deep-rainbow`.
- **Signo de la combinación de duelo.** `out = tf.subtract(v, adv)` da $Q=V-(A-\bar A)$, no $V+(A-\bar A)$ (deep-dueling y deep-rainbow, celda 2). Como $A$ se aprende libremente, solo invierte el signo de lo que aprende la cabeza de ventajas y la identificabilidad (media cero) se mantiene: es equivalente, pero confunde al leerlo. Ver [01_aproximacion_funcional_y_dqn.md §9](01_aproximacion_funcional_y_dqn.md).
- **`mse` puede no existir.** `history_losses.append(mse)` se ejecuta al acabar el primer episodio, pero `mse` solo se define tras el primer `learn()` (paso 64). Si el primer episodio termina antes de 64 pasos, salta un `NameError` (todos los deep, celda 3). Es la misma trampa señalada en el LAB ([08_entorno_best_execution.md](08_entorno_best_execution.md)). Corrección: `mse = np.nan` antes del bucle.
- **Aprendizaje extra al imprimir.** Cada `n_log` episodios se hace un `learn()` adicional solo para imprimir la pérdida: entrena un poco más los episodios múltiplos de 25 y la pérdida impresa no es la del registro. Inocuo, pero mezcla monitorización y entrenamiento.
- **La "loss" registrada** es la del último batch del episodio, no una media; y `np.round(mse)` la imprime redondeada a entero.
- **`deep-fixed` usa `predict`** para la red objetivo (`get_target_Qs`), lo que el propio `deep` desaconseja por lento, y su comentario dice que el modelo objetivo "actúa", cuando solo calcula objetivos. Además, con TF 2.15 (Keras 2) `predict` usa por defecto `verbose='auto'` y en un notebook imprime una barra "1/1 [====]" en cada llamada, es decir, una por cada `learn()`, lo que inunda la salida del entrenamiento *(no verificado ejecutando)*. Corrección: `self.target_model(s).numpy()` o `predict(s, verbose=0)`.
- **`deep-rainbow`** tiene un método `encode` que usa `self.minus_factor` y `self.div_factor`, atributos que esta clase no define (sí el `DDQNAgent` de `profit-seeking` y el de la práctica): daría `AttributeError` si se llamara, pero no se llama.
- **Pérdida MSE.** Con `targets = qs` y una sola entrada cambiada, el MSE se promedia sobre las `na` salidas y el gradiente efectivo queda dividido por `na` (4 en LunarLander). Ninguno usa Huber loss, que APP p.80 menciona, junto a *importance sampling*, como remedio de estabilidad para el *Prioritized Experience Replay*. Ver [01_aproximacion_funcional_y_dqn.md §10](01_aproximacion_funcional_y_dqn.md).

*Fuente: RLM deep, deep-fixed, deep-double, deep-dueling, deep-expected-sarsa y deep-rainbow, celdas 0-5; APP p.79-80; T06 ~01:16:37-01:18:46, ~04:14:37-04:15:06; T07 ~00:25:05-00:25:55.*

---

## 9. `insights`: exploración, γ y eficiencia

Cuatro bloques independientes, sin agente: River Swim (celdas 1-9), efecto de γ (celdas 10-18), `deque` (celdas 19-23) e `.iloc` frente a NumPy (celdas 24-38). El profesor ejecutó los dos primeros en T06 y el último en T07.

### 9.1 River Swim (celdas 1-9)

- Imagen `river_swim` (celda 2, §12). Clase `SwimRiver(steps, episode_length, total_states, dec=3)`: **política aleatoria** (en el estado 1 siempre derecha, en el último siempre izquierda, en el resto 50/50). Transiciones (`do_action`): izquierda baja un estado con probabilidad 1; derecha desde el 1 avanza con 0,6 o se queda con 0,4; desde los demás retrocede con 0,05, se queda con 0,6 o avanza con 0,35. `run()` cuenta las visitas a cada estado e imprime número y porcentaje. No hay recompensas: solo se mide **cuánto se visita cada estado**.
- Celdas 4-5: 6 estados con 100.000 pasos y 20 estados con 1.000.000 pasos (un único episodio en ambos).
- `ModSwimRiver` (celdas 6-9): misma dinámica, pero elige izquierda o derecha con pesos $1-n_{\text{prev}}/n_{\text{tot}}$ y $1-n_{\text{sig}}/n_{\text{tot}}$, donde $n$ son las visitas acumuladas a los estados vecinos: favorece ir hacia el **vecino menos visitado** (exploración por conteos). Se ejecuta con 6 estados y 100.000 pasos, y con 20 estados, 1.000.000 de pasos y **episodios de 1.000 pasos** (vuelve al estado 1 cada 1.000).

Lo que dijo el profesor (T06 ~03:26:17-03:30:18 y ~03:47:37-03:47:50): ε-greedy es el más famoso porque es el primero que se enseña, pero **no explora activamente** el entorno. Con una política 100 % aleatoria hacen falta muchísimos pasos para llegar una vez al 6, y con 8, 10 o 13 estados muchos más; con 20 estados y un millón de pasos "se atasca, no llega". El notebook permite configurar cuántos estados hay y cuenta cuántas veces se cae en cada uno *(dudoso: cita un 1 % en el último estado)*. Si la recompensa solo llegara al final, sería muy complicado. La exploración activa (dar más peso a acciones poco probadas o que llevan a estados poco visitados) exige llevar un registro, y es solo una de las técnicas. Presentó el ejemplo diciendo que los principales escollos del RL son programar un entorno con asunciones realistas, las recompensas y, sobre todo, la exploración, que llamó "un dolor de cabeza" (T06 ~03:26:42-03:27:00); lo repitió en T07: recompensas y exploración son "2 mundos" en los que ser todo lo creativo que se quiera (T07 ~00:40:00).

**APP p.72** (Exploración, River Swim) usa la misma figura con el titular "No explora activamente el entorno" (sobre ε-greedy), el aviso "¡La exploración no es fácil! + Research" y una lista de soluciones: inicialización de los valores; intrínseco (métodos *policy* o *noisy networks*); experiencia: contar o pseudocontar, eligiendo acciones menos frecuentes o subiendo ε; predicción (error cometido o confianza en el rango de los resultados, $\arg\max(V(a))+U(a)$); recompensas (*reward shaping*, *curriculum learning*, *curiosity-driven*, *auxiliary tasks*). `ModSwimRiver` es la opción "contar". Desarrollo en [04_exploracion_memoria_y_estabilidad.md §4](04_exploracion_memoria_y_estabilidad.md).

**Nota (código frente a figura).** `get_action` fuerza la derecha en el estado 1 y la izquierda en el último (también en `ModSwimRiver`), así que dos grupos de transiciones de la figura no se simulan nunca: el bucle río abajo de probabilidad 1 del estado 1 y las transiciones río arriba del último estado (0,4 retroceder y 0,6 quedarse). Con eso, el estado final se visita y se abandona en el paso siguiente, lo que reduce su porcentaje de visitas respecto a un agente que pudiera quedarse allí. Y, a diferencia del River Swim clásico, esta versión no tiene recompensas.

**Nota (simulación propia).** Reimplementando las reglas de las celdas 3 y 7 (semillas 0 y 1, resultados casi iguales), los porcentajes de visitas son:

| Caso | Último estado | Estado más lejano alcanzado |
|---|---|---|
| Aleatoria, 6 estados, 10⁵ pasos | ≈ 0,25-0,28 % | 6 |
| Aleatoria, 20 estados, 10⁶ pasos | 0 % | 13 (≈ 1 visita por millón) |
| Por conteos, 6 estados, 10⁵ pasos | ≈ 3,0 % | 6 |
| Por conteos, 20 estados, 10⁶ pasos, episodios de 1.000 | ≈ 0,013 % (≈ 130 visitas) | 20 |

Con la política aleatoria las visitas caen aproximadamente a un tercio por cada estado que se sube (37 %, 42 %, 14 %, 4,6 %, 1,5 %, 0,5 %...); con conteos la caída es mucho más suave y se llega al estado 20 aun reiniciando cada 1.000 pasos. Tiempo hasta la primera llegada y más casos en [04_exploracion_memoria_y_estabilidad.md §4](04_exploracion_memoria_y_estabilidad.md).

### 9.2 Efecto de γ (celdas 10-18)

- Imagen `gamma` (celda 11, §12): dos estados que se alternan, $S_0\to S_1$ con $r=1$ y $S_1\to S_0$ con $r=2$. Es la figura de APP p.75-76 ("Gamma o factor de descuento / valor del tiempo. ¿Qué gamma? ¿De qué depende?"), donde aparece junto a esta lista: valor del tiempo, signo de las recompensas, problema episódico o continuo, número de steps del episodio y si ese número es fijo o variable ("Mucho cuidado con los que tienen una duración (n steps) variable"). Ver [03_recompensas_y_descuento.md §11](03_recompensas_y_descuento.md).
- `run_simulation(gamma, episodes)` suma $\sum_{i}\gamma^i r_i$ con $r_i=2$ si $i$ es par y $1$ si es impar, e imprime la suma y el último factor $\gamma^{i}$.
- Pregunta "¿Quién gana?": (1) γ = 1 y 100 pasos; (2) γ = 0,99 y 1.000 pasos; (3) γ = 0,98 y "1.000.000" de pasos. Después, "¿Es 0,95 alto?" con `0.95**50`, `0.95**100` y `0.95**200`.

Lo que dijo el profesor (T06 ~03:47:57-03:50:29): los dos primeros salen más o menos equivalentes; el tercero, aunque sean un millón de pasos, vale menos que los otros dos, aproximadamente la mitad. Con 0,95, después de 50 pasos queda poco valor, después de 100 casi nada y después de 200 prácticamente 0 *(cifras confusas, dudoso: probablemente "7 % tras 50, 0,5 % tras 100 y prácticamente 0 tras 200", que cuadra con los valores exactos de la Nota siguiente)*: con γ = 0,95 las acciones de ahora no tienen en cuenta lo que pase dentro de 200 pasos. Hay que calibrar γ pensando en el horizonte temporal que deben tener en cuenta las acciones, y eso depende de cómo se definan los steps. Ver [03_recompensas_y_descuento.md §10-12](03_recompensas_y_descuento.md).

**Nota (valores exactos).**

| Caso | Suma | Último factor |
|---|---|---|
| γ = 1, 100 pasos | 150 | 1 |
| γ = 0,99, 1.000 pasos | ≈ 150,24 | ≈ 4,4·10⁻⁵ |
| γ = 0,98, 10⁷ pasos | ≈ 75,25, el límite $2{,}98/(1-0{,}98^2)$ | 0 (subdesbordamiento) |

Gana por poco el caso 2. El caso 3 vale la mitad porque el horizonte efectivo $1/(1-\gamma)$ es de 50 pasos frente a 100: los pasos de más no aportan nada. Y $0{,}95^{50}\approx0{,}077$, $0{,}95^{100}\approx0{,}0059$ y $0{,}95^{200}\approx3{,}5\cdot10^{-5}$: un 7,7 %, un 0,6 % y un 0,004 %.

**Nota (dos erratas).** La celda 16 llama a `run_simulation(0.98, 10e6)`, que son **10.000.000** de pasos, no el 1.000.000 del enunciado de la celda 13 (el profesor dijo "un millón"; el resultado es el mismo porque la serie ya ha convergido). Y el código empieza con $r=2$ ($i=0$ par) mientras que la figura empieza en $S_0$ con $r=1$: equivale a empezar en $S_1$. Con γ = 1 no cambia nada; con 0,99 la suma pasaría a ≈ 149,74.

### 9.3 `deque` frente a lista (celdas 19-23)

Con un millón de ceros, `%%timeit` de tres formas de mantener una ventana de tamaño fijo: lista con `if len(list_) == size: pop(0); append(0)` (celda 21), lista con `pop(0)` y `append(0)` sin comprobar (celda 22) y `deque(maxlen=size).append(0)` (celda 23), que descarta el más antiguo solo. Es la justificación de usar `deque` como buffer de experiencias: `list.pop(0)` desplaza todos los elementos ($O(n)$) y `deque` es $O(1)$.

Es el primer "consejo básico" de APP p.79 (Experience Replay): `from collections import deque`, `buffer = deque(maxlen=buffer_size)` y `buffer.append((s, a, r, s1))`, con la indicación "Deque → No .pop(0)". En clase lo explicó así: el buffer no es simplemente una lista a la que se hace `pop(0)`, sino una colección específica, más eficiente, para guardar las experiencias; en RL se busca estabilidad, simplicidad y eficiencia, porque hay que evaluar muchísimas veces y hacen falta sistemas "súper súper rápidos" (T06 ~04:13:11-04:13:41). Esta sección de `insights` es la demostración numérica de ese consejo; en las partes de la transcripción revisadas no se ejecutó. Ver [04_exploracion_memoria_y_estabilidad.md §6.5](04_exploracion_memoria_y_estabilidad.md).

### 9.4 `.iloc` frente a NumPy y diccionarios (celdas 24-38)

Con un vector de 10⁶ valores uniformes (`size = int(10e5)`, es decir, un millón) y `nstep = 500`, crea un DataFrame de dos columnas, un array 2 × 10⁶, un diccionario de arrays, la columna suelta y `df.to_dict()` (diccionario de diccionarios), comprueba que las 7 formas de acceso devuelven el mismo valor (celda 27) y mide cada una con `%%timeit`: `df.iloc[nstep]['col1']`, `df['col1'].iloc[nstep]`, `df['col1'].values[nstep]`, `numpy_[0][nstep]`, `dict_['col1'][nstep]`, `col1[nstep]` y `double_dict['col1'][nstep]`. Es la demo de T07 ~00:12:35-00:15:56: una variante "7 veces más rápida" que otra, convertir a array "3 veces menos", trabajar directamente con el array "8 veces menos", unos 45 nanosegundos en la más rápida, y lo que con DataFrames tarda una hora puede ser cuestión de un segundo *(cifras dudosas, sin saber qué variante compara con cuál)*. Añadió que el resultado depende del número de filas y columnas y del orden de la indexación (si `iloc` va antes o después de seleccionar la columna) (T07 ~00:14:40-00:14:46). Explicó también cómo funciona `%%timeit`: repite cada celda para estimar la media y la desviación del tiempo y ajusta el número de repeticiones, de modo que en la variante lenta hizo unas 10.000 y en la rápida unos 10 millones en el mismo tiempo; le parece un comando muy útil cuando algo va lento, y relevante en aplicaciones de trading (T07 ~00:15:19-00:15:56). Desarrollo y lectura de las cifras en [02_diseno_de_entornos.md §8](02_diseno_de_entornos.md).

**Nota (relevancia).** El entorno de la práctica trabaja con DataFrames por día (LAB), y el profesor dijo que la práctica usa pandas "por facilitarlo" (T07 ~00:12:19-00:12:31). Es el primer sitio donde buscar velocidad si el entrenamiento resulta lento.

*Fuente: RLM insights celdas 0-39; notebooks/static/river_swim.png y gamma.png; APP p.72, p.75-76, p.79; T06 ~03:26:17-03:30:46, ~03:47:37-03:50:29, ~04:13:11-04:13:41; T07 ~00:12:19-00:15:56, ~00:40:00.*

---

## 10. `profit-seeking`: el algoritmo de inversión con RL

Descripción completa, erratas y relación con las diapositivas en [05_profit_seeking.md §10](05_profit_seeking.md). Resumen:
- **Lo que dijo el profesor** (T07 ~00:42:35-00:44:00): "esto ya no funciona", porque usaba una API de BME con datos del Ibex que dejó de estar activa para los alumnos; los datos eran una especie de diccionario por día o por activo; lo hizo en una tarde, antes de ChatGPT, y no se hace responsable de que esté bien calculado; es "muy, muy cortito" y sirve para ver los componentes y contrastarlos con la práctica. El estado tiene el "Sharpe" de los últimos 15, 30 y 60 días "o algo así" (el código usa **10, 30 y 60**). CASOS p.14 (título "Profit-seeking Algorithms", con el único texto "Notebook" en el centro) marca el paso a este notebook, y p.15 es la portada de *Execution Algorithms*.
- **Datos** (`EnvData`, celda 2): cierres del Ibex y del benchmark vía `APIBMEHandler('IBEX', 'rl-algo')`; train 2017-2018 y test 2019-2020 (celda 7). Estado de 7 columnas: retorno, ranking normalizado de mediana/desviación a 10, 30 y 60 días, mediana transversal de retornos, retorno del benchmark y posición.
- **Entorno** (`AlgoEnv`, celda 3): acciones mantener/comprar/vender; ticker e inicio aleatorios; el episodio acaba al vender o al final de los datos; recompensa $\ln(1+r^{\text{activo}}_{t+1}-r^{\text{bench}}_{t+1})$ si se está invertido; acción aleatoria ponderada (0,8; 0,1; 0,1).
- **Agente** (`DDQNAgent`, celda 4): Double DQN + red objetivo + CER, el mismo que el de la práctica (§13).
- **Parámetros** (celda 9): 3.000 episodios, ε de 1 a 0,03 en el 80 %, γ = 0,995, α = 10⁻³, buffer 5.000, batch 64, 64 neuronas, `learn_after=64`, `update_target_after=640`.
- **Erratas principales**: la evaluación de "test" usa el entorno de entrenamiento (`test_data` nunca se usa); `self.position == 0` en `_sell` es una comparación; capa densa duplicada y colgada; cronología de cobro asimétrica entre compra y venta *(intención dudosa)*.

*Fuente: RLM profit-seeking celdas 0-18; src/utils/bmeapi.py; CASOS p.14-15; T07 ~00:42:13-00:44:08.*

---

## 11. `render-env-devcon`: cómo ver los entornos

- Fuera del contenedor (celda 3): `env.render()` en modo ventana durante hasta 1.000 pasos aleatorios de CartPole.
- Dentro del contenedor hay que arrancar Jupyter con `xvfb-run -a -s "-screen 0 1400x900x24" jupyter-lab` (celda 5) y dibujar los fotogramas `env.render(mode='rgb_array')` en el notebook:
  - **Opción 1** (celdas 7-8): `show_state(env, step)` crea o limpia una figura, dibuja con `imshow` y refresca con `display.clear_output(wait=True)` y `display.display(plt.gcf())`.
  - **Opción 2** (celdas 9-10), "más eficiente": `imshow` una sola vez y luego `img.set_data(...)` en cada paso. Es la que usan los notebooks de CartPole y LunarLander (`cliff-walking` usa el render de texto y `profit-seeking` no renderiza; §4.1).

**Nota:** con la instalación local (`pip install -e .`) no hace falta `xvfb`, pero en gym 0.23.1 `render(mode='rgb_array')` abre además una ventana de pygame (README, §2.2).

*Fuente: RLM render-env-devcon celdas 0-10; README.md.*

---

## 12. Imágenes de `notebooks/static/`

| Fichero | Contenido | Usado en |
|---|---|---|
| `safer-path-sutton.PNG` | Rejilla del acantilado de Sutton y Barto: S y G en la fila inferior con "The Cliff" entre ellas; $R=-1$ por paso y flechas verdes de $R=-100$ que devuelven a S; "Optimal path" (rojo, junto al borde) y "Safer path" (azul, por la fila superior) | `TD-sarsa` celda 0; `cliff-walking` celda 1 |
| `expected-sarsa-sutton.PNG` | Fórmula de Expected SARSA: $Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma\,\mathbb E_\pi[Q(S_{t+1},A_{t+1})\mid S_{t+1}]-Q(S_t,A_t)]$, desarrollada como $\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)$ | `TD-expected-sarsa` celda 0 |
| `dql-diagram-sutton.PNG` | Ejemplo del sesgo de maximización: desde A, "right" lleva a un terminal con 0 y "left" lleva a B con 0, y desde B varias acciones llevan a un terminal con recompensa $\mathcal N(-0{,}1;1)$. Curva de "% left actions from A" en 300 episodios: Q-learning (rojo) sube desde el 50 % hasta ~93 % hacia el episodio 20 y baja despacio hasta ~10 % al final; Double Q-learning (verde) baja enseguida, ~10 % hacia el episodio 50, y se queda cerca de la línea discontinua "optimal" del 5 % | `TD-double-q-learning` celda 1 |
| `dql-pseudo-code-sutton.PNG` | Pseudocódigo "Double Q-learning, for estimating $Q_1\approx Q_2\approx q_*$": ε-greedy sobre $Q_1+Q_2$; con probabilidad 0,5, $Q_1(S,A)\leftarrow Q_1+\alpha(R+\gamma Q_2(S',\arg\max_a Q_1(S',a))-Q_1)$, y si no, la simétrica | `TD-double-q-learning` celda 2 |
| `river_swim.png` | Seis estados en fila. Discontinuas (río abajo): probabilidad 1 al anterior (y quedarse en el 1). Continuas (río arriba): 0,4 quedarse y 0,6 avanzar en el 1; 0,05 retroceder, 0,6 quedarse y 0,35 avanzar en los intermedios; 0,4 retroceder y 0,6 quedarse en el 6. El estado 2 (con borde más grueso) y todas sus transiciones (las tres de río arriba y la discontinua hacia el 1) están en rojo. El código no usa el bucle discontinuo del 1 ni las de río arriba del 6 (§9.1) | `insights` celda 2; APP p.72 |
| `gamma.png` | Dos estados: $S_0\to S_1$ con $r=1$ y $S_1\to S_0$ con $r=2$ | `insights` celda 11; APP p.75-76 |

**Nota (el 5 % óptimo).** En el ejemplo de Sutton y Barto la exploración es ε = 0,1, y por eso el óptimo no es 0 % de "left" sino 5 %: la mitad de las acciones aleatorias van a la izquierda.

**Nota (mayúsculas).** Los notebooks enlazan `safer-path-sutton.png`, `expected-sarsa-sutton.png`, `dql-diagram-sutton.png` y `dql-pseudo-code-sutton.png` en minúscula, pero esos cuatro ficheros terminan en `.PNG`. En sistemas de ficheros que distinguen mayúsculas (Linux, el propio Dev Container) las imágenes no se muestran; en Windows y macOS sí. `river_swim.png` y `gamma.png` están en minúscula y funcionan siempre.

*Fuente: RLM notebooks/static/*; celdas indicadas.*

---

## 13. Qué aporta este material a la práctica

**Idea.** El agente y el bucle del LAB son los de `rl-material`, así que estos notebooks son la mejor documentación del código de la práctica. Y sus defectos son los mismos que hay que evitar en la práctica. El propio profesor, al recorrer `execution_lab`, dijo que se entrena con un número finito de episodios, "con la definición clásica que decía que no me gustaba" (10.000 o 1.000, "da igual"), y que es "el bucle típico", con algunas listas más para ir registrando (T07 ~04:49:20-04:49:45).

| Pieza de la práctica (BE dqn.py / LAB) | Dónde está en `rl-material` |
|---|---|
| `DDQNAgent`: red de dos capas ocultas densas ReLU, MSE + Adam, con red principal y red objetivo sincronizada con `update_target()` | `deep-double`, `deep-rainbow`, `profit-seeking` (celda 4, casi idéntico: BE quita la capa duplicada y añade `set_trainable` y la acción forzada en el paso 0) |
| `sample()` con CER | `deep-rainbow`, `profit-seeking` |
| `learn()` Double DQN con `targets = qs` y contador `k` para los no terminales | `deep-double` |
| `experience()` con `s1 = None` en terminal; `deque` como buffer | Todos los deep; justificación en `insights` §9.3 |
| Acción aleatoria del entorno (`env.action_sample()`) en vez de `action_space.sample()` | `profit-seeking` (ponderada 0,8/0,1/0,1) |
| `minus_factor` / `div_factor` para normalizar el estado | Atributos de `profit-seeking`; `encode` sin usar en `deep-rainbow` |
| Prellenado del buffer, `learn_after = batch_size`, `update_target_after = 10 * learn_after`, ε lineal por episodio con suelo, media móvil de recompensas | Celda de entrenamiento de todos los deep y de `profit-seeking`; prellenado frente a lo dicho en clase en §8.1 |
| API antigua de gym (4 valores) | Todos |

Qué llevarse (todo ello son **Notas/propuestas** apoyadas en lo dicho en clase):
1. **Leer primero `TD-q-learning` y `deep`**, como pide el profesor (APP p.1), y después `deep-double` y `profit-seeking`, que son el agente y el bucle de la práctica.
2. **Cambiar el ε decay por episodios** por uno guiado por evaluación greedy periódica, con tope de pasos (§4.2, §6; T06 ~03:04:52-03:05:16; T07 ~00:20:39-00:22:42, ~04:49:20). Ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).
3. **Evaluar en el conjunto de test de verdad** (errata de `profit-seeking`) y sin exploración.
4. **Inicializar `mse`** antes del bucle y registrar pérdidas medias, no la del último batch (§8.4).
5. **Normalizar el estado** (los lineales no lo hacen; `profit-seeking` estandariza con rankings) y no meter precios en bruto (T06 ~02:19:28-02:19:38).
6. **Repetir con varias semillas**: incluso con las mismas recompensas los entrenamientos varían mucho (T06 ~02:54:53-02:55:11; T06 ~01:13:17-01:13:31). Ningún notebook fija semilla.
7. **Probar variantes baratas** sobre el `DDQNAgent`: objetivo esperado (`deep-expected-sarsa`), cabeza de duelo (la de `deep-rainbow`, no la de `deep-dueling`) o Huber. El profesor anima a combinar mejoras (T07 ~00:25:05-00:25:55).
8. **Entorno de instalación**: Python 3.11 y los pines de `best-execution` (pandas 2.0.3); §2.3.

Propuesta completa, ejercicio a ejercicio, en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md); el entorno de la práctica, en [08_entorno_best_execution.md](08_entorno_best_execution.md); qué pide cada ejercicio, en [notebook_partida.md](notebook_partida.md). Los términos técnicos, en [11_glosario.md](11_glosario.md).

---

## 14. Erratas, discrepancias y dudas

| # | Dónde | Qué ocurre | Corrección o lectura razonada |
|---|---|---|---|
| 1 | TD-q-learning c.7, TD-sarsa c.3, TD-expected-sarsa c.3, TD-double-q-learning c.5 | `s1, *_, done = env.step(a)` asigna `info` a `done` | `s1, _, done, _ = env.step(a)`; probablemente los percentiles de la rejilla se calculan con estados posteriores a la caída *(no verificado ejecutando)* (§5.5) |
| 2 | Tabulares de CartPole | El fin por límite de 500 pasos se trata como terminal (`target = r`) | Distinguir truncado; ver [02_diseno_de_entornos.md §2.4](02_diseno_de_entornos.md) |
| 3 | deep-dueling c.2 | La salida estándar sobrescribe la de duelo: entrena un DQN normal | Comentar `out = Dense(...)(capa)` (§8.4) |
| 4 | deep-dueling y deep-rainbow c.2 | $V-(A-\bar A)$ en lugar de $V+(A-\bar A)$ | Equivalente en capacidad; mejor escribir `tf.add` por claridad |
| 5 | Todos los deep, c.3 | `mse` sin definir si el primer episodio acaba antes del primer `learn()` | `mse = np.nan` al principio |
| 6 | Todos los deep, c.3 | `learn()` extra en los episodios de log; pérdida registrada = último batch | Separar monitorización de entrenamiento |
| 7 | deep-fixed c.2 | `target_model.predict` (lento, y con barra de progreso, fila 24) y comentario "acts" | `self.target_model(s).numpy()` |
| 8 | deep-rainbow c.2 | `encode` usa atributos no definidos | Código muerto; borrar o definir `minus_factor`/`div_factor` |
| 9 | Lineales c.5 y deep c.5 | "Done Rew" imprime solo la última recompensa descontada | Imprimir la suma |
| 10 | Lineales c.2 | Sin término independiente ni normalización | Añadir un 1 al estado y estandarizar (§7) |
| 11 | APP p.35 frente a RLM | Gráficos de CartPole con hasta 20.000 episodios; los notebooks lineales usan LunarLander con 2.000 | Presumiblemente, gráficos de una versión anterior (§7) |
| 12 | insights c.13 y c.16 | "1.000.000" de pasos frente a `10e6` = 10⁷ | Mismo resultado (≈ 75,25); `int(1e6)` |
| 13 | insights c.11-12 | La figura empieza con $r=1$ y el código con $r=2$ | Equivale a empezar en $S_1$ (§9.2) |
| 14 | TD-q-learning c.11 | No reasigna `s` tras `reset` | `s = env.reset()` |
| 15 | Notebooks con imágenes | `.png` enlazado frente a `.PNG` en disco | Renombrar los ficheros (§12) |
| 16 | `plots.py` | Pérdida en color `lavender`, casi invisible | Cambiar el color |
| 17 | `profit-seeking` | Test evaluado con datos de train, `position == 0`, capa colgada, `getQ` roto, cronología de cobro... | Ver [05_profit_seeking.md §10.7](05_profit_seeking.md) |
| 18 | Repositorio | 17 notebooks, no 18; guardados con Python 3.8 y sin salidas | — |
| 19 | T06 ~02:18:40 frente a TD-q-learning c.7 | En clase "5 o 10 rangos"; el código usa 50 celdas en el rango p10-p90 | Ambas cosas son opciones; el nivel de discretización es un hiperparámetro (§5.1) |
| 20 | T07 ~00:22:07-00:22:24 | No queda claro cómo mostró que el acantilado se aprende hacia el episodio 30 | Dos lecturas: acortó la exploración o evaluó sin ella (§6) |
| 21 | T06 ~01:15:25-01:15:41 | "50 % del máximo más 50 % la media del resto" | El código usa la media de **todas** las acciones, que es la expectativa correcta con ε-greedy (§5.3) |
| 22 | T06 ~03:49:30-03:49:38 | Cifras confusas de 0,95ⁿ *(dudoso)* | Probablemente "7 % tras 50, 0,5 % tras 100 y prácticamente 0 tras 200", que cuadra con 7,7 %, 0,6 % y 0,004 % (§9.2) |
| 23 | cliff-walking c.11, c.18, c.23 | Recorrido greedy con `itertools.count()` sin tope: si la política cicla o choca contra un borde, no termina *(no verificado ejecutando)* | `for step in range(100)` (§6) |
| 24 | deep-fixed c.2 | `predict` con `verbose='auto'` imprime una barra por cada `learn()` en TF 2.15 *(no verificado ejecutando)* | `predict(s, verbose=0)` o `self.target_model(s).numpy()` (§8.4) |
| 25 | deep-dueling c.2 (V2 comentada) | `K` sin importar y `K.mean` sin `axis=1`: centraría con la media del batch | `K.mean(x[:,1:], axis=1, keepdims=True)` e importar `K` (§8.2) |
| 26 | insights c.3 y c.7 frente a `river_swim.png` | La política fuerza derecha en el 1 e izquierda en el último: dos transiciones de la figura no se simulan | Lectura: reduce las visitas al último estado (§9.1) |
| 27 | linear-sarsa c.2 | La clase se sigue llamando `LinearApproximation` | Cosmético (§7) |

*Fuente: las indicadas en cada fila.*
