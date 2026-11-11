# Diseño y programación de entornos de RL

> **Fuentes:**
> - **APP** (`docs/raw/MIAX14/applying_RL_MIAX.pdf`, "Applying Reinforcement Learning", 103 p.): p.48 (mapa de la segunda parte), p.49-53 (estados), p.54-55 (step), p.56-57 (acciones), p.83-86 (definición del entorno), p.87-92 (entrenamiento: p.89 y p.92 eficiencia, p.90 sobreajuste, p.91 apoyo en otros sistemas), p.93-97 (retos y entornos con IA generativa).
> - **CASOS** (`docs/raw/MIAX14/casos_practicos.pdf`, "Casos Prácticos", 63 p.): p.10-13 (profit-seeking: steps, episodios, estados, acciones), p.29-41 (execution algorithms: objetivo, componentes, generación de órdenes, steps, estado, acciones).
> - **T06** (`docs/raw/26 11 06.pdf`, transcripción automática de la clase del 06/11/2026, profesor Jesús Sanz): ~00:22-00:38 (Markov, variable omitida, impacto y estado privado), ~01:00-01:06 (material y gym antiguo), ~02:11-02:29 (estado, step, acciones), ~03:30-03:35 (curriculum learning), ~04:02-04:14 (episodios, dimensionado, eficiencia), ~04:22-04:31 (diversidad de episodios), ~04:37-04:49 (Gymnasium, done, reset, curriculum de datos).
> - **T07** (`docs/raw/26 11 07.pdf`, clase del 07/11/2026): ~00:10-00:18 (eficiencia, demo `%timeit`, libro de órdenes), ~00:29-00:34 (retos del entorno, producción), ~00:39 (acciones aleatorias ponderadas), ~02:54-03:30 (episodios, steps, estado y acciones del caso de ejecución), ~04:38-04:52 (datos, asunciones, `reset` de la práctica), ~05:00-05:05 (atributos del entorno y precio entre fotos).
> - **LAB** (`docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`): celdas 2 (datos), 4-5 y 20 (carga de un activo y evaluación en test), 7 (`BestExecutionEnv`).
> - **BE** (`docs/raw/MIAX14/best-execution/src/agents/dqn.py`): `experience`, `sample`, `act`, `learn`.
> - **RLM** (`docs/raw/MIAX14/rl-material/notebooks/rl-models-basic/TD-q-learning.ipynb` y `TD-expected-sarsa.ipynb`): uso de `gym` antiguo y discretización de CartPole; `docs/raw/MIAX14/rl-material/setup.py` (versión `gym==0.23.1`).
> - **ENUNCIADO** (`docs/raw/MIAX14/ejercicios_taller_rl.docx`): requisitos del ej. 2 que afectan al diseño del entorno.
> - **T2** (`docs/raw/referencias/deep-rl-trading-agents/docs/teoria/`): `00_fundamentos_mdp.md`, `03_dqn.md`, `07_panorama_drl.md`, `08_trading_en_clase.md`.
>
> **Convenciones:** "(T06 ~02:15)" = hora aproximada en la transcripción; todo lo que viene de ellas es paráfrasis de un texto automático muy ruidoso, y lo que no se entiende bien va marcado *(dudoso)* con las lecturas posibles. "(APP p.84)" = página del PDF. "(LAB celda 7)" = celda del notebook contando desde 0. **Nota:** = contenido propio (cálculos, correcciones, conexiones, propuestas), no del profesor ni del material. Los alumnos no se nombran ("un alumno"). Este documento cubre **cómo se construye un entorno**; la recompensa y el descuento están en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md), la exploración, el buffer y el entrenamiento en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md), y el código concreto de `BestExecutionEnv` en [08_entorno_best_execution.md](08_entorno_best_execution.md).
>
> Escrito el 10/11/2026.

---

## 1. Qué es el entorno y por qué es la pieza crítica

**Idea.** El entorno (simulador) es todo lo que no es el agente: genera las partidas, le dice al agente en qué estado está, recibe su acción, la evalúa, le devuelve la recompensa y el estado siguiente, y decide si la partida ha terminado. En la práctica es justo "lo que tenéis que programar" (T07 ~02:55:37-02:56:08).

**El diagrama de clase** (APP p.83, repetido en miniatura en p.84 y adaptado en CASOS p.30, p.33 y p.36). Un recuadro discontinuo con dos cajas azules, *Agent* y *Environment*: del entorno sale *State (s)* hacia el agente, del agente sale *Action (a)* hacia el entorno, y la flecha *Reward (r)* va del entorno al agente con un **recuadro blanco vacío** encima: la recompensa es lo que hay que definir. Debajo, la secuencia $s_t\,a_t\,r_t \to s_{t+1}\,a_{t+1}\,r_{t+1} \to \dots \to s_{t+n}\,a_{t+n}\,r_{t+n}$ y una caja "→ Step / *End of episode (Execution Time)*". En CASOS el entorno lleva la lista de funciones: genera órdenes (episodios), envía el estado actual $s_t$, recibe la acción $a_t$, la evalúa y envía $r_t$, envía $s_{t+1}$ (CASOS p.30); en p.33 aparece el estado como "vector de características" en $S_3$ (10:30 am) y en p.36 un ejemplo de ese vector (§5.7).

**Por qué es lo más difícil.** Entre los cuatro grandes retos del RL (exploración, recompensas, inestabilidad y creación del entorno) el entorno "suele suponer uno de los principales problemas" (APP p.93-95; T07 ~00:29:03-00:30:20):
- Casi siempre será **simulado**. Podría ser real (millones de personas jugando al póker, o el ajuste de los LLM con personas que dicen si una respuesta es buena o mala), pero lo normal es simular (T07 ~00:29:21-00:29:44).
- Tiene que ser **realista**, lo cual es costoso y laborioso, y tiene que **corresponderse con producción**: si entrenas robots, agentes de ejecución o de inversión en un simulador, ¿cómo sabes que luego no harán "cosas raras" en la realidad? (T07 ~00:29:44-00:30:20).
- Necesita a la vez **interacciones realistas** y **velocidad de generación de experiencias** (APP p.94-95). La p.95 lo ilustra con un libro de órdenes dibujado en tres instantes ($t=0,1,2$): niveles *ask* en rojo (en torno a 8,60-8,75 €) y *bid* en verde (8,30-8,45 €), con volúmenes por posición, el *spread* entre ambos lados y dos ejecuciones (100 títulos en el lado ask, 20 en el bid). La p.95 termina con un recuadro resaltado, «2026: la IA generativa ya ayuda a crear entornos →», que da paso a p.97 (§11); de ese recuadro dijo que es una vía de esperanza, pero muy, muy verde (T07 ~00:30:25-00:30:31).
- Un principio que el profesor repitió: los entornos y entrenamientos tienen que ser **estables, sin sesgos, sencillos y muy eficientes** (T07 ~00:10:37).

**El entorno es el que modeliza las consecuencias.** Con supervisado se etiquetan sobre un histórico estático los buenos momentos de compra y venta; pero si operas con capital relevante, al comprar mueves el precio (no habrías visto ese precio) e impactas lo que viene después, y las etiquetas quedan inservibles. En RL esas consecuencias se modelizan en el entorno; si el entorno las modeliza perfectamente, "lo tienes todo hecho" (T06 ~00:34:49-00:35:45). Por eso su regla práctica: si el problema se puede resolver con supervisado, casi mejor hacerlo con supervisado, porque en RL hay que construir un entorno y eso es "más problemático" (T06 ~00:29:14-00:29:29). Ver [00_repaso_rl.md](00_repaso_rl.md) para cuándo usar RL frente a supervisado.

**Orden de diseño.** Primero el **objetivo**, y de él salen los episodios, los steps, las características y las recompensas: el objetivo determina las recompensas, los steps, las características, todo (T06 ~04:06:49-04:06:55). En el caso de ejecución se siguió ese orden: objetivo (orden de compra, mejorar el *slippage* frente al VWAP y opcionalmente la dispersión), episodios, steps, estado, acciones y recompensas (T07 ~02:54:47-02:55:44, ~03:08:25-03:08:35; CASOS p.29-41). El mapa de la segunda parte de APP agrupa las piezas igual: qué ve y qué hace el agente (estados, acciones, recompensas), cómo aprende (ε, α, γ, *experience replay*) y cómo se pone en marcha (definición del entorno, entrenamiento, retos) (APP p.48).

---

## 2. La API Gym / Gymnasium

**Idea.** La mejor referencia para construir un entorno propio es Gym/Gymnasium: su documentación enlaza el código de cada entorno (cómo se han definido las recompensas, la arquitectura) y son entornos poco complicados, "el mejor punto de partida" (T06 ~04:37:50-04:38:23). "Luego, vuestro ENV puede contener lo que queráis" (APP p.84).

### 2.1 La interfaz (APP p.84)

La diapositiva muestra el logo de Gym → Gymnasium ("Farama, sucesor oficial desde 2022") y este código, con `info`, `terminated` y `truncated` en azul y negrita:

```python
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

- **`terminated`** → fin natural del episodio (*done* previsto).
- **`truncated`** → límite de steps o de tiempo (*done* forzado).
- Enlaces: `github.com/Farama-Foundation/Gymnasium` (`classic_control/cartpole.py`) y el equivalente antiguo de `openai/gym`. Recomienda "meterse" en los entornos de Gymnasium y leer su código (APP p.84).

Lo que explicó (T06 ~04:38:29-04:40:11):
- **`reset()`** comienza una partida nueva y devuelve el estado inicial. Lo marcado en azul en la diapositiva son los cambios de `gym` a `gymnasium`; sus propios ejemplos usan la versión antigua (T06 ~04:38:49-04:39:06).
- **`step(action)`**: "oye entorno, evalúa esta acción". Devuelve el **estado siguiente** (un *array* con las características), la **recompensa** de haber tomado esa acción, si **ha terminado** y un **diccionario `info`** para depurar, que en la mayoría de entornos viene vacío. Por convención eran 4 componentes y ahora son 5 (T06 ~04:39:14-04:39:54).
- El antiguo `done` se ha partido en dos: cuándo algo ha terminado "como debía" (final natural, done previsto) y cuándo se ha cortado por una regla (done forzado) (T06 ~04:39:57-04:40:11).

| | `gym` antiguo (material de clase) | `gymnasium` (APP p.84) |
|---|---|---|
| `reset` | `s = env.reset()` | `obs, info = env.reset(seed=...)` |
| `step` | `s1, r, done, info = env.step(a)` (4 valores) | `obs, r, terminated, truncated, info = env.step(a)` (5 valores) |
| Fin | `done` único | `terminated` (natural) / `truncated` (forzado); `done = terminated or truncated` |

### 2.2 El ejemplo de la Roomba (T06 ~04:40:11-04:40:51)

Una Roomba que debe limpiar un cuarto **termina** (`terminated`) cuando acaba de limpiarlo o se queda sin batería (un par de horas). Pero si lleva 5 minutos atascada en una esquina, se **trunca** y se empieza otro episodio: si no, llenaría el buffer de experiencias inútiles de estar atascada, aprendería de eso y se perdería el tiempo.

### 2.3 Qué usan los materiales de este taller

- **rl-material** usa la librería antigua `gym`, y el profesor avisó de que no lo había adaptado: "ya no se utiliza" y lo actual es `gymnasium` (T06 ~01:05:20-01:05:43). En efecto, sus notebooks hacen `s = env.reset()` y `s1, r, done, _ = env.step(a)` (RLM TD-q-learning, celdas 4 y 7). Detalle del material en [09_material_rl_notebooks.md](09_material_rl_notebooks.md).
- **`BestExecutionEnv`** (LAB celda 7) es una clase propia que **no hereda** de `gym.Env`: `reset()` devuelve solo la observación y `step()` devuelve 4 valores `(observation, reward, done, {})`, con `observation = None` cuando `done` es `True` (esa parte de `step` viene escrita). Las dos condiciones de parada, tiempo agotado (`is_bins_complete`) y orden completada (`is_ord_complete`), son un TODO de la función (10), y el TODO de `_compute_done` (función (11)) pide devolver `done = True` si se cumple cualquiera de las dos (LAB celda 7). Detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md).
- El ENUNCIADO permite crear el entorno desde cero, siempre que tenga todos los elementos de un entorno de RL, esté pensado para un algoritmo de ejecución, use los ficheros del taller y se documente más (ENUNCIADO, ej. 1, "No es necesario"). Ver [notebook_partida.md](notebook_partida.md).
- En el Taller 2 el entorno del enunciado (`SimpleTradingEnv`) sí heredaba de `gymnasium.Env` y el de los notebooks de trading no: comparación en [T2 08_trading_en_clase.md §11](../raw/referencias/deep-rl-trading-agents/docs/teoria/08_trading_en_clase.md) y panorama de Gym en [T2 07_panorama_drl.md §11](../raw/referencias/deep-rl-trading-agents/docs/teoria/07_panorama_drl.md).

### 2.4 `terminated` frente a `truncated` en el objetivo TD

**Nota:** la distinción no es cosmética: decide si se hace *bootstrap*. El objetivo de Q-learning/DQN debe ser

$$
y=\begin{cases} r & \text{si } s' \text{ es terminal (terminated)}\\[2pt] r+\gamma \max_{a'} Q(s',a') & \text{en otro caso, incluido truncated}\end{cases}
$$

porque un truncado no es un final real: el futuro existe, simplemente se ha dejado de simular.
- En el código del taller cualquier `done` se guarda con `s1 = None` y su objetivo es solo `r` (BE experience y learn). Para las dos condiciones de `BestExecutionEnv` eso es coherente: cuando se acaba el tiempo o se completa la orden CARE el episodio de ejecución acaba de verdad, y el estado incluye el **tiempo restante**, así que el final por tiempo es predecible desde el estado (no rompe Markov; §5.5). **Nota:** con $\tau=1-k/N$ en % y $N$ variable (400-600), el estado no da los pasos restantes en valor absoluto: con el mismo $\tau$, a un episodio de 400 bins le quedan menos pasos que a uno de 600, y con $\gamma<1$ eso cambia el valor. Es aproximadamente Markov porque `vol_care` escala con $N$; si se quiere exactitud, se puede añadir $N-k$ normalizado (p. ej. $(N-k)/600$) como característica.
- Si se añade una regla de **truncado artificial** (p. ej. cortar un episodio que "se ha atascado", al estilo Roomba), habría que guardar el `s1` real y hacer *bootstrap*, o el agente aprendería que esos estados valen solo $r$.
- El caso contrario aparece en el material: en el CartPole de `gym` antiguo el límite de 500 pasos llega como `done=True` y el código hace `target = r` (RLM TD-q-learning, celdas 6-7), aunque el estado (4 variables físicas) no contiene el tiempo. Es el sesgo que la API de `gymnasium` permite evitar.
- En `gymnasium`, además, `step` devuelve la observación final aunque `terminated` sea `True`; el `None` de `BestExecutionEnv` funciona porque el agente de BE lo trata aparte (`if s1 is None`), pero obliga a no usar la observación tras el final.

**Nota (si se rehace el entorno con `gymnasium`):** heredar de `gymnasium.Env`, declarar `action_space = spaces.Discrete(n)` y `observation_space = spaces.Box(...)`, implementar `reset(self, seed=None, options=None)` → `(obs, info)` (llamando a `super().reset(seed=seed)` para tener `self.np_random` reproducible) y `step` → 5 valores, y validarlo con `gymnasium.utils.env_checker.check_env`. Es la forma que aceptan librerías externas como Stable-Baselines3 ([T2 07_panorama_drl.md §12](../raw/referencias/deep-rl-trading-agents/docs/teoria/07_panorama_drl.md)).

---

## 3. Episodios: duración, final y truncado

### 3.1 Qué decide el final (APP p.85; T06 ~04:41:43-04:43:45)

**Idea.** La pregunta no es tanto cuánto dura el episodio sino **qué hace que `done` sea `True`**: es un booleano y hay que programar su lógica (T06 ~04:41:59-04:42:21). APP p.85 lo plantea así:
- ¿Cuánto tiene que durar un episodio? ¿Finito o continuo? ¿Se añaden condiciones de **objetivo**, de **control** o **temporales**? Es decir, ¿qué va a determinar si `done == True`?
- Final del episodio: **done previsto** o **done forzado**, este último cuando no merece la pena continuar o es imposible recuperarse → hay que **gestionar las recompensas de final de episodio**.

Lo que añadió en clase:
- Las reglas de truncado se añaden si hacen falta y siempre en función del objetivo: piensa cómo te evaluarías tú si fueras el algoritmo (T06 ~04:42:37-04:42:57).
- Con los `done` hay que decidir también si se da una recompensa en ese momento y, en el truncado, si merece la pena continuar esa partida o ya es imposible; incluso se puede **penalizar el tiempo perdido** (los 5 minutos de la Roomba en la esquina) (T06 ~04:43:04-04:43:45). El diseño de esas recompensas finales está en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

### 3.2 Ejemplos de clase

| Problema | Episodio | Step | Fuente |
|---|---|---|---|
| Ajedrez | Hasta jaque mate, tablas o la regla de movimientos sin capturas | Cada decisión | T06 ~04:04:24-04:06:38 |
| Intradía (ejecución) | El día entero (de 9 a 17) o unas 2 h; puramente temporal, de principio a fin | p. ej. cada minuto | T06 ~04:03:45-04:07:10 |
| Intradía con objetivo | Temporal, pero cortable: "si consigo un 2-3 %, me voy a casa" | | T06 ~04:07:16-04:07:37 |
| Opciones a 6 meses revisadas cada 3 meses | Doble criterio: trimestral salvo que ocurra algo, evaluando día a día | Diario | T06 ~04:08:50-04:09:39 |
| S&P 500 desde 1900 | Un único episodio: no hay diversidad (§7.3) | | T06 ~04:27:38-04:28:49 |
| Profit-seeking | X tiempo o X condición (p. ej. al deshacer posiciones) | Diario, minutos... | CASOS p.10 |
| Ejecución (caso del taller) | La orden completa (el TWAP entero) | 5 s en los datos | CASOS p.32; T07 ~03:02:28 |

- En ajedrez, el profesor y la clase reconstruyen la regla de tablas por movimientos sin capturas ni movimientos de peón, que en la transcripción aparece como "40 movimientos" (T06 ~04:04:39-04:05:24). **Nota:** la regla oficial de la FIDE es de **50** movimientos (75 para tablas automáticas); el "40" es un error de clase o de la transcripción. Para el número de steps por partida se oyen cifras contradictorias ("como 500 pasos", "en media son 50", "200 pasos") *(dudoso)*; lo que importa es el uso: steps por episodio × episodios que se quieren recordar = tamaño del buffer (§3.4).
- En bolsa no hay respuesta única: un episodio puede ser un día, un año o cinco años; depende del objetivo (T06 ~04:03:42-04:04:08, ~04:06:44-04:06:55).

### 3.3 El episodio en el caso de ejecución

**Idea.** El entorno también **genera las órdenes**, y cada orden es un episodio (CASOS p.30-31). A la pregunta de qué factores definen un episodio, la clase propuso horizonte temporal (*time frame*), número de títulos, número de particiones (órdenes hijas) y urgencia por completar la orden (T07 ~02:56:41-02:57:10). El profesor lo dio por bueno, lo llamó un "TWAP con esteroides" y lo resumió en lo que el cliente suele pedir a un TWAP: **tiempo** y **número de títulos** (T07 ~02:57:28-02:58:16). **Nota:** en la práctica las particiones quedan fijadas por las hijas de 1 título (nº de hijas = `vol_care`) y la urgencia, por la ratio `vol_care`/`episode_bins`.

- **Generación** (CASOS p.31; T07 ~02:58:30-02:59:10): tamaño en N títulos o, mejor normalizado, en **% del ADV** (volumen medio diario) de los últimos 30 días, con o sin subastas; p. ej. una orden de 1.000.000 de títulos puede ser el 10 % del volumen diario. Los parámetros se **muestrean de las distribuciones históricas** de lo que piden los clientes (en BME se hacía ese estudio, T07 ~02:57:53-02:58:12). El histograma de p.31 muestra esa distribución: eje X "N títulos / % ADV" desde 0, eje Y frecuencia, muy **asimétrica a la derecha** (muchas órdenes pequeñas y medianas y una cola larga de órdenes grandes).
- **Limitaciones** (CASOS p.31; T07 ~02:59:10-02:59:57): no generar órdenes de más del 5-10 % ADV, porque en la cola de órdenes grandes habría que "ponerse muy serio" con funciones de impacto de mercado; con órdenes pequeñas las hipótesis de ejecución se pueden relajar.
- **El "triciclo" de la práctica** (T07 ~03:00:04-03:01:26): orden de ejemplo de 400 títulos, **hijas de 1 título**, sin comisiones ni costes de transacción e **impacto 0**: así el histórico funciona como un simulador "bastante realista". Si no funciona en un escenario tan controlado, no funcionará con órdenes de un millón (T07 ~03:00:43-03:00:53). Sobre el coste 0: no es tan irreal, porque hay brokers con promociones o tarifas planas en las que los primeros títulos salen gratis; con eso y un histórico de precios se tiene un simulador "bastante realista" (T07 ~03:00:59-03:01:26) *(dudoso: transcripción muy ruidosa)*.
- **Variabilidad**: que el agente no se enfrente siempre a la misma situación (empresas, tamaños...), aunque para depurar conviene empezar con un mismo día y un solo activo, lo cual "no sirve para producción" (T07 ~03:01:44-03:02:21). Ver §9.
- **Duración**: en clase propuso episodios de **90 min a 2 h** en lugar del día entero, que es lo que dura un TWAP normal (T07 ~04:44:39-04:44:55), pero el notebook pide episodios de **400-600 bins de 5 s** (docstring y TODO de `_generate_episode_params`; implementarlo es parte de la práctica) y el propio profesor dijo que eso es "menos de una hora" y que se puede cambiar a 2-3 h (T07 ~04:46:47-04:47:02; LAB celda 7). **Nota:** 400-600 bins × 5 s = 33,3-50 min.
- **Tamaño de la orden en la práctica**: con hijas de 1 título, el tamaño de la CARE fija la flexibilidad del agente. En 1 h hay 720 bins de 5 s; una CARE de 720 títulos obligaría a comprar en todos los bins (ninguna flexibilidad) y una de 360, en uno de cada dos (T07 ~04:45:14-04:46:39). Con flexibilidad de "uno de cada diez" salía una "orden hija de 65 títulos" multiplicando 720 por "0,9" (T07 ~04:47:15-04:47:33) *(dudoso: $720\times0{,}9=648$; la cifra 65 cuadra con $720\times0{,}09\approx65$, es decir, con ejecutar en torno al 9-10 % de los bins; y lo que dice "orden hija" debe leerse como orden CARE de ~65 títulos, es decir, 65 hijas de 1 título: "65 barritas" en el gráfico final, T07 ~04:47:36)*. El docstring de `_generate_episode_params` (función (1)) pide `episode_bins` = entero uniforme en $[400, 600]$ y `vol_care` = `int(pct_bins × episode_bins)` con `pct_bins` uniforme en $[0{,}075;\,0{,}125]$; las aserciones exigen $400\le$ `episode_bins` $\le600$, `int(0,075·bins)` $\le$ `vol_care` $\le$ `int(0,125·bins)` y que `vol_care` sea `int`. Implementarlo es un TODO (LAB celda 7). El ENUNCIADO pone el ejemplo de 400 steps y 40 títulos, "1 de cada 10 steps" (ENUNCIADO, ej. 2). **Nota:** con 400-600 bins, `vol_care` queda entre 30 y 75 títulos, coherente con los "unos 50-60 títulos" que mencionó al repasar los atributos (T07 ~05:00).

**Episodio completo frente a episodios por momento de actuación** (CASOS p.32). El enfoque elegido es el "realista": duración del episodio = orden completa (+ específico, − flexibilidad: se parece más a un TWAP). La alternativa: un TWAP que actúa cada X tiempo con reglas, con momentos de actuación predefinidos, donde **cada momento de actuación es un episodio** y el RL aprende la mejor acción en él.

**Acotar el problema** (T07 ~03:02:42-03:08:18). Para el profesor fue "casi lo más importante de la clase": un TWAP decide tres dimensiones (momento, tamaño y tipo de orden), "son muchas cosas para un algoritmo". Se puede dejar que el RL decida solo una:
- **Solo el momento**: el TWAP manda una hija cada 10 min y el agente decide **dónde** dentro de cada ventana de 10 min; el resultado se parecerá mucho al TWAP porque se le deja poca libertad, pero el problema queda muy acotado. En clase lo formuló con **episodios de 10 min**: por defecto el TWAP clásico ejecuta una vez cada 10 min y el agente solo decide cuándo dentro de esa ventana; en la pizarra, las "barritas azules" del TWAP frente a ejecuciones algo desplazadas (T07 ~03:03:49-03:05:25). Es la alternativa de CASOS p.32 (arriba), con cada momento de actuación como episodio.
- **Solo el tipo de orden**, con momentos y tamaño dados por el TWAP.
- **Solo el tamaño**: mismo tipo y momento, pero tres opciones: la mitad, el mismo tamaño o "el doble... un 50 % más" *(dudoso: dice ambas cosas)*.
- Con un problema así de acotado basta un estado muy pequeño: si la única decisión es el tipo de orden, la información puede ser tan sencilla como la probabilidad de que se ejecute una orden *(dudoso: la transcripción dice "una ocasión"; por el contexto, una orden pasiva)*, y el agente aprenderá por sí mismo un umbral ("a partir del 60 % es cuando te interesa") tras promediar muchos episodios (T07 ~03:07:41-03:08:09). Estos subproblemas "bien aterrizados" tienen solución (T07 ~03:07:30-03:07:37).
- Se pueden combinar varios subsistemas, cada uno con un tema específico; la práctica simplifica a **decidir el momento a lo largo de todo el episodio**, con hijas de 1 título y siempre agresivas (T07 ~03:06:19-03:06:48). El desarrollo completo del caso está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

### 3.4 Del episodio al tamaño del buffer

Surgió de la pregunta de un alumno con unos 14 años de datos (≈3.000 días): antes de dimensionar nada hay que decidir cuánto dura un episodio, porque un día (de 9 a 17), un año o cinco años dan órdenes de magnitud distintas de episodios (T06 ~04:03:26-04:04:08). **Nota:** con episodio = día salen ~3.000 episodios distintos; en la práctica, con 60 días de train y más de 2.500 inicios posibles por día (§10), la diversidad de episodios no es el cuello de botella.

El profesor prefiere pensar el buffer en **episodios** ("mis 100 últimas partidas") más que en steps (T06 ~04:02:50-04:03:01). Por eso hay que estimar los steps por episodio: con decisiones cada minuto y usando de media el **75 %** del día (por las condiciones de parada), unas 8 h × 60 min dan los steps por episodio, y eso por 100 o 1.000 episodios da el tamaño del buffer (T06 ~04:07:41-04:08:35). La aritmética de la transcripción está mezclada ("nueve horas de ejecución son 8 h", "8 h por 65 por 60") *(dudoso)*.

**Nota (cálculo propio):**
- Con el mercado continuo español (9:00-17:30, 8,5 h) y decisión por minuto: $0{,}75\times 8{,}5\times 60\approx 383$ steps por episodio; con las "8 h" de clase, $0{,}75\times480=360$. 100 episodios ≈ 36.000-38.000 transiciones; 1.000 ≈ 360.000-383.000.
- En la práctica (step = bin de 5 s, 400-600 bins): 100 episodios ≈ 40.000-60.000 transiciones. Cómo elegir esa cifra (independencia frente a obsolescencia, CER) está en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md); el γ coherente con 400-600 pasos, en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

---

## 4. El step: qué es y cómo elegir su resolución

### 4.1 Definición y criterios (APP p.54; T06 ~02:19:54-02:24:45)

**Idea.** Salvo en los juegos por turnos, todos los problemas son continuos en el tiempo, así que hay que **definir** qué es un step (APP p.54). La diapositiva lleva un cronómetro: el ajedrez tiene un ✓ ("está claro") y CartPole, Breakout y el algoritmo de ejecución llevan una llave con "?". Pregunta: en bolsa, ¿tick a tick, segundos, minutos?; en juegos, ¿120 fps, 60, 24?

| Criterio | Ejemplos | Fuente |
|---|---|---|
| **Temporal** | ms, s, fps; cada hora, cada día | APP p.54; T06 ~02:22:53-02:22:59 |
| **Determinista / por eventos** | tick a tick; cada X volumen negociado (títulos o dólares); cada vez que se ejecuta algo en el mercado o cada vez que yo ejecuto | APP p.54; T06 ~02:22:59-02:23:19 |
| **Mixto** | "5 s o hasta que se me ejecute la orden"; "si ocurre esto, o yo hago esto, o pasan 5 s" | APP p.54; T06 ~02:23:27-02:23:35 |

- Un alumno lo definió como **el periodo en que el modelo debería recibir información nueva** para decidir (en su práctica anterior, cada pocos días *(dudoso: "cada 6 días" o "6 h")*; en Breakout, cada pocos píxeles o fotogramas) (T06 ~02:20:39-02:21:52). El profesor lo validó.
- En los entornos de Gym el step ya viene hecho; en un problema propio hay que elegirlo y darle un criterio (T06 ~02:22:02-02:22:21). En Breakout: ¿hacen falta 30, 60 o 120 imágenes por segundo, o basta con 24 para llegar a la solución perfecta? (T06 ~02:22:21-02:22:46).
- No hace falta ir tick a tick o cada milisegundo si inviertes a largo plazo (T06 ~02:23:57-02:24:03). Depende del objetivo: qué información necesita el estado y **cada cuánto necesitas refrescarla**, con la posibilidad de atender eventos: por lo general decides cada día, pero si durante el día pasa algo con ciertas características también deberías reaccionar, igual que una persona a la que avisan (T06 ~02:24:14-02:24:48).

### 4.2 Realismo frente a coste computacional (APP p.55; T06 ~02:25:04-02:25:47)

La diapositiva p.55 dibuja una **balanza** con "Realidad" (naranja claro) en un platillo y "Coste C." (coste computacional, naranja oscuro) en el otro: hay que equilibrar ambos. Texto:
- El coste de ambos cambia según el problema, igual que su ponderación.
- En **entrenamiento final o producción**, lo ideal es máxima capacidad de procesamiento y máxima disponibilidad de datos (p. ej. 60 fps en Atari, o X ms en mercado según el proveedor).
- En **research** puede ser buena idea **aproximar el problema aumentando el step**.
- "Entre step y step no quiere decir que haya interacción entre tú y el mercado" (p. ej. una orden pegada sigue viva).

En clase: muchos momentos de actuación encarecen el cómputo, así que para probar recompensas o algoritmos conviene usar **menos resolución temporal** ("más espacio en el tiempo") y, según se consolide la solución, añadir coste computacional con la esperanza de que funcione mejor (T06 ~02:25:11-02:25:47).

### 4.3 El step condiciona todo lo demás

- Hay que definir los steps **antes** que las características, porque condicionan cómo se calculan (T07 ~03:08:31-03:08:41).
- En profit-seeking: steps diarios, por minutos..., que afectan al tiempo de la orden, al impacto de mercado y a la definición de las características (CASOS p.10). Con steps diarios hay pocos datos y menos impacto, y CASOS p.13 pregunta si no convendría un supervisado; con mayor resolución hacen falta asunciones de ejecución, modelos matemáticos, umbrales de viabilidad y calibraciones (§10). Ver [05_profit_seeking.md](05_profit_seeking.md).
- En ejecución: tick a tick, ms, s...; afectan a las asunciones, a la simulación del impacto, al diseño de características y a las acciones; y no es lo mismo el research inicial que producción o la comprobación de resultados (CASOS p.34).

### 4.4 Resolución de decisión frente a resolución del simulador (CASOS p.34; T07 ~03:08:47-03:11:53)

**Idea.** El step de decisión no tiene por qué coincidir con la resolución con la que el simulador reconstruye el mercado. "Diseñar un algo (research) cada X s no significa coger fotos del libro cada X s"; se puede guardar información adicional: precios, volúmenes, ejecuciones (CASOS p.34).

- Opciones de step en ejecución: temporal, por cambio en el libro, cada 5 s, o cada vez que se reevalúa una orden pasiva pendiente (lanzo una pasiva, espero 5 s; si no se ha ejecutado, reevalúo; si se ha ejecutado, decido si lanzo otra o espero) (T07 ~03:08:52-03:09:33).
- Decidir **cada minuto** ya es "una resolución bastante buena": un broker con un millón de títulos hace su trabajo "genial" sin darle al botón cada 5 s (T07 ~03:10:17-03:10:38).
- El simulador, en cambio, debe reconstruir lo que pasa **entre** los momentos de decisión. El dibujo de CASOS p.34: una curva de precio que vale **7,12 €** en $S_t$, sube a un máximo de **7,16 €**, cae a un mínimo de **7,03 €** y acaba en **7,10 €** en $S_{t+1}$. Comparando solo las dos "fotos" (7,12 → 7,10) se concluiría que una compra limitada por debajo de 7,10 no se ejecutó, cuando el precio pasó por ese nivel con volumen suficiente y sí se habría ejecutado (T07 ~03:11:15-03:11:53; el precio exacto de la orden del ejemplo es *dudoso*, "7 con 0,7", probablemente 7,07 €). La orden del ejemplo es una **limitada pasiva por debajo del precio**, lanzada porque se espera que alguien agreda a ese nivel (T07 ~03:12:07-03:12:34).
- En la práctica, los datos ya vienen en fotos de 5 s y el step del entorno es el bin: 12 steps por minuto y **720 por hora** (T07 ~04:45:14-04:45:52; LAB celda 2). El precio al que se ejecuta dentro de un bin es una asunción del simulador (§10).

**Nota:** con datos de 5 s caben dos diseños, a valorar en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md):
- **step = bin** (el del notebook): 400-600 decisiones por episodio, horizonte largo para γ, transiciones muy correlacionadas;
- **step = k bins** (p. ej. $k=12$, un minuto) con el simulador recorriendo los $k$ bins por dentro: 33-50 decisiones por episodio, γ y buffer más manejables, y el precio de ejecución puede salir de lo ocurrido en esos $k$ bins. Cambia el significado de las acciones (p. ej. "comprar $n$ títulos en este minuto") y obliga a revisar `vol_care`.

---

## 5. El estado: diseño del vector de características

### 5.1 Qué es un estado (APP p.49 y p.53; T06 ~02:11:12-02:11:53)

**Idea.** El estado es un **vector de características**, "igual que en supervisado". Las características pueden ser **continuas o discretas** y **privadas** (dependen de mí) **o externas** (públicas, del mercado); en cuanto hay alguna continua, el espacio de estados es infinito (T06 ~02:11:19-02:11:53).

La diapositiva (APP p.49, repetida en p.53 con una columna más): "¿Qué entendemos por estado (S)?" → "Vector de características". Ejemplo: **día 3 ($S_3$) con BP**, "¿vendemos o mantenemos?", y una tabla con Retorno **7 %**, Std **16 %**, Beta **1,2**, Posición **1 (largo)**. Las características se clasifican en continuas/discretas y privadas/externas (la posición es la privada), y los estados resultantes son continuos o discretos; "también podemos usar imágenes". Ilustración de unos árboles (decorativa).

**Los estados deben** (APP p.53):
1. Cumplir la **propiedad de Markov** (§5.5).
2. Contener la **información necesaria** ("meternos en el papel", §5.2).
3. Estar **estandarizados/normalizados**: "los estados deben ser estacionarios" (§5.7).
4. Plantear **reducir la dimensionalidad** si hace falta (§5.9).
5. Ser **discretos en aprendizaje tabular** (§5.8).

### 5.2 Qué características pasar: ponerse en el papel del agente

- APP p.50: "¿qué características le pasamos al algoritmo?" = ¿cómo construimos los estados? Coger el paraguas → información del tiempo; ajedrez → jugadas actuales y pasadas; bolsa → la información necesaria para invertir. Ventaja de las máquinas: más capacidad de procesamiento → más características.
- En TFMs que supervisó y corrigió (también como tribunal) vio muchas veces características metidas sin justificación: "¿por qué metes esto?" — "para meterle más cosas" (T06 ~02:12:11-02:12:45). La pregunta es: si fueras el operador de bolsa o el jugador, **¿qué necesitarías tú?** El agente es como una "externalización de ti mismo", una persona ficticia que quieres que se comporte como tú o mejor (T06 ~02:12:50-02:13:39).
- En ejecución: "sentaos con el broker de turno" y preguntadle qué mira al ejecutar, qué le condiciona (T07 ~03:15:20-03:15:36). Y para cada característica hay que poder decir **por qué es relevante** (T07 ~03:19:10-03:19:29, ~03:21:20-03:21:28).

### 5.3 CartPole: qué pasa si falta una variable (APP p.51-52; T06 ~02:13:52-02:17:28)

APP p.51 muestra solo el carro (rectángulo negro), el palo inclinado y la marca 0 del eje, con la pregunta "¿qué características le pasamos al algoritmo para resolver el juego?". La p.52 repite el dibujo y añade los rótulos **Angle** (respecto a la vertical), **Angular velocity**, **Velocity** (flecha horizontal del carro) y **Position** (distancia al 0 del eje), junto con la pregunta "¿y si quisiéramos hacer un Q-learning? ¿Es posible?".

- La clase propuso la distancia del extremo del palo respecto a la base o al punto de inicio, la gravedad y la velocidad; faltaba algo (T06 ~02:14:00-02:15:07). Sobre la gravedad el comentario es confuso *(dudoso: parece decir que, al ser un problema sencillo y constante, no hace falta pasarla)*. Las 4 que devuelve el entorno son **posición y velocidad del carro, ángulo y velocidad angular del palo** (T06 ~02:15:11).
- Si se quita una (cogiendo el estado que devuelve `step` y eliminando una componente) cuesta mucho converger a la solución perfecta de **500 pasos** (T06 ~02:15:11-02:15:23, ~02:16:28-02:16:49).
- Alguien en clase *(dudoso: parece un alumno)* apuntó que no tienen por qué ser esas 4 exactas (p. ej. la longitud del palo) si se pueden encontrar relaciones con otras variables; el mensaje del profesor es que, si no se pueden reconstruir con el resto, en un problema tan sencillo **la falta de información impide converger** (T06 ~02:15:40-02:16:49).
- Darle mucha más información tampoco es gratis: puede complicar el problema por la dimensionalidad (hay técnicas para que el modelo aprenda en qué fijarse). "Este primer paso de pensar bien en el problema es muy, muy importante" (T06 ~02:17:05-02:17:19).

### 5.4 Si falta la variable clave, nadie aprende el patrón (T06 ~00:30:06-00:33:55)

- Alguien en clase *(dudoso: parece un alumno; la transcripción no identifica al hablante)* contó que en una práctica anterior *(dudoso: parece un taller previo de comprar/mantener/vender)* con datos aleatorios el agente aprendía a no hacer nada (no hay nada que aprender), y con datos reales funcionaba peor que con datos generados. El profesor: los precios reales no siguen patrones fijos; se mueven porque entran actores (una institución que vende "a saco") (T06 ~00:30:14-00:31:28).
- Lógicamente, si el simulador responde a un patrón, un RL bien diseñado lo aprende (T06 ~00:31:31-00:31:43); por eso rendía mejor con datos generados que con reales. **Nota:** un buen resultado en un entorno sintético mide en parte el patrón del generador; enlaza con la cautela sobre mercados sintéticos de §11 (T07 ~00:33:51).
- Si hay patrón y el vector de estado contiene las variables que lo determinan, un RL bien diseñado lo aprende. Si falta algo clave, **no lo aprende nunca, ni RL ni supervisado ni nada** (T06 ~00:31:31-00:32:19).
- Ejemplo: "baja los lunes y sube el resto de la semana". Puedes meter un millón de características: si no pasas el día de la semana, no lo aprenderá; y la solución no es hacer la red más grande o más pequeña, sino revisar qué información le pasas (T06 ~00:32:25-00:32:51).
- Variables correlacionadas con la omitida (la racha de días positivos/negativos, si ayer subió o bajó) permiten "rascar" algo: son *confounders* o proxies de la variable que falta (T06 ~00:33:21-00:33:55). **Nota:** en sentido estricto son *proxies* (variables correlacionadas con la omitida); *confounder* designa una causa común de dos variables, y el profesor lo usa en sentido amplio.

### 5.5 Propiedad de Markov y el pasado dentro del estado (T06 ~00:22:58-00:26:51)

**Idea.** Markov dice que la probabilidad de pasar al siguiente estado depende **solo del estado actual** (T06 ~00:23:02). Desarrollo formal en [T2 00_fundamentos_mdp.md §3](../raw/referencias/deep-rl-trading-agents/docs/teoria/00_fundamentos_mdp.md) y [00_repaso_rl.md](00_repaso_rl.md).

- **El coche rosa** que debe decidir si frena para no chocar: la distancia sola no basta; habría que pasarle el estado actual y el anterior para que derive implícitamente la velocidad. Hay que pasarle la **velocidad** (T06 ~00:23:25-00:24:28).
- **Pasar retornos o precios de días anteriores dentro del vector de estado no rompe Markov** (lo razonó un alumno: hoy conozco los valores del pasado). No hay que confundir "tener que remontarse a estados anteriores" con "meter información del pasado en el estado actual": si el estado dice "el retorno de los últimos días es X %", ya tiene lo que necesita (T06 ~00:25:32-00:26:24).
- En Breakout, con **dos fotogramas** se sabe hacia dónde va la bola; con uno no. Así que meter el pasado "no solamente no rompe, sino que es necesario" (T06 ~00:26:29-00:26:51).
- **Nota:** en ejecución, el **tiempo restante** es lo que hace Markov a un episodio de duración fija: sin él, dos estados con el mismo volumen pendiente serían indistinguibles aunque a uno le quede un minuto y al otro media hora. Por eso el docstring de `observation_builder` fija por defecto tiempo restante y volumen restante en porcentaje como características; construirlos es un TODO (LAB celda 7, función (3)). Con $N$ variable el tiempo en % es solo aproximadamente Markov (§2.4). La ventana `look_back = 60` del entorno existe para calcular características "roladas" (de ventana) ya en el instante $t=0$ del episodio (LAB celda 7, docstring de `__init__`).

### 5.6 Variables privadas y externas; el impacto también es interno (T06 ~00:35:48-00:37:53)

- El estado combina variables **extrínsecas/externas** y **privadas**. El impacto de tus acciones no es solo en el mercado; también es **interno**: si ya tienes BBVA, quizá no te interesa Santander por concentración, y al comprar o no Santander cambias tu propio estado. Sigue siendo interacción con el entorno (T06 ~00:35:51-00:36:48).
- Con posiciones pequeñas y sin que importe la cartera, el supervisado puede bastar; en cuanto importa en qué activos estás (riesgo de concentración), habría que modelizarlo (T06 ~00:37:19-00:37:53).
- En ejecución, dos formas de estructurar el vector (T07 ~03:16:37-03:18:18; CASOS p.35):
  - Por **dimensiones**: características de la **orden CARE**, del **activo** y de las **condiciones de mercado**. Hay que cubrir lo que explica la varianza del problema y usar el conocimiento de negocio para ir de lo alto a lo concreto.
  - **Internas/privadas** (tamaño de la orden, volumen restante, tiempo restante) frente a **externas/de mercado** (liquidez en primera posición, frecuencia de operaciones, si las ejecuciones van todas a un lado, rendimiento a corto plazo).
  - CASOS p.35 reutiliza el mismo dibujo del libro de APP p.95 (ask/bid en $t=0,1,2$, posiciones, *spread* y ejecuciones de 100 y 20 títulos) encima de la tabla de características y de esta doble clasificación.
- En profit-seeking, CASOS p.10 pide en los estados características que ayuden a explicar el problema, con el ejemplo de un "generador alpha" (*rewarded factors*, *mispriced/oversold*, riesgo específico); ver [05_profit_seeking.md](05_profit_seeking.md).

### 5.7 Normalización y estacionariedad

**Idea.** Todo normalizado, para que el agente "entienda lo mismo" con un activo u otro, con una orden u otra (T07 ~03:21:20-03:21:28). Nunca el precio en bruto (T06 ~00:25:15-00:25:25, ~02:19:28). Si se trabaja con varios activos, **estandarizar las entradas** (CASOS p.10). APP p.53: "los estados deben ser estacionarios". **Nota:** es decir, que la distribución de cada característica no cambie de escala entre días o activos.

Cómo se normaliza cada característica en el caso de ejecución (T07 ~03:18:46-03:21:57; CASOS p.35):

| Característica | Por qué importa | Normalización |
|---|---|---|
| Tamaño de la orden | Si es grande frente a lo que ejecuta el mercado hay que ser más agresivo; si es pequeña, una pasiva acabará ejecutándose | % ADV |
| Volumen restante | No es lo mismo 1 M de títulos en Santander que en otro valor | % del volumen inicial |
| Tiempo restante | | % del tiempo inicial |
| Ratio o diferencia tiempo/volumen restantes | Con un 30 % de tiempo y un 50 % de volumen por delante, "o me pongo las pilas o se me agota el tiempo" | ratio o diferencia de ambos % |
| Liquidez | ¿Habría volumen para una agresiva si la quisiera? | 1ª posición / tamaño de la orden hija (o si la cubre o no) |
| Frecuencia de operaciones | Actividad del libro | < 10 oper/min → 0 (bajo); ≥ 10 → 1 (alto) |
| Desbalanceo de operaciones | ¿Las ejecuciones van a un solo lado? | N oper. ask / N oper. totales |
| Rendimiento a corto plazo | ¿Va el precio en una dirección? | $\ln(P_t/P_a)$ |

El ejemplo de vector en $S_3$ (10:30 am) es: *spread* bid-ask **2 pbs**, probabilidad de ejecución **85 %**, tiempo restante **25 %**, volumen restante **20 %** (CASOS p.36; T07 ~03:22:35-03:22:50). El significado de cada característica para la ejecución (y la probabilidad de ejecución estimada con supervisado) está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

**Para el TWAP bastan dos.** Con tiempo y volumen restantes se puede hacer un TWAP sin ninguna característica de mercado: lo que necesita es que ambos "vayan de la mano" para que la ejecución acumulada quede en la diagonal (T07 ~04:52:12-04:52:47). Son las dos características por defecto que fija el docstring de `observation_builder`, ambas "en porcentaje"; calcularlas (`time_left`, `vol_left`) es un TODO (LAB celda 7).

**Nota:** con $N$ = `episode_bins`, $k$ = `state_pos`, $V$ = `vol_care` y $e_k$ = `exec_vol`, las dos características naturales son $\tau_k = 1-k/N$ y $\nu_k = 1-e_k/V$, ambas en $[0,1]$ y comparables entre episodios de distinta duración y tamaño; un TWAP perfecto mantiene $\nu_k\approx\tau_k$. La "diferencia o ratio" del profesor sería $\nu_k-\tau_k$ (positivo = voy retrasado) o $\nu_k/\tau_k$ (que explota cuando $\tau_k\to0$, mejor la diferencia). La forma de ponerlas en el código es un TODO de la práctica: ver [08_entorno_best_execution.md](08_entorno_best_execution.md).

### 5.8 Discretizar para métodos tabulares (T06 ~02:17:28-02:19:26)

- Con posición, ángulo y velocidades continuas, Q-learning tabular (una tabla finita) solo es posible **discretizando**: la velocidad es "alta o baja", etc. Se hace y a veces es bueno hacerlo (T06 ~02:17:28-02:18:29).
- **El nivel de discretización decide lo bien que aprende**: si discretizas mucho, el espacio es inmenso y no converge; si discretizas muy poco, hay muchísimo ruido entre "bajo" y "alto". Algo como **5 o 10 rangos** por variable; el notebook de Expected SARSA llega así a la solución perfecta (T06 ~02:18:29-02:18:59).
- Muchas veces se cree tener un problema continuo y en realidad "da pereza discretizar": no siempre hace falta el valor continuo (T06 ~02:19:13-02:19:26, frase confusa *(dudoso)*; la lectura más coherente es "no saltéis a soluciones espectaculares para problemas continuos: a veces basta discretizar").
- **Nota:** el material discretiza distinto de lo que se dijo: toma 500 episodios aleatorios, mide el rango entre los percentiles 10 y 90 de cada variable y redondea `s / (rango/50)` (`grid_step=50`), es decir, unas **50 celdas por variable dentro del 80 % central** más las de las colas, guardadas en un `defaultdict` que solo crea las celdas visitadas (RLM TD-q-learning celda 7; TD-expected-sarsa celda 3). Los "5-10 rangos" de clase son una regla general, no la del código.
- **Nota (bug del material):** el bucle de muestreo desempaqueta `s1, *_, done = env.step(a)`, que con la API de 4 valores de `gym==0.23.1` (`(obs, r, done, info)`, docs/raw/MIAX14/rl-material/setup.py) asigna a `done` el diccionario `info`, no el booleano. Ese `info` viene vacío (falso) hasta que el `TimeLimit` le añade la clave `TimeLimit.truncated` a los 500 pasos, así que cada episodio de muestreo dura hasta el límite aunque el palo haya caído antes: los percentiles incluyen estados fuera del rango válido y la rejilla es más gruesa de lo que parece (RLM TD-q-learning celda 7; TD-expected-sarsa celda 3). Detalle en [09_material_rl_notebooks.md](09_material_rl_notebooks.md).
- Más sobre discretización y sus límites en [T2 03_dqn.md §1 y §12](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md).

### 5.9 Reducir dimensionalidad y apoyarse en otros modelos

- APP p.53 propone pensar en reducir la dimensionalidad; APP p.91 añade que se puede **apoyar el aprendizaje en otros sistemas**: no supervisado (clustering, reducción de dimensionalidad), supervisado (generación de características, ayuda a la exploración) y capas de reglas para temas sensibles.
- En clase: no todo tiene que ser RL; un supervisado puede agrupar activos por perfil de liquidez para aplicar un algoritmo distinto a cada grupo, o generar características como la **probabilidad de que se ejecute una orden** en ciertas condiciones, que luego entran en el vector de estado (T07 ~00:23:11-00:24:37).
- Ejemplo de clase (T07 ~03:12:42-03:15:12): la probabilidad de ejecución de una limitada depende de lo limitada que sea; a 7,10 € podría ser del 90 % en los próximos 5 s y a 7,03 € del 0 %, porque el mercado no suele moverse 10 céntimos en 5 s. Un supervisado puede estimarla por niveles de profundidad y meterla en el estado, una por acción, y el agente elige entonces el mejor compromiso entre precio y probabilidad. "Eso sería un proyecto entero" (T07 ~03:14:12, ~03:15:09). Detalle en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).
- **Nota:** `BestExecutionEnv` calcula `n_features` llamando una vez a `observation_builder` (`_detect_num_feat`), así que añadir características solo exige tocar esa función; la red de BE se construye con ese tamaño de entrada (LAB celda 7; BE `dqn.py`).

---

## 6. Las acciones

### 6.1 Discretas o continuas (APP p.56; T06 ~02:25:51-02:27:07)

| | Ejemplos | Comentario |
|---|---|---|
| **Discretas** | Paraguas / no paraguas `[0, 1]`; comprar / vender / mantener `[0, 1, 2]` | Solo hace falta un **identificador**; es preferible `[0, 1, 2]` a `[1, -1, 0]` o `['b', 's', 'h']` |
| **Continuas** | Precio de la orden (7,65 €, 7,68 €...); precio de referencia ± pbs (−1 pb, 2 pbs...) | **No sirven para métodos basados en valor** |

- Hay algoritmos con acciones continuas: la salida de una red es una acción y otra red la evalúa (actor-crítico) (T06 ~02:25:58-02:26:22). En ejecución, acciones continuas como "mid ± pbs" son "solo para los policy" (CASOS p.37). Si necesitas sí o sí acciones continuas, descartas la mitad de los algoritmos (T07 ~00:25:05-00:25:18).
- Pero a menudo se cree necesitar acciones continuas y no hace falta; añaden complejidad. Discretizar en A, B, C simplifica bastante (T06 ~02:26:25-02:26:48).
- **Cuantas más acciones → más flexibilidad y más ruido** (APP p.56; T06 ~02:26:58-02:27:07; CASOS p.37: "+ posibilidades / + ruido").
- **No todas las acciones las toma siempre el algoritmo** → reglas, y más o menos libertad (APP p.56).
- **Nota:** el identificador importa porque una red de valor tiene una salida por acción y el índice de la acción es el índice de esa salida; `[1, -1, 0]` obliga a mapear y mezcla el identificador con un significado económico (el signo).

### 6.2 La duración del step cambia el problema de acciones (APP p.57; T06 ~02:27:23-02:29:31)

- **El volante**: girar a derecha o izquierda parece continuo, pero depende del resto de elementos. Si el step es corto (un milisegundo, incluso un segundo), basta con **mantener, girar un poco a la derecha o un poco a la izquierda**, porque en ese tiempo no puedes girar más; la concatenación de pequeños giros da el giro completo. Es lo que se hace en Breakout (T06 ~02:27:23-02:29:07).
- APP p.57: acciones discretas tomadas a muy corto plazo ≈ continuas; y hay que saber que **la acción se mantiene hasta el siguiente turno de actuar**.
- "Los elementos del RL interactúan entre sí" y **simplificar al máximo es clave, porque la estabilidad viene de simplificar el problema**: pensad bien las características y acotad (T06 ~02:29:07-02:29:31).

### 6.3 Eliminar o agrupar acciones con cuidado (APP p.57; CASOS p.38; T07 ~03:24:53-03:28:55)

APP p.57 propone un **estudio para eliminar acciones**:
- **Redundantes** → ¿hace falta *mantener* en un algoritmo de inversión?
- **De efecto similar** → ¿*mantener* y *pegada en 4ª posición* en ejecución?
- "Pensemos bien el problema antes de agrupar varias acciones o decantarnos por acciones discretas": $Q(s,\text{girar }11\,\%)=50$, $Q(s,\text{girar }12\,\%)=-100$, $Q(s,\text{girar }13\,\%)=50$. Lo ilustra con un joystick (ocho flechas alrededor) y un velero que llega a un timón. **Nota:** la lectura es que el valor puede no ser suave en la acción: agrupar 11-13 % en una sola acción "girar ~12 %" promediaría dos buenas y una catastrófica (o la ocultaría); antes de discretizar o agrupar hay que saber dónde están esos saltos.

El dato que lo sostiene en ejecución (CASOS p.38, tabla "agresividad de las órdenes a mercado enviadas por terceros", % de órdenes que llegan hasta cada posición del libro):

| Posición | Telefónica | BBVA | Santander |
|---|---|---|---|
| 0 | 0,03 | 0,05 | 0,06 |
| 1 | 91,48 | 92,36 | 89,90 |
| 2 | 5,85 | 5,46 | 6,98 |
| 3 | 1,64 | 1,35 | 1,80 |
| 4 | 0,56 | 0,44 | 0,67 |
| 5 | 0,24 | 0,17 | 0,28 |
| 6-10 | ≤ 0,11 | ≤ 0,08 | ≤ 0,14 |

Una llave agrupa las posiciones 1-2 y una flecha "Aprendía como mantener" señala la 3: una orden pegada más abajo casi nunca se ejecuta y el agente la aprendía como si fuera *mantener*. En clase: prácticamente el 90 % de las agresiones se queda en la primera posición y otro 5 % llega hasta la segunda (T07 ~03:25:31-03:25:53).

**Acciones dinámicas** (CASOS p.38; T07 ~03:26:12-03:27:46). Una acción puede significar cosas distintas según el momento del episodio: *mantener* al principio = orden **pegada en 4ª-5ª posición** (casi siempre equivale a no hacer nada, pero si alguien barre "hoy has tenido suerte" y consigues buen precio); al final = **no poner orden**, porque queda poco y hay que tener más control. El agente solo "dispara" *mantener*; el sistema lo traduce según una lógica.

**Reglas deterministas que conviven con el agente** (T07 ~03:28:09-03:29:07; CASOS p.37: "algunas acciones no elegidas por el agente, e.g., cargas de profundidad"): un TWAP real ponía "cargas de profundidad", órdenes muy abajo (9ª posición) por si alguien barría, cosa que con miles de operaciones acaba ocurriendo bastantes veces.

### 6.4 Tipos de órdenes en ejecución (CASOS p.37; T07 ~03:22:59-03:24:47)

Las acciones dependen de qué se deje decidir al agente (momento, tamaño o tipo de orden, §3.3). Si es el tipo: **limitadas** (a qué precio/posición), **agresivas** (nivel de agresión), **pegadas** (posición, n saltos o € de salto; se mueven con el libro, "no hay inacción entre steps"), **mantener** (no hacer nada hasta el siguiente step) y **cancelar** (salvo que la duración de las órdenes esté acotada al step). Se desarrollan en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md); el contexto de mercado (orden hija, libro, agresiva/pasiva) en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md).

### 6.5 Tamaño de la acción (CASOS p.39-41; CASOS p.12; T07 ~03:29:10-03:30:10)

| Opción | Ejemplo |
|---|---|
| **Fijo** | 1.000 títulos |
| **Relativo** | según un estudio de liquidez o de las operaciones de terceros |
| **Condicionado** | `min(tamaño estándar, vol_ask1 · 0,5)` |
| **Modificadores** del estándar | agresivo (× 2), conservador (× 0,5) |
| **Aleatoriedad** | para evitar ser identificado |

- El gráfico de CASOS p.40: mediana del volumen en **primera posición** del libro (bid y ask, dos líneas) del **Santander** por bins de 5 minutos, primer semestre de 2018. Empieza en ~2.100 a las 09:00, cae enseguida a ~1.350 (ask) y ~1.600-1.700 (bid) en la primera media hora, se recupera y se estabiliza en torno a **2.000 títulos** (línea discontinua) la mayor parte de la sesión y sube hasta unos 2.400-2.600 al cierre (17:30). El pie destaca que la liquidez del libro es **creciente y está descorrelacionada del volumen negociado**, que tiene forma de U.
- En clase: con unos 2.000 títulos en primera posición (menos por la mañana, más al final), hijas fijas de 1.000 títulos; y si el mercado suele negociar operaciones de unos 600 títulos, hacer algo parecido ayuda a pasar desapercibido *(dudoso: "disputar los 600 títulos")*. Así se le quita al algoritmo esa decisión (T07 ~03:29:16-03:30:10).
- En profit-seeking, las acciones de ejemplo son 0 mantener, 1 compra fuerte (100 %), 2 venta fuerte (100 %), 3 compra parcial (20 %), 4 venta parcial (20 %), con condiciones de liquidez `vol = min(cond, acción)` y otras (apalancamiento bruto o neto, cortos, mantener decisiones activas X tiempo) (CASOS p.12). Ver [05_profit_seeking.md](05_profit_seeking.md).

### 6.6 Acciones aleatorias no uniformes (APP p.57; T07 ~00:38:57-00:39:55)

La selección aleatoria no tiene por qué ser uniforme (1/N): se pueden poner pesos (p. ej. $w_{\text{mantener}} > w_{\text{comprar}}$ o $w_{\text{vender}}$) (APP p.57), o pesos según la utilidad esperada (el valor actual de cada acción) o según si una acción lleva a zonas más desconocidas del espacio (T07 ~00:39:15-00:39:55). En el entorno del taller esto vive en `action_sample`, cuyo TODO deja `p` por definir, sugiere en un comentario `np.random.choice([0, 1], p=[1-p, p])` y pregunta "¿qué distribución de probabilidad es mejor para la exploración?" (LAB celda 7, función (13)). **Nota:** para un TWAP que compra en ~10 % de los bins, una $p$ uniforme de 0,5 llenaría el buffer de episodios que acaban la orden muy pronto; una $p$ cercana a `vol_care/episode_bins` genera experiencias más parecidas a lo que se quiere aprender. Desarrollo en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

### 6.7 Acciones en el entorno de la práctica

`actions_fn = {0: _do_nothing, 1: _agg_action}`: mantener o comprar 1 título agresivamente al ask; `step` busca la función en el diccionario y lanza `ValueError` si la acción no existe (LAB celda 7). Son las "2 acciones (comprar/mantener)" que repasó el profesor (T07 ~05:00:22). El agente de BE fuerza la acción 1 en el step 0 (BE act). Detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md).

---

## 7. `reset()` y el inicio de los episodios

### 7.1 Qué tiene que hacer (APP p.86; T06 ~04:43:51-04:48:54)

APP p.86:
- **Comienzo de un nuevo episodio**; devuelve la **primera observación**.
- El inicio **no tiene (no debe) por qué ser el mismo**: distintas condiciones o puntos de partida; métodos para crear episodios o escenarios aleatorios; se podrían empezar partidas a "mitad" del episodio.
- **Inicialización de variables**: reinicio (listas vacías, valores a 0...) y carga (generación de características...).

En clase: devolver la primera observación "lo sabe todo el mundo"; lo importante es **el comienzo** (T06 ~04:43:51-04:44:15).
- En ajedrez la posición inicial es siempre la misma, pero varían el rival y la apertura; lo que no quieres es entrenar siempre contra la misma persona (T06 ~04:44:23-04:45:21).
- En bolsa, no siempre Santander: también activos más exóticos. Se busca **variabilidad** (T06 ~04:45:21-04:45:42).
- No empezar siempre a las 9 de la mañana: se puede empezar perfectamente a las 12 (T06 ~04:47:56-04:48:10).
- `reset` tiene que **limpiar** todo (la parte del estado, las listas de registros): "es una ejecución completamente nueva"; y tiene que **dejar precomputado o cargado** lo que se vaya a usar durante la partida, según lo que se haya elegido (p. ej. qué activo) (T06 ~04:48:11-04:48:54).

### 7.2 El `reset` de la práctica (T07 ~04:48:40-04:52:04; LAB celda 7)

Orden lógico que explicó: el entrenamiento hace `reset`, luego en cada paso `act` (ε-greedy) y `step` (T07 ~04:48:40-04:50:03). Dentro de `reset`:
1. Generar los **parámetros del episodio**: duración y tamaño (T07 ~04:50:18-04:50:36).
2. Construir el episodio: elegir **activo** (Santander o Repsol), **día** y **hora de inicio** (a las 13:00, a las 14:00...) (T07 ~04:50:36-04:50:52).
3. **Precalcular el VWAP de mercado** del episodio, posible base de la recompensa: se puede dejar calculado porque el agente **no impacta** en el mercado (T07 ~04:50:56-04:51:49).
4. Devolver la **primera observación**: tiempo y volumen restantes (T07 ~04:51:55-04:52:04).

En el código: `reset()` = `_reset_env_episode_params()` (variables a 0 y listas vacías) → `_generate_episode()` (que llama a `_generate_episode_params()` y elige día y hora de inicio, reintentando si no caben el `look_back` y el episodio) → `_compute_episode_market_feat()` (VWAP y volumen de mercado del episodio) → `observation_builder()` (LAB celda 7). **Nota:** el notebook carga un solo activo por instancia (`BestExecutionEnv(data, look_back)` con el `dict_["train"]` de un pickle, LAB celdas 4-5, 7 y 20); elegir entre SAN y REP dentro de `reset` exigiría pasarle ambos diccionarios. Las funciones una a una están en [08_entorno_best_execution.md](08_entorno_best_execution.md).

**Atributos repasados en clase** (T07 ~04:59:34-05:00:52): número de steps del episodio (junto con el `look_back`), `vol_care` (~50-60 títulos), el diccionario de acciones `actions_fn` con 2 acciones (comprar o mantener), `n_features = 2` (tiempo y volumen), `episode` (el DataFrame acotado al episodio) y `episode_full` = episodio + los 60 bins previos, que sirven para calcular características "roladas" desde $t=0$ (T07 ~04:59:34-04:59:54), `episode_vwap` (VWAP de mercado del episodio), `market_ep_vol` (títulos negociados por el mercado en cada bin), `state_pos` (step actual), `exec_vol` (ejecutado) e históricos de recompensas y acciones para graficar. Completar el entorno "son como 10 pasitos" (T07 ~05:01:06). Los hiperparámetros de la red (α, etc.) "son menos importantes", y si el DQN con buffer se atasca vale un Q-learning, refactorizando (T07 ~05:01:11-05:01:45).

**Nota:** el ENUNCIADO exige que el TWAP "funcione para distintos días, momentos temporales, duración del episodio y tamaño de la orden care" (ENUNCIADO, ej. 2): ese requisito se cumple precisamente aleatorizando esas cuatro cosas en `reset`, y el entrenamiento solo puede generalizar a lo que `reset` haya muestreado.

**Nota (precálculos y fugas de información):** precalcular en `reset` lo que el agente no puede alterar (VWAP del episodio, volumen de mercado por bin) es correcto con impacto 0; si se añadiera impacto, el VWAP de mercado dependería de las acciones y habría que recalcularlo. Lo precalculado con datos futuros puede usarse en la **recompensa** pero no debe entrar en el **estado**, o el agente vería el futuro; y aun en la recompensa hay que vigilar que sea "factible", ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

### 7.3 Diversidad de episodios y validez externa (T06 ~04:22:33-04:31:02)

Un alumno preguntó cómo mezclar experiencias de contextos distintos (otras partidas, otros mercados) no estropea la política. Respuesta del profesor:
- **Póker**: si llenas el buffer solo con las partidas de un jugador agresivo, hasta que entren otras experiencias el algoritmo casi siempre aprenderá de ese estilo. Añadiendo las de uno conservador, y luego las de todos los jugadores del histórico, metes "más independencia, menos sesgo" y aprende de todo (T06 ~04:24:56-04:26:04).
- Si solo tienes un algoritmo agresivo, la mejor alternativa es añadirle algo de aleatoriedad (T06 ~04:26:06-04:26:34).
- **S&P 500 desde 1900** como único episodio que siempre empieza el mismo día: el algoritmo aprende a comprar el día 1 y mantener; no hay diversidad de episodios. No hace falta empezar siempre en el día 0, y variarlo da **validez externa** (que los resultados generalicen) (T06 ~04:27:38-04:29:11).
- Muchas veces no hay históricos de algo que funcione; si los hay, se usan para mejorarlo con RL (T06 ~04:29:17-04:29:39). El llenado inicial del buffer (20-30 partidas aleatorias, no un millón) se trata en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

---

## 8. Eficiencia del código

### 8.1 Por qué en RL importa mucho más que en supervisado (T07 ~00:10:37-00:12:31; APP p.89; T06 ~04:13:11-04:14:24)

**Idea.** En supervisado hay un DataFrame ya construido con características y etiquetas y la red lo recorre. En RL el agente toma una acción y **se crean observaciones nuevas** en cada paso ("esto no es un 1,2, es un 1,7"); si cada paso modifica un DataFrame, "te va la vida" (T07 ~00:11:21-00:12:15). APP p.89 lo resume con una tabla `DF` de características `Feature 1, f2, ..., fn` por instante `t0, t1, ..., tn` (valores como 0,6, 0,8, −1; 0,1, 1,2, −1,2; −0,4, 1,5, −0,2): "DF: muy cómodo, pero muy ineficiente en RL. No es lo mismo que, e.g., en supervisado".

- Como hay que evaluar muchísimas veces, el sistema tiene que ser "súper súper rápido": algo así como que ejecutar la orden de ese día en el simulador lleve **un segundo**; **10 s** ya sería "incluso un poco lento" (T06 ~04:13:41-04:14:03) *(dudoso: frase muy ruidosa; no queda claro el umbral exacto ni si se refiere a una orden hija o a todo el episodio; la lectura más coherente es el episodio completo)*.
- El buffer no es una lista con `pop(0)` sino una colección específica, más eficiente (T06 ~04:13:11-04:13:35); en el código del taller es un `deque(maxlen=buffer_size)` (BE __init__). En RL "se busca la estabilidad, simplificar... se busca la eficiencia".
- El profesor dedicó mucho tiempo solo a hacer el código más eficiente, por ejemplo **descartando pandas por diccionarios**, "superrápidos" (T06 ~04:14:06-04:14:24).
- Los entrenamientos son muy largos: optimizar el código al máximo y la búsqueda de hiperparámetros (APP p.92; T07 ~00:28:03-00:28:09).

### 8.2 La demo con `%timeit` (T07 ~00:12:35-00:15:56)

Creó varios objetos con **el mismo valor** y midió el acceso a cada uno con `%timeit` ("lo utilizo bastante para medir los tiempos"):
- un **DataFrame** accedido por posición, con la selección de fila y columna en un orden o en el otro ("lo pongo antes, lo pongo después") *(dudoso: la transcripción dice "el hilo", que parece `.iloc`; "lo pongo antes, lo pongo después" se lee como el orden de la selección de fila y de columna)* (T07 ~00:13:50);
- el DataFrame convertido primero a **array** (`.values`) y accedido por posición;
- un **array** de NumPy directamente;
- un **diccionario**.

Cifras que se oyen *(dudosas: la transcripción las encadena sin decir qué variante compara con cuál)*: una variante "7 veces más rápida" que otra ("un entrenamiento de una hora pasaría a una hora entre 7"), convertir a array "otras 5, bueno, 3 veces menos", trabajar directamente con el array "otras 8 veces menos", y el diccionario en **unos 45 nanosegundos**. Para la variante lenta `%timeit` hizo **10.000 iteraciones** y para la rápida **10.000.000** en el mismo tiempo. Conclusión: lo que con DataFrames tarda una hora puede ser cuestión de un segundo si todo se hace con diccionarios (T07 ~00:15:03-00:15:39). Depende de cuántas filas y columnas tenga el DataFrame (T07 ~00:14:40). Recomienda `%timeit` siempre que algo vaya lento y apunta que en aplicaciones de trading de baja latencia estos tiempos sí son relevantes (T07 ~00:15:47-00:15:56) *(dudoso: la transcripción dice "cámaras de Trading")*.

**Nota (lectura de las cifras):** `%timeit` de IPython elige el número de iteraciones por bucle como la primera potencia de 10 cuyo tiempo total supera 0,2 s. Que la variante lenta use $10^4$ iteraciones implica entre ~20 y ~200 µs por acceso; que la rápida use $10^7$, entre ~20 y ~200 ns, coherente con los 45 ns del diccionario. Con la variante rápida en los 45 ns del diccionario, la diferencia entre extremos es de **~450× a ~4.400×** ($20\,\mu s/45\,ns\approx444$; $200\,\mu s/45\,ns\approx4.444$), entre 2,5 y 3,5 órdenes de magnitud. El producto de los multiplicadores oídos ($7\times3\times8\approx170\times$) se queda por debajo de ese mínimo, otra razón para no fiarse de ellos: no se pueden reconstruir con seguridad.

### 8.3 El caso real: reconstruir el libro de órdenes (T07 ~00:16:04-00:17:31)

Era un caso real, con las órdenes que llegaron a BME en un día normal (T07 ~00:16:19-00:16:22): reconstruir el libro de órdenes del **Santander** de ese día, unas **500.000-600.000 órdenes** (de compra y de venta a distintos precios), respetando la **prioridad precio-tiempo** (a igual precio se ejecuta antes la que llegó antes), con toda la lógica de casar órdenes, y **tick a tick** (cada cambio redistribuye el libro):
- en pandas: **20-30 minutos** por día;
- con diccionarios que mantienen el orden de llegada: **2-3 segundos** por día.

**Nota:** es un factor de ~400-900×, del mismo orden que el de la demo.

### 8.4 Cuándo optimizar

No siempre hace falta: si el problema es sencillo (el "triciclo"), se mantiene sencillo y se trabaja con pandas; si no, "os tenéis que poner un poco más finos" (T07 ~00:17:48-00:18:03). El notebook de best-execution, "de ahí saldrá la práctica", trabaja con pandas **para facilitarlo** (T07 ~00:12:19-00:12:31).

**Nota (propuestas concretas para la práctica):**
- Medir antes de optimizar (`%timeit` sobre `env.step`, `agent.act` y `agent.learn`): en un DQN de Keras, la llamada a la red en cada paso (`model(s).numpy()` en BE `getQs`) puede costar más que el acceso a pandas.
- En `reset`, convertir las columnas del episodio (`ask1`, `bid1`, `cumvol`...) a arrays de NumPy una sola vez y acceder en `step` por índice entero (`state_pos`), en lugar de `.iloc`/`.loc` en cada paso; precalcular mids y volúmenes por bin.
- Evitar recalcular desde cero en cada paso agregados que se pueden llevar acumulados (VWAP del algoritmo y de mercado: sumas de $p\,v$ y de $v$).
- El buffer de BE es `deque(maxlen=buffer_size)` (BE __init__), eficiente para añadir y descartar por los extremos, pero `sample` lo indexa en posiciones aleatorias (`self.buffer[i]`), y el acceso por índice en el centro de un `deque` es lineal en el tamaño. Con buffers de decenas de miles de transiciones, un buffer circular sobre arrays de NumPy o una lista con índice rotatorio muestrea más rápido. Es optimización, no un error.
- Escala del problema: 1.000 episodios × 500 steps = 500.000 llamadas a `step` y a `act`; 1 ms por paso son ~8 min, 10 ms son ~1,4 h. El "un segundo" de clase (§8.1), leído como episodio completo de ~500 pasos, equivale a ~2 ms por paso (entorno + agente): ~17 min para esos 1.000 episodios; con 10 s por episodio serían ~2,8 h.

---

## 9. Curriculum de datos: de lo sencillo a lo complejo

### 9.1 Empezar sobreajustando a propósito (T06 ~04:45:45-04:47:49; T07 ~00:18:03-00:18:48)

**Idea.** Primero simplificar e **intentar sobreajustar**: el mismo día y el mismo activo ("dame un buen Santander"); cuando lo hagas bien con Santander, te daré el resto (T06 ~04:45:45-04:46:16). Aunque "solo estoy ajustando y esto no sirve para nada", sí sirve: **es un test** de que los elementos están bien configurados (buena recompensa, buenos estados); si ni eso funciona, algo está mal (T06 ~04:46:34-04:46:56). En el entrenamiento, por la cantidad de parámetros (cada componente "tiene sus 20 papers"), se busca la sencillez y el sobreajuste inicial y a partir de ahí se evoluciona (T07 ~00:18:03-00:18:48). APP p.90 matiza que el *overfitting* es "distinto" en RL: a veces interesa sobreajustar, pero "¡no en bolsa!". **Nota:** compatible con el sobreajuste inicial como test, porque la advertencia se refiere al modelo final. Lo mismo recomienda para los TFMs: no complicarlo desde el principio, buscar algo sencillo que funcione y después la complejidad (T06 ~04:46:18-04:46:27).

### 9.2 Escalera de dificultad propuesta en clase

| Etapa | Qué cambia | Fuente |
|---|---|---|
| 0 | Mismo día, mismo activo (sobreajuste como test) | T06 ~04:45:54-04:46:56; T07 ~03:02:07-03:02:21 |
| 1 | Mismo activo, **dos días distintos**: uno alcista y otro bajista, para ver la diferencia | T06 ~04:46:56-04:47:11 |
| 2 | **Varios activos** (también exóticos) y más días | T06 ~04:45:21-04:45:42, ~04:46:56 |
| 3 | Inicios variados: **a mitad de partida**, no siempre a las 9 (también a las 12) | T06 ~04:47:20-04:47:29, ~04:47:56-04:48:10; APP p.86 |
| 4 | **Escenarios de estrés**: empezar directamente en el Covid, o **entornos sintéticos** en un momento de estrés | T06 ~04:47:29-04:47:47 |

"Puedes jugar con todo eso para testearlo, para mejorar el aprendizaje, y es superimportante"; mientras tanto se va entendiendo cómo responden los distintos elementos (T06 ~04:47:05-04:47:56).

**Curriculum learning** (T06 ~03:31:47-03:32:27, ~03:34:50-03:35:01): ir de lo fácil a lo difícil. Si el objetivo es hacer la hamburguesa, primero se aprende a desembalarla y luego a ponerla al fuego, "como de parvulitos a primaria". También caben varios agentes especializados en cosas distintas o subobjetivos ligados al objetivo principal; "se trata de romper el problema, siempre intentar simplificarlo". Aquí se presentó como técnica de exploración: ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

**El triciclo**: la orden de 400 títulos con hijas de 1 título, sin impacto ni costes, es el primer escalón deliberado del caso de ejecución; si no funciona ahí, no funcionará con órdenes reales (T07 ~03:00:30-03:00:53; ~00:17:48).

**Nota (traslado a la práctica):** la escalera encaja con los ejercicios. Para depurar el ej. 1 basta la etapa 0 (fijar día, hora, duración y tamaño en `reset`); el ej. 2 exige por enunciado la etapa 1-2 restringida a un activo (distintos días, momentos, duraciones y tamaños, bastando con mostrarlo en train, sin validación ni test) (ENUNCIADO, ej. 2); el estrés y los activos múltiples quedan para el ej. 4 o fuera. Hoja de ruta en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

---

## 10. Realismo del simulador: asunciones de ejecución

**Idea.** El simulador debe ser "lo más realista posible" (APP p.94); con datos agregados, el realismo se consigue con **asunciones explícitas**. Con mayor resolución temporal hacen falta asunciones de ejecución, modelos matemáticos, penalizaciones o umbrales de viabilidad y calibraciones (CASOS p.13); limitar el tamaño de las órdenes en % ADV ayuda con las asunciones de impacto (CASOS p.31).

- **Los datos del taller** (T07 ~04:38:35-04:44:23; LAB celda 2): diccionario por día → DataFrame con fotos cada **5 s** de la primera posición de cada lado (precio y volumen: `ask1`, `bid1`, `vask1`, `vbid1`), último precio ejecutado (`last`), volumen acumulado (`cumvol`) y fecha; sesiones de **13:00 a 17:25** (por eso el volumen acumulado no empieza en 0); train = primeros 60 días hábiles de 2018, validación = primeros 30 de 2019, test = días 31-60 de 2019 (en clase dijo que se pueden juntar). Repsol o Santander, basta uno. Extras opcionales: `orderbook.pkl` (hasta 3 posiciones) y `executions.pkl` (todas las ejecuciones con precio y nivel de agresión). **Nota:** 13:00-17:25 son 265 min = 3.180 bins de 5 s por sesión; con `look_back` de 60 bins (5 min) y episodios de hasta 600 bins (50 min), caben más de 2.500 horas de inicio distintas por día. Estructura completa en [08_entorno_best_execution.md](08_entorno_best_execution.md).
- **Asunciones de la práctica**: hijas de 1 título para asumir impacto 0 y coste 0 (T07 ~04:44:55-04:45:06); sobre por qué el coste 0 es razonable, ver el "triciclo" de §3.3.
- **Precio de ejecución entre fotos** (T07 ~05:01:56-05:05:30): el notebook ejecuta la compra al **ask al comienzo del bin**, lo que equivale a latencia 0; es un precio "tonto" y sesgado. Mejor aproximar lo que pasa entre fotos: la media de los medios de fotos consecutivas (las operaciones de esos 5 s se habrán repartido entre ambos lados) o, si se asume una latencia (p. ej. 2 s), una interpolación entre fotos. "Da exactamente igual lo que queráis hacer", pero ese tipo de razonamiento es el que hay que hacer al plantear las **asunciones de impacto** y sacar "el mayor provecho de vuestras fotos" cuando no hay información ejecución a ejecución. Es el mismo problema de las fotos de §4.4.
- **El mismo problema en el lado del mercado**: de cada bin se sabe cuántos títulos negoció el mercado (diferencia de `cumvol`) pero no a qué precio (T07 ~05:03:05-05:03:39). El notebook asume el mid de cada step para el VWAP de mercado y deja como opcional "un precio más realista" (LAB celda 7, función (4)). Mencionó también, de forma confusa, que en muchos algoritmos se piden "2 precios", uno más o menos realista (T07 ~05:02:00-05:02:16) *(dudoso: puede referirse a calcular un precio optimista y otro conservador para acotar el resultado)*. **Nota:** la asunción de precio debe ser coherente entre el agente y el benchmark; si no, el *slippage* mide la asunción y no la habilidad del agente. Detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md).
- **Nota:** `executions.pkl` permitiría sustituir la foto por las ejecuciones reales dentro de cada bin (solo Repsol, 2019, 80 días; LAB celda 2): es la vía natural para un simulador más realista en el ej. 4. Riesgos y erratas del precio de ejecución del notebook (ask al inicio del bin frente al mid del VWAP de mercado, etc.) en [08_entorno_best_execution.md](08_entorno_best_execution.md).

---

## 11. Producción y estado del arte de los entornos

- **Producción** (APP p.96; T07 ~00:30:41-00:32:41): se puede entrenar en producción pero no desplegar de forma continua aunque el modelo diga que ha "mejorado"; salida controlada (pocos usuarios, poco dinero); el simulador no asegura una representación 100 % fiel; y "el algoritmo puede hacer lo que sea con tal de maximizar sus recompensas". Aún más que en supervisado: los primeros meses hay que llevar un log para ver si la performance se desvía, porque el entorno es simulado y la recompensa es solo un **proxy** del objetivo. No basta con llegar al punto B: hay que ver **cómo** llega ("andando o a la pata coja") y que el comportamiento en producción se parezca al del entrenamiento. Desarrollo en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).
- **¿Dónde estamos? (2026)** (APP p.97; T07 ~00:30:25-00:30:41, ~00:32:47-00:33:59): el cuello de botella es generar **millones de experiencias realistas en poco tiempo**. Tres líneas: **simulación en GPU** (miles de entornos en paralelo en una sola GPU: Isaac Lab, MuJoCo MJX), **world models** (la IA genera mundos interactivos en tiempo real: Genie 3 → Genie 4, 2026) y **mercados sintéticos** (IA que genera órdenes como entorno de trading: MarS, Microsoft, ICLR 2025). "Pero el simulador también se equivoca → validar con datos reales antes de producción". Del recuadro de APP p.95 que da paso a esta diapositiva dijo que es una "vía de esperanza, pero muy, muy verde", aunque ha leído que estas herramientas ya están ayudando (APP p.95 (recuadro) y p.97; T07 ~00:30:25-00:30:31). Al comentar la simulación en GPU parece decir que antes la física de los entornos y el renderizado se repartían entre CPU y GPU con muchas ineficiencias, y que ahora se corren miles de agentes en paralelo sin salir de la GPU (T07 ~00:33:24-00:33:44) *(dudoso: tramo casi ininteligible)*. Los mercados sintéticos, "con muchas pinzas", porque responden a patrones (T07 ~00:33:51; ver §5.4).

---

## 12. Erratas, discrepancias y dudas

| Punto | Fuente | Qué pasa | Lectura o corrección razonada |
|---|---|---|---|
| Regla de tablas "de 40 movimientos" | T06 ~04:05:09 | La regla de la FIDE es de 50 movimientos sin captura ni movimiento de peón | Error de clase o de transcripción; no afecta al argumento (condición de fin no temporal) |
| Steps por partida de ajedrez (500 / 50 / 200) | T06 ~04:05:52-04:06:25 | Cifras contradictorias | *(dudoso)*; solo importa el método (steps × episodios = buffer) |
| Cálculo de steps intradía ("9 h son 8 h", "8 h × 65 × 60") | T06 ~04:07:54-04:08:18 | Aritmética mezclada | Rehecho en §3.4 (≈ 360-383 steps con decisión por minuto) |
| "720 × 0,9 = 65 títulos", "una orden hija de 65 títulos" | T07 ~04:47:21-04:47:36 | $720\times0{,}9=648$; y las hijas son de 1 título | Debe ser ≈ 0,09 (ejecutar en ~9-10 % de los bins), coherente con `vol_care` ∈ [7,5 %; 12,5 %] (LAB celda 7); los 65 títulos son la orden CARE (65 hijas, "65 barritas") || Episodios de "90 min a 2 h" frente a 400-600 bins | T07 ~04:44:45 frente a LAB celda 7 y T07 ~04:46:51 | 400-600 bins de 5 s son 33-50 min | No es contradicción: el notebook usa episodios cortos y el profesor dijo que se pueden alargar (2-3 h) |
| Multiplicadores de `%timeit` (7×, 3×, 8×) | T07 ~00:14:21-00:14:54 | No se sabe qué variante compara con cuál, y su producto (≈170×) queda por debajo del cociente que implican las iteraciones de `%timeit` (≈450-4.400×) | Solo es fiable el orden de magnitud (§8.2) |
| "5-10 rangos" para discretizar | T06 ~02:18:40 | El material usa ~50 celdas por variable en el 80 % central | Regla general frente a implementación concreta (§5.8) |
| `s1, *_, done = env.step(a)` en el muestreo de la rejilla | RLM TD-q-learning celda 7; TD-expected-sarsa celda 3 | Con la API de 4 valores, `done` recibe `info` | Cada episodio de muestreo dura hasta el `TimeLimit` (500 pasos); rango 10-90 inflado (§5.8) || Tamaño: "el doble... un 50 % más" | T07 ~03:06:07 | Dos cifras distintas para la tercera opción | *(dudoso)*; CASOS p.41 usa × 2 para el modificador agresivo |
| Precio de la limitada en el ejemplo de las fotos | T07 ~03:11:15-03:12:17 | "7 con 0,7", "a 7,7" | Probablemente 7,07 €, entre el mínimo de 7,03 € y la foto final de 7,10 € (CASOS p.34) |
| Enlace `gym.openai.com/envs/CartPole-v1/` | APP p.51-52 | Web de Gym antigua, ya no mantenida | **Nota:** la documentación actual está en `gymnasium.farama.org` (CartPole en *Classic Control*) |
| `reset()` de APP devuelve `(obs, info)`; el de LAB solo `obs` y `step` 4 valores | APP p.84 frente a LAB celda 7 | API distinta | No es error: LAB sigue la convención antigua de `gym`; ver §2.3-2.4 |
| "60 pfs en atari" | APP p.55 | Errata tipográfica | 60 fps |

---

## 13. Aplicación a la práctica

**Nota (resumen operativo; las decisiones se razonan en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md)):**
1. **Episodio** = una orden CARE de compra, con duración (400-600 bins según el docstring) y tamaño (`vol_care` ≈ 7,5-12,5 % de los bins) aleatorios, día y hora de inicio aleatorios; fin cuando se acaba el tiempo o se completa la orden. Todo ello es TODO del notebook: hay que implementarlo respetando docstrings y aserciones (§3.3; LAB celda 7).
2. **Step** = bin de 5 s en el notebook; valorar si agregar a 1 min (§4.4) y recalcular γ y buffer en consecuencia ([03](03_recompensas_y_descuento.md), [04](04_exploracion_memoria_y_estabilidad.md)).
3. **Estado** del TWAP: tiempo y volumen restantes en %, que hacen (aproximadamente) Markov el final por tiempo; valorar añadir los pasos restantes absolutos normalizados (§2.4, §5.5, §5.7); para el ej. 4 (idea A), añadir características de mercado normalizadas y justificadas una a una (§5.6-5.7; [07](07_rl_para_ejecucion.md)).
4. **Acciones**: mantener / comprar 1 título; exploración con $p$ no uniforme (§6.6).
5. **`reset`**: limpiar, muestrear los cuatro parámetros que exige el ENUNCIADO y precalcular solo lo que el agente no puede alterar, sin meter datos futuros en el estado (§7.2).
6. **Terminal frente a truncado**: con las dos condiciones del notebook, `target = r` es correcto; si se añade un truncado artificial, hacer *bootstrap* (§2.4).
7. **Curriculum**: depurar con un día y un inicio fijos, luego aleatorizar (§9.2).
8. **Eficiencia**: medir con `%timeit` y pasar a NumPy dentro del episodio si el entrenamiento es lento (§8.4).
9. **Documentar** cada decisión de diseño del entorno: el ENUNCIADO valora sobre todo la motivación, y más aún si el entorno se rehace desde cero ([notebook_partida.md](notebook_partida.md)). En clase: las decisiones arbitrarias están "fatal", y en la práctica lo que interesa es la motivación de por qué se ha hecho una cosa y no otra, como en las pruebas técnicas de las entrevistas de trabajo (T06 ~04:10:04-04:10:46).

Términos (step, episodio, `terminated`, `truncated`, ADV, orden hija, pegada, VWAP...) en [11_glosario.md](11_glosario.md).
