# Empezad por aquí: enunciado, entrega y lo que dijo el profesor

> **Fuentes:**
> - **ENUNCIADO** (`docs/raw/MIAX14/ejercicios_taller_rl.docx`): enunciado oficial de los 4 ejercicios, completo. **Es la fuente que manda.**
> - **LAB** (`docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`): notebook a completar, celdas 0-2 (portada, resumen, datos), 3-4 (imports y carga de datos) y 8-21 (TWAP, parametrización, buffer, entrenamiento y gráfico). La clase del entorno (celda 7) se desarrolla en [08_entorno_best_execution.md](08_entorno_best_execution.md).
> - **LABSTEP** (`docs/raw/MIAX14/best-execution/workshop/execution_lab-step_by_step.ipynb`): celdas 4, 6 y 8 (rango de tamaño de la orden y generación del episodio).
> - **BE** (`docs/raw/MIAX14/best-execution/`): `README.md` (motivación e instalación), `setup.py`, `src/agents/dqn.py` (constructor y `act`), `.devcontainer/devcontainer.json`, `dockerfiles/devcontainers/{docker-compose.yml, python/dockerfile, python/requirements.txt}` e `images/twap_model/twap_reproduction_dqn.png`.
> - **DATOS** (`docs/raw/MIAX14/best-execution/data/*.pkl`, escaneo estático sin cargar los pickles): nombres de columnas y versiones de los ficheros.
> - **RLM** (`docs/raw/MIAX14/rl-material/notebooks/`): `cliff-walking.ipynb` (parámetros de la demo de exploración) y `profit-seeking.ipynb` (ventanas del Sharpe).
> - **T06** (`docs/raw/26 11 06.pdf`, clase del viernes 6/11/2026) y **T07** (`docs/raw/26 11 07.pdf`, sábado 7/11/2026): transcripciones automáticas, muy ruidosas, de las clases de Jesús Sanz. Tramos usados: T06 ~00:02-00:06, ~00:59-01:07, ~01:11, ~02:57-03:00, ~04:09-04:12, ~04:52-04:59; T07 ~00:03-00:26, ~00:34-00:44, ~01:16-01:19, ~01:43-01:50, ~01:58-02:02, ~02:57-03:07, ~04:02-04:05 y ~04:27-05:10 (presentación de la práctica).
> - **APP** (`docs/raw/MIAX14/applying_RL_MIAX.pdf`): p. 1-3 (aviso, portada, agenda), p. 64 y 66 (exploración, nivel de aspiración) y p. 78 (tamaño dinámico del buffer).
> - **T2** (`docs/raw/referencias/deep-rl-trading-agents/`): `docs/teoria/notebook_partida.md` del Taller 2, como modelo de estructura.
>
> **Convenciones:** `(T07 ~04:28)` = hora aproximada en la transcripción; todo lo que viene de ellas es **paráfrasis**, y las citas literales son cortas. `(LAB celda 12)` = celda del notebook contando desde 0. `(APP p.3)` = página de las diapositivas. "*(dudoso)*" = la transcripción no permite asegurar lo dicho; se dan las lecturas posibles. **Nota:** = contenido propio (cálculos, interpretaciones, propuestas), no del enunciado ni del profesor. Cuando el ENUNCIADO y la clase difieren, se dice y **prevalece el ENUNCIADO**.
>
> Escrito el 10/11/2026.

---

## 1. La práctica en una página

**Idea.** Hay que construir, sobre datos reales del libro de órdenes de Repsol o Santander, un agente de RL que ejecute una orden CARE de compra troceándola en órdenes hijas de 1 título. Primero se completa el notebook hasta que entrena de principio a fin (ej. 1). Después se diseña la recompensa para que el agente se comporte como un TWAP (ej. 2), se sustituye el decaimiento de ε por número de episodios por uno basado en la mejora del agente (ej. 3) y se añade una mejora libre (ej. 4). Se evalúa **sobre todo la exposición y la motivación** de cada decisión; el código respalda lo expuesto (ENUNCIADO).

| Ej. | Puntos (ENUNCIADO) | Qué se pide, en una línea | Qué se toca, según el profesor (T07 ~04:58) |
|---|---|---|---|
| 1 | **3,5** | Completar `execution_lab.ipynb` (o un entorno propio) y lograr un entrenamiento y una ejecución completos | Que el notebook se ejecute hasta el final; no hace falta una recompensa concreta |
| 2 | **3,5** | Recompensa que **replique un TWAP**, con estado y recompensa explicados, gráficos y ajuste **motivado** de hiperparámetros | La función de recompensa |
| 3 | **1,5** | ε que **no dependa del número de episodios**, sino de la mejora del agente | El decaimiento de ε |
| 4 | **1,5** | Mejora libre (ideas A, B y C o cualquier tema de clase), explicando la lógica | Lo que se quiera |

- **Total: 10 puntos** con las cifras del ENUNCIADO. En clase se refirió al ej. 4 como "el otro punto", lo que no aclara su valor (T07 ~04:29); ver §12.
- **Entregables:** notebook ejecutado con la resolución y la motivación, con los bloques modificados señalados. Puede ir acompañado de ficheros `.py` y de una breve memoria aparte (ENUNCIADO, §3).
- **Exposición:** el día **18** (T07 ~05:06), de unos **10-15 minutos** por grupo (T07 ~05:06), con el orden publicado por el profesor en OneDrive (§11).
- **Contacto para dudas de alcance:** `jesus.sanz@oliverwyman.com`. Es el correo del trabajo, el que pidió usar (ENUNCIADO; T07 ~05:05).

*Fuente: ENUNCIADO; T07 ~04:27-04:31, ~04:58, ~05:05-05:07.*

---

## 2. El caso: qué problema resuelve el agente

**Idea.** El caso es un algoritmo de ejecución de tipo calendario (*schedule*): baja urgencia y mínimo impacto, solo compra y siempre completando la orden. Se mide con el VWAP *slippage*. La práctica lo reduce a una única decisión, **cuándo** lanzar cada orden hija.

**Qué pretende el taller.** Según el README del repositorio, es la base de un proyecto de unas **10 horas** para recorrer de principio a fin un problema real de RL, porque los alumnos sabían resolver tareas de RL ya definidas pero no diseñar un proyecto completo desde cero. Los alumnos construyen un **simulador de libro de órdenes deliberadamente simple** (órdenes de cierto volumen a cierto precio) y, sobre él, la tarea principal es diseñar **recompensas y modelos** de RL que rindan como los algoritmos tradicionales; la figura del README es un DQN que sigue una política TWAP (§6.2). El propio README avisa de que el simulador asume varias situaciones poco realistas, por ejemplo **impacto nulo** (BE README.md, "Motivation of the project"). LAB celda 1 lo llama un simulador "relativamente naive" pero funcional, sobre el que se aplican algoritmos clásicos y basados en IA.

**Nota:** conviene decir en la exposición qué hipótesis del simulador se mantienen (impacto nulo, sin costes, precio de ejecución) y cuáles se discuten (§12, fila 15).

### 2.1 Objetivo del caso (T07 ~01:43-01:50)

- **Tipo de algoritmo.** De los cuatro tipos (calendario, oportunistas, de precio de llegada y de enrutado o *dark pools*), el caso se centra en los de **calendario**, que son los más sencillos de entender (T07 ~01:43-01:44). Los tipos y el dilema del trader se desarrollan en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md).
- **Objetivo.** Ejecutar la orden con el menor impacto de mercado y poca urgencia, **sin tener en cuenta el precio de llegada**. Solo compra, y la orden se ejecuta **completa**: si son 10 millones de títulos, son los 10, no 8 (T07 ~01:44-01:46).
- **Métrica: VWAP *slippage*.** Compara el precio medio ponderado por volumen del algoritmo con el VWAP del mercado en el mismo periodo. La ejecución tiene éxito si el precio propio queda por debajo, porque es una compra (T07 ~01:45; LAB celda 1). La fórmula del notebook es el VWAP $\sum_i p_i v_i / \sum_i v_i$, y el *slippage* se expresa en puntos básicos (LAB celda 1).
- **Dispersión.** Una buena media no basta: un algoritmo agresivo puede tener una media excelente y una dispersión enorme. La dispersión es un factor a considerar (T07 ~01:46).
- **Benchmarks.** TWAP, VWAP y POV, programados sobre el histórico, sirven de referencia y para decidir cuándo bajar la exploración: cuanto más se acerquen los resultados del agente al *benchmark*, menos hay que explorar (T07 ~01:47). Es una pista directa para el ej. 3 (§7).
- **Mensaje de fondo.** Hay que volverse experto en los algoritmos clásicos, entender sus limitaciones y ver dónde la IA puede sustituirlos del todo o en un componente. Es "de lo casi más importante" de la presentación, y vale también para los TFM: plantear y estructurar el problema siempre es más importante (T07 ~01:48-01:50).

### 2.2 Episodios: el "TWAP con esteroides" (T07 ~02:57-03:02)

- Un episodio reproduce lo que suele pedir un cliente de TWAP: **un horizonte temporal y un número de títulos**. Conviene normalizar el tamaño como % del volumen diario: 1 000 000 de títulos puede ser el 10 % (T07 ~02:57-02:58).
- Los parámetros del episodio se muestrean de distribuciones históricas de peticiones reales (T07 ~02:59).
- **El "triciclo".** Se evitan las órdenes muy grandes, que obligarían a modelar funciones de impacto. En la práctica, orden de ejemplo de **400 títulos**, hijas de **1 título**, **sin comisiones** e **impacto 0**: con esas hipótesis el histórico es un simulador bastante realista. Si no funciona en un escenario tan controlado, no funcionará con órdenes de un millón (T07 ~03:00-03:01). Justificó el coste cero con promociones o tarifas planas en las que los primeros títulos salen gratis *(dudoso: la transcripción está muy rota)* (T07 ~03:00:59-03:01:09).
- Conviene dar variabilidad (empresas, tamaños), pero se puede empezar con un solo día y un solo activo para depurar, aunque eso no sirve para producción (T07 ~03:01-03:02).
- El episodio es la ejecución **completa** del TWAP (T07 ~03:02).

El diseño de episodios, estado, acciones y recompensa del caso completo está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

### 2.3 Acotar el problema: qué decide el agente (T07 ~03:02-03:07)

Un TWAP toma tres decisiones: **momento**, **tamaño** y **tipo de orden**. Un agente puede cubrir las tres, solo una, o pueden repartirse entre varios algoritmos. El profesor propuso tres simplificaciones:

| Alternativa | El agente decide | Fijo |
|---|---|---|
| (a) | En qué instante, dentro de cada ventana de 10 minutos, lanza la hija | Tamaño y tipo |
| (b) | El tipo de orden | Momentos (los del TWAP) y tamaño |
| (c) | El tamaño: la mitad, el mismo o un 50 % más (también dijo "el doble") | Momento y tipo |

**Simplificación de la práctica:** el agente elige **el momento** a lo largo de **todo el episodio**, no dentro de ventanas de 10 minutos; la orden es **siempre agresiva** y la hija **siempre de 1 título** (T07 ~03:06). Lo considera de lo más importante de la clase: aislar los problemas y darles un tratamiento específico (T07 ~03:07).

**Nota:** las acciones del LAB son exactamente esas dos, no hacer nada o comprar 1 título agresivo (`_do_nothing` y `_agg_action`), y el profesor las resumió como "comprar o mantener" (T07 ~05:00). Ver [08_entorno_best_execution.md](08_entorno_best_execution.md).

### 2.4 Asunciones de la práctica (T07 ~04:44-04:45; ENUNCIADO ej. 2)

| Asunción | Fuente |
|---|---|
| Sesiones de 13:00 a 17:25; el episodio no tiene por qué ser el día entero | T07 ~04:44; LAB celda 2 |
| Episodios de 400-600 *steps* por defecto; se pueden alargar a 2-3 h | T07 ~04:46-04:47; LABSTEP celda 6 |
| Hijas de 1 título | ENUNCIADO; T07 ~04:44 |
| Sin impacto de mercado ni costes de transacción | ENUNCIADO; T07 ~04:44 |
| Solo compra, orden siempre agresiva | T07 ~01:46, ~03:06 |

---

## 3. Instrucciones generales del enunciado, párrafo a párrafo

1. **Punto de partida.** Se puede usar como referencia `execution_lab.ipynb`.
2. **Organización.** Una copia del notebook por ejercicio, o todos en el mismo. En cualquier caso hay que:
   - **especificar los bloques modificados** para cada ejercicio;
   - **añadir la explicación de la lógica** seguida.
3. **No hay solución única:** la motivación es lo que demuestra el conocimiento adquirido.
4. **Hiperparámetros.** Ajustar γ, tamaño del buffer, número de episodios, etc. cuando se crea conveniente.
5. **Datos.** La práctica puede hacerse con cualquiera de **los 3 ficheros de datos**. Qué tres son no está claro (§12).
6. **Entrega final:** un **notebook ejecutado** con la resolución y la motivación, sin perjuicio de estructurar el código en otros `.py`.
7. **Motivaciones.** Pueden ir en el notebook. También se permite, y "en ocasiones es recomendable", un **documento separado** con una breve memoria de los ejercicios y el resumen de las decisiones.

**Nota:** con lo anterior, lo más seguro es un notebook por ejercicio, o uno solo con secciones por ejercicio, siempre ejecutado y con una celda de cabecera por ejercicio que diga qué celdas o funciones cambian respecto al LAB y por qué. Las decisiones se resumen en un `DECISIONES.md` o memoria aparte, como en los Talleres 1 y 2 ([T2 notebook_partida.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/notebook_partida.md)). La propuesta concreta está en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

*Fuente: ENUNCIADO, bloque inicial.*

---

## 4. Forma de evaluación

### 4.1 Lo que dice el enunciado

- Se tienen en cuenta **principalmente la exposición y la motivación** de los ejercicios.
- El código debe **soportar** lo expuesto y **puede ajustar la nota** en consecuencia.

No hay rúbrica porcentual como en los Talleres 1 y 2: solo los puntos por ejercicio (§1).

### 4.2 Lo que dijo el profesor sobre qué valora

| Tema | Qué dijo (paráfrasis) | Hora |
|---|---|---|
| Decisiones arbitrarias | Son "fatal, siempre para todo". Como en los casos para casa de las entrevistas de trabajo, lo que interesa es **por qué** se eligió cada cosa. En la práctica quiere ver la motivación de hacer una cosa y no otra, lo preguntará y vale mientras se entienda *(transcripción ruidosa, sentido claro)* | T06 ~04:10 |
| Planteamiento antes que resultados | Lo que valora (lo dijo de los TFM, enlazándolo con el día anterior) es el **planteamiento del problema** y seguir pasos lógicos en cada componente. "Podéis sacar un sobresaliente a pesar de que no consigáis resultados", porque se entienden las limitaciones de tiempo y recursos | T07 ~01:18 |
| Demostrar conocimiento | Lo importante es demostrar lo aprendido y hacer un buen planteamiento | T07 ~01:18-01:19 |
| Estructurar el problema | Aislar los problemas y darles tratamiento específico es lo más importante de la clase | T07 ~01:49-01:50, ~03:07 |
| Ej. 1 como base | Pidió señalar el principio, "el punto 1, que es de lo que depende todo" *(transcripción ruidosa)*. **Nota:** sin un entorno correcto, los ej. 2-4 no son evaluables | T07 ~04:31:19-04:31:27 |
| Ej. 4 | Basta exponer qué se recogió de clase, cuál fue el razonamiento y qué se quiso implementar | T07 ~04:31 |
| Sobre-cumplimiento | Si alguien lo hace distinto, con funciones de impacto, bromeó con pedir la matrícula de honor *(dudoso: "os pone la matrícula en aprendizaje")* | T07 ~04:56 |
| Hiperparámetros de red | Los parámetros de la red, "el alfa y el no sé qué", son menos importantes; le interesan sobre todo los que están en el notebook. **Nota:** no los enumeró; cabe entender ε, γ, el buffer y el número de episodios de la celda 12. Citó α entre los menos importantes aunque también está en la celda 12 *(dudoso: quizá se refería a lo propio de la red, como neuronas o tasa de aprendizaje del optimizador)* | T07 ~05:01 |
| Peso en la asignatura | Comentario confuso: la práctica sería "como 3 puntos" de los 10, o de los "3 primeros" *(dudoso; parece referirse a ediciones anteriores)* | T07 ~05:09 |

**Nota:** ENUNCIADO y clase coinciden en que se puntúa el razonamiento. Lo que más rinde es que cada decisión (estado, recompensa, γ, buffer, criterio de ε) tenga una justificación escrita y una evidencia (gráfico o cifra) que la respalde, y que todo el grupo pueda defenderla en la exposición.

---

## 5. Ejercicio 1 (3,5 puntos): ejecución completa de un entorno y un algoritmo de RL

### 5.1 Enunciado

> Completa el notebook de `execution_lab.ipynb` para lograr un entrenamiento y ejecución completa.

| Necesario | No es necesario |
|---|---|
| Conseguir un **entrenamiento y una ejecución completos** | **Reutilizar el notebook.** Se puede crear entorno y notebook desde cero si contienen todos los elementos de un entrenamiento de RL, están **pensados para un algoritmo de ejecución** y **usan los archivos compartidos** del taller. En ese caso hace falta **más documentación** de la estructura y los pasos |
| Incluir los **elementos necesarios de un entorno de RL** | **Reutilizar el algoritmo de Deep RL** |
| Seguir el **orden adecuado de interacción** entre componentes | **Reutilizar los paquetes y versiones** del ejemplo, lo que incluye **usar PyTorch en vez de TensorFlow** |
| **Entrenar en el orden correcto** para el algoritmo elegido | Que el entrenamiento **responda a una lógica con sentido** (eso es el ej. 2) |
| **Documentar y motivar** decisiones | Motivar en detalle la parametría (p. ej. γ), porque aún no hay un objetivo claro |

### 5.2 Qué hay que rellenar en el LAB

| Celda | Contenido | TODO |
|---|---|---|
| 7 | Clase `BestExecutionEnv` | Todas sus funciones: parámetros y generación del episodio, observación, VWAP de mercado y del algoritmo, recompensa, condiciones de parada, acciones, `step`, `action_sample`. Detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md) |
| 8-9 | Agente `TWAP(DDQNAgent)` | `act(self, s)` con un **TWAP determinista** a partir de `s = [T, V]`. El markdown cita las variantes habituales que dificultan el rastreo (aleatorizar el tamaño o el intervalo entre órdenes, limitar la cantidad a un % del volumen de participación, no agredir más allá de una posición del libro) y pide implementarlo **sin ninguna modificación**: hijas **equidistantes en el tiempo**, sin mecanismos anti-detección (LAB celda 8; ver [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md)) |
| 10-12 | Parametrización | `epsilon`, `min_epsilon`, `gamma`, `alpha`, `buffer_size`, `batch_size`, `hidden_neurons`, `nepisodes`, `epsilon_decay`. Ya vienen fijados `n_log = 25`, `learn_after = batch_size`, `update_target_after = 10 * learn_after` y `env = BestExecutionEnv(data, 60)` |
| 13-14 | Inicialización del agente | `agent = "-----"`: instanciar `DDQNAgent(env, gamma=..., epsilon=..., alpha=..., batch_size=..., buffer_size=..., hidden_neurons=...)` con los valores de la celda 12. Por defecto: γ 0,99, **ε 0,9**, α 0,01, *batch* 64, buffer 50 000, 32 neuronas. `TWAP(DDQNAgent)` hereda el mismo constructor y crea las redes aunque no las use (BE src/agents/dqn.py) |
| 15-16 | Llenado del buffer | Ya escrito: `buffer_size` pasos con la política del agente y `reset` al acabar cada episodio; imprime el progreso cada 10 000 pasos. El comentario asume ε = 1 ("actuando random"), lo que exige iniciar el agente con `epsilon = 1` en la celda 12; con otro valor, parte del buffer se llena con la política de una red sin entrenar. Aun así, en el paso 0 siempre se compra (`act` devuelve 1 si `env.state_pos == 0`) y las acciones aleatorias siguen la `p` que se defina en `action_sample`, así que el llenado no es uniforme (LAB celdas 7 y 16; BE `dqn.py`). Detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md) |
| 17-18 | Entrenamiento | Ya escrito: `act` → `step` → `experience`; `learn` cada `learn_after` pasos y `update_target` cada `update_target_after`; al acabar el episodio, **ε lineal**: `epsilon = max(epsilon - epsilon_decay, min_epsilon)`; log cada `n_log` episodios |
| 19-20 | Gráfico | `set_trainable(False)` hace que `act` sea *greedy* (`_act_no_explore`, sin ε). Se crea un `BestExecutionEnv(dict_["test"], 60)` nuevo, se corre un episodio y `plot_results(env)` dibuja sus históricos. `cum_reward` se acumula con `γ**step` **después** de incrementar `step` (exponente desde 1, a diferencia de la celda 18) y no se imprime. Para las "imágenes correctas en train" del ej. 2 hay que cambiar `dict_["test"]` por `dict_["train"]` o decir qué *split* se usa (§12, fila 8) |
| 21 | Curva de aprendizaje | Ya escrita: `pd.DataFrame(history_rewards).rolling(20).mean().plot()`, media móvil de 20 episodios de la recompensa acumulada (sin descontar) de cada episodio de entrenamiento. Incluye la exploración, así que no mide la política *greedy* (§7.2) |

Las definiciones de la celda 11 son las del propio notebook: ε entre 0 y 1; `min_epsilon` para no llegar a 0 en entrenamientos largos; `epsilon_decay` se resta "cada determinado número de episodios"; γ en $[0,1]$ por ser tareas episódicas; α estático o adaptativo pero **nunca decayendo a cero**, porque el problema es no estacionario; el buffer, para obtener datos aproximadamente i.i.d.; `learn_after` y `update_target_after` en *steps*. La red se mantiene sencilla a propósito (LAB celda 11).

### 5.3 Lo que dijo el profesor

- **Qué basta.** Rellenar el notebook vacío, que tiene los componentes necesarios, y comprobar que funciona (T07 ~04:28-04:29). Para el ej. 1 no hace falta ninguna recompensa específica: que el notebook se ejecute hasta el final (T07 ~04:58).
- **Ficheros.** `execution_lab` es el que se entrega *(dudoso: se transcribe "el soldado"; por el contexto, el notebook vaciado con huecos "complétame", es decir, `execution_lab`)*: una clase con el entorno y huecos "complétame" con *tips*. Tras fijar ε y el resto, se entrena y debe salir un gráfico; en `images/` está el ejemplo de cómo se ve un TWAP. La clase es más difícil de manejar porque hay que acceder a sus atributos (T07 ~04:33-04:34).
- **`step_by_step`.** Saca las funciones de la clase fuera, en el **orden lógico**, empezando por generar el episodio (número de *steps* y orden a ejecutar). Se completan allí y se copian a la clase, poniendo `self.` donde haga falta (T07 ~04:34-04:35). Enseñó la solución por encima y el notebook vaciado; algún nombre "se llama distinto" entre uno y otro (T07 ~04:36-04:38).
- **Orden lógico.** `reset` inicializa la partida: pone las variables a 0 o vacías, genera los parámetros del episodio (tamaño y duración), elige activo, día y hora de inicio, y **precalcula el VWAP de mercado del episodio**, que puede servir para la recompensa. Se puede precalcular porque el agente no impacta en el mercado. Después devuelve la primera observación: **tiempo restante y volumen restante** (T07 ~04:48-04:52). **Nota:** en el LAB el activo no lo elige `reset`: viene fijado por el `data` que se pasa al constructor (celdas 4, 5 y 12, `BestExecutionEnv(data, 60)`), y `_generate_episode` solo muestrea día y hora de inicio (celda 7).
- **Bucle.** Número fijo de episodios ("la definición clásica que no me gustaba"), por ejemplo 1 000 o 10 000, con listas para registrar. Dentro, `act` con ε-greedy y después `step` (T07 ~04:49-04:50).
- **`step`**, "la función más importante": recibe la acción, la evalúa y guarda la recompensa, avanza una posición en el DataFrame, decide si ha terminado (sobre todo, si se ha completado el volumen) y devuelve observación, recompensa, *done* e *info*. Calcular el VWAP de mercado parcial es **opcional**: la recompensa no tiene por qué depender del rendimiento si solo se quiere replicar un TWAP (T07 ~04:57-04:58).
- **Atributos.** Número de *steps* (con el *lookback*), volumen a ejecutar ("50 títulos, 60 títulos, algo así"), diccionario de 2 acciones (comprar o mantener), 2 características (tiempo y volumen), `episode`, `episode_full` (episodio más 60 *bins* previos, para características móviles), VWAP de mercado del episodio, títulos negociados por el mercado en cada *bin*, posición (*step* actual), ejecutado e históricos para graficar. "Son como 10 pasitos" (T07 ~04:59-05:01).
- **Algoritmo libre.** No hay requisito de algoritmo (T07 ~04:32). Si el deep learning da problemas (DQN con buffer, "un poquito más inestable"), vale un **Q-learning**, refactorizando el notebook (T07 ~05:01). Al empezar T07 mencionó que su Q-learning se podía importar o reutilizar *(dudoso)* (T07 ~00:09).

**Nota:** "ejecución completa" significa, como mínimo, entrenar hasta `nepisodes` sin errores y sacar el gráfico de `plot_results` (LAB celdas 18-20). Para que el ej. 1 sirva de base al 2, conviene que el TWAP determinista de la celda 9 también funcione y se grafique: es el *benchmark* del ej. 2 y una referencia para el ej. 3. Los posibles fallos del bucle tal como viene (por ejemplo, `mse` sin definir si se registra antes del primer `learn`) están en [08_entorno_best_execution.md](08_entorno_best_execution.md).

*Fuente: ENUNCIADO ej. 1; LAB celdas 4-21; BE `src/agents/dqn.py`; T07 ~04:28-04:38, ~04:48-05:01.*

---

## 6. Ejercicio 2 (3,5 puntos): un algoritmo de ejecución estilo TWAP

### 6.1 Enunciado

El ejercicio tiene **dos partes igualmente importantes**:

- **Primera parte.** Modificar el sistema de recompensas para **replicar el comportamiento de un TWAP**. Explicar **qué variables hay en el estado** y **cómo y por qué** se ha definido la recompensa. Incluir **gráficos** con los resultados.
- **Segunda parte.** Para asegurar un entrenamiento correcto, ahora sí, **ajustar la parametría y motivarla** (p. ej. γ, tamaño del buffer).

**Requisitos y simplificaciones:**

| Punto | Texto del enunciado (resumido) |
|---|---|
| Tamaño de las hijas | **1 título.** La orden CARE debe estar alineada: con 400 *steps*, una CARE de **40 títulos** obliga a ejecutar en **1 de cada 10 *steps***, "algo razonable" |
| Mercado | **Sin impacto** ni **costes de transacción** |
| Fidelidad | No se pide una réplica exacta, sino un **comportamiento similar**. Ejemplo: `images/twap_model/twap_reproduction_dqn.png` |
| Validación | **No hace falta** validación ni test: basta con enseñar **algunas imágenes correctas en train** |
| **Requisito** | Debe funcionar para **distintos días, momentos, duraciones del episodio y tamaños de la orden CARE**, dentro del rango esperado |
| Activo | Basta con **SAN o REP**, no hace falta que funcione en los dos |

### 6.2 La figura de referencia

`twap_reproduction_dqn.png` (BE images/twap_model/; también en el README) tiene dos paneles sobre un mismo episodio, del día "06" entre ~16:07 y ~16:42 (unos 35 minutos, ≈420 *bins* de 5 s):

- **Arriba, "Market VS Algo":** el *mid price* (rojo) oscila entre ≈15,01 y ≈15,06, con una caída brusca hacia 16:35. El VWAP del mercado (azul) y el del algoritmo (naranja) van casi pegados: suben de ≈15,025 a ≈15,048. El azul cae a ≈15,040 hacia 16:35-16:38 y vuelve a ≈15,048 al final.
- **Abajo, volumen acumulado normalizado (0-1):** el del algoritmo (naranja) sube en **escalera casi diagonal** de 0 a 1 a lo largo de todo el episodio, con barras de ejecución (naranja) **equiespaciadas**. El del mercado (azul) avanza despacio hasta ≈0,22 y salta a ≈0,97 hacia 16:38, una operación grande puntual, para llegar a 1 al final.

- **Leyendas:** arriba `vwap`, `algo` y `mid_price`; abajo `mkt cumvol`, `algo cumvol`, `mkt vol` y `algo vol`.
- **Detalles útiles del panel inferior:** los peldaños del acumulado naranja son de ≈0,02, unas 50 barras (contadas a ojo), coherente con una CARE de ~50 títulos. La primera barra está en el primer *bin*, porque `act` compra siempre en el paso 0 (BE dqn.py), y el acumulado llega a 1 justo al final del episodio.

**Nota:** la referencia consume el **100 % del tiempo**; la idea B del ej. 4 pide terminar hacia el 90 % (§8).

**Lo que hay que reproducir es el panel inferior:** la diagonal del acumulado propio y las barras regulares. El profesor lo resumió así: el tiempo y el volumen restantes tienen que "ir de la mano" para que quede la diagonal (T07 ~04:52).

### 6.3 Lo que dijo el profesor

- **Estado.** Para el TWAP bastan **tiempo restante y volumen restante**, sin ninguna característica de mercado (T07 ~04:52). Las mismas dos *features* de la clase (T07 ~05:00). Al comentar el paper de Q-learning de 2006 recordó que empezaba con **solo dos características, tiempo y volumen** (T07 ~04:02), que con solo la variable de si queda mucho tiempo para el volumen pendiente se mejoraba un 50 % el *benchmark*, y que con características adicionales como el *spread* *(dudoso: se transcribe "Spray umbilical")* se ganaba otro 7-8 % (T07 ~04:03; ver [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md)).
- **TWAP determinista.** Es un `if`: si vas **por delante** del plan, no ejecutes; si vas **igual o por detrás**, lanza otra hija. Se puede hacer así, pero **lo quiere por recompensas** (T07 ~04:53-04:54). Es una regla muy sencilla que "responde prácticamente a ese ratio o diferencia" entre volumen y tiempo restantes (T07 ~04:54:24). **Nota:** esa diferencia, normalizada, es una candidata natural de recompensa (ver [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md)). Sobre si la celda 9 es obligatoria, ver §12, fila 18.
- **Versión tabular.** Como los escenarios son finitos, valdría un **Q-learning tabular** con 2 estados (volumen por encima del tiempo, o igual o por debajo) y 2 acciones (ejecutar o no): una matriz de 4 valores (T07 ~04:54-04:55). Q-learning se repasó en los Talleres 1 y 2 ([02_td_sarsa_qlearning.md del Taller 1](../raw/referencias/rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md)) y en [00_repaso_rl.md](00_repaso_rl.md).
- **Libertad total de implementación.** El notebook usa un doble DQN, pero no es obligatorio: "podéis tirar este notebook por completo" y montar el entorno como se quiera, siempre que con esta parametría y con SAN o REP se llegue al TWAP **por recompensas** y se enseñen las imágenes (T07 ~04:55-04:56).
- **Dónde se toca.** En la función de recompensa (T07 ~04:58).
- **Tamaño y flexibilidad** (T07 ~04:45-04:48). A 5 s por *step* salen **12 *steps* por minuto y 720 por hora**. Con hijas de 1 título, una orden de 720 en 720 *steps* no deja ninguna flexibilidad: una hija por *bin*. Con 360, el ratio es 0,5 (uno sí, uno no). Con una flexibilidad del 10 %, en 1 de cada 10 *bins*, dijo multiplicar 720 "por 0,9" y obtener una orden de **65 títulos**: 65 barras en el gráfico final. Por defecto, episodios de **400-600 *steps*** (menos de una hora), aunque pueden durar 2-3 h.
- **Ejemplo ejecutado.** Salieron "1 14 214" *(dudoso)* *steps* y una orden de **55 títulos** (T07 ~04:48).

**Nota (cálculos):**
- 400-600 *steps* × 5 s = **33-50 minutos**. Las "90 min - 2 h" de las asunciones (T07 ~04:44) son la duración de un TWAP real, no la del valor por defecto.
- **65 títulos** corresponde a 720 × **0,09** = 64,8, no a 720 × 0,9 = 648. Con exactamente 1 de cada 10 serían 72. Lo coherente es "≈9-10 %", y la transcripción omite un cero.
- **Rango esperado del tamaño.** LABSTEP (celda 6) fija `episode_bins` uniforme en [400, 600] y `vol_care = int(U[0,075; 0,125] · episode_bins)`, con *asserts* que lo comprueban. La CARE va por tanto de **30 a 75 títulos**, entre el 7,5 % y el 12,5 % de los *steps*. El ejemplo del ENUNCIADO (40 de 400) y los "50-60 títulos" de clase caen dentro. Es la lectura más razonable de "dentro del rango de tamaño esperado".
- **El ejemplo de 55 títulos** es incompatible con cualquier agrupación literal de "1 14 214" (14, 114 o 214 *steps*): exigiría ≥ 440 *steps* con ratio ≤ 0,125. Una lectura posible es **514** *steps* (ratio 0,107).
- El requisito de "distintos días, momentos, duraciones y tamaños" ya lo cubre el LAB si `reset` muestrea día, hora de inicio, `episode_bins` y `vol_care` en cada episodio, como pide LABSTEP (celdas 6 y 8). Hay que **mostrarlo**: varios episodios con duraciones y tamaños distintos.

**Para profundizar:** diseño de recompensas, recompensa al final del episodio y γ según el horizonte, en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md); buffer y estabilidad, en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md); candidatas de recompensa y γ para 400-600 *steps*, en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

*Fuente: ENUNCIADO ej. 2; BE `README.md`, `src/agents/dqn.py` e `images/twap_model/twap_reproduction_dqn.png`; LABSTEP celdas 6 y 8; T07 ~04:02-04:03, ~04:44-04:58.*

---

## 7. Ejercicio 3 (1,5 puntos): exploración que no dependa del número de episodios

### 7.1 Enunciado

- Modificar el sistema de exploración para que **ε no dependa del número de episodios**. La exploración debe bajar **conforme mejora el agente**, es decir, mejora sus resultados o conoce mejor el espacio.
- La duración del entrenamiento **debería** depender de la evolución del aprendizaje, pero en este ejercicio se puede **mantener un número fijo de episodios**.
- **Nota del enunciado:** hay muchas soluciones y lo importante es explicar la lógica. **No es tan importante dar con una solución correcta** como responder a dos preguntas:
  1. **¿Cómo determinamos que el algoritmo ha mejorado** en nuestro caso?
  2. **¿Cómo trasladamos esa mejora a ε?**

### 7.2 Lo que dijo el profesor

- **Qué se toca.** El ε, "para que no sea simplemente proporcional al número de episodios" (T07 ~04:58). En la presentación se transcribió "en base a otro episodio" *(dudoso)*, casi seguro "en base a otro criterio" (T07 ~04:29).
- **Por qué no le gusta el decaimiento por episodios.** Fijar episodios y bajar ε de forma lineal o exponencial "está mal" porque es muy arbitrario: ¿por qué 10 000? La estrategia clásica empieza con ε alto y lo baja **conforme mejora el aprendizaje**. Mejorar tiene dos dimensiones: **mejores resultados** (utilidad, suma de recompensas) y **conocer el espacio** (haber visitado estados y probado acciones) (APP p.64; T07 ~00:35-00:36).
- **Motivación.** Lo enlazó con el taller anterior, donde 10 000 episodios tardaban una hora y se veía evolucionar muy despacio (T07 ~00:34:33-00:34:38).
- **Demostración en *cliff walking*** (el notebook `cliff-walking` de `rl-material`, "de ayer"; se transcribe "clipbooking", T07 ~00:19:56). Con ε decayendo linealmente de 1 a 0,05 hacia el episodio 150 de 300 (RLM cliff-walking: `nepisodes = 300`, `min_epsilon = 0.05`, `epsilon_decay = (1 - min_epsilon) / (nepisodes * 0.5)`), es decir, a una velocidad de $0{,}95/150\approx0{,}0063$ por episodio, la curva de recompensa parecía mejorar justo al ritmo de ε. La volatilidad final se debe a que sigue explorando con ε = 0,05 (T07 ~00:21:01-00:21:56). Evaluado sin ese sesgo, el agente había aprendido hacia el **episodio 30**: con 20 episodios bien evaluados ya se vería que convenía bajar la exploración y terminar (T07 ~00:20-00:22). Análisis completo en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) (§2.3).
- **Cómo medir la mejora.** Con un umbral de recompensa, el *pseudo-conteo* de lo ya visitado (activos, días de subida o bajada) o la cercanía a un *benchmark* (TWAP, VWAP, POV): cuanto más cerca, menos exploración (T07 ~00:36-00:37, ~01:47).
- **Nivel de aspiración.** Para ajustar la exploración hay que conocer el **coste de oportunidad**. Si no se conoce, se fija un objetivo o nivel de aspiración razonable con un *baseline* o *benchmark* (índices de referencia, algoritmos sin ML o soluciones clásicas, objetivos de proyecto, "vecinos"), y los niveles se suben o bajan según se alcancen (APP p.66). **Nota:** aquí el *baseline* natural es el TWAP determinista.
- **Mapa de calor.** Un alumno propuso registrar durante el entrenamiento los estados visitados y verlos como mapa de calor; el profesor lo aceptó como técnica válida y añadió que hay clases enteras solo de exploración (T07 ~00:40:12-00:40:36). Encaja con la dimensión "conocer el espacio".
- **Más allá.** ε puede variar incluso **dentro del episodio**: subirla en zonas poco conocidas y bajarla en las conocidas. Las acciones aleatorias pueden ponderarse por su utilidad esperada o hacia zonas menos visitadas. Recompensas y exploración son "dos mundos" donde ser creativos (T07 ~00:37-00:40).

**Nota:** en este caso el TWAP determinista de la celda 9 da una referencia natural de "mejora", como la distancia entre el acumulado del agente y la diagonal, o la diferencia de VWAP con el TWAP medida en episodios *greedy* periódicos. Opciones, criterios y riesgos están en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) y la propuesta en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

*Fuente: ENUNCIADO ej. 3; APP p.64 y p.66; RLM cliff-walking; T07 ~00:19-00:22, ~00:34-00:40, ~01:47, ~04:29, ~04:58.*

---

## 8. Ejercicio 4 (1,5 puntos): mejora libre

### 8.1 Enunciado

El texto lo llama "ejercicio **opcional**". Su propósito es demostrar conocimiento en la construcción del entorno o en el entrenamiento. Propone tres ideas y deja libertad para otras:

| Idea | Sobre | Qué pide | Advertencias del enunciado |
|---|---|---|---|
| **A** | Ej. 2 | Modificar la recompensa del TWAP para **mejorar sus resultados**: media del VWAP o **menor dispersión**. Exige **añadir al estado** información del libro de órdenes o de la tendencia | No se piden buenos resultados, porque pueden hacer falta más y mejores datos, pero sí un **razonamiento y unos pasos correctos** |
| **B** | Ej. 2 | Añadir recompensas para (1) **no dejar volumen pendiente** y (2) **consumir aproximadamente el 90 % de la duración** de la orden, para dejar margen | Hace falta una **calibración correcta** de las recompensas para que los resultados sean estables |
| **C** | Ej. 3 | **Condiciones de parada** basadas en el aprendizaje, **guardar los pesos** cada cierto tiempo para no perder un buen entrenamiento y poder **recuperarlo** | — |
| Libre | — | Cualquier punto de la clase de creación de entornos y entrenamiento | Hay que **especificar con detalle** la mejora y la lógica |

### 8.2 Lo que dijo el profesor

- **Libertad absoluta** para aplicar otro conocimiento del máster (T07 ~04:29).
- **Ideas que dio:**
  - Un TWAP con **menor o mayor dispersión** *(dudoso)*.
  - Un **objetivo cualquiera**, incluso una recompensa rara, para ver qué comportamiento sale.
  - **No dejar volumen pendiente** y **apurar el tiempo**: no vale quedarse en el 80 % del volumen y conviene terminar "más o menos" en el 90 %, en un rango del **80 al 95 %**. La idea: "te doy un poco de tiempo de más", pero que apure hasta el final. Es un objetivo sencillo que obliga a calibrar (T07 ~04:29-04:30).
  - Exploración o ajustar de forma dinámica el "váter" *(dudoso: casi seguro **el tamaño del buffer**; en T07 ~00:41:27 habla del "tamaño del váter" justo después de un *paper* en el que se aumenta o disminuye, que corresponde al *Dynamical System Model of ER* de APP p.78: más error → buffer mayor, menos error → buffer menor; en T06 ~04:11 habló de *papers* sobre cómo llegar al tamaño ideal del buffer e incluso ajustarlo dinámicamente)* (T07 ~04:31). Ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).
  - En T07 ~00:41 añadió que incluso "el Karaoma" puede ser dinámico y no igual para todos los episodios *(dudoso: lo más probable es γ; como justo antes hablaba del tamaño dinámico del buffer, también podría seguir refiriéndose a él)*. Sobre γ, ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).
- **Formato.** Exponer: "he recogido de clase esto, este fue mi razonamiento y esto es lo que quise implementar" (T07 ~04:31).
- **Al repasarlo de nuevo** dio otro rango, terminar entre el **70 y el 90 %** (T07 ~04:58).

**Nota:** el ENUNCIADO prevalece: **completar el 100 % del volumen** y usar **≈90 % del tiempo**. En la transcripción se oye "90 % del volumen", pero el contexto ("te doy tiempo de más", "apurar hasta el último momento") y el enunciado indican **tiempo**. Los rangos de clase (80-95 % o 70-90 %) sirven para fijar una banda de tolerancia, que hay que justificar (§12). Para la idea C, checkpoints y *early stopping* están en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md). Para la idea A, características de estado candidatas (libro, desbalanceo, tendencia) en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

*Fuente: ENUNCIADO ej. 4; APP p.78; T06 ~04:11; T07 ~00:41, ~04:29-04:31, ~04:58.*

---

## 9. Datos y ficheros del taller

**Idea.** Basta un activo, REP o SAN, y los *splits* se pueden juntar. Cada día es un DataFrame con fotos de 5 s del primer nivel del libro. La estructura detallada y sus trampas están en [08_entorno_best_execution.md](08_entorno_best_execution.md).

| Fichero | Contenido | Días | Fuente |
|---|---|---|---|
| `data/rep_data.pkl`, `data/san_data.pkl` (y sus copias en `data/new_version/`) | `dict` con claves `train`, `val`, `test` → `dict` por día → DataFrame cada 5 s, de 13:00 a 17:25, con `ask1`, `bid1`, `vask1`, `vbid1`, `last`, `cumvol` y `date` | Train: 60 primeros días hábiles de 2018. Val: 30 primeros de 2019. Test: del 31 al 60 de 2019 | LAB celda 2; T07 ~04:39-04:43; DATOS (columnas y copias en `new_version/`) |
| `data/new_version/orderbook.pkl` | Hasta el **3.er nivel** del libro, cada 5 s, **Repsol** | Train: 40 primeros días de 2019. Val: 41-60. Test: 61-80 | LAB celda 2; DATOS |
| `data/new_version/executions.pkl` | **Todas las ejecuciones** de mercado (columnas `date`, `sec`, `vol`, `price`, `nivel`, `bidask` con valores A/B), Repsol | Ídem | LAB celda 2; T07 ~04:44; DATOS (columnas) |

Lo que dijo el profesor:
- Hay **pickles antiguos y nuevos** con los mismos datos, según la versión de pandas; los nuevos "deberían funcionar para todos" (T07 ~04:38-04:39).
- Basta con **uno de los dos activos**, y train, val y test **se pueden juntar** (T07 ~04:39).
- Como la sesión empieza a las 13:00, el volumen acumulado **no empieza en 0** (T07 ~04:43).
- `orderbook` y `executions` **no hacen falta en absoluto**; son para quien quiera más detalle. No recordaba si eran de Santander o de Repsol: el LAB dice **Repsol** (T07 ~04:43-04:44; LAB celda 2).

**Nota:** la sesión de 13:00 a 17:25 son 265 minutos, ≈3.180 fotos por día. El *lookback* de 60 *bins* son 5 minutos, así que un episodio no puede empezar antes de las 13:05 (LABSTEP celda 8 descarta los inicios con `initial_position < 0`). Los valores de los pickles no se han cargado en esta síntesis: deserializar pickles de terceros ejecuta código.

---

## 10. Entorno de trabajo

### 10.1 Lo que hay en el repositorio del taller

| Vía | Cómo | Versiones | Fuente |
|---|---|---|---|
| **Dev Container** (necesita Docker) | Abrir el repo en VS Code y *Dev Containers: Reopen in Container*. `docker-compose` monta el repo en `/workspace` con `PYTHONPATH=/workspace/src:/workspace`; puerto 8080; usuario `vscode` | Imagen publicada `jesussanz/best_execution:v1`. Respaldo comentado en el `dockerfile`: `devcontainers/python:3.8` + `requirements.txt` (TF 2.5.0, numpy 1.19.2, pandas 1.4.0, matplotlib 3.6.1, jupyterlab 4.1.3) + `xvfb`, `python3-opengl`, `swig` | BE README, `.devcontainer/`, `dockerfiles/` |
| **Local** (sin Docker) | Entorno virtual de **Python 3.9-3.11** (3.11 recomendado) y `pip install -e .` desde la raíz: `python3.11 -m venv .venv` (Windows: `py -3.11 -m venv .venv`), activar (`source .venv/bin/activate`; Windows: `.venv\Scripts\activate`) y `pip install -e .`. Hay *wheels* precompiladas para Windows, macOS Intel, macOS 12+ en Apple Silicon y Linux, así que no hace falta compilador. Así `src/` se importa desde cualquier notebook (`from src.agents.dqn import DDQNAgent`) | `setup.py`: `python_requires >=3.9.2,<3.12` (TF 2.15 solo publica *wheels* para 3.9-3.11, y `cryptography` excluye 3.9.0 y 3.9.1); **tensorflow 2.15.1** (último con Keras 2, el API de `src/agents/dqn.py`); numpy 1.26.4 (TF 2.15 exige numpy < 2); **pandas 2.0.3** (los `data/*.pkl` se guardaron con pandas 1.x y `pickle.load` falla desde pandas 2.1); matplotlib 3.9.4; jupyterlab 4.5.11; además fija las dependencias transitivas `argon2-cffi-bindings==25.1.0` y `cryptography==48.0.1`, cuyas versiones nuevas dejaron de publicar *wheels* para macOS Intel | BE README ("Installation"), `setup.py` |
| Último recurso | Descomentar `sys.path.append("..")` en la celda de imports si fallan `pip install -e .` y el contenedor | — | LAB celda 3 |

### 10.2 Lo que dijo el profesor

- **T06 (sobre rl-material, mismo esquema).** Usa Docker para que el material funcione de un año a otro: hay imágenes publicadas y requisitos para construirlas, con la extensión **Dev Containers** de VS Code. "Funciona al 90 %", siempre hay un 10 % de alumnos a los que falla. La alternativa es un entorno con `pip install -e .`, que evita el `sys.path.append`. A él no le dejaba instalar TensorFlow 2.15 en su portátil nuevo por seguridad *(dudoso)*. La primera construcción tarda ~10 min por la imagen de TensorFlow (~2 GB). Para ver *renders* hay que lanzar Jupyter Lab con un comando que crea una pantalla virtual. El material usa `gym`, no `gymnasium` (T06 ~01:00-01:06).
- **T07, al empezar.** Hace falta tener **Docker** instalado; él usa el *backend* WSL/Linux, la configuración por defecto en Windows. Explicó qué es un contenedor (aísla dependencias y es más ligero que una máquina virtual), y un alumno añadió que da una copia exacta en todos los equipos y es un entregable natural para *cloud*. El `dockerfile` parte de una imagen publicada, con otro de respaldo en Python 3.8, "ya ha pasado un tiempo". Para un notebook sencillo no hace falta; para lo demás, sí (T07 ~00:03-00:08).
- **T07, en la práctica.** **No hace falta el contenedor:** otros alumnos lo hicieron sin él. Basta "Copa y Tor" *(dudoso: quizá PyTorch, aunque también podría ser Python o TensorFlow)* "y un lumpipa más" *(dudoso: ¿numpy o pandas?)*, porque solo hay que leer un pickle con datos de bolsa (T07 ~04:31:53-04:32:03). Si falla la lectura del pickle, usar el *devcontainer* o ajustar el entorno *(dudoso)* (T07 ~04:35).
- **Eficiencia.** El notebook trabaja con pandas para facilitar, pero en RL el entorno se recorre paso a paso y pandas es lento. En su demo con `%timeit`, un diccionario era órdenes de magnitud más rápido (~45 ns por acceso). Reconstruir el libro de un día de Santander pasó de 20-30 min con pandas a 2-3 s con diccionarios. Si el problema es sencillo, vale pandas (T07 ~00:10-00:18). Detalle en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

**Nota:**
- El ENUNCIADO permite **PyTorch** y otras versiones (ej. 1, "no es necesario"). El `requirements.txt` del respaldo Docker (TF 2.5, Python 3.8) no coincide con `setup.py` (TF 2.15.1, Python ≤ 3.11), y no se sabe qué lleva la imagen `v1`.
- La celda 4 del LAB lee `../data/rep_data.pkl`, la versión antigua. Con pandas ≥ 2.1 hay que usar `../data/new_version/`, o bien pandas 2.0.3 como fija `setup.py`. No verificado aquí.
- Si el equipo trabaja con Python > 3.11, TF 2.15 no se instala, y la vía natural es reescribir el agente en PyTorch. Pros y contras en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

---

## 11. Logística: grupos, correo, exposición

| Tema | Qué se dijo | Fuente |
|---|---|---|
| Formato | En T06 propuso dos opciones: hacerla **en directo** en el taller y entregarla, o tenerla **durante la semana** con una exposición. En directo hay más riesgo de que salga mal, aunque también más tiempo para corregir *(dudoso)*. Si enseñaba el ejercicio ya, quedaría grabado | T06 ~04:52-04:57 |
| Decisión | En T07 presentó el enunciado y fijó la exposición el **18**: se hace durante la semana y se expone ese día | T07 ~04:27, ~05:06 |
| Dificultad | En T06 dijo que se podría elegir si se quería más difícil o más cómodo *(dudoso)*. En T07 lo concretó en ejercicios con puntos y en la libertad del ej. 4 | T06 ~02:57; T07 ~04:29 |
| Grupos | Los del **TFM** | T06 ~04:54 |
| Qué enviar | Por **correo**, al del **trabajo** mejor que al personal, porque lo mira más a menudo: **nombre del grupo** (en T06, sus integrantes) y **preferencia de hora**, más temprano o más tarde. Su correo está en las diapositivas, la primera o la última (T06 ~04:58:56; APP p.2). En T06 aceptó también comunicarlo en persona, con un papel en clase, o por el chat de Zoom | T06 ~04:55-04:59; T07 ~05:05 |
| Orden | Por **orden de llegada** de los correos. La lista se publica el fin de semana en **OneDrive** (en T06 dijo Drive). Quien no esté en la lista va al final. Con el nombre del grupo y la preferencia (más temprano o más tarde), él asigna y comunica la hora a cada grupo; da la fecha (el 18) para que los grupos se organicen | T06 ~04:55; T07 ~05:05:54-05:06:21 |
| Seguimiento | Se podrá seguir por **Zoom**; no hace falta estar las 2-3 h, se puede conectar solo a la hora asignada | T06 ~04:55-04:56 |
| Duración | Unos **10-15 minutos** por grupo. En T06 había hablado de "5 minutos de presentación" | T07 ~05:06; T06 ~04:54 |
| Diapositivas | "No hace falta positivas. No hace falta, simplemente, como 10 min" *(dudoso; probablemente "no hacen falta diapositivas")*. Justo antes, "la exposición 18 en el restaurante al horno" *(dudoso: ininteligible; quizá el horario o el lugar de la exposición)* | T07 ~05:06:27-05:06:44 |
| Grabación | Las clases se graban. Al entrar en el detalle del llenado del buffer recomendó volver al vídeo cuando toque implementarlo | T06 ~04:19:51-04:19:57, ~04:31:23 |
| Material | Iba a subir el material al Drive y poner su correo para dudas: no quiere que nadie se atasque con un concepto tonto | T07 ~05:07 |
| Versión del enunciado | Lo presentó como "la copia original": si los alumnos daban *feedback*, se cambiaría. Iba a subirlo al Drive en ese momento | T07 ~04:27:55, ~05:07:14 |
| Agenda del taller | Día 1: repaso de RL, entornos, factores de diseño. Día 2: conceptos pendientes, *profit seeking*, algoritmos de ejecución. **Día 3: "Taller: exposición de los casos resueltos como tarea"** | APP p.3 |

**Descripción de APP p.3:** la agenda en tres bloques ("Día 1", "Día 2", "Día 3") con el icono de un aula virtual junto al Día 2 y al Día 3. La exposición del día 18 es el **Día 3**.

**Nota:**
- T06 es del viernes 6/11/2026 y T07 del sábado 7/11/2026. El "18" debería ser el **miércoles 18/11/2026**, aunque la transcripción no dice el mes.
- El ENUNCIADO **no da fecha ni canal de entrega** del notebook. Lo prudente es tenerlo todo listo **antes de la exposición** y preguntar en el correo de grupo cómo y cuándo se entrega.
- No consta qué "fin de semana" se publica la lista. Primero dijo que podía decirlo "el mismo fin de semana", luego "el próximo fin de semana" por correo, y se corrigió: el fin de semana dejará la lista en OneDrive (T07 ~05:06:09-05:06:16). Lo que corrige es el canal (correo → OneDrive); no queda claro si es el del 14-15/11. Hay que vigilar OneDrive y el correo.

---

## 12. Ambigüedades y contradicciones

| # | Tema | Enunciado | Clase | Qué prevalece / cómo resolverlo |
|---|---|---|---|---|
| 1 | **Puntos del ej. 4** | **1,5** (suma 10) | Tras "3 puntos y medio", "tres puntos y medio" y "un punto y medio", dijo "Y el otro punto es libertad absoluta" *(dudoso: puede ser "el otro apartado" o "1 punto"; si fuera 1 punto, la suma daría 9,5)* (T07 ~04:28:47-04:29:21) | **ENUNCIADO: 1,5.** Solo con 1,5 la suma da 10 |
| 2 | Carácter del ej. 4 | Lo llama "opcional", pero puntúa 1,5 | "Libertad absoluta" | Hacerlo: son 1,5 puntos de 10 |
| 3 | Idea B: qué es el 90 % | "Consuma aproximadamente un 90 % de la **duración**", sin dejar volumen pendiente | "90 % del volumen" *(transcrito)*, rango 80-95 % (T07 ~04:30); luego 70-90 % (T07 ~04:58) | **Tiempo ≈90 % con el volumen completo.** Elegir y justificar una banda; 80-95 % encaja con "aproximadamente 90 %" |
| 4 | Ej. 3: criterio | No depender del número de episodios; bajar ε según la mejora | "En base a otro episodio" *(transcrito)* | ENUNCIADO; es casi seguro "otro criterio" |
| 5 | Reutilizar el notebook | Se puede hacer desde cero con **más documentación**, pensado para ejecución y con los archivos del taller | "Podéis tirar este notebook por completo" (T07 ~04:55) | Compatibles; el ENUNCIADO añade la obligación de documentar más |
| 6 | Algoritmo | No es necesario reutilizar el de Deep RL | Vale Q-learning, incluso tabular 2×2 (T07 ~04:55, ~05:01) | Compatibles. **Nota:** un tabular 2×2 cumple, pero enseña poco de "parametría" (buffer, γ) para la segunda parte del ej. 2 |
| 7 | Datos | "Cualquiera de los **3 ficheros**" | REP o SAN; `orderbook` y `executions` "no hacen falta" (T07 ~04:39-04:44) | Ambiguo: probablemente `rep_data`, `san_data` y `orderbook` (Repsol, 3 niveles), ya que `executions` no es una foto del libro. En la práctica, **REP o SAN** |
| 8 | Validación | No hace falta: imágenes correctas en **train** | Train, val y test se pueden juntar (T07 ~04:39) | Compatibles. **Nota:** la celda 20 del LAB grafica sobre `dict_["test"]`; si se juntan los *splits*, decirlo |
| 9 | Duración del episodio | Debe funcionar con distintas duraciones (ej. 2) | Asunción de 90 min - 2 h (T07 ~04:44), pero 400-600 *steps* = 33-50 min por defecto (T07 ~04:46) | 400-600 *steps* del LAB/LABSTEP; alargar es opcional |
| 10 | Tamaño de la CARE | 40 títulos para 400 *steps* (10 %) | 720 × "0,9" = 65 (T07 ~04:47); "50-60 títulos" (T07 ~05:00); ejemplo de 55 | LABSTEP: 7,5-12,5 % de los *steps* (30-75 títulos). "0,9" debe ser 0,09 (§6.3) |
| 11 | Motivar la parametría en el ej. 1 | "Documentar / motivar decisiones" es **necesario**, pero motivar en detalle la parametría **no** | — | Motivar la estructura (entorno, orden de llamadas, algoritmo). Los hiperparámetros, en el ej. 2 |
| 12 | Duración de la exposición | No lo dice | 5 min (T06 ~04:54) → **10-15 min** (T07 ~05:06) | Lo último: 10-15 min |
| 13 | Diapositivas | Exposición y motivación son lo principal; memoria aparte "recomendable" | "No hace falta positivas" *(dudoso)* | No son obligatorias. **Nota:** algo visual (gráficos del notebook o 3-4 diapositivas) ayuda a exponer en 10-15 min |
| 14 | Canal y plazo de entrega | No los da | Solo la exposición del 18 | Preguntarlo por correo (§11) |
| 15 | Precio de ejecución | No dice nada; sin impacto ni costes | El notebook usa un precio "tonto" y **sesgado**, el `ask1` al inicio del *bin* con latencia 0. Mejor aproximar lo que ocurre entre fotos: media de medios o interpolación según la latencia (p. ej. 2 s). Son las asunciones que hay que razonar (T07 ~05:01-05:05). Ejemplo: si el mercado negocia varias decenas de títulos entre dos fotos de 5 s, no se sabe a qué precio (T07 ~05:03:14-05:03:39). El precio exacto se tendría con las ejecuciones (`executions.pkl`), pero no hace falta usarlo *(dudoso: se transcribe "el Notebook de el precio exacto")* (T07 ~05:03:51). Dijo también que en muchos algoritmos se piden "2 precios", uno más o menos realista *(dudoso)* (T07 ~05:02:00-05:02:16) | No es obligatorio cambiarlo. **Nota:** al menos hay que **reconocer el sesgo** y justificar la elección; ver [08_entorno_best_execution.md](08_entorno_best_execution.md) |
| 16 | Entorno de software | Libertad de paquetes, PyTorch incluido | Sin contenedor; "Copa y Tor" *(dudoso: PyTorch, Python o TensorFlow)* (T07 ~04:31) | Compatibles (§10) |
| 17 | Peso de la práctica | No lo dice | "3 puntos" de 10, o de los "3 primeros" *(dudoso)* (T07 ~05:09) | Desconocido; preguntarlo si importa |
| 18 | TWAP determinista (celda 9) | El ej. 1 pide completar el notebook, que lo incluye como TODO, pero no lo nombra | "Esto es importante. Esto no es necesario" *(dudoso: puede referirse a que el TWAP determinista no es obligatorio o a que no basta para el ej. 2, que lo quiere por recompensas)* (T07 ~04:53:18-04:53:24) | **Nota:** implementarlo; es barato y sirve de *benchmark* para los ej. 2 y 3 |
| 19 | Versión del enunciado | Texto fijo | "Esta es la copia original. Si me dais *feedback*, la cambiamos" (T07 ~04:27:55) | Comprobar en Drive/OneDrive que se trabaja con la última versión |

---

## 13. Lo que dijo el profesor fuera de la presentación y conviene saber

**Idea.** Contexto que explica el enfoque del taller y sus consejos.

| Tema | Qué dijo (paráfrasis) | Hora |
|---|---|---|
| Quién es | Jesús Sanz del Real, CFA (APP p.2). Alumno de la primera edición del máster, *senior manager* en una consultora (sobre todo banca). Antes estuvo en el Instituto BME haciendo algoritmos de inversión, un *robo-advisor* y el caso de ejecución del taller. Perfil mixto técnico-financiero, frente a Luis Lago, "el experto técnico" | T06 ~00:02-00:04 |
| Plan de los días | Día 1, repaso de RL y cómo definir cada componente en la práctica (p. ej. γ 0,5 o 0,9, recompensas). Día 2, dos casos financieros: inversión y, sobre todo, ejecución. El taller sería principalmente el caso de ejecución; quería ajustar la carga con los alumnos porque era su primera edición en formato taller | T06 ~00:04-00:06 |
| Origen del caso | Su TFM de 2019: BME tenía algoritmos de ejecución basados en reglas y quería meter *machine learning*. Lanzaba 72 experimentos en paralelo con recompensas y configuraciones distintas en una máquina de ~700 GB; por la mañana leía *papers* y diseñaba recompensas, por la noche entrenaba. No llegó a producción porque BME perdía cuota frente a grandes operadores y a los *brokers* les daban igual 2-3 pb, porque el dinero es del cliente | T06 ~02:57-03:00 |
| Material de apoyo | `rl-material` (Drive): algoritmos para usar; no se dedica tiempo de clase, dudas en descansos o por correo | T06 ~01:00 |
| Notebooks clave | Los dos más importantes de `rl-material` son "TD-q-learning" y "deep", base del resto; el README explica cómo ejecutarlos | APP p.1 |
| Notebook de *profit seeking* | Lo hizo "una tarde", sin hacerse responsable de que esté bien calculado. Usaba una API de datos de BME ya inactiva; el estado eran Sharpes de "15, 30, 60 días o algo así". Sirve para ver los componentes y contrastarlos con la práctica. **Nota:** el notebook usa ventanas de **10**, 30 y 60 (`for win in [10, 30, 60]` con `median_sharpe`, RLM profit-seeking) | T07 ~00:42-00:44 |
| No todo es IA | Elegir en qué pasos del problema usar RL, supervisado o reglas. Por ejemplo, un modelo de probabilidad de ejecución cuyas salidas entran en el estado. No todas las decisiones tienen que salir del agente | T07 ~00:23-00:26 |
| Combinar algoritmos | Ningún algoritmo de RL predomina. Se pueden combinar ideas, como un *Dueling Double DQN* al estilo Rainbow | T07 ~00:25 |
| Entrenar | Empezar sencillo y buscar incluso el sobreajuste inicial. Evaluar la política aparte de la exploración. Guardar registros y puntos de recuperación por si hay *catastrophic forgetting* o el agente se atasca | T07 ~00:18-00:19 |
| Volumen real de algoritmos | Cuando trabajaba en BME ("cuando yo estaba haciendo todo esto"), TWAP ~5 %, VWAP ~15 % y POV ~80 % de las órdenes recibidas. El POV cuesta más en media (~7 pb), pero con poca dispersión, y el cliente prefiere resultados estables; TWAP y VWAP cuestan ~3 pb de media, pero a veces 20 | T07 ~01:58-02:01 |

Más detalle: [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md) (TWAP/VWAP/POV, costes), [05_profit_seeking.md](05_profit_seeking.md) (notebook de inversión), [09_material_rl_notebooks.md](09_material_rl_notebooks.md) (`rl-material`) y [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) (evaluación, checkpoints).

---

## 14. Mapa de documentos y orden de lectura

| Documento | Contenido | Para qué ejercicio |
|---|---|---|
| [00_repaso_rl.md](00_repaso_rl.md) | Componentes, Markov, retorno, Q-learning, SARSA, variantes | Vocabulario; versión tabular del ej. 2 |
| [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md) | Aproximación funcional, DQN, *replay* (ER, PER, CER), *target*, Double y Dueling DQN | Ej. 1 (el `DDQNAgent`) |
| [02_diseno_de_entornos.md](02_diseno_de_entornos.md) | API, *steps*, estado, acciones, episodios, eficiencia, *curriculum* | Ej. 1 |
| [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md) | Diseño de recompensas y calibración de γ | **Ej. 2**, ej. 4 (idea B) |
| [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) | ε por mejora, buffer, *batch*, checkpoints, *early stopping* | **Ej. 3**, ej. 2 (parte 2), ej. 4 (idea C) |
| [05_profit_seeking.md](05_profit_seeking.md) | Caso de inversión | Contexto |
| [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md) | Negocio: CARE, TWAP, VWAP, POV, *slippage* | Ej. 2 y exposición |
| [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md) | El caso de ejecución como problema de RL | Ej. 2 y 4 (idea A) |
| [08_entorno_best_execution.md](08_entorno_best_execution.md) | Datos, `BestExecutionEnv` función a función, `DDQNAgent`, erratas | **Ej. 1** |
| [09_material_rl_notebooks.md](09_material_rl_notebooks.md) | Los 18 notebooks de `rl-material` | Código de referencia |
| [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md) | Propuesta razonada ejercicio a ejercicio | Todos |
| [11_glosario.md](11_glosario.md) | Términos técnicos y de negocio | Consulta |

**Orden rápido (para programar ya):** este documento → [08](08_entorno_best_execution.md) → [10](10_hoja_de_ruta_practica.md) → [03](03_recompensas_y_descuento.md) → [04](04_exploracion_memoria_y_estabilidad.md).

**Orden completo (antes de la exposición):** 00 → 01 → 02 → 03 → 04 → 06 → 07 → 08 → 10, y 05, 09 y 11 como consulta.

---

## 15. Checklist de entrega

| Bloque | Comprobación | Fuente |
|---|---|---|
| Logística | Correo enviado al **correo del trabajo** del profesor con nombre del grupo (integrantes) y preferencia de hora | T06 ~04:55-04:59; T07 ~05:05 |
| Logística | Comprobado en Drive/OneDrive que el ENUNCIADO usado es la **última versión** | T07 ~04:27:55, ~05:07 |
| Logística | Hora asignada consultada en **OneDrive**; enlace de Zoom a mano | T07 ~05:06; T06 ~04:55 |
| Logística | Preguntado **cómo y cuándo se entrega** el notebook (no lo dice el enunciado) | §11 |
| General | Notebook(s) **ejecutado(s)**, con salidas visibles | ENUNCIADO |
| General | En cada ejercicio, **bloques modificados señalados** y **lógica explicada** | ENUNCIADO |
| General | Memoria breve o `DECISIONES.md` con el resumen de decisiones (recomendable) | ENUNCIADO |
| General | Activo elegido (REP o SAN) y *split* usado, dichos y justificados | ENUNCIADO; T07 ~04:39 |
| Ej. 1 | Entorno con todos los elementos (`reset`, `step`, observación, recompensa, *done*, acciones) en el **orden correcto** | ENUNCIADO ej. 1 |
| Ej. 1 | Llenado del buffer → entrenamiento → evaluación → `plot_results`, sin errores | LAB celdas 16-20 |
| Ej. 1 | ε inicial = 1 (o justificado) antes del llenado del buffer; el paso 0 forzado a comprar, tenido en cuenta | LAB celda 16; BE `dqn.py` |
| Ej. 1-3 | Curva de recompensa de entrenamiento (celda 21) incluida y comentada | LAB celda 21 |
| Ej. 1 | TWAP determinista (celda 9) implementado y graficado como *benchmark* (**Nota**, recomendado: §12, fila 18) | LAB celdas 8-9 |
| Ej. 1 | Si hay cambio de framework (PyTorch) o de algoritmo, explicado; si es un entorno propio, **documentación extra** de la estructura | ENUNCIADO ej. 1 |
| Ej. 2 | **Recompensa** que produce la diagonal del acumulado y barras regulares, como en `twap_reproduction_dqn.png` | ENUNCIADO ej. 2 |
| Ej. 2 | **Estado** descrito y justificado (tiempo y volumen restantes, como mínimo) | ENUNCIADO; T07 ~04:52 |
| Ej. 2 | Gráficos en **varios episodios**: distintos días, horas, duraciones (400-600) y tamaños (30-75 títulos) | ENUNCIADO; LABSTEP celda 6 |
| Ej. 2 | Imágenes de evaluación sobre **train** (cambiar `dict_["test"]` en la celda 20 o justificar el *split*) | ENUNCIADO ej. 2; LAB celda 20 |
| Ej. 2 | **Parametría motivada**: γ según el horizonte, buffer, *batch*, α, episodios | ENUNCIADO ej. 2, parte 2 |
| Ej. 2 | Hijas de 1 título, sin impacto ni costes; precio de ejecución y su sesgo explicados | ENUNCIADO; T07 ~05:01-05:05 |
| Ej. 3 | ε **independiente del número de episodios** | ENUNCIADO ej. 3 |
| Ej. 3 | Respuestas explícitas a **¿cómo medimos la mejora?** y **¿cómo se traslada a ε?** | ENUNCIADO ej. 3 |
| Ej. 3 | Gráfico de ε junto a la métrica de mejora a lo largo del entrenamiento | **Nota** (propuesta) |
| Ej. 4 | Mejora elegida (A, B, C u otra), con **qué se recogió de clase, el razonamiento y lo implementado** | ENUNCIADO; T07 ~04:31 |
| Ej. 4 | Si es la idea B: volumen completo, ≈90 % del tiempo con banda justificada y calibración estable | ENUNCIADO; T07 ~04:30 |
| Exposición | Guion de **10-15 min**: problema → entorno → ejercicios → decisiones y evidencias → límites | T07 ~05:06 |
| Exposición | Cada miembro puede justificar cualquier decisión: "decisiones arbitrarias, fatal" | T06 ~04:10 |
