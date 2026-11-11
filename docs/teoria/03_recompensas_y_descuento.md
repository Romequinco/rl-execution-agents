# Recompensas y factor de descuento

> **Fuentes:**
> - **APP** (`docs/raw/MIAX14/applying_RL_MIAX.pdf`, "Applying Reinforcement Learning"): p.16-18 (objetivo del RL, utilidad, $\gamma$), p.58-62 (recompensas), p.66 (nivel de aspiración, baselines), p.72 y p.80 (recompensas como técnica de exploración), p.75-76 (gamma), p.85 (fin de episodio y recompensa final), p.96 (el algoritmo hace lo que sea por maximizar), p.99-100 (ejemplo del paraguas).
> - **CASOS** (`docs/raw/MIAX14/casos_practicos.pdf`, "Casos Prácticos"): p.6-11 (objetivos y recompensas en profit-seeking; p.10, "¿Gamma?"), p.42-44 (recompensas del algoritmo de ejecución), p.53 (perfiles conservador/moderado/agresivo).
> - **T06** (`docs/raw/26 11 06.pdf`, transcripción automática de la clase del 06/11/2026): ~00:18-00:19 (paraguas), ~00:38-00:44 (utilidad y $\gamma$), ~00:57-00:59 (recompensas como escollo), ~01:14 (el $-100$ del acantilado), ~02:29-02:57 (bloque de recompensas), ~03:22-03:24 (objetivos con números), ~03:31-03:32 (recompensas para explorar), ~03:42-03:54 (calibración de $\gamma$), ~04:33-04:35 (Montezuma).
> - **T07** (`docs/raw/26 11 07.pdf`, transcripción del 07/11/2026): ~00:31-00:33 (recompensa como proxy), ~00:40-00:41 (escollos; $\gamma$ dinámico), ~01:07-01:10 ($\gamma$ en episodios intradía y largos), ~01:19-01:21 y ~01:32-01:35 (recompensas en profit-seeking), ~01:44-01:47 (objetivo del caso de ejecución), ~03:30-03:44 (recompensas del caso de ejecución), ~03:52-03:56 (perfiles, reglas y límite del simulador), ~04:11-04:12 (utilidad esperada de la pasiva), ~04:29-04:30 (ideas del ej. 4), ~04:51-05:00 (pistas sobre la recompensa en la práctica).
> - **LAB** (`docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`): celda 7 (`_compute_reward`, `_compute_done_reward`, `_agg_action`, `_do_nothing`, `_compute_episode_market_feat`, `step`), celdas 11-12 (hiperparámetros, $\gamma$), 18, 20 y 21 (retorno en entrenamiento y evaluación, log y curva). **LABSTEP** (`.../execution_lab-step_by_step.ipynb`): celdas 15-17 y 27-28.
> - **BE** (`docs/raw/MIAX14/best-execution/src/agents/dqn.py`): objetivo de Double DQN y tratamiento del estado terminal.
> - **RLM** (`docs/raw/MIAX14/rl-material/notebooks/`): `cliff-walking.ipynb` (celdas 9, 16, 22), `profit-seeking.ipynb` (celdas 3 y 9), `rl-models-basic/TD-q-learning.ipynb` (celda 7), `rl-models-basic/deep.ipynb` (celda 3).
> - **ENUNCIADO** (`docs/raw/MIAX14/ejercicios_taller_rl.docx`): ejercicios 1, 2 y 4.
> - **T2** (`docs/raw/referencias/deep-rl-trading-agents/docs/`): `teoria/00_fundamentos_mdp.md` §1 y §4, `teoria/08_trading_en_clase.md` §4-5 y §8, `DECISIONES.md` D-1.1.
>
> **Convenciones:** "(T06 ~02:36)" = hora aproximada en la transcripción; todo lo que viene de las transcripciones es paráfrasis (son automáticas y muy ruidosas) y lo que no se entiende bien va marcado *(dudoso)* con sus lecturas posibles. "(APP p.60)" = página de las diapositivas; "(LAB celda 7)" = celda del notebook, numerada desde 0. **Nota:** = contenido propio (cálculos, correcciones, conexiones, propuestas). Notación: $r_t$ recompensa, $G_t$ retorno descontado, $\gamma$ factor de descuento, $N$ número de steps del episodio, $H=1/(1-\gamma)$ horizonte efectivo. Aquí solo se tratan recompensas y $\gamma$; exploración, $\alpha$, buffer y entrenamiento están en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).
>
> Escrito el 10/11/2026.

---

## 1. Objetivo, utilidad y retorno descontado

**Idea.** En RL no hay etiquetas: hay un objetivo, maximizar la utilidad (T06 ~02:29), y la utilidad es, por definición, la suma de recompensas descontadas a lo largo de la partida o del episodio (T06 ~00:39; APP p.18). Recompensas y $\gamma$ son las dos piezas con las que se le "explica" al agente ese objetivo.

**La pregunta de la diapositiva** (APP p.16-17). "¿Cuál es el objetivo de un sistema de RL?" con cuatro opciones: maximizar la utilidad total, maximizar la suma de recompensas descontadas, minimizar el coste de oportunidad, ninguna. En p.17 las tres primeras aparecen marcadas con un tic verde: las tres son correctas. El profesor explicó que la tercera es útil como cambio de punto de vista: en lugar de intentar optimizarlo todo, comparar alternativas y minimizar el coste de oportunidad "limpia" el problema y centra la atención en lo importante (T06 ~00:38-00:39). Añadió una frase sobre que "si lo mantienes al 0 es exactamente lo mismo" *(dudoso: lo más probable es que se refiera a que un coste de oportunidad nulo equivale a haber alcanzado la utilidad máxima, es decir, que minimizar uno y maximizar la otra son equivalentes)*.

**La fórmula** (APP p.18). La diapositiva define la utilidad total de la trayectoria (episodio) y su forma recursiva:

$$G(\tau)=r_0+\gamma r_1+\gamma^2 r_2+\dots=\sum_{t=0}^{k}\gamma^t r_t,\qquad G_t=r_t+\gamma\,G_{t+1}$$

con el rótulo "Recompensa inmediata + Valor futuro" y "PV de las recompensas futuras o PV $S_{t+1}$"; $\gamma$ es el "factor de descuento (valor del tiempo)" y la diapositiva deja la pregunta "¿$\gamma=0$ / $\gamma=1$?". A la derecha hay un icono de cronómetro.

- **Nota:** errata menor en p.18: el sumatorio llega hasta $k$ pero el último término escrito es $\gamma^t r_t$; debe ser $\gamma^k r_k$ (el índice de suma es $t$ y el límite $k$, la duración del episodio).
- **Nota:** respuesta a la pregunta: $\gamma=0$ es un agente miope que solo mira la recompensa inmediata; $\gamma=1$ no descuenta y solo es válido si los episodios son finitos (ver §11). Desarrollado en [T2 00_fundamentos_mdp.md §4](../raw/referencias/deep-rl-trading-agents/docs/teoria/00_fundamentos_mdp.md).

**Lo que dijo en clase** (T06 ~00:39-00:44):
- "Descontadas" quiere decir valor temporal del dinero, o de lo que sea el problema: no es lo mismo tener el dinero hoy que dentro de un año, y es un valor que tú le das al problema.
- La recompensa inmediata no se descuenta, porque se recibe en cuanto se actúa; la siguiente sí, y cuanto más alejada en el tiempo, mayor descuento. Cuánto depende de la dimensión temporal del step (el siguiente segundo o el próximo año) y de la urgencia de la operación.
- La fórmula larga no hace falta aprenderla: basta la recursiva. El primer término lo da el entorno; el segundo son creencias, muy imprecisas al principio, que convergen poco a poco jugando (es el *bootstrapping* de TD; ver [00_repaso_rl.md](00_repaso_rl.md)).

**Dinero no es utilidad.** En profit-seeking la diapositiva insiste en que el objetivo y las recompensas deben ir a maximizar tu utilidad, no el dinero, con el ejemplo $\text{Utility}=E[R]-0{,}5\,\sigma^2 A$ ($A$, aversión al riesgo) (CASOS p.6; T07 ~00:54). Los objetivos posibles (retorno absoluto o relativo, Sharpe, IR, TE, drawdown, combinaciones con pesos, restricciones) y el argumento de que un objetivo relativo es "menos interesante pero más fácil" (Repsol sube un 5 %: 70 % economía, 25 % sector, 5 % específico) están en [05_profit_seeking.md](05_profit_seeking.md) (CASOS p.7-9).

*Fuente: APP p.16-18; CASOS p.6-9; T06 ~00:38-00:44, ~02:29; T07 ~00:54.*

---

## 2. Lo primero: definir el objetivo

**Idea.** "Lo primero y más importante → definir tu objetivo" (APP p.58). Si el objetivo está bien aterrizado, las recompensas salen casi como consecuencia (T06 ~02:30).

**Diapositiva** (APP p.58, con un icono de diana y una paleta de pintor). Elementos clave para un buen sistema de recompensas:
- Aterrizar la idea: ¿ganar dinero en bolsa?, ¿ejecutar mi orden?
- Conocer y ponderar los *trade-off*.
- El objetivo debe ser **concreto, objetivo, factible y no abstracto**.

**Un objetivo concreto lleva números** (T06 ~03:22-03:24; APP p.66). Hay que saber a qué se aspira para medir "cómo de lejos estoy", no solo si se ha subido o bajado: baselines o benchmarks (índices de referencia como el Ibex 35, algoritmos sin ML, soluciones clásicas) o un umbral de viabilidad del proyecto (por ejemplo, si no se rascan 20 pbs no se pone en producción). Se empieza por objetivos sencillos (de nuevo el triciclo) y se suben si se alcanzan. APP p.66 lo llama "objetivo o nivel de aspiración razonable" y lo plantea para ajustar la exploración; ese uso está en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

**Lo que dijo** (T06 ~02:30-02:33):
- "Ganar dinero en bolsa" no es un objetivo: depende del riesgo, del horizonte y de la persona. Hay que aterrizar exactamente qué se busca, porque es con respecto a eso con lo que se va a medir. A veces el objetivo es complejo y necesita *proxies*.
- Tiene que poder expresarse con una fórmula matemática. Los objetivos abstractos (creatividad) son más difíciles de evaluar, aunque hoy, con los LLM, ya se evalúan con personas; por eso dice que cambiará esta diapositiva en la siguiente edición.
- **Factibilidad = ¿alguien lo hace ya?** Ganar dinero pasándole solo dos medias móviles (20 y 60) no se sabe si es posible. En cambio, hay brokers que ejecutan órdenes bien con información conocida. Si alguien tiene una estrategia y le das al agente la información que usa (las dos medias) y recompensas para capturarla, el RL la puede emular; pero entonces el objetivo es emularle a él, no ganar dinero. Siempre cabe que luego vaya un poco más allá y supere a la persona.

**El objetivo del caso de ejecución** (T07 ~01:44-01:47). Algoritmo de tipo *schedule* (calendario): poco impacto y poca urgencia, sin consideración por el precio de llegada, solo compra, y ejecutar la orden **en su totalidad** ("quiero los 10 millones, no 8"). Se mide con el **VWAP slippage**: mi precio medio ponderado frente al VWAP del mercado en el mismo periodo; si es inferior, éxito. Además conviene vigilar la **dispersión**: un algoritmo muy agresivo puede tener muy buena media y resultados muy malos en algunos casos. El detalle de algoritmos y métricas está en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md) y [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

**Alineación con el objetivo según el tipo de estrategia** (T07 ~01:32-01:35). Un alumno preguntó si la recompensa debe compararse con el mercado. Su preocupación: si el algoritmo toma la decisión correcta en un mercado muy malo, una recompensa absoluta le da mala señal, afecta a la tabla Q "de una manera errónea" y lo interpreta como mala decisión (T07 ~01:32:13-01:32:45). Respuesta: depende del objetivo. Con retorno absoluto (estrategias de un 2-3 % siempre positivo, como un depósito; o *market neutral* con largos y cortos que aísla el mercado), perder dinero es hacerlo mal. Si el objetivo es alfa, haber perdido menos que el mercado es hacerlo bien y la recompensa debe reconocerlo. "Tiene que haber una alineación entre tus recompensas y todo lo demás". Al presentar resultados hay dos lecturas, la de IA (ha aprendido, validación) y la financiera; siempre frente al benchmark, y no con una única comparación fija en el tiempo.

*Fuente: APP p.58, p.66; T06 ~02:29-02:33, ~03:22-03:24; T07 ~01:32-01:35, ~01:44-01:47.*

---

## 3. Qué hace buena a una recompensa: alineada y aprendible

**Diapositiva** (APP p.59, con la imagen de una zanahoria colgando de un palo):
1. **Guarda relación con el comportamiento u objetivo deseado**: flechas paralelas "Recompensas ↑ Resultados ↑". "Si el objetivo responde a una puntuación o fórmula, esto podría ser el objetivo. Lo mejor es que la recompensa sea el propio objetivo, pero no siempre es posible. ¿Ejemplo?". Aviso: el algoritmo es muy sensible a las recompensas.
2. **"Fácil" de aprender (posible)**: al final el algoritmo intentará descubrir cómo funciona el sistema de recompensas y aprender las acciones que lo maximizan. Pregunta clave: "¿es posible conocer la acción óptima con mi vector de características?".

**Lo que dijo** (T06 ~02:34-02:35):
- Hay que comprobar que cuando suben las recompensas los resultados van en la dirección buscada: mirar siempre las recompensas, sacar gráficos, ver comportamientos y que haya cierta correlación.
- Si una recompensa que parecía buena no aprende, no perderse probando otras a ciegas: **contraponer episodios buenos y malos** y ver cómo se repartieron las recompensas en cada uno.
- La factibilidad se aplica también a las recompensas: son proxies con los que el agente intenta descifrar el objetivo; "no va a ser fácil, pero tiene que ser posible de aprender". Prometió aterrizarlo en algoritmos de ejecución (§4).

**La recompensa es un proxy** (T07 ~00:31-00:32). En supervisado tienes etiquetas y error; en RL la recompensa es solo un proxy del objetivo. Que las recompensas sean estupendas porque se llegó al punto B no basta: hay que ver cómo se llega (andando o a la pata coja y a punto de caerse). Hay que mirar recompensas **y comportamientos**, y que el comportamiento en producción se parezca al del entrenamiento.

**Recompensas y entorno, los dos escollos** (T06 ~00:58-00:59). Definir recompensas es un escollo y el entorno es otro; si se resuelven, se tiene algo muy potente. Comentó que si se le pregunta a un asistente tipo GPT qué recompensa poner en bolsa, propone algo muy básico (el valor siguiente menos el anterior), que "lo hace fatal". En T07 insistió en que recompensas y exploración son los principales escollos y "dos mundos en que podéis poneros todo lo creativo que queráis" (T07 ~00:40; en ~04:16 vuelve a enumerar recompensas, exploración y entorno). Al cerrar el caso de ejecución repitió que el resultado se testea con métricas de cómo se ha comportado, "no solamente mirar las recompensas" (T07 ~03:52).

*Fuente: APP p.59; T06 ~00:58-00:59, ~02:33-02:35; T07 ~00:31-00:32, ~00:40, ~03:52, ~04:16.*

---

## 4. Recompensas factibles: predecibles con el estado

**Idea.** Una recompensa puede estar perfectamente alineada con el objetivo y aun así no aprenderse, si su valor depende de cosas que el agente no puede inferir de sus características. "El objetivo tiene que ser factible, pero las recompensas también tienen que ser factibles" (T07 ~03:37). Es, según el profesor, un punto "superimportante" en el que se paró a propósito (T07 ~03:41).

### 4.1 Opciones para el caso de ejecución (T07 ~03:30-03:32, ~03:38-03:39; CASOS p.43)

| Recompensa | Qué compara | Factibilidad |
|---|---|---|
| VWAP slippage **global** | Al terminar toda la ejecución, mi VWAP frente al del mercado | Muy baja: exige predecir precio y volumen futuros |
| VWAP slippage **local** | "En estos momentos lo estoy haciendo mejor" (puede que luego no) | Mayor; en media, hacerlo bien localmente debería beneficiar a largo plazo |
| Frente a la **alternativa que se quiere batir** (TWAP) | En cada momento de decisión, mi acción frente a la que habría tomado el TWAP: "quién de los dos llevaba razón" | La más local y la más asumible; "puede ser una recompensa muy interesante" |

### 4.2 Por qué el VWAP global no se aprende: el ejemplo dibujado (T07 ~03:34-03:40)

En la pizarra: el precio empieza arriba y baja; al principio, arriba, el mercado negocia bastante volumen y su VWAP queda alto; el agente (en negro) compra al principio al mismo precio que el mercado; cuando el precio empieza a bajar y el mercado negocia poco, el agente compra más *(dudoso: la transcripción dice primero que en la bajada el mercado "ejecutó mucho" y a continuación que "se ejecutó poquito"; la lectura coherente con que el agente vaya por delante es "poco")* para aprovechar la caída y va por delante (su VWAP por debajo del VWAP del mercado). Más tarde el precio cae muchísimo y, además, el mercado negocia mucho volumen en esa zona: el VWAP del mercado se desploma, pero al agente ya no le queda volumen para aprovecharlo y termina **por encima** del VWAP del mercado. Conclusión: una recompensa global implica predecir no solo el precio futuro sino también dónde va a negociar el mercado, "precio y volumen", más difícil que predecir un precio. Aunque esté alineada con el objetivo ("si mejoro esto, seguro que mejoro al TWAP clásico"), impide aprender porque el agente no puede sacar relaciones entre sus características y esa recompensa. Hay que buscar otra "igualmente alineada, pero asumible, predecible por las características que le paso".

**Analogía de inversión** (T07 ~03:40). Si para elegir entre petróleo y Repsol se le da como recompensa (o como etiqueta en supervisado) la rentabilidad absoluta a final de año, "ya tienes el problema resuelto", es decir, se le pide lo imposible. Lo razonable es preguntar si localmente lo hizo mejor o peor que el Brent.

### 4.3 El ejemplo "superespecífico": limitada frente a agresiva (T07 ~03:41-03:44)

Local en el tiempo y relativo al benchmark: el algoritmo clásico pone una limitada a 7,10; el agente apuesta por una "más limitada todavía" (7,08). Si se ejecuta, **+2**; si no y se acaba el tiempo, agresiva a 7,13 y **−3** (céntimos frente a 7,10): "hice una apuesta, me salió mal" (T07 ~03:42-03:43). Basta la probabilidad de ejecución como característica, y el objetivo se aterriza en un *proxy* factible (T07 ~03:43-03:44). Desarrollo, lectura de los precios y dudas en [07 §9.4](07_rl_para_ejecucion.md).

**El ejemplo cerrado con números** (T07 ~04:11-04:12), distinto del anterior: probabilidades supervisadas por nivel (80 %, 60 %, 50 %, 10 %) frente a la agresiva del TWAP a 7,12, con un coste de agresividad de unos 3 pb; da la utilidad esperada $0{,}8\cdot2-0{,}2\cdot3=+1{,}0$ para el nivel del 80 %. Cálculo, unidades y umbral en [07 §18](07_rl_para_ejecucion.md).

**Nota:** una versión anterior de este documento atribuía al TWAP una orden agresiva a 7,10 en el primer ejemplo; la transcripción dice "la alternativa clásica… una limitada a 7 por 10" (T07 ~03:42:13-03:42:24), y así queda en 07 §9.4.

### 4.4 Conexión con la práctica

- La recompensa por defecto del notebook es $r=(\text{VWAP}_{\text{episodio}}-p)/\text{VWAP}_{\text{episodio}}$ al comprar (LAB celda 7, docstring de `_compute_reward`; LABSTEP celda 17), donde `episode_vwap` es el VWAP del mercado **al final del episodio**, precalculado en `reset` (LAB celda 7, `_compute_episode_market_feat`). El profesor explicó que lo deja precalculado porque el algoritmo no impacta en el mercado y puede querer usarlo para la recompensa (T07 ~04:51).
- `_compute_reward(price, vol)` tiene **dos TODO**: la recompensa cuando `vol == 0` (no comprar: `_do_nothing` llama a la misma función con la pista "Precio y volumen ejecutado = 0") y la recompensa cuando `vol > 0` (por defecto la de arriba; pista "Utiliza episode_vwap y price para la reward por defecto"). Además, "Opcional: Utiliza el self y elimina los parámetros de la función" (LAB celda 7): la recompensa puede depender de `state_pos`, `exec_vol`, `vol_care` o `episode_bins`, que es lo que hace falta para premiar el seguimiento del plan TWAP. **Nota:** el valor que se dé a `vol == 0` fija el coste relativo de esperar frente a comprar y es la primera decisión de diseño del ejercicio 2.
- **Nota (sesgo de signo de la recompensa por defecto):** el precio de compra es `ask1` (LAB celda 7, `_agg_action`: "Acción agresiva de compra de un título a precio de episode['ask1']"), mientras que el VWAP del episodio se calcula con el mid ("asumimos que el precio es el mid price de cada step", `_compute_episode_market_feat`). Cada compra recibe así, en media, una recompensa negativa del orden de medio spread relativo. Si no comprar vale 0 y `_compute_done_reward` devuelve 0 (su valor por defecto), maximizar la suma lleva a comprar lo menos posible y a dejar volumen pendiente (el fallo de §8). Es un caso concreto de "lo mismo ocurre con el signo" (APP p.61, §9). Se compensa con la penalización final por lo no ejecutado (§7), con una recompensa de seguimiento del plan o comparando contra una referencia homogénea (ask frente a ask; el propio TODO admite "un precio más realista para el mkt VWAP"). Desarrollo del sesgo en [08_entorno_best_execution.md](08_entorno_best_execution.md).
- **Nota:** esa referencia es "global" en el sentido de §4.2: comparar cada compra con el VWAP final exige saber hacia dónde irá el mercado, y con el estado por defecto (tiempo y volumen restantes) no es predecible. Vale como recompensa de evaluación; como señal de aprendizaje para mejorar el TWAP (idea A del ejercicio 4) es candidata a sustituirse por una local (VWAP de mercado hasta el step, `_compute_market_vwap`) o frente al TWAP. Desarrollo en [08_entorno_best_execution.md](08_entorno_best_execution.md) y [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).
- Para replicar el TWAP (ejercicio 2) el profesor recalcó que las recompensas "no tienen que estar condicionadas a tu rendimiento": calcular el VWAP de mercado parcial "no es necesario", basta ir a la función de recompensa y calibrarla para replicar el TWAP (T07 ~04:57-04:58). Con tiempo y volumen restantes es suficiente: el TWAP solo necesita que ambos "vayan de la mano" para que salga la diagonal (T07 ~04:52).
- **La regla que hay que reproducir** (T07 ~04:53-04:56). El TWAP determinista es: si vas por delante del plan de ejecución, no ejecutes más; cuando vuelvas a estar por debajo o igual, lanza otra orden hija. Se podría programar con un `if`, pero quiere que se haga "también por recompensas". Como los escenarios son finitos, valdría incluso un Q-learning tabular con dos estados (volumen ejecutado por encima del plan, o igual/menor) y dos acciones (ejecutar o no). No es obligatorio usar el Double DQN ni el notebook: se puede rehacer entero, siempre que con esa parametría se ejecute un activo (SAN o REP) "por recompensas".
- **Nota:** una recompensa que premie la acción coherente con esa regla, o que penalice $|x_t-x^{\text{TWAP}}_t|$ (fracción ejecutada frente a la del plan), es directamente factible con el estado de tiempo y volumen restantes: cumple las dos condiciones de §3.

*Fuente: CASOS p.43; T07 ~03:30-03:44, ~04:11-04:12, ~04:51-04:58; LAB celda 7; LABSTEP celda 17.*

---

## 5. Sparse reward frente a reward shaping

**Idea.** La diferencia es la frecuencia con que se dan recompensas: en un extremo, una sola al final del episodio (*sparse*); en el otro, una en cada paso (*shaping*) (T06 ~02:36; APP p.60).

**Diapositiva** (APP p.60). "No es necesario dar una recompensa por cada acción que toma. En el otro extremo, puede recibir una sola recompensa al final del episodio". Gráfico: una balanza con "Sparse Reward" en un platillo y "Reward Shaping" en el otro. Debajo, a la izquierda un pingüino (sparse: robusto; dificultad para aprender) y a la derecha una secuencia de figuras humanas andando (shaping: uso al caso; problema de mínimos). Abajo, una hamburguesa, una barbacoa y un pescado: "No es lo mismo cocinar uno u otro. En bolsa pasa algo similar. ¿Deberíamos usar el mismo sistema para todos los activos? Exploración es muy importante. En concreto, ¡saber explorar!".

| | Sparse | Shaping |
|---|---|---|
| Ventajas | Más robusta, limpia y sencilla; suele coincidir con el objetivo final | Feedback en cada paso; adaptada al caso |
| Problemas | Cuesta más aprender (poco feedback intermedio); puede alcanzar el objetivo de una forma no deseada | Difícil de programar; mínimos locales; sesgos |

Cuál elegir "muchas veces no está clara": depende de lo complejo que sea el problema y de los problemas de exploración que se tengan (T06 ~02:36).

**Ejemplos de clase:**
- **Andar de A a B** (T06 ~02:37). Si solo se premia llegar a B, puede llegar como un pingüino, saltando, volando o haciendo el pino. Si el objetivo era llegar, perfecto; si era andar como una persona, la recompensa estaba mal definida.
- **Los brazos** (T06 ~02:40-02:41). Se añade shaping para que mantenga los brazos quietos y aprende a andar como un pingüino; se premia un movimiento concreto de brazos y, si pesa demasiado, se olvida de ir al punto y se queda moviendo los brazos en el sitio: "problema de mínimos", muy específico y difícil de programar.
- **Cocinar** (T06 ~02:41-02:42). Premiar solo "comida lista" es limpio, pero la exploración necesaria para llegar es muy difícil (desembalar, calentar, dar la vuelta; el pescado exige otros pasos, quitar raspas). Se pueden dar recompensas intermedias (por quitar el envoltorio...). También existen técnicas de exploración que no se basan en recompensas (ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md)).
- **El shaping es también una técnica de exploración.** APP p.72 agrupa bajo "Recompensas" el reward shaping y "curriculum learning, curiosity-driven, auxiliary tasks". En clase citó el feedback intermedio basado en el conocimiento que se tenga del entorno, el *curriculum learning* (aprender primero a desembalar la hamburguesa, luego a ponerla al fuego), los subobjetivos ligados al objetivo principal y las recompensas basadas en el error de predicción (T06 ~03:31-03:32); y las recompensas intrínsecas, basadas en el error, con las que se resolvió Montezuma (T06 ~04:33-04:35; APP p.80, *Exploration by Random Network Distillation*), que funcionaban muy bien en ese juego y no en el resto. Desarrollo en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).
- **Santander frente a Viscofan** (T06 ~02:42). Uno muy líquido y otro no; el mercado responde distinto a tus acciones, así que el mismo sistema de recompensas no tiene por qué servir para todos los activos.
- **Recompensa final condicionada** (T06 ~02:42-02:43). Un alumno propuso premiar solo al llegar a B si además solo los pies tocaron el suelo y los brazos no se movieron más de X grados. El profesor: en general se puede, aunque el aprendizaje sería lentísimo, y admitió no tener la solución.
- **Vídeos** (T06 ~02:43-02:46). Agentes que aprenden a correr con una recompensa muy simple, "avanzar hacia delante": llegan a esquivar y saltar obstáculos, pero mueven los brazos de forma rara ("como si celebrara un gol"); el entorno no penaliza no mover los brazos, que las personas usan para equilibrarse. Los alumnos citaron el escondite multiagente: premiados por sobrevivir, aprendían a encerrarse con bloques y quitar la rampa, sin resolver nunca el juego.

**En profit-seeking** (T07 ~01:19-01:21; CASOS p.11):
- Si el objetivo es muy específico y responde a una fórmula, de ahí sale la recompensa: al final del episodio o con shaping.
- Se puede recompensar en cada step, se actúe o no y se esté o no invertido: no estar invertido también tiene un coste de oportunidad (CASOS p.11: "Puedes dar recompensas aunque no estés invertido"; "Cada step o final episodio"; recompensas en función de alfa, *information ratio*, mejora de Sharpe).
- Si hay problemas de feedback, se aborda la exploración o se da feedback intermedio. Pero las recompensas intermedias tienen **sesgo**: si el agente ve buenos momentos para mejorar su alfa, empieza a comprar y recibe recompensas positivas; puede que al final hubieran surgido oportunidades mucho mejores y ya no le quede dinero. Se puede hacer "una ponderación de todo al final" *(dudoso: la frase está cortada; parece proponer ponderar las recompensas intermedias con el resultado final)*.
- **Nota:** el notebook `profit-seeking.ipynb` usa un shaping por step relativo al benchmark, $r_t=\ln(R^{\text{algo}}_{t+1}-R^{\text{bench}}_{t+1}+1)$ si hay posición y $0$ si no (RLM profit-seeking, celda 3, `_compute_alpha`): no aplica el coste de oportunidad cuando no se está invertido. Detalle en [05_profit_seeking.md](05_profit_seeking.md) y [09_material_rl_notebooks.md](09_material_rl_notebooks.md).

**Cuántos elementos recompensar** (T06 ~02:47). No solo importa qué se recompensa, sino cuántos elementos: eso cambia mucho el modelo; a veces lo mejor es recompensar solo al final si se cumple el objetivo. Desarrollo previo en [T2 00_fundamentos_mdp.md §1](../raw/referencias/deep-rl-trading-agents/docs/teoria/00_fundamentos_mdp.md).

*Fuente: APP p.60, p.72, p.80; CASOS p.11; T06 ~02:35-02:47, ~03:31-03:32, ~04:33-04:35; T07 ~01:19-01:21; RLM profit-seeking celda 3.*

---

## 6. Recompensas compuestas y su calibración

**Idea.** En ejecución la recompensa suele ser una función de varios términos, "Rewards (determinan comportamiento del algo)" (CASOS p.44), y el peso de cada uno determina el comportamiento. Calibrarlos es la parte difícil.

**Los términos** (CASOS p.43, con una escalera ascendente hacia un trofeo; CASOS p.44, esquema completo agente-entorno con estado de ejemplo $S_3$ 10:30: spread 2 pbs, probabilidad de ejecución 85 %, tiempo restante 25 %, volumen restante 20 %; acciones pasiva/mantener/agresiva; agente "e.g., Q-learning"):

$$r=f\big(\text{impacto de mercado},\ \%\text{ vol. CARE ejecutado},\ \text{desviación respecto al plan tradicional}\big)$$

| Término | Para qué (T07 ~03:30-03:33) |
|---|---|
| Impacto de mercado: VWAP slippage local o global | El objetivo de precio (§4) |
| % del volumen de la orden CARE ejecutado | Ejecutar la totalidad: penalizar si deja órdenes pendientes o premiar si lo ejecuta todo |
| Desviación respecto a un plan tradicional (TWAP) | "Un TWAP con esteroides" que no se aleje mucho del TWAP |
| "…" con el "e.g., equivalente al precio de lanzar orden a mercado en el último segundo" (CASOS p.43) | Interpretación: valorar lo no ejecutado al final del episodio; ver §7 |

**Por qué penalizar la desviación del TWAP** (T07 ~03:32-03:33): (1) los clientes están acostumbrados al TWAP y se asustarían con un algoritmo que "puede hacer cualquier cosa"; se les ofrece algo muy similar y un poco mejor; (2) facilita calibrar el simulador: se pasan al simulador órdenes TWAP reales y, si el impacto que da es realista, se puede asumir que también lo será para un algoritmo que ejecuta casi igual.

**El problema de calibración** (T07 ~03:34). Con mucho peso a completar la orden, puede ejecutarlo todo al principio; con mucho peso al retorno, si no ve buen precio deja volumen pendiente. Solución: mirar en cada momento qué consigue por cada dimensión, evaluar los gráficos de comportamiento e ir ajustando.

**Ponderación de subobjetivos** (T06 ~02:49-02:50). Con proxies o subobjetivos (rentabilidad y también riesgo), la ponderación es "muy, muy importante"; mal calibrada aparece el mínimo local del que "solo mueve los brazos" (o solo se mueve en círculos). Recomendación práctica: **llevar un log de cuánto ha recibido el agente por cada criterio de recompensa** y calibrar viéndolo junto con el comportamiento.

**Perfiles de agresividad** (CASOS p.53; T07 ~03:53-03:55). Órdenes CARE de compra con benchmark TWAP y distintos grados de libertad respecto al plan: conservador 5-10 %, moderado 10-30 %, agresivo 30-50 %, muy agresivo >50 %; con benchmark POV, conservador 5-10 %. El conservador se controla con **recompensas y reglas programadas** (una llave «{» en la diapositiva agrupa ambas opciones junto al perfil conservador). Por encima del 50 % "el algo RL puede hacer lo que quiera con tal de conseguir su objetivo → comportamientos atípicos". En clase mostró un conservador que, en un día en que el mercado negoció mucho volumen pronto, acabó peor que el mercado porque no tenía libertad para cerrar el hueco, y un moderado que aprendía a comprar más en ciertos momentos. La desviación también se puede limitar por regla: si se desvía demasiado, se le desautoriza y se vuelve al plan hasta que se acerque de nuevo. Resultados completos en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

**El límite de las asunciones** (T07 ~03:56). El simulador está calibrado con órdenes TWAP: cuando el algoritmo hace algo "súper agresivo" que barre posiciones del libro, no se puede asegurar que el impacto simulado sea realista, y esos resultados (incluso los del moderado) hay que tomarlos "con pinzas". **Nota:** es el reverso del segundo motivo para penalizar la desviación del plan: ese término mantiene al agente en la zona donde el simulador es fiable.

**En la práctica** (ENUNCIADO, ej. 4, idea B): "Añadir recompensas para evitar 1. Que el algoritmo deje volumen pendiente y 2. Que el algoritmo consuma aproximadamente un 90% de la duración de la orden para dejar un cierto margen. Es necesario una correcta calibración de las recompensas para asegurarnos la estabilidad de resultados". En clase lo explicó dos veces. Primero (T07 ~04:30): no vale quedarse con un 80 % del volumen ejecutado (la primera mitad de la idea, volumen pendiente), y quiere que consuma aproximadamente el 90 % *(la transcripción dice "del volumen"; por el ENUNCIADO y por lo que sigue, se refiere a la duración)*, "no exactamente", en un rango "del 80 al 95", porque da un poco de tiempo de más pero quiere que se apure hasta el último momento; avisó de que, aunque el objetivo es sencillo, obliga a aprender a calibrar el término que fija cuándo termina. Después, al resumir los ejercicios, habló de "un rango entre un 70 90 por 100" (T07 ~04:58). Hay así tres cifras: ~90 % (ENUNCIADO), 80-95 % (T07 ~04:30) y 70-90 % (T07 ~04:58). La discrepancia se trata en [notebook_partida.md](notebook_partida.md).

**Nota:** ejemplo de forma compuesta para ejecución (propuesta, no de clase): $r_t=w_1\,r^{\text{precio}}_t-w_2\,|x_t-x^{\text{TWAP}}_t|-w_3\,\mathbb{1}[\text{done}]\cdot q_{\text{pendiente}}\cdot c_{\text{agresiva}}$, con $x_t$ la fracción ejecutada. Para calibrar $w_i$ conviene estimar antes el rango de cada término en un episodio típico (APP p.61, §9) y registrar su suma por episodio por separado.

*Fuente: CASOS p.43-44, p.53; T06 ~02:49-02:50; T07 ~03:30-03:34, ~03:53-03:56, ~04:30, ~04:58; ENUNCIADO ej. 4.*

---

## 7. Recompensa al final del episodio y penalizaciones por paso

**Fin de episodio** (APP p.85). Hay que decidir qué determina `done == True`: fin previsto o forzado ("no merece la pena continuar o imposibilidad de recuperación") y, en ambos casos, "gestionar recompensas final de episodio". El detalle de terminated/truncated está en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

**En el caso de ejecución.** La penalización final natural es por lo que quede sin ejecutar, valorada como si se lanzara una orden a mercado en el último segundo (CASOS p.43). **Nota:** es una interpretación del "e.g." de CASOS p.43, que en la diapositiva cuelga del "…" que sigue al término de % de volumen ejecutado; la diapositiva no habla literalmente de "penalización final". En el notebook es la función `_compute_done_reward` (12), marcada "TODO: Free style" y que por defecto devuelve 0; `step` la suma a la recompensa del último paso cuando `done` es cierto (LAB celda 7; LABSTEP celdas 27-28). Las condiciones de parada son tiempo agotado u orden completada (`_compute_stop_conditions`, LAB celda 7).

**Nota (la recompensa final no queda en el histórico):** en `step`, `self.reward_hist.append(reward)` se ejecuta antes de `reward += self._compute_done_reward()` (LAB celda 7). La penalización o premio final llega al agente pero no aparece en `reward_hist`, así que para el log por criterio de §6 (T06 ~02:50) hay que registrarla aparte (por ejemplo, en un `done_reward_hist`). El profesor presentó los históricos de recompensas y acciones como algo "por si queréis loguearlos después" (T07 ~05:00).

**Nota:** en el agente, una experiencia con `done` se guarda con $s'=$`None` y su objetivo es solo $r$ (sin bootstrap) (BE dqn.py, experience y `learn`). Por eso la recompensa final entra entera en el último Q y se propaga hacia atrás descontada por $\gamma$ (§12). Como el tiempo restante está en el estado, tratar el fin por tiempo como terminal es coherente (el problema sigue siendo Markov).

**Penalización por paso y urgencia** (T06 ~02:50-02:51). En el acantilado (*cliff walking*) no hay premio por llegar: hay $-1$ por cada paso sin llegar. Eso **ya mete la urgencia**, así que no hace falta un $\gamma$ bajo. También se podría resolver con $+50$ al llegar a la bandera y $-1$ por paso. El $-100$ por caer lo mencionó en otro momento, al hablar de la inestabilidad: el agente del acantilado "recibió el menos 100" y, con esas señales constantes, es difícil saber cuál es el camino óptimo (T06 ~01:14) *(dudoso: la transcripción dice "señal de 1 y señal de pie"; probablemente $-1$ y $-100$)*. **Nota:** en CliffWalking de gym (el entorno de `cliff-walking.ipynb`) caer da $-100$ y devuelve al inicio; el notebook no explicita las recompensas (RLM cliff-walking).

**Las cuatro formas** (T06 ~02:53). Un alumno planteó *(dudoso: la transcripción no separa hablantes; por el tono es una pregunta)* que se pueden premiar aciertos, castigar fallos, premiar la ausencia de fallos o castigar la falta de aciertos (premiar llegar lo antes posible o castigar caerse), y preguntó cuál de las cuatro es mejor para el entrenamiento. Respuesta: por lo general la que da **más señal de forma consistente y con menos ruido**; si una alternativa da muy pocas señales, suele ser más difícil.

**Coste de oportunidad por paso** (T07 ~01:19). En inversión se puede evaluar cada paso aunque no se actúe o no se esté invertido (§5).

*Fuente: APP p.85; CASOS p.43; T06 ~01:14, ~02:50-02:53; T07 ~01:19, ~05:00; LAB celda 7; LABSTEP celdas 27-28; BE dqn.py.*

---

## 8. Efectos indeseados: el agente hace lo que sea por maximizar

**Idea.** "Hacen lo que quieran con tal de conseguir la mayor recompensa" (APP p.61); "el algoritmo puede hacer lo que sea con tal de maximizar sus recompensas" (APP p.96). Cada término mal pesado produce un comportamiento reconocible.

| Comportamiento | Causa | Fuente |
|---|---|---|
| Termina **demasiado pronto** | Bonus por terminar antes con demasiado peso ("me he pasado") | T06 ~02:50 |
| Ejecuta **todo al principio** | Demasiado peso a completar la orden | T07 ~03:34 |
| Deja **volumen pendiente** | Demasiado peso al retorno: si no ve buen precio, no compra | T07 ~03:34 |
| **Opera casi siempre al principio** del episodio | $\gamma$ mal calibrado para la longitud del episodio | T07 ~01:09-01:10 |
| Invierte pronto y se queda **sin dinero** para mejores oportunidades | Recompensas intermedias positivas | T07 ~01:20-01:21 |
| Mueve los brazos en el sitio / anda como pingüino / corre "celebrando un gol" | Shaping mal ponderado o recompensa incompleta | T06 ~02:37-02:45 |
| Se encierra y nunca juega (escondite) | Premio por sobrevivir | T06 ~02:46 |
| Toca una pared antes de ir a B | Interacción inesperada con $\gamma$ | T06 ~02:39 |
| Comportamientos "atípicos" | Demasiada libertad respecto al plan (>50 %) | CASOS p.53 |

**Reglas deterministas en lugar de recompensas** (T06 ~02:55-02:57; APP p.61 "Comportamientos muy específicos / temas sensibles → Reglas"). No todo tiene que salir de las recompensas: lo que tenga que ser al 100 % de una forma, se programa como regla y se deja fijo. Ejemplos: vender si el mercado cae un 10 % aunque el algoritmo hubiera aprendido que es momento de comprar; prohibir acciones que acerquen a dos casillas o menos del precipicio si el estado permite detectarlo. El agente puede decidir con la red, con reglas o con ambas, y las reglas también sirven para guiar el entrenamiento y acotar el espacio. En ejecución, la regla de la orden "pegada" en profundidad o la desautorización del perfil conservador son ejemplos (T07 ~03:28, ~03:53).

*Fuente: APP p.61, p.96; CASOS p.53; T06 ~02:37-02:57; T07 ~01:09-01:21, ~03:34, ~03:53.*

---

## 9. Magnitud, signo, semillas y prueba masiva

**Diapositiva** (APP p.61, iconos de metro, signos +/−, péndulo de Newton y matraces):
- **Magnitud (relativa) es muy importante**: no poner $\pm1$ o $\pm100$ "muy a la ligera"; es necesario estimar cuánto puede recibir por cada objetivo parcial; mantenerse en $[-1,1]$ puede ayudar a la estabilidad.
- **Lo mismo ocurre con el signo** (acompañado del dibujo de una cuadrícula con una figura y una bandera; probablemente alude al acantilado).
- **Ligeras modificaciones → comportamiento totalmente distinto**: "¿qué ha causado que se comporte así?"; las mismas recompensas pueden dar resultados distintos (semilla); probar ligeras modificaciones y ver comportamiento.
- **Hay que probar muchas recompensas**; el conocimiento de negocio es importante (igual que para el estado).

**Lo que dijo** (T06 ~02:48-02:49, ~02:54-02:55): la gente pone $\pm1$ o $\pm100$ muy a la ligera; para ilustrarlo volvió al ejemplo del paraguas (T06 ~02:48), en el que había dado $-100$ por mojarse y algo menos por llevar el paraguas en un día de sol porque "no es tan incómodo" (T06 ~00:19); en APP p.99-100 sus recompensas son $+100$ (seco con lluvia), $-100$, $-20$ y $+50$, magnitudes elegidas a ojo (desarrollo del ejemplo en [00_repaso_rl.md](00_repaso_rl.md)). Por lo general se dan recompensas pequeñas porque las redes funcionan mejor así *(dudoso: la transcripción mezcla la crítica de la diapositiva a poner $\pm1$/$\pm100$ a la ligera con la recomendación de usar recompensas pequeñas, "más o menos 1 o más o menos 100 términos relativos"; la lectura segura es "recompensas pequeñas, porque las redes funcionan mejor", en línea con el $[-1,1]$ de APP p.61)*. Recompensas, $\gamma$, steps y estados interactúan: "ligerísimas" modificaciones dan comportamientos totalmente distintos, e incluso con las mismas recompensas los entrenamientos varían; hay que repetir y fijar semillas para medir la inestabilidad (ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md)). Defiende empezar por lo más sencillo (primero el triciclo, luego el Ferrari): un entorno mínimo (dos puntos en un espacio) y escalar después (T06 ~02:47-02:48).

**La prueba masiva** (APP p.62). La diapositiva solo tiene dos imágenes: una cuadrícula de decenas de carpetas de experimentos y un monitor de procesos tipo htop con muchísimos núcleos al 100 %. Corresponde a su TFM de 2019 sobre algoritmos de ejecución: lanzaba 72 experimentos en paralelo con recompensas y configuraciones distintas; por las mañanas leía papers y diseñaba recompensas, por la noche lanzaba experimentos y al día siguiente veía cómo se habían comportado (T06 ~02:57-02:59). Muchas de las recomendaciones vienen de esa experiencia.

**Nota (escala en la práctica):** con la recompensa por defecto $(\text{VWAP}-p)/\text{VWAP}$, la diferencia entre el precio de un bin y el VWAP del episodio en 33-50 minutos es del orden de unos pocos a unas decenas de puntos básicos (estimación, no medida con los pickles), así que $|r|\sim10^{-4}$-$10^{-3}$ por compra: muy lejos del orden de $\pm1$ recomendado y de la escala de las penalizaciones que se añadan. Expresarla en pbs ($\times10^4$) o normalizarla ayuda a la red y a calibrar pesos. Dos consecuencias concretas en el notebook: el log de entrenamiento redondea la media de `history_rewards` a 2 decimales (`np.round(..., 2)`, LAB celda 18), así que con la recompensa por defecto mostrará 0,0 aunque el agente mejore; y la curva de la celda 21 es de `history_rewards` (suma sin descontar), no de `history_disc_rewards`. Conviene escalar la recompensa o cambiar el redondeo. La importancia de la escala de las salidas de Q ya se vio en [T2 08_trading_en_clase.md §5](../raw/referencias/deep-rl-trading-agents/docs/teoria/08_trading_en_clase.md), y en el Taller 2 se escaló la recompensa (del orden de $10^{-3}$ por paso) dividiéndola por su $\sigma$ de train ([T2 DECISIONES.md, D-1.1](../raw/referencias/deep-rl-trading-agents/docs/DECISIONES.md)).

*Fuente: APP p.61-62, p.99-100; T06 ~00:19, ~02:47-02:49, ~02:54-02:59; LAB celdas 18 y 21.*

---

## 10. Gamma: qué significa

**Idea.** $\gamma$ es el factor de descuento o "valor del tiempo" (APP p.75, con un icono de reloj y una moneda): cuánto pesa en la decisión de ahora lo que pasará más adelante.

**Ejemplo del taxi autónomo** (T06 ~00:39-00:40). Sin $\gamma$, los 50 € del trayecto de A a B valen lo mismo por la calle más recta que dando la vuelta al mundo. $\gamma$ introduce el sentido de urgencia.

**Impacto temporal de las acciones** (T06 ~03:43-03:46). Pregunta: ¿0,95 es alto o bajo? Depende de la longitud de la secuencia y del impacto temporal de las acciones: cuanto más alejada en el futuro está la información, menos impacto tiene en la decisión de ahora. Ejemplo vital: beber otra botella de agua no se decide pensando en el yo de 80 años; ese horizonte "se diluye" con $\gamma$. Sin $\gamma$, lo que pase a los 80 años pesaría en las decisiones de hoy tanto como las recompensas inmediatas.

**Matiz corregido en vivo** (T06 ~03:43-03:44). Empezó comparando el impacto de los precios de hace un mes con el de los últimos días y se corrigió: "es justo lo contrario"; $\gamma$ mide cómo el futuro influye en la decisión de ahora, no cómo influye el pasado. **Nota:** el peso del pasado corresponde a la tasa de aprendizaje $\alpha$ ([04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md)) y a las características del estado ([02_diseno_de_entornos.md](02_diseno_de_entornos.md)).

**Cuanto más bajo, más urgencia, pero ¿cuánta?** (T06 ~03:46-03:47). Un $\gamma$ más bajo fuerza a alcanzar el objetivo antes; pero si se quiere urgencia, ¿se consigue con 0,99, con 0,95 o con 0,2? "Depende": el descuento es exponencial, $\gamma^0,\gamma^1,\gamma^2,\dots$ (§12).

**$\gamma$ produce comportamientos inesperados** (T06 ~02:38-02:39). Un alumno preguntó si un $\gamma$ muy exigente lleva a arrastrarse o a estrategias ineficientes como en los vídeos de simulaciones. Respuesta: $\gamma$ a veces da comportamientos inesperados (por ejemplo, aprender que antes de ir a B hay que tocar una pared), no simplemente "salir corriendo"; sus elementos interactúan entre sí.

*Fuente: APP p.18, p.75; T06 ~00:39-00:40, ~02:38-02:39, ~03:42-03:47.*

---

## 11. De qué depende gamma y cuándo usar $\gamma=1$

**Diapositiva** (APP p.75-76, idénticas). "¿Qué gamma? ¿De qué depende?":
- Valor del tiempo.
- Signo de las recompensas.
- Problema episódico o continuo.
- Número de steps del episodio.
- Número de steps fijo o variable: "mucho cuidado con los que tienen una duración (n steps) variable".

A la derecha, un diagrama de dos estados $S_0$ y $S_1$ unidos por flechas en ciclo: de $S_0$ a $S_1$ con $r=1$ y de vuelta con $r=2$. **Nota:** es un problema continuo: con $\gamma=1$ el retorno crece sin límite ($1+2+1+2+\dots$) y todos los estados valen infinito; hace falta $\gamma<1$ para que la suma converja. Es el ejemplo del experimento de §12.

**Cuándo $\gamma=1$** (T06 ~00:40-00:42). Si no hay urgencia o la partida tiene un número de pasos fijo y conocido (diez pasos, el tres en raya: "quiero ganar, no me hace falta ganar en la segunda ronda"), $\gamma=1$ y no complicarse, porque la urgencia puede meter ruido y hacer que se comporte de forma extraña. Un alumno objetó que lo que importa casi siempre es cómo acaban las cosas; la respuesta fue que precisamente entonces se usa 1, y que solo hay que descontar cuando de verdad se quiere que algo ocurra deprisa. El notebook de la práctica lo permite explícitamente: "dado que nuestro MDP se define con tareas episódicas gamma podrá tomar valores en el intervalo [0,1]" (LAB celda 11). Teoría de la condición ($\gamma<1$ o episodios finitos) en [T2 00_fundamentos_mdp.md §4](../raw/referencias/deep-rl-trading-agents/docs/teoria/00_fundamentos_mdp.md).

**Signo de las recompensas** (T06 ~02:50-02:52, ~03:52-03:53). Con recompensas negativas por paso para ir a B, "a lo mejor la gamma ya no me hace falta": la urgencia ya está en las recompensas (acantilado, §7). Factores adicionales que enumeró al resumir: episodios finitos o continuos, si se conoce su duración y si hay recompensas adicionales por terminar antes *(transcripción poco clara)*.

**El debate del $\gamma$ agresivo** (T06 ~02:51-02:54). Razonó que, si se penaliza muchísimo por paso, "te va a pesar más alejarte temporalmente" que intentar ir por el camino óptimo, porque se quiere la recompensa cuanto antes; y que en el acantilado $\gamma$ "no juega ningún papel porque ya está en negativo" (T06 ~02:51:59-02:52:20). Con $+50$ al llegar, $-1$ por paso y un $\gamma$ "superagresivo" de 0,2 ("en dos pasos ya te has comido las recompensas"), los 50 € pierden valor y aventuró que quizá entonces no se acercaría tanto al precipicio, admitiendo no tener claro cómo se comportaría (T06 ~02:54:03-02:54:12) *(dudoso)*. La conclusión que sacó es que recompensas, $\gamma$, steps y estados interactúan y hay que pensar la solución completa (T06 ~02:54).
**Nota (contraste de la hipótesis, con el $-100$ de gym, §7):** con $\gamma=0{,}2$ el horizonte efectivo es $H=1/(1-0{,}2)=1{,}25$ pasos. El premio de 50 a $d$ pasos vale $50\cdot0{,}2^{d-1}$: 10 a dos pasos, 0,4 a cuatro, prácticamente 0 desde la salida del acantilado (a 13 pasos). Los $-1$ futuros suman como mucho $-1{,}25$, así que lejos de la meta todos los estados valen casi lo mismo y el agente no distingue direcciones: es miope, evita el $-100$ inmediato pero solo "ve" la meta en sus últimos 3-4 pasos. Con recompensas negativas, además, un $\gamma$ bajo **reduce** la urgencia (los castigos futuros pesan menos), al contrario de lo que pasa con recompensas positivas. Es la interacción signo × $\gamma$ de la diapositiva.

**Nota (duración variable en la práctica):** los episodios del notebook tienen entre 400 y 600 steps aleatorios (`_generate_episode_params`, LAB celda 7) y además pueden terminar antes si se completa la orden. Con $\gamma<1$, una misma penalización final pesa $\gamma^{N-t}$ distinto según la longitud del episodio, lo que añade ruido al objetivo de aprendizaje; es el "mucho cuidado" de APP p.75. Al tener el tiempo restante en porcentaje en el estado, el agente puede en parte compensarlo. Con recompensas por paso (shaping, p. ej. desviación del TWAP) el problema es menor que con una recompensa sparse final.

*Fuente: APP p.75-76; T06 ~00:40-00:42, ~02:50-02:54, ~03:52-03:53; LAB celdas 7 y 11.*

---

## 12. Cómo calibrar gamma hacia atrás según el horizonte

**Idea.** No se elige $\gamma$ "a ojo": se despeja a partir del horizonte que se quiere que tengan en cuenta las acciones y de cuántos steps cabe en ese horizonte (T06 ~03:50-03:52; T07 ~01:09-01:10).

### 12.1 El descuento es exponencial: lo alto no es tan alto

**El experimento de clase** (T06 ~03:47-03:49). "¿Quién gana en este entorno?": 100 steps con $\gamma=1$, 1.000 steps con $\gamma=0{,}99$ o 1.000.000 de steps con $\gamma=0{,}98$. Lo ejecutó en vivo: los dos primeros salen más o menos equivalentes y el tercero, pese a tener un millón de pasos, vale aproximadamente la mitad.

**Nota:** reproducción con el ciclo de APP p.75 (recompensas 1, 2, 1, 2...):

| Escenario | Retorno descontado |
|---|---|
| 100 steps, $\gamma=1$ | $50\cdot1+50\cdot2=150$ |
| 1.000 steps, $\gamma=0{,}99$ | $\approx\frac{1+2\gamma}{1-\gamma^2}=149{,}7$ |
| $10^6$ steps, $\gamma=0{,}98$ | $\approx\frac{1+2\cdot0{,}98}{1-0{,}98^2}=74{,}7$ |

El resultado ($\approx$ la mitad) coincide con lo dicho en clase. La razón: el horizonte efectivo $H=1/(1-\gamma)$ es 100 pasos con 0,99 y solo 50 con 0,98; los pasos más allá de unos pocos $H$ no cuentan, por muchos que haya.

**0,95 después de 50 y 200 pasos** (T06 ~03:49; repetido en T07 ~01:10). Con 0,95, tras 50 pasos el valor se queda en torno al 7 %, y tras 200 es prácticamente 0: las acciones de ahora no tienen en cuenta nada de lo que pase dentro de 200 pasos. "Parece que es muy alto y no es muy alto".
**Nota:** comprobación: $0{,}95^{50}=0{,}077$ (7,7 %), $0{,}95^{100}=0{,}006$ y $0{,}95^{200}=3{,}5\cdot10^{-5}$. La transcripción dice "siete por 100 se queda el valor después de 100, cinco después de 200" *(dudoso)*: la lectura correcta es 7 % a los 50 pasos y ~0,6 % a los 100; la vida media de 0,95 es de solo 13,5 pasos.

**Tabla de referencia (Nota, cálculo propio):** peso $\gamma^n$ de una recompensa a $n$ pasos.

| $\gamma$ | $H=1/(1-\gamma)$ | $n=10$ | $n=50$ | $n=100$ | $n=200$ | $n=400$ | $n=500$ | $n=600$ |
|---|---|---|---|---|---|---|---|---|
| 0,90 | 10 | 0,349 | 0,005 | 0 | 0 | 0 | 0 | 0 |
| 0,95 | 20 | 0,599 | 0,077 | 0,006 | 0 | 0 | 0 | 0 |
| 0,98 | 50 | 0,817 | 0,364 | 0,133 | 0,018 | 0 | 0 | 0 |
| 0,99 | 100 | 0,904 | 0,605 | 0,366 | 0,134 | 0,018 | 0,007 | 0,002 |
| 0,995 | 200 | 0,951 | 0,778 | 0,606 | 0,367 | 0,135 | 0,082 | 0,049 |
| 0,999 | 1.000 | 0,990 | 0,951 | 0,905 | 0,819 | 0,670 | 0,606 | 0,549 |
| 0,9998 | 5.000 | 0,998 | 0,990 | 0,980 | 0,961 | 0,923 | 0,905 | 0,887 |

### 12.2 El procedimiento: despejar $\gamma$

Si se quiere que una recompensa a $N$ pasos conserve una fracción $f$ de su valor:

$$\gamma^N=f\quad\Longrightarrow\quad\gamma=f^{1/N}$$

- **Depende de la definición de step** (T06 ~03:50-03:51). Si quiero que mis acciones diarias cuenten para la próxima semana y hago una acción por segundo, calculo cuántos segundos tiene una semana y despejo. En bolsa no es lo mismo una estrategia anual que opera cada día que una que opera cada minuto: la de minutos tiene muchísimas más decisiones al año y su $\gamma$ tiene que ser mucho mayor (más cercano a 1).
- **Ejemplo del 3 % anual** (T06 ~03:51-03:52). Si el coste del dinero es un 3 % al año, la recompensa a un año debe valer un 97 %; se pasa a $\gamma$ por día y por minuto según los puntos de decisión. **Nota:** con 252 sesiones, $\gamma_{\text{día}}=0{,}97^{1/252}=0{,}99988$; con 510 minutos por sesión (9:00-17:30), $\gamma_{\text{min}}=0{,}97^{1/(252\cdot510)}=0{,}99999976$. La transcripción menciona "entre 300" *(dudoso: quizá un número aproximado de días)*.
- **Episodios intradía** (T07 ~01:07-01:10). Con episodios de unas 2 h o un día, si se quiere que el algoritmo busque las mejores oportunidades durante toda la sesión (y no que consiga el retorno a las 10 de la mañana y "apagues el ordenador"), $\gamma$ debe ser alto: da igual cuándo se consiga el retorno siempre que se tenga al cierre. Si se calibra mal, "van a operar casi siempre al principio". Hay que hacerlo **hacia atrás**: cuánto dura el episodio, cuántos steps lo componen, y despejar; "me salió un gamma de 0,9998" *(dudoso: la transcripción dice "0, 99 9. Ocho"; no dijo para qué duración)*. **Nota:** 0,9998 equivale a que una recompensa a unos 3.465 steps conserve el 50 % ($\ln0{,}5/\ln0{,}9998$). No debe confundirse con el $\gamma$ diario del 3 % anual (0,99988, que redondea a 0,9999 y tiene una semivida de unos 5.776 pasos).
- **Episodios largos** (T07 ~01:08:59-01:09:20). Con episodios de 15 días, trimestrales o anuales, el coste del dinero se puede meter en $\gamma$ si de verdad interesa, aunque "lo mismo no es tan importante": depende del problema. CASOS p.10 deja "¿Gamma?" como pregunta abierta justo después de definir steps y episodios en profit-seeking: $\gamma$ se decide una vez fijada la resolución temporal.
- **$\gamma$ no tiene por qué ser fijo** (T07 ~00:41). Al repasar los componentes señaló que, igual que el tamaño del buffer, incluso "el Karaoma" puede ser dinámico *(dudoso: lo más probable es $\gamma$; podría seguir refiriéndose al buffer)*, no siempre igual para todos los episodios: todo se puede ajustar y complicar. **Nota:** en esta práctica podría ligarse a la duración de cada episodio, $\gamma=f^{1/N}$ con el $N$ de ese episodio, lo que atenúa el problema de duración variable de §11.

**Nota (aplicación a la práctica, propuesta):** el step es un bin de 5 s y los episodios tienen 400-600 bins (33-50 min). Para que una recompensa (o la penalización final) al final del episodio pese al menos un 50 % desde el primer step: $\gamma=0{,}5^{1/600}=0{,}9988$; al 10 %: $\gamma=0{,}1^{1/600}=0{,}9962$; al 5 %: $0{,}995$. Con 0,99 el final pesa un 0,2-1,8 % desde el inicio, y con 0,95 nada. Si la recompensa es por paso y local (desviación del TWAP), basta un horizonte corto, del orden de la separación entre órdenes hijas (≈10 steps cuando la CARE es el 7,5-12,5 % de los bins); $\gamma=1$ también es defendible porque el episodio es finito. El ENUNCIADO pide motivar $\gamma$ en el ejercicio 2 (en el ejercicio 1 no hace falta). Opciones razonadas en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

**Valores en el material de clase** (Nota): $\gamma=0{,}99$ en `cliff-walking.ipynb` (celdas 9, 16, 22) y por defecto en `dqn.py` (BE) y en los agentes de los notebooks; $\gamma=0{,}995$ en CartPole tabular (`TD-q-learning.ipynb`, celda 7), en `deep.ipynb` (LunarLander, celda 3) y en `profit-seeking.ipynb` (celda 9). En CartPole (500 pasos máximo) 0,995 da $H=200$ y $0{,}995^{500}=0{,}08$: coherente con la regla de clase. En el notebook de la práctica $\gamma$ es un TODO (LAB celda 12).

**Nota (errata menor del LAB):** en entrenamiento el retorno descontado se acumula con `agent.gamma ** step` antes de incrementar `step` (primer término con $\gamma^0$, LAB celda 18), pero en la evaluación de la celda 20 se incrementa `step` antes, así que el primer término se descuenta con $\gamma^1$. Solo afecta a la métrica mostrada, no al aprendizaje.

*Fuente: APP p.75; CASOS p.10; T06 ~03:46-03:52; T07 ~00:41, ~01:07-01:10; LAB celdas 7, 11, 12, 18, 20; RLM; BE dqn.py; ENUNCIADO ej. 1-2.*

---

## 13. Resumen de ejemplos de clase

| Ejemplo | Qué ilustra | Fuente |
|---|---|---|
| Taxi autónomo y 50 € | Sin $\gamma$ da igual la ruta | T06 ~00:39 |
| Tres en raya / 10 pasos | $\gamma=1$ si no hay urgencia y el episodio es finito | T06 ~00:40-00:41 |
| Medias móviles 20/60 y el broker | Factibilidad: emular a quien ya lo hace | T06 ~02:32-02:33 |
| Andar de A a B, pingüino, brazos | Sparse no garantiza el *cómo*; shaping crea mínimos | T06 ~02:37-02:41 |
| Hamburguesa y pescado | Recompensas intermedias; un sistema por activo | T06 ~02:41-02:42; APP p.60 |
| Santander frente a Viscofan | Liquidez distinta, recompensas distintas | T06 ~02:42 |
| Correr hacia delante / escondite | Recompensas simples, conductas sorprendentes | T06 ~02:43-02:46 |
| Bonus por terminar antes | Termina demasiado pronto | T06 ~02:50 |
| Acantilado ($-1$ por paso; $-100$ al caer, citado aparte) | La urgencia ya está en el signo | T06 ~01:14, ~02:50-02:53 |
| Paraguas ($\pm100$ a ojo) | Magnitudes puestas a la ligera | T06 ~00:19, ~02:48; APP p.99-100 |
| Vender si cae un 10 % | Reglas en lugar de recompensas | T06 ~02:55 |
| TFM: 72 experimentos | Probar muchas recompensas | APP p.62; T06 ~02:57 |
| Botella de agua y el yo de 80 años | Lo lejano se diluye con $\gamma$ | T06 ~03:45 |
| 100/$\gamma$=1, 1.000/0,99, $10^6$/0,98 | El descuento es exponencial | T06 ~03:48 |
| 0,95$^{50}\approx$ 7 % | "Parece alto y no lo es" | T06 ~03:49; T07 ~01:10 |
| 3 % anual → por día y minuto | Calibrar hacia atrás según el step | T06 ~03:51-03:52 |
| Episodio intradía, "apagar a las 10" | $\gamma$ mal calibrado → opera al principio | T07 ~01:08-01:10 |
| Invertir pronto y quedarse sin dinero | Sesgo de recompensas intermedias | T07 ~01:20-01:21 |
| Retorno absoluto frente a alfa | Alineación recompensa-objetivo | T07 ~01:32-01:35 |
| Dibujo del VWAP global | Recompensa no factible | T07 ~03:35-03:39 |
| Petróleo frente a Repsol | Recompensa final = problema resuelto | T07 ~03:40 |
| Limitada frente a agresiva (+2/−3) | Recompensa local y frente al benchmark | T07 ~03:42-03:44, ~04:11-04:12 |
| TWAP como regla de un `if` | Reproducirlo "por recompensas"; Q-learning tabular 2×2 | T07 ~04:53-04:56 |
| Mucho peso a completar / al retorno | Ejecuta todo al principio / deja pendiente | T07 ~03:34 |

---

## 14. Aplicación a la práctica

- **Ej. 2** (replicar el TWAP, ENUNCIADO): la recompensa no necesita medir rendimiento (T07 ~04:58); lo natural es premiar seguir el plan (desviación del TWAP, §6; la regla del `if` de §4.4) con estado de tiempo y volumen restantes, decidir el valor de no comprar (`vol == 0`, §4.4) y motivar $\gamma$ y la escala (§9, §12).
- **Requisito del ej. 2** (ENUNCIADO): debe funcionar para distintos días, momentos temporales, duración del episodio y tamaño de la orden CARE dentro del rango esperado; basta uno de los dos activos (SAN o REP). Por eso la recompensa debe definirse sobre magnitudes normalizadas (tiempo y volumen restantes en %, desviación frente al plan en fracción de `vol_care`) y $\gamma$ debe motivarse para episodios de 400 a 600 pasos (§11, §12.2). No se pide réplica exacta ni validación: basta enseñar imágenes correctas en train, como el ejemplo de `images/twap_model/twap_reproduction_dqn.png` (ENUNCIADO).
- **Ej. 4, idea A**: si se busca mejorar VWAP medio o dispersión, la recompensa debe ser **local** y factible con el estado que se añada (§4).
- **Ej. 4, idea B**: penalización por volumen pendiente (§7) y por desviarse del ~90 % de la duración, tanto por agotar el tiempo como por terminar demasiado pronto (§6, §8), calibradas con log por criterio (sin olvidar registrar aparte la recompensa final, §7).
- **Ej. 4, objetivo libre**: el objetivo de la recompensa es libre, "el objetivo en base el que queráis da igual"; mencionó incluso hacer una recompensa distinta, "aleatoria", y analizar el comportamiento resultante (T07 ~04:29) *(dudoso: lectura posible, cualquier cambio de recompensa vale si se motiva y se muestra el comportamiento)*. Ver [notebook_partida.md](notebook_partida.md).
- Propuestas concretas, pros y contras y cifras en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md); código de `_compute_reward` y `_compute_done_reward` en [08_entorno_best_execution.md](08_entorno_best_execution.md); términos en [11_glosario.md](11_glosario.md).
