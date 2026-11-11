# RL para algoritmos de inversión (profit-seeking)

> **Fuentes:**
> - **CASOS** (`docs/raw/MIAX14/casos_practicos.pdf`), p.1-15: bloque "Profit-seeking Algorithms" completo (p.3-14) y portada de "Execution Algorithms" (p.15). Se han mirado las imágenes de las p.3, 5, 6, 8-13.
> - **APP** (`docs/raw/MIAX14/applying_RL_MIAX.pdf`), solo los ejemplos de bolsa de p.9-15 (Markov, RL frente a supervisado, impacto), p.49-53 (estados) y p.58-62 (recompensas). Imágenes de p.13, 14, 49, 60 y 62.
> - **T07** (`docs/raw/26 11 07.pdf`), clase del 07/11/2026: ~00:42-01:36 (notebook, profit-seeking completo), ~03:35-03:41 (recompensas factibles, analogía Repsol/BP) y ~03:58-04:02 (de la ejecución a la inversión).
> - **T06** (`docs/raw/26 11 06.pdf`), clase del 06/11/2026: ~00:02-00:06 (presentación y agenda), ~00:22-00:38 (Markov, supervisado frente a RL en bolsa), ~02:19-02:25 (steps), ~02:29-02:33 (objetivo y factibilidad), ~02:47-02:50 y ~02:55-02:57 (recompensas, reglas), ~03:21-03:26 (benchmarks), y de pasada ~02:58-02:59 y ~04:27-04:29.
> - **RLM** (`docs/raw/MIAX14/rl-material/notebooks/profit-seeking.ipynb`, celdas 0-18, sin salidas guardadas), más `docs/raw/MIAX14/rl-material/src/utils/bmeapi.py` y `docs/raw/MIAX14/rl-material/README.md`.
> - Para comparar: **LAB** (`docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`, celdas 12-21), **BE** (`docs/raw/MIAX14/best-execution/src/agents/dqn.py`), **T1** (`docs/raw/referencias/rl-trading-agents/`) y **T2** (`docs/raw/referencias/deep-rl-trading-agents/docs/teoria/`).
>
> **Convenciones:** "(T07 ~01:12)" = hora aproximada en la transcripción automática; todo lo que viene de ella es paráfrasis y "*(dudoso)*" marca lo que no se entiende bien. "(CASOS p.9)" = página de las diapositivas. "(RLM profit-seeking, celda 3)" = celda del notebook, contando desde 0. **Nota:** = contenido propio (cálculos, definiciones estándar, correcciones, conexiones). Los errores del material se señalan con su corrección razonada. La teoría general de cada componente (Markov, recompensas, gamma, exploración, entornos) está en los documentos 00-04; aquí se recoge lo que se dijo **aplicado a inversión** y se enlaza.
>
> Escrito el 10/11/2026.

---

## 1. Qué es un algoritmo profit-seeking y qué papel tuvo en el taller

**Idea.** El trading algorítmico ("estrategias programadas para ejecutar órdenes electrónicamente", CASOS p.3) tiene dos familias que en papers y conversaciones se confunden (T07 ~00:45:06):

| Familia | Qué decide | Objetivo | Ejemplo de clase |
|---|---|---|---|
| **Profit-seeking** (algoritmos de inversión) | Sus reglas determinan **qué** comprar o vender **y cuándo** | Ganar dinero (o, mejor dicho, maximizar la utilidad, §4) | Emite la orden "cómprame un millón de títulos de Santander hoy" (T07 ~00:45:42) |
| **Execution** (algoritmos de ejecución) | Cómo materializar una decisión de inversión ya tomada | Ejecutar la orden (al mejor precio posible) | Recoge ese millón de títulos y lo ejecuta "lo mejor posible" (T07 ~00:45:50) |

La diapositiva (CASOS p.3) ilustra la primera con un icono de billetes y monedas y la segunda con una diana. Las dos familias encadenan: la salida de un algoritmo de inversión es la entrada de uno de ejecución. El bloque profit-seeking ocupa CASOS p.3-14; la p.15 es la portada del bloque "Execution Algorithms" ([06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md)).

**Papel en el taller.** El profesor, Jesús Sanz del Real, CFA (CASOS p.2), fue alumno de la primera edición del máster y hoy trabaja en una consultora, sobre todo para banca; se define con un perfil mixto técnico-financiero, ni el más técnico ni el más financiero (T06 ~00:02:09-00:04:20). Antes trabajó en el Instituto BME haciendo algoritmos de inversión y un robo-advisor, además del caso de ejecución del taller (T06 ~00:03:27-00:03:53), que nació de su TFM: el director del máster le cambió un TFM sobre evaluación de fondos por uno para probar ML en los algoritmos de ejecución basados en reglas de BME; leía papers por la mañana y lanzaba experimentos por la noche, y de ahí viene buena parte de lo que cuenta (T06 ~02:58:10-02:59:32). Detalle en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md). Las diapositivas tienen transiciones que muestran la "solución" a las preguntas de discusión, y pide no estudiarlas antes de clase (CASOS p.1); en T07 lo repitió: prefiere que participen a que vean la solución (T07 ~00:44:32). Desde el primer día anunció que el día 2 aplicaría los componentes de RL a dos casos financieros: un algoritmo de inversión ("buscan un retorno para ti; bueno, la utilidad") y, sobre todo, algoritmos de ejecución, que son la práctica (T06 ~00:05:05-00:06:08). En T07 dedicó al profit-seeking unos 20 minutos "a muy alto nivel", porque a veces interesa más que la ejecución, y hora y media a la ejecución (T07 ~00:46:03-00:46:18). En la práctica real ocupó de ~00:46 a ~01:35.

**Nota:** la ejecución (familia 2) se desarrolla en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md) y [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md). Este documento no es materia directa de los ejercicios, pero el profesor lo usó para repasar **cómo se define cada componente de RL a partir del objetivo**, que es justo lo que se valora en la práctica (§9.4).

*Fuente: CASOS p.1-4, 15; T06 ~00:02:00-00:06:26, ~02:58:10-02:59:32; T07 ~00:44:08-00:46:27.*

---

## 2. Por qué es difícil: la complejidad de los mercados

**Idea.** Si el ajedrez, con más partidas posibles que átomos en el universo, ya lo juega un RL mejor que cualquier humano, los mercados son todavía más complejos y además cambian (CASOS p.5; T07 ~00:46:27-00:47:33).

**La diapositiva** (CASOS p.5): a la izquierda tres cifras, "Átomos universo: $10^{80}$", "Ajedrez: $10^{123}$" y "Mercados financieros: ??"; a la derecha, unas piezas de ajedrez (torre, caballo, peón) "VS" un gráfico de velas japonesas. Debajo, bajo el rótulo "Algunos problemas", cinco viñetas. Dijo que el ajedrez se había "resuelto" con RL; ante la objeción de un alumno (no está resuelto: no cabe en el universo la memoria para registrar todas las partidas), precisó que se refería a que hace tiempo que un algoritmo entrenado gana a cualquiera *(dudoso quién dice cada frase: la transcripción lo atribuye todo al aula)* (T07 ~00:46:45-00:47:22). Y subrayó que el ajedrez es "tremendamente determinista": solo importa la posición, mientras que en bolsa el mercado se mueve por factores que no ves (T07 ~00:48:43-00:49:12).

| Problema (CASOS p.5) | Lo que dijo en clase |
|---|---|
| **Infinidad de variables**, (no) conocidas, (no) observables | La varianza observada puede depender de "un millón de factores"; no se puede pretender capturarla toda. Es como el CartPole: le quitas una característica y deja de funcionar, aunque a veces se puede aproximar con las demás (T07 ~00:49:12-00:49:56) |
| **No podemos sobreajustar** el modelo de RL | Si lo ajustas a un histórico, a futuro no funcionará (T07 ~00:50:04) |
| **Problemas no estacionarios** | Un algoritmo puede ser bueno en un régimen o estado de mercado y malo en otro; lo que funcionó en el histórico cambia al ponerlo en producción (T07 ~00:48:05-00:48:22, ~00:50:10-00:50:19) |
| **Capacidad de absorción** de las estrategias | Algo puede funcionar con poco capital o en mercados poco líquidos, pero tiene un límite (T07 ~00:50:21-00:50:39) |
| **Publicación → eficiencia** | Si se publica, la gente la copia y la estrategia se agota: "sacan el libro y de repente ya no a nadie más le funciona" (T07 ~00:50:39-00:50:58) |

Dos ideas más de la explicación oral:
- **Contar dos veces la información.** Es tentador meter todas las noticias y señales posibles, pero quizá ya están recogidas en el precio y se descuentan dos veces. La teoría de mercados eficientes, en sus distintas versiones, dice que el precio recoge la información pública (con o sin sesgos); el margen está en diferenciarse del resto, que procesa la misma información con sus propios sesgos (T07 ~00:50:58-00:52:10).
- **Tener razón no basta.** Alguien sugirió usar opciones para ponerse corto; el problema es que cuestan dinero y, cuanto más tiempo hay que mantenerlas, más (T07 ~00:52:22-00:52:33, *(dudoso)*). De ahí la frase: citó (de memoria, "creo que así era la frase") la idea de que el mercado puede seguir irracional más tiempo del que uno puede seguir siendo solvente: una acción claramente sobrevalorada (un ejemplo que suena a Nvidia *(dudoso)*) puede seguir subiendo mientras estás corto, y una empresa "aburrida" infravalorada puede seguir por los suelos veinte años sin darte rendimiento absoluto (T07 ~00:52:22-00:53:24).

**Nota:** la frase se atribuye habitualmente a Keynes. El profesor llamó a la cifra del ajedrez "número de Shannon" (la transcripción dice "número sano de 10 … a 123", T07 ~00:46:40). El número de Shannon clásico es $10^{120}$ (complejidad del árbol de juego); los $10^{123}$ de la diapositiva y de la clase proceden de otras estimaciones. Para el argumento da igual.

*Fuente: CASOS p.5 (texto e imagen); T07 ~00:46:27-00:53:35.*

---

## 3. ¿RL o supervisado para invertir?

**Idea.** La pregunta clave es si **nuestras decisiones cambian los datos que veremos después** (APP p.15). Si no los cambian, basta un modelo supervisado; si los cambian (impacto en el mercado o en nuestro propio estado), hace falta RL y un entorno que modele las consecuencias. La regla práctica del profesor: si se puede hacer con supervisado, hacedlo con supervisado, porque RL exige construir un entorno (APP p.12; T06 ~00:29:14-00:29:24).

**Cuándo encaja RL** (APP p.12; T06 ~00:27:14-00:29:52): **decisiones secuenciales** (como en el ajedrez, muchas decisiones seguidas en cada partida, frente a decidir el paraguas de un solo día, que es supervisado), **interacción con el entorno** (las acciones llevan por caminos distintos) y **ruido en la decisión**: un experto propone mover el caballo, otro la reina, y no hay una etiqueta clara de quién lleva razón. La diapositiva añade las consecuencias: no independencia, acciones con efecto en el entorno y el problema de asignación del crédito. Un alumno señaló que el supervisado aspira como mucho a reproducir el patrón de las etiquetas, y el profesor lo validó: sin etiquetas, RL puede llegar a comportarse mejor que quien las habría puesto (T06 ~00:27:14-00:27:47). Es la misma idea de §4.5 (emular a un operador y quizá superarle). Un problema supervisado (el ejemplo de si lloverá, con humedad y demás características y etiquetas del pasado) también se puede resolver con RL, pero exige diseñar las recompensas, que cuesta más (T06 ~00:29:24-00:29:52). La teoría está en [00_repaso_rl.md](00_repaso_rl.md).

APP p.13 abre la discusión financiera con la pregunta "¿Ejemplos financieros? ¿Supervisado o RL?" sobre un gráfico de velas que se mueve en rango, acotado por dos líneas discontinuas horizontales arriba y abajo; la respuesta se desarrolla en p.14-15.

**El gráfico del impacto** (APP p.14). Dos paneles con precio (80-110 €) del día 1 al 16:
- *Izquierda, "Supervisado: precio histórico"*: la serie baja de 100 a un mínimo de **88**, sube a un máximo local de **104** (etiquetado), cae a ~98 y acaba cerca de 105. Las etiquetas se ponen sobre el histórico: "Etiqueta 1: compra" en el mínimo y "Etiqueta 0: vende" en el máximo. Supone que nuestras órdenes no mueven el precio.
- *Derecha, "RL: precio si compramos en el mínimo"*: la curva "Histórico" (gris) y la curva "Con impacto" (azul). Al comprar en el mínimo, este sube ("Compramos: el mínimo sube") y toda la serie posterior cambia: el máximo llega antes y la serie acaba por debajo del histórico (≈96 frente a ≈105). Las etiquetas ya no valen; RL modela ese impacto.

El profesor lo explicó así: con capital relevante, si compras en ese punto no habrías visto ese precio (lo habrías subido) y además impactas lo que viene después; las consecuencias hay que modelizarlas en el entorno, y "si tu entorno modeliza perfectamente las consecuencias, lo tienes todo hecho" (T06 ~00:34:49-00:35:45, paráfrasis).

**El impacto también es interno.** El estado se define con variables extrínsecas (externas) y privadas: la cartera, la concentración. Si ya tienes BBVA, quizá no te interesa Santander aunque sea buen activo, y la etiqueta "compra Santander" deja de ser correcta para ti; cada compra te cambia el estado (T06 ~00:35:48-00:36:48). Con posiciones pequeñas y sin importar la cartera, supervisado puede bastar (T06 ~00:37:19-00:37:53).

| Supervisado (las decisiones no cambian los datos) | RL (sí los cambian) |
|---|---|
| Detección de fraude (cada operación tiene su etiqueta) | Ejecución de órdenes grandes: cada tramo mueve el precio del siguiente |
| Previsión de volatilidad o VaR: predecir el riesgo no mueve el mercado | Market making: las cotizaciones y el inventario cambian el flujo de órdenes recibido |
| Señales con posiciones pequeñas en activos líquidos: impacto despreciable | Construcción de carteras y costes: cada compra y rebalanceo condiciona las siguientes (impacto interno) |

*(APP p.15.)*

**Si hay patrón y le das la variable, lo aprende.** Un alumno contó que en una práctica anterior *(dudoso: parece un taller previo de comprar/mantener/vender)*, con datos aleatorios el agente aprendía a no hacer nada, y con datos reales funcionaba peor que con los generados. El profesor: los precios reales se mueven por actores (una institución que entra a vender) que el estado no recoge. Un RL bien diseñado aprende un patrón si el estado contiene las variables que lo determinan; si falta la clave, no lo aprende ni RL ni supervisado. Ejemplo: "baja los lunes y sube el resto de la semana"; con un millón de características pero sin el día de la semana no se aprende, y no se arregla con una red más grande. Variables correlacionadas con la omitida (la racha, si ayer subió) permiten "rascar" algo (T06 ~00:30:06-00:33:55).

**Nota:** es lo que se vio en el Taller 1 con series sintéticas: en el mercado browniano (sin patrón) y en el de regímenes (el estado no ve el régimen), Q-learning y Sarsa acabaron en "no operar" (resultados recogidos en T2 `notebook_partida.md` §5.3; para redes, el Taller 2 lo daba como lo esperable, sin resultado medido allí). Para la teoría (componentes, Markov, cuándo usar RL) ver [00_repaso_rl.md](00_repaso_rl.md).

*Fuente: APP p.12-15 (texto e imágenes de p.13 y 14); T06 ~00:26:56-00:38:04.*

---

## 4. El objetivo: dinero no es utilidad

### 4.1 Aterrizar la idea

**Idea.** Antes que nada hay que definir muy concretamente qué problema se quiere resolver, porque "a partir de aquí van a salir todos y cada uno de los componentes" (T07 ~00:54:00-00:54:27). "Ganar dinero en bolsa" no basta: depende del riesgo, del horizonte y de cada persona (T06 ~02:30:07-02:30:28). Si el objetivo está bien aterrizado, las recompensas son casi una consecuencia; si es complejo, hacen falta *proxies* (T06 ~02:30:28-02:30:47). Tiene que poder escribirse con una fórmula: concreto, objetivo, factible y no abstracto (APP p.58; T06 ~02:31:45-02:31:57). El profesor dijo que cambiará esa diapositiva en la próxima edición, porque hoy también se empiezan a evaluar cosas abstractas como la creatividad (T06 ~02:30:00, ~02:31:02-02:31:13).

**Dinero ≠ utilidad** (CASOS p.6). La diapositiva pregunta "¿Ganar dinero?" (con un icono de saco de dinero) y responde: tu objetivo y tus recompensas tienen que ir destinadas a **maximizar tu utilidad**; hay que definir el objetivo o los factores que afectan a tu utilidad final. Ejemplo:

$$
U = \mathbb{E}[R] - \tfrac12\,A\,\sigma^2
$$

con $A$ el coeficiente de aversión al riesgo. El profesor lo presentó como "puramente ilustrativo": la utilidad es el retorno esperado ponderado por la aversión al riesgo, midiendo el riesgo como la volatilidad que se sufre mientras se está invertido; y añadió que incluso esto es todavía bastante genérico (T07 ~00:54:40-00:55:10). Lo conectó con RL: el agente maximiza la suma de recompensas descontadas, así que esa suma tiene que representar tu utilidad (T07 ~00:54:32).

**Nota (ejemplo numérico propio):** con $\mathbb E[R]=8\,\%$ y $\sigma=20\,\%$ anuales, un inversor con $A=2$ tiene $U=0{,}08-0{,}5\cdot2\cdot0{,}04=4\,\%$; con $A=4$, $U=0$: le da igual esa inversión que no invertir. Dos agentes con el mismo dinero esperado pueden tener objetivos opuestos.

### 4.2 Catálogo de objetivos

CASOS p.7 ("¿Más objetivos?") y la explicación (T07 ~00:57:52-00:59:33). El profesor los llamó "más bien títulos" que objetivos, porque siguen siendo genéricos (T07 ~00:58:54-00:59:05).

| Objetivo (CASOS p.7) | Para quién o para qué (T07) | Definición habitual (**Nota**) |
|---|---|---|
| Maximizar retorno **absoluto** | Quien quiere ganar en cualquier mercado | $R_T=\prod(1+r_t)-1$ |
| Maximizar retorno **relativo** | Respecto a un índice o a otro activo; elegir mejor que otras soluciones | $R_T-R^{bench}_T$ |
| Maximizar **Sharpe** | Para luego apalancar la estrategia | $\dfrac{\bar r-r_f}{\sigma_r}$ |
| % de **trades positivos** | (solo en la diapositiva) | aciertos / operaciones |
| Maximizar **IR** (information ratio) | Retornos frente al índice lo más estables posible en el tiempo (T07 ~00:58:24-00:58:28, transcripción confusa) | $\dfrac{\overline{r-r^{b}}}{\sigma(r-r^{b})}$ |
| Minimizar **TE** (tracking error) | Simplemente seguir al Ibex 35 | $\sigma(r-r^{b})$ |
| Minimizar **drawdown** | "Estoy a punto de jubilarme y nunca quiero un −20 %" (T07 ~00:58:28-00:58:48) | $\max_t\big(1-V_t/\max_{s\le t}V_s\big)$ |
| Combinar y/o establecer **pesos** | Ponderar riesgo, retorno, etc. hasta tener "bien lo que yo quiero" (T07 ~00:59:05-00:59:24) | p. ej. $U$ de §4.1 |
| **Condiciones de operativa**, **restricciones** (p. ej. cortos), **sistemas de control** | "No quiero nunca operar en corto", "quiero estos sistemas de control puestos" (T07 ~00:59:24-00:59:33) | reglas fuera de la recompensa (§8) |

### 4.3 Del objetivo al planteamiento del problema

**Idea.** Un mismo objetivo ("ganar dinero con esa fórmula") admite planteamientos muy distintos, y el planteamiento es la "solución íntegra de principio a fin" (T07 ~01:00:26-01:00:58). El ejemplo de clase (T07 ~00:55:56-00:56:46):
1. Un algoritmo mira **activos individuales** y, para cada uno, la tarea no es predecir el precio ni el rendimiento absoluto, sino una **clasificación binaria** de "sentimiento" positivo o negativo.
2. Con esas puntuaciones se hace un **ranking** de activos.
3. Otro algoritmo, con el sentimiento y las **correlaciones** entre activos, optimiza una cartera diversificada.
4. Se prueba toda la cadena sobre el histórico.

La alternativa "heavy" sería predecir el precio de mañana: si lo consigues, ganar dinero es trivial, pero el problema es mucho más difícil que saber cuál de dos activos lo hará mejor (los dos pueden bajar mañana, pero uno menos que el otro: es alfa relativo) (T07 ~00:57:03-00:57:42).

### 4.4 Objetivos específicos y relativos: reducir la varianza a explicar

**La diapositiva** (CASOS p.8): "Objetivos: Específico + Relativo = Mejores resultados", con características y recompensas "más apropiadas".

**El ejemplo de Repsol** (CASOS p.9, con el logo de Repsol): "Repsol sube 5 % en la semana" se descompone en **economía (70 %)**, **sector (25 %)** y **específico (5 %)**. Predecir ese 5 % total obliga a dar información de cada factor, con mucho ruido en las predicciones. Predecir el rendimiento relativo es menos "interesante" pero más "fácil" (las comillas son de la diapositiva).

Explicación oral (T07 ~00:59:38-01:02:58):
- Predecir si la bolsa sube mañana depende de infinidad de factores. Si solo es Santander, se reducen; si es **Santander frente a BBVA** el año que viene, más todavía.
- Los porcentajes son ilustrativos: "más o menos" el 70 % depende de lo que haga la economía esa semana, una parte muy importante del sector y una minoría del riesgo específico de Repsol.
- Si el universo son solo petroleras y el petróleo afecta a todas por igual, el petróleo ni se considera. Pero si se compara Repsol con una petrolera que cubre su exposición al petróleo (el nombre sale deformado, "Vitris/Bridges/Britis Petroleum"; probablemente **BP**, *(dudoso)*), la diferencia entre ambas depende justo de la volatilidad del petróleo, porque Repsol "se la juega siempre". El ejemplo es del profesor y no se presenta como dato real.
- Aun así, ni "menos interesante" es cierto (si funciona, está muy bien) ni "más fácil" lo es del todo: simplemente se pasa de lo imposible a algo que "puede empezar a tener sentido".

**Nota:** los porcentajes hablan de **varianza explicada** (así lo dijo, "reduces la varianza a explicar", T07 ~01:00:26), no de que el +5 % se reparta en 3,5 + 1,25 + 0,25 puntos cada semana. La lectura práctica: un objetivo relativo elimina del problema los factores comunes (economía, sector) y deja que el estado se concentre en lo que diferencia a los activos.

**Pregunta de un alumno: rankings con variables macro** (T07 ~01:16:07-01:18:19). Un grupo quería rankear unas 500 empresas *(dudoso: "7 500" o "3 500" en la transcripción)* con señales mensuales y variables macro como el petróleo, que solo afecta a algunas y mete ruido en las demás. Respuesta:
- Hacer una **capa previa por sectores**: saber si las tecnológicas lo harán mejor o peor ya explica en torno al 50 % de la varianza (cifra ilustrativa).
- Cada empresa tiene exposiciones mezcladas a varios sectores (se citan porcentajes sueltos, *(dudoso)*), y para el petróleo hay que preguntarse si afecta a todas por igual; si es así, se olvida.
- **Rankings de rankings**: aislar primero petroleras, tecnológicas..., e ir aplicando capas.
- **Subsistemas** o agentes especializados que interactúan (uno bueno en análisis fundamental, otro en macro...), en referencia a algo comentado el día anterior *(dudoso)*.

### 4.5 Factibilidad, benchmarks y objetivos numéricos

- **¿Alguien lo hace ya?** Un broker ejecuta bien órdenes con información conocida; si un operador gana con dos medias móviles de 20 y 60 días y le das al RL esas dos medias y las recompensas adecuadas, emulará su estrategia. Pero entonces el objetivo es **emularle**, no ganar dinero; queda abierta la posibilidad de superarle (T06 ~02:32:05-02:33:44). Desarrollado en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).
- **Ponerle números.** Hay que saber a qué se aspira para saber a qué distancia se está: *baselines* y *benchmarks* según el objetivo, como batir al Ibex 35, índices de referencia, algoritmos clásicos sin ML, o un umbral de viabilidad ("necesito rascar 20 pb o no lo pongo en producción"), y averiguar qué consiguen otros. Empezar con objetivos sencillos y subirlos poco a poco, "el triciclo" (T06 ~03:22:36-03:24:31). El contexto (exploración y el mapa de Madrid) está en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

*Fuente: CASOS p.6-9 (texto e imágenes); APP p.58; T07 ~00:53:35-01:03:03, ~01:16:04-01:18:19; T06 ~02:29:43-02:33:44, ~03:21:38-03:24:31.*

---

## 5. Steps, episodios y gamma

### 5.1 Steps y episodios

**Idea.** En bolsa no hay entorno importado ("de Microsoft o de las prácticas"): hay que construirlo uno mismo (T07 ~01:03:25-01:03:36), y step y episodio salen del objetivo (CASOS p.10; T07 ~01:04:30-01:04:42).

**Método.** Primero, qué es realmente lo que se quiere; después, estructurarlo todo hasta que cobre sentido, y solo entonces materializarlo. Se puede hacer antes un poco de exploración para ver si las ideas tienen sentido; el planteamiento evita ir a ciegas (T07 ~01:04:42-01:05:29; transcripción muy ruidosa, *(dudoso)* en detalle).

| Componente | CASOS p.10 | Explicación |
|---|---|---|
| **Step** | Diarios, minutos... Afecta al **tiempo de la orden**, al **impacto de mercado** y a la **definición de características** | Cada cuánto quieres poder reposicionarte: un fondo que solo puede reorganizar la cartera una vez a la semana tiene steps semanales ("los viernes") (T07 ~01:03:47-01:04:06) |
| **Episodio** | **X tiempo** o **X condición** (p. ej. cuando deshacemos posiciones) | Un buen episodio es un buen **horizonte temporal**: si buscas resultados trimestrales, episodio trimestral (T07 ~01:03:36-01:03:47) |

**Criterios del step** (T06 ~02:22:46-02:24:45; T07 ~01:05:34-01:06:18):
- **Temporal:** cada hora, cada día, cada trimestre.
- **Por eventos o condiciones:** cada vez que se ejecuta cierto volumen en el mercado, cuando ocurren ciertos eventos, cuando yo ejecuto o vendo algo.
- **Mixto:** "si ocurre esto, o yo hago esto, o pasan 5 s". Por ejemplo, decidir una vez al día pero atender un evento intradía relevante, como quien recibe el aviso de que hay que mirar algo porque ha pasado X (T06 ~02:24:28-02:24:45).

Depende de qué información necesita el estado y con qué frecuencia se refresca (T06 ~02:24:14-02:24:28). **Regla práctica:** pregúntate cada cuánto te gustaría a ti, como persona, recibir la información, porque con RL muchas veces se intenta sustituir el trabajo que hace una persona (T06 ~02:24:45-02:25:04). Un alumno lo definió como el periodo tras el cual el modelo debería recibir información nueva; en Breakout bastan 24 fotogramas por segundo frente a 30, 60 o 120 (T06 ~02:20:39-02:22:38).

**No hace falta el milisegundo** salvo en alta frecuencia, donde se explotan ineficiencias temporales del libro y hay que estar físicamente cerca de los sistemas de órdenes. Con medias móviles de 20 y 60 días no hace falta. Además, los libros "en tiempo real" que muestran las plataformas siempre tienen una frecuencia de refresco y se saltan pasos, y un humano tarda segundos en hacer clic (T07 ~01:06:18-01:07:32; T06 ~02:23:57-02:24:14).

**Inicio de los episodios.** Un alumno planteó un ETF que cotiza 24 h; el profesor advirtió que usar todo el histórico del S&P 500 como un único episodio que siempre empieza en el mismo día enseña algo como "compra el día 1 y mantén": no hay diversidad de episodios, y conviene empezar en puntos distintos para que el resultado generalice (T06 ~04:27:38-04:29:11, *(dudoso)* en detalle). Es el mismo consejo del Taller 2, "empezar cada episodio en fechas aleatorias" (T2 `08_trading_en_clase.md` §10), y lo que hace el notebook de clase (§10.3). Ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

### 5.2 Gamma en inversión

**Idea.** CASOS p.10 deja "¿Gamma?" como pregunta abierta. En clase (T07 ~01:07:32-01:10:38):
- Si el episodio es intradía (2 horas o un día) y quieres que el algoritmo busque las mejores oportunidades a lo largo de toda la sesión (te da igual conseguir el retorno a las 10 o a las 11, siempre que lo tengas al cierre), necesitas **gamma alto**. Un gamma bajo da más peso a lo inmediato, y el problema se agrava con episodios largos.
- El caso contrario: si quieres que a las 10 de la mañana ya esté hecho y poder "apagar el ordenador desde casa" y disfrutar del día, te interesa dar más peso a lo inmediato (gamma más bajo). Si te da igual conseguir el retorno a las 10 o a las 11, siempre que lo tengas al cierre (17:00), gamma alto (T07 ~01:08:25-01:08:56).
- Con episodios trimestrales o anuales se podría pensar en el coste del dinero como descuento *(dudoso)*.
- **Si se calibra mal**, el agente "va a operar casi siempre al principio".
- Se calcula **hacia atrás**: cuánto dura el episodio, cuántos steps tiene y cómo actúa el descuento. Le salió un valor que la transcripción da como "0,99 9 ocho" (0,9998 o 0,998, *(dudoso)*).
- Recordatorio del día anterior: $0{,}95^{50}\approx7\,\%$ (exacto: 7,7 %). "Parece que es muy alto y no es muy alto."

**Nota (cálculo propio).** Si se quiere que una recompensa al final del episodio de $N$ steps conserve un peso $w$, $\gamma=w^{1/N}$:

| Caso | $N$ | $\gamma$ para $w=0{,}5$ | $\gamma$ para $w=0{,}75$ |
|---|---|---|---|
| Trimestre con step diario | 63 | 0,9891 | 0,9954 |
| Año con step diario | 252 | 0,9973 | 0,9989 |
| 2 h con step de 5 s | 1.440 | 0,99952 | 0,9998 |
| Sesión de 6,5 h con step de 5 s | 4.680 | 0,99985 | 0,99994 |

El 0,9998 de clase encaja, por ejemplo, con 1.440 steps y $w=0{,}75$; 0,998 dejaría solo un 5,6 % de peso al final de esos 1.440 steps. El notebook de clase usa $\gamma=0{,}995$ con steps diarios (horizonte efectivo $1/(1-\gamma)=200$ días; $0{,}995^{250}\approx0{,}29$). El método completo, y el caso de la práctica (episodios de 400-600 pasos), están en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md) y [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

*Fuente: CASOS p.10; T07 ~01:03:03-01:10:38; T06 ~02:19:53-02:25:04, ~04:27:38-04:29:11.*

---

## 6. El estado: características para invertir

### 6.1 Qué entra en el estado

**Markov en bolsa** (APP p.11; T06 ~00:24:37-00:26:51). El estado no es el precio de hoy. No hay que confundir la propiedad de Markov con no pasar información del pasado: pasar los últimos 30 retornos o métricas con ventana de 30 días no la rompe, y de hecho es necesario. Igual que en Breakout dos fotogramas dan la dirección de la bola. Un alumno lo razonó así: en el estado de hoy puedo conocer los valores del pasado. El profesor insistió en **normalizar las características para que sean comparables entre activos**, en lugar de usar precios en bruto (T06 ~00:25:15-00:25:25). La teoría está en [00_repaso_rl.md](00_repaso_rl.md).

**El ejemplo de BP** (APP p.49 y 53). Un estado es un vector de características. La imagen muestra "Day 3 ($S_3$)" con el logo de BP y la pregunta "¿vendemos o mantenemos?", y una tabla: **Retorno 7 %, Std 16 %, Beta 1,2, Posición 1 (largo)**. Las características pueden ser continuas o discretas y **privadas** (la posición) o **externas** (retorno, volatilidad, beta); también se pueden usar imágenes. En bolsa hay que pasar "la información necesaria para invertir", y la ventaja de las máquinas es que procesan más características (APP p.50). Los estados deben cumplir Markov, contener la información necesaria ("meternos en el papel"), estar estandarizados y ser estacionarios, y conviene pensar en reducir la dimensionalidad (APP p.53). Detalle en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

### 6.2 Las características dependen del objetivo y del horizonte

Un algoritmo intradía recibe datos de minutos; uno trimestral, no: "todo va a estar interconectado" (T07 ~01:10:38-01:10:53). CASOS p.10 da tres ideas, bajo el rótulo **"Generador alpha"** y con el icono de una matriz de correlaciones (círculos de tamaño y color variables, azules y rojos):

| Fuente de alfa (CASOS p.10) | Explicación (T07 ~01:10:57-01:13:06) |
|---|---|
| **Rewarded factors** | Factores que a largo plazo se asocian a más retorno: *momentum* (lo que mejor ha funcionado), baja volatilidad, *value* (ratios financieros) |
| **Mispriced / oversold** | *Mispriced*: infravalorado o sobrevalorado según el análisis fundamental. *Oversold*: sobrecomprado o sobrevendido, concepto más de análisis técnico. Depende de "en qué mundo os mováis" |
| **Riesgo específico** | Lo que de verdad mueve a "vuestro Repsol o vuestro Santander": los tipos de interés para Santander (con buenos tipos, en el corto plazo suele ir mejor en bolsa); la cobertura del petróleo para una petrolera (si el petróleo sube y estás cubierto, te va mejor que a tus competidores, y al revés si baja) |

**Con varios activos, estandarizar las entradas** (CASOS p.10). En el notebook de clase se hace con rankings normalizados (§10.2).

### 6.3 Conocimiento de negocio: el modelo no es magia

"El modelo no va a ser la magia negra": no se le pasa la información en bruto diciendo "haz tu magia". Hay que dársela trabajada con conocimiento de negocio, y eso es "casi condición necesaria" (T07 ~01:13:06-01:13:36). Ejemplo de su trabajo, un modelo de impago de préstamos al consumo (T07 ~01:13:49-01:15:54):
- Se procesan todas las transacciones del cliente, categorizadas por sistemas propios (una persona media en España tiene entre 300 y 700 transacciones al año con tarjetas y cuentas; aparecen casinos, por ejemplo).
- Se podrían pasar en bruto para que el sistema genere relaciones; o se construyen variables como la tasa de ahorro o el gasto en casinos.
- Quien se dedica a riesgos sabe que una variable de **recobros** (si tiene recobros, o qué porcentaje) es muy informativa y ayuda mucho al aprendizaje.
- Matiz: también se está explorando mucho pasar datos en bruto para no meter los sesgos del analista de riesgos tradicional y dejar que el algoritmo encuentre relaciones más complejas.

### 6.4 El estado interno: posición y cartera

Cuando las acciones son incrementales (§8), el estado **tiene que incluir cuánto llevas en cartera**: "información interna". Tus decisiones cambian tu estado; por eso las etiquetas supervisadas no son fáciles aquí (con un 4 % quizá quiero subir un 1 %; con un 6 %, ya no) (T07 ~01:24:35-01:25:07). Se pueden añadir características como el **riesgo esperado por concentración** de aumentar o reducir un 1 % (p. ej. si ya tienes mucha banca) y el **sentimiento** positivo o negativo calculado por otro subsistema supervisado; con las dos bien normalizadas, el algoritmo puede decidir si aumentar la exposición (T07 ~01:25:16-01:26:42).

*Fuente: APP p.11, 49-53 (texto e imagen de p.49); CASOS p.10 (texto e imagen); T06 ~00:22:58-00:26:56; T07 ~01:10:38-01:16:04, ~01:24:35-01:26:42.*

---

## 7. Recompensas

**Idea.** Si el objetivo es muy específico y responde a una fórmula, la recompensa sale de él, al final del episodio o con *reward shaping* (T07 ~01:19:09-01:19:36; APP p.59: lo mejor es que la recompensa sea el propio objetivo, cuando es posible).

**La diapositiva** (CASOS p.11; una escalera con flechas que sube hacia un trofeo). Ejemplo: objetivo **mejorar un índice de referencia (p. ej. el Ibex 35)**; recompensas en función del **alfa obtenido**, del **information ratio**, de la **mejora del Sharpe**... Además: "puedes dar recompensas aunque no estés invertido" y "cada step o final del episodio".

**Lo que añadió en clase** (T07 ~01:19:36-01:21:12):
- Ha visto mucho recompensar **solo si estás invertido**. Pero no estar invertido también tiene un **coste de oportunidad**: puedes evaluar tu situación en cada momento, actúes o no y estés o no invertido (también al mantener).
- Si hay problemas de *feedback* o de exploración, o se aborda la exploración o se dan recompensas intermedias según si los objetivos evolucionan en la buena dirección.
- **Sesgo de las recompensas intermedias:** si el objetivo es mejorar el alfa frente al índice y el agente ve buenos momentos, empieza a comprar y recibe recompensas positivas; puede que más adelante surjan oportunidades mucho mejores y ya no le quede dinero. Una solución es ponderarlo todo al final.

**Alineación con el objetivo: ¿comparar con el mercado?** (pregunta de un alumno, T07 ~01:32:04-01:34:26). La duda: si todo el mercado cae y el algoritmo actúa en la mejor condición posible, el retorno sigue siendo malo y la recompensa puede enseñarle que fue una mala decisión (T07 ~01:32:13-01:32:48; la transcripción dice "todo está lleno", *(dudoso)*, la lectura de una caída general es la que encaja con la respuesta). Depende del objetivo:
- **Retorno absoluto:** perder dinero está mal, aunque el mercado caiga. Hay quien busca un 2-3 % siempre positivo, "como un depósito"; y hay estrategias *market neutral* (largos y cortos a la vez) que aíslan el mercado y explotan solo la diferencia entre activos.
- **Alfa:** perder menos que el benchmark está bien ("he perdido mucho, pero los otros el doble"), y la recompensa lo tiene que reconocer.
- "Tiene que haber una alineación entre tus recompensas y todo lo demás."

**Recompensas factibles: el ejemplo de Repsol y BP** (T07 ~03:35-03:41, dentro del caso de ejecución). La base está en APP p.59: el agente intentará descubrir cómo funciona el sistema de recompensas y aprender las acciones que lo maximizan, así que hay que preguntarse si se puede conocer la acción óptima con el vector de características. En el ejemplo que dibujó de ejecución, el agente compra más en una bajada y va por delante del VWAP del mercado; luego el precio cae mucho más, con mucho volumen, y ya no le queda volumen para aprovecharlo, de modo que acaba peor que el mercado (T07 ~03:35:28-03:37:23). Una recompensa global ("al final de la ejecución, buen precio respecto al mercado") obliga a predecir precio y volumen futuros del mercado, y no se puede relacionar con las características aunque esté alineada con el objetivo. Hace falta una recompensa alineada **y predecible** con el estado: local. En inversión es lo mismo: si para elegir entre Repsol y el petróleo premias la rentabilidad del año, "ya tienes el problema resuelto" *(dudoso: puede querer decir que esa recompensa ya es la respuesta o que, si se pudiera aprender, el problema estaría resuelto; en ambos casos no es predecible con el estado)*. Lo aprendible es si **localmente** lo has hecho mejor o peor que la referencia (que la transcripción da como "Britis Protole", probablemente BP o el Brent, *(dudoso)*) (T07 ~03:40:27-03:41:04). El caso de ejecución (VWAP slippage global, local y frente al TWAP) está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

**Principios generales del día 1 aplicados a bolsa:**
- **¿Un mismo sistema de recompensas para todos los activos?** La diapositiva lo plantea con una barbacoa: "no es lo mismo cocinar" una hamburguesa que un pescado (APP p.60).
- **Magnitud relativa y signo.** La gente pone ±1 o ±100 "muy a la ligera" (APP p.61; T06 ~02:48:25-02:48:34). Hay que estimar cuánto puede recibir el agente por cada objetivo parcial; mantenerse en $[-1,1]$ puede ayudar a la estabilidad, porque las redes funcionan mejor con recompensas pequeñas (APP p.61; T06 ~02:48:46-02:49:09). Lo mismo vale para el signo. Si hay subobjetivos (rentabilidad y riesgo), su **ponderación** es clave, y mal calibrada aparecen mínimos locales como el robot que "solo mueve los brazos". Recomienda registrar cuánto recibe el agente por cada criterio y calibrar viendo su comportamiento (T06 ~02:49:09-02:50:27).
- **Probar muchas variantes.** Las mismas recompensas pueden dar resultados distintos según la semilla, y ligeras modificaciones, comportamientos totalmente distintos: hay que probar muchas recompensas y preguntarse qué ha causado cada comportamiento (APP p.61). APP p.62 lo ilustra sin texto: arriba, una captura con un centenar largo de carpetas de experimentos; abajo, un monitor de sistema tipo `htop` con todos los núcleos casi al 100 %. Detalle en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).
- **Simplicidad:** a veces lo mejor es recompensar solo al final si se cumple el objetivo; empezar por el triciclo antes que el Ferrari (T06 ~02:47:03-02:48:13).

**Nota:** con alfa diario como recompensa por paso (lo que hace el notebook de clase, §10.3), la suma de recompensas aproxima el alfa logarítmico acumulado durante los días cobrados ($\sum\ln(1+r^a_t-r^b_t)\approx\ln(V^a/V^b)=\sum[\ln(1+r^a_t)-\ln(1+r^b_t)]$ para retornos pequeños; en rigor es el log-crecimiento de una cartera larga-corta rebalanceada a diario, y solo cuenta desde $t+2$ tras la compra, errata 4 de §10.7), y es 0 fuera: el coste de oportunidad de §7 no se recoge. Todo lo general sobre recompensas está en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

*Fuente: CASOS p.11 (texto e imagen); APP p.59-62 (imágenes de p.60 y 62); T07 ~01:19:02-01:21:23, ~01:32:04-01:34:35, ~03:35:00-03:41:30; T06 ~02:47:03-02:50:34.*

---

## 8. Acciones y condiciones de liquidez

**La diapositiva** (CASOS p.12). Ejemplo de acciones discretas, con flechas: "=" para mantener, dos triángulos verdes o rojos para las fuertes y uno solo para las parciales.

| Acción | Significado |
|---|---|
| 0 | Mantener |
| 1 | Compra fuerte (100 %) |
| 2 | Venta fuerte (100 %) |
| 3 | Compra parcial (p. ej. 20 %) |
| 4 | Venta parcial (p. ej. 20 %) |

A la derecha: "Puedes añadir **condiciones de liquidez** para limitar el tamaño de la operación", por ejemplo $\text{vol}=\min(\text{cond},\ \text{acción})$; y otras condiciones: **apalancamiento** (bruto o neto), **posiciones cortas**, **mantener X tiempo** las decisiones activas, etc.

**En clase** (T07 ~01:21:23-01:27:51):
- Cuanto más simples y menos acciones, mejor: comprar, vender, mantener, ventas parciales.
- **Diálogo con un alumno** que quería limitar la agresividad del agente (que no encadene compras parciales): lo primero es decidir los **grados de libertad** que se le dan. Binario (comprado o vendido) con un tamaño fijo (p. ej. 5 % de la cartera), o un porcentaje "aparentemente continuo" entre 0 y 10 %.
- **Discretizar lo continuo:** acciones "aumentar 1 %", "reducir 1 %" o "mantener", preguntando varias veces: "¿quieres más Santander? Súbelo un 1 %... ¿más?... hasta el 7 %"; cuando dice mantener, se queda ahí. Exige el estado interno de §6.4.
- **Empezar siempre por el triciclo:** comprar, mantener, vender. Todo lo demás es ilustración de las posibilidades.
- **Condiciones** que se especifican fuera del agente: comprar un 5 % solo si hay volumen en el mercado, porque si no te barres el libro y consigues un precio muy malo (si la liquidez solo da para un 2 %, se compra un 2 %); apalancamiento nunca superior al 5-10 %; sin cortos; mantener las posiciones y operar poco (poca rotación).

**Reglas deterministas junto a la red** (T06 ~02:55:15-02:57:00). Lo que tenga que ser al 100 % de una forma se programa como regla y se deja fijo: p. ej. vender si el mercado cae un 10 %, aunque el agente hubiera aprendido que es momento de comprar, porque uno no quiere arriesgarse. El agente puede decidir con la red, con un conjunto de reglas o con ambas, y las reglas también sirven para acotar el espacio en el que se mueve durante el entrenamiento. Ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

**Nota:** una forma concreta de la condición de liquidez es $q_t=\min\big(q^{\text{acción}}_t,\ \kappa\cdot V_t\big)$, con $V_t$ el volumen negociado en el step y $\kappa$ la fracción máxima de participación. En la práctica de ejecución el mismo papel lo cumplen el tamaño de la orden en %ADV y el volumen disponible en el libro ([07_rl_para_ejecucion.md](07_rl_para_ejecucion.md)).

*Fuente: CASOS p.12 (texto e imagen); T07 ~01:21:23-01:28:01; T06 ~02:55:15-02:57:00.*

---

## 9. Análisis de resultados y validación

### 9.1 Lo que pide la diapositiva

CASOS p.13, "Resultados":
- Comparar contra el **benchmark fuera de train**, con métricas como Sharpe, alfa o IR.
- **Técnicas de validación** (bootstrapping, cross-validation, etc.).
- Según la agregación de los datos:
  - **Datos diarios:** pocos datos y menos impacto, así que cabe "¿considerar supervisado?".
  - **Mayor resolución:** hacen falta asunciones de ejecución, modelos matemáticos, penalizaciones (umbrales de viabilidad), calibraciones, etc.

### 9.2 Datos diarios frente a mayor resolución

(T07 ~01:28:01-01:32:04)
- Con datos diarios y poco impacto, el backtesting es mucho más sencillo y quizá ni haga falta RL: si se opera en las **subastas de cierre** cada día, podría ser un supervisado, o un sistema de reglas que optimice el peso de cada activo aunque haya estado interno (la cartera). Ejemplo: de las 500 empresas del S&P, un modelo da señal positiva a 50, y otra pieza decide cuáles entran en cartera.
- Con más materialidad hay que simular **si las órdenes se ejecutan** y su impacto (una orden grande en un activo poco líquido puede no ejecutarse).
- Los primeros algoritmos de trading con RL aparecieron en mercados de enorme liquidez, "mercados monetarios" según la transcripción (*(dudoso)*: puede referirse a los monetarios o a divisas). Allí no hay impacto en el mercado, solo en el estado interno, y lo entrenado sobre el pasado es más realista a futuro.
- En España, con poca liquidez, se impacta enseguida (la cifra de la transcripción, "1.000 €", es *(dudosa)*). Se puede **penalizar con un coste fijo** en lugar de simular el mercado con precisión: "con este volumen sé que pago unos 20 puntos básicos" de impacto, cifra que la gente del mercado conoce para un tamaño dado. Se puede empezar sin costes y luego **estresarlos** (qué pasa si a cada operación le sumo 20 pb de impacto), y **calibrar** el modelo comparando con órdenes reales en producción. El coste fijo es una simplificación de lo anterior, es decir, de los modelos matemáticos y asunciones de ejecución de CASOS p.13 (T07 ~01:31:27-01:31:35). La calibración consiste en comparar cómo se comportaron las órdenes reales frente a lo que predice el modelo o entorno, y ajustar las hipótesis en función de eso (T07 ~01:31:38-01:31:57).

**Nota:** la transcripción dice "20 puntos básicos de forma diaria". Aplicado cada día a una cartera que se mantiene, serían $0{,}2\,\%\times252\approx50\,\%$ al año, lo que no tiene sentido para una estrategia de baja rotación. La lectura razonable es 20 pb por operación (o por día en que se opera) con ese volumen. Los 20 pb coinciden con el umbral de viabilidad de T06 ~03:24:13 (§4.5).

### 9.3 Dos lecturas de los resultados y el benchmark

(T07 ~01:34:35-01:35:30)
- Hay dos formas de presentar los resultados: la de **IA** (ha aprendido o no, técnicas de validación) y la **financiera** (funciona o no).
- Errores típicos: "no funciona porque perdemos", cuando ha batido al benchmark; o "gana dinero", cuando se comporta más o menos igual que el benchmark.
- Hay más técnicas, y "una comparación fija en el tiempo no es lo más adecuado"; backtesting y benchmarks se han visto en otros módulos.

**Nota:** en series temporales la validación cruzada debe respetar el orden temporal (*walk-forward* o bloques), y el bootstrap, remuestrear por bloques o por episodios. Comparar en varios periodos y activos es la forma de responder a la "comparación fija en el tiempo".

### 9.4 Qué se valora: el planteamiento

Respondiendo a la pregunta del ranking (§4.4), el profesor explicó qué valoraba él en los TFM (añadió que "ahora va a cambiar", *(dudoso)* si se refiere a su papel en los TFM o al criterio) (T07 ~01:18:23): el **planteamiento del problema** y seguir **pasos lógicos** en cada componente. "Podéis sacar un sobresaliente a pesar de que no consigáis resultados", porque se entienden las limitaciones de tiempo y recursos; lo que se quiere es que se demuestre el conocimiento adquirido y un buen planteamiento (T07 ~01:18:19-01:19:02). Lo enlazó con lo dicho el día anterior. Para la práctica, la fuente que manda es el ENUNCIADO; cómo se traslada a la práctica de ejecución está en [notebook_partida.md](notebook_partida.md).

*Fuente: CASOS p.13 (texto e imagen); T07 ~01:18:19-01:19:02, ~01:28:01-01:35:44; T06 ~03:24:13.*

---

## 10. El notebook profit-seeking de clase (`rl-material`)

### 10.1 Qué es y qué dijo el profesor

CASOS p.14 es una diapositiva que solo dice "Notebook". Al cerrar el bloque, el profesor se refirió al notebook "que os he enseñado al principio" (T07 ~01:35:39), es decir, el que mostró a ~00:42, antes de las diapositivas: el **notebook `profit-seeking.ipynb` de `rl-material`** (T07 ~00:42:35-00:44:00). Lo que dijo:
- "Esto ya no funciona": usaba una **API del Instituto BME** con datos del Ibex que dejó de estar activa para los alumnos. Los datos eran una especie de diccionario por día o por activo (no recordaba cuál era la primera clave).
- No se hace responsable de que esté bien calculado: lo hizo "en una tarde" sin ayuda de GPT, que todavía no existía *(dudoso)* (T07 ~00:43:18).
- Es un algoritmo de inversión con RL "muy, muy cortito", para ver los componentes y contrastarlos con la práctica.
- El estado tiene el Sharpe de "los últimos 15, 30 y 60 días o algo así", y se le pide al agente que optimice.

**Corrección:** el código usa ventanas de **10, 30 y 60** días, no 15 (RLM profit-seeking, celda 2). El notebook se guardó **sin salidas**, con un kernel Python 3.8.18, y no puede ejecutarse sin la API: no hay resultados que comentar.

### 10.2 Datos y características (`EnvData`, celda 2)

- `APIBMEHandler('IBEX', 'rl-algo')` descarga el cierre diario de todos los tickers del maestro del mercado y la serie `benchmark` (`bmeapi.py`: `get_ticker_master`, `get_close_data`, `get_close_data_ticker`). La clave de usuario está vacía en el código.
- `generate_data(start, end)` recorta por años y elimina los tickers sin datos en el periodo. Entrenamiento con **2017-2018**, test con **2019-2020** (celda 7).
- Retornos simples $r_{i,t}=P_{i,t}/P_{i,t-1}-1$, también para el benchmark.
- Para cada ventana $w\in\{10,30,60\}$, un "Sharpe" robusto por activo, $\text{sh}_{i,t}^{(w)}=\dfrac{\operatorname{mediana}(r_{i,t-w+1..t})}{\operatorname{std}(r_{i,t-w+1..t})}$, que se convierte en **ranking transversal** entre activos ese día y se divide por el máximo del ranking: un valor en $(0,1]$, donde 1 es el mejor activo del día.
- Un DataFrame por ticker con 7 columnas, que son el estado (`n_features = 7`). Solo se guardan los tickers con al menos 50 filas.

| Columna | Qué es | Escala (**Nota:** orden típico de un retorno diario, no medido) |
|---|---|---|
| `returns` | Retorno diario del activo | ≈ ±0,02 |
| `sh_10`, `sh_30`, `sh_60` | Ranking normalizado del "Sharpe" a 10, 30 y 60 días | $(0,1]$ |
| `median_ret` | Mediana transversal de los retornos de todos los activos ese día ("el mercado" sin pesos) | ≈ ±0,02 |
| `bench_ret` | Retorno del benchmark | ≈ ±0,02 |
| `position` | 0 fuera, 1 invertido (información privada) | {0, 1} |

**Lectura según las diapositivas:** los rankings son a la vez un *rewarded factor* (momentum ajustado por riesgo) y la forma de **estandarizar entradas entre activos distintos** (CASOS p.10); `position` es la información interna (§6.4); `bench_ret` y `median_ret` dan el contexto de mercado para un objetivo **relativo** (§4.4).

**Nota:** solo los rankings están normalizados. `returns`, `median_ret` y `bench_ret` entran como retornos diarios en bruto (celda 2), unos dos órdenes de magnitud por debajo de los rankings, lo que va contra "estandarizar entradas" (CASOS p.10; APP p.53; T06 ~00:25:15-00:25:25). Corrección: *z-score* con media y desviación de train, o rankings también para los retornos.

### 10.3 El entorno (`AlgoEnv`, celda 3)

- **Acciones:** `0` mantener, `1` comprar, `2` vender (el "triciclo" de §8), todo o nada sobre un activo.
- **Episodio:** en cada `reset` se elige un **ticker al azar** y un **inicio al azar** `init_pos ∈ [0, len−30)`, y el episodio va de ahí al final de los datos. Termina antes si el agente **vende estando invertido**: un episodio es, como mucho, **una operación de entrada y salida**. Es el episodio "por condición, al deshacer posiciones" de CASOS p.10, con inicio aleatorio (§5.1).
- **Recompensa:** el alfa logarítmico frente al benchmark del día siguiente, solo si se está invertido:

$$
r_t=\begin{cases}\ln\!\big(1+r^{\text{activo}}_{t+1}-r^{\text{bench}}_{t+1}\big) & \text{si posición}=1\\ 0 & \text{si posición}=0\end{cases}
$$

  Es la recompensa "en función del alfa obtenido" frente al índice, dada en cada step (CASOS p.11).
- **Cronología exacta** (con $t$ = `idx`):
  - `mantener` cobra $\alpha_{t+1}$ si ya se estaba dentro.
  - `comprar` calcula la recompensa **antes** de cambiar la posición (cobra 0) y marca `position = 1` desde la fila $t+1$, así que el primer alfa cobrado es el de $t+2$.
  - `vender` cobra $\alpha_{t+1}+\alpha_{t+2}$ y devuelve `done=True`.
  - El episodio también termina si `idx + 2 == len_episode` (fin de datos, tratado como terminal: errata 13 de §10.7).
- **Acciones redundantes. Nota:** comprar estando dentro equivale a mantener (cobra $\alpha_{t+1}$) y vender estando fuera también (cobra 0, sin `done`). Las acciones efectivas dependen de `position`, y por eso esa columna es imprescindible en el estado. La acción aleatoria ponderada genera muchas de estas acciones nulas.
- **Observación:** la fila `idx` del DataFrame del episodio (los 7 valores). No hay *look-ahead*: los rankings usan retornos hasta $t$ y la recompensa, retornos posteriores.
- **Acción aleatoria ponderada:** `action_sample` elige con probabilidades $(0{,}8;\,0{,}1;\,0{,}1)$ para mantener, comprar y vender: explora sin abrir y cerrar posiciones sin parar (ponderar las acciones aleatorias, ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md)).
- Interfaz `gym` antigua: `step` devuelve 4 valores `(obs, r, done, {})`, como en `rl-material` y en el LAB.

### 10.4 El agente (`DDQNAgent`, celda 4)

Es el mismo esquema que el agente de la práctica (BE dqn.py), detallado en [08_entorno_best_execution.md](08_entorno_best_execution.md); la teoría está en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md).
- **Red:** `Input(7) → Dense(64, relu) → Dense(64, relu) → Dense(3, linear)`, MSE y Adam. **Nota:** 4.867 parámetros. Ver la errata 3 de §10.7 sobre la línea duplicada.
- **Red objetivo** (`target_model`) copiada con `update_target()`.
- **Double DQN** en `learn()`: la red principal elige $a^*=\arg\max_a Q(s',a;\theta)$ y la objetivo la evalúa, $y=r+\gamma\,Q(s',a^*;\theta^-)$; si `s1 is None` (terminal), $y=r$.
- **CER** (*combined experience replay*) en `sample()`: `batch_size − 1` transiciones al azar sin reemplazo más la última añadida al buffer.
- **ε-greedy** con `env.action_sample()` como acción aleatoria (la ponderada de §10.3).
- Los valores por defecto del constructor (γ = 0,99, ε = 0,9, α = 0,01, buffer 50.000, 32 neuronas) no se usan: la celda 9 los sobrescribe (§10.5). `minus_factor` y `div_factor` se inicializan a `None` y no se usan. Ver también la errata 14 de §10.7.

### 10.5 Hiperparámetros y bucle (celdas 9-18)

| Parámetro | Valor |
|---|---|
| ε inicial / mínimo | 1 / 0,03 |
| Decaimiento de ε | Lineal por episodio: $(1-0{,}03)/(0{,}8\cdot3000)\approx4{,}04\cdot10^{-4}$; llega al mínimo en el episodio 2.400 |
| γ | 0,995 |
| α (learning rate de Adam) | 0,001 |
| Buffer / batch | 5.000 / 64 |
| Neuronas ocultas | 64 |
| Episodios | 3.000 (log cada 250) |
| `learn_after` | `batch_size` = 64: un paso de gradiente cada 64 steps |
| `update_target_after` | 10 × 64 = 640 steps |

1. **Llenado del buffer** (celda 11): 5.000 transiciones con `action_sample` antes de entrenar. **Nota:** con estas probabilidades, se tarda de media 10 steps en comprar y otros 10 en vender, así que unos 20 steps por episodio y unos 250 episodios en el buffer inicial.
2. **Entrenamiento** (celda 13): por episodio, `act → step → experience`; aprendizaje y copia de la red objetivo según los contadores globales; al acabar el episodio baja ε y se registran la recompensa, la recompensa descontada y los steps (más un `learn()` extra cada `n_log` episodios).
3. **Curva** (celda 14): media móvil de 100 episodios de la recompensa acumulada.
4. **Evaluación** (celdas 15-17): `agent.train = False` (greedy), 100 episodios por clave `train`/`test`, suma descontada de recompensas, medias y un *boxplot* horizontal test frente a train.
5. **Gráfico** (celda 18): para el último episodio evaluado, el valor del activo y del benchmark en base 100 (producto acumulado de $1+r$), con los días invertidos (posición desplazada un día) marcados en rojo sobre la curva del activo.

**Nota:** es **el mismo bucle que el de la práctica** (LAB celdas 12-21): `learn_after = batch_size`, `update_target_after = 10 * learn_after`, ε lineal por episodio con suelo, buffer prellenado, `history_rewards` y media móvil. Entender este notebook es entender el esqueleto del LAB.

### 10.6 De las diapositivas al código

| Concepto de clase | Cómo lo concreta el notebook |
|---|---|
| Objetivo relativo, mejorar un índice (CASOS p.8-11) | Recompensa = alfa logarítmico frente a `benchmark` |
| Rewarded factors (CASOS p.10) | Rankings normalizados del "Sharpe" a 10, 30 y 60 días |
| Estandarizar entradas (CASOS p.10; APP p.53) | Solo en parte: los rankings sí; los retornos van sin escalar (§10.2) |
| Información privada en el estado (APP p.49; §6.4) | Columna `position` |
| Episodio por condición, al deshacer posiciones (CASOS p.10) | `done=True` al vender |
| Diversidad de episodios, inicio aleatorio (§5.1) | Ticker e inicio al azar en cada `reset` |
| Acciones simples, el triciclo (CASOS p.12; T07 ~01:26:47) | Mantener, comprar y vender, todo o nada |
| γ alto para horizontes largos (§5.2) | 0,995 con step diario |
| Comparar fuera de train (CASOS p.13) | Pretende evaluar train y test por separado, pero no lo hace (errata 1 de §10.7) |
| Recompensar aunque no se esté invertido (CASOS p.11) | **No**: fuera de mercado la recompensa es 0 |
| Condiciones de liquidez, costes (CASOS p.12-13) | **No** hay costes de transacción ni límite de volumen |

### 10.7 Erratas y riesgos del notebook

| # | Dónde | Qué ocurre | Corrección razonada |
|---|---|---|---|
| 1 | Celda 15, evaluación | El bucle recorre `'train'` y `'test'`, pero **siempre usa `env`**, creado con `train_data` (celda 9). `test_data` se genera (celda 7) y nunca se usa: el "test" del boxplot es otra muestra de entrenamiento | Crear `env_test = AlgoEnv(test_data)` y elegir el entorno según la clave. El LAB sí usa los datos de test (celda 20: `BestExecutionEnv(dict_["test"], 60)`), pero evalúa un solo episodio y repite el desfase de descuento de la errata 7; conviene evaluar muchos episodios de test |
| 2 | `_sell` (celda 3) | `self.position == 0` es una comparación, no una asignación: la posición no vuelve a 0 y la columna `position` se reescribe a 1. Como el episodio termina, no rompe nada; pero `alpha += self._compute_alpha(lag=2)` solo es distinto de 0 **gracias al error** (con `position = 0`, valdría 0) | Usar `self.position = 0` y decidir conscientemente si la venta cobra uno o dos días (ver errata 4). Si se quisiera que vender no terminase el episodio, el error haría que el agente siguiera "invertido" |
| 3 | `network` (celda 4) | `capa = Dense(...)(inp)` aparece dos veces. La primera capa queda **colgada** y Keras no la incluye en el modelo; la red tiene 2 capas ocultas, no 3 | Borrar la línea duplicada (sin efecto). BE `dqn.py` ya no la tiene |
| 4 | Cronología de la recompensa (celda 3) | Comprar en $t$ empieza a cobrar en $t+2$; vender en $s$ cobra $s+1$ y $s+2$. Equivale a ejecutar la compra un día tarde y la venta dos días tarde (o a cobrar un día de más al salir) *(intención dudosa)* | Fijar una convención: decisión al cierre de $t$ y ejecución al cierre de $t$ (cobrar $r_{t+1}$ desde la compra hasta la venta inclusive) o al de $t+1$, igual para compras y ventas. No es *look-ahead*, pero añade ruido no predecible a la recompensa de vender |
| 5 | Mensajes de error (celdas 2 y 3) | `f"Invalid start {action}"` y `actions_fn.keys()` usan nombres no definidos: si se lanza el error, sale un `NameError` en vez del `ValueError` | `start`/`end` y `self.actions_fn` |
| 6 | `getQ` (celda 4) | Llama a `getQs(s)` sin el argumento `model`: daría `TypeError`. Código muerto | Pasar `self.primary_model` o borrarlo |
| 7 | Descuento en la evaluación (celda 15) | `step += 1` va antes de `gamma ** step * r`, así que la primera recompensa ya se descuenta con $\gamma^1$ (en el entrenamiento, con $\gamma^0$) | Incrementar después; efecto mínimo. Además, la métrica es la recompensa **descontada**, no el alfa acumulado ni el Sharpe (CASOS p.13 pide Sharpe, alfa o IR) |
| 8 | `_compute_episode` (celda 3) | `self.episode` es un corte del DataFrame original y se escribe por `.values[...] =`. Con pandas sin *copy-on-write* modifica los datos originales (inocuo, porque se pone a 0 en cada `reset`); con *copy-on-write* (pandas 3) puede no escribir o fallar *(no verificado)* | Trabajar sobre `.copy()` o llevar la posición en un array aparte |
| 9 | Sin costes ni liquidez | Entrar y salir no cuesta nada. Con alfa diario, el agente puede aprender a operar ruido | Restar un coste por operación (p. ej. los 20 pb de §9.2, estresados) |
| 10 | Universo de activos (`bmeapi.py`) | El maestro de tickers puede ser la composición **actual** del Ibex *(no verificable sin la API)*, lo que daría sesgo de supervivencia | Usar la composición histórica en cada fecha |
| 11 | Recompensa 0 fuera de mercado | Quedarse fuera nunca penaliza, y si el agente greedy no compra, el episodio dura hasta el final de los datos | Coherente con un objetivo de alfa "cuando invierto", pero no recoge el coste de oportunidad de §7 |
| 12 | Import sin uso | `Dropout` no se usa | Cosmético |
| 13 | Fin de datos tratado como terminal (celdas 3-4, 11 y 13) | Con `idx + 2 == len_episode`, `step` pone `done=True`; `experience` guarda entonces `s1=None` y `learn` usa $y=r$, sin *bootstrap*. Es un **truncamiento** (se acaban los datos), no un final del MDP: el valor de seguir invertido se corta a 0, y el estado no recoge cuánto falta. Con $\gamma=0{,}995$ el sesgo de $Q$ a la baja cerca del final de los datos no es despreciable | Distinguir `terminated`/`truncated` (API de gymnasium) y hacer *bootstrap* en el truncamiento, o meter el tiempo restante en el estado. El LAB tiene el mismo patrón. Ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md) |
| 14 | `learn` (celda 4) | `s1_list` se construye solo con transiciones no terminales; si todo el batch fuese terminal, el array quedaría vacío y la llamada al modelo fallaría | Improbable con 64 muestras, pero conviene protegerlo (saltar la predicción si `s1_list` está vacío). El LAB hereda el patrón |

### 10.8 Qué aporta a la práctica

- **Mismo esqueleto** de agente (DDQN + CER + red objetivo + acción aleatoria ponderada del entorno) y de bucle (prellenado del buffer, ε lineal por episodio, `learn_after`, `update_target_after`) que el LAB: ver [08_entorno_best_execution.md](08_entorno_best_execution.md).
- **Mismos errores a evitar:** evaluar en el conjunto de test de verdad, revisar la cronología de recompensas y ejecuciones, y medir con métricas del objetivo y no solo con la recompensa descontada.
- **Diseño:** un ejemplo de estado con información de mercado normalizada más información privada, y de recompensa relativa a una referencia, que en la práctica será el VWAP o el TWAP ([07_rl_para_ejecucion.md](07_rl_para_ejecucion.md)).
- El resto de notebooks de `rl-material` están descritos en [09_material_rl_notebooks.md](09_material_rl_notebooks.md).

*Fuente: RLM profit-seeking, celdas 0-18; `rl-material/src/utils/bmeapi.py`; `rl-material/README.md`; LAB celdas 12-21; BE `dqn.py`; T07 ~00:42:13-00:44:08, ~01:35:39-01:35:49; CASOS p.14.*

---

## 11. De la ejecución a la inversión (T07 ~03:58-04:02)

**Idea.** Al comentar los resultados de su TFM de ejecución, el profesor cerró el círculo con la inversión. Los perfiles y gráficos están en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md); aquí, solo la conexión:
- Con órdenes pequeñas (poco impacto) y coste 0, el TWAP clásico daba un resultado frente a la medida de impacto, y el agente "conservador" quedaba en positivo, mejor que el mercado (T07 ~03:58:37-03:59:33). Avisó de que el gráfico "debería estar"… "da la vuelta", probablemente que el signo está invertido *(dudoso)* (T07 ~03:59:09-03:59:11).
- Si combinas un algoritmo de compra y otro de venta y rascas 2-3 puntos (básicos, se entiende) en cada lado, estás ganando dinero: el algoritmo de ejecución se parece entonces a uno de **alta frecuencia** o de **inversión** (T07 ~03:59:47-04:00:43).
- Pero la **capacidad de absorción** (§2) se aplica también aquí: eran órdenes hijas de un título y sin costes. Al meter costes de transacción desaparecen las ganancias, aunque la diferencia relativa se mantenga (cifras "5" y "−8", *(dudosas)*) (T07 ~04:00:49-04:01:27).
- Con un solo título y comisión cero, podría plantearse ir "rascando poco a poco" en cada sesión (T07 ~04:01:33-04:02:02).

**Nota:** es otra forma de ver §9.2: una ventaja de pocos puntos básicos frente al benchmark solo sobrevive si supera los costes y el impacto a la escala a la que se opera.

*Fuente: T07 ~03:57:43-04:02:10.*

---

## 12. Relación con los Talleres 1 y 2

**Idea.** Los Talleres 1 y 2 fueron, en la práctica, algoritmos profit-seeking de un activo sobre series sintéticas. Este bloque da el marco para releerlos, y el notebook de clase es el paso intermedio hacia el caso de ejecución.

| Aspecto | Taller 1 (tabular) | Taller 2 (deep) | Notebook profit-seeking (este taller) |
|---|---|---|---|
| Objetivo implícito | Retorno absoluto de un activo; batir a buy & hold | Igual, con DQN y A3C | **Relativo:** alfa frente a un benchmark |
| Datos | 4 generadores sintéticos, una serie nueva por episodio | Igual | Ibex real 2017-2020, activo e inicio al azar |
| Estado | `[tendencia, posición]`, discretizado | Igual (continuo, normalizado) | 7 características: rankings de Sharpe, retornos de mercado, posición |
| Acciones | Vender, mantener, comprar | Igual | Mantener, comprar, vender |
| Recompensa | Retorno por paso con posición abierta, menos comisión del 0,1 % | Igual | Alfa log diario si se está invertido; sin costes |
| Episodio | Serie completa (494 decisiones) | Igual | Hasta vender o fin de datos |
| Agente | Q-learning, Sarsa | DQN, A3C | DDQN + CER + red objetivo |

**Lecciones que conectan:**
- Sin información sobre el patrón no hay aprendizaje (browniano y regímenes: "no operar" en el Taller 1, recogido en T2 `notebook_partida.md` §5.3), justo lo que el profesor explicó con el ejemplo de los lunes (§3).
- La comisión del Taller 2 pesaba como la señal; el profesor insistió en meter costes e impacto y estresarlos (§9.2).
- "Batir a buy & hold" era un objetivo absoluto; en este taller se insiste en objetivos relativos y en comparar siempre con un benchmark (§4.4, §9.3).
- El consejo del Taller 2 de empezar episodios en fechas aleatorias y simplificar (T2 `08_trading_en_clase.md` §10) reaparece aquí (§5.1, §8).

Para profundizar:
- [08_trading_en_clase.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/08_trading_en_clase.md): el DQN de trading del Taller 2, sus errores y la experiencia del profesor del Taller 2 con fondos.
- [notebook_partida.md del Taller 2](../raw/referencias/deep-rl-trading-agents/docs/teoria/notebook_partida.md): el entorno `SimpleTradingEnv`, los generadores y los resultados tabulares.
- [notebook_partida.md del Taller 1](../raw/referencias/rl-trading-agents/docs/teoria/notebook_partida.md).

Qué se reutiliza de esos repositorios para la práctica (protocolo, estructura, PyTorch frente a TF, informe) se trata en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

*Fuente: comparación propia a partir de RLM profit-seeking y T2 `notebook_partida.md` §4-§5 y `08_trading_en_clase.md` §10-§11.*

---

## 13. Resumen: diseñar un RL de inversión según la clase

1. **Objetivo aterrizado** y escrito como fórmula; dinero ≠ utilidad (CASOS p.6).
2. Preferir objetivos **específicos y relativos**, que reducen la varianza a explicar (CASOS p.8-9).
3. **¿Hace falta RL?** Solo si las decisiones cambian los datos (impacto externo o interno); si no, supervisado o reglas (APP p.15; CASOS p.13).
4. **Step** = cada cuánto puedes y necesitas reposicionarte; **episodio** = horizonte del objetivo o condición; inicios variados (CASOS p.10).
5. **γ calculado hacia atrás** desde el número de steps; si es bajo, el agente opera al principio (T07 ~01:09:20-01:10:29).
6. **Estado:** generadores de alfa (factores, mispricing, riesgo específico), normalizados entre activos, más información privada (posición, concentración) (CASOS p.10; APP p.49).
7. **Recompensa** alineada con el objetivo **y factible** (local, predecible con el estado); cuidado con el sesgo de invertir pronto; decidir qué pasa cuando no se está invertido (CASOS p.11; T07 ~03:40).
8. **Acciones:** empezar por el triciclo; las condiciones (liquidez, apalancamiento, cortos, rotación) y las reglas sensibles, fuera de la red (CASOS p.12).
9. **Validación:** fuera de train, contra benchmark, con métricas financieras y en varios periodos; costes e impacto estresados (CASOS p.13).
10. En los TFM, según el profesor, se valoraba el **planteamiento** razonado de cada componente más que el resultado (T07 ~01:18:23-01:19:02); en la práctica manda el ENUNCIADO ([notebook_partida.md](notebook_partida.md)).
