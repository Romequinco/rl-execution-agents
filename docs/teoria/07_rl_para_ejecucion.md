# Diseño de un agente de RL para ejecución

> **Fuentes:**
> - **CASOS** (`docs/raw/MIAX14/casos_practicos.pdf`): diapositivas "Casos Prácticos" de Jesús Sanz. Aquí se usan **p.28-61** (diseño RL del algoritmo de ejecución, primeros resultados y referencias), con sus imágenes, más p.17 (fórmula del VWAP), p.19-20 (tipos de algoritmo y objetivo) y p.27 (eje del slippage), ya descritas en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md); p.62 es la despedida y p.63 la contraportada (sin contenido técnico).
> - **T07** (`docs/raw/26 11 07.pdf`): transcripción automática de la clase del 07/11/2026. Tramos: ~00:23-00:26 (combinar supervisado, RL y reglas), ~01:43-01:50 (objetivo del caso), ~02:36-04:16 (mejoras del TWAP, diseño RL completo, resultados del TFM, papers, supervisado frente a RL) y ~04:42-04:56 (cómo se concreta en el notebook: episodios, observación inicial, TWAP por recompensas).
> - **T06** (`docs/raw/26 11 06.pdf`): transcripción automática de la clase del 06/11/2026. Tramos: ~00:34-00:38 (supervisado frente a RL e impacto), ~02:57-03:00 (anécdota del TFM de 2019), ~04:04-04:09 (qué es un episodio y cálculo de steps), ~04:20-04:22 (llenado inicial del buffer), ~04:44-04:49 (reset, sobreajustar un día, estrés).
> - **APP** (`docs/raw/MIAX14/applying_RL_MIAX.pdf`): p.15 (RL en mercados), p.54-57 (steps y acciones), p.61 y p.96 (el agente hará lo que sea por maximizar), p.62 (imagen de la prueba masiva de experimentos).
> - **LAB** (`docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`), celda 7 (clase del entorno), solo para mostrar cómo se concreta cada componente en la práctica. **ENUNCIADO** (`docs/raw/MIAX14/ejercicios_taller_rl.docx`) para los requisitos.
>
> **Convenciones:** "(T07 ~03:30)" = hora aproximada en la transcripción; todo lo que viene de las transcripciones es paráfrasis, con citas literales cortas y escasas; lo que no se entiende bien va marcado *(dudoso)* con las lecturas posibles. "(CASOS p.43)" = página del PDF. "(LAB celda 7)" = celda del notebook contando desde 0. **Nota:** = contenido propio (cálculos, correcciones, conexiones), no del material de clase. pb = punto básico (1 pb = 0,01 %); "pb" con cifra ("3 pb") y "pbs" solo como plural sin cifra o en títulos de diapositiva ("VWAP Slippage (pbs)"). "Hija" = orden hija; "CARE" = orden madre del cliente. Los gráficos de las diapositivas se describen, no se copian.
>
> Escrito el 10/11/2026.
>
> **Documentos relacionados:** la parte de negocio (TWAP/VWAP/POV, dilema del trader, reglas para mejorar un TWAP) está en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md); la teoría general de steps, estados y acciones en [02_diseno_de_entornos.md](02_diseno_de_entornos.md); recompensas y gamma en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md); buffer y exploración en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md); el código del entorno de la práctica en [08_entorno_best_execution.md](08_entorno_best_execution.md); la propuesta para la práctica en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

---

## Mapa del documento

**Idea.** Jesús Sanz recorre los componentes de un problema de RL (entorno, episodios, steps, estado, acciones, recompensas, agente, buffer, análisis) aplicados a un caso real: mejorar un TWAP de compra con RL, que fue su TFM de 2019 en BME (T06 ~02:57-03:00; T07 ~03:52). La tabla resume cada decisión de diseño, la opción que él propone y cómo la concreta la práctica.

| Componente | Lo que propone el profesor (caso realista) | Cómo lo concreta la práctica (LAB / ENUNCIADO) | Sección |
|---|---|---|---|
| Objetivo | Ejecutar toda la orden CARE de compra con menor impacto; mejorar VWAP slippage frente al TWAP e igualar o mejorar la dispersión | Ej. 2: replicar un TWAP por recompensas; ej. 4 idea A: mejorarlo | §2 |
| Episodio | Orden completa; tamaño en %ADV muestreado de la distribución histórica, con límite 5-10 % ADV | 400-600 steps de 5 s; orden CARE ≈ 7,5-12,5 % de los steps | §4 |
| Qué decide el agente | Una sola dimensión del TWAP (momento, tipo o tamaño) | Solo el momento; siempre agresiva de 1 título | §5 |
| Step | Resolución de decisión ≠ resolución del simulador; guardar lo que pasa entre medias | Un step = un bin de 5 s | §6 |
| Estado | Tamaño (%ADV), volumen y tiempo restantes (%), liquidez, frecuencia, desbalanceo, rendimiento a corto, prob. de ejecución | Tiempo restante y volumen restante (%) | §7 |
| Acciones | Pasiva / mantener / agresiva (y variantes) con tamaño fijo justificado | 0 = no hacer nada, 1 = agresiva de 1 título | §8 |
| Recompensa | Local y factible (VWAP local o frente al TWAP en cada decisión) + volumen pendiente + desviación del plan | Por defecto: precio frente al VWAP de mercado del episodio | §9 |
| Agente y buffer | Cualquiera de los vistos; buffer dimensionado en episodios; inicialización bróker/TWAP/aleatoria | Double DQN con buffer CER | §10-11 |
| Resultados | Fuera de train, media y dispersión, estrés; perfiles conservador→muy agresivo | Gráficos de precio/VWAP y volumen acumulado | §12-15 |

---

## 1. Por qué RL aquí: de la explosión de reglas a aprender el "diccionario"

**Idea.** Un TWAP clásico es mejorable con reglas sensatas, pero cada regla añade condiciones y, al combinar dos o tres, el espacio de situaciones explota; además los expertos no se ponen de acuerdo sobre la regla correcta. El RL sustituye ese trabajo manual: se le da la información y aprende qué hacer en cada situación (T07 ~02:46-02:51; CASOS p.28).

- **Punto de partida.** El TWAP de libro: 10.000 títulos en 100 min → 10 hijas de 1.000 cada 10 min, tamaño y momentos fijos, sin mirar la liquidez ni lo que pasa en el mercado (T07 ~02:37-02:38). Tres decisiones "arbitrarias": momento, tamaño y tipo de orden (agresiva/a mercado, pasiva/limitada, pegada) (T07 ~02:40). El detalle de las reglas que ya usa la industria (hijas ligadas a la liquidez de la 1.ª posición, perfil de recuperación del libro por empresa, aleatoriedad para camuflarse) está en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md).
- **Reglas candidatas que surgieron en clase.** Esperar unos segundos si una ejecución previa ha abierto el spread (T07 ~02:41); dejar una limitada durante la ventana de 10 min y lanzar la agresiva solo si no se ha ejecutado (T07 ~02:45); adelantar la hija si hay actividad de trades en el mercado, "un toque de POV" (T07 ~02:46).
- **Por qué no se programan.** Con 2-3 condiciones combinadas "explota la dimensionalidad" y hay que acertar la regla exacta para cada circunstancia (T07 ~02:47). Y es ruidoso: ante un spread que se abre, un experto diría esperar y otro comprar ya porque alguien está ejecutando (T07 ~02:48, el "abogado del diablo"). Las brokers saben que el TWAP es muy mejorable, pero meter tanta condicionalidad es inviable en la práctica (T07 ~02:50).
- **Alternativa RL.** Pasar al agente la información (si hay frecuencia de trades, si el spread está por encima o por debajo de sus percentiles de los últimos minutos) y que, a partir del histórico, aprenda la decisión óptima sin que el experto tenga que decir qué es bueno o malo; en Q-learning eso es aprender el "diccionario" estado→acción (T07 ~02:49-02:51). La diapositiva lo resume como: más información → más reglas → "complex framework" con demasiadas reglas a mano → solución ML/IA, donde el RL "entrena el algo para ser bróker" (CASOS p.28).
- **Por qué es un problema de RL y no de supervisado.** En ejecución de órdenes grandes cada tramo mueve el precio del siguiente: las decisiones cambian los datos que se verán después (APP p.15). Con supervisado se etiquetaría el histórico como si nuestras órdenes no movieran el precio; si operas con capital relevante, esas etiquetas dejan de valer y hay que modelar las consecuencias en el entorno (T06 ~00:34-00:38). Desarrollo en [00_repaso_rl.md](00_repaso_rl.md).

**No todo tiene que ser RL.** El profesor insiste en combinar herramientas (T07 ~00:23-00:26; T07 ~02:51-02:54; CASOS p.28):

| Herramienta | Papel en el caso de ejecución | Fuente |
|---|---|---|
| No supervisado | Clustering de activos por perfil de liquidez para aplicar algoritmos distintos; reducción de dimensionalidad | T07 ~00:24, ~02:51; CASOS p.28 |
| Supervisado | *Feature engineering*: probabilidad de que una pasiva se ejecute en los próximos X s, como característica del estado (§7.4) | T07 ~02:51-02:54; CASOS p.28 |
| Reglas deterministas | Lo que debe cumplirse al 100 % se programa (tamaño fijo, "cargas de profundidad", desautorizar al agente si se desvía) | T06 ~02:55; T07 ~03:28, ~03:53 |
| RL | Aprender la regla de decisión en el punto del proceso donde hay ruido e interacción | T07 ~02:49-02:51 |

Construir la característica "por fuera" se lo da "masticadito" al agente y simplifica todo lo que viene después; de hecho, solo con un supervisado de probabilidades y reglas ajustadas a mano ya se mejoraría el TWAP sin RL (T07 ~02:53-02:54).

---

## 2. Objetivo aterrizado del caso

**Idea.** Antes de diseñar nada hay que aterrizar el objetivo en una fórmula; el resto (recompensas, estado, benchmark) sale de ahí (CASOS p.29; T07 ~01:44-01:47, ~02:54). Lo general sobre objetivos está en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

| Elemento | Definición en el caso | Fuente |
|---|---|---|
| Tipo de algoritmo | De calendario (*scheduled*): menor impacto, poca urgencia | T07 ~01:44; CASOS p.19-20 |
| Orden | CARE de compra (p. ej. 1 millón €); solo compra, no venta | CASOS p.29; T07 ~01:46 |
| Restricción | Ejecutarla en su totalidad ("quiero los 10 millones, no 8") | T07 ~01:45 |
| Lo que se ignora | Precio de llegada o de decisión | T07 ~01:45 |
| Métrica | VWAP slippage: mi precio medio ponderado frente al VWAP de mercado en el mismo periodo; éxito si el mío es menor | T07 ~01:45; CASOS p.29 |
| Benchmark | TWAP → mejorar el precio (VWAP) y mejorar o igualar la dispersión | CASOS p.29 |

Con el VWAP definido como en CASOS p.17:

$$
\text{VWAP}=\frac{\sum_{i}P_i\,V_i}{\sum_i V_i},\qquad
\text{slippage}_{\text{pbs}}=\frac{\text{VWAP}_{\text{mkt}}-\text{VWAP}_{\text{algo}}}{\text{VWAP}_{\text{mkt}}}\cdot 10^4
$$

**Nota:** la fórmula del slippage con este signo (positivo = he comprado más barato que el mercado = mejor) es la convención que usa el histograma de CASOS p.60, donde la derecha es "Mejor"; el material no escribe la fórmula explícitamente. La recompensa por defecto del LAB usa el mismo signo, $(\text{episode\_vwap}-\text{price})/\text{episode\_vwap}$ (LAB celda 7, `_compute_reward`).

**La dispersión importa.** Un algoritmo muy agresivo puede tener muy buena media y a la vez resultados muy dispersos (unos días muy bien, otros muy mal); un cliente prefiere saber qué va a obtener, así que se puede pedir "buena media con dispersión controlada" (T07 ~01:46). Por eso el objetivo incluye "mejorar o igualar dispersión" (CASOS p.29). Los benchmarks clásicos sirven además para saber a qué distancia se está y decidir cuándo bajar la exploración (T07 ~01:47; ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md)).

---

## 3. Componentes RL y papel del entorno

**Idea.** Lo que hay que programar es el entorno/simulador (T07 ~02:56). **Nota:** el agente puede ser cualquiera de los ya vistos (§10.1).

CASOS p.30 muestra el esquema clásico: recuadro discontinuo "Componentes RL" con *Agent* y *Environment* unidos por flechas de *State (s)*, *Reward (r)* y *Action (a)*; debajo, la secuencia $s_t\,a_t\,r_t \to s_{t+1}\,a_{t+1}\,r_{t+1}\to\dots$ que acaba en un recuadro "End of episode (Execution Time)". A la derecha, las funciones del *Environment (Simulador)*:

1. Genera órdenes (episodios).
2. Envía el estado actual $s_t$.
3. Recibe la acción $a_t$.
4. Evalúa la acción y envía la recompensa $r_t$.
5. Envía el estado siguiente $s_{t+1}$, que puede ser terminal o no (T07 ~02:55-02:56).

Las diapositivas siguientes (p.33, 36, 42, 44, 46-50) van rellenando el mismo esquema: vector de características en $S_3$ (10:30 am), acciones, recompensas, agente y finalmente la tabla Q y la política. En la API gym esto son `reset` (genera el episodio y devuelve la primera observación) y `step` (recibe la acción, devuelve recompensa, nuevo estado y fin), ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md) y, para el código de la práctica, [08_entorno_best_execution.md](08_entorno_best_execution.md).

---

## 4. Generación de órdenes (episodios)

### 4.1 Qué define un episodio: el "TWAP con esteroides"

**Idea.** El objetivo es un TWAP mejorado ("TWAP con esteroides"), así que el episodio debe parecerse a lo que un cliente pide a un TWAP: un horizonte temporal y un número de títulos (T07 ~02:56-02:58). En clase, ante la pregunta de qué define un episodio, salen el horizonte, el número de títulos y el número de particiones, con la urgencia de completar la orden (T07 ~02:56-02:57).

- En BME se estudiaba qué características tenían las peticiones reales de TWAP de los clientes y se generaban episodios con esas distribuciones (T07 ~02:57-02:58).
- **Normalizar el tamaño**: mejor en % del volumen diario que en títulos. Un millón de títulos puede ser el 10 % del volumen diario de un valor y una orden pequeña en otro (T07 ~02:58).
- El entorno genera las órdenes en N títulos o en **% ADV de los últimos 30 días** (*average daily volume*), con o sin subastas (CASOS p.31).
- **Límites**: no más de, p. ej., 5-10 % ADV, lo que ayuda con las asunciones de impacto (CASOS p.31).

**Gráfico de CASOS p.31.** Histograma de tamaños de orden generados: eje X "N títulos / % ADV" desde 0, eje Y frecuencia. Frecuencia casi nula junto a 0, subida brusca hasta un máximo en tamaños pequeños (hacia el primer sexto del eje) y cola larga a la derecha (asimétrica positiva, forma tipo log-normal): la mayoría de órdenes son pequeñas pero no minúsculas, y pocas son muy grandes. Sin valores numéricos en los ejes.

### 4.2 El triciclo: órdenes pequeñas e impacto cero

**Idea.** Empezar por el caso más controlado. Las órdenes muy grandes obligan a modelar funciones de impacto de mercado; con órdenes pequeñas se pueden relajar las hipótesis de ejecución (T07 ~02:59-03:00).

- Ejemplo de la práctica: orden de 400 títulos, hijas de 1 título, sin comisiones ni costes de transacción, impacto 0; así el histórico de precios funciona como un simulador "bastante realista" (T07 ~03:00-03:01).
- La hipótesis de coste 0 no es descabellada: hay brókeres con promociones de corretaje cero o tarifas planas en las que los primeros títulos salen gratis; con esas hipótesis el histórico de precios es un simulador bastante realista (T07 ~03:00-03:01). **Nota:** con hijas de 1 título, una comisión fija de 2 € por orden (cifra que el profesor menciona en T07 ~03:57) sería una fracción enorme del precio de una acción de ~7 €; sin coste 0 el planteamiento no tendría sentido.
- Para qué sirve el triciclo: comprobar que la mecánica del entorno funciona bien, con ejecuciones en momentos temporales distintos o con distintos tipos de orden (T07 ~03:00).
- Si no funciona en este escenario supercontrolado, no funcionará con órdenes de un millón (T07 ~03:00).
- Variabilidad: los escenarios deben ser realistas y variados (empresas, cantidades) para que el agente no se enfrente siempre a lo mismo (T07 ~03:01-03:02). Pero para depurar conviene empezar con **un mismo día y un solo activo**, lo cual "no sirve para producción" (T07 ~03:02).
- Lo mismo se dijo el día anterior sobre `reset`: el comienzo no debe ser siempre el mismo (no siempre Santander, no siempre a las 9; también a las 12), pero primero hay que intentar **sobreajustar** un día y un activo como test de que recompensa y estado están bien configurados; después dos días (uno alcista y otro bajista), varios activos y, más avanzado, partidas que empiezan a mitad de sesión o escenarios de estrés (Covid, entornos sintéticos) (T06 ~04:44-04:48). Desarrollo del curriculum en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

### 4.3 Dos enfoques para el episodio

CASOS p.32 y T07 ~03:02 plantean dos formas de definir la partida:

| Enfoque | Qué es un episodio | Pros / contras |
|---|---|---|
| **Realista (el elegido)** | La orden completa: "Duración del episodio → Completo" | *(Nota, inferencia propia por contraste con el alternativo)* cubre todo el plan; más flexible y más difícil |
| **Alternativo** | El TWAP actúa cada X tiempo con sus reglas, los momentos de actuación están predefinidos y **cada momento de actuación es un episodio**; el RL aprende la mejor forma de actuar en ese caso | **+ Específico; − Flexibilidad** (más parecido a un TWAP) |

**Nota:** en la imagen de CASOS p.32 los signos "+ Específico / − Flexibilidad (más parecido a un TWAP)" están a la derecha del bloque del enfoque **alternativo**, separados por la línea discontinua del realista; son, por tanto, los pros y contras del alternativo, no del realista. Coincide con lo que dice el profesor del ejemplo "decidir dentro de cada ventana de 10 min" (§5): resultados muy parecidos al TWAP porque se le deja poca libertad, pero problema muy acotado (T07 ~03:05).

**Episodios en ejecución intradía (T06 ~04:04-04:08).** El episodio de un algoritmo intradía es puramente temporal, de principio a fin (el día entero o unas 2 h), aunque se puede cortar por otras condiciones. Con el número de steps por episodio se dimensiona el buffer (§10). Más en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

### 4.4 Cómo se concreta en la práctica

**Idea.** El notebook aplica el triciclo: datos en bins de 5 s, hijas de 1 título, impacto y costes 0 (T07 ~04:43-04:45; ENUNCIADO, ej. 2).

- Las fotos de 5 s del pickle van de 13:00 a 17:25, de modo que `reset` sortea activo (SAN o REP), día y hora de inicio dentro de esa franja: "un día puede ser a la una… y otro a las 2 de la tarde" (T07 ~04:43, ~04:50). No hace falta usar el día entero: un episodio de 90 min-2 h es lo que dura "un TWAP normalito" (T07 ~04:44). Estructura de los datos en [08_entorno_best_execution.md](08_entorno_best_execution.md).

- Episodios de 400 a 600 steps de 5 s, es decir, menos de una hora (33-50 min); se pueden alargar a 2-3 h (T07 ~04:46-04:47). En el código, `episode_bins` es un entero uniforme en [400, 600] (LAB celda 7, `_generate_episode_params`).
- Una hora son $12\times 60=720$ bins de 5 s (T07 ~04:45). Si la orden fuera de 720 títulos con hijas de 1, habría que ejecutar en todos los bins: ninguna flexibilidad; con 360, en 1 de cada 2 (T07 ~04:46).
- Por eso la orden CARE es una fracción pequeña de los steps: `vol_care` uniforme en [0,075; 0,125] × `episode_bins` (LAB celda 7), es decir, ejecutar aproximadamente en 1 de cada 10 steps. El ENUNCIADO pone el ejemplo de 400 steps y 40 títulos (ej. 2).
- **Nota:** en T07 ~04:47 la transcripción dice multiplicar 720 "por 0 9" y obtener una hija de 65 títulos; lo coherente con el código es $720\times 0{,}09\approx 65$ títulos de orden CARE (65 ejecuciones de 1 título, "65 barritas" en el gráfico final), no 0,9 *(dudoso por la transcripción)*.
- **Nota:** este "10 % de flexibilidad" del notebook (proporción de steps en que hay que ejecutar) no es lo mismo que el "% de libertad" de los perfiles del TFM (§13), que mide cuánto puede desviarse el agente del plan TWAP.

Detalle de la generación en el código (`reset`, selección de activo, día y hora de inicio, precálculo del VWAP del episodio) en [08_entorno_best_execution.md](08_entorno_best_execution.md).

---

## 5. Acotar el problema: qué decide el agente

**Idea.** Para el profesor es "casi lo más importante de la clase": aislar los problemas y darles un tratamiento específico, con subsistemas que aborden cada uno un tema (T07 ~03:02, ~03:07). Un TWAP toma tres decisiones (momento temporal, tamaño y tipo de orden) y eso "son muchas cosas para un algoritmo": se pueden cubrir las tres, una sola o repartirlas entre varios algoritmos (T07 ~03:03).

| Alternativa | Qué fija el TWAP | Qué decide el agente | Comentario |
|---|---|---|---|
| (a) Momento dentro de la ventana | Tamaño (1.000) y tipo | Cuándo lanzar la hija dentro de cada ventana de 10 min | Resultados muy parecidos al TWAP (poca libertad) pero problema "súper específico" (T07 ~03:04-03:05) |
| (b) Tipo de orden | Momentos y tamaño | Agresiva, pasiva/limitada, pegada… (T07 ~03:05) | El que el profesor habría elegido para su TFM (§19) |
| (c) Tamaño | Momentos y tipo | Entre la mitad, el mismo o más (un 50 % más o el doble) (T07 ~03:06) | **Nota:** la transcripción dice "el doble de tamaño, un 50 por 100 más"; las dos lecturas son posibles *(dudoso)*. CASOS p.41 usa ×2 y ×0,5 como modificadores |
| **Práctica** | Tipo (siempre agresiva) y tamaño (siempre 1 título) | El momento temporal a lo largo de **todo el episodio**, no dentro de cada ventana | T07 ~03:06-03:07 |

- Ejemplo de subsistema: un agente que solo decide el tipo de orden, alimentado únicamente con la probabilidad de ejecución, aprendería tras promediar episodios algo como "a partir del 60 % interesa la pasiva" (T07 ~03:07-03:08).
- Esos subproblemas bien aterrizados "tienen solución" y sí se pueden mejorar (T07 ~03:07).
- Al presentar el caso, ya en la primera hora, el profesor había anunciado que daría alternativas para reducir el alcance del ejercicio (T07 ~00:23).

---

## 6. Steps: resolución de decisión frente a resolución del simulador

**Idea.** Hay que fijar los steps antes que las características, porque condicionan cómo se calculan (T07 ~03:08). Lo general (criterios temporal, por eventos y mixto; coste computacional) está en [02_diseno_de_entornos.md](02_diseno_de_entornos.md) (T06 ~02:19-02:29; APP p.54-55).

- Opciones en ejecución: tick a tick, ms, s… (CASOS p.34); por evento del libro, cada 5 s, o cada vez que hay que reevaluar una pasiva pendiente: lanzo una pasiva, espero 5 s, si no se ha ejecutado reevalúo y decido si relanzo otra o espero (T07 ~03:08-03:09). APP p.54 da el criterio mixto "5 s o hasta que se ejecute la orden".
- La elección afecta a las asunciones, la simulación del impacto, el diseño de características y las acciones; y no es igual en el *research* inicial que en producción o en la comprobación de resultados (CASOS p.34). En *research* se puede aumentar el step para abaratar; en producción, máxima resolución y datos (APP p.55).
- **La resolución de decisión no tiene por qué ser la del simulador** (T07 ~03:09). Decidir cada minuto ya es buena resolución para una orden de un millón: un bróker hace bien su trabajo sin "darle al botón" cada 5 s (T07 ~03:10).
- **Decidir cada X s no significa hacer fotos del libro cada X s** (CASOS p.34): se puede guardar información intermedia (precios, volúmenes, ejecuciones).

**Gráfico de CASOS p.34.** Curva de precio entre dos momentos de decisión: en $S_t$ el precio es 7,12 €, entre medias sube a 7,16 €, cae hasta 7,03 € y en $S_{t+1}$ está en 7,10 €. Si solo se comparan las dos fotos (7,12 → 7,10) se concluiría que una limitada de compra a 7,07 € no se ejecutó, cuando el precio llegó a 7,03 € con volumen suficiente para cubrirla (T07 ~03:11; la transcripción dice "una orden a 7 con 0, 7" y "el precio llegó a ser 3", así que las cifras habladas son *(dudoso)*; la diapositiva es la referencia).

**Nota:** en la práctica la resolución de decisión coincide con la del simulador (un step = un bin de 5 s del pickle) y la única acción es agresiva al ask del bin, por lo que el problema de las fotos no aparece; aparecería en cuanto se introdujeran pasivas (ej. 4), donde haría falta `executions.pkl` o los volúmenes negociados entre bins (ver [08_entorno_best_execution.md](08_entorno_best_execution.md)).

---

## 7. Estado: el vector de características

### 7.1 Cómo pensar las características

**Idea.** Sentarse con el bróker y preguntarle qué mira para ejecutar y qué le condiciona (T07 ~03:15). Después estructurar el problema por dimensiones que expliquen la varianza y, dentro de cada una, elegir las características que más importan; aquí entra el conocimiento de negocio (T07 ~03:16-03:17). Lo que interesa es "el tipo de pensamiento", no la lista concreta, que es ilustrativa (T07 ~03:22).

Dos clasificaciones (CASOS p.35; T07 ~03:16-03:17):
- **Por objeto**: orden CARE / activo a ejecutar / condiciones de mercado.
- **Por origen**: internas o privadas (dependen de ti: volumen restante, tiempo restante) / externas o de mercado. La distinción privada/externa y cómo tus acciones cambian tu propio estado se vio el día anterior (T06 ~00:36-00:38) y está en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

### 7.2 Las características de CASOS p.35

| Característica | Definición / normalización | Tipo | Por qué es relevante (clase) |
|---|---|---|---|
| Tamaño de la orden | % ADV | Privada (orden) | Una orden grande frente a lo que negocia el mercado obliga a ser más agresivo (menos pasivas, impacto seguro); una muy pequeña puede ir con pasivas (T07 ~03:19-03:20) |
| Volumen restante | % del volumen inicial | Privada | No es lo mismo un millón de títulos en Santander que en un valor menos líquido, y "me quedan 500.000" no dice nada sin saber cuánto se pidió: hay que pasar la fracción de lo pedido (T07 ~03:20) |
| Tiempo restante | % del tiempo inicial | Privada | Junto con el anterior marca la urgencia (T07 ~03:20-03:21) |
| Liquidez | 1.ª posición / tamaño de la orden hija | Mercado | Saber si hay volumen para cubrir una agresiva de la hija predefinida (T07 ~03:21) |
| Frecuencia de operaciones | < 10 oper/min → 0 (bajo); ≥ 10 → 1 (alto) | Mercado | Actividad del mercado, indicativo de liquidez (T07 ~02:46, ~03:18) |
| Desbalanceo de operaciones | N oper. en el ask / N oper. totales | Mercado | Si las ejecuciones van a un solo lado o están repartidas (T07 ~03:18) |
| Rendimiento a CP | $\ln(P_t/P_a)$ | Mercado | Si el precio va en una dirección (T07 ~03:18) |

- **Normalizar todo** para que el agente "entienda lo mismo" con cualquier activo u orden (T07 ~03:18-03:21).
- **Mejor el ratio o la diferencia** entre tiempo y volumen restantes que cada uno por separado: con un 30 % de tiempo y un 50 % de volumen por delante "o me pongo las pilas o se me agota el tiempo" (T07 ~03:21).
- Para cada característica hay que justificar por qué es relevante; de esa razón sale cómo se define (T07 ~03:19, ~03:21).
- **Nota:** CASOS p.35 no define $P_a$. Lecturas posibles: precio *anterior* (de hace unos minutos, que es lo natural para un rendimiento "a corto plazo") o precio de llegada (*arrival*) de la orden. La segunda chocaría con que el caso ignora el arrival price (§2), así que la primera parece la más probable *(dudoso)*.
- **Nota:** el desbalanceo así definido está en [0, 1] y vale 0,5 si la mitad de las operaciones agreden el ask (compras) y la otra mitad el bid; >0,5 indica presión compradora.

**Gráfico de CASOS p.35 (libro de órdenes).** Esquema del libro en t = 0, 1, 2. Ask (rojo), de la 4.ª a la 1.ª posición: 8,75 € : 325; 8,70 € : 120; 8,65 € : 60; 8,60 € : 250. Bid (verde), de la 1.ª a la 4.ª: 8,45 € : 125; 8,40 € : 30; 8,35 € : 60; 8,30 € : 270. En t = 1 una ejecución de 100 contra el ask deja la 1.ª posición en 8,60 € : 150; en t = 2 una de 20 contra el bid deja 8,45 € : 105. Llaves laterales señalan las posiciones (1.ª, 2.ª, …) y el *spread* (hueco entre 8,60 y 8,45). En clase se usa para explicar el barrido: quien quiera vender más de 105 títulos de forma agresiva venderá 105 a 8,45 y los siguientes 30 a 8,40 (T07 ~03:24). **Nota:** el spread del dibujo (0,15 € sobre ~8,5 €, unos 175 pb) es ilustrativo; en valores líquidos españoles es de pocos pbs (el propio ejemplo de p.36 usa 2 pb).

### 7.3 Ejemplo de vector de estado (CASOS p.36)

En $S_3$ (10:30 am) el estado es una tabla de cuatro columnas: **bid-ask spread 2 pb · prob. de ejecución 85 % · tiempo restante 25 % · volumen restante 20 %** (CASOS p.36; T07 ~03:22). Es "la información que tú, como experto, consideras necesaria" para ejecutar (T07 ~03:22). **Nota:** con 25 % de tiempo y 20 % de volumen restantes el agente va ligeramente por delante del plan lineal (ha ejecutado el 80 % de la orden en el 75 % del tiempo), así que no tiene urgencia; con probabilidad de ejecución alta y spread estrecho, pasiva o mantener serían razonables. La política de p.50 asigna a $S_3$ la acción a2 = mantener (§11), coherente con ir por delante del plan, aunque la tabla Q de las diapositivas es ilustrativa y no se calcula a partir de este vector.

### 7.4 La probabilidad de ejecución como característica (supervisado)

**Idea.** Una característica muy potente que se calcula fuera, con un modelo supervisado: la probabilidad de que una orden pasiva/limitada pendiente se ejecute en los próximos segundos (CASOS p.28; T07 ~02:51-02:52).

- **Entrenamiento**: con todas las pasivas del histórico y características como frecuencia de trades, volatilidad, spread, etc. (T07 ~02:52). Debe tener en cuenta no solo el precio sino la **prioridad en la cola** (no es lo mismo estar segundo que cuarto o quinto en el nivel); "no hace falta que sean redes, puede ser un árbol" (T07 ~04:08-04:09).
- **Por nivel de precio**: la probabilidad depende de lo limitada que sea la orden. Ejemplo: limitada de compra por debajo del mercado; a 7,10 € quizá un 90 % de ejecutarse en los próximos 5 s, a 7,03 € casi 0 %, porque el mercado no se mueve 10 céntimos en 5 s. El algoritmo elegiría algo intermedio como 7,08 €: buen precio y probabilidad aún alta (T07 ~03:12-03:14; las cifras son *(dudoso)* por la transcripción). **Nota:** de 7,10 a 7,03 € hay 7 céntimos; el "no se mueve 10 céntimos" es un redondeo oral o un error de transcripción *(dudoso)*.
- **Por qué elige la pasiva**: el algoritmo la elige porque ve muchos trades en el mercado y gente agrediendo en los dos lados, lo que eleva la probabilidad estimada de ejecución (T07 ~03:12-03:13). "Pasiva" es un concepto muy amplio: lo que optimiza el supervisado es a qué nivel ponerla (T07 ~03:14).
- **Vector de características**: esas probabilidades por acción (por nivel) (T07 ~03:14). Meter esa capa de ML en un punto del proceso "ya sería un proyecto entero", muy acotado, para mejorar el TWAP (T07 ~03:14-03:15).
- **Sin RL**: a mano habría que fijar umbrales (¿pasiva si p > 90 %? ¿y si es 70 % pero hay liquidez?) y vuelve la explosión de dimensionalidad; el RL aprende la regla (T07 ~02:52-02:53). Un RL al que se le pase la misma información que al supervisado probablemente calcularía internamente algo equivalente, pero dárselo hecho simplifica (T07 ~02:53-02:54).
- Los bancos ya hacen proyectos internos de este tipo (probabilidades visibles en los terminales del bróker); el RL aporta buscar una regla mejor que la humana (T07 ~04:09-04:10).

### 7.5 El estado de la práctica

La primera observación del entorno de la práctica es **tiempo restante y volumen restante** (LAB celda 7, `observation_builder`; T07 ~04:52). Para el ej. 2 (replicar un TWAP) bastan esas dos, sin ninguna característica de mercado: tienen que "ir de la mano" para que salga la diagonal (T07 ~04:52). Para el ej. 4, idea A, el ENUNCIADO pide añadir información del libro y/o de la tendencia, que es donde entran las características de §7.2. Para estas características el entorno ya prevé un *lookback*: el episodio "full" es el episodio más los 60 bins anteriores (unos 5 min), porque a veces hace falta información previa para calcular características móviles (T07 ~04:59); detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md). Desarrollo en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

---

## 8. Acciones

### 8.1 Tipos de órdenes (CASOS p.37)

| Tipo | Parámetro | Comentario |
|---|---|---|
| Limitada | Posición (precio) | Pasiva: se espera a que alguien agreda (T07 ~03:12) |
| Agresiva | Nivel de agresión | Cruza el spread y barre posiciones |
| Pegada | Posición, n saltos, € saltos | Sigue al libro: si el libro se mueve, se mueve con él ("siempre en la segunda posición") (T07 ~03:23-03:24). "No hay inacción entre steps" |
| Mantener | — | No hacer nada → siguiente step |
| Cancelar | — | Salvo que las órdenes tengan duración acotada por step |

- Más acciones → más posibilidades y más ruido (CASOS p.37); cada grado de libertad añade inestabilidad (T07 ~03:25). Hay que simplificar, como en el ejemplo del volante: con steps cortos bastan pocas acciones discretas (T07 ~03:24-03:25; T06 ~02:25-02:29). Ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md).
- Algunas acciones no las elige el agente (p. ej. "cargas de profundidad", §8.3) (CASOS p.37).
- **Acciones continuas** (p. ej. mid ± pbs, o "precio medio + X pbs") solo para métodos de *policy* (CASOS p.37, p.42); los métodos de valor necesitan acciones discretas (APP p.56).
- Versión discreta del caso: **a1 = pasiva** (en clase, "limitada a la primera posición") **| a2 = mantener | a3 = agresiva** (CASOS p.42; T07 ~03:26).
- Antes de agrupar o discretizar, estudiar si hay acciones redundantes o de efecto similar: APP p.57 pregunta precisamente si "mantener" y "pegada en 4.ª posición" son equivalentes en ejecución.

### 8.2 Agresividad de terceros: hasta dónde barren (CASOS p.38)

**Tabla de CASOS p.38** ("Agresividad de las órdenes a mercado enviadas por terceros", % de órdenes según la posición más profunda que alcanzan):

| Posición | Telefónica (%) | BBVA (%) | Santander (%) |
|---|---|---|---|
| 0 | 0,03 | 0,05 | 0,06 |
| **1** | **91,48** | **92,36** | **89,90** |
| 2 | 5,85 | 5,46 | 6,98 |
| 3 | 1,64 | 1,35 | 1,80 |
| 4 | 0,56 | 0,44 | 0,67 |
| 5 | 0,24 | 0,17 | 0,28 |
| 6 | 0,11 | 0,08 | 0,14 |
| 7 | 0,06 | 0,04 | 0,07 |
| 8 | 0,03 | 0,02 | 0,04 |
| 9 | 0,02 | 0,01 | 0,03 |
| 10 | 0,01 | 0,01 | 0,02 |

Una llave agrupa las posiciones 1-2 y una flecha con el texto "Aprendía como mantener" apunta a la posición 3. En clase: prácticamente el 90 % de las agresiones se quedan en la 1.ª posición, un 5 % adicional llega a la 2.ª y muy pocas más abajo (T07 ~03:25).

**Nota (cálculo propio):** probabilidad de que una agresión alcance al menos la posición 3: Telefónica 2,67 %, BBVA 2,12 %, Santander 3,05 %; al menos la 4: 1,03 %, 0,77 % y 1,25 %; la 9: 0,03 %, 0,02 % y 0,05 %. Cada columna (empresa) suma ≈ 100 % (Telefónica 100,03; BBVA y Santander 99,99, por redondeo). La fila "posición 0" no se explica; podría ser agresiones ejecutadas sin consumir la 1.ª posición visible (p. ej. contra liquidez oculta o dentro del spread) *(dudoso)*. La lectura de "Aprendía como mantener" es que una orden pegada en la 3.ª-4.ª posición casi nunca se ejecuta y el agente la trataba en la práctica como "mantener", lo que enlaza con APP p.57.

### 8.3 Acciones dinámicas y reglas deterministas

- **Una acción puede significar cosas distintas a lo largo del episodio** (CASOS p.38): "Mantener" se traduce al principio de la ejecución en una **pegada en 4.ª posición** (CASOS p.38; en 4.ª o 5.ª según la clase, T07 ~03:26) (a efectos prácticos es como no hacer nada, pero si alguien barre te llevas un buen precio) y al final en **no poner orden** (cerca del cierre se quiere más control) (T07 ~03:26-03:27). El agente solo sabe que eligió "mantener"; el sistema lo traduce con su propia lógica (T07 ~03:27).
- **Coexistencia con reglas**: el TWAP de BME mantenía "cargas de profundidad", órdenes muy abajo (9.ª posición) por si alguien barría el libro; con miles de trades acababa ocurriendo bastantes veces (T07 ~03:28). Es una regla determinista independiente de lo que decida el algoritmo.

### 8.4 Tamaño de las hijas (CASOS p.39-41)

| Opción | Ejemplo |
|---|---|
| Fijo | 1.000 títulos |
| Relativo | Según un estudio de liquidez o de las operaciones de terceros |
| Condicionado | $\min(\text{tamaño estándar},\ 0{,}5\cdot \text{vol}_{\text{ask1}})$ |
| Modificador agresivo | Tamaño estándar × 2 |
| Modificador conservador | Tamaño estándar × 0,5 |

Se puede añadir cierta aleatoriedad para no ser identificados (CASOS p.41; T07 ~02:44).

**Justificación del tamaño fijo en Santander** (T07 ~03:29-03:30): la 1.ª posición suele tener unos 2.000 títulos (algo menos por la mañana, más al final del día) y el mercado suele negociar unos 600 títulos por operación; con hijas fijas de 1.000 se ejecuta "lo mismo que el mercado" y se pasa desapercibido. Así se le quita esa decisión al algoritmo.

**Gráfico de CASOS p.40.** Mediana del volumen de la 1.ª posición del libro (bid en naranja y ask en azul) del Banco Santander a lo largo de la sesión, en bins de 5 min, primer semestre de 2018. Arranca a las 09:00 con un pico (~2.100), cae enseguida a ~1.300-1.700 títulos en la primera media hora, se estabiliza en torno a la línea discontinua de **2.000 títulos** la mayor parte del día y sube al final hasta ~2.400-2.600 hacia las 17:30. El pie de la diapositiva destaca que la liquidez del libro es creciente y está **descorrelacionada del volumen negociado**, que tiene forma de U.

### 8.5 Las acciones de la práctica

Dos acciones: `0 = _do_nothing` y `1 = _agg_action`, compra agresiva de 1 título al `ask1` del bin (LAB celda 7). Es la simplificación de §5: el agente solo decide el momento (T07 ~03:06). Detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md).

---

## 9. Recompensas

**Idea.** Las recompensas "determinan el comportamiento del algo" (CASOS p.44). Tienen que estar alineadas con el objetivo **y además ser factibles**, es decir, predecibles a partir de las características que se le pasan (T07 ~03:37-03:40). La teoría general (sparse frente a shaping, magnitudes, calibración) está en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

### 9.1 Candidatas (CASOS p.43-44; T07 ~03:30-03:33)

Recompensa en función de:
- **Impacto de mercado**: VWAP slippage **global** (al final de la ejecución) o **local** ("ahora mismo lo estoy haciendo mejor").
- **Desviación respecto a un plan tradicional** (el TWAP).
- **% del volumen de la orden CARE ejecutado**.
- … y otras; la diapositiva pone como ejemplo "equivalente al precio de lanzar orden a mercado en el último segundo" (CASOS p.43). **Nota:** lo más probable es que se refiera a valorar el volumen pendiente al precio de una agresiva final.
- CASOS p.44 lo resume como $f(\text{impacto},\ \%\text{vol ejecutado},\ \text{desviación del plan})$.

La imagen de p.43 es una escalera con un trofeo arriba: recompensas que van guiando hacia el objetivo.

| Recompensa | Qué compara | Ventaja | Problema |
|---|---|---|---|
| VWAP slippage global | VWAP del algo frente al de mercado al final | Es exactamente el objetivo | No factible: exige predecir precio **y** volumen futuros del mercado (§9.3) |
| VWAP slippage local | Precio de cada ejecución frente al VWAP local | Factible; en media, hacerlo bien localmente debería beneficiar a largo plazo | Puede acabar peor en escenarios concretos (T07 ~03:38) |
| Frente al TWAP en cada decisión | Mi acción frente a la que habría tomado el TWAP en ese momento ("quién llevaba razón") | La más local y "más asumible"; compara con lo que se quiere batir | *(Nota)* exige simular en paralelo la acción que habría tomado el TWAP |
| Volumen pendiente | Penalizar lo no ejecutado o premiar completarla | Cumple la restricción de §2 | Mal calibrada, ejecuta todo al principio |
| Desviación del plan TWAP | Distancia al calendario del TWAP | Comodidad del cliente; permite calibrar el simulador | *(Nota)* limita la mejora posible (perfil conservador, §13) |

**Nota:** las celdas marcadas *(Nota)* son inferencia propia; el resto procede de CASOS p.43-44 y T07 ~03:30-03:40.

### 9.2 Términos adicionales y calibración

- **Volumen pendiente**: penalizar si deja órdenes o recompensar si lo ejecuta todo (T07 ~03:32).
- **Desviación del plan**: "quiero un TWAP con esteroides, pero que no se diferencie mucho de un TWAP" (T07 ~03:32). Dos razones: (1) los clientes están acostumbrados al TWAP y un algoritmo en producción "que puede hacer cualquier cosa" les asusta, porque es su dinero; (2) **calibración del simulador**: con órdenes TWAP reales se puede contrastar si el simulador da un impacto realista y, si el algoritmo ejecuta parecido al TWAP, se puede asumir que el simulador sigue siendo realista para él (T07 ~03:32-03:33).
- **El problema es la calibración** de los pesos: con mucho peso a completar la orden, intenta ejecutar todo al principio; con mucho peso al retorno, si no ve buen precio deja volumen pendiente. Hay que mirar en cada momento qué consigue en cada dimensión y los gráficos, e ir ajustando (T07 ~03:34). Esto conecta con la idea B del ej. 4: sin volumen pendiente y sin llegar al final, terminando en torno al 90 % de la duración según el ENUNCIADO, o "en un rango entre un 70 [y] 90 %" según lo dicho en clase (T07 ~04:58). La discrepancia está en [notebook_partida.md](notebook_partida.md); la propuesta en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

### 9.3 Por qué el VWAP global no es factible (ejemplo dibujado, T07 ~03:35-03:39)

**Idea.** Una recompensa perfectamente alineada con el objetivo puede impedir el aprendizaje si el agente no puede relacionarla con sus características.

El ejemplo de la pizarra:
1. El precio arranca arriba y empieza a bajar; el mercado había negociado bastante volumen arriba, así que su VWAP va alto.
2. El agente (línea negra) aprovecha la bajada y compra más; como el mercado negocia poco en ese tramo, el agente va por delante, con su VWAP por debajo del de mercado.
3. Más tarde el precio cae mucho más y además el mercado negocia muchísimo volumen en esa zona: el VWAP de mercado baja con fuerza.
4. El agente ya ha gastado buena parte de su volumen y no le queda para aprovecharlo: acaba con un VWAP peor que el del mercado.

Recompensar con el slippage global obliga a predecir no solo el precio futuro sino **dónde va a negociar el mercado su volumen**, más difícil que predecir el precio (T07 ~03:37). Aunque mejorar esa métrica garantizaría batir al TWAP, el agente "no va a poder sacar relaciones" entre sus características y esa recompensa (T07 ~03:39-03:40). Hay que buscar otra igual de alineada pero predecible: VWAP slippage local o, todavía más local, comparar con el TWAP en cada momento de decisión (T07 ~03:38).

**Analogía de inversión** (T07 ~03:40-03:41): para elegir entre Repsol y petróleo, recompensar con la rentabilidad absoluta a final de año equivale a tener el problema resuelto; mejor saber si localmente se hizo mejor o peor que la referencia (la transcripción dice "Britis Protole": British Petroleum o el Brent *(dudoso)*). Más en [05_profit_seeking.md](05_profit_seeking.md).

### 9.4 Ejemplo de recompensa "súper específica" (T07 ~03:41-03:44)

Local en el tiempo y frente al benchmark. La alternativa clásica (el algoritmo clásico de BME) pone una limitada a 7,10 €; el agente, al ver alta la probabilidad de ejecución, apuesta por una "más limitada todavía", a 7,08 € (T07 ~03:42). En los momentos siguientes se comparan las dos órdenes, "la mía, la suya":
- Si la del agente (la limitada a 7,08) se ejecuta: recompensa **+2** (7,10 − 7,08) *(dudoso quién dice "la mía" y "la suya"; la cifra solo cuadra con la del agente)*.
- Si no se ejecuta y se acaba el tiempo, tiene que lanzar una agresiva a 7,13 € "porque si no me quedo por detrás", y la recompensa es **−3** (7,10 − 7,13): hizo una apuesta, le salió mal y ahora paga el impacto (T07 ~03:43).

Es una variante concreta del caso general de §9.1 (T07 ~03:38), donde el TWAP ejecuta la agresiva, el agente pone una pasiva y se evalúa "quién de los 2 llevaba razón".

Como característica basta la probabilidad de ejecución alta o baja. Es una recompensa basada en retorno, "pero ese retorno se consigue" (T07 ~03:43-03:44). Mensaje: el objetivo se traduce a veces en una fórmula matemática que no es factible y hay que aterrizarla en una *proxy* (T07 ~03:44).

**Nota:** las cifras cuadran con el precio de 7,10 € que el profesor da para la limitada clásica (T07 ~03:42; la transcripción dice "una limitada a 7 por 10"): $7{,}10-7{,}08=+0{,}02$ € (+2 céntimos, "puntos") y $7{,}10-7{,}13=-0{,}03$ € (−3). Es decir, en este ejemplo las unidades son céntimos frente al precio de la orden clásica (2 céntimos sobre 7,10 € son unos 28 pb, no 2). Es un ejemplo distinto del de T07 ~04:11-04:12 (§18: alternativa agresiva a 7,12 y coste de agresividad en pb); no deben mezclarse. El texto transcrito es confuso en este tramo ("el cla, el el algoritmo clásico, el B…"), así que la atribución exacta de cada orden es *(dudoso)*, aunque las cifras encajan.

### 9.5 Relación con la práctica

**Nota:** la recompensa por defecto del LAB es, en cada ejecución, $(\text{episode\_vwap}-\text{price})/\text{episode\_vwap}$, donde `episode_vwap` es el VWAP de mercado de **todo el episodio**, precalculado en `reset` (LAB celda 7; T07 ~04:51). Es una recompensa por step pero contra una referencia **global y futura**: con el estado de la práctica (solo tiempo y volumen restantes) el agente no puede predecir si el precio actual está por encima o por debajo de ese VWAP, que es justo el problema de factibilidad de §9.3. Por eso el ej. 2 exige modificar el sistema de recompensas (ENUNCIADO); una candidata natural es la desviación del plan TWAP (§9.2), y la idea A del ej. 4 exige enriquecer el estado si se quiere optimizar el precio. Ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md) y [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

El profesor lo dijo explícitamente: en `step` se puede calcular el VWAP de mercado hasta ese momento para ver parcialmente si se va mejor, pero no es necesario, porque las recompensas no tienen que estar condicionadas al rendimiento: en el ej. 2 solo se quiere replicar un TWAP (T07 ~04:57-04:58). Reparto que dio en clase: ej. 1, ejecutar el notebook hasta el final; ej. 2, tocar la función de recompensa; ej. 3, tocar el decay de epsilon para que no sea proporcional al número de episodios; ej. 4, libre (T07 ~04:58). Detalle en [notebook_partida.md](notebook_partida.md).

---

## 10. Agentes y buffer

### 10.1 Agentes (CASOS p.45)

La diapositiva dispersa como nube de palabras: Q-Learning, SARSA, Expected SARSA, DQL, DDQL, Dueling, A2C, A3C, aproximación lineal y cross-entropy. Valen todos los algoritmos vistos en los talleres anteriores (T07 ~03:44); ninguno predomina, y si se necesitan acciones continuas se descarta la mitad (T07 ~00:25). Se pueden combinar ideas, p. ej. un *Dueling Double DQN* al estilo Rainbow (T07 ~00:25). Teoría en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md) y en el Taller 2 ([03_dqn.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md), [04_doble_dqn_y_duelo.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/04_doble_dqn_y_duelo.md)). La práctica trae un Double DQN con buffer CER (ver [08_entorno_best_execution.md](08_entorno_best_execution.md)).

### 10.2 Tamaño del buffer (CASOS p.45; T07 ~03:44-03:50)

**Idea.** El trade-off es **independencia frente a obsolescencia**: partidas independientes entre sí para no introducir sesgos en la red, pero no tantas que respondan a políticas antiguas (T07 ~03:45).

- **Pensar en episodios, no en experiencias** (T07 ~03:46-03:47): ¿cuántas sesiones hacen falta para tener la diversidad necesaria? Si importan los regímenes (alcista, plano, bajista), con unas **100 sesiones** (~30 de cada) habría una buena distribución; si salen demasiado concentradas, hasta **500** para cubrir días muy alcistas, poco alcistas, planos, algo bajistas y muy bajistas (T07 ~03:47-03:48). Regla general: "unas 100", según la diversidad que se quiera.
- En cliff walking bastan las últimas 10 partidas porque hay pocas casuísticas; si algunas variables importantes aparecen muy de vez en cuando, hacen falta más partidas (T07 ~03:48).
- **Traducción a experiencias**: el buffer se configura en número de experiencias = sesiones × steps por sesión; con 10 momentos de decisión por sesión (un TWAP de 10 hijas) y 100 sesiones, un buffer de 1.000 (T07 ~03:49). También se puede configurar por partidas, registrando la duración de cada una de las últimas 100 y sacando la más antigua entera, con algo más de esfuerzo (T07 ~03:49-03:50). El día anterior se hizo el mismo cálculo: con decisiones cada minuto y usando ~75 % del día (no el 100 %, por las condiciones de parada), unas 8 h × 60 min dan los steps estimados por episodio, y eso se multiplica por los últimos 100 o 1.000 episodios con los que se quiera entrenar (T06 ~04:07-04:08; la transcripción dice "nueve horas de ejecución son 8 h" y "8 h por 65 por 60 min", cifras *(dudoso)*). **Nota:** 8 h × 60 = 480 steps; con 100 episodios, ≈ 48.000 experiencias, del mismo orden que en la práctica (≈ 50.000, ver abajo).

**Nota (cálculo propio):** en la práctica, con episodios de 400-600 steps (media 500), 100 episodios son ≈ 50.000 experiencias y 500 episodios ≈ 250.000. El razonamiento completo para elegir el tamaño está en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) y [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

### 10.3 ¿Lo llenamos entero? Inicialización (CASOS p.45; T07 ~03:50-03:51; T06 ~04:20-04:22)

Métodos de inicialización: **bróker** (su registro de ejecuciones), **algoritmo tradicional** (TWAP u otros algoritmos ejecutados históricamente) o **acciones aleatorias**, donde los pesos de las acciones "no tienen por qué ser iguales" (CASOS p.45; APP p.57: p. ej. $w_{\text{mantener}} > w_{\text{comprar}}$).

- Aquí el trade-off ya no es obsolescencia sino **independencia frente a ruido**: si las experiencias son ruidosas (aleatorias) mejor no llenarlo entero; si son útiles, se puede llenar más (T07 ~03:50-03:51; T06 ~04:21). No hace falta llenarlo entero: "a lo mejor no 100 partidas, a lo mejor 50" (T06 ~04:20).
- **Riesgo de sesgo**: si se llena con experiencias de un TWAP clásico, la red puede ajustarse a ese comportamiento y no salir de él (T07 ~03:51), igual que un agente de póker inicializado con partidas de un jugador muy agresivo (T06 ~04:21-04:22). El sesgo aparece si esas experiencias no se descartan pronto del buffer (T06 ~04:21). **Nota:** con un buffer FIFO salen a medida que entran experiencias propias del agente, así que el tamaño del llenado inicial frente al del buffer marca cuánto dura ese sesgo. Se pueden mezclar aleatorias y no aleatorias y no llenarlo del todo (T06 ~04:22).

**Nota:** en la práctica hay un riesgo concreto con la exploración aleatoria uniforme: con dos acciones equiprobables el agente ejecutaría en ~50 % de los steps y agotaría la orden (que solo necesita ~10 %) en la primera fracción del episodio, así que el buffer inicial quedaría lleno de episodios cortos que terminan por orden completada. Ponderar las acciones aleatorias (más peso a "no hacer nada", APP p.57) lo corrige. Ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

---

## 11. Del aprendizaje de Q-valores a la política (CASOS p.46-50)

**Idea.** La misión del agente es "tomar acciones, explorar, aprender" (CASOS p.46). Si se discretiza el estado, lo aprendido es literalmente el diccionario (tabla) de estados × 3 acciones (T07 ~03:51); con aproximación funcional, la red hace ese papel. La teoría de Q-learning está en [00_repaso_rl.md](00_repaso_rl.md) y en el Taller 1 ([02_td_sarsa_qlearning.md](../raw/referencias/rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md)).

Las diapositivas muestran la tabla Q (filas a1 = pasiva, a2 = mantener, a3 = agresiva; columnas estados $S_1, S_2, S_3, S_x$) en tres momentos, y después la política:

| | $S_1$ | $S_2$ | $S_3$ | $S_x$ | Rótulo |
|---|---|---|---|---|---|
| **0 reps** (p.47) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | "No sabe nada" |
| **500 reps** (p.48) | −0,3 / 0,1 / 1,1 | 0,4 / −0,3 / 0,2 | 0,1 / 0,6 / 1,4 | −1,2 / 0,2 / 0,8 | "Ya sabe algo" |
| **N reps** (p.49) | −2 / 4 / 5 | **7** / −3 / −3 | 1 / **6** / −4 | 3 / 1 / **8** | "Llega a una solución" |

(Valores en orden a1 / a2 / a3; en negrita los que la diapositiva resalta en verde.)

**Política (p.50)**, en formato *one-hot*: $S_2\to a_1$, $S_3\to a_2$, $S_x\to a_3$, la acción que maximiza Q en cada estado una vez convergen los valores (T07 ~03:51-03:52).

**Nota (errata de la diapositiva):** en p.49 el máximo de $S_1$ es $a_3=5$, pero en p.50 la columna $S_1$ aparece toda a 0 y p.49 no resalta ningún valor de $S_1$. La política correcta incluiría $S_1\to a_3$; la omisión parece una simplificación o un descuido del dibujo.

**Nota:** comparando p.48 y p.49, la acción greedy de $S_3$ cambia de $a_3$ (1,4 a las 500 repeticiones) a $a_2$ (6 al final). Ilustra por qué no conviene fijar la política (ni bajar la exploración) mientras los valores siguen moviéndose.

---

## 12. Análisis de resultados y pruebas de estrés (CASOS p.51)

**Idea.** No basta con mirar las recompensas: hay que testear bien, con métricas de cómo se ha comportado el algoritmo (T07 ~03:52).

- **Comparar Algo frente a TWAP fuera de train**, con métricas de media, dispersión… (CASOS p.51).
- **Situaciones extremas** (CASOS p.51):
  - situaciones reales (p. ej. vencimiento de contratos de derivados);
  - alterar el comportamiento de terceros (órdenes ficticias);
  - alterar variables privadas (p. ej. poco tiempo para mucho volumen restante);
  - alterar los tamaños de las órdenes fuera de los parámetros fijados;
  - …
- El día anterior se propusieron también escenarios tipo Covid o entornos sintéticos de estrés (T06 ~04:47). Validación en general en [05_profit_seeking.md](05_profit_seeking.md) y [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

**Nota:** el ENUNCIADO no pide validación ni test en el ej. 2 (basta con imágenes correctas en train), pero sí que el algoritmo funcione para distintos días, momentos, duraciones y tamaños dentro del rango; las pruebas de "alterar variables privadas" son una forma barata de demostrarlo.

---

## 13. Primeros resultados: perfiles de agresividad (CASOS p.52-53)

**Idea.** Los resultados proceden del TFM del profesor y de su continuación ("sacado del TFM… y luego continué mejorándolo") (T07 ~03:52-03:53): un TWAP con distintos grados de libertad respecto al plan tradicional.

| Bench | Perfil | Libertad respecto al plan | Cómo se controla |
|---|---|---|---|
| TWAP | Conservador | 5-10 % | Recompensas y reglas programadas |
| TWAP | Moderado | 10-30 % | — |
| TWAP | Agresivo | 30-50 % | — |
| TWAP | Muy agresivo | > 50 % | — |
| POV | Conservador | 5-10 % | — |

(CASOS p.53.) "> 50 % ya es muy agresivo": el algo RL puede hacer lo que quiera con tal de conseguir su objetivo → **comportamientos "atípicos"** a los que no estamos acostumbrados (CASOS p.53; ver §16).

- **Conservador** = no desviarse demasiado del plan tradicional, para dar seguridad a un potencial cliente (T07 ~03:53).
- La desviación se limita **por recompensas o por reglas deterministas**: si se desvía demasiado, "te desautorizo", se reajusta al plan original y, cuando vuelve a acercarse, se le devuelve el control (T07 ~03:53). No todo tiene que definirlo el agente.
- **Nota:** el material no define "% de libertad". La lectura más coherente con la clase es la desviación máxima permitida entre el volumen ejecutado por el agente y el que llevaría el TWAP en ese momento (en % de la orden) *(dudoso)*.

---

## 14. Gráficos de resultados por perfil (CASOS p.54-59)

**Formato común.** Panel superior: precio (rojo), media móvil (verde), **Mkt VWAP** (azul) y **Algo VWAP** (naranja), ambos acumulados desde el inicio de la orden. Panel inferior ("Market volume vs Algo volume over time"): volumen acumulado del mercado (azul) y del algo (naranja) en % del total del periodo, de 0 a 1, más barras de las órdenes del mercado (azul) y del algo (naranja, escaladas para que se vean, ya que son hijas pequeñas) (CASOS p.54; T07 ~03:54). En un TWAP puro la curva naranja inferior sería una recta. Los valores se leen de imágenes de baja resolución y son aproximados.

| Pág. | Perfil (bench) | Sesión y precio | Lo que muestra | Rótulo |
|---|---|---|---|---|
| p.54 | Conservador (TWAP) | Día 27, ~11:20-11:57; 4,590-4,615 | Volumen acumulado del algo casi recto, algo por delante del mercado; Algo VWAP y Mkt VWAP casi superpuestos (~4,602 al final). Se parece mucho a la recta del TWAP con pequeñas desviaciones (T07 ~03:54) | — |
| p.55 | Conservador (TWAP) | Día 27, ~11:50-12:55; 7,315-7,335 | El mercado negocia mucho volumen pronto (su curva azul va por encima) en una zona en que el precio cae; el Algo VWAP queda siempre por encima del de mercado (~7,321 frente a ~7,320 al final). El hueco se abrió desde el principio y, aunque intentaba recuperar, no tenía libertad para cerrarlo (T07 ~03:54-03:55) | "Peor precio" |
| p.56 | Moderado (TWAP) | Misma sesión que p.55 | Aprende a comprar más en ciertos momentos (escalones en la curva naranja, va por delante del mercado) y hay tramos sin ejecutar; el Algo VWAP converge al de mercado (~7,320) y empieza a cerrar el hueco (T07 ~03:55-03:56). Un TWAP no puede permitirse esos huecos | — |
| p.57 | Agresivo (TWAP) | Día 28, ~10:15-11:02; ~7,30-7,325 | Compra concentrada durante la bajada intermedia y mesetas largas sin ejecutar; va por delante del mercado casi todo el episodio (≈92 % ejecutado hacia las 10:52) y completa la orden al final, a la vez que el mercado; luego el precio sube. Algo VWAP (~7,311) por debajo del Mkt VWAP (~7,313) | "Mejor precio" |
| p.58 | Muy agresivo (TWAP) | Día 20, ~10:30-11:40; 6,05-6,08 | Meseta larga (~10:50-11:05) sin comprar mientras el precio está alto, compras masivas en la caída hacia el mínimo (~11:15-11:25) y orden completa antes del final. Algo VWAP (~6,060) bajo Mkt VWAP (~6,063) | "Mejor precio" |
| p.59 | Conservador (POV) | Día 20, ~12:10-13:05; precio cayendo de ~4,765 a ~4,735 | La curva del algo sigue de cerca a la del mercado (por debajo en el tramo final y alcanza el 100 % al cierre); VWAPs muy próximos, el del algo ligeramente por encima al final. Objetivo: precios próximos al POV ejecutando más o menos en los mismos tiempos (T07 ~03:58) | (sin rótulo) |

- Cuanto más agresivo, más momentos sin ejecutar y más momentos ejecutando mucho, con mejores precios (T07 ~03:56, ~03:57).
- **Advertencia del profesor sobre el realismo** (T07 ~03:56-03:57): con órdenes realistas estas asunciones no valen. Un algoritmo muy agresivo que concentra volumen barre posiciones, y el simulador, calibrado con TWAP, no garantiza un impacto realista en esos puntos. El agresivo no es creíble y el moderado y el conservador "con pinzas". Solo se lo cree con la configuración del triciclo: 400 títulos, hijas de 1 título, sin costes de transacción ni comisión, y siempre que no se haya colado ningún tipo de "look" (T07 ~03:57) *(dudoso: lo más probable es look-ahead bias, es decir, usar información futura)*.
- **Nota:** en la práctica, `episode_vwap` (el VWAP de mercado de todo el episodio, precalculado en `reset`) es información futura. Puede usarse en la recompensa, que se calcula fuera del agente, pero nunca en la observación (ver §9.5 y [08_entorno_best_execution.md](08_entorno_best_execution.md)).
- **Nota:** en p.55 la diferencia final es de alrededor de una décima de céntimo (≈ 1-1,5 pb sobre 7,32), del orden de lo que se discute en §15; lectura visual *(dudosa)*.

---

## 15. Histograma de VWAP slippage en test (CASOS p.60)

**Gráfico.** Título: "Algo Performance: +2.8 pb · TWAP: −1.88 pb (Sobre los más líquidos)"; en la figura, "Interval vwap histogram comparison in test: w_avg_algo = 2.8 vs w_avg_bench = −1.88". Eje X: slippage en pbs, aproximadamente de −20 a +30, con flechas "− Peor" a la izquierda y "Mejor +" a la derecha; eje Y: densidad (hasta ~0,175). Histograma azul (*Benchmark*, TWAP) muy concentrado con pico en torno a −2 pb; histograma naranja (*RL Algo*) desplazado a la derecha, con media ponderada +2,8 pb y cola derecha larga hasta ~+30. Línea negra vertical en 0 = VWAP de mercado; líneas punteadas en las dos medias ponderadas.

**Lo que se dijo en clase** (T07 ~03:58-04:02):
- Con coste 0 y poco impacto (órdenes pequeñas), el TWAP clásico da un resultado negativo frente al VWAP de mercado y el algoritmo queda en positivo, mejor que el mercado. El profesor lo atribuye al conservador ("un algo pequeñito, conservador… lo estoy haciendo mejor que el mercado") *(dudoso: no está claro si el histograma corresponde al perfil conservador o a todos)*.
- Repite, remitiendo a lo dicho antes, que el gráfico "da la vuelta" al signo habitual (T07 ~03:59; cf. T07 ~01:59 sobre CASOS p.27) *(dudoso)*. **Nota:** las etiquetas de p.60 son consistentes (derecha = mejor, slippage medido como ahorro); el comentario probablemente remite a la diapositiva de distribuciones de los algoritmos clásicos (CASOS p.27), cuyo eje sí discutió como invertido (ver [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md)).
- **De ejecución a inversión**: si un algoritmo de compra rasca 2-3 pb y uno de venta otros 2-3 pb, combinándolos se gana dinero y se parece a un algoritmo de alta frecuencia o de inversión (T07 ~03:59-04:00). Pero con hijas de 1 título y coste 0 la capacidad de absorción es mínima, y al meter costes de transacción las ganancias desaparecen, aunque la diferencia relativa se mantenga; se mencionan las cifras "5" y "−8" sin que quede claro a qué se refieren *(dudoso: quizá resultados del algo y del TWAP con costes)* (T07 ~04:00-04:01). Con un solo título de Santander y comisiones 0 se podría plantear "ir rascando poco a poco" (T07 ~04:01-04:02).
- Al final de la parte de papers vuelve sobre la idea: visto como algoritmo de inversión habría que rascar más de ~2,8 pb de media para compensar los costes (T07 ~04:05) *(dudoso: 2,8 coincide con la media del algo en p.60; puede referirse a que esa ventaja es la que tendría que cubrir los costes, o a un coste típico de ese orden)*.

**Nota (dispersión):** a simple vista la distribución naranja es bastante más ancha que la azul (de ~−10 a ~+30 frente a ~−15 a ~+5), lo que no cumple el "mejorar o igualar dispersión" de CASOS p.29: el algo gana en media a costa de más variabilidad, el patrón del "algoritmo agresivo" que el profesor describió (T07 ~01:46). Lectura visual *(dudosa)*.

**Nota (por qué el TWAP sale negativo):** una hipótesis coherente es que el TWAP compra con agresivas al ask mientras el VWAP de mercado se calcula a precios medios, de modo que paga sistemáticamente parte del spread. En la práctica pasará lo mismo: la acción agresiva usa `ask1` y el VWAP de mercado usa el mid (LAB celda 7), así que un TWAP perfecto tendrá slippage negativo del orden de medio spread. Es una hipótesis propia sobre la figura del TFM; para el notebook es un hecho del código (ver [08_entorno_best_execution.md](08_entorno_best_execution.md)).

---

## 16. Comportamientos atípicos

**Idea.** Un agente con mucha libertad hará lo que sea para conseguir su objetivo, aunque no se parezca a nada que haría un humano (CASOS p.53; APP p.61, p.96).

- En los perfiles agresivo y muy agresivo aparecen mesetas largas sin ejecutar y ráfagas que concentran casi toda la orden (§14): con impacto real y un simulador calibrado con TWAP no se puede saber si eso funcionaría (T07 ~03:56).
- Por eso los comportamientos sensibles se acotan con reglas (APP p.61): desautorizar y reajustar al plan (T07 ~03:53), forzar control al final del episodio (la acción "mantener" que al final no pone orden, §8.3).
- Y por eso el despliegue en producción debe ser controlado: el simulador no garantiza fidelidad del 100 % (APP p.96). Ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

**Nota:** para la práctica, los "atípicos" esperables son el agente que compra todo al principio (si pesa mucho completar la orden), el que deja volumen pendiente (si pesa mucho el precio) (T07 ~03:34) y el que termina antes de tiempo; son exactamente los que la idea B del ej. 4 pide evitar (ENUNCIADO).

---

## 17. Papers y estado de la industria (CASOS p.61; T07 ~04:02-04:06)

### 17.1 *Reinforcement Learning for Optimized Trade Execution* (2006)

CASOS p.61 lo atribuye a Lehman Brothers y lo califica de "muy sencillo":
- tipo *arrival price*;
- Q-learning modificado (tabular: todavía no había redes neuronales en este campo, T07 ~04:02);
- AMZN, NVDA y QCOM, 1,5 años de datos;
- órdenes pequeñas: 5-10 mil acciones en 2-8 min;
- acción: precio límite relativo al ask, "ask − a";
- **con solo 2 variables privadas, tiempo restante T e inventario restante I, mejoran el benchmark un 50 %**. El benchmark es *Submit & Leave* (CASOS p.61): dejar la orden límite (T07 ~04:02-04:03). **Nota:** en el paper, lo que no se ejecuta se completa a mercado al final del horizonte.

Con esas dos variables ya se puede ajustar la agresividad de las acciones: saber si hay mucho tiempo para el volumen que queda (T07 ~04:02-04:03). Añadiendo spread y otras variables de mercado se mejora un 7-8 % adicional (T07 ~04:03). Era un problema muy acotado, con acciones pequeñas y discretizadas porque sin discretizar no se puede aprender con Q-learning tabular; el paper es corto, unas 7 páginas según el profesor (T07 ~04:03).

**Tabla 1 del paper reproducida en p.61** ("Additional trading cost reduction when introducing market variables"):

| Variables de mercado añadidas | Reducción adicional del coste |
|---|---|
| Bid-ask spread | 7,97 % |
| Bid-ask volume misbalance | 0,13 % |
| Spread + immediate cost | 8,69 % |
| Immediate market order cost | 4,26 % |
| Signed transaction volume | 2,81 % |
| Spread + immcost + signed vol | 12,85 % |

**Nota:** el paper es de Nevmyvaka, Feng y Kearns (ICML 2006); el dato no viene del material. Dos conexiones con la práctica: (1) su estado base, tiempo e inventario restantes, es el mismo que el del LAB (§7.5); (2) el desbalanceo de volumen apenas aporta (0,13 %) mientras que el spread es lo más útil, una pista para elegir características en la idea A del ej. 4.

**Nota:** el paper optimiza frente al precio de llegada (*arrival price*, con benchmark *Submit & Leave*) y no frente al VWAP, que es la métrica del caso (§2, que ignora el arrival price), así que su 50 % no es comparable con la mejora en pbs de §15. El "otro 7 u 8 %" que dice el profesor (T07 ~04:03) coincide con las filas *Bid-ask spread* (7,97 %) y *Spread + immediate cost* (8,69 %) de la Tabla 1.

### 17.2 J.P. Morgan (DNA)

CASOS p.61 recoge tres frases de J.P. Morgan sobre su algoritmo DNA: que el RL entrena al algoritmo para aprender de las acciones que toma; que es útil para modelar la realidad porque las relaciones en la vida real suelen ser complicadas; y que da más flexibilidad y elimina posibles sesgos humanos. Según el profesor era un algoritmo muy flexible (tamaño, momento y tipo de orden) y se habría cerrado recientemente por ser caja negra; él mismo dice desconocer sus tripas *(dudoso)* (T07 ~04:03-04:05). Si se restringe a una sola dimensión, el margen de mejora es menor (T07 ~04:05).

### 17.3 Qué usa la industria

En la práctica se siguen usando TWAP, VWAP y POV; algunas firmas dicen que usan machine learning para ejecutar sus órdenes sin detallar qué (T07 ~04:04). El contexto de negocio (cuota de uso de cada algoritmo, por qué los brokers no lo mejoran) está en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md).

---

## 18. Supervisado frente a RL: la utilidad esperada como benchmark (T07 ~04:10-04:16)

**Idea.** En banca el RL se usa poco y muchas veces basta con supervisado bien planteado; la solución "matemática" con probabilidades supervisadas es el benchmark que el RL debe superar.

Ejemplo de clase (a pregunta de un alumno): 4-5 niveles posibles para una limitada, con probabilidades de ejecución dadas por un supervisado de **80 %, 60 %, 50 % y 10 %** (el último muy lejos), frente a la alternativa de una agresiva a 7,12 €, que es lo que haría el TWAP. Si la limitada no se ejecuta, al final hay que agredir, y el coste medio de la agresividad suele ser de **unos 3 pb sobre el precio medio del libro** (T07 ~04:11-04:12). Con eso se calcula de forma determinista la utilidad esperada de cada acción: "2 de ahorro por 80 % más 20 % por 3" (T07 ~04:12).

**Nota (formalización propia):** leyendo el "más 20 % por 3" como coste (signo negativo):

$$
\mathbb{E}[U(a_k)] = p_k\cdot \text{ahorro}_k - (1-p_k)\cdot \text{coste}_{\text{agresiva}}
$$

**Nota (cálculo):** con los números de clase, $0{,}8\cdot 2 - 0{,}2\cdot 3 = +1{,}0$ (el coste de 3 está en pb, T07 ~04:12:22; el "2 por 100" de ahorro es *(dudoso)*, probablemente 2 pb), mejor que agredir ya (0 frente al TWAP). Para un nivel lejano con $p=0{,}10$ habría que ahorrar más de $0{,}9\cdot 3/0{,}1 = 27$ unidades para compensar, lo que ilustra por qué el nivel más alejado casi nunca compensa.

**Nota (umbral):** con ahorro 2 y coste 3 en las mismas unidades, $U(p)=2p-3(1-p)=5p-3$: $U(0{,}8)=+1{,}0$, $U(0{,}6)=0$, $U(0{,}5)=-0{,}5$, $U(0{,}1)=-2{,}5$; el umbral de indiferencia es $p^*=3/5=60\,\%$, que coincide con la cifra ilustrativa "a partir del 60 por 100" de T07 ~03:08:04 (ver [06 §13.2](06_algoritmos_de_ejecucion.md)). Matiz: si los 3 pb son el coste de la agresividad sobre el precio medio del libro (T07 ~04:12:22), la alternativa del TWAP (agresiva a 7,12) también lo paga y la limitada no ejecutada solo pierde por la deriva del precio mientras espera, así que el umbral sería otro. La coincidencia con el 60 % no demuestra nada; que un RL con solo la probabilidad como estado converja a esta solución de DP es una hipótesis, y su valor añadido estaría en las características extra que desplazan el umbral.

- Eso es **programación dinámica** con probabilidades de transición conocidas: cada acción lleva a un escenario positivo o negativo con probabilidad conocida y recompensa conocida (T07 ~04:13). No hace falta un sistema de RL para esto, y probablemente explica el 80-90 % de la varianza (T07 ~04:13-04:14).
- El RL entra si se le añade información (frecuencia de trades, spread actual) que cambie esas esperanzas: el benchmark es la solución matemática y el RL entrenado con las características extra debería superarla (T07 ~04:13-04:14).
- El ML clásico está infrautilizado y puede coexistir con reglas y con otros sistemas supervisados y no supervisados; la pregunta es en qué capa aplicar cada cosa (T07 ~04:14-04:15). El RL tiene más potencia porque tiene en cuenta las interacciones, pero arrastra los escollos de recompensas, exploración y entorno (T07 ~04:16). Más en [00_repaso_rl.md](00_repaso_rl.md).
- **Criterio práctico con que cierra la respuesta** (T07 ~04:16): el ruido en la toma de decisiones o en las etiquetas, y la interacción con el mercado o contigo mismo (estado privado), son "papeletas a favor del aprendizaje por refuerzo". Con RL se pueden resolver la mayoría de problemas supervisados (el ejemplo del paraguas), y si sus escollos no son un problema y apetece probar, merece la pena porque es más potente.

---

## 19. Lecciones del TFM del profesor

**Contexto** (T06 ~02:57-03:00). En su TFM de 2019 (primera edición del máster) el director le ofreció trabajo y le cambió el tema, de fondos de inversión a algoritmos de ejecución: BME tenía algoritmos de ejecución basados en reglas y quería probar con machine learning. Leía papers y diseñaba recompensas por la mañana y lanzaba experimentos por la noche: llegó a lanzar 72 experimentos en paralelo, con recompensas y configuraciones distintas, en una máquina de unos 700 GB *(dudoso: la unidad puede ser de RAM o de disco)*. Muchos de sus consejos vienen de ahí. Al acabar pensó "cuántas cosas he hecho mal" y siguió profundizando con compañeros que sabían más. El proyecto no salió a producción porque BME perdía cuota frente a grandes operadores con mejores interfaces, y a los brokers les daba igual ganar 2-3 pb porque el dinero es del cliente. La imagen de APP p.62 (decenas de carpetas de experimentos y un monitor con todos los núcleos al 100 %) ilustra esa prueba masiva.

**Lo que haría distinto** (T07 ~04:06-04:10):
1. Al principio no conocía bien el funcionamiento interno del libro de órdenes; los compañeros y los brokers de BME le ayudaron a elegir características.
2. Quiso cubrir momento, tamaño y tipo a la vez; habría sido mejor **centrarse solo en el tipo de orden** (pegada o limitada) con una probabilidad de ejecución por nivel, que tenga en cuenta la prioridad en la cola, entrenada con datos históricos con un árbol, y después una evaluación correcta.
3. El error común es **intentar abarcar demasiado**.

---

## 20. Qué se lleva la práctica

**Nota:** resumen de qué decisiones de este documento ya vienen fijadas en la práctica y cuáles quedan abiertas; la propuesta razonada está en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md) y los requisitos exactos en [notebook_partida.md](notebook_partida.md).

| Decisión | Fijada por la práctica | Abierta (y dónde se usa) |
|---|---|---|
| Episodio | 400-600 steps de 5 s; orden ≈ 7,5-12,5 % de los steps; hijas de 1 título; impacto y costes 0 | Duración más larga (2-3 h), otros activos/días (requisito del ej. 2: distintos días, momentos, duraciones y tamaños) |
| Qué decide el agente | Solo el momento; agresiva de 1 título | Tipo de orden o tamaño (ej. 4, libre) |
| Estado | Tiempo y volumen restantes | Ratio/diferencia, spread, liquidez, frecuencia, desbalanceo, tendencia (ej. 4 idea A) |
| Recompensa | La por defecto (frente al VWAP del episodio) | Diseño para replicar el TWAP (ej. 2); volumen pendiente y margen del 90 % (ej. 4 idea B) |
| Agente | Double DQN (no obligatorio; vale incluso Q-learning tabular con 2 estados × 2 acciones, T07 ~04:54-04:55) | Algoritmo, PyTorch frente a TF (ENUNCIADO) |
| Buffer | Tamaño e inicialización del notebook | Dimensionar en episodios (§10), inicialización ponderada |
| Análisis | Gráficos de precio/VWAP y volumen acumulado (formato de §14) | Media y dispersión del slippage, estrés de variables privadas (§12) |

Para el ej. 2 el profesor recordó que un TWAP determinista es un `if`: si vas por delante del plan no ejecutes y, si vas igual o por detrás, lanza otra hija; se puede hacer así, pero se pide conseguirlo **por recompensas**, y se puede tirar el notebook entero y montar el entorno como se quiera mientras con esa parametría, SAN o REP, se llegue al TWAP por recompensas y se muestren las imágenes (T07 ~04:52-04:56).

---

## 21. Erratas, dudas y lecturas alternativas

| # | Dónde | Problema | Lectura propuesta |
|---|---|---|---|
| 1 | CASOS p.50 | Columna $S_1$ de la política toda a 0, aunque en p.49 su máximo es $a_3=5$ | Debería ser $S_1\to a_3$ (§11) |
| 2 | CASOS p.32 | "+ Específico / − Flexibilidad" puede leerse como del enfoque realista | Por la maquetación y la clase, son del enfoque alternativo (§4.3) |
| 3 | CASOS p.35 | $P_a$ no definido en $\ln(P_t/P_a)$ | Probablemente precio anterior (rendimiento a corto), no arrival (§7.2) |
| 4 | CASOS p.38 | Fila "posición 0" sin explicar | Posibles ejecuciones sin consumir la 1.ª posición visible *(dudoso)* |
| 5 | CASOS p.53 | "% de libertad" sin definir | Desviación máxima permitida respecto al plan TWAP *(dudoso)* |
| 6 | CASOS p.60 | La dispersión del algo parece mayor que la del TWAP | No cumple "mejorar o igualar dispersión" de p.29 (§15) |
| 7 | CASOS p.52-60 | No se indica la fuente de los resultados | La clase los atribuye al TFM del profesor y su continuación (T07 ~03:52) |
| 8 | T07 ~03:06 | Tamaños de la alternativa (c): "el doble" o "un 50 % más" | Ambas posibles; CASOS p.41 usa ×2 y ×0,5 |
| 9 | T07 ~03:43 | +2 / −3 "puntos" | Céntimos frente a la limitada clásica a 7,10 € (T07 ~03:42) (§9.4) |
| 14 | T07 ~03:13 | "El mercado no se mueve 10 céntimos en 5 s" al comparar 7,10 y 7,03 € | Son 7 céntimos; redondeo oral o error de transcripción (§7.4) |
| 10 | T07 ~04:01, ~04:05 | Cifras "5", "−8" y "2,8 para cubrir costes" | Sin referente claro (§15) |
| 11 | T07 ~04:47 | "720 por 0 9 = 65" | $720\times 0{,}09\approx 65$ (§4.4) |
| 12 | T07 ~04:05 | Cierre del algoritmo de J.P. Morgan por ser caja negra | Afirmación oral sin fuente, el profesor admite no conocerlo bien |
| 13 | T06 ~02:58 | "máquina de 700 gigas" | RAM o disco, no se precisa |
