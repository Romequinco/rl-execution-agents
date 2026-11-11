# Exploración, memoria de experiencias y estabilidad del entrenamiento

> **Fuentes:**
> - **T06** = transcripción automática de la clase del 06/11/2026 (`docs/raw/26 11 06.pdf`). Tramos usados: ~00:51-00:54 (actualización con α), ~01:12-01:20 (estabilidad, Expected SARSA, Double Q-learning), ~01:26-01:40 (experience replay, no hay algoritmo mejor, cierre sobre estabilidad), ~02:53-03:00 (semillas, reglas, experimentos del TFM y por qué no llegó a producción), ~03:00-03:42 (exploración, ruido y priorización, α), ~03:54-04:37 (buffer, PER, CER), ~04:44-04:52 (reset como test, *catastrophic forgetting*, loss).
> - **T07** = transcripción automática de la clase del 07/11/2026 (`docs/raw/26 11 07.pdf`). Tramos: ~00:18-00:42 (entrenamiento, ejemplo *cliff walking*, inestabilidad, hiperparámetros, producción, repaso de exploración), ~01:47 (benchmark para bajar la exploración), ~03:44-03:53 (buffer en el caso de ejecución), ~04:24-04:27 (aparcamiento), ~04:28-04:31 y ~04:49 (puntos de la práctica y bucle del notebook), ~05:00-05:01 (qué parámetros importan).
> - **APP** = diapositivas *Applying Reinforcement Learning* (`docs/raw/MIAX14/applying_RL_MIAX.pdf`): p.1, p.38, p.48, p.57, p.61-74, p.77-82, p.87-88, p.90-97.
> - **CASOS** = diapositivas *Casos Prácticos* (`docs/raw/MIAX14/casos_practicos.pdf`): p.45-46, p.51.
> - **ENUNCIADO** = `docs/raw/MIAX14/ejercicios_taller_rl.docx` (instrucciones generales y ejercicios 1 a 4).
> - **LAB** = `docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`, celdas 7, 11, 12, 15, 16, 18, 20. **BE dqn.py** = `docs/raw/MIAX14/best-execution/src/agents/dqn.py`.
> - **RLM** = notebooks de `docs/raw/MIAX14/rl-material/notebooks/`: `cliff-walking.ipynb` (celdas 4, 8-11), `insights.ipynb` (celdas 3-9 y 19-23), `rl-models-basic/TD-q-learning.ipynb` (celdas 2, 6-7), `rl-models-basic/deep.ipynb` (celdas 2-3).
> - **T2** = Taller 2 (`docs/raw/referencias/deep-rl-trading-agents/docs/teoria/`), para enlazar lo ya desarrollado.
>
> **Convenciones:** "(T06 ~03:03)" = hora aproximada en la transcripción; "(APP p.64)" = página de las diapositivas; "(LAB celda 18)" = celda del notebook, numeradas desde 0. Las transcripciones son automáticas y muy ruidosas: todo lo que viene de ellas es paráfrasis, y lo que no se entiende bien va marcado *(dudoso)*. **Nota:** = contenido propio (cálculos, correcciones, conexiones, propuestas), que no está en el material. Las erratas se señalan con su corrección razonada (§13). Se usa $\varepsilon$ para la exploración, $\alpha$ para el *step size* o *learning rate*, $m$ para el tamaño del buffer y $B$ para el batch.
>
> Escrito el 10/11/2026.

---

## 1. Qué cubre este documento y la idea que lo une todo

**Idea.** En la segunda parte de APP el profesor ordena las piezas de un sistema de RL en tres bloques (APP p.48): qué ve y hace el agente (estados, acciones, recompensas), **cómo aprende** (exploración $\varepsilon$, *step size* $\alpha$, $\gamma$ y *experience replay*) y cómo se pone en marcha (entorno, entrenamiento, retos). Para cada pieza da principios, errores típicos y ejemplos en bolsa. Este documento trata el segundo bloque, salvo $\gamma$, y el entrenamiento y los retos del tercero.

- Estados y acciones: [02_diseno_de_entornos.md](02_diseno_de_entornos.md). Recompensas y $\gamma$: [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).
- Mecánica de DQN (no i.i.d., "juez y parte", *fixed target*, Double DQN): [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md). Aquí solo se trata el buffer como hiperparámetro y su gestión.
- La palabra que repite el profesor en todos estos temas es **estabilidad**. Los entrenamientos son aleatorios porque dependen de los caminos recorridos y de la exploración. Muchas mejoras de los papers solo buscan dar más estabilidad. Mensaje de cierre de ese bloque (paráfrasis): olvidar los nombres de los algoritmos y quedarse con la palabra estabilidad (T06 ~01:19; lo repite en ~01:40, junto con los sesgos que corrigen las "copias").
- Otra idea transversal: cada componente "tiene sus 20 papers" sobre cómo ajustarlo, y todos pueden complicarse o volverse dinámicos (T07 ~00:18, ~00:41-00:42; T06 ~04:11).

---

## 2. Exploración y $\varepsilon$-greedy

### 2.1 Qué es una política de exploración

**Idea.** El profesor abre el tema preguntando qué es una política de exploración y cuáles se conocen (APP p.63). $\varepsilon$-greedy, por definición: en un porcentaje $1-\varepsilon$ de los pasos se ejecuta la acción que se cree óptima ahora, y en el resto se explora al azar (T06 ~03:01).

$$
a_t=\begin{cases}\arg\max_a Q(s_t,a) & \text{con prob. } 1-\varepsilon\\ \text{acción aleatoria} & \text{con prob. } \varepsilon\end{cases}
$$

- Se empieza **siempre con $\varepsilon=1$**, porque al principio el agente no sabe nada, y se va reduciendo, como en cualquier aprendizaje (T06 ~03:01-03:02).
- No interesa bajar a 0: explorar permite siempre salir de un mínimo local. **Dejar de explorar es dejar de aprender**, y eso ya es por definición la puesta en producción, donde solo se ejecuta el conocimiento capturado (T06 ~03:02).
- Q-learning es *off-policy*: aprende la política *greedy* pero actúa con $\varepsilon$-greedy, por ejemplo con un 5 % aleatorio y un 95 % *greedy* (T06 ~00:53). Para SARSA, Expected SARSA y la distinción *on/off-policy* ver [00_repaso_rl.md](00_repaso_rl.md).

En el código de clase la regla es siempre la misma: `if np.random.rand() > self.epsilon: argmax(Q) else: acción aleatoria uniforme` (RLM cliff-walking, celda 8; RLM deep, celda 2). El agente de la práctica añade dos detalles (BE dqn.py, act): en el paso 0 devuelve siempre la acción 1, y con `set_trainable(False)` actúa siempre *greedy*. Además, la acción aleatoria la toma de `env.action_sample()`, cuya distribución se deja por elegir en el LAB (§4.5). Ver [08_entorno_best_execution.md](08_entorno_best_execution.md).

### 2.2 La estrategia clásica y el *epsilon decay* por episodios

**Idea.** La estrategia clásica es empezar con mucha exploración e irla bajando **conforme mejora el aprendizaje** (*epsilon decay*). Lo que se hace en la práctica es otra cosa: fijar de antemano un número de episodios y aplicar un decaimiento lineal o exponencial. La diapositiva lo califica de "¡Está mal!" porque es muy arbitrario: "¿por qué, e.g., 10.000?" (APP p.64).

Cómo se programa (T06 ~03:03-03:04): se decide entrenar, por ejemplo, 10.000 episodios y en cada uno se reduce $\varepsilon$ un poco, según un plan, hasta llegar al 5 % al final. Es la forma "fácil y didáctica" para probar un algoritmo sencillo. Por qué no le gusta:

1. **Puede bajar demasiado pronto o demasiado tarde**: quizá el agente necesitaba más exploración, o ya no necesitaba tanta (T06 ~03:04).
2. **Depende del número de episodios**, que es arbitrario: "¿por qué 10.000 y no un millón?" (T06 ~03:04).
3. **La forma de la curva también es arbitraria.** En la pizarra compara un patrón que baja muy rápido y a mitad de entrenamiento se queda con un 5 % residual con otro que explora más al principio. No se pueden anticipar las necesidades, y la forma depende del número inicial de episodios (T06 ~03:05-03:06).
4. **No se aprende de los experimentos**: es la regla que se usa en muchas prácticas porque es sencilla y no da dolores de cabeza, pero no obliga a entender qué está pasando (T06 ~03:20).

Así está en los notebooks y en muchos ejemplos (T06 ~03:04). Al presentar el bucle de la práctica lo repite: está puesto "con la definición clásica que decía que no me gustaba", con 10.000 o 1.000 episodios, da igual (T07 ~04:49).

| Código | Decaimiento de $\varepsilon$ | Episodios |
|---|---|---|
| RLM TD-q-learning, celda 7 (CartPole tabular) | lineal de 1 a 0,05 en el 90 % de los episodios: $0{,}95/9000\approx1{,}06\cdot10^{-4}$ por episodio | 10.000 |
| RLM cliff-walking, celda 9 | lineal de 1 a 0,05 en la mitad de los episodios: $0{,}95/150\approx0{,}0063$ por episodio | 300 |
| RLM deep, celda 3 (LunarLander) | lineal de 1 a 0,05 en el 90 % de los episodios | 100 |
| LAB celdas 12 y 18 (práctica) | `epsilon = max(epsilon - epsilon_decay, min_epsilon)` al final de cada episodio; los valores son TODO | TODO |

El LAB define `epsilon_decay` como el "valor numérico mediante el cual se reduce el valor de epsilon cada determinado número de episodios", y `min_epsilon` como el mínimo "por si se quiere dejar aleatoriedad" en entrenamientos muy largos (LAB celda 11). Es justo el esquema que el ejercicio 3 pide sustituir (§3.6).

### 2.3 El ejemplo en directo: *cliff walking* (por qué la curva con exploración engaña)

**Idea.** El profesor reutiliza el *cliff walking* del día anterior para mostrar el problema (T07 ~00:19-00:23).

- Configuración (RLM cliff-walking, celda 9): Q-learning tabular, 300 episodios, $\gamma=0{,}99$, $\alpha=0{,}1$, $\varepsilon$ lineal de 1 a 0,05 con paso $0{,}95/150$, de modo que llega al mínimo hacia el episodio 150 de 300 (T07 ~00:20-00:21).
- Pregunta a la clase: mirando la curva de recompensa por episodio, ¿cuándo diríais que ha aprendido? La curva parece estabilizarse hacia el episodio 150, "qué casualidad", justo cuando $\varepsilon$ toca el mínimo. Esa curva es **la de entrenamiento, con exploración**: la parte inicial tiene recompensas malas porque explora mucho, y al final sigue habiendo volatilidad por el 5 % residual (T07 ~00:21-00:22).
- Al evaluar **sin exploración** se ve que el agente había aprendido hacia el **episodio 30**. El problema es muy sencillo y aprende muy pronto: con 20 episodios evaluados correctamente se vería que la recompensa ha subido y que habría que bajar ya la exploración y terminar el entrenamiento (T07 ~00:22).
- Conclusión: el *epsilon decay* temporal introduce un sesgo en la lectura, y por eso no le gusta (T07 ~00:20).

**Nota:** en el notebook, `total_reward` acumula $\gamma^{t}r_t$ (retorno descontado) y el *cliff walking* da $-1$ por paso y $-100$ al caer, así que la curva de entrenamiento mezcla dos efectos: la política aprendida y las caídas por exploración. Con $\varepsilon=0{,}05$ y un camino de 13 pasos junto al acantilado la caída sigue siendo frecuente, y por eso "sigue habiendo volatilidad". En el material no se ve el código de la evaluación *greedy* que el profesor ejecutó; la celda 11 solo hace un recorrido *greedy* al final con `act_no_explore`.

### 2.4 El ejemplo CartPole y los "10.000 episodios"

El notebook que el profesor señala como uno de los dos más importantes, `TD-q-learning` (APP p.1), es un **CartPole tabular**: discretiza el estado continuo con una rejilla (`grid_step=50`) cuyo rango se calcula con 500 episodios de acciones aleatorias (percentiles 10-90), y entrena 10.000 episodios con $\gamma=0{,}995$, $\alpha=0{,}1$ y $\varepsilon$ de 1 a 0,05 (RLM TD-q-learning, celdas 6-7). Es exactamente el patrón "10.000 episodios" que critica APP p.64. En T07 ~00:34 recuerda una práctica anterior de 10.000 episodios que tardaba una hora y en la que se veía evolucionar el agente muy lentamente *(dudoso: no queda claro a qué práctica se refiere)*.

**Nota:** en el Taller 2 se vio el mismo esquema en `1-cart_pole_QLearning` ($\varepsilon$ de 1 con ×0,999 por episodio y 10.000 episodios previstos), interrumpido en el episodio 2.725 con $\varepsilon\approx0{,}065$; la curva de entrenamiento crecía con muchísimo ruido, y el test *greedy* dio 426 pasos en un episodio. Ver [T2 03_dqn.md §12](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md). Es otro ejemplo de que la curva con exploración no dice cuándo ha aprendido el agente.

---

## 3. Bajar $\varepsilon$ según la mejora

### 3.1 Qué significa "mejorar"

**Idea.** Si $\varepsilon$ debe bajar "conforme mejora el aprendizaje", hay que definir mejorar. El profesor da **dos dimensiones** (APP p.64; T06 ~03:03; T07 ~00:35):

1. **Mejorar resultados**: la utilidad obtenida, es decir, la suma de recompensas descontadas de los episodios.
2. **Conocer el espacio**: haber visitado más estados y probado más acciones, aunque los resultados todavía "no hayan terminado de explotar".

Con ambas hace falta "un sistema" que configure la exploración según esos criterios, no según el tiempo: por ejemplo, al alcanzar cierto nivel de recompensa, bajar $\varepsilon$ (T07 ~00:35-00:36). El segundo criterio se mide con conteos o pseudo-conteos de lo visitado (§4.3), que tiene sentido también con estados continuos si se discretizan: un alumno objeta que con estados infinitos no se puede contar, y la respuesta es pseudo-contar aquello que importa (T07 ~00:36-00:37).

### 3.2 Evaluación *greedy* periódica

**Idea.** Lo que hay que hacer en lugar del decaimiento por episodios es **evaluar la política cada cierto tiempo** y bajar $\varepsilon$ según esa evaluación (T06 ~03:04-03:05).

- Se **para la exploración un momento** y se evalúa la política en una serie de escenarios, **en varios episodios y no en uno**, porque un único episodio es muy ruidoso. Mientras se explora no se está midiendo la política (T06 ~03:16-03:17).
- Se recopilan recompensas y resultados y se decide si va en la buena dirección y si hay una evolución que permita bajar un poco la exploración (T06 ~03:17).
- **Se mantiene $\varepsilon$ alto mientras no mejore** y se empieza a bajar cuando la "fuerza bruta" encuentra cosas con sentido (T06 ~03:17-03:18). La exploración es muy importante al principio, "cuanto más mejor", pero no es infinita: en algún momento hay que empezar a bajarla (T06 ~03:19).
- Es una evaluación "como la de cualquier otro modelo" (T07 ~00:18-00:19). La diapositiva de entrenamiento lo formula así: cada X steps, validación en datos no vistos en **modo ejecución** (*greedy*), comparando train y validación; el modo ejecución dice además si está aprendiendo, porque los resultados del entrenamiento confunden por la exploración (APP p.90).

### 3.3 Plan de bajada ligado al objetivo, con benchmarks

**Idea.** El plan de bajada de $\varepsilon$ no se hace por episodios sino **respecto al objetivo**: "ya está a mitad del objetivo, puedo bajar un poco la exploración", porque hace el aprendizaje más eficiente (T06 ~03:19). Para eso hay que ponerle números al objetivo.

- **Coste de oportunidad.** Conocerlo es "muy importante" para ajustar la exploración y el entrenamiento. Si no se conoce, se fija un objetivo o nivel de aspiración razonable, se usa un *baseline* o *benchmark*, y se ajustan los niveles al alza o a la baja según se consigan (APP p.66).
- **Qué *baselines*** (APP p.66; T06 ~03:23-03:24): índices de referencia (batir al Ibex 35), algoritmos sin ML o soluciones clásicas al problema, objetivos de viabilidad del proyecto ("necesito rascar 20 pb o no lo pongo en producción") y "vecinos", es decir, averiguar si otros ya lo hacen y qué consiguen.
- No basta saber si ha subido o bajado: hay que saber **a qué distancia se está** del objetivo (T06 ~03:22-03:23).
- **Objetivos crecientes.** Se empieza con algo muy sencillo, "el triciclo", y si se consigue, la siguiente vez se ponen objetivos más ambiciosos (T06 ~03:23).
- **En el caso de ejecución**: TWAP, VWAP y POV programados sobre los históricos dan una referencia; si el agente se acerca cada vez más al *benchmark*, se puede ir bajando la exploración, y si sigue muy lejos, no (T07 ~01:47). Ver [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md).

### 3.4 Subir $\varepsilon$ si empeora, checkpoints y doble criterio

- **Doble criterio.** Se puede poner un máximo temporal: "voy a dejarlo un tiempo a ver si aprende", para que no corra eternamente (T06 ~03:04-03:05), o parar porque se quiere ver el resultado al día siguiente (T06 ~03:20).
- **Copias en distintos momentos** (checkpoints): permiten evaluar y continuar desde cualquiera de ellas (T06 ~03:20).
- **Si los resultados empeoran**, se puede volver a subir la exploración, o volver al checkpoint anterior ("voy a recuperar esta versión del modelo, que todavía no se había perdido"), parar la ejecución y continuar desde ahí (T06 ~03:21).
- Aun así habrá que ajustar a mano, porque puede atascarse en mínimos (T06 ~03:19-03:20).

### 3.5 Exploración frente a explotación: el mapa de Madrid y el tesoro lejano

**Idea.** Explorar poco da una solución mediocre; un poco más, un buen resultado; mucho más podría encontrar "el Santo Grial", pero puede no existir, y explorar mucho tiempo sin un criterio claro arriesga perderse y no llegar a nada (T06 ~03:21-03:22). La diapositiva lo plantea como dilema: "¿cuándo considero que conozco bastante? ¿me valen estos resultados?" (APP p.64).

- **APP p.65 (imagen):** una cuadrícula grande con el agente arriba a la izquierda. Justo al lado hay una pila pequeña de monedas grises, algo más lejos una pila mediana de oro y, en la esquina opuesta, un gran tesoro. Ilustra que explotar lo cercano da poco y que el gran premio solo aparece explorando lejos.
- **APP p.67-71 (imágenes):** mapa del centro de Madrid con un coche abajo y una bandera de meta arriba. El área sombreada en rojo es la zona susceptible de exploración: con $\varepsilon=1$ todo el mapa, con flechas y un "¿?" junto al coche (p.68); con $\varepsilon=0{,}9$, casi todo (p.69); con $\varepsilon=0{,}5$, una zona amplia alrededor de la ruta, incluido el Retiro (p.70); con $\varepsilon=0{,}1$, solo dos corredores, por delante y por detrás del Retiro, hasta la bandera (p.71).
- Explicación (T06 ~03:24-03:26): al principio no se conoce el mapa. Con $\varepsilon$ muy pequeño pero infinitos intentos todo se exploraría alguna vez; al bajarlo, la exploración se concentra en las zonas que interesan y no se pierde tiempo en el centro si lo que interesa es ir por detrás del Retiro. Esto ayuda mucho al entrenamiento, porque si no, "no termina de converger nunca". Al final quedan dos alternativas, por delante o por detrás del Retiro. Si se hubiera explorado mucho más, quizá aparecería un **túnel** por el sur que conecta directamente, que no se sabía que existía.

### 3.6 Exploración excesiva o ineficiente

- Un alumno pregunta por dejar $\varepsilon=0{,}9$ "computando 27 días" *(dudoso: posiblemente alude al `uptime` de 27 días de la captura de APP p.62, ver §10)*. Respuesta: el agente se pierde explorando, y además **la exploración ineficiente hace que la red se sobreajuste a experiencias malas** (T06 ~03:06-03:07).
- Ejemplo en vídeo: un perro simulado que, a base de explorar, aprende a andar boca abajo con las patas hacia arriba. Si se le meten caminos no eficientes, la red los aprende, baja su ritmo de aprendizaje, "te dice que esa es la solución" y se queda atascada ahí. Lo deseable es que vaya olvidando esos caminos (T06 ~03:07-03:09).
- Un alumno lo compara con la búsqueda en profundidad del ajedrez, en la que explorar más ramas acaba encontrando la jugada. El profesor responde que en un ejemplo pequeño será fácil llegar, pero que permitir más exploración por tener mucho cómputo mete mucho ruido (T06 ~03:09-03:10) *(dudoso: la transcripción es muy confusa)*. Es la transición hacia la priorización de experiencias (§6.7).
- **La exploración puramente aleatoria tampoco funciona**: sin ninguna indicación de si va bien o mal, el agente da vueltas sobre lo mismo y no consigue avanzar en una dirección (T06 ~03:12-03:13).
- Un alumno pregunta si en tabular lo aprendido llegando "por suerte" no vale nada. Respuesta: con **redes** hay que tener mucho cuidado con meter ruido, porque los pesos convergen sobre las últimas experiencias. En **tabular** no pasa nada, porque visitar todos los estados infinitas veces es premisa de la convergencia: la primera llegada da valor positivo a las celdas próximas al objetivo, luego se propaga al siguiente nivel, y así hacia atrás hasta "limpiar" el entrenamiento; en una matriz de 50×50 tarda más (T06 ~03:13-03:16).

### 3.7 Cómo se traslada esto al ejercicio 3

El ejercicio 3 (1,5 puntos) pide que $\varepsilon$ no dependa del número de episodios y baje conforme mejora el agente, "mejorar resultados y/o conocer el espacio". El enunciado añade que la duración del entrenamiento "debería estar condicionada a la evolución del aprendizaje", aunque para este ejercicio permite mantener un número fijo de episodios; eso enlaza con la parada por aprendizaje de la idea C (§9.4). Lo que se evalúa es la respuesta a dos preguntas: **¿cómo determinamos que ha mejorado?** y **¿cómo trasladamos esa mejora a $\varepsilon$?** (ENUNCIADO; T07 ~04:29). La idea C del ejercicio 4 añade condiciones de parada por aprendizaje y guardado y recuperación de pesos (ENUNCIADO). Los hiperparámetros de este documento se evalúan antes: las instrucciones generales piden ajustarlos "cuando lo creáis conveniente" (gamma, tamaño del buffer, número de episodios); el ejercicio 1 no exige motivar la parametría, y la segunda parte del ejercicio 2 exige "ahora sí" ajustarla y motivarla (ENUNCIADO). Detalle en [notebook_partida.md](notebook_partida.md).

**Nota (propuesta, a decidir en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md)):** traducciones medibles de lo visto en clase.

| Dimensión | Medida posible | Cómo mueve $\varepsilon$ |
|---|---|---|
| Resultados | Media (y dispersión) de la métrica objetivo en $K$ episodios *greedy* cada $N$ episodios, sobre días fijos de validación | Bajar un escalón cuando mejora sobre la mejor evaluación previa; mantener si no; subir si empeora claramente (§3.4) |
| Distancia al *benchmark* | Diferencia frente al TWAP en la misma evaluación (en el ejercicio 2, error de seguimiento del plan TWAP) | $\varepsilon$ como función decreciente de la fracción del objetivo alcanzada ("a mitad del objetivo, bajar un poco") |
| Conocimiento del espacio | Pseudo-conteos sobre un estado discretizado (tiempo restante, volumen restante, régimen del día) | Bajar cuando la cobertura de las celdas relevantes supera un umbral; $\varepsilon$ por estado (§4.4) |

Las evaluaciones deben hacerse sin exploración ni aprendizaje y con varios episodios (§3.2). El LAB evalúa con un solo episodio de test (LAB celda 20).

---

## 4. Explorar activamente: más allá de $\varepsilon$-greedy

### 4.1 El problema: $\varepsilon$-greedy no explora activamente

**Idea.** $\varepsilon$-greedy es la técnica más famosa porque es la primera que se enseña, pero **no explora activamente el entorno**, y en problemas complejos no basta. Hay que investigar políticas de exploración adecuadas (APP p.72; T06 ~03:26). Los principales escollos del RL son el entorno con asunciones realistas, las recompensas y sobre todo la exploración, que es "un dolor de cabeza" (T06 ~03:26-03:27).

**River Swim (APP p.72; T06 ~03:27-03:30).** El diagrama muestra 6 estados en fila; el objetivo es llegar al 6, río arriba, empezando en el 1.

- Las flechas discontinuas son ir río abajo, a favor de la corriente: siempre se llega al estado anterior (probabilidad 1).
- Las continuas son ir río arriba: desde un estado intermedio, con 0,35 se avanza, con 0,6 se queda en el sitio y con 0,05 retrocede. En el estado 1, 0,6 de avanzar y 0,4 de quedarse; en el 6, 0,6 de quedarse y 0,4 de retroceder. El estado 2 aparece resaltado como "ahora estás aquí".
- Con una política 100 % aleatoria hacen falta muchísimos pasos para llegar una sola vez al 6, y con 8, 10 o 13 estados muchos más: llega un punto en que, ni con un millón de pasos, la fuerza bruta funciona. Si además la recompensa solo se da al llegar al final, "complicado".
- **Exploración activa**: si en el estado 3 se ha probado pocas veces ir río arriba, darle más peso; o preferir la acción que suele llevar a estados menos visitados (el 4) frente a la que lleva a uno muy visto (el 2). Así sí se recorre el río, pero exige llevar un registro.

El notebook `insights` programa exactamente esto (RLM insights, celdas 3-9): `SwimRiver` cuenta las visitas a cada estado con una política aleatoria (ir a la derecha en el 1 y a la izquierda en el último) y `ModSwimRiver` elige izquierda o derecha con pesos $1-n_{\text{previo}}/n_{\text{total}}$ y $1-n_{\text{siguiente}}/n_{\text{total}}$, con $n_{\text{total}}=n_{\text{previo}}+n_{\text{siguiente}}$ (las visitas de los dos vecinos, no el total del río) y $n_{\text{previo}}$ acotado inferiormente a 1 (`max(count, 1)`) (RLM insights, celda 7); es decir, favorece el estado vecino menos visitado. Se ejecuta con 6 y con 20 estados.

**Nota (cálculo propio):** reimplementando las reglas de transición del notebook (semilla 0; 30 repeticiones, 10 para 13 estados), la mediana de pasos hasta llegar por primera vez al último estado es:

| Estados | Política aleatoria | Conteo de visitas (`ModSwimRiver`) |
|---|---|---|
| 6 | ≈ 460 | ≈ 15 |
| 8 | ≈ 4.000 | ≈ 23 |
| 10 | ≈ 19.000 | ≈ 38 |
| 13 | ≈ 950.000 | ≈ 70 |

El crecimiento es aproximadamente exponencial con la política aleatoria y casi lineal con el conteo: coincide con lo que dice el profesor sobre el millón de pasos.

### 4.2 Catálogo de soluciones

APP p.72 enumera "algunas soluciones" (con "¡La exploración no es fácil! + Research") y el profesor las explica (T06 ~03:30-03:35):

| Familia | Qué hace | Comentario de clase |
|---|---|---|
| Inicialización de los valores | Valores iniciales de Q distintos de 0 | No es lo mismo inicializar con ceros que con $+1$ o $-1$: tiene impacto (T06 ~03:30) |
| Intrínseco: métodos *policy* o *noisy networks* | Redes que meten ruido en pesos o salidas para explorar | (T06 ~03:30-03:31) |
| Experiencia: contar o pseudo-contar | Elegir acciones menos frecuentes o subir $\varepsilon$ donde se ha estado poco | Lo del río; si el espacio es continuo, discretizar para contar (T06 ~03:31) |
| Predicción | Explorar donde el error es grande, o donde la confianza es baja: $\arg\max_a\,[V(a)+U(a)]$ | Las acciones nuevas suelen dar más error porque se han probado poco (T06 ~03:31) |
| Recompensas | *Reward shaping*, *curriculum learning*, *curiosity-driven*, *auxiliary tasks* | Feedback intermedio basado en el conocimiento del entorno (T06 ~03:31) |

- **Curriculum learning**: de lo fácil a lo difícil. Primero se aprende a desembalar la hamburguesa y luego a ponerla al fuego, "como de parvulitos a primaria" (T06 ~03:31-03:32).
- **Varios agentes especializados**, cada uno en una cosa, en lugar de uno para todo, y **subobjetivos** ligados al objetivo principal o recompensas basadas en el error (T06 ~03:32).
- Son técnicas complementarias, y precisamente porque la exploración es uno de los grandes escollos hay "muchísimas"; si no hubiera soluciones, ciertos problemas no se habrían resuelto (T06 ~03:32). Hay "clases enteras" solo de exploración (T06 ~03:35; T07 ~00:40).
- La consigna general: **romper y simplificar el problema** (T06 ~03:35).

**Nota:** el término $U(a)$ de la diapositiva corresponde a la familia de métodos de cota superior de confianza (UCB), en la que $U$ crece con la incertidumbre, por ejemplo $c\sqrt{\ln t/N(a)}$. La diapositiva no da la forma de $U$.

Recompensas intrínsecas y su relación con PER: §6.7. *Reward shaping*: [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md). *Curriculum* de datos (mismo día → varios días → activos → estrés): [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

### 4.3 Pseudo-conteo en bolsa y mapa de calor

- En bolsa, pseudo-contar es registrar, hasta cierto punto, qué situaciones relevantes se han vivido: "ya he ejecutado Santander en días de bajada". Lo importante es definir qué es relevante, por ejemplo encontrarse con distintos activos en días de subida o de bajada, y llevar un registro de eso (T07 ~00:37).
- Un alumno sugiere guardar los estados visitados durante el entrenamiento para visualizarlos como un mapa de calor. El profesor lo acepta como una de las muchas técnicas posibles (T07 ~00:40).
- En el resumen de componentes del caso de ejecución, la diapositiva da como misión del agente tomar acciones, explorar y aprender (CASOS p.46): la exploración forma parte explícita del diseño del agente de ejecución. Componentes completos del caso en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

### 4.4 Exploración condicionada al estado y dentro del episodio

**Idea.** La exploración no tiene que tener el mismo nivel en todos los episodios ni en todo el espacio, **ni siquiera dentro de un mismo episodio** (T07 ~00:37).

- **APP p.73 (imagen):** el mismo mapa de Madrid con $\varepsilon$ distintos por zonas: 0,75 en el oeste, lejos de la ruta; 0,5 en la parte baja izquierda; 0,15 cerca del coche; 0,1 en el centro; 0,20 al este del Retiro y 0,05 junto a la bandera.
- Ejemplo (T06 ~03:33-03:34; T07 ~00:37-00:39): por la Castellana he pasado muchas veces, mi registro lo dice, así que exploro menos; por Sol he pasado poco, así que mantengo la exploración más alta. Se puede **subir dentro de la misma partida**: empiezo explotando, en un punto explorar me desvía hacia una zona nueva, y como allí mi conocimiento es poco fiable aumento la exploración y doy más peso a las acciones aleatorias.
- Implementación: en cada paso el agente llama **a la red o a la función de exploración**. Esa función es un "subsistema en sí mismo", y puede ser puramente aleatoria, basarse en reglas como las del río o en conteos (T06 ~03:34).
- También el tamaño del buffer o $\gamma$ pueden ser dinámicos y distintos según el episodio (T07 ~00:41).

### 4.5 Cómo se elige la acción aleatoria

**Idea.** La acción "aleatoria" no tiene por qué ser uniforme $1/N$.

- Tres opciones (T07 ~00:39): **uniforme**; **ponderada por la utilidad esperada**, con una probabilidad según el valor actual de cada acción en el diccionario; o **dirigida** hacia acciones que llevan a zonas desconocidas (si ir a la izquierda lleva a zonas menos conocidas, explorar hacia la izquierda).
- Con pesos fijos: en inversión, la exploración puede dar más peso a mantener que a comprar o vender, $w_{\text{mantener}}>w_{\text{comprar/vender}}$ (APP p.57).
- En el caso de ejecución, al inicializar el buffer con acciones aleatorias, "los pesos no tienen por qué ser iguales" (CASOS p.45).
- **En la práctica la decisión es explícita.** La acción aleatoria del agente sale de `env.action_sample()` (BE dqn.py, act), y esa función (13) es un TODO con un parámetro `p`, la pista `np.random.choice([0, 1], p=[1-p, p])` y la pregunta opcional de qué distribución de probabilidad es mejor para la exploración (LAB celda 7). Es decir, el propio notebook pide elegir esa $p$.

**Nota:** el agente tiene dos acciones, `0 = no hacer nada` y `1 = orden agresiva` (LAB celda 7), y la orden CARE equivale a ejecutar en un 7,5-12,5 % de los pasos (`vol_care` en LAB celda 7). Con $p=0{,}5$ la agresiva sale en la mitad de los pasos, un ritmo de 4 a 7 veces el de un TWAP, de modo que las experiencias aleatorias se parecen poco a la política que se busca. Además, el episodio termina en cuanto se completa la orden (`done` = tiempo agotado **o** orden completada, LAB celda 7, funciones 10 y 11), así que con $p=0{,}5$ cada episodio acaba en unos $2\cdot$`vol_care` pasos, el 15-25 % del horizonte (§6.6). Elegir $p\approx$ `vol_care/episode_bins` es una aplicación directa de APP p.57 y CASOS p.45. Detalle de las condiciones de parada en [08_entorno_best_execution.md](08_entorno_best_execution.md).

### 4.6 Inicialización y sesgo inicial

- Ejemplo (T06 ~01:17-01:18): ir a la derecha da 0; ir a la izquierda da una normal de media $-0{,}1$ y desviación 1. Si la primera muestra de la izquierda es mala ($-0{,}7$), el agente queda sesgado y le cuesta salir de esa creencia; si es buena ($0{,}4$), se la cree demasiado. Incluso en un problema tan sencillo, empezar mal atasca; el profesor lo llama un problema de inicialización. Según la explicación de clase (paráfrasis), Double Q-learning usa dos diccionarios, utiliza las creencias de uno en el otro y las pondera, con lo que reduce ese sesgo (T06 ~01:18). Detalle en [00_repaso_rl.md](00_repaso_rl.md) y [T2 04_doble_dqn_y_duelo.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/04_doble_dqn_y_duelo.md).

  **Nota (precisión técnica):** el ejemplo es el del sesgo de maximización de Sutton y Barto (ej. 6.7): el $\max$ sobre estimaciones ruidosas sobreestima el valor de la izquierda aunque su media real sea $-0{,}1$, y no es solo cuestión de la primera muestra. Double Q-learning no promedia ni pondera las dos tablas para elegir el objetivo: desacopla selección y evaluación. Con $Q_A$ se elige $a^*=\arg\max_a Q_A(s',a)$ y con $Q_B$ se evalúa, $r+\gamma Q_B(s',a^*)$, y en cada paso se actualiza al azar una de las dos con los papeles intercambiados. Lo que sí suele combinarse es la acción de comportamiento, por ejemplo $\varepsilon$-greedy sobre $Q_A+Q_B$.
- En los notebooks tabulares la tabla se inicializa con ruido muy pequeño, `np.random.randn(na) * 0.001`, en lugar de ceros (RLM cliff-walking, celda 4; RLM TD-q-learning, celda 2). **Nota:** así se rompen los empates del `argmax`, que con ceros elegiría siempre la acción 0.

---

## 5. *Step size* $\alpha$

### 5.1 Qué hace

**Idea.** En la actualización tabular, $\alpha$ es el peso que se da a lo nuevo frente a la creencia anterior (T06 ~00:52):

$$
Q(s,a)\leftarrow(1-\alpha)\,Q(s,a)+\alpha\,\big[r+\gamma\max_{a'}Q(s',a')\big]
$$

Ejemplo de clase: creía 100, recibo 20, y con $\alpha=5\,\%$ actualizo a $100\cdot0{,}95+20\cdot0{,}05=96$ (T06 ~00:52). El ejemplo numérico completo de Q-learning está en [00_repaso_rl.md](00_repaso_rl.md). En redes, $\alpha$ es el *learning rate* del optimizador.

### 5.2 Grande o pequeño: el ejemplo del Retiro

**Idea.** Tamaño de $\alpha$ (APP p.74):

- **Muy grande**: puede dar mucha importancia a un hecho poco probable.
- **Menor**: recoge una mayor casuística.
- **Muy pequeño**: puede quedar atascado en mínimos locales.

APP p.74 (imagen): una carretera con curvas y, junto a ella, una señal triangular de peligro con coches parados (atasco o accidente), en línea con el ejemplo que sigue.

Ejemplo (T06 ~03:36-03:38): hay dos rutas, por delante y por detrás del Retiro, y el agente cree que la de delante es mejor. Un día explora la de detrás y tarda 30 minutos más porque hubo un accidente.

- Con un $\alpha$ grande da tanto peso a ese día que su valor queda muy negativo y ya casi nunca vuelve, salvo que le toque explorar. Pero los accidentes ocurren un 2 % de las veces y, sin ellos, la ruta ahorra unos minutos. Para saberlo hay que haber pasado varias veces.
- Con un $\alpha$ muy pequeño ocurre lo contrario: si la red ya ha convergido y un día explorando encuentra el túnel que ahorra 2 horas, ese hallazgo pasa desapercibido.
- El profesor lo usa a diario fuera de este contexto, para actualizar creencias en campañas y experimentos con clientes: no se da el 100 % de peso a lo último que ha ocurrido (T06 ~03:35-03:36, ~03:38-03:39).

### 5.3 No estacionariedad: el peso medio de la clase

**Idea.** Como los problemas no son estacionarios, $\alpha$ permite **olvidar el pasado y quedarse con lo nuevo** (T06 ~03:39). Si el problema fuera estático bastaría una media simple: muestrear un millón de veces el peso de los alumnos y promediar. Pero incluso el peso medio de la clase cambia: alguien sale, alguien come, entra gente (T06 ~03:39-03:40).

- Con una creencia de 80 kg y $\alpha=1\,\%$, observar a un alumno de 72 kg da, según la transcripción, 79,8 kg. Si entran culturistas, $\alpha$ permite que la media se vaya adaptando, a unos 85 kg (T06 ~03:40-03:41).
- Lo mismo vale para las políticas: las del principio están desactualizadas y tomaban acciones sin sentido, y se quieren olvidar en lugar de promediarlas (T06 ~03:42).

**Nota (corrección del cálculo):** con $\alpha=0{,}01$, $0{,}99\cdot80+0{,}01\cdot72=79{,}92$ kg, no 79,8. El 79,8 corresponde a $\alpha=2{,}5\,\%$ ($80-0{,}025\cdot8$). El mensaje no cambia. **Nota:** la actualización con $\alpha$ constante es una media móvil exponencial, en la que la observación de hace $k$ pasos pesa $\alpha(1-\alpha)^k$; la media simple equivale a $\alpha_n=1/n$.

### 5.4 *Alpha decay*, optimizadores y *novelty*

- Estrategia clásica: empezar alto e ir bajando conforme mejora el aprendizaje (*alpha decay*), "aunque nos despreocupamos un poco usando optimizadores (Adam, RMSProp…)" (APP p.74). En T06 ~03:35 dice que ahora pueden despreocuparse algo de $\alpha$ por las redes, pero que es importante.
- **Novelty**: también se aplica a $\alpha$, que debería ser más grande en estados nuevos. La diapositiva dice que lo correcto sería "proporcional al N veces visitado" (APP p.74). **Nota:** debe leerse como inversamente proporcional, $\alpha(s,a)=1/N(s,a)$, que es la media muestral exacta y da más peso a lo nuevo cuanto menos se ha visitado. "Proporcional a $N$" contradiría la frase anterior de la diapositiva.
- **En la práctica**: el LAB dice que, como el problema es no estacionario, $\alpha$ será un valor estático o adaptativo, pero nunca uno que decaiga a cero (LAB celda 11). En el agente, `alpha` es el `learning_rate` de Adam, 0,01 por defecto (BE dqn.py), y en el notebook `deep` se usa $10^{-4}$ (RLM deep, celda 3). Al explicar la práctica, tras recorrer los parámetros del entorno (`look_back`, `vol_care`, diccionario de acciones, etc.) y recomendar empezar por el notebook *step by step* ("como 10 pasitos"), dice que los parámetros de la red y $\alpha$ "son menos importantes"; lo que le interesa es sobre todo lo que está en el notebook *(dudoso: la transcripción dice "en este no Food"; probablemente las funciones y parámetros del entorno)* (T07 ~05:00-05:01).
- Más sobre la tasa de aprendizaje en redes (Adam frente a SGD, qué hacer si no converge): [T2 03_dqn.md §10](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md).

---

## 6. Memoria de experiencias (*replay buffer*)

### 6.1 Por qué hace falta (resumen)

**Idea.** Las redes necesitan entrenarse con observaciones i.i.d., y un minibatch con las últimas 64 experiencias está muy correlacionado: si Santander sube todo el 6 de noviembre, el agente aprendería que Santander siempre sube (T06 ~01:26-01:29). En supervisado se muestrean 64 al azar de un millón; en RL el dataset se construye jugando, así que se guarda un registro de experiencias, **FIFO** y de tamaño acotado, y se muestrea de él (APP p.38, p.77; T06 ~01:31-01:34). Tipos: uniforme, PER y CER (APP p.38). Sin buffer "se meten muchísimos sesgos o no se aprende nunca" (T06 ~03:54). Desarrollo en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md) y [T2 03_dqn.md §8](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md).

### 6.2 Tamaño: independencia frente a obsolescencia

**Idea.** El tamaño $m$ es un *trade-off* (APP p.78; T06 ~03:56-03:57; T07 ~03:45):

- **Buffer más grande → más independencia**: experiencias de días, momentos y ejecuciones muy separados entre sí, sin sesgos para la red.
- **Buffer más grande → más obsolescencia**: experiencias generadas con pesos y políticas antiguas que ya no se parecen a la actual, o que ya se seleccionaron muchas veces. Las más antiguas "salen por detrás".

APP p.78 la presenta como "probabilidad de aparición en k steps"; en rigor, $k$ son extracciones individuales (ver la Nota siguiente y §13, punto 6):

$$
P(\text{aparece al menos una vez en } k)=1-\Big(1-\frac{1}{m}\Big)^{k}\approx1-e^{-k/m}
$$

La imagen acompaña con tres pilas de "experiencias" de altura creciente.

**Nota (cálculo propio):** $k$ cuenta extracciones individuales, así que con batch $B$ y un aprendizaje cada $L$ pasos, en $T$ pasos hay $k=BT/L$. Con $m=10.000$ y 10 aprendizajes de 64, $P\approx6\,\%$; con 100 aprendizajes, $\approx47\,\%$. En la práctica (`learn_after = batch_size`, LAB celda 12), una experiencia vive $m$ pasos en el buffer y recibe en media $\frac{m}{64}\cdot\frac{63}{m}\approx1$ extracción uniforme (más la de CER si es la última). Es decir, alrededor del 37 % de las experiencias no se usan nunca, sea cual sea $m$. Aprender más a menudo, con un `learn_after` menor, sube esa cifra. Ver [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

### 6.3 La regla: pensar en episodios, no en experiencias

**Idea.** El consenso de los papers de Atari es **un millón de experiencias** (APP p.78), pero ese número viene de una regla (T06 ~03:58-04:02):

- Partidas de unos 2,5 minutos a unos 60 fotogramas (decisiones) por segundo: $60\cdot60\cdot2{,}5=9.000$ *steps* por partida.
- Las últimas 100 partidas: $9.000\cdot100=900.000\approx$ un millón. El profesor admite que no recuerda bien si eran 2 o 2,5 minutos ni si eran 60 o 30 fotogramas *(cifras dudosas)*.

Es más fácil pensar en **cuántas partidas o episodios recientes** se quieren en el buffer que en experiencias, por ejemplo las últimas 100. Así, de 64 muestras aleatorias, con alta probabilidad cada una vendrá de una partida distinta, reciente e independiente. "No me gusta poner un millón; me gusta poner: tengo mis 100 últimas partidas" (T06 ~03:59-04:03). En bolsa se hace igual con ejecuciones u operaciones (T06 ~04:01).

Cómo se calcula (T06 ~04:03-04:09; T07 ~03:46-03:50):

1. **Definir qué es un episodio**, que depende del objetivo: en ejecución intradía, el día de 9 a 17 o unas 2 horas; en ajedrez, hasta jaque mate, tablas o la regla de los movimientos sin capturas (en clase se dijo 40 *(dudoso)*; **Nota:** el reglamento FIDE fija 50 movimientos sin captura ni movimiento de peón, y con 75 son tablas automáticas, §13); en opciones a 6 meses revisadas cada 3, un trimestre salvo eventos. El episodio también puede cortarse por condiciones: "si consigo un 2-3 %, me voy a casa". Desarrollo en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).
2. **Estimar los *steps* por episodio.** Ejemplo intradía: decisiones cada minuto, usando en torno al 75 % del día porque a veces actúan condiciones de parada; con unas 8 horas, $8\cdot60$ *steps* *(cálculo dudoso en la transcripción, que mezcla 9 y 8 horas con el 75 %)*. En ajedrez cuenta unos 200 pasos por partida *(cifras dudosas: dice 500, 50 y 200)*.
3. **Decidir cuántos episodios** recientes quieres, 100 o 1.000, y multiplicar.
4. **Cuántos episodios depende de la diversidad necesaria** (T07 ~03:46-03:48): si importan los regímenes alcista, plano y bajista, con unas 100 sesiones, unas 30 de cada, ya hay una buena distribución; si quedan concentradas, hasta 500, para tener días muy alcistas, algo alcistas, planos y bajistas. Regla general: unas 100, "pero depende". En *cliff walking* bastan las últimas 10 partidas, porque hay pocas casuísticas; si hay variables importantes que aparecen muy de vez en cuando, hacen falta más.
5. En el código se fija el número de experiencias: sesiones × *steps* por sesión, por ejemplo $100\times10=1.000$ si cada sesión tiene unos 10 momentos de decisión. También se puede configurar por partidas, registrando la duración de cada una y sacando la más antigua entera, pero cuesta algo más, y con la aproximación basta (T07 ~03:48-03:50).

El tamaño también depende de **cuántos datos haya para generar experiencia**: con una serie de 20 años se puede permitir un buffer mayor; con solo 5 no conviene uno demasiado grande, porque abarcaría demasiada historia. Aclara que no habla del batch de entrenamiento (T06 ~03:55) *(dudoso: la transcripción dice "parches demasiado grandes", casi seguro por "buffers")*. Si se pregunta por 14 años de datos (unos 3.000 días), primero hay que definir cuánto dura el episodio: no hay respuesta única (T06 ~04:03-04:04). Al final del bloque pide motivar cada decisión: en entrevistas de trabajo y en la práctica le interesa por qué se eligió una opción y no otra (T06 ~04:10).

**Nota (aplicación a la práctica):** el horizonte de los episodios del LAB, `episode_bins`, es uniforme entre 400 y 600 *steps* (LAB celda 7, `_generate_episode_params`), unos 500 de media. Con la regla de clase, 100 episodios que agoten el horizonte son unos 50.000 *steps*, que es justo el `buffer_size` por defecto del agente (BE dqn.py). 30 episodios por régimen, con tres regímenes, dan unos 45.000; 500 episodios, unos 250.000. **Ojo:** 400-600 es un máximo. El episodio acaba antes si se completa la orden (LAB celda 7, `_compute_done`), así que la cuenta $100\times500$ solo vale para una política tipo TWAP que reparte la orden hasta el final. En el llenado aleatorio con $p=0{,}5$ los episodios duran unos 60-150 pasos ($2\cdot$`vol_care`, con `vol_care` entre 30 y 75), y 50.000 *steps* son varios cientos de episodios cortos (§6.6). En las celdas de ejemplo, el notebook `deep` usa `buffer_size = 10000` con LunarLander (RLM deep, celda 3).

### 6.4 Ajustar el tamaño dinámicamente

**Idea.** *Dynamical System Model of ER* (APP p.78; T06 ~04:11-04:13): se compara la estimación de la red con el valor que llega; la diferencia es el error. Si al muestrear del buffer el **error es alto**, las experiencias son nuevas para la red y conviene **ampliar el buffer** (de 100 a 200 partidas) para no descartarlas pronto y poder volver a aprender de ellas. Si los **errores ya son muy pequeños**, se reduce. "Las ideas felices son bienvenidas" (T06 ~04:12). En la práctica lo propone como posible idea para el ejercicio 4: ajustar el buffer de forma dinámica y exponer el razonamiento (T07 ~04:30-04:31).

### 6.5 Implementación: `deque`, no listas

APP p.79 da dos "consejos básicos". El primero es usar `deque(maxlen=buffer_size)` con `buffer.append((s, a, r, s1))`, **nunca** una lista con `pop(0)`. El segundo es llenar el buffer antes de entrenar (§6.6).

- Es una colección específica, más eficiente: en RL hay que evaluar muchísimas veces y se busca la eficiencia. El profesor ha dedicado mucho tiempo a hacer el código más eficiente, por ejemplo cambiando pandas por diccionarios (T06 ~04:13-04:14).
- El notebook `insights` lo mide con `%%timeit` sobre un millón de elementos: `list.pop(0)` + `append` frente a `deque.append` (RLM insights, celdas 19-23; no tiene salidas guardadas). **Nota:** `list.pop(0)` es $O(n)$ porque desplaza todos los elementos; `deque` añade y descarta por los extremos en $O(1)$.
- El agente de la práctica ya usa `deque(maxlen=buffer_size)` y guarda `(s, a, r, s1)` con `s1 = None` si el episodio terminó (BE dqn.py, experience). Más sobre eficiencia: [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

### 6.6 Llenado inicial: con qué y cuánto

**Idea.** El buffer empieza vacío. Si se entrena con las primeras 64 observaciones, estarán muy correlacionadas, porque son del mismo día o de la misma partida. Como mínimo, hay que **dejar pasar un tiempo antes de la primera actualización** para rellenarlo (T06 ~04:14-04:15).

Ejemplo (T06 ~04:15-04:18): un algoritmo de medias móviles, en un día alcista con la media corta por encima de la larga, haría que las 64 muestras reforzaran 64 veces que "comprar es bueno". Con esa misma regla habrá días en que el activo cae, sube o queda plano, y lo que se quiere es que, para ese vector de características, el batch mezcle días que respondieron bien, regular y mal, y la red saque un promedio de observaciones independientes (por ejemplo, de 100 momentos distintos, alcistas, bajistas y neutros).

**Con qué llenarlo** (APP p.79; CASOS p.45; T06 ~04:20; T07 ~03:50):

| Origen | Ventaja | Riesgo |
|---|---|---|
| Acciones aleatorias, "como mínimo"; el agente actúa al azar y aún no aprende | Diversidad sin sesgo de comportamiento | Mucho ruido |
| Experiencias reales registradas (operativa del bróker, partidas reales de póker) | Información con lógica | Sesgo hacia ese comportamiento |
| Un algoritmo sin RL ya en producción (algoritmos de ejecución por reglas, TWAP) | Experiencias útiles; permite partir de algo que funciona y mejorarlo con RL | La red puede quedarse atrapada en el comportamiento clásico |
| Mezcla de lo anterior | Equilibra ruido y sesgo | Hay que calibrar proporciones |

APP p.79 (imagen): una sartén volteando una tortilla o crepe, con un enlace a un vídeo de YouTube que no se comentó en clase *(su relación con el llenado del buffer no se explica)*.

**Cuánto** (APP p.79; T06 ~04:20-04:21; T07 ~03:50-03:51): **no hace falta llenarlo entero**, por ejemplo 50 partidas en vez de 100. Aquí el *trade-off* ya no es independencia frente a obsolescencia sino **independencia frente a ruido**: las aleatorias tienen mucho ruido, así que mejor no llenarlo entero; si responden a una lógica y aportan información, se pueden meter más.

**El riesgo del sesgo** (T06 ~04:21-04:22): llenarlo con experiencias de un comportamiento X, como un jugador de póker muy agresivo, hace muy probable que el algoritmo se sobreajuste a X si esas experiencias no se descartan pronto. Cualquier lógica que se meta condicionará mucho al algoritmo, y por eso conviene mezclar aleatorias con no aleatorias y no llenarlo del todo. En el caso de ejecución, si se llena con experiencias TWAP, la red puede no salir del comportamiento clásico (T07 ~03:51).

**Debate en clase** (T06 ~04:22-04:30). Un alumno no ve claro cómo mezclar experiencias de contextos distintos (partidas, mercados) no estropea la política. Respuestas:

- Póker: llenar solo con partidas del jugador más agresivo hace que el algoritmo seleccione casi siempre "lo suyo" hasta que entren otras. Si se añaden las de uno conservador y luego las de todos, hay más variación, más independencia y menos sesgo. Si solo se dispone de un algoritmo agresivo, lo mejor es añadirle algo de aleatoriedad.
- S&P 500 desde 1900 como único episodio: no hay diversidad (comprar el día 1 y mantener) y siempre se empieza en el mismo punto. Conviene no empezar siempre desde el día 0, para ganar validez externa (ver `reset` en [02_diseno_de_entornos.md](02_diseno_de_entornos.md)).
- A menudo no hay históricos de algo que funcione. Si se parte de pura aleatoriedad, no se espera a tener un millón de experiencias o 100 partidas, porque se tardaría muchísimo en eliminarlas: se inicializa con **20 o 30 partidas** y el agente aprende mientras el buffer se llena con partidas cada vez mejores. Si lo que se mete aporta valor, conviene más volumen.
- "Esto es prueba y error": con tantas dimensiones, o se tienen los principios claros o uno se atasca en el elemento más tonto (T06 ~04:30).

**En la práctica:** el LAB llena el buffer antes de entrenar con una política estocástica, "con el objetivo de explorar el entorno inicialmente" (LAB celda 15). El bucle recorre `range(buffer_size)` (LAB celda 16), de modo que **lo llena entero** con experiencias aleatorias. El comentario dice que $\varepsilon$ vale 1, pero eso solo es cierto si se elige `epsilon = 1` en la celda 12. Es el mismo patrón del notebook `deep` de clase (RLM deep, celda 3): $\varepsilon=1$, llena las 10.000 posiciones del buffer con acciones aleatorias antes de entrenar, aprende cada `batch_size` pasos y en cada log hace un `learn()` adicional para imprimir el `mse`; es decir, el LAB hereda el esquema del material base y no es una peculiaridad suya.

**Nota:** llenarlo entero va contra la recomendación de clase de no hacerlo cuando las experiencias son aleatorias y ruidosas. Además, si se elige $p=0{,}5$ en `action_sample` (§4.5), cada episodio completa la orden en unos $2\cdot$`vol_care` pasos (15-25 % de `episode_bins`) y termina, porque `done` es tiempo agotado **o** orden completada (LAB celda 7). El buffer se llena de episodios cortos, casi sin estados de poco tiempo restante, que son los decisivos para replicar un TWAP. Ver [08_entorno_best_execution.md](08_entorno_best_execution.md). Alternativas, como un llenado parcial o mezclado con TWAP, en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

### 6.7 Priorizar experiencias: PER y recompensas intrínsecas

**Idea.** Con mucha capacidad de cómputo uno puede permitirse explorar más y no descartar experiencias, pero entra mucho ruido y ya no basta un muestreo uniforme (T06 ~03:09-03:10). Si el buffer se llenó explorando mucho al azar, un muestreo uniforme puede no encontrar lo importante. Ejemplo (T06 ~03:10-03:12): de millones de experiencias solo 30 son relevantes. Con batches de 64 uniformes, la buena puede no salir nunca, y cuando por fin sale, la red ya ha convergido a un mínimo local rodeada de experiencias que no valían. De ahí la **priorización**: más probabilidad para las experiencias que se creen más valiosas, lo que exige definir qué determina ese valor (T06 ~03:10).

Sigue un intercambio (T06 ~03:11-03:13) *(dudoso: la transcripción no separa quién habla)*: se objeta que con tiempo infinito hasta un $\varepsilon$ de 0,01, o uno de 0,99, acabaría recorriéndolo todo; la respuesta admite el límite, pero insiste en que la exploración puramente aleatoria, sin indicación de si va bien o mal, da vueltas sobre lo mismo y no avanza en una dirección (§3.6).

**Prioritized Experience Replay** (APP p.80; arXiv 1511.05952; T06 ~04:31-04:33): la probabilidad de muestreo depende del error de la red al estimar el valor de la acción.

$$
P(i)=\frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}},\qquad p_i=|\delta_i|+\epsilon\ \ \text{(proporcional)}\quad\text{o}\quad p_i=\frac{1}{\operatorname{rank}(i)}\ \ \text{(rank-based)}
$$

donde $\delta_i$ es el error TD de la experiencia $i$.

- **Estabilidad**: *Huber loss* e *importance sampling*. **Rapidez**: un *sumtree*, que pasa el muestreo de $O(n)$ a $O(\log n)$ (APP p.80). Requiere más cómputo, y hay técnicas para acelerarlo (T06 ~04:35).
- El exponente $\alpha$ regula cuánto se pondera por error. La transcripción mezcla los extremos *(dudoso)*. En el paper, $\alpha=0$ es muestreo uniforme y $\alpha=1$ pondera totalmente por el error (§13).

**Recompensas intrínsecas (Montezuma's Revenge)** (APP p.80; arXiv 1810.12894; T06 ~04:33-04:35). En los rankings de Atari, Montezuma siempre estaba a la cola: tiene unas 20 salas, cada una con un mini objetivo (subir escaleras, coger una llave y volver a usarla), y requiere mucha más exploración que Breakout. Lo resolvieron con **recompensas intrínsecas basadas en el error de predicción**, que se dispara en los momentos nuevos o "épicos" (una moneda, una caída, una sala nueva).

- La imagen de APP p.80 (de *Exploration by Random Network Distillation*) muestra la recompensa intrínseca frente a los *steps* (0-2.500) con picos numerados del 1 al 21, cada uno asociado a una captura del juego: nuevas salas u objetos.
- Funcionaba para ese juego y no para el resto de Atari: soluciones "tontas" pueden funcionar muy bien en un caso particular y no en otros, pero se rescatan los principios. En eso se basa priorizar al aprender lo que supuso "un antes y un después" o disparó el error (T06 ~04:35).

### 6.8 *Combined Experience Replay* (CER)

**Idea.** De un batch de 64, **63 se muestrean de forma uniforme y la última es siempre la experiencia más reciente**, para darle siempre un toque de actualidad (T06 ~01:36, ~04:36). "Añadimos nuestra última experiencia a nuestro batch → ayuda a corregir el tamaño del buffer" (APP p.81).

- **Gráficos de APP p.81** (*A Deeper Look at Experience Replay*, arXiv 1712.01275). Retorno frente a episodios (0-100) con buffers de 100, 1.000, 10.000, 100.000 y 1.000.000. En **Buffer-Q** (replay uniforme), el buffer de 100 se acerca al retorno máximo hacia los episodios 40-50 y el de 1.000 algo después; los de $10^4$-$10^6$ van mucho más lentos y en el episodio 100 algunos siguen muy por debajo (en torno a $-1.000$ a $-2.000$). En **Combined-Q** todas las curvas convergen de forma parecida y rápida, hacia los episodios 30-50: CER elimina casi toda la sensibilidad al tamaño.
- **Pseudocódigo de APP p.82 (Combined-Q)**: inicializar $Q$ y el buffer $\mathcal{M}$; en cada paso, elegir la acción $\varepsilon$-greedy, ejecutarla, guardar $t=(S,A,R,S')$ en $\mathcal{M}$, muestrear un batch $\mathcal{B}$ de $\mathcal{M}$ y actualizar $Q$ con $\mathcal{B}$ **y** $t$.
- Lo que demuestra el paper, según la clase, es que la velocidad de convergencia cuando el buffer es demasiado grande se beneficia de CER. Recomendación (*dudoso* si es del profesor o la atribuye al autor del paper; la transcripción dice "recomendar a este autor es") (T06 ~04:36): **mejor pasarse de largo** con el buffer, por ejemplo 1.000 episodios en lugar de 100, y corregir la obsolescencia con el toque de actualidad de CER. **Nota:** el paper muestra que los buffers grandes degradan el aprendizaje y que CER reduce esa sensibilidad (gráficos de APP p.81); la conclusión de pasarse de largo es una lectura de clase, no una recomendación del paper.
- **En la práctica:** el `DDQNAgent` ya implementa CER en `sample()`: `batch_size - 1` índices sin reemplazo entre todos menos el último, más el último (BE dqn.py). El notebook `deep` usa replay uniforme estándar (RLM deep, celda 2). Detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md).

**Nota:** el autor que la transcripción recoge como "suto" es coherente con el paper, firmado por S. Zhang y R. S. Sutton (2017). Que en el paper se tratara de Q-learning con aproximación y en qué entornos no se ve en la diapositiva, así que aquí no se afirma.

---

## 7. Batch y frecuencia de aprendizaje

- **El batch no es el buffer**: el batch es el número de experiencias que se muestrean del buffer en cada actualización, por ejemplo 64 (T06 ~03:55, ~04:02). Las 64 deben venir de partidas distintas, recientes e independientes (§6.3). El profesor hace la distinción al hablar de cómo el tamaño del buffer depende de los datos disponibles (§6.3).
- En el LAB, `learn_after = batch_size` (se aprende cada 64 pasos si $B=64$) y `update_target_after = 10 * learn_after` (copia de pesos a la *target* cada 640 pasos) (LAB celda 12). `learn_after` es "cada cuántos steps se realiza un aprendizaje" (LAB celda 11). *Fixed target* y Double DQN: [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md).
- **Nota:** `sample()` exige que el buffer tenga al menos `batch_size` experiencias, porque muestrea `batch_size - 1` índices sin reemplazo de `len(buffer) - 1` (BE dqn.py). El llenado previo (LAB celda 16) lo garantiza.
- Valores de memoria y batch usados en el Taller 2 (1.000-20.000 y 1-128): [T2 03_dqn.md §10](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md).

---

## 8. Estabilidad e inestabilidad

### 8.1 Por qué el RL es inestable

**Idea.** Los entrenamientos son aleatorios: dependen de los caminos recorridos y de la exploración, y reciben mucho ruido y señales confusas (T06 ~01:13).

- "Incluso se dice" que, con todo perfectamente configurado, de **10 entrenamientos unos 3 fallan por completo** (T06 ~01:13). En T07 lo repite: en problemas algo más complejos, 3 de cada 10 entrenamientos perfectamente configurados no funcionan. Incluso el *cliff walking* algún día dará un camino por el medio, aunque quizá no ocurra en muchas ediciones (T07 ~00:27-00:28).
- Muchas mejoras de los papers van en la dirección de dar más estabilidad (T06 ~01:13): SARSA aprende de forma más suave y convergente aunque no siempre óptima; Expected SARSA promedia el valor siguiente según la política actual, por ejemplo 50 % la mejor y 50 % la media del resto, para tener aún más estabilidad; Double Q-learning y las "copias" que se pasan pesos cada cierto tiempo reducen sesgos (T06 ~01:14-01:18). Desarrollo en [00_repaso_rl.md](00_repaso_rl.md) y [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md). **Nota:** en rigor, con $\varepsilon$-greedy y $N$ acciones, $\mathbb{E}[Q(s',\cdot)]=(1-\varepsilon)\max_a Q(s',a)+\frac{\varepsilon}{N}\sum_a Q(s',a)$, porque la parte aleatoria también puede elegir la mejor acción; el "50 % la media del resto" de clase es una simplificación. Sobre Double Q-learning, ver la Nota de §4.6.
- **No solo el algoritmo**: todo el diseño del sistema debe buscar la estabilidad. En supervisado se vigila el sobreajuste; en RL, "usar lo que funcione" (T06 ~01:18-01:19).
- Pequeñas diferencias de código dan resultados muy distintos; se podría probar la mediana en lugar de la media del Expected SARSA. Muchos algoritmos con nombres distintos son copias de otro con un parámetro cambiado o una red duplicada (T06 ~01:16).
- APP p.88 cita a Andrej Karpathy (ex director de IA de Tesla): "Supervised learning wants to work. Reinforcement Learning must be forced to work".

### 8.2 Semillas y repetición

- Recompensas, $\gamma$, *steps* y estados interactúan, y cambios ligerísimos dan comportamientos totalmente distintos. **Incluso con las mismas recompensas** los entrenamientos varían mucho, así que hay que repetirlos varias veces o con varias semillas para saber cuán inestables son (T06 ~02:54-02:55). APP p.61 (diapositiva de recompensas): las mismas recompensas pueden dar resultados distintos según la semilla, y ligeras modificaciones dan un comportamiento totalmente distinto, lo que obliga a preguntarse qué lo ha causado.
- No hay un algoritmo mejor en general. Los que corrigen sesgos de otro suelen ir mejor, pero, por la inestabilidad, repitiendo 10 veces el orden podría invertirse (T06 ~01:38-01:40, ver [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md)).
- **Nota:** en la práctica conviene fijar semillas de `numpy`, `random` y del *framework* (TF o PyTorch) y del entorno, para reproducir cada experimento, y comparar configuraciones con varias semillas en lugar de una.

### 8.3 Ejemplos de inestabilidad y olvido vistos en clase

- **Aparcamiento** (T07 ~04:24-04:26): vídeo de un agente que aprende a aparcar. Le cuesta 15.000-20.000 repeticiones y hacia las 50.000 se "desaprende", algo que tendría que haber dominado ya. Ilustra la inestabilidad y la importancia de que el entrenamiento sea eficiente. Al empezar desde posiciones distintas generaliza, pero no ante obstáculos nuevos como un carrito de la compra; ahí es donde los modelos de lenguaje, que entienden contextos, todavía no saben traducir bien a acciones.
- **Perro boca abajo** (§3.6) y **brazos que se mueven en el sitio** por una recompensa mal ponderada ([03_recompensas_y_descuento.md](03_recompensas_y_descuento.md)): mínimos locales de los que la red no sale.

### 8.4 Reglas y otros sistemas para estabilizar

- Lo que deba ser al 100 % de una forma se programa como **regla determinista**: vender si el mercado cae un 10 %, o prohibir acciones que acerquen a dos casillas o menos del acantilado. La acción puede salir de la red, de un conjunto de reglas o de ambas, y las reglas también sirven para guiar el entrenamiento y acotar el espacio (T06 ~02:55-02:56; APP p.91).
- **Apoyarse en otros sistemas** (APP p.91; T07 ~00:23-00:26): no supervisado (*clustering*, reducción de dimensionalidad) y supervisado (generación de características, por ejemplo la probabilidad de ejecución, y ayuda en la exploración). Ejemplos de clase: agrupar activos según su perfil de liquidez para aplicar un algoritmo distinto a cada grupo, o generar la probabilidad de que se ejecute la orden en ciertas condiciones como característica del estado (T07 ~00:24). No todo es RL ni todas las decisiones tienen que salir de la IA; APP p.91 lo concreta como una "capa adicional" de reglas para temas sensibles. Ver [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).
- **Modelos**: ninguno predomina. La elección depende de las necesidades: si hacen falta acciones continuas, se descarta "la mitad" de ellos (T07 ~00:25). Hay que probar muchos, incluso cientos de variaciones propias, y combinarlos como en "rainbow", por ejemplo un Dueling Double DQN (APP p.91; T07 ~00:25).

---

## 9. Seguimiento del entrenamiento

### 9.1 Sobreajuste: al principio, buscarlo

**Idea.** El sobreajuste es "distinto" en RL: a veces interesa, "¡no en bolsa!" (APP p.90).

- Al principio se busca la sencillez e incluso el sobreajuste, y a partir de ahí se evoluciona (T07 ~00:18). Con el mismo día y el mismo activo hay que conseguir que aprenda. Si ni eso funciona, algo está mal configurado (recompensa, estados), así que sirve como test. Después se pasa a dos días (uno alcista y otro bajista), a varios activos y a escenarios de estrés (T06 ~04:45-04:47). Curriculum completo en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

### 9.2 Validación en modo ejecución

- Cada X *steps*, validación en datos no vistos en **modo ejecución** (*greedy*), comparando train y validación. Los resultados obtenidos durante el entrenamiento pueden confundir por la exploración (APP p.90; §2.3).
- Hay que mirar **comportamientos y no solo recompensas**, porque la recompensa es un proxy: no basta con que llegue al punto B, hay que ver cómo llega, andando o a la pata coja (T07 ~00:32; ~03:52).
- Para el caso de ejecución, el análisis final es frente al TWAP fuera de train, con métricas de media y dispersión, y en situaciones extremas (CASOS p.51): ver [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

### 9.3 *Catastrophic forgetting*

**APP p.88 (imagen):** curva de recompensa frente a episodios. Sube rápido, se estabiliza con pequeñas oscilaciones, cae bruscamente a menos de la mitad de la meseta ("Catastrophic forgetting"; no vuelve al origen) y después se recupera con oscilaciones fuertes hasta terminar por encima del primer máximo.

Explicación (T06 ~04:49): tras mucho entrenamiento, el agente empieza a explorar un camino malo y, por falta de independencia, olvida absolutamente todo. Para que no ocurra hay que usar *callbacks* y checkpoints que permitan recuperar buenos pesos.

### 9.4 Checkpoints y *early stopping*

- *Early stopping* y checkpoints sirven para **recuperar modelos** y **ajustar hiperparámetros**, y son "muy importantes debido a los largos entrenamientos" (APP p.90).
- Durante el entrenamiento hay que hacer paradas y registros por si hay que recuperar un modelo, no solo por *catastrophic forgetting* sino también porque parece **atascado**. Se recupera en ese punto, se ajustan los parámetros y se le da otra oportunidad desde ahí, ya parcialmente entrenado (T07 ~00:19).
- Combinado con la exploración: si empeora, volver al checkpoint anterior o subir $\varepsilon$ (§3.4).
- Es la idea C del ejercicio 4: condiciones de parada por aprendizaje, guardado periódico de pesos y posibilidad de recuperar (ENUNCIADO).
- **Nota:** el bucle del LAB no guarda pesos ni evalúa durante el entrenamiento (LAB celda 18). Lo único periódico es el log: `n_log = 25` es el único parámetro de entrenamiento ya fijado en LAB celda 12 (no es TODO; definido en LAB celda 11), y cada `n_log` episodios se imprimen las medias de pasos y recompensa de los últimos 25 episodios, el `mse` y $\varepsilon$ (LAB celda 18). Es el lugar natural para insertar la evaluación *greedy* periódica, el ajuste de $\varepsilon$ del ejercicio 3 (§3.2) y el guardado de pesos. Con el `DDQNAgent` de TensorFlow, los pesos están en `primary_model` y `target_model` (BE dqn.py), y para retomar un entrenamiento hay que guardar también $\varepsilon$, el contador de pasos y, si se quiere continuar idéntico, el buffer y el estado del optimizador.

### 9.5 Cómo leer la *loss*

**Idea.** "¿Y la loss?" (APP p.88). En supervisado el error baja hasta que se predicen bien las etiquetas, y entonces se ha terminado. En RL **el error informa de la novedad** de acciones y estados (T06 ~04:50-04:51):

- Un **pico** puede ser algo bueno: haber encontrado la llave de Montezuma (§6.7).
- Si los errores bajan a valores muy pequeños **sin picos**, puede ser que haya aprendido, o que ya no esté explorando y esté atascado.
- Por tanto, la loss no indica si el algoritmo ha terminado de aprender (T06 ~04:51).

En el código de clase, `learn` devuelve el error cuadrático (tabular) o la loss de `train_on_batch` (redes), y se registra por episodio para pintarlo con `plot_training_results` junto a las recompensas (RLM cliff-walking, celdas 9-10; LAB celda 18). **Nota:** en el LAB, en cada episodio de log se llama a `agent.learn()` solo para imprimir el `mse` (LAB celda 18), lo que añade una actualización de pesos extra; el mismo `learn()` adicional está en el notebook `deep` (RLM deep, celda 3). Además, `history_losses` guarda el último `mse` calculado, que solo existe tras el primer aprendizaje. `learn_counter` empieza en 0 en la celda 18, así que el primer `learn()` llega en el paso `learn_after` del entrenamiento. Si el primer episodio termina antes (el episodio acaba al completar la orden, y una política que compre en casi todos los pasos lo cierra en unos `vol_care` pasos, entre 30 y 75), `mse` no está definido y se produce un `NameError`. Con `learn_after = 64` el riesgo existe; con episodios que agoten el horizonte (400-600 pasos) no. Ver [08_entorno_best_execution.md](08_entorno_best_execution.md).

**Nota:** para seguir el entrenamiento, además de la loss, conviene registrar las recompensas suavizadas con una ventana móvil, la evaluación *greedy*, $\varepsilon$ y las métricas de comportamiento. En recompensas compuestas, conviene registrar cuánto aporta cada criterio, como recomienda el profesor ([03_recompensas_y_descuento.md](03_recompensas_y_descuento.md)). Ver también [T2 03_dqn.md §10](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md) (curvas suavizadas, test *greedy* con varias partidas).

---

## 10. Búsqueda de hiperparámetros y prueba y error

- "El desarrollo del aprendizaje por refuerzo funciona igual que el aprendizaje por refuerzo": por prueba y error (APP p.87). Hay que hacer muchísima prueba y error, y cuanto más claros estén los conceptos, mejor se aterriza. Se puede preguntar a Claude, pero no sustituye (T07 ~00:26).
- **Conocimiento de negocio** muy importante: el perfil mixto finanzas y datos sabe qué es realmente importante (APP p.92; T07 ~00:26-00:27).
- **Entrenamientos muy largos**: optimizar el código al máximo, optimizar la búsqueda de parámetros y apoyar el entrenamiento (APP p.92; T07 ~00:28).
- **Grid search**: era fuerza bruta y ya era inviable; con entrenamientos largos, todavía más. Hay paquetes de búsqueda más optimizada (*hypertuning*), pero tampoco funcionarán tan bien, porque hay que ver cómo interactúan los distintos elementos. Hay que ajustar "a manivela" y depender poco de esas librerías (T07 ~00:28-00:29; APP p.92).
- **Experimentación masiva** (APP p.62, imagen): una captura con decenas de carpetas de experimentos y un monitor tipo `htop` con unos 72 núcleos muy cargados, "72 running" y un *uptime* de 27 días. Corresponde a la anécdota del TFM del profesor (2019): lanzaba **72 experimentos en paralelo**, cada uno con recompensas y configuraciones distintas, en una máquina grande *(la cifra de memoria, "700 gigas", es dudosa)*. Por las mañanas leía papers y diseñaba recompensas, y por la noche lanzaba experimentos. Muchas de las cosas que cuenta vienen de esa experiencia. Al terminarlo, pese a sentirse orgulloso, vio cuántas cosas había hecho mal y siguió profundizando con compañeros que sabían más (T06 ~02:57-02:59). Por qué no llegó a producción: §11.
- **Nota:** para la práctica esto se traduce en una búsqueda pequeña y razonada, de pocas configuraciones, cada una con varias semillas, con un protocolo fijo de evaluación y un registro de decisiones. Protocolo y lo que se puede reutilizar de los Talleres 1 y 2 en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

---

## 11. Retos del RL y paso a producción

**Idea.** RL con modelos *deep* es una herramienta muy potente, pero todavía poco implantada por cuatro problemas (APP p.93; T07 ~00:29): **exploración**, **recompensas**, **inestabilidad** de los modelos y **creación del entorno o simulador**, real o simulado, con su coste y su validación para producción.

- **El simulador** (APP p.94-95; T07 ~00:29-00:30): casi siempre será simulado. Podría ser real, por ejemplo con personas jugando al póker o evaluando respuestas de un modelo de lenguaje, como en RLHF. Debe ser lo más realista posible, con interacciones realistas y velocidad de generación de experiencias, y suele ser uno de los principales problemas. Hay que comprobar que se corresponde con producción: cómo sé que un robot entrenado en simulación no hará cosas raras en la realidad. Lo mismo vale para agentes de ejecución o de inversión. La diapositiva cierra con que en 2026 la IA generativa ya ayuda a crear entornos (APP p.95); el profesor lo valora como "una vía de esperanza, pero muy verde" (T07 ~00:30).
- **Puesta en producción** (APP p.96; T07 ~00:30-00:32):
  - En producción se puede entrenar, pero **no desplegar de forma continua**, aunque el modelo diga que ha "mejorado".
  - Lo suyo es una **salida controlada**, con pocos usuarios o poco dinero. Probarlo antes en el simulador no garantiza una representación 100 % fiel.
  - Supervisada siempre, y con más cuidado que en supervisado. Allí, con etiquetas y una validación *out of time* correcta, se tiene una idea de cómo irá; aun así lo sensato es llevar un log los primeros meses para ver si la *performance* se desvía de lo entregado. En RL, además, el entorno es simulado y la recompensa es solo un proxy del objetivo.
  - Hay que mirar recompensas **y comportamientos**, y que el comportamiento en producción se parezca al del entrenamiento.
  - "El algoritmo puede hacer lo que sea con tal de maximizar sus recompensas" (APP p.96).
  - **La adopción no depende solo del rendimiento.** El algoritmo del TFM del profesor no llegó a producción: BME perdía cuota en algoritmos de ejecución frente a grandes operadores con interfaces más modernas, y los brókers preferían algo sencillo de usar aunque ejecutara algo peor, porque no es su dinero sino el del cliente (T06 ~02:59-03:00, paráfrasis). Detalle del caso en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).
- **Investigación reciente** (APP p.97; T07 ~00:32-00:34). El cuello de botella son millones de experiencias realistas en poco tiempo, y hay tres líneas: simulación en GPU con miles de entornos en paralelo (Isaac Lab, MuJoCo MJX), *world models* que generan mundos interactivos (Genie 3 → Genie 4, 2026) y mercados sintéticos, con una IA que genera órdenes como entorno de trading (MarS, Microsoft, 2025). La diapositiva cita como fuentes el anuncio de Genie 4 de Google DeepMind (27-oct-2026) y MarS de Microsoft Research (ICLR 2025). Al hablar de las ejecuciones en paralelo en GPU y de los mercados sintéticos, a estos últimos los toma "con muchas pinzas" (T07 ~00:33). La valoración de "vía de esperanza, pero muy verde" es de ~00:30, a propósito del recuadro de APP p.95 (ver arriba). La diapositiva cierra con que el simulador también se equivoca y hay que validar con datos reales antes de producción. Diseño del entorno en [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

---

## 12. Qué de todo esto toca la práctica

| Elemento | Dónde está en la práctica | Qué dijo la clase | Más detalle |
|---|---|---|---|
| $\varepsilon$, `min_epsilon`, `epsilon_decay` | LAB celdas 11-12 y 18 (decaimiento por episodio) | No por episodios; por mejora (§2-3). Es el ejercicio 3, de 1,5 puntos (T07 ~04:29), que añade que la duración del entrenamiento debería depender de la evolución del aprendizaje, aunque permite episodios fijos (ENUNCIADO; §3.7) | [notebook_partida.md](notebook_partida.md), [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md) |
| `alpha` | LAB celda 12; Adam, 0,01 por defecto (BE dqn.py) | No decae a cero (LAB celda 11); los parámetros de la red y $\alpha$, "menos importantes" que lo que está en el notebook *(dudoso: probablemente las funciones y parámetros del entorno)* (T07 ~05:01) | §5 |
| `buffer_size` | LAB celda 12; `deque`, 50.000 por defecto (BE dqn.py) | En episodios: unas 100 sesiones con diversidad de regímenes (T07 ~03:47). Se evalúa ya en el ejercicio 2, segunda parte, que exige ajustar y motivar $\gamma$ y el tamaño del buffer; en el ejercicio 1 no hace falta motivar la parametría (ENUNCIADO) | §6.3; 100 × ~500 *steps* ≈ 50.000 solo con episodios que agoten el horizonte; [notebook_partida.md](notebook_partida.md) |
| Llenado del buffer | LAB celda 16, entero y aleatorio | No hace falta llenarlo entero; independencia frente a ruido; riesgo del TWAP (§6.6) | [08_entorno_best_execution.md](08_entorno_best_execution.md) |
| Muestreo | CER en `sample()` (BE dqn.py) | CER corrige la sensibilidad al tamaño del buffer (§6.8) | [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md) |
| `batch_size`, `learn_after`, `update_target_after` | LAB celda 12 | §7 | [08_entorno_best_execution.md](08_entorno_best_execution.md) |
| `n_log` | LAB celda 12, fijado en 25 (no es TODO); el log imprime medias de los últimos 25 episodios (LAB celda 18) | — | **Nota:** lugar natural para la evaluación *greedy* periódica, para mover $\varepsilon$ y para guardar pesos (§3.2, §9.4) |
| `action_sample` (función 13) | LAB celda 7: TODO con `p`, pista `np.random.choice([0, 1], p=[1-p, p])` | Acción aleatoria no necesariamente uniforme (§4.5) | [08_entorno_best_execution.md](08_entorno_best_execution.md) |
| Evaluación | LAB celda 20: un episodio de test *greedy* | Varios episodios, sin exploración, periódica (§3.2, §9.2) | [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md) |
| Checkpoints y parada | No están | Idea C del ejercicio 4 (§9.4) | [notebook_partida.md](notebook_partida.md) |
| Buffer dinámico, exploración por estado | No están | Ideas válidas para el ejercicio 4 (T07 ~04:30-04:31; §4.4, §6.4) | [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md) |

Si el *deep* RL da problemas, el profesor acepta un Q-learning tabular, que no requiere buffer, refactorizando el agente (T07 ~05:01) *(dudoso el matiz de "un poquito más inestable" en la transcripción)*.

---

## 13. Erratas, correcciones y dudas

1. **Peso medio de la clase** (T06 ~03:41): con $\alpha=1\,\%$, de 80 kg y una observación de 72 kg sale 79,92 kg, no 79,8 (79,8 correspondería a $\alpha=2{,}5\,\%$). §5.3.
2. **"Proporcional al N veces visitado"** (APP p.74): debe ser inversamente proporcional, $\alpha=1/N(s,a)$, coherente con "más grande en estados nuevos". §5.4.
3. **Fórmula de PER** (APP p.80): el denominador aparece como $\sum_k p_i^{\alpha}$, con subíndice $i$. La forma correcta, la del paper arXiv 1511.05952, es $\sum_k p_k^{\alpha}$. El $\epsilon$ de $p_i=|\delta_i|+\epsilon$ es una constante pequeña para que ninguna experiencia tenga prioridad nula, no el $\varepsilon$ de exploración. El $\alpha$ de PER tampoco es el *learning rate*.
4. **Extremos del exponente de PER** (T06 ~04:32): la transcripción parece decir que con $\alpha=1$ el muestreo es uniforme y luego que $\alpha=1$ da todo el peso al error. Lo correcto es que $\alpha=0$ es uniforme y $\alpha=1$ pondera totalmente por la prioridad.
5. **Origen de PER** (T06 ~04:33): el profesor presenta la idea de PER como surgida de Montezuma y de las recompensas intrínsecas. **Nota:** PER (2015) es anterior a *Random Network Distillation* (2018, la imagen de APP p.80). Lo que comparten es el principio de usar el error de predicción como señal de novedad, no una relación de origen.
6. **$k$ en $1-(1-1/m)^k$** (APP p.78): la fórmula supone $k$ extracciones individuales uniformes con reemplazo. Con batches, $k$ = número de aprendizajes × tamaño del batch. §6.2.
7. **Cifras de Atari** (T06 ~03:58-04:01): el profesor duda entre 2 y 2,5 minutos y entre 60 y 30 fotogramas; con 2,5 min y 60 fps sale 9.000 *steps* por partida y 900.000 para 100 partidas, lo que redondea al millón de consenso.
8. **Steps intradía y de ajedrez** (T06 ~04:05-04:08): las cifras de la transcripción (75 % del día, 9 u 8 horas, 500/50/200 pasos de ajedrez) son confusas. El método (episodio → *steps* → número de episodios) es lo que importa.
9. **Regla de los movimientos en ajedrez** (T06 ~04:05): en clase se habla de 40 movimientos sin capturas, en un intercambio confuso en la transcripción. El reglamento FIDE fija 50 movimientos de cada jugador sin captura ni movimiento de peón (reclamables) y 75 para tablas automáticas. §6.3.
10. **Duración de los episodios del LAB** (§6.3, §6.6): 400-600 es el horizonte máximo (`episode_bins`); el episodio termina antes si se completa la orden. Las cuentas de buffer en episodios ($100\times500$) solo valen para políticas que agoten el horizonte.
11. **Double Q-learning** (T06 ~01:18): la clase lo describe como ponderar las creencias de dos diccionarios; en realidad desacopla selección ($\arg\max$ con una tabla) y evaluación (con la otra). §4.6.
12. **"27 días"** (T06 ~03:06): se interpreta como alusión al *uptime* de la captura de APP p.62 *(dudoso)*.
13. **Comentario del LAB** (LAB celda 16): "En este punto eps es 1 → actuando random" solo es cierto si en la celda 12 se elige `epsilon = 1`. El `DDQNAgent` arranca con `epsilon=0.9` por defecto (BE dqn.py).
14. **Paso 0 forzado** (BE dqn.py, act): en el primer paso el agente devuelve siempre la acción 1, también durante el llenado del buffer y la evaluación. Afecta a lo que se guarda en el buffer. Análisis en [08_entorno_best_execution.md](08_entorno_best_execution.md).
