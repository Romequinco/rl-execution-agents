# Repaso de RL tal como se vio en este taller

> **Fuentes:**
> - **APP** (`docs/raw/MIAX14/applying_RL_MIAX.pdf`, "Applying Reinforcement Learning", Jesús Sanz): Primera Parte completa, p.1-30 (portada, agenda, componentes, Markov, RL frente a supervisado, objetivo, MC/PD/TD, Q-learning, SARSA, Expected SARSA, Double Q-learning); p.31 (paso a la aproximación funcional), p.47 (papers de RL), p.51-52 (CartPole), p.74-76 (step size y gamma), p.97 (entornos con IA generativa) y el apéndice p.98-102 (repaso de componentes, V y Q, N-step TD). Se han mirado las imágenes de las páginas con gráficos (p.3, 5, 7-8, 10-14, 17-18, 20-21, 23, 25-26, 29-30, 47, 52, 75, 101-102).
> - **CASOS** (`docs/raw/MIAX14/casos_practicos.pdf`): p.45-50 (agentes, buffer y evolución de la tabla Q hasta la política en el caso de ejecución).
> - **T06** (`docs/raw/26 11 06.pdf`, transcripción automática de la clase del 06/11/2026): ~00:02-01:20 (repaso de la Primera Parte), ~02:06-02:20 (evolución de la investigación y diseño del estado), ~03:13-03:17 (convergencia tabular), ~03:35-03:42 (step size α).
> - **T07** (`docs/raw/26 11 07.pdf`, clase del 07/11/2026): ~00:08-00:10, ~00:23-00:26, ~02:46-02:54, ~03:51-03:53, ~04:09-04:27.
> - **RLM** (`docs/raw/MIAX14/rl-material/notebooks/`): `rl-models-basic/TD-q-learning.ipynb`, `TD-expected-sarsa.ipynb`, `TD-double-q-learning.ipynb` y `cliff-walking.ipynb`, solo para contrastar fórmulas con el código (el detalle de cada notebook está en [09_material_rl_notebooks.md](09_material_rl_notebooks.md)).
> - **T1/T2**: teoría de los Talleres 1 y 2 en `docs/raw/referencias/rl-trading-agents/docs/teoria/` y `docs/raw/referencias/deep-rl-trading-agents/docs/teoria/`, enlazada para profundizar.
>
> **Convenciones:** "(T06 ~00:52)" = hora aproximada (hh:mm) en la transcripción; todo lo que viene de ellas es paráfrasis, con citas literales muy cortas y entre comillas. "(APP p.22)" = página del PDF de diapositivas. "(RLM TD-q-learning, celda 6)" = celda del notebook contando desde 0. **Nota:** = contenido propio (cálculos, correcciones, conexiones), no dicho en clase. *(dudoso)* = la transcripción no permite asegurar lo dicho; se dan las lecturas posibles. Notación: $s_t, a_t, r_t$ estado, acción y recompensa en $t$; $\gamma$ factor de descuento; $\alpha$ step size; $\varepsilon$ probabilidad de explorar. Las diapositivas usan $r_t$ para la recompensa que sigue a $a_t$ (Sutton & Barto la llama $R_{t+1}$); aquí se respeta la notación de clase.
>
> Escrito el 10/11/2026.

---

## 1. Para qué fue este repaso y cómo se planteó

**Idea.** El primer día (06/11) el profesor dedicó la primera parte de la clase a un repaso "práctico" de RL, modulado según lo que pidiera el grupo, como base para la segunda parte (cómo definir en la práctica estados, acciones, recompensas, γ, α, ε) y para los dos casos financieros del segundo día (profit seeking y, sobre todo, ejecución de órdenes, que es la práctica) (T06 ~00:04-00:06; APP p.3).

- **Quién da el taller** (T06 ~00:02-00:04; APP p.2): Jesús Sanz (CFA, APP p.2) cursó la primera edición del máster; trabaja como senior manager en una consultora, sobre todo con banca, y antes estuvo en el Instituto BME con el antiguo director del máster, donde hicieron algoritmos de inversión, un robo-advisor y el caso de ejecución que se usa en el taller. Se define como un perfil mixto técnico-financiero, frente al profesor más técnico de los talleres anteriores.
- **Agenda oral** (T06 ~00:04-00:06): el día 1, repaso muy básico y después cómo fijar en la práctica cada componente (p. ej. si γ debe ser 0,5 o 0,9, qué tener en cuenta en las recompensas…); el día 2, los mismos componentes aplicados a dos casos: un algoritmo de **inversión**, que busca un retorno o una utilidad para ti, y sobre todo uno de **ejecución**, que coge tus órdenes (p. ej. comprar Santander) e intenta ejecutarlas "al mejor precio". El taller será el de ejecución; quiere alinear la carga de trabajo con el grupo, para no mandar ni mucha ni poca tarea, por ser la primera edición con formato de taller (ver [notebook_partida.md](notebook_partida.md)).
- **Aviso previo de las diapositivas** (APP p.1): no hace falta estudiarlas antes; la primera parte es un resumen de RL y la segunda puede tener algún concepto nuevo; "Esta clase está para ayudar a entender mejor el RL, no para complicarlo más". Recomienda mirar los notebooks, sobre todo **TD-q-learning** y **deep**, "que son la base del resto", y el README para ejecutarlos (ver [09_material_rl_notebooks.md](09_material_rl_notebooks.md)).
- **Agenda** (APP p.3): **Día 1**, repaso práctico de los componentes de un sistema de RL (a debate: qué habéis visto y con qué os sentís cómodos), introducción a la creación de entornos y factores a considerar en el diseño de los algoritmos de RL; **Día 2**, continuación de los conceptos teóricos pendientes, ejemplo de profit seeking y ejemplo de algoritmos de ejecución; **Día 3**, "Taller: exposición de los casos resueltos como tarea" (junto a un icono de aula virtual). Logística en [notebook_partida.md](notebook_partida.md).
- **Qué pidió el grupo** (T06 ~00:06-00:08): un alumno dijo que tras las clases anteriores no se sentía seguro y que prefería entender de forma elemental los bucles de aprendizaje, el flujo de información y los principios por los que unos algoritmos funcionan mejor que otros, más que un catálogo de algoritmos. El profesor lo aceptó y orientó el repaso a principios (en especial, la **estabilidad**, §11.1 y §12).
- **Sondeo inicial** (T06 ~00:10-00:13): mostró el diagrama agente-entorno y preguntó por estado, acción, recompensa (inmediata y futura), política y valores V/Q. Constató que la relación entre Q-valores y política no estaba clara para varios alumnos ("están ahí un poco bailando los conceptos"), y por eso dedicó tiempo al ejemplo del paraguas (§3).
- El segundo día volvió a ofrecer repasar lo que se quisiera (experience replay, algún algoritmo, Q-learning) mientras se conectaba la gente; nadie pidió nada concreto y siguió (T07 ~00:08-00:10). Dijo de Q-learning que se puede importar o reutilizar su implementación *(dudoso: parece referirse a la del material RLM)*.

---

## 2. Componentes de un sistema de RL

**Idea.** RL es un bucle: el agente observa un estado, toma una acción, el entorno le devuelve una recompensa y un nuevo estado, y así hasta el fin del episodio. La diapositiva lo dibuja como dos cajas (AGENTE y ENVIRONMENT) unidas por tres flechas (estado $s$ y recompensa $r$ del entorno al agente, acción $a$ del agente al entorno), con la secuencia $s_t a_t r_t \to s_{t+1} a_{t+1} r_{t+1} \to \dots \to s_{t+n} a_{t+n} r_{t+n} \to$ fin de episodio (APP p.5).

| Componente | Definición de clase | Ejemplo del paraguas (APP p.99) |
|---|---|---|
| **Environment** | Devuelve un nuevo estado y una recompensa tras cada acción | El tiempo atmosférico |
| **Estado** $s$ | Lo que observa el agente para decidir; en la práctica, un vector de características (§4) | Lluvia o sol |
| **Agente** | Quien decide y aprende de la experiencia | Tú, al salir de casa |
| **Acción** $a$ | Lo que puede hacer el agente en cada estado | Coger o no el paraguas |
| **Recompensa** $r$ | Señal inmediata tras actuar | +100 por ir seco con lluvia |
| **Objetivo** | Maximizar la recompensa acumulada descontada, $G_t=r_t+\gamma G_{t+1}$ (§6) | — |
| **Política** $\pi$ | Regla que dice qué acción tomar en cada estado | "Siempre paraguas" |
| **Valores** $V$ y **Q-valores** $Q$ | Valor del estado según la política / valor de cada acción en cada estado | Tablas de §3 |

En el caso de ejecución la misma descomposición aparece con contenido financiero (CASOS p.46-50): el entorno es un **simulador** que genera órdenes (episodios), envía $s_t$, recibe $a_t$, evalúa la acción, devuelve $r_t$ y envía $s_{t+1}$; el estado es un vector de características (spread, probabilidad de ejecución, tiempo restante, volumen restante); las acciones, pasiva/mantener/agresiva; la "misión" del agente es tomar acciones, explorar y aprender. Ese caso se desarrolla en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

Para la formalización (MDP, $P(s'|s,a)$, Bellman) ver [T2 00_fundamentos_mdp.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/00_fundamentos_mdp.md) §2-§7.

---

## 3. Política, V y Q: el ejemplo del paraguas

### 3.1 Las tablas de la diapositiva

**Idea.** $Q(s,a)$ da, para cada estado, el valor de cada acción disponible; $V(s)$ da el valor del estado, y **depende de la política**; la política es la "guía de actuación" en cada estado (T06 ~00:14-00:15). La diapositiva lo muestra con iconos (nube con lluvia, sol con nubes, paraguas con check verde, paraguas tachado en rojo) y cifras verdes/rojas (APP p.7-8, p.100):

**Q-valores** (no dependen aquí de la política, ver Nota):

| $Q(s,a)$ | Lluvia | Sol |
|---|---|---|
| Con paraguas | **+100** | −20 |
| Sin paraguas | −100 | **+50** |

**Valores del estado según la política:**

| $V^\pi(s)$ | Aleatoria 50 % | Siempre paraguas | Óptima $\pi^*$ |
|---|---|---|---|
| Lluvia | 0 | +100 | +100 |
| Sol | +15 | −20 | +50 |

- $V$ es la media de $Q$ ponderada por la probabilidad que la política da a cada acción (APP p.100): $V^\pi(s)=\sum_a \pi(a|s)\,Q(s,a)$. Lluvia con política aleatoria: $0{,}5\cdot100+0{,}5\cdot(-100)=0$ (APP p.100). **Nota:** sol con política aleatoria: $0{,}5\cdot(-20)+0{,}5\cdot50=15$, la cifra de la tabla.
- Con "coge siempre el paraguas" el valor del día lluvioso es 100 porque siempre se toma esa acción; con la aleatoria sería 0 porque unas veces lo coges y otras no (T06 ~00:15).
- **Política óptima** (APP p.8): $\pi^*(s)=\arg\max_a Q(s,a)$; en la diapositiva se resaltan en $Q$ el +100 (paraguas con lluvia) y el +50 (sin paraguas con sol), y la columna $\pi^*$ de $V$ vale 100 y 50. En palabras del profesor: con lluvia, coger el paraguas; con sol, no cogerlo porque pesa (T06 ~00:15).
- La política puede ser **determinista y dada** por quien diseña ("siempre paraguas") o **aprendida** por el algoritmo, que es lo que se hace en RL (T06 ~00:15).

El propio profesor lo señaló al pasar al ejemplo numérico: el de las nubes era "un ejemplito aislado de un solo pasito de secuencia", y el de Q-learning muestra qué pasa cuando el juego continúa (T06 ~00:52-00:53; §9.2). **Nota:** en este ejemplo el episodio dura un paso (sales de casa y recibes la recompensa), así que $Q(s,a)$ es la recompensa esperada y no depende de la política. En problemas de varios pasos $Q^\pi(s,a)$ también depende de $\pi$, porque incluye el valor de lo que harás después: $Q^\pi(s,a)=\mathbb{E}[r_t+\gamma V^\pi(s_{t+1})]$. Por eso en Q-learning se aprende directamente $Q^*$ (§9).

### 3.2 De los Q-valores a la política

Un alumno planteó el caso de un modelo que, en datos sin tendencia ni señal, aprende a no operar porque ninguna estrategia al alza ni a la baja le da valor, y preguntó cómo eso se convierte en una política (T06 ~00:16-00:17). Respuesta: la política óptima queda definida por la acción que maximiza en cada caso; "por detrás" el ordenador coge la que maximiza $Q$, y esa es la política (con lluvia, el paraguas; con sol, no). **Nota:** aplicado a su caso, si ninguna estrategia aporta valor, la acción de no operar acaba siendo la de mayor $Q$.

La misma idea aparece con cifras en el caso de ejecución (CASOS p.47-50), con una tabla de Q-valores de estados $S_1, S_2, S_3, S_x$ por acciones $a_1$ (pasiva), $a_2$ (mantener), $a_3$ (agresiva):

| Repeticiones | Tabla (filas $a_1$, $a_2$, $a_3$; columnas $S_1, S_2, S_3, S_x$) | Rótulo |
|---|---|---|
| 0 | todo 0 | "No sabe nada" |
| 500 | $a_1$: −0.3, 0.4, 0.1, −1.2 · $a_2$: 0.1, −0.3, 0.6, 0.2 · $a_3$: 1.1, 0.2, 1.4, 0.8 | "Ya sabe algo" |
| N | $a_1$: −2, **7**, 1, 3 · $a_2$: 4, −3, **6**, 1 · $a_3$: 5, −3, −4, **8** | "Llega a una solución" |
| Política (N) | one-hot: $S_2\to a_1$, $S_3\to a_2$, $S_x\to a_3$; columna $S_1$ toda a 0 | — |

En verde se marcan los máximos por columna (7, 6, 8). El profesor lo explicó así: si se discretiza el estado en lugar de usar aproximación funcional, queda "el diccionario que vimos ayer" de estados × 3 acciones; con las repeticiones los valores se actualizan hasta converger, y la política es la acción que maximiza en cada estado; luego hay que testearlo bien con métricas de comportamiento, no solo mirando las recompensas (T07 ~03:51-03:52).

**Nota (errata CASOS p.50):** en la tabla de N repeticiones el máximo de $S_1$ es $a_3=5$ (frente a 4 y −2), pero la política de p.50 deja la columna $S_1$ toda a 0 y no lo resalta en p.49. Lo correcto sería $\pi(S_1)=a_3$. Parece una simplificación de la diapositiva (solo se colorearon tres estados), no un concepto distinto.

### 3.3 Aprender por prueba y error: la tabla Q como "diccionario de experiencias promediadas"

**Idea.** RL es "probar, errar y hacer medias": se prueba algo, se actualiza la creencia sobre lo que debería pasar, pero no se da confianza plena a una sola experiencia, porque la misma acción en la misma situación puede dar resultados distintos; se pondera según cuántas veces se ha repetido (T06 ~00:17-00:18).

Ejemplo del recién nacido (T06 ~00:18-00:21):
1. Al principio no conoce su espacio y actúa al azar para "testear las aguas".
2. Un día de lluvia no coge el paraguas, se moja: eso se traduce en una recompensa que el algoritmo entienda, p. ej. −100.
3. Un día de sol coge el paraguas: también es malo, pero menos que mojarse, así que no se penaliza con −100 (cuánto exactamente se discutió en la parte de recompensas, ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md)).
4. Un día de lluvia lo coge: recompensa positiva.
5. Lo que queda es la tabla Q, "un diccionario de experiencias promediadas" (T06 ~00:19) que se consulta para decidir cuando vuelve a darse el estado.

**Probabilidad de transición.** La realidad no es "llueve / no llueve" sino, por ejemplo, "está nublado por la mañana": a veces llueve y a veces no; esa incertidumbre es la probabilidad de transición desde el estado nublado. Tras revivir el día nublado miles (o millones) de veces, los valores convergen a un promedio, actualizado con un factor de aprendizaje α (T06 ~00:20-00:21). "Si entendéis esto, ya tenéis la base de cómo convergen estos sistemas" (paráfrasis, T06 ~00:21).

**El principio fuera del RL** (T06 ~00:21-00:22): en su consultora hacen modelos de elasticidad de precio de productos bancarios (qué precio ofrecer según el tipo de cliente: a uno leal quizá un precio peor; a uno con actividad en otros bancos que se quiere activar, una campaña algo mejor). Se aplican los mismos principios de prueba y error y ajuste de creencias: un único experimento no se toma como 100 % creíble, porque pudo deberse a algo de ese mes, y se vuelve a testear.

---

## 4. Propiedad de Markov y qué meter en el estado

### 4.1 La propiedad

$$\mathbb{P}[S_{t+1}\mid S_t]=\mathbb{P}[S_{t+1}\mid S_1,\dots,S_t]$$

La probabilidad de pasar al siguiente estado depende solo del estado actual (APP p.9-11; T06 ~00:23).

**Test del coche** (APP p.9-10): un coche rosa a punto de chocar con uno azul debe decidir si frena. Opciones: $S=[\text{Velocidad, Hora actual}]$, $S=[\text{Distancia, Velocidad}]$ (marcada con un check verde: correcta), $S=[\text{Distancia, Hora actual}]$, "ninguna". ¿Por qué no vale $[\text{Distancia, Hora}]$? Para frenar hace falta la velocidad, y con distancia y hora solo se obtiene mirando el estado anterior:

$$v_t\approx\frac{d_{t-1}-d_t}{h_t-h_{t-1}}$$

El estado actual no basta: no cumple Markov (APP p.10; T06 ~00:23-00:24). Mensaje: así hay que pensar las características que se pasan al algoritmo.

### 4.2 Pasar información del pasado no rompe Markov (y es necesario)

APP p.11 (con un gráfico de velas y una captura de Breakout): "¡nuestro estado no es el precio de hoy!". No hay que confundir la propiedad con pasar información del pasado, que es **necesario**: últimos 30 retornos, métricas con ventana de 30 días…; en Breakout, los últimos fotogramas muestran hacia dónde va la bola.

En clase (T06 ~00:24-00:26) el profesor preguntó si pasar precios o retornos de días anteriores rompe la propiedad; un alumno razonó que no, porque en el estado de hoy se pueden conocer los valores del pasado, y el profesor lo confirmó: una cosa es tener que remontarse a estados anteriores y otra meter la información del pasado **dentro** del vector de estado actual (p. ej. "el retorno de los últimos días es X %"). Con dos fotogramas de Breakout ya se sabe la dirección de la bola; con uno no. "No solamente no rompes, sino que es necesario" (T06 ~00:26).

Matiz que añadió sobre la marcha: siempre hay que **normalizar** las características para que sean comparables entre activos; no meter precios en bruto (T06 ~00:25; repetido en ~02:19).

### 4.3 Si falta la variable clave, no se aprende (ni con RL ni con supervisado)

Un alumno contó su experiencia en una práctica anterior de comprar/mantener/vender *(dudoso: parece referirse a los talleres anteriores)*: con datos aleatorios el agente aprendía a no hacer nada (no hay nada mejor que aprender), y con datos reales funcionaba peor que con datos generados, porque los precios reales se mueven por actores (p. ej. una institución que vende "a saco") (T06 ~00:30-00:31).

Respuesta del profesor (T06 ~00:31-00:34):
- Si hay un patrón y el vector de estado contiene las variables que lo determinan, un RL bien diseñado lo aprende.
- Si falta una variable clave, **no lo aprende nunca**, ni RL ni supervisado ni nada. Ejemplo: "baja los lunes (la gente está triste) y sube el resto de la semana". Con un millón de características pero sin el día de la semana no lo aprende, y no se arregla haciendo la red más grande o más pequeña: la pregunta es qué información se le está pasando.
- Variables correlacionadas con la omitida (la racha de días positivos/negativos, si ayer subió o bajó) permiten "rascar" algo: si ayer bajó y nunca hay dos bajadas seguidas, esa parte la acertarás. Hay muchos *confounders* correlacionados con las variables omitidas.

### 4.4 Cómo elegir las características: ponerse en el lugar del agente

(T06 ~02:11-02:19, inicio de la Segunda Parte.)
- El estado es un vector de características, igual que en supervisado; pueden ser continuas o discretas, **privadas** (dependen de mí: cartera, inventario) o **externas** (públicas). En cuanto hay una continua, el espacio de estados es infinito.
- En TFMs que ha supervisado y corregido ha visto muchas veces características metidas sin justificación ("¿por qué metes esto?"). La regla: ponerse en el lugar del agente, que es "una externalización de ti mismo": ¿qué necesitarías tú como operador de bolsa o como jugador? El agente es como una persona ficticia a la que quieres parecida a ti o, mejor, una versión mejorada de ti; puede procesar más cosas que tú, aunque tú tienes la intuición (T06 ~02:13).
- **CartPole** (el juego de la escoba en la mano: que no se caiga y que no se aleje demasiado del punto de inicio): los alumnos propusieron posición, distancia, velocidad, gravedad. Las 4 características que da el entorno están en el dibujo de la diapositiva: un carro sobre un eje con el 0 marcado y flechas *Position* (distancia del carro al 0), *Velocity* (del carro), *Angle* (inclinación del palo respecto a la vertical) y *Angular velocity* (APP p.51-52; T06 ~02:15, donde las señaló: "Os ha faltado esto"). Si se quita una (por ejemplo, filtrando la observación que devuelve `step`), cuesta mucho converger a la solución perfecta de 500 pasos: un problema sencillo no converge por falta de información. No tienen por qué ser esas 4 en concreto (un alumno mencionó otras, como la longitud del palo): si el modelo puede encontrar la relación a partir de otras variables que se le pasen, vale; el problema aparece cuando la información no se puede recuperar con el resto. Más información tampoco es gratis: puede complicar por la dimensionalidad (hay técnicas para que el modelo aprenda en qué fijarse). Pensar bien el problema es el primer paso y es "muy, muy importante"; con algo tan sencillo ya se puede "meter la pata" (T06 ~02:15-02:17).
- Para usar Q-learning **tabular** con estado continuo hay que **discretizar**; el nivel de discretización determina lo que se aprende: muy fino → espacio inmenso que no converge; muy grueso ("alta/baja") → mucho ruido; algo como 5-10 rangos por variable. El notebook de CartPole con Expected SARSA llega a la solución perfecta discretizando (ver [09_material_rl_notebooks.md](09_material_rl_notebooks.md)).
- Frase final *(dudoso)*: con problemas continuos, "no paséis a soluciones que sean espectaculares"; muchas veces se cree tener un problema continuo y en realidad es pereza de discretizar, porque a veces no se necesita el valor continuo (T06 ~02:19, paráfrasis salvo la cita). La lectura literal encaja: no saltar a soluciones complejas (deep) si discretizando basta, es decir, **no descartar la versión tabular** solo porque las variables sean continuas. Es la misma idea que dio al pasar a la aproximación funcional, donde propuso regresiones como primera aproximación, mucho más rápidas, en lugar de "saltar directamente a soluciones mucho más complejas" *(dudoso: la frase está entrecortada)* (T06 ~01:22-01:23).

**Nota:** en RLM TD-q-learning (celda 7) la discretización de CartPole se hace con el rango entre los percentiles 10 y 90 de 500 episodios aleatorios dividido en `grid_step=50` celdas, y `encode` redondea cada componente a un entero (celda 6). Es decir, unas 50 celdas por dimensión dentro del rango central, más fino que los "5-10 rangos" que mencionó como orden de magnitud.

El diseño del estado en la práctica (Markov, normalización, información privada/externa, frecuencia) se desarrolla en [02_diseno_de_entornos.md](02_diseno_de_entornos.md); para el caso de ejecución, en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

---

## 5. Cuándo usar RL frente a supervisado

### 5.1 Criterios (APP p.12)

| Criterio | Qué significa |
|---|---|
| Datos y decisiones **secuenciales** | No independencia entre decisiones |
| **Interacciones** con el entorno | Las acciones tienen efecto en el entorno |
| **Ruido** en la toma de decisiones | *Credit assignment problem* y *action advantage* |

A la derecha hay imágenes de piezas de ajedrez y un gráfico de velas (arriba) y de un termostato Nest, el logo de Tesla y un robot humanoide y otro cuadrúpedo, del estilo de Atlas y Spot (abajo), sin asignación explícita a cada criterio. La diapositiva enlaza vídeos de Breakout, Walking, Driving, Park y Hide and Seek, y termina con la **regla práctica**: si basta un modelo supervisado, úsalo; RL exige construir un entorno (APP p.12).

**Nota (definiciones propias, no dadas en clase):** *credit assignment*: con recompensas retrasadas no se sabe qué acción de la secuencia causó el resultado. *Action advantage*: lo que importa es cuánto mejor es una acción frente a las demás en ese estado, $A(s,a)=Q(s,a)-V(s)$, algo que no se observa en un histórico etiquetado (ver Dueling en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md)).

Discusión en clase (T06 ~00:12-00:13, ~00:26-00:30):
- Un alumno percibía que en supervisado se fijan los criterios que debe seguir el modelo, mientras que en RL se le dan recompensas y más libertad para llegar a la solución.
- Otro alumno: RL sirve cuando no se quiere solo imitar un patrón sino mejorarlo; en supervisado la máxima aspiración es el patrón. El profesor lo destacó: no tener etiquetas permite, potencialmente, comportarse mejor que el experto que las habría puesto.
- Secuencial e interacción: en ajedrez hay muchísimas decisiones encadenadas, una infinidad de opciones, y cada jugada te lleva por caminos distintos; coger o no el paraguas un solo día, en cambio, se resuelve con supervisado porque ya conoces la respuesta del pasado.
- Ruido en las etiquetas: un experto de ajedrez dice "mueve el caballo" y otro "mueve la reina"; ¿quién tiene razón?
- Regla práctica: "si podéis pensar que con un supervisado lo podéis hacer", hacedlo con supervisado, porque en RL hay que construir un entorno. En supervisado, el problema de la lluvia se resolvería con las características relevantes del día (la humedad, todo lo relativo al tiempo) y etiquetas construidas con el histórico de si llovió o no. Un RL bien diseñado también puede abordarlo, pero hay que diseñar el sistema de recompensas y requiere más esfuerzo (T06 ~00:29-00:30).

### 5.2 En finanzas: ¿nuestras decisiones cambian los datos que veremos?

**Gráfico de APP p.14** ("impacto de nuestras órdenes"), dos paneles con el precio en € (eje 80-110) del Día 1 al Día 16:
- Izquierda, *Supervisado: precio histórico*: la serie baja hasta un mínimo de **88** (etiqueta 1: compra) y sube hasta un máximo de **104** (etiqueta 0: vende). Las etiquetas se fijan sobre el histórico y asumen que nuestras órdenes no mueven el precio.
- Derecha, *RL: precio si compramos en el mínimo*: la curva "con impacto" (azul) se separa de la "histórica" (gris): al comprar en el mínimo, este sube (rótulo "Compramos: el mínimo sube", el suelo queda en torno a 90), el máximo posterior se adelanta y queda más bajo (~102) y la serie acaba en ~96 frente a ~105 del histórico. Las etiquetas ya no valen; RL modela ese impacto.

En clase (T06 ~00:34-00:38): el supervisado etiqueta momentos de compra/venta sobre un histórico estático; si operas con capital relevante, al comprar no habrías visto ese precio y además impactas lo que viene después, de modo que las etiquetas quedan inservibles. Las consecuencias de cada acción hay que **modelizarlas en el entorno**; si el entorno modeliza perfectamente las consecuencias, "lo tienes todo hecho" en mercados. El impacto también puede ser **interno**: el estado combina variables externas y privadas (cartera, concentración: si ya tienes BBVA quizá no te interesa Santander), y tu acción de comprar cambia tu estado. Con posiciones pequeñas y sin que importe la cartera, supervisado puede bastar, aunque incluso ahí se pueden aprovechar principios de RL.

**Tabla de APP p.15** (pregunta clave: ¿nuestras decisiones cambian los datos que veremos después?):

| Supervisado: no cambian | RL: sí cambian |
|---|---|
| Detección de fraude: cada operación tiene su etiqueta | **Ejecución de órdenes grandes**: cada tramo mueve el precio del siguiente |
| Previsión de volatilidad o VaR: predecir el riesgo no mueve el mercado | Market making: las cotizaciones y el inventario cambian el flujo de órdenes recibido |
| Señales con posiciones pequeñas en activos líquidos: impacto despreciable | Construcción de carteras y costes: cada compra y rebalanceo condiciona las siguientes (impacto interno) |

APP p.13 deja abierta la pregunta "¿Ejemplos financieros? ¿Supervisado o RL?" sobre un gráfico de velas que oscila en un rango (dos líneas discontinuas de soporte y resistencia).

### 5.3 No todo es IA: combinar supervisado, RL y reglas

(T07 ~00:23-00:26.) Un problema real tiene muchos pasos y hay que decidir en cuál aplicar ML y de qué tipo: RL puede venir bien para un punto y supervisado para otro. Ejemplos: agrupar activos por perfil de liquidez para aplicar algoritmos distintos; o un modelo de probabilidad de ejecución de la orden cuyas salidas se meten como características del estado. No todas las características tienen que salir de otro modelo ni todas las decisiones del agente; se pueden **programar reglas**. Entre los algoritmos de RL ninguno predomina; si necesitas acciones continuas descartas la mitad; y se pueden combinar ideas (p. ej. un Dueling Double DQN, como hace Rainbow; ver [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md)). Un asistente de IA os sugerirá algoritmos, pero hay que probar variaciones y sentirse libre de incluir "alguna pequeña tontería" (T07 ~00:25). Cuanto más claros los conceptos, mejor se aterriza, aunque se pregunte a un asistente de IA; al final hay que hacer muchísimas pruebas. Como hay muchos parámetros y las recompensas son muy difíciles de entrenar, el **conocimiento de negocio** es "superimportante": hay que pensar qué se necesita realmente y cómo estructurar bien el problema. El perfil con esa visión de conjunto (datos y finanzas) es muy valioso en las empresas, donde a menudo el científico de datos y la gente de negocio tienen que entenderse (T07 ~00:26-00:27). Anunció además que, con el algoritmo de ejecución, daría alternativas para enfocar el ejercicio y reducir su alcance (T07 ~00:23-00:24); ver [notebook_partida.md](notebook_partida.md).

**Por qué RL en ejecución: la explosión de reglas** (T07 ~02:46-02:51). Se pueden imaginar reglas razonables para mejorar un TWAP: si hay actividad (trades) en el mercado, adelantar la ejecución dentro de la ventana de 10 minutos (un "toque de POV"); si hay poca liquidez, esperar. Pero al combinar 2 o 3 condiciones explota la dimensionalidad y hay que acertar la regla exacta para cada circunstancia. Además los expertos no se ponen de acuerdo: si el spread se abre, uno dirá que esperes a que se recupere y otro que compres ya porque la gente está ejecutando ahí; eso es ruido en la decisión. Quienes ofrecen estos algoritmos (los brókers, por el contexto *(dudoso: la transcripción solo dice "ellos")*) saben que son muy mejorables y ya han introducido alguna mejora (p. ej. intentar no ser detectados), pero no implementan todas las reglas porque es inviable meter tanta condicionalidad (T07 ~02:50). La alternativa: darle al agente la información (frecuencia de trades, spread respecto a sus percentiles de los últimos minutos) y que RL aprenda qué habría sido óptimo históricamente; así, "no necesito como experto decir que esto es bueno y esto es malo", y se le pueden dar más y más condiciones (T07 ~02:50). En Q-learning, eso es aprender el "diccionario" estado→acción. Detalle en [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md) y [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

**Supervisado como característica** (T07 ~02:51-02:54): la probabilidad de que una orden pasiva se ejecute en los próximos segundos se entrena fácilmente con supervisado sobre el histórico de pasivas (frecuencia de trades, volatilidad, spread…). A mano habría que fijar umbrales (¿pasiva si >90 %? ¿y si es 70 % pero hay liquidez?) y vuelve la explosión; con RL la regla la aprende el agente. Se le podría pasar al RL la misma información cruda y quizá calcularía internamente algo parecido, pero construir la característica por fuera ("masticadito") simplifica lo que viene después. Ejemplo de regla que aprendería el agente: con una probabilidad de ejecución muy alta, no lanzar una agresiva sino una limitada (T07 ~02:52). Un alumno lo resumió como supervisado para la probabilidad y RL para aprender las reglas, y el profesor lo confirmó (T07 ~02:53). Incluso sin RL, supervisado + reglas ajustadas ya mejoraría el TWAP.

### 5.4 El benchmark "programación dinámica" y cuándo compensa el RL

(T07 ~04:09-04:16.) Contexto previo: las probabilidades de ejecución por nivel de precio las tienen los operadores en sus terminales; calcularlas y usarlas en el algoritmo son "proyectitos" que los bancos hacen internamente. Se pueden hacer como trabajo propio (lo mencionó como posible TFM *(dudoso)*) y luego añadir RL para ver si se consigue una regla mejor que la de un humano (T07 ~04:09-04:10).

A la pregunta de un alumno sobre cuándo usar redes/supervisado frente a RL: en banca el RL se usa poco y a menudo basta con supervisado. Ejemplo: 4-5 niveles de orden limitada (los niveles pasivos parecen estar en 7,10, 7,09…) con probabilidades de ejecución dadas por un supervisado (80, 60, 50 y 10 % para el nivel muy alejado), frente a la alternativa agresiva a 7,12, que según él es lo que hace un TWAP *(dudoso: la transcripción escribe "teagua")*, cuyo coste medio por agresividad suele ser ~**3 pb** sobre el precio medio del libro (T07 ~04:11-04:12). Con eso se calcula de forma determinista la utilidad esperada de cada acción (escenario favorable × probabilidad + escenario desfavorable × su probabilidad). Eso "es como programación dinámica": probabilidades de transición y recompensas conocidas (§7). Probablemente explica el 80-90 % de la varianza. Esa solución matemática es el **benchmark** que el RL, con características extra (frecuencia de trades, spread actual), debería superar. El ML clásico está infrautilizado y puede coexistir con reglas y otros sistemas.

**Nota (cálculo ilustrativo):** la transcripción dice algo como "2 por 100 de ahorro por 80 por 100, más 20 por 100 por 3 de financiación" (T07 ~04:12) *(dudoso: mezcla "por 100" y "3" sin unidad, así que las unidades no se pueden fijar con seguridad)*. Dos lecturas:
- (i) **Ahorro de 2 céntimos** (7,12 − 7,10, coherente con los niveles de precio dichos), ≈ 28 pb sobre un precio de ~7,11, frente a un coste de agresividad de ~3 pb si la pasiva no se ejecuta y hay que cruzar después. Con $p=0{,}8$: $\mathbb{E}[U]\approx0{,}8\cdot28-0{,}2\cdot3\approx+21{,}8$ pb. La pasiva domina salvo con $p$ muy baja o mucho riesgo de timing.
- (ii) **Ahorro de 2 pb**: $\mathbb{E}[U]=0{,}8\cdot2-0{,}2\cdot3=+1{,}0$ pb frente a −3 pb de cruzar ya. Con $p=0{,}5$: $0{,}5\cdot2-0{,}5\cdot3=-0{,}5$ pb, todavía mejor que −3 pb.

En ambas, la pasiva no puede salir peor que cruzar ya mientras su peor caso (no ejecutarse y cruzar después) cueste lo mismo que la agresiva. El modelo real debe añadir el riesgo de que el precio se mueva en contra mientras esperas (riesgo de timing, ver [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md)), que es lo que hace interesante el problema.

**A favor y en contra del RL** (T07 ~04:16-04:19): a favor, el ruido en etiquetas o decisiones, las interacciones con el entorno (de mercado o internas) y los escenarios infinitos generados por un simulador, donde no hay etiquetas posibles; si se dispone de un simulador de mercado con impacto realista, se pueden crear escenarios que nadie más ha visto. En contra, sus tres escollos: **recompensas, exploración y entorno**. Con RL se puede resolver casi cualquier problema de supervisado (también el del paraguas), y es más potente porque recoge interacciones que el supervisado no ve; si esos escollos no lo son y apetece probar, adelante. Citó la "tarta" de LeCun: el bizcocho es el no supervisado, el glaseado el supervisado y la guinda el RL. Alguien en el aula, probablemente un alumno *(dudoso: todo el audio está etiquetado igual)*, añadió que lo ve útil cuando hay un patrón histórico que se sabe no óptimo y una métrica clara de qué es mejor.

---

## 6. Objetivo, retorno descontado y γ

**Test de APP p.16-17**: "¿Cuál es el objetivo de un sistema de RL?" con cuatro opciones; en p.17 aparecen con check verde las tres primeras: maximizar la utilidad total, maximizar la suma de recompensas descontadas y minimizar el coste de oportunidad (no "ninguna de las anteriores"). En clase dio por buenas todas *(dudoso: la frase es "venga, todas, pero no", T06 ~00:38)* y explicó para qué sirve la tercera: a veces conviene no intentar optimizarlo todo sino **minimizar el coste de oportunidad comparando alternativas**; limpia el problema y centra en lo importante (T06 ~00:38-00:39). Añadió que, "si esto lo mantienes al 0, es exactamente lo mismo" (T06 ~00:39) *(dudoso: lectura probable: si el valor de la alternativa de referencia se fija en 0, minimizar el coste de oportunidad equivale a maximizar la utilidad)*. **Nota:** es exactamente el enfoque de la práctica, donde la recompensa natural es la diferencia frente a un benchmark (VWAP/TWAP), ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

**Fórmulas** (APP p.18, con el icono de un reloj):

$$G(\tau)=r_0+\gamma r_1+\gamma^2 r_2+\dots+\gamma^k r_k=\sum_{t=0}^{k}\gamma^t r_t$$

$G(\tau)$ es la **utilidad total** de la trayectoria (episodio), suma de recompensas descontadas; $\gamma$ es el factor de descuento (valor del tiempo), con la pregunta "¿γ = 0 / γ = 1?". Forma recursiva:

$$G_t=r_t+\gamma G_{t+1}\quad=\quad\text{recompensa inmediata}+\text{valor futuro}$$

donde el valor futuro es el valor presente (PV) de las recompensas futuras, o el PV de $S_{t+1}$.

**Nota (errata menor APP p.18):** la diapositiva escribe el último término como $\gamma^t r_t$ y el sumatorio hasta $k$, mezclando el índice mudo con el límite; lo correcto es $\gamma^k r_k$ como arriba.

Lo dicho en clase (T06 ~00:39-00:44):
- La utilidad es, por definición, la suma de recompensas descontadas del episodio; descontar es el valor temporal del dinero: no es lo mismo tener el dinero hoy que dentro de un año.
- **Taxi autónomo:** sin γ, los 50 € de la carrera de A a B valen lo mismo yendo por la calle más recta que dando la vuelta al mundo. γ introduce urgencia.
- **Cuándo γ = 1:** si no hay urgencia o el episodio tiene un número fijo de pasos (p. ej. 10 pasos, o el tres en raya: quieres ganar, no hace falta ganar en la segunda ronda), no hay que complicarse con γ; además la urgencia puede meter ruido y comportamientos extraños. Un alumno objetó que casi siempre importa cómo acaba (ganar la partida, ganar dinero a fin de año) y el profesor lo resumió así: entonces γ = 1, no descuentes; solo si quieres que algo ocurra deprisa tiene sentido.
- La recompensa inmediata no se descuenta (se recibe al tomar la acción); las siguientes, más cuanto más lejanas, y cuánto depende de la dimensión temporal del step (un segundo o un año) y de la urgencia de la operación.
- **Forma recursiva:** la fórmula larga no hace falta aprenderla, porque se simplifica con la recursiva (T06 ~00:42-00:43): basta con recompensa inmediata + valor futuro descontado. El primer término lo da el entorno; el segundo son creencias, muy imprecisas al principio, que convergen "jugando". Esta es la base de TD y Q-learning (§7-§9).

**De qué depende γ** (APP p.75-76, dos páginas iguales): del valor del tiempo, del signo de las recompensas, de si el problema es episódico o continuo, del número de steps del episodio y de si ese número es fijo o variable ("mucho cuidado con los que tienen una duración variable"). El diagrama muestra dos estados $S_0\leftrightarrow S_1$ con $r=1$ de ida y $r=2$ de vuelta: un problema continuo sin fin.

**Nota:** en ese ciclo, sin descuento el retorno diverge ($1+2+1+2+\dots\to\infty$); con $\gamma<1$ es finito: desde $S_0$, $G=\frac{1+2\gamma}{1-\gamma^2}$; con $\gamma=0{,}9$, $G=2{,}8/0{,}19\approx14{,}7$. Por eso γ = 1 solo es seguro en problemas episódicos. Con duración variable, el signo de las recompensas por paso decide si al agente le conviene acortar (negativas) o alargar (positivas) el episodio, incluso con γ = 1; γ < 1 añade un sesgo temporal: resta peso a las penalizaciones lejanas (incentiva aplazarlas) y a las recompensas lejanas (incentiva cobrarlas antes). Por eso la diapositiva pide "mucho cuidado" con la duración variable. En el otro extremo, γ = 0 (la otra mitad de la pregunta de APP p.18) da un agente miope que solo maximiza la recompensa inmediata: equivale a un problema de un paso como el del paraguas, abordable con supervisado (§5.1). La calibración de γ hacia atrás según el horizonte (cálculos de clase para episodios de cientos de pasos) está en [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md); el repaso formal, en [T2 00_fundamentos_mdp.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/00_fundamentos_mdp.md) §4.

---

## 7. Monte Carlo, programación dinámica y diferencias temporales

### 7.1 El test de APP p.19-20

"¿Cuál de estos hace referencia al método de temporal difference (TD)?" La p.20 asigna cada fórmula a su diagrama de *backup* (árboles de Sutton & Barto con nodos de estado blancos, de acción negros y terminales "T" en verde; el tramo que se usa para actualizar va sombreado en rojo):

| Método | Actualización | Diagrama |
|---|---|---|
| **MC** | $V_{k+1}(S_t)\leftarrow(1-\alpha)V_k(S_t)+\alpha\,G_t$ | Una trayectoria completa desde $s_t$ hasta un terminal |
| **TD** (check verde) | $V_{k+1}(S_t)\leftarrow(1-\alpha)V_k(S_t)+\alpha\big(r_t+\gamma V_k(S_{t+1})\big)$ | Una sola transición $s_t\to s_{t+1}$ |
| **PD** | $V(S_t)\leftarrow\mathbb{E}_\pi\big[R_{t+1}+\gamma V(S_{t+1})\big]$ | Un nivel, pero **todas** las ramas a la vez |

**Nota (errata APP p.19-20):** la diapositiva escribe $V(R_{t+1})$ dentro de la esperanza de PD; debe ser $V(S_{t+1})$.

Explicación oral (T06 ~00:44-00:51):
- Lo presentó como una de las muchas "dimensiones" del RL (el título de APP p.19-20 es "Dimensiones del Aprendizaje por Refuerzo"), la que está "más de moda" (T06 ~00:44).
- **Monte Carlo:** se juega un episodio completo y se promedia.
- **Programación dinámica:** desde el estado actual se consideran todas las opciones ponderadas por sus probabilidades ("recompensa ponderada por la esperanza"); converge muchísimo más rápido, pero exige conocer los estados siguientes y las probabilidades de transición. En ajedrez es inviable: la transición tras mover el caballo depende del rival (un día juegas contra uno y otro día contra otro), y aun con 4 posibilidades es difícil de calcular.
- **TD:** actualiza paso a paso sin esperar al final del episodio; con una exploración buena aprende el espacio, y como va actualizando parcialmente sus valores le guía hacia qué zonas conviene explorar.
- Alguien (por el contexto, un alumno) comentó que los motores de ajedrez usarían este último enfoque, el de programación dinámica *(dudoso: la transcripción está muy degradada en ~00:47:53-00:48:12)*. La respuesta: entre los dos extremos se pueden hacer cosas intermedias, como mirar varios estados siguientes entre los más probables, y qué partes del espacio se aprenden depende de la política de exploración (T06 ~00:48-00:49). Hay, pues, puntos intermedios entre una transición y el episodio completo (mirar varios estados hacia delante); recordó que DeepMind en su programa de Go miraba unos 16 estados hacia delante entre los más probables *(dudoso: la cifra y el sistema no están claros en la transcripción)*.
- Si se pudiera explorar todo, casi no harías RL: te irías a supervisado, porque habría un número finito de escenarios.
- Dijo que estas 2-3 diapositivas son menos importantes que las primeras: lo relevante es que de ahí salen los algoritmos famosos.

### 7.2 TD: aprender el MDP "por fuerza bruta"

APP p.21 (con el diagrama TD resaltado): en TD no conocemos el MDP pero, con una política de exploración adecuada, podemos llegar a "conocerlo". Se actualizan los valores con únicamente la transición siguiente (1) $V_k(S_{t+1})$ y la recompensa observada (2) $r_t$; el paréntesis $r_t+\gamma V_k(S_{t+1})$ hace el papel de $G_t$. Iterando sobre los mismos estados se converge a la misma solución con un cierto número de repeticiones. "Ahora bien, no queremos los valores de cada estado sino de cada estado-acción (q-valores) → Q-Learning."

En clase: el algoritmo aprende implícitamente las probabilidades (que al hacer X ocurra Y, que llueva o no) porque lo ha vivido mucho, y asigna valores en función de eso; se sustituye el conocimiento matemático del MDP (sentar a matemáticos a especificar escenarios y probabilidades) por "un poquito más de fuerza bruta" dejándolo jugar (T06 ~00:50-00:51).

### 7.3 Convergencia tabular: llegar "por suerte" también enseña

A la pregunta de un alumno de si, cuando se llega a la recompensa por pura aleatoriedad, la tabla Q no aprende nada útil (T06 ~03:13-03:16):
- Con **redes neuronales** hay que tener mucho cuidado con meter ruido, porque los pesos convergen hacia las últimas experiencias que se le pasan (ver catastrophic forgetting en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md)).
- En **tabular** (un diccionario) no pasa nada: la convergencia exige visitar todos los estados (infinitas veces) y, si el espacio está bien definido, converge.
- **Propagación hacia atrás:** la primera vez que se llega al objetivo, solo las celdas próximas reciben información positiva; luego, al pasar por las del siguiente nivel, la señal parcial registrada se propaga un poco más atrás, y así sucesivamente hasta que el entrenamiento se va "limpiando" y los valores convergen a los reales. En una matriz de 50×50 eso tarda bastante más.

### 7.4 N-step TD: más velocidad, más varianza (apéndice)

APP p.101 (diagramas de *backup* de 1-step TD(0), 2-step, 3-step, …, n-step e ∞-step = Monte Carlo):

$$G_{t:t+1}=R_{t+1}+\gamma V(S_{t+1}),\qquad G_{t:t+2}=R_{t+1}+\gamma R_{t+2}+\gamma^2V(S_{t+2}),\qquad G_t=R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{T-t-1}R_T$$

- **Ventaja** (lo que la diapositiva llama "TD(1)"): más velocidad de convergencia. En la cadena $S_0\to S_1\to S_2\to S_3$ (recompensas $r_1,r_2,r_3$), con TD(0) solo se ve la información del estado siguiente y hacen falta **3 vueltas** para que la información del final llegue a $S_0$; mirando dos estados hacia delante, **2 vueltas**. ¿Por qué no TD(30)?
- **Desventaja** (APP p.102, árbol binario $s_0\to s_1,s_2\to s_3\dots s_6$ con acciones $a_0/a_1$): aumenta la varianza, porque cada paso adicional encadena una acción y una transición aleatorias más. MC no está sesgado pero tiene mucha varianza; TD está sesgado (usa su propia estimación) pero tiene poca. Los modelos de RL ya sufren inestabilidad y más varianza dificultaría el entrenamiento.

**Nota (notación):** "TD(1)" en la diapositiva significa *2-step TD* (mirar un paso más). En Sutton & Barto, TD(λ) con λ = 1 equivale a Monte Carlo, así que conviene no usar "TD(1)" en el informe con el sentido de la diapositiva. "3 vueltas": en el episodio 1 solo $S_2$ aprende $r_3$; en el 2 la señal pasa a $S_1$; en el 3 a $S_0$. Con 2-step, $S_1$ y $S_2$ ven $r_3$ en el episodio 1 y $S_0$ en el 2.

Profundización (TD(0), DP, MC, convergencia) en [T2 01_dp_y_montecarlo.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/01_dp_y_montecarlo.md) y [T2 02_td_sarsa_qlearning.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md) §1-§4.

---

## 8. El step size α (lo que se dijo en el repaso)

**Idea.** La actualización de un "diccionario" da un peso α a lo nuevo y $1-\alpha$ a lo antiguo (T06 ~00:52):

$$\text{nuevo}=(1-\alpha)\cdot\text{creencia}+\alpha\cdot\text{observación}$$

Ejemplo oral: creía que coger el paraguas valía 100 y recibo 20; con α = 5 %, nuevo valor $=100\cdot95\%+20\cdot5\%$ (T06 ~00:52). **Nota:** $=95+1=96$.

**Ruta del Retiro** (T06 ~03:36-03:38): hay dos rutas (por delante y por detrás del Retiro); creo que la de delante es mejor, un día exploro la de detrás y tardo 30 minutos más por un accidente. Con **α grande** doy tanto peso a ese día que nunca vuelvo a esa ruta (salvo que me toque explorar), aunque los accidentes ocurren un 2 % de las veces y, sin accidente, esa ruta ahorra minutos; para saberlo hay que haberla recorrido varias veces. Con **α muy pequeño**, un hallazgo muy bueno pasa desapercibido: si la red ya ha convergido y un día descubres un túnel que ahorra 2 horas, apenas mueve tu estimación. Lo usa a diario fuera del RL: al lanzar campañas o experimentos no se da 100 % de peso a lo último que ocurrió, se hacen muchos experimentos pequeños.

**No estacionariedad** (T06 ~03:39-03:42): α permite, en esencia, olvidar el pasado y quedarse con lo nuevo. Si el problema fuera estático bastaría una media simple (pesar a los alumnos un millón de veces al azar y promediar). Pero el peso medio de la clase cambia (alguien sale, otro se come el bocadillo, entra gente). Con una creencia de 80 kg y α = 1 %, una observación de 72 kg da "79,8" según el profesor; si entran culturistas de 200 kg, α hace que la media se vaya adaptando (dice "a unos 85 kg"). Lo mismo con las políticas: las iniciales están desactualizadas (se tomaban acciones sin sentido) y se quieren olvidar, no promediar.

**Nota (corrección del cálculo):** con α = 1 %, $0{,}99\cdot80+0{,}01\cdot72=79{,}92$ kg, no 79,8 (79,8 correspondería a α = 2,5 %). La idea no cambia. Con α constante, una observación de hace $k$ pasos pesa $\alpha(1-\alpha)^k$: la "memoria efectiva" es de unas $1/\alpha$ observaciones (100 con α = 1 %). Con $\alpha_n=1/n$ se recupera exactamente la media simple, que es lo que la diapositiva quiere decir (con la errata corregida, ver abajo).

**APP p.74:** α muy grande da mucha importancia a un hecho poco probable; menor α recoge más casuística; muy pequeño puede quedar atascado en mínimos locales. Estrategia clásica: empezar alto e ir bajando conforme mejora el aprendizaje (*alpha decay*), aunque con optimizadores (Adam, RMSProp) "nos despreocupamos un poco". La *novelty* también aplica a α: "Más grande en estados nuevos. Lo correcto debería ser proporcional al N veces visitado".

**Nota (errata APP p.74):** la diapositiva dice "proporcional al N veces visitado"; para que α sea mayor en estados nuevos, como dice la línea anterior, debe ser proporcional a $1/N$ (con $\alpha_n=1/N$ se obtiene la media simple).

Al empezar ese bloque el profesor dijo que ahora, con redes, pueden despreocuparse algo de α, pero que es importante (T06 ~03:35). El resto de hiperparámetros de aprendizaje está en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

---

## 9. Q-learning

### 9.1 La fórmula (APP p.22)

$$Q_{k+1}(s_t,a_t)\leftarrow\underbrace{(1-\alpha)\,Q_k(s_t,a_t)}_{\text{valor antiguo (creencia actual)}}+\alpha\,\underbrace{\big(r_t+\gamma\,\max_a Q_k(s_{t+1},a)\big)}_{\text{target / valor aprendido}}$$

Partes rotuladas en la diapositiva: valor antiguo (creencia actual), *step size* (learning rate), nuevo valor, recompensa, factor de descuento, estimación del valor óptimo futuro ($\max_a Q_k(s_{t+1},a)=V(s_{t+1})$), target. Es un algoritmo **TD** y **off-policy**: la política de exploración (random o ε-greedy) es distinta (≠) de la política de aprendizaje (greedy).

En clase (T06 ~00:51-00:54): Q-learning es "el más famoso", del que salen muchos otros; la tabla es literalmente un diccionario. Es off-policy porque se aprende lo que maximiza en cada caso (política greedy) mientras se actúa con otra, típicamente ε-greedy: con ε = 5 %, un 5 % de las veces se actúa al azar y un 95 % se hace lo que se cree mejor.

**Nota:** la forma de la diapositiva es equivalente a la de Sutton & Barto, $Q\leftarrow Q+\alpha[\text{target}-Q]$; el término entre corchetes es el error TD, cuyo cuadrado devuelve `learn` como "loss" en los notebooks (RLM TD-q-learning, celda 2: `return (target - qsa) ** 2`). Si el siguiente estado es terminal, el target es solo $r$ (misma celda: `if done: target = r`), equivalente a $Q(\text{terminal},\cdot)=0$ del pseudocódigo.

### 9.2 El ejemplo numérico ("la slide más pesada de toda la clase")

Así lo presentó (T06 ~00:52; ~00:59 "la más complicada de toda la clase"). Diagrama de APP p.23-24: el estado $S_0$ (círculo verde) tiene dos acciones (rombos azules):
- $a_0$ lleva a $S_1$ con $r=10$; $V(S_1)=50$.
- $a_1$ lleva a $S_2$ con $r=15$ ($V(S_2)=20$) **o** a $S_3$ con $r=5$ ($V(S_3)=100$); las probabilidades no se conocen.
- De cada estado salen flechas punteadas: el juego continúa.

**Situación tras 7 partidas** ($k=7$, APP p.23): $Q(s_0,a_0)=45$, $Q(s_0,a_1)=40$, de modo que el valor de $S_0$ (greedy) es 45. Los valores de los estados siguientes son el máximo de sus Q: $V(s_1)=\max_aQ_{k=7}(s_1,a)=50$, $V(s_2)=20$, $V(s_3)=100$. El profesor lo explicó así: $S_0$ vale 45 porque su diccionario da 45 a $a_0$ y 40 a $a_1$ y, con la política de coger lo que maximiza, el estado vale 45; $S_1$ vale 50 porque su mejor acción, según el diccionario, da 50 (T06 ~00:54-00:55).

**Iteración 8** (APP p.24): $\alpha=0{,}1$, $\gamma=1$ ("para simplificar, sin descuento temporal"). Aunque se cree que $a_0$ es la mejor, se está aprendiendo y las creencias iniciales no son de fiar, así que esta vez se explora y se toma $a_1$; resulta que lleva a $S_3$ (el caso bueno):

$$\text{target}=5+1\cdot100=105,\qquad Q_{k=8}(s_0,a_1)=(1-0{,}1)\cdot40+0{,}1\cdot105=36+10{,}5=46{,}5$$

Ahora $Q(s_0,a_1)=46{,}5>45=Q(s_0,a_0)$: tras una sola iteración ha cambiado la creencia de cuál es la acción maximizadora y $V(S_0)=46{,}5$ (T06 ~00:55-00:57).

**Cómo sigue** (T06 ~00:57-00:58): desde $S_3$ se hace lo mismo y se actualiza su valor (quizá a 97). En la partida 9 se empieza otra vez desde el principio; tal vez $a_1$ lleve esta vez a $S_2$ (valor 20), que es un mal resultado, y el valor vuelva a bajar ("a lo mejor 39"). Los estados por los que no se pasa no se actualizan (el 97 se queda igual).

**Notas (cálculos propios sobre el ejemplo):**
- Si en la partida 9 $a_1$ lleva a $S_2$: target $=15+20=35$ y $Q(s_0,a_1)=0{,}9\cdot46{,}5+0{,}1\cdot35=45{,}35$, no 39; el "39" fue una cifra ilustrativa. Con 45,35 la acción maximizadora sigue siendo $a_1$, por muy poco (45,35 frente a 45).
- $Q(s_0,a_0)=45$ está por debajo de su target $10+50=60$: los valores tras 7 partidas no han convergido. Si $a_0$ se repitiera, se acercaría a 60.
- Con α constante, $Q(s_0,a_1)$ oscila alrededor del target medio $p\cdot105+(1-p)\cdot35=35+70p$, donde $p$ es la probabilidad (desconocida) de ir a $S_3$. $a_1$ es mejor que $a_0$ si $35+70p>60$, es decir, si $p>25/70\approx0{,}36$. Q-learning "descubre" esa probabilidad sin conocerla, que es el sentido de §7.2.

**Pseudocódigo** (APP p.25, reproduce el recuadro de Sutton & Barto "Q-learning (off-policy TD control) for estimating π ≈ π*"): parámetros α ∈ (0,1] y ε > 0 pequeño; inicializar $Q(s,a)$ arbitrariamente salvo $Q(\text{terminal},\cdot)=0$; por cada episodio: inicializar $S$; por cada paso: elegir $A$ desde $S$ con una política derivada de $Q$ (p. ej. ε-greedy), tomar $A$, observar $R, S'$, $Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma\max_aQ(S',a)-Q(S,A)]$, $S\leftarrow S'$; hasta que $S$ sea terminal. Debajo repite la fórmula en la forma de la clase.

**En el código de clase** (RLM TD-q-learning, celdas 2 y 6-7): Q es un `defaultdict` que inicializa cada estado nuevo con valores aleatorios pequeños (`randn * 0.001`); `act` es ε-greedy; `act_no_explore` es greedy (para evaluar); en CartPole se usa γ = 0,995, α = 0,1 y ε de 1 a 0,05 lineal en el 90 % de 10 000 episodios. Detalle en [09_material_rl_notebooks.md](09_material_rl_notebooks.md).

**Comentario final del profesor** (T06 ~00:58-00:59): definir las recompensas es un escollo y el entorno es otro; si se resuelven, se tiene algo muy potente. Contó que, al preguntar a ChatGPT qué recompensa poner en bolsa, propone algo muy básico (el valor siguiente menos el anterior).

**Demo del material rl-material** (T06 ~00:59-01:07): tras el ejemplo numérico enseñó el material rl-material (subido al Drive) y cómo ejecutarlo: un contenedor Docker abierto desde VS Code con la extensión Dev Containers (la primera vez tarda, p. ej. 10 min, por descargar la imagen de TensorFlow, de unos 2 GB) o, como alternativa, un entorno propio con `pip install -e .`. Avisó de que en un ~10 % de los alumnos algo falla y hay que verlo caso por caso. Ejecutó el notebook de Q-learning *(dudoso: por el contexto, TD-q-learning sobre CartPole)* con renderizado del entorno: "es muy sencillo". Lo deja para que lo usen, aunque hay otros paquetes que quizá estén mejor (T06 ~01:05-01:07). Instalación en [notebook_partida.md](notebook_partida.md); el notebook, en [09_material_rl_notebooks.md](09_material_rl_notebooks.md).

Profundización (on/off-policy, convergencia, inicialización) en [T2 02_td_sarsa_qlearning.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md) §5 y §7, y [T1 02_td_sarsa_qlearning.md](../raw/referencias/rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md) §7.

---

## 10. SARSA y el cliff walking

APP p.26:

$$Q_{k+1}(s_t,a_t)\leftarrow(1-\alpha)\,Q_k(s_t,a_t)+\alpha\big(r_t+\gamma\,Q_k(s_{t+1},a_{t+1})\big)$$

frente a Q-learning, que usa $\gamma\max_aQ(s_{t+1},a)$. Es TD y **on-policy**.

**Nota (errata APP p.26):** la diapositiva escribe $Q_{k+1}(s_{t+1},a_{t+1})$ dentro del target (y también $Q_{k+1}$ en el recordatorio de Q-learning); debe ser $Q_k$, el valor actual de la tabla, como en p.22 y en el código (RLM cliff-walking, celda 13: `target = r + self.gamma * self.Qvalues[s1][a1]`).

Explicación (T06 ~01:07-01:10):
- El nombre viene de los datos que necesita la actualización. Q-learning recibe 4: estado, acción, recompensa inmediata y estado siguiente. SARSA necesita además la **siguiente acción realmente tomada**: $(s,a,r,s',a')$.
- En Q-learning el valor del siguiente estado es el de la acción que se cree maximizadora, por eso no necesita $a'$; SARSA deja dar un paso más y toma el valor de la acción que de verdad se dio.
- Con una política 95 % greedy / 5 % aleatoria ambos targets son muy parecidos durante el entrenamiento, pero en el 5 % de casos difieren. Es on-policy porque aprende y ejecuta con la política actual.

**Cliff walking** (figura de APP p.26, de Sutton & Barto): rejilla con inicio S y meta G en la fila inferior separados por "The Cliff"; cada paso da $R=-1$ y caer al acantilado $R=-100$ (vuelta al inicio). La flecha roja *Optimal path* va pegada al acantilado; la azul *Safer path*, por la fila de arriba. Q-learning aprende el camino óptimo (siempre lo que maximiza, moverse a la derecha junto al borde); SARSA, como su política de aprendizaje incluye la exploración y cerca del borde explorar significa caerse (−100), aprende a alejarse un poco (T06 ~01:10-01:11). Hizo la **demo** con el notebook de cliff walking: primero Q-learning, que parecía haber aprendido hacia el episodio 150 *(dudoso)*, y luego SARSA, donde el cambio de código es mínimo (indexar también por la nueva acción) y la política final es muy distinta (T06 ~01:11-01:12). Mensaje: importa menos aprender SARSA que entender que de modificaciones muy pequeñas salen algoritmos distintos (T06 ~01:07).

**Nota:** en RLM cliff-walking (celdas 9 y 16) ambos agentes usan γ = 0,99, α = 0,1 y ε de 1 a 0,05 en la primera mitad de 300 episodios; el bucle de SARSA elige `a1 = agent.act(s1)` antes de `learn` y luego `a = a1`, de modo que la acción con la que aprende es la que ejecuta. La comparación detallada está en [T2 02_td_sarsa_qlearning.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md) §8.

---

## 11. Estabilidad y Expected SARSA

### 11.1 La palabra clave: estabilidad

(T06 ~01:12-01:14.) La ventaja que se busca con SARSA es más **estabilidad**, "palabra muy, muy importante" en RL:
- Los entrenamientos son aleatorios: dependen de los caminos recorridos y de la exploración. Se dice que, con todo perfectamente configurado, de 10 ejecuciones quizá 3 fallen por completo.
- La causa: reciben mucho ruido y señales confusas. En cliff walking, al recibir −100 cada vez que explora cerca del borde, es difícil saber que ese es el camino óptimo, porque llegan constantemente señales contradictorias.
- Muchas mejoras de los papers van en la dirección de dar más estabilidad durante el entrenamiento.
- SARSA da un aprendizaje más suave y convergente, aunque no siempre óptimo. Con un γ alto quizá se acercaría más al precipicio *(dudoso: la frase sobre γ es confusa; lo que se entiende es que el camino aprendido depende también de γ)*.

### 11.2 Expected SARSA

APP p.26-28 y T06 ~01:15-01:16: en lugar del máximo (Q-learning) o de la acción realmente tomada (SARSA), se usa el **valor esperado** del siguiente estado según la política actual:

$$Q_{k+1}(s_t,a_t)\leftarrow(1-\alpha)Q_k(s_t,a_t)+\alpha\Big(r_t+\gamma\sum_a\pi(a|s_{t+1})\,Q_k(s_{t+1},a)\Big)$$

- Similar en código a Q-learning; tiene en cuenta la probabilidad de cada acción dada la política; lleva por un camino más seguro (como SARSA); computacionalmente más costoso que SARSA pero reduce la varianza; suele funcionar mejor que SARSA con el mismo número de episodios (APP p.27).
- Ejemplo oral: si la política actual es 50 % greedy y 50 % aleatoria, el valor del siguiente estado es 50 % × el de la acción maximizadora + 50 % × la media del resto, para todavía más estabilidad (T06 ~01:15).
- Se podría usar la mediana en vez de la media, probablemente con resultados similares; y si tienen otras ideas, que las prueben. Muchos algoritmos con nombres distintos son copias con un parámetro cambiado o una red duplicada (T06 ~01:16).

**Nota (matiz y código):** con ε-greedy la parte aleatoria elige uniformemente entre **todas** las acciones (incluida la greedy), así que el valor esperado es $(1-\varepsilon)\max_aQ+\varepsilon\cdot\text{media}_a Q$, con la media de todas, no "del resto". Es exactamente lo que implementa RLM TD-expected-sarsa (celda 2: `exp_value = (1 - self.epsilon) * max_value + self.epsilon * rnd_value`, con el comentario "Why not e.g., median?", que es la pregunta que lanzó en clase).

### 11.3 Resumen comparativo (APP p.28)

| | Valor futuro que usa | Política | En la práctica |
|---|---|---|---|
| **Q-learning** | El máximo: $\max_aQ(s',a)$ | Off-policy | Aprende el camino óptimo, más arriesgado |
| **SARSA** | La acción que toma: $Q(s',a')$ | On-policy | Camino más seguro, más varianza |
| **Expected SARSA** | La media: $\sum_a\pi(a|s')Q(s',a)$ | On-policy | Menos varianza, más cálculo por paso |

"Expected SARSA suele aprender con menos episodios que SARSA" (APP p.28).

---

## 12. Double Q-learning

**Por qué** (APP p.29): con una sola tabla, el max sobreestima los valores (**sesgo de maximización**). Se usan dos matrices de Q-valores, $Q_1$ y $Q_2$.

**Gráfico de APP p.29** (ejemplo de Sutton & Barto): MDP con estados A y B; desde A, "right" termina con 0 y "left" va a B, desde donde varias acciones terminan con recompensa $\mathcal{N}(-0{,}1;1)$. Eje y: % de acciones "left" desde A; eje x: episodios (1-300). La curva roja de Q-learning sube a ~90 % en los primeros episodios y baja lentamente hasta acercarse al 5 % hacia el episodio 300; la verde de Double Q-learning empieza en ~50 % y cae rápido hacia la línea discontinua "optimal" del 5 %.

**Ejemplo oral** (T06 ~01:17-01:18): problema "muy sencillo": a la derecha recibes 0; a la izquierda, una normal de media −0,1 y desviación 1. Si empiezas mal (una primera muestra de −0,7) quedas sesgado y te cuesta salir de esa creencia; si empiezas bien (0,4) empiezas a creértelo más. Incluso un problema tan sencillo tiene un problema de inicialización: si empiezas mal, te atascas.

**Nota:** el problema del ejemplo de Sutton & Barto es el optimismo: como el máximo de varias estimaciones ruidosas de media −0,1 suele ser positivo, Q-learning cree que ir a la izquierda vale más que 0 y la elige casi siempre al principio; el ejemplo oral del profesor (primera muestra −0,7 o +0,4) describe la sensibilidad a las primeras muestras, que es la misma raíz (estimaciones ruidosas usadas para elegir y evaluar). La línea "optimal" del 5 % no es 0 % porque con ε = 0,1 y dos acciones en A, la exploración elige "left" un 5 % de las veces.

**Funcionamiento** (APP p.29), si toca actualizar $Q_1$:
1. Dado $S_0$, se actúa sumando $Q_1+Q_2$ → $a_0$, se llega a $S_1$.
2. Argmax en $S_1$ con $Q_1$ → $a_1$.
3. Se evalúa $a_1$ con $Q_2$ → target.
4. Se actualiza $Q_1$.
5. Los papeles se intercambian con probabilidad 50 %.

**Pseudocódigo** (APP p.30, recuadro de Sutton & Barto "Double Q-learning, for estimating $Q_1\approx Q_2\approx q_*$"): elegir $A$ ε-greedy sobre $Q_1+Q_2$; tomar $A$, observar $R,S'$; con probabilidad 0,5:

$$Q_1(S,A)\leftarrow Q_1(S,A)+\alpha\Big(R+\gamma\,Q_2\big(S',\arg\max_aQ_1(S',a)\big)-Q_1(S,A)\Big)$$

y si no, lo simétrico intercambiando $Q_1$ y $Q_2$.

En clase (T06 ~01:18-01:19): Double Q-learning tiene dos diccionarios y usa las creencias de uno para ponderar las del otro, reduciendo el sesgo. Muchas técnicas crean copias que se pasan los pesos cada cierto tiempo (ver red objetivo y Double DQN en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md)), todo en busca de estabilidad, y no solo en el algoritmo: todo el diseño del sistema de RL debe buscarla. En supervisado se vigila no sobreajustar; en RL hay que "usar lo que funcione". Cierre del bloque: pueden olvidarse de los nombres de los algoritmos; lo que más le importa es que se queden con la palabra **estabilidad** (T06 ~01:19).

**Nota:** RLM TD-double-q-learning (celda 4) implementa exactamente esto: `act_no_explore` usa `argmax(Q1[s] + Q2[s])`; en `learn` se sortea qué tabla se actualiza, se elige `max_a` con la tabla que se actualiza y se evalúa con la otra.

---

## 13. Métodos de valor frente a métodos de política, y qué agentes se proponen

**Idea.** Hay métodos **basados en valor**, que ajustan directamente los valores (Q) y de los que se deriva la política con el argmax (la base de la clase), y métodos **basados en política**, que ajustan las probabilidades de cada acción (coger o no el paraguas) (T06 ~00:16).

| | Basados en valor | Basados en política |
|---|---|---|
| Qué se aprende | $Q(s,a)$ (o $V$) | $\pi(a|s)$ directamente |
| Política | Derivada: $\arg\max_aQ(s,a)$ (+ ε para explorar) | Explícita, normalmente estocástica |
| Algoritmos citados en el taller | Q-learning, SARSA, Expected SARSA, DQL, DDQL, Dueling, aproximación lineal | Cross-entropy; A2C y A3C (actor-crítico, combinan ambos) |

Lista de agentes posibles para el caso de ejecución (CASOS p.45): Q-learning, SARSA, Expected SARSA, DQL, DDQL, Dueling, A2C, A3C, aproximación lineal y cross-entropy, junto a las preguntas sobre el buffer (tamaño, si se llena entero, inicialización con bróker, algoritmo tradicional o acciones aleatorias, con pesos no necesariamente iguales), que se tratan en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) y [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

Lo dicho para elegir (T07 ~00:25): ninguno predomina; si necesitas acciones continuas descartas la mitad; se pueden combinar.

**Nota:** "descartar la mitad" se refiere a que los métodos de valor del taller (tabulares y familia DQN) necesitan acciones discretas para el max/argmax; con acciones continuas (p. ej. "precio medio + X pbs", CASOS p.46) habría que ir a métodos de política o actor-crítico. La práctica usa acciones discretas y un Double DQN, así que este taller se queda en métodos de valor (ver [08_entorno_best_execution.md](08_entorno_best_execution.md)). Teoría de los métodos de política en [T2 05_policy_gradient.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/05_policy_gradient.md) y [T2 06_actor_critico_y_a3c.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/06_actor_critico_y_a3c.md).

**De tabular a aproximación funcional.** Todo lo anterior es aprendizaje tabular: un diccionario con estados y valores finitos que se cambian "a mano". Con espacios infinitos (un coche autónomo) no se puede entrenar ni guardar, y dos estados casi iguales pueden tener valores y acciones muy distintos; por eso se pasa a aproximar los valores con un modelo cuyos pesos se ajustan (T06 ~01:19-01:21; APP p.31). Se desarrolla en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md).

---

## 14. Estado del campo

**Interés y madurez** (T06 ~00:09-00:10): RL es de las áreas que más interés reciben; parte del éxito de la IA generativa viene de una capa de RL que indica si una respuesta es buena o mala. Es un campo "verde", con muchísimos papers y gente probando de todo, y con más potencial ahora: antes el gran escollo era crear entornos para entrenar al agente y con la IA generativa eso es más factible; ya hay empresas estudiándolo. Aviso: ideas que funcionan muy bien en un problema pueden no funcionar en absoluto en otro distinto o incluso similar.

**Papers de RL** (APP p.47, slide con la que abrió la Segunda Parte): número de papers de DeepMind por año (fuente: su web de research filtrando por año; dejó de publicar la serie después de 2020):

| Año | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|---|---|---|
| Papers | 1 | **4** (flecha) | 16 | 52 | 76 | 127 | 206 | 234 | 164 |

A la derecha: "Hoy sigue creciendo": de ≈130 a ≈1.200 papers aceptados entre 2017 y 2025 en 5 grandes conferencias de IA (fuente citada: *Topical Phase Transitions in Artificial Intelligence Research*, arXiv 2026, fig. 9(a), valores aproximados). Debajo: "Ideas de lo más variopintas" (2015-17: Double DQN, sparse rewards → hoy: LLMs, DeepSeek-R1); "RL funciona a veces sin saber muy bien por qué → entender bien los conceptos básicos"; "En ocasiones → hay que hacerlo funcionar".

Lo dicho (T06 ~02:07-02:10): RL es antiguo, pero con las redes neuronales y convolucionales, hacia 2010 (T06 ~02:07), se vio que resolvía juegos como Breakout, cuando hasta entonces todo era Q-learning tabular para casos limitados; a partir de ahí explotó la comunidad. **Nota:** el salto suele fecharse en 2013 (DQN en Atari), que es el año que la diapositiva señala con una flecha (APP p.47; hay otra flecha tras los puntos suspensivos de 2020, que indica que la serie sigue); probablemente se refería a eso. DeepMind, que era quien llevaba la vanguardia, publicaba esos papers; la serie se corta en 2020 porque a partir de ahí su web ya no permite filtrarlos por año (APP p.47, nota 1), y para 2017-2025 usa otra fuente (las grandes conferencias de IA) (T06 ~02:08). Hasta hace poco los papers trataban lo visto en clase (Double DQN…); hoy se centran más en RL aplicado a LLMs. Los papers son muy específicos ("para este juego no funciona nada, pero con esta tontería funciona"), con mucho *storytelling* a posteriori: la explicación de por qué SARSA no cae por el precipicio es, en parte, storytelling; en la práctica se prueba una recompensa, un γ, se añade una copia de la red, se cambia el nombre y "funciona mejor". Anima a probar ideas propias en entornos y algoritmos: no es un campo cerrado como las convolucionales, y lo que funciona para una cosa puede funcionar fatal para otra. Por eso insiste en los principios básicos.

**Entornos con IA generativa (2026)** (APP p.97, "¿Dónde estamos?"): el cuello de botella es generar millones de experiencias realistas en poco tiempo. Tres líneas: simulación en GPU (miles de entornos en paralelo en una sola GPU: Isaac Lab, MuJoCo MJX); *world models* (la IA genera mundos interactivos en tiempo real: Genie 3 → Genie 4, 2026); mercados sintéticos (IA que genera órdenes como entorno de trading: MarS, Microsoft, 2025). Advertencia: el simulador también se equivoca, hay que validar con datos reales antes de producción.

**Divulgación del segundo día** (T07 ~04:18-04:27):
- Atribuyó al fundador de DeepMind *(dudoso: el nombre no se entiende en la transcripción)* la idea de que el futuro de la IA estaría en el RL, limitado por lo difícil que es simular entornos fuera de los juegos (una carretera con rotondas para un coche autónomo). Con ChatGPT, su equipo se reorientó a LLMs, y la combinación generativo + RL (feedback humano) sería el camino.
- LLMs (Gemini, Claude) jugando a Pokémon Rojo Fuego: interpretan el contexto y deciden sin RL, aunque todavía se pierden; la premisa es combinar ese contexto con feedback de recompensa.
- Los juegos de Atari son el banco de pruebas de los papers porque están muy controlados: frente a Pokémon, donde hay muchas acciones (atacar, huir, capturar, usar objetos…), en Breakout solo se mueve la plataforma a izquierda o derecha, y hay simulador, así que se pueden probar variaciones de la recompensa de forma muy controlada (T07 ~04:23-04:24).
- Vídeo de un agente que aprende a aparcar: le cuesta 15.000-20.000 repeticiones, y hacia las 50.000 deja de aparcar bien (inestabilidad; importancia de que el entrenamiento sea eficiente). Generaliza al empezar desde posiciones distintas, pero no ante obstáculos nuevos (un carrito de la compra). Si a un LLM le pasas la captura con el carrito delante, entiende que el coche no puede seguir recto porque hay un obstáculo: eso es "entender contextos". Los LLMs entienden contextos y empiezan a ser buenos generando entornos y situaciones, pero todavía no traducen bien eso en acciones (T07 ~04:26-04:27).
- Él mismo aclaró que esto viene de vídeos divulgativos y que no es experto en ello.

Más contexto histórico (DQN, AlphaGo, Atari) en [T2 07_panorama_drl.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/07_panorama_drl.md) §3 y §6; los papers de la Segunda Parte de APP, en [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md).

---

## 15. Qué llevarse a la práctica

**Nota (síntesis propia de lo anterior; la hoja de ruta completa está en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md)):**

| Principio de clase | Fuente | Consecuencia para la práctica de ejecución |
|---|---|---|
| El estado debe contener la información que decide (Markov) y estar normalizado | APP p.9-11; T06 ~00:23-00:34, ~02:11-02:19 | Justificar cada característica "poniéndose en el lugar del operador"; si falta la clave (tiempo/volumen restante, liquidez), no se aprende con ninguna red. Ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md) |
| RL se justifica por impacto e interacción | APP p.14-15; T06 ~00:34-00:38 | La ejecución de órdenes grandes es el ejemplo canónico de RL en la tabla de APP p.15; conviene decirlo en el informe |
| Primero el benchmark sencillo | T07 ~04:10-04:16; APP p.12 | Comparar siempre frente a TWAP (y, si se puede, una regla o cálculo de utilidad esperada) |
| Combinar RL, supervisado y reglas | T07 ~00:23-00:26, ~02:51-02:54 | Es legítimo forzar acciones con reglas o construir características por fuera |
| γ = 1 si el episodio tiene duración fija y no hay urgencia | T06 ~00:40-00:42; APP p.75 | Episodios de duración fija → γ cercano a 1; cuidado si la duración varía. Ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md) |
| Off-policy aprende el óptimo; on-policy, uno más seguro | APP p.26-28 | El DDQN del notebook es off-policy: su target usa el argmax de la red principal evaluado con la red objetivo, no la acción realmente tomada (BE src/agents/dqn.py, learn) |
| Double: separar quién elige y quién evalúa | APP p.29-30 | Es la base del Double DQN de [08_entorno_best_execution.md](08_entorno_best_execution.md) |
| Estabilidad por encima del nombre del algoritmo; ~3 de 10 entrenamientos pueden fallar | T06 ~01:13, ~01:19 | Varias semillas, checkpoints y evaluación greedy periódica. Ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) |
| Evaluar con métricas de comportamiento, no solo recompensas | T07 ~03:52 | Mostrar perfiles de ejecución, no solo la curva de recompensa |

---

## 16. Erratas y dudas detectadas

| Dónde | Qué dice | Corrección / lectura razonada |
|---|---|---|
| APP p.18 | $G(\tau)=\dots+\gamma^t r_t=\sum_{t=0}^{k}\gamma^t r_t$ | El último término es $\gamma^k r_k$ (§6) |
| APP p.19-20 | PD: $\mathbb{E}_\pi[R_{t+1}+\gamma V(R_{t+1})]$ | Debe ser $V(S_{t+1})$ (§7.1) |
| APP p.26 | SARSA con $Q_{k+1}(s_{t+1},a_{t+1})$ en el target | Debe ser $Q_k$, como en p.22 y en RLM cliff-walking (§10) |
| APP p.101 | "TD(1)" = mirar dos estados | Es *2-step TD*; en la literatura TD(1) ≈ MC (§7.4) |
| CASOS p.50 | Columna $S_1$ de la política toda a 0 | Con la tabla de p.49, $\pi(S_1)=a_3$ (Q = 5) (§3.2) |
| T06 ~03:41 | 80 kg, α = 1 %, observación 72 → 79,8 | Es 79,92; 79,8 sería α = 2,5 % (§8) |
| T06 ~00:58 | Partida 9 hacia $S_2$ → "a lo mejor 39" | Con los datos del ejemplo, 45,35 (§9.2) |
| T06 ~01:15 | Expected SARSA: "50 % máximo + 50 % media del resto" | Con ε-greedy, la parte aleatoria promedia todas las acciones (§11.2) |
| APP p.74 | α "proporcional al N veces visitado" | Para que α sea mayor en estados nuevos debe ser proporcional a $1/N$ (§8) |
| T06 ~02:07 | Salto del RL con redes "en 2010" | Nota: probablemente 2013, el año que la diapositiva señala (§14) |
| T06 ~00:48 | DeepMind miraba "16 estados" hacia delante en Go | *(dudoso)* cifra no verificable con el material |
| T06 ~02:19 | "no paséis a soluciones que sean espectaculares" | Lectura literal coherente: no saltar a soluciones complejas; la alternativa "espectacular" = "tabular" no encaja con la segunda mitad (§4.4) |
| T06 ~00:38 | Respuesta al test del objetivo | *(dudoso)* la diapositiva marca tres opciones correctas (§6) |
| T07 ~04:12 | Cálculo de utilidad esperada de pasiva vs agresiva ("2 por 100 de ahorro…") | *(dudoso)* unidades no fijables; con los niveles 7,10/7,12 el ahorro serían 2 céntimos (≈28 pb), no 2 pb; se dan ambas lecturas (§5.4) |
