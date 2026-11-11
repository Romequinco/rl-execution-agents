# Aproximación funcional y Deep Q-Learning

> **Fuentes:**
> - **APP** (`docs/raw/MIAX14/applying_RL_MIAX.pdf`, "Applying Reinforcement Learning", 103 p.): bloque "Aproximación Funcional" a "Resumen: Mejoras de DQN" (p.31-45), apertura de la Segunda Parte y papers de DeepMind (p.46-48), Experience Replay (p.77-82), Entrenamiento (p.87-92) y apéndice N-step TD (p.101-102). Las páginas con gráficos se han leído como imagen.
> - **CASOS** (`docs/raw/MIAX14/casos_practicos.pdf`): p.45-50 (agentes, buffer y tabla Q del caso de ejecución).
> - **T06** (`docs/raw/26 11 06.pdf`, transcripción automática de la clase del 06/11/2026, profesor Jesús Sanz): ~01:18-01:41 (tabular → funcional, lineal, replay, fixed target, Double, Dueling), ~02:06-02:11 (Segunda Parte, papers), ~03:09-03:13 (buffers enormes y priorización), ~03:30-03:35 (exploración: redes con ruido, pseudo-conteos, error de predicción), ~03:54-04:03 (tamaño del buffer), ~04:11-04:22 y ~04:29-04:31 (ajuste dinámico, `deque`, buffer inicial), ~04:31-04:37 (PER, Montezuma, CER), ~04:49-04:52 (*catastrophic forgetting* y cómo leer el error).
> - **T07** (`docs/raw/26 11 07.pdf`, clase del 07/11/2026): ~00:23-00:28 (no todo es IA; combinar algoritmos, "rainbows"; inestabilidad), ~03:44-03:52 (buffer en el caso de ejecución), ~04:08-04:09 (árbol para la probabilidad de ejecución), ~04:13-04:23 (benchmark clásico; cuándo compensa el RL; DeepMind).
> - **RLM** (`docs/raw/MIAX14/rl-material/notebooks/rl-models-basic/`): `linear-approximation`, `linear-sarsa`, `linear-expected-sarsa`, `deep`, `deep-fixed`, `deep-double`, `deep-dueling`, `deep-expected-sarsa`, `deep-rainbow` (celdas 2 y 3 de cada uno).
> - **BE** (`docs/raw/MIAX14/best-execution/src/agents/dqn.py`): el `DDQNAgent` de la práctica. **LAB** (`docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`): celdas 7 (acciones del entorno), 11, 12 y 16.
> - **T2** (`docs/raw/referencias/deep-rl-trading-agents/docs/teoria/`): `03_dqn.md`, `04_doble_dqn_y_duelo.md`, `06_actor_critico_y_a3c.md`, `07_panorama_drl.md`, para profundizar.
>
> **Convenciones:** `(T06 ~01:36)` = hora aproximada en la transcripción; todo lo que viene de ellas es paráfrasis de un texto automático muy ruidoso. `(APP p.40)` = página del PDF. `(RLM deep-double, celda 2)` = celda del notebook, contando desde 0. **Nota:** = contenido propio (cálculos, correcciones, conexiones, literatura externa) que no está en el material de clase. *(dudoso)* = la transcripción no permite asegurar lo dicho; se dan las lecturas posibles. Notación: $\theta$ = pesos de la red principal (*primary*), $\theta'$ o $\theta^-$ = pesos de la copia (*target*), $x(s)$ = vector de características del estado, $\delta$ = error TD.
>
> **Otros documentos:** Q-learning, SARSA, Expected SARSA y Double Q-learning tabulares en [00_repaso_rl.md](00_repaso_rl.md); tamaño y llenado del buffer, exploración, *catastrophic forgetting* y checkpoints en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md); el `DDQNAgent` de la práctica línea a línea en [08_entorno_best_execution.md](08_entorno_best_execution.md); los 18 notebooks de RLM en [09_material_rl_notebooks.md](09_material_rl_notebooks.md); términos en [11_glosario.md](11_glosario.md).
>
> Escrito el 10/11/2026.

---

## 1. Qué se vio y el mensaje de fondo

**Idea.** En la primera parte de la clase del 06/11 el profesor pasó del Q-learning tabular a la aproximación funcional (lineal y redes) y presentó Deep Q-Learning con los problemas que trae y las piezas que los corrigen: experience replay, red objetivo fija, Double DQN y Dueling. Insistió en que los nombres de los algoritmos importan poco. Lo que importa son dos palabras: **estabilidad** y **sesgos** (T06 ~01:19, ~01:40).

| Problema | Solución | Cómo lo resuelve | Dónde |
|---|---|---|---|
| Datos correlacionados (no iid) | Experience Replay | Entrena con muestras aleatorias del buffer | §6 |
| El target se mueve: juez y parte | Fixed Target Network | Copia $\theta^-$ que solo se actualiza cada $T$ pasos | §7 |
| Sobreestimación de $Q$ | Double DQN | Una red elige la acción, la otra la evalúa | §8 |
| Muchas acciones parecidas | Dueling DQN | Separa $Q=V(s)+A(s,a)$ | §9 |

*Fuente: tabla "Resumen: Mejoras de DQN" de APP p.45; las cuatro primeras columnas son las de la diapositiva y la columna "Dónde" es nuestra.*

Lo que dijo para cerrar ese bloque, en paráfrasis:
- Hay muchas variantes que crean dos redes o copias que se van pasando los pesos cada cierto tiempo, y "todo es en busca de esa estabilidad". Hay que buscarla no solo eligiendo algoritmo, sino en cómo se diseña y configura todo el sistema (T06 ~01:18-01:19).
- En supervisado hay que vigilar que no sobreajuste. En RL hay que "usar lo que funcione" (T06 ~01:19). La cita de Karpathy de APP p.88 dice lo mismo: "Supervised learning wants to work. Reinforcement Learning must be forced to work".
- Si de este bloque solo se recuerda una cosa, que sean la estabilidad y los sesgos. Cuando aparece "una copia" de la red, viene a resolver eso (T06 ~01:40).
- Son cambios "muy pequeños en términos de código" que dan resultados muy distintos, igual que pasaba con SARSA y Expected SARSA (T06 ~01:40).
- Al día siguiente lo cuantificó: son algoritmos que aprenden con aleatoriedad, pequeños cambios dan resultados absolutamente distintos y, aun con todo perfectamente configurado, en un problema algo complejo "3 de cada 10" entrenamientos no funcionarán (T07 ~00:27:28-00:27:54). Consecuencia para la práctica: repetir con varias semillas antes de sacar conclusiones; ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) (§8, semillas e inestabilidad).

---

## 2. Del aprendizaje tabular a la aproximación funcional

**Idea.** Hasta aquí el agente era una tabla, un diccionario literal en el que los valores se cambian a mano (T06 ~01:20). Con estados continuos o enormes eso no funciona. La alternativa es un modelo que recibe el vector de características del estado y devuelve los Q-valores. Ya no se actualizan los valores: se ajustan los parámetros $\theta$ del modelo (APP p.32).

### 2.1 Limitaciones del enfoque tabular

APP p.31 da el problema (espacios de alta dimensionalidad) y tres limitaciones, ilustradas con un coche en una carretera de montaña. El profesor las explicó así (T06 ~01:20-01:21):

| Limitación (APP p.31) | Explicación en clase |
|---|---|
| **Entrenamiento** | Estos modelos parten de que hay que vivir algo muchísimas veces para aprenderlo. Con infinitos estados es literalmente imposible. |
| **Espacio** | No cabe en memoria un diccionario de algo infinito. |
| **Puesta en producción** | Dos estados muy parecidos, uno visto y otro no, pueden tener valores muy distintos y llevar a acciones totalmente distintas. "No te puedes jugar la vida" así en un coche autónomo: la tabla no **generaliza**. |

APP p.32 lo resume: la aproximación funcional "no actualiza los valores de los estados-acciones", "modifica los parámetros (θ) del modelo de predicción" y "permite generalizar para casos no vistos". Se puede aproximar la función de valor, la de Q-valores o directamente la política (T06 ~01:21).

### 2.2 El cambio, en una imagen (APP p.33)

La diapositiva compara dos esquemas:
- **Antes:** una tabla $Q$ con 2 acciones × 4 estados: $a_1=[7,3,0,2]$ y $a_2=[4,1,2,0]$ para $s_1..s_4$.
- **Ahora:** el estado $S_2$ es un vector de 4 características $[0{,}2;\ 2;\ 1{,}5;\ 0]$ ($f_1..f_4$). Entra en un "Modelo" (iconos de "Lineal" y "Neural Networks") que devuelve $Q(s,a;\theta)$ para $S_2$: $a_1=3$, $a_2=1$. Son exactamente los valores de la columna $s_2$ de la tabla de "Antes": el modelo sustituye a la tabla y da lo mismo.
- **A la derecha**, el diagrama agente-entorno con el agente resaltado y el rótulo "Lo que modificamos es el agente": el entorno no cambia, cambia cómo el agente guarda lo que sabe.
- **Ejemplo del paraguas:** $S_x=[40{,}2\ ^{\circ}\mathrm{C},\ 12\%\ \text{humedad}]$ entra al modelo, que devuelve $-20$ para "coger paraguas" y $34$ para "no cogerlo". El 34 aparece rodeado: es la acción elegida.

En clase (T06 ~01:21-01:22): antes había una tabla, ahora un estado que es un vector de características y un modelo intermedio que da lo mismo que la tabla. **Toda la filosofía discutida se mantiene.** Si llovía, no se cogió el paraguas y llegó una mala recompensa, se ajustan los pesos para que la próxima vez se dé más valor a coger el paraguas en un día lluvioso. Para actuar se aplica el mismo principio: se toma la acción que maximiza la salida del modelo (argmax).

### 2.3 El ejemplo tabular del caso de ejecución (CASOS p.46-50)

Para el caso de ejecución las diapositivas muestran la versión tabular del mismo esquema:
- **Estado** en $S_3$ (10:30): spread bid-ask 2 pbs, probabilidad de ejecución 85 %, tiempo restante 25 %, volumen restante 20 %. **Acciones** discretas: $a_1$ pasiva, $a_2$ mantener, $a_3$ agresiva; o continuas: "precio medio + X pbs" (CASOS p.46).
- **Tabla Q** (filas $a_1..a_3$, columnas $S_1, S_2, S_3, S_x$) con 0 repeticiones: todo a 0, "no sabe nada" (p.47). Con 500 repeticiones, valores pequeños, "ya sabe algo": $a_1=[-0{,}3;\ 0{,}4;\ 0{,}1;\ -1{,}2]$, $a_2=[0{,}1;\ -0{,}3;\ 0{,}6;\ 0{,}2]$, $a_3=[1{,}1;\ 0{,}2;\ 1{,}4;\ 0{,}8]$ (p.48). Con $N$ repeticiones, "llega a una solución": $a_1=[-2;\ 7;\ 1;\ 3]$, $a_2=[4;\ -3;\ 6;\ 1]$, $a_3=[5;\ -3;\ -4;\ 8]$, con los máximos en verde en $S_2\to a_1$ (7), $S_3\to a_2$ (6) y $S_x\to a_3$ (8) (p.49).
- **Política** (p.50): matriz one-hot con un 1 en $S_2\to a_1$, $S_3\to a_2$ y $S_x\to a_3$, y la columna $S_1$ toda a 0.

El profesor lo comentó: si se discretiza el estado, queda el diccionario con 3 acciones y los estados posibles. Si no, hace falta una aproximación funcional. Tras X repeticiones los valores convergen y la política es la acción en verde, la que maximiza en cada estado (T07 ~03:51-03:52).

**Nota (errata):** en p.49 el máximo de la columna $S_1$ es $a_3=5$, pero ni está en verde ni aparece en la política de p.50, donde $S_1$ queda a ceros. Lo coherente sería $S_1\to a_3$. Probablemente es una simplificación de la diapositiva. El caso completo (estado, acciones, recompensas) está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

---

## 3. Cómo se ajustan los pesos: descenso por gradiente (APP p.34)

**Idea.** Los pesos se modifican por descenso por gradiente: se buscan los $\theta$ que minimizan una función de pérdida, el error cuadrático entre el valor verdadero y el que predice el modelo (APP p.34).

$$J(\theta)=\mathbb{E}_\pi\big[(v_\pi(S)-v_\theta(S))^2\big]$$

$$\Delta\theta=-\tfrac12\,\alpha\,\nabla_\theta J(\theta)=\alpha\,\mathbb{E}_\pi\big[(v_\pi(S)-v_\theta(S))\,\nabla_\theta v_\theta(S)\big]$$

Versión estocástica (SGD), con una sola muestra $S_t$ en lugar de la esperanza:

$$\Delta\theta_t=\alpha\,\big(v_\pi(S_t)-v_\theta(S_t)\big)\,\nabla_\theta v_\theta(S_t)$$

*Fuente: APP p.34 (las tres fórmulas, tal cual).*

**Nota (de la fórmula a lo que se programa).** $v_\pi(S_t)$ es desconocido. En la práctica se sustituye por un **objetivo** calculado con la experiencia: el retorno $G_t$ en Monte Carlo, o $R_{t+1}+\gamma\,v_\theta(S_{t+1})$ en TD(0). Para Q-learning con aproximación funcional:

$$\theta\leftarrow\theta+\alpha\Big[\underbrace{R_{t+1}+\gamma\max_{a'}Q_\theta(S_{t+1},a')}_{\text{objetivo}}-Q_\theta(S_t,A_t)\Big]\nabla_\theta Q_\theta(S_t,A_t)$$

No se deriva a través del objetivo, aunque depende de $\theta$ (por eso se llama *semigradiente*). Así el objetivo se mueve cada vez que se actualiza $\theta$. Ese es el origen formal del problema "juez y parte" de §5.2, que corrige la red objetivo de §7. Los notebooks de clase hacen esto con Keras: construyen un vector de objetivos igual a la predicción actual salvo en la acción tomada, y llaman a `train_on_batch` con pérdida `mse` (RLM deep, celda 2). Más detalle de la función de coste de DQN en [T2 03_dqn.md §5](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md).

---

## 4. Aproximación lineal

### 4.1 Qué es y cómo la explicó el profesor

**Idea.** Es la primera aproximación funcional: una regresión lineal por acción. Es "buena aproximación" y "rápida" (APP p.35). El profesor la recomendó como primer paso porque las regresiones son mucho más rápidas y a veces bastan antes de saltar a modelos más complejos (T06 ~01:22-01:23).

Cómo funciona, según la explicación en la pizarra (T06 ~01:24-01:25):
- Hay una matriz de coeficientes con una fila por acción (acción 0, 1, 2...) y una columna por característica (1, 2, 3, 4). Los pesos se suben y bajan, "no hace otra cosa más que eso".
- **Ejemplo numérico:** el estado tiene 4 características de valores 3, 0, 1,2 y 7 *(los valores de los pesos dictados, 0,2, 0,7 y 1,2, son dudosos y no llegan a las cuatro)*. Se multiplica por la fila de la acción 0 y sale una predicción de 20. El valor observado es 30, así que hay que subir un poco los pesos para que la siguiente predicción dé unos 22.
- "Es como si fuese una regresión distinta" para cada acción.

En código es exactamente eso (RLM linear-approximation, celda 2): `Qtheta = np.random.randn(na, n_features) * 0.01`, $Q(s,\cdot)=\Theta\,x(s)$ (`np.dot(self.Qtheta, s)`), y la actualización de la fila de la acción tomada es

$$\theta_a\leftarrow\theta_a+\alpha\,\big(r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big)\,x(s)$$

(`self.Qtheta[a] = self.Qtheta[a] + self.alpha * (target - qsa) * s`), que es la SGD de §3 con $\nabla_{\theta_a}Q(s,a)=x(s)$. Las variantes cambian solo el objetivo (RLM linear-sarsa y linear-expected-sarsa, celda 2):
- **Lineal Sarsa:** $r+\gamma Q(s',a')$, con $a'$ la acción que se tomará de verdad.
- **Lineal Expected Sarsa:** $r+\gamma[(1-\varepsilon)\max_{a'}Q(s',a')+\varepsilon\,\overline{Q}(s',\cdot)]$, el valor esperado bajo la política ε-greedy.

**Nota (cálculo con el ejemplo de clase).** Con SGD en un modelo lineal, la predicción para el mismo estado cambia en $\alpha\,\delta\,\lVert x\rVert^2$. Con $x=[3;\ 0;\ 1{,}2;\ 7]$, $\lVert x\rVert^2=9+0+1{,}44+49=59{,}44$ y $\delta=30-20=10$. Pasar de 20 a 22 equivale a $\alpha\approx 2/(10\cdot59{,}44)\approx0{,}0034$. Consecuencia práctica: el tamaño del paso depende de la escala de las características. Una característica de magnitud 1000 sin normalizar haría saltar la predicción, y eso es un argumento más para estandarizar el estado (ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md)). Otra matización: el "30" no es el valor verdadero sino el objetivo TD, que también se estima.

**Nota (el código).** El modelo lineal de RLM no tiene término independiente (sesgo), y el constructor acepta `mean` y `std` pero no los usa: no normaliza el estado (RLM linear-approximation, celda 2). Los tres notebooks lineales usan `LunarLander-v2` (celda 3), no CartPole como los gráficos de APP p.35.

### 4.2 Pregunta en clase: ¿y con árboles?

Un alumno preguntó si se podían usar árboles de decisión en vez de regresión o redes. El profesor respondió que entendería que sí, pero que nunca lo ha hecho: tiene la fórmula para ajustar los pesos del lineal y de las redes, pero no sabría cómo ajustar los puntos de corte de un árbol. Sugirió preguntarlo a un asistente de IA (ChatGPT) e invitó al alumno a investigarlo y contárselo ("si te apetece buscarlo y me dices") (T06 ~01:23:41-01:24:05, ~01:25:51-01:26:01). Como el alumno comentó que en su taller no habían visto la parte lineal, le remitió al notebook `linear-approximation` para comentarlo al día siguiente, y a continuación explicó el modelo lineal en la pizarra (§4.1) (T06 ~01:24:17-01:24:26). Al día siguiente, el árbol sí apareció, pero en otro papel: como modelo supervisado auxiliar para la probabilidad de ejecución ("no hace falta que sean redes, puede ser un árbol", T07 ~04:08-04:09), no como aproximador de $Q$; ver [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md). **Nota:** la dificultad de usarlo como aproximador de $Q$ es real. Un árbol no se actualiza por gradiente con cada experiencia; habría que reentrenarlo por lotes sobre el buffer (en la línea de *fitted Q-iteration*). No se trató en clase.

### 4.3 Los gráficos de APP p.35-36

**CartPole (APP p.35).** Tres curvas de recompensa por episodio:

| Variante | Eje X | Comportamiento |
|---|---|---|
| Lineal | 0-20 000 episodios | Sube casi lineal de ~40 a ~370, con ruido; no llega a 500 |
| Lineal Sarsa | 0-20 000 | Sube de ~25 a ~280, más despacio |
| Lineal Expected Sarsa | 0-10 000 | Sube a ~200 hacia el episodio 900, cae a ~120, se dispara entre 2500 y 4000 hasta **500** (el máximo de CartPole-v1) y se queda plano |

**Nota:** con el mismo modelo lineal, solo cambiar el objetivo (§4.1) pasa de no resolver el problema en 20 000 episodios a resolverlo en ~4000. Es el ejemplo de "cambio pequeño de código, resultado muy distinto" de §1.

**DQN frente al mejor aprendiz lineal (APP p.36).** Es un gráfico de barras horizontales (la diapositiva no cita su origen; **Nota:** es la figura de comparación del paper de DQN en Atari, Mnih et al., *Nature*, 2015). Muestra la puntuación normalizada respecto a un humano (100 %) en cada juego para DQN (azul) y el *Best Linear Learner* (gris). Arriba de la línea "at human-level or above" están Video Pinball (2539 %), Boxing (1707 %), Breakout (1327 %), Star Gunner (598 %), Robotank (508 %), Atlantis (449 %)... Debajo, "below human-level", quedan Asterix (69 %) y, al final, Asteroids (7 %), Frostbite (6 %), Gravitar (5 %), Private Eye (2 %) y Montezuma's Revenge (0 %). La diapositiva corta la lista intermedia. En varios juegos la barra gris es comparable a la azul (Boxing, Atlantis, Krull) y en algunos la supera (Kung-Fu Master, Tennis).

Mensajes de la diapositiva (APP p.36):
- "No lo hace tan mal": los juegos están pensados para funcionar mejor con imágenes, donde las redes convolucionales tienen ventaja.
- "En bolsa, vais a pasar características", así que lo lineal puede ser buena aproximación.
- "¿Qué entienden ellos por best linear learner?" (pregunta abierta de la diapositiva).
- "La dispersión de resultados es importante": las barras de error de DQN son amplias.

En clase solo se dijo que son comparativas de modelos lineales con otros más complejos (T06 ~01:23).

**Nota:** en el paper de Nature, "best linear learner" es el mejor de los agentes de la literatura previa del Arcade Learning Environment que usan aproximación lineal sobre conjuntos de características diseñados a mano. No es un lineal sobre píxeles. La comparación favorece algo al lineal por esas características, lo que refuerza el argumento de la diapositiva: con buenas características, lo lineal compite.

---

## 5. Deep Q-Learning y sus dos problemas

**Idea.** Sustituir el lineal por una red profunda permite encontrar relaciones más complejas y tratar datos en forma de imágenes o series temporales (APP p.37, con el dibujo de una red de 4 entradas, 5 capas ocultas y 3 salidas $y_1..y_3$, una por acción). Pero trae "una serie de problemas" (APP p.37) que no existían en el tabular.

### 5.1 Problema 1: las observaciones no son iid

APP p.38: "No podemos utilizar últimas observaciones para entrenar el modelo". Necesitamos variables aleatorias iid y, para ello, un registro de experiencias o *buffer* FIFO. Se desarrolla en §6.

### 5.2 Problema 2: juez y parte

APP p.39: los modelos DQN tienden a sobreponderar ciertas acciones. ¿Por qué? Tienen una sola red, y con ella se evalúan las acciones y se toman las decisiones: son "juez y parte". Entonces "se retroalimenta" (~*confirmation bias*).

En clase (T06 ~01:36-01:37): en la configuración más clásica, la misma red toma las acciones y las evalúa, dice si han sido buenas o malas. Se pregunta cuál es la mejor y cuánto vale, con eso vuelve a entrenar, se retroalimenta e **infla artificialmente los valores** de las acciones. Las soluciones son la red objetivo fija (§7) y Double DQN (§8).

**Nota:** hay dos fenómenos mezclados bajo "juez y parte". (a) El **objetivo móvil**: el target $r+\gamma\max Q_\theta(s',\cdot)$ cambia con cada actualización de $\theta$ (§3), y lo corrige la copia fija. (b) El **sesgo de maximización**: $\mathbb{E}[\max_a \hat Q]\ge\max_a\mathbb{E}[\hat Q]$, de modo que el máximo de estimaciones ruidosas sobreestima, y lo corrige separar quién elige y quién evalúa. La tabla de APP p.45 los separa bien (fila 2 y fila 3). La versión tabular de (b), Double Q-learning, está en [00_repaso_rl.md](00_repaso_rl.md), y el razonamiento completo en [T2 04_doble_dqn_y_duelo.md §1](../raw/referencias/deep-rl-trading-agents/docs/teoria/04_doble_dqn_y_duelo.md).

---

## 6. Experience replay

### 6.1 Por qué hace falta: la analogía con el supervisado

El profesor presentó este bloque como una "vuelta a los principios, y menos a los algoritmos" (T06 ~01:26:22): el replay es un principio de cómo se entrena, no un algoritmo nuevo.

**Idea.** Para entrenar una red, las observaciones de cada minibatch tienen que ser independientes e idénticamente distribuidas: tienen que venir de momentos distintos y no estar siempre muy correlacionadas (T06 ~01:26).

- **Ejemplo de Santander** (T06 ~01:27-01:29): si un día (el 6 de noviembre) Santander sube toda la sesión y se entrena con las últimas 64 experiencias, que son todas de ese día, la red aprende que "Santander siempre sube". Con 64 experiencias de días distintos, que unos días suba y otros baje, no depende de eso y aprende otro tipo de relación. No basta con jugar episodios distintos: hay que **aprender de varios episodios a la vez** en cada actualización, no solo del último.
- **Analogía con el supervisado** (T06 ~01:30-01:31): con un dataset de un millón de observaciones se cogen 64 al azar en cada minibatch y por eso se consideran independientes. Si se cogieran solo las 64 que más suben, se metería un sesgo muy importante: la red no aprendería ninguna relación y se sobreajustaría a eso.
- **La diferencia del RL** (T06 ~01:31-01:32): el dataset no existe de antemano, se construye sobre la marcha jugando. Se quiere aplicar el mismo principio, así que se guarda un registro de experiencias y se entrena con muestras variadas del registro en lugar de con las últimas 64.
- **Series temporales** (T06 ~01:34:54-01:35:00): en algo con tiempo, como una acción, se muestrea aleatoriamente a lo largo del tiempo y no lo más cercano al presente *(dudoso: la frase empieza por "Claro", así que pudo ser la confirmación del profesor a un comentario de un alumno; la redacción exacta no se entiende bien)*.

Más sobre la memoria de experiencias y el algoritmo DQN básico que la usa en [T2 03_dqn.md §8](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md).

### 6.2 El buffer FIFO: independencia frente a obsolescencia

**Idea.** Si se guarda todo, se acumulan millones de experiencias, y muchas quedan **obsoletas**: responden a políticas desactualizadas que ya no aportan valor (T06 ~01:32). Se quiere independencia, pero no aprender solo con lo último ni con lo más antiguo: se busca un punto intermedio (T06 ~01:32-01:33). La solución es un buffer de tamaño acotado, FIFO: entran las nuevas y salen por detrás las más antiguas (APP p.38, p.77; T06 ~01:33).

Matiz de clase (T06 ~01:34): el agente ya aprendió de esas experiencias antiguas, ajustó su política y empezó a explorar caminos que considera más eficientes. Lo que aún no ha aprendido está en lo más reciente del registro.

**Tamaño $m$.** En clase (T06 ~03:54-03:56): el buffer se mete en cualquier algoritmo de deep RL, porque sin él "metéis muchísimos sesgos o no lo aprendéis nunca". No hay que confundir su tamaño (las experiencias guardadas) con el batch de entrenamiento (las 64 que se muestrean en cada aprendizaje). Pidió no copiar sin pensar la solución típica de "entrena N episodios y pon un buffer de un millón": el tamaño depende de factores como la cantidad de datos disponibles; con una serie temporal de 5 años en vez de 20 no se pueden poner buffers demasiado grandes *(dudoso: la justificación se transcribe como "si no ves demasiada historia")*.

El dilema es independencia (buffer grande: experiencias de días y momentos muy distintos) frente a obsolescencia (experiencias generadas con pesos antiguos que no se parecen a los actuales) (APP p.78; T06 ~03:56-03:57). La diapositiva da tres datos:
- La probabilidad de aparición de una experiencia en $k$ steps es $1-(1-1/m)^k$.
- Hay un consenso de ~1 millón de experiencias en los juegos de Atari.
- El modelo dinámico (§6.3).

De dónde sale el millón, según la regla de clase (T06 ~03:57-04:01). Los papers usan juegos de Atari porque el entorno ya estaba creado y son juegos sencillos con pocas decisiones (T06 ~03:57:52-03:58:09). La cuenta, hecha con el móvil en clase: 60 fotogramas por segundo × 60 s × 2,5 min de partida = 9000 steps por partida. Con las últimas 100 partidas salen 900 000, "pues bueno, un millón" *(cifras dudosas: el profesor dudó en voz alta entre 2, 2,5 y 4 minutos y entre 60 y 30 fotogramas, T06 ~03:58:37-03:59:07, ~04:01:07)*. La recomendación es **pensar en partidas o episodios, no en experiencias**: con las últimas ~100 partidas, 64 muestras aleatorias vendrán con alta probabilidad de partidas distintas y serán a la vez recientes e independientes. En bolsa, igual, pero con ejecuciones u operaciones (T06 ~04:01-04:02). Ante la pregunta de un alumno aclaró que es un millón de steps, no de partidas, y que prefiere razonar como "mis 100 últimas partidas" (T06 ~04:02:28-04:02:50).

**Nota (no comprobado contra el material).** En el DQN de *Nature* cada acción se repite 4 fotogramas (*frame skip*), así que una partida de 9000 fotogramas daría ~2250 transiciones y el millón equivaldría a unas 440 partidas, no a 100. La regla vale como orden de magnitud y, sobre todo, como método (pensar en episodios), no como derivación exacta.

Aplicado al caso de ejecución (T07 ~03:45-03:50): se piensa en sesiones según la diversidad necesaria. Con regímenes alcista, plano y bajista, unas 100 sesiones (~30 de cada uno) o hasta 500 para cubrir matices (T07 ~03:46:48-03:48:03). En cliff walking bastan las últimas 10 partidas porque hay pocas casuísticas; si alguna variable relevante aparece muy de vez en cuando, hacen falta más partidas (T07 ~03:48:12-03:48:30). El buffer se configura en experiencias = sesiones × momentos de decisión por sesión: en su ejemplo, ~10 decisiones por sesión dan 100 × 10 = 1000, un buffer que "puede ser tan pequeñito" como eso (T07 ~03:49:13-03:49:41). También se puede configurar por partidas, lo que exige registrar la duración de cada una de las últimas 100, pero una aproximación basta ("con un proxy ya…") (T07 ~03:49:51-03:50:21).

**Nota (para la práctica).** En LAB `buffer_size` es un TODO (`buffer_size = "-----"`, LAB celda 12). Con episodios de 400-600 pasos, 100 episodios serían 40 000-60 000 experiencias. El cálculo detallado está en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) §6.3 y la decisión razonada en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

> El desarrollo práctico del tamaño (cálculo para episodios de la práctica, llenado inicial) está en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md). Aquí solo se recoge lo necesario para entender el mecanismo.

**Nota (qué significa la fórmula).** $1-(1-1/m)^k$ es la probabilidad de que una experiencia concreta salga al menos una vez en $k$ extracciones uniformes de una en una. Para $k=m$ vale $1-(1-1/m)^m\approx1-e^{-1}\approx63\,\%$. Consecuencia útil para la práctica: en los notebooks de clase y en LAB se aprende cada `learn_after = batch_size` pasos (LAB celda 12; RLM deep, celda 3). En régimen estacionario, cada experiencia vive $m$ pasos en el buffer; en ese tiempo hay $m/B$ actualizaciones, y en cada una sale con probabilidad $B/m$. Se usa, en media, **una vez** en toda su vida, sea cual sea $m$ o $B$, y una fracción $\approx e^{-1}\approx37\,\%$ **nunca se usa**. Si se quiere reutilizar más cada experiencia, hay que aprender con más frecuencia que `learn_after = batch_size`.

**Implementación** (APP p.79; T06 ~04:13):

```python
from collections import deque
buffer = deque(maxlen=buffer_size)
buffer.append((s, a, r, s1))
```

Nunca una lista con `.pop(0)`. La `deque` es una colección específica más eficiente, y en RL se busca estabilidad, simplicidad y eficiencia porque hay que evaluar un montón de veces. El profesor añadió que dedicó mucho tiempo solo a hacer el código más eficiente, por ejemplo descartando pandas en favor de diccionarios (T06 ~04:13-04:14; ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md)). **Nota:** `list.pop(0)` es $O(n)$ porque desplaza toda la lista; `deque` con `maxlen` descarta la más antigua en $O(1)$ al hacer `append`. El acceso por índice en mitad de una `deque` es $O(n)$, pero con buffers de $10^4$-$10^5$ y batches de 64 no es el cuello de botella.

### 6.3 Ajuste dinámico del tamaño: *Dynamical System Model of ER*

APP p.78: se ajusta según el error cometido. Más error → aumentar el tamaño del buffer; menos error → disminuirlo. La ilustración son tres pilas de capas de altura creciente.

En clase (T06 ~04:11-04:13): para cada componente hay "otros 20 papers" sobre su tamaño ideal y cómo ajustarlo dinámicamente. En este, el error es la diferencia entre lo que la red estima para una experiencia del buffer y el valor que realmente le llega. Si los errores son altos, las experiencias "son nuevas" para la red y necesita aprender más de ellas: se amplía el buffer (por ejemplo, de 100 a 200 partidas) para no descartarlas pronto y poder volver a seleccionarlas. Si los errores son ya muy pequeños, esas experiencias están aprendidas y se reduce el tamaño. "Ideas felices son bienvenidas".

**Nota:** la diapositiva no da referencia. En la literatura, la idea de modelar el replay como un sistema dinámico y adaptar el tamaño de la memoria con el error TD se asocia a Liu y Zou (2018), *The Effects of Memory Replay in Reinforcement Learning*. No está comprobado contra el material de clase.

### 6.4 Llenado inicial (resumen)

El buffer empieza vacío. Si se aprende con las primeras 64 experiencias, todas son del mismo día o de la misma partida y están muy correladas. Como mínimo hay que esperar a llenarlo un poco antes de la primera actualización (T06 ~04:14-04:15). El ejemplo de clase es una regla de medias móviles: en un día alcista, con la media corta por encima de la larga, las 64 muestras refuerzan 64 veces que "comprar es bueno". Lo que se quiere es que, para ese mismo vector de características, el batch mezcle días que acabaron bien, regular y mal, como "un promedio de observaciones independientes" (T06 ~04:15-04:18).

Se puede llenar con experiencias aleatorias como mínimo, con experiencias reales o con un algoritmo base (TWAP, el bróker, partidas reales de póker). No hace falta llenarlo entero. El dilema aquí es **independencia frente a ruido**, y llenarlo con un comportamiento X arriesga que la red se sobreajuste a X (APP p.79; CASOS p.45; T06 ~04:20-04:22; T07 ~03:50-03:51). APP p.79 enlaza además un vídeo junto al consejo de llenar el buffer (`https://www.youtube.com/watch?v=W_gxLKSsSIE`); no se comentó en clase y su contenido no está verificado.

Regla de clase: si las experiencias iniciales son ruidosas (aleatorias), mejor no llenarlo entero (por ejemplo 50 partidas en lugar de 100, o incluso solo 20-30), para no tardar mucho en eliminarlas y empezar pronto a rellenarlo con partidas algo mejores. Si aportan información porque responden a una lógica, se puede llenar más. También se pueden mezclar aleatorias con no aleatorias (T06 ~04:20:46-04:22:20, ~04:29:45-04:30:26; T07 ~03:50:38-03:51:06). Las acciones aleatorias del llenado no tienen por qué ser equiprobables: "w no tienen por qué ser iguales" (CASOS p.45). Desarrollo completo en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) §6.6.

### 6.5 Tres formas de muestrear (APP p.38)

| Tipo | Qué hace | En clase |
|---|---|---|
| **Uniforme** | Todas las experiencias tienen la misma probabilidad | "Lo más sencillo" (T06 ~01:35, ~04:31) |
| **Prioritized (PER)** | Más probabilidad a las experiencias más significativas | Según "lo que yo creo que me aporta la experiencia" (T06 ~01:35) |
| **Combined (CER)** | 63 uniformes + la última experiencia | "Un truco muy sencillo pero muy poderoso" (T06 ~01:36) |

### 6.6 Prioritized Experience Replay (PER)

**Por qué** (T06 ~03:09-03:12, ~04:31-04:32). Con mucha capacidad de cómputo se puede explorar más, pero entra mucho ruido, y entonces no conviene una selección uniforme. Ejemplo de clase: un buffer con millones de experiencias en el que solo 30 son relevantes. Con batches de 64 uniformes puede que esas 30 no salgan nunca. Cuando por fin salen, la red ya ha convergido y está "parada" en un mínimo local, porque la información valiosa estaba rodeada de experiencias que no valían. De ahí priorizar según el valor que se cree que tiene cada experiencia; por ejemplo, las que dieron buenos resultados tienen más probabilidad de salir.

En el debate, un alumno objetó que en el límite infinito incluso un ε de 0,01 acabaría explorándolo todo. El profesor respondió que la exploración puramente aleatoria tampoco funciona: sin indicios de si va bien o mal, el agente se queda dando vueltas sin avanzar en ninguna dirección (T06 ~03:11:57-03:12:58). *(dudoso: la transcripción no separa hablantes y frases como "en epsilon del 99 también te va a llevar" podrían ser de cualquiera de los dos; la lectura más probable es que un alumno planteó el límite infinito y el profesor respondió.)* La exploración se trata en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).

**Fórmulas** (APP p.80, paper arXiv 1511.05952):

$$P(i)=\frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}$$

- **Proporcional:** $p_i=|\delta_i|+\varepsilon$, con $\delta_i$ el error TD de la experiencia $i$ ("TD Error" en la diapositiva). **Nota:** el $\varepsilon$ pequeño (no explicado en APP) evita que una experiencia con error 0 no vuelva a salir nunca.
- **Rank-based:** $p_i=1/\mathrm{rank}(i)$, con las experiencias ordenadas por $|\delta_i|$.
- **Para aumentar la estabilidad:** Huber loss e importance sampling.
- **Para mayor rapidez:** *sumtree*, que pasa de $O(n)$ a $O(\log n)$.

**Nota (errata):** la diapositiva escribe en el denominador $\sum_k p_i^{\alpha}$; debe ser $\sum_k p_k^{\alpha}$ (se suma sobre todas las experiencias $k$ del buffer).

**El exponente α**, en clase (T06 ~04:32:36-04:32:55): la probabilidad de selección es un peso basado en el error de la red, ajustado por un alfa; con un extremo es uniforme y con el otro todo el peso va al error, "ponderarlo al 100 %". *(dudoso)* La transcripción está muy rota: "Si alfas, 1 es su uniforme de profesional fastero", y después "al Fast 1 le da todo el peso". La lectura más probable, coherente con la fórmula, es que dijo que **α = 0 da muestreo uniforme** ($p_i^0=1$) y **α = 1 prioriza totalmente por error** ("profesional fastero" ≈ "proporcional… alfa cero"). Los valores intermedios mezclan ambos. Este α no tiene nada que ver con el *learning rate* α.

**Nota (los dos remedios de estabilidad).** Muestrear con prioridad cambia la distribución de los datos y sesga la estimación. En el paper se corrige ponderando la pérdida de cada muestra con $w_i=\big(\tfrac{1}{N\,P(i)}\big)^{\beta}$ normalizado por $\max_i w_i$, con β que sube hacia 1 durante el entrenamiento: eso es el *importance sampling*. La Huber loss (cuadrática para errores pequeños, lineal para grandes) evita que unas pocas experiencias de error enorme, que son justo las priorizadas, dominen el gradiente. El *sumtree* es un árbol binario cuyas hojas guardan $p_i$ y cada nodo la suma de sus hijos: muestrear y actualizar una prioridad cuesta $O(\log n)$ en vez de recorrer todo el buffer.

**El origen de la idea: Montezuma's Revenge** (T06 ~04:33-04:36; gráfico de APP p.80). El profesor contó que PER es de "2015, 2017 o así" y que surge de otra idea, ilustrada con Montezuma's Revenge. En los gráficos de Atari ese juego siempre estaba a la cola (0 % en APP p.36) porque nadie lo resolvía. Es más complicado que Breakout: tiene unas 20 salas, cada una con subobjetivos (subir escaleras, coger la llave, volver a la sala anterior y usarla), y requiere mucha más exploración. Lo resolvieron con recompensas **intrínsecas**: favorecían la exploración usando como proxy el error de predicción, que se dispara cada vez que pasa algo nuevo o "épico" (una moneda, el marcador que sube, caer por un precipicio). Ese error era la recompensa intrínseca, no "resuelve el juego" ni "máxima puntuación". En eso se basa la idea de priorizar al aprender lo que "supuso un antes y un después" o lo inesperado. Funcionaba muy bien para ese juego y no para el resto de Atari: soluciones "tontas" que funcionan para un caso particular, de las que se rescatan los principios. Requiere más cómputo y hay técnicas para acelerarlo.

El mismo principio había aparecido antes, al hablar de exploración: hay técnicas que exploran en función del error de predicción, porque las acciones nuevas suelen dar más error al estar poco exploradas ("y eso te puede decir algo"), y otras que cuentan o pseudo-cuentan cuántas veces se ha tomado una acción o se ha llegado a ciertas zonas del espacio de estados, discretizándolo de alguna forma (T06 ~03:30:50-03:31:38). Desarrollo en [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) §4.

El gráfico de APP p.80 (paper *Exploration by Random Network Distillation*, arXiv 1810.12894) muestra la recompensa intrínseca (0-0,10) a lo largo de ~2500 steps de una partida, con 21 picos numerados que corresponden a 21 capturas de pantalla de momentos de Montezuma: entrar en una sala nueva, coger la llave, abrir una puerta... Los picos más altos (~0,10) se concentran al final (19-21); les siguen el 18 (~0,065) y el 11 (~0,058), y al principio (1-3) no pasan de ~0,05.

**Nota (cronología y autoría).** PER (Schaul et al., arXiv noviembre de 2015, ICLR 2016) es **anterior** a Random Network Distillation (Burda et al., 2018, de OpenAI, no de DeepMind). La relación que hizo el profesor es conceptual, no genealógica: ambas usan un error de predicción como señal de "esto es nuevo o importante". PER lo usa para decidir qué experiencias repasar; RND, para generar una recompensa de exploración.

### 6.7 Combined Experience Replay (CER)

**Qué es** (APP p.81; T06 ~01:36, ~04:36): de un batch de 64, 63 se eligen de forma uniforme y la última es **siempre la experiencia más reciente**, para dar siempre "un toque de actualidad" al agente. APP p.81: "Añadimos nuestra última experiencia a nuestro batch → Ayuda a corregir el tamaño del buffer". Paper: *A Deeper Look at Experience Replay* (arXiv 1712.01275).

**El gráfico de APP p.81** compara el retorno (de −5000 a 0) frente a episodios (0-100) para buffers de 100, 1000, 10 000, 100 000 y 1 000 000:
- **(b) Buffer-Q** (replay uniforme; leyenda: 100 rojo, 1000 verde, 10 000 azul, 100 000 amarillo, 1 000 000 morado): el buffer de 100 aprende el primero (≈−700 hacia el episodio 40, cerca de 0 a partir del ~60). Le siguen 1000 y, sorprendentemente, 1 000 000, que suben juntos entre los episodios ~25 y 70 y rozan 0 hacia el 70-80. Los más lentos son 10 000 y 100 000, que no despegan hasta el episodio ~45-50 y a los 100 episodios siguen en torno a −1000 y −2000. La sensibilidad al tamaño es grande y **no monótona**.
- **(c) Combined-Q** (CER): las cinco curvas convergen de forma parecida y rápida: hacia los episodios 25-40 están todas entre −1000 y −500 y a partir del ~60 rozan 0. El tamaño del buffer casi deja de importar.

**Nota:** la no monotonía del panel (b) refuerza el mensaje de la diapositiva: con replay uniforme el tamaño del buffer es un hiperparámetro delicado, y CER reduce esa sensibilidad.

**Pseudocódigo** (APP p.82, "Algorithm 3: Combined-Q"):

```text
Inicializar la función de valor Q y el replay buffer M
mientras no converja:
    obtener el estado inicial S
    mientras S no sea terminal:
        elegir A con una política ε-greedy derivada de Q
        ejecutar A, observar R y S'
        guardar la transición t = (S, A, R, S') en M
        muestrear un batch B de transiciones de M
        actualizar Q con B y t          # la transición recién vivida siempre entra
        S <- S'
```

**Lo que recomendó el profesor** (T06 ~04:36-04:37): "uno de los principales" autores del RL (*Sutton*, transcrito "suto") muestra que, cuando uno se pasa de tamaño de buffer, la velocidad de convergencia se beneficia de CER. Su recomendación: es casi mejor **pasarse de largo**, por ejemplo 1000 episodios si harían falta 100. Habrá muchas experiencias obsoletas, pero con el toque de actualidad se corrige en parte el sesgo de haber metido tantos episodios.

**Nota:** el paper es de Shangtong Zhang y Richard S. Sutton (2017), así que la atribución a Sutton es correcta como coautor. El paper muestra que el replay es sensible al tamaño del buffer por ambos lados (demasiado pequeño y demasiado grande) y que CER mitiga sobre todo el daño de un buffer grande. "Pasarse de largo" es la lectura práctica del profesor, no una conclusión literal del paper.

**Nota (código):** CER es lo que implementan el agente de la práctica y el notebook Rainbow:
```python
idx = np.hstack([np.random.choice(len(self.buffer) - 1, size=self.batch_size - 1, replace=False),
                 len(self.buffer) - 1])
```
(BE dqn.py sample; RLM deep-rainbow, celda 2). Requiere que el buffer tenga al menos `batch_size` experiencias; LAB lo garantiza al prellenarlo entero (LAB celda 16).

---

## 7. Red objetivo fija (*Fixed Q-targets*)

**Idea.** Se crea una copia de la red, $\theta'=\theta$, que se encarga de **evaluar** (red *target*). La red original va aprendiendo cada $N$ steps y cada $T$ steps se copian sus pesos en la copia (APP p.40).

**El gráfico de APP p.40** tiene dos ejes temporales:
- Arriba, "Aprendemos" ($\theta$): marcas de actualización muy frecuentes.
- Abajo, "Actualizamos copia (Target Network)" ($\theta'\leftarrow\theta$): marcas más espaciadas.

Se aprende con más frecuencia de la que se actualiza la copia. A la derecha, el esquema del paper: Input → "Prediction Network" $Q$ y "Target Network" $Q'$ ("parameter update every C iterations"), con la pérdida

$$\Big(r+\gamma\max_{a'}Q(s',a';\theta')-Q(s,a;\theta)\Big)^2$$

donde el primer término se rotula "Target" y el segundo "Prediction".

En clase (T06 ~01:37): es una modificación muy pequeña y ya es tan común que, cuando se habla de DQN, se sobreentiende su versión con fixed target. Se crea una copia, que es la que evalúa las acciones y se actualiza de forma más pausada. La red principal aprende más a menudo, y así se quita parte de la retroalimentación.

**En el código de clase y de la práctica**, $N$ = `learn_after = batch_size` y $T$ = `update_target_after = 10 * learn_after`: la copia se actualiza una vez cada 10 aprendizajes (LAB celda 12, donde `update_target_after` se define como "cada cuántos steps se copian los pesos de la red primary a la red target", LAB celda 11; RLM deep-fixed, celda 3). La copia se hace con `target_model.set_weights(model.get_weights())` (RLM deep-fixed, celda 2; BE dqn.py `update_target`).

**Nota:** con `batch_size = 64` serían $N=64$ y $T=640$ pasos de entorno. Si en la práctica se cambia `batch_size`, cambian a la vez la frecuencia de aprendizaje y la de copia, porque las dos están atadas a él. Más sobre la red objetivo y el algoritmo DQN de Atari completo en [T2 03_dqn.md §9](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md); la alternativa de actualización suave (Polyak, $\theta^-\leftarrow\tau\theta+(1-\tau)\theta^-$), que tampoco está en el material de este taller, en [T2 04_doble_dqn_y_duelo.md §2](../raw/referencias/deep-rl-trading-agents/docs/teoria/04_doble_dqn_y_duelo.md).

---

## 8. Double DQN

### 8.1 Qué cambia

**Idea.** La misma filosofía de dos redes, con una pequeña diferencia: la red **principal elige** la mejor acción siguiente y la **target la evalúa** (APP p.41; T06 ~01:37-01:38). Así se evita sobreponderar las buenas acciones.

APP p.41 describe dos redes:
- **DQN / Primary Network** (aprende y actúa): "le pasamos un estado, devuelve una acción".
- **Target Network** (evalúa): "le pasamos una acción, devuelve un valor". Es una copia congelada que se actualiza cada cierto tiempo (T06 ~01:38).

| Tarea | Fixed target | Double DQN |
|---|---|---|
| Predecir $Q(s,a)$ | Primary | Primary |
| Elegir $a'$ del target | Target | **Primary** |
| Evaluar $a'$ | Target | Target |
| Actualizar pesos | Primary | Primary |
| Copia congelada | Target | Target |

*Fuente: tabla de APP p.41.*

$$\text{DQN target}=R_{t+1}+\gamma\,Q_{\theta'_t}\big(S_{t+1},\ \arg\max_a Q_{\theta'_t}(S_{t+1},a)\big)$$

$$\text{Double DQN target}=R_{t+1}+\gamma\,Q_{\theta'_t}\big(S_{t+1},\ \arg\max_a Q_{\theta_t}(S_{t+1},a)\big)$$

La diapositiva subraya el subíndice del argmax, que es la única diferencia: $\theta'$ frente a $\theta$. Paper: *Deep Reinforcement Learning with Double Q-learning* (arXiv 1509.06461).

**Nota (lectura de las fórmulas).** La primera es la de DQN **con** red objetivo fija; como elige y evalúa la misma red, $Q_{\theta'}(S',\arg\max_a Q_{\theta'}(S',a))=\max_a Q_{\theta'}(S',a)$. El DQN "sin copia" de RLM deep (celda 2) usa $\theta$ en los dos sitios. Las frases "devuelve una acción / devuelve un valor" de la diapositiva describen el uso, no la arquitectura: las dos redes son idénticas y ambas reciben un estado y devuelven un vector de Q-valores. De la principal se toma el argmax y de la target se lee el valor en esa acción.

En código (RLM deep-double, celda 2; BE dqn.py `learn`):

```python
qs1 = self.getQs(s1_list, self.primary_model)      # la principal elige...
q_target = self.getQs(s1_list, self.target_model)  # ...la target evalúa
best_next_action = np.argmax(qs1[k])
targets[i][a] = r + self.gamma * q_target[k, best_next_action]
```

### 8.2 Las evidencias del paper (APP p.42-43)

**APP p.42.** Fila superior: estimaciones de valor frente a pasos de entrenamiento (0-200 millones) en Alien, Space Invaders, Time Pilot y Zaxxon. La curva naranja de DQN queda muy por encima de su valor real (línea horizontal naranja, el retorno descontado que obtiene de verdad su política final). La azul de Double DQN queda cerca de su línea real; en Zaxxon, por ejemplo, DQN estima ~8 frente a un valor real por debajo de 1 (la línea "DQN true value" queda en ≈0,7, incluso por debajo de la de Double DQN, ≈1,1). Fila central: Wizard of Wor y Asterix en escala logarítmica; las estimaciones de DQN se disparan (la curva llega a ~10-20 en Wizard of Wor, con la banda de varianza por encima de 100, y hasta ~80 en Asterix), mientras Double DQN se mantiene estable. Fila inferior: la puntuación en esos dos juegos. Cuando la sobreestimación de DQN se dispara, su puntuación se hunde (Asterix pasa de ~4000-6000 a casi 0 hacia los 100 millones de pasos) y Double DQN sigue más alto (~6000 en Asterix, ~3000-4000 en Wizard of Wor).

**APP p.43.** Barras verticales de puntuación normalizada por juego, Double DQN (azul claro) frente a DQN (gris oscuro), con la línea discontinua "Human" al 100 %. Double DQN supera o iguala a DQN en la gran mayoría de juegos, con mejoras enormes en Video Pinball, Atlantis, Krull, Boxing o Road Runner. En unos pocos (Robotank, Assault y, por poco, Time Pilot) la barra de DQN es más alta. En Double Dunk DQN queda por debajo de 0.

**Lo que dijo el profesor de los gráficos** (T06 ~01:38-01:39):
- DQN asociaba a ciertos juegos un valor muy superior al real, y **cuando metes sesgo también metes inestabilidad** en los entrenamientos.
- En RL hay muchísimos algoritmos y **no hay uno mejor que otro** en general. Pero los que vienen a corregir sesgos de uno existente sí suelen funcionar mejor "por regla general", como Double DQN respecto a DQN.
- Puede que en algún caso no lo haga mejor: como el RL es inestable, si se entrenara otras 10 veces podría cambiar qué curva queda por encima *(dudoso: la transcripción dice "la gráfica azul está por encima de la gris", que en APP p.43 ya es así; la lectura más probable es que repitiendo el experimento el orden podría invertirse en algunos juegos)*.
- Muchos algoritmos que se verán hablan de "copias de copias": todo busca una especie de "doble opinión" para suavizar las actualizaciones de las redes (T06 ~01:39).

Para la diferencia con Double Q-learning tabular (dos tablas que se intercambian el papel al 50 %; APP p.29-30) ver [00_repaso_rl.md](00_repaso_rl.md). Las erratas de las fórmulas de Double DQN en el material del Taller 2 y el pseudocódigo integrado están en [T2 04_doble_dqn_y_duelo.md §2 y §4](../raw/referencias/deep-rl-trading-agents/docs/teoria/04_doble_dqn_y_duelo.md).

---

## 9. Dueling DQN

**Idea.** Cuando hay acciones muy parecidas entre sí, a la red le resulta más fácil discriminar por la **ventaja** de cada acción respecto a una base que por su valor absoluto. Es "una tontería", porque la red es la misma y solo cambia un poco la salida, pero puede funcionar mucho mejor con muchas acciones o con acciones muy parecidas (T06 ~01:40).

APP p.44: $Q(s,a;\theta,\alpha,\beta)=V(s;\theta,\beta)+A(s,a;\theta,\alpha)$. **Nota** (del paper, no explicado en APP): $\theta$ son los pesos compartidos y $\beta$ y $\alpha$ los de las cabezas de valor y de ventaja; este $\alpha$ no es el *learning rate*. El dibujo del paper (*Dueling Network Architectures for Deep Reinforcement Learning*, arXiv 1511.06581) muestra arriba la red DQN, con capas convolucionales y una salida $Q(s,a)$. Debajo está la Dueling DQN, que se bifurca en dos cabezas, $V(s)$ y $A(s,a)$, recombinadas en $Q(s,a)$.

**Ejemplo de la diapositiva**, en el estado $s_6$ con 4 acciones:

| | $a_1$ | $a_2$ | $a_3$ | $a_4$ |
|---|---|---|---|---|
| DQN: $Q(s_6,\cdot)$ | 10 | 9,8 | 11 | 10,2 |
| Dueling: ventajas | 0 | −0,2 | 1 | 0,2 |
| Dueling: valor $V(s_6)$ | 10 | | | |

Cuándo usarlo (APP p.44): con muchas acciones o con acciones muy parecidas.

**Nota (identificabilidad).** $V+A$ no es único: sumar $c$ a $V$ y restarlo a todas las $A$ da la misma $Q$. El paper lo fija restando la media de las ventajas, $Q=V+\big(A-\tfrac1{|\mathcal A|}\sum_{a'}A(s,a')\big)$. Con esa convención el ejemplo sería $V=10{,}25$ y $A=[-0{,}25;\ -0{,}45;\ 0{,}75;\ -0{,}05]$. La diapositiva toma $a_1$ como acción de referencia (ventaja 0), que es la intuición de "ventaja respecto a una base" del profesor. Las diferencias entre acciones, que son lo que decide el argmax, son las mismas en ambas convenciones.

**Nota (error en el notebook `deep-dueling` de clase).** En RLM deep-dueling (celda 2), `network()` construye las dos cabezas (`adv`, `v`, `out = tf.subtract(v, adv)`), pero después ejecuta sin comentar `# Standard out` → `out = Dense(self.na, activation="linear")(capa)`, que **sobrescribe** la salida. El modelo devuelto es un DQN estándar y las cabezas de duelo quedan fuera del grafo. Ese notebook no prueba Dueling. Además, la combinación es $V-(A-\bar A)$ en lugar de $V+(A-\bar A)$. Como $A$ se aprende libremente, eso solo invierte el signo de lo que aprende la cabeza de ventaja y no cambia la capacidad del modelo, pero confunde al leerlo. En RLM deep-rainbow (celda 2) la salida estándar sí está comentada y el duelo está activo, con el mismo signo invertido. El `DDQNAgent` de la práctica no usa Dueling (BE dqn.py network). Ver [09_material_rl_notebooks.md](09_material_rl_notebooks.md).

**Nota (relevancia para ejecución).** En BE solo hay 2 acciones (`actions_fn = {0: _do_nothing, 1: _agg_action}`: no hacer nada u orden agresiva; LAB celda 7; ver [08_entorno_best_execution.md](08_entorno_best_execution.md)). Con dos acciones tan distintas, Dueling aportaría poco: no hay "muchas acciones" y no son "muy parecidas". Tendría más sentido si se amplía el espacio de acciones (p. ej., varios tamaños o niveles de agresividad), donde sí habría acciones con Q-valores muy parecidos en muchos estados, que es el segundo caso de uso de APP p.44.

---

## 10. Las mejoras en el código de clase y de la práctica

**Nota (tabla de conexión, elaboración propia a partir del código).**

| Pieza | RLM `deep` | `deep-fixed` | `deep-double` | `deep-dueling` | `deep-expected-sarsa` | `deep-rainbow` | BE `DDQNAgent` |
|---|---|---|---|---|---|---|---|
| Buffer `deque` + prellenado aleatorio | Sí | Sí | Sí | Sí | Sí | Sí | Sí (LAB celda 16) |
| Muestreo | Uniforme | Uniforme | Uniforme | Uniforme | Uniforme | **CER** | **CER** |
| Red objetivo | No | Sí (evalúa y elige) | Sí | No | No | Sí | Sí |
| Double (principal elige, target evalúa) | No | No | Sí | No | No | Sí | **Sí** |
| Dueling | No | No | No | No (bug, §9) | No | Sí | No |
| Objetivo esperado (Expected) | No | No | No | No | Sí | Sí ("Expected DDQL") | No |
| Pérdida | `mse` | `mse` | `mse` | `mse` | `mse` | `mse` | `mse` |
| Entorno | LunarLander-v2 | ídem | ídem | ídem | ídem | ídem | `BestExecutionEnv` |

Hiperparámetros comunes de RLM (celda 3): `gamma=0.995`, `alpha=0.0001`, `buffer_size=10000`, `batch_size=64`, `hidden_neurons=120`, `learn_after=batch_size`, ε de 1 a 0,05 con decaimiento lineal a lo largo del 90 % de los episodios. `nepisodes=100` en los deep, salvo Rainbow con 1000. Las redes son dos capas densas ReLU de `hidden_neurons`. Detalle en [09_material_rl_notebooks.md](09_material_rl_notebooks.md).

**Nota (pérdida y escala del paso).** Todos construyen `targets = qs` (la predicción actual) y solo cambian la entrada de la acción tomada, así que el error en las demás acciones es 0. Con `loss="mse"`, Keras promedia sobre las `na` salidas (4 en LunarLander-v2, 2 en `BestExecutionEnv`, LAB celda 7), de modo que el gradiente sobre $Q(s,a)$ queda dividido por `na`. Con SGD eso equivaldría a un learning rate `na` veces menor; pero todos compilan con Adam (`optimizer=Adam(learning_rate=alpha)`, RLM deep, celda 2; BE dqn.py l.30-33), que normaliza el gradiente por su segundo momento, así que ese factor constante se cancela casi por completo y el efecto práctico es despreciable. Ninguno usa Huber loss, que APP p.80 recomienda para estabilidad.

---

## 11. Rainbow y combinar mejoras

**Idea.** No hay un algoritmo de RL que predomine. Hay que probar muchos, cientos de variaciones ("¡Probad las vuestras!"), y **combinarlos** ("rainbow") (APP p.91).

Lo que dijo el profesor (T07 ~00:25-00:26):
- De los modelos de RL ninguno predomina. Hay que ver qué necesidades se tienen: si se necesitan sí o sí acciones continuas, se descarta la mitad.
- Un asistente de IA sugerirá unos cuantos, pero hay que sentirse libre de probar variaciones e incluir "alguna pequeña tontería".
- Se pueden hacer "rainbows". Igual que se vieron Double DQN y Dueling DQN, se puede hacer un *Dueling Double DQN* que incluya la filosofía de los dos: "se hace" y "se llama así". Es "una combinación de lo que tú quieras".

CASOS p.45 lista los agentes posibles para el caso de ejecución: Q-Learning, SARSA, Expected SARSA, DQL, DDQL, Dueling, A2C, A3C, aproximación lineal y cross-entropy. En el caso valen todos los algoritmos vistos (T07 ~03:44). A2C y A3C, de la familia actor-crítico, no se desarrollaron en este taller; ver [T2 06_actor_critico_y_a3c.md](../raw/referencias/deep-rl-trading-agents/docs/teoria/06_actor_critico_y_a3c.md).

**El "Rainbow" de clase** (RLM deep-rainbow, celdas 2-3) combina, sobre LunarLander-v2:
- Red con **duelo** (§9).
- Muestreo **CER** (§6.7).
- **Double DQN** (la principal elige, la target evalúa).
- Objetivo **esperado** al estilo Expected SARSA, con esta línea:

$$y=r+\gamma\Big[(1-\varepsilon)\,Q_{\theta'}\big(s',\arg\max_a Q_\theta(s',a)\big)+\varepsilon\,\overline{Q_{\theta'}}(s',\cdot)\Big]$$

Comentarios del código: `# Standard DDQL` (comentado) y `# Expected DDQL` (activo).

**Nota (el Rainbow de la literatura).** *Rainbow* es también el nombre propio de un agente de DeepMind (Hessel et al., 2017) que combina seis extensiones de DQN:
1. Double DQN (§8).
2. Prioritized replay (§6.6).
3. Dueling (§9).
4. **Retornos multi-paso** ($n$-step, §12).
5. **RL distribucional** (C51: aprender la distribución del retorno, no solo su media).
6. **Noisy Nets** (ruido aprendido en los pesos para explorar).

Las cuatro primeras aparecen en APP. Las Noisy Nets aparecen sin nombre en clase, como "redes que meten ruido en los pesos o en las salidas para explorar más activamente" (T06 ~03:30-03:31; *(dudoso)* que se refiriera exactamente a NoisyNet). El RL distribucional no se mencionó. El notebook de clase no incluye PER, n-step, distribucional ni Noisy Nets, y añade el objetivo esperado, que no está en el Rainbow original: es un "rainbow" en el sentido amplio del profesor, una combinación a la carta. El mapa de extensiones de DQN hasta Agent57 está en [T2 07_panorama_drl.md §3](../raw/referencias/deep-rl-trading-agents/docs/teoria/07_panorama_drl.md).

**Apoyarse en otros sistemas y "no todo es RL"** (T07 ~00:23-00:25; APP p.91). La diapositiva dice "Podemos apoyarnos de otros sistemas para facilitar el aprendizaje". Un problema real se compone de muchos pasos y hay que decidir dónde aplicar ML y de qué tipo: RL para un punto del proceso, supervisado para otro (T07 ~00:23).
- **Supervisado** (APP p.91): **generación de características** y **ayuda en la exploración**. Ejemplos de clase (T07 ~00:24:03-00:24:26): agrupar activos por perfil de liquidez para aplicarles algoritmos distintos, o un modelo de **probabilidad de ejecución** de la orden cuyas salidas entran como características en el vector de estado.
- **No supervisado** (APP p.91): clustering y reducción de la dimensionalidad.
- **No todo es aprendizaje por refuerzo** (APP p.91), dos viñetas: "Capa adicional" y "Temas sensibles → Reglas". No todas las características tienen que salir de otro modelo ni todas las decisiones de la IA; se pueden programar las reglas que se quiera (T07 ~00:24:44-00:24:55, ~00:26:06-00:26:10).

**Nota:** agrupar activos por perfil de liquidez es, en rigor, clustering (no supervisado, según la propia APP p.91), aunque en clase se puso como ejemplo de supervisado (T07 ~00:24). Sería supervisado si los grupos se definieran con una etiqueta conocida (p. ej., clasificar en líquido/ilíquido a partir de una clasificación previa).

El profesor anunció que daría alternativas para reducir el alcance del ejercicio de ejecución (T07 ~00:23-00:24; ver [notebook_partida.md](notebook_partida.md) y [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md)). Con todo, habrá "muchísimas pruebas y error"; cuanto más claros estén los conceptos, mejor se podrán aterrizar (T07 ~00:26:15-00:26:27). Insistió en el **conocimiento de negocio** (hay que pensar qué se necesita de verdad y cómo estructurar el problema) y en el valor de un perfil mixto negocio/datos, que sabe qué es realmente importante y qué no (T07 ~00:26:30-00:27:21).

---

## 12. N-step TD (apéndice de APP)

**Idea.** Entre TD(0), que mira un paso, y Monte Carlo, que mira hasta el final, están los métodos de $n$ pasos, con retorno $G_{t:t+n}=R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^nV(S_{t+n})$. Aceleran la propagación de la información (en la cadena $S_0\to S_1\to S_2\to S_3$, TD(0) necesita 3 vueltas para que la recompensa final llegue a $S_0$ y "TD(1)" solo 2), pero a costa de más varianza. A la pregunta de la diapositiva (¿por qué no, por ejemplo, TD(30)?) responde que cada paso real añadido multiplica las trayectorias posibles (árbol binario de APP p.102), y "MC → no sesgado; mucha varianza. TD → sesgado; poca varianza" (APP p.101-102, en el apéndice; no hay constancia en las transcripciones de que se explicaran en clase). La descripción completa de las dos diapositivas, los diagramas de *backup* y la nota sobre la notación "TD(1)" (en la diapositiva significa 2-step TD, no TD(λ = 1) de Sutton y Barto) están en [00_repaso_rl.md §7.4](00_repaso_rl.md).

**Nota (n-step en un DQN).** El retorno de $n$ pasos es una de las seis piezas de Rainbow (§11). Para la práctica, con episodios de cientos de pasos, un $n$ pequeño (3-5) aceleraría la propagación de la recompensa final, a costa de varianza y de que el objetivo $n$-step, calculado con las acciones de una política antigua guardadas en el buffer, deje de ser un objetivo off-policy correcto: queda sesgado salvo que se corrija (p. ej., con *importance sampling* o Retrace). El `DDQNAgent` usa el objetivo de 1 paso (BE dqn.py learn), que no tiene ese problema.

---

## 13. Entrenar un DQN: lo que dicen APP p.87-92 (resumen)

Este bloque pertenece sobre todo a [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md). Lo que toca a este documento:
- **"El desarrollo del aprendizaje por refuerzo funciona igual que el aprendizaje por refuerzo"** (APP p.87): prueba y error.
- **Curva de APP p.88:** recompensa frente a episodios. Sube rápido, cae de golpe ("*catastrophic forgetting*"), se recupera con oscilaciones y vuelve a subir. Va con la cita de Karpathy y la pregunta "¿Y la loss?". En clase (T06 ~04:49:21-04:51:57):
  - *Catastrophic forgetting*: tras mucho entrenamiento, el agente empieza a explorar un camino malo y, por la falta de independencia, lo olvida absolutamente todo. Para que no ocurra hay que tener callbacks y checkpoints con los que recuperar buenos pesos.
  - El error en RL no se interpreta como en supervisado, donde hay unas etiquetas que aprender y, cuando se predicen bien (error cero), se ha terminado. En RL el error da información sobre la **novedad** de las acciones y estados en los que se encuentra el agente.
  - Un pico de error puede ser algo bueno: haber dado con la llave de Montezuma (§6.6), "he encontrado algo útil". Por eso no indica, como en supervisado, si el algoritmo ha terminado de aprender.
  - Errores muy bajos pueden significar que ya aprendió o terminó de explorar, o que está atascado: si no aparece ningún otro pico, ya no está explorando.
  - **Nota:** hay además una razón formal para no leer la pérdida como en supervisado: el objetivo $r+\gamma\max Q(s',\cdot)$ se calcula con la propia red (o su copia) y se mueve al actualizarla (§3, §7), así que una pérdida estable no garantiza una política mejor.
- **APP p.89:** un DataFrame de características por instante ($t_0..t_n$) es "muy cómodo, pero muy ineficiente en RL", a diferencia del supervisado (ver [02_diseno_de_entornos.md](02_diseno_de_entornos.md)).
- **APP p.90:** el overfitting es "distinto" en RL, y a veces interesa, pero "¡no en bolsa!". Cada X steps hay que validar en datos no vistos, comparando train y validación en modo ejecución (greedy), porque los resultados de entrenamiento confunden por la exploración; el modo ejecución "también nos dice si está aprendiendo". Early stopping y checkpoints sirven para recuperar modelos y para ajustar hiperparámetros, y son muy importantes por lo largos que son los entrenamientos.
- **APP p.92, conclusiones:** "muy importante entender las dinámicas y componentes del RL (conceptos)"; más hiperparámetros y recompensas, difícil de entrenar, y el conocimiento de negocio es muy importante; ligeros cambios dan resultados diferentes; aleatoriedad e inestabilidad de los entrenamientos; entrenamientos muy largos, así que hay que optimizar el código, optimizar la búsqueda de parámetros (el grid search es aún más inviable) y apoyar el entrenamiento en otros sistemas (§11). Con todo, mucha prueba y error.

Consejos equivalentes del Taller 2 en [T2 03_dqn.md §10](../raw/referencias/deep-rl-trading-agents/docs/teoria/03_dqn.md) y [T2 07_panorama_drl.md §10](../raw/referencias/deep-rl-trading-agents/docs/teoria/07_panorama_drl.md).

---

## 14. Segunda Parte: los papers de DeepMind y la evolución del campo

**Idea.** Tras el descanso, el profesor abrió la Segunda Parte (APP p.46) con una diapositiva sobre cuánto se publica en RL. El mensaje: el campo explotó con las redes neuronales, sigue abierto y es muy empírico, así que hay que **entender bien los conceptos básicos** y, a veces, simplemente **"hacerlo funcionar"** (APP p.47; T06 ~02:07-02:10).

**APP p.47, papers de RL de DeepMind por año** (fuente: deepmind.com/research filtrando por año; DeepMind dejó de publicar esta serie después de 2020):

| Año | 2012 | **2013** | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|---|---|---|
| Papers | 1 | **4** | 16 | 52 | 76 | 127 | 206 | 234 | 164 |

Una flecha marca 2013 en negrita. La diapositiva añade:
- "Hoy sigue creciendo": de ≈130 a ≈1200 papers aceptados entre 2017 y 2025 en 5 grandes conferencias de IA (fuente: *Topical Phase Transitions in Artificial Intelligence Research*, arXiv 2026, Fig. 9(a), valores aproximados).
- "2015-17: Double DQN, sparse rewards → hoy: LLMs, DeepSeek-R1".
- "Ideas de lo más variopintas".
- "RL funciona a veces sin saber muy bien por qué → Entender bien los conceptos básicos".
- "En ocasiones → Hay que hacerlo funcionar".

**Lo que contó** (T06 ~02:07-02:10), en resumen: con las redes neuronales y convolucionales el RL empezó a resolver juegos como Breakout y la comunidad "explotó" *(dudoso: la transcripción dice "en el 2010"; la flecha de la diapositiva marca 2013, año del primer DQN en Atari, Mnih et al.; la versión de Nature es de 2015)*. DeepMind llevaba la vanguardia; hoy los papers van más hacia los LLM. Muchos son muy específicos ("si haces esta tontería de repente funciona todo") y el porqué se cuenta a posteriori como *storytelling*. Por eso animó a **probar ideas propias** con sentido y a apoyarse en principios y conceptos básicos. El desarrollo completo de este tramo está en [00_repaso_rl.md §14](00_repaso_rl.md).

**Nota (cálculos):** los papers de DeepMind suman 880 entre 2012 y 2020, con un pico en 2019 (234); el salto 2013→2019 es de casi 60×. El dato de las conferencias, ≈130→≈1200, supone multiplicar por ~9 en ocho años. DeepSeek-R1 (2025) se cita como ejemplo actual porque su capacidad de razonamiento se entrenó con RL a gran escala sobre un LLM.

**APP p.48, el esquema de la Segunda Parte** (sirve de índice para el resto de documentos):

| Bloque | Piezas | Documento |
|---|---|---|
| Qué ve y qué hace el agente | Estados, acciones, recompensas | [02_diseno_de_entornos.md](02_diseno_de_entornos.md), [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md) |
| Cómo aprende | Exploración (ε), step size (α), gamma (γ), experience replay | [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md), [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md), §6 de este documento |
| Cómo lo ponemos en marcha | Definición del entorno, entrenamiento, retos | [02_diseno_de_entornos.md](02_diseno_de_entornos.md), [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md) |

"Para cada pieza: principios prácticos, errores típicos y ejemplos en bolsa" (APP p.48). En clase: la Segunda Parte retoma los conceptos del principio (acciones, recompensas, gamma...) para discutir cómo definirlos (T06 ~02:06-02:07).

**Cuándo compensa el RL, DeepMind y el futuro** (T07 ~04:13-04:23). Desarrollado en [00_repaso_rl.md](00_repaso_rl.md) (bloque "A favor y en contra del RL" y §14); aquí, en resumen:
- **El benchmark es la solución clásica** (matemática o supervisada): a veces el ML clásico bien aplicado ya explica el 80 % y no hace falta montar un RL; el RL debería superarla (T07 ~04:14:13-04:15:14; ver [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md)).
- **A favor del RL:** el ruido en las etiquetas o en la toma de decisiones, las interacciones con el entorno que el supervisado no recoge y los escenarios infinitos de un simulador de mercado, donde no hay etiquetas posibles. Con RL se puede resolver casi cualquier problema supervisado (también el del paraguas) y es más potente por esas interacciones (T07 ~04:16:14-04:16:51, ~04:18:24-04:18:46).
- **En contra:** sus escollos, que son las recompensas, la exploración y el entorno; si no lo son y apetece probar, adelante (T07 ~04:16:37-04:16:57). Tarta de LeCun: bizcocho no supervisado, glaseado supervisado, guinda RL.
- Alguien en el aula, probablemente un alumno *(dudoso: todo el audio está etiquetado igual)*, añadió que lo ve útil cuando hay un patrón histórico que se sabe no óptimo y una métrica clara de qué es mejor (T07 ~04:17:44-04:18:14).
- **DeepMind** *(dudoso: la transcripción dice "Big Man" / "Bigmail")*: según el profesor, su fundador sostenía que el futuro de la IA estaba en el RL, porque los problemas reales son infinitos y no responden a etiquetas finitas; la limitación era simular entornos fuera de los juegos (una rotonda para un coche autónomo). Con ChatGPT, el equipo, financiado por Google, se reorientó en parte a los LLM, y la combinación generativo + RL (*feedback* humano) sería el camino (T07 ~04:18:59-04:21:03). **Nota:** "Bigmail" es casi seguro la transcripción de "DeepMind". La persona puede ser Demis Hassabis (cofundador y director) o, por la tesis de que el futuro está en el RL, David Silver; no se puede asegurar.
- **LLM jugando a Pokémon Rojo Fuego** *(dudoso: "germina y el Cloth", probablemente Gemini y Claude)*: interpretan contexto y deciden sin RL, aunque todavía se pierden; la idea es añadirles feedback. El profesor matizó que lo saca de vídeos divulgativos de 10 minutos y que no es experto (T07 ~04:21:48-04:23:12).

Los hitos de DeepMind (AlphaGo, AlphaZero, AlphaTensor...) vistos en el Taller 2 están en [T2 07_panorama_drl.md §6](../raw/referencias/deep-rl-trading-agents/docs/teoria/07_panorama_drl.md).

---

## 15. Conexión con la práctica (resumen)

El agente que se entrega con la práctica es un **Double DQN con CER** (BE dqn.py):
- Dos redes idénticas: dos capas densas ReLU de `hidden_neurons` y salida lineal con una neurona por acción.
- `deque(maxlen=buffer_size)` y `experience()`, que guarda `s1 = None` si el episodio termina.
- `sample()` con CER.
- `learn()` al estilo Double DQN, `update_target()` copiando pesos y pérdida `mse`.
- `act()` fuerza la acción 1 en el primer paso del episodio (`if self.env.state_pos == 0: return 1`).

LAB fija `learn_after = batch_size` y `update_target_after = 10 * learn_after` (celda 12) y prellena el buffer **entero** con experiencias aleatorias (celda 16), aunque en clase se dijo que no hace falta llenarlo entero (§6.4). La celda 11 de LAB recuerda que el *buffer size* sirve para obtener un conjunto "aproximadamente iid" y que α no debe decaer a cero porque el problema es no estacionario.

**Nota (errata en BE):** `getQ(self, s, a)` llama a `self.getQs(s)` sin el argumento `model`, que es obligatorio en el método estático `getQs(s, model)`. Si se usara, lanzaría `TypeError`; el bucle de entrenamiento no lo llama. El análisis línea a línea, la acción forzada y los riesgos están en [08_entorno_best_execution.md](08_entorno_best_execution.md). Las decisiones de diseño de la práctica (CER sí o no, tamaño del buffer en episodios, Dueling, Huber, PyTorch frente a TF) están en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md).

---

## 16. Erratas, discrepancias y puntos dudosos

| # | Dónde | Qué pone | Corrección o lectura razonada |
|---|---|---|---|
| 1 | APP p.80 | $P(i)=p_i^\alpha/\sum_k p_i^\alpha$ | El denominador es $\sum_k p_k^\alpha$ |
| 2 | T06 ~04:32 | "Si alfas, 1 es su uniforme de profesional fastero" y "al Fast 1 le da todo el peso" | *(dudoso)* Lectura probable correcta (α = 0 uniforme, α = 1 prioriza totalmente por error); solo confusa por la transcripción (§6.6) |
| 3 | APP p.41 | "DQN target" con $\arg\max_a Q_{\theta'}$ | Es DQN **con red fija**; equivale a $\max_a Q_{\theta'}(S',a)$ (§8.1) |
| 4 | APP p.41 | La target "recibe una acción y devuelve un valor" | Las dos redes reciben un estado y devuelven el vector de Q; cambia qué se lee de cada una (§8.1) |
| 5 | APP p.44 | $V+A$ con la ventaja de $a_1$ = 0 | Convención de acción de referencia; el paper resta la media de las ventajas (§9) |
| 6 | RLM deep-dueling, celda 2 | La salida estándar sobrescribe la de duelo | El notebook entrena un DQN estándar; además usa $V-(A-\bar A)$ (§9) |
| 7 | CASOS p.49-50 | La columna $S_1$ queda a ceros en la política | Su máximo es $a_3=5$; debería ser $S_1\to a_3$ (§2.3) |
| 8 | T06 ~02:07 | "En el 2010" | La flecha de APP p.47 marca 2013, año del primer DQN en Atari (§14) |
| 9 | T06 ~04:33 | PER "surge" de Montezuma/RND | RND (2018, OpenAI) es posterior a PER (2015); relación conceptual (§6.6) |
| 10 | APP p.101 | "TD(1)" = 2 pasos | Choca con la notación TD(λ); mejor "2-step TD" (§12) |
| 11 | APP p.35 frente a RLM | Gráficos en CartPole | Los notebooks lineales de clase usan LunarLander-v2 (§4.1) |
| 12 | BE dqn.py `getQ` | `self.getQs(s)` sin `model` | `TypeError` si se llama; no se usa en el bucle (§15) |
| 13 | T06 ~01:39 | "La gráfica azul por encima de la gris" tras reentrenar | *(dudoso)*; la lectura más probable es que el orden podría invertirse en algunos juegos (§8.2) |
| 14 | T06 ~04:36 | Autor de CER, "suto" | Sutton, coautor con S. Zhang (§6.7) |
| 15 | T06 ~01:24 | Pesos del ejemplo lineal (0,2, 0,7, 1,2) | Dictado incompleto; el ejemplo vale por el mecanismo 20 → 30 → 22 (§4.1) |
| 16 | T07 ~00:24 frente a APP p.91 | Agrupar activos por perfil de liquidez como ejemplo de supervisado | Agrupar es clustering, que APP p.91 sitúa en no supervisado (§11) |
| 17 | T06 ~03:58-04:01 | 60 fps × 60 s × 2,5 min × 100 partidas ≈ 1 millón | Cifras dudosas en clase; con *frame skip* 4 el millón serían ~440 partidas; vale como método, no como derivación (§6.2) |
