# Algoritmos de ejecución: el problema de negocio

> **Fuentes:**
> - **CASOS** = `docs/raw/MIAX14/casos_practicos.pdf` ("Casos Prácticos", Jesús Sanz). Leídas completas con imagen: p.3 (profit-seeking frente a ejecución) y **p.15-29** (bloque *Execution Algorithms* hasta el objetivo del caso). Consultadas para enlazar: p.31, p.37-41, p.60-61.
> - **T07** = `docs/raw/26 11 07.pdf`, transcripción automática de la clase del 07/11/2026. Tramos: ~00:23-00:26, ~00:44-00:46, **~01:35:44-02:02:24** (introducción a la ejecución, dilema del trader, tipos, TWAP/VWAP/POV, costes y uso), **~02:36:45-02:54:45** (cómo mejorar un TWAP, reglas, ML), ~02:54:45-02:58:30, ~03:02:40-03:08:25, ~03:27:56-03:30:00, ~03:42:00-03:43:50, ~03:58:37-04:16:00, ~04:52:12-04:56:30.
> - **T06** = `docs/raw/26 11 06.pdf`, transcripción de la clase del 06/11/2026. Tramos: ~00:02-00:06, ~00:34-00:38, ~02:30-02:33, ~02:57-03:00, ~03:21-03:26.
> - **APP** = `docs/raw/MIAX14/applying_RL_MIAX.pdf`, p.12-15 (cuándo usar RL; impacto de nuestras órdenes).
> - **BE** = `docs/raw/MIAX14/best-execution/README.md` (completo) y la figura `docs/raw/MIAX14/best-execution/images/twap_model/twap_reproduction_dqn.png`.
> - **T2** = `docs/raw/referencias/deep-rl-trading-agents/docs/teoria/00_fundamentos_mdp.md` (comisión del Taller 2, solo para comparar escalas).
> - **LAB** = `docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`, celdas 1 (resumen, objetivo y VWAP), 2 (datos), 7 (docstring de la recompensa), 8-9 (TWAP).
> - **ENUNCIADO** = `docs/raw/MIAX14/ejercicios_taller_rl.docx` (ejercicios 2 y 4, para conectar).
>
> **Convenciones:** "(T07 ~01:38)" = hora aproximada en la transcripción; todo lo que viene de ellas es **paráfrasis** de un texto automático muy ruidoso, y lo que no se entiende bien va marcado *(dudoso)* con sus lecturas posibles. "(CASOS p.18)" = página del PDF. "(LAB celda 8)" = celda del notebook contando desde 0. **pb** = punto básico = 0,01 % = $10^{-4}$. **Nota:** = contenido propio (cálculos, conexiones, correcciones), no del material de clase. Este documento cubre el **negocio** y los **algoritmos clásicos**; el diseño RL del caso (episodios, estado, acciones, recompensas, buffer, resultados) está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).
>
> Escrito el 10/11/2026.

---

## 1. Trading algorítmico: inversión frente a ejecución

**Idea.** "Trading algorítmico" agrupa dos problemas distintos que en papers y conversaciones se confunden: decidir **qué** comprar o vender (y cuándo) y **materializar** esa decisión en el mercado. Este taller trata del segundo (CASOS p.3; T07 ~00:44:44-00:45:50).

| | Algoritmos de inversión (*profit-seeking*) | Algoritmos de ejecución (*execution*) |
|---|---|---|
| Qué responden | Qué comprar o vender y cuándo (CASOS p.3) | Cómo ejecutar una orden ya decidida (CASOS p.3) |
| Objetivo | "Ganar dinero" (o la utilidad que defina cada uno) (CASOS p.3; T06 ~00:05:37) | "Ejecutar la orden" lo mejor posible (CASOS p.3) |
| Ejemplo de clase | El gestor decide: "cómprame 1.000.000 de títulos de Santander hoy" (T07 ~00:45:42) | Recoge esos títulos y los ejecuta al mejor precio posible (T07 ~00:45:50; T06 ~00:05:37) |
| Dónde se trata | [05_profit_seeking.md](05_profit_seeking.md) | Este documento y [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md) |

- CASOS p.3 define el conjunto como "estrategias programadas para ejecutar órdenes electrónicamente".
- Reparto del tiempo de la clase del 07/11: unos 20 min de inversión a alto nivel y hora y media de ejecución (T07 ~00:46:18). El taller (la práctica) es "principalmente" el caso de ejecución (T06 ~00:06:08).
- **Por qué este caso.** Jesús Sanz trabajó en el Instituto BME con el antiguo director del máster haciendo algoritmos de inversión, un robo-advisor y el caso del taller (T06 ~00:03:27-00:03:53). Su TFM de 2019 se reorientó (antes era sobre fondos de inversión) porque BME tenía **algoritmos de ejecución basados en reglas** y quería probar a meter machine learning; pasaba las mañanas leyendo papers y diseñando recompensas y por la noche lanzaba experimentos (72 en paralelo en una máquina de unos 700 GB) (T06 ~02:57:29-02:59:12). Muchos de los consejos del taller vienen de ahí (T06 ~02:59:12).
- **Por qué no llegó a producción** (T06 ~02:59:37-03:00:16): BME perdía cuota en algoritmos de ejecución frente a grandes operadores (cita a Morgan Stanley) con interfaces mucho mejores (bromea con una interfaz "de los años 80"), y a los brókers, hablando con ellos, les daba igual ganar 2 o 3 pb: quieren pulsar un botón y ejecutar sus decenas de órdenes de la hora de forma sencilla, porque el dinero no es suyo sino del cliente.

**Nota:** esta anécdota es el contrapunto de negocio de todo lo que sigue: una mejora de 2-3 pb es real (sobre una orden de 10 M€, 1 pb = 1.000 €; ver §3), pero quien decide si se adopta (el bróker) valora más la sencillez y la estabilidad que la media (ver también la preferencia por el POV en §11).

---

## 2. De la decisión a la orden: DMA, orden CARE y órdenes hijas

**Idea.** Una orden pequeña va directa al mercado; una orden grande no se puede lanzar de golpe porque su propio volumen movería el precio en contra, así que se trocea en el tiempo. Ese troceo y su gestión es lo que hace un algoritmo de ejecución.

### 2.1 Lo que dice el README del repositorio (BE README.md, *Introduction*)

- Una vez que un agente (persona física o institución) decide invertir o deshacer una inversión, la decisión se materializa enviando una instrucción de compra o venta a un **bróker**.
- **Órdenes DMA** (*direct market access*): las de importe pequeño se envían directamente al mercado a través del bróker; pasan sus controles, pero en general no se modifica ningún parámetro (tipo, precio, volumen, momento de envío, restricciones).
- **Órdenes CARE** ("órdenes a cuidar"): las suficientemente grandes no van directas al mercado por el impacto que tendría su volumen. Llegan al bróker para que este las **gestione a lo largo del tiempo** y evite impactar el mercado.
- Históricamente las gestionaban **traders u operadores** que troceaban a mano la orden CARE en órdenes más pequeñas (**órdenes hijas**, *child orders*), las enviaban y las gestionaban durante la sesión. Con la electronificación de los mercados, los brókers desarrollaron **algoritmos de ejecución** que hacen ese troceo y gestión de forma sistemática.
- **Definición:** un algoritmo de ejecución es un sistema automatizado que recibe una instrucción compleja del cliente y la transforma en una serie de órdenes, modificaciones y cancelaciones que se envían en el tiempo a distintos mercados y brókers, con el objetivo de cumplir la instrucción del cliente, es decir, **maximizar el resultado respecto a un benchmark concreto**.
- Los más usados son los de **slicing** (reparten el volumen de una orden grande en el tiempo) y los **Smart Order Routers** (ver §5).
- **Propósito del repositorio** (BE README.md, *Motivation of the project*): es la base de un proyecto de unas **10 horas** para recorrer el proceso completo de un problema RL real. La mayoría de problemas de la asignatura son simuladores de juegos con un MDP estacionario, y la simulación suele quedar fuera porque es programación *ad hoc* para cada problema. Aquí los alumnos construyen un **simulador de libro de órdenes naive** (colocar órdenes de un volumen y ejecutarlas a un precio) y la tarea principal es desarrollar **las recompensas y los modelos RL** para obtener algoritmos con un rendimiento **similar** al de los algoritmos tradicionales que se usan hoy. LAB celda 1 llama igualmente al simulador "relativamente naive", aunque funcional.

**Nota:** el primer listón del taller es **igualar**, no batir, al benchmark (ej. 2, replicar el TWAP); batirlo corresponde al ej. 4 (ver §7 y [notebook_partida.md](notebook_partida.md)).

### 2.2 Las diapositivas y la clase

- **CASOS p.16** (gráfico: un fajo de billetes y el logo de BP): orden CARE de **1.000 €** → "No hay necesidad" de algoritmo. En clase: con 1.000 € da igual cómo la ejecutes (T07 ~01:37:15-01:37:19).
- **CASOS p.17** (diagrama): orden CARE de **10 millones €** (fajo mayor y logo de BP) → caja *Execution Algorithm* → cinco flechas hacia una barra *Child Orders*. Debajo, una línea temporal de la sesión **09:00 am - 17:30 pm** con dos marcas: **Arrival Time** (llegada de la orden) y **End Time** (fin del horizonte), y una flecha discontinua entre ambas: el algoritmo solo actúa en esa ventana. Al pie, la fórmula del VWAP (§3).
- En clase: con 10.000.000 no puedes meter la orden de golpe; o la ejecutas como bloque o la partes en el tiempo en órdenes hijas, dentro de un tiempo finito que empieza en la llegada de la orden (T07 ~01:37:26-01:37:55).
**Nota (terminología):** en la jerga del sector la orden completa se suele llamar **orden madre** (*parent order*) frente a las hijas; el material del taller no usa ese término: habla siempre de orden CARE y de órdenes hijas (*child orders*) (BE README.md; CASOS p.16-17 y p.29; T07 ~01:37:29).

**Nota (terminología):** CASOS p.16 llama "CARE Order" a la orden de 1.000 €, pero según el propio README una orden así de pequeña iría como **DMA**; la diapositiva usa "CARE" en sentido amplio para contrastar con la de 10 M€. En el notebook, "orden CARE" es siempre la orden grande que el agente trocea (LAB celda 1).

### 2.3 Las tres decisiones de cada orden hija

Cualquier algoritmo de troceo decide, para cada hija (T07 ~02:40:19-02:40:59; CASOS p.37):

| Decisión | Opciones | Qué hace el TWAP clásico |
|---|---|---|
| **Momento** | cuándo se lanza dentro del horizonte | fijo (cada 10 min en el ejemplo) |
| **Tamaño** | cuántos títulos | fijo (orden / nº de hijas) |
| **Tipo** | **agresiva** o a mercado (cruza el spread y ejecuta contra lo que hay en el libro), **pasiva/limitada** (se queda en el libro a un precio y espera a que alguien la agreda), **pegada** (*pegged*: sigue una posición o referencia del libro) | se puede configurar para que sea siempre de un tipo, p. ej. siempre agresiva: "lanza 1.000 y lo que haya" (T07 ~02:40:38-02:40:59) |

El diseño detallado de acciones (mantener, cancelar, posiciones del libro, agresividad de terceros) está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md). La simplificación de la práctica (el agente solo decide el **momento**; la hija es siempre **agresiva** y de **1 título**) se explicó en T07 ~03:02:40-03:07:00 y se desarrolla allí.

### 2.4 Por qué el impacto convierte esto en un problema de RL

- **APP p.14** (dos gráficos, precio de 80 a 110 € entre el día 1 y el 16): a la izquierda, un modelo supervisado etiqueta sobre el histórico "compra" en el mínimo (88) y "venta" en el máximo (104), asumiendo que nuestras órdenes no mueven el precio; a la derecha, si de verdad compramos en el mínimo, el precio sube y la serie posterior cambia (curva "con impacto" frente a "histórico"): las etiquetas ya no valen y hay que modelizar el impacto.
- **APP p.15**: la pregunta clave es si nuestras decisiones cambian los datos que veremos después. Entre los casos de RL está explícitamente la **ejecución de órdenes grandes**: cada tramo mueve el precio del siguiente.
- En clase (T06 ~00:34:52-00:36:48): si operas con capital relevante, al comprar no habrías visto ese precio y además impactas lo que viene después; esas consecuencias hay que modelizarlas en el entorno, y un entorno que las modelice perfectamente "lo tiene todo hecho". El impacto también es **interno**: el estado mezcla variables externas y privadas, y tus acciones cambian tu propio estado. Con posiciones pequeñas, el supervisado puede bastar (T06 ~00:37:43).

Desarrollo general (RL frente a supervisado, estado privado/externo): [00_repaso_rl.md](00_repaso_rl.md) y [02_diseno_de_entornos.md](02_diseno_de_entornos.md).

**Nota:** en la práctica el simulador asume **impacto nulo** (BE README, nota final; ENUNCIADO ej. 2: "no hay impacto de mercado ni costes de transacción"). Lo que el agente puede mejorar es solo el **timing** (LAB celda 1: "optimizar el timing de nuestras órdenes"), no el impacto propiamente dicho. Esto es coherente con órdenes de 1 título (§14). APP p.15 da la justificación: entre los casos en que basta el supervisado pone las "señales con posiciones pequeñas", porque en activos líquidos el impacto es despreciable; por eso la simplificación de impacto nulo solo es creíble con hijas de 1 título.

---

## 3. VWAP: métrica y algoritmo

**Idea.** El VWAP (*Volume Weighted Average Price*) es el precio medio ponderado por volumen. Sirve a la vez como **métrica** (qué precio he conseguido, qué precio ha conseguido el mercado) y como nombre de un **algoritmo** (§9) (T07 ~01:38:18).

$$\mathrm{VWAP}(n)=\frac{\sum_{t=t_1}^{t_n} \mathrm{Price}_t\cdot \mathrm{Volume}_t}{\sum_{t=t_1}^{t_n}\mathrm{Volume}_t}$$

(CASOS p.17, p.20 y p.29; LAB celda 1 lo escribe como $\sum_{i=1}^{T}p_i v_i/\sum_{i=1}^{T}v_i$.)

- **Algo VWAP**: con los precios y volúmenes de **mis** hijas: cada hija tiene un volumen, se multiplica por su precio, se suma y se divide por el total (T07 ~01:38:05-01:38:26).
- **Mkt VWAP**: con todas las ejecuciones del **mercado en el mismo periodo** de ejecución (de *Arrival Time* a *End Time*), no de toda la sesión (T07 ~01:45:33).
- **VWAP slippage (pbs)**: diferencia entre ambos, en puntos básicos, usada como **medida del impacto de mercado** (CASOS p.20). LAB celda 1 la define como "la desviación, en puntos básicos, del VWAP del mercado en comparación con el VWAP de nuestro algoritmo" y la llama "el baseline habitual en el sector".
- **Criterio de éxito (compras):** si durante las (por ejemplo) 2 horas de ejecución mi precio medio ha sido **inferior** al precio medio del mercado, la ejecución ha tenido éxito (T07 ~01:45:40-01:45:49; CASOS p.20: "queremos pagar menos que el mercado").

**Nota (fórmula y signo).** Ninguna fuente escribe la fórmula del slippage con su signo. Una forma coherente con el notebook (la recompensa por defecto es "la diferencia entre el `episode_vwap` y el precio de la acción, dividida entre `episode_vwap`", LAB celda 7) y con el histograma de resultados (CASOS p.60, positivo = mejor) es, para **compras**:

$$S_{\text{compra}}=\frac{\mathrm{VWAP}_{\text{mkt}}-\mathrm{VWAP}_{\text{algo}}}{\mathrm{VWAP}_{\text{mkt}}}\cdot 10^{4}\ \text{pb}\qquad(S>0 \Rightarrow \text{he pagado menos que el mercado})$$

y con el signo cambiado para ventas. Pero hay dos convenciones en clase (ver §11.2): el profesor vino a decir que el VWAP slippage es un coste (la transcripción corta la frase), que con costes a la derecha es peor y que en el gráfico de CASOS p.27 se le dio la vuelta al eje para que "a la derecha" fuese mejor (T07 ~01:58:54-01:59:13). **En el informe hay que declarar el signo explícitamente.**

**Nota (escala).** 1 pb = 0,01 %. Sobre un precio de 7,50 € (el del ejemplo de clase), 1 pb = 0,00075 €; sobre una orden de 10 M€, 1 pb = 1.000 €, de modo que los 3 pb de coste medio de un TWAP (§11) son unos 3.000 € y los 20 pb de un mal día, 20.000 €.

---

## 4. El dilema del trader

**Idea.** Ejecutar deprisa encarece por **impacto**; ejecutar despacio expone a que el precio se mueva en contra (**riesgo de ejecución o de timing**). El punto óptimo depende de la **urgencia**.

### 4.1 La diapositiva (CASOS p.18)

Gráfico de la izquierda: eje X "Tiempo Ejecución (min)" (marcas 20, 40, 60, 80); eje Y "Coste (pbs)" (20 a 80).
- **Market Impact** (azul claro): parte de ~88-90 pb con tiempo casi nulo y **decrece** de forma convexa hasta ~15 pb a los 80-90 min.
- **Execution (timing) risk** (azul oscuro): parte de ~0 y **crece** de forma cóncava hasta ~80-83 pb.
- Ambas se cruzan hacia los **20 min a ~47 pb**.
- **Coste total (pbs)** (gris discontinuo): forma de **U** suave, empieza en ~93, mínimo de ~72 hacia los 33 min y sube a ~88 al final.

A la derecha, el título *Traders' Dilemma* y la fórmula:

$$\text{Coste total (pbs)}=\text{Market Impact}+\lambda\cdot\text{Execution Risk},\qquad \lambda=\text{nivel de urgencia}$$

"Punto óptimo → depende de la urgencia de la operación". Abajo: el nivel de urgencia sube o baja según la estrategia: *short-term alpha* (oportunidades a corto plazo), *long-term alpha* (a largo plazo), rebalanceo, necesidades de efectivo, estrategias de cobertura, etc.; y una llave que lo resume en dos factores: **Alpha Decay** ("erosión del short-term alpha una vez tomada la decisión de inversión") y **Reference Prices** ("¿hay algún precio de referencia que necesites conseguir (e.g., close, open o AP)?").

**Nota (lectura del gráfico):** la curva gris es esquemática. Con $\lambda=1$ no es la suma de las otras dos (hacia los 20 min la suma daría ~95 pb y la gris marca ~76). Con $\lambda\approx0{,}5$ se acerca a su mínimo (hacia los 33 min, ~38+0,5·63≈70 pb frente a ~72 dibujados), pero no al tramo final (a los 80 min, ~15+0,5·82≈56 pb frente a ~88): ningún $\lambda$ constante reproduce las tres curvas. Lo que transmite es cualitativo: un mínimo interior cuya posición se desplaza con $\lambda$. Con $\lambda$ alto (mucha urgencia) el riesgo pesa más y el óptimo se mueve a la izquierda (ejecutar antes); con $\lambda$ bajo, a la derecha.

### 4.2 Lo que explicó el profesor (T07 ~01:38:33-01:43:01)

- Analogía con inversión: igual que en bolsa se pondera retorno frente a riesgo según la aversión al riesgo, aquí se pondera el coste total según tu aversión a que **no se ejecute** tu orden como querías (T07 ~01:38:39-01:39:04).
- **Impacto de mercado:** si ejecuto mucho ahora, barro muchas posiciones del libro y consigo muy mal precio; si lo reparto en el tiempo, el impacto baja (T07 ~01:39:17-01:39:25).
- **Riesgo de ejecución (timing):** el precio que me interesaba era el de 7,50 € que veo ahora; si espero demasiado, puede evolucionar en la dirección que no quiero y la estrategia que respondía a ese precio no se ejecuta (T07 ~01:39:30-01:39:49).
- **Urgencia por alpha decay:** es la erosión del retorno potencial de la estrategia si pasa el tiempo hasta ejecutar. Ejemplo extremo con urgencia máxima: una estrategia que explota una pequeña ineficiencia que otros también pueden capturar; se oye algo como *high-frequency trading* *(dudoso: la transcripción dice "Shophire, Free Gunsy Trading"; la lectura HFT encaja con "urgencia máxima")* (T07 ~01:40:09-01:40:50). En el otro extremo, un inversor fundamental con un horizonte de 20 años al que le parece bien el precio de 7,50 €: reparte sus 20.000.000 durante una semana, poco a poco con su bróker (T07 ~01:40:55-01:41:19).
- **Urgencia por precios de referencia:** importa p. ej. a fondos que se miden contra una referencia (cita índices como el S&P 500 o el Ibex 35 *(dudoso)*) o contra el precio de cierre: quieren un precio muy cercano a esa referencia y por eso tienen más urgencia cerca del cierre de la sesión. Si tienes un precio muy concreto que quieres conseguir y lo ves en el mercado, más urgencia; si no, le das más tiempo, porque no sabes hacia dónde irá (T07 ~01:41:21-01:42:11).
- **Asimetría a largo plazo:** si espero, unas veces me saldrá mejor (el precio fue a favor) y otras peor; tras operar muchas veces quedo aproximadamente **neutro** respecto al riesgo de timing cuando mi alpha es de largo plazo. Lo que sí pago siempre es el **impacto**: si de forma consistente quiero comprar volúmenes enormes en la próxima hora, al final del año habré pagado muchísimo impacto (T07 ~01:42:20-01:42:51).
- El README atribuye el término a Kissell y Glantz, *Optimal Trading Strategies: Quantitative Approaches for Managing Market Impact and Trading Risk* (2003): si la orden se ejecuta rápido hay mucho impacto y precios poco competitivos; si tarda mucho, el precio puede evolucionar en contra y se asume un gran riesgo de mercado (BE README.md).

**Nota:** el README escribe "Robert Kissel"; el autor es Robert **Kissell**. Errata menor sin efecto.

**Nota (conexión con la práctica):** el caso de clase es de **baja urgencia** (§6): $\lambda$ pequeño, horizonte dado por el cliente, sin precio de llegada. Por eso el algoritmo de referencia es de calendario y no uno de *arrival price*.

---

## 5. Tipos de algoritmos de ejecución

Tabla a partir de CASOS p.19 (texto de la diapositiva) y T07 ~01:43:01-01:44:31 (explicación). En la diapositiva la fila *Scheduled* está **resaltada en azul**: es el foco del caso.

| Tipo | Qué hace (CASOS p.19) | Cuándo se usa (CASOS p.19) | Explicación en clase |
|---|---|---|---|
| **Scheduled** (TWAP, VWAP, POV) | Mandan órdenes al mercado siguiendo un calendario | Impacto reducido + menor nivel de urgencia | Tienes tanto tiempo para ejecutar y, según el algoritmo, se sigue un patrón u otro (T07 ~01:43:05-01:43:18) |
| **Opportunistic** (buscadores de liquidez) | Buscan alta liquidez a un precio favorable; cruzan mercados, incluidos *dark pools* | El activo es *thinly traded* (poco negociado) o la liquidez es episódica | Aprovechan momentos de liquidez concreta: "aquí hay mucho volumen, me lo quedo" (paráfrasis, T07 ~01:43:23) |
| **Arrival Price** | Manda más órdenes al principio para asegurar un precio próximo al de llegada (AP) | Alta aversión al riesgo o alto nivel de urgencia | Mucha urgencia; ejecutan lo máximo posible al principio para acercarse al precio que querías (T07 ~01:43:34-01:43:54) |
| **Dark Strategies / agregadores de liquidez** | Operan fuera de los *lit venues* (mercados con libro visible) | La orden tendría un impacto considerable (baja liquidez o *bid-ask spread* amplio) | Buscan en sitios no *lit*, *dark pools*, donde se esconde volumen que no se ve en pantalla (T07 ~01:44:04-01:44:15) |
| **Smart Order Routers (SORs)** | Encuentran el mejor precio entre los distintos mercados y/o el mercado con mayor probabilidad de ejecución de una orden limitada | Varios mercados ofreciendo precios | Comparan distintos *venues* donde se puede operar (T07 ~01:43:54-01:44:04) |

- El profesor se centra en los primeros, los de calendario, "más sencillos de entender" (T07 ~01:44:27-01:44:31).
- El README destaca *slicing* y *SOR* como los más usados (BE README.md).

**Nota:** los *scheduled* de CASOS p.19 (TWAP, VWAP, POV) encajan en la definición de *slicing* del README ("distribuir el volumen de una orden grande en el tiempo"), y es a los de *slicing* a los que el README asocia el dilema del trader (§4); ni el README ni CASOS p.19 hacen explícita esa equivalencia.

**Nota (glosario mínimo):** *lit venue* = mercado con libro de órdenes público; *dark pool* = plataforma sin libro visible antes de la ejecución; *AP* = *arrival price*, precio en el momento de llegada de la orden; *thinly traded* = activo con poca negociación. Ver [11_glosario.md](11_glosario.md).

---

## 6. El objetivo aterrizado del caso

**Idea.** Antes de diseñar nada, hay que aterrizar el objetivo en una fórmula medible ("1º → Aterrizar objetivo", CASOS p.20). Es la aplicación directa de lo dicho el primer día: "ganar dinero" no es un objetivo, hay que concretarlo (T06 ~02:30:00-02:31:57; ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md)).

**CASOS p.20** (con la banda *Scheduled* arriba, resaltada):
- **Nuestro objetivo:** ejecutar la orden con el **menor impacto de mercado**.
  - menor nivel de urgencia;
  - **sin consideración por el precio de llegada** o precio de decisión;
  - queremos ejecutar la orden **en su totalidad**;
  - **VWAP slippage (pbs)** como medida del impacto.
- Para **órdenes de compra** interesa un VWAP menor que el VWAP de mercado: "queremos pagar menos que el mercado".
- Recuadro "Importante: **Dispersión de resultados** → depende del bench".

**CASOS p.29** (mismo diagrama que p.17, con "Órden CARE (e.g., 1 millón €)"): objetivo del algo = ejecutar la orden CARE, en su totalidad, con el menor impacto; se asumen **órdenes de compra**; **Bench: TWAP** → debemos **mejorar precio (VWAP)** y **mejorar o igualar dispersión**.

**LAB celda 1** lo formula así: optimizar el *timing* de nuestras órdenes para minimizar el impacto de mercado, es decir, trocear la orden CARE para **comprar (vender) a los menores (mayores) precios** de la sesión en que se ejecuta.

**Nota:** el notebook contempla también la venta, pero la clase y CASOS p.29 restringen el caso a **compras** (T07 ~01:46:01), y la práctica sigue esa restricción.

En clase (T07 ~01:44:35-01:47:02 y ~02:54:47-02:55:15):
- Queremos un algoritmo de tipo *schedule* porque el objetivo es ejecutar con poco impacto y poca urgencia (T07 ~01:44:44-01:45:01).
- Ejecutar **todo**: "Quiero los 10.000.000. No quiero 8" (T07 ~01:45:16).
- Solo compra, no vende (T07 ~01:46:01).
- **Dispersión:** puedes conseguir muy buena media con un algoritmo súper agresivo que en algunos casos lo hace muy bien y en otros muy mal; la media sale genial pero la dispersión es enorme. Es un factor a considerar: pedir buena media **y** una dispersión acotada para que quien lo use sepa qué va a conseguir (T07 ~01:46:08-01:46:39).
- Resumen tras el descanso: orden de compra con impacto de mercado; objetivo, mejorar el slippage frente al VWAP en general y, **opcionalmente**, mejorar la dispersión (T07 ~02:54:47-02:55:15).

**Nota:** hay un matiz entre fuentes: CASOS p.29 pide "mejorar **o igualar** dispersión" frente al TWAP; en clase la dispersión quedó como consideración opcional. El ENUNCIADO (ej. 4, idea A) recoge ambas vías: "media del VWAP **o** una menor dispersión". Para el ej. 2 no se pide mejorar nada, solo **replicar** el TWAP (ver [notebook_partida.md](notebook_partida.md)).

---

## 7. El benchmark: por qué empezar por los algoritmos tradicionales

**CASOS p.21:** *Benchmark* = algoritmos tradicionales (TWAP, VWAP, POV). Aunque todos son *scheduled*, funcionan y sirven propósitos distintos → **seleccionar uno**. Preguntas necesarias: **¿cómo funcionan?, ¿por qué no han conseguido mejorarlos?, ¿hay margen de mejora?, ¿puede hacer algo la IA/ML?** Lo mismo en algoritmos de inversión: **localizar puntos de mejora** (p. ej. asignación de pesos, filtrado de señales).

En clase:
- **El TWAP como bench, desde el principio:** antes de explicar cómo funciona cada algoritmo, el profesor fija que en este caso el benchmark será un TWAP (paráfrasis, T07 ~01:47:54; la transcripción dice "el beso va a ser un Keywork", cuya lectura más probable es "el bench va a ser un TWAP", coherente con CASOS p.29).
- **Para qué sirven** (T07 ~01:47:08-01:47:52): si programo estas estrategias y las paso por mis históricos o simuladores, sé qué slippage consiguen; eso indica si estoy mejorando y a qué distancia estoy. Si me acerco al benchmark, **puedo ir bajando la exploración**; si sigo muy lejos, no. También dan escala: si obtengo tantos pb, ¿es bueno? Lo sé gracias a esas referencias. Conexión directa con el ej. 3: ver [04_exploracion_memoria_y_estabilidad.md](04_exploracion_memoria_y_estabilidad.md).
- **Volverse experto** (T07 ~01:48:10-01:49:02): primero entender cómo funcionan; después sus limitaciones; después ver si alguna limitación se puede abordar con IA, **sustituyendo el algoritmo entero o solo alguno de sus componentes**. Si ya fueran perfectos, no sabrías dónde meter la capa adicional y lo tendrías muy difícil.
- **Igual en inversión** (T07 ~01:49:08-01:49:30): puedes tener un sistema totalmente programado y que el ML solo haga la asignación de pesos a señales que venían de reglas o de criterio manual, o solo filtre activos buenos y malos.
- El profesor lo calificó como **de lo más importante de la presentación**, también para TFMs y para el trabajo: plantear y estructurar el problema siempre pesa más; anunció 2-3 alternativas de planteamiento (T07 ~01:49:50-01:50:09; desarrolladas en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md)).
- **Tipos de baseline en general** (T06 ~03:22:36-03:24:31): hay que ponerle números al objetivo para saber a qué distancia se está; sirven índices de referencia (batir al Ibex 35), **algoritmos clásicos sin ML**, o un **criterio de viabilidad** ("necesito rascar 20 pb o no lo pongo en producción"), además de averiguar qué consiguen otros. Empezar con objetivos sencillos y subirlos (el "triciclo").
- Un **benchmark matemático** más fino para una sub-decisión (utilidad esperada con probabilidades de ejecución) se discutió al final de la clase: ver §13.3.

---

## 8. TWAP (*Time Weighted Average Price*)

### 8.1 Funcionamiento

- El cliente da un **inicio, un fin y un número de títulos** (p. ej. de 9:00 a 17:00); el TWAP parte la orden en **trozos iguales** y los reparte **uniformemente** en ese tiempo. No es necesariamente de principio a fin de sesión: depende del tiempo que se le dé (T07 ~01:50:24-01:51:29).
- Ejemplo 1: 1.000 títulos en 100 min → una hija de 100 cada 10 min (T07 ~01:50:50-01:51:14). Se elige cada cuánto (5 o 10 min) y de ahí sale el tamaño de la hija (T07 ~01:51:20-01:51:29).
- Ejemplo 2 (tras el descanso): 10.000 títulos en 100 min → 10 hijas de 1.000 cada 10 min; tamaño fijo, tiempos fijos, sin ajustarse a la liquidez: "literalmente funciona así" (T07 ~02:37:15-02:38:06).
- **LAB celda 8:** algoritmo tradicional basado en el precio medio ponderado por el tiempo, usado para órdenes grandes; trocea para ejecutar en tiempos **equidistantes** dentro del tiempo elegido por el cliente. Su patrón es fácil de adivinar si no se modifica; soluciones habituales: **aleatorizar el tamaño** y/o el **tiempo entre envíos**; también limitar la cantidad para no superar un **% del volumen de participación** o **evitar agresiones de más de una posición** del libro. Para el taller se pide un TWAP **sin modificaciones** y sin mecanismos anti-detección; en el código es `class TWAP(DDQNAgent)` con un `act(self, s)` determinista a completar sobre el estado `[T, V]` (LAB celda 9). Detalle en [08_entorno_best_execution.md](08_entorno_best_execution.md).

### 8.2 Gráficos

- **CASOS p.22** (esquema; p.24 y p.26 lo repiten junto a los otros tres): línea de precio arriba; debajo, barras oscuras = **volumen de mercado** por intervalo de 09:00 a 17:30 (altas al principio, bajas a mediodía, muy altas al final); barras azul claro = órdenes del algoritmo, **todas de la misma altura**.
- **CASOS p.23** (ejecución real, TWAP, día "15", ~09:15-11:15):
  - Panel superior (*Price Evolution*): precio en rojo, media móvil en verde, **Mkt VWAP** en azul y **Algo VWAP** en naranja. El precio sube de ~6,92 a ~7,00 hacia las 10:15 y se estabiliza en ~6,985. El Algo VWAP queda **por encima** del Mkt VWAP casi todo el intervalo: coinciden en los primeros minutos (~09:17-09:21; entre ~09:22 y ~09:27 el de mercado va incluso un poco por encima), se separan con el naranja arriba de ~09:28 a ~09:42, casi coinciden o se entrecruzan entre ~09:43 y ~10:00, y desde ~10:05 el naranja queda claramente arriba; al final ~6,97 frente a ~6,965: el TWAP **paga más que el mercado**.
  - Panel inferior (*Market volume vs Algo volume over time*, eje en % de 0 a 100, a diferencia de p.25): volumen acumulado del mercado (azul) frente al del algoritmo (naranja, **una recta**); barras azules = órdenes de mercado; barras naranjas = órdenes del algoritmo **escaladas** (todas iguales). Un **círculo rojo** marca el inicio (~09:15-09:47). Las dos curvas van casi juntas hasta ~09:30; a partir de ahí el mercado acumula volumen **más rápido** que el TWAP, justo en el mínimo de precio (~6,92, 09:30-09:40) y con los primeros picos de volumen, y su curva queda **por encima de la recta del TWAP durante todo el resto del intervalo**; solo convergen en el 100 % hacia las 11:13-11:15. Es decir, el mercado compró antes y más barato a lo largo de casi toda la sesión, no solo dentro del círculo: por eso el Algo VWAP no llega a recuperar.
- Explicación del profesor (T07 ~01:51:40-01:54:33): la línea naranja es nuestro precio medio con las ejecuciones hasta cada momento, frente al precio medio del mercado; abajo, nuestras ejecuciones (las "varitas amarillas") están **sobredimensionadas** para que se vean, porque en realidad son mucho más pequeñas que las del mercado. Mi volumen es una recta porque está uniformemente distribuido. En el círculo el mercado ejecuta mucho justo cuando el precio está más bajo y ahí **me saca ventaja**; por eso mi precio medio queda por encima y, aunque luego me recupero algo, "siempre irá por encima" (frase del profesor, no descripción exacta del gráfico: ver arriba), cuando el objetivo era estar por debajo. Como no tiene en cuenta lo que pasa en el mercado, unas veces sale bien y otras mal: otro día la línea sería totalmente distinta, hay **dispersión**.
- **Limitación** (CASOS p.27): "no sensible al comportamiento del mercado". En clase: no tiene en cuenta si hay o no volumen, ni si hay mejor o peor precio (T07 ~01:54:37-01:54:46).

**Nota (por qué pierde en p.23):** en un día alcista en el que el mercado concentra volumen al principio (a precios bajos), cualquier calendario uniforme compra proporcionalmente menos barato que el mercado, y el Mkt VWAP queda por debajo del Algo VWAP. En un día bajista con volumen al principio pasaría lo contrario. Es la fuente de la dispersión del TWAP.

### 8.3 La figura objetivo de la práctica (BE images/twap_model/twap_reproduction_dqn.png)

El README la presenta como "ejemplo de DQN siguiendo una política TWAP" (BE README.md, *Motivation*) y el ENUNCIADO (ej. 2) la cita como lo que se pretende conseguir. Contenido:
- Panel superior *Market VS Algo*, ~16:07-16:42 (unos 35 min): `mid_price` en rojo (entre ~15,01 y ~15,06), `vwap` de mercado en azul y `algo` en naranja. Ambos VWAP suben juntos de ~15,025 a ~15,048 y **acaban prácticamente iguales**; hacia 16:35-16:38 el VWAP de mercado cae a ~15,04 mientras el mid se hunde a ~15,012, y luego vuelve a ~15,048.
- Panel inferior: `algo cumvol` (naranja) sube en **escalera casi recta** de 0 a 1, con unos **40 peldaños** de ~2,5 % cada uno; `mkt cumvol` (azul) avanza muy despacio hasta ~0,22 y hacia las 16:38 **salta a ~0,97** en un instante; barras `algo vol` (naranja) a intervalos regulares de ~50 s.
- Al cierre, el `algo` (~15,0475) queda **marginalmente por debajo** del `vwap` (~15,048): un slippage ligeramente favorable, del orden de ~0,3 pb, en el límite de lo que se puede leer a ojo en la figura.

**Nota:** ~40 hijas en ~35 min, una cada ~50 s ≈ 10 bins de 5 s: es la misma parametría que el ejemplo del ENUNCIADO (ver abajo), ahora visible directamente en la figura y no solo deducida de la duración.

**Nota:** ese salto significa que ~75 % del volumen de mercado del episodio se negoció en un solo bin de 5 s. *(dudoso)*: puede ser una operación de bloques o una operación grande, una subasta de volatilidad o un artefacto de la columna `cumvol` del pickle; no se puede distinguir sin cargar los datos. En un episodio así el Mkt VWAP lo decide casi una sola operación, y el TWAP solo empata (o gana por décimas de pb) porque el precio de ese bloque (~15,048) coincide con su media. Ilustra por qué el VWAP slippage de un TWAP tiene dispersión aunque la política sea perfecta. **Nota:** con datos en bins de 5 s (LAB celda 2), 35 min ≈ 420 steps, en línea con el ejemplo del ENUNCIADO (400 steps, orden CARE de 40 títulos, una hija cada 10 steps).

### 8.4 El TWAP determinista que pide la práctica (resumen)

El TWAP se puede escribir con una regla: si vas **por delante** del plan de ejecución, no ejecutes; si vas **igual o por detrás**, lanza otra hija; con tiempo restante y volumen restante basta, sin ninguna característica de mercado, y ambos tienen que "ir de la mano" para que salga la diagonal (T07 ~04:52:22-04:54:24). El profesor quiere que en el ej. 2 se llegue a ese comportamiento **por recompensas**; con escenarios tan finitos incluso valdría un Q-learning tabular de 2 estados × 2 acciones (T07 ~04:54:30-04:55:26). Diseño del estado y la recompensa: [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md) y [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md); Q-learning tabular: [00_repaso_rl.md](00_repaso_rl.md) y [02_td_sarsa_qlearning.md del Taller 2](../raw/referencias/deep-rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md).

---

## 9. VWAP (algoritmo) y la curva en U

### 9.1 Funcionamiento

- Motivación (T07 ~01:52:51-01:53:29): el TWAP parece básico si hablamos de impacto, porque se sabe que al principio de la jornada se negocia más volumen, a mediodía el mercado queda "desierto" y al cierre se vuelve a negociar mucho (si más o menos que en la apertura es *dudoso*, ver §9.2). Se puede aprovechar ese conocimiento y ajustar el calendario a cómo suelen comportarse los mercados en vez de hacerlo uniforme.
- Mecánica (T07 ~01:54:56-01:55:19): se calcula, con las últimas semanas, qué **porcentaje del volumen diario** se negocia de media en cada momento del día (un **perfil histórico intradía**), y se ajusta el calendario de ejecución a ese perfil.
- **CASOS p.24** (esquema VWAP): mismas barras de mercado que el TWAP; una **curva verde en U** (*U-Shape*) sobre ellas; las barras azul claro del algoritmo **siguen la U**: más altas al principio y al final, bajas a mediodía. Siguen la curva verde suave (histórica), **no** las barras de mercado de ese día: donde el día se aparta de la U (barras de mercado altas y bajas alternadas a media sesión, último tramo mucho más alto que la barra del algoritmo), el VWAP compra de más o de menos respecto al mercado. Es la "inestabilidad de la U-Shape" de p.27 vista en el propio esquema (igual en p.26).
- **Limitación** (CASOS p.27: "inestabilidad de la U-Shape"): la media de las últimas semanas es solo una media; habrá días en que el mercado negocie proporcionalmente más de lo esperado en un momento y días en que menos. Mejora al TWAP en impacto porque incorpora algo de conocimiento, pero **sigue siendo imperfecto**: un predictor imperfecto (T07 ~01:55:28-01:56:08; ~01:58:32).

### 9.2 La U, la subasta de cierre y la liquidez del libro

- El volumen negociado tiene forma de U. En T07 ~01:53:03 parece decir que al cierre se negocia todavía más que en la apertura *(dudoso: la transcripción dice "al cierre, pues todavía más en apertura", que admite también la lectura contraria)*. Lo que confirma explícitamente (T07 ~02:39:33) es que la **subasta de cierre** tiene hoy más liquidez que la apertura; volumen intradía de la sesión continua y subasta de cierre no son lo mismo.
- Un alumno planteó como **posible problema del POV** que ejecute toda la orden antes de la subasta de cierre, mucho más líquida, y se pierda esa liquidez (o ejecute a peor precio); el profesor confirmó que la subasta de cierre tiene muchísima liquidez y volvió a la limitación principal del POV, el riesgo de no ejecución (T07 ~02:39:00-02:39:45; §10.3).
- No hay que confundir **volumen negociado** con **liquidez del libro**: en Santander (primer semestre de 2018), la mediana del volumen en la primera posición del libro es baja en la apertura, ~2.000 títulos la mayor parte de la sesión y más alta al cierre: creciente y descorrelacionada del volumen negociado, que tiene forma de U (CASOS p.40). Desarrollo en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

**Nota (datos de la práctica):** los ficheros del taller cubren **13:00-17:25** (LAB celda 2): sin apertura y sin la subasta de cierre (en el mercado español la sesión continua acaba a las 17:30). En la práctica solo se ve la rama derecha de la U, con el volumen creciendo hacia el final. En los episodios que lleguen a esa franja, el volumen acumulado del mercado será convexo y un calendario uniforme irá **por delante** de él durante casi todo el episodio (como en la figura de §8.3, `algo cumvol` por encima de `mkt cumvol`), y el Mkt VWAP quedará dominado por los precios del final: si el precio sube al final, el TWAP comprará más barato que el mercado; si baja, más caro.

---

## 10. POV (*Percentage of Volume*)

### 10.1 Funcionamiento

- Idea (T07 ~01:56:10-01:56:44): en vez de predecir la liquidez, operar **precisamente cuando opera el mercado**: ejecutar un porcentaje fijo de lo que negocia el mercado.
- **POV 50 %**: por cada 10 títulos que negocia el mercado yo ejecuto 5. Es **súper agresivo**, probablemente más de lo que se encuentra en España; lo habitual es como mucho un 30 %, y más bien un 20 % o un 10 % (T07 ~01:56:29-01:56:54).
- **POV 10 %**: por cada 1.000 títulos del mercado, ejecuto 100; literalmente se sigue al mercado (T07 ~01:56:59).
- Al ejecutar en los mismos momentos que el mercado, el volumen acumulado propio **sigue al del mercado**. Cuando hay una ejecución muy fuerte del mercado, el algoritmo tiene que hacer ***catch-up***: lanzar varias hijas seguidas para recuperarse (T07 ~01:57:13-01:57:37). Como opero en los mismos momentos, mi precio por lo general no se desvía mucho del del mercado (T07 ~01:57:42-01:57:52).

### 10.2 Gráficos

- **CASOS p.24** (esquemas *POV(50%)* y *POV(30%)*): mismas barras de mercado; las del algoritmo son en cada intervalo un 50 % (una llave azul marca "50%") o un 30 % ("30%") de la barra de mercado; suben y bajan con ella.
- **CASOS p.25** ("POV (más usado)", día "15", ~09:15-10:10): panel superior con precio de ~6,92 a ~6,98; Algo VWAP **casi pegado** al Mkt VWAP, **ligeramente por encima** (paga un poco más) a lo largo de todo el intervalo, ambos ~6,951-6,953 al final. Panel inferior: el volumen acumulado del algoritmo **replica casi exactamente** el del mercado; barras naranjas del algoritmo muy densas; barras azules de mercado con picos grandes hacia las 09:23 (~80.000), 09:40 (~64.000) y sobre todo 09:54 (~110.000, con una parte más oscura de ~72.000). En este último el volumen acumulado del algoritmo se queda atrás un instante y recupera en diagonal: es el *catch-up* que describe el profesor (T07 ~01:57:26-01:57:37; §10.1).

**Nota:** en p.25 la leyenda dice "Mkt Cum Vol(%)" y "Algo Cum Vol (%)", pero el eje va en **volumen absoluto** (hasta ~110.000); probablemente la curva del algoritmo está reescalada al total del mercado para poder superponerlas. Las ejecuciones de p.23 y p.25 llevan el mismo prefijo de día ("15") y precios en el mismo rango: parece la **misma sesión** con horizontes distintos (2 h el TWAP, ~1 h el POV) *(dudoso)*.

### 10.3 Limitaciones

- **Riesgo de no ejecución** (CASOS p.27: "sigue completamente al mercado + riesgo de no ejecución"): dependo de que el mercado negocie; si no lo hace, me quedo sin ejecutar (T07 ~01:57:56). Ejemplo: con un POV del 10 % y una orden de 1.000.000 de títulos, el mercado tiene que negociar 10.000.000; si no, no completas la orden. Es su limitación principal; el porcentaje es el **nivel de agresividad** (T07 ~02:39:45-02:40:09).
- **Pagar el precio de ir segundo:** cuando ejecutas, otros ya han consumido la liquidez y operas con algo menos de liquidez (T07 ~01:57:56-01:58:18; ~02:40:09-02:40:14).

**Nota:** CASOS p.31 sugiere, como ejemplo de limitación al generar órdenes (episodios), no pasar de un 5-10 % del ADV, lo que ayuda con las asunciones de impacto. Una orden del 10 % del ADV con un POV del 10 % necesitaría que el mercado negociase **todo su volumen diario medio** dentro del horizonte: con horizontes de horas es casi imposible. Por eso un POV conservador no sirve para "ejecutar la orden en su totalidad" (requisito de §6) salvo que el horizonte sea largo o el porcentaje alto.

---

## 11. Comparativa: costes, dispersión y uso real

### 11.1 Resumen de los tres (CASOS p.27 y T07 ~01:58:18-01:58:45)

| | TWAP | VWAP | POV |
|---|---|---|---|
| Calendario | Uniforme en el tiempo | Perfil histórico de volumen (U) | % fijo del volumen negociado en tiempo real |
| Mira al mercado | No, "ajeno a lo que está pasando" | Solo vía la media histórica | Sí, lo sigue completamente |
| Desventaja (CASOS p.27) | No sensible al comportamiento del mercado | Inestabilidad de la U-Shape (predictor imperfecto) | Sigue completamente al mercado + riesgo de no ejecución; paga ir segundo |
| Coste medio (clase) | ~3 pb | ~3 pb | ~7 pb |
| Dispersión (clase) | Grande (a veces ~20 pb) | Grande | Muy pequeña |
| Uso en la plataforma del profesor | ~5 % | ~15 % | ~80 % |

### 11.2 El gráfico de distribuciones (CASOS p.27) y la corrección del profesor

- Contenido: título *VWAP Slippage (pbs)*; eje horizontal con **0 = Market** (línea discontinua verde) y una flecha "**− Peor / Mejor +**" (izquierda peor, derecha mejor). Curva **POV** (negra): **estrecha**, con su media (discontinua negra) claramente **a la izquierda** del 0. Curva **TWAP / VWAP** (azul): **muy ancha**, con su media (discontinua azul) **ligeramente a la izquierda** del 0.
- El profesor dijo que el gráfico "está mal": el VWAP slippage es un coste, y con costes **a la derecha es peor**; se le dio la vuelta porque solemos leer que a la derecha está lo mejor. En realidad el POV está donde está porque tiene **mayor coste**; los otros tienen más dispersión pero **pagan menos en media** que el POV, y el POV tiene resultados muy concentrados (T07 ~01:58:54-01:59:36).

**Nota (no es una contradicción de fondo):** la diapositiva ya usa la convención "slippage positivo = mejor" (el POV, más caro, está a la izquierda, en "peor"); el comentario del profesor se refiere a que, si se dibujase como **coste**, el eje iría al revés. Las dos lecturas dicen lo mismo: POV más caro y estable; TWAP/VWAP más baratos de media y dispersos. Lo que hay que evitar en el informe es mezclar convenciones (§3).

### 11.3 Por qué el más usado es el POV (T07 ~01:59:40-02:02:03)

- Al bróker le llegan órdenes de muchos clientes. No puede decirle a un cliente "hoy −20 pb, mañana +20": se enfadarían, y además el cliente no es siempre el mismo; no puede dar resultados súper inestables. Con un algoritmo que cuesta **unos 7 pb** de forma estable, el cliente sabe qué esperar; los otros tienen una media de 3 pb, pero a veces son 20 de coste (T07 ~01:59:56-02:00:42).
- Cuando el profesor trabajaba en esto, de las órdenes que recibían, el TWAP era ~5 %, el VWAP ~15 % y el POV ~80 % (T07 ~02:00:42-02:00:56). Por el contexto (su etapa en BME) parece la plataforma de BME *(dudoso)*.
- Ofrecían la plataforma como servicio a otros bancos: p. ej. un Sabadell cuyas filiales accedían y pedían "ejecútame la orden con un VWAP (o el que sea), de tantos títulos, de tal hora a tal hora"; el banco elegía el algoritmo, los títulos y el horario, y los sistemas la procesaban (T07 ~02:01:00-02:01:57).
- Cierre antes del descanso: "¿Creéis que se puede mejorar alguno de estos?" (T07 ~02:02:03-02:02:21; respuesta en §12).

**Nota (contraste con los resultados del caso):** en el histograma de test de CASOS p.60 el TWAP tiene una media ponderada de **−1,88 pb** frente al VWAP de mercado y el algoritmo RL **+2,8 pb** (positivo = mejor; según la propia diapositiva, "sobre los más líquidos"). Los −1,88 pb son coherentes con los "~3 pb" de coste medio de un TWAP citados en clase. El análisis de ese resultado (y la observación de que la distribución RL parece más ancha, lo que choca con "mejorar o igualar dispersión") está en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).

---

## 12. ¿Se puede mejorar un TWAP? Reglas y su explosión

**Idea.** El TWAP tiene decisiones arbitrarias (momento fijo, tamaño fijo, tipo fijo). Se le pueden añadir reglas razonables, pero al combinarlas el número de casos explota y los expertos no se ponen de acuerdo: esa es la puerta de entrada del RL.

### 12.1 Decisiones arbitrarias del TWAP (T07 ~02:40:19-02:40:59)

Momentos temporales fijos, órdenes de tamaño fijo y tipo de orden fijo (agresiva/a mercado, pasiva/limitada o pegada; p. ej. siempre agresiva: lanza 1.000 y "lo que haya").

### 12.2 Cómo estaba programado realmente el TWAP de BME (T07 ~02:42:40-02:45:01; ~03:28:12-03:30:00)

| Regla | Detalle | Por qué |
|---|---|---|
| **Tamaño de hija según la liquidez de la primera posición** | Si el libro suele tener ~2.000 títulos en primera posición, la hija es **la mitad** de lo que suele haber | No barrer el libro. En Santander la primera posición suele tener unos 2.000 títulos (más vacía por la mañana, con más liquidez al final del día), así que la hija fija es de 1.000. Además, el mercado suele negociar unos 600 títulos por operación: 1.000 se parece a lo que hace el mercado y ayuda a **pasar desapercibido**. Con un estudio así se le quita esa decisión al algoritmo y el tamaño queda fijo "y ya está" (T07 ~03:29:10-03:30:00). CASOS p.39/41 lo formaliza: tamaño fijo estándar, relativo a la liquidez, o condicionado $\min(\text{estándar},\ 0{,}5\cdot \text{vask1})$, con modificadores agresivo (×2) y conservador (×0,5); desarrollo en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md) |
| **Tiempo mínimo de recuperación del libro** | Configurado con un **perfil de liquidez por empresa**: un activo menos líquido tarda más en recuperarse | No lanzar la siguiente hija sobre un libro aún vacío |
| **Aleatoriedad en el tamaño** | Unas veces una cifra y otras otra (la transcripción dice "una 150 otras veces 1 003": ¿1.150 y 1.003?, *dudoso*) | Si siempre ejecutas el mismo tamaño, delatas a otros algoritmos que hay alguien ejecutando un plan: **camuflarse** |
| **Aleatoriedad en el intervalo** | Cada 8, 11 o 12 min en vez de cada 10, con un margen de variabilidad | Lo mismo: ocultarse |
| **Órdenes "carga de profundidad"** | Órdenes dejadas muy abajo, en la **9.ª posición** del libro, por si alguien barre | Con miles de operaciones, los barridos acaban ocurriendo bastantes veces; es una regla determinista que coexiste con el resto (CASOS p.37: "cargas de profundidad", acciones que no elige el agente) |

"Así es como estaba programado el TWAP más básico", pero quedan muchos márgenes de mejora (T07 ~02:44:48-02:45:01). La propia LAB celda 8 recoge las mismas variantes (aleatorizar tamaño y tiempos, limitar el % de participación, no agredir más de una posición).

### 12.3 Reglas de mejora propuestas en clase

| Condición | Acción propuesta | Fuente |
|---|---|---|
| Spread más amplio de lo normal | Propuesta de un alumno, aprobada por el profesor ("estupendo"): no lanzar las 1.000 sino ~500 y pasar el resto al siguiente punto del calendario *(dudoso: la transcripción está muy rota y también menciona "el mercado es muy líquido")* | T07 ~02:40:59-02:41:25 |
| Justo antes de mi hija ha habido una ejecución que ha consumido liquidez y el spread se ha abierto (cifras en la transcripción: de ~10 a ~12 pb, *dudoso*: la transcripción las da al revés, "suele ser de 12 ... ahora son 10", lo que contradice "se ha abierto") | **Esperar unos segundos** (hay 10 min de margen) a que el libro se recupere y el spread se cierre, y entonces lanzar la agresiva | T07 ~02:41:25-02:42:09 |
| Durante la ventana de 10 min | Dejar una orden **pasiva/limitada**; si se ejecuta, ya se ha cumplido el calendario sin pagar la agresividad y se pone otra limitada para la siguiente ventana; si no, agresiva en el minuto 10 | T07 ~02:45:10-02:45:45 |
| No hay operaciones en el mercado (indicador de liquidez) | Esperar | T07 ~02:46:12-02:46:24 |
| Empieza a haber gente ejecutando | **Adelantar** la hija dentro de la ventana, para conseguir el precio que está obteniendo el mercado (que es contra lo que te van a evaluar): "un toque de POV" | T07 ~02:46:24-02:47:02 |

### 12.4 La explosión de reglas (T07 ~02:47:06-02:51:25)

- **Dimensionalidad:** cada regla por separado tiene sentido, pero en cuanto mezclas 2 o 3 situaciones el espacio de escenarios explota, y con un algoritmo determinista tienes que dar con **la regla exacta para cada circunstancia** (T07 ~02:47:15-02:47:55).
- **Ruido entre expertos** (abogado del diablo): la regla de esperar a que el libro se recupere parecía buena, pero otro experto diría que si el spread se abre es porque está pasando algo, la gente está ejecutando ahí y **mejor compres ya**, porque luego el precio puede ir a peor. Programar muchas reglas añade indefinición y ruido sobre cuál es la más adecuada (T07 ~02:48:01-02:49:07).
- **Por qué los brókers no lo hacen** aunque saben que el TWAP es muy mejorable (y algunas mejoras ya las han hecho, como no ser detectados): porque es **inviable**; habría que meter demasiada condicionalidad, y con distintos escenarios de liquidez los casos "te revientan" (T07 ~02:50:21-02:51:06).
- **Conclusión:** en vez de programar las reglas a mano, darle al agente la información (si ha habido frecuencia de *trades*, si el spread está más amplio o menos respecto a los **percentiles** de los últimos minutos) y que, según lo que ha pasado históricamente, aprenda cuál habría sido la decisión óptima; ya no hace falta que el experto diga qué es bueno y qué es malo, y se le pueden dar aún más condiciones. En Q-learning eso es literalmente **aprender el "diccionario"** estado → acción (T07 ~02:49:30-02:51:25).

**CASOS p.28** resume el razonamiento en tres cajas encadenadas: **Más información** (iconos de base de datos, velas y lupa; *bid-ask spread*, frecuencia de *trades*, liquidez, …) → **Más reglas** (icono de tres planos cruzados en 3D; "Complex Framework": muchas situaciones potenciales a las que el algoritmo podría enfrentarse; demasiadas reglas a definir manualmente) → flecha "Solución" → **Machine Learning / AI** (§13).

---

## 13. Dónde entra el ML/RL

### 13.1 Las tres familias (CASOS p.28)

| Familia | Papel en ejecución (CASOS p.28) | Ejemplo de clase |
|---|---|---|
| **Aprendizaje por refuerzo** | Entrenar el algoritmo para **ser bróker** / que aprenda dichas reglas | Aprender cuándo esperar, adelantar o pasar a pasiva (§12.4) |
| **No supervisado** | *Clustering*, reducción de dimensionalidad | Agrupar activos por perfil de liquidez (líquidos/ilíquidos) para aplicar un algoritmo u otro (T07 ~00:24:03-00:24:15; ~02:51:34-02:51:40) |
| **Supervisado** | *Feature engineering*, p. ej. probabilidad de ejecución de una orden pasiva en los próximos X s | Ver §13.2 |

### 13.2 Combinar supervisado, reglas y RL

- **No todo es IA** (T07 ~00:23:02-00:26:15): un problema tiene muchos pasos; hay que decidir **dónde** aplicar ML y de qué tipo (RL en un punto, supervisado en otro); no todas las características tienen que salir de otro modelo ni todas las decisiones del agente: se pueden programar reglas. Ningún algoritmo de RL predomina (si necesitas acciones continuas descartas la mitad) y se pueden combinar ideas (un *Dueling Double DQN*, al estilo Rainbow). Ver [01_aproximacion_funcional_y_dqn.md](01_aproximacion_funcional_y_dqn.md).
- **Probabilidad de ejecución de la pasiva** (T07 ~02:51:52-02:54:39): con todas las órdenes pasivas del histórico y características como frecuencia de *trades*, volatilidad y otras (¿spread?, *dudoso*: la transcripción es ilegible en ese punto), un supervisado da la probabilidad de que una pasiva pendiente se ejecute en los próximos segundos. A mano habría que fijar umbrales ("si la probabilidad es mayor que 90 % pasiva, si no agresiva": ¿por qué 90?; ¿y si es 70 % pero hay liquidez?) y vuelve la explosión. Con la probabilidad como característica, el RL aprende la regla. El RL podría aprenderla internamente desde los datos brutos, pero construirla por fuera ("masticadito") simplifica lo que viene después. **Incluso sin RL**, supervisado + reglas ajustadas ya mejorarían el TWAP.
- **Aislar subsistemas** (T07 ~03:07:02-03:08:09): para el profesor casi lo más importante de la clase. Ejemplo: un agente que solo decide el **tipo de orden** con la probabilidad de ejecución como única información aprendería, tras promediar episodios, algo como "a partir del 60 % interesa la pasiva".
- **Factibilidad: emular a quien ya lo hace** (T06 ~02:32:05-02:33:28): hay brókers que ejecutan bien órdenes con información conocida; ese comportamiento es el que se quiere emular con RL, y entonces el objetivo es **emularles**, no "ganar dinero". Es exactamente el planteamiento del ej. 2 (replicar el TWAP). Ver [03_recompensas_y_descuento.md](03_recompensas_y_descuento.md).

### 13.3 Benchmark matemático: utilidad esperada (T07 ~04:10:26-04:16:00)

- Ante la pregunta de un alumno sobre cuándo usar supervisado frente a RL: en banca el RL se usa poco y muchas veces basta con supervisado. Coincide con la regla práctica de APP p.12: si basta un modelo supervisado, úsalo; el RL obliga a construir un entorno.
- Ejemplo: limitadas a distintos niveles con probabilidades de ejecución de un supervisado (80 %, 60 %, 50 %, 10 %) frente a la agresiva del TWAP a 7,12; si la limitada no se ejecuta, se paga la agresividad, unos **3 pb** sobre el precio medio del libro, y se calcula de forma determinista la utilidad esperada de cada acción (T07 ~04:11:11-04:12:58). Cálculo, unidades (*dudosas* en el ahorro) y umbral de indiferencia en [07 §18](07_rl_para_ejecucion.md); no confundir con el ejemplo de recompensa local de T07 ~03:42 (referencia 7,10, céntimos), en [07 §9.4](07_rl_para_ejecucion.md).
- Eso es **como programación dinámica**: probabilidades de transición conocidas y recompensas de cada escenario. Probablemente explica el 80-90 % de la varianza; es el **benchmark** (la "solución clásica matemática") que el RL, con características adicionales como la frecuencia de *trades* o el spread, debería superar (T07 ~04:13:04-04:14:31). El ML clásico está infrautilizado y puede coexistir con reglas y otros sistemas (T07 ~04:14:46-04:15:54). DP: [01_dp_y_montecarlo.md del Taller 2](../raw/referencias/deep-rl-trading-agents/docs/teoria/01_dp_y_montecarlo.md).

### 13.4 Industria y lecciones (resumen; detalle en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md))

- **Ejecución vista como inversión** (T07 ~03:58:37-04:02:02): los resultados del TFM se obtuvieron con hijas de 1 título, poco impacto y coste 0; así, el perfil conservador ya queda mejor que el mercado. Combinando un algoritmo de compra con otro de venta y rascando 2-3 pb en cada lado se "estaría ganando dinero" y se parecería a un algoritmo de alta frecuencia o de inversión, pero con **capacidad de absorción limitada**; al meter costes de transacción desaparecen las ganancias aunque la diferencia relativa se mantenga (cifras "5" y "menos 8", *dudoso*: no queda claro a qué se refieren). Con 1 título y comisiones 0 podría plantearse algo así, "ir rascando poco a poco".
- **Primer paper de RL para ejecución** (T07 ~04:02:10-04:03:53; CASOS p.61, con el logo de Lehman Brothers): *Reinforcement Learning for Optimized Trade Execution* (2006), Q-learning (tabular, "modificado" según la diapositiva) porque aún no estaban las redes. Tipo *arrival price*; AMZN, NVDA y QCOM durante 1,5 años; órdenes pequeñas (5-10 mil acciones en 2-8 min); acción "ask − a" (probablemente el precio límite como una distancia $a$ por debajo del ask *(dudoso: la diapositiva no lo detalla)*). Con solo **dos variables privadas**, tiempo e inventario restantes (T, I), mejoraba un **50 %** frente al benchmark *submit and leave* (en clase, paráfrasis: dejar la orden y, si no se ejecuta, ejecutarla mal al final); añadiendo variables de mercado mejoraba otro 7-8 % (la tabla de la diapositiva da reducciones adicionales de coste de 7,97 % con el spread, 8,69 % con spread + coste inmediato y 12,85 % combinando spread, coste inmediato y volumen firmado). Problema muy acotado, acciones discretizadas; el paper tiene unas 7 páginas.

  **Nota:** son las mismas dos variables que usa el estado `[T, V]` del TWAP del notebook (LAB celda 9): es la mejor evidencia del material de que tiempo y volumen restantes bastan para empezar. Detalle en [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md).
- En la práctica la industria sigue usando TWAP/VWAP/POV; algunas firmas dicen usar ML para ejecutar sin detallar qué (T07 ~04:04:30-04:04:54). Un algoritmo de J.P. Morgan basado en RL (CASOS p.61: "JPM DNA Algo") era, según el profesor, muy flexible (tamaño, momento y tipo) y se habría cerrado recientemente por ser caja negra *(dudoso)* (T07 ~04:04:56-04:05:21).
- Si se ve como un algoritmo de inversión (comprar con uno y vender con otro rascando pb en cada lado), habría que encontrar características y estados que rasquen en media más de ~2,8 pb para poder compensar los costes (T07 ~04:05:37-04:05:56). *(dudoso)*: lo más probable es que los 2,8 pb sean la media del algoritmo RL en el histograma de test (CASOS p.60, +2,8 pb), que acababa de enseñar, es decir, que esa mejora no bastaría; la otra lectura es que ~2,8 pb sea el nivel de costes.

  **Nota:** como escala, la comisión del Taller 2 era del 0,1 % = 10 pb por operación ([00_fundamentos_mdp.md del Taller 2](../raw/referencias/deep-rl-trading-agents/docs/teoria/00_fundamentos_mdp.md)), varias veces la mejora de 2-3 pb.
- Lección del TFM: quiso cubrir momento, tamaño y tipo a la vez; habría sido mejor centrarse solo en el tipo (pegada/limitada) con una probabilidad de ejecución por nivel entrenada con el histórico teniendo en cuenta la **prioridad en la cola**, no solo el precio, y que puede ser un árbol, sin redes. El error común es **abarcar demasiado**. Los bancos hacen proyectos internos así (probabilidades en los terminales del bróker); el RL aporta buscar una regla mejor que la humana (T07 ~04:06:49-04:10:26).

---

## 14. Qué se lleva la práctica de todo esto

**Nota (propuesta; el desarrollo está en [10_hoja_de_ruta_practica.md](10_hoja_de_ruta_practica.md)):**

| Concepto de negocio | Consecuencia para la práctica | Evidencia |
|---|---|---|
| Objetivo de baja urgencia, solo compra, ejecutar el 100 % | El episodio no debe acabar con volumen pendiente; si queda, penalizar (idea B del ej. 4) | §6; ENUNCIADO ej. 4 |
| TWAP como benchmark | Implementar el TWAP determinista (LAB celda 9) y medir su VWAP slippage en los mismos episodios que el agente; servir de referencia para el criterio de mejora de epsilon (ej. 3) | §7, §8; T07 ~01:47 |
| Slippage en pb, con signo declarado | Reportar media **y** dispersión (desviación típica o percentiles) por episodio, no solo la media | §3, §6, §11 |
| Impacto nulo, hija de 1 título | Lo único mejorable es el **timing**; no hay que modelar impacto ni barrido del libro | §2.4; ENUNCIADO ej. 2 |
| Datos 13:00-17:25 en bins de 5 s | Episodios de ~400 steps ≈ 33 min; el volumen sube hacia el final (rama derecha de la U) | §8.3, §9.2 |
| Explosión de reglas → RL | Para la idea A del ej. 4, motivar cada característica de mercado (spread, frecuencia de *trades*, desbalanceo) con la regla de §12.3 que el agente debería aprender | §12; [07_rl_para_ejecucion.md](07_rl_para_ejecucion.md) |

Qué pide exactamente cada ejercicio: [notebook_partida.md](notebook_partida.md). El código del entorno y del TWAP: [08_entorno_best_execution.md](08_entorno_best_execution.md).

---

## 15. Erratas, ambigüedades y dudas abiertas

1. **Signo del VWAP slippage.** CASOS p.27 y p.60 usan "positivo = mejor"; el profesor vino a decir que el slippage es un coste (frase cortada en la transcripción) y que el gráfico de p.27 está "dado la vuelta" (T07 ~01:58:54). Ninguna fuente da la fórmula con signo. Propuesta en §3; hay que declararla en el informe.
2. **CASOS p.16** llama "CARE" a una orden de 1.000 € que, según BE README, sería DMA (§2.2).
3. **CASOS p.18**: la curva de coste total es esquemática; no la reproduce la suma de las otras dos con ningún $\lambda$ constante (§4.1).
4. **CASOS p.25**: leyenda en % y eje en volumen absoluto (§10.2).
5. **BE README**: "Kissel" por **Kissell** (§4.2).
6. **Cifras dudosas de la transcripción:** spread de 10/12 pb en la regla de esperar (dada al revés), tamaños "una 150 / 1 003" (¿1.150 / 1.003?), la lectura "high-frequency trading" en alpha decay, los índices citados como referencia de fondos, si el volumen del cierre supera al de la apertura (§9.2), las características del supervisado de probabilidad de ejecución (¿spread?, §13.2), las unidades del ejemplo de utilidad esperada ("2 por 100" ≈ 2 pb) y el significado de sus $+2/-3$ (§13.3), las cifras "5" y "menos 8" con costes (§13.4), el cierre del algoritmo de J.P. Morgan y los ~2,8 pb a "rascar" (más probablemente la media del RL de CASOS p.60 que el nivel de costes, §13.4).
7. **Plataforma de los porcentajes de uso (5/15/80 %)**: el profesor no dice explícitamente que fuera BME; se deduce del contexto (§11.3).
8. **Dispersión como objetivo**: CASOS p.29 la exige ("mejorar o igualar"); en clase quedó como opcional (§6).
9. **Tamaño de la orden de ejemplo:** CASOS p.17 dice 10 M€ y CASOS p.29 1 M€, y en clase se habla de 10.000.000 de **títulos** (T07 ~01:45:16) y de 1.000.000 de **títulos** de Santander (T07 ~00:45:42). Son ejemplos ilustrativos; las cuentas de §3 (1 pb = 1.000 € sobre 10 M€) usan la cifra en euros.
10. **Compra y venta:** LAB celda 1 habla de comprar (vender) a los menores (mayores) precios; la clase y CASOS p.29 restringen el caso a compras (§6).
