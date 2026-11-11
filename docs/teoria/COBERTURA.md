# Cobertura del material de clase

> **Fuentes:** las de [notebook_partida.md](notebook_partida.md): T06 = `docs/raw/26 11 06.pdf` (transcripción del 06/11/2026), T07 = `docs/raw/26 11 07.pdf` (07/11/2026), APP = `docs/raw/MIAX14/applying_RL_MIAX.pdf` (103 p.), CASOS = `docs/raw/MIAX14/casos_practicos.pdf` (63 p.), LAB = `docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb` (celdas 0-21), LABSTEP = `docs/raw/MIAX14/best-execution/workshop/execution_lab-step_by_step.ipynb` (celdas 0-34), BE = `docs/raw/MIAX14/best-execution/`, DATOS = `docs/raw/MIAX14/best-execution/data/`, RLM = `docs/raw/MIAX14/rl-material/`, ENUNCIADO = `docs/raw/MIAX14/ejercicios_taller_rl.docx`.
> **Convenciones:** cada fila dice a qué documento y sección (§) de `docs/teoria/` fue cada tramo de transcripción, página, celda o fichero. Los tramos son los del índice de trabajo usado para repartir el material (tema y horas aproximadas). La comprobación se hizo **buscando en cada documento las citas que caen dentro del tramo o de la página** (p. ej. "T07 ~03:30-03:33", "CASOS p.43"): solo se anota una sección si contiene esa cita; no cuentan las cabeceras de fuentes ni las citas que abarcan más de 30 min. "también" = otros documentos que citan el tramo además del destino previsto. El glosario ([11_glosario.md](11_glosario.md)) cita casi todo y no se repite aquí salvo cuando es el único destino. Las horas son aproximadas, como en los documentos.

Escrito el 10/11/2026.

## 1. Transcripciones

### 1.1 T06 (06/11/2026)

| Tramo | Tema | Destino |
|---|---|---|
| ~00:00:02-00:02:00 | Arranque de la sesión, prueba de micrófono | descartado: Ruido de inicio de clase y charla sin contenido |
| ~00:02:00-00:06:26 | Presentación del profesor y agenda de los dos días | `notebook_partida.md` §13; `00_repaso_rl.md` §1; `05_profit_seeking.md` §1; `06_algoritmos_de_ejecucion.md` §1 |
| ~00:06:26-00:10:52 | Debate inicial: qué necesitan los alumnos; estado del campo RL | `00_repaso_rl.md` §1, §14; también `05_profit_seeking.md` §1, `notebook_partida.md` §13 |
| ~00:10:52-00:13:55 | Sondeo de conceptos básicos y RL vs supervisado (percepciones) | `00_repaso_rl.md` §1, §5.1, §14 |
| ~00:13:55-00:17:47 | V, Q y política con el ejemplo del paraguas | `00_repaso_rl.md` §1, §3.1, §3.2, §3.3, §5.1, §13; `11_glosario.md` §«M», §«P», §«Q», §«V» |
| ~00:17:47-00:21:15 | Aprendizaje por prueba y error, Q como diccionario promediado y transiciones | `00_repaso_rl.md` §3.2, §3.3; `11_glosario.md` §6, §«D», §«P»; también `03_recompensas_y_descuento.md` §9, §13 |
| ~00:21:15-00:22:58 | Principios de RL aplicados en su trabajo (elasticidades de precios bancarios) | `00_repaso_rl.md` §3.3 |
| ~00:22:58-00:26:56 | Propiedad de Markov y qué información poner en el estado | `00_repaso_rl.md` §3.3, §4.1, §4.2, §5.1, §15; `02_diseno_de_entornos.md` §5.5, §5.7; `05_profit_seeking.md` §6.1, §6.4, §10.2; `11_glosario.md` §«B», §«M», §«N» |
| ~00:26:56-00:30:06 | Cuándo usar RL frente a supervisado | `00_repaso_rl.md` §4.2, §4.3, §5.1, §15; también `02_diseno_de_entornos.md` §1, `05_profit_seeking.md` §3, §6.4 |
| ~00:30:06-00:34:05 | Datos sintéticos vs reales y por qué el agente no aprende patrones | `00_repaso_rl.md` §4.3, §5.1, §5.2, §15; `02_diseno_de_entornos.md` §5.4; `05_profit_seeking.md` §3; también `07_rl_para_ejecucion.md` §1 |
| ~00:34:05-00:38:04 | Supervisado en bolsa vs RL: impacto de las acciones y estado privado | `00_repaso_rl.md` §4.3, §5.2, §6, §15, §16; `05_profit_seeking.md` §3; `06_algoritmos_de_ejecucion.md` §2.4; `07_rl_para_ejecucion.md` §1, §7.1; `02_diseno_de_entornos.md` §1, §5.6; también `03_recompensas_y_descuento.md` §1 |
| ~00:38:04-00:44:26 | Objetivo del agente, utilidad descontada, gamma y ecuación de Bellman | `00_repaso_rl.md` §5.2, §6, §7.1, §15, §16; `03_recompensas_y_descuento.md` §1, §10, §11, §13; `11_glosario.md` §«B», §«C», §«G», §«M», §«P», §«T», …; también `05_profit_seeking.md` §3, `07_rl_para_ejecucion.md` §1, §7.1, `10_hoja_de_ruta_practica.md` §6.4 |
| ~00:44:26-00:51:22 | Monte Carlo, programación dinámica y diferencias temporales | `00_repaso_rl.md` §6, §7.1, §7.2, §9.1, §16; `11_glosario.md` §6, §«M», §«N», §«P», §«T»; también `03_recompensas_y_descuento.md` §1 |
| ~00:51:22-00:59:57 | Q-learning: actualización con alfa, off-policy y ejemplo numérico | `00_repaso_rl.md` §3.1, §7.1, §7.2, §8, §9.1, §9.2, …; `11_glosario.md` §6, §«A», §«M», §«O», §«P», §«Q», …; `03_recompensas_y_descuento.md` §3; `04_exploracion_memoria_y_estabilidad.md` §2.1, §5.1 |
| ~00:59:57-01:06:52 | Material rl-material: Docker/devcontainer, instalación y demo Q-learning | `09_material_rl_notebooks.md` §1, §1.1, §2.1, §2.3, §5.1, §5.5; `02_diseno_de_entornos.md` §2.3; `notebook_partida.md` §10.2, §13; también `00_repaso_rl.md` §9.2, `03_recompensas_y_descuento.md` §3 |
| ~01:06:52-01:12:45 | SARSA vs Q-learning y cliff walking | `00_repaso_rl.md` §9.2, §10, §11.1; `09_material_rl_notebooks.md` §1, §1.1, §5.2, §5.5, §6; también `notebook_partida.md` §10.2 |
| ~01:12:45-01:16:53 | Estabilidad del entrenamiento y Expected SARSA | `00_repaso_rl.md` §10, §11.1, §11.2, §15, §16; `03_recompensas_y_descuento.md` §7, §13; `04_exploracion_memoria_y_estabilidad.md` §8.1; también `09_material_rl_notebooks.md` §1, §4, §4.3, …, `10_hoja_de_ruta_practica.md` §2, §6.8, §10 |
| ~01:16:53-01:20:00 | Double Q-learning, sesgo de inicialización y búsqueda de estabilidad | `00_repaso_rl.md` §11.2, §12, §13, §15; `04_exploracion_memoria_y_estabilidad.md` §1, §4.6, §8.1, §13; también `01_aproximacion_funcional_y_dqn.md` §1, `09_material_rl_notebooks.md` §4, §4.3, §5.4, … |
| ~01:20:00-01:26:22 | De tabular a aproximación funcional; modelo lineal | `01_aproximacion_funcional_y_dqn.md` §2, §2.1, §2.2, §4.1, §4.2, §4.3, …; `09_material_rl_notebooks.md` §7; también `00_repaso_rl.md` §4.4, §13, `04_exploracion_memoria_y_estabilidad.md` §6.1 |
| ~01:26:22-01:36:31 | Experience replay: i.i.d., buffer FIFO, PER y CER | `01_aproximacion_funcional_y_dqn.md` §5.2, §6.1, §6.2, §6.5, §6.7; `04_exploracion_memoria_y_estabilidad.md` §6.1, §6.8; `11_glosario.md` §6, §«C», §«E», §«F», §«I», §«J», …; también `09_material_rl_notebooks.md` §7 |
| ~01:36:31-01:41:07 | Fixed Q-targets, Double DQN, sobreestimación y Dueling | `01_aproximacion_funcional_y_dqn.md` §1, §5.2, §6.5, §6.7, §7, §8.1, …; `04_exploracion_memoria_y_estabilidad.md` §6.8, §8.2 |
| ~01:41:07-02:06:44 | Descanso | descartado: Descanso: ruido y conversación irrelevante |
| ~02:06:44-02:11:09 | Segunda parte: evolución de la investigación en RL | `01_aproximacion_funcional_y_dqn.md` §14, §16; `00_repaso_rl.md` §4.4, §14, §15, §16 |
| ~02:11:09-02:19:53 | Diseño del estado: qué características pasar; CartPole y discretización | `02_diseno_de_entornos.md` §5.1, §5.2, §5.3, §5.7, §5.8, §12; `09_material_rl_notebooks.md` §5.1, §5.5, §7, §13, §14; `00_repaso_rl.md` §4.4, §15, §16; también `07_rl_para_ejecucion.md` §6 |
| ~02:19:53-02:25:04 | Definición del step: criterios temporales, por eventos y mixtos | `02_diseno_de_entornos.md` §4.1; `07_rl_para_ejecucion.md` §6, §8.1; `05_profit_seeking.md` §5.1, §5.2; también `00_repaso_rl.md` §4.4, §15, §16 |
| ~02:25:04-02:29:43 | Coste computacional, acciones discretas vs continuas y simplificación | `02_diseno_de_entornos.md` §4.2, §6.1, §6.2; `03_recompensas_y_descuento.md` §1, §2; `07_rl_para_ejecucion.md` §6, §8.1; también `05_profit_seeking.md` §5.1, §5.2 |
| ~02:29:43-02:30:00 | Introducción a recompensas: definir el objetivo | `03_recompensas_y_descuento.md` §1, §2; `05_profit_seeking.md` §4.5; también `07_rl_para_ejecucion.md` §6, §8.1 |
| ~02:30:00-02:33:35 | Definir y aterrizar el objetivo; factibilidad (emular a alguien que ya lo hace) | `03_recompensas_y_descuento.md` §2, §3, §13; `05_profit_seeking.md` §4.1, §4.5; `06_algoritmos_de_ejecucion.md` §6, §13.2 |
| ~02:33:35-02:35:34 | Comprobar que las recompensas van en la dirección del objetivo | `03_recompensas_y_descuento.md` §2, §3, §5, §13; también `05_profit_seeking.md` §4.5, `10_hoja_de_ruta_practica.md` §6.3, §9 |
| ~02:35:34-02:38:12 | Sparse reward vs reward shaping; ejemplo de andar de A a B | `03_recompensas_y_descuento.md` §3, §5, §8, §10, §13; `11_glosario.md` §«R»; también `10_hoja_de_ruta_practica.md` §6.3, §9 |
| ~02:38:12-02:41:17 | Gamma exigente y comportamientos inesperados; reward shaping excesivo (los brazos) | `03_recompensas_y_descuento.md` §5, §8, §10, §13 |
| ~02:41:17-02:43:28 | Recompensas intermedias (cocinar) y diferencias entre activos (Santander vs Viscofan) | `03_recompensas_y_descuento.md` §5, §8, §13 |
| ~02:43:28-02:47:03 | Vídeos: agente que aprende a correr; escondite multiagente | `03_recompensas_y_descuento.md` §5, §8, §9, §13 |
| ~02:47:03-02:50:34 | Simplicidad, magnitud de las recompensas, ponderación de subobjetivos y registro por criterio | `03_recompensas_y_descuento.md` §5, §6, §7, §8, §9, §11, …; `05_profit_seeking.md` §7; también `10_hoja_de_ruta_practica.md` §6.3, §6.4, §8.3 |
| ~02:50:34-02:55:15 | Signo de la recompensa y gamma: cliff walking; interacción de elementos; semillas | `03_recompensas_y_descuento.md` §6, §7, §8, §9, §11, §13; `04_exploracion_memoria_y_estabilidad.md` §8.2, §8.4; `09_material_rl_notebooks.md` §6, §13; también `05_profit_seeking.md` §7, `07_rl_para_ejecucion.md` §1, `10_hoja_de_ruta_practica.md` §2, §6.3, §6.4, … |
| ~02:55:15-02:57:00 | Reglas deterministas junto a la red | `03_recompensas_y_descuento.md` §8, §9, §13; `04_exploracion_memoria_y_estabilidad.md` §8.2, §8.4; `05_profit_seeking.md` §8; `07_rl_para_ejecucion.md` §1; también `09_material_rl_notebooks.md` §6, `10_hoja_de_ruta_practica.md` §2 |
| ~02:57:00-03:00:40 | Anuncio del enunciado para mañana; anécdota del TFM de 2019 sobre algoritmos de ejecución en BME | `notebook_partida.md` §11, §13; `06_algoritmos_de_ejecucion.md` §1; `07_rl_para_ejecucion.md` §«Mapa del documento», §19, §21; también `03_recompensas_y_descuento.md` §8, §9, §13, `04_exploracion_memoria_y_estabilidad.md` §10, §11, `05_profit_seeking.md` §1, §8 |
| ~03:00:40-03:06:15 | Epsilon-greedy y decaimiento de epsilon: por episodios (arbitrario) vs por mejora | `04_exploracion_memoria_y_estabilidad.md` §2.1, §2.2, §3.1, §3.2, §3.4, §3.6, …; `09_material_rl_notebooks.md` §1, §1.1, §4.2, §4.3, §6, §13; también `07_rl_para_ejecucion.md` §«Mapa del documento», §19, `10_hoja_de_ruta_practica.md` §7.3, §8.2, `notebook_partida.md` §13 |
| ~03:06:15-03:09:47 | Exploración excesiva o ineficiente: sobreajuste a experiencias malas | `04_exploracion_memoria_y_estabilidad.md` §2.2, §3.6, §6.7, §13; también `01_aproximacion_funcional_y_dqn.md` §6.6 |
| ~03:09:47-03:13:35 | Buffers enormes con ruido y necesidad de priorizar experiencias | `01_aproximacion_funcional_y_dqn.md` §6.6; `04_exploracion_memoria_y_estabilidad.md` §3.6, §6.7; también `00_repaso_rl.md` §7.3 |
| ~03:13:35-03:16:41 | Pregunta de un alumno: ¿sirve lo aprendido llegando por azar? Tabular vs redes | `00_repaso_rl.md` §7.3; `04_exploracion_memoria_y_estabilidad.md` §3.2, §3.6, §6.7; también `10_hoja_de_ruta_practica.md` §2, §7.2 |
| ~03:16:41-03:21:38 | Evaluación greedy periódica, bajada de epsilon según el objetivo y checkpoints | `04_exploracion_memoria_y_estabilidad.md` §2.2, §3.2, §3.3, §3.4, §3.5, §3.6; también `00_repaso_rl.md` §7.3, `10_hoja_de_ruta_practica.md` §2, §7.2, §7.3, … |
| ~03:21:38-03:26:17 | Exploración vs explotación; objetivos numéricos, baselines y benchmarks; mapa de Madrid | `03_recompensas_y_descuento.md` §2; `04_exploracion_memoria_y_estabilidad.md` §3.3, §3.4, §3.5, §4.1; `05_profit_seeking.md` §4.5, §9.2, §9.4; `06_algoritmos_de_ejecucion.md` §7; también `10_hoja_de_ruta_practica.md` §7.3, §11 |
| ~03:26:17-03:30:18 | Epsilon-greedy no explora activamente; principales escollos; ejemplo River Swim del notebook | `04_exploracion_memoria_y_estabilidad.md` §3.5, §4.1, §4.2; `09_material_rl_notebooks.md` §1, §9.1, §9.4; también `01_aproximacion_funcional_y_dqn.md` §11 |
| ~03:30:18-03:35:29 | Técnicas de exploración: inicialización, ruido en la red, pseudo-conteo, error, curriculum learning, subobjetivos | `03_recompensas_y_descuento.md` §5; `04_exploracion_memoria_y_estabilidad.md` §4.1, §4.2, §4.4, §5.2, §5.4; `01_aproximacion_funcional_y_dqn.md` §6.6, §11; `02_diseno_de_entornos.md` §9.2; `11_glosario.md` §7, §«C», §«D», §«I», §«N», §«P», …; también `00_repaso_rl.md` §8, `09_material_rl_notebooks.md` §1, §4.2, §4.3, … |
| ~03:35:29-03:42:35 | Learning rate alfa: ejemplo de la ruta del Retiro y no estacionariedad (peso medio de la clase) | `03_recompensas_y_descuento.md` §10; `04_exploracion_memoria_y_estabilidad.md` §4.2, §5.2, §5.3, §5.4, §13; `00_repaso_rl.md` §8, §16; `11_glosario.md` §7, §«A»; también `09_material_rl_notebooks.md` §1 |
| ~03:42:35-03:52:44 | Calibrar gamma según el horizonte temporal y la definición de step | `03_recompensas_y_descuento.md` §10, §11, §12, §12.1, §12.2, §13; `04_exploracion_memoria_y_estabilidad.md` §5.3; también `00_repaso_rl.md` §8, `09_material_rl_notebooks.md` §1, §9.1, §9.2, …, `10_hoja_de_ruta_practica.md` §6.4, §11 |
| ~03:52:44-03:54:00 | Factores adicionales para gamma: signo de la recompensa y episodios finitos | `03_recompensas_y_descuento.md` §11, §12, §12.2, §13; también `10_hoja_de_ruta_practica.md` §6.4, §11 |
| ~03:54:00-04:02:20 | Tamaño del buffer: independencia vs obsolescencia; regla de las ~100 partidas (Atari 1M) | `04_exploracion_memoria_y_estabilidad.md` §6.1, §6.2, §6.3, §7, §13; `01_aproximacion_funcional_y_dqn.md` §6.2, §16; también `10_hoja_de_ruta_practica.md` §6.5, §11 |
| ~04:02:20-04:09:42 | Qué es un episodio (intradía, ajedrez) y cálculo de steps por episodio para dimensionar el buffer | `02_diseno_de_entornos.md` §1, §3.2, §3.4, §12; `04_exploracion_memoria_y_estabilidad.md` §6.3, §7, §13; `07_rl_para_ejecucion.md` §4.3, §10.2; también `01_aproximacion_funcional_y_dqn.md` §6.2, `10_hoja_de_ruta_practica.md` §6.5, §11 |
| ~04:09:42-04:11:05 | Justificar las decisiones: lo que preguntará en la práctica | `notebook_partida.md` §4.2, §8.2, §15; también `01_aproximacion_funcional_y_dqn.md` §6.3, `02_diseno_de_entornos.md` §13, `04_exploracion_memoria_y_estabilidad.md` §1, §6.3, §6.4 |
| ~04:11:05-04:13:11 | Ajuste dinámico del tamaño del buffer según el error de predicción | `01_aproximacion_funcional_y_dqn.md` §6.2, §6.3; `04_exploracion_memoria_y_estabilidad.md` §1, §6.4, §6.5; también `10_hoja_de_ruta_practica.md` §8.5, `notebook_partida.md` §8.2 |
| ~04:13:11-04:14:33 | Implementación del buffer (deque) y eficiencia del entorno (pandas vs diccionarios) | `02_diseno_de_entornos.md` §8.1; `01_aproximacion_funcional_y_dqn.md` §6.2, §6.3, §6.4; también `04_exploracion_memoria_y_estabilidad.md` §6.4, §6.5, §6.6, `09_material_rl_notebooks.md` §9.3, §9.4, `10_hoja_de_ruta_practica.md` §8.5 |
| ~04:14:33-04:19:05 | Buffer vacío al inicio: warm-up y correlación de experiencias del mismo día | `04_exploracion_memoria_y_estabilidad.md` §6.5, §6.6; `01_aproximacion_funcional_y_dqn.md` §6.2, §6.4; también `09_material_rl_notebooks.md` §8.1, §8.4 |
| ~04:19:05-04:20:07 | Charla personal; al final, que el detalle del llenado del buffer se vea en el vídeo cuando toque | `notebook_partida.md` §11 (fila «Grabación»); el resto, descartado: charla personal sin contenido técnico |
| ~04:20:07-04:22:33 | Con qué rellenar el buffer inicial: aleatorio, experiencias reales o un algoritmo de reglas; riesgo de sesgo | `04_exploracion_memoria_y_estabilidad.md` §6.6; `01_aproximacion_funcional_y_dqn.md` §6.4; `07_rl_para_ejecucion.md` §10.3; también `10_hoja_de_ruta_practica.md` §6.5, §11 |
| ~04:22:33-04:31:02 | Debate: mezclar experiencias de distintos contextos; diversidad de episodios; tamaño del prellenado | `04_exploracion_memoria_y_estabilidad.md` §6.6, §6.7; `02_diseno_de_entornos.md` §3.2, §7.3; también `01_aproximacion_funcional_y_dqn.md` §6.4, §6.5, §6.6, `05_profit_seeking.md` §5.1, §5.2, `07_rl_para_ejecucion.md` §10.3 |
| ~04:31:02-04:31:50 | Gestión del tiempo de clase; las clases se graban | `notebook_partida.md` §11 (fila «Grabación»); el resto, descartado: logística de la sesión |
| ~04:31:50-04:37:19 | Prioritized Experience Replay, recompensas intrínsecas (Montezuma) y Combined Experience Replay | `01_aproximacion_funcional_y_dqn.md` §6.5, §6.6, §6.7, §16; `03_recompensas_y_descuento.md` §5; `04_exploracion_memoria_y_estabilidad.md` §6.7, §6.8, §13; también `10_hoja_de_ruta_practica.md` §6.5 |
| ~04:37:19-04:41:03 | Construcción del entorno: Gym/Gymnasium, reset y step, terminated y truncated (ejemplo Roomba) | `02_diseno_de_entornos.md` §2, §2.1; `11_glosario.md` §6, §«G», §«S», §«T»; también `01_aproximacion_funcional_y_dqn.md` §6.7 |
| ~04:41:03-04:44:07 | Qué determina el fin del episodio y las reglas de truncado | `02_diseno_de_entornos.md` §3.1, §7.1; también `07_rl_para_ejecucion.md` §4.2 |
| ~04:44:07-04:49:06 | reset: variar el estado inicial; empezar sobreajustando un día como test; escenarios de estrés | `02_diseno_de_entornos.md` §7.1, §9.1, §9.2; `04_exploracion_memoria_y_estabilidad.md` §9.1, §9.3; `07_rl_para_ejecucion.md` §4.2, §12; también `10_hoja_de_ruta_practica.md` §2, §8.4 |
| ~04:49:06-04:52:11 | Catastrophic forgetting y cómo leer la loss en RL | `04_exploracion_memoria_y_estabilidad.md` §9.3, §9.5; `11_glosario.md` §«C», §«L», §«M»; también `01_aproximacion_funcional_y_dqn.md` §13, `10_hoja_de_ruta_practica.md` §7.2, §8.4, `notebook_partida.md` §11 |
| ~04:52:11-04:58:00 | Logística de la práctica: formato (en directo o en la semana), exposición, grupos y orden | `notebook_partida.md` §11, §12, §15; también `10_hoja_de_ruta_practica.md` §12, §13 |
| ~04:58:00-05:00:13 | Cierre: contacto del profesor y despedida | `notebook_partida.md` §11, §15 |

### 1.2 T07 (07/11/2026)

| Tramo | Tema | Destino |
|---|---|---|
| ~00:00:00-00:03:51 | Charla previa al inicio de clase | descartado: Ruido y charla irrelevante antes de empezar la clase |
| ~00:03:51-00:08:03 | Montaje del entorno de trabajo: Docker y devcontainer del repo best-execution | `08_entorno_best_execution.md` §2.2; `notebook_partida.md` §10.2; `11_glosario.md` §6, §«D»; también `00_repaso_rl.md` §1, `09_material_rl_notebooks.md` §1, §2.1, §2.3 |
| ~00:08:03-00:10:19 | Ofrecimiento de repasar conceptos (experience replay, algoritmos) | `notebook_partida.md` §5.3, §10.2; `00_repaso_rl.md` §1; también `08_entorno_best_execution.md` §2.2, `10_hoja_de_ruta_practica.md` §6.6 |
| ~00:10:19-00:18:03 | Eficiencia de los entornos: pandas vs numpy vs diccionario; reconstrucción del libro de órdenes | `02_diseno_de_entornos.md` §1, §8.1, §8.2, §8.3, §8.4, §12; `08_entorno_best_execution.md` §2.3; `notebook_partida.md` §10.2, §13; también `00_repaso_rl.md` §1, `04_exploracion_memoria_y_estabilidad.md` §1, §3.2, §9.1, `09_material_rl_notebooks.md` §1, §9.4 |
| ~00:18:03-00:23:02 | Entrenamiento: sobreajuste inicial, evaluación greedy y sesgo del epsilon decay (cliff walking) | `04_exploracion_memoria_y_estabilidad.md` §1, §2.3, §3.2, §8.4, §9.1, §9.4; `09_material_rl_notebooks.md` §1, §6, §13, §14; también `00_repaso_rl.md` §5.3, §15, `01_aproximacion_funcional_y_dqn.md` §11, `02_diseno_de_entornos.md` §8.4, §9.1 |
| ~00:23:02-00:26:15 | No todo es IA: combinar supervisado, RL y reglas; elegir y combinar algoritmos | `00_repaso_rl.md` §5.3, §13, §15; `01_aproximacion_funcional_y_dqn.md` §11, §16; `06_algoritmos_de_ejecucion.md` §13.1, §13.2; `07_rl_para_ejecucion.md` §1, §5, §10.1; `notebook_partida.md` §13; también `02_diseno_de_entornos.md` §5.9, §6.1, `04_exploracion_memoria_y_estabilidad.md` §2.3, §8.4, §10, `09_material_rl_notebooks.md` §6, §8.2, §8.4, … |
| ~00:26:15-00:29:03 | Prueba y error, conocimiento de negocio, inestabilidad y búsqueda de hiperparámetros | `04_exploracion_memoria_y_estabilidad.md` §8.1, §8.4, §10, §11; también `00_repaso_rl.md` §5.3, §15, `01_aproximacion_funcional_y_dqn.md` §1, §11, `02_diseno_de_entornos.md` §8.1 |
| ~00:29:03-00:34:22 | Retos principales: entorno simulado realista, producción supervisada, recompensas como proxy | `03_recompensas_y_descuento.md` §3; `04_exploracion_memoria_y_estabilidad.md` §2.4, §9.2, §10, §11; `02_diseno_de_entornos.md` §1, §5.4, §11; también `10_hoja_de_ruta_practica.md` §2, `notebook_partida.md` §7.2 |
| ~00:34:22-00:42:13 | Repaso: exploración guiada por la mejora, pseudo-conteo y exploración dinámica | `03_recompensas_y_descuento.md` §3, §12.2; `04_exploracion_memoria_y_estabilidad.md` §1, §2.4, §3.1, §4.2, §4.3, §4.4, …; también `02_diseno_de_entornos.md` §6.6, `09_material_rl_notebooks.md` §9.1, §9.4, `10_hoja_de_ruta_practica.md` §6.4, §6.5, §7.2, … |
| ~00:42:13-00:44:08 | Notebook de profit-seeking de rl-material | `09_material_rl_notebooks.md` §1, §3.2, §10; `05_profit_seeking.md` §10.1, §10.8; `notebook_partida.md` §13; también `04_exploracion_memoria_y_estabilidad.md` §1 |
| ~00:44:08-00:46:27 | Trading algorítmico: profit-seeking frente a ejecución | `05_profit_seeking.md` §1, §10.8; `06_algoritmos_de_ejecucion.md` §1, §15; también `09_material_rl_notebooks.md` §10, `notebook_partida.md` §13 |
| ~00:46:27-00:53:35 | Por qué es difícil el profit-seeking con RL | `05_profit_seeking.md` §1, §2 |
| ~00:53:35-01:03:03 | Profit-seeking: objetivos, utilidad y planteamiento del problema | `05_profit_seeking.md` §2, §4.1, §4.2, §4.3, §4.4, §4.5; también `03_recompensas_y_descuento.md` §1 |
| ~01:03:03-01:10:38 | Profit-seeking: episodios, steps y calibración de gamma | `05_profit_seeking.md` §4.5, §5.1, §5.2, §13; `03_recompensas_y_descuento.md` §8, §12, §12.1, §12.2, §13; también `10_hoja_de_ruta_practica.md` §6.4, §8.2 |
| ~01:10:38-01:16:04 | Profit-seeking: características del estado y conocimiento de negocio | `05_profit_seeking.md` §5.2, §6.2, §6.3, §6.4; también `03_recompensas_y_descuento.md` §8, §12, §12.1, …, `10_hoja_de_ruta_practica.md` §6.4, §8.2 |
| ~01:16:04-01:19:02 | Pregunta de alumno sobre ranking con macro; criterio de evaluación de TFM y práctica | `05_profit_seeking.md` §4.4, §4.5, §6.4, §9.4, §13; `notebook_partida.md` §4.2; también `03_recompensas_y_descuento.md` §5, §7, §8, `10_hoja_de_ruta_practica.md` §1, §2 |
| ~01:19:02-01:21:23 | Profit-seeking: recompensas | `05_profit_seeking.md` §7, §9.4, §13; `03_recompensas_y_descuento.md` §5, §7, §8, §13; también `notebook_partida.md` §4.2 |
| ~01:21:23-01:28:01 | Profit-seeking: acciones discretas y continuas, estado interno y restricciones | `05_profit_seeking.md` §6.4, §7, §8, §10.6; también `03_recompensas_y_descuento.md` §5, §8, §13 |
| ~01:28:01-01:32:04 | Profit-seeking: validación, backtesting e impacto de mercado | `05_profit_seeking.md` §8, §9.2, §9.4; también `03_recompensas_y_descuento.md` §2, §13 |
| ~01:32:04-01:35:44 | Recompensa frente al mercado y presentación de resultados contra benchmark | `05_profit_seeking.md` §7, §9.2, §9.3, §9.4, §10.1, §10.8; `03_recompensas_y_descuento.md` §2, §13 |
| ~01:35:44-01:43:01 | Algoritmos de ejecución: orden madre/hijas, VWAP, dilema del trader y urgencia | `06_algoritmos_de_ejecucion.md` §2.2, §3, §4.2; `11_glosario.md` §6, §«A», §«B», §«D», §«H», §«I», …; también `03_recompensas_y_descuento.md` §2, §13, `05_profit_seeking.md` §9.4, §10.8, `notebook_partida.md` §2.1 |
| ~01:43:01-01:50:09 | Tipos de algoritmo y definición del objetivo del caso RL (VWAP slippage) | `06_algoritmos_de_ejecucion.md` §3, §5, §6, §7, §14, §15; `07_rl_para_ejecucion.md` §2, §15; `notebook_partida.md` §2.1, §2.4, §4.2, §7.2; `03_recompensas_y_descuento.md` §2; `04_exploracion_memoria_y_estabilidad.md` §3.3; también `10_hoja_de_ruta_practica.md` §2, §4.3, §6.3, … |
| ~01:50:09-01:58:18 | TWAP, VWAP y POV: funcionamiento y gráficos | `06_algoritmos_de_ejecucion.md` §7, §8.1, §8.2, §9.1, §9.2, §10.1, …; `11_glosario.md` §6, §«C», §«O», §«P», §«T», §«U», …; también `notebook_partida.md` §2.1, §4.2, §13 |
| ~01:58:18-02:02:24 | Comparativa de costes y uso real de TWAP/VWAP/POV; pregunta abierta de mejora | `06_algoritmos_de_ejecucion.md` §3, §10.3, §11.2, §11.3, §15; `07_rl_para_ejecucion.md` §15; `notebook_partida.md` §13; también `10_hoja_de_ruta_practica.md` §8.2 |
| ~02:02:24-02:30:00 | Descanso | descartado: Descanso: ruido, fragmentos ininteligibles y silencio |
| ~02:30:00-02:33:00 | Pausa / ruido de transcripción | descartado: pausa y ruido sin contenido |
| ~02:33:00-02:36:45 | Charla sobre consultoría y carrera profesional | descartado: charla personal/laboral irrelevante para el temario |
| ~02:36:45-02:42:20 | Limitaciones del TWAP clásico y del POV; primeras ideas de mejora | `06_algoritmos_de_ejecucion.md` §2.3, §8.1, §9.2, §10.3, §12.3; `07_rl_para_ejecucion.md` §1 |
| ~02:42:20-02:46:00 | Cómo está programado realmente un TWAP (BME) y otras reglas | `06_algoritmos_de_ejecucion.md` §12.2, §12.3; `07_rl_para_ejecucion.md` §1, §8.4 |
| ~02:46:00-02:51:20 | Explosión de reglas deterministas → motivación del RL | `06_algoritmos_de_ejecucion.md` §12.3, §12.4; `07_rl_para_ejecucion.md` §1, §7.2, §7.4; `00_repaso_rl.md` §5.3, §15 |
| ~02:51:20-02:54:45 | Combinar supervisado y RL: probabilidad de ejecución de la pasiva | `06_algoritmos_de_ejecucion.md` §12.4, §13.2; `07_rl_para_ejecucion.md` §1, §2, §7.4; `00_repaso_rl.md` §5.3, §15 |
| ~02:54:45-02:57:20 | Objetivo del caso y papel del entorno | `06_algoritmos_de_ejecucion.md` §6; `07_rl_para_ejecucion.md` §1, §2, §3, §4.1, §7.4; `02_diseno_de_entornos.md` §1, §3.3; también `00_repaso_rl.md` §5.3, §15, `notebook_partida.md` §2.2 |
| ~02:57:20-03:02:40 | Generación de episodios: el 'TWAP con esteroides' y el triciclo | `07_rl_para_ejecucion.md` §4.1, §4.2, §4.3, §5; `02_diseno_de_entornos.md` §3.2, §3.3, §9.2; `notebook_partida.md` §2.2; también `10_hoja_de_ruta_practica.md` §2 |
| ~03:02:40-03:07:00 | Acotar el problema: qué dimensión decide el agente (simplificación de la práctica) | `notebook_partida.md` §2.2, §2.3, §2.4; `06_algoritmos_de_ejecucion.md` §2.3; `07_rl_para_ejecucion.md` §4.2, §4.3, §5, §8.5, §21; también `02_diseno_de_entornos.md` §3.3, §12, `10_hoja_de_ruta_practica.md` §2, §8.5 |
| ~03:07:00-03:08:25 | Aislar problemas y subsistemas | `06_algoritmos_de_ejecucion.md` §2.3, §13.2, §13.3; `07_rl_para_ejecucion.md` §5, §6; también `02_diseno_de_entornos.md` §3.3, `10_hoja_de_ruta_practica.md` §2, `notebook_partida.md` §2.3, §4.2 |
| ~03:08:25-03:12:00 | Definición de steps: resolución de decisión vs resolución del simulador | `07_rl_para_ejecucion.md` §5, §6; `02_diseno_de_entornos.md` §1, §4.3, §4.4, §12; también `10_hoja_de_ruta_practica.md` §8.5 |
| ~03:12:00-03:15:20 | Probabilidad de ejecución por nivel de precio como característica | `07_rl_para_ejecucion.md` §7.1, §7.4, §8.1, §21; también `02_diseno_de_entornos.md` §4.4, §5.9, §12 |
| ~03:15:20-03:22:35 | Diseño del vector de características y normalización | `07_rl_para_ejecucion.md` §7.1, §7.2, §7.3, §7.4; `02_diseno_de_entornos.md` §5.2, §5.6, §5.7; también `10_hoja_de_ruta_practica.md` §6.2 |
| ~03:22:35-03:30:10 | Espacio de acciones: tipos de orden, acciones dinámicas y reglas deterministas | `06_algoritmos_de_ejecucion.md` §12.2; `07_rl_para_ejecucion.md` §1, §7.1, §7.2, §7.3, §8.1, §8.2, …; también `02_diseno_de_entornos.md` §5.7, §6.3, §6.5, `03_recompensas_y_descuento.md` §4.4, §6, §8 |
| ~03:30:10-03:35:00 | Recompensas: VWAP global/local, frente al TWAP, volumen pendiente y calibración | `07_rl_para_ejecucion.md` §8.4, §9.1, §9.2, §16; `03_recompensas_y_descuento.md` §4.4, §6, §8, §13; también `02_diseno_de_entornos.md` §6.5, `10_hoja_de_ruta_practica.md` §6.3, §8.2, §8.3 |
| ~03:35:00-03:41:30 | Recompensas factibles: por qué el VWAP global es difícil de aprender | `07_rl_para_ejecucion.md` §9, §9.1, §9.3, §9.4; `03_recompensas_y_descuento.md` §4, §4.2, §4.4, §13; `05_profit_seeking.md` §7, §13; también `10_hoja_de_ruta_practica.md` §2, §6.3, §8.2, … |
| ~03:41:30-03:44:30 | Ejemplo de recompensa súper específica: limitada vs agresiva del TWAP | `06_algoritmos_de_ejecucion.md` §13.3; `07_rl_para_ejecucion.md` §9.3, §9.4, §10.1, §21; `03_recompensas_y_descuento.md` §4, §4.3, §4.4, §13; también `01_aproximacion_funcional_y_dqn.md` §11, `05_profit_seeking.md` §7, `10_hoja_de_ruta_practica.md` §2, §8.2 |
| ~03:44:30-03:51:30 | Agentes y tamaño del replay buffer en el caso de ejecución | `03_recompensas_y_descuento.md` §4.3, §4.4, §13; `04_exploracion_memoria_y_estabilidad.md` §6.2, §6.3, §6.6, §12; `01_aproximacion_funcional_y_dqn.md` §2.3, §6.2, §6.4, §11; `07_rl_para_ejecucion.md` §9.4, §10.1, §10.2, §10.3, §11; también `00_repaso_rl.md` §3.2, `09_material_rl_notebooks.md` §6, `10_hoja_de_ruta_practica.md` §2, §6.5, §8.2, … |
| ~03:51:30-03:53:00 | Versión tabular y evaluación | `00_repaso_rl.md` §3.2, §15; `03_recompensas_y_descuento.md` §3; `04_exploracion_memoria_y_estabilidad.md` §6.6; también `01_aproximacion_funcional_y_dqn.md` §2.3, §6.4, `07_rl_para_ejecucion.md` §«Mapa del documento», §10.3, §11, …, `10_hoja_de_ruta_practica.md` §2, §6.5 |
| ~03:53:00-03:58:15 | Resultados del TFM del profesor: TWAP conservador, moderado y agresivo | `07_rl_para_ejecucion.md` §1, §4.2, §13, §14, §15, §16; también `03_recompensas_y_descuento.md` §6, §8, `05_profit_seeking.md` §11, `10_hoja_de_ruta_practica.md` §6.8, §8.2, §10, … |
| ~03:58:15-04:02:10 | Lectura de resultados y relación con algoritmos de inversión/HFT | `06_algoritmos_de_ejecucion.md` §13.4; `07_rl_para_ejecucion.md` §14, §15, §17.1, §21; `05_profit_seeking.md` §11; también `10_hoja_de_ruta_practica.md` §6.2, `notebook_partida.md` §6.3 |
| ~04:02:10-04:06:00 | Papers: Q-learning para ejecución (2006) y algoritmo de JP Morgan | `06_algoritmos_de_ejecucion.md` §13.4; `07_rl_para_ejecucion.md` §15, §17.1, §17.2, §17.3, §21; también `05_profit_seeking.md` §11, `10_hoja_de_ruta_practica.md` §6.2, `notebook_partida.md` §6.3 |
| ~04:06:00-04:10:30 | Lecciones del TFM: no abarcar demasiado | `06_algoritmos_de_ejecucion.md` §13.4; `07_rl_para_ejecucion.md` §7.4, §19; también `00_repaso_rl.md` §5.4, §15, `01_aproximacion_funcional_y_dqn.md` §4.2 |
| ~04:10:30-04:16:00 | Supervisado vs RL: utilidad esperada como programación dinámica y benchmark | `00_repaso_rl.md` §5.4, §15, §16; `06_algoritmos_de_ejecucion.md` §13.3; `07_rl_para_ejecucion.md` §7.4, §18, §19; también `01_aproximacion_funcional_y_dqn.md` §14, `03_recompensas_y_descuento.md` §4.3, §4.4, §13 |
| ~04:16:00-04:21:00 | Cuándo compensa el RL; LeCun, DeepMind, simuladores | `00_repaso_rl.md` §5.4, §14, §15; `01_aproximacion_funcional_y_dqn.md` §14; también `03_recompensas_y_descuento.md` §3, `07_rl_para_ejecucion.md` §18 |
| ~04:21:00-04:27:20 | LLMs jugando, Atari y ejemplo del aparcamiento | `00_repaso_rl.md` §14; `04_exploracion_memoria_y_estabilidad.md` §8.3; también `01_aproximacion_funcional_y_dqn.md` §14, `10_hoja_de_ruta_practica.md` §10, §13, `notebook_partida.md` §1, §11 |
| ~04:27:20-04:31:40 | Presentación de la práctica: los 4 ejercicios y su puntuación | `notebook_partida.md` §1, §4.2, §5.3, §7.2, §8.2, §11, …; también `00_repaso_rl.md` §14, `03_recompensas_y_descuento.md` §6, §14, `04_exploracion_memoria_y_estabilidad.md` §3.7, §6.4, §12 |
| ~04:31:40-04:36:00 | Requisitos técnicos y ficheros del taller (execution_lab y step_by_step) | `notebook_partida.md` §1, §4.2, §5.3, §8.2, §10.2, §12, …; `08_entorno_best_execution.md` §1, §2.2, §13; también `04_exploracion_memoria_y_estabilidad.md` §6.4, §12, `10_hoja_de_ruta_practica.md` §1, §3.1, §5.2, … |
| ~04:36:00-04:38:30 | Parámetros de la orden en el step_by_step | `notebook_partida.md` §5.3, §9; `08_entorno_best_execution.md` §6.1, §13; también `10_hoja_de_ruta_practica.md` §3.1 |
| ~04:38:30-04:45:00 | Datos SAN/REP, estructura de los pickles y asunciones de la práctica | `notebook_partida.md` §2.4, §5.3, §6.3, §9, §12, §15; `08_entorno_best_execution.md` §1, §3.1, §3.2, §3.4, §3.5, §6.2, …; `02_diseno_de_entornos.md` §3.3, §10, §12; también `07_rl_para_ejecucion.md` §4.4, `10_hoja_de_ruta_practica.md` §3.1, §4.4, §5.2, … |
| ~04:45:00-04:48:25 | Cálculo de steps y tamaño de la orden (flexibilidad) | `notebook_partida.md` §2.4, §5.3, §6.3, §12; `08_entorno_best_execution.md` §3.5, §12.4, §15; también `02_diseno_de_entornos.md` §3.3, §4.4, §10, …, `07_rl_para_ejecucion.md` §4.4, §21, `10_hoja_de_ruta_practica.md` §8.5 |
| ~04:48:25-04:52:10 | Bucle de entrenamiento y función reset | `notebook_partida.md` §5.3, §6.2, §6.3, §15; `08_entorno_best_execution.md` §3.5, §5, §6.2, §6.3, §6.4, §11, …; `02_diseno_de_entornos.md` §7.2; también `03_recompensas_y_descuento.md` §4.4, `04_exploracion_memoria_y_estabilidad.md` §2.2, `07_rl_para_ejecucion.md` §4.4, §7.5, §9.5, … |
| ~04:52:10-04:56:34 | Observación (tiempo y volumen restantes) y TWAP determinista / tabular | `notebook_partida.md` §4.2, §5.3, §6.2, §6.3, §12, §15; `08_entorno_best_execution.md` §6.3, §6.6, §7, §9.6; `06_algoritmos_de_ejecucion.md` §8.4; `07_rl_para_ejecucion.md` §7.5, §20; también `02_diseno_de_entornos.md` §5.7, `03_recompensas_y_descuento.md` §4.4, §13, `10_hoja_de_ruta_practica.md` §2, §5.2, §5.3, … |
| ~04:56:34-05:01:00 | Función step, qué toca cada ejercicio y lookback | `notebook_partida.md` §1, §2.3, §4.2, §5.3, §6.3, §7.2, …; `08_entorno_best_execution.md` §4, §5, §6.5, §6.6, §6.11, §11, …; también `02_diseno_de_entornos.md` §3.3, §6.7, §7.2, `03_recompensas_y_descuento.md` §4.4, §6, §7, …, `04_exploracion_memoria_y_estabilidad.md` §5.4 |
| ~05:01:00-05:05:40 | Atributos del entorno, alternativa Q-learning y precio de ejecución entre fotos | `notebook_partida.md` §1, §4.2, §5.3, §11, §12, §15; `08_entorno_best_execution.md` §1, §6.7, §8, §9.6, §15; `02_diseno_de_entornos.md` §7.2, §10; también `04_exploracion_memoria_y_estabilidad.md` §5.4, §12, `10_hoja_de_ruta_practica.md` §6.6, §8.2, §8.5, … |
| ~05:05:40-05:07:10 | Logística: grupos, correo, exposición del día 18 | `notebook_partida.md` §1, §11, §12, §15; también `10_hoja_de_ruta_practica.md` §8.2, §8.5, §12, … |
| ~05:07:10-05:10:58 | Dudas finales y cierre | `notebook_partida.md` §1, §4.2, §11, §12, §15; también `10_hoja_de_ruta_practica.md` §13, §15 |

## 2. Diapositivas, por bloques temáticos

### 2.1 APP ("Applying Reinforcement Learning")

| Páginas | Tema | Destino |
|---|---|---|
| p.1-3 | Portada, aviso previo y agenda del Taller (3 días) | `notebook_partida.md` §11, §13; `00_repaso_rl.md` §1; `09_material_rl_notebooks.md` §1, §1.1, §13; también `04_exploracion_memoria_y_estabilidad.md` §2.4, `10_hoja_de_ruta_practica.md` §6.6 |
| p.4-8 | Componentes de un sistema RL; V, π y Q con el ejemplo del paraguas | `00_repaso_rl.md` §2, §3.1; `11_glosario.md` §«G», §«P», §«Q», §«V» |
| p.9-11 | Propiedad de Markov: ejemplo del coche y aplicación a bolsa y juegos | `00_repaso_rl.md` §4.1, §4.2, §15; `05_profit_seeking.md` §6.1, §6.4; `11_glosario.md` §«B», §«M» |
| p.12-15 | Cuándo usar RL frente a supervisado; impacto de nuestras órdenes | `00_repaso_rl.md` §5.1, §5.2, §15; `06_algoritmos_de_ejecucion.md` §2.4, §13.3; `07_rl_para_ejecucion.md` §1; `05_profit_seeking.md` §3, §13 |
| p.16-18 | Objetivo del RL, retorno descontado y gamma | `00_repaso_rl.md` §6, §16; `11_glosario.md` §«B», §«C», §«G», §«O», §«R», §«U»; también `03_recompensas_y_descuento.md` §1, §10 |
| p.19-21 | MC, programación dinámica y TD (diagramas de backup) | `00_repaso_rl.md` §7.1, §7.2, §16; `11_glosario.md` §7, §«B», §«M», §«P», §«T» |
| p.22-25 | Q-learning: fórmula, ejemplo numérico y pseudocódigo | `00_repaso_rl.md` §9.2; `09_material_rl_notebooks.md` §5.1, §5.5 |
| p.26-28 | SARSA, Expected SARSA y resumen comparativo | `00_repaso_rl.md` §10, §11.2, §11.3, §15, §16; `09_material_rl_notebooks.md` §5.3, §5.5, §6 |
| p.29-30 | Double Q-learning | `00_repaso_rl.md` §12, §15; también `01_aproximacion_funcional_y_dqn.md` §8.2, `09_material_rl_notebooks.md` §5.4, §5.5 |
| p.31-36 | Aproximación funcional, descenso por gradiente y aproximación lineal | `01_aproximacion_funcional_y_dqn.md` §2, §2.1, §3, §4.1, §4.3, §6.6, …; `09_material_rl_notebooks.md` §7, §14; también `00_repaso_rl.md` §13 |
| p.37-40 | Deep Q-learning: problemas, experience replay y fixed target | `01_aproximacion_funcional_y_dqn.md` §5, §5.1, §5.2, §6.2, §7; `04_exploracion_memoria_y_estabilidad.md` §6.1 |
| p.41-45 | Double DQN, Dueling DQN y resumen de mejoras | `01_aproximacion_funcional_y_dqn.md` §1, §5.2, §8.1, §8.2, §9, §16; `11_glosario.md` §5, §7, §«D», §«P», §«V» |
| p.46-48 | Segunda parte: papers de DeepMind y mapa de piezas a ajustar | `01_aproximacion_funcional_y_dqn.md` §14, §16; `04_exploracion_memoria_y_estabilidad.md` §1; también `00_repaso_rl.md` §14, `02_diseno_de_entornos.md` §1 |
| p.49-53 | Estados: vector de características y requisitos | `02_diseno_de_entornos.md` §5.1, §5.2, §5.3, §5.7, §5.9, §12; `05_profit_seeking.md` §6.1, §6.4, §10.2, §10.6, §13; también `00_repaso_rl.md` §4.4, `09_material_rl_notebooks.md` §5.1, §5.5 |
| p.54-55 | Definición del step | `02_diseno_de_entornos.md` §4.1, §12; `07_rl_para_ejecucion.md` §6 |
| p.56-57 | Acciones: discretas/continuas, eliminación y aleatoriedad | `02_diseno_de_entornos.md` §6.1, §6.2, §6.3, §6.6; `07_rl_para_ejecucion.md` §8.1, §8.2, §10.3; `04_exploracion_memoria_y_estabilidad.md` §4.5; también `10_hoja_de_ruta_practica.md` §6.5, §11 |
| p.58-62 | Recompensas: objetivo, sparse vs shaping, magnitud y prueba masiva | `03_recompensas_y_descuento.md` §2, §3, §4.4, §5, §6, §8, …; `04_exploracion_memoria_y_estabilidad.md` §3.6, §8.2, §10, §13; `07_rl_para_ejecucion.md` §16, §19; `05_profit_seeking.md` §4.1, §4.5, §7; también `10_hoja_de_ruta_practica.md` §6.3, §8.3 |
| p.63-67 | Exploración: ε-greedy, crítica del decay por episodios y coste de oportunidad | `03_recompensas_y_descuento.md` §2; `04_exploracion_memoria_y_estabilidad.md` §2.1, §2.2, §2.4, §3.1, §3.3, §3.5; también `10_hoja_de_ruta_practica.md` §2, §7.2, §7.3, …, `notebook_partida.md` §7.2 |
| p.68-73 | Exploración ilustrada en el mapa de Madrid; River Swim y alternativas a ε-greedy | `03_recompensas_y_descuento.md` §5; `04_exploracion_memoria_y_estabilidad.md` §3.5, §4.1, §4.2, §4.4; `11_glosario.md` §7, §«C», §«D», §«E», §«N», §«P», …; también `09_material_rl_notebooks.md` §9.1, §9.4, §12, `10_hoja_de_ruta_practica.md` §7.3, §8.5 |
| p.74-76 | Step size (α) y gamma | `03_recompensas_y_descuento.md` §10, §11, §12.1, §12.2; `04_exploracion_memoria_y_estabilidad.md` §5.2, §5.4, §13; `00_repaso_rl.md` §6, §8, §15, §16; también `09_material_rl_notebooks.md` §4.2, §4.3, §9.2, …, `10_hoja_de_ruta_practica.md` §6.2, §6.4, §8.5 |
| p.77-82 | Experience Replay: tamaño del buffer, PER y CER | `01_aproximacion_funcional_y_dqn.md` §6.2, §6.3, §6.4, §6.6, §6.7, §10, …; `03_recompensas_y_descuento.md` §5; `04_exploracion_memoria_y_estabilidad.md` §6.1, §6.2, §6.3, §6.4, §6.5, §6.6, …; también `08_entorno_best_execution.md` §9.3, `09_material_rl_notebooks.md` §8.1, §8.4, §9.3, …, `10_hoja_de_ruta_practica.md` §6.5, §8.5 |
| p.83-86 | Definición del entorno: API gymnasium, episodios y reset | `02_diseno_de_entornos.md` §1, §2, §2.1, §3.1, §7.1, §9.2, …; `11_glosario.md` §«D», §«E», §«G», §«R», §«T»; también `03_recompensas_y_descuento.md` §7, `10_hoja_de_ruta_practica.md` §6.3 |
| p.87-92 | Entrenamiento: inestabilidad, eficiencia, validación y conclusiones | `04_exploracion_memoria_y_estabilidad.md` §3.2, §8.1, §8.4, §9.1, §9.2, §9.3, …; `02_diseno_de_entornos.md` §5.9, §8.1, §9.1; `01_aproximacion_funcional_y_dqn.md` §1, §11, §13, §16; también `10_hoja_de_ruta_practica.md` §2, §7.2, §8.4 |
| p.93-97 | Retos del RL: exploración, recompensas, simulador, producción; entornos con IA generativa 2026 | `03_recompensas_y_descuento.md` §8; `04_exploracion_memoria_y_estabilidad.md` §11; `02_diseno_de_entornos.md` §1, §5.6, §10, §11; `07_rl_para_ejecucion.md` §16; también `00_repaso_rl.md` §14 |
| p.98-100 | Apéndice: repaso de componentes, políticas, V y Q | `00_repaso_rl.md` §2, §3.1; `11_glosario.md` §«A», §«E», §«P», §«R», §«V»; también `03_recompensas_y_descuento.md` §9, §13 |
| p.101-102 | N-step TD: ventaja (velocidad) y desventaja (varianza) | `00_repaso_rl.md` §7.4, §16; `01_aproximacion_funcional_y_dqn.md` §12, §16 |
| p.103 | Cierre | descartado: Diapositiva de cierre/contacto sin contenido |

### 2.2 CASOS ("Casos Prácticos")

| Páginas | Tema | Destino |
|---|---|---|
| p.1-2 | Portada y aviso del bloque Casos Prácticos | descartado: Portada, aviso genérico y datos de contacto; sin contenido técnico ni de la práctica |
| p.3-5 | Algorithmic trading: profit-seeking vs execution; dificultades del profit-seeking | `05_profit_seeking.md` §1, §2; `06_algoritmos_de_ejecucion.md` §1; `11_glosario.md` §«A», §«C», §«E», §«P», §«S» |
| p.6-9 | Profit-seeking: objetivo, utilidad y objetivos relativos | `05_profit_seeking.md` §1, §4.1, §4.2, §4.4, §4.5, §10.6, …; `11_glosario.md` §2, §7, §«A», §«I», §«O», §«S», …; también `03_recompensas_y_descuento.md` §1 |
| p.10-13 | Profit-seeking: estados/episodios, recompensas, acciones y validación de resultados | `05_profit_seeking.md` §1, §5.1, §5.2, §6.2, §6.4, §7, …; `02_diseno_de_entornos.md` §3.2, §4.3, §5.6, §5.7, §6.5, §10; `03_recompensas_y_descuento.md` §5, §12.2 |
| p.14-15 | Notebook de profit-seeking y portada de Execution Algorithms | `05_profit_seeking.md` §1, §10.1, §10.8; `09_material_rl_notebooks.md` §10 |
| p.16-18 | Orden CARE, órdenes hijas, VWAP y Traders' Dilemma | `06_algoritmos_de_ejecucion.md` §2.2, §3, §15; `07_rl_para_ejecucion.md` §2; `11_glosario.md` §2, §5, §7, §«A», §«C», §«D», … |
| p.19-21 | Tipos de algoritmos de ejecución, objetivo aterrizado (VWAP slippage) y benchmark | `06_algoritmos_de_ejecucion.md` §3, §5, §6, §7; `07_rl_para_ejecucion.md` §2; `11_glosario.md` §«A», §«D», §«L», §«O», §«S», §«T», … |
| p.22-26 | Algoritmos tradicionales TWAP, VWAP y POV (esquemas y gráficos reales) | `06_algoritmos_de_ejecucion.md` §8.2, §9.1, §10.2, §15; `11_glosario.md` §«C», §«T», §«U», §«V» |
| p.27-28 | Desventajas de los algoritmos tradicionales y papel del ML/RL | `06_algoritmos_de_ejecucion.md` §3, §8.2, §9.1, §10.3, §11.1, §12.4, …; `07_rl_para_ejecucion.md` §1, §7.4, §15 |
| p.29-32 | Diseño RL del caso: objetivo, componentes y generación de órdenes (episodios) | `06_algoritmos_de_ejecucion.md` §2.2, §3, §6, §7, §10.3, §15; `07_rl_para_ejecucion.md` §2, §3, §4.1, §4.3, §15, §21; `02_diseno_de_entornos.md` §1, §3.2, §3.3, §10 |
| p.33-36 | Steps, libro de órdenes y vector de estado | `07_rl_para_ejecucion.md` §6, §7.1, §7.2, §7.3, §21; `02_diseno_de_entornos.md` §1, §4.3, §4.4, §5.6, §5.7, §12; también `10_hoja_de_ruta_practica.md` §8.5 |
| p.37-41 | Acciones: tipos, agresividad de terceros y tamaño de órdenes | `06_algoritmos_de_ejecucion.md` §2.3, §9.2, §12.2; `07_rl_para_ejecucion.md` §5, §8.1, §8.2, §8.3, §8.4, §21; `02_diseno_de_entornos.md` §1, §6.1, §6.3, §6.5, §12 |
| p.42-44 | Recompensas del algoritmo de ejecución | `07_rl_para_ejecucion.md` §8.1, §9, §9.1; `03_recompensas_y_descuento.md` §4.4, §6, §7; también `10_hoja_de_ruta_practica.md` §6.3, §8.3 |
| p.45-50 | Agentes, buffer y aprendizaje de Q-valores hasta la política | `07_rl_para_ejecucion.md` §10.3, §11, §21; `00_repaso_rl.md` §2, §3.2, §13, §16; `01_aproximacion_funcional_y_dqn.md` §2.3, §6.4, §11, §16; `04_exploracion_memoria_y_estabilidad.md` §4.3, §4.5, §6.6; también `10_hoja_de_ruta_practica.md` §6.5, §11 |
| p.51-53 | Análisis de resultados, estrés y perfiles de agresividad | `07_rl_para_ejecucion.md` §12, §13, §16, §21; `03_recompensas_y_descuento.md` §6, §8; `04_exploracion_memoria_y_estabilidad.md` §9.2; también `08_entorno_best_execution.md` §7, `10_hoja_de_ruta_practica.md` §8.2, §8.5 |
| p.54-59 | Gráficos de resultados por perfil (precio/VWAP y volumen acumulado) | `07_rl_para_ejecucion.md` §14, §21; también `08_entorno_best_execution.md` §12.4 |
| p.60 | Histograma de VWAP slippage en test: algo RL vs TWAP | `06_algoritmos_de_ejecucion.md` §3, §11.3, §13.4, §15; `07_rl_para_ejecucion.md` §2, §21; también `10_hoja_de_ruta_practica.md` §8.2 |
| p.61 | Referencias: Lehman Brothers 2006 y J.P. Morgan | `06_algoritmos_de_ejecucion.md` §13.4; `07_rl_para_ejecucion.md` §17.1, §17.2; también `10_hoja_de_ruta_practica.md` §8.2 |
| p.62-63 | Cierre y contacto | descartado: Diapositivas de cierre y contacto sin contenido |

## 3. Diapositivas, página a página

Secciones que citan cada página (páginas consecutivas con el mismo destino, agrupadas).

### 3.1 APP

| Página(s) | Destino |
|---|---|
| p.1 | `00_repaso_rl.md` §1; `04_exploracion_memoria_y_estabilidad.md` §2.4; `09_material_rl_notebooks.md` §1, §1.1, §13; `10_hoja_de_ruta_practica.md` §6.6; `11_glosario.md` §«R»; `notebook_partida.md` §13 |
| p.2 | `00_repaso_rl.md` §1; `notebook_partida.md` §11, §13 |
| p.3 | `00_repaso_rl.md` §1; `notebook_partida.md` §11 |
| p.4 | descartado: portadilla "Primera Parte" (sin contenido) |
| p.5 | `00_repaso_rl.md` §2 |
| p.6 | descartado: separador "¿Alguna duda?" (sin contenido) |
| p.7 | `00_repaso_rl.md` §3.1; `11_glosario.md` §«P», §«Q», §«V» |
| p.8 | `00_repaso_rl.md` §3.1; `11_glosario.md` §«G», §«P», §«V» |
| p.9-10 | `00_repaso_rl.md` §4.1, §15; `11_glosario.md` §«M» |
| p.11 | `00_repaso_rl.md` §4.1, §4.2, §15; `05_profit_seeking.md` §6.1, §6.4; `11_glosario.md` §«B», §«M» |
| p.12 | `00_repaso_rl.md` §5.1, §15; `05_profit_seeking.md` §3; `06_algoritmos_de_ejecucion.md` §13.3; `11_glosario.md` §«C», §«S» |
| p.13 | `00_repaso_rl.md` §5.2; `05_profit_seeking.md` §3 |
| p.14 | `00_repaso_rl.md` §5.2, §15; `05_profit_seeking.md` §3; `06_algoritmos_de_ejecucion.md` §2.4; `11_glosario.md` §«S» |
| p.15 | `00_repaso_rl.md` §5.2, §15; `05_profit_seeking.md` §3, §13; `06_algoritmos_de_ejecucion.md` §2.4; `07_rl_para_ejecucion.md` §1; `11_glosario.md` §«S» |
| p.16-17 | `00_repaso_rl.md` §6; `03_recompensas_y_descuento.md` §1; `11_glosario.md` §«C», §«O» |
| p.18 | `00_repaso_rl.md` §6, §16; `03_recompensas_y_descuento.md` §1, §10; `11_glosario.md` §«B», §«G», §«R», §«U» |
| p.19-20 | `00_repaso_rl.md` §7.1, §16; `11_glosario.md` §7, §«B», §«M», §«P», … |
| p.21 | `00_repaso_rl.md` §7.2; `11_glosario.md` §«M», §«P», §«T» |
| p.22 | `09_material_rl_notebooks.md` §5.1, §5.5; `11_glosario.md` §«O», §«Q» |
| p.23-25 | `00_repaso_rl.md` §9.2; `09_material_rl_notebooks.md` §5.1, §5.5 |
| p.26 | `00_repaso_rl.md` §10, §11.2, §15, §16; `09_material_rl_notebooks.md` §5.5, §6; `11_glosario.md` §«C», §«E», §«O», §«S» |
| p.27 | `00_repaso_rl.md` §11.2, §15; `09_material_rl_notebooks.md` §5.3, §5.5, §6; `11_glosario.md` §«E» |
| p.28 | `00_repaso_rl.md` §11.2, §11.3, §15; `09_material_rl_notebooks.md` §5.3, §5.5, §6; `11_glosario.md` §«E» |
| p.29 | `00_repaso_rl.md` §12, §15; `01_aproximacion_funcional_y_dqn.md` §8.2; `09_material_rl_notebooks.md` §5.4, §5.5; `11_glosario.md` §«D», §«S» |
| p.30 | `00_repaso_rl.md` §12, §15; `01_aproximacion_funcional_y_dqn.md` §8.2; `09_material_rl_notebooks.md` §5.4, §5.5; `11_glosario.md` §«D» |
| p.31 | `00_repaso_rl.md` §13; `01_aproximacion_funcional_y_dqn.md` §2.1; `09_material_rl_notebooks.md` §7; `11_glosario.md` §«A», §«T» |
| p.32 | `01_aproximacion_funcional_y_dqn.md` §2, §2.1; `09_material_rl_notebooks.md` §7; `11_glosario.md` §«A», §«G» |
| p.33 | `09_material_rl_notebooks.md` §7 |
| p.34 | `01_aproximacion_funcional_y_dqn.md` §3; `09_material_rl_notebooks.md` §7; `11_glosario.md` §«D» |
| p.35 | `01_aproximacion_funcional_y_dqn.md` §4.1, §4.3, §16; `09_material_rl_notebooks.md` §7, §14; `11_glosario.md` §«A» |
| p.36 | `01_aproximacion_funcional_y_dqn.md` §4.3, §6.6; `09_material_rl_notebooks.md` §7 |
| p.37 | `01_aproximacion_funcional_y_dqn.md` §5; `11_glosario.md` §«D» |
| p.38 | `01_aproximacion_funcional_y_dqn.md` §5.1, §6.2; `04_exploracion_memoria_y_estabilidad.md` §6.1; `11_glosario.md` §«D», §«E», §«F», §«I» |
| p.39 | `01_aproximacion_funcional_y_dqn.md` §5.2; `11_glosario.md` §«D», §«J» |
| p.40 | `01_aproximacion_funcional_y_dqn.md` §7; `11_glosario.md` §«R» |
| p.41 | `01_aproximacion_funcional_y_dqn.md` §8.1, §16; `11_glosario.md` §7, §«D», §«P» |
| p.42-43 | `01_aproximacion_funcional_y_dqn.md` §8.2; `11_glosario.md` §«D» |
| p.44 | `01_aproximacion_funcional_y_dqn.md` §9, §16; `11_glosario.md` §5, §«D», §«V» |
| p.45 | `01_aproximacion_funcional_y_dqn.md` §1, §5.2 |
| p.46 | `01_aproximacion_funcional_y_dqn.md` §14 |
| p.47 | `00_repaso_rl.md` §14; `01_aproximacion_funcional_y_dqn.md` §14, §16 |
| p.48 | `01_aproximacion_funcional_y_dqn.md` §14; `02_diseno_de_entornos.md` §1; `04_exploracion_memoria_y_estabilidad.md` §1 |
| p.49 | `02_diseno_de_entornos.md` §5.1; `05_profit_seeking.md` §6.1, §6.4, §10.6, §13 |
| p.50 | `02_diseno_de_entornos.md` §5.2; `05_profit_seeking.md` §6.1, §6.4 |
| p.51 | `00_repaso_rl.md` §4.4; `02_diseno_de_entornos.md` §5.3, §12; `05_profit_seeking.md` §6.4; `09_material_rl_notebooks.md` §5.1, §5.5; `11_glosario.md` §«C» |
| p.52 | `00_repaso_rl.md` §4.4; `02_diseno_de_entornos.md` §12; `05_profit_seeking.md` §6.4; `09_material_rl_notebooks.md` §5.1, §5.5; `11_glosario.md` §«C» |
| p.53 | `02_diseno_de_entornos.md` §5.1, §5.7, §5.9; `05_profit_seeking.md` §6.1, §6.4, §10.2, §10.6; `09_material_rl_notebooks.md` §5.1, §5.5; `11_glosario.md` §«E» |
| p.54 | `02_diseno_de_entornos.md` §4.1; `07_rl_para_ejecucion.md` §6; `11_glosario.md` §«S», §«T» |
| p.55 | `02_diseno_de_entornos.md` §12; `07_rl_para_ejecucion.md` §6 |
| p.56 | `02_diseno_de_entornos.md` §6.1; `07_rl_para_ejecucion.md` §8.1; `11_glosario.md` §«A» |
| p.57 | `02_diseno_de_entornos.md` §6.2, §6.3, §6.6; `04_exploracion_memoria_y_estabilidad.md` §4.5; `07_rl_para_ejecucion.md` §8.1, §8.2, §10.3; `10_hoja_de_ruta_practica.md` §6.5, §11; `11_glosario.md` §«A», §«M» |
| p.58 | `03_recompensas_y_descuento.md` §2; `05_profit_seeking.md` §4.1, §4.5 |
| p.59 | `03_recompensas_y_descuento.md` §3; `05_profit_seeking.md` §7 |
| p.60 | `03_recompensas_y_descuento.md` §5, §13; `05_profit_seeking.md` §7; `11_glosario.md` §«R» |
| p.61 | `03_recompensas_y_descuento.md` §4.4, §6, §8, §9; `04_exploracion_memoria_y_estabilidad.md` §8.2; `05_profit_seeking.md` §7; `07_rl_para_ejecucion.md` §16; `10_hoja_de_ruta_practica.md` §6.3, §8.3; `11_glosario.md` §«C», §«M», §«R», §«S» |
| p.62 | `03_recompensas_y_descuento.md` §9, §13; `04_exploracion_memoria_y_estabilidad.md` §3.6, §10, §13; `05_profit_seeking.md` §7; `07_rl_para_ejecucion.md` §19 |
| p.63 | `04_exploracion_memoria_y_estabilidad.md` §2.1 |
| p.64 | `04_exploracion_memoria_y_estabilidad.md` §2.2, §2.4, §3.1, §3.5; `10_hoja_de_ruta_practica.md` §7.2, §7.3, §11; `11_glosario.md` §«E», §«M»; `notebook_partida.md` §7.2 |
| p.65 | `04_exploracion_memoria_y_estabilidad.md` §3.5 |
| p.66 | `03_recompensas_y_descuento.md` §2; `04_exploracion_memoria_y_estabilidad.md` §3.3; `10_hoja_de_ruta_practica.md` §2, §7.2, §7.3, §11; `11_glosario.md` §«C», §«N»; `notebook_partida.md` §7.2 |
| p.67 | `04_exploracion_memoria_y_estabilidad.md` §3.5 |
| p.68-71 | `04_exploracion_memoria_y_estabilidad.md` §3.5; `11_glosario.md` §«E» |
| p.72 | `03_recompensas_y_descuento.md` §5; `04_exploracion_memoria_y_estabilidad.md` §4.1, §4.2; `09_material_rl_notebooks.md` §9.1, §9.4, §12; `11_glosario.md` §7, §«C», §«D», §«E», … |
| p.73 | `04_exploracion_memoria_y_estabilidad.md` §4.4; `10_hoja_de_ruta_practica.md` §7.3, §8.5; `11_glosario.md` §«E» |
| p.74 | `00_repaso_rl.md` §8, §16; `04_exploracion_memoria_y_estabilidad.md` §5.2, §5.4, §13; `09_material_rl_notebooks.md` §4.2, §4.3; `11_glosario.md` §5, §7, §«A», §«N» |
| p.75 | `00_repaso_rl.md` §6, §15; `03_recompensas_y_descuento.md` §10, §11, §12.1, §12.2; `09_material_rl_notebooks.md` §9.2, §9.4, §12; `10_hoja_de_ruta_practica.md` §6.2, §6.4, §8.5; `11_glosario.md` §«G» |
| p.76 | `00_repaso_rl.md` §6; `03_recompensas_y_descuento.md` §11; `09_material_rl_notebooks.md` §9.2, §9.4, §12; `11_glosario.md` §«G» |
| p.77 | `01_aproximacion_funcional_y_dqn.md` §6.2; `04_exploracion_memoria_y_estabilidad.md` §6.1 |
| p.78 | `01_aproximacion_funcional_y_dqn.md` §6.2, §6.3; `04_exploracion_memoria_y_estabilidad.md` §6.2, §6.3, §6.4, §13; `10_hoja_de_ruta_practica.md` §6.5, §8.5; `11_glosario.md` §«M»; `notebook_partida.md` §8.2 |
| p.79 | `01_aproximacion_funcional_y_dqn.md` §6.2, §6.4; `04_exploracion_memoria_y_estabilidad.md` §6.5, §6.6; `09_material_rl_notebooks.md` §8.1, §8.4, §9.3, §9.4; `10_hoja_de_ruta_practica.md` §6.5 |
| p.80 | `01_aproximacion_funcional_y_dqn.md` §6.6, §10, §16; `03_recompensas_y_descuento.md` §5; `04_exploracion_memoria_y_estabilidad.md` §6.7, §13; `09_material_rl_notebooks.md` §8.4; `10_hoja_de_ruta_practica.md` §8.5; `11_glosario.md` §5, §7, §«H», §«I», … |
| p.81 | `01_aproximacion_funcional_y_dqn.md` §6.7; `04_exploracion_memoria_y_estabilidad.md` §6.8; `08_entorno_best_execution.md` §9.3; `10_hoja_de_ruta_practica.md` §6.5, §8.5; `11_glosario.md` §«C» |
| p.82 | `01_aproximacion_funcional_y_dqn.md` §6.7; `04_exploracion_memoria_y_estabilidad.md` §6.8 |
| p.83 | `02_diseno_de_entornos.md` §1 |
| p.84 | `02_diseno_de_entornos.md` §2, §2.1, §12; `11_glosario.md` §«D», §«G», §«T» |
| p.85 | `02_diseno_de_entornos.md` §3.1; `03_recompensas_y_descuento.md` §7; `10_hoja_de_ruta_practica.md` §6.3; `11_glosario.md` §«E», §«R» |
| p.86 | `02_diseno_de_entornos.md` §7.1, §9.2; `11_glosario.md` §«E», §«R» |
| p.87 | `01_aproximacion_funcional_y_dqn.md` §13; `04_exploracion_memoria_y_estabilidad.md` §10; `11_glosario.md` §«H», §«P» |
| p.88 | `01_aproximacion_funcional_y_dqn.md` §1, §13; `04_exploracion_memoria_y_estabilidad.md` §8.1, §9.3, §9.5; `10_hoja_de_ruta_practica.md` §7.2, §8.4; `11_glosario.md` §«C», §«E», §«L» |
| p.89 | `01_aproximacion_funcional_y_dqn.md` §13; `02_diseno_de_entornos.md` §8.1; `11_glosario.md` §«E» |
| p.90 | `01_aproximacion_funcional_y_dqn.md` §13; `02_diseno_de_entornos.md` §9.1; `04_exploracion_memoria_y_estabilidad.md` §3.2, §9.1, §9.2, §9.4; `10_hoja_de_ruta_practica.md` §2, §7.2, §8.4; `11_glosario.md` §«C», §«E», §«S» |
| p.91 | `01_aproximacion_funcional_y_dqn.md` §11, §16; `02_diseno_de_entornos.md` §5.9; `04_exploracion_memoria_y_estabilidad.md` §8.4; `11_glosario.md` §7, §«R» |
| p.92 | `01_aproximacion_funcional_y_dqn.md` §13; `02_diseno_de_entornos.md` §8.1; `04_exploracion_memoria_y_estabilidad.md` §10; `10_hoja_de_ruta_practica.md` §2; `11_glosario.md` §«H» |
| p.93 | `02_diseno_de_entornos.md` §1; `04_exploracion_memoria_y_estabilidad.md` §11; `11_glosario.md` §«P», §«R» |
| p.94 | `02_diseno_de_entornos.md` §1, §10; `04_exploracion_memoria_y_estabilidad.md` §11 |
| p.95 | `02_diseno_de_entornos.md` §1, §5.6, §11; `04_exploracion_memoria_y_estabilidad.md` §11 |
| p.96 | `02_diseno_de_entornos.md` §11; `03_recompensas_y_descuento.md` §8; `04_exploracion_memoria_y_estabilidad.md` §11; `07_rl_para_ejecucion.md` §16; `11_glosario.md` §«C», §«P», §«R» |
| p.97 | `00_repaso_rl.md` §14; `02_diseno_de_entornos.md` §11; `04_exploracion_memoria_y_estabilidad.md` §11; `11_glosario.md` §3, §«M» |
| p.98 | descartado: portadilla "Appendix"; su contenido son p.99-102 |
| p.99 | `00_repaso_rl.md` §2; `03_recompensas_y_descuento.md` §9, §13; `11_glosario.md` §«A», §«E», §«R» |
| p.100 | `00_repaso_rl.md` §3.1; `03_recompensas_y_descuento.md` §9, §13; `11_glosario.md` §«P», §«V» |
| p.101 | `00_repaso_rl.md` §7.4, §16; `01_aproximacion_funcional_y_dqn.md` §12, §16; `11_glosario.md` §5, §7, §«N» |
| p.102 | `00_repaso_rl.md` §7.4; `01_aproximacion_funcional_y_dqn.md` §12; `11_glosario.md` §«M», §«N», §«T» |
| p.103 | descartado: Diapositiva de cierre/contacto sin contenido |

### 3.2 CASOS

| Página(s) | Destino |
|---|---|
| p.1 | `05_profit_seeking.md` §1; `06_algoritmos_de_ejecucion.md` §15 |
| p.2 | `05_profit_seeking.md` §1 |
| p.3 | `05_profit_seeking.md` §1; `06_algoritmos_de_ejecucion.md` §1; `11_glosario.md` §«A», §«P» |
| p.4 | `05_profit_seeking.md` §1 |
| p.5 | `05_profit_seeking.md` §1, §2; `11_glosario.md` §«C», §«E», §«S» |
| p.6 | `03_recompensas_y_descuento.md` §1; `05_profit_seeking.md` §1, §4.1, §4.5, §13; `11_glosario.md` §2, §«A», §«U» |
| p.7 | `03_recompensas_y_descuento.md` §1; `05_profit_seeking.md` §1, §4.2, §4.5; `11_glosario.md` §7, §«I», §«O», §«S», … |
| p.8-9 | `03_recompensas_y_descuento.md` §1; `05_profit_seeking.md` §1, §4.4, §4.5, §10.6, …; `11_glosario.md` §«O» |
| p.10 | `02_diseno_de_entornos.md` §3.2, §4.3, §5.6, §5.7; `03_recompensas_y_descuento.md` §12.2; `05_profit_seeking.md` §1, §5.1, §5.2, §6.2, …; `11_glosario.md` §«N», §«R» |
| p.11 | `03_recompensas_y_descuento.md` §5; `05_profit_seeking.md` §1, §7, §10.3, §10.6, …; `11_glosario.md` §5, §«A» |
| p.12 | `02_diseno_de_entornos.md` §6.5; `05_profit_seeking.md` §1, §8, §10.6, §13; `11_glosario.md` §«C» |
| p.13 | `02_diseno_de_entornos.md` §4.3, §10; `05_profit_seeking.md` §1, §9.1, §9.2, §9.4, … |
| p.14 | `05_profit_seeking.md` §1, §10.1, §10.8; `09_material_rl_notebooks.md` §10 |
| p.15 | `05_profit_seeking.md` §1; `09_material_rl_notebooks.md` §10 |
| p.16 | `06_algoritmos_de_ejecucion.md` §2.2, §15; `11_glosario.md` §7, §«C», §«O» |
| p.17 | `06_algoritmos_de_ejecucion.md` §2.2, §3, §15; `07_rl_para_ejecucion.md` §2; `11_glosario.md` §«A», §«C», §«O», §«V» |
| p.18 | `06_algoritmos_de_ejecucion.md` §15; `11_glosario.md` §2, §5, §«A», §«C», … |
| p.19 | `06_algoritmos_de_ejecucion.md` §5; `07_rl_para_ejecucion.md` §2; `11_glosario.md` §«A», §«D», §«L», §«O», … |
| p.20 | `06_algoritmos_de_ejecucion.md` §3, §6; `07_rl_para_ejecucion.md` §2; `11_glosario.md` §«A», §«D», §«O», §«V» |
| p.21 | `06_algoritmos_de_ejecucion.md` §7 |
| p.22 | `06_algoritmos_de_ejecucion.md` §8.2 |
| p.23 | `06_algoritmos_de_ejecucion.md` §8.2; `11_glosario.md` §«T» |
| p.24 | `06_algoritmos_de_ejecucion.md` §9.1, §10.2; `11_glosario.md` §«U», §«V» |
| p.25 | `06_algoritmos_de_ejecucion.md` §10.2, §15; `11_glosario.md` §«C» |
| p.26 | `06_algoritmos_de_ejecucion.md` §8.2 (repite el esquema de p.22 y p.24; se dice expresamente) |
| p.27 | `06_algoritmos_de_ejecucion.md` §3, §8.2, §9.1, §10.3, …; `07_rl_para_ejecucion.md` §15; `11_glosario.md` §7, §«P», §«U», §«V» |
| p.28 | `06_algoritmos_de_ejecucion.md` §12.4, §13.1; `07_rl_para_ejecucion.md` §1, §7.4; `11_glosario.md` §«B», §«C», §«F», §«P» |
| p.29 | `02_diseno_de_entornos.md` §1; `06_algoritmos_de_ejecucion.md` §2.2, §3, §6, §7, …; `07_rl_para_ejecucion.md` §2, §15; `11_glosario.md` §«B», §«D», §«T» |
| p.30 | `02_diseno_de_entornos.md` §1, §3.3; `07_rl_para_ejecucion.md` §3; `11_glosario.md` §«E» |
| p.31 | `02_diseno_de_entornos.md` §1, §3.3, §10; `06_algoritmos_de_ejecucion.md` §10.3; `07_rl_para_ejecucion.md` §4.1; `11_glosario.md` §«A», §«G» |
| p.32 | `02_diseno_de_entornos.md` §1, §3.2, §3.3; `07_rl_para_ejecucion.md` §4.3, §21; `11_glosario.md` §5, §«E» |
| p.33 | `02_diseno_de_entornos.md` §1 |
| p.34 | `02_diseno_de_entornos.md` §1, §4.3, §4.4, §12; `07_rl_para_ejecucion.md` §6; `10_hoja_de_ruta_practica.md` §8.5; `11_glosario.md` §«S» |
| p.35 | `02_diseno_de_entornos.md` §1, §5.6, §5.7; `07_rl_para_ejecucion.md` §7.1, §7.2, §21; `11_glosario.md` §3, §7, §«B», §«D», … |
| p.36 | `02_diseno_de_entornos.md` §1, §5.7; `07_rl_para_ejecucion.md` §7.3; `11_glosario.md` §«E», §«S» |
| p.37 | `02_diseno_de_entornos.md` §1, §6.1, §6.3; `06_algoritmos_de_ejecucion.md` §2.3, §12.2; `07_rl_para_ejecucion.md` §8.1; `11_glosario.md` §5, §«A», §«C», §«M», … |
| p.38 | `02_diseno_de_entornos.md` §1, §6.3; `07_rl_para_ejecucion.md` §8.2, §8.3, §21; `11_glosario.md` §«A», §«M» |
| p.39 | `02_diseno_de_entornos.md` §1; `06_algoritmos_de_ejecucion.md` §12.2; `11_glosario.md` §«C», §«O» |
| p.40 | `02_diseno_de_entornos.md` §1, §6.5; `06_algoritmos_de_ejecucion.md` §9.2; `07_rl_para_ejecucion.md` §8.4; `11_glosario.md` §5, §«C», §«L», §«O» |
| p.41 | `02_diseno_de_entornos.md` §1, §12; `07_rl_para_ejecucion.md` §5, §8.4, §21; `11_glosario.md` §«C», §«O» |
| p.42 | `07_rl_para_ejecucion.md` §8.1; `11_glosario.md` §«A» |
| p.43 | `03_recompensas_y_descuento.md` §4.4, §6, §7; `07_rl_para_ejecucion.md` §9.1; `10_hoja_de_ruta_practica.md` §6.3, §8.3; `11_glosario.md` §«D», §«R», §«V» |
| p.44 | `03_recompensas_y_descuento.md` §6; `07_rl_para_ejecucion.md` §9, §9.1; `10_hoja_de_ruta_practica.md` §6.3; `11_glosario.md` §«D», §«R» |
| p.45 | `00_repaso_rl.md` §13; `01_aproximacion_funcional_y_dqn.md` §6.4, §11; `04_exploracion_memoria_y_estabilidad.md` §4.5, §6.6; `07_rl_para_ejecucion.md` §10.3; `10_hoja_de_ruta_practica.md` §6.5, §11; `11_glosario.md` §«A», §«I», §«M» |
| p.46 | `00_repaso_rl.md` §2, §13; `01_aproximacion_funcional_y_dqn.md` §2.3; `04_exploracion_memoria_y_estabilidad.md` §4.3; `07_rl_para_ejecucion.md` §11 |
| p.47-48 | `00_repaso_rl.md` §2, §3.2 |
| p.49 | `00_repaso_rl.md` §2, §3.2; `01_aproximacion_funcional_y_dqn.md` §16 |
| p.50 | `00_repaso_rl.md` §2, §3.2, §16; `01_aproximacion_funcional_y_dqn.md` §16; `07_rl_para_ejecucion.md` §21; `11_glosario.md` §«O» |
| p.51 | `04_exploracion_memoria_y_estabilidad.md` §9.2; `07_rl_para_ejecucion.md` §12; `08_entorno_best_execution.md` §7; `10_hoja_de_ruta_practica.md` §8.2, §8.5; `11_glosario.md` §«P», §«V» |
| p.52 | `07_rl_para_ejecucion.md` §21; `11_glosario.md` §«P» |
| p.53 | `03_recompensas_y_descuento.md` §6, §8; `07_rl_para_ejecucion.md` §13, §16, §21; `10_hoja_de_ruta_practica.md` §8.2; `11_glosario.md` §7, §«C», §«L», §«P» |
| p.54 | `07_rl_para_ejecucion.md` §14, §21; `08_entorno_best_execution.md` §12.4; `11_glosario.md` §«P» |
| p.55-59 | `07_rl_para_ejecucion.md` §21; `11_glosario.md` §«P» |
| p.60 | `06_algoritmos_de_ejecucion.md` §3, §11.3, §13.4, §15; `07_rl_para_ejecucion.md` §2, §21; `10_hoja_de_ruta_practica.md` §8.2; `11_glosario.md` §7, §«V» |
| p.61 | `06_algoritmos_de_ejecucion.md` §13.4; `07_rl_para_ejecucion.md` §17.1, §17.2; `10_hoja_de_ruta_practica.md` §8.2; `11_glosario.md` §3, §«A», §«D», §«J», … |
| p.62-63 | descartado: Diapositivas de cierre y contacto sin contenido |

## 4. Notebook LAB (`execution_lab.ipynb`)

| Celda | Contenido | Destino |
|---|---|---|
| 0 | Título y autor (markdown) | `08_entorno_best_execution.md` §1 |
| 1 | Resumen: orden CARE, hijas, VWAP slippage (markdown) | `08_entorno_best_execution.md` §1, §6.4; `06_algoritmos_de_ejecucion.md` §2.1, §2.2, §3, §6; `notebook_partida.md` §2, §2.1 |
| 2 | Datos: ficheros, días y particiones (markdown) | `08_entorno_best_execution.md` §3.1, §3.2, §3.4; `02_diseno_de_entornos.md` §4.4, §10; `notebook_partida.md` §2.4, §9 |
| 3 | Imports | `08_entorno_best_execution.md` §2.1; `notebook_partida.md` §10.1 |
| 4 | Carga del pickle | `08_entorno_best_execution.md` §2.1, §3.1, §3.2; `02_diseno_de_entornos.md` §7.2 |
| 5 | `data = dict_["train"]` y exploración | `08_entorno_best_execution.md` §3.2; `02_diseno_de_entornos.md` §7.2 |
| 6 | Título "Entorno" (markdown, sin texto) | `08_entorno_best_execution.md` §4 (la sección que abre) |
| 7 | `class BestExecutionEnv` con las 13 funciones y los TODO | `08_entorno_best_execution.md` §4, §5, §6.1-§6.14, §12.2, §15, §16; `02_diseno_de_entornos.md` §2.3, §3.3, §5.5-§5.9; `03_recompensas_y_descuento.md` §4.4, §7; `07_rl_para_ejecucion.md` §4.4, §7.5, §8.5, §9.5; `10_hoja_de_ruta_practica.md` §5-§8 |
| 8 | TWAP (markdown) | `08_entorno_best_execution.md` §7; `06_algoritmos_de_ejecucion.md` §8.1, §12.2 |
| 9 | `class TWAP(DDQNAgent)` a completar | `08_entorno_best_execution.md` §7; `06_algoritmos_de_ejecucion.md` §8.1, §8.4; `notebook_partida.md` §5.3 |
| 10 | Parametrización del problema (markdown) | `08_entorno_best_execution.md` §8 |
| 11 | Explicación de epsilon, gamma, buffer… (markdown) | `08_entorno_best_execution.md` §8; `04_exploracion_memoria_y_estabilidad.md` §2.2, §5.4, §7; `03_recompensas_y_descuento.md` §11 |
| 12 | Hiperparámetros | `08_entorno_best_execution.md` §8; `04_exploracion_memoria_y_estabilidad.md` §2.2, §6.2; `01_aproximacion_funcional_y_dqn.md` §6.2, §7; `10_hoja_de_ruta_practica.md` §6.4, §6.6 |
| 13-14 | Inicialización del agente (`agent = "-----"`) | `08_entorno_best_execution.md` §9.1, §16; `notebook_partida.md` §5.3 |
| 15-16 | Llenado inicial del buffer | `08_entorno_best_execution.md` §10; `04_exploracion_memoria_y_estabilidad.md` §6.6; `10_hoja_de_ruta_practica.md` §6.5 |
| 17-18 | Bucle de entrenamiento | `08_entorno_best_execution.md` §11, §15; `04_exploracion_memoria_y_estabilidad.md` §2.2, §9.4; `03_recompensas_y_descuento.md` §9; `10_hoja_de_ruta_practica.md` §7.2 |
| 19-20 | Evaluación y `plot_results` | `08_entorno_best_execution.md` §12.1, §15; `10_hoja_de_ruta_practica.md` §7.6, §9 |
| 21 | Media móvil de las recompensas | `08_entorno_best_execution.md` §12.1; `03_recompensas_y_descuento.md` §9; `10_hoja_de_ruta_practica.md` §9 |

## 5. Notebook LABSTEP (`execution_lab-step_by_step.ipynb`)

Todas las celdas se describen en conjunto en `08_entorno_best_execution.md` §13, y cada función en su apartado de §6.

| Celdas | Contenido | Destino |
|---|---|---|
| 0 | Título | `08_entorno_best_execution.md` §13 |
| 1-3 | Imports, `rep_data.pkl`, `data = dict_['train']`, `look_back` | `08_entorno_best_execution.md` §2.1, §13 |
| 4 | `__init__` como variables globales | `08_entorno_best_execution.md` §4, §13 |
| 5-6 | (1) `_generate_episode_params` | `08_entorno_best_execution.md` §6.1, §13; `notebook_partida.md` §2.4, §6.3; `10_hoja_de_ruta_practica.md` §6.1, §6.2 |
| 7-8 | (2) `_generate_episode` | `08_entorno_best_execution.md` §6.2, §13; `notebook_partida.md` §6.3, §9 |
| 9-10 | (3) `observation_builder` | `08_entorno_best_execution.md` §6.3, §13 |
| 11-12 | (4) `_compute_episode_market_feat` | `08_entorno_best_execution.md` §6.4, §13 |
| 13-14 | (5) `_compute_market_vwap` | `08_entorno_best_execution.md` §6.5, §13 |
| 15-17 | (6) `_compute_reward` | `08_entorno_best_execution.md` §6.6, §13; `03_recompensas_y_descuento.md` §4.4 |
| 18-20 | (7-8) `_agg_action` y `_compute_algo_vwap` | `08_entorno_best_execution.md` §6.7, §6.8, §13 |
| 21-22 | (9) `_do_nothing` | `08_entorno_best_execution.md` §6.9, §13 |
| 23-24 | (10) `_compute_stop_conditions` | `08_entorno_best_execution.md` §6.10, §13 |
| 25-26 | (11) `_compute_done` | `08_entorno_best_execution.md` §6.11, §13 |
| 27-28 | (12) `_compute_done_reward` | `08_entorno_best_execution.md` §6.12, §13, §15; `03_recompensas_y_descuento.md` §7 |
| 29-30 | (13) `action_sample` | `08_entorno_best_execution.md` §6.13, §13 |
| 31-34 | `step` y su prueba | `08_entorno_best_execution.md` §5, §13, §15 |

## 6. Material RLM (`docs/raw/MIAX14/rl-material/`)

| Notebook o fichero | Destino |
|---|---|
| `TD-q-learning` | `09_material_rl_notebooks.md` §5.1, §5.5, §13; `00_repaso_rl.md` §9.1, §9.2; `02_diseno_de_entornos.md` §2.3, §2.4; `04_exploracion_memoria_y_estabilidad.md` §2.2, §2.4 |
| `TD-sarsa` | `09_material_rl_notebooks.md` §5.2, §5.5 |
| `TD-expected-sarsa` | `09_material_rl_notebooks.md` §5.3, §5.5; `00_repaso_rl.md` §11.2 |
| `TD-double-q-learning` | `09_material_rl_notebooks.md` §5.4, §5.5; `00_repaso_rl.md` §12 |
| `cliff-walking` | `09_material_rl_notebooks.md` §6; `00_repaso_rl.md` §10; `03_recompensas_y_descuento.md` §7; `04_exploracion_memoria_y_estabilidad.md` §2.1-§2.3 |
| `linear-approximation` | `09_material_rl_notebooks.md` §7; `01_aproximacion_funcional_y_dqn.md` §4.1 |
| `linear-sarsa` | `09_material_rl_notebooks.md` §7; `01_aproximacion_funcional_y_dqn.md` §4.1 |
| `linear-expected-sarsa` | `09_material_rl_notebooks.md` §7; `01_aproximacion_funcional_y_dqn.md` §4.1 |
| `deep` | `09_material_rl_notebooks.md` §8.1, §8.4, §13; `01_aproximacion_funcional_y_dqn.md` §3, §6.2; `04_exploracion_memoria_y_estabilidad.md` §2.1, §5.4 |
| `deep-fixed` | `09_material_rl_notebooks.md` §8.2, §8.4; `01_aproximacion_funcional_y_dqn.md` §7 |
| `deep-double` | `09_material_rl_notebooks.md` §8.2, §8.4; `01_aproximacion_funcional_y_dqn.md` §8.1 |
| `deep-dueling` | `09_material_rl_notebooks.md` §8.2, §8.4; `01_aproximacion_funcional_y_dqn.md` §9 |
| `deep-expected-sarsa` | `09_material_rl_notebooks.md` §8.2, §8.4; `01_aproximacion_funcional_y_dqn.md` §10 |
| `deep-rainbow` | `09_material_rl_notebooks.md` §8.1-§8.4; `01_aproximacion_funcional_y_dqn.md` §6.7, §11 |
| `insights` | `09_material_rl_notebooks.md` §9.1-§9.4; `04_exploracion_memoria_y_estabilidad.md` §4.1, §6.5 |
| `profit-seeking` | `09_material_rl_notebooks.md` §10; `05_profit_seeking.md` §10.1-§10.8; `03_recompensas_y_descuento.md` §5 |
| `render-env-devcon` | `09_material_rl_notebooks.md` §11 |
| `README.md`, `setup.py` | `09_material_rl_notebooks.md` §1.1, §2.1-§2.3 |
| `src/utils/plots.py` | `09_material_rl_notebooks.md` §3.1 |
| `src/utils/bmeapi.py` | `09_material_rl_notebooks.md` §3.2, §10; `05_profit_seeking.md` §10.8 |
| `notebooks/static/*` | `09_material_rl_notebooks.md` §12 |

## 7. Repositorio BE (`docs/raw/MIAX14/best-execution/`)

| Fichero | Destino |
|---|---|
| `README.md` | `08_entorno_best_execution.md` §1, §2.1; `06_algoritmos_de_ejecucion.md` §2.1; `notebook_partida.md` §2 |
| `setup.py` | `08_entorno_best_execution.md` §2.1; `10_hoja_de_ruta_practica.md` §3.1; `notebook_partida.md` §10.1 |
| `src/agents/dqn.py` | `08_entorno_best_execution.md` §9.1-§9.6; `01_aproximacion_funcional_y_dqn.md` §6.7, §7, §8.1, §9; `04_exploracion_memoria_y_estabilidad.md` §2.1, §4.5, §5.4 |
| `src/report/report.py` | `08_entorno_best_execution.md` §12.3 |
| `images/twap_model/twap_reproduction_dqn.png` | `08_entorno_best_execution.md` §12.4; `notebook_partida.md` §6.2; `06_algoritmos_de_ejecucion.md` §8.3 |
| `images/twap_model/data_exploration/*.PNG` | `08_entorno_best_execution.md` §1, §3.2, §3.3 |
| `.devcontainer/devcontainer.json`, `dockerfiles/**` (con `requirements.txt`) | `08_entorno_best_execution.md` §2.2; `notebook_partida.md` §10.1; `10_hoja_de_ruta_practica.md` §3.1 |
| `LICENSE` | `10_hoja_de_ruta_practica.md` §4.2, §10 |
| `.gitignore` | `08_entorno_best_execution.md` §15 |
| `data/*.pkl`, `data/new_version/*.pkl` (DATOS) | `08_entorno_best_execution.md` §3.1-§3.4; `notebook_partida.md` §9, §12 |
| `workshop/*.ipynb` | §4 y §5 de este documento |
| `src/__init__.py`, `src/agents/__init__.py`, `src/report/__init__.py` | descartados: vacíos (0 bytes) |

## 8. ENUNCIADO, párrafo a párrafo

Párrafos numerados por orden en el `.docx`.

| Párr. | Contenido | Destino |
|---|---|---|
| 1-2 | Título; `execution_lab.ipynb` como referencia | `notebook_partida.md` §1, §3 (punto 1) |
| 3-5 | Una copia por ejercicio o todo junto; marcar los bloques modificados y explicar la lógica | `notebook_partida.md` §3 (punto 2); `10_hoja_de_ruta_practica.md` §4.2, §12 |
| 6 | No hay solución única; la motivación demuestra el conocimiento | `notebook_partida.md` §3 (punto 3), §4.2; `10_hoja_de_ruta_practica.md` §11 |
| 7 | Ajustar hiperparámetros (γ, buffer, episodios) | `notebook_partida.md` §3 (punto 4) |
| 8 | Cualquiera de los 3 ficheros de datos | `notebook_partida.md` §3 (punto 5), §9, §12; `08_entorno_best_execution.md` §3.1 |
| 9-10 | Entrega: notebook ejecutado; motivaciones en el notebook o en un documento aparte | `notebook_partida.md` §3 (puntos 6-7), §15; `10_hoja_de_ruta_practica.md` §12, §14 |
| 11-13 | Forma de evaluación (exposición y motivación; el código la respalda) | `notebook_partida.md` §4.1; `10_hoja_de_ruta_practica.md` §2 |
| 14-26 | Ejercicio 1 (3,5 p.): necesario y no necesario | `notebook_partida.md` §5.1; `10_hoja_de_ruta_practica.md` §5.1; `08_entorno_best_execution.md` §16 |
| 27-33 | Ejercicio 2 (3,5 p.): primera y segunda parte | `notebook_partida.md` §6.1; `10_hoja_de_ruta_practica.md` §6.1 |
| 34-38 | Ejercicio 2: requisitos (hijas de 1, CARE alineada, sin impacto, figura de referencia, robustez) | `notebook_partida.md` §6.1, §6.2; `08_entorno_best_execution.md` §3.5, §12.4; `06_algoritmos_de_ejecucion.md` §8.3, §8.4; `10_hoja_de_ruta_practica.md` §6.1 |
| 39-42 | Ejercicio 3 (1,5 p.): ε que no dependa del número de episodios | `notebook_partida.md` §7.1; `10_hoja_de_ruta_practica.md` §7.1 |
| 43-49 | Ejercicio 4 (1,5 p.): ideas A y B | `notebook_partida.md` §8.1; `10_hoja_de_ruta_practica.md` §8.1-§8.3 |
| 50-51 | Ejercicio 4: idea C | `notebook_partida.md` §8.1; `10_hoja_de_ruta_practica.md` §8.4 |
| 52 | Libertad para incorporar otros puntos de la clase | `notebook_partida.md` §8.1; `10_hoja_de_ruta_practica.md` §8.5 |
| 53 | Contacto para dudas | `notebook_partida.md` §1, §11 (solo el correo del trabajo; el personal se omite a propósito) |

## 9. Tramos y páginas descartados

| Fuente | Tramo | Tema | Motivo |
|---|---|---|---|
| T06 | ~00:00:02-00:02:00 | Arranque de la sesión, prueba de micrófono | Ruido de inicio de clase y charla sin contenido |
| T06 | ~01:41:07-02:06:44 | Descanso | Descanso: ruido y conversación irrelevante |
| T06 | ~04:19:05-04:20:07 | Charla personal | Sin contenido técnico; solo el aviso de ver el vídeo (~04:19:57) pasa a `notebook_partida.md` §11 |
| T06 | ~04:31:02-04:31:50 | Gestión del tiempo de clase | Logística; solo «se graba» (~04:31:23) pasa a `notebook_partida.md` §11 |
| T07 | ~00:00:00-00:03:51 | Charla previa al inicio de clase | Ruido y charla irrelevante antes de empezar la clase |
| T07 | ~02:02:24-02:30:00 | Descanso | Descanso: ruido, fragmentos ininteligibles y silencio |
| T07 | ~02:30:00-02:33:00 | Pausa / ruido de transcripción | pausa y ruido sin contenido |
| T07 | ~02:33:00-02:36:45 | Charla sobre consultoría y carrera profesional | charla personal/laboral irrelevante para el temario |
| APP | p.4 | Portadilla "Primera Parte" | Sin contenido |
| APP | p.6 | Separador "¿Alguna duda?" | Sin contenido |
| APP | p.98 | Portadilla "Appendix" | Sin contenido; el apéndice son p.99-102 |
| APP | p.103 | Cierre | Diapositiva de cierre/contacto sin contenido |
| CASOS | p.1-2 | Portada y aviso del bloque Casos Prácticos | Portada, aviso genérico y datos de contacto; sin contenido técnico ni de la práctica |
| CASOS | p.62-63 | Cierre y contacto | Diapositivas de cierre y contacto sin contenido |

## 10. PENDIENTES

Ninguno. Todos los tramos de T06 y T07 con contenido, todas las páginas de APP y CASOS no descartadas, las 22 celdas de LAB, las 35 de LABSTEP, los 18 notebooks de RLM, los ficheros de BE y los párrafos del ENUNCIADO tienen al menos una sección que los cita.

**Nota (límites de la comprobación):** (1) es automática por citas: confirma que la sección cita el tramo, no que recoja todo lo que se dijo en él. (2) Los tramos con dos destinos previstos (`03`/`04` y `06`/`07`) a menudo solo se citan en uno: es el reparto buscado, no una omisión. (3) APP p.4, 6 y 98 no se citan en ningún documento, pero son portadillas sin contenido (§9). (4) CASOS p.26 no se cita con su número, pero [06_algoritmos_de_ejecucion.md](06_algoritmos_de_ejecucion.md) §8.2 dice que repite el esquema de p.22 y p.24.
