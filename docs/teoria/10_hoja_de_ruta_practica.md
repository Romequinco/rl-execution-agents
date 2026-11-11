# Hoja de ruta para la práctica

> **Fuentes:**
> - **ENUNCIADO** (`docs/raw/MIAX14/ejercicios_taller_rl.docx`): completo. **Es la fuente que manda.**
> - **Documentos de esta carpeta**, leídos antes de escribir este y citados con su sección: [notebook_partida.md](notebook_partida.md) (requisitos, logística, ambigüedades), [00](00_repaso_rl.md), [01](01_aproximacion_funcional_y_dqn.md), [02](02_diseno_de_entornos.md), [03](03_recompensas_y_descuento.md), [04](04_exploracion_memoria_y_estabilidad.md), [05](05_profit_seeking.md), [06](06_algoritmos_de_ejecucion.md), [07](07_rl_para_ejecucion.md), [08](08_entorno_best_execution.md) y [09](09_material_rl_notebooks.md). Casi toda la evidencia de este documento está desarrollada allí; aquí se resume en 1-3 líneas y se enlaza.
> - **T06** (`docs/raw/26 11 06.pdf`) y **T07** (`docs/raw/26 11 07.pdf`), transcripciones automáticas de las clases de Jesús Sanz. Tramos leídos para este documento: T06 ~00:02-00:06, ~00:40-00:41, ~00:59-01:07, ~02:57-03:00, ~03:04-03:05, ~03:17-03:20, ~03:49, ~04:09-04:11, ~04:20-04:21, ~04:29-04:30, ~04:52-05:00; T07 ~00:03-00:26, ~00:41-00:44, ~01:10, ~01:16-01:19, ~01:43-01:50, ~01:58-02:02, ~02:57-03:07, ~03:31-03:38 y ~04:27-05:10. Las demás horas que se citan se han tomado de los documentos 02-08, que las transcriben y contrastan.
> - **APP** (`docs/raw/MIAX14/applying_RL_MIAX.pdf`): p.1-3 (aviso, agenda); p.57, 61, 64, 66, 73, 75, 78-81, 85, 88, 90 y 92, a través de [03](03_recompensas_y_descuento.md) y [04](04_exploracion_memoria_y_estabilidad.md) (p.66 comprobada en el texto extraído). **CASOS** (`docs/raw/MIAX14/casos_practicos.pdf`): p.34, 43-45, 51, 53, 60 y 61, a través de [07](07_rl_para_ejecucion.md) (p.60 comprobada en el texto extraído).
> - **LAB** (`docs/raw/MIAX14/best-execution/workshop/execution_lab.ipynb`), **LABSTEP** (`.../execution_lab-step_by_step.ipynb`) y **BE** (`docs/raw/MIAX14/best-execution/`: `src/agents/dqn.py`, `src/report/report.py`, `setup.py`, `README.md`, `LICENSE`), a través de [08](08_entorno_best_execution.md) y comprobados en los puntos que se citan.
> - **T1** (`docs/raw/referencias/rl-trading-agents/`): [`README.md`](../raw/referencias/rl-trading-agents/README.md), [`docs/DECISIONES.md`](../raw/referencias/rl-trading-agents/docs/DECISIONES.md), [`docs/teoria/notebook_partida.md`](../raw/referencias/rl-trading-agents/docs/teoria/notebook_partida.md). **T2** (`docs/raw/referencias/deep-rl-trading-agents/`): [`README.md`](../raw/referencias/deep-rl-trading-agents/README.md), [`docs/DECISIONES.md`](../raw/referencias/deep-rl-trading-agents/docs/DECISIONES.md), [`docs/teoria/notebook_partida.md`](../raw/referencias/deep-rl-trading-agents/docs/teoria/notebook_partida.md), `requirements.txt`, `notebooks/common/dqn.py`.
> - Estado de arranque de este repositorio: `README.md`, `.gitattributes`, `.gitignore` y el primer commit.
>
> **Convenciones:** `(T07 ~04:52)` = hora aproximada en la transcripción, siempre paráfrasis; *(dudoso)* = lectura insegura. `(LAB celda 7)` = celda contando desde 0. `T1 D-0.1` / `T2 D-0.2` = decisión del registro del Taller 1 / Taller 2 (numeraciones distintas entre sí). **Nota:** y **Propuesta:** = contenido propio (cálculos, simulaciones sin datos, recomendaciones); casi todo este documento lo es, y por eso cada propuesta enlaza la evidencia de clase o del código en que se apoya. **No se cierran decisiones**: para cada bloque se dan opciones, pros y contras y una recomendación motivada, que el grupo debe cerrar en `docs/DECISIONES.md` (§4.4). Notación: $N$ = `episode_bins`, $n$ = `vol_care`, $t$ = `state_pos`, $e_t$ = `exec_vol`, $T_t=1-t/N$ (tiempo restante), $V_t=1-e_t/n$ (volumen restante), $d_t=V_t-T_t$ (positivo = retrasado respecto al plan TWAP), $\rho=n/N$ (fracción de *steps* en que hay que comprar, entre 0,075 y 0,125).
>
> Escrito el 10/11/2026.

---

## 1. La propuesta en una página

**Idea.** El enunciado puntúa sobre todo **la exposición y la motivación** (ENUNCIADO, "Forma de evaluación"), y el profesor repitió que valora el planteamiento por encima del resultado (T07 ~01:18). La hoja de ruta, por tanto, no persigue el mejor agente posible: persigue **un entorno correcto, una réplica del TWAP conseguida por recompensas y defendible con cálculos, y un ε guiado por una métrica de mejora explícita**, con cada decisión registrada y respaldada por un gráfico o una cifra.

| Ej. (puntos) | Lo imprescindible (ENUNCIADO) | Propuesta recomendada | Evidencia que hay que enseñar | Detalle |
|---|---|---|---|---|
| 1 (3,5) | Entrenamiento y ejecución completos, elementos de un entorno RL, orden correcto, documentar | Completar las 13 funciones vía LABSTEP, sin cambiar la estructura de `BestExecutionEnv`; TWAP determinista de la celda 9 como *benchmark*; corregir las erratas que rompen el bucle (E4, E5); agente Double DQN, en TF tal cual o portado a PyTorch con la misma interfaz | Orden de llamadas `reset`→`act`→`step`; pruebas de invariantes; `plot_results` del TWAP determinista y de un episodio del agente; curva de entrenamiento | §5 |
| 2 (3,5) | Recompensa que **replique un TWAP**; estado y recompensa explicados; gráficos; **parametría motivada**; funciona con distintos días, horas, duraciones y tamaños | Estado $[T,V]$ (opcional $d$). Recompensa **local de seguimiento del plan**, $-\lvert d\rvert$ por *step*, con la regla del `if` como control; γ y *buffer* calculados a partir de la duración del episodio y de la separación entre hijas; `action_sample` con $p\approx\rho$ | Varios `plot_results` en *train* con $N$ y $n$ distintos; trayectorias $V$ frente a $T$ de un conjunto fijo de episodios; mapa de la política *greedy* sobre $(T,V)$; tabla de γ y *buffer* probados | §6 |
| 3 (1,5) | ε que no dependa del número de episodios; responder **cómo se mide la mejora** y **cómo se traslada a ε** | Evaluación *greedy* periódica en un conjunto fijo de episodios; mejora = fracción recorrida hacia el nivel del TWAP determinista; ε por escalones o función continua de esa fracción, con suelo, margen de ruido y posibilidad de subir | Gráfico de ε y de la métrica frente a episodios; comparación con el decaimiento lineal con el mismo presupuesto; mapa de calor de estados visitados | §7 |
| 4 (1,5) | Mejora libre, explicando qué se recogió de clase, el razonamiento y lo implementado | **B** (sin volumen pendiente y ≈90 % del tiempo) y **C** (parada por aprendizaje, *checkpoints* y recuperación), baratas y medibles; **A** (estado con libro y tendencia, recompensa local de precio) si queda tiempo | Distribución del instante de finalización y del volumen pendiente; curva con *checkpoints* y punto de parada; reanudación demostrada | §8 |

**Nota:** el orden de trabajo importa más que en los talleres anteriores, porque todo depende del ej. 1: el profesor pidió señalar "el punto 1, que es de lo que depende todo" *(transcripción ruidosa)* (T07 ~04:31). Calendario orientativo en §13.

---

## 2. Principios de clase que ordenan la propuesta

| Principio | Fuente | Qué implica aquí |
|---|---|---|
| Decisiones arbitrarias, "fatal, siempre para todo"; preguntará el porqué de cada elección | T06 ~04:10 | Cada parámetro con una línea de motivación y una evidencia (§11) |
| Sobresaliente posible sin resultados si el planteamiento es bueno | T07 ~01:18 | En el ej. 4 basta un razonamiento correcto (lo dice también el ENUNCIADO para la idea A) |
| Empezar por el triciclo y sobreajustar a propósito como test | T07 ~03:00-03:02; T06 ~04:45-04:47; [02 §9](02_diseno_de_entornos.md) | Depurar con un día y una hora fijos antes de aleatorizar |
| Aislar el problema; el agente decide una sola dimensión | T07 ~03:02-03:07; [07 §5](07_rl_para_ejecucion.md) | No ampliar acciones en los ej. 1-3; solo el momento (la ampliación queda como opción del ej. 4, §8.5) |
| La recompensa debe estar alineada **y** ser factible desde el estado | T07 ~03:37-03:44; [03 §4](03_recompensas_y_descuento.md) | Con $[T,V]$, recompensas de seguimiento del plan, no de precio |
| Para el TWAP bastan tiempo y volumen restantes, "de la mano" | T07 ~04:52 | Estado mínimo en el ej. 2 |
| "Quiero que lo hagáis también por recompensas" | T07 ~04:54 | La regla del `if` vale como *benchmark*, no como solución del ej. 2 |
| Benchmarks clásicos para medir la distancia y decidir cuándo bajar la exploración | T07 ~01:47; APP p.66 | El TWAP determinista es el nivel de aspiración del ej. 3 |
| Evaluar sin exploración, en varios episodios | T06 ~03:16-03:17; T07 ~00:20-00:22; APP p.90 | Conjunto fijo de episodios de evaluación *greedy* (§4.3) |
| Mirar comportamientos, no solo recompensas | T07 ~00:31-00:32, ~03:52 | Gráficos de ejecución y trayectorias, no solo la curva de recompensa |
| Inestabilidad: ~3 de cada 10 entrenamientos bien configurados fallan | T06 ~01:13; T07 ~00:27-00:28 | Varias semillas por configuración |
| Ajustar "a manivela", no con *grid search* masivo | T07 ~00:28-00:29; APP p.92 | Pocas configuraciones razonadas, un factor cada vez |
| No todo es RL: reglas y supervisado donde convenga | T07 ~00:23-00:26; T06 ~02:55-02:56 | Legítimo forzar acciones con reglas (p. ej. el perfil conservador de la idea A) si se dice |

---

## 3. Entorno de software: framework y versiones

### 3.1 Las restricciones

- **El código del taller** exige Python `>=3.9.2,<3.12`, TensorFlow 2.15.1 (Keras 2, la API de `dqn.py`), numpy 1.26.4 y **pandas 2.0.3**, porque los pickles de `data/` se guardaron con pandas 1.x y no se leen desde pandas 2.1 (BE setup.py; [08 §2.1](08_entorno_best_execution.md)). La receta de respaldo del Dev Container (`requirements.txt`) y el *kernel* registrado en los notebooks son de la pila antigua (Python 3.8, TF 2.5), y el contenido de la imagen publicada `jesussanz/best_execution:v1` no se conoce ([notebook_partida §10](notebook_partida.md)): lo primero es incoherente y lo segundo está sin verificar frente a `setup.py` (E19 en [08 §15](08_entorno_best_execution.md)).
- **Los pickles nuevos** (`data/new_version/`) son los mismos datos serializados con pandas 2.x, y "deberían funcionar para todos" (T07 ~04:38-04:39; [08 §3.1](08_entorno_best_execution.md)). No se ha comprobado cargándolos.
- **El ENUNCIADO libera el *stack*:** no es necesario reutilizar el algoritmo de Deep RL ni "los paquetes y versionados del ejemplo. Esto incluye el uso de PyTorch en vez de TensorFlow" (ej. 1, "No es necesario"). En clase: no hace falta el contenedor, otros alumnos lo hicieron sin él (T07 ~04:31-04:32).
- **El estado de partida del grupo.** Los Talleres 1 y 2 se hicieron con Python 3.13 (T1 y T2 `README.md`, `py -3.13`), y el 2 con PyTorch en CPU (T2 D-0.2; T2 `requirements.txt`). El `README.md` de arranque de este repositorio ya anuncia Python 3.11 y PyTorch. **Nota:** al escribir este documento, el equipo de trabajo solo tiene instalado Python 3.13 (`py -0`), en el que TF 2.15 no se instala.

### 3.2 Opciones

| Opción | Qué es | A favor | En contra |
|---|---|---|---|
| **A. LAB tal cual** | `py -3.11` + `pip install -e .` de BE: TF 2.15.1, pandas 2.0.3, pickles antiguos (LAB celda 4) | Cero traducción; lo que el profesor espera ver; los TODO se rellenan sin tocar el agente | Hay que instalar Python 3.11; Keras es lento en el bucle paso a paso (abajo); el código no se parece al de los Talleres 1 y 2 |
| **B. Python 3.11 + PyTorch** | Mismo entorno que A sin TF (pandas 2.0.3, pickles antiguos o nuevos) y un `DDQNAgent` reescrito en PyTorch con **la misma interfaz** (`act`, `experience`, `sample` con CER, `learn`, `update_target`, `set_trainable`, compra forzada en el paso 0) | Las celdas 13-21 del LAB siguen funcionando; velocidad de PyTorch; reutiliza el código del Taller 2; coherente con el README del repo | Hay que traducir y probar el agente (unas 100 líneas, [08 §9](08_entorno_best_execution.md)); declarar el cambio de *framework* (ENUNCIADO ej. 1) |
| **C. Python 3.13 + PyTorch** | El entorno del Taller 2; pandas ≥ 2.2 obligatoriamente y pickles de `new_version/` | No hay que instalar otro Python; mismo `requirements` que el Taller 2 | Depende de que los pickles nuevos carguen con pandas reciente (no verificado); imposible ejecutar el agente original de TF para comparar |

**Coste esperable (Nota, orden de magnitud).** En el Taller 2 se midió, con una red de 64 neuronas en CPU, que evaluar un estado cuesta 0,3 ms en PyTorch, 10,8 ms con `model(x)` en Keras y 276 ms con `predict` (T2 D-0.2). El `DDQNAgent` llama a `model(s).numpy()` en cada `act` (BE dqn.py, getQs). Con episodios de ~500 *steps*: ≈5,4 s por episodio solo en `act` con Keras frente a ≈0,15 s con PyTorch; 1.000 episodios serían ≈1,5 h frente a ≈2,5 min, sin contar `learn`, el entorno en pandas ni las evaluaciones *greedy* del ej. 3, que multiplican los *steps* (§7.5). Son medidas de otro equipo: hay que repetirlas con `%timeit` antes de decidir ([02 §8.4](02_diseno_de_entornos.md)).

**Propuesta (entorno):** con el agente en PyTorch, el cuello de botella puede pasar a ser el entorno en pandas, sobre todo con las evaluaciones *greedy* del ej. 3, que casi doblan los *steps* (§7.5). En `reset`, convertir una sola vez las columnas del episodio (`ask1`, `bid1`, *mid*, volumen por *bin*) a arrays de NumPy y acceder en `step` por índice entero (`state_pos`), no con `.iloc`/`.loc`; llevar el VWAP del algoritmo y el de mercado como sumas acumuladas de $p\,v$ y de $v$. En la demo de clase con `%timeit`, el acceso por diccionario (~45 ns) o por array era órdenes de magnitud más rápido que por DataFrame, y reconstruir el libro de un día de Santander pasó de 20-30 min con pandas a 2-3 s con diccionarios (T07 ~00:12-00:17; [02 §8.2-8.4](02_diseno_de_entornos.md); RLM `insights` celdas 24-38, [09 §9.4](09_material_rl_notebooks.md)). Si la medición lo pide, cambiar también el `deque` que `sample` indexa al azar por un buffer circular sobre arrays ([02 §8.4](02_diseno_de_entornos.md)). Si se aplica, es un bloque modificado más que hay que declarar, y conviene comprobar que la versión con arrays reproduce exactamente los históricos de la versión con pandas en unos cuantos episodios.

**Propuesta:** **B**. Primero se valida el entorno completo con el agente TF original en una sesión corta (opción A, si se consigue el 3.11 con TF) o directamente con el agente PyTorch; después se entrena todo con PyTorch. Motivos: (1) velocidad, que en el ej. 3 se vuelve crítica; (2) el ENUNCIADO lo permite expresamente; (3) mantener la interfaz del `DDQNAgent` permite decir "solo cambia el agente" y señalar los bloques modificados con precisión, como pide el ENUNCIADO. Si los pickles nuevos cargan con pandas reciente, **C** es equivalente y evita el segundo Python. Sea cual sea, va a `DECISIONES.md` con la medición propia.

**Nota (seguridad de los datos):** cargar un pickle ejecuta código. Los de `data/` vienen del repositorio del taller, así que el riesgo es aceptable, pero conviene cargarlos una vez, comprobar columnas y tipos (`astype(float)`: SAN puede traer una columna entera, [08 §3.1](08_entorno_best_execution.md)) y, si se quiere, guardarlos en un formato sin código (`parquet`, `npz`) para el resto del trabajo.

---

## 4. Estructura del repositorio, protocolo y registro de decisiones

### 4.1 Qué se reutiliza de los Talleres 1 y 2

| Pieza | Origen | Adaptación al Taller 3 |
|---|---|---|
| Código común como única fuente de verdad; un notebook por línea de trabajo; resultados en CSV con prefijo; un notebook final que solo lee `results/` | T1 D-0.2; T2 D-0.4, D-4.1; T2 `README.md` | Aquí el entregable **es un notebook ejecutado** (ENUNCIADO), no solo un PDF: los notebooks deben leerse solos, con la motivación dentro, y señalar los bloques modificados |
| Registro de decisiones `D-X.n` con fecha, contexto, alternativas, evidencia y estado; `merge=union` | T1 y T2 `DECISIONES.md` | `.gitattributes` de este repo ya trae `docs/DECISIONES.md merge=union` y `*.ipynb -merge` |
| Protocolo: semillas fijas, validación separada, test usado una vez, 5 semillas para el resultado final | T1 D-0.1, D-0.4; T2 D-0.1, D-2.4 | El ej. 2 no exige validación ni test (basta *train*, ENUNCIADO); se conserva la idea de un **conjunto fijo de episodios de evaluación** (§4.3) |
| Ajuste por etapas y regla de parsimonia o margen de ruido | T1 D-2.3; T2 D-2.3 | Para elegir γ, *buffer* y el criterio de ε sin elegir por azar |
| Dos curvas: con exploración y *greedy* periódica; la convergencia se mide en la *greedy* | T2 D-0.3 | Es exactamente lo que necesita el ej. 3 (§7) |
| Normalizar observación y recompensa (recompensas de ~1e-3 por paso) | T2 D-1.1 | Si se usa una recompensa de precio (idea A), pasarla a pb o dividir por su σ de *train* ([03 §9](03_recompensas_y_descuento.md)) |
| Fin de episodio como terminal ($y=r$) | T1 D-2.1; T2 D-1.2 | Aquí es correcto sin matices: el tiempo restante está en el estado ([08 §5](08_entorno_best_execution.md)) |
| PyTorch en CPU, con mediciones | T2 D-0.2 | §3 |
| Extras con un factor cada vez y sin reajustar después de mirar | T1 D-1.4; T2 D-3.1, D-3.2 | Para el ej. 4 y para los barridos de γ y *buffer* |
| Trampas conocidas: RNG compartido, volver a ejecutar la celda que crea el modelo, quedarse sin tiempo, funciones en celdas con multiproceso en Windows | T2 `notebook_partida.md` §9 (trampas 5, 14, 15, 16) | §10 |
| Informe en Markdown → PDF (`build_informe.py`) | T1 D-3.1; T2 D-0.5, D-4.2 | Aquí no hay límite de 2 páginas; la memoria es opcional pero "en ocasiones recomendable" (ENUNCIADO); §12 |
| Presentación HTML → PDF (una diapositiva HTML por fichero en `results/presentacion/slides/`, `build_presentacion.py`), 6 diapositivas + anexos A-F | T2 `README.md` (`results/presentacion`) | §12 |
| Trabajo en paralelo: un editor por notebook a la vez, notebooks subidos ejecutados, `notebooks/scratch_*.ipynb` sin versionar; `run_grid` con caché que depende de la configuración y de una `version` (reabrir no reentrena; `RECALCULAR = True` para forzar) | T2 `README.md` ("Cómo trabajar en paralelo y contrato común"); T2 D-0.4; T1 `README.md` | Imprescindible para los barridos de γ y *buffer* y para las semillas; el reparto entre miembros está en §13 |

**Lo que no se reutiliza:** la rúbrica porcentual y el PDF de 2 páginas (este taller no los tiene, [notebook_partida §4](notebook_partida.md)); los mercados sintéticos y sus métricas (mediana del retorno, *drawdown*). Aquí las métricas son de ejecución (§4.3).

### 4.2 Estructura propuesta

**Propuesta** (sobre la estructura del Taller 2, T2 `README.md`):

```
notebooks/
  ej1_execution_lab.ipynb        # copia del LAB completada; cabecera con bloques modificados
  ej2_twap_recompensas.ipynb     # estado, recompensa, barridos de γ y buffer, gráficos
  ej3_exploracion.ipynb          # criterio de mejora y ε; comparación con el lineal
  ej4_extras.ipynb               # ideas B y C (y A si da tiempo)
  common/
    env.py          # BestExecutionEnv con las 13 funciones (una sola versión, con opciones)
    agents.py       # DDQNAgent (PyTorch, interfaz de BE) y TWAP determinista
    rewards.py      # candidatas de recompensa, seleccionables por configuración
    protocol.py     # semillas, conjunto fijo de evaluación, métricas, guardado de resultados
    plots.py        # plot_results corregido (figura nueva por llamada), trayectorias, mapas
    test_env.py     # invariantes del entorno (§5.2, paso 4)
docs/DECISIONES.md
results/            # CSV por notebook, figures/, models/ (checkpoints)
memoria.md          # opcional: memoria breve de decisiones (§12)
```

- **Un notebook por ejercicio** es lo más claro para "especificar los bloques que han sido modificados" (ENUNCIADO). Cada uno empieza con una celda **"Qué cambia respecto al ejercicio anterior"** con la lista de funciones o celdas tocadas y el motivo.
- **Alternativa:** un único `execution_lab.ipynb` con secciones por ejercicio, como permite el ENUNCIADO. Más fiel al original, pero los cambios de recompensa y de ε conviven y cuesta mostrar cada ejercicio por separado.
- **Si el entorno se mueve a `.py`**, el ENUNCIADO lo permite ("sin perjuicio de que alumno quiera estructurar mejor su código haciendo uso de otros ficheros .py"), pero el notebook del ej. 1 debe enseñar las funciones o importarlas de forma visible y explicadas. **No** se recomienda rehacer el entorno desde cero: el ENUNCIADO exigiría entonces "mayor documentación sobre la estructura y los pasos seguidos" (ej. 1).

**Nota (licencias):** el código del taller (`best-execution`) se distribuye bajo GNU LGPL 2.1 (BE LICENSE), y el código propio de este repositorio, bajo MIT (`README.md` de este repo, sección "Licencia"). Los ficheros derivados de BE (`env.py` con la estructura y los *docstrings* de `BestExecutionEnv`, el puerto de `DDQNAgent`, `plot_results`) no son "código propio" en ese sentido: deben conservar el aviso de *copyright* y la referencia a la LGPL 2.1, e ir marcados como derivados. Se recomienda una cabecera en cada uno de esos ficheros y una línea en el `README.md` que diga que la licencia MIT no los cubre. No es asesoramiento legal: conviene revisarlo antes de publicar (§10, fila 22).

### 4.3 Protocolo de evaluación

**Propuesta:**

1. **Conjunto fijo de episodios de evaluación** sobre *train* (lo que pide el ej. 2): una rejilla que cubra los extremos del rango, por ejemplo $N\in\{400, 500, 600\}$ × $\rho\in\{0{,}075;\ 0{,}10;\ 0{,}125\}$ × 2-3 días y horas al azar con semilla fija, es decir, 18-27 episodios. Así se **demuestra** el requisito de "distintos días, momentos temporales, duración del episodio y tamaño de la orden" (ENUNCIADO ej. 2) en lugar de afirmarlo. Exige un `reset` con parámetros opcionales forzados (un bloque modificado más, que hay que declarar).
2. **Evaluación *greedy***, sin aprender, con `agent.env = env` reasignado (errata E5, [08 §12.1](08_entorno_best_execution.md)) y con generadores aleatorios propios para entorno y agente, para que la exploración no consuma la secuencia del entorno (lección de T2 `notebook_partida.md` §9, trampa 5).
3. **Métricas por episodio** (Nota):

| Métrica | Definición | Para qué |
|---|---|---|
| Completado | $e_{\text{fin}}=n$ (sí/no) y volumen pendiente $n-e_{\text{fin}}$ | Objetivo del caso: la orden completa (T07 ~01:45) |
| Instante de finalización | $t_{\text{fin}}/N$ | Idea B (≈90 %) |
| Error de seguimiento | $\max_t \lvert e_t - t\,n/N\rvert$ y su media, **en títulos** | "Comportamiento similar" al TWAP; el TWAP determinista da ≤ 1 título |
| Número de compras y separación media | $n$ compras; media y desviación de los intervalos | Barras "equiespaciadas" de la figura de referencia ([notebook_partida §6.2](notebook_partida.md)) |
| VWAP *slippage* frente al mercado | $(\text{VWAP}_{mkt}-\text{VWAP}_{algo})/\text{VWAP}_{mkt}\cdot10^4$, con signo y precio de mercado declarados | Métrica del caso (LAB celda 1; [07 §2](07_rl_para_ejecucion.md)) |
| *Slippage* frente al TWAP determinista | Diferencia en pb en el mismo episodio | Benchmark de la idea A ([06 §14](06_algoritmos_de_ejecucion.md)) |

4. **Dispersión además de media** en todas las tablas (T07 ~01:46; [06 §14](06_algoritmos_de_ejecucion.md)).
5. **Semillas:** 3 para elegir y 5 para el resultado que se expone, como en los Talleres 1 y 2 (T1 D-2.3; T2 D-0.1), si el presupuesto de cómputo lo permite. Si no, al menos 3 y decirlo.
6. **Una sola definición de precio de mercado** en recompensa, gráfico y tablas: el entorno usa el *mid* y `stats_df` el `ask1` (errata E3, [08 §12.2](08_entorno_best_execution.md)).

### 4.4 Decisiones que hay que abrir en `DECISIONES.md`

**Propuesta** (mismo formato que T1 y T2; los números son orientativos):

| ID | Decisión | Opciones principales | Dónde se razona |
|---|---|---|---|
| D-0.1 | Entorno de software y *framework* | A, B, C | §3 |
| D-0.2 | Estructura y notebooks por ejercicio | Uno por ejercicio / uno único | §4.2 |
| D-0.3 | Protocolo de evaluación y métricas | Rejilla fija en *train*; métricas de §4.3 | §4.3 |
| D-1.1 | Activo y *split* | REP o SAN; *train* o *splits* juntos (T07 ~04:39) | §5.3 |
| D-1.2 | Precio de ejecución y de mercado | `ask1` al inicio del *bin* (sesgo declarado) / *mid* medio entre fotos / latencia / **calcular dos precios** (el del LAB y uno realista) y dar el *slippage* con ambos | §6.8, §8.2 |
| D-1.3 | Estado | $[T,V]$ / $[T,V,d]$ / otros | §6.2 |
| D-1.4 | `action_sample` | $p=0{,}5$ / $p\approx\rho$ / otra | §6.5 |
| D-2.1 | Algoritmo | Double DQN / Q-learning tabular | §5.3, §6.6 |
| D-2.2 | Recompensa del TWAP | R1-R6 | §6.3 |
| D-2.3 | γ | Tabla de §6.4 | §6.4 |
| D-2.4 | *Buffer*: tamaño, llenado, frecuencia de aprendizaje | §6.5 | §6.5 |
| D-2.5 | Criterio de mejora y regla de ε | §7 | §7 |
| D-3.x | Mejoras del ej. 4 | B, C, A | §8 |

---

## 5. Ejercicio 1: entorno y entrenamiento completos

### 5.1 Requisitos exactos

Completar `execution_lab.ipynb` (o un entorno propio pensado para ejecución y con los datos del taller) hasta lograr un entrenamiento y una ejecución completos; incluir los elementos de un entorno de RL; seguir el orden adecuado de interacción; entrenar en el orden correcto para el algoritmo elegido; documentar y motivar. No hace falta reutilizar el notebook, el algoritmo ni los paquetes (PyTorch permitido), ni que el entrenamiento "responda a una lógica con sentido", ni motivar en detalle la parametría (ENUNCIADO ej. 1). En clase: para el ej. 1 basta "una ejecución hasta el final del Notebook", sin recompensa específica (T07 ~04:58). Detalle en [notebook_partida §5](notebook_partida.md).

### 5.2 Plan en orden

**Propuesta:**

1. **Cargar y mirar los datos** de un activo: claves, días, columnas, longitud de cada día (≈3.181 filas de 13:00 a 17:25, [08 §3.1](08_entorno_best_execution.md)), `cumvol` que no empieza en 0 (T07 ~04:43).
2. **Completar el LABSTEP función a función**, en su orden (1)-(13), con el `print` de cada celda como prueba, y pasar cada cálculo al `def` con `global` donde se reasigne (E27, [08 §13](08_entorno_best_execution.md)). Es el método que propuso el profesor (T07 ~04:34-04:35).
3. **Copiar a la clase** con `self.` y vigilar las trampas ya identificadas en [08 §6](08_entorno_best_execution.md): `list(data.keys())` (E13); límite superior 601 e `int(...)` truncando en (1) (E14); `init_time` como etiqueta existente del índice; alineación $m_{t-1}v_t$ y `.values` en (4); registrar también en `_do_nothing` (E16); `>=` en la condición de tiempo (E17); rellenar `actions_hist` en `step`, que ninguna función rellena (E7) y hace falta para la separación entre compras (§4.3) y para auditar la política; volumen del primer *bin* (E11) y primer precio NaN en el VWAP parcial (E12) ([08 §15](08_entorno_best_execution.md); §10, filas 23-25).
4. **Pruebas de invariantes** (Nota), antes de entrenar nada: con la política "comprar siempre" el episodio dura $n$ pasos y acaba por orden completada; con "no comprar nunca" (salvo el paso 0 forzado) dura $N$ y queda $n-1$ pendiente; con el TWAP determinista $e_{\text{fin}}=n$, se completa hacia $t\approx N-N/n$ ([08 §7](08_entorno_best_execution.md)) y $\max_t\lvert e_t-tn/N\rvert\le1$; longitudes de `price_hist`, `vol_hist` y `algo_vwap_hist` iguales a los pasos; sin NaN en los VWAP; los *asserts* de (1) pasan en miles de `reset`.
5. **TWAP determinista** (celda 9) con la regla del profesor: si vas por delante del plan, no ejecutes; si vas igual o por detrás, lanza una hija (T07 ~04:53-04:54). Comparar con enteros ($e\cdot N\le t\cdot n$) para evitar empates en coma flotante ([08 §7](08_entorno_best_execution.md)). Graficarlo con `plot_results`: debe salir la diagonal de la figura de referencia.
6. **Agente:** instanciar el `DDQNAgent` (celda 14) o su versión PyTorch (§3.2). Corregir antes del primer entrenamiento: `mse = np.nan` antes del bucle y no entrenar en el *log* (E4); `agent.env = env` en la evaluación (E5); `plt.show()` o figura nueva en cada `plot_results` ([08 §12.3](08_entorno_best_execution.md)).
7. **Parámetros provisionales** (el ej. 1 no exige motivarlos en detalle, ENUNCIADO): ε = 1 antes del llenado del *buffer* para que el comentario de la celda 16 sea cierto (punto 13 de [04 §13](04_exploracion_memoria_y_estabilidad.md)); `action_sample` con $p\approx\rho$ (§6.5); pocos episodios.
8. **Ejecutar el notebook completo** hasta la celda 21 y guardarlo ejecutado.

### 5.3 Decisiones mínimas del ej. 1

- **Activo:** REP o SAN. **Propuesta:** REP, porque `orderbook.pkl` y `executions.pkl` son de Repsol (LAB celda 2) y permitirían extender el estado o el precio en la idea A sin cambiar de activo. SAN es igual de válido para los ej. 1-3; la duda de si SAN trae una columna entera se resuelve al cargar ([08 §3.1](08_entorno_best_execution.md)).
- ***Split*:** el ej. 2 pide imágenes en *train* (ENUNCIADO) y la celda 20 grafica sobre `dict_["test"]`. **Propuesta:** entrenar y graficar en *train*; si se usan val o test, como extra y dicho ([notebook_partida §12](notebook_partida.md), fila 8).
- **Algoritmo:** el Double DQN del LAB. El tabular 2×2 que propuso el profesor (T07 ~04:54-04:55) es válido, pero deja poco que motivar en la segunda parte del ej. 2 (no hay *buffer*), así que se propone como **control** barato en el ej. 2 (§6.6), no como solución principal.

### 5.4 Qué enseñar

El esquema de llamadas ([08 §5](08_entorno_best_execution.md)) con una frase por función; la salida de las pruebas del paso 4; `plot_results` del TWAP determinista y de un episodio del agente tras un entrenamiento corto (con recompensa por defecto, su comportamiento no tiene por qué parecerse a nada, y conviene decirlo); la curva de la celda 21 con la advertencia de que incluye exploración.

**Nota (qué cabe esperar con la recompensa por defecto):** la recompensa $(\text{VWAP}_{ep}-p)/\text{VWAP}_{ep}$ usa el VWAP final del episodio, que no es predecible con $[T,V]$, y paga en media ≈½ *spread* por compra (≈−1,7 pb en REP si el *spread* es de un *tick* a ~15 €; no verificado con los datos, porque los pickles no se han cargado, [08 §6.6](08_entorno_best_execution.md)), así que con $r=0$ al no comprar y sin penalización final el óptimo es **no comprar** ([08 §6.6](08_entorno_best_execution.md); [03 §4.4](03_recompensas_y_descuento.md)). Es un buen arranque para la exposición del ej. 2: "esto es lo que hace la recompensa por defecto y por qué".

---

## 6. Ejercicio 2: un TWAP por recompensas

### 6.1 Requisitos exactos

Dos partes igual de importantes: (1) modificar la recompensa para replicar el comportamiento de un TWAP, explicando qué variables hay en el estado y cómo y por qué se define la recompensa, con gráficos; (2) ajustar y motivar la parametría (γ, *buffer*, etc.). Hijas de 1 título y CARE alineada (40 títulos en 400 *steps*: 1 de cada 10); sin impacto ni costes; réplica no exacta pero comportamiento similar a `images/twap_model/twap_reproduction_dqn.png`; sin validación ni test, basta *train*; **debe funcionar con distintos días, momentos, duraciones y tamaños** dentro del rango; basta un activo (ENUNCIADO ej. 2). El rango esperado del tamaño es el de (1): $N\in[400,600]$, $\rho\in[0{,}075;\ 0{,}125]$, $n$ entre 30 y 75 títulos (LABSTEP celda 6; [notebook_partida §6.3](notebook_partida.md)).

### 6.2 Estado

| Opción | Vector | A favor | En contra |
|---|---|---|---|
| **E1** (LAB) | $[T,V]$ | Lo que dijo el profesor (T07 ~04:52) y lo que usa el paper de 2006 como base (T07 ~04:02; [07 §17.1](07_rl_para_ejecucion.md)); Markov para el objetivo de seguimiento; ya normalizado en $[0,1]$ | La red tiene que aprender la frontera $V=T$ a partir de dos entradas; con 2 neuronas de entrada y ReLU es fácil, pero no inmediato |
| **E2** | $[T,V,d]$ con $d=V-T$ | El profesor recomendó pasar "el ratio o la diferencia" entre tiempo y volumen restantes (T07 ~03:21; [02 §5.7](02_diseno_de_entornos.md)); la frontera queda en $d=0$, lineal en una sola entrada | Información redundante (no añade nada que no esté en $T$ y $V$); hay que cambiar `n_features` y comprobar `_detect_num_feat` |
| **E3** | $[T,d]$ | Mínimo y con la frontera explícita | Se pierde $V$, que importa al final (no es lo mismo $d=0$ con $V=0$ que con $V=0{,}5$) |
| **E4** | E1 o E2 + $1/n$ | Cada compra mueve $V$ en $1/n$, distinto según el episodio: con $1/n$ la red puede saber cuánto "pesa" un título | Una entrada más, casi constante dentro del episodio |
| **E5** | E1 o E2 + $(N-t)/600$ | Hace exacto el Markov del final por tiempo con γ < 1 y $N$ variable (400-600): con el mismo $T$, a un episodio de 400 *bins* le quedan menos pasos que a uno de 600 ([02 §2.4](02_diseno_de_entornos.md)); responde al aviso de APP p.75 sobre duraciones variables | Una entrada más; con R2 y γ bajo (G1-G2) la diferencia es pequeña; hay que cambiar `n_features` |
| ~~Ratio $V/T$~~ | — | — | Explota cuando $T\to0$ ([02 §5.7](02_diseno_de_entornos.md)) |

**Propuesta:** **E1** como base, porque es lo pedido y basta; **E2** como variante barata si la red tarda en aprender la frontera, justificada con la cita de clase; **E5** si se usa γ alto (G3-G4) o la idea B. En la exposición, una frase sobre Markov: con R2, $T$ y $V$ capturan casi todo lo relevante; falta la escala del episodio ($1/n$, $1/N$), que fija cuánto mueve una compra el estado y el umbral de compra óptimo ($d>\frac{1}{2n}-\frac{1}{N}$, §6.3), así que el estado es solo **aproximadamente** de Markov entre episodios de distinto tamaño. Por eso E4 (añadir $1/n$ o $\rho$) es la corrección natural si la red falla en los extremos del rango (LABSTEP celda 6: $N\in[400,600]$, $\rho\in[0{,}075;\ 0{,}125]$). Las características de mercado no aportan nada porque **el precio no entra en el objetivo** del ej. 2 (T07 ~04:57-04:58).

### 6.3 Recompensas candidatas para replicar el TWAP

**Idea.** El profesor dio dos pistas: la regla del TWAP "responde prácticamente a ese ratio o diferencia" entre volumen y tiempo restantes (T07 ~04:54), y las recompensas del ej. 2 "no tienen que estar condicionadas a tu rendimiento" (T07 ~04:58). En el caso realista, la desviación respecto al plan tradicional era uno de los tres términos de la recompensa (CASOS p.43-44; [03 §6](03_recompensas_y_descuento.md)). Todo apunta a una recompensa **local, factible con $[T,V]$ y sin precio**.

| ID | Recompensa | A favor | En contra | Veredicto |
|---|---|---|---|---|
| **R0** | Por defecto: $(\text{VWAP}_{ep}-p)/\text{VWAP}_{ep}$ al comprar, 0 si no | Viene en el LAB | No factible con $[T,V]$; sesgo de −½ *spread*; óptimo = no comprar ([08 §6.6](08_entorno_best_execution.md)) | Solo como contraejemplo |
| **R1** | Acuerdo con la regla: $0$ si la acción coincide con la del TWAP determinista en ese estado, $-1$ si no | Densa, trivialmente factible, sin escala que calibrar; aplica la recompensa local de clase en la que "comparas tu acción con la acción que habría tomado" la alternativa, en cada decisión (T07 ~03:31, ~03:38; [03 §4.1](03_recompensas_y_descuento.md)) | Es casi aprendizaje supervisado de la regla: "por recompensas" en la forma, pero con poco que razonar; no enseña nada sobre γ | Control de que el *pipeline* aprende |
| **R2** | Seguimiento del plan: $r_t=-\lvert d_{t+1}\rvert$, medido tras la acción | Codifica "ir de la mano" (T07 ~04:52) sin decir qué acción tomar; factible; escala ya acotada en $[-1,0]$ (APP p.61, [03 §9](03_recompensas_y_descuento.md)); base natural de las ideas A y B | Pequeño incentivo a acelerar las últimas hijas (Nota abajo); hay que fijar en qué instante se mide $d$ | **Recomendada** |
| **R3** | Cuadrática: $r_t=-c\,d_{t+1}^2$ | Penaliza más las desviaciones grandes; suave | Casi no penaliza el error de un título (≈$1/n^2$), así que tolera escalones; hay que calibrar $c$ | Variante de R2 |
| **R4** | Solo terminal: penalización por volumen pendiente al final | *Sparse*, limpia, coincide con "ejecutar la orden completa" (T07 ~01:45) | No dice **cuándo**: cualquier calendario que complete es óptimo; con el fin por orden completada, comprar todo al principio es tan bueno como el TWAP: el "ejecuta todo al principio" de T07 ~03:34 | Insuficiente sola |
| **R5** | R0 + penalización terminal por pendiente | Alineada con el objetivo de precio | Sigue sin ser factible con $[T,V]$: el calendario sale aleatorio | No replica el TWAP |
| **R6** | Compuesta: R2 + término terminal (pendiente o finalización) [+ $w\cdot$precio local] | Es el "TWAP con esteroides" de clase (CASOS p.43-44); permite pasar al ej. 4 sin cambiar de marco | Más pesos que calibrar; registrar cada término por separado (T06 ~02:49-02:50) | Para el ej. 4 (ideas A y B) |

**Nota (cálculo propio, sin datos, con la mecánica del LAB: compra forzada en $t=0$, fin al completar o al agotar el tiempo):**

- **R2 con γ = 0 ya produce un TWAP.** Tras la acción, $d$ pasa a $d+1/N$ si no se compra y a $d+1/N-1/n$ si se compra; comprar da más recompensa inmediata si y solo si $d>\frac{1}{2n}-\frac{1}{N}$, es decir, cuando el agente va retrasado más de medio título (descontada la deriva $1/N$ de un paso). Es la regla del `if` con un desplazamiento de medio título: un TWAP "redondeado". Matiz: el umbral depende de $n$ y $N$, que no están en $[T,V]$, así que la red solo puede **aproximarlo** (§6.2, E4).
- **Penalización acumulada por episodio** ($\sum_t\lvert d_{t+1}\rvert$): con $N=500$, $n=50$, el TWAP determinista suma 4,43 y la política "comprar si $d_t>\frac{1}{2n}-\frac{1}{N}$" 2,52; con $N=400$, $n=30$, 5,72 y 3,39; con $N=600$, $n=75$, 3,46 y 2,01. Con el umbral sin el $-1/N$ ($d_t>\frac{1}{2n}$) saldría algo más: 2,54 / 3,47 / 2,13. Comprarlo todo al principio suma 22,95 ($N=500$, $n=50$): R2 penaliza con claridad el "todo al principio".
- **Artefacto final.** Como el episodio termina al completar la orden, terminar antes ahorra penalizaciones futuras. Partiendo del TWAP determinista, acelerar las últimas 3-5 hijas mejora la suma un 2-7 % (4,43 → 4,32 con 3 hijas, $N=500$, $n=50$; 5,72 → 5,35 con 5 hijas, $N=400$, $n=30$), y acelerar 10 ya empeora. Es pequeño, pero puede verse como un racimo de compras al final. Remedios: (a) en `_compute_done_reward`, cuando se completa antes de tiempo, sumar la penalización que habría acumulado el resto del horizonte con $V=0$, $\sum_{k}T_k\approx T_{\text{fin}}^2N/2$; (b) no terminar al completar y dejar que el episodio corra hasta $N$ sin poder comprar más. (a) es más limpio y es el "gestionar recompensas final de episodio" de APP p.85.
- **R1 no tiene ese artefacto:** acelerar exige desobedecer la regla, con $-1$ por paso.

**Detalles de implementación que hay que decidir y documentar** (Nota):

- **Dónde se calcula.** `step` llama a la acción, y por tanto a `_compute_reward`, **antes** de `state_pos += 1` (LAB celda 7). Para medir $d$ tras la acción hay que usar `state_pos + 1` dentro de la recompensa o calcularla en `step` tras el incremento. Lo segundo modifica `step`: hay que señalarlo.
- **Usar `self` y quitar los parámetros** de `_compute_reward`, como sugiere el TODO opcional (LAB celda 7).
- **El valor de `vol == 0`.** Con R2 no hay un valor fijo: la penalización depende del estado. Con R0/R5 es la primera decisión del ejercicio ([03 §4.4](03_recompensas_y_descuento.md)).
- **La recompensa terminal no entra en `reward_hist`** (E8): registrarla aparte si se quiere el *log* por criterio ([03 §7](03_recompensas_y_descuento.md)).

**Propuesta:** **R2** como recompensa del ej. 2, con el remedio (a) del artefacto final, y **R1** como experimento de control de una sola figura ("con la regla como recompensa el agente la aprende en X episodios; con R2, que no le dice qué hacer, también"). R4 y R0 se enseñan como contraejemplos breves: son la forma más directa de motivar R2 con evidencia propia, en la línea de "contraponer episodios buenos y malos" (T06 ~02:34-02:35). Si se quiere una sola frase para la exposición: "penalizamos en cada paso la distancia entre el volumen y el tiempo restantes; con eso, comprar solo compensa cuando se va retrasado, que es exactamente la regla del TWAP".

### 6.4 γ calculado para episodios de 400-600 *steps*

**Idea.** γ se calibra hacia atrás: cuánto tiempo deben "ver" las acciones y cuántos *steps* caben en ese tiempo (T06 ~03:50-03:52; T07 ~01:09-01:10; [03 §12](03_recompensas_y_descuento.md)). Con γ = 0,95, tras 50 pasos el valor queda en un 7 % (T06 ~03:49); al día siguiente lo recordó así: "parece que es muy alto y no es muy alto" (T07 ~01:10).

**Peso $\gamma^k$ de una recompensa a $k$ *steps*** (Nota, cálculo propio; 10 *steps* ≈ separación entre hijas, 40-50 ≈ 4-5 hijas, 400-600 = episodio):

| γ | $H=1/(1-\gamma)$ | $k=10$ | $k=40$ | $k=50$ | $k=400$ | $k=500$ | $k=600$ |
|---|---|---|---|---|---|---|---|
| 0,90 | 10 | 0,349 | 0,015 | 0,005 | 0 | 0 | 0 |
| 0,95 | 20 | 0,599 | 0,129 | 0,077 | 0 | 0 | 0 |
| 0,99 | 100 | 0,904 | 0,669 | 0,605 | 0,018 | 0,007 | 0,002 |
| 0,995 | 200 | 0,951 | 0,818 | 0,778 | 0,135 | 0,082 | 0,049 |
| 0,998 | 500 | 0,980 | 0,923 | 0,905 | 0,449 | 0,368 | 0,301 |
| 0,999 | 1.000 | 0,990 | 0,961 | 0,951 | 0,670 | 0,606 | 0,549 |
| 1 | ∞ | 1 | 1 | 1 | 1 | 1 | 1 |

**γ que conserva una fracción $f$ al final del episodio**, $\gamma=f^{1/N}$: con $f=0{,}5$, 0,99827 / 0,99861 / 0,99885 para $N$ = 400 / 500 / 600; con $f=0{,}1$, 0,99426 / 0,99541 / 0,99617; con $f=0{,}05$, 0,99254 / 0,99403 / 0,99502.

**Opciones:**

| Opción | Razonamiento | A favor | En contra |
|---|---|---|---|
| **G1.** γ ≈ 0,9-0,95 | Con R2 la consecuencia de no comprar se agota en la siguiente hija, unos $1/\rho$ ≈ 8-13 *steps*; $H$ ≈ 10-20 cubre esa ventana | Q pequeños y estables; propagación rápida | Si se añade un término terminal (idea B), no se ve desde donde se decide |
| **G2.** γ = 0,99 (por defecto en BE) | $H$ = 100 ≈ 8-12 hijas; el TWAP solo necesita mirar unas pocas hijas hacia delante | Conserva el valor por defecto con una motivación explícita; margen frente a G1 | Una penalización final pesa 0,2-1,8 % desde el inicio: inútil para la idea B |
| **G3.** γ ≈ 0,995-0,998 | Que un término al final del episodio pese un 5-45 % desde el principio | Necesario si la recompensa tiene peso terminal (pendiente, finalización al 90 %) | Q de mayor magnitud y más lentos de propagar con DQN |
| **G4.** γ = 1 | Episodio finito y sin urgencia que meter en el descuento: "No te compliques la vida con el gamma" y γ = 1, sin descontar (T06 ~00:40-00:41); el LAB admite $[0,1]$ (LAB celda 11) | Coherente con la clase; con R2 la urgencia ya está en la recompensa negativa, como el −1 por paso del acantilado (T06 ~02:50-02:51) | Con DQN, el *bootstrapping* sobre 400-600 pasos sin descuento puede acumular sobreestimación; "mucho cuidado" con duraciones variables (APP p.75) |
| **G5.** γ por episodio, $\gamma=f^{1/N}$ | "Incluso el Karaoma puede ser dinámico" (T07 ~00:41) *(dudoso: lo más probable es γ; podría seguir refiriéndose al buffer, del que hablaba justo antes)*; [03 §12.2](03_recompensas_y_descuento.md) | Mismo peso relativo del final en todas las duraciones | Más complejo; mejor como idea del ej. 4 |

**Propuesta:** con R2 sola, **G2 (0,99)** motivada por la separación entre hijas, más un barrido corto **{0,9; 0,99; 0,999}** con 3 semillas como evidencia de la segunda parte del ej. 2: lo esperable (Nota) es que los tres repliquen el TWAP, porque R2 es local, y que 0,999 sea más lento o inestable. Si se añade un término terminal (remedio (a) de §6.3 o la idea B), pasar a **G3 (≈0,995)** y decirlo. La frase de la exposición: "elegimos γ mirando cuántos pasos hay entre dos hijas, no por costumbre; con nuestra recompensa la urgencia ya está en la penalización".

**Nota (por qué aquí no aplica el "γ alto intradía" de clase):** el aviso de que un γ mal calibrado hace "operar casi siempre al principio" (T07 ~01:09-01:10) es para recompensas de retorno, donde conviene esperar a la mejor oportunidad. Con R2 no hay oportunidades que esperar, solo un plan que seguir. Volverá a importar en la idea A.

### 6.5 *Buffer*, llenado inicial, `action_sample` y frecuencia de aprendizaje

**Cuántas experiencias tiene un episodio** (Nota, simulación sin datos con la mecánica del LAB, 2.000 episodios por caso):

| Política | $N=500$, $n=50$ | $N=400$, $n=30$ | $N=600$, $n=75$ |
|---|---|---|---|
| Aleatoria con $p=0{,}5$ | 99 *steps*, completa el 100 % | 58, 100 % | 149, 100 % |
| Aleatoria con $p=\rho$ | 468, completa el 59 % (pendiente medio 2,3) | 365, 61 % (1,6) | 571, 55 % (2,9) |
| Aleatoria con $p=1{,}2\rho$ | 408, 95 % (0,2) | 322, 89 % (0,3) | 495, 97 % (0,1) |
| TWAP determinista | ≈ 491 | ≈ 388 | ≈ 593 |

Con $p=0{,}5$ los episodios aleatorios se quedan en el 15-25 % del horizonte y nunca visitan $T$ pequeño, que es donde se decide si la orden se completa (E10, [08 §6.13](08_entorno_best_execution.md); [04 §4.5](04_exploracion_memoria_y_estabilidad.md)). La acción aleatoria no tiene por qué ser uniforme (APP p.57; CASOS p.45; T07 ~00:39), y el propio TODO pregunta qué distribución es mejor (LAB celda 7).

**Propuesta para `action_sample`:** $p=\rho$ = `vol_care / episode_bins`, o algo por encima ($1{,}1$-$1{,}2\,\rho$) para que la exploración no deje sistemáticamente volumen pendiente. Motivación: recorre todo el horizonte y genera estados con retraso y adelanto en las mismas proporciones.

**Tamaño del *buffer* en episodios** ("tengo mis 100 últimas partidas", T06 ~03:59-04:03; [04 §6.3](04_exploracion_memoria_y_estabilidad.md)). Independencia frente a obsolescencia (APP p.78; T07 ~03:45):

| Opción | Experiencias | Episodios tipo TWAP (~450 *steps*) | A favor | En contra |
|---|---|---|---|---|
| **B1** | 10.000-20.000 | ≈ 20-45 | Con $[T,V]$ y R2 hay pocas casuísticas, como en el *cliff walking*, donde bastan las últimas 10 partidas (T07 ~03:48); poca obsolescencia | Situaciones raras (mucho retraso cerca del final) que solo genera la exploración salen pronto del *buffer*: riesgo de olvidar cómo recuperarse |
| **B2** | 50.000 (por defecto en BE) | ≈ 100-110 | La regla general de "unas 100" (T07 ~03:47-03:48); coincide con el valor por defecto, ahora motivado | Más obsolescencia al principio, cuando la política cambia deprisa |
| **B3** | 100.000-250.000 | ≈ 220-550 | Diversidad de regímenes de mercado si el estado los incluye (100-500 sesiones, T07 ~03:47-03:48); "mejor pasarse de largo" y que CER corrija la obsolescencia (T06 ~04:36 *(dudoso: quizá atribuido al autor del paper)*; APP p.81) | Innecesario con $[T,V]$; más memoria y `sample` más lento sobre `deque` ([08 §9.3](08_entorno_best_execution.md)) |

**Propuesta:** **B2 en el ej. 2**, con el argumento de clase y la cuenta "100 episodios × ~450-500 *steps*"; aclarando que con $[T,V]$ la diversidad que importa no son los regímenes de mercado sino las combinaciones de $N$, $n$ y fase del episodio. B1 como punto del barrido de la segunda parte. **B3 solo en la idea A**, donde el estado sí incluye mercado. El `DDQNAgent` ya usa CER, que reduce la sensibilidad al tamaño (APP p.81; [01 §6.7](01_aproximacion_funcional_y_dqn.md)).

**Llenado inicial.** El LAB lo llena **entero** con acciones aleatorias (LAB celda 16), y en clase se dijo que no hace falta: no llenarlo entero, quizá 50 partidas en lugar de 100 (T06 ~04:20-04:21), y con aleatorias, ruidosas, solo un poco, 20-30 partidas, para que el agente actúe y aprenda a la vez (T06 ~04:29-04:30; T07 ~03:50-03:51; [04 §6.6](04_exploracion_memoria_y_estabilidad.md)).

| Opción | A favor | En contra |
|---|---|---|
| **L1.** Entero, aleatorio (LAB) | Sin tocar el código | Va contra lo dicho en clase; con $p=0{,}5$, cientos de episodios cortos e inútiles |
| **L2.** 20-30 episodios aleatorios con $p\approx\rho$ (≈10.000-15.000 *steps*) | Lo recomendado en clase; cubre el horizonte | Hay que cambiar `range(buffer_size)` por un número de pasos o episodios (bloque modificado) |
| **L3.** Mezcla con experiencias del TWAP determinista | Experiencias "con lógica" (APP p.79; CASOS p.45) | En el ej. 2 contamina la prueba: la red podría ajustarse al comportamiento clásico y no salir de él (T07 ~03:51), es decir, reproducir el TWAP por el *buffer* y no por la recompensa. **Nota:** con un *buffer* FIFO el sesgo dura lo que tardan en salir esas experiencias ([07 §10.3](07_rl_para_ejecucion.md)) |

**Precedente** (Nota): en el Taller 2 el DQN usó una memoria circular de 20.000 transiciones (unos 40 episodios), empezó a entrenar con 1.000 y daba un paso de gradiente cada 4 de entorno (T2 D-2.2), y convergió en todas las semillas en los mercados con señal (T2 `README.md`). Es un argumento propio, con evidencia medida, para L2 y para el punto B1 del barrido.

**Propuesta:** **L2**. Si se probara L3, enseñar la ablación sin ella para demostrar que es la recompensa la que produce el TWAP.

**Frecuencia de aprendizaje.** Con `learn_after = batch_size = 64` hay unas 8 actualizaciones por episodio de 500 pasos y cada experiencia se muestrea, en media, una vez; ≈37 % no se usan nunca ([04 §6.2](04_exploracion_memoria_y_estabilidad.md); [08 §11](08_entorno_best_execution.md)). **Propuesta:** probar `learn_after` 4-16 (en el Taller 2 se aprendía cada 4 pasos, T2 D-2.2) y `update_target_after` en torno a un episodio (500-1.000 pasos). Es barato con PyTorch y probablemente reduce el número de episodios necesario; motivarlo con la curva *greedy* (§7).

### 6.6 Resto de la parametría

| Parámetro | Por defecto | Propuesta y motivo | Fuente |
|---|---|---|---|
| `alpha` (tasa de Adam) | 0,01 (BE) | 1e-3: el valor del Taller 2; 0,01 es alto para Adam con una red pequeña. El profesor lo considera "menos importante" (T07 ~05:01) | T2 `README.md` (α = 0,001); T2 `notebooks/common/dqn.py` (`lr = 1e-3`); RLM `deep` usa 1e-4 ([09 §8](09_material_rl_notebooks.md)) |
| `hidden_neurons` | 32 | 32 basta para 2-3 entradas; no se barre | LAB celda 11 ("agentes sencillos"); T2 D-2.2 (la red "lo más simple posible") |
| `batch_size` | 64 | 64 | BE `dqn.py` |
| `nepisodes` | TODO | Presupuesto de cómputo (§3.2) y la curva *greedy*: parar de añadir episodios cuando la métrica se estabiliza (en el ej. 3 se permite un número fijo, ENUNCIADO) | ENUNCIADO ej. 3 |
| ε en el ej. 2 | Lineal por episodio (LAB celda 18) | Se puede mantener en el ej. 2 y sustituir en el ej. 3; o adelantar el criterio del ej. 3 y decirlo | LAB celda 18 |
| Algoritmo de control | — | **Q-learning tabular 2×2** (estados $d>0$ / $d\le0$; acciones comprar / no) con R2: "una matriz de 4" que el profesor puso como ejemplo (T07 ~04:54-04:55). Media figura: muestra que el problema es trivial con la representación adecuada y que la parte difícil del ej. 2 es la recompensa. Código reutilizable: `QLearningAgent` del Taller 1 (`src/agents.py`, interfaz `act / update / end_episode`, T1 D-0.2) o RLM `TD-q-learning`, uno de los dos notebooks más importantes según APP p.1 ([09 §5.1](09_material_rl_notebooks.md)); el profesor ofreció incluso importar o reutilizar el suyo (T07 ~00:09) *(dudoso: la transcripción dice "el poolearning")* | [00 §9](00_repaso_rl.md); [T1 02_td_sarsa_qlearning.md](../raw/referencias/rl-trading-agents/docs/teoria/02_td_sarsa_qlearning.md) |

**Cómo presentar la segunda parte** (Nota): una tabla con γ, *buffer*, llenado, `p` de exploración, `learn_after` y `alpha`, cada uno con su valor, su motivo en una línea y la figura que lo respalda; y un barrido mínimo (un factor cada vez, 3 semillas) sobre γ y *buffer*, que son los dos que nombra el ENUNCIADO. Es el procedimiento por etapas con margen de ruido de los talleres anteriores (T1 D-2.3; T2 D-2.3), reducido a lo esencial.

### 6.7 Evidencias del ej. 2

1. **Rejilla de `plot_results`** (Nota): 4-6 episodios de *train* con $N$ y $\rho$ en los extremos y en el centro, y días y horas distintos, cada uno con su título (día, hora, $N$, $n$). Es el "algunas imágenes correctas" del ENUNCIADO, y la variedad demuestra el requisito.
2. **Trayectorias**: $V_t$ frente a $T_t$ de los 18-27 episodios del conjunto de evaluación, superpuestas sobre la diagonal, para el agente y para el TWAP determinista.
3. **Mapa de la política *greedy*** sobre una rejilla de $(T,V)$: debería verse "comprar" por encima de la diagonal y "no comprar" por debajo. Es la forma más directa de enseñar que el agente aprendió la regla y no un calendario fijo.
4. **Tabla de métricas** (§4.3) del agente frente al TWAP determinista: completado, $t_{\text{fin}}/N$, error de seguimiento máximo y medio en títulos, separación entre compras; media y dispersión.
5. **Curvas**: recompensa de entrenamiento (con exploración) y métrica *greedy* periódica, separadas (T2 D-0.3).
6. **Contraejemplos**: un episodio con R0 (no compra o compra al azar) y otro con R4 (compra al principio).
7. **Barrido de la segunda parte**: γ y *buffer* con 3 semillas.

### 6.8 Riesgos del ej. 2

- **Evaluar con la acción forzada desactivada** por no reasignar `agent.env` (E5): en test la primera acción no se fuerza y el VWAP del algoritmo puede salir NaN ([08 §12.1](08_entorno_best_execution.md)).
- **Precio irrelevante, pero visible.** Con R2 el precio no interviene; aun así el gráfico enseña VWAP del algoritmo frente al de mercado, y con `stats_df` al `ask1` frente a un entorno al *mid* (E3) el *slippage* aparente difiere en ≈½ *spread*. Decir cuál se muestra. **Nota:** un TWAP perfecto saldrá con *slippage* negativo del orden de medio *spread* frente a un VWAP al *mid* ([07 §15](07_rl_para_ejecucion.md)).
- **`plot_results` superpone figuras** si se llama varias veces en una celda ([08 §12.3](08_entorno_best_execution.md)).
- **Confundir la curva de entrenamiento con el aprendizaje**: incluye exploración (T07 ~00:20-00:22).
- **Semillas**: una buena figura de una sola semilla no demuestra nada (T06 ~01:13).
- **Sobreinterpretar**: con impacto 0 e hijas de 1 título, el resultado no dice nada de órdenes reales (T07 ~03:56-03:57). Decirlo antes de que lo pregunten.

---

## 7. Ejercicio 3: exploración según la mejora

### 7.1 Requisitos exactos

ε no debe depender del número de episodios; la exploración baja conforme mejora el agente (resultados y/o conocimiento del espacio); la duración del entrenamiento "debería" depender del aprendizaje, pero se puede mantener un número fijo de episodios; lo importante es responder **¿cómo determinamos que el algoritmo ha mejorado?** y **¿cómo trasladamos esta mejora a nuestro epsilon?** (ENUNCIADO ej. 3). En clase: tocar ε "para que no sea simplemente proporcional al número de episodios" (T07 ~04:58). Contexto completo en [notebook_partida §7](notebook_partida.md) y [04 §2-4](04_exploracion_memoria_y_estabilidad.md).

### 7.2 ¿Cómo medimos la mejora?

| ID | Métrica | Dimensión (APP p.64) | A favor | En contra |
|---|---|---|---|---|
| **M1** | Error de seguimiento *greedy* (máximo y medio, en títulos) y % de episodios completados, en el conjunto fijo de evaluación | Resultados | Interpretable y ligada al objetivo del ej. 2; no depende de la escala de la recompensa | Hay que fijar umbrales |
| **M2** | Retorno *greedy* medio con la recompensa del ej. 2, normalizado entre el inicial y el del TWAP determinista con la misma recompensa: $\pi_k=\dfrac{G_k-G_0}{G_{\text{TWAP}}-G_0}$ | Resultados + *benchmark* | Es el "nivel de aspiración" con un *baseline* clásico (APP p.66) y la "distancia al *benchmark*" de clase (T07 ~01:47); da una fracción en $[0,1]$ | Si la recompensa cambia (ej. 4), cambia el nivel; el TWAP no es el máximo de R2 (el "redondeado" lo supera, §6.3), así que $\pi$ puede pasar de 1: se resuelve con un nivel de aspiración móvil (§7.3) |
| **M3** | Cobertura: fracción de celdas visitadas de una rejilla de $(T,d)$, o conteos por celda | Conocer el espacio | El pseudo-conteo de clase (T07 ~00:36-00:37); el mapa de calor que propuso un alumno y el profesor aceptó (T07 ~00:40) | Por sí solo no dice si la política es buena |
| **M4** | Recompensa de entrenamiento suavizada | Resultados | Gratis | Sesgada por ε: es el error del *cliff walking* (T07 ~00:20-00:22; [04 §2.3](04_exploracion_memoria_y_estabilidad.md)) |
| **M5** | Pérdida o error TD | — | Gratis | No indica si ha terminado de aprender: un pico puede ser bueno y una pérdida baja puede ser estancamiento (T06 ~04:50-04:51; APP p.88) |

Cómo evaluar (T06 ~03:16-03:17; APP p.90): parar la exploración y probar la política en varios episodios, no en uno. **Nota:** siempre los mismos, como en el Taller 2 (unas 20 series de validación siempre iguales, T2 D-0.3), para que las evaluaciones sean comparables entre sí. **Propuesta:** el conjunto fijo de §4.3 (o un subconjunto de 10-20 episodios), cada `n_log = 25` episodios, que ya es el punto del bucle donde se imprime el *log* (LAB celda 18) y la frecuencia que usó el Taller 2 (cada 25-50 episodios sobre ~20 series, T2 D-0.3).

### 7.3 ¿Cómo se traslada a ε?

| ID | Regla | De dónde sale | A favor | En contra |
|---|---|---|---|---|
| **X1. Escalones** | Si la métrica mejora sobre la mejor previa más que un margen, $\varepsilon\leftarrow\max(\varepsilon_{min},\ k_\downarrow\,\varepsilon)$; si empeora claramente, $\varepsilon\leftarrow\min(1,\ k_\uparrow\,\varepsilon)$ o volver al *checkpoint*; si no, se mantiene | "Mantener ε alto mientras no mejore" (T06 ~03:17-03:18); subir ε o recuperar la versión anterior si empeora (T06 ~03:21) | Simple de explicar y de dibujar | Dos factores y un margen que fijar |
| **X2. Continua** | $\varepsilon=\varepsilon_{min}+(1-\varepsilon_{min})\,(1-\mathrm{clip}(\pi,0,1))$ con $\pi$ de M2 | "Ya está a mitad del objetivo, puedo bajar un poco la exploración" (T06 ~03:19) | Responde literalmente a la pregunta del ENUNCIADO: la mejora se traduce en ε mediante una función explícita | Ruido de la evaluación → ε oscila; suavizar $\pi$ (media móvil) o permitir solo bajadas |
| **X3. Por estado** | $\varepsilon(s)$ decreciente con los conteos de la celda de $s$, $\varepsilon(s)=\max(\varepsilon_{min},\ 1/\sqrt{1+N(s)/c})$ | Explorar menos por la Castellana y más por Sol, incluso dentro del episodio (T07 ~00:37-00:39; APP p.73) | La dimensión "conocer el espacio"; enlaza con la idea libre del ej. 4 | Requiere discretizar el estado y llevar los conteos en el agente; más difícil de evaluar |
| **X4. Combinada** | ε global de X1/X2 multiplicado por un factor local de X3 | Ambas dimensiones de APP p.64 | La más completa | La más difícil de calibrar y de exponer en 2 minutos |

**Salvaguardas** (Nota): suelo $\varepsilon_{min}>0$, porque dejar de explorar es dejar de aprender (T06 ~03:02); margen de ruido para no reaccionar al azar (T2 D-2.3); tope de episodios ("máximo temporal", T06 ~03:04-03:05; "doble criterio", T06 ~03:20), que el ENUNCIADO permite; y qué pasa si nunca mejora: con X1, ε se queda en 1 y el agente no sale del azar. Hay que poner un plan B (por ejemplo, un decaimiento lento de fondo, declarado, o detener y revisar la configuración).

**Nivel de aspiración móvil** (APP p.66: ajustar los niveles al alza o a la baja según se consigan; "objetivos crecientes", T06 ~03:23; [04 §3.3](04_exploracion_memoria_y_estabilidad.md)). **Propuesta:** cuando $\pi\ge1$ durante $P$ evaluaciones seguidas, se sube el nivel: por ejemplo, $G_{\text{TWAP}}$ se sustituye por el retorno de la política "comprar si $d_t>\frac{1}{2n}-\frac{1}{N}$", que es la óptima miope de R2 con γ = 0 (§6.3: 2,52 frente a 4,43 de penalización con $N=500$, $n=50$), y $\pi$ se recalcula. Así el criterio sigue informando después de igualar al TWAP, y X2 no deja ε en el suelo en cuanto se iguala al TWAP si aún queda margen. Si se sube el nivel, hay que marcarlo en el gráfico de ε y decirlo.

**Propuesta:** **M1 + M2 con X1** como solución principal, por ser la que mejor se explica y se defiende: "cada 25 episodios evaluamos la política sin exploración en 20 episodios fijos; si recorre más camino hacia el nivel del TWAP determinista que nuestro margen de ruido, bajamos ε un escalón; si empeora, lo subimos y recuperamos el último buen modelo". **X2** como variante si se quiere una función explícita. **M3** se registra siempre (cuesta poco) para dibujar el mapa de calor y, si da tiempo, **X3** va al ej. 4 como exploración por estado.

### 7.4 Comparación que hay que enseñar

**Propuesta:** mismo presupuesto de episodios, mismas semillas y mismo conjunto de evaluación para (a) el ε lineal del LAB y (b) el ε por mejora. Medir episodios hasta alcanzar un umbral de M1 (por ejemplo, error máximo ≤ 1-2 títulos en el 90 % de los episodios) y la métrica final.

**Medida de convergencia** (reutilizada de los talleres anteriores): el primer punto de la curva *greedy* que recorre el 90 % del camino entre su valor inicial y el final, dado en episodios, pasos de entorno y segundos (T2 D-0.3; T1 D-0.3 usaba la misma regla sobre la curva con exploración). La comparación se hace como un **duelo a configuración idéntica**, con las mismas semillas y los mismos episodios (T1 D-2.5). **Argumento de partida:** con decaimiento por episodios, ε depende del presupuesto y no del aprendizaje; en el Taller 2, con 500 episodios, ε acabó en 0,08 sin llegar a su mínimo de 0,01 (T2 D-2.2). No hace falta que (b) gane: el ENUNCIADO dice que no es tan importante dar con una solución correcta como explicar la lógica. Si (b) no mejora en un problema tan sencillo, es un resultado que se explica solo (como en el *cliff walking*, el agente aprende mucho antes de lo que dice la curva, T07 ~00:22).

### 7.5 Evidencias y coste

- **Gráfico doble**: ε y la métrica *greedy* frente a episodios, con los escalones marcados; la recompensa de entrenamiento en otro panel.
- **Mapa de calor** de los estados visitados (M3) al principio, a mitad y al final.
- **Tabla** (a) frente a (b).
- **Coste** (Nota): 20 episodios *greedy* cada 25 de entrenamiento casi doblan los *steps* simulados (sin `learn`). Con Keras puede ser prohibitivo (§3.2); con PyTorch, asumible. Alternativas: 10 episodios, cada 50, o evaluación por lotes.

### 7.6 Riesgos del ej. 3

- **Evaluar con la política exploratoria** o con un único episodio (LAB celda 20 evalúa uno solo; [04 §3.7](04_exploracion_memoria_y_estabilidad.md)).
- **Usar como "mejora" algo que depende de ε**, como la recompensa de entrenamiento (M4): circular.
- **Reaccionar al ruido**: sin margen, ε sube y baja al azar.
- **Olvidar que `action_sample` define qué es explorar**: con $p=0{,}5$, más ε significa más episodios cortos (§6.5).
- **No guardar ε y contadores en los *checkpoints***: al recuperar un modelo, ε debe volver con él ([04 §9.4](04_exploracion_memoria_y_estabilidad.md)).

---

## 8. Ejercicio 4: mejoras

### 8.1 Requisitos y cómo elegir

El ENUNCIADO lo llama "opcional" pero puntúa 1,5 (la clase dijo "el otro punto", *(dudoso)*, [notebook_partida §12](notebook_partida.md), filas 1 y 2). Propone tres ideas (A, B, C) y deja libertad para cualquier punto de la clase de entornos y entrenamiento, especificando con detalle la mejora y su lógica. La exposición debe decir qué se recogió de clase, cuál fue el razonamiento y qué se implementó (T07 ~04:31).

**Propuesta de criterio:** coste bajo, medible con el protocolo ya montado y ligado a algo dicho en clase. Por eso **B y C primero** y **A si queda tiempo**.

### 8.2 Idea A: mejorar el precio (media o dispersión) con información de mercado

**Lo que pide:** modificar la recompensa del TWAP para mejorar media del VWAP o dispersión, añadiendo al estado información del libro y/o de la tendencia; no se piden buenos resultados, sí un razonamiento y pasos correctos (ENUNCIADO ej. 4).

**Propuesta (qué mejorar):** la **dispersión** antes que la media. En clase, el POV costaba unos 7 pb de media de forma estable, frente a unos 3 pb de media de los otros con colas de hasta 20 pb de coste, y era el más usado (~80 % de las órdenes) porque al cliente no se le pueden dar resultados "súper inestables" (T07 ~01:59-02:01; [06 §11.3](06_algoritmos_de_ejecucion.md)); y a los brókeres les daban igual 2-3 pb porque el dinero es del cliente (T06 ~02:59-03:00). Con ventajas de pocos pb que caen dentro del ruido (Riesgos, abajo), atacar la dispersión es más defendible. Recompensa posible: R2 + penalización de la desviación local del precio de compra frente al *mid* reciente; métrica: desviación típica o percentil 95 del *slippage* frente al TWAP en los mismos episodios.

**Planteamiento propuesto** (Nota, sobre [07 §7, §9 y §13](07_rl_para_ejecucion.md)):

1. **Estado**: $[T,V,d]$ + características normalizadas y causales, cada una justificada por la regla de bróker que debería aprender ([06 §12.3](06_algoritmos_de_ejecucion.md)): *spread* relativo (en *ticks* o pb), desbalanceo del primer nivel $v^{bid}_1/(v^{bid}_1+v^{ask}_1)$, rendimiento a corto $\ln(m_t/m_{t-k})$ con $k$ dentro del `look_back` de 60 *bins*, actividad (volumen negociado en los últimos $k$ *bins* frente a su media), hora del día. El paper de 2006 encontró que el *spread* era lo que más aportaba (7,97 %) y el desbalanceo casi nada (0,13 %) (CASOS p.61; [07 §17.1](07_rl_para_ejecucion.md)). Con `orderbook.pkl` (REP 2019, 3 niveles) se puede añadir profundidad, pero son otros días y otra partición.
2. **Recompensa local y factible**: precio de cada compra frente a una referencia **local** (VWAP de mercado hasta el *step*, `_compute_market_vwap`, o el *mid* de los últimos $k$ *bins*), o frente a la compra que habría hecho el TWAP; nunca frente al VWAP final del episodio, que no se puede predecir (T07 ~03:35-03:44; [03 §4](03_recompensas_y_descuento.md)). En pb, para que tenga escala comparable a R2.
3. **Restricción de libertad**: mantener R2 con un peso que limite la desviación del plan (el perfil "conservador" de 5-10 % de libertad, CASOS p.53), o una regla que desautorice al agente si se aleja más de un umbral (T07 ~03:53). Así el simulador sigue siendo creíble (T07 ~03:56) y el cliente no se asusta (T07 ~03:32-03:33).
4. **Evaluación**: *slippage* frente al TWAP determinista en los mismos episodios, media **y** dispersión; mejor en val/test que en *train*, porque aquí sí importa la generalización (CASOS p.51).
   - **Nota (tamaño del conjunto):** elegir el número de episodios $K$ para que el error estándar $\sigma_{slip}/\sqrt{K}$ quede por debajo de la diferencia que se busca; con diferencias de ~2-3 pb harán falta cientos de episodios, no los 18-27 de la rejilla de §4.3, que bastan para el error de seguimiento del ej. 2, casi determinista. En el Taller 1, con 50 series la mediana se movía hasta 10 puntos y con 200 el error estándar bajaba a ≈±4 (T1 D-0.1). Las variantes se eligen en val y el test se mira una sola vez; lo que se elija después de verlo se mide en episodios nuevos (T1 D-0.4; T2 D-2.4).
   - **Controles** (Nota, lecciones de T1 y T2): (a) programar como *benchmark* 2-3 reglas de bróker sobre las mismas características ([06 §12.3](06_algoritmos_de_ejecucion.md); p. ej. "adelantar la hija si el *spread* está en su mínimo reciente y se va retrasado"). En el Taller 2 las reglas de un umbral fueron "el listón real" y ni DQN ni A3C las batieron en test (T2 D-0.3; T2 `README.md`), y en el Taller 1, 70 reglas fijas mostraron que no había ventaja simple con aquel estado (T1 D-2.4). (b) **Placebo**: la misma característica permutada entre *bins*; si el agente "mejora" igual, la mejora no viene de la información (T1 D-1.4; T2 D-3.2).
   - **Propuesta (dos precios):** en clase se dijo que en muchos algoritmos se piden "2 precios", uno más o menos realista (T07 ~05:02) *(dudoso)*. Registrar en cada compra el `ask1` al inicio del *bin* (el del LAB, sesgado) y el *mid* medio entre las dos fotos (o el interpolado con ~2 s de latencia, T07 ~05:04-05:05); entrenar con uno y dar el *slippage* frente al TWAP con los dos. Si la ventaja de la idea A desaparece con el precio realista, es un artefacto de la asunción.

**Riesgos:** fugas de información (precalcular en `reset` está bien para la recompensa, nunca para el estado, [02 §7.2](02_diseno_de_entornos.md)); `_detect_num_feat` no llama a (4) y rompe el constructor si las características dependen de él (E15); γ: con recompensas de precio vuelve el problema de "operar al principio" (T07 ~01:09-01:10), así que G3; el sesgo del precio de ejecución (E2) importa ahora: decidir si se usa el *ask* al inicio del *bin* o una aproximación entre fotos (*mid* medio, latencia; T07 ~05:01-05:05; [08 §6.7](08_entorno_best_execution.md)); y, sobre todo, que las ventajas de pocos pb están dentro del ruido (en el TFM del profesor el algoritmo obtenía +2,8 pb frente al VWAP de mercado y el TWAP −1,88 pb, unos 4,7 pb de diferencia, con una distribución visiblemente más ancha *(lectura visual, dudosa)*, CASOS p.60; [07 §15](07_rl_para_ejecucion.md)). *Buffer* B3, por la diversidad de regímenes.

### 8.3 Idea B: sin volumen pendiente y ≈90 % de la duración

**Lo que pide:** añadir recompensas para que (1) no quede volumen pendiente y (2) se consuma aproximadamente el 90 % de la duración de la orden, con una calibración que dé resultados estables (ENUNCIADO). En clase: no vale quedarse en el 80 % del volumen; terminar "más o menos" en un rango del 80 al 95 %, porque "te doy un poco de tiempo de más" (T07 ~04:30); al repasarlo, 70-90 % (T07 ~04:58). **Nota:** prevalece el ENUNCIADO: volumen al 100 % y **tiempo** ≈90 % ([notebook_partida §12](notebook_partida.md), fila 3).

**Opciones de diseño:**

| Opción | Qué es | A favor | En contra |
|---|---|---|---|
| **B-1. Plan comprimido** | R2 con el plan $x^*_t=\min(1,\ t/(0{,}9N))$ en lugar de $t/N$, más penalización terminal por pendiente | Reutiliza todo el ej. 2; el agente aprende un TWAP de 0,9N; la "calibración" se reduce a un parámetro con significado | Termina hacia el 0,9N de forma casi determinista: cumple, pero es poco "libre" |
| **B-2. Banda de finalización** | R2 con el plan original + término terminal: penalización por pendiente y penalización si $t_{\text{fin}}/N$ sale de la banda [0,80; 0,95] (o creciente con la distancia a 0,90) | Lo que se dijo en clase: un rango, no un punto | Es una recompensa *sparse*: exige γ alto (G3) y más episodios; competencia entre términos que hay que calibrar |
| **B-3. Penalización por pendiente como agresiva final** | Valorar lo no ejecutado al precio de una agresiva en el último segundo (CASOS p.43) | Lo más cercano al caso real | Necesita precio en la recompensa; mejor en combinación con A |

**Calibración** (Nota): estimar antes el rango de cada término en un episodio típico (APP p.61): con R2 la penalización acumulada de un TWAP es de 2-6 por episodio (§6.3), así que la penalización por título pendiente tiene que ser del orden de la penalización acumulada o mayor (≥ 1 por título) para que dejar uno sin comprar nunca compense; y la de terminar fuera de banda, comparable. Registrar cada término por separado (T06 ~02:49-02:50) y ajustar mirando los comportamientos: con mucho peso a completar, puede ejecutar todo al principio; con mucho peso al precio, si no ve buen precio puede dejar volumen pendiente (T07 ~03:34; [07 §16](07_rl_para_ejecucion.md)). **Nota:** con R2 sola, seguir el plan no deja volumen pendiente; el riesgo propio de B es el contrario, que un término terminal con mucho peso frente a R2 empuje a acelerar las últimas hijas (el artefacto final de §6.3) o a terminar fuera de banda.

**Evidencias:** histograma de $t_{\text{fin}}/N$ y del volumen pendiente sobre el conjunto de evaluación, con la banda marcada; un par de `plot_results` en los que la línea naranja llega a 1 hacia el 90 % del eje; la "estabilidad" que pide el ENUNCIADO, como dispersión de $t_{\text{fin}}/N$ entre episodios y entre semillas.

**Propuesta:** **B-1** como primera versión (rápida y robusta), y **B-2** si se quiere enseñar la calibración de verdad, que es lo que el profesor destacó ("es un objetivo sencillo, pero siempre tenéis que aprender a calibrar", T07 ~04:30). Elegir y justificar la banda: 80-95 % encaja con "aproximadamente 90 %".

### 8.4 Idea C: parada por aprendizaje, *checkpoints* y recuperación

**Lo que pide:** condiciones de parada en base al aprendizaje, guardar los pesos cada cierto tiempo para no perder un buen entrenamiento y poder recuperarlo (ENUNCIADO). En clase: hacer paradas y registros por si hay *catastrophic forgetting* o el agente se atasca, recuperar y darle otra oportunidad (T07 ~00:18-00:19; T06 ~04:49; APP p.88, p.90; [04 §9.3-9.4](04_exploracion_memoria_y_estabilidad.md)).

**Propuesta:**

- **Guardar** en cada evaluación de §7.2: el **mejor** modelo según la métrica *greedy* (no el último) y un *checkpoint* periódico. Contenido: pesos de la red principal y de la *target*, estado del optimizador, ε, contadores (`learn_counter`, episodio), estados de los generadores aleatorios y, si se quiere reanudar idéntico, el *buffer* ([04 §9.4](04_exploracion_memoria_y_estabilidad.md)). En PyTorch, `state_dict` + un JSON; en TF, `save_weights` de `primary_model` y `target_model`.
- **Parar** cuando (a) se alcanza el nivel de aspiración (umbral de M1, o $\pi\ge1$ en M2) durante $P$ evaluaciones seguidas, o (b) no hay mejora en $P$ evaluaciones (paciencia), o (c) se agota un tope de episodios.
- **Evidencia previa** (Nota): en el Taller 2, quedarse con el mejor punto según la evaluación *greedy* subió la peor semilla del mercado de reversión de +5,4 % a +24,5 % (comprobación en series nuevas, solo informativa), eliminó las semillas malas y se presentó como la mejora más prometedora, aunque no superó el margen de ruido (T2 D-2.5). Con estado de mercado (idea A), guardar también las constantes de normalización junto a los pesos (T2 `notebook_partida.md` §9, trampa 9).
- **Recuperar**: cargar el mejor *checkpoint* para la evaluación final y, si el entrenamiento empeora (caída brusca de la métrica), volver al último bueno y subir ε (enlaza con X1).
- **Demostrarlo**: interrumpir un entrenamiento a mitad, reanudarlo desde el *checkpoint* y enseñar que la curva continúa; y una figura con la métrica, los puntos de guardado y el punto de parada.

**Nota:** es la idea más barata y la que más se apoya en lo ya construido para el ej. 3; combinada con este, responde también a la frase del ENUNCIADO de que "la duración del entrenamiento debería estar condicionada a la evolución del aprendizaje" (ej. 3).

### 8.5 Otras ideas de clase, por si se prefiere

| Idea | Fuente | Coste | Comentario |
|---|---|---|---|
| ε por estado / dentro del episodio (X3) | T07 ~00:37-00:39; APP p.73 | Medio | Prolonga el ej. 3 |
| Tamaño de *buffer* dinámico según el error | APP p.78; T06 ~04:11-04:13; T07 ~04:31 ("el váter", casi seguro el *buffer*, *(dudoso)*) | Medio | Ampliar el *buffer* cuando el error TD es alto y reducirlo cuando es bajo |
| γ por episodio, $\gamma=f^{1/N}$ | T07 ~00:41 *(dudoso: "Karaoma")*; [03 §12.2](03_recompensas_y_descuento.md) | Bajo | Neutraliza la duración variable (APP p.75) |
| *Step* de $k$ *bins* (p. ej. 1 minuto) | CASOS p.34; T07 ~03:09-03:10; [02 §4.4](02_diseno_de_entornos.md) | Alto | Menos decisiones por episodio (33-50), γ y *buffer* más manejables; cambia acciones y `vol_care` |
| Precio de ejecución más realista | T07 ~05:01-05:05; [08 §6.7](08_entorno_best_execution.md) | Bajo-medio | *Mid* medio entre fotos o interpolación por latencia; o `executions.pkl` (REP 2019) |
| Variantes del agente (Dueling, objetivo esperado, Huber, PER) | T07 ~00:25; APP p.80-81; [01](01_aproximacion_funcional_y_dqn.md); [09 §13](09_material_rl_notebooks.md) | Medio | Poco que ganar con $[T,V]$; mejor con la idea A. En el Taller 2, duelo, quitar la *target*, *clipping* y recompensa log no superaron el margen, y sí lo superó ampliar el estado: "representación antes que arquitectura" (T2 D-3.2). Con R2 en $[-1,0]$, el *reward clipping* a ±1 no hace nada y Huber con δ = 1 es casi MSE (T2 `notebook_partida.md` §9, trampa 8) |
| Pruebas de estrés de variables privadas (empezar retrasado, poco tiempo para mucho volumen) y **fuera del rango de entrenamiento** (p. ej. $N=300$ o $700$, $\rho=0{,}05$ o $0{,}15$) | CASOS p.51 ("alterar los tamaños de las órdenes fuera de los parámetros fijados"); [07 §12](07_rl_para_ejecucion.md) | Bajo | Demuestra que el agente se recupera y que aprendió la regla sobre $(T,V)$ y no un calendario ligado al rango; refuerza el requisito del ej. 2. Hace falta un `reset` con parámetros forzados que se salte los *asserts* de (1) (LABSTEP celda 6), y hay que declararlo. Con R2 y estado $[T,V]$ se espera que generalice; si no lo hace, es un hallazgo que se puede exponer |
| Que el agente decida otra dimensión: (a) momento dentro de cada ventana de 10 min de un TWAP, o (c) tamaño de la hija (mitad, igual, +50 % o el doble) | T07 ~03:03-03:06; [07 §5, §8.4](07_rl_para_ejecucion.md); [02 §6.5](02_diseno_de_entornos.md) | Medio-alto | **Nota:** con impacto 0 el tamaño no tiene coste, así que (c) solo tiene sentido con una penalización o un impacto (fila siguiente); (a) acota la libertad como el perfil conservador. Es la ampliación de acciones que §2 deja fuera de los ej. 1-3 |
| Impacto de mercado simple (p. ej. el precio empeora si la hija consume más que `vask1`, o un impacto temporal proporcional a las hijas recientes) | T07 ~04:56 (quien lo haga distinto, con funciones de impacto: "os pone la matrícula", *(dudoso)*); [02 §5.6](02_diseno_de_entornos.md) (el impacto también es interno); [07 §4.2](07_rl_para_ejecucion.md) (el triciclo lo evita) | Medio | Solo como ej. 4, porque el ej. 2 exige impacto 0 (ENUNCIADO). Combina bien con la idea A o con tamaños variables; hay que declarar que el modelo de impacto es una asunción no calibrada |
| Episodios de 2-3 h | T07 ~04:46-04:47; E21 ([08 §15](08_entorno_best_execution.md)) | Bajo | Hay que recalcular γ ($\gamma=f^{1/N}$ con $N$ ≈ 1.440-2.160 *bins* de 5 s) y el *buffer* en episodios, y decirlo |
| Recompensa "rara" y analizar el comportamiento | T07 ~04:29 *(dudoso)* | Bajo | Válido si se motiva y se enseña el comportamiento |

---

## 9. Mapa de evidencias

| Ej. | Gráfico o tabla | Qué demuestra | Fuente de la idea |
|---|---|---|---|
| 1 | Esquema de llamadas y salida de las pruebas de invariantes | Elementos del entorno y orden correcto (ENUNCIADO ej. 1) | [08 §5](08_entorno_best_execution.md) |
| 1 | `plot_results` del TWAP determinista | Entorno correcto y *benchmark* | LAB celdas 8-9 |
| 1 | `plot_results` del agente y curva de la celda 21 | Entrenamiento y ejecución completos | LAB celdas 18-21 |
| 2 | Rejilla de 4-6 `plot_results` con $N$, $n$, día y hora distintos | Comportamiento similar a la figura de referencia y requisito de variedad | ENUNCIADO ej. 2; BE `images/twap_model/` |
| 2 | Trayectorias $V$ frente a $T$ y mapa de la política *greedy* | Que el agente aprendió la regla | Nota (§6.7) |
| 2 | Tabla de métricas frente al TWAP determinista (media y dispersión) | Calidad de la réplica | §4.3 |
| 2 | Contraejemplos R0 y R4 | Por qué esta recompensa y no otra | T06 ~02:34-02:35 |
| 2 | Tabla de parametría con motivos y barrido de γ y *buffer* | Segunda parte | ENUNCIADO ej. 2; §6.4-6.6 |
| 3 | ε y métrica *greedy* frente a episodios | Respuestas a las dos preguntas | ENUNCIADO ej. 3 |
| 3 | Comparación lineal frente a mejora | Qué aporta el criterio | §7.4 |
| 3 | Mapa de calor de visitas | Dimensión "conocer el espacio" | T07 ~00:40 |
| 4 | Histograma de $t_{\text{fin}}/N$ y de pendiente (B); curva con *checkpoints* y reanudación (C); *slippage* frente al TWAP con dispersión (A) | Cada idea | §8 |

---

## 10. Riesgos y trampas

| # | Riesgo | Consecuencia | Prevención | Fuente |
|---|---|---|---|---|
| 1 | No se puede instanciar el entorno hasta completar (1)-(3) | Bloqueo al empezar | Empezar por el LABSTEP | [08 §6.14](08_entorno_best_execution.md) |
| 2 | `mse` indefinido, `learn()` extra en el *log*, redondeos que imprimen 0 | `NameError` o *logs* engañosos | Corregir E4 | [08 §11](08_entorno_best_execution.md) |
| 3 | `agent.env` no reasignado en evaluación | Sin compra forzada; NaN | `agent.env = env` (E5) | [08 §12.1](08_entorno_best_execution.md) |
| 4 | `action_sample` con $p=0{,}5$ y *buffer* lleno de aleatorias | Episodios cortos; sin estados de final de episodio | $p\approx\rho$; llenado parcial | §6.5 |
| 5 | Recompensa por defecto | Óptimo = no comprar | R2 | §6.3 |
| 6 | Medir $d$ antes del incremento de `state_pos` sin darse cuenta | Recompensa desfasada un paso | Decidir y documentar el instante | §6.3 |
| 7 | Artefacto de acelerar al final con R2 | Racimo de compras finales | Término terminal (remedio a) | §6.3 |
| 8 | Recompensa terminal fuera de `reward_hist` | *Logs* que no cuadran | Registrar aparte | [03 §7](03_recompensas_y_descuento.md) |
| 9 | Comparaciones en coma flotante en el TWAP determinista | Compras retrasadas un paso | Enteros | [08 §7](08_entorno_best_execution.md) |
| 10 | `plot_results` superpone figuras | Gráficos ilegibles | Figura nueva por llamada | [08 §12.3](08_entorno_best_execution.md) |
| 11 | Pickles y versiones de pandas | Error al cargar | §3 | [08 §3.1](08_entorno_best_execution.md) |
| 12 | VWAP de mercado al *ask* en el gráfico y al *mid* en el entorno | *Slippage* distinto según dónde se mire | Una sola definición | E3 |
| 13 | `episode_vwap` (futuro) en el estado | Fuga de información | Solo en la recompensa, y aun así no factible | [02 §7.2](02_diseno_de_entornos.md) |
| 14 | `_detect_num_feat` sin (4) al ampliar el estado | El constructor falla | Llamar a (4) o calcular desde `episode_full` | E15 |
| 15 | Una sola semilla | Conclusiones por azar (3 de 10 fallan) | 3-5 semillas | T06 ~01:13 |
| 16 | Generador aleatorio compartido entre entorno y agente | Experimentos no comparables al cambiar ε | Generadores propios | T2 `notebook_partida.md` §9 |
| 17 | Volver a ejecutar la celda que crea el agente | Perder el entrenamiento | *Checkpoints* (idea C) | T2 `notebook_partida.md` §9, trampa 14 |
| 18 | Quedarse sin tiempo de cómputo | Sin barridos ni semillas | PyTorch, `learn_after` razonable, entorno con arrays de NumPy, medir antes | §3.2 |
| 19 | Multiproceso con funciones definidas en celdas (Windows) | Errores al paralelizar semillas | Código en `common/` | T2 `README.md`; T2 `notebook_partida.md` §9 |
| 20 | Sobreinterpretar resultados con impacto 0 | Pregunta incómoda en la exposición | Decir los límites del simulador | T07 ~03:56-03:57 |
| 21 | Trabajar con una versión vieja del enunciado | Requisitos cambiados | Comprobar Drive/OneDrive | T07 ~04:27 |
| 22 | Repositorio público con los datos y el código del taller: pickles (~139 MB de datos de mercado) y ficheros derivados de BE | Posible problema de licencia de los datos *(no verificado)*; código LGPL 2.1 mezclado con código MIT | Confirmarlo antes de publicar; cabeceras LGPL en los ficheros derivados (§4.2) | E28 ([08 §15](08_entorno_best_execution.md)); BE `LICENSE`; `README.md` de este repo |
| 23 | `actions_hist` vacío (E7) | Sin acciones para medir la separación entre compras ni para graficar | `self.actions_hist.append(action)` en `step` (bloque modificado que hay que declarar) | [08 §15](08_entorno_best_execution.md) |
| 24 | VWAP parcial NaN en el primer paso (E12) | Recompensa local de la idea A con NaN | Rellenar el primer precio | [08 §15](08_entorno_best_execution.md) |
| 25 | Volumen del primer *bin* perdido (E11) | VWAP de mercado ligeramente sesgado | Calcular el primer `diff` con `episode_full` | [08 §15](08_entorno_best_execution.md) |
| 26 | El DQN no converge de forma estable en el plazo | Ej. 2 sin réplica | Plan B aceptado en clase: Q-learning tabular como solución principal, refactorizando el agente (T07 ~05:01), con el DQN como intento documentado | [04 §12](04_exploracion_memoria_y_estabilidad.md) |

---

## 11. Cómo motivar cada decisión

**Idea.** Lo que preguntará el profesor es "por qué esto y no lo otro" (T06 ~04:10). Una motivación completa tiene cinco piezas.

**Plantilla** (Nota, la de los registros de T1 y T2):

1. **Contexto**: qué problema resuelve la decisión.
2. **Alternativas** consideradas (al menos dos).
3. **Argumento de clase**: la cita (sigla y hora o página) que la apoya.
4. **Evidencia propia**: un cálculo o una figura.
5. **Límite**: qué no demuestra y cuándo habría que revisarla.

**Ejemplos ya armados con este documento:**

| Decisión | Alternativas | Argumento de clase | Evidencia | Límite |
|---|---|---|---|---|
| Recompensa R2 | R0, R1, R4 | Factibilidad (T07 ~03:37); "ir de la mano" (T07 ~04:52); sin rendimiento (T07 ~04:58) | γ = 0 ya da un TWAP "redondeado" (que la red aproxima, sin conocer $n$); contraejemplos R0/R4 (§6.3) | Ligero incentivo a acelerar al final |
| γ = 0,99 | 0,9 / 0,999 / 1 | Calibrar hacia atrás (T06 ~03:50-03:52) | Tabla de §6.4; barrido | Si se añade un término terminal, subir |
| *Buffer* 50.000 | 10-20k / 100-250k | "Mis 100 últimas partidas" (T06 ~04:02); 100 sesiones (T07 ~03:47) | 100 × ~450-500 *steps*; barrido | Con estado de mercado, más |
| `action_sample` $p\approx\rho$ | 0,5 | Pesos no uniformes (APP p.57; CASOS p.45) | Tabla de duración de episodios (§6.5) | Deja algo de volumen pendiente si $p=\rho$ exacto |
| Llenado de 20-30 episodios | Entero (LAB) | T06 ~04:20-04:21, ~04:29-04:30 | Curva *greedy*; precedente de T2 D-2.2 (§6.5) | — |
| ε por mejora (M1+M2, X1) | Lineal; X2; X3 | APP p.64, p.66; T06 ~03:17-03:21; T07 ~01:47 | Gráfico doble; comparación (§7.4) | Depende del margen y del coste de evaluar |

---

## 12. Exposición y memoria

**Lo que se sabe:** exposición el día 18, de unos 10-15 minutos por grupo, con orden publicado en OneDrive (T07 ~05:06), seguible por Zoom (T06 ~04:55-04:56); diapositivas probablemente no obligatorias ("no hace falta positivas", *(dudoso)*) (T07 ~05:06; [notebook_partida §11](notebook_partida.md)). La memoria separada es opcional, pero "en ocasiones recomendable" (ENUNCIADO).

**Guion propuesto** (Nota, 12-13 minutos):

| Min. | Bloque | Contenido | Apoyo |
|---|---|---|---|
| 1 | Problema | Orden CARE de compra, hijas de 1 título, impacto 0, solo decide el momento; métrica | [07 §2, §5](07_rl_para_ejecucion.md) |
| 2 | Ej. 1 | Entorno: `reset` → `act` → `step`; asunciones de precio; TWAP determinista | Esquema + figura |
| 4 | Ej. 2 | Estado; por qué R2 (contraejemplos); γ y *buffer* con su cálculo; rejilla de episodios y mapa de política | §6.7 |
| 2-3 | Ej. 3 | Cómo medimos la mejora y cómo pasa a ε; comparación con el lineal | §7.5 |
| 2 | Ej. 4 | B y C (y A): qué se recogió de clase, razonamiento, qué se implementó, resultado | §8 |
| 1 | Límites | Lo que este simulador no permite concluir y qué haríamos después | T07 ~03:56-03:57 |

**Propuesta:** 6-8 diapositivas con las figuras del notebook (reutilizando el sistema HTML → PDF del Taller 2, `build_presentacion.py`, T2 `README.md`, o algo más simple), una memoria breve en Markdown → PDF con la tabla de decisiones (reutilizando `build_informe.py` sin el límite de páginas, T1 D-3.1; T2 D-0.5) y los notebooks ejecutados como entregable principal. Como en el Taller 2 (6 diapositivas + anexos A-F, T2 `README.md`), preparar 3-5 **anexos** que no se presentan, con la evidencia de cada decisión de §11 (tabla de γ, barrido de *buffer*, contraejemplos R0/R4, comparación de ε, tiempos medidos), para responder a "por qué esto y no lo otro" sin improvisar. Ensayar con reloj y repartir bloques. El profesor preguntará el porqué de cada elección (T06 ~04:10); **Nota:** conviene que cualquier miembro pueda responder.

---

## 13. Calendario orientativo

**Nota:** hoy es el martes 10/11/2026 y la exposición, el miércoles 18/11 ([notebook_partida §11](notebook_partida.md)). Ocho días.

| Días | Objetivo | Salida |
|---|---|---|
| 10-11/11 | Entorno de software (§3), carga de datos, LABSTEP completo, pruebas de invariantes, TWAP determinista. **Logística:** comprobar que se envió el correo (al del trabajo) con el nombre del grupo y la preferencia de hora, porque el orden se asigna por llegada de los correos; consultar la lista en OneDrive (quien no figure va al final); preguntar en ese correo cómo y cuándo se entrega el notebook y si hay una versión nueva del enunciado (T06 ~04:55; T07 ~04:27, ~05:05-05:07; [notebook_partida §11](notebook_partida.md)) | Ej. 1 casi listo; D-0.1, D-1.1, D-1.2 |
| 12/11 | Agente (PyTorch o TF), bucle completo, correcciones E4/E5, medición de tiempos | Ej. 1 ejecutado de principio a fin |
| 13-14/11 | Ej. 2: R2, conjunto de evaluación, figuras, barrido de γ y *buffer* | Ej. 2 con evidencias; D-1.3, D-1.4, D-2.2-D-2.4 |
| 15/11 | Ej. 3: métrica, regla de ε, comparación | Ej. 3; D-2.5 |
| 16/11 | Ej. 4: C (sale casi gratis del ej. 3) y B; A si sobra tiempo | Ej. 4; D-3.x |
| 17/11 | Semillas finales, memoria, diapositivas, ensayo | Entregables |
| 18/11 | Exposición | — |

Si algo se retrasa, se recorta por el final (A, luego B), nunca por el ej. 1.

**Reparto** (Nota): todo depende del ej. 1, pero no todo espera a que esté acabado. Mientras una persona completa el entorno (ej. 1), otra puede escribir `protocol.py` (conjunto de evaluación, métricas, M1-M3) y `plots.py` sobre el TWAP determinista, y una tercera el puerto PyTorch del agente con la interfaz de BE. Desde el 13/11, el ej. 2, el ej. 3 y las ideas B y C pueden avanzar en paralelo sobre el mismo `env.py`, con un editor por notebook a la vez y caché de entrenamientos (§4.1).

---

## 14. Checklist final

| Bloque | Comprobación | Dónde |
|---|---|---|
| Logística | Correo al profesor (del trabajo) con nombre del grupo y preferencia de hora; hora asignada consultada en OneDrive; Zoom a mano | [notebook_partida §11](notebook_partida.md) |
| Logística | Preguntado cómo y cuándo se entrega el notebook; versión del enunciado comprobada | [notebook_partida §11-12](notebook_partida.md) |
| General | Notebooks **ejecutados**, uno por ejercicio (o secciones), con la celda "qué cambia" y los bloques modificados señalados | ENUNCIADO; §4.2 |
| General | `DECISIONES.md` con las decisiones de §4.4 cerradas o abiertas, con evidencia | §4.4 |
| General | Entorno de software documentado (Python, *framework*, pickles usados) | §3 |
| General | Activo (REP o SAN) y *split* usados, dichos y justificados | ENUNCIADO; T07 ~04:39; §5.3 |
| General | Ficheros derivados de BE con su aviso LGPL 2.1 | §4.2 |
| Ej. 1 | 13 funciones completas; pruebas de invariantes en verde; TWAP determinista graficado | §5.2 |
| Ej. 1 | Cambio a PyTorch (o de algoritmo) explicado y bloques del agente señalados; si el entorno se reescribe, documentación adicional de estructura y pasos | ENUNCIADO ej. 1; §3.2, §4.2 |
| Ej. 1-3 | Curva de la celda 21 incluida y comentada como curva con exploración | LAB celda 21; §5.4 |
| Ej. 1 | E4 y E5 corregidas; ε = 1 o justificado en el llenado; paso 0 forzado tenido en cuenta | §5.2 |
| Ej. 1 | Entrenamiento hasta `nepisodes` sin errores y `plot_results` final | LAB celdas 18-21 |
| Ej. 2 | Estado explicado (y por qué no hacen falta características de mercado) | §6.2 |
| Ej. 2 | Recompensa explicada con su porqué y contraejemplos | §6.3 |
| Ej. 2 | Rejilla de episodios en *train* con $N$ y $n$ distintos; trayectorias; mapa de política; tabla de métricas | §6.7 |
| Ej. 2 | γ, *buffer*, llenado, `p`, `learn_after`, α motivados; barrido de γ y *buffer* | §6.4-6.6 |
| Ej. 3 | ε independiente del número de episodios; respuestas escritas a las dos preguntas | §7 |
| Ej. 3 | Gráfico de ε y métrica; comparación con el lineal | §7.4-7.5 |
| Ej. 4 | Para cada idea: qué se recogió de clase, razonamiento, implementación, evidencia | §8 |
| Ej. 4 | B: volumen completo, banda de finalización elegida y justificada, estabilidad medida. C: guardado, parada y reanudación demostrados | §8.3-8.4 |
| Rigor | 3-5 semillas en lo que se expone; media y dispersión; una sola definición de precio | §4.3 |
| Rigor | Límites del simulador dichos (impacto 0, precio al *ask* al inicio del *bin*, sin costes) | §6.8, §10 |
| Exposición | Guion de 10-15 min ensayado; todos pueden defender cualquier decisión | §12 |

---

## 15. Dudas abiertas

1. **Puntos del ej. 4** (1,5 en el ENUNCIADO frente a "el otro punto" en clase): se asume 1,5 ([notebook_partida §12](notebook_partida.md), fila 1).
2. **Idea B: tiempo o volumen al 90 %**: se asume tiempo, por el ENUNCIADO; la banda (80-95 % o 70-90 %) queda a elección justificada.
3. **Los "3 ficheros de datos"** del ENUNCIADO: probablemente `rep_data`, `san_data` y `orderbook`; no afecta si se usa REP o SAN.
4. **¿Es obligatorio el TWAP determinista de la celda 9?** "Esto es importante. Esto no es necesario" (T07 ~04:53) es ambiguo; se propone implementarlo igualmente.
5. **Canal y fecha de entrega del notebook**: no los dice el ENUNCIADO.
6. **Si la exposición necesita diapositivas**: "no hace falta positivas" *(dudoso)*.
7. **Si los pickles de `new_version/` cargan con pandas ≥ 2.2** (opción C de §3): no verificado.
8. **Rendimiento real de Keras frente a PyTorch en este equipo**: las cifras de §3.2 son del Taller 2; hay que medir.
9. **Si las conclusiones de las simulaciones de §6.3 y §6.5 se mantienen con el agente y los datos reales**: son cálculos sin datos con la mecánica del LAB; hay que confirmarlas en el ej. 2.
10. **Publicación de los datos del taller en un repositorio público** (§10, fila 22).
11. **Peso de la práctica en la asignatura** ("3 puntos" *(dudoso)*, T07 ~05:09): no cambia el plan.

Términos técnicos en [11_glosario.md](11_glosario.md).
