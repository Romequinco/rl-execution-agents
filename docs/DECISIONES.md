# Registro de decisiones del proyecto

Formato ligero tipo ADR, como en los Talleres 1 y 2. ID `D-X.n`: `X` es el bloque (0 general, 1 entorno y ej. 1, 2 ej. 2,
3 ej. 3, 4 ej. 4) y `n` un correlativo. El estado es `ABIERTA` (pendiente de confirmar con los datos reales o con el equipo)
o `CERRADA`. Las decisiones se añaden al final de su bloque. La base teórica de cada una está en
[`teoria/10_hoja_de_ruta_practica.md`](teoria/10_hoja_de_ruta_practica.md).

---

## Bloque 0 · General

### D-0.1 · PyTorch en lugar del TensorFlow del taller

- **Fecha:** 2026-11-11
- **Decisión:** `DDQNAgent` se porta a PyTorch en CPU (`notebooks/common/agents.py`) con la misma interfaz, los mismos
  hiperparámetros por defecto, Combined Experience Replay y el objetivo Double DQN.
- **Justificación:** el enunciado lo permite («incluye el uso de PyTorch en vez de TensorFlow»). TF 2.15 solo instala en
  Python 3.9-3.11 y el equipo trabaja con 3.13; en el Taller 2 (T2 D-0.2) evaluar un estado costaba 0,3 ms en PyTorch frente
  a 10,8 ms en Keras. Aquí 200 episodios de ~500 pasos entrenan en unos 10 s.
- **Estado:** CERRADA

### D-0.2 · Activo: Repsol

- **Fecha:** 2026-11-11
- **Decisión:** se trabaja con `rep_data.pkl` (carpeta `new_version/`, compatible con pandas 2).
- **Justificación:** el enunciado admite cualquiera de los dos activos; `orderbook.pkl` y `executions.pkl` son de Repsol, así
  que la idea A del ej. 4 puede usar el libro a 3 niveles sin cambiar de activo.
- **Estado:** CERRADA

### D-0.3 · Un entorno con opciones, código en `common/` y rejilla fija de evaluación

- **Fecha:** 2026-11-11
- **Decisión:** una sola `BestExecutionEnv` (`common/env.py`) con las 13 funciones del LAB; la recompensa se inyecta desde
  `common/rewards.py`. Los notebooks importan el código y declaran en su primera celda qué bloques cambian. La evaluación
  usa siempre la misma rejilla: N ∈ {400, 500, 600} × ρ ∈ {0,075; 0,10; 0,125} × 3 días de train con hora de inicio al azar
  (semilla 0), 27 episodios.
- **Justificación:** el enunciado permite ficheros `.py` y pide especificar los bloques modificados; una copia del entorno
  por ejercicio haría difícil ver qué cambia. La rejilla demuestra con cifras el requisito del ej. 2 (distintos días,
  horas, duraciones y tamaños) y da a la métrica del ej. 3 episodios comparables entre evaluaciones.
- **Estado:** CERRADA

### D-0.4 · Licencias: el código derivado del taller sigue bajo LGPL-2.1

- **Fecha:** 2026-11-11
- **Decisión:** `env.py`, `agents.py` y `plots.plot_results` derivan de best-execution (LGPL-2.1) y lo dicen en su cabecera;
  el resto del código propio es MIT.
- **Estado:** CERRADA

### D-0.5 · Pruebas sobre los datos reales

- **Fecha:** 2026-11-11
- **Decisión:** `common/test_env.py` (18 pruebas) usa los días de train de REP. Nada de datos sintéticos.
- **Justificación:** las invariantes se comprueban sobre los mismos datos que usan los notebooks. Los datos cumplen lo
  supuesto: claves `datetime.date`, una fila cada 5 s de 13:00 a 17:25 (3.177-3.181 filas), sin NaN y `cumvol` nunca
  decreciente; spread mediano 3,3 pb (un tick de 0,005 € sobre ~15 €).
- **Estado:** CERRADA

---

## Bloque 1 · Entorno y ej. 1

### D-1.1 · Llenado parcial del buffer: 25 episodios aleatorios

- **Fecha:** 2026-11-11
- **Decisión:** antes de aprender se llenan 25 episodios (~12.000 pasos) con acciones aleatorias, no `buffer_size` pasos.
- **Justificación:** recomendación de clase (llenarlo solo en parte, 20-30 episodios) para no anclar al agente en
  experiencias de una política que no es la suya.
- **Estado:** CERRADA

### D-1.2 · Acción aleatoria con p = ρ = n/N

- **Fecha:** 2026-11-11
- **Decisión:** `action_sample` compra con probabilidad igual al ritmo del TWAP.
- **Justificación:** con p = 0,5 la orden se acaba en ~2n pasos (unos 100 de 500) y el agente casi nunca ve la parte final
  del episodio. Con p = ρ la exploración recorre todo el horizonte.
- **Estado:** CERRADA

### D-1.3 · Generadores aleatorios propios de entorno y agente

- **Fecha:** 2026-11-11
- **Decisión:** el entorno y el agente tienen cada uno su `numpy.random.Generator` con semilla.
- **Justificación:** si comparten el global, cambiar la exploración cambia también los episodios y las comparaciones entre
  configuraciones dejan de ser justas (lección del Taller 2).
- **Estado:** CERRADA

### D-1.4 · TWAP determinista: compra si V ≥ T

- **Fecha:** 2026-11-11
- **Decisión:** `TWAPAgent` compra cuando va igual o por detrás del plan (volumen restante ≥ tiempo restante).
- **Justificación:** es la regla que dio el profesor en clase. En la rejilla sigue el plan con error ≤ 1 título y completa
  todas las órdenes; es el benchmark de los ej. 2-4.
- **Estado:** CERRADA

---

## Bloque 2 · Ej. 2

### D-2.1 · Estado [T, V]

- **Fecha:** 2026-11-11
- **Decisión:** el estado del LAB, tiempo y volumen restantes en tanto por uno (`obs_mode="TVd"` añade V − T si hiciera falta).
- **Justificación:** la regla del TWAP es una frontera en ese plano; el ej. 2 no busca precio, así que no hacen falta
  variables de mercado.
- **Estado:** CERRADA

### D-2.2 · Recompensa de seguimiento del plan con cierre

- **Fecha:** 2026-11-12
- **Decisión:** $r_t = -\min(|e_{t+1} - (t+1)\,n/N|,\ c)/c$ con c = 20, más la penalización de cierre de D-2.6
  (`rewards.TrackingCierre`). Control: `rewards.TWAPRule` (±1 según coincida con la regla).
- **Justificación:** local, densa y predecible con [T, V] (la desviación es n·(T − V)), acotada en [−1, 0] y sin decir qué
  acción tomar. La recompensa por defecto es un slippage de ~1e-4 por compra frente a un VWAP futuro: en la rejilla paga más
  a «todo al final» (+0,020) que al TWAP (−0,009). **Evidencia (REP train, rejilla de 27, semilla 0):** error máximo 0,90
  títulos (TWAP determinista 0,90), error medio 0,27 (TWAP 0,44), 27/27 órdenes completas, acaba al 98,9 % del tiempo,
  intervalo entre compras 1,01 ± 0,07 veces N/n. Control `TWAPRule`: 0,91.
- **Estado:** CERRADA

### D-2.3 · γ = 0,9

- **Fecha:** 2026-11-12
- **Decisión:** γ = 0,9.
- **Justificación:** horizonte efectivo de ~10 pasos, la separación entre compras. **Evidencia (barridos con 3 semillas,
  `results/ej2_sweeps_etapa*.csv`):** con c = 20, γ = 0,9 → 0,91; 0,95 → 1,18; 0,99 → 4,57. Con c = 5 y sin cierre, γ = 0,99
  acaba la orden al 17 % del tiempo (atajo, ver D-2.6); con cierre usa el 100 % pero se queda en 21 títulos: con Q-valores
  de ~100 pasos la diferencia entre acciones queda enterrada.
- **Estado:** CERRADA

### D-2.4 · Aprender cada 8 pasos y copiar la red objetivo cada 1.000

- **Fecha:** 2026-11-12
- **Decisión:** `learn_after = 8`, `update_target_after = 1000` (el LAB usa 64 y 640).
- **Justificación:** con 64 hay ~8 actualizaciones por episodio. **Evidencia:** 64/640 da 15,8 (c = 5) y 6,3 (c = 20) títulos,
  sin ninguna semilla por debajo de 2 ni órdenes completas; 8/1.000 da 0,91 con c = 20.
- **Estado:** CERRADA

### D-2.5 · Buffer de 10.000 experiencias

- **Fecha:** 2026-11-12
- **Decisión:** 10.000 (~20 episodios), en lugar de los 50.000 de partida (regla de ~100 episodios).
- **Justificación:** **evidencia:** con c = 5, 10.000 → 0,90 en las 3 semillas frente a 3,04 de media; con c = 20, 0,90 en
  las 3 frente a 0,90-0,94, y baja de 2 títulos antes (episodio 75 frente a 92). Olvida antes el llenado aleatorio; el
  problema es estacionario (la política óptima no depende del día), así que el riesgo de olvido catastrófico es bajo.
- **Estado:** CERRADA

### D-2.6 · Penalización de cierre

- **Fecha:** 2026-11-12
- **Decisión:** si la orden se completa en t_f < N, `done` añade $\sum_{k=1}^{N-t_f} \gamma^k\, r(e_{t_f}, t_f + k)$, los
  pasos que faltan como si el agente se quedara quieto, descontados con su γ.
- **Justificación:** al completar la orden el episodio acaba y deja de penalizar: con c = 5 y sin descontar, «todo al
  principio» cobra −47,6 frente a −43,0 del TWAP pese a un error de 44,7 títulos. Con el cierre, el retorno descontado es el
  de un episodio de N pasos, como la métrica. Evidencia en D-2.3 (γ = 0,99: 17 % → 100 % del tiempo; γ = 0,95: 12,5 → 4,0).
- **Estado:** CERRADA

### D-2.7 · Recorte c = 20

- **Fecha:** 2026-11-12
- **Decisión:** c = 20 títulos (partida: 5).
- **Justificación:** con c pequeño la recompensa satura lejos del plan y no indica cómo volver. **Evidencia (c = 5 de
  partida, 3 semillas):** c = 1 → 28,0; c = 5 → 3,04 (0,90 / 1,33 / 6,87, solo el 33 % de órdenes completas); c = 20 → 0,91 en
  las 3 semillas, por debajo de 2 títulos hacia el episodio 92. La política aprendida coincide con V = T en todo el plano.
- **Estado:** CERRADA

---

## Bloque 3 · Ej. 3

### D-3.1 · Métrica de mejora: error máximo de seguimiento en evaluación greedy

- **Fecha:** 2026-11-12
- **Decisión:** cada 25 episodios se evalúa la política greedy (sin ε ni aprendizaje) en los 27 episodios fijos de la
  rejilla; E = error máximo de seguimiento medio, en títulos. Se registran además % completadas, títulos pendientes, t_fin y
  slippage (informativo).
- **Justificación:** la recompensa de entrenamiento va pegada a ε (correlación −0,85 con la config. final): mezcla política,
  exploración y episodios de distinto N y n (T07 ~00:20-00:22). El slippage no separa políticas: la aleatoria gana al TWAP
  en 11 de 27 episodios y completa el 48 %. El error medio diluye no terminar: una política que deja de comprar en el último
  10 % da 0,63 de error medio (TWAP 0,44) y 4,78 de máximo (TWAP 0,90). La evaluación es determinista (mismos pesos → mismo
  E), así que un salto de E es un cambio real de la política.
- **Estado:** CERRADA

### D-3.2 · Progreso entre E0 (primera evaluación) y E* (TWAP determinista)

- **Fecha:** 2026-11-12
- **Decisión:** $p = \mathrm{clip}((E_0 - E)/(E_0 - E^*), 0, 1)$ con E0 = primera evaluación del propio agente y E* = 0,90
  (TWAP determinista en la rejilla). Alternativas: E0 = aleatoria (7,18) o «no comprar» (49,89, fijo).
- **Justificación:** nivel de aspiración con un benchmark clásico (APP p.66): dice a qué distancia está el agente, no solo si
  mejora. La aleatoria como E0 deja p = 0 aunque el agente aprenda. **Riesgo confirmado:** E0 varía mucho entre semillas
  (config. final: 6,4-15,0; config. inestable: 29,9-48,9) y la primera evaluación da p = 0 por construcción, lo que mantiene
  ε = 1 al menos 50 episodios.
- **Revisión (punto 5):** se sustituye por **E0 fijo = 49,89** (política «no comprar», = n medio de la rejilla) y **progreso
  en escala logarítmica**, $p = \mathrm{clip}((\ln E_0 - \ln E)/(\ln E_0 - \ln E^*), 0, 1)$
  (`ImprovementEpsilon(base_error=…, log_scale=True)`). El error se mueve en órdenes de magnitud: con escala lineal el tramo
  fácil 50 → 7 títulos ocupa el 88 % de la escala y ε baja demasiado pronto; en log cada división del error por dos suma
  0,17. Evidencia en D-3.4.
- **Estado:** CERRADA

### D-3.3 · Traslado a ε: mejor progreso, ε_min y regla de subida

- **Fecha:** 2026-11-12
- **Decisión:** $\varepsilon = 1 - 0{,}95\,p_{ref}$, con $p_{ref}$ el mejor progreso alcanzado y ε_min = 0,05. Si p cae más de
  0,1 bajo $p_{ref}$ en 2 evaluaciones seguidas, $p_{ref}$ ← p actual (ε sube hasta lo que corresponde, no a 1)
  (`exploration.ImprovementEpsilon`; corregido para que la paciencia cuente evaluaciones realmente seguidas).
- **Justificación:** con el último progreso ε da bandazos (0,52 → 0,89 → 0,18 en la serie inestable). Volver a 1 llenaría el
  buffer de experiencias aleatorias y haría olvidar. ε nunca a 0 en entrenamiento (T06 ~03:02). Sensibilidad margen ×
  paciencia: con paciencia 1 se reacciona al bache pasajero del episodio 200; 0,1 × 2 reacciona solo al empeoramiento
  sostenido.
- **Estado:** CERRADA

### D-3.4 · Comparación justa con el ε lineal en dos escenarios

- **Fecha:** 2026-11-12
- **Decisión:** solo cambia el plan de ε; igual configuración, llenado del buffer, secuencia de episodios, ε inicial = 1,
  ε_min, 300 episodios y evaluación cada 25 (también para el lineal). 3 semillas. Escenario A = config. final del ej. 2;
  B = config. de partida (`Tracking` c = 5, buffer 50.000), inestable.
- **Evidencia (E final / área bajo la curva, media de 3 semillas):** A: lineal 0,90 ± 0,00 / 1,62; mejora 0,96 ± 0,09 / 3,44
  (≤ 2 títulos en el episodio 75 frente al 92). B: lineal 5,65 ± 8,18 / 22,3; mejora 1,07 ± 0,28 / 13,1 (3/3 semillas ≤ 2
  frente a 2/3). ε por mejora ayuda cuando el aprendizaje es inestable y es algo más lento cuando el agente aprende rápido,
  por el arranque con ε = 1 que impone E0 (D-3.2).
- **Variantes de E0 (punto 5, 3 semillas; E final / área / episodio con E ≤ 2):** A: c (E0 fijo, lineal) 0,92 / 2,05 / 83;
  d (E0 fijo, log) 0,96 / 1,93 / **67**. B: c 1,40 / 19,3 / 275; d **0,96** / 13,9 / 242. d es la única entre las dos
  mejores en ambos escenarios; en A ninguna variante supera al lineal (1,62 de área). En B la diferencia entre a y d cae
  dentro del ruido de 3 semillas.
- **Estado:** CERRADA. Variante recomendada: d.

---

## Bloque 4 · Ej. 4

### D-4.1 · Idea B: plan al 90 % y penalización del volumen pendiente

- **Fecha:** 2026-11-11 (cerrada 2026-11-13)
- **Decisión:** `TrackingCierre(γ = 0,9, c = 20, horizon = 0,9, pending_penalty = 1)` (`ej4.make_reward`): el seguimiento del
  ej. 2 contra un TWAP que acaba en 0,9·N, con el cierre y −k·pendiente/n al final. Banda aceptada: 80-95 % del tiempo
  con el 100 % del volumen. El esqueleto inicial usaba `Tracking` sin cierre (atajo de acabar pronto, D-2.6) y se corrigió.
- **Calibración previa (políticas fijas, rejilla de train):** dejar el último 5 % sin comprar empeora el seguimiento en
  6,21 por episodio; el término terminal añade 0,04 (k = 1), 0,41 (k = 10) y 2,07 (k = 50). Con γ = 0,9, el final pesa
  0,007 visto desde 0,9·N.
- **Evidencia (`scripts/ej4_ideaB.py`, 7 configuraciones × 3 semillas, 300 episodios):** plan a N (ej. 2) → t/N = 0,988,
  0 % en banda. Plan a 0,9·N con γ = 0,9 y k ∈ {0, 1, 10, 50} → 100 % completas, t/N = 0,887 ± 0,005, 100 % en banda,
  error máx. 0,889 en las 3 semillas (= TWAP a 0,9·N); k no cambia nada. γ = 0,995 (G3) empeora: error 2,6-3,3 y, con
  k = 10, solo 1/3 semillas completa todo. **Test (una vez):** 3/3 semillas 100 % completas, t/N 0,884-0,889, 100 % en banda.
- **Justificación:** el plan comprimido ya castiga lo pendiente en cada paso del último 10 %; k = 1 deja explícita la
  penalización que pide el enunciado sin coste. La recomendación de γ alto vale para señales terminales dispersas; aquí la
  señal es densa (coherente con D-2.3).
- **Estado:** CERRADA

### D-4.2 · Idea C: parada por aprendizaje, checkpoints y recuperación

- **Fecha:** 2026-11-11 (cerrada 2026-11-13)
- **Decisión:** `ej4.Trainer` en lugar de `protocol.train` (que no se toca por los hashes de los ej. 1-3). Parada por
  (a) error ≤ 1,5 títulos en 2 evaluaciones seguidas, (b) 6 evaluaciones sin mejorar 0,05 (más que las 2 del ε por mejora,
  para no cortar la reexploración) o (c) tope de 600 episodios. Evaluación greedy cada 25 episodios frente al plan de
  0,9·N; en cada una se guarda `*_last.pt` y, con cualquier mejora, `*_best.pt`. El checkpoint guarda el estado completo:
  pesos, optimizador, ε, buffer, generadores (agente, entorno, torch), learn_counter, mejor métrica, paciencia y plan de ε.
  Exploración: ε por mejora, variante d (D-3.4).
- **Problemas corregidos de `protocol.train`:** `best` se reiniciaba al reanudar (se podía sobrescribir `*_best.pt` con un
  modelo peor); learn_counter volvía a 0; no se guardaban buffer, generadores ni el plan de ε; el objetivo valía con una sola
  evaluación.
- **Evidencia (`scripts/ej4_ideaC.py`, 3 semillas, config. de D-4.1):** parada por objetivo en los episodios 125, 125 y 150
  (frente a 300), con error final 0,92 / 0,89 / 0,91 (300 episodios sin parada: 0,889). Reanudación: interrumpida en el
  episodio 60, un agente nuevo con otra semilla carga el checkpoint del 50 y reproduce exactamente la curva, la parada y los
  pesos. La semilla 2 olvida entre los episodios 25 y 75 (3,8 → 14,1 títulos, 0 % completas en el 75) y `*_best.pt` conserva
  el modelo del 25. Tests en `common/test_ej4.py`.
- **Estado:** CERRADA

