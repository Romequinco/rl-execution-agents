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
- **Decisión:** `common/test_env.py` (16 pruebas) usa los días de train de REP. Nada de datos sintéticos.
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

### D-3.1 · ε por mejora medida en evaluación greedy

- **Fecha:** 2026-11-11
- **Decisión:** cada 25 episodios, evaluación greedy en la rejilla; métrica E = error máximo de seguimiento medio. Progreso
  $p = \mathrm{clip}((E_0 - E)/(E_0 - E^*), 0, 1)$ entre la primera evaluación y el TWAP determinista;
  $\varepsilon = \varepsilon_{max} - (\varepsilon_{max} - \varepsilon_{min})\,p_{ref}$ con $p_{ref}$ el mejor progreso. Si
  empeora más de 0,1 en 2 evaluaciones seguidas, $p_{ref}$ baja y ε sube (`exploration.ImprovementEpsilon`).
- **Justificación:** responde a las dos preguntas del enunciado con una métrica y un benchmark explícitos, como pidió el
  profesor (evaluar sin explorar, en varios episodios, contra un objetivo). **Evidencia (REP train, semilla 0):** baja de 2 títulos en
  el episodio 150, frente al 250 del ε lineal, y termina con 0,95 ± 0,07 frente a 0,97 ± 0,11.
- **Estado:** ABIERTA (comparar con 3 semillas sobre los datos reales)

---

## Bloque 4 · Ej. 4

### D-4.1 · Idea B: plan al 90 % y penalización del volumen pendiente

- **Fecha:** 2026-11-11
- **Decisión:** `Tracking(horizon=0.9, pending_penalty=k)`, con k calibrado en {0, 1, 5}.
- **Estado:** ABIERTA

### D-4.2 · Idea C: parada por aprendizaje, checkpoints y recuperación

- **Fecha:** 2026-11-11
- **Decisión:** parada si el error baja de 1,5 títulos o si no mejora 0,05 en 4 evaluaciones; se guardan `*_best.pt` y
  `*_last.pt` en cada evaluación y se recupera con `DDQNAgent.load` y `start_episode`.
- **Estado:** ABIERTA
