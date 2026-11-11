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

### D-0.5 · Pruebas con datos sintéticos

- **Fecha:** 2026-11-11
- **Decisión:** `common/test_env.py` (16 pruebas) y la prueba de humo de los notebooks usan datos sintéticos con el mismo
  esquema que los pickles (`RL_DATA=synthetic`). Los resultados de `results/` salen siempre de los datos reales.
- **Justificación:** las invariantes del entorno no dependen de los precios, y así se pueden comprobar sin deserializar
  pickles de terceros.
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

### D-2.2 · Recompensa de seguimiento del plan

- **Fecha:** 2026-11-11
- **Decisión:** $r_t = -\min(|e_{t+1} - (t+1)\,n/N|,\ 5)/5$ (`rewards.Tracking`). Control: `rewards.TWAPRule` (±1 según
  coincida con la regla).
- **Justificación:** local y factible (depende solo del estado y la acción), acotada en [−1, 0] y sin decir qué acción
  tomar. La recompensa por defecto mide el slippage frente al VWAP de todo el episodio, que el agente no puede anticipar.
- **Estado:** ABIERTA (confirmar con los datos reales)

### D-2.3 · γ = 0,9

- **Fecha:** 2026-11-11
- **Decisión:** γ = 0,9, con barrido {0,9; 0,95; 0,99} × 3 semillas en el notebook.
- **Justificación:** una compra afecta al error hasta la siguiente, unos N/n ≈ 10 pasos; con γ = 0,9 el horizonte efectivo
  es de 10 pasos. **Evidencia preliminar (datos sintéticos, 300 episodios, 1 semilla):** γ = 0,9 replica el TWAP (error
  máximo medio 0,97 títulos, igual que el determinista); γ = 0,95 se queda en 9,8 y γ = 0,99 en más de 20: con ~500 pasos los
  Q-valores son grandes y la diferencia entre acciones queda enterrada.
- **Estado:** ABIERTA (confirmar con el barrido sobre los datos reales)

### D-2.4 · Aprender cada 8 pasos y copiar la red objetivo cada 1.000

- **Fecha:** 2026-11-11
- **Decisión:** `learn_after = 8`, `update_target_after = 1000` (el LAB usa 64 y 640).
- **Justificación:** con 64 hay unas 8 actualizaciones por episodio y en 300 episodios el agente no llega a aprender la
  frontera (evidencia sintética). El barrido {8, 64} queda en el notebook.
- **Estado:** ABIERTA

### D-2.5 · Buffer de 50.000 experiencias

- **Fecha:** 2026-11-11
- **Decisión:** 50.000 ≈ 100 episodios × ~500 pasos, la regla de clase; barrido {10.000, 50.000}.
- **Estado:** ABIERTA

---

## Bloque 3 · Ej. 3

### D-3.1 · ε por mejora medida en evaluación greedy

- **Fecha:** 2026-11-11
- **Decisión:** cada 25 episodios, evaluación greedy en la rejilla; métrica E = error máximo de seguimiento medio. Progreso
  $p = \mathrm{clip}((E_0 - E)/(E_0 - E^*), 0, 1)$ entre la primera evaluación y el TWAP determinista;
  $\varepsilon = \varepsilon_{max} - (\varepsilon_{max} - \varepsilon_{min})\,p_{ref}$ con $p_{ref}$ el mejor progreso. Si
  empeora más de 0,1 en 2 evaluaciones seguidas, $p_{ref}$ baja y ε sube (`exploration.ImprovementEpsilon`).
- **Justificación:** responde a las dos preguntas del enunciado con una métrica y un benchmark explícitos, como pidió el
  profesor (evaluar sin explorar, en varios episodios, contra un objetivo). **Evidencia preliminar (sintética):** alcanza el
  error del TWAP hacia el episodio 200, frente al 250 del ε lineal.
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
