# Búsqueda ampliada de parámetros · Ejercicio 1

Fecha: 13/11/2026.

**Resultado: conservar la configuración actual.** El reward original del LAB, con recompensa terminal 0, permanece intacto.

## Procedimiento

- Criba: 19 configuraciones, 2000 episodios y semilla 0; cada alternativa cambia un solo parámetro.
- Confirmación: dos alternativas finalistas y la base; entrenamientos nuevos de 4000 episodios con semillas 0, 1 y 2.
- Epsilon de 1 a 0,02 en 2000 episodios; llenado inicial de 25 episodios; estado [T,V] y compra inicial forzada.
- Evaluación greedy: 27 episodios en la criba; 108 por semilla en la confirmación (27 originales + 81 adicionales, con más días y ventanas de train).
- Selección solo con train. Test no se ha utilizado para elegir parámetros en esta búsqueda.
- Criterio prefijado: completar 324/324 órdenes, mejorar al menos 0,25 pb de media y mejorar en al menos dos semillas. El margen de 0,25 pb es práctico; no es un contraste estadístico.
- Los desempates de la criba se ordenan por nombre. La base se incluye siempre en la confirmación.

## Valores explorados

| Parámetro | Valores |
|---|---|
| `gamma` | 0.9, 0.95, 0.98, 0.99, 0.995, 0.999 |
| `alpha` | 0.0001, 0.0003, 0.001, 0.003 |
| `buffer_size` | 10000, 50000, 100000 |
| `batch_size` | 32, 64, 128 |
| `hidden_neurons` | 32, 64, 128 |
| `learn_after` | 16, 64, 128 |
| `update_target_after` | 320, 640, 1280 |

## Confirmación

| Configuración | Completas | Slippage medio (pb) | Mejora frente a base (pb) | Semillas que mejoran |
|---|---:|---:|---:|---:|
| base | 324/324 | 1.110110 | 0.000000 | 0/3 |
| alpha_1e-4 | 216/324 | No comparable: incompleta | — | 0/3 |
| target_1280 | 324/324 | 1.114245 | 0.004135 | 1/3 |

Mayor slippage es mejor: (VWAP mercado − VWAP algoritmo) / VWAP mercado × 10 000. Esta tabla usa la rejilla ampliada; sus medias no se comparan directamente con las de los 27 episodios originales.

`target_1280` coincide con BuyAll en las tres semillas de esta evaluación. La mejora de aproximadamente 0,004 pb procede de la pequeña diferencia de la semilla 0 de la base y no supera el criterio prefijado.

`alpha=1e-4` completa las órdenes y mejora el precio en la criba de 2000 episodios/semilla 0. A los 4000 episodios esa semilla deja volumen pendiente en los 108 casos; las otras dos completan todo. Se descarta por esa inconsistencia.

## Configuración conservada

| Parámetro | Valor |
|---|---|
| `gamma` | 0.99 |
| `alpha` | 0.001 |
| `buffer_size` | 50000 |
| `batch_size` | 64 |
| `hidden_neurons` | 64 |
| `learn_after` | 64 |
| `update_target_after` | 640 |
| `nepisodes` | 4000 |
| `seed` | 0 |

## Verificación y alcance

Se auditan los 28 entrenamientos (74 000 episodios): presupuestos, resultados finitos, identidad del reward y hashes de código y datos. Las tres semillas de la base y de alpha bajo reproducen exactamente los entrenamientos anteriores. El modelo actual coincide con el confirmado de la base/semilla 0 y el notebook ejecutado sigue válido. Los 18 tests existentes han pasado durante esta búsqueda.

Es una búsqueda de factores individuales con criba en una semilla y confirmación en tres. No se prueban todas las combinaciones ni se demuestra un óptimo global. La completitud observada está limitada a los episodios evaluados. El error de seguimiento TWAP no se utiliza para seleccionar parámetros del ejercicio 1.

Cada ejecución conserva entrenamiento, evaluación y checkpoint. Reproducir desde la raíz:

```powershell
..\.venv\Scripts\python.exe scripts/ej1_search.py
..\.venv\Scripts\python.exe scripts/ej1_search_audit.py
```
