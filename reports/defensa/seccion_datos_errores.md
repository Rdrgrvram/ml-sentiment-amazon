# 3.2, 3.3 y 3.7 Datos, preparación y análisis de errores

## Fuente y subconjunto

**Amazon Reviews 2023 - Electronics, McAuley Lab (UCSD), accedido el 16/10/2026.** Se leyeron por streaming
aproximadamente 1,5 millones de reseñas. Tras eliminar textos de menos de 20 caracteres y duplicados, se tomó
con semilla 42 una muestra balanceada de 49 998 observaciones: 16 666 por clase.

## Variables

| Variable | Uso y significado |
|---|---|
| review | Título y texto; entrada principal del modelo. |
| label | Sentimiento débil: 1–2 negativo, 3 neutro, 4–5 positivo. |
| rating | Calificación original de 1 a 5 estrellas. |
| asin | Identificador del producto. |
| parent_asin | Identificador de familia o variante. |
| helpful_vote | Votos de utilidad de la reseña. |
| verified_purchase | Indicador de compra verificada. |
| fecha | Fecha derivada del timestamp original. |

## Diagnóstico de calidad

- El subconjunto tiene cero faltantes y cero duplicados exactos en `review`.
- Las fechas abarcan de 2000 a 2023.
- La mediana es 41 palabras; el percentil 99 es 476 palabras. Las 499 observaciones sobre ese percentil se
  consideran extremas, pero no se eliminan sin evidencia de que sean inválidas.
- El dataset original está fuertemente concentrado en cinco estrellas; el submuestreo equilibra las clases,
  pero modifica su prevalencia natural.
- La etiqueta es débil y las reseñas de tres estrellas suelen mezclar elogios y quejas.

## Hallazgos del EDA y decisiones

1. **Desbalance original:** justificó el submuestreo balanceado y el uso de F1 macro.
2. **Neutro semánticamente mixto:** anticipó que sería la clase más difícil y motivó reportar su F1.
3. **Cola larga de longitud:** se conservaron reseñas extensas, mientras TF-IDF normaliza su influencia.
4. **Negaciones informativas:** se conservaron `not`, `no`, `never` y bigramas como `not work`.
5. **Títulos que revelan estrellas:** expresiones como `Five Stars` se eliminaron; mantenerlas elevaba el
   F1 macro de manera artificial de aproximadamente 0,748 a 0,765.

## Partición y prevención de leakage

La partición global es 64 % sub-train, 16 % validación y 20 % test, estratificada y con semilla 42. TF-IDF,
SVD y cualquier transformación aprendida se ajustan solamente con los datos permitidos en cada etapa. Los
hiperparámetros y el modelo se eligen mediante validación; test se utiliza una sola vez para el reporte final.
No se incluyen `rating` ni `label` como entrada predictiva.

## Métricas y modelo elegido

F1 macro es la métrica principal porque pondera por igual las tres clases. Accuracy se reporta como métrica
secundaria y F1 neutro permite vigilar la clase más ambigua. La regresión logística con C=1 se eligió por su
F1 macro de validación (0,741), no por test. En test obtuvo F1 macro 0,748 y F1 neutro 0,653.

## Análisis de errores

El modelo cometió 2 523 errores en 10 000 ejemplos de test. Las confusiones dominantes fueron:

| Real → predicción | Casos |
|---|---:|
| neutro → negativo | 753 |
| negativo → neutro | 734 |
| positivo → neutro | 450 |
| neutro → positivo | 388 |
| positivo → negativo | 108 |
| negativo → positivo | 90 |

La frontera principal es, por tanto, neutro frente a las clases polares. Varios errores de alta confianza son
textos muy breves como `excellent`, `works well` o `for the money ok`: el lenguaje expresa una polaridad clara,
pero el rating que generó la etiqueta puede ser neutro o negativo. Esto evidencia ruido de etiqueta, falta de
contexto y opiniones mixtas, no solamente una limitación del clasificador.

## Limitaciones y mejoras

El modelo no comprende ironía, contexto del producto ni cambios de opinión dentro de una reseña. Una mejora
futura debe revisar manualmente una muestra de errores, evaluar resultados por longitud/producto/fecha y
considerar etiquetas humanas o modelos contextuales en la Parte II.
