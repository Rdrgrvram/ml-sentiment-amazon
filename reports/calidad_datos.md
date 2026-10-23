# Diagnóstico de calidad de datos

- **Fuente:** Amazon Reviews 2023 - Electronics, McAuley Lab (UCSD).
- **Fecha de acceso declarada por el equipo:** 16/10/2026.
- **Subconjunto final:** 49,998 reseñas y 8 variables.
- **Muestreo:** balanceado por clase después de leer por streaming aproximadamente 1,5 millones de reseñas.
- **Duplicados exactos por texto en el subconjunto:** 0.
- **Faltantes totales:** 0.
- **Rango de fechas:** 2000-01-11 03:01:26 a 2023-03-20 00:47:02.053000.
- **Longitud mediana:** 41 palabras; percentil 99: 476 palabras (2610 caracteres).
- **Reseñas sobre el percentil 99 de longitud:** 499; se consideran extremas para diagnóstico, no se eliminan automáticamente.

## Distribución de la muestra final

- positivo: 16,666
- negativo: 16,666
- neutro: 16,666

## Distribución por estrellas

- 1 estrellas: 10,815
- 2 estrellas: 5,851
- 3 estrellas: 16,666
- 4 estrellas: 3,088
- 5 estrellas: 13,578

## Riesgos y sesgos

1. La etiqueta es débil porque se deriva de estrellas; el texto puede expresar sentimientos mixtos o no coincidir con el rating.
2. El dataset original está fuertemente sesgado hacia 5 estrellas; el balanceo mejora la comparación de clases, pero ya no representa la prevalencia real.
3. Las reseñas extremadamente largas pueden dominar conteos o contener varios temas; se conservan para no descartar información sin evidencia.
4. Productos, fechas y usuarios no están necesariamente distribuidos de forma uniforme; los resultados no deben generalizarse a todas las categorías de Amazon.
5. Los títulos automáticos como 'Five Stars' se eliminan antes del modelado porque revelan directamente la etiqueta.

## Decisiones

- Se eliminan duplicados antes del muestreo.
- Se descartan reseñas de menos de 20 caracteres durante la construcción del subconjunto.
- No se imputan textos; el subconjunto final no presenta faltantes en la entrada principal.
- La partición es estratificada y fija. Todo transformador aprendido se ajusta solo con entrenamiento.
