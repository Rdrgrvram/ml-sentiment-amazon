# 3.1 Comprensión del problema

## Contexto y necesidad

Las tiendas digitales reciben volúmenes de reseñas que superan la capacidad de revisión manual. Clasificar
automáticamente su sentimiento permite resumir satisfacción, localizar problemas recurrentes y priorizar
la inspección de comentarios críticos. El proyecto estudia reseñas de productos electrónicos de Amazon y
clasifica el texto como negativo, neutro o positivo.

## Objetivo y pregunta guía

El objetivo es construir una solución clásica, reproducible e interpretable que prediga el sentimiento a
partir del texto y explorar patrones temáticos sin utilizar las etiquetas durante el agrupamiento.

**Pregunta guía:** ¿qué combinación de representación y modelo clásico generaliza mejor al clasificar
sentimiento, y qué patrones temáticos aparecen al analizar las reseñas sin supervisión?

## Criterios de éxito

1. Superar de forma clara el baseline Dummy en F1 macro.
2. Comparar al menos tres modelos clásicos bajo la misma partición y elegir mediante validación.
3. Mantener el conjunto test aislado hasta la evaluación final.
4. Reportar métricas por clase y analizar errores, especialmente en la clase neutra.
5. Obtener patrones no supervisados interpretables sin presentarlos como categorías reales.
6. Mantener configuración, semilla, dependencias y salidas suficientes para reproducir el proceso.

## Restricciones y riesgos

- La etiqueta proviene de estrellas y es débil: un rating puede no coincidir completamente con el texto.
- El inglés informal, la ironía, textos muy cortos y opiniones mixtas generan ambigüedad.
- La muestra balanceada permite comparar clases, pero no conserva la prevalencia real del dataset original.
- Los recursos de cómputo y el tiempo limitan la búsqueda de AutoML y los modelos no supervisados.
- Los clusters dependen de la representación y de los parámetros; son hallazgos exploratorios.

## Continuidad y mejora respecto del EC1

El EC1 aportó la formulación inicial, el subconjunto, el EDA y los primeros modelos supervisados. En EC2 se
fortaleció el protocolo: se fijó una partición 64/16/20, se separó validación de test, se incorporó el
preprocesamiento dentro de pipelines, se eliminaron títulos automáticos que filtraban estrellas, se añadieron
análisis de errores, UMAP–HDBSCAN y AutoML como benchmark. Por ello, la mejora no consiste solamente en una
métrica, sino en mayor validez metodológica, interpretabilidad y reproducibilidad.

## Apoyos externos e integridad académica

Se utilizaron pandas, NumPy, scikit-learn, LightGBM, XGBoost, FLAML, UMAP-learn, HDBSCAN, Matplotlib,
Seaborn, PyArrow, Joblib y PyYAML. Se usaron asistentes de IA como apoyo para revisar y estructurar código,
documentación y pruebas. El equipo conserva la responsabilidad de verificar los resultados, comprender cada
decisión y defenderla. El dataset y su documentación pertenecen a Amazon Reviews 2023 del McAuley Lab.
