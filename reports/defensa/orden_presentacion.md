# Presentación de defensa en 10 diapositivas

Duración objetivo: 10 a 12 minutos. Las cifras deben copiarse de `reports/` después de ejecutar
`python -m src.validate_project`.

1. **Problema y objetivo, 50 segundos.** Clasificar sentimiento en reseñas de electrónica y explorar temas.
2. **Dataset y fuente, 55 segundos.** Amazon Reviews 2023 Electronics, McAuley Lab, acceso 16/10/2026,
   muestra balanceada de 49 998 reseñas.
3. **EDA y calidad, 65 segundos.** Desbalance original, etiqueta débil, longitud y ausencia de faltantes o
   duplicados en el subconjunto.
4. **Preparación y validación, 65 segundos.** Limpieza, TF-IDF, partición 64/16/20 y prevención de leakage.
5. **Baseline y modelos, 60 segundos.** Dummy, LogReg, SVM lineal y LightGBM bajo el mismo protocolo.
6. **Resultados supervisados, 65 segundos.** Tabla comparativa y elección de LogReg C=1 por validación.
7. **Errores y limitaciones, 65 segundos.** Confusión de neutro, textos cortos y desacuerdo texto-estrellas.
8. **No supervisado, 85 segundos.** UMAP–HDBSCAN final, cobertura, ruido, temas y comparación con benchmarks.
9. **AutoML, 65 segundos.** Configuración, resultado frente al modelo manual y trade-offs.
10. **Conclusiones y Parte II, 55 segundos.** Evidencia principal, limitaciones y mejoras futuras.

## Reparto sugerido

- Mauricio: diapositivas 1 a 4.
- Rodrigo: diapositivas 5, 6 y 9.
- Laura: diapositivas 7 y 8.
- Mayte: diapositiva 10, control del tiempo y transiciones.

Todos deben estudiar `guia_preguntas.md`: el docente puede dirigir cualquier pregunta a cualquier integrante.
