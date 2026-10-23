# Guía breve para las preguntas de defensa

1. **¿Por qué se puede resolver con ML?** Hay texto histórico con una señal de etiqueta derivada del rating y
   patrones léxicos que permiten aprender una función de clasificación.
2. **¿Qué hallazgo del EDA fue más importante?** El desbalance original y la ambigüedad de tres estrellas;
   motivaron el muestreo balanceado, F1 macro y el seguimiento de F1 neutro.
3. **¿Cómo evitaron leakage?** Separación fija 64/16/20; transformadores ajustados solo con entrenamiento;
   test aislado; exclusión de rating y eliminación de títulos `Five Stars`.
4. **¿Por qué F1 macro?** Da el mismo peso a negativo, neutro y positivo, independientemente de frecuencia.
5. **¿Cuál fue el baseline?** Dummy de clase más frecuente, F1 macro 0,167; establece el piso sin aprender texto.
6. **¿Qué modelo generalizó mejor?** TF-IDF + LogReg C=1, elegido por F1 de validación 0,741; en test 0,748.
7. **¿Qué errores persisten?** Principalmente neutro↔negativo y neutro↔positivo por opiniones mixtas,
   textos cortos y desacuerdo entre texto y estrellas.
8. **¿Qué significan los clusters?** Son patrones exploratorios de vocabulario y temas; no son categorías
   reales ni clases de sentimiento. El 24,5 % de ruido representa casos sin pertenencia densa clara.
9. **¿Qué encontró AutoML?** Favoreció nuevamente una familia lineal; se compara con el modelo manual bajo
   el mismo holdout y muestra el costo de automatización frente a control e interpretabilidad.
10. **¿Qué cambiarían para Parte II?** Etiquetas humanas, auditoría por segmentos, representaciones
    contextuales, tracking de experimentos, monitoreo y recuperación semántica integrada.
