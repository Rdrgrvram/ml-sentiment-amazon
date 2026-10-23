# Notebooks y fuente oficial de resultados

- `01_datos_y_eda.ipynb` documenta la descarga, limpieza, muestreo y EDA inicial. No es necesario volver a
  descargar 1,5 millones de reseñas: `data/electronics_50k.parquet` ya contiene el subconjunto final.
- `02_modelos_sentimiento.ipynb` conserva el desarrollo exploratorio inicial. Sus resultados no deben usarse
  como cifras finales porque preceden a la separación 64/16/20 y a la corrección del leakage por títulos
  automáticos de estrellas.

La fuente oficial y reproducible de la entrega son los comandos:

```bash
python -m src.data_quality
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.unsupervised
python -m src.automl
python -m src.report_automl
```

Las tablas y figuras que deben citarse en informe y presentación están en `reports/`.
