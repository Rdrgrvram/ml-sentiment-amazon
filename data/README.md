# Datos

- `electronics_50k.parquet`: muestra balanceada de 49 998 reseñas (16 666 por clase) de
  Amazon Reviews 2023 – Electronics, McAuley Lab (UCSD), accedido el 16/10/2026. Columnas: `review` (título + texto), `label`, `rating`, `asin`,
  `parent_asin`, `helpful_vote`, `verified_purchase`, `fecha`. Rango de fechas: 2000–2023.
- `split.csv`: índice de cada fila y si va a `train` o `test`. Lo crea `python -m src.preprocess`
  y lo usan todos los scripts para comparar sobre la misma partición.

## Cómo regenerar la muestra
Correr `notebooks/01_datos_y_eda.ipynb` (pensado para Google Colab). Lee por streaming las primeras
~1,5 M líneas de:
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 → `raw/review_categories/Electronics.jsonl`
- respaldo: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz

Semilla 42. Distribución original de las clases antes de balancear: ~1,09 M positivo, ~200 k negativo,
~100 k neutro (muy cargada a 5★).

El diagnóstico reproducible y la tabla de variables se generan con `python -m src.data_quality` y quedan en
`reports/calidad_datos.md` y `reports/tabla_variables.csv`.
