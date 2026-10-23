# Datos

- `electronics_50k.parquet`: muestra balanceada de 49 998 reseñas (16 666 por clase) de
  Amazon Reviews 2023 – Electronics. Columnas: `review` (título + texto), `label`, `rating`, `asin`,
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
