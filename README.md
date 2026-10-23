# Sentimiento en reseñas de electrónica — Proyecto Integrador Parte I (EC2)

**SIS-351 Machine Learning · UCB · Grupo 3 · Defensa: viernes 23/10/2026, 17:45**

Clasificar el sentimiento (negativo / neutro / positivo) de reseñas de productos electrónicos de Amazon
a partir del texto, siguiendo CRISP-DM: baseline + modelos clásicos, aprendizaje no supervisado y AutoML
como benchmark.

## Datos
- **Fuente:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD), categoría
  *Electronics*, archivo `raw/review_categories/Electronics.jsonl`.
- **Subconjunto:** se leen por streaming las primeras ~1,5 M reseñas; se limpian (reseñas < 20 caracteres y
  duplicadas fuera) y se toma una **muestra balanceada de 49 998 reseñas** (16 666 por clase).
  Ver `notebooks/01_datos_y_eda.ipynb` y `data/README.md`.
- **Etiqueta (débil):** desde las estrellas → 1–2★ negativo · 3★ neutro · 4–5★ positivo.

## Estructura
```
data/        electronics_50k.parquet (muestra), split.csv (partición fija), README con cómo obtenerla
notebooks/   01_datos_y_eda.ipynb, 02_modelos_sentimiento.ipynb (EDA y experimentos explicados)
src/         preparación, entrenamiento, evaluación, no supervisado y AutoML
models/      sentiment_logreg.joblib (mejor modelo manual)
reports/     tablas de resultados, figures/ (gráficos), defensa/ (informe y PPT)
config/      config.yaml (semilla, partición, hiperparámetros de todos los experimentos)
parte2/      búsqueda semántica y demo Streamlit (fuera del alcance del EC2; se usan en la Parte II)
```

## Cómo ejecutar
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
# macOS: brew install libomp   # requerido por LightGBM/XGBoost

python -m src.data_quality     # tabla de variables y diagnóstico reproducible
python -m src.preprocess      # crea data/split.csv (80/20 estratificado, semilla 42)
python -m src.train           # Dummy + SVM + LightGBM + LogReg con C ajustado en validación (≈ 17 min; --rapido sin LightGBM)
python -m src.evaluate        # matriz de confusión, reporte por clase, errores, palabras por clase
python -m src.unsupervised    # SVD + UMAP + HDBSCAN (método no supervisado final)
python -m src.automl          # FLAML, misma partición, 15 min (--budget 120 para una prueba)
python -m src.report_automl   # gráficos de evidencia de AutoML
python -m src.validate_project # comprobación rápida antes de entregar
```
Orden recomendado: `data_quality → preprocess → train → evaluate → unsupervised → automl`.
Todo lee `config/config.yaml`. Los notebooks conservan el desarrollo exploratorio; las cifras oficiales
provienen de `src/` y `reports/`.

El análisis no supervisado final usa una muestra estratificada de 8 000 reseñas: TF-IDF reduce primero a
100 componentes con SVD y luego UMAP aprende una representación no lineal de 10 dimensiones. HDBSCAN
detecta regiones densas sin fijar `k` y permite dejar reseñas ambiguas como ruido. Con
`min_cluster_size=100` se obtuvieron 30 clusters, 75,5 % de cobertura y 24,5 % de ruido. La silueta de
0,568 se calcula dentro del espacio UMAP y no se compara directamente con la de K-Means sobre SVD.

Los experimentos K-Means, GMM y DBSCAN se conservan como benchmarks:
```bash
python -m src.unsupervised --metodo kmeans --k 12
python -m src.unsupervised_gmm
python -m src.unsupervised_dbscan
```

## Metodología y decisiones
- **Partición:** train / validación / test = 64 / 16 / 20 %, estratificado y con semilla fija (`data/split.csv`
  separa el test). Los hiperparámetros y el modelo final se eligen **con validación**; después el modelo se
  reentrena con todo train y el test se usa **una sola vez** para el reporte final.
- **Sin data leakage:**
  - el TF-IDF va dentro del `Pipeline`, así que se ajusta solo con train;
  - duplicados eliminados antes de muestrear;
  - se quitan los **títulos automáticos de Amazon** ("Five Stars", "One Star"…), que traen la etiqueta escrita
    dentro del texto (6,9 % de las reseñas). Con ellos el F1 macro era 0,765; sin ellos baja a 0,748, que es
    la cifra honesta.
- **Limpieza:** minúsculas, sin URLs ni números, negaciones conservadas (`don't → do not`) y `! ?`.
- **Métrica principal:** F1 macro (las 3 clases pesan igual; el neutro es la clase difícil).

## Resultados
Selección con **validación** (16 % de los datos); el test (20 %) se usa una sola vez, al final.

| Modelo | F1 macro validación | F1 macro test | F1 neutro test | Tiempo | Observación |
|---|---|---|---|---|---|
| Dummy (clase más frecuente) | 0,167 | 0,167 | 0,000 | 13 s | Baseline / piso |
| **TF-IDF + LogReg (C=1)** | **0,741** | **0,748** | **0,653** | 17 s | **Elegido por validación** |
| TF-IDF + LightGBM | 0,728 | 0,732 | 0,633 | 259 s | |
| TF-IDF + SVM lineal | 0,731 | 0,737 | 0,632 | 8 s | |
| AutoML FLAML (lrl1, 15 min) | 0,716 | 0,724 | 0,630 | 946 s | Benchmark holdout comparable |

C de la LogReg elegido en validación entre {0,5; 1; 2; 4; 8} (`reports/tuning_logreg_val.csv`).
Tiempos: entrenamiento en sub-train + reentrenamiento con todo train; LightGBM usa 2 núcleos.
Detalle en `reports/comparativa_final.csv`; evidencia de AutoML en `reports/automl_resultado.json`,
`reports/automl_trials.csv`, `reports/automl.log` y `reports/figures/automl_curva.png`.

## Estado del EC2
- [x] Fuente, subconjunto y etiqueta documentados
- [x] EDA básico (distribución original, longitudes, palabras por clase)
- [x] Baseline + 3 modelos clásicos + tuning con validación holdout
- [x] Pipeline sin leakage y partición fija
- [x] Diagnóstico de calidad, tabla de variables y hallazgos del EDA
- [x] Análisis de errores interpretado (`reports/errores_logreg.csv`)
- [x] No supervisado: UMAP + HDBSCAN, comparación con K-Means/GMM/DBSCAN e interpretación
- [x] AutoML con 15 min y comparación con el mejor manual (`src/automl.py`, `src/report_automl.py`)
- [ ] Ensamblar el informe técnico (8–12 pág.) y la presentación final

## Parte II (fuera del EC2)
`parte2/03_busqueda_semantica.ipynb` (sentence-transformers + coseno) y `parte2/streamlit_app.py` (demo).
Requieren `parte2/requirements.txt`.
