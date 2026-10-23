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
src/         preprocess.py, train.py, evaluate.py, unsupervised.py, automl.py
models/      sentiment_logreg.joblib (mejor modelo manual)
reports/     tablas de resultados, figures/ (gráficos), defensa/ (informe y PPT)
config/      config.yaml (semilla, partición, hiperparámetros de todos los experimentos)
parte2/      búsqueda semántica y demo Streamlit (fuera del alcance del EC2; se usan en la Parte II)
```

## Cómo ejecutar
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.preprocess      # crea data/split.csv (80/20 estratificado, semilla 42)
python -m src.train           # Dummy + LogReg + SVM + LightGBM + tuning (≈ 10 min; --rapido sin LightGBM)
python -m src.evaluate        # matriz de confusión, reporte por clase, errores, palabras por clase
python -m src.unsupervised    # LSA + KMeans, selección de k por silueta, interpretación
python -m src.automl          # FLAML, misma partición, 15 min (--budget 120 para una prueba)
python -m src.report_automl   # gráficos de evidencia de AutoML
```
Orden recomendado: `preprocess → train → evaluate → unsupervised → automl`. Todo lee `config/config.yaml`.

## Metodología y decisiones
- **Partición:** 80 % train / 20 % test estratificado, fija en `data/split.csv`. El ajuste de hiperparámetros
  usa validación cruzada de 3 folds **solo dentro de train**; el test se usa una vez al final.
- **Sin data leakage:**
  - el TF-IDF va dentro del `Pipeline`, así que se ajusta solo con train;
  - duplicados eliminados antes de muestrear;
  - se quitan los **títulos automáticos de Amazon** ("Five Stars", "One Star"…), que traen la etiqueta escrita
    dentro del texto (6,9 % de las reseñas). Con ellos el F1 macro era 0,765; sin ellos baja a 0,748, que es
    la cifra honesta.
- **Limpieza:** minúsculas, sin URLs ni números, negaciones conservadas (`don't → do not`) y `! ?`.
- **Métrica principal:** F1 macro (las 3 clases pesan igual; el neutro es la clase difícil).

## Resultados (test, 10 000 reseñas, sin títulos automáticos)
| Modelo | Accuracy | F1 macro | F1 neutro | Tiempo | Observación |
|---|---|---|---|---|---|
| Dummy (clase más frecuente) | 0,333 | 0,167 | 0,000 | 7 s | Baseline / piso |
| **TF-IDF + LogReg ajustada (C=1)** | **0,748** | **0,748** | **0,653** | 118 s* | Mejor modelo manual |
| TF-IDF + LogReg (C=2) | 0,745 | 0,745 | 0,647 | 18 s | |
| TF-IDF + SVM lineal | 0,739 | 0,738 | 0,634 | 10 s | |
| TF-IDF + LightGBM | 0,732 | 0,732 | 0,632 | 499 s | |
| AutoML FLAML (lrl1, 15 min) | 0,720 | 0,720 | 0,625 | 954 s | Benchmark |

\* incluye la búsqueda de C con CV de 3 folds. Detalle en `reports/comparativa_final.csv`;
evidencia de AutoML en `reports/automl_resultado.json`, `reports/automl_trials.csv`, `reports/automl.log`
y `reports/figures/automl_curva.png`.

## Estado del EC2
- [x] Fuente, subconjunto y etiqueta documentados
- [x] EDA básico (distribución original, longitudes, palabras por clase)
- [x] Baseline + 3 modelos clásicos + tuning con CV
- [x] Pipeline sin leakage y partición fija
- [ ] Diagnóstico de calidad de datos y 3–5 hallazgos del EDA en el informe
- [ ] Análisis de errores interpretado (`reports/errores_logreg.csv`)
- [ ] No supervisado: elegir k, interpretar clusters, limitaciones (`src/unsupervised.py`)
- [x] AutoML con 15 min y comparación con el mejor manual (`src/automl.py`, `src/report_automl.py`)
- [ ] Informe técnico (8–12 pág.) y presentación en `reports/defensa/`

## Parte II (fuera del EC2)
`parte2/03_busqueda_semantica.ipynb` (sentence-transformers + coseno) y `parte2/streamlit_app.py` (demo).
Requieren `parte2/requirements.txt`.
