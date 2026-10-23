# 3.6 AutoML como benchmark (Rodrigo)

## Configuración
- **Herramienta:** FLAML 2.x (`src/automl.py`, parámetros en `config/config.yaml → automl`).
- **Misma partición** que los modelos manuales: sub-train/validación/test = 64/16/20, estratificado y con
  semilla 42. El test no se usa durante la búsqueda.
- **Representación:** TF-IDF con unigramas y bigramas, reducido con TruncatedSVD a 300 componentes. Durante
  la búsqueda ambos transformadores se ajustan exclusivamente con sub-train y solo transforman validación.
- **Búsqueda:** métrica `macro_f1`, holdout de validación común, presupuesto de 900 s,
  estimadores candidatos: LightGBM, XGBoost, Random Forest, Extra Trees y regresión logística L1.
- **Evidencia:** `reports/automl_resultado.json` (mejor configuración y métricas), `reports/automl.log` y
  `reports/automl_trials.csv` (todos los trials), `reports/figures/automl_curva.png`.

## Resultados
| Modelo | F1 macro validación | F1 macro test | F1 neutro test | Tiempo |
|---|---|---|---|---|
| TF-IDF + LogReg C=1 (mejor manual, elegido por validación) | **0,741** | **0,748** | **0,653** | 17 s |
| TF-IDF + LightGBM (manual) | 0,728 | 0,732 | 0,633 | 259 s |
| **Mejor de AutoML: FLAML → regresión logística L1 (C=1,898)** | 0,716 | 0,724 | 0,630 | 946 s |
| Dummy (baseline) | 0,167 | 0,167 | 0,000 | 13 s |

FLAML registró 15 trials en 15 minutos: 10 de LightGBM (F1 de validación hasta 0,696) y 5 de regresión
logística L1 (hasta 0,716), que fue la mejor familia. El presupuesto no alcanzó para producir trials válidos
de todas las familias candidatas.

## Discusión de trade-offs
- **Rendimiento:** AutoML quedó 2,4 puntos de F1 en test y 2,5 en validación por debajo del mejor modelo manual. La causa principal es la
  representación: al comprimir el TF-IDF de 100 000 términos a 300 componentes con SVD se pierde información
  léxica fina (negaciones, bigramas como "stopped working"), que es justo lo que aprovecha la regresión
  logística manual sobre el TF-IDF disperso completo.
- **Hallazgo coincidente:** sin que se lo indicáramos, AutoML también terminó eligiendo un **modelo lineal**
  por encima de los árboles con boosting. Esto respalda nuestra conclusión manual: en texto representado con
  TF-IDF, los modelos lineales bastan y los árboles no compensan su costo.
- **Tiempo:** AutoML usó 946 s contra 17 s de la LogReg manual (unas 56 veces más) para un resultado peor.
  Con más presupuesto o más núcleos podría explorar mejor las familias pendientes.
- **Interpretabilidad:** la LogReg manual sobre TF-IDF permite leer las palabras que empujan a cada clase.
  El modelo de AutoML opera sobre componentes SVD, que no se pueden leer como palabras.
- **Control del proceso:** en el enfoque manual decidimos la limpieza, los n-gramas y el ajuste de C. En
  AutoML solo controlamos la representación de entrada, el presupuesto y la lista de estimadores.
- **Reproducibilidad:** ambos usan semilla fija, la misma validación y configuración versionada. El resultado
  de AutoML todavía depende del presupuesto y del hardware, que determinan cuántos trials alcanza a explorar.

## Conclusión
El mejor modelo manual (TF-IDF + regresión logística, C=1) supera al benchmark de AutoML en rendimiento,
tiempo e interpretabilidad, y se mantiene como modelo elegido para la Parte I. AutoML confirma de forma
independiente que la familia lineal es la adecuada para este problema.
