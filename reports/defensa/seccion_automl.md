# 3.6 AutoML como benchmark (Rodrigo)

## Configuración
- **Herramienta:** FLAML 2.x (`src/automl.py`, parámetros en `config/config.yaml → automl`).
- **Misma partición** que los modelos manuales: `data/split.csv` (80/20 estratificado, semilla 42). El test
  no se usa durante la búsqueda.
- **Representación:** TF-IDF (unigramas y bigramas, mismo preprocesamiento que los modelos manuales, ajustado
  solo con train) reducido con TruncatedSVD a 300 componentes, porque los estimadores de FLAML trabajan con
  matrices densas de tamaño moderado.
- **Búsqueda:** métrica `macro_f1`, validación cruzada de 3 folds dentro de train, presupuesto de 900 s,
  estimadores candidatos: LightGBM, XGBoost, Random Forest, Extra Trees y regresión logística L1.
- **Evidencia:** `reports/automl_resultado.json` (mejor configuración y métricas), `reports/automl.log` y
  `reports/automl_trials.csv` (todos los trials), `reports/figures/automl_curva.png`.

## Resultados
| Modelo | F1 macro (test) | F1 neutro | Tiempo |
|---|---|---|---|
| TF-IDF + LogReg ajustada (mejor manual) | **0,748** | **0,653** | 118 s |
| TF-IDF + LightGBM (manual) | 0,732 | 0,632 | 499 s |
| **Mejor de AutoML: FLAML → regresión logística L1 (C=1)** | 0,720 | 0,625 | 954 s |
| Dummy (baseline) | 0,167 | 0,000 | 7 s |

FLAML ejecutó 11 trials en 15 minutos: 9 de LightGBM (F1 de validación hasta 0,694), 1 de XGBoost (0,701) y
1 de regresión logística L1 (0,721), que fue el mejor. No llegó a probar Random Forest ni Extra Trees.

## Discusión de trade-offs
- **Rendimiento:** AutoML quedó 2,8 puntos de F1 por debajo del mejor modelo manual. La causa principal es la
  representación: al comprimir el TF-IDF de 100 000 términos a 300 componentes con SVD se pierde información
  léxica fina (negaciones, bigramas como "stopped working"), que es justo lo que aprovecha la regresión
  logística manual sobre el TF-IDF disperso completo.
- **Hallazgo coincidente:** sin que se lo indicáramos, AutoML también terminó eligiendo un **modelo lineal**
  por encima de los árboles con boosting. Esto respalda nuestra conclusión manual: en texto representado con
  TF-IDF, los modelos lineales bastan y los árboles no compensan su costo.
- **Tiempo:** AutoML usó 954 s contra 118 s de la LogReg ajustada (8 veces más) para un resultado peor. Con
  2 núcleos solo alcanzó a evaluar 11 configuraciones; con más presupuesto o más núcleos podría acercarse.
- **Interpretabilidad:** la LogReg manual sobre TF-IDF permite leer las palabras que empujan a cada clase.
  El modelo de AutoML opera sobre componentes SVD, que no se pueden leer como palabras.
- **Control del proceso:** en el enfoque manual decidimos la limpieza, los n-gramas y el ajuste de C. En
  AutoML solo controlamos la representación de entrada, el presupuesto y la lista de estimadores.
- **Reproducibilidad:** ambos son reproducibles (semilla fija, misma partición, configuración en
  `config.yaml`), pero el resultado de AutoML depende del presupuesto de tiempo y del hardware: con otra
  máquina explora un número distinto de trials.

## Conclusión
El mejor modelo manual (TF-IDF + regresión logística, C=1) supera al benchmark de AutoML en rendimiento,
tiempo e interpretabilidad, y se mantiene como modelo elegido para la Parte I. AutoML confirma de forma
independiente que la familia lineal es la adecuada para este problema.
