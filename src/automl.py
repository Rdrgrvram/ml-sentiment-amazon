"""AutoML como benchmark (FLAML) con la MISMA partición que los modelos manuales.

Uso:  python -m src.automl                 (presupuesto de config: 15 min)
      python -m src.automl --budget 120    (prueba rápida)

Features: TF-IDF (ajustado solo en train) -> TruncatedSVD, porque los estimadores de
FLAML trabajan mejor con matrices densas de tamaño moderado.
Salidas: reports/automl_resultado.json, reports/automl.log, filas añadidas a
         reports/comparativa_final.csv
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from src.preprocess import ROOT, get_split, load_config
from src.train import metricas, tfidf


def main(budget: int | None = None):
    cfg = load_config()
    a = cfg["automl"]
    budget = budget or a["time_budget_s"]
    rep = ROOT / cfg["paths"]["reports"]
    X_tr, X_te, y_tr, y_te = get_split(cfg)

    feats = Pipeline([("tfidf", tfidf(cfg)),
                      ("svd", TruncatedSVD(n_components=a["svd_components"], random_state=cfg["seed"]))])
    t0 = time.time()
    F_tr = feats.fit_transform(X_tr)      # ajustado solo con train
    F_te = feats.transform(X_te)

    automl = AutoML()
    automl.fit(F_tr, np.asarray(y_tr, dtype=object), task="classification", metric=a["metric"],
               time_budget=budget, estimator_list=a["estimators"], eval_method="cv", n_splits=3,
               seed=cfg["seed"], log_file_name=str(rep / "automl.log"), verbose=1)
    pred = automl.predict(F_te)
    seg = round(time.time() - t0, 1)

    resultado = {"herramienta": "FLAML", "presupuesto_s": budget, "metrica_busqueda": a["metric"],
                 "features": f"TF-IDF + SVD({a['svd_components']})",
                 "mejor_estimador": automl.best_estimator, "mejor_config": automl.best_config,
                 "f1_cv": 1 - automl.best_loss, **metricas(y_te, pred), "seg_total": seg}
    (rep / "automl_resultado.json").write_text(json.dumps(resultado, indent=2, default=str), encoding="utf-8")
    print(json.dumps(resultado, indent=2, default=str))

    # Tabla comparativa final: modelos manuales + mejor de AutoML
    manual = pd.read_csv(rep / "resultados_modelos.csv")
    fila = {"modelo": f"AutoML FLAML ({automl.best_estimator})", "accuracy": resultado["accuracy"],
            "f1_macro": resultado["f1_macro"], "f1_neutro": resultado["f1_neutro"], "seg": seg}
    comp = pd.concat([manual[~manual.modelo.str.startswith("AutoML")], pd.DataFrame([fila])])
    comp.sort_values("f1_macro", ascending=False).to_csv(rep / "comparativa_final.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, help="segundos de búsqueda (por defecto: config)")
    main(ap.parse_args().budget)
