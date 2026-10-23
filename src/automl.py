"""AutoML como benchmark (FLAML) con la MISMA partición que los modelos manuales.

Uso:  python -m src.automl                 (presupuesto de config: 15 min)
      python -m src.automl --budget 120    (prueba rápida)

Protocolo: usa el mismo sub-train/validación/test 64/16/20 que los modelos manuales.
TF-IDF y SVD se ajustan solo con sub-train durante la búsqueda; así la validación no
participa en la representación y el test permanece completamente aislado.
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import ROOT, get_split, load_config
from src.train import metricas, tfidf


def main(budget: int | None = None):
    cfg = load_config()
    a = cfg["automl"]
    budget = budget or a["time_budget_s"]
    rep = ROOT / cfg["paths"]["reports"]
    X_tr, X_te, y_tr, y_te = get_split(cfg)

    # Exactamente la misma validación estratificada que src.train.
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_tr, y_tr, test_size=cfg["data"]["val_size"],
        stratify=y_tr, random_state=cfg["seed"]
    )

    feats_busqueda = Pipeline([
        ("tfidf", tfidf(cfg)),
        ("svd", TruncatedSVD(n_components=a["svd_components"], random_state=cfg["seed"])),
    ])
    t0 = time.time()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        F_sub = feats_busqueda.fit_transform(X_sub)  # nunca ve validación ni test
        F_val = feats_busqueda.transform(X_val)
    if not (np.isfinite(F_sub).all() and np.isfinite(F_val).all()):
        raise ValueError("SVD produjo valores no finitos durante la búsqueda AutoML")

    automl = AutoML()
    automl.fit(
        F_sub, np.asarray(y_sub, dtype=object),
        X_val=F_val, y_val=np.asarray(y_val, dtype=object),
        task="classification", metric=a["metric"], time_budget=budget,
        estimator_list=a["estimators"], eval_method="holdout", retrain_full=False,
        seed=cfg["seed"], log_file_name=str(rep / "automl.log"), verbose=1,
    )
    f1_val = 1 - automl.best_loss

    # Reajuste final: la configuración ya fue elegida. La representación se ajusta
    # ahora con todo train y el estimador elegido se reentrena una sola vez antes de test.
    feats_final = Pipeline([
        ("tfidf", tfidf(cfg)),
        ("svd", TruncatedSVD(n_components=a["svd_components"], random_state=cfg["seed"])),
    ])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        F_tr = feats_final.fit_transform(X_tr)
        F_te = feats_final.transform(X_te)
    if not (np.isfinite(F_tr).all() and np.isfinite(F_te).all()):
        raise ValueError("SVD produjo valores no finitos durante el reajuste AutoML")
    final = AutoML()
    final.fit(
        F_tr, np.asarray(y_tr, dtype=object), task="classification", metric=a["metric"],
        estimator_list=[automl.best_estimator],
        starting_points={automl.best_estimator: automl.best_config},
        max_iter=1, eval_method="holdout", retrain_full=True,
        seed=cfg["seed"], verbose=0,
    )
    pred = final.predict(F_te)
    seg = round(time.time() - t0, 1)

    resultado = {"herramienta": "FLAML", "presupuesto_s": budget, "metrica_busqueda": a["metric"],
                 "protocolo": "holdout 64/16/20; features ajustadas solo con sub-train durante búsqueda",
                 "features": f"TF-IDF + SVD({a['svd_components']})",
                 "mejor_estimador": automl.best_estimator, "mejor_config": automl.best_config,
                 "f1_val": f1_val, **metricas(y_te, pred), "seg_total": seg}
    (rep / "automl_resultado.json").write_text(json.dumps(resultado, indent=2, default=str), encoding="utf-8")
    print(json.dumps(resultado, indent=2, default=str))

    actualizar_comparativa(cfg, resultado)


def actualizar_comparativa(cfg: dict, resultado: dict | None = None):
    """Tabla final: modelos manuales + mejor de AutoML (sin volver a correr AutoML)."""
    rep = ROOT / cfg["paths"]["reports"]
    if resultado is None:
        resultado = json.loads((rep / "automl_resultado.json").read_text(encoding="utf-8"))
    manual = pd.read_csv(rep / "resultados_modelos.csv")
    f1_val = resultado.get("f1_val", resultado.get("f1_cv"))
    fila = {"modelo": f"AutoML FLAML ({resultado['mejor_estimador']})",
            "f1_val": f1_val,
            "accuracy": resultado["accuracy"], "f1_macro": resultado["f1_macro"],
            "f1_neutro": resultado["f1_neutro"], "seg": resultado["seg_total"],
            "elegido_por_validacion": False}
    comp = pd.concat([manual[~manual.modelo.str.startswith("AutoML")], pd.DataFrame([fila])])
    comp.sort_values("f1_val", ascending=False).to_csv(rep / "comparativa_final.csv", index=False)
    print(comp.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, help="segundos de búsqueda (por defecto: config)")
    ap.add_argument("--solo-comparativa", action="store_true",
                    help="no corre AutoML; rearma comparativa_final.csv con el último resultado")
    args = ap.parse_args()
    if args.solo_comparativa:
        actualizar_comparativa(load_config())
    else:
        main(args.budget)
