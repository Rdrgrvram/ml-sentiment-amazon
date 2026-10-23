"""Baseline + 3 modelos clásicos + tuning del mejor lineal (reproduce el notebook 02).

Uso:  python -m src.train            (todos los modelos, ~8 min por LightGBM)
      python -m src.train --rapido   (sin LightGBM)

Salidas: reports/resultados_modelos.csv, models/sentiment_logreg.joblib
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import ROOT, get_split, load_config


def tfidf(cfg: dict, max_features: int | None = None) -> TfidfVectorizer:
    t = cfg["tfidf"]
    return TfidfVectorizer(ngram_range=tuple(t["ngram_range"]), min_df=t["min_df"],
                           max_df=t["max_df"], sublinear_tf=t["sublinear_tf"],
                           max_features=max_features or t["max_features"])


def build_models(cfg: dict, rapido: bool = False) -> dict[str, Pipeline]:
    m, seed = cfg["models"], cfg["seed"]
    # El TF-IDF va dentro del Pipeline: se ajusta solo con train (evita data leakage).
    modelos = {
        "Dummy (baseline)": Pipeline([("tfidf", tfidf(cfg, 1000)),
                                      ("clf", DummyClassifier(strategy=m["dummy"]["strategy"]))]),
        "TF-IDF + LogReg": Pipeline([("tfidf", tfidf(cfg)),
                                     ("clf", LogisticRegression(C=m["logreg"]["C"], max_iter=m["logreg"]["max_iter"]))]),
        "TF-IDF + SVM lineal": Pipeline([("tfidf", tfidf(cfg)), ("clf", LinearSVC(C=m["svm"]["C"]))]),
    }
    if not rapido:
        lg = m["lightgbm"]
        modelos["TF-IDF + LightGBM"] = Pipeline([
            ("tfidf", tfidf(cfg, lg["max_features"])),
            ("clf", LGBMClassifier(n_estimators=lg["n_estimators"], learning_rate=lg["learning_rate"],
                                   num_leaves=lg["num_leaves"], random_state=seed, verbose=-1))])
    return modelos


def metricas(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_neutro": f1_score(y_true, y_pred, labels=["neutro"], average="macro")}


def main(rapido: bool = False):
    cfg = load_config()
    X_tr, X_te, y_tr, y_te = get_split(cfg)

    filas = []
    for nombre, modelo in build_models(cfg, rapido).items():
        t0 = time.time()
        modelo.fit(X_tr, y_tr)
        filas.append({"modelo": nombre, **metricas(y_te, modelo.predict(X_te)),
                      "seg": round(time.time() - t0, 1)})
        print(f"{nombre:22s} F1 macro = {filas[-1]['f1_macro']:.4f}")

    # Ajuste de hiperparámetros con validación cruzada SOLO sobre train.
    tcfg = cfg["tuning"]
    base = build_models(cfg, rapido=True)["TF-IDF + LogReg"]
    t0 = time.time()
    grid = GridSearchCV(base, {f"clf__{tcfg['param']}": tcfg["grid"]},
                        scoring=tcfg["scoring"], cv=tcfg["cv"], n_jobs=-1)
    grid.fit(X_tr, y_tr)
    mejor = grid.best_estimator_
    filas.append({"modelo": f"TF-IDF + LogReg ajustada ({grid.best_params_})",
                  **metricas(y_te, mejor.predict(X_te)), "seg": round(time.time() - t0, 1)})
    print("Mejor:", grid.best_params_, "F1 CV:", round(grid.best_score_, 4))

    res = pd.DataFrame(filas).sort_values("f1_macro", ascending=False)
    (ROOT / cfg["paths"]["reports"]).mkdir(exist_ok=True)
    res.to_csv(ROOT / cfg["paths"]["reports"] / "resultados_modelos.csv", index=False)
    pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score"]].to_csv(
        ROOT / cfg["paths"]["reports"] / "tuning_logreg_cv.csv", index=False)
    (ROOT / cfg["paths"]["models"]).mkdir(exist_ok=True)
    joblib.dump(mejor, ROOT / cfg["paths"]["models"] / "sentiment_logreg.joblib")
    print(res.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="omite LightGBM")
    main(ap.parse_args().rapido)
