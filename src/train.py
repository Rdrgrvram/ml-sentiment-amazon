"""Baseline + 3 modelos clásicos + tuning, con selección por VALIDACIÓN (el test no se usa para elegir).

Protocolo (train / validación / test = 64 / 16 / 20 %, estratificado, semilla fija):
  1. train se divide en sub-train y validación (data/split.csv ya separa el test).
  2. Cada modelo y cada valor de C de la LogReg se entrena en sub-train y se evalúa en validación.
  3. Se elige el modelo con mejor F1 macro de validación.
  4. Solo después, cada modelo se reentrena con todo train y se evalúa UNA vez en test (reporte final).

Uso:  python -m src.train            (todos los modelos, ≈ 20 min por LightGBM)
      python -m src.train --rapido   (sin LightGBM)

Salidas: reports/resultados_modelos.csv, reports/tuning_logreg_val.csv,
         models/sentiment_logreg.joblib (modelo elegido, reentrenado con todo train)
"""
from __future__ import annotations

import argparse
import time

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import ROOT, get_split, load_config


def tfidf(cfg: dict, max_features: int | None = None) -> TfidfVectorizer:
    t = cfg["tfidf"]
    return TfidfVectorizer(ngram_range=tuple(t["ngram_range"]), min_df=t["min_df"],
                           max_df=t["max_df"], sublinear_tf=t["sublinear_tf"],
                           max_features=max_features or t["max_features"])


def logreg(cfg: dict, C: float) -> Pipeline:
    return Pipeline([("tfidf", tfidf(cfg)),
                     ("clf", LogisticRegression(C=C,
                                                max_iter=cfg["models"]["logreg"]["max_iter"],
                                                random_state=cfg["seed"]))])


def build_models(cfg: dict, rapido: bool = False) -> dict[str, Pipeline]:
    m, seed = cfg["models"], cfg["seed"]
    # El TF-IDF va dentro del Pipeline: se ajusta solo con los datos de entrenamiento (evita leakage).
    modelos = {
        "Dummy (baseline)": Pipeline([("tfidf", tfidf(cfg, 1000)),
                                      ("clf", DummyClassifier(strategy=m["dummy"]["strategy"]))]),
        "TF-IDF + SVM lineal": Pipeline([("tfidf", tfidf(cfg)), ("clf", LinearSVC(C=m["svm"]["C"]))]),
    }
    if not rapido:
        lg = m["lightgbm"]
        modelos["TF-IDF + LightGBM"] = Pipeline([
            ("tfidf", tfidf(cfg, lg["max_features"])),
            ("clf", LGBMClassifier(n_estimators=lg["n_estimators"], learning_rate=lg["learning_rate"],
                                   num_leaves=lg["num_leaves"], random_state=seed,
                                   n_jobs=lg.get("n_jobs", 2), verbose=-1))])
    return modelos


def metricas(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_neutro": f1_score(y_true, y_pred, labels=["neutro"], average="macro")}


def main(rapido: bool = False):
    cfg = load_config()
    X_tr, X_te, y_tr, y_te = get_split(cfg)
    val_size = cfg["data"].get("val_size", 0.2)
    X_sub, X_val, y_sub, y_val = train_test_split(X_tr, y_tr, test_size=val_size,
                                                  stratify=y_tr, random_state=cfg["seed"])
    print(f"sub-train={len(X_sub)}  validación={len(X_val)}  test={len(X_te)} (test no se usa para elegir)")

    # 1) Tuning de C de la LogReg en validación
    tun = []
    for C in cfg["tuning"]["grid"]:
        p = logreg(cfg, C).fit(X_sub, y_sub).predict(X_val)
        tun.append({"C": C, "f1_val": f1_score(y_val, p, average="macro")})
        print(f"LogReg C={C:<4} F1 val = {tun[-1]['f1_val']:.4f}")
    tun = pd.DataFrame(tun)
    C_best = float(tun.loc[tun.f1_val.idxmax(), "C"])

    candidatos = build_models(cfg, rapido)
    candidatos[f"TF-IDF + LogReg (C={C_best:g}, ajustada en val)"] = logreg(cfg, C_best)

    # 2) Selección en validación y 3) reporte final en test
    filas, finales = [], {}
    for nombre, modelo in candidatos.items():
        t0 = time.time()
        m_val = clone(modelo).fit(X_sub, y_sub)
        f1_val = f1_score(y_val, m_val.predict(X_val), average="macro")
        m_fin = clone(modelo).fit(X_tr, y_tr)          # reentrenado con todo train
        finales[nombre] = m_fin
        filas.append({"modelo": nombre, "f1_val": f1_val, **metricas(y_te, m_fin.predict(X_te)),
                      "seg": round(time.time() - t0, 1)})
        print(f"{nombre:40s} F1 val = {f1_val:.4f}  F1 test = {filas[-1]['f1_macro']:.4f}")

    res = pd.DataFrame(filas).sort_values("f1_val", ascending=False)
    elegido = res.iloc[0].modelo
    res["elegido_por_validacion"] = res.modelo == elegido
    print(f"\nElegido por validación: {elegido}")

    rep = ROOT / cfg["paths"]["reports"]; rep.mkdir(exist_ok=True)
    res.to_csv(rep / "resultados_modelos.csv", index=False)
    tun.to_csv(rep / "tuning_logreg_val.csv", index=False)
    (ROOT / cfg["paths"]["models"]).mkdir(exist_ok=True)
    joblib.dump(finales[elegido], ROOT / cfg["paths"]["models"] / "sentiment_logreg.joblib")
    print(res.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="omite LightGBM")
    main(ap.parse_args().rapido)
