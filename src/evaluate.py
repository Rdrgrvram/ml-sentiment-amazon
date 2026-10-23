"""Evaluación y análisis de errores del modelo elegido (LogReg ajustada).

Uso:  python -m src.evaluate

Salidas: reports/figures/confusion.png, reports/classification_report.txt,
         reports/errores_logreg.csv, reports/palabras_por_clase.txt
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.preprocess import ROOT, get_split, load_config


def main():
    cfg = load_config()
    rep, fig_dir = ROOT / cfg["paths"]["reports"], ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    _, X_te, _, y_te = get_split(cfg)
    modelo = joblib.load(ROOT / cfg["paths"]["models"] / "sentiment_logreg.joblib")
    pred = modelo.predict(X_te)

    txt = classification_report(y_te, pred, digits=3)
    (rep / "classification_report.txt").write_text(txt, encoding="utf-8")
    print(txt)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax[0], cmap="Blues", colorbar=False)
    ax[0].set_title("LogReg ajustada — conteos")
    ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax[1], cmap="Blues", normalize="true",
                                            values_format=".2f", colorbar=False)
    ax[1].set_title("Normalizada por fila (recall)")
    plt.tight_layout(); plt.savefig(fig_dir / "confusion.png", dpi=120); plt.close()

    # Interpretabilidad: palabras con más peso por clase
    vec, clf = modelo.named_steps["tfidf"], modelo.named_steps["clf"]
    nombres = np.array(vec.get_feature_names_out())
    lineas = [f"{c.upper():9s} " + ", ".join(nombres[np.argsort(clf.coef_[i])[-15:][::-1]])
              for i, c in enumerate(clf.classes_)]
    (rep / "palabras_por_clase.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))

    # Errores: qué confunde y con qué confianza
    proba = modelo.predict_proba(X_te).max(axis=1)
    err = pd.DataFrame({"texto": X_te.values, "real": y_te.values, "pred": pred, "confianza": proba})
    err = err[err.real != err.pred].sort_values("confianza", ascending=False)
    err.to_csv(rep / "errores_logreg.csv", index=False)
    print("\nErrores por tipo:\n", err.groupby(["real", "pred"]).size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
