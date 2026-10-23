"""Gráficos de evidencia de AutoML (correr después de src.automl).

Uso:  python -m src.report_automl

Salidas: reports/figures/automl_curva.png (mejor F1 de validación vs tiempo),
         reports/figures/comparativa_final.png, reports/automl_trials.csv
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocess import ROOT, load_config


def main():
    cfg = load_config()
    rep, fig_dir = ROOT / cfg["paths"]["reports"], ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Cada línea del log de FLAML es un trial (JSON)
    trials = []
    with open(rep / "automl.log", encoding="utf-8") as f:
        for linea in f:
            try:
                d = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if "learner" in d:
                trials.append({"trial": d.get("record_id"), "estimador": d["learner"],
                               "seg": d.get("wall_clock_time"), "f1_val": 1 - d["validation_loss"],
                               "config": json.dumps(d.get("config"))})
    t = pd.DataFrame(trials)
    t.to_csv(rep / "automl_trials.csv", index=False)
    print(t.groupby("estimador")["f1_val"].agg(["count", "max"]).sort_values("max", ascending=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    for est, g in t.groupby("estimador"):
        ax.scatter(g.seg, g.f1_val, s=14, label=est, alpha=0.7)
    ax.plot(t.seg, t.f1_val.cummax(), color="black", lw=1.5, label="mejor hasta el momento")
    ax.set_xlabel("segundos de búsqueda"); ax.set_ylabel("F1 macro (validación holdout)")
    ax.set_title("FLAML: trials por estimador"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "automl_curva.png", dpi=120); plt.close()

    comp = pd.read_csv(rep / "comparativa_final.csv").sort_values("f1_macro")
    colores = ["#c97b2a" if m.startswith("AutoML") else "#3b6ea5" for m in comp.modelo]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(comp.modelo, comp.f1_macro, color=colores)
    for y, v in enumerate(comp.f1_macro):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("F1 macro (test)"); ax.set_title("Modelos manuales vs. AutoML")
    plt.tight_layout(); plt.savefig(fig_dir / "comparativa_final.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
