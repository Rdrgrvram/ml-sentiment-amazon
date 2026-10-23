"""Validaciones rápidas de integridad antes de entregar.

Uso: python -m src.validate_project
No entrena modelos ni modifica resultados.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocess import ROOT, load_config


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"OK  {message}")


def main() -> None:
    cfg = load_config()
    requeridos = [
        "data/electronics_50k.parquet",
        "data/split.csv",
        "models/sentiment_logreg.joblib",
        "reports/resultados_modelos.csv",
        "reports/comparativa_final.csv",
        "reports/classification_report.txt",
        "reports/errores_logreg.csv",
        "reports/tabla_variables.csv",
        "reports/calidad_datos.md",
        "reports/no_supervisado_seleccion.csv",
        "reports/no_supervisado_resumen.csv",
        "reports/no_supervisado_top_terminos.txt",
        "reports/figures/no_supervisado_final.png",
        "reports/automl_resultado.json",
    ]
    faltantes = [p for p in requeridos if not (ROOT / p).exists()]
    require(not faltantes, f"artefactos requeridos presentes (faltan: {faltantes})")

    df = pd.read_parquet(ROOT / cfg["data"]["path"])
    require(len(df) == 49_998, "dataset final con 49 998 filas")
    require(df["review"].isna().sum() == 0, "texto sin faltantes")
    require(df.duplicated("review").sum() == 0, "texto sin duplicados exactos")
    require(set(df["label"].astype(str)) == {"negativo", "neutro", "positivo"},
            "tres clases esperadas")

    split = pd.read_csv(ROOT / cfg["data"]["split_file"], index_col=0)["split"]
    require(len(split) == len(df), "split cubre todas las filas")
    require(set(split) == {"train", "test"}, "split contiene solo train/test")
    require(int((split == "test").sum()) == 10_000, "test fijo de 10 000 filas")

    modelos = pd.read_csv(ROOT / "reports/resultados_modelos.csv")
    elegidos = modelos[modelos["elegido_por_validacion"].astype(str).str.lower() == "true"]
    require(len(elegidos) == 1, "exactamente un modelo elegido por validación")
    require("LogReg" in elegidos.iloc[0]["modelo"], "modelo manual elegido es LogReg")

    ns = pd.read_csv(ROOT / "reports/no_supervisado_seleccion.csv")
    fila = ns.loc[ns.min_cluster_size == cfg["umap_hdbscan"]["selected_min_cluster_size"]]
    require(len(fila) == 1, "configuración final no supervisada presente")
    require(int(fila.iloc[0].clusters) == 30, "resultado final no supervisado tiene 30 clusters")

    automl = json.loads((ROOT / "reports/automl_resultado.json").read_text(encoding="utf-8"))
    require(automl.get("mejor_estimador") is not None, "AutoML registra mejor estimador")
    if "protocolo" in automl:
        require("64/16/20" in automl["protocolo"], "AutoML registra protocolo sin leakage")
    else:
        print("AVISO  AutoML es un resultado histórico; volver a ejecutar src.automl.")

    print("\nVALIDACIÓN COMPLETA")


if __name__ == "__main__":
    main()
