"""Carga de datos, limpieza de texto y partición train/test compartida.

Uso:  python -m src.preprocess      (crea data/split.csv si no existe)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
NEG = re.compile(r"n't\b")
# Amazon pone "Five Stars", "One Star", etc. como título automático cuando el usuario no
# escribe uno: eso es la etiqueta escrita dentro del texto (data leakage). Se quita.
TITULO_ESTRELLAS = re.compile(r"^\s*(one|two|three|four|five) stars?\b[.!]*\s*", re.I)


def load_config(path: str | os.PathLike = ROOT / "config" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def limpiar(t: str) -> str:
    """Minúsculas, sin URLs ni números, conservando negaciones (don't -> do not) y ! ?.

    Debe ser idéntica a la del notebook 02 y a la de parte2/streamlit_app.py.
    """
    t = t.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = NEG.sub(" not", t)
    t = re.sub(r"[^a-z\s!?']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / cfg["data"]["path"])
    df[cfg["data"]["label_col"]] = df[cfg["data"]["label_col"]].astype(str)
    texto = df[cfg["data"]["text_col"]]
    if cfg["data"].get("quitar_titulo_estrellas", True):
        texto = texto.str.replace(TITULO_ESTRELLAS, "", regex=True)
    df["clean"] = texto.map(limpiar)
    return df


def get_split(cfg: dict, df: pd.DataFrame | None = None):
    """Devuelve X_train, X_test, y_train, y_test siempre con la misma partición.

    La primera vez guarda los índices en data/split.csv; después los reutiliza, así
    los modelos manuales, AutoML y el análisis de errores comparan sobre el mismo test.
    """
    df = load_data(cfg) if df is None else df
    split_path = ROOT / cfg["data"]["split_file"]
    y = df[cfg["data"]["label_col"]]
    if split_path.exists():
        split = pd.read_csv(split_path, index_col=0)["split"]
    else:
        tr, te = train_test_split(df.index, test_size=cfg["data"]["test_size"],
                                  stratify=y, random_state=cfg["seed"])
        split = pd.Series("train", index=df.index, name="split")
        split.loc[te] = "test"
        split.to_frame().to_csv(split_path)
    tr = split[split == "train"].index
    te = split[split == "test"].index
    return df.loc[tr, "clean"], df.loc[te, "clean"], y.loc[tr], y.loc[te]


if __name__ == "__main__":
    cfg = load_config()
    X_tr, X_te, y_tr, y_te = get_split(cfg)
    print(f"train={len(X_tr)}  test={len(X_te)}")
    print(y_tr.value_counts().to_string())
