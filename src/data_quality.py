"""Diagnóstico reproducible de calidad y tabla de variables del dataset EC2.

Uso: python -m src.data_quality
Salidas: reports/tabla_variables.csv, reports/calidad_datos.md
"""
from __future__ import annotations

import pandas as pd

from src.preprocess import ROOT, load_config


DESCRIPCIONES = {
    "review": "Título y texto de la reseña; entrada principal del modelo.",
    "label": "Sentimiento débil derivado del rating: 1-2 negativo, 3 neutro, 4-5 positivo.",
    "rating": "Calificación original de 1 a 5 estrellas.",
    "asin": "Identificador del producto reseñado.",
    "parent_asin": "Identificador del producto padre o familia de variantes.",
    "helpful_vote": "Cantidad de votos que marcaron la reseña como útil.",
    "verified_purchase": "Indica si Amazon registró la compra como verificada.",
    "fecha": "Fecha de la reseña convertida desde el timestamp original.",
}


def main() -> None:
    cfg = load_config()
    ruta = ROOT / cfg["data"]["path"]
    rep = ROOT / cfg["paths"]["reports"]
    rep.mkdir(exist_ok=True)
    df = pd.read_parquet(ruta)

    tabla = pd.DataFrame({
        "variable": df.columns,
        "tipo": [str(df[c].dtype) for c in df.columns],
        "faltantes": [int(df[c].isna().sum()) for c in df.columns],
        "valores_unicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "descripcion": [DESCRIPCIONES.get(c, "") for c in df.columns],
    })
    tabla.to_csv(rep / "tabla_variables.csv", index=False)

    longitudes = df["review"].astype(str).str.len()
    palabras = df["review"].astype(str).str.split().str.len()
    q99_chars = float(longitudes.quantile(0.99))
    q99_words = float(palabras.quantile(0.99))
    duplicados = int(df.duplicated("review").sum())
    estrellas = df["rating"].value_counts().sort_index()
    clases = df["label"].astype(str).value_counts()

    lineas = [
        "# Diagnóstico de calidad de datos",
        "",
        "- **Fuente:** Amazon Reviews 2023 - Electronics, McAuley Lab (UCSD).",
        "- **Fecha de acceso declarada por el equipo:** 16/10/2026.",
        f"- **Subconjunto final:** {len(df):,} reseñas y {df.shape[1]} variables.",
        "- **Muestreo:** balanceado por clase después de leer por streaming aproximadamente 1,5 millones de reseñas.",
        f"- **Duplicados exactos por texto en el subconjunto:** {duplicados:,}.",
        f"- **Faltantes totales:** {int(df.isna().sum().sum()):,}.",
        f"- **Rango de fechas:** {df['fecha'].min()} a {df['fecha'].max()}.",
        f"- **Longitud mediana:** {palabras.median():.0f} palabras; percentil 99: {q99_words:.0f} palabras ({q99_chars:.0f} caracteres).",
        f"- **Reseñas sobre el percentil 99 de longitud:** {int((palabras > q99_words).sum()):,}; se consideran extremas para diagnóstico, no se eliminan automáticamente.",
        "",
        "## Distribución de la muestra final",
        "",
        *[f"- {k}: {int(v):,}" for k, v in clases.items()],
        "",
        "## Distribución por estrellas",
        "",
        *[f"- {int(k)} estrellas: {int(v):,}" for k, v in estrellas.items()],
        "",
        "## Riesgos y sesgos",
        "",
        "1. La etiqueta es débil porque se deriva de estrellas; el texto puede expresar sentimientos mixtos o no coincidir con el rating.",
        "2. El dataset original está fuertemente sesgado hacia 5 estrellas; el balanceo mejora la comparación de clases, pero ya no representa la prevalencia real.",
        "3. Las reseñas extremadamente largas pueden dominar conteos o contener varios temas; se conservan para no descartar información sin evidencia.",
        "4. Productos, fechas y usuarios no están necesariamente distribuidos de forma uniforme; los resultados no deben generalizarse a todas las categorías de Amazon.",
        "5. Los títulos automáticos como 'Five Stars' se eliminan antes del modelado porque revelan directamente la etiqueta.",
        "",
        "## Decisiones",
        "",
        "- Se eliminan duplicados antes del muestreo.",
        "- Se descartan reseñas de menos de 20 caracteres durante la construcción del subconjunto.",
        "- No se imputan textos; el subconjunto final no presenta faltantes en la entrada principal.",
        "- La partición es estratificada y fija. Todo transformador aprendido se ajusta solo con entrenamiento.",
    ]
    (rep / "calidad_datos.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
