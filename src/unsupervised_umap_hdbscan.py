"""TF-IDF -> SVD -> UMAP -> HDBSCAN para estructura temática no lineal.

UMAP se usa con 10 dimensiones para clustering, no solo como dibujo 2D.
HDBSCAN puede dejar reseñas ambiguas como ruido y no requiere fijar k.

Uso:
    pip install umap-learn hdbscan
    python -m src.unsupervised_umap_hdbscan
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

try:
    import hdbscan
    import umap
except ImportError as exc:
    raise SystemExit(
        "Faltan dependencias. Ejecuta: pip install umap-learn hdbscan"
    ) from exc

from src.preprocess import ROOT, get_split, load_config


def muestra_estratificada(X, y, n: int, seed: int):
    if n >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True)
    X_m, _, y_m, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return X_m.reset_index(drop=True), y_m.reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    c, seed = cfg["umap_hdbscan"], cfg["seed"]
    rep = ROOT / cfg["paths"]["reports"]
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    X_tr, _, y_tr, _ = get_split(cfg)
    X, y = muestra_estratificada(X_tr, y_tr, c["sample_size"], seed)
    print(f"Muestra estratificada: {len(X)} reseñas")

    stop = sorted(ENGLISH_STOP_WORDS - {"not", "no", "never", "nor", "cannot"})
    vec = TfidfVectorizer(
        stop_words=stop,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.5,
        max_features=20000,
        sublinear_tf=True,
    )
    T = vec.fit_transform(X)
    componentes = min(c["svd_components"], T.shape[1] - 1)
    svd = TruncatedSVD(n_components=componentes, random_state=seed)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Z = Normalizer(copy=False).fit_transform(svd.fit_transform(T))
    if not np.isfinite(Z).all():
        raise ValueError("SVD produjo valores no finitos")
    print(f"Varianza explicada por SVD({componentes}): "
          f"{svd.explained_variance_ratio_.sum():.3f}")

    reducer = umap.UMAP(
        n_components=c["umap_components"],
        n_neighbors=c["n_neighbors"],
        min_dist=c["min_dist"],
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        low_memory=True,
    )
    U = reducer.fit_transform(Z)
    print(f"UMAP generado: {U.shape[1]} dimensiones")

    filas: list[dict] = []
    resultados: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tam in c["min_cluster_sizes"]:
        modelo = hdbscan.HDBSCAN(
            min_cluster_size=tam,
            min_samples=c["min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        etiquetas = modelo.fit_predict(U)
        mascara = etiquetas != -1
        clusters = len(set(etiquetas[mascara]))
        ruido = float((~mascara).mean())
        sil = np.nan
        if clusters >= 2 and mascara.sum() > clusters:
            n_sil = min(c["silhouette_sample"], int(mascara.sum()))
            sil = silhouette_score(
                U[mascara], etiquetas[mascara], sample_size=n_sil, random_state=seed
            )
        persistencia = (
            float(np.mean(modelo.cluster_persistence_))
            if len(modelo.cluster_persistence_) else np.nan
        )
        cobertura = 1.0 - ruido
        puntaje = sil * cobertura if np.isfinite(sil) else np.nan
        filas.append({
            "min_cluster_size": tam,
            "min_samples": c["min_samples"],
            "clusters": clusters,
            "ruido_pct": ruido * 100,
            "cobertura": cobertura,
            "silueta_sin_ruido": sil,
            "silueta_x_cobertura": puntaje,
            "persistencia_media": persistencia,
            "probabilidad_media_asignados": (
                float(modelo.probabilities_[mascara].mean()) if mascara.any() else np.nan
            ),
        })
        resultados[tam] = (etiquetas, modelo.probabilities_)
        sil_txt = f"{sil:.4f}" if np.isfinite(sil) else "no aplica"
        print(f"min_cluster_size={tam:3d} clusters={clusters:3d} "
              f"ruido={ruido:6.1%} silueta={sil_txt} "
              f"persistencia={persistencia:.3f}")

    seleccion = pd.DataFrame(filas)
    seleccion.to_csv(rep / "umap_hdbscan_seleccion.csv", index=False)
    validos = seleccion[
        seleccion["silueta_x_cobertura"].notna()
        & seleccion["clusters"].between(2, 30)
    ]
    if validos.empty:
        print("UMAP + HDBSCAN no encontró una solución interpretable de 2-30 clusters.")
        print("La tabla de diagnóstico se guardó igualmente.")
        return

    tam_config = c.get("selected_min_cluster_size")
    if tam_config is not None and tam_config in resultados:
        tam = int(tam_config)
    else:
        mejor = validos.loc[validos["silueta_x_cobertura"].idxmax()]
        tam = int(mejor["min_cluster_size"])
    etiquetas, probabilidades = resultados[tam]
    print(f"Configuración elegida: min_cluster_size={tam}")

    nombres = np.array(vec.get_feature_names_out())
    lineas: list[str] = []
    resumen: list[dict] = []
    for cluster in sorted(set(etiquetas) - {-1}):
        mascara = etiquetas == cluster
        centro = np.asarray(T[mascara].mean(axis=0)).ravel()
        terminos = nombres[np.argsort(centro)[-c["top_terms"]:][::-1]]
        lineas.append(f"Cluster {cluster}: " + ", ".join(terminos))
        proporciones = y[mascara].value_counts(normalize=True)
        resumen.append({
            "cluster": cluster,
            "n": int(mascara.sum()),
            "probabilidad_media": float(probabilidades[mascara].mean()),
            "negativo": proporciones.get("negativo", 0.0),
            "neutro": proporciones.get("neutro", 0.0),
            "positivo": proporciones.get("positivo", 0.0),
        })
    (rep / "umap_hdbscan_top_terminos.txt").write_text(
        "\n".join(lineas), encoding="utf-8"
    )
    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(rep / "umap_hdbscan_resumen.csv", index=False)
    # Salidas canónicas del método final para informe y presentación.
    seleccion.to_csv(rep / "no_supervisado_seleccion.csv", index=False)
    resumen_df.to_csv(rep / "no_supervisado_resumen.csv", index=False)
    (rep / "no_supervisado_top_terminos.txt").write_text(
        "\n".join(lineas), encoding="utf-8"
    )
    print("\n".join(lineas))

    # Proyección 2D separada y solo exploratoria; no se usa para entrenar HDBSCAN.
    U2 = umap.UMAP(
        n_components=2,
        n_neighbors=c["n_neighbors"],
        min_dist=c["min_dist"],
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        low_memory=True,
    ).fit_transform(Z)
    colores = etiquetas.astype(float)
    colores[etiquetas == -1] = np.nan
    plt.figure(figsize=(8, 6))
    plt.scatter(U2[etiquetas == -1, 0], U2[etiquetas == -1, 1],
                s=3, c="lightgray", alpha=0.35, label="ruido")
    plt.scatter(U2[etiquetas != -1, 0], U2[etiquetas != -1, 1],
                s=4, c=colores[etiquetas != -1], cmap="tab20", alpha=0.75)
    plt.title(f"UMAP 2D exploratorio; HDBSCAN ({len(set(etiquetas) - {-1})} clusters)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(fig_dir / "umap_hdbscan_2d.png", dpi=140)
    plt.savefig(fig_dir / "no_supervisado_final.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
