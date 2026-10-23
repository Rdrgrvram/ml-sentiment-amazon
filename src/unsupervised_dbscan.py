"""Comparación no supervisada: TF-IDF -> LSA -> DBSCAN.

DBSCAN se ejecuta en una muestra estratificada para limitar memoria y tiempo.
No sobrescribe resultados de K-Means ni GMM.

Uso: python -m src.unsupervised_dbscan
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer

from src.preprocess import ROOT, get_split, load_config


def muestra_estratificada(X, y, n: int, seed: int):
    """Devuelve una muestra reproducible que conserva la proporción de clases."""
    if n >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True)
    X_m, _, y_m, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return X_m.reset_index(drop=True), y_m.reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    u, d, seed = cfg["unsupervised"], cfg["dbscan"], cfg["seed"]
    rep = ROOT / cfg["paths"]["reports"]
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    X_tr, _, y_tr, _ = get_split(cfg)
    X, y = muestra_estratificada(X_tr, y_tr, d["sample_size"], seed)
    print(f"Muestra estratificada: {len(X)} reseñas")

    stop = sorted(ENGLISH_STOP_WORDS - {"not", "no", "never", "nor", "cannot"})
    vec = TfidfVectorizer(
        stop_words=stop, ngram_range=(1, 2), min_df=3, max_df=0.5,
        max_features=20000, sublinear_tf=True,
    )
    T = vec.fit_transform(X)
    componentes = min(u["svd_components"], T.shape[1] - 1)
    svd = TruncatedSVD(n_components=componentes, random_state=seed)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Z = Normalizer(copy=False).fit_transform(svd.fit_transform(T))
    if not np.isfinite(Z).all():
        raise ValueError("SVD produjo valores no finitos")
    print(f"Varianza explicada por {componentes} componentes: "
          f"{svd.explained_variance_ratio_.sum():.3f}")

    # En vectores normalizados, la distancia euclídea conserva el orden de la
    # distancia coseno y resulta más directa para DBSCAN.
    nn = NearestNeighbors(n_neighbors=d["min_samples"], metric="euclidean", n_jobs=-1)
    distancias, _ = nn.fit(Z).kneighbors(Z)
    distancia_k = distancias[:, -1]
    eps_values = sorted({float(np.quantile(distancia_k, q)) for q in d["eps_quantiles"]})

    filas: list[dict] = []
    resultados: dict[float, np.ndarray] = {}
    for eps in eps_values:
        etiquetas = DBSCAN(
            eps=eps,
            min_samples=d["min_samples"],
            metric="euclidean",
            n_jobs=-1,
        ).fit_predict(Z)
        mascara = etiquetas != -1
        clusters = len(set(etiquetas[mascara]))
        ruido = float((~mascara).mean())
        sil = np.nan
        if clusters >= 2 and mascara.sum() > clusters:
            sil = silhouette_score(Z[mascara], etiquetas[mascara], metric="euclidean")
        cobertura = 1.0 - ruido
        # Puntaje auxiliar: premia separación sin ignorar cuántos puntos cubre.
        puntaje = sil * cobertura if np.isfinite(sil) else np.nan
        filas.append({
            "eps": eps,
            "min_samples": d["min_samples"],
            "clusters": clusters,
            "ruido_pct": ruido * 100,
            "cobertura": cobertura,
            "silueta_sin_ruido": sil,
            "silueta_x_cobertura": puntaje,
        })
        resultados[eps] = etiquetas
        sil_txt = f"{sil:.4f}" if np.isfinite(sil) else "no aplica"
        print(f"eps={eps:.4f} clusters={clusters:3d} ruido={ruido:6.1%} "
              f"silueta_sin_ruido={sil_txt}")

    seleccion = pd.DataFrame(filas)
    seleccion.to_csv(rep / "dbscan_seleccion_eps.csv", index=False)
    validos = seleccion[
        seleccion["silueta_x_cobertura"].notna() &
        seleccion["clusters"].between(2, 30)
    ]
    if validos.empty:
        print("DBSCAN no encontró una solución interpretable (2-30 clusters).")
        print("La tabla de diagnóstico se guardó igualmente.")
    else:
        mejor = validos.loc[validos["silueta_x_cobertura"].idxmax()]
        eps = float(mejor["eps"])
        etiquetas = resultados[eps]
        print(f"eps elegido = {eps:.4f}")

        nombres = np.array(vec.get_feature_names_out())
        lineas = []
        resumen = []
        for cluster in sorted(set(etiquetas) - {-1}):
            mascara = etiquetas == cluster
            centro = np.asarray(T[mascara].mean(axis=0)).ravel()
            terminos = nombres[np.argsort(centro)[-d["top_terms"]:][::-1]]
            lineas.append(f"Cluster {cluster}: " + ", ".join(terminos))
            proporciones = y[mascara].value_counts(normalize=True)
            resumen.append({
                "cluster": cluster,
                "n": int(mascara.sum()),
                "negativo": proporciones.get("negativo", 0.0),
                "neutro": proporciones.get("neutro", 0.0),
                "positivo": proporciones.get("positivo", 0.0),
            })
        (rep / "dbscan_top_terminos.txt").write_text("\n".join(lineas), encoding="utf-8")
        pd.DataFrame(resumen).to_csv(rep / "dbscan_resumen.csv", index=False)
        print("\n".join(lineas))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    seleccion.plot(x="eps", y="clusters", marker="o", ax=ax[0], legend=False,
                   title="DBSCAN: número de clusters")
    seleccion.plot(x="eps", y="ruido_pct", marker="o", ax=ax[1], legend=False,
                   title="DBSCAN: porcentaje de ruido")
    seleccion.plot(x="eps", y="silueta_sin_ruido", marker="o", ax=ax[2], legend=False,
                   title="DBSCAN: silueta sin ruido")
    plt.tight_layout()
    plt.savefig(fig_dir / "dbscan_seleccion_eps.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
