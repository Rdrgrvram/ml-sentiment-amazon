"""Aprendizaje no supervisado: TF-IDF -> TruncatedSVD (LSA) -> KMeans.

Uso:  python -m src.unsupervised           (elige k por silueta)
      python -m src.unsupervised --k 6     (fija k)

Se ajusta solo con train y NO usa la etiqueta para agrupar: la etiqueta se usa
después, solo para interpretar (cruce cluster x sentimiento).
Salidas: reports/figures/kmeans_seleccion_k.png, reports/figures/clusters_svd2d.png,
         reports/clusters_resumen.csv, reports/clusters_top_terminos.txt
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from src.preprocess import ROOT, get_split, load_config


def main(k_fijo: int | None = None):
    cfg = load_config()
    u, seed = cfg["unsupervised"], cfg["seed"]
    rep, fig_dir = ROOT / cfg["paths"]["reports"], ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    X_tr, _, y_tr, _ = get_split(cfg)

    # Para agrupar por TEMA se quitan stopwords (conservando negaciones); si no, los
    # clusters se forman por palabras vacías ("the", "it", "to").
    stop = sorted(ENGLISH_STOP_WORDS - {"not", "no", "never", "nor", "cannot"})
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, max_df=0.5,
                          max_features=20000, sublinear_tf=True)
    T = vec.fit_transform(X_tr)
    svd = TruncatedSVD(n_components=u["svd_components"], random_state=seed)
    Z = Normalizer(copy=False).fit_transform(svd.fit_transform(T))   # LSA normalizado ~ coseno
    print(f"Varianza explicada por {u['svd_components']} componentes: {svd.explained_variance_ratio_.sum():.3f}")

    # Selección de k: inercia (codo) + silueta
    filas = []
    for k in u["k_range"]:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=u["silhouette_sample"], random_state=seed)
        filas.append({"k": k, "inercia": km.inertia_, "silueta": sil})
        print(f"k={k:2d} inercia={km.inertia_:.0f} silueta={sil:.4f}")
    sel = pd.DataFrame(filas)
    sel.to_csv(rep / "kmeans_seleccion_k.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sel.plot(x="k", y="inercia", marker="o", ax=ax[0], legend=False, title="Codo (inercia)")
    sel.plot(x="k", y="silueta", marker="o", ax=ax[1], legend=False, title="Silueta")
    plt.tight_layout(); plt.savefig(fig_dir / "kmeans_seleccion_k.png", dpi=120); plt.close()

    k = k_fijo or int(sel.loc[sel.silueta.idxmax(), "k"])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    print(f"k elegido = {k}")

    # Interpretación: términos más representativos de cada centroide (vuelta al espacio TF-IDF)
    centros = svd.inverse_transform(km.cluster_centers_)
    nombres = np.array(vec.get_feature_names_out())
    lineas = [f"Cluster {c}: " + ", ".join(nombres[np.argsort(centros[c])[-u['top_terms']:][::-1]])
              for c in range(k)]
    (rep / "clusters_top_terminos.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))

    # Cruce con la etiqueta (solo para interpretar, no se usó para agrupar)
    cruce = pd.crosstab(km.labels_, y_tr.values, normalize="index").round(3)
    cruce.insert(0, "n", pd.Series(km.labels_).value_counts().sort_index().values)
    cruce.index.name = "cluster"
    cruce.to_csv(rep / "clusters_resumen.csv")
    print(cruce.to_string())

    # Vista 2D (2 primeras componentes) sobre una muestra
    idx = np.random.default_rng(seed).choice(len(Z), size=min(5000, len(Z)), replace=False)
    plt.figure(figsize=(7, 5))
    plt.scatter(Z[idx, 0], Z[idx, 1], c=km.labels_[idx], s=4, cmap="tab10")
    plt.xlabel("Componente 1"); plt.ylabel("Componente 2"); plt.title(f"KMeans (k={k}) sobre LSA")
    plt.tight_layout(); plt.savefig(fig_dir / "clusters_svd2d.png", dpi=120); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int)
    main(ap.parse_args().k)
