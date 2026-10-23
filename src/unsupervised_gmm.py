"""Comparación no supervisada: TF-IDF -> LSA -> Gaussian Mixture.

No reemplaza el experimento de K-Means ni sobrescribe sus archivos. El GMM
produce pertenencias probabilísticas, útiles cuando los temas se solapan.

Uso: python -m src.unsupervised_gmm
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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer

from src.preprocess import ROOT, get_split, load_config


def main() -> None:
    cfg = load_config()
    u, g, seed = cfg["unsupervised"], cfg["gmm"], cfg["seed"]
    rep = ROOT / cfg["paths"]["reports"]
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    X_tr, _, y_tr, _ = get_split(cfg)

    stop = sorted(ENGLISH_STOP_WORDS - {"not", "no", "never", "nor", "cannot"})
    vec = TfidfVectorizer(
        stop_words=stop, ngram_range=(1, 2), min_df=5, max_df=0.5,
        max_features=20000, sublinear_tf=True,
    )
    T = vec.fit_transform(X_tr)
    svd = TruncatedSVD(n_components=u["svd_components"], random_state=seed)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Z = Normalizer(copy=False).fit_transform(svd.fit_transform(T))
    if not np.isfinite(Z).all():
        raise ValueError("SVD produjo valores no finitos")
    print(f"Varianza explicada por {u['svd_components']} componentes: "
          f"{svd.explained_variance_ratio_.sum():.3f}")

    filas: list[dict] = []
    modelos: dict[int, GaussianMixture] = {}
    muestra = min(u["silhouette_sample"], len(Z))
    for k in g["k_range"]:
        modelo = GaussianMixture(
            n_components=k,
            covariance_type=g["covariance_type"],
            max_iter=g["max_iter"],
            n_init=1,
            random_state=seed,
            reg_covar=1e-5,
        ).fit(Z)
        etiquetas = modelo.predict(Z)
        sil = silhouette_score(Z, etiquetas, sample_size=muestra, random_state=seed)
        filas.append({
            "k": k,
            "bic": modelo.bic(Z),
            "aic": modelo.aic(Z),
            "silueta": sil,
            "convergio": modelo.converged_,
            "iteraciones": modelo.n_iter_,
        })
        modelos[k] = modelo
        print(f"k={k:2d} BIC={filas[-1]['bic']:.0f} AIC={filas[-1]['aic']:.0f} "
              f"silueta={sil:.4f} convergió={modelo.converged_}")

    seleccion = pd.DataFrame(filas)
    seleccion.to_csv(rep / "gmm_seleccion_k.csv", index=False)
    # En BIC, menor es mejor; evita elegir solo por una silueta que crece al fragmentar.
    k_elegido = int(seleccion.loc[seleccion.bic.idxmin(), "k"])
    modelo = modelos[k_elegido]
    etiquetas = modelo.predict(Z)
    probabilidades = modelo.predict_proba(Z)
    confianza = probabilidades.max(axis=1)
    print(f"k elegido por BIC = {k_elegido}")
    print(f"Confianza media de asignación = {confianza.mean():.3f}")

    nombres = np.array(vec.get_feature_names_out())
    centros_tfidf = svd.inverse_transform(modelo.means_)
    lineas = [
        f"Componente {c}: " + ", ".join(
            nombres[np.argsort(centros_tfidf[c])[-g["top_terms"]:][::-1]]
        )
        for c in range(k_elegido)
    ]
    (rep / "gmm_top_terminos.txt").write_text("\n".join(lineas), encoding="utf-8")

    cruce = pd.crosstab(etiquetas, y_tr.to_numpy(), normalize="index").round(3)
    cruce.insert(0, "n", pd.Series(etiquetas).value_counts().sort_index().values)
    cruce.insert(1, "confianza_media", [confianza[etiquetas == c].mean()
                                         for c in cruce.index])
    cruce.index.name = "componente"
    cruce.to_csv(rep / "gmm_resumen.csv")
    print("\n".join(lineas))
    print(cruce.to_string())

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    seleccion.plot(x="k", y="bic", marker="o", ax=ax[0], legend=False, title="GMM: BIC (menor es mejor)")
    seleccion.plot(x="k", y="aic", marker="o", ax=ax[1], legend=False, title="GMM: AIC (menor es mejor)")
    seleccion.plot(x="k", y="silueta", marker="o", ax=ax[2], legend=False, title="GMM: silueta")
    plt.tight_layout()
    plt.savefig(fig_dir / "gmm_seleccion_k.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
