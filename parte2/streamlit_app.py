"""Demo: análisis de sentimiento + búsqueda semántica en reseñas de electrónica.
Ejecutar desde la raíz del proyecto:  streamlit run app/streamlit_app.py
Requiere: models/sentiment_logreg.joblib (notebook 02), data/electronics_50k.parquet (01)
y data/embeddings_minilm.npy (03)."""
import re
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
NEG = re.compile(r"n't\b")

def limpiar(t: str) -> str:          # misma limpieza que el notebook 02
    t = t.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = NEG.sub(" not", t)
    t = re.sub(r"[^a-z\s!?']", " ", t)
    return re.sub(r"\s+", " ", t).strip()

@st.cache_resource
def cargar_clasificador():
    return joblib.load(RAIZ / "models/sentiment_logreg.joblib")

@st.cache_resource
def cargar_buscador():
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    df = pd.read_parquet(RAIZ / "data/electronics_50k.parquet")
    df["label"] = df["label"].astype(str)
    E = np.load(RAIZ / "data/embeddings_minilm.npy")
    return modelo, df, E

st.set_page_config(page_title="Reseñas de electrónica", page_icon="💬", layout="wide")
st.title("💬 Sentimiento y búsqueda semántica en reseñas de electrónica")
st.caption("Amazon Reviews 2023 · Electronics · muestra balanceada de 50 000 reseñas")

tab1, tab2 = st.tabs(["Clasificar una reseña", "Buscar reseñas"])
COLORES = {"negativo": "🔴", "neutro": "🟡", "positivo": "🟢"}

with tab1:
    texto = st.text_area("Escribe una reseña (en inglés)",
                         "The battery died after two days, totally not worth the money")
    if st.button("Clasificar", type="primary") and texto.strip():
        clf = cargar_clasificador()
        proba = clf.predict_proba([limpiar(texto)])[0]
        clases = list(clf.classes_)
        pred = clases[int(np.argmax(proba))]
        st.subheader(f"{COLORES.get(pred, '')} {pred.capitalize()}")
        st.bar_chart(pd.Series(proba, index=clases, name="probabilidad"))

with tab2:
    c1, c2, c3 = st.columns([3, 1, 1])
    consulta = c1.text_input("Consulta", "doesn't hold a charge")
    filtro = c2.selectbox("Sentimiento", ["todos", "negativo", "neutro", "positivo"])
    k = c3.slider("Resultados", 3, 20, 5)
    if consulta.strip():
        modelo, df, E = cargar_buscador()
        q = modelo.encode([consulta], normalize_embeddings=True)[0]
        sims = E @ q
        if filtro != "todos":
            sims = np.where(df["label"].values == filtro, sims, -1)
        idx = np.argsort(-sims)[:k]
        for i in idx:
            r = df.iloc[i]
            with st.container(border=True):
                st.markdown(f"**{COLORES.get(r['label'], '')} {r['label']}** · {int(r['rating'])}★ · similitud {sims[i]:.3f}")
                st.write(r["review"][:600])
