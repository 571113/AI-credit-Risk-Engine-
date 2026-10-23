from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "data" / "processed" / "priority_summary.csv"

st.set_page_config(page_title="AI Credit Risk Engine", layout="wide")
st.title("AI Credit Risk Engine")
st.caption("Tableau de bord de priorisation du risque crédit")

if not FILE.exists():
    st.error("Fichier priority_summary.csv introuvable.")
    st.stop()

df = pd.read_csv(FILE)
total = int(df["dossiers"].sum())
weighted_probability = (
    (df["probabilite_moyenne"] * df["dossiers"]).sum() / total
    if total else 0
)
anomalies = int(df["profils_atypiques"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Dossiers analysés", f"{total:,}".replace(",", " "))
c2.metric("Probabilité moyenne estimée", f"{weighted_probability:.2%}")
c3.metric("Profils atypiques", f"{anomalies:,}".replace(",", " "))

st.subheader("Nombre de dossiers par priorité")
st.bar_chart(df.set_index("priorite")["dossiers"])

st.subheader("Risque estimé par priorité")
display_df = df.rename(columns={
    "priorite": "Priorité",
    "dossiers": "Nombre de dossiers",
    "probabilite_moyenne": "Probabilité moyenne estimée",
    "profils_atypiques": "Profils atypiques",
})
st.dataframe(
    display_df.style.format({
        "Probabilité moyenne estimée": "{:.2%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.info(
    "Les probabilités sont des estimations du modèle. "
    "Le taux de défaut réel n'est pas affiché car les étiquettes "
    "de défaut ne sont pas disponibles dans ce fichier."
)
