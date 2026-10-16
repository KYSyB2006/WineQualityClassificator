"""
app.py — petite application web qui prédit si un vin est "bon" ou "pas bon".
On la lance avec :  streamlit run app.py
L'utilisateur entre les mesures d'un vin, clique sur un bouton, et l'app affiche la prédiction.
"""
import streamlit as st     # streamlit = librairie pour faire une app web en Python, sans HTML
import pandas as pd
import joblib              # pour recharger le modèle sauvegardé par train.py

# 1. On charge le modèle, le scaler et la liste des colonnes (une seule fois, au démarrage).
#    @st.cache_resource dit à Streamlit : "garde ça en mémoire, ne le recharge pas à chaque clic".
@st.cache_resource
def charger():
    modele   = joblib.load("modele_vin.joblib")
    scaler   = joblib.load("scaler_vin.joblib")
    colonnes = joblib.load("colonnes.joblib")
    return modele, scaler, colonnes

modele, scaler, colonnes = charger()

# 2. Le titre de la page
st.title("Prédiction de la qualité d'un vin")
st.write("Entrez les mesures du vin, puis cliquez sur **Prédire**.")

# 3. Les champs de saisie. Pour chaque variable, un champ numérique avec une valeur par défaut
#    (la moyenne du jeu de données). min_value / max_value = bornes autorisées.
st.subheader("Mesures physico-chimiques")
col1, col2 = st.columns(2)   # on met les champs sur 2 colonnes, pour que ce soit plus joli

with col1:
    fixed_acidity    = st.number_input("Acidité fixe",            3.8, 16.0, 7.2)
    volatile_acidity = st.number_input("Acidité volatile",        0.08, 1.6, 0.34)
    citric_acid      = st.number_input("Acide citrique",          0.0, 1.7, 0.32)
    residual_sugar   = st.number_input("Sucre résiduel",          0.6, 66.0, 5.4)
    chlorides        = st.number_input("Chlorures",               0.009, 0.62, 0.056, format="%.3f")
    free_so2         = st.number_input("Dioxyde de soufre libre", 1.0, 289.0, 30.0)

with col2:
    total_so2        = st.number_input("Dioxyde de soufre total", 6.0, 440.0, 115.0)
    density          = st.number_input("Densité",                 0.98, 1.04, 0.995, format="%.4f")
    pH               = st.number_input("pH",                      2.7, 4.1, 3.2)
    sulphates        = st.number_input("Sulfates",                0.22, 2.0, 0.53)
    alcohol          = st.number_input("Alcool (°)",              8.0, 15.0, 10.5)

# Le type de vin : un menu déroulant, qu'on transforme en 0/1 (comme à l'entraînement)
type_vin = st.selectbox("Type de vin", ["Rouge", "Blanc"])
type_num = 1 if type_vin == "Rouge" else 0   # rouge = 1, blanc = 0

# 4. Le bouton. Tout ce qui est dans le "if" ne s'exécute que quand on clique.
if st.button("Prédire"):
    # a) On range les valeurs saisies dans un tableau, dans le MÊME ordre que les colonnes du modèle
    valeurs = {
        "fixed acidity": fixed_acidity, "volatile acidity": volatile_acidity,
        "citric acid": citric_acid, "residual sugar": residual_sugar,
        "chlorides": chlorides, "free sulfur dioxide": free_so2,
        "total sulfur dioxide": total_so2, "density": density, "pH": pH,
        "sulphates": sulphates, "alcohol": alcohol, "type": type_num,
    }
    donnee = pd.DataFrame([valeurs])[colonnes]   # [colonnes] garantit le bon ordre

    # b) On standardise avec le MÊME scaler qu'à l'entraînement, puis on prédit
    donnee_norm = scaler.transform(donnee)
    prediction  = modele.predict(donnee_norm)[0]            # 0 ou 1
    proba_bon   = modele.predict_proba(donnee_norm)[0][1]   # probabilité d'être "bon"

    # c) On affiche le résultat
    if prediction == 1:
        st.success(f"✅ Bon vin (probabilité : {proba_bon:.0%})")
    else:
        st.error(f"❌ Pas bon vin (probabilité d'être bon : {proba_bon:.0%})")
