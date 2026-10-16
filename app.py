"""
app.py — application web qui prédit si un vin est "bon" (note >= 7) ou "pas bon".
On la lance avec :  streamlit run app.py

Elle utilise les fichiers créés par train.py :
    modele_vin.joblib, scaler_vin.joblib, colonnes.joblib
et les deux fichiers de données (winequality-red.csv, winequality-white.csv).

4 pages (menu en haut) : Prédire · Les données · Le modèle · À propos.
"""
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 0. Réglages de la page (toujours la PREMIÈRE commande Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Qualité du vin", page_icon="🍷", layout="wide")

# Palette de la maquette : bordeaux = marque / "pas bon", vert = "bon"
BORDEAUX, VERT, OR = "#6E1A33", "#1E6B57", "#D9A62E"
GRIS_PAS_BON = "#8E827E"
BLEU, ORANGE = "#3B6EA5", "#B5562B"       # corrélations positives / négatives
ENCRE, TEXTE_2 = "#221418", "#5A4A4E"
SEUIL = 0.5                              # seuil de décision du modèle (predict)

# ---------------------------------------------------------------------------
# 1. Style (polices, cartes, badges, jauge, verdict qui reste visible)
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Fraunces', Georgia, serif !important; color: var(--acc-titre) !important; font-weight: 600 !important; }}
.block-container {{ max-width: 1280px; padding-top: 2.5rem; }}

.titre-etape {{ display:flex; align-items:center; gap:10px; font-family:'Fraunces', Georgia, serif;
               font-size:22px; font-weight:600; color:var(--acc-titre); margin:0 0 6px; }}
.puce {{ width:28px; height:28px; border-radius:999px; background:var(--acc-plein); color:#fff;
         font:600 14px 'IBM Plex Sans', sans-serif; display:inline-flex; align-items:center; justify-content:center; }}
.groupe {{ font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:{TEXTE_2};
           font-weight:600; margin:18px 0 0; }}
.nom-mesure {{ font-size:15px; font-weight:500; color:{ENCRE}; }}
.unite {{ font-size:13px; color:{TEXTE_2}; font-weight:400; }}
.aide {{ font-size:14px; color:{TEXTE_2}; margin:0 0 4px; }}

.badge {{ display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px;
          font-size:13px; font-weight:600; }}
.badge-bon {{ background:#E2F0EA; color:{VERT}; }}
.badge-serre {{ background:#EFE8E5; color:#4A3B3F; }}
.badge-pasbon {{ background:var(--acc-fond); color:var(--acc-texte); }}
.pourcent {{ font-family:'Fraunces', Georgia, serif; font-size:56px; font-weight:600; line-height:1;
             font-variant-numeric:tabular-nums; }}
.jauge {{ position:relative; height:12px; border-radius:6px; background:#E3DBD8; border:1px solid #948884;
          box-sizing:border-box; margin-top:18px; }}
.jauge-plein {{ position:absolute; left:-1px; top:-1px; bottom:-1px; border-radius:6px; }}
.jauge-seuil {{ position:absolute; left:50%; top:-7px; width:2px; height:24px; background:{ENCRE}; }}
.jauge-legende {{ display:flex; justify-content:space-between; font-size:12px; color:{TEXTE_2}; margin-top:8px; }}

.ligne-raison {{ display:flex; gap:10px; align-items:flex-start; font-size:15px; line-height:1.4; margin:0 0 10px; }}
.rond-ok {{ flex:none; width:24px; height:24px; border-radius:999px; background:{VERT};
            display:inline-flex; align-items:center; justify-content:center; }}
.rond-alerte {{ flex:none; width:24px; height:24px; border-radius:999px; border:2px solid var(--acc-plein);
                box-sizing:border-box; display:inline-flex; align-items:center; justify-content:center; }}
.etsi {{ display:flex; justify-content:space-between; gap:12px; padding:12px 14px; background:#fff;
         border-radius:10px; font-size:15px; margin-bottom:8px; }}
.petit {{ font-size:13px; color:{TEXTE_2}; line-height:1.5; }}

.tuile {{ border:1px solid #E6DEDB; border-radius:16px; padding:20px; }}
.tuile-titre {{ font-size:14px; color:{TEXTE_2}; }}
.tuile-valeur {{ font-family:'Fraunces', Georgia, serif; font-size:40px; font-weight:600; color:var(--acc-titre);
                 font-variant-numeric:tabular-nums; }}
.tuile-sous {{ font-size:13px; color:{TEXTE_2}; }}

/* Le panneau de verdict reste visible pendant le défilement (écran large uniquement) */
@media (min-width: 960px) {{
  [data-testid="stColumn"]:has(.st-key-verdict) {{ position: sticky; top: 4rem; align-self: flex-start;
      max-height: calc(100vh - 5rem); overflow-y: auto; overscroll-behavior: contain; padding-right: 4px; }}
}}
/* La valeur est déjà dans le champ : on masque celle affichée au-dessus de la barre */
[data-testid="stSliderThumbValue"] {{ display: none; }}
h3 {{ font-size: 22px !important; }}
/* Barre de verdict fixée en bas de l'écran, sur téléphone uniquement */
.barre-mobile {{ display: none; }}
@media (max-width: 640px) {{
  .barre-mobile {{ display: flex; position: fixed; left: 0; right: 0; bottom: 0; z-index: 999; gap: 12px;
                   align-items: center; padding: 10px 16px 14px; background: #FFFFFF;
                   border-top: 1px solid #CFC5C1; box-shadow: 0 -4px 16px rgba(34,20,24,.08); }}
  .block-container {{ padding-bottom: 7rem; }}
}}
/* Fond gris clair de la carte "Et si" */
.st-key-etsi {{ background:#F6F3F1; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Chargement du modèle et des données (une seule fois, grâce au cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    modele   = joblib.load("modele_vin.joblib")
    scaler   = joblib.load("scaler_vin.joblib")
    colonnes = joblib.load("colonnes.joblib")
    return modele, scaler, colonnes


@st.cache_data
def charger_donnees():
    """Même préparation que train.py : fusion, type (rouge = 1), doublons retirés, cible."""
    rouge = pd.read_csv("winequality-red.csv", sep=";")
    blanc = pd.read_csv("winequality-white.csv", sep=";")
    rouge["type"], blanc["type"] = 1, 0
    df = pd.concat([rouge, blanc], ignore_index=True).drop_duplicates().reset_index(drop=True)
    df["bon_vin"] = (df["quality"] >= 7).astype(int)
    df["Classe"] = np.where(df["bon_vin"] == 1, "Bon (7 et plus)", "Pas bon (moins de 7)")
    df["Couleur"] = np.where(df["type"] == 1, "Rouge", "Blanc")
    return df


modele, scaler, colonnes = charger_modele()
df = charger_donnees()

# Chaque mesure : (nom affiché, unité, min, max, pas, format) — bornes = valeurs possibles
VARIABLES = {
    "alcohol":              ("Alcool",           "% vol.", 8.0,   15.0,  0.1,    "%.1f"),
    "density":              ("Densité",          "g/cm³",  0.98,  1.04,  0.0001, "%.4f"),
    "volatile acidity":     ("Acidité volatile", "g/L",    0.08,  1.6,   0.01,   "%.2f"),
    "citric acid":          ("Acide citrique",   "g/L",    0.0,   1.7,   0.01,   "%.2f"),
    "fixed acidity":        ("Acidité fixe",     "g/L",    3.8,   16.0,  0.1,    "%.1f"),
    "pH":                   ("pH",               "",       2.7,   4.1,   0.01,   "%.2f"),
    "chlorides":            ("Chlorures",        "g/L",    0.009, 0.62,  0.001,  "%.3f"),
    "residual sugar":       ("Sucre résiduel",   "g/L",    0.6,   66.0,  0.1,    "%.1f"),
    "total sulfur dioxide": ("SO₂ total",        "mg/L",   6.0,   440.0, 1.0,    "%.0f"),
    "sulphates":            ("Sulfates",         "g/L",    0.22,  2.0,   0.01,   "%.2f"),
    "free sulfur dioxide":  ("SO₂ libre",        "mg/L",   1.0,   289.0, 1.0,    "%.0f"),
}
# Groupes, de la mesure la plus influente à la moins influente pour le modèle
GROUPES = {
    "Alcool et densité":  ["alcohol", "density"],
    "Acidité":            ["volatile acidity", "citric acid", "fixed acidity", "pH"],
    "Sucre et sel":       ["chlorides", "residual sugar"],
    "Soufre et sulfates": ["total sulfur dioxide", "sulphates", "free sulfur dioxide"],
}
MESURES = list(VARIABLES)


# ---------------------------------------------------------------------------
# 3. Fonctions utiles
# ---------------------------------------------------------------------------
def nombre_fr(x, fmt="%.2f"):
    return (fmt % x).replace(".", ",")


def predire(valeurs, type_num):
    """Probabilité d'être un bon vin, exactement comme dans train.py / test_model.py."""
    donnee = pd.DataFrame([{**valeurs, "type": type_num}])[colonnes]
    return modele.predict_proba(scaler.transform(donnee))[0][1]


def arrondir(var, x):
    _, _, mini, maxi, pas, _ = VARIABLES[var]
    x = float(np.clip(x, mini, maxi))
    return round(round(x / pas) * pas, 6)


def ecrire_valeurs(valeurs):
    """Met à jour la barre ET le champ de chaque mesure (ils partagent la même valeur)."""
    for var in MESURES:
        v = arrondir(var, valeurs[var])
        st.session_state[f"barre_{var}"] = v
        st.session_state[f"champ_{var}"] = v


def couleur_actuelle():
    return st.session_state.get("couleur") or "Blanc"


# Callbacks : quand l'un des deux bouge, l'autre suit
def depuis_barre(var):
    st.session_state[f"champ_{var}"] = st.session_state[f"barre_{var}"]


def depuis_champ(var):
    st.session_state[f"barre_{var}"] = st.session_state[f"champ_{var}"]


def vin_typique():
    meme = df[df["Couleur"] == couleur_actuelle()]
    ecrire_valeurs(meme[MESURES].median())
    st.session_state.pop("vrai_vin", None)


def vin_au_hasard():
    vin = df[df["Couleur"] == couleur_actuelle()].sample(1).iloc[0]
    ecrire_valeurs(vin)
    st.session_state["vrai_vin"] = int(vin["quality"])


def reinitialiser():
    st.session_state["couleur"] = "Blanc"
    vin_typique()


@st.cache_data
def lien_alcool_densite(couleur):
    """Droite densité = a × alcool + b sur les vrais vins de cette couleur, et l'écart habituel."""
    meme = df[df["Couleur"] == couleur]
    a, b = np.polyfit(meme["alcohol"], meme["density"], 1)
    ecart = np.std(meme["density"] - (a * meme["alcohol"] + b))
    return a, b, ecart


def ajuster_densite():
    a, b, _ = lien_alcool_densite(couleur_actuelle())
    alcool = st.session_state["barre_alcohol"]
    v = arrondir("density", a * alcool + b)
    st.session_state["barre_density"] = v
    st.session_state["champ_density"] = v


def verre_svg(proba, couleur_vin, largeur=96):
    niveau = 88 - proba * 80                       # le vin monte avec la probabilité
    return f"""<svg width="{largeur}" height="{largeur*150/110:.0f}" viewBox="0 0 110 150" aria-hidden="true">
      <defs><clipPath id="bol"><path d="M18 8 Q14 70 55 88 Q96 70 92 8 Z"/></clipPath></defs>
      <rect x="0" y="{niveau:.1f}" width="110" height="{92 - niveau:.1f}" fill="{couleur_vin}" clip-path="url(#bol)"/>
      <path d="M18 8 Q14 70 55 88 Q96 70 92 8 Z" fill="none" stroke="#4A3B3F" stroke-width="2.5"/>
      <line x1="55" y1="88" x2="55" y2="136" stroke="#4A3B3F" stroke-width="3"/>
      <ellipse cx="55" cy="140" rx="28" ry="5" fill="none" stroke="#4A3B3F" stroke-width="2.5"/></svg>"""


ICONE_OK = ('<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.4" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-10"/></svg>')
ICONE_ALERTE = ('<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" '
                'stroke-linecap="round"><path d="M12 6v8"/><path d="M12 18h.01"/></svg>')


# Valeurs de départ : un vin blanc typique (uniquement au tout premier chargement)
if "barre_alcohol" not in st.session_state:
    st.session_state["couleur"] = "Blanc"
    ecrire_valeurs(df[df["Couleur"] == "Blanc"][MESURES].median())


# ---------------------------------------------------------------------------
# Thème qui suit la couleur du vin : bordeaux pour le rouge, doré clair pour le blanc
# (titres, pastilles, barres, choix sélectionné, verdict "pas bon")
# ---------------------------------------------------------------------------
THEMES = {
    # plein : barres, pastilles, jauge (contraste ≥ 3:1 sur blanc)
    # titre : grands textes (≥ 3:1) · texte : petits textes (≥ 4,5:1) · fond : fond des badges
    "Rouge": {"plein": "#6E1A33", "titre": "#3A0E1D", "texte": "#6E1A33", "fond": "#F4E1E7",
              "filtre": "none"},
    "Blanc": {"plein": "#B8860B", "titre": "#B8860B", "texte": "#8A6500", "fond": "#FBF3E3",
              # filtre CSS qui recolore les barres (dessinées par Streamlit) en doré #B8860B
              "filtre": "brightness(0) saturate(100%) invert(23%) sepia(100%) saturate(4481%) "
                        "hue-rotate(52deg) brightness(151%) contrast(91%)"},
}
THEME = THEMES[couleur_actuelle()]
st.markdown(f"""
<style>
:root {{ --acc-plein:{THEME['plein']}; --acc-titre:{THEME['titre']}; --acc-texte:{THEME['texte']};
         --acc-fond:{THEME['fond']}; }}
[data-testid="stSlider"] [role="group"] {{ filter: {THEME['filtre']}; }}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{ color:{THEME['texte']} !important;
    border-color:{THEME['plein']} !important; background:{THEME['fond']} !important; }}
[data-testid="stNumberInputContainer"]:focus-within {{ border-color:{THEME['plein']} !important; }}
</style>""", unsafe_allow_html=True)


# ===========================================================================
# PAGE 1 — PRÉDIRE
# ===========================================================================
def page_predire():
    st.title("Ce vin sera-t-il jugé bon ?")
    st.markdown('<p class="aide" style="font-size:17px">Réglez les mesures d\'un vin. Le modèle (Random Forest) '
                'estime en direct la probabilité qu\'il obtienne une note de 7/10 ou plus.</p>',
                unsafe_allow_html=True)

    gauche, droite = st.columns([1.7, 1], gap="large")

    # ------------------------- Colonne gauche : saisie -------------------------
    with gauche:
        with st.container(border=True):
            st.markdown('<div class="titre-etape"><span class="puce">1</span>Point de départ</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.segmented_control("Couleur du vin", ["Rouge", "Blanc"], key="couleur")
            with c2:
                st.markdown(f'<div class="aide">Remplir avec un vin {couleur_actuelle().lower()}</div>',
                            unsafe_allow_html=True)
                b1, b2, b3 = st.columns([1, 1.3, 1])
                b1.button("Vin typique", on_click=vin_typique, width="stretch")
                b2.button("Vin au hasard", on_click=vin_au_hasard, icon=":material/casino:",
                          width="stretch")
                b3.button("Réinitialiser", on_click=reinitialiser, icon=":material/restart_alt:",
                          type="tertiary", width="stretch")

        with st.container(border=True):
            st.markdown('<div class="titre-etape"><span class="puce">2</span>Mesures du vin</div>'
                        '<p class="aide">Glissez la barre ou tapez la valeur dans le champ : les deux restent '
                        'synchronisés. Mesures classées de la plus à la moins influente pour le modèle.</p>',
                        unsafe_allow_html=True)
            for groupe, variables in GROUPES.items():
                st.markdown(f'<div class="groupe">{groupe}</div>', unsafe_allow_html=True)
                for var in variables:
                    nom, unite, mini, maxi, pas, fmt = VARIABLES[var]
                    c_nom, c_barre, c_champ = st.columns([1.3, 3, 1.2], vertical_alignment="center")
                    c_nom.markdown(f'<div class="nom-mesure">{nom} <span class="unite">{unite}</span></div>',
                                   unsafe_allow_html=True)
                    c_barre.slider(f"{nom} (barre)", mini, maxi, step=pas, format=fmt, key=f"barre_{var}",
                                   on_change=depuis_barre, args=(var,), label_visibility="collapsed")
                    c_champ.number_input(f"{nom} (valeur)", mini, maxi, step=pas, format=fmt,
                                         key=f"champ_{var}", on_change=depuis_champ, args=(var,),
                                         label_visibility="collapsed")

    # ------------------------- Calcul de la prédiction -------------------------
    couleur = couleur_actuelle()
    type_num = 1 if couleur == "Rouge" else 0
    valeurs = {var: st.session_state[f"barre_{var}"] for var in MESURES}
    proba = predire(valeurs, type_num)
    meme = df[df["Couleur"] == couleur]
    bons = meme[meme["bon_vin"] == 1]

    # ------------------------- Barre de verdict pour téléphone -------------------------
    teinte_m = VERT if proba >= SEUIL else THEME["plein"]
    st.markdown(f"""
<div class="barre-mobile" role="status" aria-live="polite">
  <div style="flex:1; min-width:0;">
    <div style="display:flex; justify-content:space-between; align-items:baseline;">
      <span style="font-size:14px; font-weight:600; color:{teinte_m};">{'✓ Prédit : bon vin' if proba >= SEUIL else '✕ Prédit : pas bon'}</span>
      <span style="font-family:'Fraunces', Georgia, serif; font-size:28px; font-weight:600; color:{teinte_m};">{proba:.0%}</span>
    </div>
    <div class="jauge" style="margin-top:6px; height:8px;"><div class="jauge-plein" style="width:{proba*100:.1f}%; background:{teinte_m};"></div>
    <div class="jauge-seuil" style="top:-5px; height:16px;"></div></div>
  </div>
</div>""".replace(" %", "\u202f%").replace("%<", "\u202f%<"), unsafe_allow_html=True)

    # ------------------------- Colonne droite : verdict -------------------------
    with droite:
        with st.container(border=True, key="verdict"):
            st.markdown("### Verdict")
            ecart_seuil = round((proba - SEUIL) * 100)
            if abs(proba - SEUIL) < 0.10:                       # entre 40 % et 60 % : le modèle hésite
                badge = '<span class="badge badge-serre">≈ Résultat serré</span>'
                teinte = "#4A3B3F"
                phrase = (f"Classé « {'bon' if proba >= SEUIL else 'pas bon'} » de justesse : entre 40 % "
                          "et 60 %, le modèle hésite. Voir « Ce qui fait pencher la balance ».")
            elif proba >= SEUIL:
                badge = '<span class="badge badge-bon">✓ Prédit : bon vin</span>'
                teinte = VERT
                phrase = f"{ecart_seuil} points au-dessus du seuil : résultat net."
            else:
                badge = '<span class="badge badge-pasbon">✕ Prédit : pas bon</span>'
                teinte = THEME["plein"]
                phrase = f"{-ecart_seuil} points sous le seuil."
            couleur_vin = BORDEAUX if couleur == "Rouge" else OR
            st.markdown(f"""
<div style="display:flex; gap:20px; align-items:center;">
  {verre_svg(proba, couleur_vin)}
  <div>{badge}
    <div class="pourcent" style="color:{teinte}; margin-top:8px;">{proba:.0%}</div>
    <div class="petit" style="font-size:14px;">de chances d'être noté 7/10 ou plus</div>
  </div>
</div>
<div class="jauge"><div class="jauge-plein" style="width:{proba*100:.1f}%; background:{teinte};"></div>
<div class="jauge-seuil"></div></div>
<div class="jauge-legende"><span>0 %</span><span>seuil : 50 %</span><span>100 %</span></div>
<p style="margin:14px 0 8px; font-size:14px; color:#4A3B3F;">{phrase}</p>
""".replace(" %", " %"), unsafe_allow_html=True)
            if "vrai_vin" in st.session_state:
                note = st.session_state["vrai_vin"]
                juste = (note >= 7) == (proba >= SEUIL)
                st.markdown(f"Vraie note des dégustateurs : **{note}/10** — "
                            f"{'le modèle a raison' if juste else 'le modèle se trompe'}.")

            # Avertissement : alcool et densité trop éloignés de ce qu'on observe
            a, b, ecart = lien_alcool_densite(couleur)
            if abs(valeurs["density"] - (a * valeurs["alcohol"] + b)) > 3 * ecart:
                st.warning("**Combinaison rare : la prédiction est moins fiable.** Dans les vrais vins, "
                           "plus il y a d'alcool, plus le vin est léger : cette densité ne correspond pas "
                           "à ce degré d'alcool.", icon=":material/warning:")
                st.button("Ajuster la densité automatiquement", on_click=ajuster_densite)

        # Ce qui fait pencher la balance : les 4 mesures les plus importantes pour le modèle
        with st.container(border=True):
            st.markdown("### Ce qui fait pencher la balance")
            importance = pd.Series(modele.feature_importances_, index=colonnes).drop("type")
            lignes = ""
            for var in importance.sort_values(ascending=False).index[:4]:
                nom, unite, *_r, fmt = VARIABLES[var]
                q1, q3 = bons[var].quantile([0.25, 0.75])
                v = valeurs[var]
                texte_val = f"<b>{nom} {nombre_fr(v, fmt)}</b>"
                if q1 <= v <= q3:
                    lignes += (f'<div class="ligne-raison"><span class="rond-ok">{ICONE_OK}</span>'
                               f'<span>{texte_val} : comme la plupart des bons vins</span></div>')
                else:
                    sens = "plus élevé" if v > q3 else "plus bas"
                    lignes += (f'<div class="ligne-raison"><span class="rond-alerte" style="color:var(--acc-plein);">{ICONE_ALERTE}</span>'
                               f'<span>{texte_val} : {sens} que la plupart des bons vins</span></div>')
            st.markdown(lignes, unsafe_allow_html=True)
            st.markdown(f'<p class="petit">Comparaison avec les bons vins {couleur.lower()}s du jeu de '
                        'données : une aide à la lecture, pas l\'explication exacte du modèle.</p>',
                        unsafe_allow_html=True)

        # Et si… ? : le même vin avec une modification réaliste, recalculée par le modèle
        with st.container(border=True, key="etsi"):
            st.markdown("### Et si… ?")
            st.markdown('<p class="aide">Le même vin avec une modification réaliste.</p>',
                        unsafe_allow_html=True)
            scenarios = []
            for delta in (+0.5, -0.5):                 # l'alcool bouge, la densité suit
                alc = arrondir("alcohol", valeurs["alcohol"] + delta)
                dens = arrondir("density", valeurs["density"] + a * (alc - valeurs["alcohol"]))
                p = predire({**valeurs, "alcohol": alc, "density": dens}, type_num)
                scenarios.append((f"Alcool {nombre_fr(alc, '%.1f')} % et densité {nombre_fr(dens, '%.4f')}", p))
            for var in importance.sort_values(ascending=False).index[:6]:
                q1, q3 = bons[var].quantile([0.25, 0.75])
                if var not in ("alcohol", "density") and not (q1 <= valeurs[var] <= q3):
                    mediane = arrondir(var, bons[var].median())
                    p = predire({**valeurs, var: mediane}, type_num)
                    scenarios.append((f"{VARIABLES[var][0]} à {nombre_fr(mediane, VARIABLES[var][5])}", p))
                    break
            html = ""
            for texte, p in scenarios:
                teinte_p = VERT if p >= proba else THEME["texte"]
                html += (f'<div class="etsi"><span>{texte}</span>'
                         f'<b style="color:{teinte_p}; font-variant-numeric:tabular-nums;">{p:.0%}</b></div>')
            st.markdown(html.replace(" %", " %"), unsafe_allow_html=True)


# ===========================================================================
# PAGE 2 — LES DONNÉES
# ===========================================================================
def page_donnees():
    st.title("L'alcool est le meilleur indice de qualité")
    st.markdown('<p class="aide" style="font-size:17px">Et seul 1 vin sur 5 est jugé bon. Analyse de 5 320 vins '
                'portugais (Vinho Verde) notés par des dégustateurs, rouges et blancs réunis, doublons retirés.</p>',
                unsafe_allow_html=True)

    choix = st.segmented_control("Afficher", ["Tous les vins", "Rouges", "Blancs"], default="Tous les vins")
    vue = df if choix in (None, "Tous les vins") else df[df["Couleur"] == choix[:-1]]

    k1, k2, k3, k4 = st.columns(4)
    tuiles = [
        (k1, "Vins analysés", f"{len(vue):,}".replace(",", " "),
         f"{(vue['type'] == 1).sum():,} rouges · {(vue['type'] == 0).sum():,} blancs".replace(",", " ")),
        (k2, "Bons vins (note ≥ 7)", f"{vue['bon_vin'].mean():.0%}".replace("%", " %"), "classe minoritaire"),
        (k3, "Note moyenne", nombre_fr(vue["quality"].mean(), "%.1f") + " / 10",
         f"notes de {vue['quality'].min()} à {vue['quality'].max()}"),
        (k4, "Alcool moyen", nombre_fr(vue["alcohol"].mean(), "%.1f") + " %",
         nombre_fr(vue[vue['bon_vin'] == 1]['alcohol'].mean(), "%.1f") + " % chez les bons vins"),
    ]
    for col, titre, valeur, sous in tuiles:
        col.markdown(f'<div class="tuile"><div class="tuile-titre">{titre}</div>'
                     f'<div class="tuile-valeur">{valeur}</div><div class="tuile-sous">{sous}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    couleurs_classe = alt.Scale(domain=["Pas bon (moins de 7)", "Bon (7 et plus)"], range=[GRIS_PAS_BON, VERT])
    g1, g2 = st.columns(2, gap="large")
    with g1:
        with st.container(border=True):
            st.markdown("### Seul 1 vin sur 5 atteint 7/10")
            st.markdown('<p class="aide">Nombre de vins par note. Les classes sont déséquilibrées : le modèle est '
                        'jugé sur le F1, pas sur l\'exactitude.</p>', unsafe_allow_html=True)
            barres = alt.Chart(vue).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("quality:O", title="Note des dégustateurs", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count():Q", title="Nombre de vins"),
                color=alt.Color("Classe:N", scale=couleurs_classe, legend=alt.Legend(orient="bottom", title=None)),
                tooltip=[alt.Tooltip("quality:O", title="Note"), alt.Tooltip("count():Q", title="Vins")])
            texte = barres.mark_text(dy=-8, color=ENCRE).encode(text="count():Q")
            st.altair_chart((barres + texte).properties(height=300), width="stretch")
    with g2:
        with st.container(border=True):
            st.markdown("### L'alcool sépare le mieux les deux classes")
            st.markdown('<p class="aide">Degré d\'alcool : la boîte couvre la moitié centrale des vins '
                        '(25 % à 75 %), le trait est la médiane.</p>', unsafe_allow_html=True)
            boite = alt.Chart(vue).mark_boxplot(size=36, extent="min-max", median={"color": ENCRE}).encode(
                y=alt.Y("Classe:N", title=None, sort=["Pas bon (moins de 7)", "Bon (7 et plus)"],
                        axis=alt.Axis(labelLimit=220)),
                x=alt.X("alcohol:Q", title="Alcool (% vol.)", scale=alt.Scale(zero=False)),
                color=alt.Color("Classe:N", scale=couleurs_classe, legend=None))
            st.altair_chart(boite.properties(height=300), width="stretch")

    with st.container(border=True):
        st.markdown("### Lien de chaque mesure avec la note")
        st.markdown('<p class="aide">Corrélation avec la note. Une corrélation négative n\'est pas « mauvaise » : '
                    'la mesure baisse simplement quand la note monte.</p>', unsafe_allow_html=True)
        corr = vue[MESURES + ["quality"]].corr()["quality"].drop("quality").reset_index()
        corr.columns = ["var", "Corrélation"]
        corr["Mesure"] = corr["var"].map(lambda v: VARIABLES[v][0])
        corr["Sens"] = np.where(corr["Corrélation"] > 0, "Monte avec la note", "Baisse quand la note monte")
        graph = alt.Chart(corr).mark_bar(cornerRadius=4).encode(
            x=alt.X("Corrélation:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
            y=alt.Y("Mesure:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
            color=alt.Color("Sens:N", scale=alt.Scale(domain=["Baisse quand la note monte", "Monte avec la note"],
                                                        range=[ORANGE, BLEU]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Mesure", alt.Tooltip("Corrélation:Q", format=".2f")])
        st.altair_chart(graph.properties(height=400), width="stretch")

    with st.expander("Voir les données brutes"):
        st.dataframe(vue.drop(columns=["bon_vin"]), width="stretch", height=300)


# ===========================================================================
# PAGE 3 — LE MODÈLE
# ===========================================================================
def page_modele():
    st.title("Un Random Forest de " + str(modele.n_estimators) + " arbres")
    gauche, droite = st.columns([1, 1.2], gap="large")
    with gauche:
        st.markdown("Chaque arbre pose une série de questions (« plus de 11 % d'alcool ? », « acidité volatile "
                    "sous 0,4 ? »…) et vote *bon* ou *pas bon*. La forêt fait la moyenne des votes : c'est la "
                    "probabilité affichée. Des arbres un peu différents se trompent à des endroits différents ; "
                    "en votant ensemble, leurs erreurs se compensent.")
        p = modele.get_params()
        st.markdown("#### Réglages retenus après optimisation")
        st.table(pd.DataFrame({
            "Réglage": ["Nombre d'arbres", "Profondeur maximale", "Variables par question", "Poids des classes"],
            "Valeur": [str(p["n_estimators"]), str(p["max_depth"]), str(p["max_features"]), str(p["class_weight"])],
            "Rôle": ["Plus d'arbres = vote plus stable", "Empêche chaque arbre d'apprendre par cœur",
                     "Rend les arbres différents entre eux", "Donne plus de poids aux bons vins, plus rares"],
        }).set_index("Réglage"))
    with droite:
        st.markdown("#### Ce qui compte le plus pour le modèle")
        imp = pd.DataFrame({"var": colonnes, "Importance": modele.feature_importances_})
        imp["Mesure"] = imp["var"].map(lambda v: VARIABLES[v][0] if v in VARIABLES else "Couleur du vin")
        g = alt.Chart(imp).mark_bar(cornerRadius=4, color=BORDEAUX).encode(
            x=alt.X("Importance:Q", axis=alt.Axis(format="%"), title=None),
            y=alt.Y("Mesure:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
            tooltip=["Mesure", alt.Tooltip("Importance:Q", format=".1%")])
        st.altair_chart((g + g.mark_text(align="left", dx=4, color=ENCRE).encode(
            text=alt.Text("Importance:Q", format=".0%"))).properties(height=380), width="stretch")

    figures = [(f, t) for f, t in [("fig_cv_avant_apres.png", "F1 en validation croisée, avant / après optimisation"),
                                   ("fig_correlation.png", "Matrice de corrélation")] if os.path.exists(f)]
    if figures:
        st.markdown("#### Résultats du notebook")
        for col, (fichier, titre) in zip(st.columns(len(figures)), figures):
            col.image(fichier, caption=titre, width="stretch")


# ===========================================================================
# PAGE 4 — À PROPOS
# ===========================================================================
def page_apropos():
    st.title("Le projet")
    gauche, droite = st.columns([1.3, 1], gap="large")
    with gauche:
        st.markdown("""
Projet de Machine Learning — **EFREI Paris, ING2-BDML, 2026/2027**.

**Objectif :** prédire si un vin est *bon* (note ≥ 7) ou *pas bon* à partir de ses mesures physico-chimiques.

**Données :** jeu *Wine Quality* de l'UCI (vins portugais « Vinho Verde »), rouges et blancs réunis,
soit 5 320 vins après suppression des doublons. Une colonne `type` garde la couleur du vin.

**Démarche :**
1. Analyse des données (distributions, corrélations, valeurs extrêmes, déséquilibre des classes).
2. Comparaison de 5 modèles (régression logistique, k-NN, arbre, Naive Bayes, SVM), jugés sur le F1.
3. Optimisation des hyperparamètres (Grid Search, validation croisée à 5 plis).
4. Modèle final : Random Forest avec `class_weight='balanced'`.
""")
    with droite:
        st.info("Le jeu de données est **déséquilibré** : environ 4 vins sur 5 sont « pas bons ». Le modèle a été "
                "réglé pour ne pas rater les bons vins, quitte à faire quelques fausses alertes.")
        st.warning("Les notes viennent de dégustateurs pour des vins portugais d'une seule région : le modèle ne "
                   "s'applique pas tel quel à d'autres vins.")


# ---------------------------------------------------------------------------
# Menu en haut de page
# ---------------------------------------------------------------------------
pages = [
    st.Page(page_predire, title="Prédire", icon=":material/wine_bar:", default=True),
    st.Page(page_donnees, title="Les données", icon=":material/bar_chart:", url_path="donnees"),
    st.Page(page_modele, title="Le modèle", icon=":material/account_tree:", url_path="modele"),
    st.Page(page_apropos, title="À propos", icon=":material/info:", url_path="a-propos"),
]
st.navigation(pages, position="top").run()

