"""
test_model.py — tests unitaires du modèle sauvegardé.
On les lance avec :  pytest
Un test unitaire = une petite fonction qui vérifie automatiquement qu'une partie du code
se comporte comme prévu. Si une vérification échoue, pytest le signale en rouge.
Chaque fonction qui commence par "test_" est un test que pytest exécute.
"""
import joblib
import numpy as np
import pandas as pd

# On charge une fois le modèle, le scaler et les colonnes (comme le fait l'application)
modele   = joblib.load("modele_vin.joblib")
scaler   = joblib.load("scaler_vin.joblib")
colonnes = joblib.load("colonnes.joblib")


def _preparer(valeurs):
    """Petite aide : transforme un dictionnaire de mesures en données standardisées,
    dans le bon ordre de colonnes (exactement comme dans l'application)."""
    donnee = pd.DataFrame([valeurs])[colonnes]
    return scaler.transform(donnee)


# Un vin exemple (les moyennes du jeu de données), qu'on réutilise dans plusieurs tests
VIN_EXEMPLE = {
    "fixed acidity": 7.2, "volatile acidity": 0.34, "citric acid": 0.32,
    "residual sugar": 5.4, "chlorides": 0.056, "free sulfur dioxide": 30.0,
    "total sulfur dioxide": 115.0, "density": 0.995, "pH": 3.2,
    "sulphates": 0.53, "alcohol": 10.5, "type": 0,
}


def test_nombre_de_colonnes():
    # Le modèle doit attendre exactement 12 variables (11 mesures + le type)
    assert len(colonnes) == 12


def test_la_prediction_est_0_ou_1():
    # La prédiction doit être soit 0 (pas bon), soit 1 (bon), rien d'autre
    prediction = modele.predict(_preparer(VIN_EXEMPLE))[0]
    assert prediction in (0, 1)


def test_les_probabilites_font_100_pourcent():
    # Les deux probabilités (pas bon + bon) doivent s'additionner à 1
    probas = modele.predict_proba(_preparer(VIN_EXEMPLE))[0]
    assert np.isclose(probas.sum(), 1.0)


def test_un_bon_vin_est_bien_detecte():
    # Un vin avec beaucoup d'alcool et peu d'acidité volatile a de bonnes chances d'être "bon".
    # On vérifie que sa probabilité d'être bon est plus élevée que celle d'un vin médiocre.
    bon_vin     = {**VIN_EXEMPLE, "alcohol": 13.5, "volatile acidity": 0.2}
    mauvais_vin = {**VIN_EXEMPLE, "alcohol": 9.0,  "volatile acidity": 0.8}
    proba_bon     = modele.predict_proba(_preparer(bon_vin))[0][1]
    proba_mauvais = modele.predict_proba(_preparer(mauvais_vin))[0][1]
    assert proba_bon > proba_mauvais
