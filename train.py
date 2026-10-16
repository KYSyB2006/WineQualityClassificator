"""
train.py — entraîne le modèle final et le sauvegarde dans des fichiers.
On lance ce script UNE fois. Il crée 'modele_vin.joblib', 'scaler_vin.joblib' et
'colonnes.joblib', que l'application web (app.py) chargera ensuite.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib   # joblib = librairie pour sauvegarder/charger des objets Python (ici notre modèle)

# 1. On charge les deux fichiers et on les réunit, comme dans le notebook
rouge = pd.read_csv("winequality-red.csv", sep=";")
blanc = pd.read_csv("winequality-white.csv", sep=";")
rouge["type"] = 1   # rouge = 1
blanc["type"] = 0   # blanc = 0
df = pd.concat([rouge, blanc], ignore_index=True)

# 2. On enlève les doublons (comme dans le notebook)
df = df.drop_duplicates().reset_index(drop=True)

# 3. Cible binaire : bon vin si note >= 7
df["bon_vin"] = (df["quality"] >= 7).astype(int)

# 4. X = les variables, y = la cible
X = df.drop(columns=["quality", "bon_vin"])
y = df["bon_vin"]

# 5. On standardise (met toutes les variables à la même échelle)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# 6. On entraîne le modèle final sur TOUTES les données (une fois le modèle choisi,
#    on l'entraîne sur tout pour qu'il soit le plus performant possible).
#    On reprend les réglages trouvés par le Grid Search dans le notebook.
modele = RandomForestClassifier(
    n_estimators=100,          # nombre d'arbres dans la forêt
    max_depth=12,              # profondeur max de chaque arbre
    max_features="sqrt",       # nb de variables tirées au hasard à chaque question
    class_weight="balanced",   # gère le déséquilibre (plus de poids aux bons vins)
    random_state=42,           # pour obtenir toujours le même modèle
)
modele.fit(X_norm, y)

# 7. On sauvegarde le modèle, le scaler et la liste des colonnes (dans le bon ordre)
joblib.dump(modele, "modele_vin.joblib")
joblib.dump(scaler, "scaler_vin.joblib")
joblib.dump(list(X.columns), "colonnes.joblib")

print("Modèle entraîné et sauvegardé.")
print("Colonnes attendues par le modèle :", list(X.columns))
