# Qualité du vin — classification

Projet de Machine Learning (EFREI Paris, ING2-BDML, 2026/2027) : prédire si un vin est **bon** (note ≥ 7)
ou **pas bon** à partir de ses mesures physico-chimiques. Jeu de données *Wine Quality* (UCI),
rouges et blancs réunis (5 320 vins après suppression des doublons). Modèle final : Random Forest
avec `class_weight="balanced"`.

## Contenu

| Fichier | Rôle |
|---|---|
| `projet_vin.ipynb` | Analyse, comparaison des modèles, optimisation |
| `train.py` | Entraîne le modèle final et crée `modele_vin.joblib`, `scaler_vin.joblib`, `colonnes.joblib` |
| `app.py` | Application web Streamlit (Prédire · Les données · Le modèle · À propos) |
| `.streamlit/config.toml` | Thème de l'application |
| `test_model.py` | Tests unitaires (pytest) |
| `Dockerfile` | Image Docker de l'application |
| `.github/workflows/ci.yml` | Intégration continue GitHub Actions |

## Lancer en local

```bash
pip install -r requirements.txt
python train.py          # crée les fichiers .joblib
python -m pytest -q      # 4 tests
streamlit run app.py     # http://localhost:8501
```

## Lancer avec Docker

```bash
docker build -t qualite-vin .
docker run -p 8501:8501 qualite-vin
```

La construction entraîne le modèle et lance les tests : si un test échoue, l'image n'est pas créée.
`docker ps` affiche `healthy` quand l'application répond.

## Intégration continue

À chaque push et pull request, GitHub Actions installe les librairies, entraîne le modèle, lance les
tests, construit l'image Docker et vérifie que l'application démarre. Sur la branche principale, l'image
est publiée sur `ghcr.io` :

```bash
docker run -p 8501:8501 ghcr.io/kysyb2006/winequalityclassificator:latest
```

## Auteurs

Awêro OKETOKOUN & Belva
