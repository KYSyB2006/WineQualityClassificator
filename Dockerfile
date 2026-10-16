# Dockerfile — recette pour mettre notre application dans un "conteneur".
# Un conteneur = une boîte qui contient Python + nos librairies + notre code, pour que
# n'importe qui puisse lancer l'app sans rien installer, juste avec Docker.

# 1. On part d'une image Python légère (slim = version allégée)
FROM python:3.11-slim

# 2. On se place dans un dossier /app à l'intérieur du conteneur
WORKDIR /app

# 3. On copie d'abord la liste des librairies, puis on les installe.
#    (On copie requirements.txt seul en premier : Docker garde cette étape en cache
#     tant que le fichier ne change pas, donc les rebuilds sont plus rapides.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. On copie le reste : les données, le script d'entraînement et l'application
COPY winequality-red.csv winequality-white.csv train.py app.py ./

# 5. On entraîne le modèle PENDANT la construction de l'image.
#    Avantage : le modèle est créé dans le même environnement que celui où tournera l'app,
#    donc aucun problème de version.
RUN python train.py

# 6. L'app Streamlit tourne sur le port 8501 : on l'ouvre
EXPOSE 8501

# 7. La commande lancée au démarrage du conteneur.
#    --server.address=0.0.0.0 permet d'accéder à l'app depuis l'extérieur du conteneur.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
