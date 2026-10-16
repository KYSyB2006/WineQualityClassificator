# Dockerfile — recette pour mettre notre application dans un "conteneur".
# Un conteneur = une boîte qui contient Python + nos librairies + notre code, pour que
# n'importe qui puisse lancer l'app sans rien installer, juste avec Docker.
#   Construire : docker build -t qualite-vin .
#   Lancer     : docker run -p 8501:8501 qualite-vin   puis ouvrir http://localhost:8501

# 1. On part d'une image Python légère (slim = version allégée)
FROM python:3.11-slim

# Python n'écrit pas de fichiers .pyc et affiche ses messages immédiatement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. On se place dans un dossier /app à l'intérieur du conteneur
WORKDIR /app

# 3. On copie d'abord la liste des librairies, puis on les installe.
#    (Docker garde cette étape en cache tant que requirements.txt ne change pas.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. On copie le reste : données, entraînement, tests, application, thème et figures
COPY winequality-red.csv winequality-white.csv train.py test_model.py app.py ./
COPY .streamlit ./.streamlit
COPY fig_cv_avant_apres.png fig_correlation.png ./

# 5. On entraîne le modèle PENDANT la construction de l'image, puis on lance les tests.
#    Si un test échoue, la construction s'arrête : on ne peut pas livrer une image cassée.
RUN python train.py && python -m pytest -q test_model.py -p no:cacheprovider

# 6. L'app Streamlit tourne sur le port 8501 : on l'ouvre
EXPOSE 8501

# 7. Docker vérifie toutes les 30 s que l'app répond (statut "healthy" dans docker ps)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# 8. La commande lancée au démarrage du conteneur.
#    --server.address=0.0.0.0 permet d'accéder à l'app depuis l'extérieur du conteneur.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
