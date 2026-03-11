# Utiliser une image Python légère
FROM python:3.11-slim

# Définir le dossier de travail dans le container
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer toutes les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du projet
COPY . .

# Lancer l'API FastAPI avec uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]