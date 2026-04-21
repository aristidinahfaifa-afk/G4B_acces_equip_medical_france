# 🌿 PollenGuard — Prédiction du Risque Pollinique à Rennes

> Projet Data Science · 2A ENSAI · 2025–2026

---
Ce travail a été réalisé  dans le cadre du cours de python pour la datascience dispensée en 2ème année à l'ENSAI de Rennes.

---

##  Contexte et problématique

En France, près de **30% de la population souffre d'allergie pollinique**, un chiffre en hausse constante avec le changement climatique. Les allergènes polliniques (bouleau, graminées) provoquent des rhinites, conjonctivites et crises d'asthme dont la sévérité est directement liée à la concentration de pollen dans l'atmosphère.

**Question de recherche :**
> Peut-on prédire le niveau de risque allergique journalier à Rennes à partir des conditions météorologiques et des concentrations polliniques récentes ?

**Réponse apportée :** Deux modèles de Machine Learning (un par type de pollen) permettent de classifier le risque du lendemain en deux niveaux (Faible / À risque), avec une application web accessible au grand public pour consulter les prévisions en temps réel.

---

##  Structure du projet

```
G4B_acces_equip_medical_france/
│
├── app/
│   └── main.py                    # Application Streamlit 
│
├── data/
│   ├── raw/                       # Données brutes téléchargées
│   └── clean/                     # Données nettoyées et enrichies
│
├── models/
│   ├── modele_bouleau.pkl         # Modèle Random Forest — pollen bouleau
│   ├── modele_graminees.pkl       # Modèle Random Forest — pollen graminées
│   └── features.pkl               # Liste ordonnée des features du modèle
│
├── Rapport_final.ipynb     # EDA : visualisations et statistiques
├── resources/
│   ├── logo_ensai.png      
│   └── allergy.jpg
│
├── utils.py                       # Fonctions partagées (import, prédiction...)
├── pyproject.toml                 # Dépendances gérées par uv
├── python-version                 # Version de python
├── uv.lock                        # Versions exactes des packages (ne pas modifier)
└── README.md
```

---

##  Méthodologie

### Sources de données

Toutes les données proviennent de l'**API Open-Meteo** :

| Source | Variables | Granularité |
|--------|-----------|-------------|
| [Open-Meteo Archive](https://archive-api.open-meteo.com) | Température, précipitations, vitesse du vent | Horaire → agrégé en journalier |
| [Open-Meteo Air Quality](https://air-quality-api.open-meteo.com) | Pollen bouleau (`birch_pollen`), pollen graminées (`grass_pollen`) | Horaire → agrégé en journalier |

**Couverture temporelle :** 2021-01-01 → 2026-04-19  
**Localisation :** Rennes (48.11°N, 1.67°W)

### Construction de la variable cible
En utilisant la variable concentration de chaque pollen, on a c


> Les variables laggées (`_lag`) et les moyennes glissantes (`_roll`, `_moy`) capturent l'inertie biologique : la floraison réagit à l'accumulation de chaleur sur plusieurs jours, pas à la température instantanée.

### Modèles

| Pollen | Algorithme | Split |
|--------|-----------|-------|
| Bouleau | Random Forest Classifier | Entraînement 2021–2024 · Test 2025–2026 |
| Graminées | Random Forest Classifier | Entraînement 2021–2024 · Test 2025–2026 |

> Le split est **temporel** (pas aléatoire) pour simuler les conditions réelles de prévision — on ne peut pas utiliser le futur pour entraîner un modèle de prévision.

---

## 🚀 Installation et utilisation

### Prérequis

- [Python 3.13+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) — gestionnaire de packages moderne

```bash
# Installer uv si pas encore installé
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Cloner le dépôt

```bash
git clone https://github.com/aristidinahfaifa-afk/G4B_acces_equip_medical_france.git
cd G4B_acces_equip_medical_france
```

### 2. Installer l'environnement

```bash
uv sync
```

Cette commande crée automatiquement le `.venv`, installe toutes les dépendances et garantit que tout le monde utilise les mêmes versions exactes (via `uv.lock`).

### 3. Lancer les notebooks

```bash
# Activer l'environnement
source .venv/bin/activate   # Linux/Mac
# ou
.venv\Scripts\activate      # Windows

# Lancer Jupyter
uv run jupyter lab
```

Ordre d'exécution recommandé :
1. `Analyse_exploratoire.ipynb` — import des données, nettoyage, visualisations
2. `Analyse_explor_modelisation.ipynb` — feature engineering, entraînement, évaluation

### 4. Lancer l'application Streamlit

```bash
uv run streamlit run app/main.py
```

L'application est accessible sur `http://localhost:8501`

---

## 🌐 Application en ligne

L'application est déployée et accessible publiquement :

👉 **[Lien Streamlit à compléter]**

Elle permet de :
- Consulter le bulletin allergie pour n'importe quelle date depuis 2021
- Voir les prévisions J+1, J+2, J+3
- Analyser les indicateurs météo-polliniques (GDD, lessivage, anomalie saisonnière...)
- Comparer la saison en cours avec les années précédentes (2021–2025)

---

## 📦 Données et reproductibilité

### Stratégie de chargement des données

Le projet utilise un système de **triple fallback** pour garantir la reproductibilité :

```
1. Cache local  → si data/raw/meteo.csv existe déjà
2. API          → téléchargement depuis Open-Meteo (gratuit, sans clé)
3. S3           → si l'API est indisponible (données de secours sur MinIO SSPCloud)
```

Si tu exécutes le notebook pour la première fois :
- Les données sont téléchargées automatiquement depuis l'API
- Elles sont sauvegardées en cache local pour les prochaines exécutions
- En cas d'indisponibilité de l'API, les données de secours sont chargées depuis S3

### Ajouter une dépendance (développeurs)

```bash
uv add nom_du_package

# Puis commit les fichiers de configuration
git add pyproject.toml uv.lock
git commit -m "add: nom_du_package"
git push
```

### Mettre à jour après un `git pull`

```bash
git pull
uv sync
```

---


## Equipe
Ce projet a été réalisé par: 
- AIFA ARISTIDINA
-KENNE YONTA Lesline
- Rose Valentin

---

##  Limites

- Modèles entraînés sur **Rennes uniquement** — non généralisables directement à d'autres villes
- Les données pollen de l'API Open-Meteo sont issues d'un **modèle atmosphérique**, pas de mesures terrain (contrairement au Réseau Sentinelles)
- Les prévisions à J+2 et J+3 dépendent des prévisions météo qui sont elles-mêmes imprécises
- L'API archive publie ses données avec **1 à 2 jours de délai**

---

## 📄 Licence

Projet académique — usage éducatif uniquement.  
Données : [Open-Meteo](https://open-meteo.com/) (CC BY 4.0)