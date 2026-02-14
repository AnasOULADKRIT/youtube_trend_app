# 🎥 Application Streamlit - Analyse YouTube Trending Videos

## 📋 Description

Cette application Streamlit interactive permet d'explorer, visualiser et prédire la popularité des vidéos YouTube trending. Elle est basée sur votre projet d'analyse de données YouTube.

## ✨ Fonctionnalités

### 🏠 Page d'Accueil
- Vue d'ensemble des statistiques clés
- Métriques principales (nombre de vidéos, vues totales, likes)
- Objectifs du projet

### 📈 Dataset
- **Aperçu** : Visualisation du dataset avec slider pour ajuster le nombre de lignes
- **Statistiques** : Statistiques descriptives des variables numériques
- **Filtres** : Filtrage interactif par catégorie, popularité, et plage de vues
- **Téléchargement** : Export des données complètes ou filtrées en CSV

### 📊 Visualisations
- Distribution des vues (échelle logarithmique)
- Matrice de corrélation entre variables
- Taux d'engagement par catégorie
- Impact de l'heure et du jour de publication
- Top 10 des chaînes les plus populaires
- Analyse temporelle des tendances

### 🤖 Modèle ML
- **Entraînement** : Configuration et entraînement de modèles (Random Forest, Logistic Regression)
- **Résultats** : Matrice de confusion, rapport de classification, importance des variables
- **Prédiction** : Interface pour prédire la popularité d'une nouvelle vidéo

### ℹ️ À propos de Streamlit
- Explication des avantages de Streamlit
- Cas d'usage dans le projet
- Ressources pour aller plus loin

## 🚀 Installation et Lancement

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation

1. Installez les dépendances :
```bash
pip install -r requirements.txt
```

2. Lancez l'application :
```bash
streamlit run app_youtube_trend.py
```

3. L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 🎮 Éléments Interactifs

L'application utilise plusieurs widgets Streamlit pour l'interactivité :

- **Sliders** : Pour ajuster le nombre de lignes affichées, les hyperparamètres du modèle
- **Selectbox** : Pour choisir les visualisations, les algorithmes, les jours/heures
- **Multiselect** : Pour sélectionner plusieurs colonnes ou catégories
- **Radio buttons** : Pour filtrer par popularité
- **Buttons** : Pour entraîner le modèle et faire des prédictions
- **Number inputs** : Pour entrer les caractéristiques d'une vidéo
- **Download buttons** : Pour télécharger les données

## 📊 Structure des Données

Le dataset simulé contient les colonnes suivantes :
- `video_id` : Identifiant unique de la vidéo
- `title` : Titre de la vidéo
- `channel_title` : Nom de la chaîne
- `category_id` : Catégorie de la vidéo
- `views` : Nombre de vues
- `likes` : Nombre de likes
- `dislikes` : Nombre de dislikes
- `comment_count` : Nombre de commentaires
- `tags_count` : Nombre de tags
- `title_length` : Longueur du titre
- `description_length` : Longueur de la description
- `publish_hour` : Heure de publication
- `publish_day` : Jour de publication
- `trending_date` : Date de trending
- `is_popular` : Variable cible (1 = populaire, 0 = non populaire)
- `engagement_rate` : Taux d'engagement calculé
- `like_ratio` : Ratio de likes

## 🔧 Personnalisation

Pour utiliser vos propres données YouTube :

1. Remplacez la fonction `load_data()` dans le fichier `app_youtube_trend.py`
2. Chargez votre CSV avec `pd.read_csv("votre_fichier.csv")`
3. Assurez-vous que les colonnes correspondent aux features utilisées

Exemple :
```python
@st.cache_data
def load_data():
    df = pd.read_csv("youtube_data.csv")
    # Preprocessing selon votre dataset
    return df
```

## 💡 Pourquoi Streamlit ?

### Avantages dans ce projet :

1. **Rapidité de développement** : Application complète en quelques heures
2. **Interactivité native** : Widgets intégrés sans JavaScript
3. **Visualisations riches** : Support de Matplotlib, Plotly, Seaborn
4. **Déploiement facile** : Streamlit Cloud gratuit
5. **Communication efficace** : Partage des résultats avec les stakeholders

### Intérêt en Data Science :

- **Prototypage rapide** : Tester des idées rapidement
- **Démonstration** : Présenter les résultats de manière interactive
- **Exploration** : Permettre aux utilisateurs d'explorer les données
- **Déploiement** : Mettre en production des modèles ML simplement
- **Collaboration** : Faciliter le feedback et l'itération

## 📚 Ressources Streamlit

- [Documentation officielle](https://docs.streamlit.io)
- [API Reference](https://docs.streamlit.io/library/api-reference)
- [Forum communautaire](https://discuss.streamlit.io)
- [Gallery d'applications](https://streamlit.io/gallery)
- [30 Days of Streamlit](https://30days.streamlit.app)

## 🎯 Extensions Possibles

- Intégration avec l'API YouTube pour des données en temps réel
- Ajout de modèles de Deep Learning (LSTM, Transformers)
- Analyse de sentiment des commentaires
- Prédiction du nombre de vues exact (régression)
- Dashboard temps réel avec mise à jour automatique
- Export des visualisations en PDF
- Système de recommandation de tags/titres

## 📝 Notes

- Les données sont générées aléatoirement pour la démonstration
- Le modèle est entraîné à chaque session (pas de sauvegarde persistante)
- Pour un usage en production, ajoutez la persistance des modèles avec joblib/pickle

## 🤝 Contribution

Ce projet est basé sur l'analyse YouTube Trending Videos de Kaggle. N'hésitez pas à l'adapter selon vos besoins !

---

**Créé avec ❤️ et Streamlit**
