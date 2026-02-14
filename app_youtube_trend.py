import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse YouTube Trending Videos",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF0000;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #282828;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1e88e5;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🎥 Analyse des Vidéos YouTube Trending</h1>', unsafe_allow_html=True)

# Fonction pour charger les données (simulées pour démo)
@st.cache_data
def load_data():
    """Génère un dataset simulé basé sur votre projet"""
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'video_id': [f'VID_{i:05d}' for i in range(n_samples)],
        'title': [f'Video Title {i}' for i in range(n_samples)],
        'channel_title': np.random.choice(['Channel A', 'Channel B', 'Channel C', 'Channel D'], n_samples),
        'category_id': np.random.choice([1, 10, 15, 17, 20, 22, 23, 24, 25, 26], n_samples),
        'views': np.random.exponential(100000, n_samples).astype(int),
        'likes': np.random.exponential(5000, n_samples).astype(int),
        'dislikes': np.random.exponential(500, n_samples).astype(int),
        'comment_count': np.random.exponential(1000, n_samples).astype(int),
        'tags_count': np.random.poisson(15, n_samples),
        'title_length': np.random.normal(50, 15, n_samples).astype(int),
        'description_length': np.random.normal(500, 200, n_samples).astype(int),
        'publish_hour': np.random.randint(0, 24, n_samples),
        'publish_day': np.random.randint(0, 7, n_samples),
        'trending_date': pd.date_range('2024-01-01', periods=n_samples, freq='H')
    }
    
    df = pd.DataFrame(data)
    
    # Créer la variable cible 'is_popular' (basée sur les vues)
    df['is_popular'] = (df['views'] > df['views'].median()).astype(int)
    
    # Créer des features dérivées
    df['engagement_rate'] = (df['likes'] + df['comment_count']) / (df['views'] + 1)
    df['like_ratio'] = df['likes'] / (df['likes'] + df['dislikes'] + 1)
    
    return df

# Charger les données
df = load_data()

# Sidebar pour navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choisissez une section:",
    ["🏠 Accueil", "📈 Dataset", "📊 Visualisations", "🤖 Modèle ML", "ℹ️ À propos de Streamlit"]
)

# ============================================
# PAGE 1: ACCUEIL
# ============================================
if page == "🏠 Accueil":
    st.markdown('<h2 class="sub-header">Bienvenue dans l\'application d\'analyse YouTube !</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📹 Vidéos analysées", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("👁️ Vues totales", f"{df['views'].sum():,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("👍 Likes totaux", f"{df['likes'].sum():,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Objectif du Projet
    
    Ce projet vise à développer un **modèle de classification** pour prédire si une vidéo YouTube 
    deviendra populaire ou non, en se basant sur diverses caractéristiques :
    
    - 📊 Métriques d'engagement (vues, likes, commentaires)
    - 🏷️ Métadonnées (titre, description, tags)
    - ⏰ Informations temporelles (heure/jour de publication)
    - 📁 Catégorie de la vidéo
    
    ### 📚 Source des Données
    Dataset basé sur **YouTube Trending Videos** disponible sur Kaggle.
    """)
    
    st.info("💡 Utilisez le menu à gauche pour naviguer entre les différentes sections de l'application.")

# ============================================
# PAGE 2: DATASET
# ============================================
elif page == "📈 Dataset":
    st.markdown('<h2 class="sub-header">Exploration du Dataset</h2>', unsafe_allow_html=True)
    
    # Onglets pour organiser l'information
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Aperçu", "📊 Statistiques", "🔍 Filtres", "📥 Téléchargement"])
    
    with tab1:
        st.markdown("#### Aperçu des données")
        
        # Slider pour choisir le nombre de lignes à afficher
        n_rows = st.slider("Nombre de lignes à afficher:", 5, 100, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dimensions du dataset:**")
            st.write(f"- Nombre de lignes: {df.shape[0]:,}")
            st.write(f"- Nombre de colonnes: {df.shape[1]}")
        
        with col2:
            st.markdown("**Types de données:**")
            st.write(df.dtypes.value_counts())
    
    with tab2:
        st.markdown("#### Statistiques descriptives")
        
        # Selectbox pour choisir les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Sélectionnez les colonnes à analyser:",
            numeric_cols,
            default=numeric_cols[:5]
        )
        
        if selected_cols:
            st.dataframe(df[selected_cols].describe().T, use_container_width=True)
            
            # Distribution de la variable cible
            st.markdown("#### Distribution de la variable cible")
            fig, ax = plt.subplots(figsize=(8, 4))
            df['is_popular'].value_counts().plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
            ax.set_xlabel('Popularité (0 = Non populaire, 1 = Populaire)')
            ax.set_ylabel('Nombre de vidéos')
            ax.set_title('Distribution des vidéos populaires vs non populaires')
            plt.xticks(rotation=0)
            st.pyplot(fig)
            plt.close()
    
    with tab3:
        st.markdown("#### Filtrer les données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtre par catégorie
            categories = st.multiselect(
                "Catégories:",
                options=sorted(df['category_id'].unique()),
                default=sorted(df['category_id'].unique())[:3]
            )
        
        with col2:
            # Filtre par popularité
            popularity = st.radio(
                "Filtrer par popularité:",
                options=["Toutes", "Populaires uniquement", "Non populaires uniquement"]
            )
        
        # Filtre par vues
        min_views, max_views = st.slider(
            "Plage de vues:",
            int(df['views'].min()),
            int(df['views'].max()),
            (int(df['views'].min()), int(df['views'].max()))
        )
        
        # Appliquer les filtres
        filtered_df = df.copy()
        
        if categories:
            filtered_df = filtered_df[filtered_df['category_id'].isin(categories)]
        
        if popularity == "Populaires uniquement":
            filtered_df = filtered_df[filtered_df['is_popular'] == 1]
        elif popularity == "Non populaires uniquement":
            filtered_df = filtered_df[filtered_df['is_popular'] == 0]
        
        filtered_df = filtered_df[(filtered_df['views'] >= min_views) & (filtered_df['views'] <= max_views)]
        
        st.write(f"**Nombre de vidéos après filtrage:** {len(filtered_df):,}")
        st.dataframe(filtered_df.head(20), use_container_width=True)
    
    with tab4:
        st.markdown("#### Télécharger les données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bouton pour télécharger le dataset complet
            csv_full = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger le dataset complet (CSV)",
                data=csv_full,
                file_name='youtube_trending_full.csv',
                mime='text/csv',
            )
        
        with col2:
            # Bouton pour télécharger le dataset filtré
            if 'filtered_df' in locals():
                csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les données filtrées (CSV)",
                    data=csv_filtered,
                    file_name='youtube_trending_filtered.csv',
                    mime='text/csv',
                )

# ============================================
# PAGE 3: VISUALISATIONS
# ============================================
elif page == "📊 Visualisations":
    st.markdown('<h2 class="sub-header">Visualisations Clés</h2>', unsafe_allow_html=True)
    
    # Selectbox pour choisir le type de visualisation
    viz_type = st.selectbox(
        "Choisissez une visualisation:",
        [
            "Distribution des vues",
            "Corrélation entre variables",
            "Engagement par catégorie",
            "Impact de l'heure de publication",
            "Top 10 chaînes",
            "Analyse temporelle"
        ]
    )
    
    if viz_type == "Distribution des vues":
        st.markdown("#### Distribution des vues (échelle logarithmique)")
        
        fig = px.histogram(
            df, 
            x='views', 
            nbins=50, 
            title='Distribution des vues par vidéo',
            color_discrete_sequence=['#FF6B6B'],
            labels={'views': 'Nombre de vues', 'count': 'Fréquence'}
        )
        fig.update_xaxis(type="log")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Moyenne", f"{df['views'].mean():,.0f}")
        with col2:
            st.metric("Médiane", f"{df['views'].median():,.0f}")
        with col3:
            st.metric("Maximum", f"{df['views'].max():,.0f}")
    
    elif viz_type == "Corrélation entre variables":
        st.markdown("#### Matrice de corrélation")
        
        # Sélection des variables numériques
        corr_vars = ['views', 'likes', 'dislikes', 'comment_count', 
                     'tags_count', 'engagement_rate', 'like_ratio']
        
        corr_matrix = df[corr_vars].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Matrice de corrélation des variables numériques', fontsize=14, pad=20)
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 Les corrélations positives (rouge) indiquent que les variables augmentent ensemble, "
                "tandis que les corrélations négatives (bleu) indiquent une relation inverse.")
    
    elif viz_type == "Engagement par catégorie":
        st.markdown("#### Taux d'engagement moyen par catégorie")
        
        engagement_by_cat = df.groupby('category_id').agg({
            'engagement_rate': 'mean',
            'views': 'mean'
        }).reset_index()
        
        fig = px.bar(
            engagement_by_cat, 
            x='category_id', 
            y='engagement_rate',
            title='Taux d\'engagement moyen par catégorie',
            color='engagement_rate',
            color_continuous_scale='Viridis',
            labels={'category_id': 'Catégorie ID', 'engagement_rate': 'Taux d\'engagement'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Vues moyennes par catégorie
        fig2 = px.bar(
            engagement_by_cat, 
            x='category_id', 
            y='views',
            title='Nombre de vues moyen par catégorie',
            color='views',
            color_continuous_scale='Blues',
            labels={'category_id': 'Catégorie ID', 'views': 'Vues moyennes'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Impact de l'heure de publication":
        st.markdown("#### Popularité selon l'heure de publication")
        
        popularity_by_hour = df.groupby('publish_hour')['is_popular'].mean().reset_index()
        
        fig = px.line(
            popularity_by_hour,
            x='publish_hour',
            y='is_popular',
            title='Probabilité de devenir populaire selon l\'heure de publication',
            markers=True,
            labels={'publish_hour': 'Heure de publication', 'is_popular': 'Taux de popularité'}
        )
        fig.update_traces(line_color='#FF6B6B', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Analyse par jour de la semaine:**")
        
        days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        popularity_by_day = df.groupby('publish_day')['is_popular'].mean().reset_index()
        popularity_by_day['day_name'] = popularity_by_day['publish_day'].apply(lambda x: days[x])
        
        fig2 = px.bar(
            popularity_by_day,
            x='day_name',
            y='is_popular',
            title='Probabilité de devenir populaire selon le jour de la semaine',
            color='is_popular',
            color_continuous_scale='RdYlGn',
            labels={'day_name': 'Jour de la semaine', 'is_popular': 'Taux de popularité'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Top 10 chaînes":
        st.markdown("#### Top 10 des chaînes les plus populaires")
        
        top_channels = df.groupby('channel_title').agg({
            'views': 'sum',
            'likes': 'sum',
            'video_id': 'count'
        }).reset_index()
        top_channels.columns = ['channel_title', 'total_views', 'total_likes', 'video_count']
        top_channels = top_channels.sort_values('total_views', ascending=False).head(10)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_channels['channel_title'],
            y=top_channels['total_views'],
            name='Vues totales',
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title='Top 10 des chaînes par nombre de vues',
            xaxis_title='Chaîne',
            yaxis_title='Nombre de vues',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(top_channels, use_container_width=True)
    
    elif viz_type == "Analyse temporelle":
        st.markdown("#### Évolution temporelle des tendances")
        
        # Agrégation par date
        df['date'] = df['trending_date'].dt.date
        time_series = df.groupby('date').agg({
            'video_id': 'count',
            'views': 'mean',
            'likes': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_series['date'],
            y=time_series['video_id'],
            mode='lines',
            name='Nombre de vidéos',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig.update_layout(
            title='Nombre de vidéos trending par jour',
            xaxis_title='Date',
            yaxis_title='Nombre de vidéos',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 4: MODÈLE ML
# ============================================
elif page == "🤖 Modèle ML":
    st.markdown('<h2 class="sub-header">Prédiction de la Popularité</h2>', unsafe_allow_html=True)
    
    # Tabs pour organiser le contenu
    tab1, tab2, tab3 = st.tabs(["🎯 Entraînement", "📊 Résultats", "🔮 Prédiction"])
    
    with tab1:
        st.markdown("#### Configuration du modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Choisissez un algorithme:",
                ["Random Forest", "Logistic Regression"]
            )
            
            test_size = st.slider("Taille de l'ensemble de test:", 0.1, 0.5, 0.2, 0.05)
        
        with col2:
            if model_choice == "Random Forest":
                n_estimators = st.slider("Nombre d'arbres:", 10, 200, 100, 10)
                max_depth = st.slider("Profondeur maximale:", 5, 50, 20, 5)
            else:
                st.info("Régression logistique avec paramètres par défaut")
        
        if st.button("🚀 Entraîner le modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                # Préparation des données
                features = ['views', 'likes', 'comment_count', 'tags_count', 
                           'title_length', 'engagement_rate', 'like_ratio', 
                           'publish_hour', 'publish_day', 'category_id']
                
                X = df[features]
                y = df['is_popular']
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Normalisation
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    model = LogisticRegression(max_iter=500, random_state=42)
                
                model.fit(X_train_scaled, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Stocker dans session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['features'] = features
                st.session_state['accuracy'] = accuracy
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['X_test'] = X_test
                
                st.success(f"✅ Modèle entraîné avec succès ! Accuracy: {accuracy:.4f}")
    
    with tab2:
        st.markdown("#### Résultats du modèle")
        
        if 'model' in st.session_state:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Accuracy", f"{st.session_state['accuracy']:.4f}")
            
            # Matrice de confusion
            st.markdown("#### Matrice de confusion")
            cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non Populaire', 'Populaire'],
                       yticklabels=['Non Populaire', 'Populaire'])
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Réalité')
            ax.set_title('Matrice de confusion')
            st.pyplot(fig)
            plt.close()
            
            # Rapport de classification
            st.markdown("#### Rapport de classification")
            report = classification_report(
                st.session_state['y_test'], 
                st.session_state['y_pred'],
                target_names=['Non Populaire', 'Populaire'],
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            
            # Feature importance (si Random Forest)
            if hasattr(st.session_state['model'], 'feature_importances_'):
                st.markdown("#### Importance des variables")
                
                importance_df = pd.DataFrame({
                    'Feature': st.session_state['features'],
                    'Importance': st.session_state['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Importance des variables dans la prédiction',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Veuillez d'abord entraîner un modèle dans l'onglet 'Entraînement'.")
    
    with tab3:
        st.markdown("#### Prédire la popularité d'une nouvelle vidéo")
        
        if 'model' in st.session_state:
            st.markdown("Entrez les caractéristiques de votre vidéo:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                views_input = st.number_input("Vues", min_value=0, value=10000, step=1000)
                likes_input = st.number_input("Likes", min_value=0, value=500, step=50)
                comments_input = st.number_input("Commentaires", min_value=0, value=100, step=10)
            
            with col2:
                tags_input = st.number_input("Nombre de tags", min_value=0, value=15, step=1)
                title_length_input = st.number_input("Longueur du titre", min_value=1, value=50, step=1)
                category_input = st.selectbox("Catégorie", sorted(df['category_id'].unique()))
            
            with col3:
                publish_hour_input = st.slider("Heure de publication", 0, 23, 14)
                publish_day_input = st.selectbox(
                    "Jour de publication",
                    options=list(range(7)),
                    format_func=lambda x: ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 
                                           'Vendredi', 'Samedi', 'Dimanche'][x]
                )
            
            if st.button("🔮 Prédire", type="primary"):
                # Calculer les features dérivées
                engagement_rate = (likes_input + comments_input) / (views_input + 1)
                like_ratio = likes_input / (likes_input + 1)
                
                # Créer le vecteur de features
                input_data = np.array([[
                    views_input, likes_input, comments_input, tags_input,
                    title_length_input, engagement_rate, like_ratio,
                    publish_hour_input, publish_day_input, category_input
                ]])
                
                # Normaliser
                input_scaled = st.session_state['scaler'].transform(input_data)
                
                # Prédire
                prediction = st.session_state['model'].predict(input_scaled)[0]
                proba = st.session_state['model'].predict_proba(input_scaled)[0]
                
                # Afficher le résultat
                st.markdown("---")
                st.markdown("### 🎯 Résultat de la prédiction")
                
                if prediction == 1:
                    st.success(f"✅ Cette vidéo sera probablement **POPULAIRE** ! "
                             f"(Confiance: {proba[1]:.2%})")
                else:
                    st.error(f"❌ Cette vidéo sera probablement **NON POPULAIRE**. "
                           f"(Confiance: {proba[0]:.2%})")
                
                # Graphique de probabilité
                fig = go.Figure(go.Bar(
                    x=['Non Populaire', 'Populaire'],
                    y=proba,
                    marker_color=['#FF6B6B', '#4ECDC4']
                ))
                fig.update_layout(
                    title='Probabilités de prédiction',
                    yaxis_title='Probabilité',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Veuillez d'abord entraîner un modèle dans l'onglet 'Entraînement'.")

# ============================================
# PAGE 5: À PROPOS DE STREAMLIT
# ============================================
elif page == "ℹ️ À propos de Streamlit":
    st.markdown('<h2 class="sub-header">Pourquoi Streamlit ?</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Les avantages de Streamlit dans un projet Data Science
    
    Streamlit est une bibliothèque Python open-source qui permet de créer rapidement 
    des applications web interactives pour la data science et le machine learning.
    
    #### ✨ Points forts de Streamlit :
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Simplicité et rapidité** 🏃‍♂️
        - Code en pur Python, pas de HTML/CSS/JavaScript nécessaire
        - Développement d'une app complète en quelques heures
        - Rechargement automatique lors des modifications
        
        **2. Interactivité native** 🎮
        - Widgets intégrés (sliders, selectbox, buttons...)
        - Mise à jour en temps réel
        - Gestion automatique de l'état
        
        **3. Visualisations puissantes** 📊
        - Support natif de Matplotlib, Plotly, Altair
        - Affichage de dataframes interactifs
        - Cartes, graphiques, et plus encore
        """)
    
    with col2:
        st.markdown("""
        **4. Déploiement facile** 🌐
        - Streamlit Cloud gratuit
        - Déploiement en un clic
        - Partage simple avec des collègues/clients
        
        **5. Collaboration** 🤝
        - Démonstration de résultats aux parties prenantes
        - Prototypage rapide d'idées
        - Communication efficace des insights
        
        **6. Extensibilité** 🔧
        - Composants personnalisés
        - Intégration avec toute bibliothèque Python
        - API riche et bien documentée
        """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Cas d'usage dans ce projet")
    
    use_cases = [
        ("Exploration de données", "Présentation interactive du dataset avec filtres et statistiques"),
        ("Visualisation", "Affichage de graphiques complexes avec interactivité"),
        ("Modélisation", "Configuration et entraînement de modèles ML en direct"),
        ("Prédiction", "Interface utilisateur pour faire des prédictions en temps réel"),
        ("Communication", "Partage des résultats de manière visuelle et compréhensible")
    ]
    
    for title, description in use_cases:
        with st.expander(f"📌 {title}"):
            st.write(description)
    
    st.markdown("---")
    
    st.markdown("### 📚 Ressources pour aller plus loin")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Documentation**
        - [Streamlit Docs](https://docs.streamlit.io)
        - [API Reference](https://docs.streamlit.io/library/api-reference)
        """)
    
    with col2:
        st.markdown("""
        **Communauté**
        - [Forum Streamlit](https://discuss.streamlit.io)
        - [Gallery d'apps](https://streamlit.io/gallery)
        """)
    
    with col3:
        st.markdown("""
        **Tutoriels**
        - [30 Days of Streamlit](https://30days.streamlit.app)
        - [YouTube Channel](https://youtube.com/@streamlit)
        """)
    
    st.success("🎉 Streamlit transforme l'analyse de données en expérience interactive !")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📊 Application créée avec Streamlit | 🎥 Analyse YouTube Trending Videos</p>
        <p><i>Projet d'analyse de données et Machine Learning</i></p>
    </div>
""", unsafe_allow_html=True)
