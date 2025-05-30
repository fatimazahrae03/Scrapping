import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
import json
import pickle
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from paste import ProductAnalyzer

import warnings
warnings.filterwarnings('ignore')

class DashboardDataManager:
    """Gestionnaire de données pour le dashboard avec MongoDB"""
    
    def __init__(self, mongo_uri: str = 'mongodb://localhost:27017/', 
                 db_name: str = 'products_db', 
                 collection_name: str = 'products',
                 data_path: Optional[str] = None):
        """
        Initialiser le gestionnaire de données
        
        Args:
            mongo_uri: URI de connexion MongoDB
            db_name: Nom de la base de données
            collection_name: Nom de la collection
            data_path: Chemin vers le fichier de données (optionnel, pour compatibilité)
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.data_path = data_path
        self.df = None
        self.original_df = None
        self.logger = self._setup_logger()
        
        # Connexion MongoDB
        self.client = None
        self.db = None
        self.collection = None
        self.connect_to_mongodb()
        
        # Charger les données au démarrage
        try:
            self.load_data_from_mongodb()
        except Exception as e:
            self.logger.warning(f"Impossible de charger les données MongoDB: {e}")
            # Fallback vers les données d'exemple uniquement si MongoDB échoue
            self.load_sample_data()
    
    def _setup_logger(self) -> logging.Logger:
        """Configurer le logger"""
        logger = logging.getLogger('DashboardDataManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def connect_to_mongodb(self) -> bool:
        """Se connecter à MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.logger.info("✅ Connecté à MongoDB")
            return True
        except ConnectionFailure as e:
            self.logger.error(f"❌ Échec de la connexion MongoDB: {e}")
            return False
    
    def load_data_from_mongodb(self, filters: Dict = None) -> pd.DataFrame:
        """
        Charger les données depuis MongoDB
        
        Args:
            filters: Filtres MongoDB à appliquer
            
        Returns:
            DataFrame contenant les données
        """
        try:
            if not self.collection:
                raise Exception("Pas de connexion MongoDB")
            
            # Appliquer les filtres si fournis
            query = filters if filters else {}
            
            # Récupérer les produits depuis MongoDB
            cursor = self.collection.find(query)
            products = list(cursor)
            
            if not products:
                self.logger.warning("Aucun produit trouvé dans MongoDB")
                return pd.DataFrame()
            
            # Convertir en DataFrame
            self.df = pd.DataFrame(products)
            
            # Convertir ObjectId en string
            if '_id' in self.df.columns:
                self.df['_id'] = self.df['_id'].astype(str)
            
            # Traitement des données pour assurer la compatibilité
            self._process_mongodb_data()
            
            self.original_df = self.df.copy()
            self.logger.info(f"Données MongoDB chargées: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
            
            return self.df
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement MongoDB: {str(e)}")
            raise e
    
    def _process_mongodb_data(self):
        """Traiter les données MongoDB pour assurer la compatibilité"""
        if self.df is None or self.df.empty:
            return
        
        # Convertir les dates
        if 'created_at' in self.df.columns:
            self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        
        # S'assurer que les colonnes numériques sont correctes
        numeric_columns = ['price', 'stock_quantity', 'rating', 'review_count', 'discount_percentage']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # S'assurer que les colonnes booléennes sont correctes
        boolean_columns = ['available']
        for col in boolean_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(bool, errors='ignore')
        
        # Ajouter synthetic_score s'il n'existe pas
        if 'synthetic_score' not in self.df.columns:
            self.df['synthetic_score'] = np.random.uniform(0, 1, len(self.df))
    
    def load_data(self, file_path: Optional[str] = None, mongodb_filters: Dict = None) -> pd.DataFrame:
        """
        Charger les données (MongoDB par défaut, fichier en option)
        
        Args:
            file_path: Chemin vers le fichier de données (optionnel)
            mongodb_filters: Filtres MongoDB à appliquer
            
        Returns:
            DataFrame contenant les données
        """
        try:
            # Priorité à MongoDB
            if self.collection:
                return self.load_data_from_mongodb(mongodb_filters)
            
            # Fallback vers le fichier si fourni
            if file_path or self.data_path:
                return self._load_data_from_file(file_path or self.data_path)
            
            # Dernière option: données d'exemple
            return self.load_sample_data()
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement: {str(e)}")
            return self.load_sample_data()
    
    def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
        """Charger les données depuis un fichier (méthode originale)"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            self.df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            self.df = pd.read_excel(file_path)
        elif file_extension == '.json':
            self.df = pd.read_json(file_path)
        elif file_extension == '.pkl':
            with open(file_path, 'rb') as f:
                self.df = pickle.load(f)
        else:
            raise ValueError(f"Format de fichier non supporté: {file_extension}")
        
        self.original_df = self.df.copy()
        return self.df
    
    def load_sample_data(self) -> pd.DataFrame:
        """
        Créer des données d'exemple pour les tests (méthode originale inchangée)
        
        Returns:
            DataFrame avec des données d'exemple
        """
        np.random.seed(42)
        
        n_products = 1000
        platforms = ['Amazon', 'eBay', 'Shopify', 'WooCommerce', 'Magento']
        regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa']
        vendors = [f'Vendor_{i}' for i in range(1, 51)]
        categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Beauty', 'Toys']
        
        data = {
            '_id': [f'prod_{i}' for i in range(n_products)],
            'title': [f'Product {i} - {np.random.choice(categories)}' for i in range(n_products)],
            'price': np.random.uniform(10, 500, n_products).round(2),
            'synthetic_score': np.random.uniform(0, 1, n_products).round(3),
            'available': np.random.choice([True, False], n_products, p=[0.7, 0.3]),
            'stock_quantity': np.random.randint(0, 200, n_products),
            'platform': np.random.choice(platforms, n_products),
            'store_region': np.random.choice(regions, n_products),
            'store_domain': [f'store{i % 20}.com' for i in range(n_products)],
            'vendor': np.random.choice(vendors, n_products),
            'category': np.random.choice(categories, n_products),
            'rating': np.random.uniform(1, 5, n_products).round(1),
            'review_count': np.random.randint(0, 1000, n_products),
            'created_at': pd.date_range(
                start='2023-01-01', 
                end='2024-12-31', 
                periods=n_products
            ).to_pydatetime(),
            'discount_percentage': np.random.uniform(0, 50, n_products).round(1)
        }
        
        self.df = pd.DataFrame(data)
        self.original_df = self.df.copy()
        self.logger.info(f"Données d'exemple créées: {len(self.df)} produits")
        
        return self.df
    
    def refresh_data_from_mongodb(self, filters: Dict = None) -> pd.DataFrame:
        """
        Actualiser les données depuis MongoDB
        
        Args:
            filters: Filtres MongoDB à appliquer
            
        Returns:
            DataFrame actualisé
        """
        return self.load_data_from_mongodb(filters)
    
    def get_products_dataframe(self) -> pd.DataFrame:
        """
        Obtenir le DataFrame des produits
        
        Returns:
            DataFrame des produits
        """
        if self.df is None or self.df.empty:
            if self.collection:
                self.load_data_from_mongodb()
            else:
                self.load_sample_data()
        return self.df.copy()
    
    def calculate_synthetic_score(self, criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculer un score synthétique basé sur les critères (méthode originale inchangée)
        
        Args:
            criteria: Dictionnaire des critères et poids
            
        Returns:
            DataFrame avec les scores calculés
        """
        if self.df is None or self.df.empty:
            self.get_products_dataframe()
        
        df_scored = self.df.copy()
        
        # Récupérer les poids
        weights = criteria.get('weights', {})
        price_preference = criteria.get('price_preference', 'low')
        
        # Normalisation des prix
        if 'price' in df_scored.columns:
            price_max = df_scored['price'].max()
            price_min = df_scored['price'].min()
            if price_max > price_min:
                if price_preference == 'high':
                    df_scored['price_score'] = (df_scored['price'] - price_min) / (price_max - price_min)
                else:
                    df_scored['price_score'] = 1 - (df_scored['price'] - price_min) / (price_max - price_min)
            else:
                df_scored['price_score'] = 1.0
        else:
            df_scored['price_score'] = 0.5
        
        # Score de disponibilité
        if 'available' in df_scored.columns:
            df_scored['availability_score'] = df_scored['available'].astype(float)
        else:
            df_scored['availability_score'] = 1.0
        
        # Score de stock
        if 'stock_quantity' in df_scored.columns:
            stock_max = df_scored['stock_quantity'].fillna(0).max()
            if stock_max > 0:
                df_scored['stock_score'] = df_scored['stock_quantity'].fillna(0) / stock_max
            else:
                df_scored['stock_score'] = 0.5
        else:
            df_scored['stock_score'] = 0.5
        
        # Score de popularité du vendeur
        if 'vendor' in df_scored.columns:
            vendor_counts = df_scored['vendor'].value_counts()
            vendor_max = vendor_counts.max()
            df_scored['vendor_score'] = df_scored['vendor'].map(vendor_counts) / vendor_max
        else:
            df_scored['vendor_score'] = 0.5
        
        # Score de nouveauté
        if 'created_at' in df_scored.columns:
            try:
                df_scored['created_at'] = pd.to_datetime(df_scored['created_at'])
                max_date = df_scored['created_at'].max()
                min_date = df_scored['created_at'].min()
                if max_date != min_date:
                    df_scored['recency_score'] = (df_scored['created_at'] - min_date) / (max_date - min_date)
                else:
                    df_scored['recency_score'] = 1.0
            except:
                df_scored['recency_score'] = 0.5
        else:
            df_scored['recency_score'] = 0.5
        
        # Score de rating
        if 'rating' in df_scored.columns:
            df_scored['rating_score'] = df_scored['rating'] / 5.0
        else:
            df_scored['rating_score'] = 0.5
        
        # Calcul du score synthétique
        df_scored['synthetic_score'] = (
            weights.get('price', 0.25) * df_scored['price_score'] +
            weights.get('availability', 0.20) * df_scored['availability_score'] +
            weights.get('stock', 0.15) * df_scored['stock_score'] +
            weights.get('vendor_popularity', 0.15) * df_scored['vendor_score'] +
            weights.get('recency', 0.10) * df_scored['recency_score'] +
            weights.get('rating', 0.15) * df_scored['rating_score']
        )
        
        return df_scored
    
    def apply_ml_scoring(self, method: str = 'random_forest', target_column: str = 'synthetic_score') -> pd.DataFrame:
        """
        Appliquer un scoring ML (méthode originale inchangée)
        
        Args:
            method: Méthode ML à utiliser
            target_column: Colonne cible pour l'entraînement
            
        Returns:
            DataFrame avec les scores ML
        """
        if self.df is None or self.df.empty:
            self.get_products_dataframe()
        
        df_ml = self.df.copy()
        
        try:
            # Préparer les features numériques
            numeric_features = []
            for col in ['price', 'stock_quantity', 'rating', 'review_count', 'discount_percentage']:
                if col in df_ml.columns:
                    df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce').fillna(0)
                    numeric_features.append(col)
            
            # Encoder les features catégorielles
            categorical_features = []
            for col in ['platform', 'store_region', 'category', 'vendor']:
                if col in df_ml.columns:
                    # Encoder en utilisant le mode le plus simple
                    df_ml[f'{col}_encoded'] = pd.Factorize(df_ml[col])[0]
                    categorical_features.append(f'{col}_encoded')
            
            # Préparer le dataset d'entraînement
            feature_columns = numeric_features + categorical_features
            
            if len(feature_columns) < 2:
                # Pas assez de features, utiliser un score aléatoire amélioré
                df_ml['ml_score'] = np.random.uniform(0.1, 0.9, len(df_ml))
                return df_ml
            
            X = df_ml[feature_columns].fillna(0)
            
            # Si pas de colonne cible, en créer une
            if target_column not in df_ml.columns:
                # Créer un target basé sur une combinaison de features
                df_ml[target_column] = (
                    (df_ml.get('rating', 3) / 5.0) * 0.3 +
                    (1 - (df_ml.get('price', 100) / df_ml.get('price', 100).max())) * 0.3 +
                    (df_ml.get('available', True).astype(float)) * 0.2 +
                    (df_ml.get('stock_quantity', 50) / df_ml.get('stock_quantity', 50).max()) * 0.2
                )
            
            y = df_ml[target_column].fillna(0.5)
            
            # Appliquer le modèle ML
            if method == 'random_forest':
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                model.fit(X, y)
                df_ml['ml_score'] = model.predict(X)
                
                # Sauvegarder l'importance des features
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance = feature_importance
                
            else:
                # Méthodes alternatives simplifiées
                df_ml['ml_score'] = np.random.uniform(0.1, 0.9, len(df_ml))
            
            # Normaliser les scores
            df_ml['ml_score'] = np.clip(df_ml['ml_score'], 0, 1)
            
        except Exception as e:
            self.logger.warning(f"Erreur dans apply_ml_scoring: {e}")
            # Score de fallback
            df_ml['ml_score'] = np.random.uniform(0.1, 0.9, len(df_ml))
        
        return df_ml
    
    def cluster_products(self, n_clusters: int = 5, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Appliquer un clustering sur les produits (méthode originale inchangée)
        
        Args:
            n_clusters: Nombre de clusters
            features: Liste des features à utiliser
            
        Returns:
            DataFrame avec les clusters
        """
        if self.df is None or self.df.empty:
            self.get_products_dataframe()
        
        df_clustered = self.df.copy()
        
        try:
            # Sélectionner les features pour le clustering
            if features is None:
                features = []
                for col in ['price', 'synthetic_score', 'stock_quantity', 'rating', 'review_count']:
                    if col in df_clustered.columns:
                        features.append(col)
            
            if len(features) < 2:
                # Clustering simple basé sur les quantiles
                if 'synthetic_score' in df_clustered.columns:
                    df_clustered['cluster'] = pd.qcut(
                        df_clustered['synthetic_score'], 
                        q=min(n_clusters, len(df_clustered)), 
                        labels=False, 
                        duplicates='drop'
                    ).fillna(0)
                else:
                    # Clustering aléatoire
                    df_clustered['cluster'] = np.random.randint(0, n_clusters, len(df_clustered))
                return df_clustered
            
            # Préparer les données
            X = df_clustered[features].fillna(0)
            
            # Normaliser
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Appliquer KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_clustered['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Sauvegarder le modèle de clustering
            self.kmeans_model = kmeans
            self.cluster_scaler = scaler
            self.cluster_features = features
            
        except Exception as e:
            self.logger.warning(f"Erreur dans cluster_products: {e}")
            # Clustering de fallback
            df_clustered['cluster'] = np.random.randint(0, n_clusters, len(df_clustered))
        
        return df_clustered
    
    def get_top_k_products(self, k: int = 20, score_column: str = 'synthetic_score', 
                          filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Obtenir les top K produits (méthode originale inchangée)
        
        Args:
            k: Nombre de produits à retourner
            score_column: Colonne de score à utiliser
            filters: Filtres à appliquer
            
        Returns:
            DataFrame des top K produits
        """
        if self.df is None or self.df.empty:
            self.get_products_dataframe()
        
        df_filtered = self.df.copy()
        
        # Appliquer les filtres si fournis
        if filters:
            df_filtered = self.filter_data_advanced(df_filtered, filters)
        
        # Vérifier que la colonne de score existe
        if score_column not in df_filtered.columns:
            if 'synthetic_score' in df_filtered.columns:
                score_column = 'synthetic_score'
            else:
                # Créer un score aléatoire
                df_filtered['temp_score'] = np.random.uniform(0, 1, len(df_filtered))
                score_column = 'temp_score'
        
        # Retourner les top K
        top_k = df_filtered.nlargest(k, score_column)
        
        return top_k
    
    def filter_data_advanced(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Appliquer des filtres avancés (méthode originale inchangée)
        
        Args:
            df: DataFrame à filtrer
            filters: Dictionnaire des filtres
            
        Returns:
            DataFrame filtré
        """
        filtered_df = df.copy()
        
        # Filtre de prix
        if 'price_range' in filters and 'price' in filtered_df.columns:
            min_price, max_price = filters['price_range']
            filtered_df = filtered_df[
                (filtered_df['price'] >= min_price) & 
                (filtered_df['price'] <= max_price)
            ]
        
        # Filtre de disponibilité
        if filters.get('available_only', False) and 'available' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['available'] == True]
        
        # Filtre de stock minimum
        if 'min_stock' in filters and 'stock_quantity' in filtered_df.columns:
            min_stock = filters['min_stock']
            filtered_df = filtered_df[filtered_df['stock_quantity'].fillna(0) >= min_stock]
        
        # Filtre de plateforme
        if 'platforms' in filters and filters['platforms'] and 'platform' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['platform'].isin(filters['platforms'])]
        
        # Filtre de région
        if 'regions' in filters and filters['regions'] and 'store_region' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['store_region'].isin(filters['regions'])]
        
        # Filtre de catégorie
        if 'categories' in filters and filters['categories'] and 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # Filtre de rating minimum
        if 'min_rating' in filters and 'rating' in filtered_df.columns:
            min_rating = filters['min_rating']
            filtered_df = filtered_df[filtered_df['rating'].fillna(0) >= min_rating]
        
        return filtered_df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Obtenir un résumé des données (méthode originale légèrement modifiée)
        
        Returns:
            Dictionnaire contenant les statistiques des données
        """
        if self.df is None or self.df.empty:
            self.get_products_dataframe()
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'date_range': None,
            'data_source': 'MongoDB' if self.collection else 'File/Sample'
        }
        
        # Ajouter la plage de dates si disponible
        if 'created_at' in self.df.columns:
            try:
                date_col = pd.to_datetime(self.df['created_at'])
                summary['date_range'] = {
                    'min_date': date_col.min().isoformat(),
                    'max_date': date_col.max().isoformat()
                }
            except:
                pass
        
        return summary
    
    def clean_data(self) -> pd.DataFrame:
        """
        Nettoyer les données (méthode originale inchangée)
        
        Returns:
            DataFrame nettoyé
        """
        if self.df is None or self.df.empty:
            self.get_products_dataframe()
        
        original_rows = len(self.df)
        
        # Supprimer les doublons
        self.df = self.df.drop_duplicates()
        
        # Nettoyer les prix négatifs
        if 'price' in self.df.columns:
            self.df = self.df[self.df['price'] >= 0]
        
        # Nettoyer les scores hors limites
        if 'synthetic_score' in self.df.columns:
            self.df = self.df[
                (self.df['synthetic_score'] >= 0) & 
                (self.df['synthetic_score'] <= 1)
            ]
        
        # Nettoyer les quantités de stock négatives
        if 'stock_quantity' in self.df.columns:
            self.df = self.df[self.df['stock_quantity'] >= 0]
        
        cleaned_rows = len(self.df)
        self.logger.info(f"Nettoyage terminé: {original_rows - cleaned_rows} lignes supprimées")
        
        return self.df
    
    def filter_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Filtrer les données selon les critères spécifiés
        
        Args:
            filters: Dictionnaire des filtres à appliquer
            
        Returns:
            DataFrame filtré
        """
        return self.filter_data_advanced(self.df, filters)
    
    def get_statistics(self, column: str) -> Dict[str, Any]:
        """
        Obtenir les statistiques pour une colonne (méthode originale inchangée)
        
        Args:
            column: Nom de la colonne
            
        Returns:
            Dictionnaire des statistiques
        """
        if self.df is None or column not in self.df.columns:
            self.get_products_dataframe()
            if column not in self.df.columns:
                return {}
        
        col_data = self.df[column]
        
        if col_data.dtype in ['int64', 'float64']:
            return {
                'count': col_data.count(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75)
            }
        else:
            return {
                'count': col_data.count(),
                'unique': col_data.nunique(),
                'most_frequent': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                'frequency': col_data.value_counts().head(10).to_dict()
            }
    
    
    
    def export_data(self, file_path: str, format: str = 'csv') -> bool:
        """
        Exporter les données
        
        Args:
            file_path: Chemin de destination
            format: Format d'export ('csv', 'excel', 'json')
            
        Returns:
            True si l'export a réussi
        """
        try:
            if self.df is None:
                raise ValueError("Aucune donnée à exporter")
            
            if format.lower() == 'csv':
                self.df.to_csv(file_path, index=False)
            elif format.lower() in ['excel', 'xlsx']:
                self.df.to_excel(file_path, index=False)
            elif format.lower() == 'json':
                self.df.to_json(file_path, orient='records', indent=2)
            else:
                raise ValueError(f"Format non supporté: {format}")
            
            self.logger.info(f"Données exportées vers {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export: {str(e)}")
            return False
    
    def reset_data(self) -> pd.DataFrame:
        """
        Remettre les données à leur état original
        
        Returns:
            DataFrame original
        """
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.logger.info("Données remises à l'état original")
        
        return self.df


# Classe utilitaire pour compatibilité
class DashboardUtils:
    """Utilitaires généraux pour le dashboard"""
    
    @staticmethod
    def format_currency(value: float, currency: str = '$') -> str:
        """Formater une valeur monétaire"""
        return f"{currency}{value:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Formater un pourcentage"""
        return f"{value * 100:.{decimals}f}%"
    
    @staticmethod
    def format_number(value: float, decimals: int = 2) -> str:
        """Formater un nombre avec séparateurs de milliers"""
        return f"{value:,.{decimals}f}"


# Fonctions utilitaires pour la compatibilité
def load_dashboard_data(file_path: str) -> pd.DataFrame:
    """Charger les données du dashboard"""
    manager = DashboardDataManager()
    return manager.load_data(file_path)


def get_sample_data() -> pd.DataFrame:
    """Obtenir des données d'exemple"""
    manager = DashboardDataManager()
    return manager.load_sample_data()


def format_metric(value: float, metric_type: str = 'number') -> str:
    """Formater une métrique selon son type"""
    if metric_type == 'currency':
        return DashboardUtils.format_currency(value)
    elif metric_type == 'percentage':
        return DashboardUtils.format_percentage(value)
    else:
        return DashboardUtils.format_number(value)
    
def format_currency(amount, currency_symbol="€", decimal_places=2):
    """
    Formater un montant en devise
    
    Args:
        amount (float): Montant à formater
        currency_symbol (str): Symbole de la devise
        decimal_places (int): Nombre de décimales
    
    Returns:
        str: Montant formaté
    """
    if amount is None or np.isnan(amount):
        return "N/A"
    
    try:
        # Formater avec séparateurs de milliers
        formatted = f"{amount:,.{decimal_places}f}".replace(",", " ")
        return f"{formatted} {currency_symbol}"
    except (ValueError, TypeError):
        return f"{amount} {currency_symbol}"

def format_percentage(value, decimal_places=1):
    """
    Formater une valeur en pourcentage
    
    Args:
        value (float): Valeur décimale (0.25 pour 25%)
        decimal_places (int): Nombre de décimales
    
    Returns:
        str: Pourcentage formaté
    """
    if value is None or np.isnan(value):
        return "N/A"
    
    try:
        percentage = value * 100
        return f"{percentage:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return f"{value}%"

def get_color_palette(palette_name="default", n_colors=10):
    """
    Obtenir une palette de couleurs pour les graphiques
    
    Args:
        palette_name (str): Nom de la palette
        n_colors (int): Nombre de couleurs nécessaires
    
    Returns:
        list: Liste des couleurs en format hex
    """
    palettes = {
        "default": [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ],
        "modern": [
            "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#34495e", "#e67e22", "#95a5a6", "#f1c40f"
        ],
        "pastel": [
            "#a8dadc", "#457b9d", "#1d3557", "#f1faee", "#e63946",
            "#ffd166", "#06ffa5", "#fb8500", "#219ebc", "#023047"
        ],
        "vibrant": [
            "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7",
            "#dda0dd", "#98d8c8", "#f7dc6f", "#bb8fce", "#85c1e9"
        ],
        "business": [
            "#2c3e50", "#3498db", "#e74c3c", "#f39c12", "#27ae60",
            "#8e44ad", "#34495e", "#e67e22", "#16a085", "#f1c40f"
        ]
    }
    
    # Sélectionner la palette
    if palette_name in palettes:
        colors = palettes[palette_name]
    else:
        colors = palettes["default"]
    
    # Étendre la palette si nécessaire
    if n_colors > len(colors):
        # Répéter les couleurs ou utiliser une palette Plotly
        extended_colors = colors * (n_colors // len(colors) + 1)
        return extended_colors[:n_colors]
    
    return colors[:n_colors]

def format_number(number, suffix="", decimal_places=0):
    """
    Formater un nombre avec des abréviations (K, M, B)
    
    Args:
        number (float): Nombre à formater
        suffix (str): Suffixe à ajouter
        decimal_places (int): Nombre de décimales
    
    Returns:
        str: Nombre formaté
    """
    if number is None or np.isnan(number):
        return "N/A"
    
    try:
        abs_number = abs(number)
        
        if abs_number >= 1_000_000_000:
            formatted = f"{number / 1_000_000_000:.{decimal_places}f}B"
        elif abs_number >= 1_000_000:
            formatted = f"{number / 1_000_000:.{decimal_places}f}M"
        elif abs_number >= 1_000:
            formatted = f"{number / 1_000:.{decimal_places}f}K"
        else:
            formatted = f"{number:.{decimal_places}f}"
        
        return f"{formatted}{suffix}"
    except (ValueError, TypeError):
        return str(number)

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """
    Crée une carte de métrique HTML stylisée pour Streamlit
    
    Args:
        title (str): Titre de la métrique
        value (str): Valeur de la métrique
        delta (str, optional): Variation (ex: "+5%", "-2%")
        delta_color (str): Couleur du delta ("normal", "inverse")
    
    Returns:
        str: Code HTML de la carte de métrique
    """
    
    # Couleur du delta
    if delta:
        if delta_color == "inverse":
            delta_color_class = "red" if delta.startswith("+") else "green"
        else:
            delta_color_class = "green" if delta.startswith("+") else "red"
    else:
        delta_color_class = ""
    
    # Code HTML de la carte
    card_html = f"""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        color: white;
    ">
        <h4 style="
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
            margin: 0 0 0.5rem 0;
            font-weight: 500;
        ">{title}</h4>
        <div style="
            font-size: 1.8rem;
            font-weight: bold;
            margin: 0;
            color: white;
        ">{value}</div>
        {f'<div style="color: {delta_color_class}; font-size: 0.8rem; margin-top: 0.3rem;">{delta}</div>' if delta else ''}
    </div>
    """
    
    return card_html
def format_currency(value):
    """
    Formate une valeur numérique en devise
    
    Args:
        value (float): Valeur à formater
    
    Returns:
        str: Valeur formatée en euros
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f} €"

def format_number(value):
    """
    Formate un nombre avec des séparateurs de milliers
    
    Args:
        value (int/float): Nombre à formater
    
    Returns:
        str: Nombre formaté
    """
    if value is None or pd.isna(value):
        return "0"
    return f"{int(value):,}".replace(",", " ")

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """
    Crée une carte de métrique HTML stylisée pour Streamlit
    
    Args:
        title (str): Titre de la métrique
        value (str): Valeur de la métrique
        delta (str, optional): Variation (ex: "+5%", "-2%")
        delta_color (str): Couleur du delta ("normal", "inverse")
    
    Returns:
        str: Code HTML de la carte de métrique
    """
    
    # Couleur du delta
    if delta:
        if delta_color == "inverse":
            delta_color_class = "red" if delta.startswith("+") else "green"
        else:
            delta_color_class = "green" if delta.startswith("+") else "red"
    else:
        delta_color_class = ""
    
    # Code HTML de la carte
    card_html = f"""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        color: white;
    ">
        <h4 style="
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
            margin: 0 0 0.5rem 0;
            font-weight: 500;
        ">{title}</h4>
        <div style="
            font-size: 1.8rem;
            font-weight: bold;
            margin: 0;
            color: white;
        ">{value}</div>
        {f'<div style="color: {delta_color_class}; font-size: 0.8rem; margin-top: 0.3rem;">{delta}</div>' if delta else ''}
    </div>
    """
    
    return card_html

def get_color_palette():
    """
    Retourne une palette de couleurs pour les graphiques
    
    Returns:
        list: Liste de couleurs hexadécimales
    """
    return [
        '#667eea', '#764ba2', '#f093fb', '#f5576c',
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
        '#ffecd2', '#fcb69f', '#a8edea', '#fed6e3',
        '#fad0c4', '#ffd1ff', '#c2e9fb', '#a1c4fd'
    ]

def create_donut_chart(values, labels, title):
    """
    Crée un graphique en donut (camembert troué)

    Args:
        values (list): Valeurs pour chaque section
        labels (list): Labels pour chaque section
        title (str): Titre du graphique

    Returns:
        plotly.graph_objects.Figure: Graphique Plotly
    """
    colors = get_color_palette()

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors[:len(labels)],
        textinfo='label+percent',
        textposition='outside',
        showlegend=True
    )])

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        font=dict(size=12)
    )

    return fig

def create_bar_chart(x_data, y_data, title, x_title="", y_title="", orientation="v"):
    """
    Crée un graphique en barres
    
    Args:
        x_data (list): Données pour l'axe X
        y_data (list): Données pour l'axe Y
        title (str): Titre du graphique
        x_title (str): Titre de l'axe X
        y_title (str): Titre de l'axe Y
        orientation (str): Orientation des barres ("v" pour vertical, "h" pour horizontal)
    
    Returns:
        plotly.graph_objects.Figure: Graphique Plotly
    """
    colors = get_color_palette()
    
    if orientation == "h":
        fig = go.Figure(data=[go.Bar(
            x=y_data,
            y=x_data,
            orientation='h',
            marker_color=colors[0],
            text=y_data,
            textposition='outside'
        )])
    else:
        fig = go.Figure(data=[go.Bar(
            x=x_data,
            y=y_data,
            marker_color=colors[0],
            text=y_data,
            textposition='outside'
        )])
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        showlegend=False
    )
    
    # Amélioration de l'affichage des axes
    if orientation == "v":
        fig.update_xaxes(tickangle=0)
    
    return fig

def create_line_chart(x_data, y_data, title, x_title="", y_title="", line_name=""):
    """
    Crée un graphique en ligne
    
    Args:
        x_data (list): Données pour l'axe X
        y_data (list): Données pour l'axe Y
        title (str): Titre du graphique
        x_title (str): Titre de l'axe X
        y_title (str): Titre de l'axe Y
        line_name (str): Nom de la ligne pour la légende
    
    Returns:
        plotly.graph_objects.Figure: Graphique Plotly
    """
    colors = get_color_palette()
    
    fig = go.Figure(data=[go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers',
        name=line_name,
        line=dict(color=colors[0], width=3),
        marker=dict(size=6, color=colors[1])
    )])
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        hovermode='x unified'
    )
    
    return fig

def create_scatter_plot(x_data, y_data, title, x_title="", y_title="", size_data=None, color_data=None):
    """
    Crée un graphique de dispersion (scatter plot)
    
    Args:
        x_data (list): Données pour l'axe X
        y_data (list): Données pour l'axe Y
        title (str): Titre du graphique
        x_title (str): Titre de l'axe X
        y_title (str): Titre de l'axe Y
        size_data (list, optional): Données pour la taille des points
        color_data (list, optional): Données pour la couleur des points
    
    Returns:
        plotly.graph_objects.Figure: Graphique Plotly
    """
    colors = get_color_palette()
    
    scatter = go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(
            size=size_data if size_data else 10,
            color=color_data if color_data else colors[0],
            colorscale='Viridis' if color_data else None,
            showscale=True if color_data else False
        )
    )
    
    fig = go.Figure(data=[scatter])
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig
    
@st.cache_resource
def init_analyzer():
    """Initialiser l'analyseur de produits avec cache Streamlit"""
    try:
        analyzer = ProductAnalyzer()
        if analyzer.client:
            return analyzer
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de l'analyseur: {e}")
        return None

def load_custom_css():
    """Charger le CSS personnalisé pour le dashboard"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2em;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card h3 {
        color: #333;
        margin: 0 0 0.5rem 0;
        font-size: 1.1em;
    }
    
    .metric-card .big-number {
        font-size: 2.5em;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .stAlert > div {
        border-radius: 10px;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
