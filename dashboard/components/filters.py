import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class FilterComponents:
    """Composants de filtrage pour le dashboard"""
    
    @staticmethod
    def create_date_filter(df: pd.DataFrame, date_column: str = 'created_at') -> Tuple[date, date]:
        """
        Créer un filtre de date
        
        Args:
            df: DataFrame contenant les données
            date_column: Nom de la colonne de date
            
        Returns:
            Tuple contenant (date_début, date_fin)
        """
        if date_column not in df.columns:
            st.warning(f"Colonne '{date_column}' non trouvée dans les données")
            return date.today() - timedelta(days=30), date.today()
        
        # Convertir la colonne en datetime si nécessaire
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            min_date = df[date_column].min().date()
            max_date = df[date_column].max().date()
        except:
            st.warning(f"Impossible de convertir la colonne '{date_column}' en date")
            return date.today() - timedelta(days=30), date.today()
        
        st.subheader("📅 Filtre de Date")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Date de début",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="start_date_filter"
            )
        
        with col2:
            end_date = st.date_input(
                "Date de fin",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="end_date_filter"
            )
        
        # Validation des dates
        if start_date > end_date:
            st.error("La date de début doit être antérieure à la date de fin")
            return min_date, max_date
        
        return start_date, end_date
    
    @staticmethod
    def create_platform_filter(df: pd.DataFrame, platform_column: str = 'platform') -> List[str]:
        """
        Créer un filtre de plateforme
        
        Args:
            df: DataFrame contenant les données
            platform_column: Nom de la colonne de plateforme
            
        Returns:
            Liste des plateformes sélectionnées
        """
        if platform_column not in df.columns:
            st.warning(f"Colonne '{platform_column}' non trouvée dans les données")
            return []
        
        unique_platforms = sorted(df[platform_column].dropna().unique())
        
        if not unique_platforms:
            st.warning("Aucune plateforme trouvée dans les données")
            return []
        
        st.subheader("🏪 Filtre de Plateforme")
        
        # Option pour sélectionner toutes les plateformes
        select_all = st.checkbox("Sélectionner toutes les plateformes", value=True, key="select_all_platforms")
        
        if select_all:
            selected_platforms = unique_platforms
        else:
            selected_platforms = st.multiselect(
                "Choisir les plateformes",
                options=unique_platforms,
                default=unique_platforms[:3] if len(unique_platforms) >= 3 else unique_platforms,
                key="platform_multiselect"
            )
        
        return selected_platforms
    
    @staticmethod
    def create_price_filter(df: pd.DataFrame, price_column: str = 'price') -> Tuple[float, float]:
        """
        Créer un filtre de prix
        
        Args:
            df: DataFrame contenant les données
            price_column: Nom de la colonne de prix
            
        Returns:
            Tuple contenant (prix_min, prix_max)
        """
        if price_column not in df.columns:
            st.warning(f"Colonne '{price_column}' non trouvée dans les données")
            return 0.0, 100.0
        
        # Nettoyer les données de prix
        price_data = df[price_column].dropna()
        if price_data.empty:
            st.warning("Aucune donnée de prix disponible")
            return 0.0, 100.0
        
        min_price = float(price_data.min())
        max_price = float(price_data.max())
        
        st.subheader("💰 Filtre de Prix")
        
        price_range = st.slider(
            "Gamme de prix",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=(max_price - min_price) / 100 if max_price > min_price else 1.0,
            key="price_range_slider"
        )
        
        return price_range
    
    @staticmethod
    def create_score_filter(df: pd.DataFrame, score_column: str = 'synthetic_score') -> Tuple[float, float]:
        """
        Créer un filtre de score synthétique
        
        Args:
            df: DataFrame contenant les données
            score_column: Nom de la colonne de score
            
        Returns:
            Tuple contenant (score_min, score_max)
        """
        if score_column not in df.columns:
            st.warning(f"Colonne '{score_column}' non trouvée dans les données")
            return 0.0, 1.0
        
        score_data = df[score_column].dropna()
        if score_data.empty:
            st.warning("Aucune donnée de score disponible")
            return 0.0, 1.0
        
        min_score = float(score_data.min())
        max_score = float(score_data.max())
        
        st.subheader("⭐ Filtre de Score")
        
        score_range = st.slider(
            "Gamme de score",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
            step=(max_score - min_score) / 100 if max_score > min_score else 0.01,
            key="score_range_slider"
        )
        
        return score_range
    
    @staticmethod
    def create_availability_filter(df: pd.DataFrame, availability_column: str = 'available') -> str:
        """
        Créer un filtre de disponibilité
        
        Args:
            df: DataFrame contenant les données
            availability_column: Nom de la colonne de disponibilité
            
        Returns:
            Option de disponibilité sélectionnée
        """
        if availability_column not in df.columns:
            st.warning(f"Colonne '{availability_column}' non trouvée dans les données")
            return "Tous"
        
        st.subheader("✅ Filtre de Disponibilité")
        
        availability_options = ["Tous", "Disponibles seulement", "Non disponibles seulement"]
        
        selected_availability = st.radio(
            "Statut de disponibilité",
            options=availability_options,
            index=0,
            key="availability_radio"
        )
        
        return selected_availability
    
    @staticmethod
    def create_vendor_filter(df: pd.DataFrame, vendor_column: str = 'vendor', max_vendors: int = 10) -> List[str]:
        """
        Créer un filtre de vendeur
        
        Args:
            df: DataFrame contenant les données
            vendor_column: Nom de la colonne de vendeur
            max_vendors: Nombre maximum de vendeurs à afficher
            
        Returns:
            Liste des vendeurs sélectionnés
        """
        if vendor_column not in df.columns:
            st.warning(f"Colonne '{vendor_column}' non trouvée dans les données")
            return []
        
        # Obtenir les vendeurs les plus fréquents
        vendor_counts = df[vendor_column].value_counts().head(max_vendors)
        top_vendors = vendor_counts.index.tolist()
        
        if not top_vendors:
            st.warning("Aucun vendeur trouvé dans les données")
            return []
        
        st.subheader("👥 Filtre de Vendeur")
        
        selected_vendors = st.multiselect(
            f"Choisir les vendeurs (Top {max_vendors})",
            options=top_vendors,
            default=top_vendors[:5] if len(top_vendors) >= 5 else top_vendors,
            key="vendor_multiselect"
        )
        
        return selected_vendors
    
    @staticmethod
    def create_region_filter(df: pd.DataFrame, region_column: str = 'store_region') -> List[str]:
        """
        Créer un filtre de région
        
        Args:
            df: DataFrame contenant les données
            region_column: Nom de la colonne de région
            
        Returns:
            Liste des régions sélectionnées
        """
        if region_column not in df.columns:
            st.warning(f"Colonne '{region_column}' non trouvée dans les données")
            return []
        
        unique_regions = sorted(df[region_column].dropna().unique())
        
        if not unique_regions:
            st.warning("Aucune région trouvée dans les données")
            return []
        
        st.subheader("🌍 Filtre de Région")
        
        selected_regions = st.multiselect(
            "Choisir les régions",
            options=unique_regions,
            default=unique_regions,
            key="region_multiselect"
        )
        
        return selected_regions
    
    @staticmethod
    def create_stock_filter(df: pd.DataFrame, stock_column: str = 'stock_quantity') -> str:
        """
        Créer un filtre de niveau de stock
        
        Args:
            df: DataFrame contenant les données
            stock_column: Nom de la colonne de stock
            
        Returns:
            Catégorie de stock sélectionnée
        """
        if stock_column not in df.columns:
            st.warning(f"Colonne '{stock_column}' non trouvée dans les données")
            return "Tous"
        
        st.subheader("📦 Filtre de Stock")
        
        stock_categories = [
            "Tous",
            "Stock Faible (0-10)",
            "Stock Moyen (11-50)",
            "Stock Élevé (51-100)",
            "Stock Très Élevé (100+)"
        ]
        
        selected_stock_category = st.selectbox(
            "Niveau de stock",
            options=stock_categories,
            index=0,
            key="stock_category_select"
        )
        
        return selected_stock_category


# Fonction principale pour créer tous les filtres
def create_comprehensive_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Créer tous les filtres pour le dashboard
    
    Args:
        df: DataFrame contenant les données
        
    Returns:
        Dictionnaire contenant tous les filtres appliqués
    """
    filters = {}
    
    st.sidebar.header("🔍 Filtres")
    
    # Filtre de date
    if 'created_at' in df.columns:
        with st.sidebar.expander("📅 Date", expanded=True):
            filters['date_range'] = FilterComponents.create_date_filter(df)
    
    # Filtre de plateforme
    if 'platform' in df.columns:
        with st.sidebar.expander("🏪 Plateforme", expanded=True):
            filters['platforms'] = FilterComponents.create_platform_filter(df)
    
    # Filtre de prix
    if 'price' in df.columns:
        with st.sidebar.expander("💰 Prix", expanded=False):
            filters['price_range'] = FilterComponents.create_price_filter(df)
    
    # Filtre de score
    if 'synthetic_score' in df.columns:
        with st.sidebar.expander("⭐ Score", expanded=False):
            filters['score_range'] = FilterComponents.create_score_filter(df)
    
    # Filtre de disponibilité
    if 'available' in df.columns:
        with st.sidebar.expander("✅ Disponibilité", expanded=False):
            filters['availability'] = FilterComponents.create_availability_filter(df)
    
    # Filtre de vendeur
    if 'vendor' in df.columns:
        with st.sidebar.expander("👥 Vendeur", expanded=False):
            filters['vendors'] = FilterComponents.create_vendor_filter(df)
    
    # Filtre de région
    if 'store_region' in df.columns:
        with st.sidebar.expander("🌍 Région", expanded=False):
            filters['regions'] = FilterComponents.create_region_filter(df)
    
    # Filtre de stock
    if 'stock_quantity' in df.columns:
        with st.sidebar.expander("📦 Stock", expanded=False):
            filters['stock_category'] = FilterComponents.create_stock_filter(df)
    
    return filters


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Appliquer les filtres au DataFrame
    
    Args:
        df: DataFrame original
        filters: Dictionnaire des filtres à appliquer
        
    Returns:
        DataFrame filtré
    """
    filtered_df = df.copy()
    
    # Appliquer le filtre de date
    if 'date_range' in filters and 'created_at' in filtered_df.columns:
        start_date, end_date = filters['date_range']
        filtered_df['created_at'] = pd.to_datetime(filtered_df['created_at'])
        mask = (filtered_df['created_at'].dt.date >= start_date) & (filtered_df['created_at'].dt.date <= end_date)
        filtered_df = filtered_df[mask]
    
    # Appliquer le filtre de plateforme
    if 'platforms' in filters and filters['platforms'] and 'platform' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['platform'].isin(filters['platforms'])]
    
    # Appliquer le filtre de prix
    if 'price_range' in filters and 'price' in filtered_df.columns:
        min_price, max_price = filters['price_range']
        filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]
    
    # Appliquer le filtre de score
    if 'score_range' in filters and 'synthetic_score' in filtered_df.columns:
        min_score, max_score = filters['score_range']
        filtered_df = filtered_df[(filtered_df['synthetic_score'] >= min_score) & (filtered_df['synthetic_score'] <= max_score)]
    
    # Appliquer le filtre de disponibilité
    if 'availability' in filters and 'available' in filtered_df.columns:
        availability = filters['availability']
        if availability == "Disponibles seulement":
            filtered_df = filtered_df[filtered_df['available'] == True]
        elif availability == "Non disponibles seulement":
            filtered_df = filtered_df[filtered_df['available'] == False]
    
    # Appliquer le filtre de vendeur
    if 'vendors' in filters and filters['vendors'] and 'vendor' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['vendor'].isin(filters['vendors'])]
    
    # Appliquer le filtre de région
    if 'regions' in filters and filters['regions'] and 'store_region' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['store_region'].isin(filters['regions'])]
    
    # Appliquer le filtre de stock
    if 'stock_category' in filters and 'stock_quantity' in filtered_df.columns:
        stock_category = filters['stock_category']
        if stock_category != "Tous":
            if stock_category == "Stock Faible (0-10)":
                filtered_df = filtered_df[(filtered_df['stock_quantity'] >= 0) & (filtered_df['stock_quantity'] <= 10)]
            elif stock_category == "Stock Moyen (11-50)":
                filtered_df = filtered_df[(filtered_df['stock_quantity'] >= 11) & (filtered_df['stock_quantity'] <= 50)]
            elif stock_category == "Stock Élevé (51-100)":
                filtered_df = filtered_df[(filtered_df['stock_quantity'] >= 51) & (filtered_df['stock_quantity'] <= 100)]
            elif stock_category == "Stock Très Élevé (100+)":
                filtered_df = filtered_df[filtered_df['stock_quantity'] > 100]
    
    return filtered_df


# Fonctions individuelles pour la compatibilité
def create_date_filter(df: pd.DataFrame, date_column: str = 'created_at') -> Tuple[date, date]:
    """Fonction wrapper pour create_date_filter"""
    return FilterComponents.create_date_filter(df, date_column)


def create_platform_filter(df: pd.DataFrame, platform_column: str = 'platform') -> List[str]:
    """Fonction wrapper pour create_platform_filter"""
    return FilterComponents.create_platform_filter(df, platform_column)
def create_advanced_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Créer des filtres avancés pour le dashboard
    
    Args:
        df: DataFrame contenant les données
        
    Returns:
        Dictionnaire contenant tous les filtres appliqués
    """
    filters = {}
    
    # Filtre de date
    if 'created_at' in df.columns:
        filters['date_range'] = FilterComponents.create_date_filter(df)
    
    # Filtre de plateforme
    if 'platform' in df.columns:
        filters['platforms'] = FilterComponents.create_platform_filter(df)
    
    # Filtre de prix
    if 'price' in df.columns:
        filters['price_range'] = FilterComponents.create_price_filter(df)
    
    # Filtre de score
    if 'synthetic_score' in df.columns:
        filters['score_range'] = FilterComponents.create_score_filter(df)
    
    # Filtre de disponibilité
    if 'available' in df.columns:
        filters['availability'] = FilterComponents.create_availability_filter(df)
    
    # Filtre de vendeur
    if 'vendor' in df.columns:
        filters['vendors'] = FilterComponents.create_vendor_filter(df)
    
    # Filtre de région
    if 'store_region' in df.columns:
        filters['regions'] = FilterComponents.create_region_filter(df)
    
    # Filtre de stock
    if 'stock_quantity' in df.columns:
        filters['stock_category'] = FilterComponents.create_stock_filter(df)
    
    return filters
def create_geography_filters():
    """
    Créer les filtres spécifiques à l'analyse géographique
    
    Returns:
        dict: Dictionnaire contenant tous les filtres sélectionnés
    """
    filters = {}
    
    # Filtre par régions
    st.sidebar.subheader("🌍 Sélection des Régions")
    
    # Option pour sélectionner toutes les régions ou des régions spécifiques
    region_selection_type = st.sidebar.radio(
        "Type de sélection:",
        ["Toutes les régions", "Régions spécifiques"],
        key="region_selection_type"
    )
    
    if region_selection_type == "Régions spécifiques":
        # Liste des régions communes (à adapter selon vos données)
        available_regions = [
            "Île-de-France", "Auvergne-Rhône-Alpes", "Nouvelle-Aquitaine",
            "Occitanie", "Hauts-de-France", "Grand Est", "Provence-Alpes-Côte d'Azur",
            "Pays de la Loire", "Normandie", "Bretagne", "Bourgogne-Franche-Comté",
            "Centre-Val de Loire", "Corse"
        ]
        
        selected_regions = st.sidebar.multiselect(
            "Sélectionner les régions:",
            available_regions,
            default=available_regions[:5],  # Sélectionner les 5 premières par défaut
            key="selected_regions"
        )
        
        if selected_regions:
            filters['regions'] = selected_regions
    
    # Filtre par gamme de prix
    st.sidebar.subheader("💰 Gamme de Prix")
    
    price_filter_enabled = st.sidebar.checkbox("Activer le filtre de prix", key="price_filter_enabled")
    
    if price_filter_enabled:
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            min_price = st.number_input(
                "Prix min (€):",
                min_value=0.0,
                max_value=10000.0,
                value=0.0,
                step=10.0,
                key="min_price"
            )
        
        with col2:
            max_price = st.number_input(
                "Prix max (€):",
                min_value=0.0,
                max_value=10000.0,
                value=1000.0,
                step=10.0,
                key="max_price"
            )
        
        if min_price < max_price:
            filters['min_price'] = min_price
            filters['max_price'] = max_price
        else:
            st.sidebar.error("Le prix minimum doit être inférieur au prix maximum")
    
    # Filtre de disponibilité
    st.sidebar.subheader("📦 Disponibilité")
    
    availability_only = st.sidebar.checkbox(
        "Produits disponibles uniquement",
        value=True,
        key="availability_only"
    )
    filters['availability_only'] = availability_only
    
    # Filtre par niveau de stock
    st.sidebar.subheader("📊 Niveau de Stock")
    
    stock_filter_enabled = st.sidebar.checkbox("Filtrer par stock", key="stock_filter_enabled")
    
    if stock_filter_enabled:
        stock_threshold = st.sidebar.slider(
            "Stock minimum:",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
            key="stock_threshold"
        )
        filters['min_stock'] = stock_threshold
    
    # Filtre par score de performance
    st.sidebar.subheader("⭐ Score de Performance")
    
    score_filter_enabled = st.sidebar.checkbox("Filtrer par score", key="score_filter_enabled")
    
    if score_filter_enabled:
        min_score = st.sidebar.slider(
            "Score minimum:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            key="min_score"
        )
        filters['min_score'] = min_score
    
    # Filtre par vendeur
    st.sidebar.subheader("🏪 Vendeurs")
    
    vendor_filter_enabled = st.sidebar.checkbox("Filtrer par vendeurs", key="vendor_filter_enabled")
    
    if vendor_filter_enabled:
        # Liste des vendeurs les plus populaires (à adapter selon vos données)
        popular_vendors = [
            "Allibirds", "Shark"
        ]
        
        selected_vendors = st.sidebar.multiselect(
            "Sélectionner les vendeurs:",
            popular_vendors,
            key="selected_vendors"
        )
        
        if selected_vendors:
            filters['vendors'] = selected_vendors
    
    return filters
