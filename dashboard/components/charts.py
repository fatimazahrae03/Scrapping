import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from datetime import datetime

class ChartComponents:
    """Composants de visualisation pour le dashboard"""
    
    @staticmethod
    def create_kpi_cards(df: pd.DataFrame) -> None:
        """Créer les cartes KPI principales"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products = len(df)
            st.metric(
                label="Produits Total",
                value=f"{total_products:,}",
                delta=None
            )
        
        with col2:
            available_products = df['available'].sum() if 'available' in df.columns else 0
            availability_rate = (available_products / total_products * 100) if total_products > 0 else 0
            st.metric(
                label="Produits Disponibles",
                value=f"{available_products:,}",
                delta=f"{availability_rate:.1f}%"
            )
        
        with col3:
            avg_price = df['price'].mean() if 'price' in df.columns else 0
            price_std = df['price'].std() if 'price' in df.columns else 0
            st.metric(
                label="Prix Moyen",
                value=f"${avg_price:.2f}",
                delta=f"±${price_std:.2f}"
            )
        
        with col4:
            avg_score = df['synthetic_score'].mean() if 'synthetic_score' in df.columns else 0
            max_score = df['synthetic_score'].max() if 'synthetic_score' in df.columns else 0
            st.metric(
                label="Score Moyen",
                value=f"{avg_score:.3f}",
                delta=f"Max: {max_score:.3f}"
            )
    
    @staticmethod
    def price_distribution_chart(df: pd.DataFrame) -> go.Figure:
        """Graphique de distribution des prix"""
        if 'price' not in df.columns:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Distribution des Prix', 'Box Plot des Prix'),
            vertical_spacing=0.12
        )
        
        # Histogramme
        fig.add_trace(
            go.Histogram(
                x=df['price'],
                nbinsx=30,
                name='Distribution',
                opacity=0.7,
                marker_color='skyblue'
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=df['price'],
                name='Prix',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="Analyse de la Distribution des Prix"
        )
        
        return fig
    
    @staticmethod
    def score_vs_price_scatter(df: pd.DataFrame) -> go.Figure:
        """Scatter plot Score vs Prix"""
        if 'price' not in df.columns or 'synthetic_score' not in df.columns:
            return go.Figure()
        
        # Ajouter une couleur basée sur la disponibilité
        color = df['available'].map({True: 'Disponible', False: 'Indisponible'}) if 'available' in df.columns else 'blue'
        
        fig = px.scatter(
            df,
            x='price',
            y='synthetic_score',
            color=color if isinstance(color, pd.Series) else None,
            size='stock_quantity' if 'stock_quantity' in df.columns else None,
            hover_data=['title', 'vendor'] if all(col in df.columns for col in ['title', 'vendor']) else None,
            title="Relation Prix vs Score Synthétique",
            labels={
                'price': 'Prix ($)',
                'synthetic_score': 'Score Synthétique'
            }
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def platform_comparison_chart(df: pd.DataFrame) -> go.Figure:
        """Graphique de comparaison par plateforme"""
        if 'platform' not in df.columns:
            return go.Figure()
        
        platform_stats = df.groupby('platform').agg({
            'price': ['mean', 'count'],
            'synthetic_score': 'mean',
            'available': 'sum'
        }).round(2)
        
        platform_stats.columns = ['avg_price', 'product_count', 'avg_score', 'available_count']
        platform_stats = platform_stats.reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Nombre de Produits par Plateforme',
                'Prix Moyen par Plateforme',
                'Score Moyen par Plateforme',
                'Produits Disponibles par Plateforme'
            )
        )
        
        # Nombre de produits
        fig.add_trace(
            go.Bar(x=platform_stats['platform'], y=platform_stats['product_count'], name='Produits'),
            row=1, col=1
        )
        
        # Prix moyen
        fig.add_trace(
            go.Bar(x=platform_stats['platform'], y=platform_stats['avg_price'], name='Prix Moyen'),
            row=1, col=2
        )
        
        # Score moyen
        fig.add_trace(
            go.Bar(x=platform_stats['platform'], y=platform_stats['avg_score'], name='Score Moyen'),
            row=2, col=1
        )
        
        # Disponibilité
        fig.add_trace(
            go.Bar(x=platform_stats['platform'], y=platform_stats['available_count'], name='Disponibles'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        return fig
    
    @staticmethod
    def vendor_performance_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Graphique de performance des vendeurs"""
        if 'vendor' not in df.columns:
            return go.Figure()
        
        vendor_stats = df.groupby('vendor').agg({
            'synthetic_score': ['mean', 'count'],
            'price': 'mean',
            'available': 'sum'
        }).round(3)
        
        vendor_stats.columns = ['avg_score', 'product_count', 'avg_price', 'available_count']
        vendor_stats = vendor_stats.reset_index()
        
        # Filtrer les top vendeurs
        top_vendors = vendor_stats.nlargest(top_n, 'avg_score')
        
        fig = px.scatter(
            top_vendors,
            x='avg_price',
            y='avg_score',
            size='product_count',
            color='available_count',
            hover_name='vendor',
            title=f"Performance des Top {top_n} Vendeurs",
            labels={
                'avg_price': 'Prix Moyen ($)',
                'avg_score': 'Score Moyen',
                'product_count': 'Nombre de Produits',
                'available_count': 'Produits Disponibles'
            }
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def geographic_heatmap(df: pd.DataFrame) -> go.Figure:
        """Heatmap géographique des performances"""
        if 'store_region' not in df.columns:
            return go.Figure()
        
        geo_stats = df.groupby('store_region').agg({
            'synthetic_score': 'mean',
            'price': 'mean',
            'available': ['sum', 'count']
        }).round(3)
        
        geo_stats.columns = ['avg_score', 'avg_price', 'available_sum', 'total_products']
        geo_stats['availability_rate'] = (geo_stats['available_sum'] / geo_stats['total_products'] * 100).round(1)
        geo_stats = geo_stats.reset_index()
        
        fig = px.bar(
            geo_stats,
            y='store_region',
            x='avg_score',
            color='availability_rate',
            orientation='h',
            title="Performance par Région",
            labels={
                'store_region': 'Région',
                'avg_score': 'Score Moyen',
                'availability_rate': 'Taux de Disponibilité (%)'
            }
        )
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def stock_analysis_chart(df: pd.DataFrame) -> go.Figure:
        """Analyse des stocks"""
        if 'stock_quantity' not in df.columns:
            return go.Figure()
        
        # Catégoriser les stocks
        df_stock = df.copy()
        df_stock['stock_category'] = pd.cut(
            df_stock['stock_quantity'],
            bins=[0, 10, 50, 100, float('inf')],
            labels=['Stock Faible (0-10)', 'Stock Moyen (11-50)', 'Stock Élevé (51-100)', 'Stock Très Élevé (100+)']
        )
        
        stock_distribution = df_stock['stock_category'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=stock_distribution.index,
                values=stock_distribution.values,
                hole=.3,
                textinfo='label+percent',
                title="Distribution des Niveaux de Stock"
            )
        ])
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def trend_analysis_chart(df: pd.DataFrame) -> go.Figure:
        """Analyse des tendances temporelles"""
        if 'created_at' not in df.columns:
            return go.Figure()
        
        df_trend = df.copy()
        df_trend['created_at'] = pd.to_datetime(df_trend['created_at'])
        df_trend['month'] = df_trend['created_at'].dt.to_period('M')
        
        monthly_stats = df_trend.groupby('month').agg({
            'synthetic_score': 'mean',
            'price': 'mean',
            'available': 'sum'
        }).round(2)
        
        monthly_stats = monthly_stats.reset_index()
        monthly_stats['month'] = monthly_stats['month'].astype(str)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Score Moyen Mensuel', 'Prix Moyen Mensuel', 'Produits Disponibles Mensuels'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_stats['month'], y=monthly_stats['synthetic_score'], 
                      mode='lines+markers', name='Score'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_stats['month'], y=monthly_stats['price'], 
                      mode='lines+markers', name='Prix'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_stats['month'], y=monthly_stats['available'], 
                      mode='lines+markers', name='Disponibles'),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        return fig
    
    @staticmethod
    def clustering_visualization(df: pd.DataFrame) -> go.Figure:
        """Visualisation des clusters de produits"""
        if 'cluster' not in df.columns:
            return go.Figure()
        
        fig = px.scatter_3d(
            df,
            x='price',
            y='synthetic_score',
            z='stock_quantity' if 'stock_quantity' in df.columns else 'available',
            color='cluster',
            size='stock_quantity' if 'stock_quantity' in df.columns else None,
            hover_data=['title', 'vendor'] if all(col in df.columns for col in ['title', 'vendor']) else None,
            title="Visualisation des Clusters de Produits",
            labels={
                'price': 'Prix ($)',
                'synthetic_score': 'Score Synthétique',
                'stock_quantity': 'Stock',
                'cluster': 'Cluster'
            }
        )
        
        fig.update_layout(height=600)
        return fig
    
    @staticmethod
    def comparison_radar_chart(df: pd.DataFrame, product_ids: List[str]) -> go.Figure:
        """Graphique radar pour comparer des produits"""
        if not product_ids or '_id' not in df.columns:
            return go.Figure()
        
        # Sélectionner les produits à comparer
        selected_products = df[df['_id'].astype(str).isin(product_ids)]
        
        if selected_products.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        categories = ['Prix (normalisé)', 'Score', 'Stock (normalisé)', 'Disponibilité']
        
        for idx, (_, product) in enumerate(selected_products.iterrows()):
            # Normaliser les valeurs pour le radar
            price_norm = 1 - (product.get('price', 0) / df['price'].max()) if 'price' in df.columns else 0
            score_norm = product.get('synthetic_score', 0)
            stock_norm = product.get('stock_quantity', 0) / df['stock_quantity'].max() if 'stock_quantity' in df.columns else 0
            availability = float(product.get('available', 0))
            
            values = [price_norm, score_norm, stock_norm, availability]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=product.get('title', f'Produit {idx+1}')[:30] + '...',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Comparaison Radar des Produits Sélectionnés",
            height=500
        )
        
        return fig


# Function to create overview charts - This is the missing function
def create_overview_charts(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create all overview charts for the dashboard
    
    Args:
        df: DataFrame containing product data
        
    Returns:
        Dictionary containing all charts for the overview page
    """
    charts = {}
    
    # Price distribution chart
    charts['price_distribution'] = ChartComponents.price_distribution_chart(df)
    
    # Score vs Price scatter plot
    charts['score_vs_price'] = ChartComponents.score_vs_price_scatter(df)
    
    # Platform comparison
    charts['platform_comparison'] = ChartComponents.platform_comparison_chart(df)
    
    # Stock analysis
    charts['stock_analysis'] = ChartComponents.stock_analysis_chart(df)
    
    # Trend analysis (if date column exists)
    charts['trend_analysis'] = ChartComponents.trend_analysis_chart(df)
    
    # Geographic heatmap (if region column exists)
    charts['geographic_heatmap'] = ChartComponents.geographic_heatmap(df)
    
    return charts


# Alternative function for creating overview charts with simplified layout
def create_overview_charts_simple(df: pd.DataFrame) -> None:
    """
    Create overview charts and display them directly in Streamlit
    
    Args:
        df: DataFrame containing product data
    """
    # Create KPI cards first
    st.subheader("📊 Indicateurs Clés de Performance")
    ChartComponents.create_kpi_cards(df)
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Distribution des Prix")
        price_chart = ChartComponents.price_distribution_chart(df)
        if price_chart.data:
            st.plotly_chart(price_chart, use_container_width=True)
        else:
            st.info("Données de prix non disponibles")
    
    with col2:
        st.subheader("📈 Score vs Prix")
        scatter_chart = ChartComponents.score_vs_price_scatter(df)
        if scatter_chart.data:
            st.plotly_chart(scatter_chart, use_container_width=True)
        else:
            st.info("Données de score non disponibles")
    
    # Platform comparison chart
    st.subheader("🏪 Comparaison par Plateforme")
    platform_chart = ChartComponents.platform_comparison_chart(df)
    if platform_chart.data:
        st.plotly_chart(platform_chart, use_container_width=True)
    else:
        st.info("Données de plateforme non disponibles")
    
    # Stock analysis and geographic data in two columns
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📦 Analyse des Stocks")
        stock_chart = ChartComponents.stock_analysis_chart(df)
        if stock_chart.data:
            st.plotly_chart(stock_chart, use_container_width=True)
        else:
            st.info("Données de stock non disponibles")
    
    with col4:
        st.subheader("🌍 Performance par Région")
        geo_chart = ChartComponents.geographic_heatmap(df)
        if geo_chart.data:
            st.plotly_chart(geo_chart, use_container_width=True)
        else:
            st.info("Données géographiques non disponibles")
    
    # Trend analysis (full width)
    st.subheader("📅 Analyse des Tendances")
    trend_chart = ChartComponents.trend_analysis_chart(df)
    if trend_chart.data:
        st.plotly_chart(trend_chart, use_container_width=True)
    else:
        st.info("Données temporelles non disponibles")
def create_top_products_chart(df: pd.DataFrame, 
                            metric: str = 'synthetic_score', 
                            top_n: int = 10,
                            title: Optional[str] = None,
                            color_column: Optional[str] = None) -> go.Figure:
    """
    Créer un graphique des top produits
    
    Args:
        df: DataFrame contenant les données des produits
        metric: Métrique à utiliser pour le classement ('synthetic_score', 'price', etc.)
        top_n: Nombre de produits à afficher
        title: Titre personnalisé du graphique
        color_column: Colonne à utiliser pour la couleur (ex: 'platform')
        
    Returns:
        Figure Plotly
    """
    if df.empty:
        st.warning("Aucune donnée disponible pour créer le graphique")
        return go.Figure()
    
    # Vérifier que la métrique existe
    if metric not in df.columns:
        st.error(f"La colonne '{metric}' n'existe pas dans les données")
        return go.Figure()
    
    # Préparer les données
    df_sorted = df.nlargest(top_n, metric).copy()
    
    # Créer un nom de produit si pas disponible
    if 'product_name' not in df_sorted.columns:
        if 'title' in df_sorted.columns:
            df_sorted['product_name'] = df_sorted['title']
        else:
            df_sorted['product_name'] = df_sorted.index.astype(str)
    
    # Tronquer les noms trop longs
    df_sorted['product_name_short'] = df_sorted['product_name'].str[:50] + \
                                    df_sorted['product_name'].str[50:].apply(lambda x: '...' if x else '')
    
    # Définir le titre
    if title is None:
        metric_labels = {
            'synthetic_score': 'Score Synthétique',
            'price': 'Prix',
            'rating': 'Note',
            'reviews_count': 'Nombre d\'avis'
        }
        metric_label = metric_labels.get(metric, metric.replace('_', ' ').title())
        title = f"Top {top_n} Produits par {metric_label}"
    
    # Créer le graphique
    if color_column and color_column in df_sorted.columns:
        fig = px.bar(
            df_sorted,
            x='product_name_short',
            y=metric,
            color=color_column,
            title=title,
            labels={
                'product_name_short': 'Produit',
                metric: metric.replace('_', ' ').title()
            },
            hover_data=['product_name'] if 'product_name' in df_sorted.columns else None
        )
    else:
        fig = px.bar(
            df_sorted,
            x='product_name_short',
            y=metric,
            title=title,
            labels={
                'product_name_short': 'Produit',
                metric: metric.replace('_', ' ').title()
            },
            hover_data=['product_name'] if 'product_name' in df_sorted.columns else None
        )
    
    # Personnaliser l'apparence
    fig.update_layout(
        xaxis_title="Produits",
        yaxis_title=metric.replace('_', ' ').title(),
        xaxis_tickangle=-45,
        height=600,
        showlegend=bool(color_column),
        hovermode='x unified'
    )
    
    # Améliorer les hover tooltips
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>' +
                     f'{metric.replace("_", " ").title()}: %{{y}}<br>' +
                     '<extra></extra>'
    )
    
    return fig


def create_score_distribution_chart(df: pd.DataFrame, 
                                  score_column: str = 'synthetic_score',
                                  bins: int = 20,
                                  title: Optional[str] = None) -> go.Figure:
    """
    Créer un histogramme de distribution des scores
    
    Args:
        df: DataFrame contenant les données
        score_column: Nom de la colonne de score
        bins: Nombre de bins pour l'histogramme
        title: Titre personnalisé
        
    Returns:
        Figure Plotly
    """
    if df.empty or score_column not in df.columns:
        st.warning(f"Données insuffisantes pour créer la distribution de {score_column}")
        return go.Figure()
    
    if title is None:
        title = f"Distribution des {score_column.replace('_', ' ').title()}"
    
    fig = px.histogram(
        df,
        x=score_column,
        nbins=bins,
        title=title,
        labels={score_column: score_column.replace('_', ' ').title()},
        marginal="box"  # Ajouter un box plot au-dessus
    )
    
    fig.update_layout(
        xaxis_title=score_column.replace('_', ' ').title(),
        yaxis_title="Nombre de Produits",
        height=500
    )
    
    return fig


def create_ml_feature_importance_chart(feature_importance: Dict[str, float],
                                     title: str = "Importance des Caractéristiques ML",
                                     top_n: int = 15) -> go.Figure:
    """
    Créer un graphique d'importance des caractéristiques ML
    
    Args:
        feature_importance: Dictionnaire {feature_name: importance_score}
        title: Titre du graphique
        top_n: Nombre de caractéristiques à afficher
        
    Returns:
        Figure Plotly
    """
    if not feature_importance:
        st.warning("Aucune donnée d'importance des caractéristiques disponible")
        return go.Figure()
    
    # Convertir en DataFrame et trier
    df_importance = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in feature_importance.items()
    ]).nlargest(top_n, 'importance')
    
    # Créer le graphique
    fig = px.bar(
        df_importance,
        x='importance',
        y='feature',
        orientation='h',
        title=title,
        labels={
            'importance': 'Importance',
            'feature': 'Caractéristique'
        }
    )
    
    fig.update_layout(
        height=max(400, top_n * 25),  # Ajuster la hauteur selon le nombre de features
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig
def create_map_chart(df, value_column='synthetic_score', title="Map Visualization"):
    """Create a map visualization (placeholder for actual map)"""
    if 'store_region' not in df.columns:
        st.warning("Données de région non disponibles pour la carte")
        return None
    
    # Since we don't have actual coordinates, create a bar chart instead
    region_data = df.groupby('store_region')[value_column].mean().reset_index()
    
    fig = px.bar(
        region_data,
        x='store_region',
        y=value_column,
        title=title,
        color=value_column,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Région",
        yaxis_title=value_column.replace('_', ' ').title(),
        height=400
    )
    
    return fig

def create_regional_comparison_chart(df, metrics=['price', 'synthetic_score']):
    """Create regional comparison chart"""
    if 'store_region' not in df.columns:
        return None
    
    comparison_data = df.groupby('store_region')[metrics].mean().reset_index()
    
    fig = go.Figure()
    
    colors = get_color_palette()
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=comparison_data['store_region'],
            y=comparison_data[metric],
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title="Comparaison Régionale",
        xaxis_title="Région",
        barmode='group',
        height=400
    )
    
    return fig

def create_price_heatmap(df):
    """Create price heatmap by region and category"""
    if 'store_region' not in df.columns:
        return None
    
    # Use platform as category if available
    category_col = 'platform' if 'platform' in df.columns else 'vendor'
    
    if category_col not in df.columns:
        return None
    
    # Get top categories
    top_categories = df[category_col].value_counts().head(8).index
    df_filtered = df[df[category_col].isin(top_categories)]
    
    # Create pivot table
    heatmap_data = df_filtered.groupby(['store_region', category_col])['price'].mean().unstack(fill_value=0)
    
    fig = px.imshow(
        heatmap_data,
        title=f"Heatmap des Prix par Région et {category_col.title()}",
        labels=dict(x=category_col.title(), y="Région", color="Prix Moyen"),
        color_continuous_scale="RdYlBu_r"
    )
    
    fig.update_layout(height=500)
    return fig

def create_trend_chart(df, date_column='created_at', value_column='synthetic_score'):
    """Create trend chart over time"""
    if date_column not in df.columns:
        return None
    
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by month
    df['month'] = df[date_column].dt.to_period('M')
    trend_data = df.groupby('month')[value_column].mean().reset_index()
    trend_data['month'] = trend_data['month'].astype(str)
    
    fig = px.line(
        trend_data,
        x='month',
        y=value_column,
        title=f"Évolution de {value_column.replace('_', ' ').title()}",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Mois",
        yaxis_title=value_column.replace('_', ' ').title(),
        height=400
    )
    
    return fig

def create_distribution_chart(df, column='price', bins=30):
    """Create distribution histogram"""
    if column not in df.columns:
        return None
    
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        title=f"Distribution de {column.replace('_', ' ').title()}",
        marginal="box"
    )
    
    fig.update_layout(
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title="Fréquence",
        height=400
    )
    
    return fig

def create_correlation_heatmap(df, numeric_columns=None):
    """Create correlation heatmap"""
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        return None
    
    correlation_matrix = df[numeric_columns].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Matrice de Corrélation",
        labels=dict(color="Corrélation"),
        color_continuous_scale="RdBu_r"
    )
    
    fig.update_layout(height=500)
    return fig

def create_top_products_chart(df, n=10, score_column='synthetic_score'):
    """Create top products horizontal bar chart"""
    if score_column not in df.columns or 'title' not in df.columns:
        return None
    
    top_products = df.nlargest(n, score_column)
    
    # Truncate long titles
    top_products['short_title'] = top_products['title'].apply(
        lambda x: x[:30] + '...' if len(str(x)) > 30 else str(x)
    )
    
    colors = px.colors.qualitative.Set3[:len(top_products)]
    
    fig = go.Figure(go.Bar(
        x=top_products[score_column],
        y=top_products['short_title'],
        orientation='h',
        marker_color=colors,
        text=top_products[score_column].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Top {n} Produits",
        xaxis_title=score_column.replace('_', ' ').title(),
        yaxis_title="Produits",
        height=max(400, n * 30)
    )
    
    return fig

def create_vendor_performance_chart(df):
    """Create vendor performance chart"""
    if 'vendor' not in df.columns or 'synthetic_score' not in df.columns:
        return None
    
    vendor_stats = df.groupby('vendor').agg({
        'synthetic_score': 'mean',
        'price': 'count'
    }).rename(columns={'price': 'product_count'})
    
    # Filter vendors with at least 5 products
    vendor_stats = vendor_stats[vendor_stats['product_count'] >= 5]
    
    if vendor_stats.empty:
        return None
    
    vendor_stats = vendor_stats.reset_index()
    
    fig = px.scatter(
        vendor_stats,
        x='product_count',
        y='synthetic_score',
        hover_name='vendor',
        size='product_count',
        title="Performance des Vendeurs",
        labels={
            'product_count': 'Nombre de Produits',
            'synthetic_score': 'Score Moyen'
        }
    )
    
    fig.update_layout(height=500)
    return fig

def create_platform_comparison_chart(df):
    """Create platform comparison chart"""
    if 'platform' not in df.columns:
        return None
    
    platform_stats = df.groupby('platform').agg({
        'synthetic_score': 'mean',
        'price': ['mean', 'count'],
        'available': 'sum'
    }).round(2)
    
    platform_stats.columns = ['Score_Moyen', 'Prix_Moyen', 'Nb_Produits', 'Produits_Disponibles']
    platform_stats = platform_stats.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Score Moyen', 'Prix Moyen', 'Nombre de Produits', 'Produits Disponibles'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = get_color_palette()
    
    # Score moyen
    fig.add_trace(
        go.Bar(x=platform_stats['platform'], y=platform_stats['Score_Moyen'],
               name='Score', marker_color=colors[0]),
        row=1, col=1
    )
    
    # Prix moyen
    fig.add_trace(
        go.Bar(x=platform_stats['platform'], y=platform_stats['Prix_Moyen'],
               name='Prix', marker_color=colors[1]),
        row=1, col=2
    )
    
    # Nombre de produits
    fig.add_trace(
        go.Bar(x=platform_stats['platform'], y=platform_stats['Nb_Produits'],
               name='Produits', marker_color=colors[2]),
        row=2, col=1
    )
    
    # Produits disponibles
    fig.add_trace(
        go.Bar(x=platform_stats['platform'], y=platform_stats['Produits_Disponibles'],
               name='Disponibles', marker_color=colors[3]),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Comparaison des Plateformes"
    )
    
    return fig

def create_stock_analysis_chart(df):
    """Create stock analysis chart"""
    if 'stock_quantity' not in df.columns:
        return None
    
    # Define stock levels
    df['stock_level'] = pd.cut(
        df['stock_quantity'],
        bins=[-1, 0, 10, 50, 100, float('inf')],
        labels=['Rupture', 'Faible (1-10)', 'Moyen (11-50)', 'Élevé (51-100)', 'Très Élevé (100+)']
    )
    
    stock_counts = df['stock_level'].value_counts()
    
    fig = px.pie(
        values=stock_counts.values,
        names=stock_counts.index,
        title="Répartition des Niveaux de Stock",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    fig.update_layout(height=400)
    return fig