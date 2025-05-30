import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from datetime import datetime, timedelta

class KPIMetrics:
    """Classe pour gérer les indicateurs clés de performance (KPI)"""
    
    def __init__(self):
        self.metrics_config = {
            'total_products': {'icon': '📦', 'color': '#1f77b4'},
            'available_products': {'icon': '✅', 'color': '#2ca02c'},
            'avg_price': {'icon': '💰', 'color': '#ff7f0e'},
            'top_vendors': {'icon': '🏪', 'color': '#d62728'},
            'stock_alerts': {'icon': '⚠️', 'color': '#ff0000'},
            'platform_diversity': {'icon': '🌐', 'color': '#9467bd'}
        }
    
    def display_main_kpis(self, df: pd.DataFrame):
        """Affiche les KPI principaux en haut du dashboard"""
        if df.empty:
            st.warning("Aucune donnée disponible pour les KPI")
            return
        
        # Calcul des KPI
        total_products = len(df)
        available_products = int(df['available'].sum()) if 'available' in df.columns else 0
        avg_price = df['price'].mean() if 'price' in df.columns else 0
        unique_vendors = df['vendor'].nunique() if 'vendor' in df.columns else 0
        low_stock_count = len(df[df['stock_quantity'] <= 5]) if 'stock_quantity' in df.columns else 0
        platforms_count = df['platform'].nunique() if 'platform' in df.columns else 0
        
        # Affichage en colonnes
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            self._display_kpi_card(
                "Produits Total",
                total_products,
                "📦",
                "#1f77b4"
            )
        
        with col2:
            availability_rate = (available_products / total_products * 100) if total_products > 0 else 0
            self._display_kpi_card(
                "Disponibles",
                f"{available_products} ({availability_rate:.1f}%)",
                "✅",
                "#2ca02c"
            )
        
        with col3:
            self._display_kpi_card(
                "Prix Moyen",
                f"${avg_price:.2f}",
                "💰",
                "#ff7f0e"
            )
        
        with col4:
            self._display_kpi_card(
                "Vendeurs",
                unique_vendors,
                "🏪",
                "#d62728"
            )
        
        with col5:
            self._display_kpi_card(
                "Stock Faible",
                low_stock_count,
                "⚠️",
                "#ff0000" if low_stock_count > 0 else "#2ca02c"
            )
        
        with col6:
            self._display_kpi_card(
                "Plateformes",
                platforms_count,
                "🌐",
                "#9467bd"
            )
    
    def _display_kpi_card(self, title: str, value: Any, icon: str, color: str):
        """Affiche une carte KPI individuelle"""
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, {color}22 0%, {color}11 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {color};
            margin: 0.5rem 0;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <div>
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.2rem;">{title}</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: {color};">{value}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def display_performance_metrics(self, df: pd.DataFrame):
        """Affiche les métriques de performance détaillées"""
        st.subheader("📊 Métriques de Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_price_distribution_metrics(df)
            self._display_stock_health_metrics(df)
        
        with col2:
            self._display_vendor_performance_metrics(df)
            self._display_platform_metrics(df)
    
    def _display_price_distribution_metrics(self, df: pd.DataFrame):
        """Métriques de distribution des prix"""
        if 'price' not in df.columns:
            return
        
        st.markdown("**💰 Distribution des Prix**")
        
        price_stats = df['price'].describe()
        
        metrics_data = {
            'Minimum': f"${price_stats['min']:.2f}",
            'Médiane': f"${price_stats['50%']:.2f}",
            'Moyenne': f"${price_stats['mean']:.2f}",
            'Maximum': f"${price_stats['max']:.2f}",
            'Écart-type': f"${price_stats['std']:.2f}"
        }
        
        for label, value in metrics_data.items():
            st.metric(label, value)
    
    def _display_stock_health_metrics(self, df: pd.DataFrame):
        """Métriques de santé du stock"""
        if 'stock_quantity' not in df.columns:
            return
        
        st.markdown("**📦 Santé du Stock**")
        
        total_stock = df['stock_quantity'].sum()
        low_stock = len(df[df['stock_quantity'] <= 5])
        out_of_stock = len(df[df['stock_quantity'] == 0])
        high_stock = len(df[df['stock_quantity'] > 100])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stock Total", f"{total_stock:,}")
            st.metric("Stock Faible", low_stock)
        with col2:
            st.metric("Rupture Stock", out_of_stock)
            st.metric("Stock Élevé", high_stock)
    
    def _display_vendor_performance_metrics(self, df: pd.DataFrame):
        """Métriques de performance des vendeurs"""
        if 'vendor' not in df.columns:
            return
        
        st.markdown("**🏪 Performance Vendeurs**")
        
        vendor_stats = df.groupby('vendor').agg({
            'price': 'mean',
            'available': 'sum',
            'stock_quantity': 'sum'
        }).sort_values('available', ascending=False)
        
        top_vendor = vendor_stats.index[0] if not vendor_stats.empty else "N/A"
        vendor_count = len(vendor_stats)
        avg_products_per_vendor = len(df) / vendor_count if vendor_count > 0 else 0
        
        st.metric("Top Vendeur", top_vendor)
        st.metric("Nb Vendeurs", vendor_count)
        st.metric("Moy/Vendeur", f"{avg_products_per_vendor:.1f}")
    
    def _display_platform_metrics(self, df: pd.DataFrame):
        """Métriques des plateformes"""
        if 'platform' not in df.columns:
            return
        
        st.markdown("**🌐 Plateformes**")
        
        platform_stats = df['platform'].value_counts()
        dominant_platform = platform_stats.index[0] if not platform_stats.empty else "N/A"
        platform_diversity = len(platform_stats)
        
        st.metric("Plateforme Dominante", dominant_platform)
        st.metric("Diversité", platform_diversity)
        if not platform_stats.empty:
            st.metric("Part Dominante", f"{platform_stats.iloc[0]/len(df)*100:.1f}%")
    
    def display_trend_analysis(self, df: pd.DataFrame):
        """Analyse des tendances temporelles"""
        if 'created_at' not in df.columns:
            return
        
        st.subheader("📈 Analyse des Tendances")
        
        # Conversion en datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        
        # Tendances par jour
        daily_trends = df.groupby('date').agg({
            'price': 'mean',
            'available': 'sum',
            'stock_quantity': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique des prix moyens
            fig_price = px.line(
                daily_trends, 
                x='date', 
                y='price',
                title="Évolution Prix Moyen",
                labels={'price': 'Prix Moyen ($)', 'date': 'Date'}
            )
            fig_price.update_layout(height=300)
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            # Graphique de disponibilité
            fig_avail = px.bar(
                daily_trends.tail(10), 
                x='date', 
                y='available',
                title="Produits Disponibles (10 derniers jours)",
                labels={'available': 'Nb Disponibles', 'date': 'Date'}
            )
            fig_avail.update_layout(height=300)
            st.plotly_chart(fig_avail, use_container_width=True)
    
    def display_alert_system(self, df: pd.DataFrame):
        """Système d'alertes pour les KPI critiques"""
        st.subheader("🚨 Alertes Système")
        
        alerts = []
        
        # Alerte stock faible
        if 'stock_quantity' in df.columns:
            low_stock_products = df[df['stock_quantity'] <= 5]
            if len(low_stock_products) > 0:
                alerts.append({
                    'type': 'warning',
                    'title': 'Stock Faible',
                    'message': f"{len(low_stock_products)} produits ont un stock ≤ 5",
                    'details': low_stock_products[['title', 'vendor', 'stock_quantity']].head(3)
                })
        
        # Alerte prix élevés
        if 'price' in df.columns:
            price_threshold = df['price'].quantile(0.95)
            expensive_products = df[df['price'] >= price_threshold]
            if len(expensive_products) > 0:
                alerts.append({
                    'type': 'info',
                    'title': 'Prix Élevés',
                    'message': f"{len(expensive_products)} produits dans le top 5% des prix",
                    'details': expensive_products[['title', 'vendor', 'price']].head(3)
                })
        
        # Alerte indisponibilité
        if 'available' in df.columns:
            unavailable_products = df[df['available'] == False]
            if len(unavailable_products) > len(df) * 0.2:  # Plus de 20% indisponibles
                alerts.append({
                    'type': 'error',
                    'title': 'Indisponibilité Élevée',
                    'message': f"{len(unavailable_products)} produits indisponibles ({len(unavailable_products)/len(df)*100:.1f}%)",
                    'details': unavailable_products[['title', 'vendor']].head(3)
                })
        
        # Affichage des alertes
        if not alerts:
            st.success("✅ Aucune alerte système")
        else:
            for alert in alerts:
                if alert['type'] == 'error':
                    st.error(f"🚨 **{alert['title']}**: {alert['message']}")
                elif alert['type'] == 'warning':
                    st.warning(f"⚠️ **{alert['title']}**: {alert['message']}")
                else:
                    st.info(f"ℹ️ **{alert['title']}**: {alert['message']}")
                
                # Affichage des détails en expander
                if 'details' in alert and not alert['details'].empty:
                    with st.expander(f"Voir détails ({alert['title']})"):
                        st.dataframe(alert['details'], use_container_width=True)
    
    def display_comparison_metrics(self, df: pd.DataFrame, comparison_field: str = 'platform'):
        """Métriques de comparaison entre différents groupes"""
        if comparison_field not in df.columns:
            return
        
        st.subheader(f"🔄 Comparaison par {comparison_field.title()}")
        
        # Calcul des métriques par groupe
        comparison_stats = df.groupby(comparison_field).agg({
            'price': ['mean', 'median', 'std'],
            'available': ['sum', 'count'],
            'stock_quantity': 'sum' if 'stock_quantity' in df.columns else 'count'
        }).round(2)
        
        # Flatten column names
        comparison_stats.columns = ['_'.join(col).strip() for col in comparison_stats.columns]
        comparison_stats = comparison_stats.reset_index()
        
        # Calcul du taux de disponibilité
        if 'available_sum' in comparison_stats.columns and 'available_count' in comparison_stats.columns:
            comparison_stats['availability_rate'] = (
                comparison_stats['available_sum'] / comparison_stats['available_count'] * 100
            ).round(1)
        
        # Affichage du tableau de comparaison
        st.dataframe(comparison_stats, use_container_width=True)
        
        # Graphique de comparaison
        if 'price_mean' in comparison_stats.columns:
            fig = px.bar(
                comparison_stats,
                x=comparison_field,
                y='price_mean',
                title=f"Prix Moyen par {comparison_field.title()}",
                labels={'price_mean': 'Prix Moyen ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)


# Standalone functions for backward compatibility
def display_kpi_metrics(df: pd.DataFrame):
    """Fonction standalone pour afficher les KPI principaux"""
    kpi_metrics = KPIMetrics()
    kpi_metrics.display_main_kpis(df)


def display_trend_metrics(df: pd.DataFrame):
    """Fonction standalone pour afficher les métriques de tendance"""
    kpi_metrics = KPIMetrics()
    kpi_metrics.display_trend_analysis(df)


def display_performance_metrics(df: pd.DataFrame):
    """Fonction standalone pour afficher les métriques de performance"""
    kpi_metrics = KPIMetrics()
    kpi_metrics.display_performance_metrics(df)


def display_alert_system(df: pd.DataFrame):
    """Fonction standalone pour afficher le système d'alertes"""
    kpi_metrics = KPIMetrics()
    kpi_metrics.display_alert_system(df)


def display_comparison_metrics(df: pd.DataFrame, comparison_field: str = 'platform'):
    """Fonction standalone pour afficher les métriques de comparaison"""
    kpi_metrics = KPIMetrics()
    kpi_metrics.display_comparison_metrics(df, comparison_field)

def display_top_products_metrics(df: pd.DataFrame, 
                                filters: Optional[Dict[str, Any]] = None) -> None:
    """
    Afficher les métriques clés pour les top produits
    
    Args:
        df: DataFrame contenant les données des produits
        filters: Dictionnaire des filtres appliqués (optionnel)
    """
    if df.empty:
        st.warning("Aucune donnée disponible pour afficher les métriques")
        return
    
    # Calculer les métriques principales
    total_products = len(df)
    
    # Score moyen
    avg_score = df['synthetic_score'].mean() if 'synthetic_score' in df.columns else 0
    
    # Prix moyen
    avg_price = df['price'].mean() if 'price' in df.columns else 0
    
    # Produits disponibles
    available_products = df['available'].sum() if 'available' in df.columns else 0
    availability_rate = (available_products / total_products * 100) if total_products > 0 else 0
    
    # Top plateforme
    top_platform = df['platform'].mode().iloc[0] if 'platform' in df.columns and not df['platform'].empty else "N/A"
    
    # Affichage des métriques en colonnes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Produits",
            value=f"{total_products:,}",
            help="Nombre total de produits dans la sélection"
        )
    
    with col2:
        st.metric(
            label="⭐ Score Moyen",
            value=f"{avg_score:.2f}" if avg_score > 0 else "N/A",
            help="Score synthétique moyen des produits"
        )
    
    with col3:
        st.metric(
            label="💰 Prix Moyen",
            value=f"{avg_price:.2f} €" if avg_price > 0 else "N/A",
            help="Prix moyen des produits"
        )
    
    with col4:
        st.metric(
            label="✅ Disponibilité",
            value=f"{availability_rate:.1f}%",
            help="Pourcentage de produits disponibles"
        )
    
    # Métriques supplémentaires en seconde ligne
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        # Meilleur score
        best_score = df['synthetic_score'].max() if 'synthetic_score' in df.columns else 0
        st.metric(
            label="🏆 Meilleur Score",
            value=f"{best_score:.2f}" if best_score > 0 else "N/A",
            help="Score synthétique le plus élevé"
        )
    
    with col6:
        # Prix le plus bas
        min_price = df['price'].min() if 'price' in df.columns else 0
        st.metric(
            label="💸 Prix Min",
            value=f"{min_price:.2f} €" if min_price > 0 else "N/A",
            help="Prix le plus bas"
        )
    
    with col7:
        # Prix le plus élevé
        max_price = df['price'].max() if 'price' in df.columns else 0
        st.metric(
            label="💎 Prix Max",
            value=f"{max_price:.2f} €" if max_price > 0 else "N/A",
            help="Prix le plus élevé"
        )
    
    with col8:
        # Plateforme principale
        st.metric(
            label="🏪 Top Plateforme",
            value=top_platform,
            help="Plateforme avec le plus de produits"
        )
    
    # Métriques additionnelles si disponibles
    if 'vendor' in df.columns:
        st.markdown("---")
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            unique_vendors = df['vendor'].nunique()
            st.metric(
                label="👥 Vendeurs Uniques",
                value=f"{unique_vendors:,}",
                help="Nombre de vendeurs différents"
            )
        
        with col10:
            if 'reviews_count' in df.columns:
                avg_reviews = df['reviews_count'].mean()
                st.metric(
                    label="📝 Avis Moyens",
                    value=f"{avg_reviews:.0f}" if avg_reviews > 0 else "N/A",
                    help="Nombre moyen d'avis par produit"
                )
        
        with col11:
            if 'rating' in df.columns:
                avg_rating = df['rating'].mean()
                st.metric(
                    label="⭐ Note Moyenne",
                    value=f"{avg_rating:.1f}/5" if avg_rating > 0 else "N/A",
                    help="Note moyenne des produits"
                )
        
        with col12:
            if 'stock_quantity' in df.columns:
                avg_stock = df['stock_quantity'].mean()
                st.metric(
                    label="📦 Stock Moyen",
                    value=f"{avg_stock:.0f}" if avg_stock > 0 else "N/A",
                    help="Quantité moyenne en stock"
                )
    
    # Afficher les filtres appliqués si fournis
    if filters:
        with st.expander("🔍 Filtres Appliqués", expanded=False):
            for filter_name, filter_value in filters.items():
                if filter_value and filter_value != "Tous":
                    st.write(f"**{filter_name.replace('_', ' ').title()}:** {filter_value}")


def display_product_comparison_metrics(df: pd.DataFrame, 
                                     product_ids: list,
                                     id_column: str = 'id') -> None:
    """
    Afficher les métriques de comparaison entre produits sélectionnés
    
    Args:
        df: DataFrame contenant les données
        product_ids: Liste des IDs de produits à comparer
        id_column: Nom de la colonne ID
    """
    if not product_ids:
        st.info("Sélectionnez des produits pour voir la comparaison")
        return
    
    selected_products = df[df[id_column].isin(product_ids)]
    
    if selected_products.empty:
        st.warning("Aucun produit trouvé avec les IDs sélectionnés")
        return
    
    st.subheader("📊 Comparaison des Produits Sélectionnés")
    
    # Tableau de comparaison
    comparison_data = []
    for _, product in selected_products.iterrows():
        comparison_data.append({
            'Produit': product.get('product_name', product.get('title', f"Produit {product[id_column]}")),
            'Score': product.get('synthetic_score', 0),
            'Prix': f"{product.get('price', 0):.2f} €",
            'Plateforme': product.get('platform', 'N/A'),
            'Disponible': '✅' if product.get('available', False) else '❌',
            'Note': f"{product.get('rating', 0):.1f}/5" if product.get('rating', 0) > 0 else 'N/A'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)


def display_trend_metrics(df: pd.DataFrame, 
                         date_column: str = 'created_at',
                         metric_column: str = 'synthetic_score') -> None:
    """
    Afficher les métriques de tendance dans le temps
    
    Args:
        df: DataFrame contenant les données
        date_column: Nom de la colonne de date
        metric_column: Nom de la colonne métrique à analyser
    """
    if date_column not in df.columns or metric_column not in df.columns:
        return
    
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Calculer les tendances
    current_period = df_copy[df_copy[date_column] >= df_copy[date_column].max() - pd.Timedelta(days=7)]
    previous_period = df_copy[(df_copy[date_column] >= df_copy[date_column].max() - pd.Timedelta(days=14)) & 
                             (df_copy[date_column] < df_copy[date_column].max() - pd.Timedelta(days=7))]
    
    if not current_period.empty and not previous_period.empty:
        current_avg = current_period[metric_column].mean()
        previous_avg = previous_period[metric_column].mean()
        change = ((current_avg - previous_avg) / previous_avg * 100) if previous_avg != 0 else 0
        
        st.metric(
            label=f"📈 Tendance {metric_column.replace('_', ' ').title()} (7j)",
            value=f"{current_avg:.2f}",
            delta=f"{change:+.1f}%",
            help="Évolution par rapport à la semaine précédente"
        )

def display_geography_metrics(df):
    """
    Afficher les métriques principales pour l'analyse géographique
    
    Args:
        df (pd.DataFrame): DataFrame des produits avec données géographiques
    """
    if df.empty:
        st.warning("Aucune donnée disponible pour afficher les métriques")
        return
    
    # Créer 4 colonnes pour les métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Nombre total de produits
        total_products = len(df)
        st.metric(
            label="🛍️ Total Produits",
            value=f"{total_products:,}",
            delta=None
        )
    
    with col2:
        # Nombre de régions couvertes
        if 'store_region' in df.columns:
            total_regions = df['store_region'].nunique()
            st.metric(
                label="🌍 Régions Couvertes",
                value=total_regions,
                delta=None
            )
        else:
            st.metric(label="🌍 Régions", value="N/A")
    
    with col3:
        # Prix moyen global
        if 'price' in df.columns:
            avg_price = df['price'].mean()
            st.metric(
                label="💰 Prix Moyen",
                value=f"{avg_price:.2f}€",
                delta=None
            )
        else:
            st.metric(label="💰 Prix Moyen", value="N/A")
    
    with col4:
        # Taux de disponibilité global
        if 'available' in df.columns:
            availability_rate = df['available'].mean() * 100
            st.metric(
                label="📦 Disponibilité",
                value=f"{availability_rate:.1f}%",
                delta=None
            )
        else:
            st.metric(label="📦 Disponibilité", value="N/A")
    
    # Deuxième ligne de métriques
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        # Score moyen de performance
        if 'synthetic_score' in df.columns:
            avg_score = df['synthetic_score'].mean()
            st.metric(
                label="⭐ Score Moyen",
                value=f"{avg_score:.3f}",
                delta=None
            )
        else:
            st.metric(label="⭐ Score Moyen", value="N/A")
    
    with col6:
        # Nombre de vendeurs uniques
        if 'vendor' in df.columns:
            unique_vendors = df['vendor'].nunique()
            st.metric(
                label="🏪 Vendeurs",
                value=unique_vendors,
                delta=None
            )
        else:
            st.metric(label="🏪 Vendeurs", value="N/A")
    
    with col7:
        # Stock moyen
        if 'stock_quantity' in df.columns:
            avg_stock = df['stock_quantity'].mean()
            st.metric(
                label="📊 Stock Moyen",
                value=f"{avg_stock:.0f}",
                delta=None
            )
        else:
            st.metric(label="📊 Stock Moyen", value="N/A")
    
    with col8:
        # Région la plus représentée
        if 'store_region' in df.columns:
            top_region = df['store_region'].value_counts().index[0]
            top_region_count = df['store_region'].value_counts().iloc[0]
            st.metric(
                label="🏆 Top Région",
                value=top_region[:10] + "..." if len(top_region) > 10 else top_region,
                delta=f"{top_region_count} produits"
            )
        else:
            st.metric(label="🏆 Top Région", value="N/A")

