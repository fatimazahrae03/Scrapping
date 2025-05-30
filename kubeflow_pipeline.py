"""
Kubeflow Pipeline pour l'orchestration ML du système E-commerce - VERSION ULTRA-CORRIGÉE
Compatible avec Kubeflow Pipelines 1.8.x (API v1) - Format Argo fixé
Correction du problème "unknown template format"
"""

import kfp
from kfp import dsl
from kfp.components import func_to_container_op, create_component_from_func
from typing import NamedTuple
import json
import os


# =============================================================================
# COMPOSANTS KUBEFLOW V1.8.x (API v1) - VERSION ULTRA-CORRIGÉE
# =============================================================================

def data_extraction_component(
    stores_config: str,
    scraping_config: str
) -> NamedTuple('ExtractionOutput', [('total_products', int), ('stores_processed', int), ('output_path', str)]):
    """
    Composant d'extraction de données A2A
    Utilise votre système UnifiedExtractionPipeline existant
    """
    import json
    import pandas as pd
    from collections import namedtuple
    import os
    
    # Simulation de votre pipeline d'extraction existant
    print("🚀 Démarrage de l'extraction A2A...")
    
    # Parsing des configurations
    stores = json.loads(stores_config)
    scraping_params = json.loads(scraping_config)
    
    # Simulation de l'extraction (remplacer par votre code réel)
    extracted_data = []
    total_products = 0
    
    for store in stores:
        print(f"📦 Extraction depuis {store['domain']} ({store['platform']})...")
        
        # Ici vous appelleriez votre agent A2A approprié
        if store['platform'] == 'shopify':
            products_count = 150  # Simulation
        else:
            products_count = 80   # Simulation
            
        total_products += products_count
        
        # Données simulées pour le pipeline
        for i in range(products_count):
            extracted_data.append({
                'store_domain': store['domain'],
                'platform': store['platform'],
                'title': f"Product {i} from {store['domain']}",
                'price': 29.99 + (i % 100),
                'available': i % 4 != 0,
                'stock_quantity': i % 50,
                'vendor': f"Vendor_{i % 10}"
            })
    
    # Sauvegarde des données extraites
    
    df = pd.DataFrame(extracted_data)
    output_path = '/tmp/extracted_products.csv'
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Extraction terminée: {total_products} produits de {len(stores)} magasins")
    
    ExtractionOutput = namedtuple('ExtractionOutput', ['total_products', 'stores_processed', 'output_path'])
    return ExtractionOutput(total_products, len(stores), output_path)

def data_storage_component(
    extraction_output_path: str,
    mongodb_config: str
) -> NamedTuple('StorageOutput', [('stored_products', int), ('quality_ratio', float)]):
    """
    Composant de stockage MongoDB
    Utilise votre système ProductsDB existant
    """
    import pandas as pd
    import json
    from collections import namedtuple
    
    print("💾 Démarrage du stockage MongoDB...")
    
    # Chargement des données extraites
    df = pd.read_csv(extraction_output_path)
    config = json.loads(mongodb_config)
    
    print(f"📊 Traitement de {len(df)} produits...")
    
    # Simulation du stockage (remplacer par votre ProductsDB)
    # db = ProductsDB(config['connection_string'])
    # db.insert_products_batch(df.to_dict('records'))
    
    # Nettoyage et validation des données
    df_clean = df.dropna(subset=['title', 'price'])
    df_clean = df_clean[df_clean['price'] > 0]
    
    stored_products = len(df_clean)
    quality_ratio = stored_products / len(df) if len(df) > 0 else 0
    
    print(f"✅ Stockage terminé: {stored_products} produits stockés")
    
    StorageOutput = namedtuple('StorageOutput', ['stored_products', 'quality_ratio'])
    return StorageOutput(stored_products, quality_ratio)

def ml_scoring_component(
    stored_products: int,
    ml_config: str
) -> NamedTuple('MLOutput', [('model_accuracy', float), ('top_k_count', int), ('scored_data_path', str)]):
    """
    Composant de scoring ML
    Utilise votre ProductAnalyzer existant
    """
    import json
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pickle
    from collections import namedtuple
    
    print("🧠 Démarrage du scoring ML...")
    
    # Configuration ML
    config = json.loads(ml_config)
    
    # Simulation de données depuis MongoDB (remplacer par votre ProductAnalyzer)
    np.random.seed(42)
    n_products = min(int(stored_products), 1000)  # Conversion explicite en int
    
    # Génération de données simulées
    data = {
        'price': np.random.uniform(10, 500, n_products),
        'stock_quantity': np.random.randint(0, 100, n_products),
        'available': np.random.choice([True, False], n_products, p=[0.8, 0.2]),
        'vendor_popularity': np.random.uniform(0, 1, n_products),
        'platform_score': np.random.uniform(0.5, 1, n_products)
    }
    
    df = pd.DataFrame(data)
    
    # Feature engineering
    df['price_score'] = 1 / (1 + df['price'] / 100)
    df['availability_score'] = df['available'].astype(float)
    df['stock_score'] = np.minimum(df['stock_quantity'] / 50, 1)
    
    # Calcul du score synthétique (cible)
    weights = config.get('weights', {
        'price': 0.3, 'availability': 0.25, 'stock': 0.2, 
        'vendor_popularity': 0.15, 'platform': 0.1
    })
    
    df['synthetic_score'] = (
        df['price_score'] * weights['price'] +
        df['availability_score'] * weights['availability'] +
        df['stock_score'] * weights['stock'] +
        df['vendor_popularity'] * weights['vendor_popularity'] +
        df['platform_score'] * weights['platform']
    )
    
    # Préparation des features pour ML
    X = df[['price', 'stock_quantity', 'vendor_popularity', 'platform_score']].copy()
    X['available'] = df['available'].astype(int)
    y = df['synthetic_score']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    model_type = config.get('model_type', 'random_forest')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    print(f"🎯 Entraînement du modèle {model_type}...")
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = max(0, 1 - mse)
    
    # Prédiction sur l'ensemble complet
    df['ml_score'] = model.predict(X)
    df['final_score'] = (df['synthetic_score'] + df['ml_score']) / 2
    
    # Sélection des Top-K
    k = config.get('top_k', 50)
    top_products = df.nlargest(k, 'final_score')
    
    # Sauvegarde des résultats
    scored_data_path = '/tmp/scored_products.csv'
    top_products.to_csv(scored_data_path, index=False)
    
    # Sauvegarde du modèle
    with open('/tmp/ml_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Scoring ML terminé: Top-{k} produits sélectionnés")
    print(f"📈 Précision du modèle: {accuracy:.3f}")
    
    MLOutput = namedtuple('MLOutput', ['model_accuracy', 'top_k_count', 'scored_data_path'])
    return MLOutput(float(accuracy), k, scored_data_path)

def validation_component(
    scored_data_path: str,
    validation_config: str
) -> NamedTuple('ValidationOutput', [('quality_score', float), ('recommendations_count', int)]):
    """
    Composant de validation et contrôle qualité
    """
    import pandas as pd
    import json
    from collections import namedtuple
    
    print("🔍 Démarrage de la validation...")
    
    # Chargement des données scorées
    df = pd.read_csv(scored_data_path)
    config = json.loads(validation_config)
    
    # Métriques de qualité
    quality_checks = {
        'data_completeness': (df.isnull().sum().sum() == 0),
        'price_validity': (df['price'] > 0).all(),
        'score_distribution': df['final_score'].std() > 0.1,
        'top_products_available': (df.head(10)['available'].astype(bool)).mean() > 0.7
    }
    
    quality_score = sum(quality_checks.values()) / len(quality_checks)
    
    # Génération du rapport
    report = {
        'quality_score': quality_score,
        'total_products_analyzed': len(df),
        'top_products_count': config.get('top_k', 50),
        'average_score': float(df['final_score'].mean()),
        'quality_checks': quality_checks,
        'recommendations': []
    }
    
    # Recommandations basées sur l'analyse
    if quality_score < 0.8:
        report['recommendations'].append("Améliorer la qualité des données d'entrée")
    if df['final_score'].std() < 0.1:
        report['recommendations'].append("Diversifier les critères de scoring")
    
    # Sauvegarde du rapport
    with open('/tmp/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Validation terminée - Score qualité: {quality_score:.2f}")
    
    ValidationOutput = namedtuple('ValidationOutput', ['quality_score', 'recommendations_count'])
    return ValidationOutput(quality_score, len(report['recommendations']))

# =============================================================================
# CRÉATION DES COMPOSANTS AVEC ANCIENNE API - VERSION ULTRA-CORRIGÉE
# =============================================================================

# Utilisation de l'ancienne méthode func_to_container_op pour assurer la compatibilité
extraction_op = func_to_container_op(
    data_extraction_component,
    base_image='python:3.8-slim',
    packages_to_install=['pandas==1.5.3', 'numpy==1.24.3']
)

storage_op = func_to_container_op(
    data_storage_component,
    base_image='python:3.8-slim',
    packages_to_install=['pandas==1.5.3']
)

ml_scoring_op = func_to_container_op(
    ml_scoring_component,
    base_image='python:3.8-slim',
    packages_to_install=['pandas==1.5.3', 'numpy==1.24.3', 'scikit-learn==1.2.2']
)

validation_op = func_to_container_op(
    validation_component,
    base_image='python:3.8-slim',
    packages_to_install=['pandas==1.5.3']
)

# =============================================================================
# PIPELINE PRINCIPAL - VERSION ULTRA-CORRIGÉE POUR KUBEFLOW 1.8.x
# =============================================================================

@dsl.pipeline(
    name="ecommerce-ml-pipeline-ultra-fixed",
    description="Pipeline ML complet pour l'analyse e-commerce - Version ultra-corrigée pour Kubeflow 1.8.x"
)
def ecommerce_ml_pipeline_ultra_fixed(
    stores_config: str = '[]',
    scraping_config: str = '{}',
    mongodb_config: str = '{}',
    ml_config: str = '{}',
    validation_config: str = '{}'
):
    """
    Pipeline principal orchestrant toutes les étapes du processus ML e-commerce
    Version ultra-corrigée pour Kubeflow 1.8.x avec ancienne API
    """
    
    # Étape 1: Extraction des données A2A
    extraction_task = extraction_op(
        stores_config=stores_config,
        scraping_config=scraping_config
    )
    
    # Étape 2: Stockage MongoDB
    storage_task = storage_op(
        extraction_output_path=extraction_task.outputs['output_path'],
        mongodb_config=mongodb_config
    )
    
    # Étape 3: Scoring ML et sélection Top-K
    ml_task = ml_scoring_op(
        stored_products=storage_task.outputs['stored_products'],
        ml_config=ml_config
    )
    
    # Étape 4: Validation et contrôle qualité
    validation_task = validation_op(
        scored_data_path=ml_task.outputs['scored_data_path'],
        validation_config=validation_config
    )

# =============================================================================
# UTILITAIRES DE CONFIGURATION
# =============================================================================

def create_default_configs():
    """
    Crée les configurations par défaut pour le pipeline
    """
    
    stores_config = [
        {
            "domain": "allbirds.com",
            "name": "Allbirds",
            "platform": "shopify",
            "region": "US",
            "currency": "USD",
            "priority": 1
        },
        {
            "domain": "gymshark.com", 
            "name": "Gymshark",
            "platform": "shopify",
            "region": "UK",
            "currency": "GBP",
            "priority": 2
        }
    ]
    
    scraping_config = {
        "max_retries": 3,
        "timeout_seconds": 30,
        "delay_between_requests": 1.5,
        "selenium_options": {
            "headless": True,
            "window_size": "1920,1080"
        },
        "max_products_per_store": 1000
    }
    
    mongodb_config = {
        "connection_string": "mongodb://localhost:27017",
        "database": "ecommerce_products",
        "collection": "products"
    }
    
    ml_config = {
        "model_type": "random_forest",
        "top_k": 50,
        "weights": {
            "price": 0.3,
            "availability": 0.25,
            "stock": 0.2,
            "vendor_popularity": 0.15,
            "platform": 0.1
        }
    }
    
    validation_config = {
        "top_k": 50,
        "quality_threshold": 0.8,
        "enable_recommendations": True
    }
    
    return {
        'stores_config': json.dumps(stores_config),
        'scraping_config': json.dumps(scraping_config),
        'mongodb_config': json.dumps(mongodb_config),
        'ml_config': json.dumps(ml_config),
        'validation_config': json.dumps(validation_config)
    }

# =============================================================================
# COMPILATION ULTRA-CORRIGÉE POUR KUBEFLOW 1.8.x
# =============================================================================

def compile_pipeline_ultra_fixed():
    """
    Compilation ultra-corrigée pour Kubeflow 1.8.x
    Utilise l'ancienne API de compilation pour assurer la compatibilité
    """
    pipeline_package_path = 'ecommerce_ml_pipeline_ultra_fixed.yaml'
    
    try:
        print(f"📋 Version KFP installée: {kfp.__version__}")
        
        # Utilisation de l'ancienne méthode de compilation pour Kubeflow 1.8.x
        # IMPORTANT: On évite le nouveau Compiler() qui peut causer des problèmes
        import kfp.compiler as compiler
        
        # Compilation avec l'ancienne API
        compiler.Compiler().compile(
            pipeline_func=ecommerce_ml_pipeline_ultra_fixed,
            package_path=pipeline_package_path
        )
        
        print(f"✅ Pipeline compilé avec succès: {pipeline_package_path}")
        
        # Vérification approfondie du contenu
        with open(pipeline_package_path, 'r') as f:
            content = f.read()
            
        print("🔍 Analyse du fichier généré:")
        
        if "apiVersion: argoproj.io/v1alpha1" in content:
            print("✅ Format Argo Workflows v1alpha1 - Compatible Kubeflow 1.8.x")
        elif "kind: Workflow" in content:
            print("✅ Format Workflow détecté - Compatible")
        else:
            print("❌ Format non reconnu - Possible incompatibilité")
            
        if "pipelineSpec" in content:
            print("⚠️  Nouveau format v2 détecté - Risque d'incompatibilité")
        
        # Vérification de la structure Argo
        if "templates:" in content and "dag:" in content:
            print("✅ Structure DAG Argo correcte")
        else:
            print("⚠️  Structure DAG non détectée")
            
        return pipeline_package_path
        
    except Exception as e:
        print(f"❌ Erreur de compilation: {e}")
        import traceback
        traceback.print_exc()
        return None

def compile_and_run_ultra_fixed():
    """
    Compile et exécute le pipeline avec version ultra-corrigée
    """
    
    # Vérification de compatibilité
    print("🔧 Vérification de la compatibilité...")
    try:
        client = kfp.Client(host='http://localhost:8080')
        print("✅ Connexion Kubeflow OK")
    except Exception as e:
        print(f"❌ Problème de connexion: {e}")
        return None
    
    # Compilation du pipeline
    pipeline_package_path = compile_pipeline_ultra_fixed()
    if not pipeline_package_path:
        return None
    
    # Configuration par défaut
    configs = create_default_configs()
    
    try:
        # Création d'une expérience avec nom unique
        import time
        experiment_name = f"ecommerce-ml-ultra-fixed-{int(time.time())}"
        
        experiment = client.create_experiment(name=experiment_name)
        print(f"📁 Nouvelle expérience créée: {experiment_name}")
        
        # Soumission du pipeline
        print("🚀 Lancement du pipeline ultra-fixé...")
        
        run = client.run_pipeline(
            experiment_id=experiment.id,
            job_name=f"ecommerce-ml-run-ultra-fixed-{int(time.time())}",
            pipeline_package_path=pipeline_package_path,
            params=configs
        )
        
        print(f"✅ Pipeline lancé avec succès!")
        print(f"🆔 Run ID: {run.id}")
        print(f"🔗 URL de suivi: http://localhost:8080/#/runs/details/{run.id}")
        
        return run
        
    except Exception as run_error:
        print(f"❌ Erreur lors du lancement: {run_error}")
        
        # Diagnostic avancé
        print("\n🔍 DIAGNOSTIC AVANCÉ:")
        
        # Vérification du contenu YAML
        try:
            with open(pipeline_package_path, 'r') as f:
                yaml_content = f.read()
                
            print(f"📄 Taille du fichier YAML: {len(yaml_content)} caractères")
            
            # Recherche de patterns problématiques
            if "pipelineSpec" in yaml_content:
                print("⚠️  Format v2 détecté dans le YAML - Incompatible avec Kubeflow 1.8.x")
            if "apiVersion: argoproj.io/v1alpha1" not in yaml_content:
                print("❌ En-tête Argo manquant")
            if "kind: Workflow" not in yaml_content:
                print("❌ Type Workflow manquant")
                
        except Exception as yaml_error:
            print(f"❌ Impossible de lire le YAML: {yaml_error}")
        
        # Solutions détaillées
        print("\n💡 SOLUTIONS DÉTAILLÉES:")
        print("1. Vérifier la version Kubeflow server:")
        print("   kubectl get pods -n kubeflow | grep ml-pipeline")
        print("2. Vérifier les logs du serveur ML Pipeline:")
        print("   kubectl logs -n kubeflow deployment/ml-pipeline")
        print("3. Essayer avec une version KFP encore plus ancienne:")
        print("   pip install kfp==1.8.12")
        
        return None

def test_pipeline_locally():
    """
    Test local du pipeline sans Kubeflow
    """
    print("🧪 TEST LOCAL DU PIPELINE ULTRA-FIXÉ")
    print("=" * 40)
    
    configs = create_default_configs()
    
    try:
        print("1️⃣ Test de l'extraction...")
        extraction_result = data_extraction_component(
            configs['stores_config'], 
            configs['scraping_config']
        )
        print(f"   ✅ {extraction_result.total_products} produits extraits")
        
        print("2️⃣ Test du stockage...")
        storage_result = data_storage_component(
            extraction_result.output_path,
            configs['mongodb_config']
        )
        print(f"   ✅ {storage_result.stored_products} produits stockés")
        
        print("3️⃣ Test du ML scoring...")
        ml_result = ml_scoring_component(
            storage_result.stored_products,
            configs['ml_config']
        )
        print(f"   ✅ Top-{ml_result.top_k_count} produits scorés")
        
        print("4️⃣ Test de la validation...")
        validation_result = validation_component(
            ml_result.scored_data_path,
            configs['validation_config']
        )
        print(f"   ✅ Score qualité: {validation_result.quality_score:.2f}")
        
        print("\n🎉 Tous les composants fonctionnent correctement!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

def emergency_downgrade_kfp():
    """
    Instructions pour downgrade d'urgence de KFP
    """
    instructions = """
    🚨 DOWNGRADE D'URGENCE DE KFP
    
    Le problème "unknown template format" indique souvent une incompatibilité
    entre la version KFP client et le serveur Kubeflow.
    
    💊 SOLUTION RADICALE:
    
    1. Désinstaller complètement KFP:
       pip uninstall kfp kfp-server-api kfp-pipeline-spec
    
    2. Installer la version la plus compatible:
       pip install kfp==1.8.12
    
    3. Vérifier l'installation:
       python -c "import kfp; print(kfp.__version__)"
    
    4. Redémarrer complètement Python et relancer
    
    📋 SI LE PROBLÈME PERSISTE:
    
    Option A - Downgrade encore plus:
       pip install kfp==1.7.2
    
    Option B - Vérifier la version du serveur Kubeflow:
       kubectl get pods -n kubeflow -l app=ml-pipeline -o yaml | grep image
    
    Option C - Compilation manuelle:
       python kubeflow_pipeline_ultra_fixed.py --compile-only
       Puis copier/coller le YAML dans l'interface web
    """
    print(instructions)

if __name__ == "__main__":
    print("🏗️  KUBEFLOW PIPELINE E-COMMERCE ML - VERSION ULTRA-CORRIGÉE")
    print("=" * 70)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test local sans Kubeflow
        success = test_pipeline_locally()
        if success:
            print("\n📦 Compilation du pipeline...")
            compile_pipeline_ultra_fixed()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--compile-only':
        # Compilation uniquement
        result = compile_pipeline_ultra_fixed()
        if result:
            print(f"\n📁 Fichier généré: {result}")
            print("💡 Vous pouvez maintenant l'uploader via l'interface web Kubeflow")
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--emergency':
        # Instructions de downgrade d'urgence
        emergency_downgrade_kfp()
        
    else:
        # Exécution normale avec version ultra-corrigée
        print("🔄 Compilation et exécution du pipeline ultra-fixé...")
        run = compile_and_run_ultra_fixed()
        
        if run:
            print(f"\n🎉 Pipeline déployé avec succès!")
            print(f"🆔 Run ID: {run.id}")
        else:
            print("\n⚠️  Échec du déploiement")
            print("\n💡 COMMANDES DE DIAGNOSTIC:")
            print("   python kubeflow_pipeline_ultra_fixed.py --test")
            print("   python kubeflow_pipeline_ultra_fixed.py --compile-only") 
            print("   python kubeflow_pipeline_ultra_fixed.py --emergency")
    
    print("\n💡 Commandes utiles:")
    print("   kubectl get pods -n kubeflow")
    print("   kubectl logs -n kubeflow deployment/ml-pipeline")
    print("   kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80")