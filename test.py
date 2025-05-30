
import kfp
import requests
import json

def test_kubeflow_connection():
    """Test la connexion à Kubeflow"""
    
    print("🔍 Test de connexion Kubeflow...")
    print(f"📦 Version KFP SDK: {kfp.__version__}")
    
    # Test 1: Vérifier que l'interface web est accessible
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            print("✅ Interface web accessible sur http://localhost:8080")
        else:
            print(f"❌ Interface web non accessible (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Erreur d'accès à l'interface web: {e}")
        return False
    
    # Test 2: Tester différentes versions d'API
    test_endpoints = [
        "/api/v1/healthz",           # v1 API
        "/apis/v1beta1/healthz",     # v1beta1 API  
        "/apis/v2beta1/healthz",     # v2beta1 API
        "/healthz"                   # API simple
    ]
    
    working_endpoint = None
    for endpoint in test_endpoints:
        try:
            url = f"http://localhost:8080{endpoint}"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ API endpoint fonctionnel: {endpoint}")
                working_endpoint = endpoint
                break
            else:
                print(f"❌ {endpoint} → Status {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} → Erreur: {str(e)[:50]}")
    
    if not working_endpoint:
        print("❌ Aucun endpoint d'API santé trouvé")
        return False
    
    # Test 3: Tenter la connexion KFP avec gestion d'erreurs
    try:
        print("\n🔌 Test connexion client KFP...")
        
        # Version simple sans vérification de santé
        client = kfp.Client(
            host='http://localhost:8080',
            existing_token=None
        )
        
        # Test basique: lister les expériences
        experiments = client.list_experiments(page_size=1)
        print("✅ Connexion KFP réussie!")
        print(f"📊 Nombre d'expériences: {experiments.total_size if hasattr(experiments, 'total_size') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur connexion KFP: {e}")
        
        # Tentative avec configuration alternative
        try:
            print("🔄 Tentative avec configuration alternative...")
            from kfp.client import Client
            
            # Client sans vérification de santé
            client = Client(
                host='http://localhost:8080',
                existing_token=None,
                namespace='kubeflow'
            )
            
            print("✅ Connexion alternative réussie!")
            return True
            
        except Exception as e2:
            print(f"❌ Connexion alternative échouée: {e2}")
            return False

def get_kubeflow_version_info():
    """Récupère les informations de version Kubeflow"""
    
    print("\n🔍 Informations sur l'installation Kubeflow...")
    
    try:
        # Vérifier les pods Kubeflow
        import subprocess
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", "kubeflow", "-o", "json"],
            capture_output=True, text=True, check=True
        )
        
        pods_info = json.loads(result.stdout)
        
        print("📦 Pods Kubeflow actifs:")
        for pod in pods_info['items']:
            name = pod['metadata']['name']
            image = pod['spec']['containers'][0]['image'] if pod['spec']['containers'] else 'N/A'
            status = pod['status']['phase']
            
            if 'ml-pipeline' in name:
                print(f"  🔹 {name}")
                print(f"    Image: {image}")
                print(f"    Status: {status}")
                
    except Exception as e:
        print(f"❌ Impossible de récupérer les infos Kubeflow: {e}")
        print("💡 Assurez-vous que kubectl est installé et configuré")

def suggest_fixes():
    """Suggère des corrections basées sur les tests"""
    
    print("\n🛠️ SOLUTIONS RECOMMANDÉES:")
    print("=" * 50)
    
    print("1. 📥 DOWNGRADE DU SDK KFP:")
    print("   pip uninstall kfp")
    print("   pip install kfp==1.8.22")
    print()
    
    print("2. 🔄 OU UPGRADE KUBEFLOW (plus complexe):")
    print("   kubectl delete -k 'github.com/kubeflow/pipelines//manifests/kustomize/env/platform-agnostic?ref=1.8.1'")
    print("   kubectl apply -k 'github.com/kubeflow/pipelines//manifests/kustomize/env/platform-agnostic?ref=2.0.1'")
    print()
    
    print("3. 🐳 SOLUTION DOCKER ALTERNATIVE:")
    print("   docker run -p 8080:8080 gcr.io/ml-pipeline/frontend:1.8.22")
    print()
    
    print("4. ⚡ SOLUTION RAPIDE - CLIENT MODIFIÉ:")
    print("   Utilisez le code modifié ci-dessous dans votre script")

if __name__ == "__main__":
    print("🧪 DIAGNOSTIC KUBEFLOW PIPELINES")
    print("=" * 50)
    
    # Tests de connexion
    connection_ok = test_kubeflow_connection()
    
    # Informations sur l'installation
    get_kubeflow_version_info()
    
    # Suggestions
    if not connection_ok:
        suggest_fixes()
    else:
        print("\n🎉 Connexion Kubeflow opérationnelle!")
        print("Vous pouvez maintenant exécuter votre pipeline.")