from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, BulkWriteError
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
import os
from bson import ObjectId

class ProductsDB:
    """MongoDB storage for extracted products"""
    
    def __init__(self, connection_string='mongodb://localhost:27017/', 
                 database_name='products_db', collection_name='products'):
        """Initialize MongoDB connection"""
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Connect to MongoDB database"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            self.logger.info(f"✅ Connected to MongoDB: {self.database_name}.{self.collection_name}")
            return True
        except ConnectionFailure as e:
            self.logger.error(f"❌ MongoDB connection failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error connecting to MongoDB: {e}")
            return False
    
    def create_indexes(self):
        """Create indexes for better performance"""
        try:
            indexes = [
                ('store_domain', 1),
                ('platform', 1),
                ('available', 1),
                ('price', 1),
                ('product_id', 1),
                ('vendor', 1)
            ]
            
            for field, direction in indexes:
                self.collection.create_index([(field, direction)])
            
            # Compound indexes for common queries
            self.collection.create_index([('store_domain', 1), ('available', 1)])
            self.collection.create_index([('platform', 1), ('price', 1)])
            
            self.logger.info("✅ Indexes created/verified")
            return True
        except Exception as e:
            self.logger.error(f"❌ Index creation failed: {e}")
            return False
    
    def clean_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate product data for MongoDB"""
        cleaned = {}
        
        # Helper function to safely get and clean values
        def safe_get(key, default=None):
            value = product_data.get(key, default)
            
            # Handle NaN values
            if pd.isna(value):
                return default
            
            # Convert to appropriate type
            if value is not None and value != '':
                return value
            
            return default
        
        def safe_get_float(key, default=None):
            value = product_data.get(key, default)
            if pd.isna(value) or value == '' or value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_get_int(key, default=None):
            value = product_data.get(key, default)
            if pd.isna(value) or value == '' or value is None:
                return default
            try:
                return int(float(value))  # Convert through float first
            except (ValueError, TypeError):
                return default
        
        def safe_get_bool(key, default=False):
            value = product_data.get(key, default)
            if pd.isna(value) or value == '' or value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return bool(value)
        
        # Clean all fields - MongoDB is flexible with data types
        cleaned['store_domain'] = safe_get('store_domain')
        cleaned['store_region'] = safe_get('store_region')
        cleaned['platform'] = safe_get('platform')
        cleaned['product_id'] = safe_get('product_id')
        cleaned['title'] = safe_get('title')
        cleaned['handle'] = safe_get('handle')  
        cleaned['vendor'] = safe_get('vendor')
        cleaned['price'] = safe_get_float('price')
        cleaned['compare_at_price'] = safe_get_float('compare_at_price')
        cleaned['available'] = safe_get_bool('available')
        cleaned['stock_quantity'] = safe_get_int('stock_quantity')
        cleaned['sku'] = safe_get('sku')
        cleaned['image_src'] = safe_get('image_src')
        cleaned['body_html'] = safe_get('body_html')
        
        # Handle tags - can be string or list
        tags_value = product_data.get('tags', '')
        if pd.isna(tags_value):
            cleaned['tags'] = []
        elif isinstance(tags_value, list):
            cleaned['tags'] = [str(tag).strip() for tag in tags_value if str(tag).strip()]
        elif isinstance(tags_value, str) and tags_value.strip():
            cleaned['tags'] = [tag.strip() for tag in tags_value.split(',') if tag.strip()]
        else:
            cleaned['tags'] = []
        
        # Add timestamps
        cleaned['created_at'] = datetime.utcnow()
        cleaned['updated_at'] = datetime.utcnow()
        
        # Remove None values to keep documents clean
        cleaned = {k: v for k, v in cleaned.items() if v is not None}
        
        return cleaned
    
    def insert_product(self, product_data: Dict[str, Any]) -> Optional[str]:
        """Insert single product into database"""
        try:
            # Clean data first
            cleaned_data = self.clean_product_data(product_data)
            
            # Insert document
            result = self.collection.insert_one(cleaned_data)
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"❌ Product insertion failed: {e}")
            self.logger.error(f"❌ Product data: {product_data}")
            return None
    
    def insert_products_batch(self, products: List[Dict[str, Any]], batch_size=1000):
        """Insert multiple products at once with better error handling"""
        if not products:
            return 0
            
        total_inserted = 0
        total_products = len(products)
        errors = []
        
        # Process in batches
        for i in range(0, total_products, batch_size):
            batch = products[i:i + batch_size]
            
            try:
                # Prepare all documents for this batch
                documents = []
                for j, product_data in enumerate(batch):
                    try:
                        # Clean data first
                        cleaned_data = self.clean_product_data(product_data)
                        documents.append(cleaned_data)
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error processing product {i+j+1}: {e}")
                        errors.append(f"Product {i+j+1}: {str(e)}")
                        continue
                
                # Insert this batch
                if documents:
                    try:
                        result = self.collection.insert_many(documents, ordered=False)
                        batch_inserted = len(result.inserted_ids)
                        total_inserted += batch_inserted
                        
                        print(f"✅ Batch {i//batch_size + 1}: {batch_inserted} products inserted (Total: {total_inserted}/{total_products})")
                        
                    except BulkWriteError as bwe:
                        # Handle partial success in bulk operations
                        batch_inserted = bwe.details.get('nInserted', 0)
                        total_inserted += batch_inserted
                        
                        print(f"⚠️ Batch {i//batch_size + 1}: {batch_inserted} products inserted with some errors")
                        for error in bwe.details.get('writeErrors', []):
                            errors.append(f"Bulk error: {error.get('errmsg', 'Unknown error')}")
                
            except Exception as e:
                self.logger.error(f"❌ Batch insertion failed for batch {i//batch_size + 1}: {e}")
                errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
                continue
        
        if errors:
            self.logger.warning(f"⚠️ {len(errors)} errors occurred during insertion")
            
        self.logger.info(f"✅ {total_inserted} products inserted successfully out of {total_products}")
        return total_inserted
    
    def get_products_count(self):
        """Get total number of products in database"""
        try:
            return self.collection.count_documents({})
        except Exception as e:
            self.logger.error(f"❌ Count query failed: {e}")
            return 0
    
    def get_products_by_store(self, store_domain: str):
        """Get products from specific store"""
        try:
            cursor = self.collection.find({'store_domain': store_domain})
            products = list(cursor)
            # Convert ObjectId to string for JSON serialization
            for product in products:
                product['_id'] = str(product['_id'])
            return products
        except Exception as e:
            self.logger.error(f"❌ Store query failed: {e}")
            return []
    
    def get_available_products(self):
        """Get only available products"""
        try:
            cursor = self.collection.find({'available': True})
            products = list(cursor)
            # Convert ObjectId to string
            for product in products:
                product['_id'] = str(product['_id'])
            return products
        except Exception as e:
            self.logger.error(f"❌ Available products query failed: {e}")
            return []
    
    def get_products_by_price_range(self, min_price: float, max_price: float):
        """Get products within price range"""
        try:
            cursor = self.collection.find({
                'price': {'$gte': min_price, '$lte': max_price}
            })
            products = list(cursor)
            for product in products:
                product['_id'] = str(product['_id'])
            return products
        except Exception as e:
            self.logger.error(f"❌ Price range query failed: {e}")
            return []
    
    def search_products(self, search_term: str, limit=100):
        """Search products by title or description"""
        try:
            # Create text search (requires text index)
            try:
                self.collection.create_index([('title', 'text'), ('body_html', 'text')])
            except:
                pass  # Index might already exist
            
            cursor = self.collection.find(
                {'$text': {'$search': search_term}},
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit)
            
            products = list(cursor)
            for product in products:
                product['_id'] = str(product['_id'])
            return products
        except Exception as e:
            # Fallback to regex search
            cursor = self.collection.find({
                '$or': [
                    {'title': {'$regex': search_term, '$options': 'i'}},
                    {'body_html': {'$regex': search_term, '$options': 'i'}}
                ]
            }).limit(limit)
            
            products = list(cursor)
            for product in products:
                product['_id'] = str(product['_id'])
            return products
    
    def get_stats(self):
        """Get database statistics using aggregation pipeline"""
        try:
            stats = {}
            
            # Total products
            stats['total_products'] = self.collection.count_documents({})
            
            # Available products
            stats['available_products'] = self.collection.count_documents({'available': True})
            
            # Products by platform
            pipeline = [
                {'$group': {'_id': '$platform', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}}
            ]
            platform_stats = list(self.collection.aggregate(pipeline))
            stats['by_platform'] = {item['_id']: item['count'] for item in platform_stats}
            
            # Products by store
            pipeline = [
                {'$group': {'_id': '$store_domain', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': 10}  # Top 10 stores
            ]
            store_stats = list(self.collection.aggregate(pipeline))
            stats['by_store'] = {item['_id']: item['count'] for item in store_stats}
            
            # Price statistics
            pipeline = [
                {'$match': {'price': {'$exists': True, '$ne': None}}},
                {'$group': {
                    '_id': None,
                    'avg_price': {'$avg': '$price'},
                    'min_price': {'$min': '$price'},
                    'max_price': {'$max': '$price'}
                }}
            ]
            price_stats = list(self.collection.aggregate(pipeline))
            if price_stats:
                stats['price_stats'] = {
                    'average': round(price_stats[0]['avg_price'], 2),
                    'minimum': price_stats[0]['min_price'],
                    'maximum': price_stats[0]['max_price']
                }
            
            # Database size (approximate)
            db_stats = self.db.command("collStats", self.collection_name)
            stats['collection_size_mb'] = round(db_stats.get('size', 0) / (1024 * 1024), 2)
            stats['index_size_mb'] = round(db_stats.get('totalIndexSize', 0) / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Stats query failed: {e}")
            return {'error': str(e)}
    
    def export_to_csv(self, output_file='exported_products.csv', limit=None):
        """Export products to CSV"""
        try:
            cursor = self.collection.find({})
            if limit:
                cursor = cursor.limit(limit)
                
            products = list(cursor)
            
            if not products:
                print("❌ No products found to export")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(products)
            
            # Convert ObjectId to string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            
            # Handle datetime columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains datetime objects
                    if any(isinstance(x, datetime) for x in df[col].dropna()):
                        df[col] = df[col].astype(str)
            
            df.to_csv(output_file, index=False)
            print(f"✅ {len(products)} products exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")
            return False
    
    def backup_collection(self, backup_file='backup_products.json'):
        """Backup entire collection to JSON"""
        try:
            cursor = self.collection.find({})
            products = list(cursor)
            
            # Convert ObjectId and datetime to string
            for product in products:
                product['_id'] = str(product['_id'])
                if 'created_at' in product:
                    product['created_at'] = product['created_at'].isoformat()
                if 'updated_at' in product:
                    product['updated_at'] = product['updated_at'].isoformat()
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(products, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {len(products)} products backed up to {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backup failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.logger.info("🔌 MongoDB connection closed")

# === UTILITY FUNCTIONS ===

def analyze_csv_structure(csv_file: str):
    """Analyze CSV structure to understand the data"""
    try:
        print(f"🔍 Analyzing CSV structure: {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"📊 Total rows: {len(df)}")
        print(f"📊 Total columns: {len(df.columns)}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check for NaN values
        print(f"\n🔍 NaN values per column:")
        nan_summary = []
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_summary.append(f"  • {col}: {nan_count} NaN values")
        
        if nan_summary:
            print('\n'.join(nan_summary))
        else:
            print("  • No NaN values found!")
        
        # Sample data
        print(f"\n📝 First 3 rows:")
        for i in range(min(3, len(df))):
            print(f"\nRow {i+1}:")
            for col in df.columns:
                value = df.iloc[i][col]
                print(f"  • {col}: {value} (type: {type(value)})")
        
        return df
        
    except Exception as e:
        print(f"❌ CSV analysis failed: {e}")
        return None

def store_csv_to_database(csv_file: str, connection_string='mongodb://localhost:27017/', 
                         database_name='products_db', collection_name='products'):
    """Enhanced function to store CSV data to MongoDB"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Analyze CSV first
        df = analyze_csv_structure(csv_file)
        if df is None:
            return False
        
        # Convert to records
        print(f"\n💾 Converting to records...")
        products = df.to_dict('records')
        print(f"📊 Found {len(products)} products in CSV")
        
        # Connect to database
        print(f"\n🔌 Connecting to MongoDB...")
        db = ProductsDB(connection_string, database_name, collection_name)
        if not db.connect():
            print("❌ Failed to connect to MongoDB")
            print("💡 Make sure MongoDB is running: mongod")
            return False
            
        # Create indexes
        print(f"🏗️ Creating indexes...")
        db.create_indexes()
        
        # Insert products
        print("💾 Storing products to MongoDB...")
        inserted = db.insert_products_batch(products, batch_size=1000)
        
        # Show stats
        print(f"\n📊 Getting final statistics...")
        stats = db.get_stats()
        print(f"\n🎉 Storage completed!")
        print(f"📊 Total products in database: {stats.get('total_products', 0)}")
        print(f"✅ Available products: {stats.get('available_products', 0)}")
        print(f"📈 Successfully inserted: {inserted} out of {len(products)}")
        print(f"💾 Collection size: {stats.get('collection_size_mb', 0)} MB")
        print(f"🔍 Index size: {stats.get('index_size_mb', 0)} MB")
        
        if stats.get('by_platform'):
            print("\n📈 By Platform:")
            for platform, count in stats['by_platform'].items():
                print(f"  • {platform}: {count} products")
        
        if stats.get('price_stats'):
            price_stats = stats['price_stats']
            print(f"\n💰 Price Statistics:")
            print(f"  • Average: ${price_stats.get('average', 0)}")
            print(f"  • Min: ${price_stats.get('minimum', 0)}")
            print(f"  • Max: ${price_stats.get('maximum', 0)}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_database_setup():
    """Quick setup for testing"""
    
    # Test data
    sample_products = [
        {
            'store_domain': 'test-store.com',
            'store_region': 'US',
            'platform': 'shopify',
            'product_id': '123',
            'title': 'Amazing Headphones',
            'handle': 'amazing-headphones',
            'vendor': 'AudioTech',
            'price': 99.99,
            'compare_at_price': 129.99,
            'available': True,
            'stock_quantity': 50,
            'sku': 'AUDIO-001',
            'tags': 'electronics, audio, headphones',
            'image_src': 'https://example.com/headphones.jpg',
            'body_html': '<p>High-quality wireless headphones with noise cancellation</p>'
        },
        {
            'store_domain': 'fashion-store.com',
            'store_region': 'EU',
            'platform': 'woocommerce',
            'product_id': '456',
            'title': 'Premium T-Shirt',
            'handle': 'premium-t-shirt',
            'vendor': 'FashionBrand',
            'price': 29.99,
            'available': True,
            'stock_quantity': 100,
            'sku': 'SHIRT-002',
            'tags': 'clothing, fashion, cotton',
            'image_src': 'https://example.com/tshirt.jpg',
            'body_html': '<p>Comfortable cotton t-shirt in various colors</p>'
        },
        {
            'store_domain': 'tech-gadgets.com',
            'store_region': 'US',
            'platform': 'shopify',
            'product_id': '789',
            'title': 'Smart Watch Pro',
            'handle': 'smart-watch-pro',
            'vendor': 'TechCorp',
            'price': 299.99,
            'compare_at_price': 399.99,
            'available': False,
            'stock_quantity': 0,
            'sku': 'WATCH-003',
            'tags': 'electronics, wearables, smartwatch',
            'image_src': 'https://example.com/smartwatch.jpg',
            'body_html': '<p>Advanced smartwatch with health monitoring</p>'
        }
    ]
    
    try:
        print("🚀 Setting up test MongoDB database...")
        
        db = ProductsDB('mongodb://localhost:27017/', 'test_products_db', 'products')
        if not db.connect():
            print("❌ Failed to connect to MongoDB")
            print("💡 Make sure MongoDB is running!")
            return
            
        db.create_indexes()
        inserted = db.insert_products_batch(sample_products)
        
        print(f"✅ {inserted} test products inserted")
        
        # Show stats
        stats = db.get_stats()
        print(f"📊 Database stats:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        # Test search
        print(f"\n🔍 Testing search for 'headphones':")
        results = db.search_products('headphones')
        print(f"Found {len(results)} results")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

# === USAGE EXAMPLES ===

if __name__ == "__main__":
    
    # Fichier CSV généré par ton script d'extraction
    csv_file = 'unified_extracted_products.csv'
    
    # Configuration MongoDB
    connection_string = 'mongodb://localhost:27017/'  # Default local MongoDB
    database_name = 'products_db'
    collection_name = 'products'
    
    print("🚀 Stockage des produits extraits en MongoDB...")
    print("=" * 60)
    
    # Vérifier si le fichier CSV existe
    if os.path.exists(csv_file):
        print(f"📖 Fichier trouvé: {csv_file}")
        
        # Stocker les VRAIS produits extraits
        success = store_csv_to_database(csv_file, connection_string, database_name, collection_name)
        
        if success:
            print("\n🎉 STOCKAGE TERMINÉ AVEC SUCCÈS !")
            print(f"🗄️ Base de données: {database_name}")
            print(f"📚 Collection: {collection_name}")
            
            # Quick query examples
            print("\n🔍 Tests rapides:")
            db = ProductsDB(connection_string, database_name, collection_name)
            if db.connect():
                print(f"• Total produits: {db.get_products_count()}")
                
                # Test search
                search_results = db.search_products('test', limit=5)
                print(f"• Recherche 'test': {len(search_results)} résultats")
                
                db.close()
        else:
            print("\n❌ Erreur lors du stockage")
    else:
        print(f"❌ Fichier CSV non trouvé: {csv_file}")
        print("💡 Assure-toi d'avoir lancé le script d'extraction d'abord !")
        print("\n🧪 Lancement du test avec des données de démonstration...")
        quick_database_setup()
    
    print("\n" + "="*60)
    print("🎯 INSTALLATION ET UTILISATION:")
    print("="*60)
    print("1. Installer MongoDB: https://www.mongodb.com/try/download/community")
    print("2. Installer pymongo: pip install pymongo")
    print("3. Démarrer MongoDB: mongod")
    print("4. Utiliser: store_csv_to_database('ton_fichier.csv')")
    print("="*60)
    print("🌟 AVANTAGES MONGODB:")
    print("• Pas de limite d'espace (extensible)")
    print("• Recherche texte intégrée")
    print("• Requêtes complexes avec aggregation")
    print("• Données flexibles (pas de schéma fixe)")
    print("• Très rapide pour les gros volumes")
    print("="*60)