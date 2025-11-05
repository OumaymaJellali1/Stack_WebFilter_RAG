
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    VECTORDB_AVAILABLE = True
    logger.info(" ChromaDB et sentence-transformers disponibles")
except ImportError:
    VECTORDB_AVAILABLE = False
    logger.warning(" ChromaDB ou sentence-transformers non installés")

class StackOverflowVectorStore:
    """
    Gestionnaire de base vectorielle pour cacher les résultats Stack Overflow.
    Utilise ChromaDB (gratuit, local) + sentence-transformers (gratuit, local).
    
    Avantages:
    - Recherche sémantique sur les résultats SO
    - Cache local pour éviter requêtes répétées
    - 100% gratuit et local (pas de coût API)
    """
    
    def __init__(self, 
                 collection_name: str = "stackoverflow_cache",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialise la base vectorielle.
        
        Args:
            collection_name: Nom de la collection ChromaDB
            persist_directory: Répertoire de stockage local
            embedding_model: Modèle sentence-transformers (gratuit, local)
        """
        if not VECTORDB_AVAILABLE:
            logger.error("ChromaDB non disponible - installez: pip install chromadb sentence-transformers")
            self.available = False
            return
        
        self.available = True
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Créer répertoire si nécessaire
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialiser ChromaDB (local, persistant)
        logger.info(f" Initialisation ChromaDB: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialiser modèle d'embeddings (local, gratuit)
        logger.info(f" Chargement modèle embeddings: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Créer ou récupérer collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f" Collection '{collection_name}' récupérée")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Stack Overflow search results cache"}
            )
            logger.info(f" Collection '{collection_name}' créée")
    
    def add_results(self, results: List[Dict], query: str) -> bool:
        """
        Ajoute des résultats SO à la base vectorielle.
        
        Args:
            results: Liste de résultats Stack Overflow
            query: Requête d'origine (pour métadonnées)
        
        Returns:
            True si succès
        """
        if not self.available or not results:
            return False
        
        try:
            documents = []
            metadatas = []
            ids = []
            
            for i, result in enumerate(results):
                # Créer document texte
                doc_text = f"{result['title']} {result['body'][:500]}"
                documents.append(doc_text)
                
                # Métadonnées
                metadatas.append({
                    "question_id": str(result['question_id']),
                    "title": result['title'],
                    "score": result['score'],
                    "link": result['link'],
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                })
                
                # ID unique
                ids.append(f"so_{result['question_id']}_{i}")
            
            # Générer embeddings (local)
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Ajouter à ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            logger.info(f" {len(results)} résultats ajoutés au cache vectoriel")
            return True
            
        except Exception as e:
            logger.error(f" Erreur ajout cache: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Recherche sémantique dans le cache.
        
        Args:
            query: Question de recherche
            top_k: Nombre de résultats
        
        Returns:
            Liste de résultats similaires
        """
        if not self.available:
            return []
        
        try:
            # Générer embedding de la query
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Recherche dans ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Formater résultats
            formatted_results = []
            if results['metadatas'] and len(results['metadatas']) > 0:
                for i, metadata in enumerate(results['metadatas'][0]):
                    formatted_results.append({
                        "question_id": metadata['question_id'],
                        "title": metadata['title'],
                        "score": metadata['score'],
                        "link": metadata['link'],
                        "distance": results['distances'][0][i] if results['distances'] else 0,
                        "from_cache": True
                    })
            
            logger.info(f" Trouvé {len(formatted_results)} résultats similaires dans le cache")
            return formatted_results
            
        except Exception as e:
            logger.error(f" Erreur recherche cache: {e}")
            return []
    
    def get_cache_stats(self) -> Dict:
        """Retourne statistiques du cache"""
        if not self.available:
            return {"available": False}
        
        try:
            count = self.collection.count()
            return {
                "available": True,
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except:
            return {"available": True, "error": "Impossible de récupérer stats"}
    
    def clear_cache(self) -> bool:
        """Vide le cache (utile pour tests)"""
        if not self.available:
            return False
        
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Stack Overflow search results cache"}
            )
            logger.info(" Cache vidé")
            return True
        except Exception as e:
            logger.error(f" Erreur vidage cache: {e}")
            return False


def demo_vectordb():
    """Démonstration du vector store"""
    
    print("\n" + "="*80)
    print(" DÉMONSTRATION CHROMADB VECTOR STORE")
    print("="*80)
    
    # Initialiser
    vector_store = StackOverflowVectorStore()
    
    if not vector_store.available:
        print(" VectorDB non disponible")
        return
    
    # Exemples de résultats SO
    fake_results = [
        {
            "question_id": 123,
            "title": "How to implement JWT authentication in Flask",
            "body": "I want to secure my Flask API with JWT tokens...",
            "score": 45,
            "link": "https://stackoverflow.com/q/123"
        },
        {
            "question_id": 456,
            "title": "Flask JWT refresh tokens implementation",
            "body": "Best practices for refresh tokens in Flask...",
            "score": 32,
            "link": "https://stackoverflow.com/q/456"
        }
    ]
    
    # Ajouter au cache
    print("\n Ajout de résultats au cache...")
    vector_store.add_results(fake_results, "flask jwt authentication")
    
    # Stats
    print("\n Statistiques du cache:")
    stats = vector_store.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Recherche sémantique
    print("\n Recherche sémantique: 'secure flask api with tokens'")
    similar = vector_store.search_similar("secure flask api with tokens", top_k=2)
    
    for i, result in enumerate(similar, 1):
        print(f"\n  [{i}] {result['title']}")
        print(f"      Distance: {result['distance']:.4f}")
        print(f"      Link: {result['link']}")

if __name__ == "__main__":
    demo_vectordb()