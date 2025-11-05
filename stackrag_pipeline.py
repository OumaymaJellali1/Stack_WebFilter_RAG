
import os
import sys
import json
import ast
import requests
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config_prompts as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def safe_parse_llm_output(text: str):
    """Parse robuste des sorties LLM"""
    text = text.strip()
    if not text:
        return []
    
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    
    text = text.replace("```json", "").replace("```python", "").replace("```", "").strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if text.startswith("[") and text.endswith("]"):
            try:
                return ast.literal_eval(text)
            except:
                return [text]
        elif text.startswith("{") and text.endswith("}"):
            try:
                return ast.literal_eval(text)
            except:
                return {}
        return [text]

# INITIALISATION LLM (Groq)

try:
    from langchain_groq import ChatGroq
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        logger.error(" GROQ_API_KEY manquante dans .env")
        sys.exit(1)
    
    logger.info(" Initialisation de Groq API (llama-3.3-70b-versatile)")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY,
        max_tokens=2048
    )
    
    logger.info("LLM Groq prêt")

except ImportError:
    logger.error("langchain-groq non installé")
    sys.exit(1)
except Exception as e:
    logger.error(f"Erreur Groq: {e}")
    sys.exit(1)

# VECTOR DATABASE (ChromaDB)

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    VECTORDB_AVAILABLE = True
except ImportError:
    VECTORDB_AVAILABLE = False
    logger.warning("ChromaDB non disponible")


class VectorDatabase:
    """Base vectorielle pour stocker les résultats SO"""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.available = VECTORDB_AVAILABLE
        if not self.available:
            return
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # FIX: Spécifier explicitement le device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"  Device utilisé: {device}")
        
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device
        )
        
        try:
            self.collection = self.client.get_collection("stackoverflow")
        except:
            self.collection = self.client.create_collection("stackoverflow")
    
    def store_results(self, results: List[Dict], query: str):
        """Stocke les résultats dans la base vectorielle"""
        if not self.available or not results:
            return
        
        try:
            documents = []
            metadatas = []
            ids = []
            
            for result in results:
                doc_text = f"{result['title']} {result['body'][:500]}"
                documents.append(doc_text)
                
                metadatas.append({
                    "question_id": str(result['question_id']),
                    "title": result['title'],
                    "score": result['score'],
                    "link": result['link'],
                    "tags": ",".join(result.get('tags', [])),
                    "query": query
                })
                
                ids.append(f"so_{result['question_id']}")
            
            embeddings = self.embedding_model.encode(documents).tolist()
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            logger.info(f"   {len(results)} résultats stockés dans Vector DB")
            
        except Exception as e:
            logger.error(f"    Erreur stockage: {e}")
    
    def search_similar(self, query: str, top_k: int = 10) -> List[Dict]:
        """Recherche sémantique dans le cache"""
        if not self.available:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            formatted = []
            if results['metadatas'] and len(results['metadatas']) > 0:
                for metadata in results['metadatas'][0]:
                    formatted.append({
                        "question_id": int(metadata['question_id']),
                        "title": metadata['title'],
                        "score": metadata['score'],
                        "link": metadata['link'],
                        "tags": metadata.get('tags', '').split(','),
                        "from_cache": True
                    })
            
            return formatted
            
        except Exception as e:
            logger.error(f"    Erreur recherche: {e}")
            return []

# TOOL 1: KEYWORD EXTRACTOR (avec complexité)

class KeywordExtractorTool:
    """
    TOOL 1 du diagramme:
    - Check complexity
    - Divide into sub-questions if complex
    - Extract keywords
    """
    
    def __init__(self, llm):
        self.llm = llm
        
        # Chains LLM
        self.complexity_checker = (
            PromptTemplate.from_template(config.QUESTION_COMPLEXITY_CHECKER_PROMPT)
            | llm
            | StrOutputParser()
        )
        
        self.question_divider = (
            PromptTemplate.from_template(config.QUESTION_DIVIDER_PROMPT)
            | llm
            | StrOutputParser()
        )
        
        self.keyword_extractor = (
            PromptTemplate.from_template(config.KEYWORD_EXTRACTOR_PROMPT)
            | llm
            | StrOutputParser()
        )
    
    def process(self, user_question: str) -> Dict:
        """
        Traite la question utilisateur
        
        Returns:
            {
                "is_complex": bool,
                "sub_questions": List[str],
                "keywords": List[str],
                "original_question": str
            }
        """
        logger.info(" TOOL 1: KEYWORD EXTRACTOR")
        logger.info(f"   Question: {user_question[:80]}...")
        
        # Étape 1: Check complexity
        complexity_result = self.complexity_checker.invoke({"question": user_question})
        is_complex = "TRUE" in complexity_result.strip().upper()
        
        logger.info(f"   Complexité: {'OUI' if is_complex else 'NON'}")
        
        # Étape 2: Divide if complex
        sub_questions = []
        if is_complex:
            division_result = self.question_divider.invoke({"question": user_question})
            sub_questions = safe_parse_llm_output(division_result)
            if not isinstance(sub_questions, list):
                sub_questions = [user_question]
            logger.info(f"    {len(sub_questions)} sous-questions")
        else:
            sub_questions = [user_question]
        
        # Étape 3: Extract keywords
        all_keywords = []
        for sq in sub_questions:
            keyword_result = self.keyword_extractor.invoke({"question": sq})
            keywords = safe_parse_llm_output(keyword_result)
            
            if not isinstance(keywords, list):
                keywords = self._fallback_keywords(sq)
            elif len(keywords) < 4:
                keywords.extend(self._fallback_keywords(sq))
                keywords = keywords[:4]
            elif len(keywords) > 4:
                keywords = keywords[:4]
            
            all_keywords.extend(keywords)
        
        # Garder uniques (max 4)
        unique_keywords = list(dict.fromkeys(all_keywords))[:4]
        
        logger.info(f"   🔑 Keywords: {unique_keywords}")
        
        return {
            "is_complex": is_complex,
            "sub_questions": sub_questions,
            "keywords": unique_keywords,
            "original_question": user_question
        }
    
    def _fallback_keywords(self, question: str) -> List[str]:
        """Extraction fallback"""
        stop_words = {
            'how', 'what', 'why', 'when', 'where', 'can', 'could', 'should',
            'i', 'to', 'the', 'a', 'an', 'in', 'on', 'is', 'are', 'do', 'does',
            'with', 'using', 'for', 'from', 'this', 'that', 'my', 'implement'
        }
        
        words = [w.strip('?.,!') for w in question.lower().split() 
                if w not in stop_words and len(w) > 2]
        words.sort(key=len, reverse=True)
        
        return words[:4]

# TOOL 2: SEARCH AND STORAGE (WebFilter + Stack Overflow + Vector DB)


class SearchAndStorageTool:
    """
    TOOL 2 du diagramme:
    - Reformule avec WebFilter
    - Recherche Stack Overflow
    - Stocke dans Vector DB
    """
    
    def __init__(self, llm, vector_db: VectorDatabase):
        self.llm = llm
        self.vector_db = vector_db
        
        # WebFilter reformulation
        self.query_reformulator = (
            PromptTemplate.from_template(config.QUERY_REFORMULATOR_PROMPT)
            | llm
            | StrOutputParser()
        )
        
        self.operator_selector = (
            PromptTemplate.from_template(config.SEARCH_OPERATOR_SELECTOR_PROMPT)
            | llm
            | StrOutputParser()
        )
        
        # Stack Overflow API
        self.so_api_url = "https://api.stackexchange.com/2.3/search/advanced"
        self.api_key = os.getenv("STACKOVERFLOW_API_KEY", None)
    
    def process(self, keywords: List[str], original_question: str, max_results: int = 15) -> Dict:
        """
        Recherche et stockage
        
        Returns:
            {
                "queries": List[Dict],
                "total_results": int,
                "results": List[Dict],
                "stored_in_db": bool
            }
        """
        logger.info("TOOL 2: SEARCH AND STORAGE")
        
        # Étape 1: WebFilter - Reformuler queries
        queries = self._reformulate_queries(keywords, original_question)
        logger.info(f"    {len(queries)} requêtes générées")
        
        # Étape 2: Recherche Stack Overflow
        all_results = []
        for query_config in queries:
            results = self._search_stackoverflow(query_config, max_results)
            all_results.extend(results)
        
        # Dédupliquer
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r["question_id"] not in seen_ids:
                seen_ids.add(r["question_id"])
                unique_results.append(r)
        
        logger.info(f"    {len(unique_results)} résultats uniques")
        
        # Étape 3: Stocker dans Vector DB
        self.vector_db.store_results(unique_results, original_question)
        
        return {
            "queries": queries,
            "total_results": len(unique_results),
            "results": unique_results,
            "stored_in_db": self.vector_db.available
        }
    
    def _reformulate_queries(self, keywords: List[str], original_question: str) -> List[Dict]:
        """Reformulation WebFilter"""
        try:
            # Reformulation LLM
            reformulated = self.query_reformulator.invoke({
                "keywords": str(keywords),
                "original_question": original_question
            })
            base_queries = safe_parse_llm_output(reformulated)
            
            if not isinstance(base_queries, list) or len(base_queries) == 0:
                base_queries = [" ".join(keywords[:3])]
            
            base_queries = base_queries[:3]
            
        except Exception as e:
            logger.warning(f"   Erreur reformulation: {e}")
            base_queries = [" ".join(keywords[:3])]
        
        # Opérateurs
        try:
            operator_result = self.operator_selector.invoke({
                "question": original_question
            })
            operators = safe_parse_llm_output(operator_result)
            
            if not isinstance(operators, dict):
                operators = {"min_score": 1, "accepted_only": False}
                
        except:
            operators = {"min_score": 1, "accepted_only": False}
        
        # Construction queries finales
        queries = []
        for query in base_queries:
            queries.append({
                "base_query": query,
                "search_params": operators
            })
        
        return queries
    
    def _search_stackoverflow(self, query_config: Dict, max_results: int) -> List[Dict]:
        """Recherche Stack Overflow API"""
        base_query = query_config["base_query"]
        search_params = query_config["search_params"]
        
        params = {
            "site": "stackoverflow",
            "q": base_query,
            "pagesize": max_results,
            "order": "desc",
            "sort": "relevance",
            "filter": "withbody"
        }
        
        min_score = search_params.get("min_score", 1)
        if min_score > 1:
            params["min"] = min_score
        
        if search_params.get("accepted_only", False):
            params["accepted"] = "True"
        
        if self.api_key:
            params["key"] = self.api_key
        
        try:
            response = requests.get(self.so_api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            questions = data.get("items", [])
            
            # Retry sans filtres si aucun résultat
            if len(questions) == 0 and (min_score > 1 or search_params.get("accepted_only")):
                params.pop("min", None)
                params.pop("accepted", None)
                
                response = requests.get(self.so_api_url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                questions = data.get("items", [])
            
            formatted = []
            for q in questions:
                formatted.append({
                    "question_id": q.get("question_id"),
                    "title": q.get("title", ""),
                    "body": q.get("body", "")[:1000],
                    "score": q.get("score", 0),
                    "answer_count": q.get("answer_count", 0),
                    "is_answered": q.get("is_answered", False),
                    "link": q.get("link", ""),
                    "tags": q.get("tags", []),
                    "creation_date": q.get("creation_date", 0)
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"   Erreur SO API: {e}")
            return []

# TOOL 3: GATHER EVIDENCE (Recherche Vector DB + Scoring + BM25)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning(" BM25 non disponible")

class GatherEvidenceTool:
    """
    TOOL 3 du diagramme:
    - Recherche similaires dans Vector DB
    - Score pertinence avec LLM
    - Re-ranking BM25
    - Check evidence suffisante
    """
    
    def __init__(self, llm, vector_db: VectorDatabase):
        self.llm = llm
        self.vector_db = vector_db
        
        # Scoring LLM
        self.relevance_scorer = (
            PromptTemplate.from_template(config.RELEVANCE_SCORING_PROMPT)
            | llm
            | StrOutputParser()
        )
    
    def process(self, original_question: str, so_results: List[Dict], top_k: int = 5) -> Dict:
        """
        Rassemble et filtre les evidences
        
        Returns:
            {
                "vector_results": List[Dict],
                "scored_results": List[Dict],
                "top_k_results": List[Dict],
                "evidence_status": str,
                "enough_evidence": bool
            }
        """
        logger.info(" TOOL 3: GATHER EVIDENCE")
        
        # Étape 1: Recherche similaires dans Vector DB
        vector_results = self.vector_db.search_similar(original_question, top_k=10)
        logger.info(f"    {len(vector_results)} résultats Vector DB")
        
        # Combiner SO + Vector DB
        all_results = so_results + vector_results
        
        # Dédupliquer
        seen_ids = set()
        unique = []
        for r in all_results:
            if r["question_id"] not in seen_ids:
                seen_ids.add(r["question_id"])
                unique.append(r)
        
        logger.info(f"    {len(unique)} résultats uniques combinés")
        
        # Étape 2: Scoring avec LLM
        scored_results = self._score_relevance(unique, original_question)
        
        # Étape 3: Re-ranking BM25
        top_results = self._rerank_bm25(scored_results, original_question, top_k)
        
        # Étape 4: Check evidence
        enough_evidence = len(top_results) >= 3 and any(r.get('final_score', 0) > 6 for r in top_results)
        evidence_status = "sufficient" if enough_evidence else "insufficient"
        
        logger.info(f"   {'' if enough_evidence else ''} Evidence: {evidence_status}")
        
        return {
            "vector_results": vector_results,
            "scored_results": scored_results,
            "top_k_results": top_results,
            "evidence_status": evidence_status,
            "enough_evidence": enough_evidence,
            "average_score": sum(r.get('final_score', 0) for r in top_results) / len(top_results) if top_results else 0
        }
    
    def _score_relevance(self, results: List[Dict], question: str) -> List[Dict]:
        """Score avec LLM"""
        scored = []
        
        for i, result in enumerate(results):
            try:
                title = result.get('title', '')
                body = result.get('body', '')[:300]
                tags = result.get('tags', [])
                score = result.get('score', 0)
                
                score_result = self.relevance_scorer.invoke({
                    "question": question,
                    "title": title,
                    "tags": ", ".join(tags) if tags else "None",
                    "score": score,
                    "body_preview": body
                })
                
                relevance_score = self._parse_score(score_result)
                result["webfilter_score"] = relevance_score
                scored.append(result)
                
            except Exception as e:
                result["webfilter_score"] = 5.0
                scored.append(result)
        
        return scored
    
    def _parse_score(self, score_text: str) -> float:
        """Parse score LLM"""
        import re
        numbers = re.findall(r'\d+\.?\d*', score_text.strip())
        
        if numbers:
            return max(0.0, min(10.0, float(numbers[0])))
        return 5.0
    
    def _rerank_bm25(self, results: List[Dict], query: str, top_k: int) -> List[Dict]:
        """Re-ranking BM25"""
        if not BM25_AVAILABLE or not results:
            sorted_results = sorted(results, key=lambda x: x.get("webfilter_score", 0), reverse=True)
            return sorted_results[:top_k]
        
        # BM25
        corpus = [f"{r['title']} {r.get('body', '')}".lower().split() for r in results]
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Combiner scores (70% WebFilter + 30% BM25)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        
        for i, result in enumerate(results):
            webfilter = result.get("webfilter_score", 5.0)
            bm25_norm = (bm25_scores[i] / max_bm25) * 10
            
            result["bm25_score"] = bm25_norm
            result["final_score"] = 0.7 * webfilter + 0.3 * bm25_norm
        
        ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)
        
        return ranked[:top_k]

# TOOL 4: ANSWER GENERATOR (StackRAG avec citations)

class AnswerGeneratorTool:
    """
    TOOL 4 du diagramme:
    - Génère réponse avec citations
    - Utilise evidences filtrées
    """
    
    def __init__(self, llm):
        self.llm = llm
        
        self.generator = (
            PromptTemplate.from_template(config.FINAL_GENERATION_PROMPT)
            | llm
            | StrOutputParser()
        )
    
    def process(self, original_question: str, evidences: List[Dict]) -> Dict:
        """
        Génère réponse finale
        
        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "num_sources": int,
                "timestamp": str
            }
        """
        logger.info(" TOOL 4: ANSWER GENERATOR")
        
        if not evidences:
            return {
                "answer": "Désolé, aucune evidence pertinente trouvée pour répondre à votre question.",
                "sources": [],
                "num_sources": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Construire contexte
        context_parts = []
        sources = []
        
        for i, evidence in enumerate(evidences, 1):
            context_parts.append(f"[Source {i}]")
            context_parts.append(f"Title: {evidence['title']}")
            context_parts.append(f"Content: {evidence.get('body', '')[:800]}")
            context_parts.append(f"Score: {evidence['score']}, Answers: {evidence.get('answer_count', 0)}")
            context_parts.append("")
            
            sources.append({
                "source_id": i,
                "title": evidence['title'],
                "link": evidence['link'],
                "score": evidence['score'],
                "final_score": evidence.get('final_score', 0)
            })
        
        context = "\n".join(context_parts)
        
        # Générer réponse
        try:
            answer = self.generator.invoke({
                "question": original_question,
                "context": context,
                "num_sources": len(evidences)
            })
            
            logger.info("  Réponse générée")
            
            return {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"   Erreur génération: {e}")
            return {
                "answer": f"Erreur lors de la génération: {str(e)}",
                "sources": sources,
                "num_sources": 0,
                "error": str(e)
            }

# LLM AGENT ORCHESTRATOR (Architecture du diagramme)


class LLMAgentOrchestrator:
    """
    Agent LLM central qui coordonne les 4 outils selon le diagramme
    """
    
    def __init__(self, llm):
        self.llm = llm
        
        # Initialiser Vector DB
        self.vector_db = VectorDatabase()
        
        # Initialiser les 4 outils
        self.keyword_extractor = KeywordExtractorTool(llm)
        self.search_storage = SearchAndStorageTool(llm, self.vector_db)
        self.gather_evidence = GatherEvidenceTool(llm, self.vector_db)
        self.answer_generator = AnswerGeneratorTool(llm)
    
    def process_question(self, user_question: str, max_results: int = 15, top_k: int = 5) -> Dict:
        """
        Pipeline complet selon le diagramme
        
        Flow:
        Input → Tool 1 (Keywords) → Tool 2 (Search+Storage) → Tool 3 (Evidence) 
        → Tool 4 (Answer) → Output
        """
        logger.info("="*80)
        logger.info(" LLM AGENT - STACKRAG PIPELINE")
        logger.info("="*80)
        
        pipeline_results = {
            "input_question": user_question,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # ===== TOOL 1: KEYWORD EXTRACTOR =====
            logger.info("\n" + "="*80)
            tool1_output = self.keyword_extractor.process(user_question)
            pipeline_results["tool1_keywords"] = tool1_output
            
            # ===== TOOL 2: SEARCH AND STORAGE =====
            logger.info("\n" + "="*80)
            tool2_output = self.search_storage.process(
                keywords=tool1_output["keywords"],
                original_question=user_question,
                max_results=max_results
            )
            pipeline_results["tool2_search"] = tool2_output
            
            # ===== TOOL 3: GATHER EVIDENCE =====
            logger.info("\n" + "="*80)
            tool3_output = self.gather_evidence.process(
                original_question=user_question,
                so_results=tool2_output["results"],
                top_k=top_k
            )
            pipeline_results["tool3_evidence"] = tool3_output
            
            # ===== VÉRIFICATION EVIDENCE =====
            if not tool3_output["enough_evidence"]:
                logger.warning(" Evidence insuffisante, retour vers recherche...")
              
            
            # ===== TOOL 4: ANSWER GENERATOR =====
            logger.info("\n" + "="*80)
            tool4_output = self.answer_generator.process(
                original_question=user_question,
                evidences=tool3_output["top_k_results"]
            )
            pipeline_results["tool4_answer"] = tool4_output
            
            logger.info("\n" + "="*80)
            logger.info(" PIPELINE TERMINÉ")
            logger.info("="*80)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f" ERREUR PIPELINE: {e}")
            import traceback
            traceback.print_exc()
            raise

# FONCTION PRINCIPALE


def run_stackrag_pipeline(question: str, max_results: int = 15, top_k: int = 5) -> Dict:
    """Point d'entrée principal"""
    orchestrator = LLMAgentOrchestrator(llm)
    return orchestrator.process_question(question, max_results, top_k)

# TEST

if __name__ == "__main__":
    test_question = "How do I implement JWT authentication in Flask with refresh tokens?"
    
    print("\n" + "="*80)
    print(" TEST DU PIPELINE")
    print("="*80)
    print(f"\nQuestion: {test_question}\n")
    
    results = run_stackrag_pipeline(test_question, max_results=10, top_k=5)
    
    # Affichage résultats
    print("\n" + "="*80)
    print(" RÉSULTATS")
    print("="*80)
    
    print("\n TOOL 1 - Keywords:")
    print(f"   Complexe: {results['tool1_keywords']['is_complex']}")
    print(f"   Sous-questions: {len(results['tool1_keywords']['sub_questions'])}")
    print(f"   Keywords: {results['tool1_keywords']['keywords']}")
    
    print("\n TOOL 2 - Search & Storage:")
    print(f"   Requêtes: {len(results['tool2_search']['queries'])}")
    print(f"   Résultats trouvés: {results['tool2_search']['total_results']}")
    print(f"   Stocké en DB: {results['tool2_search']['stored_in_db']}")
    
    print("\n TOOL 3 - Evidence:")
    print(f"   Résultats Vector DB: {len(results['tool3_evidence']['vector_results'])}")
    print(f"   Score moyen: {results['tool3_evidence']['average_score']:.2f}/10")
    print(f"   Evidence status: {results['tool3_evidence']['evidence_status']}")
    print(f"   Top K sélectionnés: {len(results['tool3_evidence']['top_k_results'])}")
    
    print("\n TOOL 4 - Answer:")
    print(f"   Sources utilisées: {results['tool4_answer']['num_sources']}")
    print(f"\n{'='*80}")
    print("RÉPONSE FINALE:")
    print(f"{'='*80}\n")
    print(results['tool4_answer']['answer'])
    
    print(f"\n{'='*80}")
    print(" SOURCES:")
    print(f"{'='*80}")
    for src in results['tool4_answer']['sources']:
        print(f"\n[{src['source_id']}] {src['title']}")
        print(f"    Score: {src['final_score']:.2f}/10")
        print(f"    Link: {src['link']}")