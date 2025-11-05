import os
import sys
import json
import ast
import time
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
    text = text.strip()
    if not text:
        return []
    
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    if text.startswith("```python"):
        text = text.replace("```python", "").replace("```", "").strip()
    
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

try:
    from langchain_groq import ChatGroq
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY manquante dans .env")
        sys.exit(1)
    
    logger.info("⚡ Initialisation de Groq API (llama-3.3-70b-versatile)")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY,
        max_tokens=2048
    )
    
    logger.info(" LLM Groq prêt")

except ImportError:
    logger.error(" langchain-groq non installé")
    sys.exit(1)
except Exception as e:
    logger.error(f" Erreur Groq: {e}")
    sys.exit(1)

# ÉTAPES 1-4 

class QuestionReceiver:
    @staticmethod
    def receive_question(question: str) -> Dict:
        logger.info("📥 ÉTAPE 1: Réception")
        if not question.strip():
            raise ValueError("Question vide")
        return {
            "original_question": question.strip(),
            "timestamp": datetime.now().isoformat(),
            "word_count": len(question.split()),
            "char_count": len(question)
        }

class ComplexityAnalyzer:
    def __init__(self, llm):
        self.llm = llm
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

    def analyze_and_decompose(self, question: str) -> Dict:
        logger.info("🔍 ÉTAPE 2: Analyse complexité")
        
        complexity_result = self.complexity_checker.invoke({"question": question})
        is_complex = "TRUE" in complexity_result.strip().upper()
        
        result = {
            "is_complex": is_complex,
            "original_question": question,
            "sub_questions": []
        }

        if is_complex:
            decomposition_result = self.question_divider.invoke({"question": question})
            try:
                sub_questions = safe_parse_llm_output(decomposition_result)
                if isinstance(sub_questions, list):
                    result["sub_questions"] = sub_questions
                else:
                    result["sub_questions"] = [question]
            except:
                result["sub_questions"] = [question]
        else:
            result["sub_questions"] = [question]

        return result

class KeywordExtractor:
    """Extraction stricte de 4 mots-clés"""
    
    def __init__(self, llm):
        self.llm = llm
        self.extractor = (
            PromptTemplate.from_template(config.KEYWORD_EXTRACTOR_PROMPT)
            | llm
            | StrOutputParser()
        )

    def extract_keywords(self, questions: List[str]) -> Dict:
        logger.info("🔑 ÉTAPE 3: Extraction mots-clés (exactement 4)")
        keywords_map = {}
        all_keywords = []
        
        for question in questions:
            try:
                result = self.extractor.invoke({"question": question})
                keywords = safe_parse_llm_output(result)
                
                # Forcer exactement 4 mots-clés
                if not isinstance(keywords, list):
                    keywords = self._simple_keyword_extraction(question)
                elif len(keywords) < 4:
                    keywords.extend(self._simple_keyword_extraction(question))
                    keywords = keywords[:4]
                elif len(keywords) > 4:
                    keywords = keywords[:4]
                
                keywords_map[question] = keywords
                all_keywords.extend(keywords)
                
                logger.info(f"    Question: {question[:50]}...")
                logger.info(f"    Keywords: {keywords}")
                
            except Exception as e:
                logger.warning(f"   Erreur extraction: {e}")
                keywords = self._simple_keyword_extraction(question)
                keywords_map[question] = keywords
                all_keywords.extend(keywords)
        
        # Garder mots-clés uniques mais limiter à 4 principaux
        unique_keywords = list(dict.fromkeys(all_keywords))[:4]
        
        return {
            "keywords_by_question": keywords_map,
            "all_keywords": all_keywords,
            "unique_keywords": unique_keywords,
            "total_count": len(unique_keywords)
        }

    def _simple_keyword_extraction(self, question: str) -> List[str]:
        """Extraction fallback intelligente"""
        stop_words = {
            'how', 'what', 'why', 'when', 'where', 'can', 'could', 'should',
            'i', 'to', 'the', 'a', 'an', 'in', 'on', 'is', 'are', 'do', 'does',
            'with', 'using', 'for', 'from', 'this', 'that', 'my', 'implement',
            'create', 'make', 'get', 'set', 'use'
        }
        
        words = question.lower().split()
        
        # Filtrer et prioriser mots longs (plus susceptibles d'être techniques)
        keywords = [w.strip('?.,!') for w in words 
                   if w not in stop_words and len(w) > 2]
        
        # Trier par longueur (mots techniques sont souvent plus longs)
        keywords.sort(key=len, reverse=True)
        
        return keywords[:4]



class WebFilterReformulator:
    """Reformulation simplifiée pour Stack Overflow"""
    
    def __init__(self, llm):
        self.llm = llm
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

    def reformulate_queries(self, keywords: List[str], original_question: str) -> Dict:
        logger.info(" ÉTAPE 4: Reformulation simplifiée")
        
        # Reformulation LLM
        try:
            reformulated = self.query_reformulator.invoke({
                "keywords": str(keywords),
                "original_question": original_question
            })
            base_queries = safe_parse_llm_output(reformulated)
            
            if not isinstance(base_queries, list) or len(base_queries) == 0:
                base_queries = self._create_fallback_queries(keywords)
            
            # Limiter à 3 requêtes max
            base_queries = base_queries[:3]
            
        except Exception as e:
            logger.warning(f"Erreur reformulation: {e}")
            base_queries = self._create_fallback_queries(keywords)

        # Sélection opérateurs (minimaliste)
        try:
            operator_result = self.operator_selector.invoke({
                "question": original_question
            })
            operators = safe_parse_llm_output(operator_result)
            
            if not isinstance(operators, dict):
                operators = {"min_score": 1, "accepted_only": False}
                
        except Exception as e:
            logger.warning(f" Erreur opérateurs: {e}")
            operators = {"min_score": 1, "accepted_only": False}

        # Construction requêtes finales
        final_queries = []
        for query in base_queries:
            enhanced_query = self._build_enhanced_query(query, operators)
            final_queries.append(enhanced_query)
            logger.info(f"   Query: {enhanced_query['full_query']}")

        return {
            "base_queries": base_queries,
            "operators": operators,
            "final_queries": final_queries,
            "query_count": len(final_queries)
        }

    def _create_fallback_queries(self, keywords: List[str]) -> List[str]:
        """Génère requêtes simples en cas d'échec LLM"""
        if len(keywords) < 2:
            return [" ".join(keywords)]
        
        queries = []
        
        # Requête avec tous les mots-clés
        queries.append(" ".join(keywords[:3]))
        
        # Requête avec 2 premiers (souvent les plus importants)
        if len(keywords) >= 2:
            queries.append(" ".join(keywords[:2]))
        
        return queries

    def _build_enhanced_query(self, base_query: str, operators: Dict) -> Dict:
        """Construction requête avec filtres minimaux"""
        query_components = {
            "base_query": base_query,
            "site": "stackoverflow.com",
            "filters": []
        }
        
        # Seuls filtres: score minimum et accepted
        min_score = operators.get("min_score", 1)
        if min_score > 1:
            query_components["filters"].append(f"score:>={min_score}")
        
        if operators.get("accepted_only", False):
            query_components["filters"].append("accepted:yes")
        
        # Construction requête finale simple
        full_query = base_query
        query_components["full_query"] = full_query
        query_components["search_params"] = {
            "min_score": min_score,
            "accepted_only": operators.get("accepted_only", False)
        }
        
        return query_components

# ÉTAPE 5: RECHERCHE & FILTRES (Stack Overflow API + WebFilter dynamique)

class StackOverflowSearcher:
    """Recherche simplifiée et plus flexible"""
    
    def __init__(self):
        self.base_url = "https://api.stackexchange.com/2.3/search/advanced"
        self.api_key = os.getenv("STACKOVERFLOW_API_KEY", None)
        
    def search_questions(self, query_config: Dict, max_results: int = 15) -> List[Dict]:
        """Recherche avec paramètres simplifiés"""
        base_query = query_config.get("base_query", "")
        search_params = query_config.get("search_params", {})
        
        logger.info(f" Recherche: '{base_query}'")
        
        params = {
            "site": "stackoverflow",
            "q": base_query,
            "pagesize": max_results,
            "order": "desc",
            "sort": "relevance",  # Tri par pertinence (meilleur pour nos cas)
            "filter": "withbody"
        }
        
        # Filtres optionnels
        min_score = search_params.get("min_score", 1)
        if min_score > 1:
            params["min"] = min_score
        
        if search_params.get("accepted_only", False):
            params["accepted"] = "True"
        
        if self.api_key:
            params["key"] = self.api_key
            
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            questions = data.get("items", [])
            logger.info(f"   {len(questions)} résultats trouvés")
            
            # Si aucun résultat avec filtres, réessayer sans filtres
            if len(questions) == 0 and (min_score > 1 or search_params.get("accepted_only")):
                logger.info("   Aucun résultat, réessai sans filtres...")
                params.pop("min", None)
                params.pop("accepted", None)
                
                response = requests.get(self.base_url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                questions = data.get("items", [])
                logger.info(f"   {len(questions)} résultats trouvés (sans filtres)")
            
            # Formater résultats
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
            
        except requests.RequestException as e:
            logger.error(f"   Erreur API: {e}")
            return []
        except Exception as e:
            logger.error(f"   Erreur: {e}")
            return []
# ÉTAPES 6-7: FILTRAGE & RE-RANKING (WebFilter RL + BM25)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    logger.warning(" rank-bm25 non installé, BM25 désactivé")
    BM25_AVAILABLE = False

class WebFilterRelevanceScorer:
    """Agent WebFilter pour scorer la pertinence"""
    
    def __init__(self, llm):
        self.llm = llm
        self.relevance_scorer = (
            PromptTemplate.from_template(config.RELEVANCE_SCORING_PROMPT)
            | llm
            | StrOutputParser()
        )
    
    def score_results(self, results: List[Dict], original_question: str) -> Dict:
        """
        Score chaque résultat avec WebFilter (RL)
        
        RETOURNE UN DICT avec:
        - scored_results: liste des résultats avec scores
        - average_score: score moyen
        - successful_scores: nombre de succès
        - failed_scores: nombre d'échecs
        - total_processed: nombre total traité
        """
        logger.info(" ÉTAPE 6: Scoring pertinence (WebFilter RL)")
        
        if not results:
            logger.warning("   Aucun résultat à scorer")
            return {
                "scored_results": [],
                "average_score": 0.0,
                "successful_scores": 0,
                "failed_scores": 0,
                "total_processed": 0
            }
        
        scored_results = []
        total_score = 0.0
        successful_scores = 0
        failed_scores = 0
        
        for i, result in enumerate(results):
            try:
                # Construire contexte pour le LLM
                title = result.get('title', 'No title')
                body = result.get('body', '')
                tags = result.get('tags', [])
                score = result.get('score', 0)
                
                # Limiter body à 300 caractères
                body_preview = body[:300] + "..." if len(body) > 300 else body
                
                logger.info(f"  [{i+1}/{len(results)}] Scoring: {title[:60]}...")
                
                # Demander score au LLM
                score_result = self.relevance_scorer.invoke({
                    "question": original_question,
                    "title": title,
                    "tags": ", ".join(tags) if tags else "None",
                    "score": score,
                    "body_preview": body_preview
                })
                
                # Parser le score
                relevance_score = self._parse_score(score_result)
                logger.info(f"      Score: {relevance_score:.1f}/10")
                
                # Ajouter le score au résultat
                result["webfilter_score"] = relevance_score
                scored_results.append(result)
                
                total_score += relevance_score
                successful_scores += 1
                
            except Exception as e:
                logger.error(f"   Erreur scoring #{i+1}: {e}")
                
                # Score neutre en cas d'erreur
                result["webfilter_score"] = 5.0
                scored_results.append(result)
                failed_scores += 1
        
        # Calculer moyenne
        avg_score = total_score / successful_scores if successful_scores > 0 else 0.0
        
        logger.info(f"   Scoring terminé")
        logger.info(f"      Moyenne: {avg_score:.2f}/10")
        logger.info(f"      Succès: {successful_scores}/{len(results)}")
        logger.info(f"      Échecs: {failed_scores}")
        
        # RETOURNER UN DICT COMPLET
        return {
            "scored_results": scored_results,
            "average_score": round(avg_score, 2),
            "successful_scores": successful_scores,
            "failed_scores": failed_scores,
            "total_processed": len(results)
        }
    
    def _parse_score(self, score_result: str) -> float:
        """Parse robuste du score LLM"""
        import re
        
        score_text = score_result.strip()
        
        # Méthode 1: Extraire le premier nombre
        numbers = re.findall(r'\d+\.?\d*', score_text)
        
        if numbers:
            relevance_score = float(numbers[0])
            # Clamp entre 0 et 10
            return max(0.0, min(10.0, relevance_score))
        
        # Méthode 2: Mots-clés
        score_keywords = {
            'perfect': 10.0, 'excellent': 9.0, 'high': 8.0,
            'good': 7.0, 'relevant': 6.0, 'moderate': 5.0,
            'low': 4.0, 'poor': 3.0, 'irrelevant': 2.0
        }
        
        score_lower = score_text.lower()
        for keyword, score_val in score_keywords.items():
            if keyword in score_lower:
                return score_val
        
        # Fallback
        logger.warning(f"      Aucun score trouvé, utilisation de 5.0 par défaut")
        return 5.0



class BM25Reranker:
    """Re-ranking avec BM25 (AMÉLIORÉ avec logs détaillés)"""
    
    def __init__(self):
        self.available = BM25_AVAILABLE
    
    def rerank(self, results: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
        """Re-rank avec BM25 si disponible"""
        logger.info("ÉTAPE 7: Re-ranking BM25")
        
        if not results:
            logger.warning("   Aucun résultat à re-ranker")
            return []
        
        if not self.available:
            logger.warning("   BM25 non disponible, tri par WebFilter score uniquement")
            sorted_results = sorted(
                results, 
                key=lambda x: x.get("webfilter_score", 0), 
                reverse=True
            )
            
            # LOG des scores
            logger.info(f"  Résultats triés par WebFilter score:")
            for i, r in enumerate(sorted_results[:top_k], 1):
                logger.info(f"    [{i}] Score: {r.get('webfilter_score', 0):.2f} - {r['title'][:50]}")
            
            return sorted_results[:top_k]
        
        # Créer corpus pour BM25
        corpus = []
        for r in results:
            text = f"{r['title']} {r['body']}"
            corpus.append(text.lower().split())
        
        # BM25
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        
        logger.info(f"  BM25 scores calculés:")
        logger.info(f"    Min: {min(bm25_scores):.2f}, Max: {max(bm25_scores):.2f}")
        
        # Combiner scores: 70% WebFilter + 30% BM25
        for i, result in enumerate(results):
            webfilter_score = result.get("webfilter_score", 5.0)
            bm25_score = bm25_scores[i]
            
            # Normaliser BM25 (0-10)
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            normalized_bm25 = (bm25_score / max_bm25) * 10
            
            # Score combiné
            result["bm25_score"] = normalized_bm25
            result["final_score"] = 0.7 * webfilter_score + 0.3 * normalized_bm25
            
            logger.info(f"    [{i+1}] WF: {webfilter_score:.2f}, BM25: {normalized_bm25:.2f}, Final: {result['final_score']:.2f}")
        
        # Trier par score final
        ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)
        
        logger.info(f"   Top {top_k} résultats sélectionnés:")
        for i, r in enumerate(ranked[:top_k], 1):
            logger.info(f"    [{i}] Final: {r['final_score']:.2f} - {r['title'][:60]}")
        
        return ranked[:top_k]
# ÉTAPE 8: GÉNÉRATION FINALE (StackRAG avec citations)

class StackRAGGenerator:
    """Génération de réponse finale avec sources"""
    
    def __init__(self, llm):
        self.llm = llm
        self.generator = (
            PromptTemplate.from_template(config.FINAL_GENERATION_PROMPT)
            | llm
            | StrOutputParser()
        )
    
    def generate_answer(self, 
                       original_question: str,
                       top_results: List[Dict]) -> Dict:
        """Génère réponse finale avec citations"""
        logger.info("ÉTAPE 8: Génération finale StackRAG")
        
        # Construire contexte
        context_parts = []
        sources = []
        
        for i, result in enumerate(top_results, 1):
            context_parts.append(f"[Source {i}]")
            context_parts.append(f"Title: {result['title']}")
            context_parts.append(f"Content: {result['body'][:800]}")
            context_parts.append(f"Score: {result['score']}, Answers: {result['answer_count']}")
            context_parts.append("")
            
            sources.append({
                "source_id": i,
                "title": result['title'],
                "link": result['link'],
                "score": result['score'],
                "final_score": result.get('final_score', 0)
            })
        
        context = "\n".join(context_parts)
        
        # Générer réponse
        try:
            answer = self.generator.invoke({
                "question": original_question,
                "context": context,
                "num_sources": len(top_results)
            })
            
            logger.info("Réponse générée avec succès")
            
            return {
                "answer": answer,
                "sources": sources,
                "num_sources_used": len(sources),
                "generation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f" Erreur génération: {e}")
            return {
                "answer": "Erreur lors de la génération de la réponse.",
                "sources": sources,
                "num_sources_used": 0,
                "error": str(e)
            }


# ORCHESTRATEUR COMPLET (Étapes 1-8)

class StackRAGOrchestratorFull:
    """Pipeline complet StackRAG (8 étapes)"""
    
    def __init__(self, llm):
        self.llm = llm
        
        # Étapes 1-4
        self.question_receiver = QuestionReceiver()
        self.complexity_analyzer = ComplexityAnalyzer(llm)
        self.keyword_extractor = KeywordExtractor(llm)
        self.web_filter = WebFilterReformulator(llm)
        
        # Étapes 5-8
        self.stackoverflow_searcher = StackOverflowSearcher()
        self.relevance_scorer = WebFilterRelevanceScorer(llm)
        self.bm25_reranker = BM25Reranker()
        self.answer_generator = StackRAGGenerator(llm)
    
    def process_question(self, question: str, max_results: int = 10, top_k: int = 5) -> Dict:
        """Exécute le pipeline complet"""
        logger.info("="*80)
        logger.info("DÉMARRAGE PIPELINE STACKRAG COMPLET (8 ÉTAPES)")
        logger.info("="*80)
        
        pipeline_results = {}
        
        try:
            # ===== ÉTAPES 1-4 (Préparation) =====
            pipeline_results["step1_reception"] = self.question_receiver.receive_question(question)
            
            pipeline_results["step2_complexity"] = self.complexity_analyzer.analyze_and_decompose(
                pipeline_results["step1_reception"]["original_question"]
            )
            
            questions_to_process = pipeline_results["step2_complexity"]["sub_questions"]
            pipeline_results["step3_keywords"] = self.keyword_extractor.extract_keywords(questions_to_process)
            
            pipeline_results["step4_reformulation"] = self.web_filter.reformulate_queries(
                keywords=pipeline_results["step3_keywords"]["unique_keywords"],
                original_question=pipeline_results["step1_reception"]["original_question"]
            )
            
            # ===== ÉTAPE 5: Recherche =====
            logger.info("="*80)
            all_search_results = []
            for query_config in pipeline_results["step4_reformulation"]["final_queries"]:
                results = self.stackoverflow_searcher.search_questions(query_config, max_results)
                all_search_results.extend(results)
            
            # Dédupliquer par question_id
            seen_ids = set()
            unique_results = []
            for r in all_search_results:
                if r["question_id"] not in seen_ids:
                    seen_ids.add(r["question_id"])
                    unique_results.append(r)
            
            pipeline_results["step5_search"] = {
                "total_results": len(unique_results),
                "results": unique_results
            }
            
            # ===== ÉTAPE 6: Scoring WebFilter =====
            logger.info("="*80)

            # Appeler score_results qui retourne maintenant un DICT
            scoring_result = self.relevance_scorer.score_results(
            unique_results,
            pipeline_results["step1_reception"]["original_question"]
)

# Extraire les résultats scorés ET les statistiques
            scored_results = scoring_result["scored_results"]

            pipeline_results["step6_scoring"] = {
            "scored_count": scoring_result["total_processed"],
            "average_score": scoring_result["average_score"],  
            "successful_scores": scoring_result["successful_scores"],
            "failed_scores": scoring_result["failed_scores"]
}

            logger.info(f" Stats Étape 6: Moyenne={scoring_result['average_score']:.2f}, "
            f"Succès={scoring_result['successful_scores']}, "
            f"Échecs={scoring_result['failed_scores']}")
            
            # ===== ÉTAPE 7: Re-ranking BM25 =====
            logger.info("="*80)
            top_results = self.bm25_reranker.rerank(
                scored_results,
                pipeline_results["step1_reception"]["original_question"],
                top_k
            )
            
            pipeline_results["step7_reranking"] = {
                "top_k": len(top_results),
                "results": top_results
            }
            
            # ===== ÉTAPE 8: Génération =====
            logger.info("="*80)
            final_answer = self.answer_generator.generate_answer(
                pipeline_results["step1_reception"]["original_question"],
                top_results
            )
            
            pipeline_results["step8_generation"] = final_answer
            
            logger.info("="*80)
            logger.info(" PIPELINE COMPLET TERMINÉ")
            logger.info("="*80)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f" ERREUR PIPELINE: {e}")
            import traceback
            traceback.print_exc()
            raise

def run_full_stackrag_pipeline(question: str, max_results: int = 10, top_k: int = 5) -> Dict:
    """Point d'entrée principal"""
    orchestrator = StackRAGOrchestratorFull(llm)
    return orchestrator.process_question(question, max_results, top_k)

# Test
if __name__ == "__main__":
    test_question = "How do I implement JWT authentication in Flask with refresh tokens?"
    results = run_full_stackrag_pipeline(test_question)
    
    print("\n" + "="*80)
    print(" RÉSUMÉ FINAL")
    print("="*80)
    print(f"\nRÉPONSE:\n{results['step8_generation']['answer']}\n")
    print(f"\n SOURCES ({results['step8_generation']['num_sources_used']}):")
    for src in results['step8_generation']['sources']:
        print(f"  [{src['source_id']}] {src['title']}")
        print(f"      {src['link']}")