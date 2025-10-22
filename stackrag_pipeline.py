# stackrag_pipeline.py - Pipeline avec Groq API (LangChain moderne)
import os
import sys
import json
import ast
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import logging

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config_prompts as config

# ============================================
# Configuration du logging
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Charger les variables d'environnement
# ============================================
load_dotenv()

# ============================================
# Fonction utilitaire pour parser la sortie LLM
# ============================================
def safe_parse_llm_output(text: str):
    text = text.strip()
    if not text:
        return []
    
    # Nettoyer les balises markdown
    if text.startswith("```") and text.endswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    
    # Nettoyer les balises json/python
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

# ============================================
# Initialisation du modèle Groq
# ============================================
try:
    from langchain_groq import ChatGroq
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY manquante dans .env")
        logger.info("📝 Obtenez votre clé gratuite sur: https://console.groq.com")
        sys.exit(1)
    
    logger.info("⚡ Initialisation de Groq API (llama-3.3-70b-versatile)")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY,
        max_tokens=1024
    )
    
    logger.info("✅ LLM Groq prêt (ultra-rapide + gratuit)")

except ImportError:
    logger.error("❌ langchain-groq non installé")
    logger.info("📦 Installez avec: pip install langchain-groq langchain-core")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Erreur d'initialisation Groq: {e}")
    sys.exit(1)

# ============================================
# ÉTAPE 1: Réception de la question
# ============================================
class QuestionReceiver:
    @staticmethod
    def receive_question(question: str) -> Dict:
        logger.info("📥 ÉTAPE 1: Réception de la question")
        if not question.strip():
            raise ValueError("La question ne peut pas être vide")
        question_data = {
            "original_question": question.strip(),
            "timestamp": datetime.now().isoformat(),
            "word_count": len(question.split()),
            "char_count": len(question)
        }
        logger.info(f"✅ Question reçue: {question_data['word_count']} mots")
        return question_data

# ============================================
# ÉTAPE 2: Analyse de la complexité
# ============================================
class ComplexityAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        # Utilisation moderne de LangChain (LCEL - LangChain Expression Language)
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
        logger.info("🔍 ÉTAPE 2: Analyse de la complexité")
        
        # Invocation moderne
        complexity_result = self.complexity_checker.invoke({"question": question})
        complexity_text = complexity_result.strip().upper()
        is_complex = "TRUE" in complexity_text or "COMPLEX" in complexity_text
        
        logger.info(f"Question complexe: {is_complex}")

        result = {
            "is_complex": is_complex,
            "original_question": question,
            "sub_questions": []
        }

        if is_complex:
            logger.info("📋 Décomposition en sous-questions...")
            decomposition_result = self.question_divider.invoke({"question": question})
            try:
                sub_questions = safe_parse_llm_output(decomposition_result)
                if isinstance(sub_questions, list):
                    result["sub_questions"] = sub_questions
                    logger.info(f"✅ {len(sub_questions)} sous-questions créées")
                else:
                    result["sub_questions"] = [question]
            except:
                logger.warning("⚠️ Échec de la décomposition")
                result["sub_questions"] = [question]
        else:
            result["sub_questions"] = [question]

        return result

# ============================================
# ÉTAPE 3: Extraction des mots-clés
# ============================================
class KeywordExtractor:
    def __init__(self, llm):
        self.llm = llm
        self.extractor = (
            PromptTemplate.from_template(config.KEYWORD_EXTRACTOR_PROMPT)
            | llm
            | StrOutputParser()
        )

    def extract_keywords(self, questions: List[str]) -> Dict:
        logger.info("🔑 ÉTAPE 3: Extraction des mots-clés")
        keywords_map = {}
        all_keywords = []
        
        for i, question in enumerate(questions):
            logger.info(f"Extraction pour la question {i+1}/{len(questions)}")
            try:
                result = self.extractor.invoke({"question": question})
                keywords = safe_parse_llm_output(result)
                if not isinstance(keywords, list):
                    keywords = self._simple_keyword_extraction(question)
                keywords_map[question] = keywords
                all_keywords.extend(keywords)
                logger.info(f"✅ {len(keywords)} mots-clés extraits")
            except Exception as e:
                logger.error(f"❌ Erreur: {e}")
                keywords = self._simple_keyword_extraction(question)
                keywords_map[question] = keywords
                all_keywords.extend(keywords)
        
        unique_keywords = list(set(all_keywords))
        return {
            "keywords_by_question": keywords_map,
            "all_keywords": all_keywords,
            "unique_keywords": unique_keywords,
            "total_count": len(unique_keywords)
        }

    def _simple_keyword_extraction(self, question: str) -> List[str]:
        stop_words = {'how','what','why','when','where','can','i','to','the','a','an','in','on','is','are','do','does','with','using'}
        words = question.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:4]  # 4 mots-clés par concept

# ============================================
# ÉTAPE 4: Reformulation avec WebFilter
# ============================================
class WebFilterReformulator:
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
        logger.info("🔄 ÉTAPE 4: Reformulation avec WebFilter")
        logger.info("📝 4.1 - Création des requêtes de base...")
        
        try:
            reformulated = self.query_reformulator.invoke({
                "keywords": str(keywords),
                "original_question": original_question
            })
            base_queries = safe_parse_llm_output(reformulated)
            if not isinstance(base_queries, list):
                base_queries = [" ".join(keywords[:4])]
            logger.info(f"✅ {len(base_queries)} requêtes créées")
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            base_queries = [" ".join(keywords[:4])]

        logger.info("🔧 4.2 - Sélection des opérateurs...")
        try:
            operator_result = self.operator_selector.invoke({
                "question": original_question,
                "current_date": datetime.now().strftime("%d/%m/%Y")
            })
            operators = safe_parse_llm_output(operator_result)
            if not isinstance(operators, dict):
                operators = {
                    "use_after_date": False,
                    "use_intitle": False,
                    "min_score": 0,
                    "accepted_only": False
                }
            logger.info(f"✅ Opérateurs sélectionnés")
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            operators = {
                "use_after_date": False,
                "use_intitle": False,
                "min_score": 0,
                "accepted_only": False
            }

        logger.info("🏗️ 4.3 - Construction des requêtes finales...")
        final_queries = []
        for query in base_queries:
            enhanced_query = self._build_enhanced_query(query, operators)
            final_queries.append(enhanced_query)

        return {
            "base_queries": base_queries,
            "operators": operators,
            "final_queries": final_queries,
            "query_count": len(final_queries)
        }

    def _build_enhanced_query(self, base_query: str, operators: Dict) -> Dict:
        query_components = {
            "base_query": base_query,
            "site": "stackoverflow.com",
            "filters": []
        }
        
        if operators.get("use_after_date") and operators.get("after_date"):
            query_components["filters"].append(f"after:{operators['after_date']}")
        if operators.get("use_intitle") and operators.get("intitle_terms"):
            intitle = " ".join(operators["intitle_terms"])
            query_components["filters"].append(f"intitle:{intitle}")
        if operators.get("min_score", 0) > 0:
            query_components["filters"].append(f"score:>={operators['min_score']}")
        if operators.get("accepted_only"):
            query_components["filters"].append("accepted:yes")
        
        full_query = base_query
        if query_components["filters"]:
            full_query += " " + " ".join(query_components["filters"])
        query_components["full_query"] = full_query
        
        return query_components

# ============================================
# ORCHESTRATEUR PRINCIPAL
# ============================================
class StackRAGOrchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.question_receiver = QuestionReceiver()
        self.complexity_analyzer = ComplexityAnalyzer(llm)
        self.keyword_extractor = KeywordExtractor(llm)
        self.web_filter = WebFilterReformulator(llm)

    def process_question(self, question: str) -> Dict:
        logger.info("="*80)
        logger.info("🚀 DÉMARRAGE DU PIPELINE STACKRAG")
        logger.info("="*80)
        
        pipeline_results = {}
        
        try:
            # Étape 1
            pipeline_results["step1_reception"] = self.question_receiver.receive_question(question)
            
            # Étape 2
            pipeline_results["step2_complexity"] = self.complexity_analyzer.analyze_and_decompose(
                pipeline_results["step1_reception"]["original_question"]
            )
            
            # Étape 3
            questions_to_process = pipeline_results["step2_complexity"]["sub_questions"]
            pipeline_results["step3_keywords"] = self.keyword_extractor.extract_keywords(questions_to_process)
            
            # Étape 4
            pipeline_results["step4_reformulation"] = self.web_filter.reformulate_queries(
                keywords=pipeline_results["step3_keywords"]["unique_keywords"],
                original_question=pipeline_results["step1_reception"]["original_question"]
            )
            
            logger.info("="*80)
            logger.info("✅ PIPELINE COMPLÉTÉ AVEC SUCCÈS")
            logger.info("="*80)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ ERREUR PIPELINE: {e}")
            raise

def run_stackrag_pipeline(question: str) -> Dict:
    orchestrator = StackRAGOrchestrator(llm)
    return orchestrator.process_question(question)

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    test_question = "How do I implement a REST API with authentication using Flask and JWT tokens in Python?"
    results = run_stackrag_pipeline(test_question)
    print(json.dumps(results, indent=2, ensure_ascii=False))