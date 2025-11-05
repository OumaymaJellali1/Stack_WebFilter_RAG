
import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, List
import sys

# Import du pipeline
try:
    from stackrag_pipeline import (
        LLMAgentOrchestrator,
        llm,
        VectorDatabase
    )
except ImportError:
    st.error(" Impossible d'importer stackrag_pipeline.py")
    st.stop()

st.set_page_config(
    page_title="StackRAG - Live Pipeline",
    page_icon="🤖",
    layout="wide"
)

# ===== STYLES CSS =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .step-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .step-pending {
        background: #e0e0e0;
        color: #666;
    }
    .step-running {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        animation: pulse 2s infinite;
    }
    .step-completed {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-step {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #fff;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .answer-box {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #FF6B35;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B35;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .progress-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 30px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown('<h1 class="main-header">🤖 StackRAG - Pipeline en Temps Réel</h1>', unsafe_allow_html=True)

# ===== ONGLETS PRINCIPAUX =====
tab1, tab2 = st.tabs(["🤖 Pipeline StackRAG", "💾 ChromaDB Explorer"])

# ===== TAB 2: CHROMADB EXPLORER =====
with tab2:
    st.markdown("### 💾 Explorateur ChromaDB")
    
    try:
        from stackrag_pipeline import VectorDatabase
        
        vector_db = VectorDatabase()
        
        if not vector_db.available:
            st.error("❌ ChromaDB n'est pas disponible")
        else:
            # Récupérer toutes les données
            try:
                result = vector_db.collection.get()
                total_docs = len(result['ids'])
                
                st.success(f"✅ Connecté à ChromaDB - **{total_docs} documents** stockés")
                
                if total_docs > 0:
                    # Statistiques
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📄 Total Documents", total_docs)
                    
                    with col2:
                        # Compter les queries uniques
                        queries = set(m.get('query', '') for m in result['metadatas'])
                        st.metric("🔍 Queries Uniques", len(queries))
                    
                    with col3:
                        # Score moyen SO
                        avg_score = sum(int(m.get('score', 0)) for m in result['metadatas']) / total_docs
                        st.metric("⭐ Score SO Moyen", f"{avg_score:.1f}")
                    
                    st.markdown("---")
                    
                    # Afficher les documents
                    st.markdown("### 📚 Documents Stockés")
                    
                    for i, (doc_id, metadata, document) in enumerate(zip(
                        result['ids'],
                        result['metadatas'],
                        result['documents']
                    ), 1):
                        with st.expander(f"[{i}] {metadata.get('title', 'No title')}", expanded=False):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**ID:** `{doc_id}`")
                                st.markdown(f"**Query origine:** {metadata.get('query', 'N/A')}")
                                st.markdown(f"**Tags:** {metadata.get('tags', 'N/A')}")
                                st.markdown(f"**Lien:** [{metadata.get('link', 'N/A')}]({metadata.get('link', '#')})")
                            
                            with col2:
                                st.metric("Score SO", metadata.get('score', 0))
                                st.metric("Question ID", metadata.get('question_id', 'N/A'))
                            
                            st.markdown("**Contenu (preview):**")
                            st.text(document[:500] + "..." if len(document) > 500 else document)
                    
                    # Bouton pour vider la DB
                    st.markdown("---")
                    if st.button("🗑️ Vider ChromaDB", type="secondary"):
                        if st.checkbox("⚠️ Confirmer la suppression"):
                            try:
                                vector_db.client.delete_collection("stackoverflow")
                                vector_db.collection = vector_db.client.create_collection("stackoverflow")
                                st.success("✅ ChromaDB vidée avec succès!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Erreur: {e}")
                
                else:
                    st.info("ℹ️ ChromaDB est vide. Lancez une recherche dans l'onglet Pipeline pour la remplir.")
                    
            except Exception as e:
                st.error(f"❌ Erreur lecture ChromaDB: {e}")
    
    except Exception as e:
        st.error(f"❌ Erreur: {e}")

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### ⚙️ Configuration Pipeline")
    
    max_results = st.slider(
        "📊 Max résultats SO",
        min_value=5,
        max_value=30,
        value=15,
        help="Nombre max de résultats Stack Overflow par requête"
    )
    
    top_k = st.slider(
        "🏆 Top K Evidence",
        min_value=3,
        max_value=10,
        value=5,
        help="Nombre d'evidences à utiliser pour la génération"
    )
    
    st.markdown("---")
    st.markdown("### 💡 Questions Exemples")
    
    examples = [
        "How to implement JWT authentication in Flask?",
        "Compare React hooks vs class components",
        "MongoDB schema design best practices",
        "Handle async/await errors in JavaScript",
        "Rate limiting in Express.js with Redis",
        "Django REST framework authentication",
        "PostgreSQL indexing for large tables"
    ]
    
    for ex in examples:
        if st.button(f"📝 {ex[:35]}...", key=ex, use_container_width=True):
            st.session_state.selected_example = ex
            st.session_state.clear_results = True

# ===== MAIN INTERFACE =====
st.markdown("### 💬 Posez votre question technique")

default_value = st.session_state.get('selected_example', '')

user_question = st.text_area(
    "Question:",
    value=default_value,
    placeholder="Ex: How do I implement JWT authentication in Flask?",
    height=100,
    key="question_input"
)

col1, col2 = st.columns([3, 1])

with col1:
    submit_button = st.button("🚀 Lancer le Pipeline", type="primary", use_container_width=True)

with col2:
    if st.button("🗑️ Effacer", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ===== FONCTION POUR EXÉCUTER LE PIPELINE AVEC STEPS =====
def run_pipeline_with_steps(question: str, max_results: int, top_k: int):
    """Exécute le pipeline et met à jour l'UI à chaque étape"""
    
    # Conteneurs pour chaque étape
    progress_container = st.container()
    step1_container = st.container()
    step2_container = st.container()
    step3_container = st.container()
    step4_container = st.container()
    
    results = {
        "input_question": question,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Initialiser l'orchestrateur
        orchestrator = LLMAgentOrchestrator(llm)
        
        # ===== BARRE DE PROGRESSION =====
        with progress_container:
            progress_placeholder = st.empty()
            
        def update_progress(step: int, total: int = 4):
            percentage = (step / total) * 100
            progress_placeholder.markdown(f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%">
                    Étape {step}/{total} - {percentage:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ===== TOOL 1: KEYWORD EXTRACTOR =====
        update_progress(0)
        
        with step1_container:
            st.markdown("""
            <div class="step-container step-running">
                <div class="step-header">🔑 TOOL 1: Keyword Extractor</div>
                <p>⏳ Analyse de la complexité et extraction des mots-clés...</p>
            </div>
            """, unsafe_allow_html=True)
        
        tool1_output = orchestrator.keyword_extractor.process(question)
        results["tool1_keywords"] = tool1_output
        
        with step1_container:
            st.markdown(f"""
            <div class="step-container step-completed">
                <div class="step-header">✅ TOOL 1: Keyword Extractor</div>
                <div class="sub-step">
                    <strong>Complexité:</strong> {'✅ Complexe' if tool1_output['is_complex'] else '✅ Simple'}
                </div>
                <div class="sub-step">
                    <strong>Sous-questions:</strong> {len(tool1_output['sub_questions'])}
                </div>
                <div class="sub-step">
                    <strong>Keywords extraits:</strong> {', '.join(tool1_output['keywords'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if tool1_output['is_complex']:
                with st.expander("📋 Voir les sous-questions", expanded=False):
                    for i, sq in enumerate(tool1_output['sub_questions'], 1):
                        st.markdown(f"{i}. {sq}")
        
        update_progress(1)
        time.sleep(0.3)
        
        # ===== TOOL 2: SEARCH AND STORAGE =====
        with step2_container:
            st.markdown("""
            <div class="step-container step-running">
                <div class="step-header">🔎 TOOL 2: Search and Storage</div>
                <p>⏳ Reformulation WebFilter, recherche Stack Overflow et stockage...</p>
            </div>
            """, unsafe_allow_html=True)
        
        tool2_output = orchestrator.search_storage.process(
            keywords=tool1_output["keywords"],
            original_question=question,
            max_results=max_results
        )
        results["tool2_search"] = tool2_output
        
        with step2_container:
            # Détection si SO a échoué
            so_results = tool2_output['total_results']
            so_warning = ""
            if so_results == 0:
                so_warning = " ⚠️ (Aucun nouveau résultat)"
            
            st.markdown(f"""
            <div class="step-container step-completed">
                <div class="step-header">✅ TOOL 2: Search and Storage</div>
                <div class="sub-step">
                    <strong>Requêtes WebFilter générées:</strong> {len(tool2_output['queries'])}
                </div>
                <div class="sub-step">
                    <strong>🆕 Nouveaux résultats Stack Overflow:</strong> {so_results}{so_warning}
                </div>
                <div class="sub-step">
                    <strong>Stocké en Vector DB:</strong> {'✅ Oui' if tool2_output['stored_in_db'] else '❌ Non'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Warning si aucun résultat SO
            if so_results == 0:
                st.warning("⚠️ **Aucun nouveau résultat de Stack Overflow.** Le pipeline va utiliser le cache Vector DB (résultats de recherches précédentes).")
                with st.expander("🔧 Pourquoi 0 résultats ?"):
                    st.markdown("""
                    **Causes possibles :**
                    - 🚫 Rate limiting API Stack Overflow
                    - 🔑 API Key manquante/invalide
                    - 🌐 Problème de connexion réseau
                    - 🔍 Mots-clés trop spécifiques (aucune correspondance)
                    
                    **Solutions :**
                    - Vérifier la variable `STACKOVERFLOW_API_KEY` dans `.env`
                    - Attendre quelques minutes (rate limit)
                    - Reformuler la question avec des termes plus généraux
                    """)
            
            with st.expander("🔍 Détails des requêtes WebFilter", expanded=False):
                for i, query in enumerate(tool2_output['queries'], 1):
                    st.markdown(f"**Requête {i}:** `{query['base_query']}`")
                    st.json(query['search_params'])
        
        update_progress(2)
        time.sleep(0.3)
        
        # ===== TOOL 3: GATHER EVIDENCE =====
        with step3_container:
            st.markdown("""
            <div class="step-container step-running">
                <div class="step-header">📊 TOOL 3: Gather Evidence</div>
                <p>⏳ Recherche Vector DB, scoring LLM et re-ranking BM25...</p>
            </div>
            """, unsafe_allow_html=True)
        
        tool3_output = orchestrator.gather_evidence.process(
            original_question=question,
            so_results=tool2_output["results"],
            top_k=top_k
        )
        results["tool3_evidence"] = tool3_output
        
        with step3_container:
            avg_score = tool3_output.get('average_score', 0)
            evidence_status = tool3_output.get('evidence_status', 'unknown')
            vector_results = len(tool3_output['vector_results'])
            
            # Calculer origine des résultats
            so_results_count = tool2_output['total_results']
            cache_used = vector_results > 0 and so_results_count == 0
            
            origin_text = ""
            if cache_used:
                origin_text = " 💾 (depuis cache)"
            elif so_results_count > 0:
                origin_text = f" 🆕 ({so_results_count} nouveaux + {vector_results} cache)"
            
            st.markdown(f"""
            <div class="step-container step-completed">
                <div class="step-header">✅ TOOL 3: Gather Evidence</div>
                <div class="sub-step">
                    <strong>📦 Résultats Vector DB (cache):</strong> {vector_results}{origin_text}
                </div>
                <div class="sub-step">
                    <strong>Score moyen:</strong> {avg_score:.2f}/10
                </div>
                <div class="sub-step">
                    <strong>Evidence status:</strong> {evidence_status.upper()} {'✅' if evidence_status == 'sufficient' else '⚠️'}
                </div>
                <div class="sub-step">
                    <strong>Top K sélectionnés:</strong> {len(tool3_output['top_k_results'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Info si uniquement cache utilisé
            if cache_used:
                st.info("ℹ️ **Résultats provenant uniquement du cache Vector DB** (recherches précédentes). Aucun nouveau résultat Stack Overflow n'a été trouvé.")
            
            with st.expander("📈 Top K Results avec scores", expanded=False):
                for i, res in enumerate(tool3_output['top_k_results'], 1):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("WebFilter", f"{res.get('webfilter_score', 0):.1f}")
                    with col2:
                        st.metric("BM25", f"{res.get('bm25_score', 0):.1f}")
                    with col3:
                        st.metric("Final", f"{res.get('final_score', 0):.2f}")
                    with col4:
                        st.metric("SO Score", res.get('score', 0))
                    
                    st.markdown(f"**{i}. {res['title']}**")
                    st.markdown(f"[🔗 Lien]({res['link']})")
                    st.markdown("---")
        
        update_progress(3)
        time.sleep(0.3)
        
        # ===== TOOL 4: ANSWER GENERATOR =====
        with step4_container:
            st.markdown("""
            <div class="step-container step-running">
                <div class="step-header">✨ TOOL 4: Answer Generator</div>
                <p>⏳ Génération de la réponse finale avec citations...</p>
            </div>
            """, unsafe_allow_html=True)
        
        tool4_output = orchestrator.answer_generator.process(
            original_question=question,
            evidences=tool3_output["top_k_results"]
        )
        results["tool4_answer"] = tool4_output
        
        with step4_container:
            st.markdown(f"""
            <div class="step-container step-completed">
                <div class="step-header">✅ TOOL 4: Answer Generator</div>
                <div class="sub-step">
                    <strong>Sources utilisées:</strong> {tool4_output['num_sources']}
                </div>
                <div class="sub-step">
                    <strong>Réponse générée:</strong> ✅ Complète
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        update_progress(4)
        
        return results
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'exécution: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ===== EXÉCUTION DU PIPELINE =====
if submit_button and user_question:
    st.markdown("---")
    st.markdown("## 🔄 Exécution du Pipeline")
    
    results = run_pipeline_with_steps(user_question, max_results, top_k)
    
    if results:
        st.session_state.pipeline_results = results
        
        # ===== AFFICHAGE FINAL =====
        st.markdown("---")
        st.markdown("## ✨ Résultat Final")
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        tool1 = results.get("tool1_keywords", {})
        tool2 = results.get("tool2_search", {})
        tool3 = results.get("tool3_evidence", {})
        tool4 = results.get("tool4_answer", {})
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🔑</h3>
                <h2>{}</h2>
                <p>Keywords</p>
            </div>
            """.format(len(tool1.get('keywords', []))), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔍</h3>
                <h2>{}</h2>
                <p>SO Results</p>
            </div>
            """.format(tool2.get('total_results', 0)), unsafe_allow_html=True)
        
        with col3:
            avg_score = tool3.get('average_score', 0)
            st.markdown("""
            <div class="metric-card">
                <h3>🎯</h3>
                <h2>{:.1f}/10</h2>
                <p>Avg Score</p>
            </div>
            """.format(avg_score), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>📚</h3>
                <h2>{}</h2>
                <p>Sources</p>
            </div>
            """.format(tool4.get('num_sources', 0)), unsafe_allow_html=True)
        
        # Réponse finale
        st.markdown("### 📝 Réponse Générée")
        answer = tool4.get("answer", "Aucune réponse générée")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        
        # Sources
        st.markdown("### 📚 Sources Utilisées")
        sources = tool4.get("sources", [])
        if sources:
            for src in sources:
                st.markdown(f"""
                <div class="source-card">
                    <strong>[Source {src['source_id']}]</strong> {src['title']}<br>
                    <small>Score final: {src.get('final_score', 0):.2f}/10 | ⭐ SO Score: {src.get('score', 0)}</small><br>
                    <a href="{src['link']}" target="_blank">🔗 Voir sur Stack Overflow</a>
                </div>
                """, unsafe_allow_html=True)
        
        # Export JSON
        st.markdown("---")
        with st.expander("📋 Export JSON Complet"):
            json_str = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 Télécharger JSON",
                data=json_str,
                file_name=f"stackrag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.json(results)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 <strong>StackRAG - Pipeline en Temps Réel</strong></p>
    <p><small>Affichage étape par étape avec progression visuelle</small></p>
    <p><small>4 outils spécialisés | 100% Gratuit | Groq + SO API + ChromaDB + BM25</small></p>
</div>
""", unsafe_allow_html=True)