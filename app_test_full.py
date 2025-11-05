# app_test_full.py - Interface Streamlit complète (Étapes 1-8)

import streamlit as st
import json
from stackrag_pipeline import run_full_stackrag_pipeline
import logging
from datetime import datetime

st.set_page_config(
    page_title="StackRAG Full Pipeline",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .step-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .source-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B35;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .answer-box {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #FF6B35;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔍 StackRAG - Pipeline Complet</h1>', unsafe_allow_html=True)

with st.expander("ℹ️ Architecture du Pipeline (8 Étapes)", expanded=False):
    st.markdown("""
    ### 🏗️ Pipeline StackRAG Complet avec WebFilter & RL
    
    **Phase 1: Préparation (Étapes 1-4)**
    1. **📥 Réception**: Validation & analyse basique
    2. **🔍 Complexité**: Décomposition si nécessaire (LLM)
    3. **🔑 Mots-clés**: Extraction intelligente (LLM)
    4. **🔄 WebFilter**: Reformulation avec opérateurs avancés (LLM)
    
    **Phase 2: Recherche & Filtrage (Étapes 5-7)**
    5. **🔎 Recherche**: Stack Overflow API (gratuite)
    6. **🎯 Scoring**: WebFilter avec RL pour pertinence (LLM)
    7. **📊 Re-ranking**: BM25 + scores combinés
    
    **Phase 3: Génération (Étape 8)**
    8. **✨ StackRAG**: Réponse finale avec citations (LLM)
    
    ---
    
    ### 🛠️ Technologies Utilisées (100% Gratuites)
    
    - **LLM**: Groq (llama-3.3-70b) - Ultra-rapide, gratuit
    - **API Search**: Stack Exchange API - Gratuite
    - **Re-ranking**: BM25 (rank-bm25) - Open-source
    - **Base Vector**: ChromaDB - Locale, gratuite
    - **Embeddings**: sentence-transformers - Local, gratuit
    
    ---
    
    ### 🎯 Avantages du Système
    
    ✅ **WebFilter avec RL**: Scoring intelligent de pertinence  
    ✅ **BM25 Re-ranking**: Optimisation des résultats  
    ✅ **Citations**: Chaque réponse cite ses sources  
    ✅ **Multi-requêtes**: Recherches parallèles optimisées  
    ✅ **100% Gratuit**: Aucun coût API externe  
    """)

# ===== Sidebar Configuration =====
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    max_results = st.slider(
        "📊 Résultats max par requête",
        min_value=5,
        max_value=30,
        value=10,
        help="Nombre de résultats à récupérer de Stack Overflow"
    )
    
    top_k = st.slider(
        "🏆 Top K final",
        min_value=3,
        max_value=10,
        value=5,
        help="Nombre de meilleurs résultats à utiliser pour la réponse"
    )
    
    st.markdown("---")
    st.markdown("### 💡 Exemples de questions")
    
    examples = [
        "How to implement JWT authentication in Flask?",
        "Compare React hooks vs class components",
        "Best practices for MongoDB schema design",
        "How to handle async/await in JavaScript?",
        "Implement rate limiting in Express.js"
    ]
    
    for ex in examples:
        if st.button(f"📝 {ex[:35]}...", key=ex, use_container_width=True):
            st.session_state.selected_example = ex

# ===== Main Interface =====
st.markdown("### 💬 Posez votre question technique")

# Utiliser l'exemple sélectionné si disponible
default_value = st.session_state.get('selected_example', '')

user_prompt = st.text_area(
    "Question:",
    value=default_value,
    placeholder="Ex: How do I implement authentication with JWT in a Python Flask API?",
    height=100,
    key="question_input"
)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    submit_button = st.button("🚀 Lancer le Pipeline Complet", type="primary", use_container_width=True)

with col2:
    if st.button("🗑️ Effacer", use_container_width=True):
        st.session_state.clear()
        st.rerun()

with col3:
    show_json = st.checkbox("📋 JSON", value=False)

# ===== Processing =====
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

if submit_button and user_prompt:
    with st.spinner("⏳ Exécution du pipeline complet (cela peut prendre 30-60 secondes)..."):
        try:
            results = run_full_stackrag_pipeline(
                user_prompt,
                max_results=max_results,
                top_k=top_k
            )
            st.session_state.pipeline_results = results
            st.success("✅ Pipeline complété avec succès!")
            
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ===== Results Display =====
if st.session_state.pipeline_results:
    results = st.session_state.pipeline_results
    
    st.markdown("---")
    
    # ===== RÉPONSE FINALE EN HAUT =====
    st.markdown("## ✨ Réponse StackRAG")
    
    step8 = results.get("step8_generation", {})
    answer = step8.get("answer", "Aucune réponse générée")
    
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
    
    # ===== SOURCES =====
    st.markdown("### 📚 Sources Utilisées")
    
    sources = step8.get("sources", [])
    for src in sources:
        st.markdown(f"""
        <div class="source-card">
            <strong>[Source {src['source_id']}]</strong> {src['title']}<br>
            <small>Score: {src.get('final_score', 0):.2f} | ⭐ {src['score']}</small><br>
            <a href="{src['link']}" target="_blank">🔗 Voir sur Stack Overflow</a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== MÉTRIQUES GLOBALES =====
    st.markdown("## 📊 Métriques du Pipeline")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔍</h3>
            <h2>{results.get('step5_search', {}).get('total_results', 0)}</h2>
            <p>Résultats trouvés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = results.get('step6_scoring', {}).get('average_score', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯</h3>
            <h2>{avg_score:.1f}/10</h2>
            <p>Score moyen</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏆</h3>
            <h2>{len(sources)}</h2>
            <p>Top sources</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔑</h3>
            <h2>{results.get('step3_keywords', {}).get('total_count', 0)}</h2>
            <p>Mots-clés</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== DÉTAILS PAR ÉTAPE =====
    st.markdown("## 🔍 Détails du Pipeline")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📥 Étape 1",
        "🔍 Étape 2",
        "🔑 Étape 3",
        "🔄 Étape 4",
        "🔎 Étape 5",
        "🎯 Étape 6-7",
        "✨ Étape 8",
        "📋 JSON"
    ])
    
    with tab1:
        st.markdown('<div class="step-header"><h3>📥 ÉTAPE 1: Réception</h3></div>', unsafe_allow_html=True)
        step1 = results.get("step1_reception", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mots", step1.get("word_count", 0))
        with col2:
            st.metric("Caractères", step1.get("char_count", 0))
        with col3:
            st.metric("Timestamp", step1.get("timestamp", "N/A")[:19])
        st.write(f"**Question:** {step1.get('original_question', '')}")
    
    with tab2:
        st.markdown('<div class="step-header"><h3>🔍 ÉTAPE 2: Complexité</h3></div>', unsafe_allow_html=True)
        step2 = results.get("step2_complexity", {})
        is_complex = step2.get("is_complex", False)
        if is_complex:
            st.success("✅ Question complexe - Décomposée")
        else:
            st.info("ℹ️ Question simple")
        st.markdown("**Sous-questions:**")
        for i, sq in enumerate(step2.get("sub_questions", []), 1):
            st.markdown(f"{i}. {sq}")
    
    with tab3:
        st.markdown('<div class="step-header"><h3>🔑 ÉTAPE 3: Mots-clés</h3></div>', unsafe_allow_html=True)
        step3 = results.get("step3_keywords", {})
        st.metric("Mots-clés uniques", step3.get("total_count", 0))
        keywords = step3.get("unique_keywords", [])
        st.write(", ".join([f"`{k}`" for k in keywords]))
    
    with tab4:
        st.markdown('<div class="step-header"><h3>🔄 ÉTAPE 4: WebFilter</h3></div>', unsafe_allow_html=True)
        step4 = results.get("step4_reformulation", {})
        st.metric("Requêtes générées", step4.get("query_count", 0))
        
        for i, query_obj in enumerate(step4.get("final_queries", []), 1):
            with st.expander(f"Requête {i}", expanded=False):
                st.code(query_obj.get('full_query', ''), language="text")
    
    with tab5:
        st.markdown('<div class="step-header"><h3>🔎 ÉTAPE 5: Recherche Stack Overflow</h3></div>', unsafe_allow_html=True)
        step5 = results.get("step5_search", {})
        st.metric("Résultats uniques", step5.get("total_results", 0))
        
        if st.checkbox("Afficher tous les résultats bruts", key="show_raw"):
            for i, res in enumerate(step5.get("results", [])[:10], 1):
                with st.expander(f"[{i}] {res['title']}", expanded=False):
                    st.write(f"**Score SO:** {res['score']} | **Réponses:** {res['answer_count']}")
                    st.write(f"**Tags:** {', '.join(res['tags'])}")
                    st.write(f"**Lien:** {res['link']}")
    
    with tab6:
        st.markdown('<div class="step-header"><h3>🎯 ÉTAPES 6-7: Scoring & Re-ranking</h3></div>', unsafe_allow_html=True)
        
        st.subheader("📊 Étape 6: WebFilter RL Scoring")
        step6 = results.get("step6_scoring", {})
        st.metric("Score moyen WebFilter", f"{step6.get('average_score', 0):.2f}/10")
        
        st.subheader("🏆 Étape 7: BM25 Re-ranking")
        step7 = results.get("step7_reranking", {})
        st.write(f"**Top {step7.get('top_k', 0)} résultats sélectionnés:**")
        
        for i, res in enumerate(step7.get("results", []), 1):
            with st.expander(f"[{i}] {res['title']} - Score: {res.get('final_score', 0):.2f}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("WebFilter Score", f"{res.get('webfilter_score', 0):.1f}/10")
                    st.metric("SO Score", res['score'])
                with col2:
                    st.metric("BM25 Score", f"{res.get('bm25_score', 0):.1f}/10")
                    st.metric("Final Score", f"{res.get('final_score', 0):.2f}/10")
                st.write(f"**Lien:** {res['link']}")
    
    with tab7:
        st.markdown('<div class="step-header"><h3>✨ ÉTAPE 8: Génération StackRAG</h3></div>', unsafe_allow_html=True)
        st.write(f"**Sources utilisées:** {step8.get('num_sources_used', 0)}")
        st.write(f"**Timestamp:** {step8.get('generation_timestamp', 'N/A')[:19]}")
        
        st.markdown("### 📝 Réponse complète:")
        st.markdown(answer)
    
    with tab8:
        st.markdown('<div class="step-header"><h3>📋 JSON Complet</h3></div>', unsafe_allow_html=True)
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Télécharger JSON complet",
            data=json_str,
            file_name=f"stackrag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        st.json(results)

# ===== Footer =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔍 <strong>StackRAG Pipeline</strong> - Powered by Groq, Stack Overflow API & BM25</p>
    <p><small>Toutes les technologies utilisées sont gratuites et open-source</small></p>
</div>
""", unsafe_allow_html=True)