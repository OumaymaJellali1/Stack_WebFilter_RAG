
import streamlit as st
import json
from stackrag_pipeline import run_stackrag_pipeline
import logging

st.set_page_config(
    page_title="StackRAG Pipeline Test",
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
    }
    .step-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔍 StackRAG Pipeline - Test (Étapes 1-4)</h1>', unsafe_allow_html=True)

with st.expander(" À propos du pipeline", expanded=False):
    st.markdown("""
    ### Pipeline StackRAG avec WebFilter
    
    Ce système traite votre question en 4 étapes:
    
    1. **Réception**: Validation et préparation
    2. ** Analyse**: Décomposition si nécessaire
    3. ** Mots-clés**: Extraction intelligente
    4. ** WebFilter**: Requêtes optimisées
    """)

st.markdown("###  Posez votre question technique")
user_prompt = st.text_area(
    "Entrez votre question:",
    placeholder="Ex: Comment implémenter une API REST avec JWT en Python?",
    height=100
)

col1, col2 = st.columns([1, 1])

with col1:
    submit_button = st.button(" Lancer le pipeline", type="primary")

with col2:
    if st.button(" Effacer"):
        st.rerun()

if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

if submit_button and user_prompt:
    with st.spinner(" Traitement en cours..."):
        try:
            results = run_stackrag_pipeline(user_prompt)
            st.session_state.pipeline_results = results
            st.success("Pipeline complété avec succès!")
            
        except Exception as e:
            st.error(f" Erreur: {str(e)}")

if st.session_state.pipeline_results:
    results = st.session_state.pipeline_results
    
    st.markdown("---")
    st.markdown("## Résultats du Pipeline")
    
    tab1, tab2, tab3, tab4, tab_all = st.tabs([
        " Étape 1",
        " Étape 2",
        " Étape 3",
        " Étape 4",
        " JSON"
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
            st.success(" Question complexe - Décomposée")
        else:
            st.info(" Question simple")
        st.markdown("**Sous-questions:**")
        for i, sq in enumerate(step2.get("sub_questions", []), 1):
            st.markdown(f"{i}. {sq}")
    
    with tab3:
        st.markdown('<div class="step-header"><h3> ÉTAPE 3: Mots-clés</h3></div>', unsafe_allow_html=True)
        step3 = results.get("step3_keywords", {})
        st.metric("Mots-clés uniques", step3.get("total_count", 0))
        keywords = step3.get("unique_keywords", [])
        st.write(", ".join([f"`{k}`" for k in keywords]))
    
    with tab4:
        st.markdown('<div class="step-header"><h3> ÉTAPE 4: WebFilter</h3></div>', unsafe_allow_html=True)
        step4 = results.get("step4_reformulation", {})
        st.metric("Requêtes générées", step4.get("query_count", 0))
        
        st.markdown("**Requêtes finales:**")
        for i, query_obj in enumerate(step4.get("final_queries", []), 1):
            with st.expander(f"Requête {i}", expanded=True):
                st.code(query_obj.get('full_query', ''), language="text")
    
    with tab_all:
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label=" Télécharger JSON",
            data=json_str,
            file_name="stackrag_results.json",
            mime="application/json"
        )
        st.json(results)

with st.sidebar:
    st.markdown("###  Exemples")
    examples = [
        "How to sort a list in Python?",
        "Implement JWT authentication in Flask",
        "Compare React hooks vs Vue composition API"
    ]
    for ex in examples:
        if st.button(f" {ex[:30]}...", key=ex):
            st.session_state.example = ex
            st.rerun()
