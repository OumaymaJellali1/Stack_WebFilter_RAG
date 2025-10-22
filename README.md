# 🔍 StackRAG Pipeline

Pipeline intelligent de Q&A avec Stack Overflow (Étapes 1-4)

## 🚀 Installation

Tout a été créé automatiquement par le script d'installation!

## 🎯 Utilisation

### 1. Installer Ollama (recommandé)

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2

# Mac
brew install ollama
ollama pull llama2

# Windows: Télécharger depuis https://ollama.ai/download
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l'application

```bash
# Terminal 1: Démarrer Ollama
ollama serve

# Terminal 2: Lancer Streamlit
streamlit run app_test.py
```

## 📋 Structure

- `config_prompts.py` - Configuration des prompts
- `stackrag_pipeline.py` - Pipeline principal (Étapes 1-4)
- `app_test.py` - Interface Streamlit
- `requirements.txt` - Dépendances
- `.env.example` - Configuration

## 🧪 Test rapide

```python
from stackrag_pipeline import run_stackrag_pipeline

results = run_stackrag_pipeline("How to sort a list in Python?")
print(results)
```

## 📊 Architecture

1. **Réception** - Validation de la question
2. **Analyse** - Décomposition si complexe (StackRAG)
3. **Mots-clés** - Extraction intelligente
4. **WebFilter** - Requêtes avec opérateurs avancés

## 🔮 Prochaines étapes

- Étape 5: Stack Overflow API
- Étape 6: Ranking BM25
- Étape 7: Génération réponse

## 📝 Licence

MIT
