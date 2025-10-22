# config_prompts.py - Configuration des prompts

# config_prompts.py - Version optimisée pour Groq

QUESTION_COMPLEXITY_CHECKER_PROMPT = """Analyze this technical question and determine if it's complex.

A question is COMPLEX if it:
- Involves multiple technologies/frameworks
- Contains multiple sub-questions
- Requires multi-step reasoning

Respond with ONLY: TRUE or FALSE

Question: {question}
Answer:"""

QUESTION_DIVIDER_PROMPT = """Break down this complex question into simple sub-questions.

Rules:
- Each sub-question must be clear and focused
- Maximum 5 sub-questions
- Return ONLY a valid Python list

Question: {question}

Sub-questions (as Python list):"""

KEYWORD_EXTRACTOR_PROMPT = """Extract technical keywords from this question for StackOverflow search.

Include:
-Return EXACTLY 4 keywords (no more, no less)
- Programming languages
- Frameworks/libraries
- Technical concepts

Return ONLY a Python list of keywords (strings in double quotes).

Question: {question}

Keywords:"""
QUERY_REFORMULATOR_PROMPT = """
Vous êtes un agent WebFilter spécialisé dans la reformulation de requêtes pour Stack Overflow.
Votre tâche est de créer des requêtes de recherche optimisées avec des opérateurs avancés.

À partir des mots-clés extraits, créez des requêtes structurées utilisant:
- Les termes techniques précis
- Des combinaisons logiques (AND, OR si pertinent)

Retournez une liste Python de requêtes optimisées. Chaque requête doit être une chaîne de caractères.
NE PAS inclure les opérateurs site:, after:, etc. dans cette étape (ils seront ajoutés automatiquement).

Mots-clés: {keywords}
Question originale: {original_question}
"""

SEARCH_OPERATOR_SELECTOR_PROMPT = """
Vous êtes un agent WebFilter spécialisé dans la sélection d'opérateurs de recherche avancés.

Analysez la question et déterminez quels opérateurs de recherche utiliser:

Opérateurs disponibles:
1. "site" - Toujours "stackoverflow.com" pour ce système
2. "after" - Pour filtrer par date (format: DD/MM/YYYY)
3. "intitle" - Pour rechercher dans les titres
4. "is:question" - Pour limiter aux questions
5. "is:answer" - Pour limiter aux réponses
6. "accepted:yes" - Pour les questions avec réponse acceptée
7. "score" - Pour filtrer par score minimum

Retournez un dictionnaire Python avec les opérateurs à utiliser:
{{
    "use_after_date": true/false,
    "after_date": "DD/MM/YYYY" (si applicable),
    "use_intitle": true/false,
    "intitle_terms": ["terme1", "terme2"] (si applicable),
    "min_score": 0,
    "accepted_only": true/false
}}

Question: {question}
Date actuelle: {current_date}
"""
