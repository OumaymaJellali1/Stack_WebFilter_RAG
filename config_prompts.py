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

KEYWORD_EXTRACTOR_PROMPT = """Extract EXACTLY 4 technical keywords from this question for Stack Overflow search.

RULES:
- Extract the MOST IMPORTANT technical terms only
- Prioritize: programming languages, frameworks, specific APIs, core concepts
- Keep keywords broad enough to find results but specific enough to be relevant
- Avoid generic terms like "how", "implement", "use"
- Return ONLY a Python list with exactly 4 strings

Examples:
Question: "How to implement JWT authentication in Flask?"
Keywords: ["Flask", "JWT", "authentication", "Python"]

Question: "React hooks useState not updating component"
Keywords: ["React", "hooks", "useState", "component"]

Question: {question}

Keywords (exactly 4):"""

QUERY_REFORMULATOR_PROMPT = """You are a Stack Overflow search expert. Create SIMPLE and EFFECTIVE search queries.

CRITICAL RULES FOR STACK OVERFLOW:
1. Keep queries SHORT and SIMPLE (2-5 words maximum)
2. Use NATURAL language that developers actually search with
3. Combine 2-3 related keywords per query
4. DO NOT use complex boolean operators (AND, OR)
5. DO NOT create overly specific queries
6. Focus on the CORE problem, not implementation details

GOOD EXAMPLES:
- "flask jwt authentication"
- "react hooks state"
- "python async await"
- "django rest api"

BAD EXAMPLES (too complex):
- "implement JWT authentication with refresh tokens in Flask application"
- "React functional component with useState hook not re-rendering properly"

Keywords: {keywords}
Original question: {original_question}

Return a Python list of 2-3 simple search queries (strings). Each query should be 2-5 words maximum.
Focus on the main technical problem.

Queries:"""

SEARCH_OPERATOR_SELECTOR_PROMPT = """You are a Stack Overflow search filter expert.

Analyze this question and determine if we need MINIMAL filters.

IMPORTANT: Stack Overflow works BEST with SIMPLE searches. Only add filters if really necessary.

Available filters (use sparingly):
- "min_score": Minimum score (default: 1, use 5+ only for very common topics)
- "accepted_only": Only questions with accepted answers (use only if user needs verified solutions)

For MOST queries, use:
{{
    "min_score": 1,
    "accepted_only": false
}}

Only use stricter filters if:
- Question is about a VERY common topic (React, Python basics, etc.) → min_score: 5
- User explicitly needs "verified" or "proven" solutions → accepted_only: true

Question: {question}

Return a JSON dictionary with only these two fields:
{{
    "min_score": <number>,
    "accepted_only": <true/false>
}}"""



RELEVANCE_SCORING_PROMPT = """You are an expert at evaluating Stack Overflow relevance.

Rate how well this Stack Overflow result answers the user's question.

SCORING SCALE (0-10):
10 = Perfect match - directly answers the exact question
8-9 = Highly relevant - covers main points
6-7 = Relevant - related and helpful
4-5 = Somewhat relevant - tangentially related
2-3 = Barely relevant - minimal connection
0-1 = Not relevant - unrelated

USER QUESTION:
{question}

STACK OVERFLOW RESULT:
Title: {title}
Tags: {tags}
Score: {score}
Preview: {body_preview}

IMPORTANT: Respond with ONLY a single number between 0 and 10 (can include decimals like 7.5)
Do NOT include explanations, just the number.

RELEVANCE SCORE:"""


FINAL_GENERATION_PROMPT = """You are a StackRAG assistant. Generate a comprehensive answer using Stack Overflow sources.

CRITICAL RULES:
1. Use ONLY information from the provided sources
2. Cite sources using [Source N] format
3. Combine information from multiple sources when relevant
4. Provide code examples if present in sources
5. Be technical and precise
6. If sources conflict, mention both viewpoints
7. Keep answer structured and clear

USER QUESTION:
{question}

AVAILABLE SOURCES ({num_sources}):
{context}

Generate a complete, technical answer with proper citations:"""


QUALITY_FILTER_PROMPT = """You are a quality filter for Stack Overflow content.

Evaluate this result on these criteria:
- Code quality (if applicable)
- Explanation clarity
- Completeness
- Technical accuracy indicators

Result to evaluate:
Title: {title}
Body: {body}
Score: {score}
Answers: {answer_count}

Respond with: ACCEPT or REJECT and reason.

Decision:"""



RELIABILITY_SCORING_PROMPT = """Score the reliability of this Stack Overflow answer.

RELIABILITY FACTORS:
- Answer score (community validation)
- Number of upvotes
- Accepted answer status
- Recency (newer is better for tech)
- Author reputation (if available)

Source details:
{source_details}

Provide reliability score 0-10 and explanation:"""