"""
question_generator.py  —  100% local, zero API keys, zero cost.

Uses only:
  - Python standard library (re, random, collections)
  - scikit-learn (already in your project for the classifier)

How it works:
  1. Splits text into meaningful sentences
  2. Scores sentences by TF-IDF importance (finds the most content-rich ones)
  3. Extracts key terms (nouns, definitions, comparisons) via regex patterns
  4. Applies 6 question templates per sentence type:
       - Definition  → "What is X?"
       - Purpose     → "What is the purpose of X?"
       - Comparison  → "How does X differ from Y?"
       - Process     → "How does X work?"
       - Fill-blank  → "_____ is used for Y."
       - True/False  → derived from factual sentences
  5. Assigns Bloom's level and difficulty by sentence complexity score
"""

import re
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Bloom's taxonomy keyword signals ─────────────────────────────────────────

BLOOM_SIGNALS = {
    "Remember (L1)":  ["define", "what is", "list", "name", "identify", "recall", "state"],
    "Understand (L2)":["explain", "describe", "summarise", "classify", "interpret", "means"],
    "Apply (L3)":     ["use", "apply", "implement", "demonstrate", "calculate", "perform", "execute"],
    "Analyze (L4)":   ["analyze", "compare", "contrast", "distinguish", "examine", "why", "how does"],
    "Evaluate (L5)":  ["evaluate", "assess", "justify", "argue", "critique", "advantage", "disadvantage"],
    "Create (L6)":    ["design", "develop", "build", "propose", "construct", "generate", "plan"],
}

BLOOM_ORDER = ["Remember (L1)", "Understand (L2)", "Apply (L3)",
               "Analyze (L4)", "Evaluate (L5)", "Create (L6)"]


def detect_bloom(sentence: str) -> str:
    sl = sentence.lower()
    for level in reversed(BLOOM_ORDER):
        if any(kw in sl for kw in BLOOM_SIGNALS[level]):
            return level
    # Fall back by sentence length — longer sentences tend to be higher order
    words = len(sentence.split())
    if words > 30: return "Analyze (L4)"
    if words > 20: return "Understand (L2)"
    return "Remember (L1)"


# ── Difficulty scoring ───────────────────────────────────────────────────────

def score_difficulty(sentence: str) -> str:
    """Score based on sentence length, subordinate clauses, and jargon density."""
    words = sentence.split()
    length_score = len(words)
    clause_score = sentence.count(",") + sentence.count(";") + sentence.count("(")
    # Count capitalised technical terms (likely jargon)
    jargon_score = sum(1 for w in words if len(w) > 8)
    total = length_score * 0.4 + clause_score * 3 + jargon_score * 2
    if total < 20: return "easy"
    if total < 40: return "medium"
    return "hard"


# ── Sentence cleaning & splitting ───────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip slide numbers, bullet counters, REVIEW labels, emoji, and section headers."""
    # Remove REVIEW-N labels
    text = re.sub(r'\bREVIEW-\d+\b', '', text, flags=re.IGNORECASE)
    # Remove emoji number bullets (e.g. 1️⃣)
    text = re.sub(r'\d+\ufe0f\u20e3', '', text)
    # Strip trailing digit clusters that appear after sentences (slide cross-refs)
    # Matches: "...system. 9 10 1 Next" or "...system.1 Next" 
    text = re.sub(r'([a-z.!?])[.\s]*(\d{1,2}\s*){1,6}(?=[A-Z]|\s*$)', r'\1 ', text)
    # Remove ". 4." style patterns
    text = re.sub(r'\.\s*\d{1,2}\s*\.', '.', text)
    # Remove known section header prefixes
    headers = (r'Objectives|Problem statement|Expected results|Future work|'
               r'Existing solutions|Gaps in existing solutions|Literature Survey|'
               r'Proposed work|Abstract|Conclusion|References|Technologies Used')
    text = re.sub(rf'\b({headers})\s*:\s*', ' ', text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """Split text into clean, useful sentences."""
    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Pre-split on em-dash / en-dash patterns used in PPTX bullets like "Term – definition"
    # Turn "Heading – body text" into "Heading. Body text." so each becomes its own sentence
    text = re.sub(r'\s+[–—]\s+', '. ', text)

    # Split on sentence-ending punctuation followed by a capital
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    NOISE_PATTERNS = [
        re.compile(r'\b(professor|assistant professor|department|college|university|guide|submitted by|prepared by|roll no|sem\b|aiml|cse\b)', re.IGNORECASE),
        re.compile(r'\b(rv college|bangalore|karnataka|rvce)\b', re.IGNORECASE),
        re.compile(r'^\s*\d+[\s\d]*$'),  # pure number lines
    ]

    # Strip common PPTX slide heading prefixes fused to sentence body
    HEADING_PREFIXES = re.compile(
        r'^(Adaptive Assessment Mechanism|Difficulty Level Classification|'
        r'Intelligent Feedback System|No Adaptive Feedback Mechanism|'
        r'Absence of Difficulty Validation|Traditional Question Paper Preparation|'
        r'Student Performance Analysis|Limited Intelligence|Partial'
        r'[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)\s+',
        re.IGNORECASE
    )

    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < 25:
            continue
        if any(p.search(s) for p in NOISE_PATTERNS):
            continue
        # Strip fused slide-heading prefix — heading is typically 2-5 title-case words
        # followed immediately by a new sentence starting with a capital+lowercase
        s = re.sub(
            r'^(?:[A-Z][A-Za-z]+\s+){2,5}(?=[A-Z][a-z])',
            '',
            s
        ).strip()
        if len(s) < 25:
            continue
        # Further split on mid-sentence verb boundary
        parts = re.split(
            r'(?<=[a-z])\s+(?=[A-Z][a-z]{2,}\s+(?:is|are|was|were|has|have|does|do|can|will|should|allows|enables|provides|uses|used|generate|perform|helps|help)\b)',
            s
        )
        for part in parts:
            part = part.strip()
            if len(part) >= 25 and not any(p.search(part) for p in NOISE_PATTERNS):
                sentences.append(part)

    return sentences


# ── Key term extraction ──────────────────────────────────────────────────────

def extract_subject(sentence: str) -> str:
    """Extract the most likely subject/topic of a sentence."""
    # Strip leading connectors and numbering noise
    sentence = re.sub(r'^(Hence|Therefore|Thus|In this|As a result|Based on this)[,\s]+', '', sentence, flags=re.IGNORECASE).strip()
    sentence = re.sub(r'^\d+[.)]\s*', '', sentence).strip()

    # Common English verbs that should never be a subject
    BAD_SUBJECTS = {
        'performs', 'provides', 'uses', 'allows', 'enables', 'helps', 'helps',
        'implements', 'applies', 'generates', 'handles', 'stores', 'supports',
        'integrates', 'manages', 'processes', 'returns', 'creates', 'builds',
        'loads', 'runs', 'sends', 'receives', 'reads', 'writes', 'checks',
        'validates', 'trains', 'classifies', 'extracts', 'converts', 'displays',
    }

    def _is_clean(s: str) -> bool:
        generic = {"the", "this", "these", "it", "its", "in", "by", "for",
                   "traditional", "existing", "based", "most", "such", "each",
                   "hence", "therefore", "thus", "there", "used"}
        first_word = s.split()[0].lower() if s.split() else ''
        return (len(s) > 3
                and s.lower() not in generic
                and first_word not in BAD_SUBJECTS
                and not re.search(r'\b\d+\b|REVIEW', s))

    # Pattern 1: "X is/are/was/refers to..."
    m = re.match(r'^([A-Z][^,.(]{2,50?}?)\s+(?:is|are|was|were|refers to|means|denotes)\s', sentence)
    if m:
        subj = re.sub(r'\s+(also|already|often|always|now|then|just|only)\s*$', '', m.group(1)).strip()
        # If subject looks like "Heading Body" (two separate title-case phrases), take only first phrase
        # e.g. "Adaptive Assessment Mechanism Question difficulty" → "Question difficulty"
        # Split on transition from title-case run to another title-case word
        title_parts = re.split(r'(?<=[a-z])\s+(?=[A-Z])', subj)
        if len(title_parts) > 1:
            # Take the last meaningful part (the actual topic, not the slide heading)
            for part in reversed(title_parts):
                part = part.strip()
                if _is_clean(part) and len(part) > 4:
                    subj = part
                    break
        if _is_clean(subj):
            return subj

    # Pattern 2: "X is used for..."
    m = re.match(r'^([A-Z][A-Za-z\s\-()]{3,45?}?)\s+(?:is|are)\s+used\s', sentence)
    if m:
        subj = m.group(1).strip()
        if _is_clean(subj):
            return subj

    # Pattern 3: capitalised noun phrase (1–4 words, no digits, not a verb)
    m = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', sentence)
    if m:
        candidate = m.group(1).strip()
        if _is_clean(candidate):
            return candidate

    return ""


def extract_key_terms(sentence: str, top_n: int = 3) -> list[str]:
    """Return the most important multi-char words (rough noun-phrase extraction)."""
    words = re.findall(r'\b[A-Za-z][a-zA-Z\-]{4,}\b', sentence)
    stopwords = {"this", "that", "these", "those", "which", "where", "their",
                 "there", "about", "would", "could", "should", "have", "been",
                 "with", "from", "into", "such", "also", "more", "each",
                 "other", "system", "based", "using", "used", "level", "student",
                 "question", "questions", "learning", "students", "generate",
                 "generated", "approach", "method", "model", "models"}
    filtered = [w for w in words if w.lower() not in stopwords and len(w) > 4]
    # Return highest-frequency terms
    counts = Counter(w.lower() for w in filtered)
    return [w for w, _ in counts.most_common(top_n)]


# ── Distractor generation ────────────────────────────────────────────────────

def make_distractors(correct: str, diff: str, all_terms: list[str], all_definitions: list[str] = None) -> list[str]:
    """
    Build 3 plausible wrong MCQ options.
    Priority: other real definitions from the corpus → term-based phrases → generic fallback.
    """
    result = []
    correct_lower = correct.lower()

    # 1. Use other real sentence fragments from the corpus as distractors
    if all_definitions:
        shuffled = all_definitions[:]
        random.shuffle(shuffled)
        for defn in shuffled:
            d = defn[:120].rstrip(".,")
            if (d.lower() != correct_lower[:len(d)]
                    and correct_lower[:30] not in d.lower()
                    and len(d) > 15
                    and len(result) < 3):
                result.append(d)

    # 2. Term-swap distractors: take correct answer, swap out key terms
    if len(result) < 3 and all_terms:
        terms_not_in_correct = [t for t in all_terms if t.lower() not in correct_lower]
        for term in terms_not_in_correct[:4]:
            if len(result) >= 3:
                break
            # Replace first long word in correct answer with a different term
            swapped = re.sub(r'\b[A-Za-z]{6,}\b', term, correct[:80], count=1)
            if swapped != correct[:80] and swapped not in result:
                result.append(swapped.rstrip(".,") + ("..." if len(correct) > 80 else ""))

    # 3. Generic fallback pool (only if we still need options)
    generic_pool = {
        "easy": [
            "It stores data in a relational database table",
            "It renders the graphical user interface components",
            "It handles HTTP network communication between client and server",
            "It manages file input and output on the local filesystem",
            "It compresses raw data for long-term storage",
        ],
        "medium": [
            "It applies rule-based pattern matching without any training data",
            "It uses unsupervised clustering to group similar features together",
            "It relies on hard-coded heuristics rather than learned parameters",
            "It performs dimensionality reduction using principal component analysis",
            "It implements a greedy search over a fixed predefined vocabulary",
        ],
        "hard": [
            "It uses a Bayesian inference framework with conjugate prior distributions",
            "It applies gradient boosting over an ensemble of weak decision stumps",
            "It relies on attention masks derived from absolute positional encodings",
            "It implements backpropagation through time on stacked LSTM cells",
            "It uses kernel density estimation on the high-dimensional feature manifold",
        ],
    }
    pool = generic_pool.get(diff, generic_pool["medium"])[:]
    random.shuffle(pool)
    for d in pool:
        if len(result) >= 3:
            break
        if correct_lower[:20] not in d.lower():
            result.append(d)

    return result[:3]


# ── Question templates ───────────────────────────────────────────────────────

def sanitise_answer(text: str) -> str:
    """Remove trailing slide-number digits, REVIEW labels, and stray punctuation."""
    text = text.strip()
    text = re.sub(r'\s*REVIEW-\d+\b.*$', '', text, flags=re.IGNORECASE)
    # Strip trailing digits whether before or after a period: "system 1." or "system. 1" or "system 1"
    text = re.sub(r'\s+\d+(\s+\d+)*\s*\.?\s*$', '', text)
    text = re.sub(r'\.\s*\d+(\s+\d+)*\s*$', '', text)
    text = text.rstrip(".,;: –-")
    return text.strip()

def make_definition_question(sentence: str, diff: str, bloom: str, all_terms: list[str], all_definitions: list[str] = None) -> dict | None:
    """Sentence like 'X is/are Y' → 'What is X?'"""
    m = re.match(
        r'^(.{5,60}?)\s+(?:is|are|was|refers to|means|denotes)\s+(.{10,})',
        sentence, re.IGNORECASE
    )
    if not m:
        return None
    subject = m.group(1).strip().rstrip(",.")
    definition = m.group(2).strip().rstrip(".")

    # Reject subjects that are clearly noise
    bad_subject_patterns = [r'\b\d+\b', r'REVIEW', r'^Hence', r'^There',
                            r'^It ', r'^The system$', r'^Questions?$',
                            r'^Generated questions?$', r'^Student performance$']
    if any(re.search(p, subject, re.IGNORECASE) for p in bad_subject_patterns):
        return None

    # Cap subject at 4 words — anything longer is likely a PPTX heading+body fusion
    subject_words = subject.split()
    if len(subject_words) > 4:
        # Try splitting on lowercase→Capital boundary
        parts = re.split(r'(?<=[a-z])\s+(?=[A-Z])', subject)
        if len(parts) >= 2:
            candidate = ' '.join(parts[len(parts)//2:]).strip()
            if 3 < len(candidate.split()) <= 4 and not re.search(r'\b\d+\b|REVIEW', candidate):
                subject = candidate
            else:
                return None  # Can't salvage this fused subject
        else:
            return None  # All title-case blob — reject

    if len(definition) < 10 or len(subject) < 3:
        return None

    q_text = f"What is {subject}?"
    correct = sanitise_answer(definition[:120] + ('...' if len(definition) > 120 else ''))
    if len(correct) < 8:
        return None

    distractors = make_distractors(correct, diff, all_terms, all_definitions)
    # Sanitise distractors too
    distractors = [sanitise_answer(d) for d in distractors]
    options = [correct] + distractors
    random.shuffle(options)

    return {
        "question": q_text,
        "type": "MCQ",
        "difficulty": diff,
        "bloom": bloom if bloom in ["Remember (L1)", "Understand (L2)"] else "Remember (L1)",
        "options": [f"{chr(65+i)}. {o}" for i, o in enumerate(options)],
        "answer": f"{chr(65 + options.index(correct))}. {correct}",
    }


def make_purpose_question(sentence: str, diff: str, bloom: str, all_terms: list[str], all_definitions: list[str] = None) -> dict | None:
    """Sentence with 'used for/to' → 'What is X used for?'"""
    m = re.search(
        r'([A-Z][A-Za-z\s\-]{3,40}?)\s+(?:is|are)\s+used\s+(?:for|to)\s+(.{10,80})',
        sentence, re.IGNORECASE
    )
    if not m:
        return None
    subject = m.group(1).strip()
    purpose = m.group(2).strip().rstrip(".,")

    q_text = f"What is {subject} used for?"
    correct = sanitise_answer(purpose)
    if len(correct) < 8:
        return None
    distractors = make_distractors(correct, diff, all_terms, all_definitions)
    distractors = [sanitise_answer(d) for d in distractors]
    options = [correct] + distractors
    random.shuffle(options)

    return {
        "question": q_text,
        "type": "MCQ",
        "difficulty": diff,
        "bloom": "Understand (L2)",
        "options": [f"{chr(65+i)}. {o}" for i, o in enumerate(options)],
        "answer": f"{chr(65 + options.index(correct))}. {correct}",
    }


def make_process_question(sentence: str, diff: str, bloom: str, all_terms: list[str], all_definitions: list[str] = None) -> dict | None:
    """Sentences describing a mechanism → 'How does X function?'"""
    sl = sentence.lower()
    triggers = ["is used", "are used", "performs", "enables", "allows", "provides",
                "implements", "applies", "works by", "functions as"]
    if not any(t in sl for t in triggers):
        return None
    subject = extract_subject(sentence)
    if not subject or len(subject.split()) > 6:
        return None
    subject = re.sub(r'\s+(is|are|was|were|has|have|does|do|will|can|may|should)\s*$', '', subject).strip()
    if len(subject) < 4:
        return None
    q_text = f"How does {subject} function in this system?"
    correct = sanitise_answer(sentence[:150] + ("..." if len(sentence) > 150 else ""))
    if len(correct) < 15:
        return None
    distractors = make_distractors(correct, diff, all_terms, all_definitions)
    distractors = [sanitise_answer(d) for d in distractors]
    options = [correct] + distractors
    random.shuffle(options)

    return {
        "question": q_text,
        "type": "MCQ",
        "difficulty": diff,
        "bloom": bloom if bloom in ["Apply (L3)", "Analyze (L4)"] else "Analyze (L4)",
        "options": [f"{chr(65+i)}. {o}" for i, o in enumerate(options)],
        "answer": f"{chr(65 + options.index(correct))}. {correct}",
    }


def make_fill_blank(sentence: str, diff: str, bloom: str, all_terms: list[str]) -> dict | None:
    """Replace a key term with a blank."""
    sentence = sanitise_answer(sentence)
    terms = extract_key_terms(sentence, top_n=5)
    if not terms:
        return None
    terms.sort(key=len, reverse=True)
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(sentence):
            blanked = pattern.sub("_______", sentence, count=1)
            return {
                "question": blanked,
                "type": "Fill",
                "difficulty": diff,
                "bloom": "Remember (L1)",
                "options": [],
                "answer": term,
            }
    return None


def make_true_false(sentence: str, diff: str, bloom: str) -> dict | None:
    """Convert a factual statement into a True/False question."""
    sentence = sanitise_answer(sentence)
    if len(sentence) < 40 or len(sentence) > 220:
        return None
    # Reject sentences starting with a verb (no subject)
    first_word = sentence.split()[0].lower()
    BAD_STARTS = {'used', 'uses', 'performs', 'provides', 'allows', 'enables',
                  'handles', 'stores', 'generates', 'implements', 'applies',
                  'designed', 'built', 'based', 'integrated', 'trained'}
    if first_word in BAD_STARTS:
        return None
    # Must contain a proper finite verb
    if not re.search(r'\b(is|are|was|were|does|do|can|will|has|have|should|'
                     r'allows|enables|provides|uses|used|helps|generates|performs)\b',
                     sentence, re.IGNORECASE):
        return None
    if " not " in sentence.lower() or " no " in sentence.lower():
        inverted = re.sub(r'\b(not|no)\b', '', sentence, count=1, flags=re.IGNORECASE)
        inverted = re.sub(r'\s+', ' ', inverted).strip()
        return {
            "question": f"True or False: {inverted}",
            "type": "TF",
            "difficulty": "easy",
            "bloom": "Remember (L1)",
            "options": ["True", "False"],
            "answer": "False",
        }
    return {
        "question": f"True or False: {sentence.rstrip('.')}.",
        "type": "TF",
        "difficulty": diff,
        "bloom": "Remember (L1)",
        "options": ["True", "False"],
        "answer": "True",
    }


def make_short_answer(sentence: str, diff: str, bloom: str) -> dict | None:
    """Generate an open-ended short answer question."""
    sentence = sanitise_answer(sentence)
    if len(sentence) < 30:
        return None

    VERB_STARTS = {
        'develop', 'implement', 'integrate', 'build', 'create', 'design',
        'generate', 'train', 'evaluate', 'test', 'support', 'improve',
        'help', 'track', 'store', 'display', 'allow', 'enable', 'provide',
        'use', 'apply', 'perform', 'classify', 'extract', 'convert',
        'analyse', 'analyze', 'compare', 'explain', 'describe', 'define',
        'maintain', 'manage', 'handle', 'process', 'ensure', 'monitor',
        'collect', 'record', 'update', 'calculate', 'compute', 'fetch',
        'send', 'receive', 'load', 'save', 'deploy', 'run', 'execute',
    }
    first_word = sentence.split()[0].lower()
    if first_word in VERB_STARTS or first_word.rstrip('es').rstrip('d') in VERB_STARTS:
        return None

    subject = extract_subject(sentence)
    if not subject or len(subject) < 4:
        return None
    subject = re.sub(r'\s+(is|are|was|for|to|and|or|of|in|on|at)\s*$', '', subject).strip()
    if len(subject) < 4:
        return None

    # Reject single generic or negative words as subjects
    GENERIC_SUBJECTS = {
        'rule', 'system', 'model', 'method', 'approach', 'process',
        'result', 'output', 'input', 'data', 'lack', 'partial', 'limited',
        'limitations', 'limitation', 'absence', 'issues', 'issue',
        'existing', 'traditional', 'current', 'proposed',
    }
    if subject.lower() in GENERIC_SUBJECTS:
        return None
    # Reject if subject is a single lowercase common word
    if len(subject.split()) == 1 and subject[0].islower():
        return None

    templates = [
        f"Explain the role of {subject} in this system.",
        f"What is the significance of {subject} in this context?",
        f"Describe how {subject} contributes to the overall system.",
    ]
    return {
        "question": random.choice(templates),
        "type": "Short",
        "difficulty": diff,
        "bloom": bloom if bloom in ["Analyze (L4)", "Evaluate (L5)", "Create (L6)"] else "Analyze (L4)",
        "options": [],
        "answer": sentence[:200],
    }


# ── TF-IDF sentence ranking ───────────────────────────────────────────────────

def rank_sentences(sentences: list[str], top_n: int) -> list[str]:
    """Return the most informationally dense sentences using TF-IDF."""
    if len(sentences) <= top_n:
        return sentences
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vec.fit_transform(sentences)
        scores = matrix.sum(axis=1).A1  # sum of TF-IDF weights per sentence
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = sorted([i for i, _ in indexed[:top_n]])
        return [sentences[i] for i in top_indices]
    except Exception:
        return sentences[:top_n]


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_questions(
    text: str,
    count: int = 10,
    difficulty_mode: str = "adaptive",
    question_type: str = "Mixed (All Types)",
    bloom_focus: str = "Auto (System Decides)",
) -> list[dict]:
    """
    Generate `count` questions from `text` using local NLP only.
    Returns a list of question dicts compatible with the frontend.
    """
    text = clean_text(text)
    sentences = split_sentences(text)
    if not sentences:
        return []

    candidate_count = min(len(sentences), max(count * 5, 40))
    ranked = rank_sentences(sentences, candidate_count)

    # Global term list for distractor swapping
    all_terms = extract_key_terms(" ".join(ranked), top_n=30)

    # Collect real answer fragments for corpus-aware distractors
    all_definitions = []
    for s in ranked:
        m = re.match(r'^.{5,60}?\s+(?:is|are|was|refers to|means|denotes)\s+(.{10,120})', s, re.IGNORECASE)
        if m:
            all_definitions.append(m.group(1).strip().rstrip(".,"))
        mp = re.search(r'(?:is|are)\s+used\s+(?:for|to)\s+(.{10,80})', s, re.IGNORECASE)
        if mp:
            all_definitions.append(mp.group(1).strip().rstrip(".,"))
    # Deduplicate
    all_definitions = list(dict.fromkeys(all_definitions))

    # Difficulty distribution for adaptive mode
    third = max(count // 3, 1)
    diff_cycle = ["easy"] * third + ["medium"] * third + ["hard"] * third
    diff_cycle += ["medium"] * (count - len(diff_cycle))  # pad to count
    random.shuffle(diff_cycle)

    type_map = {
        "MCQ Only":          ["MCQ"],
        "True/False Only":   ["TF"],
        "Fill in the Blank": ["Fill"],
        "Short Answer":      ["Short"],
        "Mixed (All Types)": ["MCQ", "MCQ", "MCQ", "TF", "Fill", "Short"],
    }
    type_pool = type_map.get(question_type, ["MCQ", "MCQ", "MCQ", "TF", "Fill", "Short"])

    questions: list[dict] = []
    used_sentences: set[str] = set()
    # Full pool: ranked first, then remaining, repeated to ensure we can fill count
    sentence_pool = ranked + [s for s in sentences if s not in ranked]
    # Allow each sentence to be used for at most one question type
    sentence_type_used: dict[str, set] = {}  # sentence -> set of types already tried

    q_id = 0
    attempts = 0
    max_attempts = max(len(sentence_pool) * 6, count * 20)  # generous budget

    while len(questions) < count and attempts < max_attempts:
        sentence = sentence_pool[attempts % len(sentence_pool)]
        attempts += 1

        diff = difficulty_mode if difficulty_mode != "adaptive" else score_difficulty(sentence)
        bloom = bloom_focus if bloom_focus != "Auto (System Decides)" else detect_bloom(sentence)
        desired_type = type_pool[q_id % len(type_pool)]

        # Skip if this sentence already produced this exact type
        if sentence_type_used.get(sentence, set()).issuperset({desired_type}):
            continue

        q = None
        if desired_type == "MCQ":
            q = (make_definition_question(sentence, diff, bloom, all_terms, all_definitions) or
                 make_purpose_question(sentence, diff, bloom, all_terms, all_definitions) or
                 make_process_question(sentence, diff, bloom, all_terms, all_definitions))
        elif desired_type == "TF":
            q = make_true_false(sentence, diff, bloom)
        elif desired_type == "Fill":
            q = make_fill_blank(sentence, diff, bloom, all_terms)
        elif desired_type == "Short":
            q = make_short_answer(sentence, diff, bloom)

        # Track what we've tried for this sentence
        sentence_type_used.setdefault(sentence, set()).add(desired_type)

        if q:
            q["id"] = q_id
            questions.append(q)
            q_id += 1

    # Final fallback: fill remaining slots — try every sentence with every type
    fallback_types = ["MCQ", "TF", "Fill", "Short"]
    for sentence in sentence_pool * 2:  # allow two passes
        if len(questions) >= count:
            break
        for ftype in fallback_types:
            if len(questions) >= count:
                break
            if ftype in sentence_type_used.get(sentence, set()):
                continue
            diff = difficulty_mode if difficulty_mode != "adaptive" else score_difficulty(sentence)
            bloom = bloom_focus if bloom_focus != "Auto (System Decides)" else detect_bloom(sentence)
            q = None
            if ftype == "MCQ":
                q = (make_definition_question(sentence, diff, bloom, all_terms, all_definitions) or
                     make_purpose_question(sentence, diff, bloom, all_terms, all_definitions) or
                     make_process_question(sentence, diff, bloom, all_terms, all_definitions))
            elif ftype == "TF":
                q = make_true_false(sentence, diff, bloom)
            elif ftype == "Fill":
                q = make_fill_blank(sentence, diff, bloom, all_terms)
            elif ftype == "Short":
                q = make_short_answer(sentence, diff, bloom)
            sentence_type_used.setdefault(sentence, set()).add(ftype)
            if q:
                q["id"] = q_id
                questions.append(q)
                q_id += 1

    return questions[:count]