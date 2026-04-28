import re

from app.config import MAX_CONTEXT_CHUNKS


STOPWORDS = {
    "a", "about", "an", "and", "answer", "are", "as", "by", "case", "did",
    "do", "does", "document", "explain", "for", "from", "give", "has",
    "have", "how", "in", "into", "is", "main", "mean", "means", "meant",
    "of", "on", "or", "provided", "simple", "study", "tell", "that", "the",
    "their", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "word", "words",
}

LIST_WORDS = {
    "type", "types", "kind", "kinds", "category", "categories",
    "classification", "classifications", "classified", "form", "forms",
    "list", "section", "sections", "part", "parts", "component",
    "components", "element", "elements", "characteristic", "characteristics",
    "feature", "features", "trait", "traits", "quality", "qualities",
}

PROBLEM_WORDS = {
    "problem", "problems", "challenge", "challenges", "issue", "issues",
    "difficulty", "difficulties", "barrier", "barriers", "constraint",
    "constraints",
}

EXPLANATORY_WORDS = {
    "role", "roles", "impact", "impacts", "importance", "important",
    "contribution", "contributions", "effect", "effects", "function",
    "functions", "benefit", "benefits", "significance", "purpose",
    "responsibility", "responsibilities",
}

LOW_VALUE_MARKERS = {
    "review questions", "further readings", "table of contents",
    "learning objectives", "chapter objectives", "self assessment",
    "fill in the blanks", "chapter overview", "here we have provided",
    "to better comprehend the ideas", "students should review the chapter",
    "- how, when, and where is dates",
}


def filter_relevant_chunks(results, query=None, max_chunks=None):
    seen = set()
    filtered = []

    for chunk in results:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        key = chunk.get("id") or (chunk.get("page"), chunk.get("chunk_index"), text[:120])
        if key in seen:
            continue
        seen.add(key)

        item = dict(chunk)
        item["quality_score"] = _chunk_quality_score(text)
        filtered.append(item)

    if not query:
        return _sort_chunks(filtered)[:max_chunks] if max_chunks else _sort_chunks(filtered)

    scored = []
    for chunk in filtered:
        score, evidence = _relevance_score(chunk, query)
        if not evidence:
            continue
        item = dict(chunk)
        item["guardrail_score"] = score
        item["score"] = float(item.get("score", 0) or 0) + score
        scored.append(item)

    scored = _sort_chunks(scored)
    if not scored:
        return []

    limit = max_chunks or _context_limit_for_query(query)
    selected = scored[:limit]

    # Keep final context in PDF order so list and section answers retain flow.
    return sorted(selected, key=lambda c: (c.get("page", 0), c.get("chunk_index", 0)))


def _relevance_score(chunk, query):
    text = chunk.get("text", "")
    text_lower = text.lower()
    topic_terms = _topic_terms(query)
    subject_terms = _subject_terms(query)
    query_terms = _query_terms(query)

    topic_hits = _count_term_hits(topic_terms, text_lower)
    subject_hits = _count_term_hits(subject_terms, text_lower)
    query_hits = _count_term_hits(query_terms, text_lower)
    quality = _chunk_quality_score(text)

    score = (
        float(chunk.get("score", 0) or 0)
        + query_hits * 0.8
        + topic_hits * 2.2
        + subject_hits * 3.2
        + quality
    )

    if _is_section_question(query):
        score += _cue_score(text_lower, LIST_WORDS | {"following", "include", "includes", "basis"})
    if _is_problem_question(query):
        score += _cue_score(text_lower, PROBLEM_WORDS | {"following"})
    if _is_explanatory_question(query):
        score += _cue_score(text_lower, EXPLANATORY_WORDS | {"creates", "provides", "helps", "accelerate", "improve"})
        if re.search(r"(?:^|\n)\s*\d{1,3}[\).]\s+", text):
            score += 6.0
    if _is_definition_question(query):
        score += _cue_score(text_lower, {"is", "are", "means", "refers", "defined", "process", "tendency"})
    if _is_when_question(query) and re.search(r"\b\d{3,4}\b", text_lower):
        score += 2.0

    evidence = _has_evidence(query, text_lower, topic_terms, subject_terms, topic_hits, subject_hits, score)
    return score, evidence


def _has_evidence(query, text_lower, topic_terms, subject_terms, topic_hits, subject_hits, score):
    if _is_low_value_text(text_lower) and subject_hits == 0:
        return False

    if subject_terms:
        required = min(2, len(subject_terms))
        if subject_hits >= required:
            return True
        if _is_explanatory_question(query) and topic_hits > 0:
            return True
        if topic_hits >= max(1, min(2, len(topic_terms))):
            return True
        return False

    if topic_terms:
        return topic_hits > 0

    return score > 1.0


def _sort_chunks(chunks):
    return sorted(
        chunks,
        key=lambda c: (
            -float(c.get("score", 0) or 0),
            -float(c.get("guardrail_score", 0) or 0),
            -float(c.get("quality_score", 0) or 0),
            c.get("page", 0),
            c.get("chunk_index", 0),
        ),
    )


def _context_limit_for_query(query):
    if _is_section_question(query) or _is_problem_question(query):
        return MAX_CONTEXT_CHUNKS + 2
    if _is_explanatory_question(query):
        return MAX_CONTEXT_CHUNKS
    return min(MAX_CONTEXT_CHUNKS, 7)


def _query_terms(query):
    terms = _topic_terms(query)
    if _is_section_question(query):
        terms.extend(LIST_WORDS)
    if _is_problem_question(query):
        terms.extend(PROBLEM_WORDS)
    if _is_explanatory_question(query):
        terms.extend(EXPLANATORY_WORDS)
    return list(dict.fromkeys(terms))


def _is_section_question(query):
    return bool(re.search(
        r"\b(types?|kinds?|categories|classifications?|forms?|list|"
        r"characteristics?|features?|traits?|qualities|sections?|parts?|"
        r"components?|elements?)\b",
        query.lower(),
    )) or _is_problem_question(query)


def _is_problem_question(query):
    return bool(re.search(
        r"\b(problems?|challenges?|issues?|difficulties|barriers|faced by|before)\b",
        query.lower(),
    ))


def _is_explanatory_question(query):
    return bool(re.search(
        r"\b(role|roles|impact|impacts|importance|important|contribution|contributions|"
        r"effect|effects|function|functions|benefit|benefits|significance|purpose|"
        r"responsibility|responsibilities|how|why|explain|describe|discuss)\b",
        query.lower(),
    ))


def _is_definition_question(query):
    return bool(re.search(
        r"^\s*(?:what\s+(?:is|does)|define|meaning\s+of)\b",
        query.lower(),
    )) and not re.search(r"\b(role|importance|impact|purpose|benefit|effect|types?|problems?)\b", query.lower())


def _is_when_question(query):
    return bool(re.search(r"^\s*(?:when|what\s+year|what\s+date)\b", query.lower()))


def _is_describe_question(query):
    return bool(re.search(
        r"\b(?:describe|tell|explain|how\s+(?:to|does|did|is|are)|"
        r"what\s+is\s+the\s+(?:process|method|way|procedure))\b",
        query.lower(),
    ))


def _topic_terms(query):
    ignored = STOPWORDS | LIST_WORDS | PROBLEM_WORDS | EXPLANATORY_WORDS | {
        "faced", "basis", "based", "define",
    }
    return [
        token for token in _content_terms(_normalize_question(query))
        if token not in ignored
    ]


def _subject_terms(query):
    query_lower = _normalize_question(query)
    patterns = [
        r"\b(?:role|roles|importance|impact|effect|effects|benefits?|purpose|contributions?)\s+of\s+(.+?)(?:\s+in\s+|\s+for\s+|\?|$)",
        r"\b(?:types?|kinds?|categories|classifications?|forms?|sections?|parts?|components?|elements?)\s+of\s+(.+?)(?:\?|$)",
        r"\b(?:problems?|challenges?|issues?|difficulties|barriers)\s+(?:faced\s+by|of|before|related\s+to)\s+(.+?)(?:\?|$)",
        r"\b(?:what\s+is|define|meaning\s+of)\s+(.+?)(?:\?|$)",
        r"\b(?:main|major|primary)\s+(?:issue|problem|reason|cause)\s+(?:in|of|with)\s+(.+?)(?:\?|$)",
    ]

    ignored = LIST_WORDS | PROBLEM_WORDS | EXPLANATORY_WORDS | {"case", "study", "simple", "words"}
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            terms = [term for term in _content_terms(match.group(1)) if term not in ignored]
            if terms:
                return terms

    return _topic_terms(query)


def _content_terms(text):
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", (text or "").lower())
        if token not in STOPWORDS
    ]


def _normalize_question(query):
    query = (query or "").lower()
    query = re.sub(r"\bin\s+simple\s+words\b", "", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def _matches_topic(text, terms):
    text_lower = text.lower()
    matches = _count_term_hits(terms, text_lower)
    required = 1 if len(terms) <= 2 else max(2, len(terms) // 2)
    return matches >= required


def _matches_topic_lenient(text, terms):
    return _count_term_hits(terms, text.lower()) >= 1


def _count_term_hits(terms, text_lower):
    return sum(1 for term in terms if any(
        re.search(rf"\b{re.escape(variant)}\b", text_lower)
        for variant in _term_variants(term)
    ))


def _cue_score(text_lower, cues):
    return sum(1.1 for cue in cues if re.search(rf"\b{re.escape(cue)}\b", text_lower))


def _chunk_quality_score(text):
    lowered = text.lower()
    score = 0.0

    if any(marker in lowered for marker in LOW_VALUE_MARKERS):
        score -= 4.0
    if lowered.count("?") >= 2:
        score -= 2.0
    if re.search(r"\bstudents?\s+(?:will|can)\s+(?:learn|understand|develop|gain)\b", lowered):
        score -= 1.5
    if re.search(r"\b(?:is|are|means|refers to|defined as|include|includes|following)\b", lowered):
        score += 1.0
    if re.search(r"\n\s*(?:\d{1,3}[\).]|[-*])\s+", text):
        score += 1.2

    return score


def _is_low_value_text(text_lower):
    return any(marker in text_lower for marker in LOW_VALUE_MARKERS)


def _term_variants(term):
    variants = {term}

    if term.endswith("s") and len(term) > 4:
        variants.add(term[:-1])
    elif len(term) > 3:
        variants.add(term + "s")
    if term.endswith("ies") and len(term) > 5:
        variants.add(term[:-3] + "y")
    if term.endswith("y") and len(term) > 5:
        variants.add(term[:-1] + "ies")
        variants.add(term[:-1] + "ical")
    if term.endswith("ical") and len(term) > 6:
        variants.add(term[:-4] + "y")
    if term.endswith("isation") and len(term) > 8:
        root = term[:-7]
        variants.update({root + "ise", root + "ization", root + "ize"})
    if term.endswith("ization") and len(term) > 8:
        root = term[:-7]
        variants.update({root + "isation", root + "ise", root + "ize"})
    if term.endswith("ise") and len(term) > 5:
        root = term[:-3]
        variants.update({root + "isation", root + "ize", root + "ization"})
    if term.endswith("ize") and len(term) > 5:
        root = term[:-3]
        variants.update({root + "ization", root + "ise", root + "isation"})
    if term == "important":
        variants.add("importance")
    if term == "entrepreneur":
        variants.update({"entrepreneurs", "entrepreneurship"})
    if term == "entrepreneurship":
        variants.update({"entrepreneur", "entrepreneurs"})
    if term == "india":
        variants.add("indian")
    if term == "indian":
        variants.add("india")
    if term == "science":
        variants.add("scientific")
    if term == "scientific":
        variants.add("science")

    return variants
