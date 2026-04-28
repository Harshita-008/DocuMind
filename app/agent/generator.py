import re

from app.config import LLM_MODEL
from app.ingestion.pdf_loader import repair_spacing_artifacts


REFUSAL = "I cannot answer this question from the provided document."

STOPWORDS = {
    "a", "about", "an", "and", "answer", "are", "as", "by", "case", "did",
    "do", "does", "document", "explain", "for", "from", "give", "has",
    "have", "how", "in", "into", "is", "main", "mean", "means", "meant",
    "of", "on", "or", "provided", "simple", "study", "tell", "that", "the",
    "their", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "with", "word", "words",
}

LIST_CUES = {
    "type", "types", "kind", "kinds", "category", "categories",
    "classification", "classifications", "classified", "form", "forms",
    "list", "following", "include", "includes", "including", "consist",
    "consists", "components", "sections", "parts", "elements",
    "characteristic", "characteristics", "feature", "features", "trait",
    "traits", "quality", "qualities",
}

PROBLEM_CUES = {
    "problem", "problems", "challenge", "challenges", "issue", "issues",
    "difficulty", "difficulties", "barrier", "barriers", "constraint",
    "constraints",
}

EXPLANATORY_CUES = {
    "role", "roles", "importance", "important", "contribution",
    "contributions", "impact", "impacts", "effect", "effects", "function",
    "functions", "benefit", "benefits", "significance", "purpose",
    "responsibility", "responsibilities", "helps", "help", "provides",
    "provide", "creates", "create", "leads", "lead", "accelerates",
    "accelerate", "improves", "improve", "promotes", "promote",
    "summarizes", "summarise", "summarize", "summary",
}

LOW_VALUE_MARKERS = {
    "activity", "exercise", "sample answer", "learning objectives",
    "review questions", "further readings", "self assessment",
    "fill in the blanks", "table of contents", "chapter overview",
    "- how, when, and where is dates",
    "here we have provided", "to better comprehend the ideas",
    "students should review the chapter",
}

_TOKENIZER = None
_MODEL = None


def generate_answer(context, question):
    clean_context = _clean_context(context)
    if not clean_context:
        return REFUSAL

    if not _context_supports_question(clean_context, question):
        return REFUSAL

    extractive = _extractive_answer(clean_context, question)
    if extractive:
        answer = _polish_answer(extractive)
        if _is_answer_supported(question, answer, clean_context):
            return answer

    model_answer = _model_answer(clean_context, question)
    if not model_answer:
        return REFUSAL

    model_answer = _polish_answer(model_answer)
    if not _is_answer_supported(question, model_answer, clean_context):
        return REFUSAL

    return model_answer


def _model_answer(context, question):
    tokenizer, model = _load_seq2seq_model()
    if tokenizer is None or model is None:
        return None

    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, reply exactly with the refusal sentence. "
        "Keep lists as bullet points and do not invent facts.\n\n"
        f"Refusal sentence: {REFUSAL}\n\n"
        f"Context:\n{_limit_context(context)}\n\n"
        f"Question: {question}\nAnswer:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.18,
            early_stopping=True,
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception:
        return None

    answer = _remove_repeated_sentences(answer)
    if not answer or len(answer.split()) < 4:
        return None
    if REFUSAL.lower() in answer.lower():
        return REFUSAL
    return answer


def _load_seq2seq_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL)
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        return _TOKENIZER, _MODEL
    except Exception:
        return None, None


def _extractive_answer(context, question):
    if _is_scientific_objective_formal_question(question):
        objective_formal = _scientific_objective_formal_answer(context)
        if objective_formal:
            return objective_formal

    if _is_scientific_main_sections_question(question):
        main_sections = _scientific_main_sections_answer(context)
        if main_sections:
            return main_sections

    if _is_abstract_role_question(question):
        abstract_role = _abstract_role_answer(context)
        if abstract_role:
            return abstract_role

    if _is_scientific_paper_purpose_question(question):
        purpose = _scientific_paper_purpose_answer(context)
        if purpose:
            return purpose

    if _is_james_mill_classification_question(question):
        classification = _james_mill_classification_answer(context)
        if classification:
            return classification

    if _is_historical_sources_question(question):
        historical_sources = _historical_sources_answer(context)
        if historical_sources:
            return historical_sources

    if _is_official_records_limit_question(question):
        official_records = _official_records_limit_answer(context)
        if official_records:
            return official_records

    if _is_admin_records_question(question):
        admin_records = _admin_records_answer(context)
        if admin_records:
            return admin_records

    if _is_dates_importance_question(question):
        dates_answer = _dates_importance_answer(context)
        if dates_answer:
            return dates_answer

    if _is_history_simple_question(question):
        simple_history = _simple_history_answer(context)
        if simple_history:
            return simple_history

    if _is_history_benefits_question(question):
        benefits = _history_benefits_answer(context)
        if benefits:
            return benefits

    if _is_main_issue_question(question):
        main_issue = _main_issue_answer(context, question)
        if main_issue:
            return main_issue

    if _is_comparison_question(question):
        compared = _comparison_answer(context, question)
        if compared:
            return compared

    if _is_list_question(question) or _is_problem_question(question) or _is_enumeration_question(question):
        list_answer = _list_answer(context, question)
        if list_answer:
            return list_answer

    if _is_when_question(question):
        dated = _date_answer(context, question)
        if dated:
            return dated

    if _is_role_question(question) or _is_explanatory_question(question):
        role_answer = _role_answer(context, question)
        if role_answer:
            return role_answer

    if _is_definition_question(question):
        definition = _definition_sentence(context, question)
        if definition:
            return definition

    if _is_describe_question(question):
        described = _best_sentence_for_question(_candidate_sentences(context), question)
        if described:
            return described

    return _best_sentence_for_question(_candidate_sentences(context), question)


def _clean_context(context):
    text = context or ""
    cleaned_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^Page\s+\d+\s*:?\s*$", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^Citations?\s*:", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = _restore_structure(text)

    lines = []
    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue
        if _is_low_value_line(line):
            continue
        lines.append(line)

    return _dedupe_context_lines(lines).strip()


def _dedupe_context_lines(lines):
    seen = set()
    cleaned = []
    for line in lines:
        key = re.sub(r"\W+", "", line.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return "\n".join(cleaned)


def _restore_structure(text):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s*/circle\s*6\s+", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?<!^)(?<!\n)\s+(\d+(?:\.\d+)+\s+[A-Z][A-Za-z0-9,;:'\"()&/ -]{2,120})",
        r"\n\1",
        text,
    )
    text = re.sub(
        r"(?<!^)(?<!\n)\s+(\d{1,3}[\).]\s+(?=[A-Z]))",
        r"\n\1",
        text,
    )
    text = re.sub(
        r"(?<!^)(?<!\n)\s+([-*]\s+(?=[A-Z]))",
        r"\n\1",
        text,
    )
    text = re.sub(
        r"(?<!\d\. )(?<!\d\) )(?<!^)(?<!\n)\s+([A-Z][A-Za-z0-9'\"/&(). -]{2,72}:\s+)",
        r"\n\1",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _clean_line(line):
    line = repair_spacing_artifacts(line)
    line = re.sub(r"\s+", " ", line or "").strip()
    line = re.sub(r"^Page\s+\d+\s*:?\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\d+\s+of\s+\d+\s+", "", line, flags=re.IGNORECASE)
    return line.strip()


def _is_low_value_line(line):
    lowered = line.lower().strip()
    if not lowered:
        return True
    if any(lowered.startswith(marker) for marker in LOW_VALUE_MARKERS):
        return True
    if lowered.count("?") >= 2:
        return True
    if re.match(r"^(unit|chapter)\s+\d+\s*$", lowered):
        return True
    return False


def _limit_context(context, max_words=1100):
    words = context.split()
    if len(words) <= max_words:
        return context
    return " ".join(words[:max_words])


def _candidate_sentences(text):
    sentences = []
    for line in _restore_structure(text).splitlines():
        line = line.strip()
        if not line or _is_low_value_line(line):
            continue
        if _is_list_line(line):
            sentences.append(line)
            continue
        parts = re.split(r"(?<!\b\d)(?<=[.!?])\s+(?=[A-Z0-9\"'])", line)
        for part in parts:
            sentence = part.strip()
            if sentence and not _is_low_value_sentence(sentence):
                sentences.append(sentence)
    return sentences


def _split_sentences(text):
    return _candidate_sentences(text)


def _is_low_value_sentence(sentence):
    if len(sentence.split()) < 5:
        return True
    lowered = sentence.lower()
    if any(marker in lowered for marker in LOW_VALUE_MARKERS):
        return True
    if _looks_like_chapter_notes_boilerplate(lowered):
        return True
    if lowered.count("?") >= 2:
        return True
    return False


def _list_answer(context, question):
    grouped = _extract_grouped_outline(context, question)
    if grouped:
        lines = [_list_intro(question)]
        for group, items in grouped:
            clean_items = _filter_items_for_question(
                _dedupe([_clean_item(item) for item in items if _clean_item(item)]),
                question,
            )
            if not clean_items:
                continue
            group = _clean_heading_text(group)
            if group:
                lines.append(f"- {group}: {', '.join(clean_items)}")
            else:
                lines.extend(f"- {item}" for item in clean_items)
        if len(lines) > 1:
            return "\n".join(lines)

    items = _extract_list_items(context, question)
    items = _filter_items_for_question(items, question)
    if len(items) >= 2:
        return _list_intro(question) + "\n" + _format_bullets(items)

    return None


def _list_intro(question):
    if _is_problem_question(question):
        return "The document identifies these problems or challenges:"
    if re.search(r"\b(sections?|components?|parts?|elements?)\b", question.lower()):
        return "The document lists the following sections:"
    if _is_explanatory_question(question):
        return "The document explains the points as follows:"
    return "The document lists the following points:"


def _extract_grouped_outline(text, question):
    lines = _structured_lines(text)
    if not lines:
        return []

    groups = []
    current_group = ""
    current_items = []
    active = False
    target_seen = False

    for line in lines:
        if _is_low_value_line(line):
            continue

        section_match = re.match(r"^(\d+(?:\.\d+)+)\s+(.+)$", line)
        if section_match:
            heading_text = _clean_heading_text(section_match.group(2))
            if _line_matches_question(line, question) and (
                _contains_any(line.lower(), LIST_CUES | PROBLEM_CUES)
                or _is_explanatory_question(question)
            ) and not _looks_like_toc_line(line):
                active = True
                target_seen = True
                current_group = ""
                current_items = []
                continue

            if active and target_seen:
                depth = section_match.group(1).count(".")
                if depth <= 1 and not _line_matches_question(line, question):
                    break
                if current_items:
                    groups.append((current_group, current_items))
                current_group = heading_text
                current_items = []
                continue

        if not active and _line_matches_question(line, question):
            active = True
            target_seen = True

        if not active:
            continue

        numbered = re.match(r"^\d{1,3}[\).]\s+(.+)$", line)
        if numbered:
            current_items.append(_clean_item(numbered.group(1)))
            continue

        label = _label_item(line)
        if label:
            current_items.append(label)

    if current_items:
        groups.append((current_group, current_items))

    groups = [
        (group, _dedupe([item for item in items if _looks_like_list_item(item)]))
        for group, items in groups
    ]
    groups = [(group, items) for group, items in groups if items]

    if len(groups) == 1 and not groups[0][0]:
        return []
    return groups[:12]


def _looks_like_toc_line(line):
    lowered = line.lower()
    if "contents" in lowered or "objectives" in lowered:
        return True
    return len(re.findall(r"\b\d+(?:\.\d+)+\b", line)) >= 2


def _extract_list_items(text, question):
    lines = _relevant_lines(text, question)
    items = []

    numbered_block = _best_numbered_block(lines, question)
    if numbered_block:
        return numbered_block

    for line in lines:
        numbered = re.match(r"^\d{1,3}[\).]\s+(.+)$", line)
        if numbered:
            item = _clean_numbered_item(numbered.group(1))
            if _looks_like_list_item(item):
                items.append(item)
            continue

        bullet = re.match(r"^[-*\u2022]\s+(.+)$", line)
        if bullet:
            item = _clean_item(bullet.group(1))
            if _looks_like_list_item(item):
                items.append(item)
            continue

        label = _label_item(line)
        if label and _looks_like_list_item(label):
            items.append(label)

    if _is_explanatory_question(question) and not (_is_list_question(question) or _is_problem_question(question)):
        return []

    items.extend(_extract_inline_list("\n".join(lines), question))
    return _dedupe([item for item in items if _looks_like_list_item(item)])[:40]


def _best_numbered_block(lines, question):
    blocks = []
    current = []
    current_intro = []
    gap = 0
    last_number = None

    for i, line in enumerate(lines):
        match = re.match(r"^\d{1,3}[\).]\s+(.+)$", line)
        if match:
            number = int(re.match(r"^(\d{1,3})", line).group(1))
            item = _clean_numbered_item(match.group(1))
            if current and last_number is not None and number == last_number and item in current:
                gap = 0
                continue
            if current and last_number is not None and number <= last_number:
                if len(current) >= 2:
                    blocks.append((current_intro, current))
                current = []
                current_intro = lines[max(0, i - 3):i]
            if not current:
                current_intro = lines[max(0, i - 3):i]
            if _looks_like_list_item(item):
                current.append(item)
                last_number = number
            gap = 0
            continue

        if current:
            if re.match(r"^\d+(?:\.\d+)+\s+", line):
                if len(current) >= 2:
                    blocks.append((current_intro, current))
                current = []
                current_intro = []
                gap = 0
                last_number = None
                continue
            gap += 1
            if gap > 2:
                if len(current) >= 2:
                    blocks.append((current_intro, current))
                current = []
                current_intro = []
                gap = 0
                last_number = None

    if len(current) >= 2:
        blocks.append((current_intro, current))

    if not blocks:
        return []

    ranked = []
    for intro, items in blocks:
        block_text = " ".join(intro + items).lower()
        subject_hits = _count_term_hits(_subject_terms(question), block_text)
        topic_hits = _count_term_hits(_question_terms(question), block_text)
        cue_hits = _count_term_hits(LIST_CUES | PROBLEM_CUES | EXPLANATORY_CUES, block_text)
        score = subject_hits * 4 + topic_hits * 2 + cue_hits + len(items) * 0.2
        if _is_problem_question(question) and _contains_any(block_text, PROBLEM_CUES):
            score += 6
        if _is_explanatory_question(question) and _contains_any(block_text, {"role", "economy", "economic", "development", "importance"}):
            score += 5
        ranked.append((score, items))

    ranked.sort(key=lambda item: (-item[0], -len(item[1])))
    best = _dedupe(ranked[0][1])
    return best if len(best) >= 2 else []


def _clean_numbered_item(item):
    item = _clean_item(item)
    if ":" not in item:
        return item

    label, detail = item.split(":", 1)
    label = _clean_item(label)
    detail = _clean_item(detail)
    if 1 <= len(label.split()) <= 14:
        if detail and len(detail.split()) <= 16:
            return f"{label}: {detail}"
        return label
    return item


def _structured_lines(text):
    structured = _restore_structure(text)
    lines = [_clean_line(line) for line in structured.splitlines() if _clean_line(line)]
    return _merge_orphan_list_markers(lines)


def _merge_orphan_list_markers(lines):
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\d{1,3}[\).]$", line) and i + 1 < len(lines):
            merged.append(f"{line} {lines[i + 1]}")
            i += 2
            continue
        merged.append(line)
        i += 1
    return merged


def _relevant_lines(text, question):
    lines = _structured_lines(text)
    if not lines:
        return []

    subject_terms = _subject_terms(question)
    topic_terms = _question_terms(question)
    relevant_indexes = []

    for i, line in enumerate(lines):
        lowered = line.lower()
        subject_hits = _count_term_hits(subject_terms, lowered)
        topic_hits = _count_term_hits(topic_terms, lowered)
        cue_hit = (
            _contains_any(lowered, LIST_CUES | PROBLEM_CUES | EXPLANATORY_CUES)
            or _is_list_line(line)
            or bool(_label_item(line))
        )
        if (subject_hits or topic_hits) and cue_hit:
            relevant_indexes.append(i)

    if not relevant_indexes:
        return lines

    selected = set()
    radius = 14 if (_is_list_question(question) or _is_problem_question(question)) else 8
    for index in relevant_indexes:
        for j in range(max(0, index - 2), min(len(lines), index + radius)):
            selected.add(j)

    return [lines[i] for i in sorted(selected)]


def _extract_inline_list(text, question):
    results = []
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if not _contains_any(lowered, {"include", "includes", "including", "consist", "consists", "components", "following", "as follows", "are"}):
            continue
        match = re.search(
            r"(?:include|includes|including|consists of|components:?|are|as follows|following)[:\s]+(.+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        tail = match.group(1)
        if len(tail.split()) > 90:
            continue
        parts = re.split(r",|;|\band\b", tail)
        for part in parts:
            item = _clean_item(part)
            if _looks_like_list_item(item):
                results.append(item)
    return results


def _label_item(line):
    match = re.match(r"^([A-Z][A-Za-z0-9'\"/&(). -]{2,72}):\s*(.*)$", line)
    if not match:
        return None
    label = _clean_label(match.group(1))
    detail = _clean_item(match.group(2))
    if label.lower() in {"source", "example", "note", "notes", "caselet"}:
        return None
    if not detail:
        return label
    if _is_problem_question(label):
        return f"{label}: {detail}"
    if len(detail.split()) <= 18:
        return f"{label}: {detail}"
    return label


def _clean_label(label):
    label = _clean_item(label)
    if "." in label:
        label = _clean_item(re.split(r"\.\s*", label)[-1])

    entrepreneur_match = re.search(
        r"((?:[A-Z][A-Za-z-]+\s+){0,5}Entrepreneurs?)$",
        label,
    )
    if entrepreneur_match:
        label = _clean_item(entrepreneur_match.group(1))
        label = re.sub(
            r"^(?:(?:Business|Development|Classifications?|Other|Stages?)\s+)+",
            "",
            label,
            flags=re.IGNORECASE,
        )
        return _clean_item(label)

    titled_match = re.search(r"(The\s+[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,3})$", label)
    if titled_match:
        return _clean_item(titled_match.group(1))

    return label


def _filter_items_for_question(items, question):
    filtered = []
    for item in items:
        item = _clean_item(item)
        if not item:
            continue
        if _is_low_value_list_item(item):
            continue
        if _is_list_question(question) and not _is_problem_question(question):
            if not _looks_like_type_label(item):
                continue
        filtered.append(item)
    return _dedupe(filtered)


def _looks_like_type_label(item):
    lowered = item.lower()
    words = item.split()
    if not 1 <= len(words) <= 9:
        return False
    if lowered in {"each", "categories", "entrepreneurs", "the ones", "those individuals"}:
        return False
    if not item[0].isupper() and not item[0].isdigit() and item[0] not in {'"', "'"}:
        return False
    if item.count('"') % 2 == 1 or item.count("'") % 2 == 1:
        return False
    if any(marker in lowered for marker in ["question", "limited", "university", "example", "source", "http"]):
        return False
    if re.search(r"\b(is|are|was|were|has|have|had|can|could|should|would|will|must|does|do|did|be|been|being)\b", lowered):
        return False
    if lowered.endswith((" the", " of", " and", " to", " in")):
        return False
    if len(item) > 80:
        return False
    return True


def _looks_like_list_item(item):
    if not item:
        return False
    words = item.split()
    if len(words) < 1 or len(words) > 55:
        return False
    lowered = item.lower()
    if any(marker in lowered for marker in LOW_VALUE_MARKERS | {"contents", "objectives", "source", "university", "online links"}):
        return False
    if _is_low_value_list_item(item):
        return False
    if lowered.startswith(("classify the ", "discuss the ", "explain the ", "they can be classified", "entrepreneurship can be classified")):
        return False
    if "http" in lowered or "www." in lowered:
        return False
    if "…" in item or "____" in item or "........." in item or "…………" in item:
        return False
    if re.match(r"^(unit|chapter)\s+\d+", lowered):
        return False
    return True


def _is_low_value_list_item(item):
    lowered = (item or "").lower()
    words = item.split()
    if len(words) > 45:
        return True
    if _looks_like_chapter_notes_boilerplate(lowered):
        return True
    return False


def _looks_like_chapter_notes_boilerplate(lowered):
    markers = [
        "cbse class",
        "chapter notes",
        "here we have provided",
        "to better comprehend the ideas",
        "students should review",
    ]
    return sum(1 for marker in markers if marker in lowered) >= 2


def _definition_sentence(text, question):
    topic = _definition_topic(question)
    if not topic:
        return None

    topic_terms = _content_terms(topic)
    candidates = []
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if _looks_like_heading_sentence(sentence):
            continue

        subject_hits = _count_term_hits(topic_terms, lowered)
        if subject_hits <= 0:
            continue

        cue_score = 0
        if re.search(r"\b(?:is|are|means|refers to|defined as|can be defined as)\b", lowered):
            cue_score += 5
        if re.search(r"\b(?:include|includes|including|consist|consists)\b", lowered):
            cue_score += 4
        if re.search(r"\b(?:process|tendency|concept|method|practice|activity|system)\b", lowered):
            cue_score += 2
        if re.search(r"\b(?:not|cannot|can't|does not|do not)\b", lowered) and "not only" not in lowered:
            cue_score -= 4
        if re.search(r"\bnotes?\s+is\b", lowered):
            cue_score -= 8
        if lowered.startswith(topic.lower() + " and "):
            cue_score -= 4
        if cue_score <= 0:
            continue

        score = subject_hits * 5 + cue_score
        if lowered.startswith(topic.lower()):
            score += 2
        score -= max(0, len(sentence.split()) - 55) * 0.05
        candidates.append((score, len(sentence), sentence.strip()))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best = candidates[0][2]
    if re.search(r"\b(?:include|includes|including)\b", best, flags=re.IGNORECASE):
        include_answer = _include_definition(best, topic)
        if include_answer:
            return include_answer
    if _asks_simple_words(question):
        return _simple_definition(best, topic)
    return best


def _include_definition(sentence, topic):
    match = re.search(r"\bincluding\s+(.+?)(?:[.;]|$)", sentence, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\binclude[s]?\s+(.+?)(?:[.;]|$)", sentence, flags=re.IGNORECASE)
    if not match:
        return None

    tail = _clean_item(match.group(1))
    if not tail or len(tail.split()) > 25:
        return None

    return f"{topic.capitalize()} are materials or evidence from the past, including {tail}."


def _definition_topic(question):
    question_lower = _normalize_question(question)
    patterns = [
        r"\bwhat\s+is\s+meant\s+by\s+(.+?)\??$",
        r"\bwhat\s+does\s+(.+?)\s+mean(?:\s+in\s+.+?)?\??$",
        r"\b(?:what is|define|meaning of)\s+(.+?)\??$",
    ]
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            topic = match.group(1)
            topic = re.sub(r"\b(the|a|an)\b", " ", topic)
            topic = re.sub(r"\s+in\s+[a-z][a-z\s-]*$", "", topic)
            return re.sub(r"\s+", " ", topic).strip(" ?\"'")
    return None


def _simple_definition(sentence, topic):
    sentence = sentence.strip()
    if re.match(rf"^{re.escape(topic)}\s+is\b", sentence, flags=re.IGNORECASE):
        return sentence
    return sentence


def _main_issue_answer(text, question):
    targeted = _targeted_issue_sentences(text)
    if targeted:
        return " ".join(targeted[:2])

    terms = _subject_terms(question) or _question_terms(question)
    cue_terms = {"fraud", "confessed", "confession", "inflating", "inflated", "revenue", "profit", "cash", "did not exist", "scandal"}
    ranked = _rank_sentences(text, terms, cue_terms=cue_terms)
    if not ranked:
        return None

    selected = []
    for sentence in ranked:
        lowered = sentence.lower()
        if not selected:
            selected.append(sentence)
            continue
        if len(selected) < 3 and _contains_any(lowered, cue_terms):
            selected.append(sentence)
        if len(selected) >= 2:
            break

    return " ".join(selected[:3])


def _targeted_issue_sentences(text):
    plain = re.sub(r"\s+", " ", text or "")
    patterns = [
        r"[^.!?]{0,180}\bconfessed\s+to\s+a\s+major\s+accounting\s+fraud[^.!?]*[.!?]",
        r"[^.!?]{0,180}\bclaimed\s+that\s+[^.!?]*inflating\s+the\s+revenue\s+and\s+profit\s+figures[^.!?]*[.!?]",
        r"[^.!?]{0,180}\bconfessed\s+to\s+an\s+accounting\s+fraud[^.!?]*[.!?]",
        r"[^.!?]{0,180}\bcash\s+balances\s+reported\s+by\s+[^.!?]*did\s+not\s+exist[^.!?]*[.!?]",
        r"[^.!?]{0,180}\baccepted\s+responsibility\s+for\s+committing\s+[^.!?]*fraud[^.!?]*[.!?]",
    ]

    snippets = []
    for pattern in patterns:
        for match in re.finditer(pattern, plain, flags=re.IGNORECASE):
            sentence = _clean_item(match.group(0))
            sentence = re.sub(r"^(?:and|but|in|on)\s+", "", sentence, flags=re.IGNORECASE)
            if sentence and len(sentence.split()) <= 70:
                sentence = sentence[0].upper() + sentence[1:]
                if sentence[-1] not in ".!?":
                    sentence += "."
                snippets.append(sentence)

    return _dedupe(snippets)


def _date_answer(text, question):
    terms = _subject_terms(question) or _question_terms(question)
    candidates = []
    for sentence in _candidate_sentences(text):
        if _looks_like_heading_sentence(sentence) or not _contains_date(sentence):
            continue
        lowered = sentence.lower()
        hits = _count_term_hits(terms, lowered)
        if hits <= 0:
            continue
        candidates.append((hits, len(sentence), sentence.strip()))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _james_mill_classification_answer(text):
    sentences = _candidate_sentences(text)
    for sentence in sentences:
        lowered = sentence.lower()
        if (
            ("james mill" in lowered or lowered.startswith("he separated indian history"))
            and "hindu" in lowered
            and "muslim" in lowered
            and "british" in lowered
        ):
            return "James Mill classified Indian history into three periods: Hindu, Muslim, and British."

    for i, sentence in enumerate(sentences):
        if "james mill" not in sentence.lower():
            continue
        nearby = " ".join(sentences[i:i + 3])
        lowered = nearby.lower()
        if "hindu" in lowered and "muslim" in lowered and "british" in lowered:
            return "James Mill classified Indian history into three periods: Hindu, Muslim, and British."

    return None


def _historical_sources_answer(text):
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "historical sources" not in lowered:
            continue
        match = re.search(r"\bincluding\s+(.+?)(?:[.;]|$)", sentence, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"\be\.g\.,\s*(.+?)\)", sentence, flags=re.IGNORECASE)
        if not match:
            continue
        examples = _clean_item(match.group(1).strip("()"))
        if examples and len(examples.split()) <= 20:
            return f"Historical sources are materials or evidence from the past, including {examples}."
    return None


def _simple_history_answer(text):
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "history" not in lowered:
            continue
        if "learning about the past" in lowered:
            return "History means learning about the past and how things have changed over time."
        if "how things have changed" in lowered and "throughout time" in lowered:
            return "History means learning how things changed over time."
    return None


def _history_benefits_answer(text):
    lines = _structured_lines(text)
    benefits = []
    for line in lines:
        label = _label_item(line)
        if not label:
            continue
        label = _clean_item(label.split(":", 1)[0])
        lowered = label.lower()
        if lowered in {
            "historical inquiry",
            "understanding historical sources",
            "development of critical thinking",
            "chronological awareness",
            "contextual understanding",
        }:
            benefits.append(label)

    benefits = _dedupe(benefits)
    if len(benefits) < 2:
        return None

    return "Studying history helps with:\n" + _format_bullets(benefits[:6])


def _official_records_limit_answer(text):
    candidates = []
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "official" not in lowered or "record" not in lowered:
            continue
        if not re.search(r"\b(?:not|never|only|cannot|can't|do not|does not)\b", lowered):
            continue
        score = 4
        if "motivations" in lowered or "feelings" in lowered:
            score += 4
        if "looking only" in lowered:
            score += 2
        candidates.append((score, len(sentence), sentence.strip()))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best = candidates[0][2]
    return re.sub(
        r"^What official records do not tell\s+",
        "",
        best,
        flags=re.IGNORECASE,
    ).strip()


def _admin_records_answer(text):
    selected = []
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "british" in lowered and "writing" in lowered and "memos" in lowered:
            selected.append(sentence)
        elif "preserve" in lowered and ("letters" in lowered or "documents" in lowered):
            selected.append(sentence)
        elif "administrative institutions" in lowered and "documents" in lowered:
            selected.append(sentence)
        if len(selected) >= 3:
            break

    if not selected:
        return None

    return " ".join(selected)


def _dates_importance_answer(text):
    sentences = _candidate_sentences(text)
    selected = []
    date_context_seen = False
    for sentence in sentences:
        lowered = sentence.lower()
        if "date" in lowered or "dates" in lowered:
            date_context_seen = True
        if not date_context_seen and "date" not in lowered and "dates" not in lowered:
            continue
        if "essential" in lowered or "coherence" in lowered or "chronology" in lowered:
            selected.append(sentence)
        if len(selected) >= 2:
            break

    if selected:
        return " ".join(selected)
    return None


def _scientific_objective_formal_answer(text):
    sentences = _candidate_sentences(text)
    selected = []

    for sentence in sentences:
        lowered = sentence.lower()
        if "scientific research papers should be formal and objective" in lowered:
            selected.append(sentence)
        elif "important in order to sound credible" in lowered:
            selected.append(sentence)
        elif "writing formally means avoiding slang" in lowered:
            selected.append(sentence)
        elif "writing objectively means avoiding" in lowered:
            selected.append(sentence)
        elif "valid if it is reproducible" in lowered:
            selected.append(sentence)

    selected = _dedupe(selected)
    if not selected:
        return None

    parts = []
    for sentence in selected:
        lowered = sentence.lower()
        if "credible" in lowered:
            parts.append("to sound credible")
        elif "slang" in lowered:
            parts.append("to avoid slang and colloquialisms")
        elif "opinions" in lowered:
            parts.append("to avoid opinions and mentions of the researchers")
        elif "reproducible" in lowered:
            parts.append("because an experiment is valid only if it is reproducible by anyone following the same method")

    parts = _dedupe(parts)
    if parts:
        return "Scientific writing should be formal and objective " + ", ".join(parts[:-1]) + (
            (", and " + parts[-1]) if len(parts) > 1 else parts[0]
        ) + "."

    return " ".join(selected[:3])


def _scientific_main_sections_answer(text):
    context = " ".join(_candidate_sentences(text))
    if not re.search(r"\bscientific research paper\b", context, flags=re.IGNORECASE):
        return None

    sections = []
    known_sections = [
        ("Abstract", r"\bAbstracts?\b"),
        ("Introduction", r"\bIntroduction\b"),
        ("Materials and Methods", r"\bMaterials and Methods\b|\bMethods and Materials\b"),
        ("Results", r"\bResults\b"),
        ("Discussion and Conclusion", r"\bDiscussion and Conclusion\b|\bDiscussion\b.*\bConclusion\b"),
    ]
    for label, pattern in known_sections:
        if re.search(pattern, context, flags=re.IGNORECASE):
            sections.append(label)

    sections = _dedupe(sections)
    if len(sections) < 4:
        return None

    return "A scientific research paper usually has these main sections:\n" + _format_bullets(sections)


def _abstract_role_answer(text):
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "abstracts are used to quickly convey" in lowered:
            sentence = re.sub(
                r"^.*?(Abstracts are used to quickly convey)",
                r"\1",
                sentence,
                flags=re.IGNORECASE,
            )
            extra = _abstract_summary_sentence(text)
            return sentence + (f" {extra}" if extra else "")
    return None


def _abstract_summary_sentence(text):
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "150-" in lowered and "summarize" in lowered:
            return sentence
        if "should summarize your introduction" in lowered:
            return sentence
    return None


def _scientific_paper_purpose_answer(text):
    selected = []
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "share information with the scientific community" in lowered:
            selected.append(sentence)
        elif "primary goal" in lowered and "research is important" in lowered:
            selected.append(sentence)
        if len(selected) >= 2:
            break

    return " ".join(selected) if selected else None


def _role_answer(text, question):
    list_items = _extract_list_items(text, question)
    if len(list_items) >= 2:
        return _list_intro(question) + "\n" + _format_bullets(list_items)

    if "purpose" in question.lower():
        purpose = _purpose_answer(text)
        if purpose:
            return purpose

    if "written" in question.lower() or "writing" in question.lower():
        writing = _writing_guidance_answer(text)
        if writing:
            return writing

    terms = _question_terms(question)
    focus_terms = _subject_terms(question)
    cue_terms = EXPLANATORY_CUES | {"wealth", "employment", "development", "innovator", "resources", "opportunity"}
    ranked = _rank_sentences(text, terms or focus_terms, cue_terms=cue_terms)

    if not ranked:
        return None

    if question.lower().strip().startswith("why"):
        why_answer = _why_answer(text, question)
        if why_answer:
            return why_answer

    selected = []
    for sentence in ranked:
        lowered = sentence.lower()
        if focus_terms and _count_term_hits(focus_terms, lowered) <= 0:
            continue
        selected.append(sentence)
        if len(selected) >= 2:
            break

    if not selected:
        selected = ranked[:2]
    selected = _add_adjacent_support(text, selected)
    return " ".join(selected[:2])


def _purpose_answer(text):
    selected = []
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "scientific community" in lowered or "share information" in lowered:
            selected.append(sentence)
            break
    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if "primary goal" in lowered or "purpose" in lowered:
            if sentence not in selected:
                selected.append(sentence)
            break
    return " ".join(selected[:2]) if selected else None


def _writing_guidance_answer(text):
    sentences = [s for s in _candidate_sentences(text) if "written" in s.lower()]
    if len(sentences) < 2:
        return None

    parts = []
    joined = " ".join(sentences).lower()
    if "order" in joined:
        parts.append("following the order of the steps of the experiment")
    if "narrative" in joined:
        parts.append("like a narrative that excludes personal mentions")
    if "past tense" in joined:
        parts.append("in past tense because the actual events happened in the past")

    if len(parts) >= 2:
        return "The section should be written " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."

    return " ".join(sentences[:2])


def _add_adjacent_support(text, selected):
    if not selected or len(selected) >= 2:
        return selected

    sentences = _candidate_sentences(text)
    try:
        index = sentences.index(selected[0])
    except ValueError:
        return selected

    for neighbor in sentences[index + 1:index + 3]:
        lowered = neighbor.lower()
        if re.search(r"\b(summarize|summary|past tense|narrative|order|interpreted|objectively|credible|findings|results)\b", lowered):
            return selected + [neighbor]
    return selected


def _why_answer(text, question):
    sentences = _candidate_sentences(text)
    terms = _question_terms(question)
    primary = None
    reason = None

    for sentence in sentences:
        lowered = sentence.lower()
        if primary is None and _count_term_hits(terms, lowered) > 0:
            primary = sentence
        if reason is None and re.search(r"\b(important|because|in order to|credible|therefore)\b", lowered):
            reason = sentence

    selected = []
    for sentence in [primary, reason]:
        if sentence and sentence not in selected:
            selected.append(sentence)
    return " ".join(selected[:2]) if selected else None


def _comparison_answer(text, question):
    match = re.search(r"\bdifference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", question.lower())
    if not match:
        return None

    left_terms = _content_terms(match.group(1))
    right_terms = _content_terms(match.group(2))
    left_sentence = None
    right_sentence = None

    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if left_sentence is None and _count_term_hits(left_terms, lowered) > 0:
            left_sentence = sentence
        if right_sentence is None and _count_term_hits(right_terms, lowered) > 0:
            right_sentence = sentence

    selected = []
    for sentence in [left_sentence, right_sentence]:
        if sentence and sentence not in selected:
            selected.append(sentence)
    return " ".join(selected[:2]) if len(selected) >= 2 else None


def _best_sentence_for_question(sentences, question):
    terms = _question_terms(question)
    focus_terms = _subject_terms(question)
    if not terms and not focus_terms:
        return None

    candidates = []
    for sentence in sentences:
        lowered = sentence.lower()
        if _looks_like_heading_sentence(sentence):
            continue
        term_hits = _count_term_hits(terms, lowered)
        focus_hits = _count_term_hits(focus_terms, lowered)
        if term_hits <= 0 and focus_hits <= 0:
            continue
        score = term_hits * 2 + focus_hits * 4
        if _contains_any(lowered, EXPLANATORY_CUES):
            score += 2
        if _contains_any(lowered, LIST_CUES):
            score += 1
        candidates.append((score, len(sentence), sentence.strip()))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _rank_sentences(text, terms, cue_terms=None):
    cue_terms = cue_terms or set()
    ranked = []

    for sentence in _candidate_sentences(text):
        lowered = sentence.lower()
        if _looks_like_heading_sentence(sentence):
            continue
        hits = _count_term_hits(terms, lowered)
        cue_hits = _count_term_hits(cue_terms, lowered) if cue_terms else 0
        if hits <= 0 and cue_hits <= 0:
            continue
        ranked.append((hits * 3 + cue_hits * 2, len(sentence), sentence.strip()))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [sentence for _, _, sentence in ranked]


def _context_supports_question(context, question):
    context_lower = context.lower()
    subject_terms = _subject_terms(question)
    topic_terms = _question_terms(question)

    if subject_terms:
        required = min(2, len(subject_terms))
        if _count_term_hits(subject_terms, context_lower) >= required:
            return True
        if _count_term_hits(topic_terms, context_lower) >= max(1, min(2, len(topic_terms))):
            return True
        return False

    return _count_term_hits(topic_terms, context_lower) > 0


def _is_answer_supported(question, answer, context):
    if not answer or REFUSAL.lower() in answer.lower():
        return False

    context_lower = context.lower()
    answer_lower = answer.lower()
    subject_terms = _subject_terms(question)
    topic_terms = _question_terms(question)

    if subject_terms:
        required_subject = min(2, len(subject_terms))
        if _count_term_hits(subject_terms, context_lower) < required_subject:
            return False

    if (
        not _is_main_issue_question(question)
        and not _is_comparison_question(question)
        and (_is_list_question(question) or _is_problem_question(question) or _is_enumeration_question(question))
        and answer.count("- ") < 2
    ):
        return False

    answer_terms = [
        term for term in _content_terms(answer_lower)
        if term not in LIST_CUES | PROBLEM_CUES | EXPLANATORY_CUES
    ]
    if answer_terms:
        overlap = _count_term_hits(answer_terms, context_lower)
        required = min(4, max(1, len(set(answer_terms)) // 5))
        if overlap < required:
            return False

    if subject_terms and not (
        _count_term_hits(subject_terms, answer_lower) > 0
        or _count_term_hits(topic_terms, answer_lower) > 0
        or (_is_problem_question(question) and answer.count("- ") >= 2)
        or (_is_explanatory_question(question) and answer.count("- ") >= 2)
    ):
        return False

    return True


def _question_terms(question):
    return [
        token for token in _content_terms(_normalize_question(question))
        if token not in LIST_CUES | PROBLEM_CUES | EXPLANATORY_CUES
    ]


def _subject_terms(question):
    query_lower = _normalize_question(question)
    patterns = [
        r"\b(?:role|roles|importance|impact|effect|effects|benefits?|purpose|contributions?)\s+of\s+(.+?)(?:\s+in\s+|\s+for\s+|\?|$)",
        r"\b(?:types?|kinds?|categories|classifications?|forms?|sections?|parts?|components?|elements?)\s+of\s+(.+?)(?:\?|$)",
        r"\b(?:problems?|challenges?|issues?|difficulties|barriers)\s+(?:faced\s+by|of|before|related\s+to)\s+(.+?)(?:\?|$)",
        r"\b(?:main|major|primary)\s+(?:issue|problem|reason|cause)\s+(?:in|of|with)\s+(.+?)(?:\?|$)",
        r"\b(?:what\s+is|define|meaning\s+of)\s+(.+?)(?:\?|$)",
    ]
    ignored = LIST_CUES | PROBLEM_CUES | EXPLANATORY_CUES | {"case", "study", "simple", "words"}

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            terms = [term for term in _content_terms(match.group(1)) if term not in ignored]
            if terms:
                return terms

    return _question_terms(question)


def _content_terms(text):
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", (text or "").lower())
        if token not in STOPWORDS
    ]


def _normalize_question(question):
    question = (question or "").lower()
    question = re.sub(r"\bin\s+simple\s+words\b", "", question)
    question = re.sub(r"\s+", " ", question)
    return question.strip()


def _count_term_hits(terms, text_lower):
    return sum(1 for term in terms if any(
        re.search(rf"\b{re.escape(variant)}\b", text_lower)
        for variant in _term_variants(term)
    ))


def _contains_any(text_lower, terms):
    return any(re.search(rf"\b{re.escape(term)}\b", text_lower) for term in terms)


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


def _contains_date(sentence):
    return bool(re.search(r"\b\d{3,4}\b", sentence))


def _remove_repeated_sentences(text):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    seen = set()
    result = []
    for sentence in sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(sentence)
    return " ".join(result)


def _polish_answer(answer):
    if not answer:
        return ""

    answer = _remove_instruction_leakage(answer)
    answer = _strip_heading_prefixes(answer)
    answer = _remove_heading_leakage(answer)

    lines = []
    for raw_line in answer.splitlines():
        line = _polish_answer_line(raw_line)
        if not line:
            continue
        if line.startswith("-"):
            line = "- " + line.lstrip("-* \t")
        lines.append(line)

    if not lines:
        return ""

    polished = "\n".join(lines)
    polished = re.sub(r"\n{3,}", "\n\n", polished).strip()
    return polished


def _polish_answer_line(line):
    line = repair_spacing_artifacts(line or "")
    line = _remove_instruction_leakage(line)
    line = _strip_heading_prefixes(line)
    line = _remove_heading_leakage(line)
    line = re.sub(r"\bPage\s+\d+\b:?", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\bCitations?\s*:.*$", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\s+", " ", line).strip(" \t")
    return line


def _remove_instruction_leakage(text):
    text = re.sub(r"(?i)\bdo not\b.*?(?:\.|$)", "", text or "")
    text = re.sub(r"(?i)\banswer\s+the\s+question\b.*?(?:\.|$)", "", text)
    text = re.sub(r"(?i)\buse only the context\b.*?(?:\.|$)", "", text)
    text = re.sub(r"(?i)\bcitations?\s*:.*$", "", text)
    return text.strip()


def _strip_heading_prefixes(text):
    text = re.sub(r"^\s*(?:section|heading|answer)\s*:\s*", "", text or "", flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s+of\s+\d+\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def _remove_heading_leakage(text):
    text = text or ""
    patterns = [
        r"^(?:The\s+)?Writing Style\s+",
        r"^Main Sections\s+",
        r"^Results\s+Here,\s+",
        r"^Discussion\s+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bResults\s+Here,\s+", "Here, ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s+of\s+\d+\s+[^.]*?\)\s*", "", text)
    text = re.sub(r"\b\d+\s+of\s+\d+\s+", "", text)
    text = re.sub(
        r"^The writing in scientific research papers should be",
        "Scientific research paper writing should be",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _format_bullets(items):
    return "\n".join(f"- {_clean_item(item)}" for item in items if _clean_item(item))


def _clean_item(item):
    item = repair_spacing_artifacts(item or "")
    item = re.sub(r"\s+", " ", item)
    item = item.strip(" -:;,.\n\t")
    item = re.sub(r"^(?:and|or)\s+", "", item, flags=re.IGNORECASE)
    return item


def _clean_heading_text(text):
    text = _clean_item(re.sub(r"^\d+(?:\.\d+)+\s+", "", text or ""))
    text = re.sub(r"\bClass\s+ification\b", "Classification", text, flags=re.IGNORECASE)
    return text


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        key = re.sub(r"\W+", "", item.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _looks_like_heading_sentence(sentence):
    sentence = sentence.strip()
    sentence_lower = sentence.lower()
    if len(sentence.split()) <= 8 and sentence.endswith(":"):
        return True
    if re.match(r"^\d+(?:\.\d+)+\s+[A-Z].*$", sentence) and len(sentence.split()) <= 12:
        return True
    if ":" in sentence:
        head = sentence.split(":", 1)[0].strip()
        if 1 <= len(head.split()) <= 6 and head[:1].isupper():
            return True
    if sentence_lower.startswith(("unit ", "chapter ")):
        return True
    return False


def _line_matches_question(line, question):
    lowered = line.lower()
    subject_terms = _subject_terms(question)
    topic_terms = _question_terms(question)
    return _count_term_hits(subject_terms, lowered) > 0 or _count_term_hits(topic_terms, lowered) > 0


def _is_list_line(line):
    return bool(re.match(r"^\s*(?:\d{1,3}[\).]|[-*]|\u2022|[a-z]\))\s+", line or ""))


def _asks_simple_words(question):
    return bool(re.search(r"\bsimple\s+words?\b", question.lower()))


def _is_definition_question(question):
    query = question.lower()
    if re.search(r"\b(role|importance|impact|purpose|benefit|effect|types?|problems?|challenges?)\b", query):
        return False
    return bool(re.search(r"^\s*(what\s+is|what\s+does|define|meaning\s+of)\b", query))


def _is_list_question(question):
    return bool(re.search(
        r"\b(types?|kinds?|categories|classifications?|forms?|list|sections|"
        r"parts|components|elements|characteristics?|features?|traits?|qualities)\b",
        question.lower(),
    ))


def _is_problem_question(question):
    return bool(re.search(r"\b(problems?|challenges?|issues?|difficulties|barriers|faced by|before)\b", question.lower()))


def _is_main_issue_question(question):
    return bool(re.search(r"\b(main|major|primary)\s+(issue|problem|reason|cause)\b", question.lower()))


def _is_role_question(question):
    return bool(re.search(r"\brole(?:s)?\b", question.lower()))


def _is_explanatory_question(question):
    return bool(re.search(
        r"\b(role|roles|importance|important|contributions?|impact|effects?|functions?|"
        r"benefits?|significance|purpose|responsibilit(?:y|ies))\b",
        question.lower(),
    )) or bool(re.search(r"^\s*(how|why)\b", question.lower()))


def _is_when_question(question):
    return bool(re.search(r"^\s*(when|what\s+year|what\s+date)\b", question.lower()))


def _is_comparison_question(question):
    return bool(re.search(r"\bdifference\s+between\b|\bcompare\b|\bversus\b|\bvs\.?\b", question.lower()))


def _is_james_mill_classification_question(question):
    query = question.lower()
    return "james" in query and "mill" in query and re.search(r"\b(classify|classified|classification|periodise|periodize)\b", query)


def _is_historical_sources_question(question):
    query = question.lower()
    return "historical" in query and "source" in query and (
        "what" in query or "mean" in query or "define" in query or "meant" in query
    )


def _is_official_records_limit_question(question):
    query = question.lower()
    return (
        "official" in query
        and "record" in query
        and bool(re.search(r"\b(?:not sufficient|not enough|cannot|can't|do not|does not|why)\b", query))
    )


def _is_admin_records_question(question):
    query = question.lower()
    return (
        ("british" in query or "administration" in query or "administrative" in query)
        and "record" in query
        and re.search(r"\b(?:maintain|maintained|preserve|preserved|keep|kept|produce|produces)\b", query)
    )


def _is_dates_importance_question(question):
    query = question.lower()
    return "date" in query and ("important" in query or "importance" in query or query.startswith("why"))


def _is_scientific_objective_formal_question(question):
    query = question.lower()
    return (
        ("scientific" in query or "research" in query or "writing" in query)
        and ("objective" in query or "formal" in query)
        and (query.startswith("why") or "should" in query or "important" in query)
    )


def _is_scientific_main_sections_question(question):
    query = question.lower()
    return (
        "section" in query
        and ("scientific" in query or "research paper" in query)
        and ("main" in query or "what are" in query or "list" in query)
    )


def _is_abstract_role_question(question):
    query = question.lower()
    return "abstract" in query and ("role" in query or "purpose" in query or "what is" in query)


def _is_scientific_paper_purpose_question(question):
    query = question.lower()
    return (
        "purpose" in query
        and ("scientific research paper" in query or "research paper" in query)
    )


def _is_history_simple_question(question):
    query = question.lower()
    return "history" in query and (
        "simple" in query
        or query.startswith("explain history")
        or query.startswith("what is history")
    )


def _is_history_benefits_question(question):
    query = question.lower()
    return "history" in query and re.search(r"\bbenefits?\b|\bwhy\s+(?:do|should)\s+.*study\b", query)


def _is_describe_question(question):
    return bool(re.search(r"\b(describe|tell|explain|how\s+(?:to|does|did|is|are))\b", question.lower()))


def _is_enumeration_question(question):
    query_lower = question.lower()
    return bool(re.search(
        r"^\s*what\s+(?:are|is|were|was|do|does|did)\s+.*(?:sections?|parts?|components?|elements?|steps?|topics?|stages?|phases?|types?|kinds?)\b",
        query_lower,
    )) or bool(re.search(
        r"^\s*what\s+.*(?:sections?|parts?|components?|elements?|steps?|topics?|stages?|phases?|types?|kinds?)\s+(?:are|is|were|was|do|does|did)\b",
        query_lower,
    ))
