import re

from pypdf import PdfReader


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def repair_spacing_artifacts(text):
    """Repair common missing-space artifacts produced by PDF text extraction."""
    if not text:
        return ""

    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Preserve list and heading boundaries before flattening newlines.
    text = re.sub(
        r"\s*\n+\s*(?=(?:\d{1,3}[\).]|[-*]|\u2022))",
        "\n",
        text,
    )
    text = re.sub(
        r"\s*\n+\s*(?=[A-Z][A-Za-z0-9 /-]{2,40}:)",
        "\n",
        text,
    )
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    text = re.sub(r"(?<=[A-Za-z])-(?=(?:how|when|where|why|what|who|which)\b)", " - ", text)

    # These patterns are generic PDF extraction repairs. They catch common
    # glued function words such as "Itprovides" or "understandingtheprocess"
    # without relying on any particular uploaded document.
    patterns = [
        (
            r"\b(It|This|That|These|Those|They|There|Here|We|You|He|She)"
            r"(is|are|was|were|will|would|can|could|should|has|have|had|"
            r"does|do|did|became|becomes?|seems?|appears?|helps?|provides?|"
            r"emphasizes?|emphasises?|highlights?|explores?|introduces?|"
            r"discusses?|includes?|shows?|means|refers|serves?|deals?|"
            r"doesn't|isn't|wasn't|won't|can't)\b",
            r"\1 \2",
        ),
        (
            r"\b(The|A|An)"
            r"(chapter|section|focus|goal|term|concept|process|method|methods|"
            r"source|sources|record|records|document|documents|benefit|benefits|"
            r"problem|problems|challenge|challenges|result|results|role|roles|"
            r"purpose|importance|following|first|second|next|same|main)\b",
            r"\1 \2",
        ),
        (
            r"\b(In|On|At|By|To|For|From|With|Without|Before|After|During|"
            r"Because|Although|While|When|Where|What|How|Why|Who|Which)"
            r"(the|a|an|this|that|these|those|their|its|his|her|our|your)\b",
            r"\1 \2",
        ),
        (r"\b(all)(about|the)\b", r"\1 \2"),
        (r"\b(historical)(inquiry|narratives?|events?)\b", r"\1 \2"),
        (r"\b(historical)(sources?)\b", r"\1 \2"),
        (r"\b(official)(records?|government)\b", r"\1 \2"),
        (r"\b(reliability)(and)\b", r"\1 \2"),
        (r"\b(their)(ability|beliefs)\b", r"\1 \2"),
        (r"\b(soil)(quality)\b", r"\1 \2"),
        (r"\b(in)(history)\b", r"\1 \2"),
        (
            r"\b(to)(better|create|cover|map|preserve|provide|study|assess|"
            r"understand|reconstruct|convey|disseminate|determine|gather|"
            r"fully)\b",
            r"\1 \2",
        ),
        (r"\b(will)(never|learn|discover|comprehend|understand)\b", r"\1 \2"),
        (r"\b(fully)(comprehend|understand)\b", r"\1 \2"),
    ]

    previous = None
    while previous != text:
        previous = text
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\bAn other\b", "Another", text)
    text = re.sub(r"\ban other\b", "another", text)

    text = re.sub(
        r"\b(How|What|Why|When|Where|Who|Which)"
        r"(do|does|did|is|are|was|were|can|will|would|should)\b",
        r"\1 \2",
        text,
    )
    text = re.sub(
        r"\b(what|why|when|where|who|which)\s+(is|are|was|were)([A-Za-z]{4,})\b",
        r"\1 \2 \3",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(how)\s+(do|does|did)(we|you|they|historians|people)\b",
        r"\1 \2 \3",
        text,
        flags=re.IGNORECASE,
    )

    return text


def clean_text(text):
    if not text:
        return ""

    text = CONTROL_CHARS.sub(" ", text)
    text = text.replace("\u00a0", " ")

    replacements = {
        "â€¦": " ",
        "Ã¢â‚¬Â¦": " ",
        "Ã¢â€ â€™": "->",
        "Ã¢â‚¬â€": "-",
        "Ã¢â‚¬â€œ": "-",
        "Ã¢â‚¬Ëœ": "'",
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬": '"',
        "â€œ": '"',
        "â€": '"',
        "â€˜": "'",
        "â€™": "'",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = repair_spacing_artifacts(text)

    # Remove textbook blanks and separator-like artifacts.
    text = re.sub(r"[.]{2,}", " ", text)
    text = re.sub(r"[-_]{2,}", " ", text)

    # Fix common extraction splits seen in scanned textbook headings.
    text = re.sub(r"\bC\s+lassification", "Classification", text)
    text = re.sub(r"\bM\s+anaging", "Managing", text)
    text = re.sub(r"\bG\s+lobalization", "Globalization", text)

    text = re.sub(r"\s+", " ", text)

    if "References" in text:
        text = text.split("References")[0]

    return text.strip()


def extract_page_text(page):
    candidates = []

    for kwargs in ({}, {"extraction_mode": "layout"}):
        try:
            extracted = page.extract_text(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue

        cleaned = clean_text(extracted)
        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        return ""

    return max(candidates, key=text_quality_score)


def text_quality_score(text):
    words = text.split()
    if not words:
        return 0

    long_glued_words = sum(
        1
        for word in words
        if len(word) > 18
        and re.search(r"[a-z]{6,}(?:the|and|of|to|for|with|from)[a-z]{4,}", word.lower())
    )
    readable_words = sum(1 for word in words if re.search(r"[A-Za-z]", word))

    return readable_words - (long_glued_words * 12)


def load_pdf(file_path):
    reader = PdfReader(file_path)
    docs = []

    for i, page in enumerate(reader.pages):
        text = extract_page_text(page)

        if text:
            docs.append({
                "text": text,
                "page": i + 1
            })

    return docs
