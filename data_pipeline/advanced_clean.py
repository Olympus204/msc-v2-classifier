import re
import ast

FULL_PATTERN = re.compile(r"^[0-9]{2}[A-Z][0-9]{2}$")
ROOT_PATTERN = re.compile(r"^[0-9]{2}$")

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = text.replace("\u00ad", "")
    # Replace escaped characters
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove leftover braces
    text = text.replace("{", " ").replace("}", " ")

    # Replace backslashes used as linebreaks
    text = text.replace("\\", " ")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def clean_keywords(keywords_raw):
    if not keywords_raw:
        return []
    
    if isinstance(keywords_raw, str):
        keywords_raw = keywords_raw.strip()

    
    if isinstance(keywords_raw, list):
        cleaned = []
        for kw in keywords_raw:
            if not isinstance(kw, str):
                continue
            text = clean_text(kw).strip()
            if text:
                cleaned.append(text)
        return cleaned
    
    parts = re.split(r"[;,]", keywords_raw)
    cleaned = []
    for kw in parts:
        text = clean_text(kw).strip()
        if text:
            cleaned.append(text)
    return cleaned

def clean_msc_list(msc_raw):
    if not msc_raw:
        return[]
    
    if isinstance(msc_raw, list):
        cleaned = []
        for m in msc_raw:
            code = m.strip().upper()
            if FULL_PATTERN.match(code):
                cleaned.append(code)
        return cleaned
    
    parts = re.split(r"[;,]", msc_raw)
    cleaned = []
    for p in parts:
        code = p.strip().upper()
        if FULL_PATTERN.match(code):
            cleaned.append(code)
    return cleaned