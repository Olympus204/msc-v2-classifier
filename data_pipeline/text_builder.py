import re

def build_text(title, keywords):
    parts = []

    if title:
        parts.append(f"[Title] {title}")

    if keywords:
        if isinstance(keywords, str):
            kw = [k.strip() for k in re.split(r"[;,]", keywords)]
        else:
            kw = keywords
        kw = [k for k in kw if k]
        if kw:
            parts.append(f"[Keywords] {', '.join(kw)}")
        

    return "\n".join(parts)