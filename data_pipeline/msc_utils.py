import re
import ast
from data_pipeline.advanced_clean import clean_text

print("MSC UTILS LOADED")

MSC_PATTERN = re.compile(r"\b\d{2}[A-Z]\d{2}\b")

def clean_msc_list(msc_raw):
    if not msc_raw:
        return []

    if isinstance(msc_raw, list):
        text = " ".join(msc_raw)
    else:
        text = str(msc_raw)

    return MSC_PATTERN.findall(text.upper())


MSC_FULL_PATTERN = re.compile(r"\b\d{2}[A-Z]\d{2}\b")
MSC_MID_PATTERN  = re.compile(r"\b\d{2}[A-Z]\b")
MSC_ROOT_PATTERN = re.compile(r"\b\d{2}\b")

def extract_msc_codes(msc_raw):
    if not msc_raw:
        return [], [], []

    if isinstance(msc_raw, list):
        text = " ".join(str(x) for x in msc_raw)
    else:
        text = str(msc_raw)

    text = text.upper()

    full = sorted(set(MSC_FULL_PATTERN.findall(text)))
    mid  = sorted(set(MSC_MID_PATTERN.findall(text)))
    root = sorted(set(MSC_ROOT_PATTERN.findall(text)))

    return full, mid, root



def normalise_list_field(raw):
    if isinstance(raw, list):
        return raw
    return raw



def clean_refs(refs_raw):
    if not refs_raw:
        return []

    if isinstance(refs_raw, list):
        refs = refs_raw
    else:
        refs = [refs_raw]

    out = []
    for r in refs:
        if isinstance(r, str):
            t = clean_text(r).strip()
            if t:
                out.append(t)

    return sorted(set(out))
