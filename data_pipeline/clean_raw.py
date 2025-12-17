from data_pipeline.msc_utils import normalise_list_field, clean_refs, extract_msc_codes
from data_pipeline.text_builder import build_text
from data_pipeline.advanced_clean import clean_text, clean_keywords, clean_msc_list
import re

FULL_PATTERN = re.compile(r"^[0-9]{2}[A-Z][0-9]{2}$")
ROOT_PATTERN = re.compile(r"^[0-9]{2}$")

print("USING MSC UTILS FROM:", __file__)


def clean_row(row):
    title = clean_text(row.get("title",""))
    keywords = clean_keywords(row.get("keywords",""))
    msc_raw = row.get("msc","")
    refs_raw = normalise_list_field(row.get("refs", ""))
    if isinstance(refs_raw, str):
        ref_msc = extract_msc_codes(refs_raw)[0]
    else:
        ref_msc = []
    msc_raw = normalise_list_field(msc_raw)

    full_codes, mid_codes, root_codes = extract_msc_codes(msc_raw)

    if not full_codes:
        return None

    
    primary_full = full_codes[0]
    primary_mid = primary_full[:3]
    primary_root = primary_full[:2]

    mid_codes  = sorted({c[:3] for c in full_codes})
    root_codes = sorted({c[:2] for c in full_codes})


    text = build_text(title, keywords)

    if not text:
        return None
    return {
        "id": str(row.get("id","")),
        "text": text,
        "ref_msc": ref_msc,
        "primary_full": primary_full,
        "primary_mid": primary_mid,
        "primary_root": primary_root,
        "full_codes": full_codes,
        "mid_codes": mid_codes,
        "root_codes": root_codes,
    }