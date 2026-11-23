"""Cultural/regional MENA gazetteer and Arabic normalization for domain stratification."""

from __future__ import annotations

import re
from typing import Iterable, Set, Tuple

MENA_ONTOLOGY: Set[str] = {
    # Modern Arab States
    "مصر", "العراق", "سوريا", "سورية", "لبنان", "الأردن", "فلسطين", "السعودية", "الإمارات",
    "الكويت", "قطر", "البحرين", "عمان", "اليمن", "المغرب", "الجزائر", "تونس",
    "ليبيا", "السودان", "موريتانيا", "الصومال", "جيبوتي", "جزر القمر",
    # Major Cities
    "القاهرة", "دمشق", "بغداد", "القدس", "مكة", "المدينة", "الرياض", "بيروت",
    "صنعاء", "مسقط", "الدوحة", "المنامة", "طرابلس", "الخرطوم", "الرباط",
    "نواكشوط", "مقديشو", "فاس", "مراكش", "القيروان", "حلب", "الموصل", "البصرة",
    "الكوفة", "الإسكندرية", "قرطبة", "غرناطة", "إشبيلية", "طنجة", "وهران",
    "قسنطينة", "صيدا", "صور", "يافا", "حيفا", "نابلس", "غزة", "الخليل", "سامراء",
    # Historical Dynasties and Polities
    "الأموية", "العباسية", "الفاطمية", "الأيوبية", "المملوكية", "المرابطية",
    "الموحدية", "الأندلس", "الأغالبة", "الأدارسة", "الحمدانيين", "الزيانيين",
    "الحفصيين", "السلجوقية", "الزنكية", "الرستميين", "الطولونية", "الإخشيدية",
    # Demonyms and Attributes
    "عربي", "عربية", "مصري", "عراقي", "سوري", "لبناني", "أردني", "فلسطيني",
    "سعودي", "كويتي", "إماراتي", "قطري", "بحريني", "عماني", "يمني", "مغربي",
    "جزائري", "تونسي", "ليبي", "سوداني", "مسلم", "إسلامي",
}

_ARABIC_WORD_RE = re.compile(r"[\u0621-\u064A]+")
_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")  # Diacritics + tatweel
_ALEF_NORM_TABLE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})

# Keep 'المدينة' definite-only so generic 'مدينة' (e.g., New York City) never false-matches Medina
_KEEP_DEFINITE_ONLY: Set[str] = {"المدينه"}


def normalize_arabic_for_match(text: str) -> str:
    """Normalize Arabic diacritics, tatweel, alef/hamza, ya, and ta-marbuta."""
    text = _ARABIC_DIACRITICS_RE.sub("", str(text))
    return text.translate(_ALEF_NORM_TABLE)


def _prepare_ontology(ontology: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    single_words: Set[str] = set()
    multi_phrases: Set[str] = set()
    for term in ontology:
        norm = " ".join(_ARABIC_WORD_RE.findall(normalize_arabic_for_match(term)))
        if not norm:
            continue
        if " " in norm:
            multi_phrases.add(norm)
        else:
            single_words.add(norm)
            if norm.startswith("ال") and len(norm) > 3 and norm not in _KEEP_DEFINITE_ONLY:
                single_words.add(norm[2:])
    return single_words, multi_phrases


_DEFAULT_SINGLE_ONTOLOGY, _DEFAULT_PHRASE_ONTOLOGY = _prepare_ontology(MENA_ONTOLOGY)


def is_mena_entity(text: str, ontology: Iterable[str] = MENA_ONTOLOGY) -> bool:
    """Return True if any normalized Arabic whole word or phrase matches the MENA ontology."""
    if not isinstance(text, str):
        return False

    norm_text = normalize_arabic_for_match(text)
    tokens = _ARABIC_WORD_RE.findall(norm_text)
    if not tokens:
        return False

    if ontology is MENA_ONTOLOGY:
        single_ont, phrase_ont = _DEFAULT_SINGLE_ONTOLOGY, _DEFAULT_PHRASE_ONTOLOGY
    else:
        single_ont, phrase_ont = _prepare_ontology(ontology)

    words = set(tokens)
    words_base = {w[2:] if (w.startswith("ال") and len(w) > 3) else w for w in words}
    if (words | words_base) & single_ont:
        return True

    if phrase_ont:
        joined = f" {' '.join(tokens)} "
        if any(f" {phrase} " in joined for phrase in phrase_ont):
            return True

    return False
