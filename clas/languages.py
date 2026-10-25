from dataclasses import dataclass


@dataclass(frozen=True)
class Lang:
    code: str  # ISO 639-1
    name: str 
    flores: str 
    joshi: int 
    instruction: str  # "Respond in X." written in that language

REGISTRY = {l.code: l for l in [
    Lang("ar", "Arabic",     "arb_Arab", 5, "أجب بالعربية."),
    Lang("cs", "Czech",      "ces_Latn", 4, "Odpovězte v češtině."),
    Lang("de", "German",     "deu_Latn", 5, "Antworte auf Deutsch."),
    Lang("el", "Greek",      "ell_Grek", 3, "Απαντήστε στα ελληνικά."),
    Lang("en", "English",    "eng_Latn", 5, "Respond in English."),
    Lang("es", "Spanish",    "spa_Latn", 5, "Responde en español."),
    Lang("fa", "Persian",    "pes_Arab", 4, "به فارسی پاسخ دهید."),
    Lang("fr", "French",     "fra_Latn", 5, "Réponds en français."),
    Lang("hi", "Hindi",      "hin_Deva", 4, "हिंदी में उत्तर दें।"),
    Lang("it", "Italian",    "ita_Latn", 4, "Rispondi in italiano."),
    Lang("ja", "Japanese",   "jpn_Jpan", 5, "日本語で答えてください。"),
    Lang("ka", "Georgian",   "kat_Geor", 3, "უპასუხეთ ქართულად."),
    Lang("ko", "Korean",     "kor_Hang", 4, "한국어로 답하세요."),
    Lang("mt", "Maltese",    "mlt_Latn", 2, "Wieġeb bil-Malti."),
    Lang("no", "Norwegian",  "nob_Latn", 4, "Svar på norsk."),
    Lang("ru", "Russian",    "rus_Cyrl", 4, "Ответь на русском."),
    Lang("sk", "Slovak",     "slk_Latn", 4, "Odpovedajte po slovensky."),
    Lang("sw", "Swahili",    "swh_Latn", 2, "Jibu kwa Kiswahili."),
    Lang("th", "Thai",       "tha_Thai", 3, "ตอบในภาษาไทย"),
    Lang("tr", "Turkish",    "tur_Latn", 4, "Türkçe cevap verin."),
    Lang("uk", "Ukrainian",  "ukr_Cyrl", 4, "Дайте відповідь українською мовою."),
    Lang("vi", "Vietnamese", "vie_Latn", 4, "Trả lời bằng tiếng Việt."),
    Lang("zh", "Chinese",    "zho_Hans", 5, "用中文回答。"),
]}

CANDIDATES = [
    Lang("lb", "Luxembourgish", "ltz_Latn", 1, "Äntwert op Lëtzebuergesch."),
    Lang("ga", "Irish",         "gle_Latn", 2, "Freagair as Gaeilge."),
]

ALL = list(REGISTRY)

FLORES_PLUS_OVERRIDES = {
    "zho_Hans": "cmn_Hans", 
    "zho_Hant": "cmn_Hant",
}


def flores_config(code: str, dataset: str) -> str:
    """FLORES config name for a language under a given HF dataset."""
    f = REGISTRY[code].flores
    return FLORES_PLUS_OVERRIDES.get(f, f) if "flores_plus" in dataset else f

CLAS_CODE_OVERRIDES = {"cs": "cz"}


def clas_codes(code: str) -> list:
    """Identifiers to try when looking a language up in CLaS-Bench."""
    return list(dict.fromkeys([code, CLAS_CODE_OVERRIDES.get(code, code)]))

HIGH = ["ar", "de", "en", "es", "fr", "ja", "zh", "ru", "hi", "ko"]
LOW = ["cs", "el", "fa", "vi", "uk", "mt", "no", "sw", "th", "tr"]

SPARE = [c for c in ALL if c not in HIGH + LOW]

TRACK_LANGS = HIGH + LOW  # T3 sources
T3_TARGETS = ["ja", "de", "ru"]  # participants pick one

# FLORES-200 codes the LID classifier may emit for a registry language.
LID_ALIASES = {
    "zho_Hant": "zh", "yue_Hant": "zh", "cmn_Hans": "zh", "cmn_Hant": "zh",
    "fas_Arab": "fa",
    "arz_Arab": "ar", "ary_Arab": "ar", "acm_Arab": "ar",
    "apc_Arab": "ar", "ars_Arab": "ar", "aeb_Arab": "ar",
}

def get(code: str) -> Lang:
    return REGISTRY[code]


def flores_to_code() -> dict:
    """Map FLORES-200 labels (as emitted by fastText LID) to registry codes."""
    return {**{l.flores: l.code for l in REGISTRY.values()}, **LID_ALIASES}


def pairs(track: str, t3_target: str = "ja") -> list:
    """Return the (source, target) pairs evaluated for a given track.

    T1: HIGH -> HIGH, cross-lingual only
    T2: HIGH -> LOW, cross-lingual only
    T3: every track language -> one selected target
    Bonus (refusal): HIGH -> HIGH, including same-language pairs
    """
    track = track.lower()
    if track == "t1":
        return [(s, t) for s in HIGH for t in HIGH if s != t]
    if track == "t2":
        return [(s, t) for s in HIGH for t in LOW if s != t]
    if track == "t3":
        return [(s, t3_target) for s in TRACK_LANGS if s != t3_target]
    if track == "bonus":
        return [(s, t) for s in HIGH for t in HIGH]
    raise ValueError(f"unknown track: {track}")