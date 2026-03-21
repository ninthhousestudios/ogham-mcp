"""Date and entity extraction for memory enrichment.

Pure regex, no LLM calls. Runs at store_memory time to enrich
metadata (dates) and tags (entities) automatically.
"""

import logging
import re
from datetime import datetime

import parsedatetime
from stop_words import AVAILABLE_LANGUAGES, get_stop_words

logger = logging.getLogger(__name__)

# parsedatetime calendar for relative date parsing (zero deps, English)
_PDT_CAL = parsedatetime.Calendar()

# --- Stopwords (34 languages, loaded once) ---

_STOP_WORDS: set[str] = set()
for _lang in AVAILABLE_LANGUAGES:
    _STOP_WORDS.update(get_stop_words(_lang))

# --- Date extraction ---

_ISO_DATE = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b")

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_NATURAL_DATE = re.compile(
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}\b"
    rf"|\b\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}\b",
    re.IGNORECASE,
)

# Patterns for relative date phrases to feed to parsedatetime
_RELATIVE_PHRASES = re.compile(
    r"\b((?:last|next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|week|month|year)"
    r"|yesterday|today|tomorrow"
    r"|\d+\s+(?:days?|weeks?|months?|years?)\s+ago"
    r"|in\s+\d+\s+(?:days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)

# --- Multilingual day names → day index (0=Sun..6=Sat) ---
# Covers 16 languages matching the importance scoring dictionaries.
# Each entry maps a lowercase day name to its dow index.

_DAY_NAMES: dict[str, int] = {}
_DAY_NAMES_BY_LANG: dict[str, dict[str, int]] = {
    "en": {
        "sunday": 0,
        "monday": 1,
        "tuesday": 2,
        "wednesday": 3,
        "thursday": 4,
        "friday": 5,
        "saturday": 6,
    },
    "de": {
        "sonntag": 0,
        "montag": 1,
        "dienstag": 2,
        "mittwoch": 3,
        "donnerstag": 4,
        "freitag": 5,
        "samstag": 6,
        # Adverbial forms ("montags" = "on Mondays")
        "sonntags": 0,
        "montags": 1,
        "dienstags": 2,
        "mittwochs": 3,
        "donnerstags": 4,
        "freitags": 5,
        "samstags": 6,
    },
    "fr": {
        "dimanche": 0,
        "lundi": 1,
        "mardi": 2,
        "mercredi": 3,
        "jeudi": 4,
        "vendredi": 5,
        "samedi": 6,
    },
    "es": {
        "domingo": 0,
        "lunes": 1,
        "martes": 2,
        "miércoles": 3,
        "miercoles": 3,
        "jueves": 4,
        "viernes": 5,
        "sábado": 6,
        "sabado": 6,
    },
    "it": {
        "domenica": 0,
        "lunedì": 1,
        "lunedi": 1,
        "martedì": 2,
        "martedi": 2,
        "mercoledì": 3,
        "mercoledi": 3,
        "giovedì": 4,
        "giovedi": 4,
        "venerdì": 5,
        "venerdi": 5,
        "sabato": 6,
    },
    "pt": {
        "domingo": 0,
        "segunda": 1,
        "segunda-feira": 1,
        "terça": 2,
        "terça-feira": 2,
        "terca": 2,
        "terca-feira": 2,
        "quarta": 3,
        "quarta-feira": 3,
        "quinta": 4,
        "quinta-feira": 4,
        "sexta": 5,
        "sexta-feira": 5,
        "sábado": 6,
        "sabado": 6,
    },
    "nl": {
        "zondag": 0,
        "maandag": 1,
        "dinsdag": 2,
        "woensdag": 3,
        "donderdag": 4,
        "vrijdag": 5,
        "zaterdag": 6,
    },
    "pl": {
        "niedziela": 0,
        "poniedziałek": 1,
        "poniedzialek": 1,
        "wtorek": 2,
        "środa": 3,
        "sroda": 3,
        "czwartek": 4,
        "piątek": 5,
        "piatek": 5,
        "sobota": 6,
    },
    "tr": {
        "pazar": 0,
        "pazartesi": 1,
        "salı": 2,
        "sali": 2,
        "çarşamba": 3,
        "carsamba": 3,
        "perşembe": 4,
        "persembe": 4,
        "cuma": 5,
        "cumartesi": 6,
    },
    "ru": {
        "воскресенье": 0,
        "понедельник": 1,
        "вторник": 2,
        "среда": 3,
        "четверг": 4,
        "пятница": 5,
        "суббота": 6,
    },
    "uk": {
        "неділя": 0,
        "понеділок": 1,
        "вівторок": 2,
        "середа": 3,
        "четвер": 4,
        "п'ятниця": 5,
        "субота": 6,
    },
    "ga": {  # Irish
        "domhnach": 0,
        "luan": 1,
        "máirt": 2,
        "mairt": 2,
        "céadaoin": 3,
        "ceadaoin": 3,
        "déardaoin": 4,
        "deardaoin": 4,
        "aoine": 5,
        "satharn": 6,
        # With "dé" prefix
        "dé luain": 1,
        "dé máirt": 2,
        "dé céadaoin": 3,
        "dé haoine": 5,
        "dé sathairn": 6,
    },
    "ar": {
        "الأحد": 0,
        "الاثنين": 1,
        "الثلاثاء": 2,
        "الأربعاء": 3,
        "الخميس": 4,
        "الجمعة": 5,
        "السبت": 6,
    },
    "zh": {
        "星期日": 0,
        "星期天": 0,
        "星期一": 1,
        "星期二": 2,
        "星期三": 3,
        "星期四": 4,
        "星期五": 5,
        "星期六": 6,
        "周日": 0,
        "周一": 1,
        "周二": 2,
        "周三": 3,
        "周四": 4,
        "周五": 5,
        "周六": 6,
    },
    "ja": {
        "日曜日": 0,
        "月曜日": 1,
        "火曜日": 2,
        "水曜日": 3,
        "木曜日": 4,
        "金曜日": 5,
        "土曜日": 6,
        "日曜": 0,
        "月曜": 1,
        "火曜": 2,
        "水曜": 3,
        "木曜": 4,
        "金曜": 5,
        "土曜": 6,
    },
    "ko": {
        "일요일": 0,
        "월요일": 1,
        "화요일": 2,
        "수요일": 3,
        "목요일": 4,
        "금요일": 5,
        "토요일": 6,
    },
    "hi": {"रविवार": 0, "सोमवार": 1, "मंगलवार": 2, "बुधवार": 3, "गुरुवार": 4, "शुक्रवार": 5, "शनिवार": 6},
}

# Flatten into single lookup
for _lang_days in _DAY_NAMES_BY_LANG.values():
    _DAY_NAMES.update(_lang_days)

# Multilingual "every" / "each" / "weekly" keywords
_EVERY_WORDS: set[str] = {
    # English
    "every",
    "each",
    "weekly",
    # German
    "jeden",
    "jede",
    "jedes",
    "wöchentlich",
    "wochentlich",
    # French
    "chaque",
    "tous les",
    "hebdomadaire",
    # Spanish
    "cada",
    "todos los",
    "semanal",
    # Italian
    "ogni",
    "settimanale",
    # Portuguese
    "cada",
    "todo",
    "toda",
    "semanal",
    # Dutch
    "elke",
    "iedere",
    "wekelijks",
    # Polish
    "każdy",
    "kazdy",
    "każda",
    "kazda",
    "co tydzień",
    # Turkish
    "her",
    "haftalık",
    "haftalik",
    # Russian
    "каждый",
    "каждая",
    "каждое",
    "еженедельно",
    # Ukrainian
    "кожний",
    "кожна",
    "кожне",
    "щотижня",
    # Irish
    "gach",
    # Arabic
    "كل",
    # Chinese
    "每",
    "每周",
    "每個",
    # Japanese
    "毎",
    "毎週",
    # Korean
    "매",
    "매주",
    # Hindi
    "हर",
    "प्रत्येक",
    "साप्ताहिक",
}


def extract_recurrence(content: str) -> list[int] | None:
    """Extract recurring day-of-week patterns from content.

    Returns sorted list of day indices (0=Sun..6=Sat) or None.
    Supports 16 languages via static dictionary lookup.
    No LLM calls.
    """
    content_lower = content.lower()
    days_found: set[int] = set()

    # Check if content contains an "every" keyword
    has_every = any(word in content_lower for word in _EVERY_WORDS)

    if not has_every:
        # German adverbial forms ("montags") imply recurrence without "jeden"
        for name, idx in _DAY_NAMES.items():
            if name.endswith("s") and name in _DAY_NAMES_BY_LANG.get("de", {}):
                if re.search(rf"\b{re.escape(name)}\b", content_lower):
                    days_found.add(idx)
        if not days_found:
            return None

    # Scan for day names in content
    if not days_found:
        # Longest names first to avoid partial matches (e.g. "dé luain" before "luan")
        for name, idx in sorted(_DAY_NAMES.items(), key=lambda x: -len(x[0])):
            if len(name) < 2:
                continue
            # CJK/Arabic/Cyrillic/Devanagari: substring match is fine (no spaces)
            if name[0].isascii() and name[0].isalpha():
                # Alphabetic: use word boundary to avoid "vendredi" matching "di"
                if re.search(rf"\b{re.escape(name)}\b", content_lower):
                    days_found.add(idx)
            elif name in content_lower:
                days_found.add(idx)

    return sorted(days_found) if days_found else None


TEMPORAL_KEYWORDS = frozenset(
    {
        "when",
        "date",
        "time",
        "ago",
        "last",
        "before",
        "after",
        "during",
        "since",
        "until",
        "yesterday",
        "tomorrow",
        "week",
        "month",
        "year",
    }
)


def extract_dates(content: str) -> list[str]:
    """Extract date strings from content. Returns sorted normalised ISO dates."""
    dates: set[str] = set()

    for match in _ISO_DATE.finditer(content):
        dates.add(match.group(1).replace("/", "-"))

    for match in _NATURAL_DATE.finditer(content):
        try:
            raw = re.sub(r"(st|nd|rd|th)", "", match.group(0))
            raw = raw.replace(",", "").strip()
            for fmt in ("%B %d %Y", "%d %B %Y", "%b %d %Y", "%d %b %Y"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    dates.add(dt.strftime("%Y-%m-%d"))
                    break
                except ValueError:
                    continue
        except Exception:
            logger.debug("Failed to parse natural date: %s", match.group(0))
            continue

    # Relative dates via parsedatetime ("last Tuesday", "yesterday", "two weeks ago")
    # Only attempt if no absolute dates found and content has temporal keywords
    if not dates:
        for phrase in _RELATIVE_PHRASES.findall(content):
            try:
                result, status = _PDT_CAL.parse(phrase)
                if status:
                    dt = datetime(*result[:6])
                    dates.add(dt.strftime("%Y-%m-%d"))
            except Exception:
                logger.debug("Failed to parse relative date: %s", phrase)
                continue

    return sorted(dates)


def has_temporal_intent(query: str) -> bool:
    """Check if a query has temporal intent (when, date, time, etc.)."""
    words = set(query.lower().split())
    return bool(words & TEMPORAL_KEYWORDS)


# --- Multi-hop temporal detection + entity extraction ---

_MULTI_HOP_PATTERNS = re.compile(
    r"how many (?:months?|weeks?|days?|years?) (?:between|since|before|after|passed)"
    r"|which (?:happened|came|occurred|did I do|event|did I|trip|device|gift|project) .{0,20}first"
    r"|what (?:is|was) the order"
    r"|how long (?:between|since|before|after|had passed|did it take)"
    r"|which .{0,30} (?:earlier|later|before|after)"
    r"|how (?:old|long) was I when",
    re.IGNORECASE,
)

# Pattern to extract entity anchors from "between X and Y" / "first, X or Y" queries
_BETWEEN_PATTERN = re.compile(
    r"between\s+(.+?)\s+and\s+(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)
_OR_PATTERN = re.compile(
    r"(?:first|earlier|later),?\s+(.+?)\s+or\s+(?:the\s+)?(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)
_SINCE_PATTERN = re.compile(
    r"(?:since|after|before)\s+(.+?)(?:\s+(?:did|was|had|have|how|when|$))",
    re.IGNORECASE,
)
# "how many days before X did Y" → extract X and Y
_BEFORE_DID_PATTERN = re.compile(
    r"(?:before|after)\s+(.+?)\s+did\s+(?:I\s+)?(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)
# "how long had I been X when Y" → extract X and Y
_BEEN_WHEN_PATTERN = re.compile(
    r"been\s+(.+?)\s+when\s+(?:I\s+)?(.+?)(?:\s*[?.!]|$)",
    re.IGNORECASE,
)


_ORDERING_PATTERNS = re.compile(
    r"what is the order of"
    r"|from earliest to latest"
    r"|from latest to earliest"
    r"|in (?:chronological|reverse) order",
    re.IGNORECASE,
)


def is_ordering_query(query: str) -> bool:
    """Detect if a query asks for chronological ordering of multiple events."""
    return bool(_ORDERING_PATTERNS.search(query))


def is_multi_hop_temporal(query: str) -> bool:
    """Detect if a query requires multi-hop temporal reasoning (comparing two events)."""
    return bool(_MULTI_HOP_PATTERNS.search(query))


def extract_query_anchors(query: str) -> list[str]:
    """Extract entity anchors from a multi-hop temporal query.

    Returns a list of 1-2 noun phrases representing the events being compared.
    Uses regex patterns -- no LLM or NLP library needed.
    """
    anchors = []

    # "between X and Y"
    match = _BETWEEN_PATTERN.search(query)
    if match:
        anchors = [match.group(1).strip(), match.group(2).strip()]

    # "X or Y" (from "which happened first, X or Y")
    if not anchors:
        match = _OR_PATTERN.search(query)
        if match:
            anchors = [match.group(1).strip(), match.group(2).strip()]

    # "before X did Y" / "after X did Y"
    if not anchors:
        match = _BEFORE_DID_PATTERN.search(query)
        if match:
            anchors = [match.group(1).strip(), match.group(2).strip()]

    # "been X when Y"
    if not anchors:
        match = _BEEN_WHEN_PATTERN.search(query)
        if match:
            anchors = [match.group(1).strip(), match.group(2).strip()]

    # "since/after X" (single anchor fallback)
    if not anchors:
        match = _SINCE_PATTERN.search(query)
        if match:
            anchors = [match.group(1).strip()]

    # Clean up common filler words from anchors
    cleaned = []
    for a in anchors:
        a = re.sub(r"^(?:the|my|a|an|I|i)\s+", "", a, flags=re.IGNORECASE)
        a = re.sub(r"\s+(?:the|my|a|an)\s+", " ", a, flags=re.IGNORECASE)
        a = a.strip().rstrip(",.")
        if len(a) > 2:
            cleaned.append(a)

    return cleaned


# --- Temporal query resolution ---

# Patterns for temporal range phrases ("in January", "last March", "four months ago")
_MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def resolve_temporal_query(
    query: str,
    reference_date: datetime | None = None,
) -> tuple[str, str] | None:
    """Resolve a temporal query into a date range (start, end) as ISO strings.

    Uses parsedatetime first (free, handles ~80% of cases).
    Falls back to LLM (GPT-4o-mini via litellm) for complex expressions.
    Returns None if no temporal range can be resolved.
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Try parsedatetime first
    result = _resolve_with_parsedatetime(query, reference_date)
    if result:
        return result

    # Try month name extraction ("in January", "last March")
    result = _resolve_month_reference(query, reference_date)
    if result:
        return result

    # LLM fallback (optional -- graceful if litellm not installed)
    if has_temporal_intent(query):
        result = _resolve_with_llm(query, reference_date)
        if result:
            return result

    return None


def _resolve_with_parsedatetime(query: str, reference_date: datetime) -> tuple[str, str] | None:
    """Try parsedatetime for relative date expressions."""
    # Look for range-like phrases
    range_patterns = [
        # "between X and Y"
        re.compile(r"between\s+(.+?)\s+and\s+(.+?)(?:\s*[?.!]|$)", re.IGNORECASE),
        # "from X to Y"
        re.compile(r"from\s+(.+?)\s+to\s+(.+?)(?:\s*[?.!]|$)", re.IGNORECASE),
    ]

    for pattern in range_patterns:
        match = pattern.search(query)
        if match:
            start_result, start_status = _PDT_CAL.parse(match.group(1), reference_date)
            end_result, end_status = _PDT_CAL.parse(match.group(2), reference_date)
            if start_status and end_status:
                start = datetime(*start_result[:6])
                end = datetime(*end_result[:6])
                return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # Single point phrases → create a ±window
    for phrase in _RELATIVE_PHRASES.findall(query):
        result, status = _PDT_CAL.parse(phrase, reference_date)
        if status:
            dt = datetime(*result[:6])
            # Create a 7-day window around the resolved date
            from datetime import timedelta as _td

            start = dt - _td(days=3)
            end = dt + _td(days=4)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # "N months/weeks ago" → create a month/week window
    _WORD_NUMBERS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "a": 1,
        "an": 1,
    }
    ago_match = re.search(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)"
        r"\s+(months?|weeks?|years?)\s+ago",
        query,
        re.IGNORECASE,
    )
    if ago_match:
        raw = ago_match.group(1).lower()
        num = _WORD_NUMBERS.get(raw, None)
        if num is None:
            num = int(raw)
        unit = ago_match.group(2).lower().rstrip("s")
        from datetime import timedelta

        if unit == "month":
            # Approximate: go back N months, create a 30-day window
            start = reference_date - timedelta(days=num * 30 + 15)
            end = (
                reference_date - timedelta(days=(num - 1) * 30 - 15) if num > 1 else reference_date
            )
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        elif unit == "week":
            start = reference_date - timedelta(weeks=num, days=3)
            end = reference_date - timedelta(weeks=num - 1) if num > 1 else reference_date
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        elif unit == "year":
            start = reference_date.replace(year=reference_date.year - num, month=1, day=1)
            end = reference_date.replace(year=reference_date.year - num, month=12, day=31)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    return None


def _resolve_month_reference(query: str, reference_date: datetime) -> tuple[str, str] | None:
    """Resolve 'in January', 'last March', 'this April' to a month range."""
    query_lower = query.lower()

    for month_name, month_num in _MONTH_NAMES.items():
        if month_name not in query_lower:
            continue
        # Determine the year
        if "last" in query_lower or "previous" in query_lower:
            year = reference_date.year - 1
        elif "next" in query_lower:
            year = reference_date.year + 1
        else:
            # Default: most recent occurrence of that month
            year = reference_date.year
            if month_num > reference_date.month:
                year -= 1

        import calendar

        _, last_day = calendar.monthrange(year, month_num)
        start = f"{year}-{month_num:02d}-01"
        end = f"{year}-{month_num:02d}-{last_day:02d}"
        return (start, end)

    return None


def _resolve_with_llm(query: str, reference_date: datetime) -> tuple[str, str] | None:
    """LLM fallback for complex temporal expressions. Optional -- returns None if unavailable.

    Uses TEMPORAL_LLM_MODEL env var to determine the model. If empty, LLM
    fallback is disabled (parsedatetime only). Self-hosters can set this to
    an Ollama model (e.g. "ollama/llama3.2") or any litellm-compatible string.
    """
    from ogham.config import settings

    model = settings.temporal_llm_model
    if not model:
        logger.debug("TEMPORAL_LLM_MODEL not set, skipping LLM temporal resolution")
        return None

    try:
        import litellm
    except ImportError:
        logger.debug("litellm not installed, skipping LLM temporal resolution")
        return None

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Today is {reference_date.strftime('%Y-%m-%d')}. "
                        "Extract the date range from the user's query. "
                        'Return ONLY JSON: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} '
                        "or null if no date range can be determined."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=50,
            temperature=0,
        )
        import json

        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        if data and "start" in data and "end" in data:
            return (data["start"], data["end"])
    except Exception:
        logger.debug("LLM temporal resolution failed for: %s", query[:60])

    return None


# --- Shared regex patterns (used by both importance and entity extraction) ---

_CAMEL_CASE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+\b")
_FILE_PATH = re.compile(r"(?:\.{0,2}/)?(?:[\w@.-]+/)+[\w@.-]+\.\w+")
_ERROR_TYPE = re.compile(r"\b\w*(?:Error|Exception)\b")

# --- Importance scoring (8 languages) ---

_DECISION_WORDS_BY_LANG = {
    "en": {
        "decided",
        "chose",
        "choosing",
        "switched",
        "migrated",
        "selected",
        "picked",
        "opted",
        "replaced",
        "adopted",
    },
    "de": {
        "entschieden",
        "gewählt",
        "gewechselt",
        "migriert",
        "ausgewählt",
        "ersetzt",
        "umgestiegen",
        "beschlossen",
        "festgelegt",
        "übernommen",
    },
    "fr": {
        "décidé",
        "choisi",
        "migré",
        "sélectionné",
        "remplacé",
        "adopté",
        "opté",
        "basculé",
        "changé",
        "retenu",
    },
    "it": {
        "deciso",
        "scelto",
        "migrato",
        "selezionato",
        "sostituito",
        "adottato",
        "optato",
        "cambiato",
        "passato",
        "configurato",
    },
    "es": {
        "decidido",
        "elegido",
        "migrado",
        "seleccionado",
        "reemplazado",
        "adoptado",
        "optado",
        "cambiado",
        "escogido",
        "configurado",
    },
    "ar": {"قرر", "اختار", "انتقل", "حدد", "استبدل", "اعتمد", "بدّل", "غيّر", "هاجر", "تبنى"},
    "tr": {
        "karar",
        "seçti",
        "geçti",
        "taşıdı",
        "değiştirdi",
        "benimsedi",
        "tercih",
        "belirledi",
        "yapılandırdı",
        "uyguladı",
    },
    "zh": {"决定", "选择", "迁移", "替换", "采用", "切换", "配置", "选定", "更换", "转换"},
    "pt": {
        "decidido",
        "escolhido",
        "migrado",
        "selecionado",
        "substituído",
        "adotado",
        "optado",
        "mudado",
        "trocado",
        "configurado",
    },
    "ja": {"決定", "選択", "移行", "置換", "採用", "切替", "設定", "選定", "変更", "移管"},
    "ko": {"결정", "선택", "마이그레이션", "교체", "채택", "전환", "구성", "설정", "변경", "이전"},
    "ru": {
        "решили",
        "выбрали",
        "перешли",
        "мигрировали",
        "заменили",
        "приняли",
        "перенесли",
        "настроили",
        "изменили",
        "установили",
    },
    "pl": {
        "zdecydowali",
        "wybrali",
        "zmigrowali",
        "zamienili",
        "przyjęli",
        "przeszli",
        "zmienili",
        "skonfigurowali",
        "zastąpili",
        "wdrożyli",
    },
    "uk": {
        "вирішили",
        "обрали",
        "мігрували",
        "замінили",
        "прийняли",
        "перейшли",
        "змінили",
        "налаштували",
        "встановили",
        "впровадили",
    },
    "hi": {
        "निर्णय",
        "चुना",
        "माइग्रेट",
        "बदला",
        "अपनाया",
        "चयन",
        "स्थापित",
        "कॉन्फ़िगर",
        "परिवर्तन",
        "स्विच",
    },
    "nl": {
        "besloten",
        "gekozen",
        "gemigreerd",
        "geselecteerd",
        "vervangen",
        "overgestapt",
        "gewijzigd",
        "geconfigureerd",
        "aangenomen",
        "ingesteld",
    },
}

_ERROR_WORDS_BY_LANG = {
    "en": {
        "error",
        "exception",
        "failed",
        "failure",
        "bug",
        "crash",
        "broken",
        "traceback",
        "timeout",
        "denied",
    },
    "de": {
        "fehler",
        "ausnahme",
        "fehlgeschlagen",
        "absturz",
        "defekt",
        "zeitüberschreitung",
        "abgelehnt",
        "abgebrochen",
        "ungültig",
        "kaputt",
    },
    "fr": {
        "erreur",
        "exception",
        "échoué",
        "échec",
        "bogue",
        "plantage",
        "cassé",
        "délai",
        "refusé",
        "invalide",
    },
    "it": {
        "errore",
        "eccezione",
        "fallito",
        "guasto",
        "crash",
        "rotto",
        "scaduto",
        "negato",
        "invalido",
        "interrotto",
    },
    "es": {
        "error",
        "excepción",
        "fallido",
        "fallo",
        "roto",
        "caída",
        "tiempo",
        "denegado",
        "inválido",
        "bloqueado",
    },
    "ar": {"خطأ", "استثناء", "فشل", "عطل", "انهيار", "مكسور", "مهلة", "مرفوض", "غير صالح", "توقف"},
    "tr": {
        "hata",
        "istisna",
        "başarısız",
        "arıza",
        "çöktü",
        "bozuk",
        "zaman aşımı",
        "reddedildi",
        "geçersiz",
        "durdu",
    },
    "zh": {"错误", "异常", "失败", "故障", "崩溃", "超时", "拒绝", "无效", "中断", "报错"},
    "pt": {
        "erro",
        "exceção",
        "falhou",
        "falha",
        "bug",
        "travou",
        "quebrado",
        "timeout",
        "negado",
        "inválido",
    },
    "ja": {
        "エラー",
        "例外",
        "失敗",
        "障害",
        "クラッシュ",
        "タイムアウト",
        "拒否",
        "無効",
        "中断",
        "バグ",
    },
    "ko": {"오류", "예외", "실패", "장애", "크래시", "타임아웃", "거부", "무효", "중단", "버그"},
    "ru": {
        "ошибка",
        "исключение",
        "сбой",
        "отказ",
        "крах",
        "таймаут",
        "отклонено",
        "недействительно",
        "прервано",
        "баг",
    },
    "pl": {
        "błąd",
        "wyjątek",
        "niepowodzenie",
        "awaria",
        "crash",
        "timeout",
        "odmowa",
        "nieprawidłowy",
        "przerwano",
        "usterka",
    },
    "uk": {
        "помилка",
        "виняток",
        "збій",
        "відмова",
        "крах",
        "таймаут",
        "відхилено",
        "недійсний",
        "перервано",
        "баг",
    },
    "hi": {"त्रुटि", "अपवाद", "विफल", "विफलता", "क्रैश", "टाइमआउट", "अस्वीकृत", "अमान्य", "बाधित", "बग"},
    "nl": {
        "fout",
        "uitzondering",
        "mislukt",
        "storing",
        "crash",
        "kapot",
        "timeout",
        "geweigerd",
        "ongeldig",
        "afgebroken",
    },
}

_ARCHITECTURE_WORDS_BY_LANG = {
    "en": {
        "design",
        "pattern",
        "refactor",
        "architecture",
        "restructure",
        "modular",
        "decouple",
        "abstract",
        "interface",
        "migrate",
    },
    "de": {
        "entwurf",
        "muster",
        "refaktorisierung",
        "architektur",
        "umstrukturierung",
        "modular",
        "entkoppeln",
        "abstraktion",
        "schnittstelle",
        "migration",
    },
    "fr": {
        "conception",
        "modèle",
        "refactorisation",
        "architecture",
        "restructuration",
        "modulaire",
        "découpler",
        "abstraction",
        "interface",
        "migration",
    },
    "it": {
        "progettazione",
        "modello",
        "refactoring",
        "architettura",
        "ristrutturazione",
        "modulare",
        "disaccoppiare",
        "astrazione",
        "interfaccia",
        "migrazione",
    },
    "es": {
        "diseño",
        "patrón",
        "refactorización",
        "arquitectura",
        "reestructuración",
        "modular",
        "desacoplar",
        "abstracción",
        "interfaz",
        "migración",
    },
    "ar": {
        "تصميم",
        "نمط",
        "إعادة هيكلة",
        "بنية",
        "معمارية",
        "وحدات",
        "فصل",
        "تجريد",
        "واجهة",
        "ترحيل",
    },
    "tr": {
        "tasarım",
        "kalıp",
        "yeniden düzenleme",
        "mimari",
        "yeniden yapılandırma",
        "modüler",
        "ayrıştırma",
        "soyutlama",
        "arayüz",
        "geçiş",
    },
    "zh": {"设计", "模式", "重构", "架构", "解耦", "模块化", "抽象", "接口", "迁移", "拆分"},
    "pt": {
        "projeto",
        "padrão",
        "refatoração",
        "arquitetura",
        "reestruturação",
        "modular",
        "desacoplar",
        "abstração",
        "interface",
        "migração",
    },
    "ja": {
        "設計",
        "パターン",
        "リファクタリング",
        "アーキテクチャ",
        "再構成",
        "モジュール",
        "分離",
        "抽象",
        "インターフェース",
        "移行",
    },
    "ko": {
        "설계",
        "패턴",
        "리팩토링",
        "아키텍처",
        "재구성",
        "모듈",
        "분리",
        "추상화",
        "인터페이스",
        "마이그레이션",
    },
    "ru": {
        "проектирование",
        "паттерн",
        "рефакторинг",
        "архитектура",
        "реструктуризация",
        "модульный",
        "разделение",
        "абстракция",
        "интерфейс",
        "миграция",
    },
    "pl": {
        "projekt",
        "wzorzec",
        "refaktoryzacja",
        "architektura",
        "restrukturyzacja",
        "modularny",
        "rozdzielenie",
        "abstrakcja",
        "interfejs",
        "migracja",
    },
    "uk": {
        "проєктування",
        "патерн",
        "рефакторинг",
        "архітектура",
        "реструктуризація",
        "модульний",
        "розділення",
        "абстракція",
        "інтерфейс",
        "міграція",
    },
    "hi": {
        "डिजाइन",
        "पैटर्न",
        "रीफैक्टरिंग",
        "आर्किटेक्चर",
        "पुनर्गठन",
        "मॉड्यूलर",
        "विभाजन",
        "एब्स्ट्रैक्शन",
        "इंटरफेस",
        "माइग्रेशन",
    },
    "nl": {
        "ontwerp",
        "patroon",
        "refactoring",
        "architectuur",
        "herstructurering",
        "modulair",
        "ontkoppelen",
        "abstractie",
        "interface",
        "migratie",
    },
}

# Flatten all languages into single sets for fast lookup
_DECISION_WORDS: set[str] = set()
for _words in _DECISION_WORDS_BY_LANG.values():
    _DECISION_WORDS.update(_words)

_ERROR_WORDS: set[str] = set()
for _words in _ERROR_WORDS_BY_LANG.values():
    _ERROR_WORDS.update(_words)

_ARCHITECTURE_WORDS: set[str] = set()
for _words in _ARCHITECTURE_WORDS_BY_LANG.values():
    _ARCHITECTURE_WORDS.update(_words)


def _content_has_signal(content: str, word_set: set[str]) -> bool:
    """Check if content contains any word from the signal set."""
    content_lower = content.lower()
    return any(word in content_lower for word in word_set)


def compute_importance(content: str, tags: list[str] | None = None) -> float:
    """Score content importance based on signals. Returns 0.0-1.0.

    Checks 8 languages for decision, error, and architecture keywords.
    No LLM needed.
    """
    score = 0.2  # base score

    if _content_has_signal(content, _DECISION_WORDS):
        score += 0.3
    if _content_has_signal(content, _ERROR_WORDS) or _ERROR_TYPE.search(content):
        score += 0.2
    if _content_has_signal(content, _ARCHITECTURE_WORDS):
        score += 0.2
    if _FILE_PATH.search(content):
        score += 0.1
    if "```" in content or "`" in content:
        score += 0.1
    if len(content) > 500:
        score += 0.1
    if tags and len(tags) >= 3:
        score += 0.1

    return min(score, 1.0)


# --- Entity extraction ---
_PUNCT = str.maketrans("", "", ".,!?:;\"'()")


def extract_entities(content: str) -> list[str]:
    """Extract named entities from content for tagging.

    Returns sorted list of prefixed tags, capped at 15:
      person:FirstName LastName
      entity:CamelCaseName
      file:path/to/file.ext
      error:SomeError
    """
    entities: set[str] = set()

    for m in _CAMEL_CASE.finditer(content):
        entities.add(f"entity:{m.group(0)}")

    for i, m in enumerate(_FILE_PATH.finditer(content)):
        if i >= 5:
            break
        entities.add(f"file:{m.group(0)}")

    for m in _ERROR_TYPE.finditer(content):
        entities.add(f"error:{m.group(0)}")

    # Person names: two consecutive capitalised words not in stopwords
    words = content.split()
    for i in range(len(words) - 1):
        w1 = words[i].translate(_PUNCT)
        w2 = words[i + 1].translate(_PUNCT)
        if (
            w1
            and w2
            and w1[0].isupper()
            and w2[0].isupper()
            and w1.isalpha()
            and w2.isalpha()
            and w1.lower() not in _STOP_WORDS
            and w2.lower() not in _STOP_WORDS
            and len(w1) > 1
            and len(w2) > 1
        ):
            entities.add(f"person:{w1} {w2}")

    return sorted(entities)[:15]
