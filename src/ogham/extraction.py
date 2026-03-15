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
