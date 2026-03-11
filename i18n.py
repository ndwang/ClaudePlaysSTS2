"""Internationalization support for the STS2 client.

Translations live in locales/*.json (one file per language).
Each file is a flat JSON object mapping string keys to format templates.
Templates use Python str.format() syntax, e.g. "HP:{hp}/{max_hp}".
"""

from __future__ import annotations

import json
from pathlib import Path

LOCALES_DIR = Path(__file__).parent / "locales"

_lang = "en"
_strings: dict[str, dict[str, str]] = {}


def _load_locale(lang: str) -> dict[str, str]:
    """Load a locale JSON file, returning an empty dict on failure."""
    path = LOCALES_DIR / f"{lang}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


def _ensure_loaded(lang: str) -> None:
    if lang not in _strings:
        _strings[lang] = _load_locale(lang)


def available_langs() -> list[str]:
    """Return list of available language codes based on locales/*.json files."""
    return sorted(p.stem for p in LOCALES_DIR.glob("*.json"))


def set_lang(lang: str) -> None:
    """Set the active language (e.g. 'en', 'zh')."""
    global _lang
    _ensure_loaded(lang)
    if not _strings.get(lang):
        raise ValueError(f"Unsupported language: {lang}. Available: {available_langs()}")
    _lang = lang


def get_lang() -> str:
    return _lang


def t(_key: str, **kwargs) -> str:
    """Look up a translated string by key, formatting with kwargs if provided."""
    _ensure_loaded(_lang)
    _ensure_loaded("en")
    strings = _strings.get(_lang, {})
    template = strings.get(_key) or _strings["en"].get(_key, _key)
    return template.format(**kwargs) if kwargs else template


def get_overlay_strings() -> dict[str, str]:
    """Return overlay label strings for the current language."""
    _ensure_loaded("en")
    return {k.removeprefix("overlay."): t(k)
            for k in _strings["en"] if k.startswith("overlay.")}


def write_overlay_locale(obs_dir: Path | None = None) -> None:
    """Write overlay locale strings to obs/locale.json for HTML consumption."""
    if obs_dir is None:
        obs_dir = Path(__file__).parent / "obs"
    obs_dir.mkdir(exist_ok=True)
    locale_file = obs_dir / "locale.json"
    locale_file.write_text(
        json.dumps(get_overlay_strings(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
