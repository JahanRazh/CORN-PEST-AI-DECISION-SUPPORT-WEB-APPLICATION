"""
Knowledge base: loads the expert-authored Excel workbook into memory.

Sheet1 is the curated pest profile table - one row per pest, holding damage
symptoms, recommended active ingredients, IRAC mode-of-action groups,
biological control options, application timing, IPM guidance, environmental
considerations and economic treatment thresholds.

Sheet2 is a product-level reference table (pest -> active ingredient -> trade
name -> pre-harvest interval -> IRAC group -> threshold). It uses a merged-cell
layout where the pest name appears only on the first row of each block, so the
pest column is forward-filled during parsing.

The workbook is read once at start-up and cached; a reload() hook is exposed so
an updated Excel file can be picked up without restarting Flask.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

import pandas as pd

from app import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: dict[str, Any] = {"pests": None, "products": None, "error": None}

# The workbook was authored on Windows and contains cp1252 bytes that arrive as
# replacement characters, e.g. 1-1/4" appears as "1-1/4<?>". Repair them so the
# thresholds read correctly in the UI.
_ENCODING_FIXES = {
    "�": '"',   # replacement char used for the inch mark
    " ": " ",   # non-breaking space
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
}


def clean_text(value: Any) -> str:
    """Normalise a spreadsheet cell into clean display text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "-"}:
        return ""
    for bad, good in _ENCODING_FIXES.items():
        text = text.replace(bad, good)
    # The source cells pad entries with runs of spaces for visual alignment.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_list(value: Any) -> list[str]:
    """Split a semicolon/comma separated cell into a de-duplicated list."""
    text = clean_text(value)
    if not text:
        return []
    parts = re.split(r"[;\n]+", text)
    items: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = part.strip(" ,.·")
        if not item:
            continue
        # Normalise casing: the sheet mixes UPPERCASE and Title Case for the
        # same ingredient (e.g. PERMETHRIN vs Permethrin).
        if item.isupper() and len(item) > 3:
            item = item.title()
        key = item.lower().replace("-", " ").replace(".", "")
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def _parse_moa_groups(value: Any) -> list[str]:
    """Extract IRAC mode-of-action group codes, preserving order."""
    text = clean_text(value)
    if not text:
        return []
    groups = re.findall(r"Group\s*([0-9]+[A-Za-z]?)", text, flags=re.IGNORECASE)
    if not groups:
        groups = re.findall(r"\b([0-9]{1,2}[A-Za-z]?)\b", text)
    result: list[str] = []
    for group in groups:
        code = group.upper()
        if code not in result:
            result.append(code)
    return result


def _load_pests() -> list[dict[str, Any]]:
    frame = pd.read_excel(config.KNOWLEDGE_BASE_PATH, sheet_name="Sheet1")
    records: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        common_name = clean_text(row.get("Pest Common Name"))
        if not common_name:
            continue

        active_ingredients = split_list(row.get("Recommended Active Ingredients"))
        records.append(
            {
                "common_name": common_name,
                "scientific_name": clean_text(row.get("Scientific Name")),
                "pest_group": clean_text(row.get("Pest Group")),
                "damage_symptoms": clean_text(row.get("Major Damage Symptoms")),
                "damage_symptom_list": split_list(row.get("Major Damage Symptoms")),
                "active_ingredients": active_ingredients,
                "moa_groups": _parse_moa_groups(row.get("IRAC Mode of Action")),
                "moa_raw": clean_text(row.get("IRAC Mode of Action")),
                "biological_controls": split_list(row.get("Biological Control Options")),
                "application_timing": clean_text(row.get("Application Timing")),
                "ipm_recommendation": clean_text(row.get("IPM Recommendation")),
                "environmental_consideration": clean_text(
                    row.get("Environmental Consideration")
                ),
                "treatment_guideline": clean_text(row.get("Treatment Guideline")),
                "slug": slugify(common_name),
            }
        )

    logger.info("Knowledge base: %d pest profiles loaded", len(records))
    return records


def _load_products() -> list[dict[str, Any]]:
    """Parse the product reference table (Sheet2)."""
    frame = pd.read_excel(
        config.KNOWLEDGE_BASE_PATH, sheet_name="Sheet2", header=1
    )
    frame.columns = [
        "pest",
        "active_ingredient",
        "trade_name",
        "phi_days",
        "moa",
        "guideline",
    ][: len(frame.columns)]

    # Merged cells leave the pest name only on the first row of each block.
    frame["pest"] = frame["pest"].ffill()

    records: list[dict[str, Any]] = []
    current_section = ""

    for _, row in frame.iterrows():
        pest = clean_text(row.get("pest"))
        ingredient = clean_text(row.get("active_ingredient"))
        if not pest or not ingredient:
            continue

        # "Pre-Mixes" is a sub-heading inside a pest block, not a product.
        if ingredient.lower() in {"pre-mixes", "premixes", "pre mixes"}:
            current_section = "Pre-Mix"
            continue
        if ingredient.lower() == "active ingredient(s)":
            continue

        # A restricted-use product is marked with a trailing asterisk.
        restricted = ingredient.endswith("*")
        records.append(
            {
                "pest": pest,
                "active_ingredient": ingredient.rstrip("*").strip(),
                "restricted_use": restricted,
                "trade_name": clean_text(row.get("trade_name")),
                "phi_days": _parse_phi(row.get("phi_days")),
                "phi_raw": clean_text(row.get("phi_days")),
                "moa_group": clean_text(row.get("moa")),
                "guideline": clean_text(row.get("guideline")),
                "product_type": current_section or "Single active ingredient",
            }
        )
        current_section = ""

    logger.info("Knowledge base: %d product records loaded", len(records))
    return records


def _parse_phi(value: Any) -> int | None:
    """Pre-harvest interval in days; the sheet footnotes some values."""
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _ensure_loaded() -> None:
    if _cache["pests"] is not None or _cache["error"]:
        return
    with _lock:
        if _cache["pests"] is not None or _cache["error"]:
            return
        try:
            if not config.KNOWLEDGE_BASE_PATH.exists():
                raise FileNotFoundError(
                    f"Knowledge base workbook not found at "
                    f"{config.KNOWLEDGE_BASE_PATH}"
                )
            _cache["pests"] = _load_pests()
            _cache["products"] = _load_products()
        except Exception as exc:
            logger.exception("Knowledge base failed to load")
            _cache["error"] = str(exc)


def reload() -> None:
    """Drop the cache so an edited workbook is re-read on next access."""
    with _lock:
        _cache["pests"] = None
        _cache["products"] = None
        _cache["error"] = None
    _ensure_loaded()


def is_ready() -> bool:
    _ensure_loaded()
    return _cache["pests"] is not None


def load_error() -> str | None:
    _ensure_loaded()
    return _cache["error"]


def all_pests() -> list[dict[str, Any]]:
    _ensure_loaded()
    return _cache["pests"] or []


def all_products() -> list[dict[str, Any]]:
    _ensure_loaded()
    return _cache["products"] or []


def find_pest(common_name: str) -> dict[str, Any] | None:
    """Exact (case-insensitive) lookup of a pest profile by common name."""
    target = (common_name or "").strip().lower()
    for pest in all_pests():
        if pest["common_name"].lower() == target:
            return pest
    return None


def find_pest_by_slug(slug: str) -> dict[str, Any] | None:
    for pest in all_pests():
        if pest["slug"] == slug:
            return pest
    return None


def products_for(pest_labels: list[str]) -> list[dict[str, Any]]:
    """Return Sheet2 products whose pest label matches any of the given names."""
    wanted = {label.strip().lower() for label in pest_labels if label}
    return [p for p in all_products() if p["pest"].strip().lower() in wanted]


def stats() -> dict[str, Any]:
    """Summary counts for the knowledge base page."""
    pests = all_pests()
    products = all_products()
    ingredients = {p["active_ingredient"].lower() for p in products}
    biologicals: set[str] = set()
    for pest in pests:
        biologicals.update(b.lower() for b in pest["biological_controls"])
    return {
        "pest_count": len(pests),
        "product_count": len(products),
        "unique_active_ingredients": len(ingredients),
        "biological_option_count": len(biologicals),
        "pest_groups": sorted({p["pest_group"] for p in pests if p["pest_group"]}),
        "restricted_use_count": sum(1 for p in products if p["restricted_use"]),
    }
