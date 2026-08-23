"""
Expert mapping layer: validates the AI class against the agricultural pest
entity described in the knowledge base.

The classifier was trained on folder names ("Army Worm-Spodoptera frugiperda"),
the curated pest sheet uses agronomic common names ("Fall Armyworm"), and the
product reference sheet uses yet another vocabulary in which several species
are grouped together ("Armyworm species", "Cutworms (Black, Dingy, Variegated,
Claybacked)"). Nothing downstream can be trusted until those three vocabularies
are reconciled, which is this layer's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services import knowledge_base

# Model class name -> knowledge base entities.
#
#   kb_name       : matching row in Sheet1 (curated pest profile)
#   display_name  : the agronomic name shown to the user
CLASS_MAPPING: dict[str, dict[str, Any]] = {
    "Army Worm-Spodoptera frugiperda": {
        "display_name": "Fall Armyworm",
        "kb_name": "Fall Armyworm",
        "scientific_name": "Spodoptera frugiperda",
        "aliases": ["Fall Army Worm", "FAW"],
    },
    "Beet Army Worm-Spodoptera exigua": {
        "display_name": "Beet Armyworm",
        "kb_name": "Beet Armyworm",
        "scientific_name": "Spodoptera exigua",
        "aliases": ["Small Mottled Willow Moth"],
    },
    "Black Cut Worm-Agrotis ypsilon": {
        "display_name": "Black Cutworm",
        "kb_name": "Black Cutworm",
        "scientific_name": "Agrotis ipsilon",
        "aliases": ["Agrotis ypsilon", "Greasy Cutworm"],
    },
    "Corn Aphid-Rhopalosiphum maidis": {
        "display_name": "Corn Aphid",
        "kb_name": "Corn Aphid",
        "scientific_name": "Rhopalosiphum maidis",
        "aliases": ["Corn Leaf Aphid", "Green Corn Aphid"],
    },
    "Corn Borer-Ostrinia furnacalis": {
        "display_name": "Corn Borer",
        "kb_name": "Corn Borer",
        "scientific_name": "Ostrinia furnacalis",
        "aliases": ["Asian Corn Borer", "Ostrinia nubilalis"],
    },
    "Corn Ear Worm-Helicoverpa armigera": {
        "display_name": "Corn Earworm",
        "kb_name": "Corn Earworm",
        "scientific_name": "Helicoverpa armigera",
        "aliases": ["Cotton Bollworm", "Helicoverpa zea", "Old World Bollworm"],
    },
    "Corn Grasshopper-Oxya chinensis": {
        "display_name": "Corn Grasshopper",
        "kb_name": "Corn Grasshopper",
        "scientific_name": "Oxya chinensis",
        "aliases": ["Rice Grasshopper"],
    },
    "Flea Beetle-Phyllotreta spp": {
        "display_name": "Flea Beetle",
        "kb_name": "Flea Beetle",
        "scientific_name": "Phyllotreta spp.",
        "aliases": ["Corn Flea Beetle", "Chaetocnema pulicaria"],
    },
    "White Grub-Holotrichia spp": {
        "display_name": "White Grub",
        "kb_name": "White Grub",
        "scientific_name": "Holotrichia spp.",
        "aliases": ["Cockchafer larvae", "Scarab larvae"],
    },
    "Wire Worm-Agriotes lineatus": {
        "display_name": "Wireworm",
        "kb_name": "Wireworm",
        "scientific_name": "Limonius spp.",
        "aliases": ["Agriotes lineatus", "Click beetle larvae"],
    },
}


@dataclass
class MappingResult:
    """Outcome of validating one AI class against the knowledge base."""

    ai_class: str
    display_name: str
    scientific_name: str
    matched: bool
    match_method: str
    pest_profile: dict[str, Any] | None = None
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai_class": self.ai_class,
            "display_name": self.display_name,
            "scientific_name": self.scientific_name,
            "matched": self.matched,
            "match_method": self.match_method,
            "aliases": self.aliases,
        }


def _fuzzy_kb_lookup(ai_class: str) -> tuple[dict[str, Any] | None, str]:
    """Fallback matcher for a class name absent from CLASS_MAPPING.

    Trained folder names follow a "Common Name-Scientific name" convention, so
    the portion before the hyphen is compared token-wise against the knowledge
    base common names.
    """
    common_part = ai_class.split("-")[0].strip().lower()
    condensed = common_part.replace(" ", "")

    for pest in knowledge_base.all_pests():
        kb_name = pest["common_name"].lower()
        if kb_name.replace(" ", "") == condensed:
            return pest, "normalised name match"

    # Token overlap, e.g. "corn ear worm" vs "corn earworm".
    tokens = set(common_part.split())
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for pest in knowledge_base.all_pests():
        kb_tokens = set(pest["common_name"].lower().split())
        if not kb_tokens:
            continue
        overlap = len(tokens & kb_tokens) / len(tokens | kb_tokens)
        if overlap > best[0]:
            best = (overlap, pest)
    if best[0] >= 0.5:
        return best[1], f"token similarity {best[0]:.0%}"

    return None, "no match"


def map_class(ai_class: str) -> MappingResult:
    """Resolve an AI class name to a validated agricultural pest entity."""
    entry = CLASS_MAPPING.get(ai_class)

    if entry:
        profile = knowledge_base.find_pest(entry["kb_name"])
        match_method = "expert mapping table"

        if profile is None:
            # The mapping table names a row the workbook no longer contains.
            profile, fallback_method = _fuzzy_kb_lookup(ai_class)
            match_method = f"expert mapping table -> {fallback_method}"

        return MappingResult(
            ai_class=ai_class,
            display_name=entry["display_name"],
            scientific_name=entry.get("scientific_name", ""),
            matched=profile is not None,
            match_method=match_method,
            pest_profile=profile,
            aliases=entry.get("aliases", []),
        )

    # Class is not in the mapping table at all - the model has been retrained
    # with new classes and the mapping table was not updated.
    profile, method = _fuzzy_kb_lookup(ai_class)
    return MappingResult(
        ai_class=ai_class,
        display_name=profile["common_name"] if profile else ai_class.split("-")[0].strip(),
        scientific_name=profile["scientific_name"] if profile else "",
        matched=profile is not None,
        match_method=method,
        pest_profile=profile,
        aliases=[],
    )


def coverage_report() -> dict[str, Any]:
    """Diagnostic: how completely the mapping table covers the trained classes."""
    from app.services import model_service

    class_names = model_service.get_class_names()
    rows = []
    for name in class_names:
        result = map_class(name)
        rows.append(
            {
                "ai_class": name,
                "display_name": result.display_name,
                "matched": result.matched,
                "match_method": result.match_method,
            }
        )
    return {
        "total_classes": len(class_names),
        "mapped": sum(1 for r in rows if r["matched"]),
        "rows": rows,
    }
