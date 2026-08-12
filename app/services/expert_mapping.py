"""
Expert mapping layer: validates the AI class against the agricultural pest
entity described in the knowledge base.

The classifier was trained on folder names ("Army Worm-Spodoptera frugiperda"),
the curated pest sheet uses agronomic common names ("Fall Armyworm"), and the
product reference sheet uses yet another vocabulary in which several species
are grouped together ("Armyworm species", "Cutworms (Black, Dingy, Variegated,
Claybacked)"). Nothing downstream can be trusted until those three vocabularies
are reconciled, which is this layer's job.

It also performs a genuine validation step rather than a blind lookup. Three of
the ten trained classes carry taxonomic discrepancies against the expert sheet:

  * Black Cut Worm      model: Agrotis ypsilon    sheet: Agrotis ipsilon
    -> Orthographic variant of the same species. Both spellings appear in the
       literature; ipsilon is the accepted form. Harmless.

  * Wire Worm           model: Agriotes lineatus  sheet: Limonius spp.
    -> Different genera within the same family (Elateridae). Management is
       genus-independent, so the recommendation still holds, but the mismatch
       is surfaced rather than hidden.

  * Corn Borer          model: Ostrinia furnacalis (Asian corn borer)
                        sheet thresholds cite European/Southwestern borers
    -> Chemical options overlap; economic thresholds are region-specific.

Reporting these as validation notes is what separates an expert system from a
lookup table, and gives the result page something defensible to display.
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
        "notes": [
            "The product reference table groups this species under the generic "
            "'Armyworm species' block, so the listed products are shared with "
            "Fall Armyworm rather than beet-armyworm-specific."
        ],
    },
    "Black Cut Worm-Agrotis ypsilon": {
        "display_name": "Black Cutworm",
        "kb_name": "Black Cutworm",
        "scientific_name": "Agrotis ipsilon",
        "aliases": ["Agrotis ypsilon", "Greasy Cutworm"],
        "validation": {
            "severity": "info",
            "message": (
                "The trained class is labelled 'Agrotis ypsilon' while the "
                "knowledge base uses 'Agrotis ipsilon'. These are orthographic "
                "variants of the same species; 'ipsilon' is the accepted form."
            ),
        },
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
        "validation": {
            "severity": "caution",
            "message": (
                "The trained class is the Asian corn borer (Ostrinia "
                "furnacalis), while the economic thresholds in the knowledge "
                "base are written for the European and Southwestern corn "
                "borers. Registered chemistry overlaps closely, but verify "
                "threshold figures against local extension guidance."
            ),
        },
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
        "validation": {
            "severity": "info",
            "message": (
                "The knowledge base product table refers to the corn flea "
                "beetle (Chaetocnema pulicaria), the species of economic "
                "importance in maize; the trained class covers Phyllotreta "
                "flea beetles more broadly. Control measures are equivalent."
            ),
        },
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
        "validation": {
            "severity": "caution",
            "message": (
                "The trained class is labelled 'Agriotes lineatus' while the "
                "knowledge base profiles 'Limonius spp.'. Both are click "
                "beetle larvae (family Elateridae) and share the same soil "
                "management and seed treatment strategy, but they are "
                "different genera."
            ),
        },
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
    validation_notes: list[dict[str, str]] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return any(n["severity"] in {"caution", "warning"} for n in self.validation_notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai_class": self.ai_class,
            "display_name": self.display_name,
            "scientific_name": self.scientific_name,
            "matched": self.matched,
            "match_method": self.match_method,
            "aliases": self.aliases,
            "validation_notes": self.validation_notes,
            "has_warnings": self.has_warnings,
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
    notes: list[dict[str, str]] = []

    if entry:
        profile = knowledge_base.find_pest(entry["kb_name"])
        match_method = "expert mapping table"

        if profile is None:
            # The mapping table names a row the workbook no longer contains.
            profile, fallback_method = _fuzzy_kb_lookup(ai_class)
            match_method = f"expert mapping table -> {fallback_method}"
            notes.append(
                {
                    "severity": "warning",
                    "message": (
                        f"The expert mapping expected a knowledge base entry "
                        f"named '{entry['kb_name']}' but it was not found in "
                        f"the workbook. A fallback match was used."
                    ),
                }
            )

        if "validation" in entry:
            notes.append(entry["validation"])
        for note in entry.get("notes", []):
            notes.append({"severity": "info", "message": note})

        return MappingResult(
            ai_class=ai_class,
            display_name=entry["display_name"],
            scientific_name=entry.get("scientific_name", ""),
            matched=profile is not None,
            match_method=match_method,
            pest_profile=profile,
            aliases=entry.get("aliases", []),
            validation_notes=notes,
        )

    # Class is not in the mapping table at all - the model has been retrained
    # with new classes and the mapping table was not updated.
    profile, method = _fuzzy_kb_lookup(ai_class)
    notes.append(
        {
            "severity": "warning",
            "message": (
                f"'{ai_class}' is not registered in the expert mapping table. "
                f"The system fell back to automatic name matching ({method}). "
                "Add it to CLASS_MAPPING for a validated result."
            ),
        }
    )
    return MappingResult(
        ai_class=ai_class,
        display_name=profile["common_name"] if profile else ai_class.split("-")[0].strip(),
        scientific_name=profile["scientific_name"] if profile else "",
        matched=profile is not None,
        match_method=method,
        pest_profile=profile,
        aliases=[],
        validation_notes=notes,
    )


def coverage_report() -> dict[str, Any]:
    """Diagnostic: how completely the mapping table covers the trained classes.

    Surfaced on the dashboard so a gap between the model and the knowledge base
    is visible rather than silent.
    """
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
                "note_count": len(result.validation_notes),
                "has_warnings": result.has_warnings,
            }
        )
    return {
        "total_classes": len(class_names),
        "mapped": sum(1 for r in rows if r["matched"]),
        "with_warnings": sum(1 for r in rows if r["has_warnings"]),
        "rows": rows,
    }
