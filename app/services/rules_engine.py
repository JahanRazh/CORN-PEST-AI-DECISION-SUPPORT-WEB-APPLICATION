"""
Rule-based recommendation engine.

Takes a validated pest identity and produces three coordinated recommendation tracks:

    chemical    ranked pesticide products with IRAC rotation guidance
    biological  biocontrol agents
    ipm         sequenced integrated pest management actions

Every rule that fires records a human-readable trace entry, so the result page
can explain *why* a recommendation was made rather than presenting an opaque
list. That trace is the "decision reasoning" element of the explainable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app import config

# Soil-dwelling pests: foliar rescue sprays do not reach them.
SOIL_PESTS = {"White Grub", "Wireworm"}


@dataclass
class Recommendation:
    """Full context-aware output of the rule engine."""

    pest_display_name: str
    action_level: str                       # monitor | treat_soon | treat_now | preventive
    action_headline: str
    action_detail: str
    urgency_score: int                      # 0-100, drives the UI urgency meter
    chemical_options: list[dict[str, Any]] = field(default_factory=list)
    biological_options: list[dict[str, Any]] = field(default_factory=list)
    ipm_actions: list[dict[str, Any]] = field(default_factory=list)
    environmental_notes: list[str] = field(default_factory=list)
    reasoning: list[dict[str, str]] = field(default_factory=list)
    threshold_guidance: str = ""
    timing_guidance: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pest_display_name": self.pest_display_name,
            "action_level": self.action_level,
            "action_headline": self.action_headline,
            "action_detail": self.action_detail,
            "urgency_score": self.urgency_score,
            "chemical_options": self.chemical_options,
            "biological_options": self.biological_options,
            "ipm_actions": self.ipm_actions,
            "environmental_notes": self.environmental_notes,
            "reasoning": self.reasoning,
            "threshold_guidance": self.threshold_guidance,
            "timing_guidance": self.timing_guidance,
            "context": self.context,
            "sources": self.sources,
        }


def _score_chemical(product: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    """Rank one product; returns (score, advantages, cautions)."""
    score = 50.0
    advantages: list[str] = []
    cautions: list[str] = []

    if product.get("restricted_use"):
        score -= 8
        cautions.append("Restricted-use pesticide - a certified applicator is required")

    phi = product.get("phi_days")
    if phi is not None and phi <= 7:
        score += 4
        advantages.append(f"Short pre-harvest interval ({phi} days)")

    if product.get("trade_name"):
        advantages.append(f"Available as {product['trade_name']}")

    return score, advantages, cautions


def _build_chemical_options(mapping_result) -> list[dict[str, Any]]:
    """Rank product-level records; fall back to the curated ingredient list."""
    options: list[dict[str, Any]] = []
    if mapping_result.pest_profile:
        profile = mapping_result.pest_profile
        moa_groups = profile.get("moa_groups", [])
        for index, ingredient in enumerate(profile.get("active_ingredients", [])):
            pseudo = {
                "active_ingredient": ingredient,
                "trade_name": "",
                "moa_group": moa_groups[index] if index < len(moa_groups) else "",
                "phi_days": None,
                "restricted_use": False,
                "product_type": "Curated recommendation",
            }
            score, advantages, cautions = _score_chemical(pseudo)
            options.append(
                {
                    **pseudo,
                    "score": round(score, 1),
                    "advantages": advantages,
                    "cautions": cautions,
                    "permitted": True,
                    "source": "Curated pest profile (Sheet1)",
                }
            )

    options.sort(key=lambda o: (o["permitted"], o["score"]), reverse=True)
    return options


def _build_biological_options(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not profile:
        return []

    options: list[dict[str, Any]] = []
    for agent in profile.get("biological_controls", []):
        options.append(
            {
                "agent": agent,
                "suitability": "suitable",
                "note": "A potential biological control for this pest.",
                "source": "Curated pest profile (Sheet1)",
            }
        )
    return options


def generate(
    mapping_result,
    *,
    confidence: float = 1.0,
) -> Recommendation:
    """Apply the rule set and return a fully explained recommendation."""
    profile = mapping_result.pest_profile
    pest_name = mapping_result.display_name
    reasoning: list[dict[str, str]] = []
    sources: list[str] = []

    reasoning.append(
        {
            "rule": "Pest identity",
            "detail": (
                f"The AI classification was validated against the knowledge base "
                f"as {pest_name}"
                + (f" ({mapping_result.scientific_name})" if mapping_result.scientific_name else "")
                + f", matched via {mapping_result.match_method}."
            ),
        }
    )

    if profile:
        sources.append("Curated pest profile (Excel Sheet1)")

    action_level = "treat_now"
    urgency = 75
    headline = "Treat now - threshold exceeded"
    detail = (
        "Based on the detection, verify against the economic threshold and "
        "apply at the timing indicated below for maximum efficacy."
    )

    # Soil pests past the planting window override the action level: no rescue
    # treatment exists, so telling the user to spray would be wrong.
    if pest_name in SOIL_PESTS:
        action_level = "preventive"
        urgency = min(urgency, 45)
        headline = "No rescue treatment - plan preventively"
        detail = (
            "Larvae of this pest feed on roots below ground and cannot be "
            "reached by a foliar spray. Record the affected areas "
            "and plan seed treatment or soil-applied control for the next "
            "planting; treat now only if replanting into infested ground."
        )
        reasoning.append(
            {
                "rule": "Soil pest override",
                "detail": (
                    "The knowledge base states no rescue treatment is available "
                    "for this pest once the crop is established, so the "
                    "recommendation switches from curative to preventive."
                ),
            }
        )

    # --- Rule: low AI confidence tempers the advice ---------------------
    if confidence < 0.75:
        urgency = max(urgency - 10, 10)
        reasoning.append(
            {
                "rule": "Confidence moderation",
                "detail": (
                    f"Model confidence is {confidence * 100:.1f}%, below the 75% "
                    "high-confidence band, so field verification is advised "
                    "before committing to a chemical application."
                ),
            }
        )

    chemical_options = _build_chemical_options(mapping_result)
    biological_options = _build_biological_options(profile)

    # --- Rule: IRAC rotation --------------------------------------------
    distinct_moa = []
    for option in chemical_options:
        group = (option.get("moa_group") or "").strip()
        if group and group not in distinct_moa:
            distinct_moa.append(group)
    if len(distinct_moa) > 1:
        reasoning.append(
            {
                "rule": "Resistance management (IRAC rotation)",
                "detail": (
                    f"{len(distinct_moa)} distinct modes of action are available "
                    f"({', '.join(distinct_moa[:6])}). Rotate between groups "
                    "across consecutive generations - never apply the same group "
                    "twice in a row against successive generations."
                ),
            }
        )

    ipm_actions = _build_ipm_actions(
        profile,
        action_level=action_level,
        pest_name=pest_name,
        biological_options=biological_options,
        distinct_moa=distinct_moa,
    )

    environmental_notes: list[str] = []
    if profile and profile.get("environmental_consideration"):
        environmental_notes.append(profile["environmental_consideration"])

    return Recommendation(
        pest_display_name=pest_name,
        action_level=action_level,
        action_headline=headline,
        action_detail=detail,
        urgency_score=urgency,
        chemical_options=chemical_options,
        biological_options=biological_options,
        ipm_actions=ipm_actions,
        environmental_notes=environmental_notes,
        reasoning=reasoning,
        threshold_guidance=profile.get("treatment_guideline", "") if profile else "",
        timing_guidance=profile.get("application_timing", "") if profile else "",
        context={},
        sources=sources or ["Curated pest profile (Excel Sheet1)"],
    )


def _build_ipm_actions(
    profile: dict[str, Any] | None,
    *,
    action_level: str,
    pest_name: str,
    biological_options: list[dict[str, Any]],
    distinct_moa: list[str],
) -> list[dict[str, Any]]:
    """Assemble the sequenced IPM programme: monitor, cultural, bio, chemical."""
    actions: list[dict[str, Any]] = [
        {
            "step": 1,
            "category": "Monitor",
            "title": "Scout and confirm the infestation level",
            "detail": (
                profile.get("treatment_guideline")
                or "Inspect at least 20 plants at five points across the field "
                   "and record the percentage showing fresh damage."
            )
            if profile
            else "Inspect at least 20 plants at five points across the field.",
        }
    ]

    cultural = {
        "Fall Armyworm": "Destroy crop residue and avoid staggered planting dates, "
                         "which give overlapping generations a continuous host.",
        "Beet Armyworm": "Remove weed hosts around field margins where egg masses "
                         "are laid before the crop emerges.",
        "Black Cutworm": "Control winter weeds at least two weeks before planting "
                         "to remove egg-laying sites, and consider deep ploughing.",
        "Corn Aphid": "Avoid excess nitrogen, which produces soft growth that "
                      "favours aphid colonies, and manage moisture stress.",
        "Corn Borer": "Shred and plough down stalks after harvest to destroy "
                      "overwintering larvae inside the stubble.",
        "Corn Earworm": "Use pheromone traps to time treatment to moth flight, "
                        "and prefer tight-husk varieties where available.",
        "Corn Grasshopper": "Treat field margins and grassy borders where nymphs "
                            "congregate before they move into the crop.",
        "Flea Beetle": "Plant after soil warms so seedlings grow through the "
                       "vulnerable stage quickly; control grassy weed hosts.",
        "White Grub": "Rotate away from grass crops, and plough infested fields to "
                      "expose grubs to predators and desiccation.",
        "Wireworm": "Rotate out of grassland, improve drainage, and use bait "
                    "stations before planting to assess population density.",
    }
    if pest_name in cultural:
        actions.append(
            {
                "step": 2,
                "category": "Cultural",
                "title": "Apply cultural and preventive measures",
                "detail": cultural[pest_name],
            }
        )

    if biological_options:
        favoured = [b for b in biological_options if b["suitability"] != "limited"]
        agents = ", ".join(b["agent"] for b in (favoured or biological_options)[:3])
        actions.append(
            {
                "step": len(actions) + 1,
                "category": "Biological",
                "title": "Deploy biological control first where feasible",
                "detail": (
                    f"Consider {agents}. Biological agents are most effective "
                    "against early larval stages and conserve the natural "
                    "enemies already working in the field."
                ),
            }
        )

    if action_level == "preventive":
        chemical_detail = (
            "No effective rescue chemistry exists at this stage. Plan a seed "
            "treatment or soil-applied insecticide for the next planting in "
            "the affected areas."
        )
    else:
        chemical_detail = (
            "Apply the highest-ranked permitted product from the chemical list, "
            "targeting the timing window stated in the pest profile. Ensure "
            "thorough coverage of the whorl or feeding site."
        )
        if profile and profile.get("application_timing"):
            chemical_detail += f" Optimal timing: {profile['application_timing']}."

    actions.append(
        {
            "step": len(actions) + 1,
            "category": "Chemical",
            "title": "Chemical intervention decision",
            "detail": chemical_detail,
        }
    )

    if len(distinct_moa) > 1:
        actions.append(
            {
                "step": len(actions) + 1,
                "category": "Resistance",
                "title": "Rotate modes of action",
                "detail": (
                    f"Alternate between IRAC groups {', '.join(distinct_moa[:4])} "
                    "on successive generations. Repeated use of one group is the "
                    "primary driver of field-evolved resistance."
                ),
            }
        )
    elif profile and profile.get("ipm_recommendation"):
        actions.append(
            {
                "step": len(actions) + 1,
                "category": "Resistance",
                "title": "Resistance management",
                "detail": profile["ipm_recommendation"],
            }
        )

    actions.append(
        {
            "step": len(actions) + 1,
            "category": "Follow-up",
            "title": "Re-scout after treatment",
            "detail": (
                "Assess control 3-5 days after application. If live larvae "
                "persist above threshold, re-treat with a different mode of "
                "action rather than repeating the same product."
            ),
        }
    )

    return actions
