"""
Rule-based recommendation engine.

Takes a validated pest identity plus the field context supplied by the user
(growth stage, observed severity, weather, days to harvest, beneficial-insect
presence) and produces three coordinated recommendation tracks:

    chemical    ranked pesticide products with IRAC rotation guidance
    biological  biocontrol agents suited to the current conditions
    ipm         sequenced integrated pest management actions

Every rule that fires records a human-readable trace entry, so the result page
can explain *why* a recommendation was made rather than presenting an opaque
list. That trace is the "decision reasoning" element of the explainable output.

The rules encode standard IPM doctrine:
  - treat only when the economic threshold is met (low severity -> monitor)
  - prefer selective chemistry while beneficials are active
  - respect the pre-harvest interval as harvest approaches
  - rotate IRAC mode-of-action groups to manage resistance
  - match application timing to the crop growth stage
  - adjust for weather (rain-fastness, heat, humidity and biocontrol efficacy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app import config

# Active ingredients considered selective / softer on beneficial insects.
# Used when the user reports beneficial insects present in the field.
SELECTIVE_INGREDIENTS = {
    "chlorantraniliprole",
    "cyclaniliprole",
    "flubendiamide",
    "methoxyfenozide",
    "novaluron",
    "indoxacarb",
    "spinetoram",
    "spinosad",
    "emamectin benzoate",
    "bacillus thuringiensis",
}

# Broad-spectrum groups that are hardest on pollinators and natural enemies.
BROAD_SPECTRUM_MOA = {"1A", "1B", "3", "3A", "4A"}

# Ingredients with documented pollinator risk; flagged during tasseling.
POLLINATOR_RISK = {
    "imidacloprid",
    "thiamethoxam",
    "clothianidin",
    "acetamiprid",
    "chlorpyrifos",
    "dimethoate",
    "malathion",
    "carbaryl",
}

# Soil-dwelling pests: foliar rescue sprays do not reach them.
SOIL_PESTS = {"White Grub", "Wireworm"}

# Pests that bore into tissue, where timing before entry is critical.
BORING_PESTS = {"Corn Borer", "Corn Earworm", "Stalk Borer"}

# Weather-driven modifiers for biological control agents.
BIOCONTROL_WEATHER = {
    "beauveria bassiana": {
        "favours": {"humid", "cool"},
        "hinders": {"dry"},
        "reason": "entomopathogenic fungi require high humidity to germinate on the cuticle",
    },
    "metarhizium anisopliae": {
        "favours": {"humid", "cool"},
        "hinders": {"dry"},
        "reason": "fungal conidia need moisture and moderate temperatures to infect",
    },
    "bacillus thuringiensis": {
        "favours": {"cool", "humid"},
        "hinders": {"rainy"},
        "reason": "Bt is degraded by UV and washed off by rain; apply late in the day",
    },
    "entomopathogenic nematodes": {
        "favours": {"humid", "rainy", "cool"},
        "hinders": {"dry"},
        "reason": "nematodes need soil moisture to move toward their host",
    },
    "steinernema carpocapsae": {
        "favours": {"humid", "rainy"},
        "hinders": {"dry"},
        "reason": "requires moist soil for movement and survival",
    },
    "trichogramma": {
        "favours": {"humid", "cool"},
        "hinders": {"rainy"},
        "reason": "egg parasitoid releases fail in heavy rain and extreme heat",
    },
}


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


def _severity_weight(severity: str) -> int:
    for level in config.SEVERITY_LEVELS:
        if level["key"] == severity:
            return level["weight"]
    return 2


def _stage_order(stage: str) -> int:
    for item in config.GROWTH_STAGES:
        if item["key"] == stage:
            return item["order"]
    return 2


def _label_for(collection: list[dict[str, Any]], key: str, default: str = "") -> str:
    for item in collection:
        if item["key"] == key:
            return item["label"]
    return default or key


def _stage_alignment(pest_name: str, stage: str, timing_text: str) -> tuple[bool, str]:
    """Judge whether the current growth stage matches the pest's damage window."""
    order = _stage_order(stage)
    timing = (timing_text or "").lower()

    if pest_name in SOIL_PESTS:
        if order <= 1:
            return True, (
                "Soil pests are managed before or at planting; the crop is still "
                "within that window."
            )
        return False, (
            "Soil-dwelling larvae feed below ground and the crop is past the "
            "planting window, so no rescue treatment can reach them this season."
        )

    if "pre-planting" in timing and order <= 1:
        return True, "The pest's control window is pre-planting or seedling stage."

    if pest_name == "Corn Earworm":
        if order >= 4:
            return True, "Silking has begun, which is the critical earworm window."
        return False, (
            "Earworm damage occurs at silking; the crop has not reached that "
            "stage yet, so scouting rather than spraying is appropriate."
        )

    if pest_name in BORING_PESTS and order >= 2:
        return True, (
            "The crop is in the vegetative to reproductive range where borer "
            "larvae are still exposed before tunnelling."
        )

    if order <= 3:
        return True, (
            "Foliar-feeding larvae are exposed and reachable at the current "
            "growth stage."
        )

    return True, "Treatment remains feasible at the current growth stage."


def _score_chemical(
    product: dict[str, Any],
    *,
    stage: str,
    beneficials_present: bool,
    days_to_harvest: int | None,
    weather: str,
) -> tuple[float, list[str], list[str]]:
    """Rank one product; returns (score, advantages, cautions)."""
    ingredient = product["active_ingredient"].lower()
    moa = (product.get("moa_group") or "").upper().replace(" ", "")
    score = 50.0
    advantages: list[str] = []
    cautions: list[str] = []

    is_selective = any(sel in ingredient for sel in SELECTIVE_INGREDIENTS)
    if is_selective:
        score += 20
        advantages.append("Selective chemistry - lower impact on natural enemies")
    if beneficials_present and not is_selective:
        score -= 18
        cautions.append(
            "Broad-spectrum: you reported beneficial insects present in the field"
        )
    if any(group in moa for group in BROAD_SPECTRUM_MOA) and beneficials_present:
        score -= 6

    if product.get("restricted_use"):
        score -= 8
        cautions.append("Restricted-use pesticide - a certified applicator is required")

    # Pre-harvest interval handling.
    phi = product.get("phi_days")
    if phi is not None and days_to_harvest is not None:
        if phi > days_to_harvest:
            score -= 60
            cautions.append(
                f"Pre-harvest interval of {phi} days exceeds the {days_to_harvest} "
                "days remaining before harvest - not permissible"
            )
        elif phi > days_to_harvest - 7:
            score -= 12
            cautions.append(
                f"Pre-harvest interval of {phi} days leaves little margin before "
                f"harvest in {days_to_harvest} days"
            )
        else:
            score += 6
            advantages.append(f"Pre-harvest interval of {phi} days fits the schedule")
    elif phi is not None and phi <= 7:
        score += 4
        advantages.append(f"Short pre-harvest interval ({phi} days)")

    # Pollinator protection during pollen shed.
    if _stage_order(stage) == 4 and any(risk in ingredient for risk in POLLINATOR_RISK):
        score -= 15
        cautions.append(
            "Elevated pollinator risk during tasseling and silking - avoid "
            "application while pollen is shedding"
        )

    # Weather effects on the application itself.
    if weather == "rainy":
        score -= 5
        cautions.append("Rain may wash off the application - check the rain-fast period")
    if weather == "dry" and "spinosad" in ingredient:
        cautions.append("Ensure thorough coverage; residues degrade quickly in strong sun")

    if product.get("trade_name"):
        advantages.append(f"Available as {product['trade_name']}")

    return score, advantages, cautions


def _build_chemical_options(
    mapping_result,
    *,
    stage: str,
    beneficials_present: bool,
    days_to_harvest: int | None,
    weather: str,
) -> list[dict[str, Any]]:
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
            score, advantages, cautions = _score_chemical(
                pseudo,
                stage=stage,
                beneficials_present=beneficials_present,
                days_to_harvest=days_to_harvest,
                weather=weather,
            )
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


def _build_biological_options(
    profile: dict[str, Any] | None, weather: str, stage: str
) -> list[dict[str, Any]]:
    if not profile:
        return []

    options: list[dict[str, Any]] = []
    for agent in profile.get("biological_controls", []):
        lowered = agent.lower()
        suitability = "suitable"
        note = "Compatible with the reported field conditions."

        for keyword, rule in BIOCONTROL_WEATHER.items():
            if keyword in lowered:
                if weather in rule["hinders"]:
                    suitability = "limited"
                    note = (
                        f"Reduced efficacy in {weather} conditions - {rule['reason']}."
                    )
                elif weather in rule["favours"]:
                    suitability = "favoured"
                    note = f"Conditions favour this agent - {rule['reason']}."
                break

        # Nematodes and soil fungi need to go on before or at planting.
        if "nematode" in lowered and _stage_order(stage) > 2:
            suitability = "limited"
            note = (
                "Soil applications are most effective before planting or during "
                "early root development."
            )

        options.append(
            {
                "agent": agent,
                "suitability": suitability,
                "note": note,
                "source": "Curated pest profile (Sheet1)",
            }
        )

    order = {"favoured": 0, "suitable": 1, "limited": 2}
    options.sort(key=lambda o: order.get(o["suitability"], 3))
    return options


def generate(
    mapping_result,
    *,
    growth_stage: str = "vegetative",
    severity: str = "moderate",
    weather: str = "humid",
    days_to_harvest: int | None = None,
    beneficials_present: bool = False,
    confidence: float = 1.0,
) -> Recommendation:
    """Apply the rule set and return a fully explained recommendation."""
    profile = mapping_result.pest_profile
    pest_name = mapping_result.display_name
    reasoning: list[dict[str, str]] = []
    sources: list[str] = []

    weight = _severity_weight(severity)
    stage_label = _label_for(config.GROWTH_STAGES, growth_stage)
    severity_label = _label_for(config.SEVERITY_LEVELS, severity)
    weather_label = _label_for(config.WEATHER_CONDITIONS, weather)

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

    # --- Rule 1: growth stage alignment -----------------------------------
    stage_ok, stage_reason = _stage_alignment(
        pest_name, growth_stage, profile.get("application_timing", "") if profile else ""
    )
    reasoning.append(
        {
            "rule": "Growth stage alignment",
            "detail": f"Crop stage reported as {stage_label}. {stage_reason}",
        }
    )

    # --- Rule 2: severity vs economic threshold ---------------------------
    if weight <= 1:
        action_level = "monitor"
        urgency = 20
        headline = "Monitor - below the economic threshold"
        detail = (
            "Infestation is currently isolated. IPM doctrine is to scout rather "
            "than spray at this level: treating below the economic threshold "
            "adds cost, removes natural enemies and accelerates resistance."
        )
    elif weight == 2:
        action_level = "treat_soon"
        urgency = 50
        headline = "Prepare to treat - approaching threshold"
        detail = (
            "Damage is spreading. Confirm against the economic threshold below, "
            "and have the chosen product ready so treatment can go on at the "
            "correct larval stage."
        )
    elif weight == 3:
        action_level = "treat_now"
        urgency = 75
        headline = "Treat now - threshold exceeded"
        detail = (
            "Infestation is widespread enough to justify intervention. Apply at "
            "the timing indicated below for maximum efficacy."
        )
    else:
        action_level = "treat_now"
        urgency = 92
        headline = "Urgent - severe infestation"
        detail = (
            "Field-wide damage warrants immediate intervention combined with "
            "follow-up scouting to confirm control and detect re-infestation."
        )

    reasoning.append(
        {
            "rule": "Severity vs economic threshold",
            "detail": (
                f"Severity reported as '{severity_label}' (weight {weight}/4), "
                f"which maps to the '{headline.split(' - ')[0].lower()}' action level."
            ),
        }
    )

    # Soil pests past the planting window override the action level: no rescue
    # treatment exists, so telling the user to spray would be wrong.
    if pest_name in SOIL_PESTS and not stage_ok:
        action_level = "preventive"
        urgency = min(urgency, 45)
        headline = "No rescue treatment - plan preventively"
        detail = (
            "Larvae of this pest feed on roots below ground and cannot be "
            "reached by a foliar spray at this stage. Record the affected areas "
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

    # --- Rule 3: low AI confidence tempers the advice ---------------------
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

    # --- Rule 4: pre-harvest interval -------------------------------------
    if days_to_harvest is not None:
        reasoning.append(
            {
                "rule": "Pre-harvest interval filter",
                "detail": (
                    f"Harvest is {days_to_harvest} days away. Products whose "
                    "pre-harvest interval exceeds that window are marked as not "
                    "permissible and pushed to the bottom of the ranking."
                ),
            }
        )

    # --- Rule 5: beneficial insect protection -----------------------------
    if beneficials_present:
        reasoning.append(
            {
                "rule": "Natural enemy conservation",
                "detail": (
                    "Beneficial insects were reported present, so selective "
                    "chemistry (IRAC groups 5, 18, 22A, 28) is ranked above "
                    "broad-spectrum pyrethroids, carbamates and neonicotinoids."
                ),
            }
        )

    # --- Rule 6: weather ---------------------------------------------------
    reasoning.append(
        {
            "rule": "Environmental conditions",
            "detail": (
                f"Weather reported as {weather_label}, which adjusts biological "
                "control suitability and application advice."
            ),
        }
    )

    chemical_options = _build_chemical_options(
        mapping_result,
        stage=growth_stage,
        beneficials_present=beneficials_present,
        days_to_harvest=days_to_harvest,
        weather=weather,
    )
    biological_options = _build_biological_options(profile, weather, growth_stage)

    # --- Rule 7: IRAC rotation --------------------------------------------
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
        stage_ok=stage_ok,
        severity_weight=weight,
        beneficials_present=beneficials_present,
        biological_options=biological_options,
        distinct_moa=distinct_moa,
    )

    environmental_notes: list[str] = []
    if profile and profile.get("environmental_consideration"):
        environmental_notes.append(profile["environmental_consideration"])
    if _stage_order(growth_stage) == 4:
        environmental_notes.append(
            "Crop is pollinating: avoid applications during pollen shed and "
            "spray in the early morning or late evening when bees are inactive."
        )
    if weather == "rainy":
        environmental_notes.append(
            "Wet conditions raise the risk of runoff into waterways; observe "
            "buffer zones and delay application if heavy rain is forecast."
        )
    if weather == "dry":
        environmental_notes.append(
            "Under hot, dry conditions plants are already stressed and spray "
            "drift travels further; apply in cooler hours with reduced pressure."
        )

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
        context={
            "growth_stage": growth_stage,
            "growth_stage_label": stage_label,
            "severity": severity,
            "severity_label": severity_label,
            "weather": weather,
            "weather_label": weather_label,
            "days_to_harvest": days_to_harvest,
            "beneficials_present": beneficials_present,
        },
        sources=sources or ["Curated pest profile (Excel Sheet1)"],
    )


def _build_ipm_actions(
    profile: dict[str, Any] | None,
    *,
    action_level: str,
    pest_name: str,
    stage_ok: bool,
    severity_weight: int,
    beneficials_present: bool,
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

    if action_level == "monitor":
        chemical_detail = (
            "Hold off on chemical treatment. The infestation is below the "
            "economic threshold, and spraying now would cost more than the "
            "damage prevented while removing natural enemies."
        )
    elif action_level == "preventive":
        chemical_detail = (
            "No effective rescue chemistry exists at this stage. Plan a seed "
            "treatment or soil-applied insecticide for the next planting in "
            "the affected areas."
        )
    elif not stage_ok:
        chemical_detail = (
            "The crop is outside this pest's main damage window, so delay "
            "chemical treatment and continue scouting rather than spraying now."
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

    if beneficials_present:
        actions.append(
            {
                "step": len(actions) + 1,
                "category": "Conservation",
                "title": "Protect the natural enemies present",
                "detail": (
                    "Leave untreated refuge strips where practical so predators "
                    "and parasitoids can recolonise the treated area."
                ),
            }
        )

    return actions
