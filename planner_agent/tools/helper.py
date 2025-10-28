# helper.py
# ---------- UPDATED ----------
# Helpers: imputation, aggregation, data quality penalty, transport proxy, haversine.
# NOTE: This file was updated to return per-age price tiers and compute budget aggregations.
import re
from difflib import get_close_matches
from typing import Dict, Any, Tuple, List, Iterable
from math import radians, cos, sin, asin, sqrt

from planner_agent.tools.config import senior_multiplier


# reuse your haversine from core if available; reimplement minimal here to avoid circular imports
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    try:
        R = 6371.0
        dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 2 * R * asin(sqrt(a))
    except Exception:
        return 0.0

def impute_price(candidate: dict, medians_by_type: dict = None) -> dict:
    medians_by_type = medians_by_type or {}
    tp = candidate.get("ticket_price_sgd") or {}
    adult_v = 0.0

    if isinstance(tp, dict) and tp.get("adult") not in (None, ""):
        try:
            adult_v = float(tp.get("adult") or 0.0)
            child_v = float(tp.get("child")) if tp.get("child") not in (None, "") else round(adult_v * child_multiplier,2)
            senior_v = float(tp.get("senior")) if tp.get("senior") not in (None, "") else round(
                adult_v * senior_multiplier, 2)
            return {"adult": adult_v, "child": child_v, "senior": senior_v, "method": "known_tiers",
                    "uncertainty_pct": 0.0}
        except Exception:
            pass

    it_type = (candidate.get("type") or "").lower()
    it_tags = set(token_to_canonical(x) for x in (candidate.get("tags") or []))
    it_terms = set(token_to_canonical(x) for x in (candidate.get("terms") or []))
    # unify candidate vocabulary
    cand_vocab = set(it_terms) | {it_type} | set(it_tags)

    # 2) median_by_type (expected structure: medians_by_type[type] = {"adult":..,"child":..,"senior":..} or scalar adult)
    for t in cand_vocab:
        if medians_by_type and t in medians_by_type:
            m = medians_by_type[t]
            if isinstance(m, dict):
                adult_v = float(m.get("adult", 0.0))
                child_v = float(m.get("child", adult_v * child_multiplier))
                senior_v = float(m.get("senior", adult_v * senior_multiplier))
            else:
                adult_v = float(m)
                child_v = round(adult_v * child_multiplier, 2)
                senior_v = round(adult_v * senior_multiplier, 2)
            return {"adult": adult_v, "child": child_v, "senior": senior_v, "method": "median_by_type",
                    "uncertainty_pct": 0.25}

    # 3) fallback adult
    fallback_table = {
        "park": 0.0,
        "garden": 0.0,
        "museum": 20.0,
        "zoo": 50.0,
        "theme_park": 85.0,
        "temple": 0.0,
        "landmark": 35.0,
        "dining": 20.0,
        "food": 20.0,
        "restaurant": 35.0
    }
    adult_v = 15.0
    for t in cand_vocab:
        tmp = float(fallback_table.get(t, 15.0))
        if tmp > adult_v:
            adult_v = tmp

    return {
        "adult": adult_v,
        "child": round(adult_v * child_multiplier, 2),
        "senior": round(adult_v * senior_multiplier, 2),
        "method": "fallback_table",
        "uncertainty_pct": 0.40
    }

def dq_penalty_for_item(candidate: dict) -> float:
    """
    UPDATED: data-quality penalty to nudge planner/human review when metadata missing.
    Caps at 0.5.
    """
    penalty = 0.0
    # check known price presence
    tp = candidate.get("ticket_price_sgd")
    if not tp:
        penalty += 0.15
    if candidate.get("low_carbon_score") in (None, 0):
        penalty += 0.05
    if not candidate.get("opening_hours"):
        penalty += 0.10
    return min(0.5, penalty)

def aggregate_budget_range(scheduled_items: List[dict], medians_by_type: dict = None, group_counts: dict = None) -> Dict[str, Any]:
    """
    UPDATED: Accepts a list of scheduled items (each is a dict with key 'item'),
    computes expected/min/max totals for the group using impute_price and group_counts.
    group_counts: {"adults":1,"children":0,"seniors":0}
    Returns dict with expected/min/max, unknown_frac and uncertainty_ratio.
    """
    medians_by_type = medians_by_type or {}
    group_counts = group_counts or {"adults":1,"children":0,"seniors":0}
    adults = group_counts.get("adult", 1)
    children = group_counts.get("child", 0)
    seniors = group_counts.get("senior", 0)

    expected = 0.0
    min_t = 0.0
    max_t = 0.0
    unknown_count = 0
    n = max(1, len(scheduled_items))
    for s in scheduled_items:
        it = s.get("item") if isinstance(s, dict) else s
        price = impute_price(it, medians_by_type)
        # compute expected total for this item for group
        expected_item = price["adult"] * adults + price["child"] * children + price["senior"] * seniors
        expected += expected_item
        # min/max by uncertainty_pct
        u = price.get("uncertainty_pct", 0.0)
        min_t += expected_item * (1 - u)
        max_t += expected_item * (1 + u)
        if price.get("method") != "known_tiers" and price.get("method") != "known":
            unknown_count += 1
    unknown_frac = unknown_count / n
    uncertainty_ratio = (max_t - expected) / max(1e-6, expected)
    return {"expected": expected, "min": min_t, "max": max_t, "unknown_frac": unknown_frac, "uncertainty_ratio": uncertainty_ratio}

def transport_proxy_options(from_item: dict, to_item: dict) -> List[dict]:
    """
    Simple transport options generator used when real transport API is not yet connected.
    Returns plausible transport modes with estimated duration_min, cost_sgd, co2_kg.
    """
    from_geo = from_item.get("geo") or {}
    to_geo = to_item.get("geo") or {}
    try:
        f_lat, f_lon = float(from_geo.get("latitude")), float(from_geo.get("longitude"))
        t_lat, t_lon = float(to_geo.get("latitude")), float(to_geo.get("longitude"))
        dist = _haversine_km(f_lat, f_lon, t_lat, t_lon)
    except Exception:
        dist = 2.0

    opts = []
    if dist <= 1.5:
        opts.append({"mode": "walk", "duration_min": int(dist * 20 + 5), "co2_kg": 0.0, "cost_sgd": 0.0})
    if dist <= 10:
        opts.append({"mode": "metro", "duration_min": max(8, int(dist * 5 + 5)), "co2_kg": round(dist * 0.06, 3), "cost_sgd": 1.5})
    opts.append({"mode": "taxi", "duration_min": max(5, int(dist * 3 + 5)), "co2_kg": round(dist * 0.2, 3), "cost_sgd": round(dist * 1.5 + 3, 2)})
    return opts

def compute_daily_travel_summary(day_plan: dict) -> Dict[str, Any]:
    """
    Compute total travel minutes, cost, co2 for a day from 'transport_options' list per hop.
    """
    total_min = 0
    total_cost = 0.0
    total_co2 = 0.0
    for hop in day_plan.get("transport_options", []):
        options = hop.get("options") or []
        # pick preferred option; default: metro if available, else first
        opt = next((o for o in options if o.get("mode") == "metro"), options[0] if options else None)
        if opt:
            total_min += int(opt.get("duration_min", 0))
            total_cost += float(opt.get("cost_sgd", 0.0))
            total_co2 += float(opt.get("co2_kg", 0.0))
    return {"total_travel_min": total_min, "total_travel_cost_sgd": round(total_cost, 2), "total_travel_co2_kg": round(total_co2, 3)}

def safe_item_name(slot_obj):
    """
    Robustly extract item name from a slot which may be None, or a dict possibly missing keys.
    Returns None if name not available.
    """
    if not slot_obj or not isinstance(slot_obj, dict):
        return None
    item = slot_obj.get("item")
    if not item or not isinstance(item, dict):
        return None
    return item.get("name")

def normalize_token(tok: str) -> str:
    """
    Normalize a single token:
      - lowercase
      - replace hyphens with underscore
      - remove common suffixes like '-friendly'
      - remove non-alpha chars at ends
      - naive singularization for trailing 's' (only when length>3)
    """
    if not tok:
        return ""
    s = tok.lower().strip()
    s = s.replace("-", "_")
    # remove '-friendly' style tokens: family_friendly -> family
    s = re.sub(r'(_?friendly)$', '', s)
    # remove punctuation
    s = re.sub(r'[^a-z0-9_ ]+', '', s)
    s = s.strip()
    # simple plural handling: museums -> museum (only naive)
    if len(s) > 3 and s.endswith("s"):
        s_sing = s[:-1]
        # guard against turning 'gas'->'ga'
        if len(s_sing) >= 3:
            s = s_sing
    return s

# small synonyms mapping — extendable
SYNONYMS = {
    "museum": {"museum", "museums", "gallery"},
    "park": {"park", "parks", "garden"},
    "theme_park": {"theme_park", "themepark", "theme", "amusement_park"},
    "family": {"family", "family_friendly", "familyfriendly", "family-friendly"},
    "photo_spot": {"photo_spot", "photo_spot", "photo", "photo-spot"},
    "attraction": {"attraction", "attractions"},
    "interactive": {"interactive", "hands_on"},
    "wheelchair_friendly": {"wheelchair_friendly", "accessible"},
    # add more domain-specific synonyms here
}

def build_synonym_map(synonyms: dict = None):
    """Invert SYNONYMS to map token -> canonical key"""
    synonyms = synonyms or SYNONYMS
    inv = {}
    for canon, variants in synonyms.items():
        for v in variants:
            inv[normalize_token(v)] = canon
    return inv

_SYNONYM_INV = build_synonym_map()

def token_to_canonical(tok: str) -> str:
    """
    Normalizes token and maps synonyms to canonical token if available.
    """
    n = normalize_token(tok)
    return _SYNONYM_INV.get(n, n)

# ---------- Interest scoring function ----------
def compute_interest_score(user_interests: Iterable[str], candidate_terms: Iterable[str], candidate_type: str, candidate_tags: Iterable[str]) -> float:
    """
    Compute a weighted interest score in [0..1].
    Strategy:
     - Normalize all inputs
     - Exact canonical matches -> weight 1.0
     - Synonym canonical matches (via synonyms map) -> weight 0.95 (treated as exact by canon)
     - Partial token overlap (substring or token prefix) -> weight 0.6
     - Fuzzy matches via difflib (cutoff) -> weight 0.5
    Returns average match fraction scaled by these weights vs number of user interests.
    """
    # build normalized canonical sets
    ui = set(token_to_canonical(x) for x in (user_interests or []))
    # defensive: if user provided a single dict set, handle it
    ui = {x for x in ui if x}
    it_terms = set(token_to_canonical(x) for x in (candidate_terms or []))
    it_type = token_to_canonical(candidate_type or "")
    it_tags = set(token_to_canonical(x) for x in (candidate_tags or []))
    # unify candidate vocabulary
    cand_vocab = set(it_terms) | {it_type} | set(it_tags)

    if not ui:
        return 0.6  # neutral when user provided no explicit interests

    score_sum = 0.0
    for interest in ui:
        if not interest:
            continue
        # Exact canonical match
        if interest in cand_vocab:
            score_sum += 1.0
            continue
        # Partial substring / token prefix match (e.g., 'museum' vs 'museum_exhibit')
        found_partial = False
        for cv in cand_vocab:
            if cv and (interest in cv or cv in interest):
                score_sum += 0.6
                found_partial = True
                break
        if found_partial:
            continue
        # Fuzzy match fallback with difflib
        # get_close_matches returns similar strings; cutoff tuned to 0.8 for high similarity
        candidates_list = [c for c in cand_vocab if c]
        close = get_close_matches(interest, candidates_list, n=2, cutoff=0.8)
        if close:
            score_sum += 0.5
            continue
        # no match
        score_sum += 0.0

    # normalized by number of interests (so more interests makes it harder to reach 1.0)
    return float(score_sum) / max(1, len(ui))