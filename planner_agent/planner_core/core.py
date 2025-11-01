from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional
from math import radians, cos, sin, asin, sqrt
from datetime import datetime, timedelta
import itertools
import re
import collections

from planner_agent.tools.config import child_multiplier
from planner_agent.tools.helper import dq_penalty_for_item, compute_interest_score, _haversine_km, token_to_canonical, \
    impute_price, _date_span_days, _start_date, _safe_get, _to_lower_set, _accom_latlon, _get_geo, \
    _get_open_hours_for_weekday, _candidate_interest_terms, _is_accessible, _normalize, _diet_fit_score, _cluster_id, \
    _pace_minutes, _minutes_for_item, _is_dining, _slot_times

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ----------------------------
# 1) score_candidates
# ----------------------------
"""
12 rules

Uninterest hard filter — drop any candidate whose tokens / type / tags intersect the user's uninterests.

Distance uses accommodation as origin — if accommodation lat/lon present, compute haversine distance to candidate; otherwise fallback distance = 5.0 km default.

Interest score (soft match) — Jaccard-ish overlap between user interests and candidate terms (type/tags/name); neutral default = 0.6 when no interests provided.

Accessibility boost — _is_accessible returns boolean and accessibility component = 1.0 if accessible else 0.7.

Eco (low_carbon) normalized score — normalized from low_carbon_score to 0..1 using dataset min/max.

Distance normalized/inverted — closer = higher score (normalize & invert).

Price normalized/inverted — lower adult ticket price = higher score (normalize & invert).

Opening hours normalized — compute open-hours for the itinerary weekday and normalize (more open hours → higher score).

Rating normalized — rating scaled into 0..1.

Diet fit soft scoring — for dining: if diet preference present, check tags/name for synonyms (vegan/halal/etc.) → score 1.0 if match, else 0.5; non-dining returns neutral 0.7. (So dining filtered by diet only via score, not filtered out.)

Weighted composite score with configurable weights — default weight vector is used but can be overridden from requirements.weights.

Expose metadata / clusterization — each scored item includes cluster ID (geo_cluster or grid fallback), distance, price and token terms for downstream decisions.
"""

def score_candidates(payload: dict, weights: Optional[dict] = None) -> Dict[str, dict]:
    """
    Compute composite score per candidate using only schema, not sample values.
    Uses user-provided weights if present: payload["requirements"]["weights"].
    Feature weights default (sum ≈ 1.0):
      - interest, access, eco, distance, price, opening, rating, diet
    """
    req = payload.get("requirements", {})
    cands = ((payload.get("retrieval", {}) or {}).get("places_matrix", {}) or {}).get("nodes", []) or []
    days = _date_span_days(req)
    start_dt = _start_date(req)
    weekday = start_dt.strftime("%A").lower()

    # Allow weight overrides from payload
    default_weights = {
        "interest": 0.26,
        "access": 0.10,
        "eco": 0.18,
        "distance": 0.18,
        "price": 0.10,
        "opening": 0.06,
        "rating": 0.07,
        "diet": 0.05,
    }
    payload_weights = _safe_get(req, ["weights"], {}) or {}
    if weights is None:
        weights = {**default_weights, **payload_weights}

    # Preferences
    interests = _to_lower_set(_safe_get(req, ["optional", "interests"], []))
    uninterests = _to_lower_set(_safe_get(req, ["optional", "uninterests"], []))
    diet_pref =_safe_get(req, ["optional", "dietary_preferences"])
    accom_lat, accom_lon = _accom_latlon(req)

    # Precompute bounds
    dist_vals, price_vals, open_vals, rating_vals, eco_vals = [], [], [], [], []
    number_adult = _safe_get(req,["optional","adult"]) or 1
    number_child = _safe_get(req,["optional","child"]) or 0
    number_senior = _safe_get(req,["optional","senior"]) or 0
    for it in cands:
        lat, lon = _get_geo(it)
        dist = _haversine_km(accom_lat, accom_lon, lat, lon) if (accom_lat is not None and lat is not None) else 5.0
        dist_vals.append(dist)
        price = impute_price(it)
        total_price = price["adult"] * number_adult + price["child"] * number_child + price["senior"] * number_senior
        price_vals.append(total_price)
        open_vals.append(_get_open_hours_for_weekday(it, weekday))
        rating_vals.append(float(it.get("rating", 0.0) or 0.0))
        eco_vals.append(float(it.get("low_carbon_score", 0.0) or 0.0))

    dmin, dmax = (min(dist_vals or [0.0]), max(dist_vals or [1.0]))
    pmin, pmax = (min(price_vals or [0.0]), max(price_vals or [1.0]))
    omin, omax = (min(open_vals or [0.0]), max(open_vals or [24.0]))
    rmin, rmax = (min(rating_vals or [0.0]), max(rating_vals or [5.0]))
    emin, emax = (min(eco_vals or [0.0]), max(eco_vals or [10.0]))

    out: Dict[str, dict] = {}
    for it in cands:
        pid = it.get("place_id") or it.get("id") or f"item_{len(out)+1}"

        # Hard filter: uninterests
        it_terms = _candidate_interest_terms(it)
        it_type = str(it.get("type") or "").lower()
        it_tags = _to_lower_set(it.get("tags", []))
        if uninterests & (it_terms | {it_type} | it_tags):
            continue

        # Components
        lat, lon = _get_geo(it)
        dist = _haversine_km(accom_lat, accom_lon, lat, lon) if (accom_lat is not None and lat is not None) else 5.0

        """interest_score = 0.0
        if interests:
            # Jaccard-ish overlap between user interests and candidate terms/types/tags
            overlap = interests & (it_terms | {it_type} | it_tags)
            interest_score = len(overlap) / max(len(interests), 1)
        else:
            interest_score = 0.6  # neutral if no explicit interests"""

        interest_score = compute_interest_score(interests, it_terms, it_type, it_tags)

        access_score = 1.0 if _is_accessible(it) else 0.7
        eco_score = _normalize(float(it.get("low_carbon_score", 0.0) or 0.0), emin, max(emax, emin + 0.01))
        distance_score = _normalize(dist, dmin, max(dmax, dmin + 0.01), invert=True)
        price = impute_price(it)
        total_price = price["adult"] * number_adult + price["child"] * number_child + price["senior"] * number_senior
        price_score = _normalize(total_price, pmin, max(pmax, pmin + 0.01), invert=True)
        opening_score = _normalize(_get_open_hours_for_weekday(it, weekday), omin, max(omax, omin + 0.01))

        rating_score = _normalize(float(it.get("rating", 0.0) or 0.0), rmin, max(rmax, rmin + 0.01))
        diet_score = _diet_fit_score(it, diet_pref)

        score = (
            weights.get("interest", 0) * interest_score +
            weights.get("access", 0) * access_score +
            weights.get("eco", 0) * eco_score +
            weights.get("distance", 0) * distance_score +
            weights.get("price", 0) * price_score +
            weights.get("opening", 0) * opening_score +
            weights.get("rating", 0) * rating_score +
            weights.get("diet", 0) * diet_score
        )
        # inside score_candidates loop (where 'it' is the candidate) -- after computing `score`:
        # apply a data-quality penalty to nudge planner/human review when metadata missing
        try:
            dq_pen = dq_penalty_for_item(it)
            # reduce score multiplicatively by (1 - dq_pen)
            score = score * (1.0 - dq_pen)
        except Exception:
            # if helper not available no penalty applied (backwards compatibility)
            pass

        out[pid] = {
            "score": round(float(score), 6),
            "components": {
                "interest": round(interest_score, 4),
                "access": round(access_score, 4),
                "eco": round(eco_score, 4),
                "distance": round(distance_score, 4),
                "price": round(price_score, 4),
                "opening": round(opening_score, 4),
                "rating": round(rating_score, 4),
                "diet": round(diet_score, 4),
            },
            "est_price_sgd": total_price,
            "distance_km": round(dist, 3),
            "cluster": _cluster_id(it),
            "type": it_type,
            "terms": sorted(list(it_terms | it_tags | {it_type})),
            "item": it,
        }

    return out

# ----------------------------
# 2) shortlist
# ----------------------------
"""
11 rules

Days <= 0 => empty shortlist (early exit).

Budget enforcement while selecting — running_cost compared to budget_total to avoid exceeding budget when adding attractions/dining. (Note: budget variable read from requirements["budget"] in this implementation.)

Pace → minutes capacity — pace string mapped to minutes/day via _pace_minutes (e.g., "slow" → 300 min) then subtract a daily_buffer_min (45 min) to get usable minutes/day; floor at 60 min.

One lunch per day guarantee — pick one dining item per day, prefer different clusters (spread) when possible; top-up if not enough dining candidates.

Reserve lunch time & cost before attractions — lunches consume lunch_minutes_total and running_cost deducted from total usable minutes / budget before picking attractions.

Attraction total time budget — remaining minutes after lunches allocated to attractions total (not per day).

Dominant type time cap (~60%) — any one attraction type may not consume more than ≈60% of the attraction time budget (implemented as minutes per type). This is a soft cap enforced during selection.

Two-pass selection — pass 1: prefer items that add new interest terms (diversity) or until half of available time used; pass 2: top up to fill remaining time (still obey type cap & budget).

Duration inference — visit duration uses candidate fields (visit_minutes, est_duration_min, duration_min, etc.) or type-based fallbacks; dining uses _lunch_minutes. These durations drive time checks.

Selected_time + dur and running_cost checks — each candidate must fit remaining total minutes and budget before being accepted.

Interest diversity nudging — prefer items that cover uncovered interest terms (if interests provided).
"""
def shortlist(payload: dict, scored: Dict[str, dict]) -> Dict[str, List[dict]]:
    """
    Improved pace-aware shortlist builder.

    Key improvements over original:
      - Robust price handling via helper.impute_price and group_counts (adults/children/seniors).
      - Deterministic cluster_key fallback when candidate.cluster missing (rounded geo).
      - Better 'one lunch per day' selection: prefer unique clusters, then fill by best scores.
      - Avoids object-identity checks; uses stable id/place_id to detect duplicates.
      - Normalizes types for time caps.
    """

    req = payload.get("requirements", {})

    # Days
    days = _date_span_days(req)
    if days <= 0:
        return {"attractions": [], "dining": []}

    # Budget (try several keys)
    budget_total = float(_safe_get(req, ["budget_total_sgd"]) or _safe_get(req, ["budget"]) or 1e12)
    adults = _safe_get(req, ["optional", "adult"]) or 1
    children = _safe_get(req, ["optional", "child"]) or 0
    seniors = _safe_get(req, ["optional", "senior"]) or 0

    # Interests (for diversity nudging) - keep as lowercase set
    interests = _to_lower_set(_safe_get(req, ["optional", "interests"], []))

    # Pace to time capacity
    per_day_inminute = _pace_minutes(req.get("pace"))         # 360 / 420 / 540
    daily_buffer_min = 45                                     # transit/slack each day (tunable)
    usable_minutes_per_day = max(60, per_day_inminute - daily_buffer_min)
    total_usable_minutes = usable_minutes_per_day * days

    # Sort by score (descending) — scored values are expected to contain 'score' and 'item'
    items_sorted = sorted(scored.values(), key=lambda x: x.get("score", 0.0), reverse=True)

    # Helper: robust place id
    def place_id_of(entry):
        item = entry.get("item") or {}
        return item.get("place_id") or item.get("id") or item.get("name")

    # Helper: cluster key fallback using rounded lat/lon (2 decimal ~ ~1km). ensures deterministic cluster for all items.
    def cluster_key(entry):
        item = entry.get("item") or {}
        geo = item.get("geo") or {}
        try:
            lat = float(geo.get("latitude"))
            lon = float(geo.get("longitude"))
            # round to 2 decimal places to get coarse clusters; tune if needed
            return f"{round(lat, 2)}_{round(lon, 2)}"
        except Exception:
            # fallback: if item has explicit cluster, use it; else use canonical id
            return str(item.get("cluster") or place_id_of(entry) or "unknown_cluster")

    # Split dining / non-dining pools; also enrich entries with cluster and group_est_price
    dining_pool = []
    non_dining_pool = []
    for c in items_sorted:
        item = c.get("item") or {}
        # compute group cost (adult/child/senior) using impute_price helper
        price_tiers = impute_price(item, medians_by_type=None)  # method, uncertainty included
        group_est_price = price_tiers["adult"] * adults + price_tiers["child"] * children + price_tiers["senior"] * seniors
        c["_group_est_price"] = group_est_price
        # compute estimated minutes via your _minutes_for_item helper; fall back to 60 min if missing
        try:
            dur = _minutes_for_item(c)  # keep existing API that accepts either entry or {"item":...}
            c["_est_minutes"] = int(dur or 60)
        except Exception:
            c["_est_minutes"] = 60

        # ensure cluster presence
        c["_cluster"] = cluster_key(c)

        if _is_dining(item):
            dining_pool.append(c)
        else:
            non_dining_pool.append(c)

    # ---- Dining selection: prefer spread across clusters ----
    dining_selected = []
    used_place_ids = set()
    used_clusters = set()

    # strategy:
    # 1) pick the highest scored dining per distinct cluster until we have min(days, distinct_clusters)
    # 2) then fill remaining slots by highest-scoring remaining diners

    # build best per-cluster map (score descending)
    cluster_to_best = {}
    for d in dining_pool:
        cl = d["_cluster"]
        pid = place_id_of(d)
        current = cluster_to_best.get(cl)
        if current is None or d.get("score", 0) > current.get("score", 0):
            cluster_to_best[cl] = d

    # pick best per cluster in score order
    best_cluster_entries = sorted(cluster_to_best.values(), key=lambda x: x.get("score", 0), reverse=True)
    for e in best_cluster_entries:
        if len(dining_selected) >= days:
            break
        pid = place_id_of(e)
        if pid in used_place_ids:
            continue
        dining_selected.append(e)
        used_place_ids.add(pid)
        used_clusters.add(e["_cluster"])

    # top-up by overall best dining (score) excluding already selected
    if len(dining_selected) < days:
        for d in dining_pool:
            if len(dining_selected) >= days:
                break
            pid = place_id_of(d)
            if pid in used_place_ids:
                continue
            dining_selected.append(d)
            used_place_ids.add(pid)
            used_clusters.add(d["_cluster"])

    # compute lunch minutes and running cost from selected lunches
    lunch_selected = dining_selected[:days]
    lunch_minutes_total = sum(int(x.get("_est_minutes", 60)) for x in lunch_selected)
    running_cost = sum(float(x.get("_group_est_price", 0.0) or 0.0) for x in lunch_selected)

    # Remaining minutes for attractions (non-dining)
    remaining_minutes_for_attractions = max(0, total_usable_minutes - lunch_minutes_total)

    # Soft type balance: cap ≈ 60% of total attraction *time*
    dominant_time_cap = max(60, int(remaining_minutes_for_attractions * 0.6))
    time_by_type = collections.Counter()  # minutes scheduled per type (normalized lowercase)

    covered_interest_terms = set()
    selected = []
    selected_time = 0
    # membership detection by place_id
    selected_place_ids = set()

    def violates_type_time_cap(x: dict, dur: int) -> bool:
        t = ((x.get("type") or "") or "").lower()
        if time_by_type[t] == 0:
            return False
        return (time_by_type[t] + dur) > dominant_time_cap

    # First pass: diversity-first (items that introduce new interest coverage)
    for x in non_dining_pool:
        pid = place_id_of(x)
        if pid in selected_place_ids:
            continue
        dur = int(x.get("_est_minutes", 60))
        if selected_time + dur > remaining_minutes_for_attractions:
            continue
        if running_cost + float(x.get("_group_est_price", 0.0) or 0.0) > budget_total:
            continue
        if violates_type_time_cap(x, dur):
            continue

        terms = set(map(str.lower, (x.get("terms") or [])))
        adds_new = bool((terms - covered_interest_terms) & interests) if interests else True

        # be more selective early if user has interests: prefer those that add new interest coverage
        if adds_new or selected_time < remaining_minutes_for_attractions * 0.5:
            selected.append(x)
            selected_time += dur
            time_by_type[((x.get("type") or "") or "").lower()] += dur
            running_cost += float(x.get("_group_est_price", 0.0) or 0.0)
            covered_interest_terms |= terms
            selected_place_ids.add(pid)

        if selected_time >= remaining_minutes_for_attractions:
            break

    # Second pass: top-up to fill remaining time (still respect budget & type cap)
    if selected_time < remaining_minutes_for_attractions:
        for x in non_dining_pool:
            pid = place_id_of(x)
            if pid in selected_place_ids:
                continue
            dur = int(x.get("_est_minutes", 60))
            if selected_time + dur > remaining_minutes_for_attractions:
                continue
            if running_cost + float(x.get("_group_est_price", 0.0) or 0.0) > budget_total:
                continue
            if violates_type_time_cap(x, dur):
                continue
            selected.append(x)
            selected_time += dur
            time_by_type[((x.get("type") or "") or "").lower()] += dur
            running_cost += float(x.get("_group_est_price", 0.0) or 0.0)
            selected_place_ids.add(pid)
            if selected_time >= remaining_minutes_for_attractions:
                break

    # Final output: keep same shape as before (lists of scored entries). downstream stages expect these dicts.
    return {
        "attractions": selected,
        "dining": lunch_selected
    }
# ----------------------------
# 3) assign_to_days
# ----------------------------
"""
9 rules

Use accommodation lat/lon as origin, fallback to first item (seed) if absent.

Order clusters by proximity to accommodation — clusters sorted by minimum distance of their members to accom.

Round-robin cluster cycling to fill days — itinerary cycles clusters to spread days across clusters and reduce long hops.

Morning: pick unused attraction from target cluster; fallback to any unused.

Lunch: prefer same cluster; else any unused lunch.

Afternoon: prefer attraction with different terms than morning (diversity); fallback to any unused.

Item reuse prevention — used_att_ids and used_lunch_ids tracked so same place_id not reused across slots/days.

Metrics computed — total ticket spend (morning+afternoon), approx distance (haversine between sequence + return), interest terms covered, accessible stops counted. These are used in orchestrator gating.

Afternoon/morning ticket cost counted but lunch excluded from ticket spend (implicit rule in bookkeeping).
"""
def assign_to_days(payload: dict, shortlist_out: Dict[str, List[dict]]) -> Tuple[Dict[str, dict], Dict[str, Any]]:
    """
    Builds a D-day plan with Morning/Lunch/Afternoon per day.
    Cluster-aware; minimizes hops using nearest clusters first.
    Uses accommodation location if present. If not, uses first item as origin.
    """
    req = payload.get("requirements", {})
    number_adult = _safe_get(req,["optional", "adult"]) or 1
    number_child = _safe_get(req,["optional", "child"]) or 0
    number_senior = _safe_get(req,["optional", "senior"]) or 0
    days = _date_span_days(req)
    start_dt = _start_date(req)
    slot_times = _slot_times(req)
    accom_lat, accom_lon = _accom_latlon(req)

    atts = shortlist_out.get("attractions", [])[:]
    lunches = shortlist_out.get("dining", [])[:]

    # Fallback origin if no accommodation
    if accom_lat is None or accom_lon is None:
        seed = (atts + lunches)[0]["item"] if (atts or lunches) else {}
        accom_lat, accom_lon = _get_geo(seed)

    # Order clusters by proximity to accommodation
    def cluster_distance(cluster: str) -> float:
        members = [x for x in (atts + lunches) if x["cluster"] == cluster]
        if not members:
            return 1e9
        return min(_haversine_km(accom_lat, accom_lon, *_get_geo(m["item"])) for m in members)

    clusters = sorted({x["cluster"] for x in (atts + lunches)}, key=cluster_distance)
    if not clusters:
        clusters = ["cluster_unknown"]

    atts_by_cluster = {c: [] for c in clusters}
    for a in atts:
        atts_by_cluster.setdefault(a["cluster"], []).append(a)

    lunch_by_cluster = {c: [] for c in clusters}
    for l in lunches:
        lunch_by_cluster.setdefault(l["cluster"], []).append(l)

    itinerary: Dict[str, dict] = {}
    total_tickets_cost, total_distance_km = 0.0, 0.0
    all_interest_terms, accessible_hits = set(), 0

    used_att_ids, used_lunch_ids = set(), set()
    cluster_cycle = itertools.cycle(clusters)

    for d in range(days):
        date_str = (start_dt + timedelta(days=d)).date().isoformat()
        cluster = next(cluster_cycle)

        # Ensure cluster with available attractions
        for _ in range(len(clusters)):
            if any(a["item"]["place_id"] not in used_att_ids for a in atts_by_cluster.get(cluster, [])):
                break
            cluster = next(cluster_cycle)

        # Morning
        morning = next((a for a in atts_by_cluster.get(cluster, []) if a["item"]["place_id"] not in used_att_ids), None)
        if morning is None:
            morning = next((a for a in itertools.chain.from_iterable(atts_by_cluster.values()) if a["item"]["place_id"] not in used_att_ids), None)

        # Lunch (prefer same cluster)
        lunch = next((l for l in lunch_by_cluster.get(cluster, []) if l["item"]["place_id"] not in used_lunch_ids), None)
        if lunch is None:
            lunch = next((l for l in lunches if l["item"]["place_id"] not in used_lunch_ids), None)

        # Afternoon (prefer different terms than morning)
        afternoon = None
        morning_terms = set(morning["terms"]) if morning else set()
        for a in atts_by_cluster.get(cluster, []) + list(itertools.chain.from_iterable(atts_by_cluster.values())):
            if a["item"]["place_id"] in used_att_ids:
                continue
            if morning and a["item"]["place_id"] == morning["item"]["place_id"]:
                continue
            # favor diversity in terms
            if morning and len(set(a["terms"]) - morning_terms) == 0:
                continue
            afternoon = a
            break
        if afternoon is None:
            afternoon = next((a for a in itertools.chain.from_iterable(atts_by_cluster.values())
                              if a["item"]["place_id"] not in used_att_ids and
                                 (not morning or a["item"]["place_id"] != morning["item"]["place_id"])), None)

        itinerary[date_str] = {
            "morning": {"time": slot_times["morning"], "item": morning["item"] if morning else None},
            "lunch": {"time": slot_times["lunch"], "item": lunch["item"] if lunch else None},
            "afternoon": {"time": slot_times["afternoon"], "item": afternoon["item"] if afternoon else None},
        }

        # Metrics bookkeeping
        sequence = []
        for key in ["morning", "lunch", "afternoon"]:
            it = itinerary[date_str][key]["item"]
            if it: sequence.append(it)

        prev_lat, prev_lon = accom_lat, accom_lon
        for it in sequence:
            lat, lon = _get_geo(it)
            if prev_lat is not None and lat is not None:
                total_distance_km += _haversine_km(prev_lat, prev_lon, lat, lon)
            prev_lat, prev_lon = lat, lon
        if prev_lat is not None and accom_lat is not None:
            total_distance_km += _haversine_km(prev_lat, prev_lon, accom_lat, accom_lon)

        for key in ["morning", "afternoon"]:
            it = itinerary[date_str][key]["item"]
            if it:
                used_att_ids.add(it["place_id"])
                price = impute_price(it)
                total_tickets_cost += price["adult"] * number_adult + price["child"] * number_child + price["senior"] * number_senior
                all_interest_terms |= _candidate_interest_terms(it)
                if _is_accessible(it):
                    accessible_hits += 1

        if itinerary[date_str]["lunch"]["item"]:
            used_lunch_ids.add(itinerary[date_str]["lunch"]["item"]["place_id"])

    metrics = {
        "days": days,
        "estimated_adult_ticket_spend_sgd": round(total_tickets_cost, 2),
        "approx_distance_km": round(total_distance_km, 1),
        "interest_terms_covered": sorted(list(all_interest_terms))[:40],  # trim for readability
        "accessible_stops": accessible_hits,
    }
    return itinerary, metrics

# ----------------------------
# 4) explain
# ----------------------------
def explain(requirements: Dict, itinerary: Dict[str, dict], metrics: Dict[str, Any]) -> str:
    """
    Human-readable explanation that stays generic and auditable.
    """
    lines = []
    lines.append("Plan Overview")
    lines.append(f"- {metrics.get('days', 0)} days with Morning / Lunch / Afternoon slots.")
    lines.append(f"- Estimated adult ticket spend ≈ SGD {metrics.get('estimated_adult_ticket_spend_sgd', 0)}.")
    lines.append(f"- Approx. travel distance ≈ {metrics.get('approx_distance_km', 0)} km.")
    lines.append(f"- Accessible stops counted: {metrics.get('accessible_stops', 0)}.")
    if metrics.get("interest_terms_covered"):
        lines.append(f"- Interest terms covered include: {', '.join(metrics['interest_terms_covered'][:12])}...")
    lines.append("")

    number_adult = _safe_get(requirements,["optional", "adult"]) or 1
    number_child = _safe_get(requirements,["optional", "child"]) or 0
    number_senior = _safe_get(requirements,["optional", "senior"]) or 0

    for date, plan in sorted(itinerary.items()):
        lines.append(f"{date}")
        for slot in ["morning", "lunch", "afternoon"]:
            entry = plan[slot]["item"]
            t = plan[slot]["time"]
            if not entry:
                lines.append(f"  • {slot.title()} {t}: (open slot)")
                continue
            name = entry.get("name", "Unknown")
            typ = str(entry.get("type", "attraction")).title()
            cluster = str(entry.get("geo_cluster_id") or _cluster_id(entry))
            price = impute_price(entry)
            total_price= price["adult"] * number_adult + price["child"] * number_child + price["senior"] * number_senior

            access = "✓ accessible" if _is_accessible(entry) else "—"
            lines.append(f"  • {slot.title()} {t}: {name}  [{typ}, {cluster}] — ticket ~SGD {total_price:.0f} {access}")
        lines.append("")

    lines.append("Why these picks?")
    lines.append("- Selections favor places that match stated interests, are closer to the accommodation when known,")
    lines.append("  are budget-aware when you provided a budget, include one lunch per day, and balance attraction types.")
    lines.append("- Ordering is cluster-aware to reduce transit time; afternoon differs in vibe when possible.")
    lines.append("- Ordering is cluster-aware to reduce transit time; afternoon differs in vibe when possible.")
    return "\n".join(lines)
