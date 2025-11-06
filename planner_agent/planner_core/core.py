from __future__ import annotations

import datetime
import logging
from datetime import timedelta
from typing import Dict, List, Tuple, Any, Optional
from math import radians, cos, sin, asin, sqrt
import itertools
import re
import collections
from planner_agent.tools.helper import dq_penalty_for_item, compute_interest_score, _haversine_km, token_to_canonical, \
    impute_price, _date_span_days, _start_date, _safe_get, _to_lower_set, _accom_latlon, _get_geo, \
    _get_open_hours_for_weekday, _candidate_interest_terms, _is_accessible, _normalize, _diet_fit_score, _cluster_id, \
    _pace_minutes, _minutes_for_item, _is_dining, _slot_times, _default_fix_for_gate, _generate_alternatives, \
    _compute_item_score, pick_by_cluster_roundrobin, eco_score_from_low_carbon

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
    group_counts = req.get("travelers") or {"adults": 1, "children": 0, "seniors": 0}
    number_adult = group_counts.get("adults", 1)
    number_child = group_counts.get("children", 0)
    number_senior = group_counts.get("senior", 0)
    for it in cands:
        lat, lon = _get_geo(it)
        dist = _haversine_km(accom_lat, accom_lon, lat, lon) if (accom_lat is not None and lat is not None) else 5.0
        dist_vals.append(dist)
        price = impute_price(it)
        total_price = price["adults"] * number_adult + price["children"] * number_child + price["senior"] * number_senior
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
        eco_score = eco_score_from_low_carbon (float(it.get("low_carbon_score", 5.0) or 5.0))
        distance_score = _normalize(dist, dmin, max(dmax, dmin + 0.01), invert=True)
        price = impute_price(it)
        total_price = price["adults"] * number_adult + price["children"] * number_child + price["senior"] * number_senior
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
def shortlistbak(payload: dict, scored: Dict[str, dict]) -> Dict[str, List[dict]]:
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
    group_counts = req.get("travelers")
    adults = group_counts.get("adults", 1)
    children = group_counts.get("children", 0)
    seniors = group_counts.get("senior", 0)

    # Interests (for diversity nudging) - keep as lowercase set
    interests = _to_lower_set(_safe_get(req, ["optional", "interests"], []))

    # Pace to time capacity
    per_day_inminute = _pace_minutes(req.get("pace"))         # 360 / 420 / 540
    daily_buffer_min = 30                                     # transit/slack each day (tunable)
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
        group_est_price = price_tiers["adults"] * adults + price_tiers["children"] * children + price_tiers["senior"] * seniors
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
        cl = d["cluster"]
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
        if e["cluster"] in used_clusters:
            continue
        dining_selected.append(e)
        used_place_ids.add(pid)
        used_clusters.add(e["cluster"])

    #best_cluster_entries = sorted(cluster_to_best.values(), key=lambda x: x.get("score", 0), reverse=True)
    #used_clusters = pick_by_cluster_roundrobin(best_cluster_entries, days)

    # top-up by overall best dining (score) excluding already selected
    if len(dining_selected) < days:
        for d in dining_pool:
            if len(dining_selected) >= days:
                break
            pid = place_id_of(d)
            if pid in used_place_ids:
                continue
            if e["cluster"] in used_clusters:
                continue
            dining_selected.append(d)
            used_place_ids.add(pid)
            used_clusters.add(d["cluster"])

    # compute lunch minutes and running cost from selected lunches
    lunch_selected = dining_selected[:days]
    lunch_minutes_total = sum(int(x.get("_est_minutes", 60)) for x in lunch_selected)
    running_cost = sum(float(x.get("_group_est_price", 0.0) or 0.0) for x in lunch_selected)

    # Remaining minutes for attractions (non-dining)
    remaining_minutes_for_attractions = max(0, total_usable_minutes - lunch_minutes_total)

    # Soft type balance: cap ≈ 60% of total attraction *time*
    dominant_time_cap = max(60, int(remaining_minutes_for_attractions * 0.8))
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
        """if violates_type_time_cap(x, dur):
            continue"""

        terms = set(map(str.lower, (x.get("terms") or [])))
        adds_new = bool((terms - covered_interest_terms) & interests) if interests else True

        # be more selective early if user has interests: prefer those that add new interest coverage
        if adds_new or selected_time < remaining_minutes_for_attractions * 0.6:
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
            if running_cost  > budget_total:
                continue
            """if violates_type_time_cap(x, dur):
                continue"""
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

def shortlist(payload: dict, scored: Dict[str, dict]) -> Dict[str, List[dict]]:
    """
    Improved pace-aware shortlist builder with spatially-aligned lunch selection.

    Major improvements:
      - Determines attraction cluster centroids and prefers lunches near those centroids.
      - Avoids selecting far-away lunch stops by using a tunable max-distance (lunch_max_distance_km).
      - Deduplicates by place_id.
      - Preserves previous time/budget logic for attractions selection.
    """
    req = payload.get("requirements", {}) or {}

    # Days
    days = _date_span_days(req)
    if days <= 0:
        return {"attractions": [], "dining": []}

    # Budget and group
    budget_total = float(_safe_get(req, ["budget_total_sgd"]) or _safe_get(req, ["budget"]) or 1e12)
    group_counts = req.get("travelers", {}) or {}
    adults = group_counts.get("adults", 1)
    children = group_counts.get("children", 0)
    seniors = group_counts.get("senior", 0)

    # Interests (for diversity nudging) - keep as lowercase set
    interests = _to_lower_set(_safe_get(req, ["optional", "interests"], []))

    # Pace to time capacity
    per_day_inminute = _pace_minutes(req.get("pace"))         # 360 / 420 / 540
    daily_buffer_min = 30                                     # transit/slack each day (tunable)
    usable_minutes_per_day = max(60, per_day_inminute - daily_buffer_min)
    total_usable_minutes = usable_minutes_per_day * days

    # Tunables (override via requirements if desired)
    lunch_max_distance_km = float(req.get("lunch_max_distance_km", 12.0))  # prefer lunches within this distance from cluster centroid
    lunch_search_radius_km = float(req.get("lunch_search_radius_km", 20.0))  # fallback search radius when looking for near lunches

    # Sort scored items by score desc
    items_sorted = sorted(scored.values(), key=lambda x: x.get("score", 0.0), reverse=True)

    # Helper: robust place id
    def place_id_of(entry):
        item = entry.get("item") or {}
        return item.get("place_id") or item.get("id") or item.get("name")

    # Helper: cluster key fallback using rounded lat/lon
    def cluster_key(entry):
        item = entry.get("item") or {}
        geo = item.get("geo") or {}
        try:
            lat = float(geo.get("latitude"))
            lon = float(geo.get("longitude"))
            return f"{round(lat,2)}_{round(lon,2)}"
        except Exception:
            return str(item.get("cluster") or place_id_of(entry) or "unknown_cluster")

    # Helper: safe geo pull
    def geo_of(entry):
        item = entry.get("item") or {}
        geo = item.get("geo") or {}
        try:
            return float(geo.get("latitude")), float(geo.get("longitude"))
        except Exception:
            return None, None

    # Haversine distance helper (km)
    def _dist_km(a_lat, a_lon, b_lat, b_lon):
        if None in (a_lat, a_lon, b_lat, b_lon):
            return float("inf")
        return _haversine_km(a_lat, a_lon, b_lat, b_lon)

    # Build dining / non-dining pools and enrich entries (group_est_price, _est_minutes, _cluster)
    dining_pool = []
    non_dining_pool = []
    # temporary map for deduplication by place_id (keep highest score)
    seen_place_ids = {}

    for c in items_sorted:
        item = c.get("item") or {}

        # compute group cost (use impute_price helper if available)
        try:
            price_tiers = impute_price(item)
            group_est_price = price_tiers["adults"] * adults + price_tiers["children"] * children + price_tiers["senior"] * seniors
        except Exception:
            group_est_price = float(c.get("_group_est_price", 0.0) or 0.0)
        c["_group_est_price"] = group_est_price

        # estimated minutes
        try:
            dur = _minutes_for_item(c)
            c["_est_minutes"] = int(dur or 60)
        except Exception:
            c["_est_minutes"] = int(c.get("_est_minutes") or 60)

        # cluster fallback
        c["_cluster"] = cluster_key(c)

        pid = place_id_of(c)
        if pid:
            prev = seen_place_ids.get(pid)
            if prev is None or c.get("score", 0) > prev.get("score", 0):
                seen_place_ids[pid] = c
        else:
            # items without pid, keep them appended with generated id
            # they are less stable but still usable
            temp_id = f"NOID_{id(c)}"
            c["_temp_id"] = temp_id
            seen_place_ids[temp_id] = c

    # Reconstruct deduped lists preserving score order by taking values from seen_place_ids
    deduped = sorted(seen_place_ids.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    for c in deduped:
        if _is_dining(c.get("item") or {}):
            dining_pool.append(c)
        else:
            non_dining_pool.append(c)

    # ---- Build cluster centroids from attraction candidates ----
    cluster_coords = {}   # cluster -> list of (lat, lon)
    cluster_score = collections.Counter()  # cluster -> cumulative score
    for a in non_dining_pool:
        cl = a.get("_cluster")
        lat, lon = geo_of(a)
        if lat is None or lon is None:
            continue
        cluster_coords.setdefault(cl, []).append((lat, lon))
        cluster_score[cl] += float(a.get("score", 0.0) or 0.0)

    cluster_centroid = {}
    for cl, pts in cluster_coords.items():
        if not pts:
            continue
        avg_lat = sum(p[0] for p in pts) / len(pts)
        avg_lon = sum(p[1] for p in pts) / len(pts)
        cluster_centroid[cl] = (avg_lat, avg_lon)

    # Order clusters by descending cumulative attraction score (priority where attractions concentrate)
    prioritized_clusters = [c for c, _ in cluster_score.most_common()]

    # ---- Dining selection: choose lunches aligned to attraction clusters ----
    dining_selected = []
    used_place_ids = set()
    used_clusters = set()

    # Build mapping cluster -> lunches (with geo)
    lunches_by_cluster = {}
    for d in dining_pool:
        cl = d.get("_cluster")
        lunches_by_cluster.setdefault(cl, []).append(d)
    # ensure lunches sorted by score in each cluster
    for cl in lunches_by_cluster:
        lunches_by_cluster[cl].sort(key=lambda x: x.get("score", 0), reverse=True)

    # 1) Try to pick at most one lunch per prioritized cluster (up to days)
    for cl in prioritized_clusters:
        if len(dining_selected) >= days:
            break
        centroid = cluster_centroid.get(cl)
        # prefer same-cluster lunch that is within lunch_max_distance_km of centroid
        candidate = None
        # first try same-cluster lunches
        for l in lunches_by_cluster.get(cl, []):
            pid = place_id_of(l)
            if pid in used_place_ids:
                continue
            lat, lon = geo_of(l)
            if centroid and lat is not None:
                dist = _dist_km(centroid[0], centroid[1], lat, lon)
            else:
                dist = float("inf")
            # accept if within preferred threshold
            if dist <= lunch_max_distance_km:
                candidate = l
                break
        # if no same-cluster within threshold, try nearest lunch across all dining_pool
        if candidate is None:
            nearest = None
            nearest_d = float("inf")
            for l in dining_pool:
                pid = place_id_of(l)
                if pid in used_place_ids:
                    continue
                lat, lon = geo_of(l)
                if centroid and lat is not None:
                    dkm = _dist_km(centroid[0], centroid[1], lat, lon)
                else:
                    dkm = float("inf")
                # constrained to reasonable search radius
                if dkm <= lunch_search_radius_km and dkm < nearest_d:
                    nearest = l
                    nearest_d = dkm
            candidate = nearest

        if candidate:
            pid = place_id_of(candidate)
            if pid not in used_place_ids:
                dining_selected.append(candidate)
                used_place_ids.add(pid)
                used_clusters.add(candidate.get("_cluster"))

    # 2) Top-up remaining lunch slots by highest scoring lunches that are reasonably close to any cluster centroid
    if len(dining_selected) < days:
        # prepare list of lunchtime candidates sorted by score
        for l in dining_pool:
            if len(dining_selected) >= days:
                break
            pid = place_id_of(l)
            if pid in used_place_ids:
                continue
            # compute min distance to any cluster centroid (if no centroid, accept)
            min_d = float("inf")
            for cl, centroid in cluster_centroid.items():
                latc, lonc = centroid
                lat, lon = geo_of(l)
                if lat is None:
                    continue
                dkm = _dist_km(latc, lonc, lat, lon)
                if dkm < min_d:
                    min_d = dkm
            # Accept if near at least one attraction cluster or if there are no clusters (fallback)
            if (cluster_centroid and min_d <= lunch_search_radius_km) or not cluster_centroid:
                dining_selected.append(l)
                used_place_ids.add(pid)
                used_clusters.add(l.get("_cluster"))

    # 3) Final fallback: if still not enough lunches, pick top scoring remaining lunches ignoring distance
    if len(dining_selected) < days:
        for l in dining_pool:
            if len(dining_selected) >= days:
                break
            pid = place_id_of(l)
            if pid in used_place_ids:
                continue
            dining_selected.append(l)
            used_place_ids.add(pid)
            used_clusters.add(l.get("_cluster"))

    # Now compute lunch minutes and running cost (for attraction budget/time calculations)
    lunch_selected = dining_selected[:days]
    lunch_minutes_total = sum(int(x.get("_est_minutes", 60)) for x in lunch_selected)
    running_cost = sum(float(x.get("_group_est_price", 0.0) or 0.0) for x in lunch_selected)

    # Remaining minutes for attractions (non-dining)
    remaining_minutes_for_attractions = max(0, total_usable_minutes - lunch_minutes_total)

    # Soft type balance: cap ≈ 80% of remaining attraction time (tunable)
    dominant_time_cap = max(60, int(remaining_minutes_for_attractions * 0.8))
    time_by_type = collections.Counter()  # minutes scheduled per type (normalized lowercase)

    covered_interest_terms = set()
    selected = []
    selected_time = 0
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

        terms = set(map(str.lower, (x.get("terms") or [])))
        adds_new = bool((terms - covered_interest_terms) & interests) if interests else True

        # be more selective early if user has interests: prefer those that add new interest coverage
        if adds_new or selected_time < remaining_minutes_for_attractions * 0.6:
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
            if running_cost > budget_total:
                continue
            selected.append(x)
            selected_time += dur
            time_by_type[((x.get("type") or "") or "").lower()] += dur
            running_cost += float(x.get("_group_est_price", 0.0) or 0.0)
            selected_place_ids.add(pid)
            if selected_time >= remaining_minutes_for_attractions:
                break

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
def assign_to_daysBak(payload: dict, shortlist_out: Dict[str, List[dict]]) -> Tuple[Dict[str, dict], Dict[str, Any]]:
    """
    Builds a D-day plan with Morning/Lunch/Afternoon per day.
    Cluster-aware; minimizes hops using nearest clusters first.
    Uses accommodation location if present. If not, uses first item as origin.
    """
    req = payload.get("requirements", {})
    group_counts = req.get("travelers")
    number_adult = group_counts.get("adults", 1)
    number_child = group_counts.get("children", 0)
    number_senior = group_counts.get("senior", 0)
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
                total_tickets_cost += price["adults"] * number_adult + price["children"] * number_child + price["senior"] * number_senior
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

import itertools
from datetime import timedelta
from typing import Dict, List, Tuple, Any

def assign_to_days(payload: dict, shortlist_out: Dict[str, List[dict]]) -> Tuple[Dict[str, dict], Dict[str, Any]]:
    """
    Same behavior as before but:
      - Deduplicates attraction candidates by place_id (keep highest score)
      - Immediately reserves a candidate when it is selected (so it cannot be reselected later)
      - Lunch is reserved immediately when chosen
    """
    req = payload.get("requirements", {})
    group_counts = req.get("travelers", {}) or {}
    number_adult = group_counts.get("adults", 1)
    number_child = group_counts.get("children", 0)
    number_senior = group_counts.get("senior", 0)
    days = _date_span_days(req)
    start_dt = _start_date(req)
    slot_times = _slot_times(req)
    accom_lat, accom_lon = _accom_latlon(req)

    raw_atts = shortlist_out.get("attractions", [])[:]
    lunches = shortlist_out.get("dining", [])[:]

    # --- Deduplicate attractions by place_id (keep highest score) ---
    atts_by_id: Dict[str, dict] = {}
    for a in raw_atts:
        pid = a.get("item", {}).get("place_id")
        if not pid:
            # keep items without place_id as-is (they won't be deduped)
            continue
        prev = atts_by_id.get(pid)
        if prev is None or a.get("score", 0) > prev.get("score", 0):
            atts_by_id[pid] = a
    # Reconstruct atts list: include deduped ones plus any original items that had no place_id
    atts = [v for v in atts_by_id.values()] + [a for a in raw_atts if not a.get("item", {}).get("place_id")]

    # fallback origin if no accommodation
    if accom_lat is None or accom_lon is None:
        seed = (atts + lunches)[0]["item"] if (atts or lunches) else {}
        accom_lat, accom_lon = _get_geo(seed)

    def est_minutes_for(cand: dict) -> int:
        return int(cand.get("_est_minutes") or cand.get("item", {}).get("duration_recommended_minutes") or 90)

    # Capacity tuning by pace (can override in requirements)
    pace = (req.get("pace") or "").lower()
    if req.get("slot_capacity_minutes"):  # optional explicit override
        morning_capacity = afternoon_capacity = int(req.get("slot_capacity_minutes"))
    else:
        if pace in ("slow", "relaxed"):
            morning_capacity = afternoon_capacity = 120
        elif pace in ("fast", "packed"):
            morning_capacity = afternoon_capacity = 300
        elif pace == "moderate":
            morning_capacity = afternoon_capacity = 180
        else:  # normal or unspecified
            morning_capacity = afternoon_capacity = 180

    # Order clusters by proximity to accommodation
    def cluster_distance(cluster: str) -> float:
        members = [x for x in (atts + lunches) if x.get("cluster") == cluster]
        if not members:
            return 1e9
        return min(_haversine_km(accom_lat, accom_lon, *_get_geo(m["item"])) for m in members)

    clusters = sorted({x.get("cluster") for x in (atts + lunches) if x.get("cluster") is not None}, key=cluster_distance)
    if not clusters:
        clusters = ["cluster_unknown"]

    # group candidates by cluster and sort by score desc
    atts_by_cluster = {c: [] for c in clusters}
    for a in atts:
        c = a.get("cluster") or "cluster_unknown"
        atts_by_cluster.setdefault(c, []).append(a)
    for c in list(atts_by_cluster.keys()):
        atts_by_cluster[c].sort(key=lambda x: x.get("score", 0), reverse=True)

    lunch_by_cluster = {c: [] for c in clusters}
    for l in lunches:
        c = l.get("cluster") or "cluster_unknown"
        lunch_by_cluster.setdefault(c, []).append(l)
    for c in list(lunch_by_cluster.keys()):
        lunch_by_cluster[c].sort(key=lambda x: x.get("score", 0), reverse=True)

    itinerary: Dict[str, dict] = {}
    total_tickets_cost, total_distance_km = 0.0, 0.0
    all_interest_terms, accessible_hits = set(), 0

    used_att_ids, used_lunch_ids = set(), set()
    cluster_cycle = itertools.cycle(clusters)

    for d in range(days):
        date_str = (start_dt + timedelta(days=d)).date().isoformat()
        # pick starting cluster for day
        cluster = next(cluster_cycle)
        # advance until cluster with unused attractions (or stop after full rotation)
        for _ in range(len(clusters)):
            if any(a.get("item", {}).get("place_id") not in used_att_ids for a in atts_by_cluster.get(cluster, [])):
                break
            cluster = next(cluster_cycle)

        # ---------- PICK MULTIPLE FOR MORNING ----------
        morning_items = []
        remaining = morning_capacity
        morning_pool = atts_by_cluster.get(cluster, []) + list(itertools.chain.from_iterable(atts_by_cluster.values()))
        # filter pool to candidates not already used
        morning_pool = [c for c in morning_pool if c.get("item", {}).get("place_id") not in used_att_ids]

        for cand in morning_pool:
            pid = cand.get("item", {}).get("place_id")
            # skip if no pid but already used by identity (rare)
            if pid and pid in used_att_ids:
                continue
            est = est_minutes_for(cand)
            if est <= remaining or (remaining >= 30 and est <= remaining + 20):
                morning_items.append(cand)
                # reserve immediately to prevent re-selection across days/slots
                if pid:
                    used_att_ids.add(pid)
                remaining -= est
            if remaining <= 0:
                break

        # fallback single if none selected but there are unused in pool
        if not morning_items:
            fallback = next((a for a in morning_pool if a.get("item", {}).get("place_id") not in used_att_ids), None)
            if fallback:
                pid = fallback.get("item", {}).get("place_id")
                morning_items.append(fallback)
                if pid:
                    used_att_ids.add(pid)

        # ---------- PICK LUNCH (single) ----------
        lunch = next((l for l in lunch_by_cluster.get(cluster, []) if l.get("item", {}).get("place_id") not in used_lunch_ids), None)
        if lunch is None:
            lunch = next((l for l in lunches if l.get("item", {}).get("place_id") not in used_lunch_ids), None)
        if lunch and lunch.get("item", {}).get("place_id"):
            used_lunch_ids.add(lunch["item"]["place_id"])

        # ---------- PICK MULTIPLE FOR AFTERNOON ----------
        afternoon_items = []
        remaining = afternoon_capacity
        afternoon_pool = atts_by_cluster.get(cluster, []) + list(itertools.chain.from_iterable(atts_by_cluster.values()))
        # avoid items already reserved (either earlier days or morning of same day)
        afternoon_pool = [c for c in afternoon_pool if c.get("item", {}).get("place_id") not in used_att_ids]

        for cand in afternoon_pool:
            pid = cand.get("item", {}).get("place_id")
            if pid and pid in used_att_ids:
                continue
            est = est_minutes_for(cand)
            if est <= remaining or (remaining >= 30 and est <= remaining + 20):
                afternoon_items.append(cand)
                if pid:
                    used_att_ids.add(pid)
                remaining -= est
            if remaining <= 0:
                break

        # fallback single if none found
        if not afternoon_items:
            fallback = next((a for a in afternoon_pool if a.get("item", {}).get("place_id") not in used_att_ids), None)
            if fallback:
                pid = fallback.get("item", {}).get("place_id")
                afternoon_items.append(fallback)
                if pid:
                    used_att_ids.add(pid)

        # Save itinerary for the day
        itinerary[date_str] = {
            "morning": {"time": slot_times["morning"], "items": [c["item"] for c in morning_items]},
            "lunch": {"time": slot_times["lunch"], "items": [lunch["item"]] if lunch else []},
            "afternoon": {"time": slot_times["afternoon"], "items": [c["item"] for c in afternoon_items]},
        }

        # --- bookkeeping: distance/cost/accessibility/interest_terms ---
        sequence = []
        for mi in morning_items:
            sequence.append(mi["item"])
        if lunch and lunch.get("item"):
            sequence.append(lunch["item"])
        for ai in afternoon_items:
            sequence.append(ai["item"])

        prev_lat, prev_lon = accom_lat, accom_lon
        for it in sequence:
            lat, lon = _get_geo(it)
            if prev_lat is not None and lat is not None:
                total_distance_km += _haversine_km(prev_lat, prev_lon, lat, lon)
            prev_lat, prev_lon = lat, lon
        if prev_lat is not None and accom_lat is not None:
            total_distance_km += _haversine_km(prev_lat, prev_lon, accom_lat, accom_lon)

        for it in sequence:
            pid = it.get("place_id")
            if not pid:
                continue
            price = impute_price(it)
            total_tickets_cost += price["adults"] * number_adult + price["children"] * number_child + price["senior"] * number_senior
            all_interest_terms |= _candidate_interest_terms(it)
            if _is_accessible(it):
                accessible_hits += 1

    metrics = {
        "days": days,
        "estimated_adult_ticket_spend_sgd": round(total_tickets_cost, 2),
        "approx_distance_km": round(total_distance_km, 1),
        "interest_terms_covered": sorted(list(all_interest_terms))[:40],
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
    group_counts = requirements.get("travelers")
    number_adult = group_counts.get("adults", 1)
    number_child = group_counts.get("children", 0)
    number_senior = group_counts.get("senior", 0)

    for date, plan in sorted(itinerary.items()):
        lines.append(f"{date}")
        for slot in ["morning", "lunch", "afternoon"]:
            for entry in plan[slot]["items"]:
                t = plan[slot]["time"]
                if not entry:
                    lines.append(f"  • {slot.title()} {t}: (open slot)")
                    continue
                name = entry.get("name", "Unknown")
                typ = str(entry.get("type", "attraction")).title()
                cluster = str(entry.get("geo_cluster_id") or _cluster_id(entry))
                price = impute_price(entry)
                total_price= price["adults"] * number_adult + price["children"] * number_child + price["senior"] * number_senior
    
                access = "✓ accessible" if _is_accessible(entry) else "—"
                lines.append(f"  • {slot.title()} {t}: {name}  [{typ}, {cluster}] — ticket ~SGD {total_price:.0f} {access}")
        lines.append("")

    lines.append("Why these picks?")
    lines.append("- Selections favor places that match stated interests, are closer to the accommodation when known,")
    lines.append("  are budget-aware when you provided a budget, include one lunch per day, and balance attraction types.")
    lines.append("- Ordering is cluster-aware to reduce transit time; afternoon differs in vibe when possible.")
    lines.append("- Ordering is cluster-aware to reduce transit time; afternoon differs in vibe when possible.")
    return "\n".join(lines)

def explainbak(requirements: Dict, itinerary: Dict[str, dict], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a structured explanation + human-readable summary.
    Structured fields are machine-actionable and auditable.
    """
    # Metadata for reproducibility
    meta = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "planner_version": "planner-v1.2",   # update dynamically
        "ruleset": ["budget_gate_v1", "carbon_gate_v1"]
    }

    # Top-level human summary
    human_lines = []
    human_lines.append("Plan overview:")
    human_lines.append(f"- {metrics.get('days', 0)} days; ~{metrics.get('approx_distance_km', 0)} km travel.")
    human_lines.append(f"- Estimated cost: SGD {metrics.get('total_cost_sgd', 0):.0f}; carbon ≈ {metrics.get('carbon_kg', 0):.1f} kg.")
    if metrics.get("issues"):
        human_lines.append(f"- Flags: {', '.join(metrics['issues'])}")
    human_text = "\n".join(human_lines)

    # Structured per-slot decisions
    decisions: List[Dict[str, Any]] = []
    number_adult = (requirements.get("optional") or {}).get("adults", 1)
    number_child = (requirements.get("optional") or {}).get("children", 0)
    number_senior = (requirements.get("optional") or {}).get("senior", 0)

    for date, plan in sorted(itinerary.items()):
        for slot in ["morning", "lunch", "afternoon"]:
            entry = plan[slot].get("item")
            t = plan[slot].get("time")
            if not entry:
                decisions.append({
                    "date": date,
                    "slot": slot,
                    "time": t,
                    "choice": None,
                    "reason_list": ["open_slot"],
                    "score": None,
                    "confidence": 0.25
                })
                continue

            name = entry.get("name", "Unknown")
            typ = entry.get("type", "attraction")
            cluster = entry.get("geo_cluster_id") or entry.get("cluster") or "unknown"
            price = impute_price(entry)
            total_price = price["adults"] * number_adult + price["children"] * number_child + price["senior"] * number_senior

            # Compose a small decision trace
            reason_list = []
            # Example: match interest
            if any(term.lower() in (requirements.get("interests") or []) for term in (entry.get("tags") or [])):
                reason_list.append("matches_interest")
            # Proximity rule
            if metrics.get("accommodation_cluster") and cluster == metrics["accommodation_cluster"]:
                reason_list.append("proximal_to_accommodation")
            # Eco preference
            if requirements.get("eco_preference") == "low" and entry.get("preferred_for_eco"):
                reason_list.append("eco_friendly")
            # Budget sensitivity
            if metrics["estimated_remaining_budget_sgd"] is not None:
                if total_price > metrics.get("estimated_remaining_budget_sgd", float('inf')):
                    reason_list.append("cost_risky")
            # Accessibility
            if _is_accessible(entry):
                reason_list.append("accessible")

            # Score: small composite (0..1) — tune to your scoring function
            score = _compute_item_score(entry=entry, requirements=requirements, metrics=metrics)

            decisions.append({
                "date": date,
                "slot": slot,
                "time": t,
                "choice": {"name": name, "type": typ, "cluster": cluster},
                "ticket_est_sgd": total_price,
                "reason_list": reason_list,
                "score": round(score, 3),
                "confidence": round(0.6 + 0.4 * score, 3),  # simple mapping: higher score → higher confidence
                "provenance": entry.get("source") or "catalog",
                "alternatives": _generate_alternatives(entry, requirements, metrics, top_n=2)
            })

    # Gate summary + fixes (if any gate failed)
    gates = metrics.get("gates", {})
    gate_results = []
    suggested_actions = []
    for gname, ginfo in gates.items():
        gate_results.append({"gate": gname, "status": ginfo.get("status"), "explanation": ginfo.get("explanation")})
        if ginfo.get("status") == "FAILED":
            # Provide a short actionable fix
            suggested_actions.append({
                "gate": gname,
                "fix": ginfo.get("fix_suggestion") or _default_fix_for_gate(gname)
            })

    structured = {
        "meta": meta,
        "summary_metrics": {
            "days": metrics.get("days"),
            "total_cost_sgd": metrics.get("total_cost_sgd"),
            "carbon_kg": metrics.get("carbon_kg"),
            "accessible_stops": metrics.get("accessible_stops")
        },
        "decisions": decisions,
        "gates": gate_results,
        "suggested_actions": suggested_actions
    }

    return {"human_text": human_text, "structured": structured}

