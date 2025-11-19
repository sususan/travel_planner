from __future__ import annotations

import datetime
import logging
from datetime import timedelta
from typing import Dict, List, Tuple, Any, Optional
from math import radians, cos, sin, asin, sqrt
import itertools
import re
import collections

import requests

from planner_agent.tools.config import TRANSPORT_ADAPTERAPI_ENDPOINT, X_API_Key, Transport_Agent_Folder
from planner_agent.tools.helper import dq_penalty_for_item, compute_interest_score, _haversine_km, token_to_canonical, \
    impute_price, _date_span_days, _start_date, _safe_get, _to_lower_set, _accom_latlon, _get_geo, \
    _get_open_hours_for_weekday, _candidate_interest_terms, _is_accessible, _normalize, _diet_fit_score, _cluster_id, \
    _pace_minutes, _minutes_for_item, _is_dining, _slot_times, _default_fix_for_gate, _generate_alternatives, \
    _compute_item_score, pick_by_cluster_roundrobin, eco_score_from_low_carbon, aggregate_budget_range, _lunch_minutes
from planner_agent.tools.s3io import get_json_data, put_json, update_json_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ----------------------------
# 1) score_candidates
# ----------------------------

def score_candidates(bucket: str, key: str, weights: None) -> dict:
    """
    Compute composite score per candidate using only schema, not sample values.
    Uses user-provided weights if present: payload["requirements"]["weights"].
    Feature weights default (sum ≈ 1.0):
      - interest, access, eco, distance, price, opening, rating, diet
    """
    print("!!score_candidates!!")
    print(f"bucket={bucket}")
    print(f"key={key}")
    payload = get_json_data(bucket, key)
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
    diet_pref = _safe_get(req, ["optional", "dietary_preferences"])
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
        total_price = price["adults"] * number_adult + price["children"] * number_child + price[
            "senior"] * number_senior
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
        pid = it.get("place_id") or it.get("id") or f"item_{len(out) + 1}"

        # Hard filter: uninterests
        it_terms = _candidate_interest_terms(it)
        it_type = str(it.get("type") or "").lower()
        it_tags = _to_lower_set(it.get("tags", []))
        if uninterests & (it_terms | {it_type} | it_tags):
            continue

        # Components
        lat, lon = _get_geo(it)
        dist = _haversine_km(accom_lat, accom_lon, lat, lon) if (accom_lat is not None and lat is not None) else 5.0

        interest_score = compute_interest_score(interests, it_terms, it_type, it_tags)

        access_score = 1.0 if _is_accessible(it) else 0.7
        eco_score = eco_score_from_low_carbon(float(it.get("low_carbon_score", 5.0) or 5.0))
        distance_score = _normalize(dist, dmin, max(dmax, dmin + 0.01), invert=True)
        price = impute_price(it)
        total_price = price["adults"] * number_adult + price["children"] * number_child + price[
            "senior"] * number_senior
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
    payload["scored"] = out
    #fileName = "../process/20251107T200155_02bb2fc0.json"
    #print(f"fileName: {key} : payload: {payload}")
    update_json_data(bucket, key, payload)
    return {"status" : "success"}

# ----------------------------
# 2) shortlist
# ----------------------------

def shortlist(bucket: str, key: str) -> dict:
    """
    Improved pace-aware shortlist builder with spatially-aligned lunch selection.

    Major improvements:
      - Determines attraction cluster centroids and prefers lunches near those centroids.
      - Avoids selecting far-away lunch stops by using a tunable max-distance (lunch_max_distance_km).
      - Deduplicates by place_id.
      - Preserves previous time/budget logic for attractions selection.
    """
    print("!!shortlist!!")
    print(f"key={key}")
    payload = get_json_data(bucket, key)
    #print(f"payload={payload}")
    req = payload.get("requirements", {}) or {}
    #print(f"requirements{req}")
    scored = payload.get("scored", {}) or {}
    #print(f"scored{scored}")
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
    per_day_inminute = _pace_minutes(req.get("pace"))  # 360 / 420 / 540
    daily_buffer_min = 30  # transit/slack each day (tunable)
    usable_minutes_per_day = max(60, per_day_inminute - daily_buffer_min)
    total_usable_minutes = usable_minutes_per_day * days

    # Tunables (override via requirements if desired)
    lunch_max_distance_km = float(
        req.get("lunch_max_distance_km", 12.0))  # prefer lunches within this distance from cluster centroid
    lunch_search_radius_km = float(
        req.get("lunch_search_radius_km", 20.0))  # fallback search radius when looking for near lunches

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
            return f"{round(lat, 2)}_{round(lon, 2)}"
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
            group_est_price = price_tiers["adults"] * adults + price_tiers["children"] * children + price_tiers[
                "senior"] * seniors
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
    cluster_coords = {}  # cluster -> list of (lat, lon)
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
        # if running_cost + float(x.get("_group_est_price", 0.0) or 0.0) > budget_total:
        # continue

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
            # if running_cost > budget_total:
            # continue
            selected.append(x)
            selected_time += dur
            time_by_type[((x.get("type") or "") or "").lower()] += dur
            running_cost += float(x.get("_group_est_price", 0.0) or 0.0)
            selected_place_ids.add(pid)
            if selected_time >= remaining_minutes_for_attractions:
                break

    payload["shortlist"] = {
        "attractions": selected,
        "dining": lunch_selected
    }
    update_json_data(bucket, key, payload)
    return {"status" : "success"}

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

def assign_to_days(bucket: str, key: str) -> dict:
    """
    Same behavior as before but:
      - Deduplicates attraction candidates by place_id (keep highest score)
      - Immediately reserves a candidate when it is selected (so it cannot be reselected later)
      - Lunch is reserved immediately when chosen
    """
    print("!!assign_to_days!!")
    payload = get_json_data(bucket, key)
    req = payload.get("requirements", {})
    shortlist_out = payload.get("shortlist", {})
    group_counts = req.get("travelers", {}) or {}
    number_adult = group_counts.get("adults", 1)
    number_child = group_counts.get("children", 0)
    number_senior = group_counts.get("senior", 0)
    days = _date_span_days(req)
    start_dt = _start_date(req)
    slot_times = _slot_times(req)
    accom_lat, accom_lon = _accom_latlon(req)
    # print("shortlist_out:", shortlist_out)
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

    clusters = sorted({x.get("cluster") for x in (atts + lunches) if x.get("cluster") is not None},
                      key=cluster_distance)
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
        lunch = next(
            (l for l in lunch_by_cluster.get(cluster, []) if l.get("item", {}).get("place_id") not in used_lunch_ids),
            None)
        if lunch is None:
            lunch = next((l for l in lunches if l.get("item", {}).get("place_id") not in used_lunch_ids), None)
        if lunch and lunch.get("item", {}).get("place_id"):
            used_lunch_ids.add(lunch["item"]["place_id"])

        # ---------- PICK MULTIPLE FOR AFTERNOON ----------
        afternoon_items = []
        remaining = afternoon_capacity
        afternoon_pool = atts_by_cluster.get(cluster, []) + list(
            itertools.chain.from_iterable(atts_by_cluster.values()))
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
            total_tickets_cost += price["adults"] * number_adult + price["children"] * number_child + price[
                "senior"] * number_senior
            all_interest_terms |= _candidate_interest_terms(it)
            if _is_accessible(it):
                accessible_hits += 1

    metrics = {
        "days": days,
        "estimated_spend_sgd": round(total_tickets_cost, 2),
        "approx_distance_km": round(total_distance_km, 1),
        "interest_terms_covered": sorted(list(all_interest_terms))[:40],
        "accessible_stops": accessible_hits,
    }
    payload["itinerary"] = itinerary
    payload["metrics"] = metrics
    update_json_data(bucket, key, payload)
    return {"status" : "success"}

def call_transport_agent_api(bucket_name: str, key: str, sender_agent: str, session: str) -> Dict:
    """
    Makes an API call to the specified endpoint using the provided data.
    :param bucket_name: Name of the S3 bucket
    :param key: Path to the file in the S3 bucket
    :param sender_agent: Sender agent name
    :param session: Session identifier
    :return: Response from the API as a dictionary
    """
    url = TRANSPORT_ADAPTERAPI_ENDPOINT + "/transport"
    print(f"Calling Transport Agent API: {url}")
    headers = {"Content-Type": "application/json", "X-API-Key": X_API_Key}
    payload = {
        "bucket_name": bucket_name,
        "key": Transport_Agent_Folder + "/" + "20251107T200155_02bb2fc0.json",
        "sender_agent": sender_agent,
        "session": session
    }
    print(f"Calling Transport Agent API payload: {payload}")
    try:
        response = requests.post(url, json=payload, headers=headers)
        #print(f"Transport Agent response: {response.json()}")
        transport_options = {}
        if response:
            response_data = response.json() if response else {}
            #print(f"Transport Agent response: {response_data}")
            statusCode = response.status_code
            if statusCode == 200 or statusCode == 202:
                transport_options = response_data.get("result", {})
            else:
                transport_options = {}
                logger.warning(f"Transport Agent returned non-200 status code: {statusCode}")

        file_payload = get_json_data(bucket_name, key)
        file_payload["transport_options"] = transport_options
        update_json_data(bucket_name, key, file_payload)
        return transport_options
    except requests.RequestException as e:
        logging.error(f"Transport Agent API call failed: {e}")
        return {"status" : "error"}

def validate_itinerary(bucket: str, key: str) -> Dict[str, Any]:
    """
    Validate gates (budget, coverage, pace) and return gates dict.
    This is intentionally deterministic; agentic decisions only occur in PlannerAgent.
    """
    print("!!validate_itinerary!!")
    print(f"key={key}")
    payload = get_json_data(bucket, key)
    req = payload.get("requirements", {})
    itinerary = payload.get("itinerary", {})
    metrics = payload.get("metrics", {})
    budget_cap = float(req.get("budget_total_sgd"))
    gates = {"budget_ok": True, "coverage_ok": True, "pace_ok": True, "uncertainty_escalate": False}

    # Build scheduled items list for budget aggregation
    scheduled_items = []
    for date, plan in itinerary.items():
        for slot in ("morning", "afternoon", "lunch"):
            for item in plan.get(slot, {}).get("items", []):
                if item:
                    scheduled_items.append({"item": item})
    # number_adult = _safe_get(req, ["optional", "adults"]) or 1
    # number_child = _safe_get(req, ["optional", "children"]) or 0
    # number_senior = _safe_get(req, ["optional", "senior"]) or 0
    group_counts = req.get("travelers")  # , {"adults":number_adult,"children":number_child,"senior":number_senior})
    agg = aggregate_budget_range(scheduled_items, medians_by_type=None, group_counts=group_counts)
    total_transport_cost = sum(plan.get("metrics", {}).get("total_travel_cost_sgd", 0.0) for plan in itinerary.values())
    expected_total = agg["expected"] + total_transport_cost
    max_total = agg["max"] + total_transport_cost

    if agg["unknown_frac"] >= 0.30 or agg["uncertainty_ratio"] >= 0.10:
        gates["uncertainty_escalate"] = True

    gates["budget_ok"] = expected_total <= budget_cap

    # coverage gate (simple): require some interest terms covered if requested
    if req.get("optional", {}).get("interests"):
        # metrics may include interest_terms_covered; fallback to True if unknown
        gates["coverage_ok"] = bool(metrics.get("interest_terms_covered"))

    # pace/time: ensure each day activity_time + travel_time <= pace_limit
    pace_minutes = _pace_minutes(req.get("pace"))
    for date, plan in itinerary.items():
        activity_minutes = 0
        for slot in ("morning", "afternoon"):
            it = plan.get(slot, {}).get("item")
            if it:
                activity_minutes += _minutes_for_item({"item": it})
        lunch_item = plan.get("lunch", {}).get("item")
        if lunch_item:
            activity_minutes += _lunch_minutes({"item": lunch_item})
        travel_min = plan.get("metrics", {}).get("total_travel_min", 0)
        if activity_minutes + travel_min > pace_minutes:
            gates["pace_ok"] = False
            break

    gates["all_ok"] = gates["budget_ok"] and gates["coverage_ok"] and gates[
        "pace_ok"]  # and not gates["uncertainty_escalate"]
    gates["expected_spend_sgd"] = round(expected_total, 2)
    gates["max_spend_sgd"] = round(max_total, 2)
    # gates["unknown_frac"] = agg["unknown_frac"]
    # gates["uncertainty_ratio"] = agg["uncertainty_ratio"]
    payload["gates"] = gates
    update_json_data(bucket, key, payload)
    return gates


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
                total_price = price["adults"] * number_adult + price["children"] * number_child + price[
                    "senior"] * number_senior

                access = "✓ accessible" if _is_accessible(entry) else "—"
                lines.append(
                    f"  • {slot.title()} {t}: {name}  [{typ}, {cluster}] — ticket ~SGD {total_price:.0f} {access}")
        lines.append("")

    lines.append("Why these picks?")
    lines.append("- Selections favor places that match stated interests, are closer to the accommodation when known,")
    lines.append(
        "  are budget-aware when you provided a budget, include one lunch per day, and balance attraction types.")
    lines.append("- Ordering is cluster-aware to reduce transit time; afternoon differs in vibe when possible.")
    lines.append("- Ordering is cluster-aware to reduce transit time; afternoon differs in vibe when possible.")
    return "\n".join(lines)

