from orchestrator import plan_itinerary

if __name__ == "__main__":
    import json
    with open("../inputs/attraction-output.json", "r") as f:
        payload = json.load(f)
    #scored = score_candidates(payload)                      # accepts payload["requirements"]["weights"] if present
    #sl = shortlist(payload, scored)                         # respects budget/type caps if provided
    #itinerary, metrics = assign_to_days(payload, sl)        # any duration; fills missing with (open slot)
    #print(explain(itinerary, metrics))
    print(plan_itinerary(payload, ""))