from orchestrator import plan_itinerary
from planner_agent.planner_core.core import score_candidates, shortlist, assign_to_days, explain

if __name__ == "__main__":
    import json
    with open("../data/attraction-output-V2.json", "r") as f:
        payload = json.load(f)
    #scored = score_candidates(payload)                      # accepts payload["requirements"]["weights"] if present
    #sl = shortlist(payload, scored)                         # respects budget/type caps if provided
    #itinerary, metrics = assign_to_days(payload, sl)        # any duration; fills missing with (open slot)
    #print(explain(itinerary, metrics))
    print(plan_itinerary(payload, ""))