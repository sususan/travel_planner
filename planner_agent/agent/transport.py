# transport.py
# ---------- NEW ----------
# Provides attach_transport_options() that the orchestrator calls.
# Replace the CrewAITransportAdapter.run_trip_query with your real transport provider or CrewAI sub-agent.

from typing import Dict, Any, List

from planner_agent.tools.helper import transport_proxy_options, compute_daily_travel_summary


# TransportAdapter base: you can implement a CrewAI-backed adapter by subclassing this.
class TransportAdapter:
    def get_options_for_hop(self, from_item: dict, to_item: dict) -> List[dict]:
        """
        Return list of transport options for a hop. Override to call real API.
        """
        # Default: use local proxy
        return transport_proxy_options(from_item, to_item)

def attach_transport_options(itinerary: Dict[str, Any], transport_adapter: TransportAdapter = None) -> Dict[str, Any]:
    """
    For each day, attach transport options and compute day-level travel summary.
    """
    transport_adapter = transport_adapter or TransportAdapter()
    for date, plan in list(itinerary.items()):
        hops = []
        # define hops based on expected morning/lunch/afternoon structure
        slots = [("morning", "lunch"), ("lunch", "afternoon")]
        for a, b in slots:
            from_item = plan.get(a, {}).get("item")
            to_item = plan.get(b, {}).get("item")
            if from_item and to_item:
                opts = transport_adapter.get_options_for_hop(from_item, to_item)
                hops.append({"from_slot": a, "to_slot": b, "from": from_item.get("name"), "to": to_item.get("name"), "options": opts})
        plan["transport_options"] = hops
        # compute summary
        plan.setdefault("metrics", {}).update(compute_daily_travel_summary(plan))
    return itinerary
