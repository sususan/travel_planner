import time

from orchestrator import plan_itinerary, call_transport_agent_api, validate_itinerary, PLANNER_CREW_ADAPTER
from planner_agent.agent.planner_agent import PlannerAgent
from planner_agent.agent.transport import attach_transport_options
from planner_agent.planner_core.core import score_candidates, shortlist, assign_to_days, explain
from planner_agent.tools.config import Transport_Agent_Folder, MAX_AGENT_ITERATIONS, S3_BUCKET

if __name__ == "__main__":
    import json
    with open("../inputs/research_output.json", "r") as f:
        payload = json.load(f)
    #scored = score_candidates(payload)                      # accepts payload["requirements"]["weights"] if present
    #sl = shortlist(payload, scored)                         # respects budget/type caps if provided
    #itinerary, metrics = assign_to_days(payload, sl)        # any duration; fills missing with (open slot)
    #print(explain(itinerary, metrics))
    session= "001"
    fileName = "attraction-output-v2.json"
    gates = []
    t0 = time.time()
    # Stage 1: Heuristic plan
    scored = score_candidates(payload)
    sl = shortlist(payload, scored)
    attractions = sl.get("attractions", {})
    dining = sl.get("dining", {})
    itinerary, metrics = assign_to_days(payload, sl)
    print(f"Heuristic Itinerary: {itinerary}")
    # Add itinerary to payload and upload to S3
    payload["itinerary"] = itinerary
    # Upload to Transport Agent bucket
    # Call Transport Agent
    transport_options = {}
    response = call_transport_agent_api(S3_BUCKET, fileName, "Planner Agent", session)
    if len(response) != 0:
        response_data = response.json() if response else {}
        print(f"Transport Agent response: {response_data}")
        statusCode = response.status_code
        if statusCode == 200 or statusCode == 202:
            transport_options = response_data.get("transport", {})
            itinerary = attach_transport_options(itinerary, transport_options)
    # Stage 3: Validation & possible repair loop
    gates=  validate_itinerary(itinerary, metrics, payload)
    planner_agent = PlannerAgent(crew_adapter=PLANNER_CREW_ADAPTER)
    iterations = 0
    # Ensure the loop runs at least once
    while (iterations == 0 or not gates.get("all_ok")) and iterations < MAX_AGENT_ITERATIONS:
        iterations += 1
        # Agentic repair: pass current itinerary, gates, metrics, shortlist
        new_it, new_metrics = planner_agent.run(payload.get("requirements", {}), attractions, dining, itinerary, transport_options, metrics, gates)
        print(f"Itinerary by planner: {new_it}")
        if not new_it:
            # planner couldn't repair -> break and return best-effort
            break
        # Stage 2 again: reattach transport options in case structure changed (planner may have swapped items)
        # Update the new itinerary
        payload["itinerary"] = new_it
        # Call Transport Agent
        response = call_transport_agent_api(S3_BUCKET, fileName, "Planner Agent", session)
        if len(response) != 0:
            response_data = response.json() if response else {}
            print(f"Transport Agent response: {response_data}")
            statusCode = response.status_code
            if statusCode == 200 or statusCode == 202:
                transport_options = response_data.get("transport", {})
                new_it = attach_transport_options(itinerary, transport_options)
                itinerary = new_it
                payload["itinerary"] = itinerary

        metrics = new_metrics or metrics
        gates = validate_itinerary(itinerary, metrics, payload)
        # loop continues until gates pass or max iterations exhausted

    requirements = payload.get("requirements", {})
    itinerary = payload.get("itinerary", {})
    explanation = explain(requirements, itinerary, metrics)
    # Upload to Summarizer Agent
    payload["gates"] = gates
    payload["metrics"] = metrics
    payload["explanation"] = explanation
    print(f"Final output: {payload}")
    # Upload to Summarizer Agent bucket