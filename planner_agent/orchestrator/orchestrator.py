# orchestrator.py
# ---------- UPDATED (AGENTIC FLOW) ----------
# Orchestrates the 4-stage workflow and loops between Transport Retrieval and Plan Validation/Repair.
import json
import logging
import time
import traceback
from typing import Dict, Any
import requests
import boto3
from planner_agent.agent.transport import TransportAdapter, attach_transport_options
from planner_agent.planner_core.core import score_candidates, shortlist, assign_to_days, explain
from planner_agent.tools.config import MAX_AGENT_ITERATIONS, Summarizer_Agent_Folder, Final_ADAPTERAPI_ENDPOINT, \
    X_API_Key, Transport_Agent_Folder, TRANSPORT_ADAPTERAPI_ENDPOINT, TransportAgentARN
from planner_agent.tools.final_agent_helper import create_pdf_bytes
from planner_agent.tools.helper import aggregate_budget_range, _lunch_minutes, _pace_minutes, _minutes_for_item
from planner_agent.agent.planner_agent import PlannerAgent, CrewAIAdapter as PlannerCrewAdapter
from planner_agent.agent.final_agent import CrewAIAdapterForFinal, CrewAIAdapterForFinal
from planner_agent.tools.s3io import update_json_data, get_json_data, upload_pdf_to_s3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configurable params
TRANSPORT_ADAPTER = TransportAdapter()  # swap with real adapter if you have one
PLANNER_CREW_ADAPTER = PlannerCrewAdapter()  # replace with PlannerCrewAdapter() wired to CrewAI
FINAL_CREW_ADAPTER = CrewAIAdapterForFinal()  # replace with CrewAIAdapterForFinal()

def validate_itinerary(itinerary: Dict[str, Any], metrics: Dict[str, Any], payload: dict) -> Dict[str, Any]:
    """
    Validate gates (budget, coverage, pace) and return gates dict.
    This is intentionally deterministic; agentic decisions only occur in PlannerAgent.
    """
    req = payload.get("requirements", {})
    budget_cap = float(req.get("budget_total_sgd"))
    gates = {"budget_ok": True, "coverage_ok": True, "pace_ok": True, "uncertainty_escalate": False}

    # Build scheduled items list for budget aggregation
    scheduled_items = []
    for date, plan in itinerary.items():
        for slot in ("morning", "afternoon", "lunch"):
            for item in plan.get(slot, {}).get("items", []):
                if item:
                    scheduled_items.append({"item": item})
    #number_adult = _safe_get(req, ["optional", "adults"]) or 1
    #number_child = _safe_get(req, ["optional", "children"]) or 0
    #number_senior = _safe_get(req, ["optional", "senior"]) or 0
    group_counts = req.get("travelers")#, {"adults":number_adult,"children":number_child,"senior":number_senior})
    agg = aggregate_budget_range(scheduled_items, medians_by_type=None, group_counts=group_counts)
    total_transport_cost = sum(plan.get("metrics", {}).get("total_travel_cost_sgd", 0.0) for plan in itinerary.values())
    expected_total = agg["expected"] + total_transport_cost
    max_total = agg["max"] + total_transport_cost

    if agg["unknown_frac"] >= 0.30 or agg["uncertainty_ratio"] >= 0.10:
        gates["uncertainty_escalate"] = True

    gates["budget_ok"] = max_total <= budget_cap

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

    gates["all_ok"] = gates["budget_ok"] and gates["coverage_ok"] and gates["pace_ok"] # and not gates["uncertainty_escalate"]
    #gates["expected_total_spend_sgd"] = round(expected_total, 2)
    #gates["max_total_spend_sgd"] = round(max_total, 2)
    #gates["unknown_frac"] = agg["unknown_frac"]
    #gates["uncertainty_ratio"] = agg["uncertainty_ratio"]
    return gates

def plan_itinerary(bucket_name: str,key: str, session: str) -> Dict[str, Any]:
    """
    Orchestrator main flow (agentic-ready).
    1) Heuristic Plan (core)
    2) Transport Retrieval (attach transport)
    3) Validate -> if fail call planner agent -> reattach transport -> revalidate (loop)
    4) FinalAgent formatting
    """
    gates = []
    t0 = time.time()
    payload = get_json_data(bucket_name, key)
    fileName = key.split('/')[-1]
    # Stage 1: Heuristic plan
    scored = score_candidates(payload)
    sl = shortlist(payload, scored)
    itinerary, metrics = assign_to_days(payload, sl)
    logger.info(f"Heuristic Itinerary: {itinerary}")
    attractions = sl.get("attractions", {})
    dining = sl.get("dining", {})
    # Add itinerary to payload and upload to S3
    payload["itinerary"] = itinerary
    # Upload to Transport Agent bucket
    transport_options= {}
    update_json_data(bucket_name,Transport_Agent_Folder + "/" + fileName, payload)
    # Call Transport Agent
    response = call_transport_agent_api(bucket_name, fileName, "Planner Agent", session)
    response_data = lambda_synchronous_call(TransportAgentARN, bucket_name, fileName, "Planner Agent", session)
    """if len(response_data) != 0:
        transport_options = response_data.get("transport", {})
        itinerary = attach_transport_options(itinerary, transport_options)
        payload["itinerary"] = itinerary"""
    if len(response) != 0:
        response_data = response.json() if response else {}
        logger.info(f"Transport Agent response: {response_data}")
        statusCode = response.status_code
        transport_options= {}
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
        new_it, new_metrics = planner_agent.run(payload.get("requirements", {}), attractions, dining, itinerary, transport_options,
                                                metrics, gates)
        logger.info(f"Itinerary by Planner Agent: {itinerary}")
        if not new_it:
            # planner couldn't repair -> break and return best-effort
            break
        # Stage 2 again: reattach transport options in case structure changed (planner may have swapped items)
        # Update the new itinerary
        payload["itinerary"] = new_it
        update_json_data(bucket_name, Transport_Agent_Folder + "/" + fileName, payload)
        # Call Transport Agent
        response = call_transport_agent_api(bucket_name, fileName, "Planner Agent", session)
        """response_data = lambda_synchronous_call(TransportAgentARN, bucket_name, fileName, "Planner Agent", session)
        if len(response_data) != 0:
            transport_options = response_data.get("transport", {})
            new_it = attach_transport_options(itinerary, transport_options)
            itinerary = new_it
            payload["itinerary"]= itinerary"""
        if len(response) != 0:
            response_data = response.json() if response else {}
            logger.info(f"Transport Agent response: {response_data}")
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
    # Upload to Summarizer Agent bucket
    update_json_data(bucket_name, Summarizer_Agent_Folder + "/" + fileName, payload)
    # Call summarizer
    return sumarrizer(payload)

    """return {
        # "scored": scored,
        # "shortlist": sl,
        #"itinerary": itinerary,
        # "metrics": metrics,
        "gates": gates,
        "explanation": explanation,
        "planner_iterations": iterations,
        # "final_output": final_payload
    }"""
    #return final_payload

def sumarrizer(payload: dict):
    try:
        # Ask FinalAgent to run (keeps existing behavior)
        requirements = payload.get("requirements", {})
        itinerary = payload.get("itinerary", {})
        metrics = payload.get("metrics", {})
        explanation = payload.get("explanation", {})
        gates = payload.get("gates", {})
        final_agent = CrewAIAdapterForFinal(crew_adapter=FINAL_CREW_ADAPTER)
        response = final_agent.run(requirements, itinerary, metrics, explanation, gates)
        logger.info("Final agent returned payload (kept in logs for debugging)")

        # Build human-readable text (includes explanation and gates)
        human_text = response.get("human_summary", "")
        # Create PDF
        pdf_bytes = create_pdf_bytes(human_text, title="Final Itinerary (Human-readable)")

        # Upload PDF to S3 under final_outputs/
        pdf_key = f"final_outputs/{fileName.rsplit('.', 1)[0]}.pdf"
        try:
            presigned_url = upload_pdf_to_s3(bucket_name, pdf_key, pdf_bytes)
            logger.info(f"Uploaded PDF to s3://{bucket_name}/{pdf_key}")
            return {
                "statusCode": 200,
                "message": human_text,
                "s3_pdf_key": pdf_key,
                "s3_pdf_presigned_url": presigned_url,
                "session": session,
                "summary": {
                    "gates": gates,
                    "explanation": explanation,
                    "metrics": metrics
                }
            }
        except Exception as e:
            # fallback: return base64 PDF in response body
            logger.exception(f"Error: {e}")
            traceback.print_exc()
            statusCode = 500
            response = f"Summarizer Agent Failed with: {e}"

    except Exception as e:
        logger.exception(f"Error: {e}")
        traceback.print_exc()
        statusCode = 500
        return {"error": f"PlSummarizeranner Agent Failed with: {e}"}

def call_transport_agent_api(bucket_name: str, key: str, sender_agent: str, session: str):
    """
    Makes an API call to the specified endpoint using the provided data.
    :param bucket_name: Name of the S3 bucket
    :param key: Path to the file in the S3 bucket
    :param sender_agent: Sender agent name
    :param session: Session identifier
    :return: Response from the API as a dictionary
    """
    url = TRANSPORT_ADAPTERAPI_ENDPOINT + "/transport"
    headers = {"Content-Type": "application/json", "X-API-Key": X_API_Key}
    payload = {
        "bucket_name": bucket_name,
        "key": Transport_Agent_Folder +"/"+ key,
        "sender_agent": sender_agent,
        "session": session
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        return response
    except requests.RequestException as e:
        logging.error(f"Transport Agent API call failed: {e}")
        return {}


def call_transport_final_api(bucket_name: str, key: str, sender_agent: str, session: str):
    """
    Makes an API call to the specified endpoint using the provided data.

    :param bucket_name: Name of the S3 bucket
    :param key: Path to the file in the S3 bucket
    :param sender_agent: Sender agent name
    :param session: Session identifier
    :return: Response from the API as a dictionary
    """
    url = Final_ADAPTERAPI_ENDPOINT + "/planner/final"
    headers = {"Content-Type": "application/json"}
    payload = {
        "bucket_name": bucket_name,
        "key": Summarizer_Agent_Folder + "/" + key,
        "sender_agent": sender_agent,
        "session": session
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        # response.raise_for_status()  # Raise an exception for HTTP errors
        return response
    except requests.RequestException as e:
        logging.error(f"Final Agent API call failed: {e}")
        return {}


def lambda_synchronous_call(function_name: str, bucket_name: str, key: str, sender_agent: str, session: str) -> Dict[str, Any]:
    payload = {
        "bucket_name": bucket_name,
        "key": Transport_Agent_Folder + "/" + key,
        "sender_agent": sender_agent,
        "session": session
    }
    logger.info(f"Invoking Lambda function {function_name} with payload: {payload}")
    """
    Invokes an AWS Lambda function synchronously.

    :param function_name: Name of the Lambda function to invoke
    :param payload: Payload dictionary to pass to the Lambda function
    :return: Response from the Lambda function as a dictionary
    """
    client = boto3.client('lambda')
    try:
        # Call the Lambda function
        response = client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        # Read and parse the response payload
        response_payload = json.loads(response['Payload'].read().decode('utf-8'))
        logger.info(f"Lambda response payload: {response_payload}")
        return response_payload
    except Exception as e:
        logging.error(f"Lambda invocation failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import json
    with open("../inputs/research_output.json", "r") as f:
        payload = json.load(f)
    #scored = score_candidates(payload)                      # accepts payload["requirements"]["weights"] if present
    #sl = shortlist(payload, scored)                         # respects budget/type caps if provided
    #itinerary, metrics = assign_to_days(payload, sl)        # any duration; fills missing with (open slot)
    #print(explain(itinerary, metrics))
    session= "001"
    fileName = "research_output.json"
    gates = []
    t0 = time.time()
    gates = []
    t0 = time.time()
    bucket_name=""
    #payload = get_json_data(bucket_name, key)
    #fileName = key.split('/')[-1]
    # Stage 1: Heuristic plan
    scored = score_candidates(payload)
    sl = shortlist(payload, scored)
    itinerary, metrics = assign_to_days(payload, sl)
    logger.info(f"Heuristic Itinerary: {itinerary}")
    attractions = sl.get("attractions", {})
    dining = sl.get("dining", {})
    # Add itinerary to payload and upload to S3
    payload["itinerary"] = itinerary
    # Upload to Transport Agent bucket
    transport_options = {}
    #update_json_data(bucket_name, Transport_Agent_Folder + "/" + fileName, payload)
    # Call Transport Agent
    response = call_transport_agent_api(bucket_name, fileName, "Planner Agent", session)
    #response_data = lambda_synchronous_call(TransportAgentARN, bucket_name, fileName, "Planner Agent", session)
    """if len(response_data) != 0:
        transport_options = response_data.get("transport", {})
        itinerary = attach_transport_options(itinerary, transport_options)
        payload["itinerary"] = itinerary"""
    if len(response) != 0:
        response_data = response.json() if response else {}
        logger.info(f"Transport Agent response: {response_data}")
        statusCode = response.status_code
        transport_options = {}
        if statusCode == 200 or statusCode == 202:
            transport_options = response_data.get("transport", {})
            itinerary = attach_transport_options(itinerary, transport_options)
    # Stage 3: Validation & possible repair loop
    gates = validate_itinerary(itinerary, metrics, payload)
    planner_agent = PlannerAgent(crew_adapter=PLANNER_CREW_ADAPTER)
    iterations = 0
    # Ensure the loop runs at least once
    while (iterations == 0 or not gates.get("all_ok")) and iterations < MAX_AGENT_ITERATIONS:
        iterations += 1
        # Agentic repair: pass current itinerary, gates, metrics, shortlist
        new_it, new_metrics = planner_agent.run(payload.get("requirements", {}), attractions, dining, itinerary,
                                                transport_options,
                                                metrics, gates)
        logger.info(f"Itinerary by Planner Agent: {itinerary}")
        if not new_it:
            # planner couldn't repair -> break and return best-effort
            break
        # Stage 2 again: reattach transport options in case structure changed (planner may have swapped items)
        # Update the new itinerary
        payload["itinerary"] = new_it
        #update_json_data(bucket_name, Transport_Agent_Folder + "/" + fileName, payload)
        # Call Transport Agent
        response = call_transport_agent_api(bucket_name, fileName, "Planner Agent", session)
        """response_data = lambda_synchronous_call(TransportAgentARN, bucket_name, fileName, "Planner Agent", session)
        if len(response_data) != 0:
            transport_options = response_data.get("transport", {})
            new_it = attach_transport_options(itinerary, transport_options)
            itinerary = new_it
            payload["itinerary"]= itinerary"""
        if len(response) != 0:
            response_data = response.json() if response else {}
            logger.info(f"Transport Agent response: {response_data}")
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
    # Upload to Summarizer Agent bucket
    update_json_data(bucket_name, Summarizer_Agent_Folder + "/" + fileName, payload)
    # Call summarizer
    print(sumarrizer(payload))