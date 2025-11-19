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
from botocore.config import Config
from planner_agent.planner_core.core import score_candidates, shortlist, assign_to_days, explain, \
    call_transport_agent_api, validate_itinerary
from planner_agent.tools.config import MAX_AGENT_ITERATIONS, Summarizer_Agent_Folder, Final_ADAPTERAPI_ENDPOINT, \
    X_API_Key, Transport_Agent_Folder, TRANSPORT_ADAPTERAPI_ENDPOINT, TransportAgentARN, S3_BUCKET
from planner_agent.tools.final_agent_helper import create_pdf_bytes, create_pdf_bytes_plain_from_html
from planner_agent.tools.helper import aggregate_budget_range, _lunch_minutes, _pace_minutes, _minutes_for_item
from planner_agent.agent.planner_agent import PlannerAgent
from planner_agent.agent.final_agent import  CrewAIAdapter
from planner_agent.tools.s3io import update_json_data, get_json_data, upload_pdf_to_s3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configurable params
PLANNER_CREW_ADAPTER = PlannerAgent()  # replace with PlannerCrewAdapter() wired to CrewAI
FINAL_CREW_ADAPTER = CrewAIAdapter()  # replace with CrewAIAdapterForFinal()

def plan_itinerary(bucket_name: str, key: str, session: str) -> Dict[str, Any]:
    planner_agent = PlannerAgent(crew_adapter=PLANNER_CREW_ADAPTER)
    payload = get_json_data(bucket_name, key)
    planner_agent = PlannerAgent()
    planner_agent.run(payload, bucket_name, key, session)
    # Get updated payload
    payload = get_json_data(bucket_name, key)
    summarize = sumarrizer(payload, transport_options, bucket_name, key, session)
    print(f"Summarizer Agent returned payload: {summarize}")
    return summarize

def plan_itinerarybak(bucket_name: str, key: str, session: str) -> Dict[str, Any]:
    """
    Orchestrator main flow (agentic-ready).
    1) Heuristic Plan (core)
    2) Transport Retrieval (attach transport)
    3) Validate -> if fail call planner agent -> reattach transport -> revalidate (loop)
    4) FinalAgent formatting
    """
    gates = []
    t0 = time.time()
    transport_options= {}
    payload = get_json_data(bucket_name, key)
    fileName = key.split('/')[-1]
    # Stage 1: Heuristic plan
    scored = score_candidates(payload)
    sl = shortlist(payload, scored)
    itinerary, metrics = assign_to_days(payload, sl)
    print(f"Heuristic Itinerary: {itinerary}")
    attractions = sl.get("attractions", {})
    dining = sl.get("dining", {})
    # Add itinerary to payload and upload to S3
    payload["itinerary"] = itinerary
    # Upload to Transport Agent bucket
    update_json_data(bucket_name,Transport_Agent_Folder + "/" + fileName, payload)
    # Call Transport Agent
    #accomodation = payload.get("requirements", {}).get("accomodation", {})
    response = call_transport_agent_api(bucket_name, fileName, "Planner Agent", session)
    #response_data = lambda_synchronous_call(TransportAgentARN, bucket_name, fileName, "Planner Agent", session)
    """if len(response_data) != 0:
        transport_options = response_data.get("result", {})
        itinerary = attach_transport_options(itinerary, transport_options)
        payload["itinerary"] = itinerary"""
    if response:
        response_data = response.json() if response else {}
        print(f"Transport Agent response: {response_data}")
        statusCode = response.status_code
        if statusCode == 200 or statusCode == 202:
            transport_options = response_data.get("result", {})
            #itinerary = attach_transport_options(itinerary, transport_options)
    # Stage 3: Validation & possible repair loop
    gates=  validate_itinerary(itinerary, metrics, payload)
    planner_agent = PlannerAgent(crew_adapter=PLANNER_CREW_ADAPTER)
    iterations = 0
    # Ensure the loop runs at least once
    while (iterations == 0 or not gates.get("all_ok")) and iterations < MAX_AGENT_ITERATIONS:
        iterations += 1
        # Agentic repair: pass current itinerary, gates, metrics, shortlist
        new_it, new_metrics = planner_agent.plan(payload)
                               #.run(payload.get("requirements", {}), attractions, sl, dining, itinerary, transport_options,  metrics, gates))
        print(f"Itinerary by Planner Agent: {new_it}")
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
            transport_options = response_data.get("result", {})
            new_it = attach_transport_options(itinerary, transport_options)
            itinerary = new_it
            payload["itinerary"]= itinerary"""
        if response:
            response_data = response.json() if response else {}
            print(f"Transport Agent response: {response_data}")
            statusCode = response.status_code
            if statusCode == 200 or statusCode == 202:
                transport_options = response_data.get("result", {})
                #new_it = attach_transport_options(itinerary, transport_options)
                #itinerary = new_it
                #payload["itinerary"] = itinerary

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
    print("Calling Summarizer Agent")

    summarize = sumarrizer(payload, transport_options, bucket_name, fileName, session)
    print(f"Summarizer Agent returned payload: {summarize}")
    return summarize

    #return final_payload

def sumarrizer(payload: dict, transport_options: dict, bucket_name: str, fileName: str, session: str = ""):
    global pdf_key
    try:
        # Ask FinalAgent to run (keeps existing behavior)
        requirements = payload.get("requirements", {})
        itinerary = payload.get("itinerary", {})
        metrics = payload.get("metrics", {})
        explanation = payload.get("explanation", {})
        gates = payload.get("gates", {})
        final_agent = CrewAIAdapter()
        response = final_agent.run(itinerary, transport_options, metrics,  gates, requirements, explanation)
        print(f"Final Agent response: {response}")

        # Build human-readable text (includes explanation and gates)
        human_text = response.get("human_summary", "")
        follow_up = response.get("follow_up", "")
        presigned_url= ""
        #if gates["all_ok"] == 'true':
        # Create PDF
        pdf_bytes = create_pdf_bytes_plain_from_html(human_text, title="Your Complete Trip Guide") #create_pdf_bytes(human_text, title="Final Itinerary (Human-readable)")

        # Upload PDF to S3 under final_outputs/
        pdf_key = f"{fileName.rsplit('.', 1)[0]}.pdf"
        presigned_url = upload_pdf_to_s3(bucket_name, pdf_key, pdf_bytes)
        print(f"Uploaded PDF to s3://{bucket_name}/{pdf_key}")
        try:
            return {
                "itinerary_summary": human_text,
                "follow_up": follow_up,
                "s3_pdf_key": pdf_key,
                "s3_pdf_presigned_url": presigned_url,
                "gates": gates,
                "explanation": explanation
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
    print(f"Invoking Lambda function {function_name} with payload: {payload}")
    """
    Invokes an AWS Lambda function synchronously.

    :param function_name: Name of the Lambda function to invoke
    :param payload: Payload dictionary to pass to the Lambda function
    :return: Response from the Lambda function as a dictionary
    """
    cfg = Config(connect_timeout=60, read_timeout=900)

    client = boto3.client('lambda', config=cfg)

    try:
        # Call the Lambda function
        response = client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        # Read and parse the response payload
        response_payload = json.loads(response['Payload'].read().decode('utf-8'))
        print(f"Lambda response payload: {response_payload}")
        return response_payload
    except Exception as e:
        logging.error(f"Lambda invocation failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import json
    with open("../inputs/20251107T200155_02bb2fc0.json", "r") as f:
        payload = json.load(f)
    planner_agent = PlannerAgent()
    response = planner_agent.run(payload, S3_BUCKET, "../inputs/20251107T200155_02bb2fc0.json", "02bb2fc0")
    print(f"Planner Agent response: {response}")

    with open("../inputs/20251107T200155_02bb2fc0.json", "r") as f:
        payload = json.load(f)
    transport_options = payload.get("transport_options", {})
    requirements = payload.get("requirements", {})
    itinerary = payload.get("itinerary", {})
    metrics = payload.get("metrics", {})
    explanation = payload.get("explanation", {})
    gates = payload.get("gates", {})
    final_agent = CrewAIAdapter()
    response = final_agent.run(itinerary, transport_options, metrics, gates, requirements, explanation)
    print(f"Final Agent response: {response}")

    # Build human-readable text (includes explanation and gates)
    human_text = response.get("human_summary", "")
    follow_up = response.get("follow_up", "")
    payload["human_summary"] = human_text
    payload["follow_up"] = follow_up
    update_json_data("", "../inputs/20251107T200155_02bb2fc0.json", payload)
    presigned_url = ""
    # if gates["all_ok"] == 'true':
    # Create PDF
    pdf_bytes = create_pdf_bytes_plain_from_html(human_text,
                                                 title="Your Complete Trip Guide")  # create_pdf_bytes(human_text, title="Final Itinerary (Human-readable)")

    # Save PDF bytes to a local file
    with open("trip_guide.pdf", "wb") as pdf_file:
        pdf_file.write(pdf_bytes)

    #update_json_data(bucket_name, Summarizer_Agent_Folder + "/" + fileName, payload)