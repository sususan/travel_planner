import json
import logging
import traceback
from datetime import time

from planner_agent.agent.final_agent import FinalAgent
from planner_agent.orchestrator.orchestrator import plan_itinerary, FINAL_CREW_ADAPTER
from planner_agent.planner_core.core import explain
from planner_agent.tools.s3io import get_json_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info("!! lambda_handler !!")
    try:
        t0 = time.time()
        parsed_json = None
        json_input = event.get("body", "").strip("")
        logger.info(f"JSON input: {json_input}")
        if json_input:
            parsed_json = json.loads(json_input)
            bucket_name = parsed_json.get("bucket_name")
            key = parsed_json.get("key")
            sender_agent = parsed_json.get("sender_agent")
            session = parsed_json.get("session")
            if bucket_name and key:
                logger.info(f"Fetching JSON file from S3 bucket '{bucket_name}' with key '{key}'")
                payload = get_json_data(key)
                fileName = key.split('/')[-1]
                # Call orchestrator to plan itinerary
                # Stage 4: Final formatting using final agent
                final_agent = FinalAgent(crew_adapter=FINAL_CREW_ADAPTER)
                requirements = payload.get("requirements", {})
                itinerary = payload.get("itinerary", {})
                metrics = payload.get("metrics", {})
                explanation = explain(requirements, itinerary, metrics)
                gates = payload.get("gates", {})
                final_payload = final_agent.run(itinerary, metrics, explanation, gates, payload)

                logger.info(f"Fetched JSON content: {json.dumps(parsed_json, indent=2)}")
            else:
                logger.error("Missing 'bucket_name' or 'key' in event")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON input: {str(e)}")
    except Exception as e:
        logger.exception(f"Error: {e}")
        traceback.print_exc()

    return {
        #"scored": scored,
        #"shortlist": sl,
        "itinerary": itinerary,
        "metrics": metrics,
        "explanation": explanation,
        "gates": gates,
        #"planner_iterations": iterations,
        "final_output": final_payload,
        "elapsed_ms": int((time.time() - t0) * 1000)
    }