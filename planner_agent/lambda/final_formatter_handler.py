# /mnt/data/final_formatter_handler.py
import json
import logging
import traceback
import base64
import time
from planner_agent.agent.final_agent import CrewAIAdapterForFinal
from planner_agent.orchestrator.orchestrator import FINAL_CREW_ADAPTER
from planner_agent.planner_core.core import explain
from planner_agent.tools.final_agent_helper import create_pdf_bytes
from planner_agent.tools.s3io import get_json_data, upload_pdf_to_s3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ------------ Helper functions ----------------

# ------------ Lambda handler ----------------

def lambda_handler(event, context):
    logger.info("!! lambda_handler !!")
    statusCode = 200
    response_payload = None
    try:
        t0 = time.time()
        parsed_json = None
        json_input = event.get("body", "").strip("") if isinstance(event.get("body", ""), str) else ""
        logger.info(f"JSON input: {json_input}")
        if json_input:
            parsed_json = json.loads(json_input)
            bucket_name = parsed_json.get("bucket_name")
            key = parsed_json.get("key")
            sender_agent = parsed_json.get("sender_agent")
            session = parsed_json.get("session")
            if bucket_name and key:
                logger.info(f"Fetching JSON file from S3 bucket '{bucket_name}' with key '{key}'")
                payload = get_json_data(bucket_name, key)
                fileName = key.split('/')[-1]
                # Stage 4: Final formatting using final agent
                final_agent = CrewAIAdapterForFinal(crew_adapter=FINAL_CREW_ADAPTER)
                requirements = payload.get("requirements", {})
                itinerary = payload.get("itinerary", {})
                metrics = payload.get("metrics", {})
                explanation = explain(requirements, itinerary, metrics)
                gates = payload.get("gates", {})

                # Ask FinalAgent to run (keeps existing behavior)
                response = final_agent.run(requirements, itinerary, metrics, explanation, gates)
                logger.info("Final agent returned payload (kept in logs for debugging)")

                # Build human-readable text (includes explanation and gates)
                human_text = response.get("human_summary", "")
                # Create PDF
                pdf_bytes = create_pdf_bytes(human_text, title="Final Itinerary (Human-readable)")

                # Upload PDF to S3 under final_outputs/
                pdf_key = f"final_outputs/{fileName.rsplit('.',1)[0]}.pdf"
                try:
                    presigned_url = upload_pdf_to_s3(bucket_name, pdf_key, pdf_bytes)
                    logger.info(f"Uploaded PDF to s3://{bucket_name}/{pdf_key}")
                    response_payload = {
                        "statusCode": 200,
                        "message": human_text,
                        "s3_pdf_key": pdf_key,
                        "s3_pdf_presigned_url": presigned_url,
                        "session": session,
                        "summary": {
                            "gates": gates,
                            "explanation":explanation,
                            "metrics": metrics
                        }
                    }
                except Exception as e:
                    # fallback: return base64 PDF in response body
                    logger.exception(f"Error: {e}")
                    traceback.print_exc()
                    statusCode = 500
                    response = f"Summarizer Agent Failed with: {e}"
            else:
                logger.error("Missing 'bucket_name' or 'key' in event")
                statusCode = 500
                response_payload = {"error": "Missing 'bucket_name' or 'key' in event"}

        else:
            logger.error("Empty request body")
            statusCode = 400
            response_payload = {"error": "Empty request body"}

    except Exception as e:
        logger.exception(f"Error: {e}")
        traceback.print_exc()
        statusCode = 500
        response_payload = {"error": f"PlSummarizeranner Agent Failed with: {e}"}

    # Return JSON body so callers can inspect presigned URL or base64 PDF
    return {
        "statusCode": statusCode,
        "body": json.dumps(response_payload)
    }
if __name__ == "__main__":
    with open("../outputs_panner/20251031T003447_f312ea72.json", "r") as f:
        payload = json.load(f)

    final_agent = CrewAIAdapterForFinal(crew_adapter=FINAL_CREW_ADAPTER)
    requirements = payload.get("requirements", {})
    itinerary = payload.get("itinerary", {})
    metrics = payload.get("metrics", {})
    explanation = explain(requirements, itinerary, metrics)
    gates = payload.get("gates", {})

    # Ask FinalAgent to run (keeps existing behavior)
    response = final_agent.run(requirements, itinerary, metrics, explanation, gates)
    print(response)