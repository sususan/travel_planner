# lambda/handler.py
from __future__ import annotations

import base64
import json, os, logging
import traceback
from botocore.exceptions import ClientError
from planner_agent.orchestrator.orchestrator import plan_itinerary
from planner_agent.tools.config import Transport_Agent_Folder, Summarizer_Agent_Folder
from planner_agent.tools.s3io import get_json_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info("!! lambda_handler !!")
    session = ""
    statusCode = 200
    response = ""
    try:
        parsed_json = None
        if not isinstance(event, dict):
            json_input = event.get("body", "").strip("")
            parsed_json = json.loads(json_input)
        else:
            parsed_json = event

        if parsed_json is not None:
            bucket_name = parsed_json.get("bucket_name")
            key = parsed_json.get("key")
            sender_agent = parsed_json.get("sender_agent")
            session = parsed_json.get("session")
            if bucket_name and key:
                logger.info(f"Fetching JSON file from S3 bucket '{bucket_name}' with key '{key}'")
                #payload = get_json_data(key)
                fileName = key.split('/')[-1]
                # Call orchestrator to plan itinerary
                ret = plan_itinerary(bucket_name,key, session)
                logger.info(f"Plan itinerary returned: {ret}")
                statusCode = 200
                response = {
                    "statusCode": 200,
                    "message": "Itinerary planned successfully.",
                    "session": session,
                    'input_location': key,
                    'output_location': Summarizer_Agent_Folder + '/' + fileName,
                    'summary': ret
                }
            else:
                logger.error("Missing 'bucket_name' or 'key' in event")
                statusCode = 500
                response =  "Missing 'bucket_name' or 'key' in event"

    except Exception as e:
        logger.exception(f"Error: {e}")
        traceback.print_exc()
        statusCode = 500
        response = f"Planner Agent Failed with: {e}"

    return {
        "statusCode": statusCode,
        "body": json.dumps(response)
    }
# API Gateway REST/HTTP event compatible
