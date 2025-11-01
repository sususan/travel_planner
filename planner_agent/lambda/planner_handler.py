# lambda/handler.py
from __future__ import annotations

import base64
import json, os, logging
import traceback

from botocore.exceptions import ClientError

from planner_agent.orchestrator.orchestrator import plan_itinerary
from planner_agent.tools.config import Transport_Agent_Folder
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
                ret = plan_itinerary(payload, fileName)
                logger.info(f"Plan itinerary returned: {ret}")
                statusCode = 200
                response = {
                    "statusCode": 200,
                    "message": "Itinerary planned successfully.",
                    "session": session,
                    'input_location': key,
                    'output_location': Transport_Agent_Folder + '/' + fileName,
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
        "body": response
    }
# API Gateway REST/HTTP event compatible

def s3_event_handler(event, context):
    """
    Handle S3 event triggers.
    """
    try:
        logger.info("!! s3_event_handler !!")
        logger.info(f"Received S3 event: {json.dumps(event)}")

        # Validate and extract bucket name and object key
        records = event.get("Records", [])
        if not records or "s3" not in records[0]:
            return {"statusCode": 400, "body": json.dumps({"status": "error", "message": "Invalid S3 event structure"})}

        s3_info = records[0].get("s3", {})
        bucket_name = s3_info.get("bucket", {}).get("name")
        object_key = s3_info.get("object", {}).get("key")

        if not bucket_name or not object_key:
            return {"statusCode": 400,
                    "body": json.dumps({"status": "error", "message": "Missing bucket name or object key"})}

        # Perform necessary processing (e.g., logging, moving files, updating metadata)
        logger.info(f"Processing file from bucket '{bucket_name}' with key '{object_key}'")
        #payload = json.loads(s3_info.get("object", {}).get("body"))
        payload = get_json_data(object_key)
        result = plan_itinerary(payload, object_key)
        logger.info("S3 event processed successfully")
        return {"statusCode": 200,
                "body": json.dumps({"status": "success", "message": "S3 event processed successfully"})}

    except ClientError as e:
        logger.error(f"AWS S3 Client Error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"status": "error", "message": "AWS S3 error occurred"})}

    except Exception as e:
        logger.exception(f"Error handling S3 event: {e}")
        return {"statusCode": 500, "body": json.dumps({"status": "error", "message": "Unexpected error occurred"})}

def count_by_type(activities):
    """Count activities by type for summary"""
    type_count = {}
    for activity in activities:
        activity_type = activity.get('type', 'unknown')
        type_count[activity_type] = type_count.get(activity_type, 0) + 1
    return type_count


def min_cost(activities):
    """Get minimum cost from activities (handle nulls)"""
    costs = [act.get('cost_sgd', 0) for act in activities if act.get('cost_sgd') is not None]
    return min(costs) if costs else 0


def max_cost(activities):
    """Get maximum cost from activities (handle nulls)"""
    costs = [act.get('cost_sgd', 0) for act in activities if act.get('cost_sgd') is not None]
    return max(costs) if costs else 0


def summarize_interests(activities, required_interests=None):
    """Summarize which user interests are covered"""
    if required_interests is None:
        required_interests = ['family', 'parks', 'museums', 'educational']

    coverage = {}
    for interest in required_interests:
        # Simple heuristic - you can enhance this based on your scoring logic
        count = 0
        for activity in activities:
            if (interest in activity.get('tags', []) or
                    interest in activity.get('name', '').lower() or
                    interest in activity.get('description', '').lower()):
                count += 1
        coverage[interest] = count

    return coverage
