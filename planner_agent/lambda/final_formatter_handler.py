import logging

from planner_agent.orchestrator.orchestrator import plan_itinerary
from planner_agent.tools.s3io import get_json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info("!! lambda_handler !!")
    name = event.get("name", "world")
    return {
        "statusCode": 200,
        "body": f"Hello, {name}!"
    }