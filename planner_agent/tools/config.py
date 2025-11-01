import os
from xmlrpc.client import Transport

TRANSPORT_ENABLED = True
USE_CREW = False
child_multiplier: float = 0.5
senior_multiplier: float = 0.8
MAX_AGENT_ITERATIONS = 0 # number of times to let planner agent try to repair
S3_BUCKET = os.getenv("S3_BUCKET", "iss-travel-planner")
Transport_Agent_Folder = "transport_agent"