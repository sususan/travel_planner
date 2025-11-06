import os


TRANSPORT_ENABLED = True
USE_CREW = False
child_multiplier: float = 0.5
senior_multiplier: float = 0.8
MAX_AGENT_ITERATIONS = 1 # number of times to let planner agent try to repair
S3_BUCKET = os.getenv("S3_BUCKET", "iss-travel-planner")
Transport_Agent_Folder = "transport_agent/active"
Summarizer_Agent_Folder = "summarizer_agent"
TRANSPORT_ADAPTERAPI_ENDPOINT = "https://29jxwf52gb.execute-api.ap-southeast-1.amazonaws.com/prod"
Final_ADAPTERAPI_ENDPOINT = "https://s0k25s7kqk.execute-api.ap-southeast-1.amazonaws.com/prod"
X_API_Key = "pYSqYcivG1504xjeQAskn2iyVS7fZ2Uj14lK1w8v" #os.getenv("X-API-Key")
TransportAgentARN = 'arn:aws:lambda:ap-southeast-1:641675857341:function:transport-queue-handler-prod'
LLM_MODEL = os.getenv("CREW_LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")