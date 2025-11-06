import os

TRANSPORT_ENABLED = True
USE_CREW = False
child_multiplier: float = 0.5
senior_multiplier: float = 0.8
MAX_AGENT_ITERATIONS = 1 # number of times to let planner agent try to repair
S3_BUCKET = os.getenv("S3_BUCKET", "iss-travel-planner")
Transport_Agent_Folder = "transport_agent"
Summarizer_Agent_Folder = "summarizer_agent"
TRANSPORT_ADAPTERAPI_ENDPOINT = "https://zjjh6gp7x9.execute-api.ap-southeast-1.amazonaws.com/prod"
Final_ADAPTERAPI_ENDPOINT = "https://s0k25s7kqk.execute-api.ap-southeast-1.amazonaws.com/prod"
X_API_Key = "ELyVlpoxgyHVV7MgqH8waT2Byab36oY7rxjZ5CSd" #os.getenv("X-API-Key")
TransportAgentARN = 'arn:aws:lambda:ap-southeast-1:641675857341:function:transport-queue-handler-prod'

LLM_MODEL = os.getenv("CREW_LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

