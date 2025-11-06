import requests

from planner_agent.tools.config import TRANSPORT_ADAPTERAPI_ENDPOINT, X_API_Key, Transport_Agent_Folder, S3_BUCKET


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
        return {}


if __name__ == "__main__":
    response = call_transport_agent_api(S3_BUCKET, "attraction-output-v2.json", "Planner Agent", "123")
    response_data = response.json() if response else {}
    print(response.status_code)
    print(response_data)
    retrieval_summary = response_data.get("transport", {})