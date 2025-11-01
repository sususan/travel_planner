import os, json, datetime
import boto3

from planner_agent.tools.config import S3_BUCKET

_s3 = boto3.client("s3")

def put_json(key: str, obj: dict) -> str:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env not set")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_key = f"{ts}-{key}.json"
    _s3.put_object(Bucket=S3_BUCKET, Key=full_key, Body=json.dumps(obj).encode("utf-8"), ContentType="application/json")
    return full_key

def get_json(key: str) -> dict:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env not set")
    full_key = f"{key}"
    obj = _s3.get_object(Bucket=S3_BUCKET, Key=full_key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def update_json(key: str, payload: dict):
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env not set")

    s3 = boto3.client('s3')
    bucket_name = S3_BUCKET  # Replace with your bucket name
    file_key = f"{key}"  # Replace with the file path in the bucket
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=json.dumps(payload))