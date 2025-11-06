import os, json, datetime
import boto3


_s3 = boto3.client("s3")

def put_json(bucket_name: str,key: str, obj: dict) -> str:
    if not bucket_name:
        raise RuntimeError("S3_BUCKET env not set")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_key = f"{ts}-{key}.json"
    _s3.put_object(Bucket=bucket_name, Key=full_key, Body=json.dumps(obj).encode("utf-8"), ContentType="application/json")
    return full_key

def get_json_data (bucket_name: str, key: str) -> dict:
    if not bucket_name:
        raise RuntimeError("S3_BUCKET env not set")
    full_key = f"{key}"
    obj = _s3.get_object(Bucket=bucket_name, Key=full_key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def update_json_data(bucket_name: str, key: str, payload: dict):
    if not bucket_name:
        raise RuntimeError("S3_BUCKET env not set")

    s3 = boto3.client('s3')
    bucket_name = bucket_name  # Replace with your bucket name
    file_key = f"{key}"  # Replace with the file path in the bucket
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=json.dumps(payload))


def upload_pdf_to_s3(bucket_name, key, pdf_bytes):
    """
    Upload bytes to S3 and return a presigned URL (1 hour).
    """
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket_name, Key="summarizer_agent/pdf/"+key, Body=pdf_bytes, ContentType="application/pdf")
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": key},
        ExpiresIn=3600
    )
    return url