from typing import Literal

import boto3
from mypy_boto3_s3 import S3Client
from pydantic import BaseModel
from settings import settings


class MinioStorageConfig(BaseModel):
    strategy: Literal["minio"] = "minio"
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str


def get_minio_client(config: MinioStorageConfig | None = None) -> S3Client:
    effective_config = (
        config
        if config is not None
        else MinioStorageConfig(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket_name=settings.MINIO_BUCKET,
        )
    )

    client = boto3.client(
        "s3",
        endpoint_url=f"http://{effective_config.endpoint}",
        aws_access_key_id=effective_config.access_key,
        aws_secret_access_key=effective_config.secret_key,
    )

    try:
        client.head_bucket(Bucket=effective_config.bucket_name)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=effective_config.bucket_name)

    return client
