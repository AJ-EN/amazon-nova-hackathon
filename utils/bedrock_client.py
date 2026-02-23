import boto3
import os
from botocore.config import Config


def get_bedrock_client():
    """
    Returns a configured Bedrock runtime client for model invocation.
    Credentials are resolved automatically from ~/.aws/credentials
    via the boto3 credential chain - no .env file or hardcoded keys needed.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        config=Config(
            read_timeout=3600,   # long timeout needed for Nova Act and extended thinking
            connect_timeout=30,
            retries={"max_attempts": 3},
        ),
    )


def get_bedrock_agent_client():
    """
    Returns a Bedrock agent client for Knowledge Base operations.
    This is a different boto3 service from bedrock-runtime - runtime is for
    invoking models, agent is for managing and querying Knowledge Bases.
    """
    return boto3.client(
        "bedrock-agent",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )
