import boto3
import os
from dotenv import load_dotenv
from botocore.config import Config

# Load environment variables from .env file
load_dotenv()


def get_bedrock_client():
    """
    Returns a configured Bedrock runtime client.
    Every agent in this project imports and uses this function -
    it's the single source of truth for how we connect to AWS.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        config=Config(
            read_timeout=3600,    # 60 minutes; Nova Act tasks can be long.
            connect_timeout=30,
            retries={"max_attempts": 3}
        ),
    )
