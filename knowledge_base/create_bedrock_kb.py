"""
Creates a Bedrock Knowledge Base backed by S3 + OpenSearch Serverless + Nova Embeddings.

Usage:
    python knowledge_base/create_bedrock_kb.py

Steps performed:
    1. Creates S3 bucket and uploads policy documents
    2. Creates IAM role for Bedrock KB
    3. Creates OpenSearch Serverless collection (vector store)
    4. Creates Bedrock Knowledge Base with Nova Multimodal Embeddings
    5. Creates S3 data source and starts ingestion
    6. Saves KB ID to kb_config.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
ACCOUNT_ID = None  # resolved at runtime
BUCKET_NAME = "priorauth-agent-kb-617903186897"
KB_NAME = "priorauth-payer-policies"
KB_DESCRIPTION = "Payer-specific prior authorization policy criteria for medical necessity evaluation."
EMBEDDING_MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/amazon.nova-2-multimodal-embeddings-v1:0"
ROLE_NAME = "PriorAuthKBBedrockRole"
COLLECTION_NAME = "priorauth-policies"
INDEX_NAME = "bedrock-knowledge-base-default-index"
POLICY_DOCS_DIR = Path(__file__).resolve().parent / "policy_docs"
CONFIG_PATH = Path(__file__).resolve().parent / "kb_config.json"


def get_account_id() -> str:
    global ACCOUNT_ID
    sts = boto3.client("sts", region_name=REGION)
    ACCOUNT_ID = sts.get_caller_identity()["Account"]
    return ACCOUNT_ID


# ── S3 ──────────────────────────────────────────────────────────────

def create_s3_bucket(s3) -> None:
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"[S3] Bucket exists: {BUCKET_NAME}")
    except ClientError:
        print(f"[S3] Creating bucket: {BUCKET_NAME}")
        params = {"Bucket": BUCKET_NAME}
        if REGION != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
        s3.create_bucket(**params)
        print(f"[S3] Created: {BUCKET_NAME}")


def upload_policy_docs(s3) -> int:
    count = 0
    for doc_path in sorted(POLICY_DOCS_DIR.glob("*.txt")):
        key = f"policies/{doc_path.name}"
        s3.upload_file(str(doc_path), BUCKET_NAME, key)
        print(f"  Uploaded: s3://{BUCKET_NAME}/{key}")
        count += 1
    return count


# ── IAM ─────────────────────────────────────────────────────────────

def create_iam_role(iam) -> str:
    role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": ACCOUNT_ID},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:knowledge-base/*"
                    },
                },
            }
        ],
    }

    try:
        iam.get_role(RoleName=ROLE_NAME)
        print(f"[IAM] Role exists: {ROLE_NAME}")
    except ClientError:
        print(f"[IAM] Creating role: {ROLE_NAME}")
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Bedrock KB role for PriorAuth agent.",
        )

    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{BUCKET_NAME}",
                    f"arn:aws:s3:::{BUCKET_NAME}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": [EMBEDDING_MODEL_ARN],
            },
            {
                "Effect": "Allow",
                "Action": ["aoss:APIAccessAll"],
                "Resource": [
                    f"arn:aws:aoss:{REGION}:{ACCOUNT_ID}:collection/*"
                ],
            },
        ],
    }

    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="PriorAuthKBAccess",
        PolicyDocument=json.dumps(inline_policy),
    )
    print(f"[IAM] Role ready: {role_arn}")
    print("[IAM] Waiting for propagation...")
    time.sleep(10)
    return role_arn


# ── OpenSearch Serverless ───────────────────────────────────────────

def create_opensearch_collection(aoss) -> str:
    # Check if collection already exists
    collections = aoss.list_collections(
        collectionFilters={"name": COLLECTION_NAME}
    ).get("collectionSummaries", [])

    if collections:
        collection = collections[0]
        collection_id = collection["id"]
        print(f"[AOSS] Collection exists: {collection_id}")
    else:
        # Create encryption policy first (required)
        enc_policy_name = f"{COLLECTION_NAME}-enc"
        try:
            aoss.create_security_policy(
                name=enc_policy_name,
                type="encryption",
                policy=json.dumps({
                    "Rules": [
                        {
                            "Resource": [f"collection/{COLLECTION_NAME}"],
                            "ResourceType": "collection",
                        }
                    ],
                    "AWSOwnedKey": True,
                }),
            )
            print(f"[AOSS] Created encryption policy: {enc_policy_name}")
        except ClientError as e:
            if "ConflictException" in str(type(e).__name__) or "already exists" in str(e).lower() or "409" in str(e):
                print(f"[AOSS] Encryption policy exists: {enc_policy_name}")
            else:
                raise

        # Network policy (allow public access for demo)
        net_policy_name = f"{COLLECTION_NAME}-net"
        try:
            aoss.create_security_policy(
                name=net_policy_name,
                type="network",
                policy=json.dumps([
                    {
                        "Rules": [
                            {
                                "Resource": [f"collection/{COLLECTION_NAME}"],
                                "ResourceType": "collection",
                            }
                        ],
                        "AllowFromPublic": True,
                    }
                ]),
            )
            print(f"[AOSS] Created network policy: {net_policy_name}")
        except ClientError as e:
            if "ConflictException" in str(type(e).__name__) or "already exists" in str(e).lower() or "409" in str(e):
                print(f"[AOSS] Network policy exists: {net_policy_name}")
            else:
                raise

        # Create the collection
        print(f"[AOSS] Creating collection: {COLLECTION_NAME}")
        response = aoss.create_collection(
            name=COLLECTION_NAME,
            type="VECTORSEARCH",
            description="Vector store for PriorAuth payer policies.",
        )
        collection_id = response["createCollectionDetail"]["id"]
        print(f"[AOSS] Created collection: {collection_id}")

    # Wait for ACTIVE
    for _ in range(60):
        batch = aoss.batch_get_collection(ids=[collection_id])
        details = batch.get("collectionDetails", [{}])[0]
        status = details.get("status", "CREATING")
        if status == "ACTIVE":
            collection_arn = details["arn"]
            print(f"[AOSS] Collection ACTIVE: {collection_arn}")

            # Data access policy (allows Bedrock role + current user)
            access_policy_name = f"{COLLECTION_NAME}-access"
            try:
                aoss.create_access_policy(
                    name=access_policy_name,
                    type="data",
                    policy=json.dumps([
                        {
                            "Rules": [
                                {
                                    "Resource": [f"collection/{COLLECTION_NAME}"],
                                    "Permission": [
                                        "aoss:CreateCollectionItems",
                                        "aoss:DeleteCollectionItems",
                                        "aoss:UpdateCollectionItems",
                                        "aoss:DescribeCollectionItems",
                                    ],
                                    "ResourceType": "collection",
                                },
                                {
                                    "Resource": [f"index/{COLLECTION_NAME}/*"],
                                    "Permission": [
                                        "aoss:CreateIndex",
                                        "aoss:DeleteIndex",
                                        "aoss:UpdateIndex",
                                        "aoss:DescribeIndex",
                                        "aoss:ReadDocument",
                                        "aoss:WriteDocument",
                                    ],
                                    "ResourceType": "index",
                                },
                            ],
                            "Principal": [
                                f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}",
                                f"arn:aws:iam::{ACCOUNT_ID}:user/nova-hackathon-ayush",
                            ],
                        }
                    ]),
                )
                print(f"[AOSS] Created access policy: {access_policy_name}")
            except ClientError as e:
                if "already exists" in str(e).lower() or "409" in str(e):
                    print(f"[AOSS] Access policy exists: {access_policy_name}")
                else:
                    raise

            return collection_arn
        if status == "FAILED":
            print(f"[AOSS] Collection FAILED")
            sys.exit(1)
        print(f"  Collection status: {status}, waiting...")
        time.sleep(10)

    print("[AOSS] Timed out waiting for collection")
    sys.exit(1)


# ── Bedrock KB ──────────────────────────────────────────────────────

def create_knowledge_base(bedrock_agent, role_arn: str, collection_arn: str) -> str:
    existing = bedrock_agent.list_knowledge_bases(maxResults=100)
    for kb in existing.get("knowledgeBaseSummaries", []):
        if kb["name"] == KB_NAME:
            kb_id = kb["knowledgeBaseId"]
            print(f"[KB] Already exists: {kb_id}")
            return kb_id

    print(f"[KB] Creating: {KB_NAME}")
    response = bedrock_agent.create_knowledge_base(
        name=KB_NAME,
        description=KB_DESCRIPTION,
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": EMBEDDING_MODEL_ARN,
                "embeddingModelConfiguration": {
                    "bedrockEmbeddingModelConfiguration": {"dimensions": 1024}
                },
            },
        },
        storageConfiguration={
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": {
                "collectionArn": collection_arn,
                "fieldMapping": {
                    "metadataField": "AMAZON_BEDROCK_METADATA",
                    "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                    "vectorField": "bedrock-knowledge-base-default-vector",
                },
                "vectorIndexName": INDEX_NAME,
            },
        },
    )

    kb_id = response["knowledgeBase"]["knowledgeBaseId"]
    print(f"[KB] Created: {kb_id}")

    for _ in range(30):
        status = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
        state = status["knowledgeBase"]["status"]
        if state == "ACTIVE":
            print(f"[KB] ACTIVE: {kb_id}")
            return kb_id
        if state in ("FAILED", "DELETE_IN_PROGRESS"):
            reasons = status["knowledgeBase"].get("failureReasons", ["unknown"])
            print(f"[KB] FAILED: {reasons}")
            sys.exit(1)
        print(f"  KB status: {state}, waiting...")
        time.sleep(5)

    print("[KB] Timed out")
    sys.exit(1)


def create_data_source_and_ingest(bedrock_agent, kb_id: str) -> None:
    existing = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id, maxResults=10)
    ds_id = None
    for ds in existing.get("dataSourceSummaries", []):
        if ds["name"] == "payer-policy-docs":
            ds_id = ds["dataSourceId"]
            print(f"[DS] Already exists: {ds_id}")
            break

    if ds_id is None:
        print("[DS] Creating S3 data source...")
        response = bedrock_agent.create_data_source(
            knowledgeBaseId=kb_id,
            name="payer-policy-docs",
            description="Payer PA policy documents from S3.",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{BUCKET_NAME}",
                    "inclusionPrefixes": ["policies/"],
                },
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": {
                    "chunkingStrategy": "FIXED_SIZE",
                    "fixedSizeChunkingConfiguration": {
                        "maxTokens": 300,
                        "overlapPercentage": 20,
                    },
                }
            },
        )
        ds_id = response["dataSource"]["dataSourceId"]
        print(f"[DS] Created: {ds_id}")

    # Start ingestion
    print("[Ingest] Starting ingestion job...")
    response = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id, dataSourceId=ds_id
    )
    job_id = response["ingestionJob"]["ingestionJobId"]

    for _ in range(60):
        status = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )
        state = status["ingestionJob"]["status"]
        if state == "COMPLETE":
            stats = status["ingestionJob"].get("statistics", {})
            print(f"[Ingest] COMPLETE: {stats}")
            return
        if state == "FAILED":
            reasons = status["ingestionJob"].get("failureReasons", ["unknown"])
            print(f"[Ingest] FAILED: {reasons}")
            sys.exit(1)
        print(f"  Ingestion: {state}, waiting...")
        time.sleep(5)

    print("[Ingest] Timed out")
    sys.exit(1)


def save_config(kb_id: str) -> None:
    config = {"knowledge_base_id": kb_id, "region": REGION}
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nSaved config to: {CONFIG_PATH}")
    print(f"Set env var:  export BEDROCK_KB_ID={kb_id}")


def main() -> None:
    account_id = get_account_id()
    print(f"Account: {account_id} | Region: {REGION}\n")

    s3 = boto3.client("s3", region_name=REGION)
    iam = boto3.client("iam", region_name=REGION)
    aoss = boto3.client("opensearchserverless", region_name=REGION)
    bedrock_agent = boto3.client("bedrock-agent", region_name=REGION)

    # Step 1: S3
    create_s3_bucket(s3)
    count = upload_policy_docs(s3)
    print(f"Uploaded {count} policy documents\n")

    # Step 2: IAM
    role_arn = create_iam_role(iam)

    # Step 3: OpenSearch Serverless
    collection_arn = create_opensearch_collection(aoss)

    # Step 4: Bedrock KB
    kb_id = create_knowledge_base(bedrock_agent, role_arn, collection_arn)

    # Step 5: Data source + ingestion
    create_data_source_and_ingest(bedrock_agent, kb_id)

    # Step 6: Save config
    save_config(kb_id)
    print(f"\nDone! Knowledge Base ID: {kb_id}")


if __name__ == "__main__":
    main()
