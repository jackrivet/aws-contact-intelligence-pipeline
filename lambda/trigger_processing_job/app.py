import os
import uuid
import datetime as dt
import logging
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sagemaker = boto3.client("sagemaker")

@dataclass

class ProcessingJobConfig:
    role_arn: str
    image_uri: str
    entrypoint_script: str
    model_path: str
    tokenizer_path: str
    output_prefix: str
    athena_database: str
    athena_output: str
    s3_bucket: str
    s3_prefix: str
    region: str = os.environ.get("AWS_REGION", "us-east-1")
    script_s3_uri: str = ""
    instance_type: str = "ml.m5.4xlarge"
    instance_count: int = 1
    volume_gb: int = 64
    max_runtime_seconds: int = 14_400 
def load_config_from_env() -> ProcessingJobConfig:
    """Load job configuration from Lambda environment variables.
    Env vars should be defined in the Lambda configuration or IaC.
    """
    required = [
        "ROLE_ARN",
        "IMAGE_URI",
        "ENTRYPOINT_SCRIPT",
        "MODEL_PATH",
        "TOKENIZER_PATH",
        "OUTPUT_PREFIX",
        "ATHENA_DATABASE",
        "ATHENA_OUTPUT",
        "S3_BUCKET",
        "S3_PREFIX",
        "SCRIPT_S3_URI",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return ProcessingJobConfig(
        role_arn=os.environ["ROLE_ARN"],
        image_uri=os.environ["IMAGE_URI"],
        entrypoint_script=os.environ["ENTRYPOINT_SCRIPT"],
        model_path=os.environ["MODEL_PATH"],
        tokenizer_path=os.environ["TOKENIZER_PATH"],
        output_prefix=os.environ["OUTPUT_PREFIX"],
        athena_database=os.environ["ATHENA_DATABASE"],
        athena_output=os.environ["ATHENA_OUTPUT"],
        s3_bucket=os.environ["S3_BUCKET"],
        s3_prefix=os.environ["S3_PREFIX"],
        script_s3_uri=os.environ["SCRIPT_S3_URI"],
        region=os.environ.get("AWS_REGION", "us-east-1"),
    )
def build_job_name(prefix: str = "call-intent-batch") -> str:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}-{timestamp}-{suffix}"
def create_processing_job(cfg: ProcessingJobConfig, job_name: str) -> str:
    """Create the SageMaker Processing job and return its name."""
    logger.info("Creating SageMaker Processing job %s", job_name)
    try:
        sagemaker.create_processing_job(
            ProcessingJobName=job_name,
            RoleArn=cfg.role_arn,
            AppSpecification={
                "ImageUri": cfg.image_uri,
                "ContainerEntrypoint": [
                    "python3",
                    cfg.entrypoint_script,
                ],
                "ContainerArguments": [
                    "--model_path",
                    cfg.model_path,
                    "--tokenizer_path",
                    cfg.tokenizer_path,
                    "--output_prefix",
                    cfg.output_prefix,
                    "--athena_database",
                    cfg.athena_database,
                    "--athena_output",
                    cfg.athena_output,
                    "--s3_bucket",
                    cfg.s3_bucket,
                    "--s3_prefix",
                    cfg.s3_prefix,
                    "--region",
                    cfg.region,
                ],
            },
            ProcessingInputs=[
                {
                    "InputName": "script-input",
                    "S3Input": {
                        "S3Uri": cfg.script_s3_uri,
                        "LocalPath": "/opt/ml/processing/input/",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            ],
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": cfg.instance_count,
                    "InstanceType": cfg.instance_type,
                    "VolumeSizeInGB": cfg.volume_gb,
                }
            },
            StoppingCondition={
                "MaxRuntimeInSeconds": cfg.max_runtime_seconds,
            },
        )
    except ClientError as e:
        logger.exception("Failed to create processing job %s", job_name)
        raise
    return job_name
def lambda_handler(event, context):
    cfg = load_config_from_env()
    job_name = build_job_name(prefix="call-intent-deberta-batch")
    job_name = create_processing_job(cfg, job_name)
    logger.info("Started processing job %s", job_name)
    return {
        "statusCode": 200,
        "body": f"Started processing job {job_name}",
    }
