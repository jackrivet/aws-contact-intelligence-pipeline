This function is responsible for **orchestrating batch inference** by launching an Amazon SageMaker Processing job.
It performs the following steps:
Generates a unique job name using timestamp + UUID.
Calls `sagemaker.create_processing_job()` with the configured container image.
Passes model paths, tokenizer paths, output prefixes, and Athena parameters to the job.
Returns the job name as the execution result.
### Environment Variables
The Lambda is configured using environment variables, typically provided by IaC (AWS CDK, Terraform, or SAM):
| Variable            | Description |
|---------------------|-------------|
| `ROLE_ARN`          | IAM role used by the Processing Job. |
| `IMAGE_URI`         | ECR URI of the SageMaker Processing container. |
| `ENTRYPOINT_SCRIPT` | Path inside the container to the batch inference script. |
| `MODEL_PATH`        | S3 path to model artifacts. |
| `TOKENIZER_PATH`    | S3 path to tokenizer. |
| `OUTPUT_PREFIX`     | S3 prefix for prediction output. |
| `ATHENA_DATABASE`   | Athena database to query. |
| `ATHENA_OUTPUT`     | Location for Athena query results. |
| `S3_BUCKET`         | Bucket containing inference inputs. |
| `S3_PREFIX`         | Prefix for inference inputs. |
| `SCRIPT_S3_URI`     | S3 prefix containing the batch-inference setup files. |
| `AWS_REGION`        | AWS region (defaults to `us-east-1`). |
