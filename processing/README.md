# SageMaker Processing - Batch Inference
This directory contains the processing script executed inside a SageMaker
Processing job. It loads model artifacts from S3, queries Athena for new records,
performs inference, and writes Parquet outputs to S3.
## Entry Point
batch_inference.py
## Arguments
--model_path 
--tokenizer_path 
--output_prefix 
--athena_database 
--athena_output 
--s3_bucket 
--s3_prefix 
--region 
--queue_name 
## Docker Build
This script is packaged into the Docker image located in `docker/processing/`.
