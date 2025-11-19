# aws-contact-intelligence-pipeline
This repository contains a production-grade pipeline for call-intent classification. It provides:
- Transformer-based model training using HuggingFace + SageMaker Studio (or SageMaker Training)
- Custom Docker image for SageMaker Processing, built and published with CodeBuild → ECR 
- Batch inference over Amazon Connect / Salesforce voice-call transcripts via SageMaker Processing 
- Lambda orchestration for scheduled batch runs (typically through EventBridge) 
- Outputs written to S3 and queried downstream by Athena / QuickSight
  <img width="1800" height="1836" alt="Call Driver Classification Process Diagram" src="https://github.com/user-attachments/assets/406cb77a-09ee-421d-96db-7f9ec6806dbf" />
## High-Level Architecture
### 1. Training (`training/`)
Fine-tunes `microsoft/deberta-v3-large` on labeled call notes.
Produces HuggingFace-compatible artifacts: 
- model weights 
- tokenizer 
- label mapping 
- Artifacts are uploaded to S3 for batch inference.
---
### 2. Processing / Batch Inference (`processing/` + `docker/processing/`)
A custom Docker image (Python 3.10 slim + PyTorch, HF Transformers, PyAthena) is built and pushed to ECR.
A SageMaker Processing job uses this image and the `batch_inference.py` script to:
- Load the fine-tuned model + tokenizer from S3 
- Query Athena for new call records 
- Run batch inference 
- Write predictions as partitioned Parquet files back to S3 

### 3. Orchestration Lambda (`lambda/trigger_processing_job/`)
The Lambda function launches the SageMaker Processing job.
**Process:**
- Generate a unique job name 
- Pass model paths, tokenizer paths, Athena connection values 
- Provide S3 output prefix 
- Specify the ECR container image 
- Trigger the job via EventBridge 
---
### 4. Analytics / Reporting
Downstream reporting includes:
- Athena queries referencing S3 parquet outputs 
- QuickSight dashboards for call-driver reporting 
---
## Repository Structure
```
aws-contact-intelligence-pipeline/
├── docker/
│   └── processing/
│       ├── Dockerfile
│       ├── buildspec.yml
│       └── requirements.txt
│
├── lambda/
│   └── trigger_processing_job/
│       ├── app.py
│       ├── requirements.txt
│       └── README.md
│
├── processing/
│   ├── batch_inference.py
│   └── README.md
│
├── training/
│   ├── train.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
├── LICENSE
└── README.md
```
