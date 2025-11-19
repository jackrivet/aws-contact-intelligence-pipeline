# Model Training – Call Intent Classification
This directory contains the full training pipeline used to fine-tune a transformer-based call intent classification model. It supports both local GPU training and SageMaker Training Jobs and produces model artifacts consumed directly by the SageMaker Processing batch inference workflow.
The model architecture is DeBERTa-v3-large.
## Overview
This training pipeline performs the following steps:
- Load labeled call data containing “note” (text) and “label” (category).
- Encode labels into integer IDs.
- Tokenize text using DeBERTa-v3-large (or others).
- Train/test split.
- Fine-tune using HuggingFace Trainer.
- Compute evaluation metrics (macro-F1, accuracy).
- Save the final model, tokenizer, and label mapping in a SageMaker-compatible structure.

## Training With DeBERTa-v3-large in SageMaker Unified Studio
Uploaded the base checkpoint “microsoft/deberta-v3-large” into SageMaker Unified Studio. This enables:
- GPU-accelerated fine-tuning directly inside Studio
- Rapid iteration and experimentation
- Direct export of trained artifacts to S3
## SageMaker Training Job Example
```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="sagemaker/training",
    role="arn:aws:iam::<ACCOUNT_ID>:role/SageMakerRole",
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
    hyperparameters={
        "data_path": "/opt/ml/input/data/train/labeled_calls.csv",
        "model_checkpoint": "microsoft/deberta-v3-large",
        "output_dir": "/opt/ml/model",
    },
)
estimator.fit({"train": "s3://your-bucket/training-data/"})
```
## Dataset Requirements
Your labeled dataset must include:
- note | the call note text 
- label  | the human-assigned category label 

The training script:
- drops nulls
- encodes labels into integers 0..N
- saves the mapping for inference

## Summary
This directory implements the complete training pipeline for the DeBERTa-v3-large call intent classifier. It supports:
Local GPU training
SageMaker Unified Studio execution
Managed SageMaker Training Jobs
Export to S3 for use in SageMaker Processing batch inference
Artifacts generated here integrate directly with processing/
