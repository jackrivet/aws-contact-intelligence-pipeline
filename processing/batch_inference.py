
"""
This script is executed inside an Amazon SageMaker Processing job.
It:
Downloads model + tokenizer artifacts from S3
Queries Athena for new call records since the last processed timestamp
Runs batched inference using a Hugging Face transformer
Writes predictions back to S3 as partitioned Parquet files
"""

import argparse
import io
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Config

@dataclass
class InferenceConfig:
    model_path: str
    tokenizer_path: str
    output_prefix: str
    athena_database: str
    athena_output: str
    s3_bucket: str
    s3_prefix: str
    region: str
    queue_name: Optional[str] = None  # Filter on queuename
    batch_size: int = 32
    model_local_dir: str = "/opt/ml/processing/input/local_model"
    tokenizer_local_dir: str = "/opt/ml/processing/input/local_tokenizer"
    athena_table: str = "unique_voicecall_prod"
    text_column: str = "detailed_call_notes__c"
    created_date_column: str = "createddate"


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--athena_database", type=str, required=True)
    parser.add_argument("--athena_output", type=str, required=True)
    parser.add_argument("--s3_bucket", type=str, required=True)
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="call-intent-batch-output/",
        help="S3 prefix where prediction Parquet files are stored/checked.",
    )
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument(
        "--queue_name",
        type=str,
        default=None,
        help="Optional queue name filter for the Athena query.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model inference.",
    )
    args = parser.parse_args()
    return InferenceConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_prefix=args.output_prefix,
        athena_database=args.athena_database,
        athena_output=args.athena_output,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        region=args.region,
        queue_name=args.queue_name,
        batch_size=args.batch_size,
    )

def get_s3_client(region: str):
    return boto3.client("s3", region_name=region)

def download_s3_folder(s3_client, bucket: str, prefix: str, local_path: str) -> None:
    os.makedirs(local_path, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = os.path.basename(key)
            if not filename:
                continue
            dest_path = os.path.join(local_path, filename)
            s3_client.download_file(bucket, key, dest_path)

def extract_latest_timestamp_from_prefixes(
    s3_client, bucket: str, prefix: str
) -> Optional[datetime]:
    """Scan existing prediction files under S3 prefix and extract timestamp."""
    paginator = s3_client.get_paginator("list_objects_v2")
    timestamp_strings: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", key)
            if match:
                timestamp_strings.append(match.group(1))
    if not timestamp_strings:
        print("No prior prediction files found under prefix; processing full dataset.")
        return None
    last_dt = max(datetime.fromisoformat(ts) for ts in timestamp_strings)
    print(f"Last processed timestamp detected from S3 prefix: {last_dt}")
    return last_dt

# Athena 

def build_athena_query(cfg: InferenceConfig, last_processed: Optional[datetime]) -> str:
    date_clause = ""
    if last_processed is not None:
        date_clause = (
            f"AND from_iso8601_timestamp({cfg.created_date_column}) "
            f"> TIMESTAMP '{last_processed.isoformat(sep=' ')}'"
        )
    queue_clause = ""
    if cfg.queue_name:
        queue_clause = f"AND queuename = '{cfg.queue_name}'"
      
    query = f"
    SELECT *
    FROM "{cfg.athena_database}"."{cfg.athena_table}"
    WHERE {cfg.text_column} IS NOT NULL
    {queue_clause}
    {date_clause}; 
    "
    print("Athena query:\n", query)
    return query

def run_athena_query(cfg: InferenceConfig, query: str) -> pd.DataFrame:
    cursor = connect(
        s3_staging_dir=cfg.athena_output,
        region_name=cfg.region,
        schema_name=cfg.athena_database,
        cursor_class=PandasCursor,
    ).cursor()
    df = cursor.execute(query).as_pandas()
    print(f"Fetched {len(df)} rows from Athena.")
    return df

# Inference

def load_model_and_tokenizer(cfg: InferenceConfig, s3_client):
    model_s3_prefix = cfg.model_path.replace(f"s3://{cfg.s3_bucket}/", "")
    tokenizer_s3_prefix = cfg.tokenizer_path.replace(f"s3://{cfg.s3_bucket}/", "")
    download_s3_folder(s3_client, cfg.s3_bucket, model_s3_prefix, cfg.model_local_dir)
    download_s3_folder(
        s3_client, cfg.s3_bucket, tokenizer_s3_prefix, cfg.tokenizer_local_dir
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(cfg.tokenizer_local_dir)
    model = DistilBertForSequenceClassification.from_pretrained(
        cfg.model_local_dir, local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Loaded model and tokenizer. Using device: {device}")
    return model, tokenizer, device, model_s3_prefix

def load_label_mapping(s3_client, bucket: str, model_s3_prefix: str) -> dict:
    label_map_key = os.path.join(model_s3_prefix, "label_mapping.csv")
    obj = s3_client.get_object(Bucket=bucket, Key=label_map_key)
    label_map = pd.read_csv(obj["Body"])
    id2label = dict(zip(label_map["Label ID"], label_map["Category"]))
    print(f"Loaded {len(id2label)} label mappings.")
    return id2label

def run_inference(
    df: pd.DataFrame,
    text_column: str,
    model,
    tokenizer,
    device,
    batch_size: int,
    id2label: dict,
) -> pd.DataFrame:
    df = df.copy()
    df.dropna(subset=[text_column], inplace=True)
    texts = df[text_column].tolist()
    predicted_labels: List[int] = []
    print(f"Running inference on {len(texts)} records with batch_size={batch_size}")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts, truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            preds = torch.argmax(outputs.logits, dim=1).tolist()
            predicted_labels.extend(preds)
    df["Predicted Category"] = [id2label[p] for p in predicted_labels]
    return df

# Output

def write_predictions_by_date(
    cfg: InferenceConfig, s3_client, df: pd.DataFrame
) -> None:
    df = df.copy()
    df["call_date"] = pd.to_datetime(df[cfg.created_date_column], errors="coerce")

    if cfg.output_prefix.startswith("s3://"):
        output_prefix_clean = cfg.output_prefix.replace(f"s3://{cfg.s3_bucket}/", "")
    else:
        output_prefix_clean = cfg.output_prefix
    base_columns = [
        "id",
        "vendorcallkey",
        cfg.text_column,
        "call_date",
        "Predicted Category",
    ]
    for date_value, group in df.groupby(df["call_date"].dt.date):
        unique_id = uuid.uuid4().hex
        timestamp_str = datetime.utcnow().replace(microsecond=0).isoformat()
        output_key = f"{output_prefix_clean}{timestamp_str}_{unique_id}.parquet"
        group = group.copy()
        group["call_date"] = group["call_date"].astype(str)
        table = pa.Table.from_pandas(group[base_columns])
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        s3_client.put_object(
            Bucket=cfg.s3_bucket,
            Key=output_key,
            Body=buffer.getvalue(),
        )
        print(f"Wrote Parquet predictions for date {date_value} to s3://{cfg.s3_bucket}/{output_key}")
    print("Batch classification saved as Parquet to S3 with unique daily filenames.")

# Main

def main():
    cfg = parse_args()
    s3_client = get_s3_client(cfg.region)
    last_processed = extract_latest_timestamp_from_prefixes(
        s3_client, cfg.s3_bucket, cfg.s3_prefix
    )
    query = build_athena_query(cfg, last_processed)
    df = run_athena_query(cfg, query)
    if df.empty:
        print("No new records to process. Exiting.")
        return
      
    model, tokenizer, device, model_s3_prefix = load_model_and_tokenizer(
        cfg, s3_client
    )
    id2label = load_label_mapping(s3_client, cfg.s3_bucket, model_s3_prefix)
    df_with_preds = run_inference(
        df,
        text_column=cfg.text_column,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=cfg.batch_size,
        id2label=id2label,
    )
    write_predictions_by_date(cfg, s3_client, df_with_preds)
  
if __name__ == "__main__":
    main()
