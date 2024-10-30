from google.cloud import storage
import argparse
from datasets import load_dataset
import os
from datetime import datetime

def upload_to_gcs(bucket_name: str, source_dir: str, destination_prefix: str):
    """Upload dataset files to GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create GCS path by replacing local path with GCS prefix
            gcs_path = os.path.join(
                destination_prefix,
                os.path.relpath(local_path, source_dir)
            )
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset and upload to GCS')
    parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--bucket-name', required=True, help='Bucket name (without gs://)')
    parser.add_argument('--dataset', default='imdb', help='Dataset name from Hugging Face')
    parser.add_argument('--sample-size', type=int, default=None, 
                       help='Number of examples to sample (for testing)')
    
    args = parser.parse_args()

    # Create timestamp for versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    
    if args.sample_size:
        dataset['train'] = dataset['train'].select(range(args.sample_size))
        if 'validation' in dataset:
            dataset['validation'] = dataset['validation'].select(range(args.sample_size // 5))
        elif 'test' in dataset:
            dataset['test'] = dataset['test'].select(range(args.sample_size // 5))

    # Save dataset locally first
    local_path = f"temp_dataset_{timestamp}"
    print(f"Saving dataset to {local_path}")
    dataset.save_to_disk(local_path)

    # Upload to GCS
    gcs_prefix = f"datasets/{args.dataset}_{timestamp}"
    print(f"Uploading to GCS bucket: {args.bucket_name}")
    upload_to_gcs(args.bucket_name, local_path, gcs_prefix)

    # Update config with dataset path
    config_update = f"""
Update your configs/config.yaml with:

pipeline:
  data_path: "gs://{args.bucket_name}/{gcs_prefix}"
"""
    print(config_update)

    # Cleanup
    import shutil
    shutil.rmtree(local_path)
    print("Cleaned up temporary files")

if __name__ == "__main__":
    main()