import argparse
import sys
from google.cloud import aiplatform
from kfp import compiler
from src.pipeline import create_fine_tuning_pipeline
from src.config import load_config
import os

def main():
    parser = argparse.ArgumentParser(description='Run LLaMA fine-tuning pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data-path', type=str, required=True,
                      help='Path to dataset (e.g., "your_username/your_dataset")')
    parser.add_argument('--hf-token', type=str,
                      help='Hugging Face token (or set HF_TOKEN env var)')
    args = parser.parse_args()

    # Get token from args or environment
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if not hf_token:
        print("Please provide Hugging Face token via --hf-token or HF_TOKEN environment variable",
              file=sys.stderr)
        sys.exit(1)

    try:
        # Load configuration
        config = load_config(args.config)
        config.data_path = args.data_path

        print(f"Initializing Vertex AI with project {config.project_id} in {config.region}")
        # Initialize Vertex AI
        aiplatform.init(
            project=config.project_id,
            location=config.region
        )

        # Compile the pipeline
        pipeline_path = "pipeline.json"
        print("Compiling pipeline...")
        compiler.Compiler().compile(
            pipeline_func=create_fine_tuning_pipeline,
            package_path=pipeline_path
        )
        print(f"Pipeline compiled successfully to {pipeline_path}")

        # Create and run pipeline job
        print("Creating pipeline job...")
        job = aiplatform.PipelineJob(
            display_name=config.pipeline_name,
            template_path=pipeline_path,
            pipeline_root=f"{config.bucket}/pipeline_root",
            parameter_values={
                "project_id": config.project_id,
                "data_path": config.data_path,
                "region": config.region,
                "base_model": config.base_model,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "machine_type": config.machine_type,
                "hf_token": hf_token
            }
        )

        print("Starting pipeline job...")
        job.run()
        print(f"Pipeline job started successfully. Monitor progress at:")
        print(f"https://console.cloud.google.com/vertex-ai/pipelines/runs?project={config.project_id}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()