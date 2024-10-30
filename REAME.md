# LLaMA Vertex AI Fine-tuning Pipeline

A streamlined pipeline for fine-tuning LLaMA models on Google Cloud's Vertex AI platform.

## Prerequisites

1. Google Cloud Project with Vertex AI API enabled
2. Google Cloud credentials configured
3. Python 3.9+
4. A dataset on Hugging Face (or other supported format)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ethanshenley/llama-vertex-pipeline.git
cd llama-vertex-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Update `configs/config.yaml` with your project settings:
```yaml
project:
  id: "your-project-id"
  region: "us-central1"
  bucket: "gs://your-bucket"
```

## Usage

Run the pipeline with your dataset:
```bash
python run_pipeline.py --data-path "your_username/your_dataset"
```

The pipeline will:
1. Fine-tune the LLaMA model on your dataset
2. Upload the trained model to Vertex AI Model Registry
3. Deploy the model to an endpoint

## Monitor Progress

1. Go to Google Cloud Console
2. Navigate to Vertex AI > Pipelines
3. Find your pipeline run under the name specified in your config

## Configuration

Edit `configs/config.yaml` to customize:
- Model parameters (epochs, batch size, etc.)
- Machine type for training
- Base model version
- Pipeline name and settings

## Troubleshooting

Common issues:
1. **GPU Quota**: Ensure you have sufficient quota for A2 GPUs in your project
2. **Dataset Format**: Ensure your dataset has "text" and "label" columns
3. **Permissions**: Verify your service account has required permissions

## License

MIT