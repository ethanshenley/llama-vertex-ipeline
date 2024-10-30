from google.cloud import aiplatform
from kfp import dsl, compiler
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.endpoint import (
    EndpointCreateOp,
    ModelDeployOp
)
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from kfp.components import importer_node

@dsl.component(
    base_image="gcr.io/deeplearning-platform-release/pytorch-gpu.1-13",
    packages_to_install=[
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "huggingface_hub>=0.16.4",
        "bitsandbytes>=0.41.1",
        "scipy>=1.10.0",
        "peft>=0.4.0",
        "psutil"  # For memory monitoring
    ],
    output_component_file="fine_tune_llama_component.yaml"
)
def fine_tune_llama(
    project_id: str,
    data_path: str,
    base_model: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str,
    hf_token: str
) -> str:
    """Fine-tune LLaMA model component."""
    import os
    import torch
    import psutil
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        BitsAndBytesConfig,
        DataCollatorWithPadding
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    from datasets import load_dataset
    from huggingface_hub import login, HfFolder
    import numpy as np

    def print_gpu_memory():
        """Print GPU memory usage."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
                print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                print(f"GPU {i} memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    def print_ram_usage():
        """Print RAM usage."""
        print(f"RAM usage: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
        print(f"Total RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")

    try:
        # Print initial system information
        print("\n=== System Information ===")
        print_gpu_memory()
        print_ram_usage()
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Current CUDA device name: {torch.cuda.get_device_name()}")
        
        # Set reduced batch size and gradient accumulation steps
        effective_batch_size = 2  # Reduced from 4
        gradient_accumulation_steps = 16  # Increased from 4
        print(f"\nUsing effective batch size of {effective_batch_size} with {gradient_accumulation_steps} gradient accumulation steps")
        
        # Set environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print("\n=== Starting Model Loading ===")
        HfFolder.save_token(hf_token)
        login(token=hf_token)
        
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token=hf_token,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            print("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("\nConfiguring model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        
        print("\nLoading base model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            token=hf_token,
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        print("\n=== Memory After Model Load ===")
        print_gpu_memory()
        print_ram_usage()
        
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        
        print("\nConfiguring LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        
        print("\nPreparing model for training...")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        
        print("\nLoading dataset...")
        dataset = load_dataset("imdb")
        print(f"Train size: {len(dataset['train'])}")
        print(f"Test size: {len(dataset['test'])}")
        
        def preprocess_function(examples):
            result = tokenizer(
                examples["text"],
                padding="max_length",
                max_length=512,
                truncation=True
            )
            result["labels"] = examples["label"]
            return result
        
        print("\nPreprocessing dataset...")
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Running tokenizer on dataset"
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = np.mean(predictions == labels)
            return {"accuracy": accuracy}
        
        print("\nSetting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=effective_batch_size,
            per_device_eval_batch_size=effective_batch_size,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=1,  # Log every step
            save_strategy="steps",
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            gradient_checkpointing=True,
            fp16=True,
            optim="paged_adamw_32bit",
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            log_level="info",
            logging_first_step=True,
            logging_nan_inf_filter=False  # To see if we get NaN losses
        )
        
        print("\nInitializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        print("\n=== Starting Training ===")
        print("You should see training progress immediately after this...")
        trainer.train()
        
        print("\nSaving model...")
        trainer.save_model(output_dir)
        model.save_pretrained(f"{output_dir}/adapter")
        tokenizer.save_pretrained(output_dir)
        
        return output_dir
        
    except Exception as e:
        print(f"\nError during model loading/training: {str(e)}")
        print(f"Error type: {type(e)}")
        print("\nFull traceback:")
        import traceback
        print(traceback.format_exc())
        print("\nFinal memory state:")
        print_gpu_memory()
        print_ram_usage()
        raise e

# Pipeline definition (no changes needed)
@dsl.pipeline(
    name="llama-fine-tuning-pipeline",
    description="Pipeline for fine-tuning LLaMA model"
)
def create_fine_tuning_pipeline(
    project_id: str,
    data_path: str,
    region: str,
    base_model: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    machine_type: str,
    hf_token: str
):
    """Define the fine-tuning pipeline."""
    
    # Fine-tune model
    fine_tune_task = (fine_tune_llama(
        project_id=project_id,
        data_path=data_path,
        base_model=base_model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=f"gs://{project_id}-artifacts/model",
        hf_token=hf_token
    )
    .set_cpu_limit('4')
    .set_memory_limit('16G')
    .set_accelerator_type('NVIDIA_TESLA_T4')
    .set_accelerator_limit(1))
    
    # Create UnmanagedContainerModel artifact
    container_spec = {
        'imageUri': 'gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-12'
    }
    
    model_spec = importer_node.importer(
        artifact_uri=fine_tune_task.outputs["Output"],
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            'containerSpec': container_spec
        }
    )
    
    # Upload model to registry
    model_upload = ModelUploadOp(
        project=project_id,
        location=region,
        display_name=f"llama-fine-tuned-{dsl.PIPELINE_JOB_ID_PLACEHOLDER}",
        unmanaged_container_model=model_spec.outputs["artifact"]
    )
    
    # Create endpoint
    endpoint_create = EndpointCreateOp(
        project=project_id,
        location=region,
        display_name=f"llama-endpoint-{dsl.PIPELINE_JOB_ID_PLACEHOLDER}"
    )
    
    # Deploy model
    ModelDeployOp(
        endpoint=endpoint_create.outputs["endpoint"],
        model=model_upload.outputs["model"]
    )