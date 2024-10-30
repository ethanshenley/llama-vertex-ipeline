import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class PipelineConfig:
    project_id: str
    region: str
    bucket: str
    pipeline_name: str
    machine_type: str
    base_model: str
    display_name: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    data_path: Optional[str] = None

def load_config(config_path: str) -> PipelineConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return PipelineConfig(
        project_id=config_dict['project']['id'],
        region=config_dict['project']['region'],
        bucket=config_dict['project']['bucket'],
        pipeline_name=config_dict['pipeline']['name'],
        machine_type=config_dict['pipeline']['machine_type'],
        base_model=config_dict['model']['base_model'],
        display_name=config_dict['model']['display_name'],
        num_epochs=config_dict['training']['num_epochs'],
        batch_size=config_dict['training']['batch_size'],
        learning_rate=config_dict['training']['learning_rate']
    )