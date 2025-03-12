import argparse
from typing import Optional, Union, List, Dict, Any
import os
import yaml
from pydantic import BaseModel, Field, validator, field_validator
from schema import PARAM_BY_NAME, MODEL_BASE_PATH, DEFAULT_BEAM_WIDTH, DEFAULT_BRANCHING_FACTOR, DEFAULT_TRAINING_DATA_PATH


class ConfigModel(BaseModel):
    """Configuration model with validation using pydantic"""
    # Application parameters
    verbose: bool = Field(default=PARAM_BY_NAME["verbose"].default)
    
    # Model paths
    policy_model: str = Field(default=PARAM_BY_NAME["policy_model"].default)
    pp_model: str = Field(default=PARAM_BY_NAME["pp_model"].default)
    policy_model_dir: Optional[str] = None
    pp_model_dir: Optional[str] = None
    
    # Generation parameters
    max_tokens: int = Field(default=PARAM_BY_NAME["max_tokens"].default, gt=0)
    max_depth: int = Field(default=PARAM_BY_NAME["max_depth"].default, gt=0)
    max_iterations: int = Field(default=PARAM_BY_NAME["max_iterations"].default, gt=0)
    
    # Data parameters
    data_folder: str = Field(default=PARAM_BY_NAME["data_folder"].default)
    task_index: int = Field(default=PARAM_BY_NAME["task_index"].default, gt=0)
    task_name: str = Field(default=PARAM_BY_NAME["task_name"].default)
    all_tasks: bool = Field(default=PARAM_BY_NAME["all_tasks"].default)
    
    # Search parameters
    search_mode: str = Field(default=PARAM_BY_NAME["search_mode"].default)
    temperature: float = Field(
        default=PARAM_BY_NAME["temperature"].default, 
        ge=0.0, 
        le=1.0
    )
    seed: int = Field(default=PARAM_BY_NAME["seed"].default)
    deterministic: bool = Field(default=PARAM_BY_NAME["deterministic"].default)
    
    # Hardware parameters
    gpus: int = Field(default=PARAM_BY_NAME["gpus"].default, ge=0)
    dtype: str = Field(default=PARAM_BY_NAME["dtype"].default)
    
    # Output parameters
    output_dir: str = Field(default=PARAM_BY_NAME["output_dir"].default)
    hint: str = Field(default=PARAM_BY_NAME["hint"].default)
    
    # Beam search specific parameters
    beam_width: int = Field(default=PARAM_BY_NAME["beam_width"].default, gt=0)
    branching_factor: int = Field(default=PARAM_BY_NAME["branching_factor"].default, gt=0)
    
    # SLURM parameters - not used in Python app, but included for completeness
    mem: Optional[str] = None
    cpus: Optional[int] = None
    partition: Optional[str] = None
    exclude: Optional[str] = None
    nodelist: Optional[str] = None
    time: Optional[str] = None
    
    @field_validator('search_mode')
    def validate_search_mode(cls, v):
        if v.lower() not in ['beam_search', 'mcts']:
            raise ValueError(f"Search mode '{v}' not supported. Use 'beam_search' or 'mcts'.")
        return v
    
    @field_validator('dtype')
    def validate_dtype(cls, v):
        if v not in ['float16', 'bfloat16', 'float32']:
            raise ValueError(f"Data type '{v}' not supported. Use 'float16', 'bfloat16', or 'float32'.")
        return v


class Config:
    """Configuration manager that handles loading from different sources"""
    
    def __init__(self, args: argparse.Namespace):
        # Start with empty config
        config_data = {}
        
        # First load config from file if provided
        if hasattr(args, 'config_file') and args.config_file:
            config_data.update(self._load_from_file(args.config_file))
        
        # Then override with command line arguments
        for param_name in PARAM_BY_NAME:
            # Skip SLURM-specific parameters
            if param_name in ['mem', 'cpus', 'partition', 'exclude', 'nodelist', 'time']:
                continue
                
            if hasattr(args, param_name) and getattr(args, param_name) is not None:
                config_data[param_name] = getattr(args, param_name)
        
        # Initialize pydantic model with data
        self._model = ConfigModel(**config_data)
        
        # Add computed fields
        self._model.policy_model_dir = os.path.join(MODEL_BASE_PATH, "policy")
        self._model.pp_model_dir = os.path.join(MODEL_BASE_PATH, "pp")

    def _load_from_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
                
            # Convert dashed keys to underscore
            return {k.replace('-', '_'): v for k, v in config_data.items()}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return self._model.model_dump(exclude_none=True)
        
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        try:
            # Convert underscores back to dashes for YAML
            data = {k.replace('_', '-'): v for k, v in self.to_dict().items()}
            
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"Config saved to {file_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def __getattr__(self, name):
        """Delegate attribute access to the pydantic model."""
        try:
            return getattr(self._model, name)
        except AttributeError:
            raise AttributeError(f"Config has no attribute '{name}'")
