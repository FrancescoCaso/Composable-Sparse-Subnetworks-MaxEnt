import logging
from typing import Dict
import wandb
import os
import json
import hashlib
import numpy as np
from train_utils import ClassesList
from pathlib import Path

def convert_to_serializable(obj):
    """Convert non-serializable types (e.g., numpy types) to serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, ClassesList):
        if obj.is_multidataset:
            raise NotImplementedError("Multidataset classes are not supported for JSON serialization.")
        return obj.classes
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def generate_config_hash(config: dict) -> str:
    """Generates a unique hash for a given configuration dictionary."""
    if '_id' in config:
        config.pop('_id')
    config_str = json.dumps(config, sort_keys=True, default=convert_to_serializable)
    # return hashlib.md5(config_str.encode()).hexdigest()  # Generate MD5 hash
    return hashlib.sha256(config_str.encode()).hexdigest()


class Logger:
    def __init__(self, name: str, logs_directory: Path = Path('./logs/'), results_directory: Path = Path('./results/'), config_file_path: Path = None):
        log_file_path = logs_directory/ f"{name}.log"

        if (log_file_path).exists():
            log_file_path.unlink()
        self.logger = logging.getLogger(name)
        self.name = name
            
        self.logs_directory = logs_directory
        self.results_directory = results_directory
        self.config_file_path = config_file_path
        self._configure_local_logger(name)
        
    def _configure_local_logger(self, name: str):
        """Configures the local logger with console and file handlers."""
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # File handler
        file_handler = logging.FileHandler(self.logs_directory/f"{name}.log", mode='w', encoding="utf-8")
        
        # Formatter
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M"
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)

    def __call__(self, stats: Dict[str, float]):
        """Logs the statistics both to the console and file."""
        message = " | ".join(f"{key}: {value}" for key, value in stats.items())
        self.logger.info(message)
    
    def finish(self):
        if self.config_file_path:
            # rename config_file_path to remove _raw
            new_name = self.config_file_path.with_name(self.config_file_path.stem.replace("_raw", "") + ".yaml")
            if new_name.exists() and not self.config_file_path.exists():
                pass
            else:
                self.config_file_path.rename(new_name)
            self.logger.info(f"Config file saved to {new_name}")
        self.logger.info('finish')

class WandbLogger(Logger):
    def __init__(self, name: str, logs_directory: str, results_directory: str, config_file_path: str = None, **kwargs):
        super().__init__(name, logs_directory, results_directory, config_file_path)
        # self.project_name = project_name
        # Initialize the Wandb run
        wandb.init(name=name, **kwargs)
        
    def __call__(self, stats: Dict[str, float]):
        """Logs statistics to both the local logger and Wandb."""
        super().__call__(stats)  # Log to local logger
        wandb.log(stats, step = stats['step'])         # Log to Wandb

    def finish(self):
        """Ends the Wandb run."""
        super().finish()
        wandb.finish()

from typing import Dict, Union
import yaml

# Register the representer with PyYAML: keep just the classes
yaml.add_representer(ClassesList, lambda dumper, data: dumper.represent_list(data.classes))

# Assuming Logger and WandbLogger classes are imported here

def initialize_logger_from_config(config: dict) -> Union[Logger, WandbLogger]:
    
    logger_config = config.get("logger", {})
    logger_type = logger_config.get("type", "local").lower()
    str_classes = ''.join(map(str, config["classes"]))
    choosen_class = ''.join(map(str, [config["classes"][ind] for ind, val in enumerate(config["weights"]) if val==1]))
    logs_directory = Path('./lth_logs/') if config["lth"] else Path('./logs/')
    results_directory = Path('./lth_results/') if config["lth"] else Path('./results/')
    
    logs_directory = logs_directory / config["dataset_name"] / str_classes / choosen_class
    results_directory = results_directory / config["dataset_name"] / str_classes / choosen_class
    results_directory.mkdir(parents=True, exist_ok=True)
    logs_directory.mkdir(parents=True, exist_ok=True)
    

    config_hash = generate_config_hash(config)
    name = f"{config_hash}"

    config_file_path = Path(results_directory) / f"{name}_config_raw.yaml"
    with open(config_file_path, "w") as config_file:
        yaml.dump(config, config_file)
    assert config_file_path.exists(), f"Config file {config_file_path} was not created."
    if logger_type == "wandb":
        
        kwargs = {
            "config": config,
            **(logger_config.get("wandb_args", {})) 
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        return WandbLogger(name=name, logs_directory=logs_directory, results_directory=results_directory, config_file_path=config_file_path, **kwargs)
    else:
        return Logger(name=name, logs_directory=logs_directory, results_directory=results_directory, config_file_path=config_file_path)