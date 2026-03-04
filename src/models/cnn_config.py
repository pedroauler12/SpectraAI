"""
Configuração de hiperparâmetros para o modelo CNN via YAML.
Suporta múltiplas configurações para experimentação iterativa.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any


def get_config_path(config_name: str = "baseline") -> Path:
    """Retorna o caminho do arquivo de configuração YAML."""
    config_dir = Path(__file__).parent / "configs"
    config_file = config_dir / f"{config_name}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {config_file}")
    
    return config_file


def load_config(config_name: str = "baseline") -> Dict[str, Any]:
    """
    Carrega configuração de um arquivo YAML.
    
    Args:
        config_name: Nome do arquivo (sem .yaml)
                    Ex: "baseline", "higher_dropout"
    
    Returns:
        Dict com as configurações carregadas
    
    Exemplo:
        >>> config = load_config("baseline")
        >>> config["model"]["filters"]
        [32, 64]
    """
    config_path = get_config_path(config_name)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Converter caminhos relativos para absolutos (relativos ao repo root)
    # config_path está em: {repo_root}/src/models/configs/{name}.yaml
    # Então repo_root = config_path.parent.parent.parent.parent (sobe 4 níveis)
    repo_root = config_path.parent.parent.parent.parent
    
    if "data" in config:
        for key in ["dataset_path", "codes_path", "normalizer_path"]:
            if key in config["data"]:
                rel_path = config["data"][key]
                if not Path(rel_path).is_absolute():
                    config["data"][key] = str(repo_root / rel_path)
    
    if "output" in config:
        for key in ["models_dir", "logs_dir"]:
            if key in config["output"]:
                rel_path = config["output"][key]
                if not Path(rel_path).is_absolute():
                    config["output"][key] = str(repo_root / rel_path)
    
    return config


def list_available_configs() -> list:
    """Lista todas as configurações disponíveis."""
    config_dir = Path(__file__).parent / "configs"
    configs = [f.stem for f in config_dir.glob("*.yaml")]
    return sorted(configs)


def save_experiment_config(config: Dict[str, Any], 
                          experiment_dir: Path) -> None:
    """
    Salva a configuração usada em um experimento.
    Facilita rastreamento posterior.
    
    Args:
        config: Dicionário de configuração
        experiment_dir: Diretório onde salvar
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = experiment_dir / "config_used.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_model_config(config_name: str = "baseline") -> Dict[str, Any]:
    """Retorna apenas a seção 'model' da configuração."""
    config = load_config(config_name)
    return config.get("model", {})


def get_training_config(config_name: str = "baseline") -> Dict[str, Any]:
    """Retorna apenas a seção 'training' da configuração."""
    config = load_config(config_name)
    return config.get("training", {})


def get_data_config(config_name: str = "baseline") -> Dict[str, Any]:
    """Retorna apenas a seção 'data' da configuração."""
    config = load_config(config_name)
    return config.get("data", {})


def get_output_config(config_name: str = "baseline") -> Dict[str, Any]:
    """Retorna apenas a seção 'output' da configuração."""
    config = load_config(config_name)
    return config.get("output", {})
