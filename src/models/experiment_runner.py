"""
Pipeline para executar experimentos CNN com diferentes configurações.
Gerencia treinamento, logging e salvamento de resultados.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .cnn_config import (
    load_config,
    get_model_config,
    get_training_config,
    get_data_config,
    get_output_config,
    save_experiment_config,
    list_available_configs,
)
from .cnn_builder import build_cnn_model
from .cnn_data_prep import prepare_cnn_inputs


class ExperimentRunner:
    """Executa e gerencia experimentos de treinamento CNN."""
    
    def __init__(self, config_name: str = "baseline"):
        """
        Inicializa runner com uma configuração.
        
        Args:
            config_name: Nome da configuração YAML (sem .yaml)
        """
        self.config_name = config_name
        self.config = load_config(config_name)
        self.model = None
        self.history = None
        self.experiment_dir = None
        
        print(f"✓ ExperimentRunner inicializado com config: {config_name}")
        print(f"  Configurações disponíveis: {list_available_configs()}")
    
    def prepare_data(self, limit_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega e prepara dados para treinamento.
        
        Args:
            limit_samples: Limita número de amostras (para teste rápido)
        
        Returns:
            X_train, y_train
        """
        data_cfg = self.config["data"]
        model_cfg = self.config["model"]
        
        print(f"\n📂 Carregando dados...")
        
        # Carregar dataset
        df = pd.read_csv(data_cfg["dataset_path"])
        if limit_samples:
            df = df.sample(n=limit_samples, random_state=42)
        
        # Carregar labels
        with open(data_cfg["codes_path"]) as f:
            codes = json.load(f)
        
        # Preparar com normalizer precomputado
        import numpy as np
        npz = np.load(data_cfg["normalizer_path"])
        precomputed_normalizer = {
            "method": "zscore",
            "mean": npz["mean"],
            "std": npz["std"],
        }
        
        result = prepare_cnn_inputs(
            df,
            labels=None,
            extracted_codes_path=data_cfg["codes_path"],
            normalization=data_cfg["normalization_method"],
            normalizer=precomputed_normalizer,
            data_format="channels_last",
        )
        
        X = result['X']
        y = result['y']
        
        # Remover label inválido (-1)
        valid_mask = y != -1
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"✓ Dados carregados: {X.shape}, labels: {y.shape}")
        print(f"  Classes: {np.unique(y)}, distribuição: {np.bincount(y)}")
        
        return X, y
    
    def build_model(self) -> keras.Model:
        """Constrói modelo CNN com configuração."""
        model_cfg = self.config["model"]
        training_cfg = self.config["training"]
        
        print(f"\n🏗️ Construindo modelo...")
        
        filters = model_cfg["filters"]
        self.model = build_cnn_model(
            input_shape=tuple(model_cfg["input_shape"]),
            n_classes=model_cfg["num_classes"],
            conv1_filters=filters[0],
            conv2_filters=filters[1],
            kernel_size=model_cfg["kernel_size"],
            l2_regularizer=model_cfg["l2_regularizer"],
            conv_dropout_rate=model_cfg["conv_dropout_rate"],
            dropout_rate=model_cfg["dense_dropout_rate"],
            dense_units=model_cfg["dense_units"],
        )
        
        # Compilar
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=training_cfg["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print(f"✓ Modelo compilado!")
        return self.model
    
    def create_experiment_dir(self) -> Path:
        """Cria diretório para salvar resultados do experimento."""
        output_cfg = self.config["output"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exp_dir = Path(output_cfg["models_dir"]) / f"{self.config_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_dir = exp_dir
        
        # Salvar cópia da config
        save_experiment_config(self.config, exp_dir)
        
        print(f"\n💾 Experimento: {exp_dir.name}")
        return exp_dir
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Treina modelo.
        
        Args:
            X_train: Features
            y_train: Labels
            verbose: 0=silent, 1=progress bar, 2=one line per epoch
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        training_cfg = self.config["training"]
        
        print(f"\n🚀 Iniciando treinamento...")
        print(f"  Batch size: {training_cfg['batch_size']}")
        print(f"  Epochs: {training_cfg['epochs']}")
        print(f"  Learning rate: {training_cfg['learning_rate']}")
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=training_cfg["batch_size"],
            epochs=training_cfg["epochs"],
            validation_split=training_cfg["validation_split"],
            verbose=verbose,
        )
        
        print(f"✓ Treinamento concluído!")
        return self.history.history
    
    def save_results(self) -> None:
        """Salva modelo e histórico de treinamento."""
        if self.experiment_dir is None:
            self.create_experiment_dir()
        
        output_cfg = self.config["output"]
        
        if output_cfg["save_model"] and self.model:
            model_path = self.experiment_dir / "model.h5"
            self.model.save(model_path)
            print(f"✓ Modelo salvo: {model_path}")
        
        if output_cfg["save_history"] and self.history:
            history_path = self.experiment_dir / "history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f, indent=2)
            print(f"✓ Histórico salvo: {history_path}")
    
    def run_full_pipeline(self,
                         limit_samples: int = None,
                         verbose: int = 1) -> Dict[str, Any]:
        """
        Executa pipeline completo: preparar dados → build → treinar → salvar.
        
        Args:
            limit_samples: Limita amostras (para testes rápidos)
            verbose: Verbosidade do treinamento
        
        Returns:
            Dicionário com resultado do experimento
        """
        print(f"\n{'='*60}")
        print(f"EXPERIMENTO: {self.config_name}")
        print(f"{'='*60}")
        
        # 1. Preparar dados
        X_train, y_train = self.prepare_data(limit_samples)
        
        # 2. Build modelo
        self.build_model()
        
        # 3. Criar diretório
        self.create_experiment_dir()
        
        # 4. Treinar
        history_dict = self.train(X_train, y_train, verbose=verbose)
        
        # 5. Salvar
        self.save_results()
        
        # 6. Resumo
        result = {
            "config_name": self.config_name,
            "experiment_dir": str(self.experiment_dir),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "final_train_loss": float(history_dict["loss"][-1]),
            "final_train_acc": float(history_dict["accuracy"][-1]),
            "final_val_loss": float(history_dict["val_loss"][-1]),
            "final_val_acc": float(history_dict["val_accuracy"][-1]),
            "epochs_run": len(history_dict["loss"]),
        }
        
        # 7. Log em CSV
        self._log_result_to_csv(result)
        
        print(f"\n{'='*60}")
        print(f"RESUMO DO EXPERIMENTO:")
        print(f"{'='*60}")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        return result
    
    def _log_result_to_csv(self, result: Dict[str, Any]) -> None:
        """Salva resultado em CSV com append (rastreamento de experimentos)."""
        output_cfg = self.config["output"]
        csv_path = Path(output_cfg["models_dir"]) / "experiments_log.csv"
        
        # Preparar dados para CSV
        row = pd.DataFrame([{
            "timestamp": result["timestamp"],
            "config_name": result["config_name"],
            "train_loss": result["final_train_loss"],
            "train_acc": result["final_train_acc"],
            "val_loss": result["final_val_loss"],
            "val_acc": result["final_val_acc"],
            "epochs": result["epochs_run"],
            "experiment_dir": result["experiment_dir"],
        }])
        
        # Append ou criar novo
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            df_updated = pd.concat([df_existing, row], ignore_index=True)
            df_updated.to_csv(csv_path, index=False)
            print(f"✓ Resultado adicionado ao log: {csv_path}")
        else:
            row.to_csv(csv_path, index=False)
            print(f"✓ Novo log de experimentos criado: {csv_path}")


def run_multiple_experiments(config_names: list,
                            limit_samples: int = None) -> list:
    """
    Executa múltiplos experimentos sequencialmente.
    
    Args:
        config_names: Lista de nomes de config
        limit_samples: Limita amostras para teste
    
    Returns:
        Lista com resultados de cada experimento
    """
    results = []
    
    for config_name in config_names:
        try:
            runner = ExperimentRunner(config_name)
            result = runner.run_full_pipeline(limit_samples=limit_samples, verbose=1)
            results.append(result)
        except Exception as e:
            print(f"❌ Erro em {config_name}: {e}")
            results.append({
                "config_name": config_name,
                "error": str(e),
            })
    
    return results


if __name__ == "__main__":
    # Exemplo: rodar experimento baseline
    runner = ExperimentRunner("baseline")
    runner.run_full_pipeline(limit_samples=50)  # Teste rápido com 50 amostras
