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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    average_precision_score,
)

from .cnn_config import (
    load_config,
    get_model_config,
    get_training_config,
    get_data_config,
    get_output_config,
    save_experiment_config,
    list_available_configs,
)
from .cnn_builder import build_cnn_model, build_deep_cnn_model
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
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
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

        if len(filters) > 2:
            # Arquitetura profunda com N blocos convolucionais
            self.model = build_deep_cnn_model(
                input_shape=tuple(model_cfg["input_shape"]),
                n_classes=model_cfg["num_classes"],
                filters_list=filters,
                kernel_size=model_cfg["kernel_size"],
                l2_regularizer=model_cfg["l2_regularizer"],
                conv_dropout_rate=model_cfg["conv_dropout_rate"],
                dropout_rate=model_cfg["dense_dropout_rate"],
                dense_units=model_cfg["dense_units"],
            )
        else:
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
        Treina modelo com split train/val para calcular métricas posteriores.
        
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
        val_split = training_cfg["validation_split"]
        
        # Fazer split manual para ter acesso ao conjunto de validação
        val_size = int(len(X_train) * val_split)
        indices = np.random.RandomState(42).permutation(len(X_train))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_tr = X_train[train_indices]
        y_tr = y_train[train_indices]
        self.X_val = X_train[val_indices]
        self.y_val = y_train[val_indices]
        self.X_train = X_tr  # Guardar para referência
        self.y_train = y_tr
        
        print(f"\n🚀 Iniciando treinamento...")
        print(f"  Batch size: {training_cfg['batch_size']}")
        print(f"  Epochs: {training_cfg['epochs']}")
        print(f"  Learning rate: {training_cfg['learning_rate']}")
        print(f"  Train size: {len(X_tr)}, Val size: {len(self.X_val)}")
        
        self.history = self.model.fit(
            X_tr, y_tr,
            batch_size=training_cfg["batch_size"],
            epochs=training_cfg["epochs"],
            validation_data=(self.X_val, self.y_val),
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
            model_path = self.experiment_dir / "model.keras"
            self.model.save(model_path)
            print(f"✓ Modelo salvo: {model_path}")
        
        if output_cfg["save_history"] and self.history:
            history_path = self.experiment_dir / "history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f, indent=2)
            print(f"✓ Histórico salvo: {history_path}")
    
    def _calculate_validation_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas detalhadas no conjunto de validação.
        
        Returns:
            Dicionário com todas as métricas calculadas
        """
        if self.model is None or self.X_val is None or self.y_val is None:
            return {}
        
        # Fazer predições
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        
        # Para classificação binária ou multi-classe
        if y_pred_proba.shape[1] == 1:
            # Saída com um neurônio (sigmoid)
            y_pred = (y_pred_proba.flatten() > 0.5).astype(int)
            y_prob = y_pred_proba.flatten()
        else:
            # Saída com múltiplos neurônios (softmax)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_prob = np.max(y_pred_proba, axis=1)  # Confiança da predição
        
        y_true = self.y_val
        
        # Calcular métricas básicas
        metrics = {
            "val_accuracy": float(np.mean(y_pred == y_true)),
            "val_precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "val_recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "val_f1": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            "val_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
        
        # AUC ROC - funciona para binário e multi-classe (one-vs-rest)
        try:
            if len(np.unique(y_true)) == 2:
                # Binário
                metrics["val_auc_roc"] = float(roc_auc_score(y_true, y_prob))
                metrics["val_pr_auc"] = float(average_precision_score(y_true, y_prob))
            else:
                # Multi-classe
                metrics["val_auc_roc"] = float(roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='weighted'
                ))
                metrics["val_pr_auc"] = None  # PR-AUC é mais complexo para multi-classe
        except Exception as e:
            print(f"  ⚠️ Não foi possível calcular AUC: {e}")
            metrics["val_auc_roc"] = None
            metrics["val_pr_auc"] = None
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Para binário, extrair TP, FP, TN, FN
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["val_cm_tp"] = int(tp)
            metrics["val_cm_fp"] = int(fp)
            metrics["val_cm_tn"] = int(tn)
            metrics["val_cm_fn"] = int(fn)
            # Especificidade e Sensibilidade
            metrics["val_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
            metrics["val_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        else:
            # Multi-classe: salvar matriz em JSON no arquivo de resultado
            metrics["val_cm_string"] = str(cm.tolist())
        
        return metrics
    
    def run_full_pipeline(self,
                         limit_samples: int = None,
                         verbose: int = 1) -> Dict[str, Any]:
        """
        Executa pipeline completo: preparar dados → build → treinar → salvar → métricas.
        
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
        
        # 6. Calcular métricas detalhadas
        validation_metrics = self._calculate_validation_metrics()
        
        # 7. Resumo
        result = {
            "config_name": self.config_name,
            "experiment_dir": str(self.experiment_dir),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "final_train_loss": float(history_dict["loss"][-1]),
            "final_train_acc": float(history_dict["accuracy"][-1]),
            "final_val_loss": float(history_dict["val_loss"][-1]),
            "final_val_acc": float(history_dict["val_accuracy"][-1]),
            "epochs_run": len(history_dict["loss"]),
            **validation_metrics,  # Adicionar todas as métricas calculadas
        }
        
        # 8. Log em CSV
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
        """
        Salva resultado em CSV com append (rastreamento de experimentos).
        Salva automaticamente todas as métricas do dicionário result.
        """
        output_cfg = self.config["output"]
        csv_path = Path(output_cfg["models_dir"]) / "experiments_log.csv"
        
        # Preparar dados para CSV - usar todas as métricas do result
        row_data = {
            "timestamp": result["timestamp"],
            "config_name": result["config_name"],
            "train_loss": result.get("final_train_loss"),
            "train_acc": result.get("final_train_acc"),
            "val_loss": result.get("final_val_loss"),
            "val_acc": result.get("final_val_acc"),
            "epochs": result.get("epochs_run"),
            # Métricas adicionadas
            "val_accuracy": result.get("val_accuracy"),
            "val_precision": result.get("val_precision"),
            "val_recall": result.get("val_recall"),
            "val_f1": result.get("val_f1"),
            "val_balanced_accuracy": result.get("val_balanced_accuracy"),
            "val_auc_roc": result.get("val_auc_roc"),
            "val_pr_auc": result.get("val_pr_auc"),
            # Matriz de confusão (binário)
            "val_cm_tp": result.get("val_cm_tp"),
            "val_cm_fp": result.get("val_cm_fp"),
            "val_cm_tn": result.get("val_cm_tn"),
            "val_cm_fn": result.get("val_cm_fn"),
            # Especificidade e Sensibilidade
            "val_specificity": result.get("val_specificity"),
            "val_sensitivity": result.get("val_sensitivity"),
            # Matriz de confusão (multi-classe)
            "val_cm_string": result.get("val_cm_string"),
            "experiment_dir": result["experiment_dir"],
        }
        
        row = pd.DataFrame([row_data])
        
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
