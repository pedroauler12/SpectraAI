"""
Pipeline para executar experimentos de Transfer Learning com MobileNetV2.
Gerencia treinamento em duas fases (head + fine-tuning), logging e
salvamento de resultados no experiments_log.csv.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .cnn_builder import build_transfer_model, unfreeze_backbone_layers
from .cnn_config import load_config, save_experiment_config
from .cnn_data_prep import prepare_grouped_cnn_splits
from .cnn_tf_data_pipeline import build_train_val_test_tf_data


SEED = 42


class TransferLearningExperimentRunner:
    """Executa e gerencia experimentos de transfer learning com MobileNetV2."""

    def __init__(self, config_name: str = "tl_baseline"):
        self.config_name = config_name
        self.config = load_config(config_name)
        self.model = None
        self.experiment_dir = None
        self.training_time = None

        print(f"TransferLearningExperimentRunner inicializado: {config_name}")

    def prepare_data(self, limit_samples: int | None = None) -> dict:
        """
        Carrega e prepara dados com split treino/val/teste agrupado por image_id.

        Returns:
            Dicionario com X_train, y_train, X_val, y_val, X_test, y_test, shape_info
        """
        data_cfg = self.config["data"]

        print("\nCarregando dados...")
        df = pd.read_csv(data_cfg["dataset_path"])
        if limit_samples:
            df = df.sample(n=min(limit_samples, len(df)), random_state=SEED)

        split_data = prepare_grouped_cnn_splits(
            df,
            extracted_codes_path=data_cfg["codes_path"],
            test_size=data_cfg.get("test_size", 0.2),
            val_size=data_cfg.get("val_size", 0.2),
            seed=SEED,
        )

        n_train = len(split_data["y_train"])
        n_val = len(split_data["y_val"])
        n_test = len(split_data["y_test"])
        print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test}")

        return split_data

    def build_tf_data(self, split_data: dict) -> dict:
        """Constroi pipelines tf.data com normalizacao e augmentation."""
        model_cfg = self.config["model"]
        training_cfg = self.config["training"]
        data_cfg = self.config["data"]

        resize_to = tuple(model_cfg.get("resize_to", [160, 160]))

        tf_data = build_train_val_test_tf_data(
            split_data["X_train"],
            split_data["y_train"],
            split_data["X_val"],
            split_data["y_val"],
            split_data["X_test"],
            split_data["y_test"],
            batch_size=training_cfg["batch_size"],
            normalization=data_cfg.get("normalization_method", "zscore"),
            resize_to=resize_to,
            data_format="channels_last",
            target_channels=split_data["shape_info"]["n_channels"],
            augment_train=True,
            seed=SEED,
        )
        return tf_data

    def build_model(self, input_shape: tuple) -> keras.Model:
        """Constroi modelo de transfer learning (backbone congelado)."""
        model_cfg = self.config["model"]
        training_cfg = self.config["training"]

        print("\nConstruindo modelo Transfer Learning...")
        self.model, model_info = build_transfer_model(
            input_shape=input_shape,
            learning_rate=training_cfg["head_learning_rate"],
            dropout_rate=model_cfg.get("dropout_rate", 0.25),
        )
        print(f"  Backbone: {model_info['backbone']}")
        print(f"  Dropout: {model_cfg.get('dropout_rate', 0.25)}")
        return self.model

    def create_experiment_dir(self) -> Path:
        """Cria diretorio para salvar resultados do experimento."""
        output_cfg = self.config["output"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exp_dir = Path(output_cfg["models_dir"]) / f"{self.config_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = exp_dir

        save_experiment_config(self.config, exp_dir)
        print(f"\nExperimento: {exp_dir.name}")
        return exp_dir

    def train_two_phases(self, tf_data: dict, verbose: int = 1) -> dict:
        """
        Treina modelo em duas fases: head training + fine-tuning.

        Returns:
            Dicionario com historicos consolidados e metadados de treinamento.
        """
        training_cfg = self.config["training"]
        model_cfg = self.config["model"]

        # --- Fase 1: Head Training (backbone congelado) ---
        print("\n--- Fase 1: Head Training ---")
        print(f"  LR: {training_cfg['head_learning_rate']}")
        print(f"  Epochs: {training_cfg['head_epochs']}")

        cb_phase1 = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=training_cfg.get("early_stopping_patience_head", 3),
                restore_best_weights=True,
                verbose=int(verbose > 0),
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=int(verbose > 0),
            ),
        ]

        start_time = time.time()

        hist_p1 = self.model.fit(
            tf_data["train_ds"],
            validation_data=tf_data["val_ds"],
            epochs=training_cfg["head_epochs"],
            callbacks=cb_phase1,
            verbose=verbose,
        )
        p1_epochs = len(hist_p1.history["loss"])

        # --- Fase 2: Fine-Tuning ---
        print(f"\n--- Fase 2: Fine-Tuning ---")
        print(f"  LR: {training_cfg['fine_tune_learning_rate']}")
        print(f"  Epochs: {training_cfg['fine_tune_epochs']}")
        print(f"  Camadas descongeladas: {model_cfg.get('fine_tune_last_layers', 20)}")

        unfreeze_info = unfreeze_backbone_layers(
            self.model,
            fine_tune_last_layers=model_cfg.get("fine_tune_last_layers", 20),
            learning_rate=training_cfg["fine_tune_learning_rate"],
        )

        cb_phase2 = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=training_cfg.get("early_stopping_patience_ft", 4),
                restore_best_weights=True,
                verbose=int(verbose > 0),
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=int(verbose > 0),
            ),
        ]

        hist_p2 = self.model.fit(
            tf_data["train_ds"],
            validation_data=tf_data["val_ds"],
            epochs=p1_epochs + training_cfg["fine_tune_epochs"],
            initial_epoch=p1_epochs,
            callbacks=cb_phase2,
            verbose=verbose,
        )
        p2_epochs = len(hist_p2.history["loss"])

        self.training_time = time.time() - start_time

        # Consolidar historicos
        full_history = {}
        for key in hist_p1.history:
            full_history[key] = hist_p1.history[key] + hist_p2.history.get(key, [])

        print(f"\n  Head: {p1_epochs} ep | FT: {p2_epochs} ep | Total: {p1_epochs + p2_epochs} ep")
        print(f"  Tempo: {self.training_time:.1f}s")

        return {
            "full_history": full_history,
            "head_epochs": p1_epochs,
            "ft_epochs": p2_epochs,
            "total_epochs": p1_epochs + p2_epochs,
            "unfreeze_info": unfreeze_info,
        }

    def evaluate_on_test(self, tf_data: dict) -> dict:
        """Avalia modelo no conjunto de teste e calcula metricas."""
        print("\nAvaliando no conjunto de teste...")

        y_prob = self.model.predict(tf_data["test_ds"], verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = np.concatenate([y.numpy() for _, y in tf_data["test_ds"]])

        metrics = {
            "val_accuracy": float(accuracy_score(y_true, y_pred)),
            "val_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "val_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "val_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "val_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

        try:
            metrics["val_auc_roc"] = float(roc_auc_score(y_true, y_prob))
            metrics["val_pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception as e:
            print(f"  Nao foi possivel calcular AUC: {e}")
            metrics["val_auc_roc"] = None
            metrics["val_pr_auc"] = None

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["val_cm_tp"] = int(tp)
            metrics["val_cm_fp"] = int(fp)
            metrics["val_cm_tn"] = int(tn)
            metrics["val_cm_fn"] = int(fn)
            metrics["val_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["val_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        print(f"  Acc: {metrics['val_accuracy']:.4f}  F1: {metrics['val_f1']:.4f}  "
              f"AUC: {metrics.get('val_auc_roc', 'N/A')}")

        return metrics

    def _log_result_to_csv(self, result: dict) -> None:
        """Salva resultado em experiments_log.csv com append."""
        output_cfg = self.config["output"]
        csv_path = Path(output_cfg["models_dir"]) / "experiments_log.csv"

        row_data = {
            "timestamp": result["timestamp"],
            "config_name": result["config_name"],
            "train_loss": result.get("final_train_loss"),
            "train_acc": result.get("final_train_acc"),
            "val_loss": result.get("final_val_loss"),
            "val_acc": result.get("final_val_acc"),
            "epochs": result.get("total_epochs"),
            "val_accuracy": result.get("val_accuracy"),
            "val_precision": result.get("val_precision"),
            "val_recall": result.get("val_recall"),
            "val_f1": result.get("val_f1"),
            "val_balanced_accuracy": result.get("val_balanced_accuracy"),
            "val_auc_roc": result.get("val_auc_roc"),
            "val_pr_auc": result.get("val_pr_auc"),
            "val_cm_tp": result.get("val_cm_tp"),
            "val_cm_fp": result.get("val_cm_fp"),
            "val_cm_tn": result.get("val_cm_tn"),
            "val_cm_fn": result.get("val_cm_fn"),
            "val_specificity": result.get("val_specificity"),
            "val_sensitivity": result.get("val_sensitivity"),
            "experiment_dir": result.get("experiment_dir"),
            "training_time_seconds": result.get("training_time_seconds"),
            "augmentation_enabled": True,
        }

        row = pd.DataFrame([row_data])

        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            df_updated = pd.concat([df_existing, row], ignore_index=True)
            df_updated.to_csv(csv_path, index=False)
            print(f"Resultado adicionado ao log: {csv_path}")
        else:
            row.to_csv(csv_path, index=False)
            print(f"Novo log de experimentos criado: {csv_path}")

    def run_full_pipeline(
        self,
        limit_samples: int | None = None,
        verbose: int = 1,
    ) -> dict:
        """
        Executa pipeline completo de transfer learning:
        preparar dados -> build -> treinar (2 fases) -> avaliar -> log CSV.

        Args:
            limit_samples: Limita amostras (para testes rapidos)
            verbose: Verbosidade do treinamento

        Returns:
            Dicionario com resultado do experimento
        """
        print(f"\n{'='*60}")
        print(f"EXPERIMENTO TL: {self.config_name}")
        print(f"{'='*60}")

        # 1. Preparar dados
        split_data = self.prepare_data(limit_samples)

        # 2. Construir pipelines tf.data
        tf_data = self.build_tf_data(split_data)

        # 3. Build modelo
        self.build_model(input_shape=tf_data["train_meta"]["input_shape"])

        # 4. Criar diretorio
        self.create_experiment_dir()

        # 5. Treinar em duas fases
        train_result = self.train_two_phases(tf_data, verbose=verbose)
        full_history = train_result["full_history"]

        # 6. Avaliar no teste
        test_metrics = self.evaluate_on_test(tf_data)

        # 7. Salvar modelo (se configurado)
        output_cfg = self.config["output"]
        if output_cfg.get("save_model", False) and self.experiment_dir:
            model_path = self.experiment_dir / "best_model.keras"
            self.model.save(model_path)
            print(f"Modelo salvo: {model_path}")

        if output_cfg.get("save_history", False) and self.experiment_dir:
            history_path = self.experiment_dir / "history.json"
            serializable = {
                k: [float(v) for v in vals]
                for k, vals in full_history.items()
            }
            with open(history_path, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"Historico salvo: {history_path}")

        # 8. Montar resultado
        result = {
            "config_name": self.config_name,
            "experiment_dir": str(self.experiment_dir),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "final_train_loss": float(full_history["loss"][-1]),
            "final_train_acc": float(full_history["accuracy"][-1]),
            "final_val_loss": float(full_history["val_loss"][-1]),
            "final_val_acc": float(full_history["val_accuracy"][-1]),
            "head_epochs": train_result["head_epochs"],
            "ft_epochs": train_result["ft_epochs"],
            "total_epochs": train_result["total_epochs"],
            "training_time_seconds": self.training_time,
            **test_metrics,
        }

        # 9. Log em CSV
        self._log_result_to_csv(result)

        # 10. Resumo
        print(f"\n{'='*60}")
        print("RESUMO:")
        print(f"{'='*60}")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Limpar sessao para liberar memoria
        tf.keras.backend.clear_session()

        return result
