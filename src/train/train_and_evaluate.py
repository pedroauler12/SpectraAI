"""
Funções utilitárias para treinamento e avaliação de modelos.

Este módulo contém funções genéricas para treinar e avaliar modelos de machine learning,
com logging padronizado de métricas e retorno estruturado.
"""

import logging
import time
from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str = 'classification',
    average: str = 'binary',
    model_name: str = None
) -> Dict[str, Any]:
    """
    Função genérica para treinar e avaliar modelos de machine learning.
    
    Esta função treina um modelo nos dados de treino, faz predições no conjunto
    de validação e calcula métricas relevantes de acordo com o tipo de tarefa.
    Inclui logging detalhado e retorno padronizado de resultados.
    
    Parameters
    ----------
    model : Any
        Modelo de machine learning com métodos fit() e predict().
        Para classificação probabilística, também deve ter predict_proba().
    X_train : np.ndarray
        Features do conjunto de treino.
    y_train : np.ndarray
        Labels/targets do conjunto de treino.
    X_val : np.ndarray
        Features do conjunto de validação.
    y_val : np.ndarray
        Labels/targets do conjunto de validação.
    task_type : str, optional
        Tipo de tarefa: 'classification' ou 'regression' (default: 'classification').
    average : str, optional
        Tipo de média para métricas de classificação multiclasse.
        Opções: 'binary', 'micro', 'macro', 'weighted' (default: 'binary').
    model_name : str, optional
        Nome do modelo para logging. Se None, usa o nome da classe.
    
    Returns
    -------
    Dict[str, Any]
        Dicionário contendo:
        - 'model': modelo treinado
        - 'predictions': predições no conjunto de validação
        - 'metrics': dicionário com métricas calculadas
        - 'training_time': tempo de treinamento em segundos
        - 'inference_time': tempo de inferência em segundos
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(n_estimators=100)
    >>> results = train_and_evaluate(model, X_train, y_train, X_val, y_val)
    >>> print(f"Acurácia: {results['metrics']['accuracy']:.4f}")
    
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression()
    >>> results = train_and_evaluate(
    ...     model, X_train, y_train, X_val, y_val, 
    ...     task_type='regression'
    ... )
    >>> print(f"R²: {results['metrics']['r2']:.4f}")
    """
    
    # Determina nome do modelo
    if model_name is None:
        model_name = model.__class__.__name__
    
    logger.info(f"{'='*60}")
    logger.info(f"Iniciando treinamento: {model_name}")
    logger.info(f"Tipo de tarefa: {task_type}")
    logger.info(f"Tamanho do treino: {X_train.shape}")
    logger.info(f"Tamanho da validação: {X_val.shape}")
    logger.info(f"{'='*60}")
    
    # Treinamento
    logger.info("Treinando modelo...")
    start_train = time.time()
    
    try:
        model.fit(X_train, y_train)
        training_time = time.time() - start_train
        logger.info(f"✓ Treinamento concluído em {training_time:.2f}s")
    except Exception as e:
        logger.error(f"✗ Erro no treinamento: {str(e)}")
        raise
    
    # Inferência
    logger.info("Realizando predições no conjunto de validação...")
    start_inference = time.time()
    
    try:
        y_pred = model.predict(X_val)
        inference_time = time.time() - start_inference
        logger.info(f"✓ Inferência concluída em {inference_time:.2f}s")
    except Exception as e:
        logger.error(f"✗ Erro na inferência: {str(e)}")
        raise
    
    # Cálculo de métricas
    logger.info("Calculando métricas...")
    metrics = {}
    
    try:
        if task_type == 'classification':
            metrics = _calculate_classification_metrics(
                y_val, y_pred, model, X_val, average
            )
        elif task_type == 'regression':
            metrics = _calculate_regression_metrics(y_val, y_pred)
        else:
            raise ValueError(f"task_type '{task_type}' não reconhecido. Use 'classification' ou 'regression'.")
        
        logger.info(f"✓ Métricas calculadas")
    except Exception as e:
        logger.error(f"✗ Erro ao calcular métricas: {str(e)}")
        raise
    
    # Log das métricas
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTADOS - {model_name}")
    logger.info(f"{'='*60}")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"{metric_name:.<30} {metric_value:.6f}")
        else:
            logger.info(f"{metric_name:.<30} {metric_value}")
    logger.info(f"Tempo de treinamento:............ {training_time:.2f}s")
    logger.info(f"Tempo de inferência:............. {inference_time:.2f}s")
    logger.info(f"{'='*60}\n")
    
    # Retorno padronizado
    results = {
        'model': model,
        'predictions': y_pred,
        'metrics': metrics,
        'training_time': training_time,
        'inference_time': inference_time
    }
    
    return results


def _calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: Any,
    X_val: np.ndarray,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Calcula métricas para problemas de classificação.
    
    Parameters
    ----------
    y_true : np.ndarray
        Labels verdadeiros.
    y_pred : np.ndarray
        Predições do modelo.
    model : Any
        Modelo treinado (usado para calcular AUC-ROC se houver predict_proba).
    X_val : np.ndarray
        Features de validação.
    average : str
        Tipo de média para métricas multiclasse.
    
    Returns
    -------
    Dict[str, float]
        Dicionário com métricas calculadas.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Tenta calcular AUC-ROC se o modelo tiver predict_proba
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_val)
            
            # Para classificação binária
            if y_proba.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            # Para classificação multiclasse
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, 
                    multi_class='ovr', 
                    average=average
                )
        except Exception as e:
            logger.warning(f"Não foi possível calcular ROC-AUC: {str(e)}")
    
    return metrics


def _calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas para problemas de regressão.
    
    Parameters
    ----------
    y_true : np.ndarray
        Valores verdadeiros.
    y_pred : np.ndarray
        Predições do modelo.
    
    Returns
    -------
    Dict[str, float]
        Dicionário com métricas calculadas.
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics


if __name__ == "__main__":
    """
    Bloco de teste da função train_and_evaluate.
    
    Este exemplo demonstra o uso da função com dados sintéticos
    para classificação e regressão.
    """
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    
    print("\n" + "="*70)
    print("TESTE 1: CLASSIFICAÇÃO BINÁRIA")
    print("="*70 + "\n")
    
    # Gera dados sintéticos para classificação
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Divide em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treina e avalia Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    results_clf = train_and_evaluate(
        model=rf_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type='classification',
        model_name='Random Forest Classifier'
    )
    
    print("\n" + "="*70)
    print("TESTE 2: REGRESSÃO")
    print("="*70 + "\n")
    
    # Gera dados sintéticos para regressão
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    # Divide em treino e validação
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Treina e avalia Regressão Linear
    lr_model = LinearRegression()
    results_reg = train_and_evaluate(
        model=lr_model,
        X_train=X_train_reg,
        y_train=y_train_reg,
        X_val=X_val_reg,
        y_val=y_val_reg,
        task_type='regression',
        model_name='Linear Regression'
    )
