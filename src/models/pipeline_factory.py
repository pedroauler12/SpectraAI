from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath("src"))

from utils.metrics import classification_metrics, regression_metrics

def create_classifier_pipeline(model_name: str, **kwargs):
    """
    Pipeline para modelos de classificação
    
    :param model_name: Selecione o modelo svm, random_forest ou logisticregression
    :type model_name: str
    """
    if model_name == "svm":
        model = SVC(**kwargs)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**kwargs)
    elif model_name == "logisticregression":
        model = LogisticRegression(**kwargs)
    else:
        raise ValueError("Modelo de classificação não suportado")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    return pipeline


def train_classification(X, y, model_name="random_forest", **model_params):
    """ Use para treinar os modelos de classificação, selecionado previamente"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = create_classifier_pipeline(model_name, **model_params)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = classification_metrics(y_test, y_pred, y_prob)

    return pipeline, metrics



## apenas para o caso de decidirmos usar regressão (no futuro)##


def create_regressor_pipeline(model_name: str):
    if model_name == "svm":
        model = SVR()
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_name == "linear":
        model = LinearRegression()
    else:
        raise ValueError("Modelo de regressão não suportado")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    return pipeline


def train_regression(X, y, model_name="random_forest"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = create_regressor_pipeline(model_name)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    return pipeline, metrics

