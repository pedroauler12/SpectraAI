import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

"""
Uso no notebook :
após separação dos dados de treino e teste:

best_model, gs_results = run_rf_gridsearch(X_train, y_train)
gs_results
"""

def rf_gridsearch(X_train, y_train, seed=42):

    start_time = time.time()

    rf = RandomForestClassifier(
        random_state=seed,
        n_jobs=1
    )

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring={
        "accuracia": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"},
        refit="roc_auc",
        n_jobs=1,
        verbose=2,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    elapsed_time = time.time() - start_time

    results = {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "total_candidates": len(grid.cv_results_["params"]),
        "execution_time_seconds": round(elapsed_time, 2)
    }

    with open("rf_gridsearch_heavy_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return grid.best_estimator_, results
