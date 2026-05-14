"""Model factories — extracted verbatim from the original
``run_comprehensive_comparison.py``. Hyperparameters are pinned to the
values that produced the published AUCs (seed=42).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import RND


def get_lr_pipe():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight=None, random_state=RND),
    )


def get_rf_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RND,
        class_weight="balanced",
        n_jobs=-1,
    )


def get_xgb_model():
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RND,
        eval_metric="logloss",
    )


def get_mlp_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=RND,
        early_stopping=True,
    )


def get_pipeline(model_type: str):
    if model_type == "Neural Network":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), get_mlp_model())
    if model_type == "Stepwise Logistic Regression":
        return None
    if model_type == "Random Forest":
        return make_pipeline(SimpleImputer(strategy="median"), get_rf_model())
    if model_type == "XGBoost":
        return make_pipeline(SimpleImputer(strategy="median"), get_xgb_model())
    raise ValueError(f"Unknown model: {model_type}")
