import os
from contextlib import contextmanager
from typing import List, Tuple, Union

import pandas as pd

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


@contextmanager
def _mlflow_run():
    enabled = os.getenv("ENABLE_MLFLOW", "0") == "1"
    if not enabled or mlflow is None:
        yield None
        return
    try:
        with mlflow.start_run():
            yield mlflow
    except Exception:
        # Never block training if MLflow is misconfigured.
        yield None


class DelayModel:
    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union(
        Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame
    ):
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        return

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        with _mlflow_run() as run:
            if run is not None and self._model is not None:
                run.log_param("model_class", self._model.__class__.__name__)
                run.log_param("n_features", features.shape[1])
                run.log_param("n_rows", features.shape[0])
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return
