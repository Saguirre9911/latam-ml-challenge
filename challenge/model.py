import os
from contextlib import contextmanager
from datetime import datetime
from typing import List, Tuple, Union

import pandas as pd
from sklearn.linear_model import LogisticRegression

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
    TOP_10_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
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
        base_required = {"OPERA", "TIPOVUELO", "MES"}
        missing = base_required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing columns for preprocessing: {sorted(missing)}")

        data_copy = data.copy()

        if target_column is not None:
            target_required = {"Fecha-I", "Fecha-O"}
            missing_target = target_required.difference(data.columns)
            if missing_target:
                raise ValueError(
                    f"Missing columns for target preprocessing: {sorted(missing_target)}"
                )
            data_copy["period_day"] = data_copy["Fecha-I"].apply(self._get_period_day)
            data_copy["high_season"] = data_copy["Fecha-I"].apply(self._is_high_season)
            data_copy["min_diff"] = data_copy.apply(self._get_min_diff, axis=1)
            data_copy["delay"] = (data_copy["min_diff"] > 15).astype(int)

        features = pd.concat(
            [
                pd.get_dummies(data_copy["OPERA"], prefix="OPERA"),
                pd.get_dummies(data_copy["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data_copy["MES"], prefix="MES"),
            ],
            axis=1,
        )
        for col in self.TOP_10_FEATURES:
            if col not in features.columns:
                features[col] = 0

        features = features[self.TOP_10_FEATURES]

        if target_column is not None:
            if target_column not in data_copy.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            target = data_copy[[target_column]]
            return features, target

        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        target_series = target.squeeze()
        n_y0 = (target_series == 0).sum()
        n_y1 = (target_series == 1).sum()

        class_weight = {
            1: n_y0 / len(target_series),
            0: n_y1 / len(target_series),
        }

        self._model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
        )
        self._model.fit(features, target_series)

        with _mlflow_run() as run:
            if run is not None:
                run.log_param("model_class", self._model.__class__.__name__)
                run.log_param("n_features", features.shape[1])
                run.log_param("n_rows", features.shape[0])
                run.log_param("class_weight_0", class_weight[0])
                run.log_param("class_weight_1", class_weight[1])
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0 for _ in range(len(features))]

        predictions = self._model.predict(features)
        return [int(pred) for pred in predictions]

    @staticmethod
    def _get_period_day(date: str) -> str:
        date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", "%H:%M").time()
        evening_max = datetime.strptime("23:59", "%H:%M").time()
        night_min = datetime.strptime("00:00", "%H:%M").time()
        night_max = datetime.strptime("4:59", "%H:%M").time()

        if morning_min < date_time < morning_max:
            return "mañana"
        if afternoon_min < date_time < afternoon_max:
            return "tarde"
        if (evening_min < date_time < evening_max) or (night_min < date_time < night_max):
            return "noche"
        return "noche"

    @staticmethod
    def _is_high_season(fecha: str) -> int:
        fecha_year = int(fecha.split("-")[0])
        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
        range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=fecha_year)
        range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=fecha_year)
        range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=fecha_year)
        range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=fecha_year)
        range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=fecha_year)
        range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=fecha_year)
        range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=fecha_year)
        range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=fecha_year)

        if (
            range1_min <= fecha_dt <= range1_max
            or range2_min <= fecha_dt <= range2_max
            or range3_min <= fecha_dt <= range3_max
            or range4_min <= fecha_dt <= range4_max
        ):
            return 1
        return 0

    @staticmethod
    def _get_min_diff(row: pd.Series) -> float:
        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        return (fecha_o - fecha_i).total_seconds() / 60
