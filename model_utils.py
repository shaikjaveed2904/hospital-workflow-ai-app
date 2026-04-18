"""
model_utils.py
==============

This module contains helper functions for training and using predictive models
within the hospital workflow optimization app.  Four separate models are
provided:

1. **Patient Flow Prediction**: Estimates the number of arriving patients by
   day-of-week and hour-of-day using linear regression.  Useful for forecasting
   demand and identifying peak periods.
2. **Wait Time Prediction**: Predicts how long a patient will wait based on
   appointment time, department, doctor and patient priority using a random
   forest regressor.
3. **Doctor Workload Forecasting**: Predicts how many appointments a doctor
   will have at a given hour using linear regression on aggregated workload
   data.
4. **No‑Show Prediction**: Predicts the probability that a patient will miss
   their appointment using logistic regression.

Each training function returns a fitted scikit‑learn pipeline that includes
categorical preprocessing.  The prediction functions accept the fitted model
and relevant features to output a prediction.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor


def _extract_time_features(df: pd.DataFrame,
                           date_col: str = "appointment_date",
                           time_col: str = "appointment_time") -> pd.DataFrame:
    """Internal helper to derive time related features (hour, day of week, month).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing appointment date and time columns.
    date_col : str
        Name of the column containing dates in YYYY-MM-DD format.
    time_col : str
        Name of the column containing times in HH:MM format.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with 'hour', 'day_of_week', and 'month'
        columns added.
    """
    df_copy = df.copy()
    # Combine date and time into a datetime for easier feature extraction
    dt = pd.to_datetime(df_copy[date_col] + " " + df_copy[time_col])
    df_copy["hour"] = dt.dt.hour
    df_copy["day_of_week"] = dt.dt.dayofweek  # Monday=0
    df_copy["month"] = dt.dt.month
    return df_copy


def train_flow_model(df: pd.DataFrame) -> Pipeline:
    """Train a model to forecast patient arrivals by hour and day.

    The model uses linear regression on aggregated counts of appointments per
    day-of-week and hour-of-day.  Only date/time features are used.

    Parameters
    ----------
    df : pd.DataFrame
        The appointment data; must contain 'appointment_date' and 'appointment_time'.

    Returns
    -------
    Pipeline
        A fitted scikit-learn pipeline that predicts counts of arrivals.
    """
    # Extract time features
    df_time = _extract_time_features(df)

    # Aggregate counts by day-of-week and hour
    agg = df_time.groupby(["day_of_week", "hour"]).size().reset_index(name="count")

    X = agg[["day_of_week", "hour"]]
    y = agg["count"]

    # Use one-hot encoding for the day of week; hour remains numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("day", OneHotEncoder(handle_unknown="ignore"), ["day_of_week"]),
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    model.fit(X, y)
    return model


def predict_flow(model: Pipeline, day_of_week: int, hour: int) -> float:
    """Predict patient arrival count given day-of-week and hour.

    Parameters
    ----------
    model : Pipeline
        Fitted flow prediction model.
    day_of_week : int
        Day of the week (0=Monday, 6=Sunday).
    hour : int
        Hour of day (0-23).

    Returns
    -------
    float
        Predicted number of arrivals for the given slot.
    """
    X_input = pd.DataFrame({"day_of_week": [day_of_week], "hour": [hour]})
    return float(model.predict(X_input)[0])


def train_wait_time_model(df: pd.DataFrame) -> Pipeline:
    """Train a model to predict patient wait times.

    The model uses appointment time, department, doctor name and patient priority
    to estimate wait time via a random forest regressor.

    Parameters
    ----------
    df : pd.DataFrame
        The appointment data; must include columns: 'appointment_date',
        'appointment_time', 'department', 'doctor_name', 'patient_priority',
        and 'wait_time_minutes'.

    Returns
    -------
    Pipeline
        A fitted scikit-learn pipeline that predicts wait times.
    """
    df_feat = _extract_time_features(df)

    X = df_feat[["hour", "day_of_week", "department", "doctor_name", "patient_priority"]]
    y = df_feat["wait_time_minutes"]

    categorical_features = ["department", "doctor_name", "patient_priority"]
    numeric_features = ["hour", "day_of_week"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return model


def predict_wait_time(model: Pipeline,
                      hour: int,
                      day_of_week: int,
                      department: str,
                      doctor_name: str,
                      patient_priority: str) -> float:
    """Predict wait time for a new appointment slot.

    Parameters
    ----------
    model : Pipeline
        Fitted wait time model.
    hour : int
        Appointment hour (0-23).
    day_of_week : int
        Day of the week (0=Monday).
    department : str
        Department name.
    doctor_name : str
        Doctor name.
    patient_priority : str
        Patient priority (e.g., 'normal' or 'urgent').

    Returns
    -------
    float
        Predicted wait time in minutes.
    """
    X_input = pd.DataFrame({
        "hour": [hour],
        "day_of_week": [day_of_week],
        "department": [department],
        "doctor_name": [doctor_name],
        "patient_priority": [patient_priority],
    })
    return float(model.predict(X_input)[0])


def train_workload_model(df: pd.DataFrame) -> Pipeline:
    """Train a model to forecast doctor workload (number of appointments).

    The model uses day-of-week, hour and doctor name to predict how many
    appointments a doctor will have during a specific time slot.  A linear
    regression is used on aggregated workloads.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data with columns 'doctor_name', 'appointment_date',
        'appointment_time'.

    Returns
    -------
    Pipeline
        A fitted scikit-learn pipeline for workload forecasting.
    """
    df_time = _extract_time_features(df)
    # Aggregate counts by doctor, day-of-week and hour
    agg = df_time.groupby(["doctor_name", "day_of_week", "hour"]).size().reset_index(name="count")
    X = agg[["doctor_name", "day_of_week", "hour"]]
    y = agg["count"]

    categorical_features = ["doctor_name"]
    numeric_features = ["day_of_week", "hour"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    model.fit(X, y)
    return model


def predict_workload(model: Pipeline,
                     doctor_name: str,
                     day_of_week: int,
                     hour: int) -> float:
    """Predict doctor workload for a particular time slot.

    Parameters
    ----------
    model : Pipeline
        Fitted workload model.
    doctor_name : str
        Name of the doctor.
    day_of_week : int
        Day of the week (0=Monday).
    hour : int
        Hour of day.

    Returns
    -------
    float
        Predicted number of appointments for the doctor in that slot.
    """
    X_input = pd.DataFrame({
        "doctor_name": [doctor_name],
        "day_of_week": [day_of_week],
        "hour": [hour],
    })
    return float(model.predict(X_input)[0])


def train_no_show_model(df: pd.DataFrame) -> Pipeline:
    """Train a classifier to predict no-shows.

    This function uses logistic regression to estimate the probability that
    a patient will not show up.  Features include time of day, day of week,
    department, doctor and patient priority.  Only completed or scheduled
    appointments are considered.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data with 'no_show' (0/1) target column and relevant
        feature columns.

    Returns
    -------
    Pipeline
        A fitted scikit-learn pipeline for no-show prediction.
    """
    df_feat = _extract_time_features(df)
    X = df_feat[["hour", "day_of_week", "department", "doctor_name", "patient_priority"]]
    y = df_feat["no_show"]

    categorical_features = ["department", "doctor_name", "patient_priority"]
    numeric_features = ["hour", "day_of_week"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=200)),
    ])
    model.fit(X, y)
    return model


def predict_no_show_probability(model: Pipeline,
                                hour: int,
                                day_of_week: int,
                                department: str,
                                doctor_name: str,
                                patient_priority: str) -> float:
    """Predict the probability that a patient will not show up.

    Parameters
    ----------
    model : Pipeline
        Fitted no-show classification model.
    hour : int
        Appointment hour (0-23).
    day_of_week : int
        Day of the week (0=Monday).
    department : str
        Department name.
    doctor_name : str
        Doctor name.
    patient_priority : str
        Patient priority.

    Returns
    -------
    float
        Predicted probability of no-show (between 0 and 1).
    """
    X_input = pd.DataFrame({
        "hour": [hour],
        "day_of_week": [day_of_week],
        "department": [department],
        "doctor_name": [doctor_name],
        "patient_priority": [patient_priority],
    })
    proba = model.predict_proba(X_input)[0]
    # Probability of class 1 (no-show)
    return float(proba[1])