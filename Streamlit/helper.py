"""
All the non-UI logic for the Heart Disease Risk app lives here:
- feature metadata (used to build the input form in app.py)
- building a model-ready input DataFrame from raw form values
- loading the trained pipelines (.pkl files, produced by the notebooks)
- running predictions for either the Binary or Multiclass model
- static metrics/reports captured from the notebooks, used on the
  "Model Performance" page

Keeping this separate from app.py means the ML/data logic can be tested,
reused, or swapped out without touching any Streamlit code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent

MODEL_PATHS = {
    "binary": MODELS_DIR / "final_tuned_2_logistic_regression.pkl",
    "multiclass": MODELS_DIR / "final_tuned_multi_class_logistic_regression.pkl",
}

# --------------------------------------------------------------------------
# Feature schema
# --------------------------------------------------------------------------
# Order matters: it must match the column order the pipelines were trained on
# (numerical_cols + categorical_cols, exactly as in the notebooks).

NUMERICAL_COLS = [
    "age",
    "resting_blood_pressure",
    "cholesterol_level",
    "maximum_heart_rate_achieved",
    "st_depression_during_exercise",
]

CATEGORICAL_COLS = [
    "sex",
    "chest_pain_type",
    "fasting_blood_sugar",
    "resting_ecg_result",
    "exercise_induced_chest_pain",
    "slope_of_exercise_ecg",
    "number_of_major_vessels",
    "thalassemia_test_result",
]

FEATURE_ORDER = NUMERICAL_COLS + CATEGORICAL_COLS


@dataclass
class NumericField:
    label: str
    help: str
    min_value: float
    max_value: float
    default: float
    step: float = 1.0
    normal_range: str = ""  # shown as a caption under the field, e.g. "Typical: 90-120"


@dataclass
class CategoricalField:
    label: str
    help: str
    options: dict  # {raw_value: display_label}
    default: Any


NUMERIC_FIELDS: dict[str, NumericField] = {
    "age": NumericField(
        label="Age",
        help="Patient age in years.",
        min_value=1, max_value=120, default=54, step=1,
        normal_range="Dataset range: 29-77 years",
    ),
    "resting_blood_pressure": NumericField(
        label="Resting Blood Pressure (mm Hg)",
        help="Blood pressure measured on admission to hospital.",
        min_value=60, max_value=260, default=130, step=1,
        normal_range="Normal: ~90-120 · Elevated: 120-129 · High: 130+",
    ),
    "cholesterol_level": NumericField(
        label="Serum Cholesterol (mg/dl)",
        help="Cholesterol level from blood test.",
        min_value=80, max_value=700, default=246, step=1,
        normal_range="Desirable: <200 · Borderline: 200-239 · High: 240+",
    ),
    "maximum_heart_rate_achieved": NumericField(
        label="Maximum Heart Rate Achieved",
        help="Highest heart rate reached during a stress/exercise test.",
        min_value=60, max_value=250, default=150, step=1,
        normal_range="Rough guide: ~220 minus age, at peak effort",
    ),
    "st_depression_during_exercise": NumericField(
        label="ST Depression (Exercise vs. Rest)",
        help="ST depression induced by exercise relative to rest (oldpeak).",
        min_value=0.0, max_value=7.0, default=1.0, step=0.1,
        normal_range="0 = none · higher values suggest more ischemia",
    ),
}

CATEGORICAL_FIELDS: dict[str, CategoricalField] = {
    "sex": CategoricalField(
        label="Sex",
        help="Biological sex of the patient.",
        options={1: "Male", 0: "Female"},
        default=1,
    ),
    "chest_pain_type": CategoricalField(
        label="Chest Pain Type",
        help="Type of chest pain experienced.",
        options={
            1: "Typical angina",
            2: "Atypical angina",
            3: "Non-anginal pain",
            4: "Asymptomatic",
        },
        default=4,
    ),
    "fasting_blood_sugar": CategoricalField(
        label="Fasting Blood Sugar > 120 mg/dl",
        help="Whether fasting blood sugar exceeds 120 mg/dl.",
        options={1: "Yes", 0: "No"},
        default=0,
    ),
    "resting_ecg_result": CategoricalField(
        label="Resting ECG Result",
        help="Resting electrocardiographic result.",
        options={
            0: "Normal",
            1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy",
        },
        default=0,
    ),
    "exercise_induced_chest_pain": CategoricalField(
        label="Exercise-Induced Angina",
        help="Whether exercise induced chest pain (angina).",
        options={1: "Yes", 0: "No"},
        default=0,
    ),
    "slope_of_exercise_ecg": CategoricalField(
        label="Slope of Peak Exercise ST Segment",
        help="Slope of the ST segment during peak exercise.",
        options={1: "Upsloping", 2: "Flat", 3: "Downsloping"},
        default=1,
    ),
    "number_of_major_vessels": CategoricalField(
        label="Major Vessels Colored by Fluoroscopy",
        help="Number of major vessels (0-3) visible via fluoroscopy.",
        options={0: "0", 1: "1", 2: "2", 3: "3"},
        default=0,
    ),
    "thalassemia_test_result": CategoricalField(
        label="Thalassemia Test Result",
        help="Result of the thalassemia blood disorder test.",
        options={3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
        default=3,
    ),
}


def default_form_values() -> dict:
    """Sensible pre-filled defaults for the input form."""
    values = {name: field.default for name, field in NUMERIC_FIELDS.items()}
    values.update({name: field.default for name, field in CATEGORICAL_FIELDS.items()})
    return values


def build_input_dataframe(form_values: dict) -> pd.DataFrame:
    """Turn raw form values into a single-row DataFrame the pipeline expects."""
    row = {col: [form_values[col]] for col in FEATURE_ORDER}
    return pd.DataFrame(row, columns=FEATURE_ORDER)


# --------------------------------------------------------------------------
# Model loading & prediction
# --------------------------------------------------------------------------

BINARY_LABELS = {0: "No Heart Disease", 1: "Heart Disease Present"}
SEVERITY_LABELS = {
    0: "No Disease",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Very Severe",
}


class ModelNotFoundError(FileNotFoundError):
    pass


def list_model_files() -> list[str]:
    """Every file currently sitting in the models/ folder (for debugging)."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_file())


def model_status() -> dict:
    """Connected/not-connected status for exactly the two models the app uses.

    Returns e.g. {"binary": {"label": "Binary model", "connected": True, "path": ...}, ...}
    """
    labels = {"binary": "Binary model", "multiclass": "Multiclass model"}
    return {
        key: {"label": labels[key], "connected": path.exists(), "path": path}
        for key, path in MODEL_PATHS.items()
    }


def load_model(model_type: str):
    """Load a trained pipeline (.pkl) from disk.

    model_type: "binary" or "multiclass"
    """
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Unknown model_type '{model_type}'")

    path = MODEL_PATHS[model_type]
    if not path.exists():
        found = list_model_files()
        found_msg = (
            f"Files currently in '{MODELS_DIR}/': {found}"
            if found
            else f"The folder '{MODELS_DIR}/' is empty (or does not exist)."
        )
        raise ModelNotFoundError(
            f"Could not find '{path.name}' at {path}. {found_msg} "
            f"Double-check the exact filename (including .pkl) and that it's "
            f"inside the models/ subfolder next to app.py."
        )
    return joblib.load(path)


def predict(model_type: str, model, form_values: dict) -> dict:
    """Run a prediction and return a small, UI-friendly result dict."""
    input_df = build_input_dataframe(form_values)

    pred = model.predict(input_df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]

    if model_type == "binary":
        label = BINARY_LABELS[int(pred)]
        classes = [0, 1]
        class_labels = [BINARY_LABELS[c] for c in classes]
    else:
        label = SEVERITY_LABELS[int(pred)]
        classes = sorted(SEVERITY_LABELS.keys())
        class_labels = [SEVERITY_LABELS[c] for c in classes]

    proba_dict = None
    if proba is not None:
        # model.classes_ tells us the real order of the probability columns
        model_classes = list(model.classes_)
        proba_dict = {
            (BINARY_LABELS if model_type == "binary" else SEVERITY_LABELS)[int(c)]: float(p)
            for c, p in zip(model_classes, proba)
        }

    return {
        "prediction": int(pred),
        "label": label,
        "probabilities": proba_dict,
        "input": input_df,
    }


# --------------------------------------------------------------------------
# Static metrics captured from the notebooks (used on the Model Performance page)
# --------------------------------------------------------------------------

MODEL_METRICS = {
    "binary": {
        "Logistic Regression": {"accuracy": 0.9016, "macro_f1": 0.9014, "weighted_f1": 0.9018, "params": "Default"},
        "KNN": {"accuracy": 0.8689, "macro_f1": 0.8688, "weighted_f1": 0.8686, "params": "Default"},
        "Decision Tree": {"accuracy": 0.7377, "macro_f1": 0.7376, "weighted_f1": 0.7380, "params": "Default"},
        "Random Forest": {"accuracy": 0.8525, "macro_f1": 0.8523, "weighted_f1": 0.8527, "params": "Default"},
        "XGBoost": {"accuracy": 0.8689, "macro_f1": 0.8688, "weighted_f1": 0.8690, "params": "Default"},
        "Tuned Logistic Regression": {"accuracy": 0.8852, "macro_f1": 0.8851, "weighted_f1": 0.8854, "params": "Tuned"},
    },
    "multiclass": {
        "Logistic Regression": {"accuracy": 0.5410, "macro_f1": 0.3590, "weighted_f1": 0.5618, "params": "Default"},
        "KNN": {"accuracy": 0.3770, "macro_f1": 0.2193, "weighted_f1": 0.4272, "params": "Default"},
        "Decision Tree": {"accuracy": 0.4262, "macro_f1": 0.2680, "weighted_f1": 0.4537, "params": "Default"},
        "Random Forest": {"accuracy": 0.4754, "macro_f1": 0.1706, "weighted_f1": 0.4614, "params": "Default"},
        "XGBoost": {"accuracy": 0.5410, "macro_f1": 0.2775, "weighted_f1": 0.5290, "params": "Default"},
        "Tuned Logistic Regression": {"accuracy": 0.5410, "macro_f1": 0.3590, "weighted_f1": 0.5618, "params": "Tuned"},
    },
}

# Per-class classification report for the final/tuned model on each task,
# for the "class-level performance" table on the Model Performance page.
CLASS_REPORTS = {
    "binary": {
        "labels": ["No Heart Disease", "Heart Disease Present"],
        "rows": [
            {"class": "No Heart Disease", "precision": 0.93, "recall": 0.85, "f1": 0.89, "support": 33},
            {"class": "Heart Disease Present", "precision": 0.84, "recall": 0.93, "f1": 0.88, "support": 28},
        ],
    },
    "multiclass": {
        "labels": ["No Disease", "Mild", "Moderate", "Severe", "Very Severe"],
        "rows": [
            {"class": "No Disease", "precision": 0.89, "recall": 0.76, "f1": 0.82, "support": 33},
            {"class": "Mild", "precision": 0.29, "recall": 0.36, "f1": 0.32, "support": 11},
            {"class": "Moderate", "precision": 0.20, "recall": 0.14, "f1": 0.17, "support": 7},
            {"class": "Severe", "precision": 0.25, "recall": 0.29, "f1": 0.27, "support": 7},
            {"class": "Very Severe", "precision": 0.17, "recall": 0.33, "f1": 0.22, "support": 3},
        ],
    },
}

TOP_FEATURES = {
    "binary": {
        "Decision Tree": [
            ("thalassemia_test_result_3.0", 0.3285),
            ("number_of_major_vessels", 0.1127),
            ("chest_pain_type_4.0", 0.1094),
            ("cholesterol_level", 0.0939),
            ("st_depression_during_exercise", 0.0763),
        ],
        "Random Forest": [
            ("number_of_major_vessels", 0.1050),
            ("thalassemia_test_result_3.0", 0.1021),
            ("maximum_heart_rate_achieved", 0.1004),
            ("thalassemia_test_result_7.0", 0.0875),
            ("st_depression_during_exercise", 0.0818),
        ],
        "XGBoost": [
            ("thalassemia_test_result_3.0", 0.4973),
            ("chest_pain_type_1.0", 0.1100),
            ("chest_pain_type_4.0", 0.0696),
            ("number_of_major_vessels", 0.0650),
            ("sex", 0.0315),
        ],
    },
    "multiclass": {
        "Decision Tree": [
            ("thalassemia_test_result_3.0", 0.1207),
            ("maximum_heart_rate_achieved", 0.1207),
            ("st_depression_during_exercise", 0.0982),
            ("number_of_major_vessels", 0.0858),
            ("cholesterol_level", 0.0849),
        ],
        "Random Forest": [
            ("number_of_major_vessels", 0.0979),
            ("maximum_heart_rate_achieved", 0.0921),
            ("st_depression_during_exercise", 0.0907),
            ("age", 0.0854),
            ("cholesterol_level", 0.0783),
        ],
        "XGBoost": [
            ("thalassemia_test_result_3.0", 0.2479),
            ("resting_ecg_result_1.0", 0.1008),
            ("thalassemia_test_result_6.0", 0.0644),
            ("number_of_major_vessels", 0.0507),
            ("chest_pain_type_1.0", 0.0505),
        ],
    },
}
