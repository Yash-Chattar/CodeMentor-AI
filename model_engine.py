"""Leakage-aware future-performance modelling for CodeMentor AI.

Classification target: the NEXT week's Weak/Average/Strong category.
Regression target: the NEXT week's accuracy.
All features are available at the end of the current week.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)

TOPICS = ["Loops", "Arrays", "Strings", "Sorting", "Linked Lists", "Recursion", "Trees", "Dynamic Programming"]
DIFFICULTY = {"Loops": .10, "Arrays": .15, "Strings": .20, "Sorting": .30,
              "Linked Lists": .40, "Recursion": .45, "Trees": .55, "Dynamic Programming": .70}
REQUIRED_COLUMNS = ["student_id", "week", "topic", "attempts", "problems_solved", "hints_used", "avg_time_min"]
LABELS = ["Weak", "Average", "Strong"]

# Predict the next week's outcome, not the current week's label.
CLASS_FEATURES = [
    "topic", "week", "accuracy", "previous_accuracy", "prev_avg3",
    "improvement", "rolling_std3", "attempts", "problems_solved",
    "hints_used", "avg_time_min", "solved_rate", "hint_rate"
]
REG_FEATURES = CLASS_FEATURES.copy()


def generate_demo_data(seed: int = 42, students: int = 60, weeks: int = 12) -> pd.DataFrame:
    """Create explicitly synthetic data for demos/testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(1, students + 1):
        ability = rng.normal(.68, .12)
        for topic in TOPICS:
            start = ability - .5 * DIFFICULTY[topic] + rng.normal(0, .10)
            gain = rng.uniform(.01, .035)
            for week in range(1, weeks + 1):
                accuracy = float(np.clip((start + gain * week) * 100 + rng.normal(0, 6), 5, 100))
                attempts = int(rng.integers(3, 12))
                solved = int(np.clip(round(attempts * accuracy / 100), 0, attempts))
                hints = int(max(0, round((1 - accuracy / 100) * attempts * .8 + rng.normal(0, .7))))
                time_min = float(np.clip(32 - accuracy * .2 + DIFFICULTY[topic] * 20 + rng.normal(0, 3), 5, 60))
                rows.append({"student_id": f"demo_{sid:03d}", "week": week, "topic": topic,
                             "attempts": attempts, "problems_solved": solved,
                             "hints_used": min(hints, attempts), "avg_time_min": round(time_min, 1),
                             "accuracy": round(accuracy, 1)})
    return validate_and_prepare(pd.DataFrame(rows))


def validate_and_prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate logs, consolidate student/topic/week rows and derive missing accuracy."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"user_id": "student_id", "learner_id": "student_id", "student": "student_id",
               "problems_attempted": "attempts", "solved": "problems_solved",
               "hints": "hints_used", "avg_time": "avg_time_min", "time_per_problem": "avg_time_min"}
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for c in ["week", "attempts", "problems_solved", "hints_used", "avg_time_min"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["topic"] = df["topic"].astype(str).str.strip()
    if df["student_id"].eq("").any() or df["topic"].eq("").any():
        raise ValueError("Student IDs and topic names cannot be blank.")
    if df[["week", "attempts", "problems_solved", "hints_used", "avg_time_min"]].isna().any().any():
        raise ValueError("Numeric fields contain blank or non-numeric values.")
    if (df["week"] < 1).any() or (df["attempts"] <= 0).any():
        raise ValueError("Week must be >= 1 and attempts must be > 0.")
    if ((df["problems_solved"] < 0) | (df["problems_solved"] > df["attempts"])).any():
        raise ValueError("Problems solved must be between zero and attempts.")
    if ((df["hints_used"] < 0) | (df["avg_time_min"] <= 0)).any():
        raise ValueError("Hints cannot be negative and average time must be positive.")
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        if df["accuracy"].isna().any() or ((df["accuracy"] < 0) | (df["accuracy"] > 100)).any():
            raise ValueError("Accuracy values must be numeric and between 0 and 100.")
    else:
        df["accuracy"] = 100 * df["problems_solved"] / df["attempts"]

    df["hints_used"] = np.minimum(df["hints_used"], df["attempts"])
    df["week"] = df["week"].astype(int)
    df["_weighted_time"] = df["avg_time_min"] * df["attempts"]
    df["_weighted_accuracy"] = df["accuracy"] * df["attempts"]
    df = df.groupby(["student_id", "week", "topic"], as_index=False).agg(
        attempts=("attempts", "sum"), problems_solved=("problems_solved", "sum"),
        hints_used=("hints_used", "sum"), _weighted_time=("_weighted_time", "sum"),
        _weighted_accuracy=("_weighted_accuracy", "sum"))
    df["avg_time_min"] = df["_weighted_time"] / df["attempts"]
    df["accuracy"] = df["_weighted_accuracy"] / df["attempts"]
    df = df.drop(columns=["_weighted_time", "_weighted_accuracy"])
    return df.sort_values(["student_id", "topic", "week"]).reset_index(drop=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create strictly historical features and next-week targets."""
    out = df.copy().sort_values(["student_id", "topic", "week"]).reset_index(drop=True)
    group = out.groupby(["student_id", "topic"], sort=False)
    out["previous_accuracy"] = group["accuracy"].shift(1)
    out["prev_avg3"] = group["accuracy"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    out["rolling_std3"] = group["accuracy"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    out["improvement"] = out["accuracy"] - out["previous_accuracy"]
    out["solved_rate"] = out["problems_solved"] / out["attempts"].clip(lower=1)
    out["hint_rate"] = out["hints_used"] / out["attempts"].clip(lower=1)
    next_week = group["week"].shift(-1)
    candidate_next_accuracy = group["accuracy"].shift(-1)
    out["next_accuracy"] = candidate_next_accuracy.where(next_week == out["week"] + 1)
    out["next_performance"] = pd.cut(
        out["next_accuracy"], bins=[-0.001, 50, 75, 100.001],
        labels=LABELS, right=False
    ).astype("object")
    return out


def _preprocessor(numeric_features, categorical_features, scale=True):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ("num", Pipeline(numeric_steps), numeric_features)
    ])


def train_and_evaluate(raw: pd.DataFrame) -> dict:
    """Chronological train/validation/test evaluation for next-week outcomes."""
    df = add_targets(raw)
    df = df.dropna(subset=["next_accuracy"]).copy()
    weeks = sorted(df.week.unique())
    if len(weeks) < 5:
        raise ValueError("At least five distinct weeks are required for train/validation/test evaluation.")

    test_start = weeks[-3]
    pretest = df[df.week < test_start]
    test = df[df.week >= test_start]
    val_week = sorted(pretest.week.unique())[-1]
    fit = pretest[pretest.week < val_week]
    val = pretest[pretest.week == val_week]
    if fit.empty or val.empty or test.empty:
        raise ValueError("Not enough chronological data for train/validation/test evaluation.")

    cat = ["topic"]
    cls_num = [c for c in CLASS_FEATURES if c not in cat]
    Xfit, yfit = fit[CLASS_FEATURES], fit["next_performance"].astype(str)
    Xval, yval = val[CLASS_FEATURES], val["next_performance"].astype(str)
    Xtest, ytest = test[CLASS_FEATURES], test["next_performance"].astype(str)

    cls_models = {
        "Logistic Regression": LogisticRegression(max_iter=2500, class_weight="balanced", C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, max_features=.8,
                                                 random_state=42, n_jobs=-1),
        "Random Forest (balanced)": RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                    max_features=.8, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, max_features=.8,
                                             class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=.035,
                                                          max_depth=2, random_state=42),
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                                    alpha=.02, learning_rate_init=.001, max_iter=300,
                                    early_stopping=False, validation_fraction=.12,
                                    n_iter_no_change=25, random_state=42)
    }

    def cls_pipe(name, estimator):
        scale = name in ("Logistic Regression", "ANN (MLP)")
        prep = _preprocessor(cls_num, cat, scale=scale)
        return Pipeline([("prep", prep), ("model", estimator)])

    cls_validation, cls_fitted, cls_predictions, cls_results = {}, {}, {}, []
    for name, estimator in cls_models.items():
        pipe = cls_pipe(name, estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xfit, yfit)
        val_pred = pipe.predict(Xval)
        cls_validation[name] = {
            "Accuracy": accuracy_score(yval, val_pred),
            "F1_macro": f1_score(yval, val_pred, average="macro", zero_division=0)
        }

    optimized_params = {"selection": "chronological validation comparison"}
    best_cls_name = max(cls_validation, key=lambda n: (cls_validation[n]["Accuracy"], cls_validation[n]["F1_macro"]))
    cls_train_X, cls_train_y = pretest[CLASS_FEATURES], pretest["next_performance"].astype(str)
    for name, estimator in cls_models.items():
        pipe = cls_pipe(name, estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(cls_train_X, cls_train_y)
        pred = pipe.predict(Xtest)
        cls_fitted[name], cls_predictions[name] = pipe, pred
        cls_results.append({
            "Model": name,
            "Validation_Accuracy": cls_validation[name]["Accuracy"],
            "Validation_F1_macro": cls_validation[name]["F1_macro"],
            "Test_Accuracy": accuracy_score(ytest, pred),
            "Test_Precision_macro": precision_score(ytest, pred, average="macro", zero_division=0),
            "Test_Recall_macro": recall_score(ytest, pred, average="macro", zero_division=0),
            "Test_F1_macro": f1_score(ytest, pred, average="macro", zero_division=0)
        })
    cls_table = pd.DataFrame(cls_results).sort_values(
        ["Validation_Accuracy", "Validation_F1_macro"], ascending=False
    ).reset_index(drop=True)
    best_cls_pred = cls_predictions[best_cls_name]

    # Regression uses the same historical/current-week predictors and next-week target.
    reg_cat = ["topic"]
    reg_num = [c for c in REG_FEATURES if c not in reg_cat]
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=220, min_samples_leaf=2,
                                                max_features=.8, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=220, min_samples_leaf=2,
                                            max_features=.8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=.03,
                    max_depth=2, loss="huber", random_state=42),
        "ANN (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                    alpha=.02, learning_rate_init=.001, max_iter=400,
                    early_stopping=False, validation_fraction=.12,
                    n_iter_no_change=30, random_state=42)
    }
    Xrfit, yrfit = fit[REG_FEATURES], fit["next_accuracy"]
    Xrval, yrval = val[REG_FEATURES], val["next_accuracy"]
    reg_validation, reg_fitted, reg_predictions, reg_results = {}, {}, {}, []

    for name, estimator in reg_models.items():
        pipe = Pipeline([("prep", _preprocessor(reg_num, reg_cat, scale=name in ("Linear Regression", "Ridge Regression", "ANN (MLP)"))),
                         ("model", estimator)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xrfit, yrfit)
        pv = np.clip(pipe.predict(Xrval), 0, 100)
        reg_validation[name] = mean_absolute_error(yrval, pv)

    best_reg_name = min(reg_validation, key=reg_validation.get)
    reg_train_X, reg_train_y = pretest[REG_FEATURES], pretest["next_accuracy"]
    reg_test_X, reg_test_y = test[REG_FEATURES], test["next_accuracy"]
    baseline = test["accuracy"].to_numpy()
    reg_results.append({
        "Model": "Persistence baseline", "Validation_MAE": np.nan,
        "Test_MAE": mean_absolute_error(reg_test_y, baseline),
        "Test_RMSE": mean_squared_error(reg_test_y, baseline) ** .5,
        "Test_R2": r2_score(reg_test_y, baseline)
    })
    for name, estimator in reg_models.items():
        pipe = Pipeline([("prep", _preprocessor(reg_num, reg_cat, scale=name in ("Linear Regression", "Ridge Regression", "ANN (MLP)"))),
                         ("model", estimator)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(reg_train_X, reg_train_y)
        pred = np.clip(pipe.predict(reg_test_X), 0, 100)
        reg_fitted[name], reg_predictions[name] = pipe, pred
        reg_results.append({
            "Model": name, "Validation_MAE": reg_validation[name],
            "Test_MAE": mean_absolute_error(reg_test_y, pred),
            "Test_RMSE": mean_squared_error(reg_test_y, pred) ** .5,
            "Test_R2": r2_score(reg_test_y, pred)
        })
    reg_table = pd.DataFrame(reg_results).sort_values("Validation_MAE", na_position="last").reset_index(drop=True)

    return {
        "data": df, "classification_results": cls_table, "regression_results": reg_table,
        "best_classifier_name": best_cls_name, "best_classifier": cls_fitted[best_cls_name],
        "best_regressor_name": best_reg_name, "best_regressor": reg_fitted[best_reg_name],
        "classification_test_actual": ytest.to_numpy(), "classification_test_pred": best_cls_pred,
        "regression_test_actual": reg_test_y.to_numpy(), "regression_test_pred": reg_predictions[best_reg_name],
        "confusion_matrix": confusion_matrix(ytest, best_cls_pred, labels=LABELS),
        "classification_report": classification_report(ytest, best_cls_pred, labels=LABELS,
                                                        zero_division=0, output_dict=True),
        "test_start_week": test_start, "optimized_classifier_params": optimized_params,
        "classification_models": cls_fitted, "regression_models": reg_fitted,
        "classification_target_description": "Next week's performance category",
        "regression_target_description": "Next week's accuracy (%)"
    }


def predict_next_accuracy(model, history: pd.DataFrame, topic: str) -> float:
    """Forecast next-week accuracy for a topic from the student's available history."""
    h = history[history.topic == topic].sort_values("week")
    if h.empty:
        return float("nan")
    group = h.iloc[-1]
    previous = float(h.accuracy.iloc[-2]) if len(h) > 1 else np.nan
    prev_avg3 = float(h.accuracy.tail(3).mean())
    rolling_std3 = float(h.accuracy.tail(3).std()) if len(h) > 1 else np.nan
    row = pd.DataFrame([{
        "topic": topic, "week": int(group.week), "accuracy": float(group.accuracy),
        "previous_accuracy": previous, "prev_avg3": prev_avg3,
        "improvement": float(group.accuracy - previous) if np.isfinite(previous) else np.nan,
        "rolling_std3": rolling_std3, "attempts": float(group.attempts),
        "problems_solved": float(group.problems_solved), "hints_used": float(group.hints_used),
        "avg_time_min": float(group.avg_time_min),
        "solved_rate": float(group.problems_solved / max(group.attempts, 1)),
        "hint_rate": float(group.hints_used / max(group.attempts, 1))
    }])
    return float(np.clip(model.predict(row[REG_FEATURES])[0], 0, 100))
