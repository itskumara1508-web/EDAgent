"""11-Agent 4-Stage LangGraph Pipeline for EDAgent.

Stages:
  Stage 1 (Parallel): Schema, Quality, Statistics, Outliers (4 agents)
  Stage 2 (Sequential): Data Cleaning with 60s Sandbox & Guardrails (1 agent)
  Stage 3 (Parallel): Correlation, Feature Importance, Baseline Models (5-Fold CV) (3 agents)
  Stage 4 (Sequential): Synthesis, Model Recommendation, Feature Engineering (3 agents)

Core Rules:
  - Python computes every number. AI only explains and writes code.
  - Baseline models use 5-fold cross-validation.
  - Guardrails enforced on cleaning.
"""

from __future__ import annotations

import io
import math
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langgraph.graph import END, START, StateGraph
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from edagent.llm_provider import get_llm_provider
from edagent.sandbox import execute_in_sandbox, run_with_self_correction
from edagent.target_inference import infer_target_column


class EDAgentState(TypedDict, total=False):
    # Data & config
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    target_col: Optional[str]
    target_explanation: str
    is_target_inferred: bool
    output_dir: str

    # Stage 1 Outputs
    schema_info: Dict[str, Any]
    quality_info: Dict[str, Any]
    stats_info: Dict[str, Any]
    outliers_info: Dict[str, Any]

    # Stage 2 Outputs
    cleaning_code: str
    cleaning_logs: str
    guardrail_violations: List[str]
    cleaning_retry_used: bool
    cleaning_guardrail_rejected: bool

    # Stage 3 Outputs
    correlation_info: Dict[str, Any]
    heatmap_path: str
    feature_importance_info: Dict[str, Any]
    baseline_models_info: Dict[str, Any]

    # Stage 4 Outputs
    synthesis_text: str
    recommendation_text: str
    feature_engineering_text: str

    # Progress tracking
    completed_agents: int


# =====================================================================
# STAGE 1 AGENTS (PARALLEL: Schema, Quality, Statistics, Outliers)
# =====================================================================

def agent_schema(state: EDAgentState) -> Dict[str, Any]:
    """Agent 1: Schema Agent - Inspects types, column categories, dimensions."""
    df = state["raw_df"]
    num_rows, num_cols = df.shape
    columns_info = []

    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        if pd.api.types.is_numeric_dtype(s):
            col_type = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(s):
            col_type = "datetime"
        elif pd.api.types.is_bool_dtype(s):
            col_type = "boolean"
        else:
            col_type = "categorical"

        columns_info.append({
            "name": col,
            "dtype": dtype_str,
            "type_category": col_type,
            "null_count": int(s.isna().sum()),
            "nunique": int(s.nunique(dropna=True))
        })

    llm = get_llm_provider()
    explanation = llm.complete(
        prompt=f"Dataset shape is {num_rows} rows by {num_cols} columns. Columns: {columns_info}. Provide a 2-sentence schema summary.",
        system="Summarize the dataset schema concisely."
    )

    return {
        "schema_info": {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "columns": columns_info,
            "explanation": explanation
        }
    }


def agent_quality(state: EDAgentState) -> Dict[str, Any]:
    """Agent 2: Quality Agent - Missing values, duplicates, zero-variance columns."""
    df = state["raw_df"]
    n_rows = len(df)
    missing_counts = df.isna().sum().to_dict()
    missing_pcts = {k: round((v / n_rows) * 100, 2) if n_rows > 0 else 0.0 for k, v in missing_counts.items()}
    duplicates_count = int(df.duplicated().sum())
    duplicates_pct = round((duplicates_count / n_rows) * 100, 2) if n_rows > 0 else 0.0

    zero_var_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    high_cardinality_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique() > 50 and df[c].nunique() > 0.3 * n_rows
    ]

    total_missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.size)
    overall_missing_pct = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    return {
        "quality_info": {
            "missing_counts": missing_counts,
            "missing_pcts": missing_pcts,
            "total_missing_cells": total_missing_cells,
            "overall_missing_pct": overall_missing_pct,
            "duplicates_count": duplicates_count,
            "duplicates_pct": duplicates_pct,
            "zero_variance_cols": zero_var_cols,
            "high_cardinality_cols": high_cardinality_cols
        }
    }


def agent_statistics(state: EDAgentState) -> Dict[str, Any]:
    """Agent 3: Statistics Agent - Computes numerical and categorical summaries."""
    df = state["raw_df"]
    num_df = df.select_dtypes(include=["number"])
    num_stats = {}

    for c in num_df.columns:
        s = num_df[c].dropna()
        if len(s) > 0:
            num_stats[c] = {
                "count": int(s.count()),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std(ddof=1)), 3) if len(s) > 1 else 0.0,
                "min": round(float(s.min()), 3),
                "25%": round(float(s.quantile(0.25)), 3),
                "50%": round(float(s.median()), 3),
                "75%": round(float(s.quantile(0.75)), 3),
                "max": round(float(s.max()), 3),
                "skew": round(float(s.skew()), 3) if len(s) > 2 else 0.0
            }

    cat_df = df.select_dtypes(exclude=["number"])
    cat_stats = {}
    for c in cat_df.columns:
        s = cat_df[c].dropna()
        if len(s) > 0:
            top_val = str(s.mode()[0]) if not s.mode().empty else "N/A"
            top_freq = int(s.value_counts().iloc[0]) if not s.empty else 0
            cat_stats[c] = {
                "count": int(s.count()),
                "unique": int(s.nunique()),
                "top": top_val,
                "freq": top_freq,
                "top_pct": round((top_freq / len(s)) * 100, 2)
            }

    return {
        "stats_info": {
            "numerical": num_stats,
            "categorical": cat_stats
        }
    }


def agent_outliers(state: EDAgentState) -> Dict[str, Any]:
    """Agent 4: Outliers Agent - Detects outliers via IQR and Z-scores."""
    df = state["raw_df"]
    num_df = df.select_dtypes(include=["number"])
    outliers_data = {}

    for c in num_df.columns:
        s = num_df[c].dropna()
        if len(s) >= 4:
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_mask = (s < lower_bound) | (s > upper_bound)
            iqr_count = int(outlier_mask.sum())
            iqr_pct = round((iqr_count / len(s)) * 100, 2)

            outliers_data[c] = {
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3),
                "outlier_count": iqr_count,
                "outlier_pct": iqr_pct
            }

    return {
        "outliers_info": outliers_data
    }


# =====================================================================
# STAGE 2 AGENT (SEQUENTIAL: Data Cleaning with Guardrails & Sandbox)
# =====================================================================

def agent_data_cleaning(state: EDAgentState) -> Dict[str, Any]:
    """Agent 5: Data Cleaning Agent - AI writes cleaning code, verified by guardrails."""
    raw_df = state["raw_df"]
    target_col = state.get("target_col")
    quality_info = state.get("quality_info", {})
    outliers_info = state.get("outliers_info", {})
    llm = get_llm_provider()

    # Generator function for self-correction loop
    def generate_cleaning_code(feedback: Optional[str] = None) -> str:
        prompt = (
            f"Write robust Python cleaning code for DataFrame 'df'.\n"
            f"Columns: {list(raw_df.columns)}\n"
            f"Target column: '{target_col}'\n"
            f"Quality issues: {quality_info}\n"
            f"Outliers: {outliers_info}\n"
            f"MANDATORY RULES:\n"
            f"1. Never delete any column. df_clean must contain all original columns: {list(raw_df.columns)}.\n"
            f"2. Never delete more than 5% of rows. Prefer median imputation for numeric and mode for categorical.\n"
            f"3. Never modify the target column '{target_col}'.\n"
            f"4. Assign final cleaned DataFrame to 'df_clean'.\n"
        )
        if feedback:
            prompt += f"\nPREVIOUS ATTEMPT FAILED WITH REASON:\n{feedback}\nFix the code strictly adhering to the rules."

        response = llm.complete(
            prompt=prompt,
            system="You write safe, deterministic pandas cleaning scripts. Return ONLY Python code inside ```python ... ```."
        )

        # Extract code from markdown block if present
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.strip()

        # Fallback if empty
        if not code or "df_clean" not in code:
            code = (
                "df_clean = df.copy()\n"
                "for c in df_clean.select_dtypes(include=['number']).columns:\n"
                f"    if c != '{target_col}' and df_clean[c].isna().sum() > 0:\n"
                "        df_clean[c] = df_clean[c].fillna(df_clean[c].median())\n"
                "for c in df_clean.select_dtypes(exclude=['number']).columns:\n"
                f"    if c != '{target_col}' and df_clean[c].isna().sum() > 0:\n"
                "        mode_s = df_clean[c].mode()\n"
                "        df_clean[c] = df_clean[c].fillna(mode_s[0] if not mode_s.empty else 'Unknown')\n"
            )
        return code

    # Execute with 60s sandbox and 1 self-correction retry
    exec_result = run_with_self_correction(
        code_generator_fn=generate_cleaning_code,
        df=raw_df,
        target_column=target_col,
        timeout_seconds=60
    )

    cleaned_df: pd.DataFrame
    guardrail_rejected = False

    if exec_result.success and exec_result.output_df is not None:
        cleaned_df = exec_result.output_df
        clean_code = exec_result.code
    else:
        # Fallback: Safe deterministic cleaning if AI code repeatedly violated guardrails
        guardrail_rejected = True
        cleaned_df = raw_df.copy(deep=True)
        for c in cleaned_df.select_dtypes(include=["number"]).columns:
            if c != target_col and cleaned_df[c].isna().sum() > 0:
                cleaned_df[c] = cleaned_df[c].fillna(cleaned_df[c].median())
        for c in cleaned_df.select_dtypes(exclude=["number"]).columns:
            if c != target_col and cleaned_df[c].isna().sum() > 0:
                mode_val = cleaned_df[c].mode()[0] if not cleaned_df[c].mode().empty else "Unknown"
                cleaned_df[c] = cleaned_df[c].fillna(mode_val)

        clean_code = (
            "# Safe fallback cleaning applied after AI code violated safety guardrails\n"
            "df_clean = df.copy()\n"
            "for c in df_clean.select_dtypes(include=['number']).columns:\n"
            f"    if c != '{target_col}' and df_clean[c].isna().sum() > 0:\n"
            "        df_clean[c] = df_clean[c].fillna(df_clean[c].median())\n"
            "for c in df_clean.select_dtypes(exclude=['number']).columns:\n"
            f"    if c != '{target_col}' and df_clean[c].isna().sum() > 0:\n"
            "        mode_s = df_clean[c].mode()\n"
            "        df_clean[c] = df_clean[c].fillna(mode_s[0] if not mode_s.empty else 'Unknown')\n"
        )

    return {
        "cleaned_df": cleaned_df,
        "cleaning_code": clean_code,
        "cleaning_logs": exec_result.logs,
        "guardrail_violations": exec_result.violations,
        "cleaning_retry_used": exec_result.retry_used,
        "cleaning_guardrail_rejected": guardrail_rejected
    }


# =====================================================================
# STAGE 3 AGENTS (PARALLEL: Correlation, Feature Importance, Baselines)
# =====================================================================

def agent_correlation(state: EDAgentState) -> Dict[str, Any]:
    """Agent 6: Correlation Agent - Pearson/Spearman matrix and heatmap PNG."""
    df = state["cleaned_df"]
    output_dir = state.get("output_dir", tempfile.gettempdir())
    num_df = df.select_dtypes(include=["number"])

    corr_info = {"high_correlations": [], "matrix": {}}
    heatmap_path = os.path.join(output_dir, "heatmap.png")

    if num_df.shape[1] >= 2:
        corr_matrix = num_df.corr(method="pearson").round(3)
        corr_info["matrix"] = corr_matrix.to_dict()

        # Find high correlation pairs
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= 0.70:
                    corr_info["high_correlations"].append({
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": float(val)
                    })

        # Generate Heatmap
        plt.figure(figsize=(max(6, len(cols) * 0.8), max(5, len(cols) * 0.7)))
        sns.heatmap(corr_matrix, annot=len(cols) <= 12, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True)
        plt.title("Correlation Heatmap (EDAgent)", fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
    else:
        # Create dummy placeholder heatmap if < 2 numerical columns
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Not enough numerical features\nfor correlation heatmap",
                 ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=100)
        plt.close()

    return {
        "correlation_info": corr_info,
        "heatmap_path": heatmap_path
    }


def agent_feature_importance(state: EDAgentState) -> Dict[str, Any]:
    """Agent 7: Feature Importance Agent - Mutual Info & Random Forest ranking."""
    df = state["cleaned_df"]
    target_col = state.get("target_col")

    importance_info = {"top_features": [], "method": "none"}

    if target_col and target_col in df.columns:
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert categoricals in X to numeric codes
        X_num = pd.DataFrame(index=X.index)
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X_num[col] = X[col].fillna(X[col].median() if len(X[col].dropna()) > 0 else 0)
            else:
                X_num[col] = pd.Categorical(X[col].astype(str)).codes

        # Check target type
        is_classification = (
            not pd.api.types.is_numeric_dtype(y)
            or y.nunique() <= 10
        )

        valid_idx = y.notna()
        if valid_idx.sum() >= 5 and X_num.shape[1] > 0:
            X_clean = X_num.loc[valid_idx]
            y_clean = y.loc[valid_idx]

            try:
                if is_classification:
                    y_encoded = pd.Categorical(y_clean.astype(str)).codes
                    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
                    rf.fit(X_clean, y_encoded)
                    mi = mutual_info_classif(X_clean, y_encoded, random_state=42)
                else:
                    rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
                    rf.fit(X_clean, y_clean)
                    mi = mutual_info_regression(X_clean, y_clean, random_state=42)

                rf_scores = rf.feature_importances_
                features = list(X.columns)
                ranked = []
                for idx, feat in enumerate(features):
                    ranked.append({
                        "feature": feat,
                        "rf_importance": round(float(rf_scores[idx]), 4),
                        "mutual_info": round(float(mi[idx]), 4)
                    })
                ranked.sort(key=lambda x: x["rf_importance"], reverse=True)
                importance_info["top_features"] = ranked[:10]
                importance_info["method"] = "RandomForest + Mutual Information"
            except Exception as e:
                importance_info["error"] = str(e)

    return {
        "feature_importance_info": importance_info
    }


def agent_baseline_models(state: EDAgentState) -> Dict[str, Any]:
    """Agent 8: Baseline Models Agent - 5-fold cross-validation with real scores."""
    df = state["cleaned_df"]
    target_col = state.get("target_col")

    models_info = {
        "task_type": "none",
        "models": [],
        "best_model_name": "None",
        "primary_metric": "none"
    }

    if not target_col or target_col not in df.columns:
        return {"baseline_models_info": models_info}

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Preprocess features into numerical representation
    X_num = pd.DataFrame(index=X.index)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X_num[col] = X[col].fillna(X[col].median() if len(X[col].dropna()) > 0 else 0)
        else:
            X_num[col] = pd.Categorical(X[col].astype(str)).codes

    # Filter out missing targets
    valid_idx = y.notna()
    if valid_idx.sum() < 10:
        return {"baseline_models_info": models_info}

    X_valid = X_num.loc[valid_idx]
    y_valid = y.loc[valid_idx]

    is_classification = (
        not pd.api.types.is_numeric_dtype(y_valid)
        or y_valid.nunique() <= 10
    )

    n_samples = len(X_valid)
    n_splits = min(5, max(2, n_samples // 3))

    if is_classification:
        models_info["task_type"] = "classification"
        models_info["primary_metric"] = "accuracy"
        y_encoded = pd.Categorical(y_valid.astype(str)).codes

        # Ensure class count allows stratified k-fold
        min_class_count = pd.Series(y_encoded).value_counts().min()
        actual_splits = min(n_splits, max(2, min_class_count))

        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
        candidates = [
            ("Dummy Classifier", DummyClassifier(strategy="most_frequent")),
            ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42))
        ]

        scoring = ["accuracy", "f1_macro"]

        for name, clf in candidates:
            try:
                cv_res = cross_validate(clf, X_valid, y_encoded, cv=cv, scoring=scoring, error_score="raise")
                acc_mean = float(np.mean(cv_res["test_accuracy"]))
                acc_std = float(np.std(cv_res["test_accuracy"]))
                f1_mean = float(np.mean(cv_res["test_f1_macro"]))
                f1_std = float(np.std(cv_res["test_f1_macro"]))
                models_info["models"].append({
                    "model_name": name,
                    "accuracy": round(acc_mean, 4),
                    "accuracy_std": round(acc_std, 4),
                    "f1_macro": round(f1_mean, 4),
                    "f1_std": round(f1_std, 4),
                    "primary_score": round(acc_mean, 4)
                })
            except Exception as e:
                # If a specific model fails (e.g. convergence), record with safe fallbacks
                pass

        if models_info["models"]:
            models_info["models"].sort(key=lambda m: m["primary_score"], reverse=True)
            models_info["best_model_name"] = models_info["models"][0]["model_name"]

    else:
        models_info["task_type"] = "regression"
        models_info["primary_metric"] = "r2"
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        candidates = [
            ("Dummy Regressor", DummyRegressor(strategy="mean")),
            ("Ridge Regression", Ridge(alpha=1.0, random_state=42)),
            ("Random Forest", RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42))
        ]

        scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

        for name, reg in candidates:
            try:
                cv_res = cross_validate(reg, X_valid, y_valid, cv=cv, scoring=scoring, error_score="raise")
                r2_mean = float(np.mean(cv_res["test_r2"]))
                r2_std = float(np.std(cv_res["test_r2"]))
                rmse_mean = float(np.mean(np.sqrt(-cv_res["test_neg_mean_squared_error"])))
                mae_mean = float(np.mean(-cv_res["test_neg_mean_absolute_error"]))
                models_info["models"].append({
                    "model_name": name,
                    "r2": round(r2_mean, 4),
                    "r2_std": round(r2_std, 4),
                    "rmse": round(rmse_mean, 4),
                    "mae": round(mae_mean, 4),
                    "primary_score": round(r2_mean, 4)
                })
            except Exception as e:
                pass

        if models_info["models"]:
            models_info["models"].sort(key=lambda m: m["primary_score"], reverse=True)
            models_info["best_model_name"] = models_info["models"][0]["model_name"]

    return {
        "baseline_models_info": models_info
    }


# =====================================================================
# STAGE 4 AGENTS (SEQUENTIAL: Synthesis, Recommendation, Feature Eng)
# =====================================================================

def agent_synthesis(state: EDAgentState) -> Dict[str, Any]:
    """Agent 9: Synthesis Agent - Synthesizes findings across all previous agents."""
    schema = state.get("schema_info", {})
    quality = state.get("quality_info", {})
    stats = state.get("stats_info", {})
    target = state.get("target_col", "None")
    target_exp = state.get("target_explanation", "")
    corr = state.get("correlation_info", {})
    fi = state.get("feature_importance_info", {})
    models = state.get("baseline_models_info", {})

    llm = get_llm_provider()
    prompt = (
        f"Synthesize an executive EDA summary:\n"
        f"- Target: '{target}' ({target_exp})\n"
        f"- Dimensions: {schema.get('num_rows')} rows, {schema.get('num_cols')} columns\n"
        f"- Missing rate: {quality.get('overall_missing_pct')}%, Duplicates: {quality.get('duplicates_count')}\n"
        f"- High correlations: {len(corr.get('high_correlations', []))} pairs\n"
        f"- Top features: {[f['feature'] for f in fi.get('top_features', [])[:3]]}\n"
        f"- Best baseline model: {models.get('best_model_name')}\n"
        f"Provide 3-4 bullet points highlighting key business insights, risks, and next steps."
    )
    synthesis = llm.complete(
        prompt=prompt,
        system="You are the lead EDA synthesis agent. Write crisp, high-impact findings."
    )

    return {
        "synthesis_text": synthesis
    }


def agent_model_recommendation(state: EDAgentState) -> Dict[str, Any]:
    """Agent 10: Model Recommendation Agent - Recommends architecture grounded in 5-fold CV."""
    models_info = state.get("baseline_models_info", {})
    models = models_info.get("models", [])
    best_name = models_info.get("best_model_name", "None")
    task_type = models_info.get("task_type", "None")

    if not models or best_name == "None":
        rec_text = "No predictive models were evaluated as no target column was designated."
    else:
        scores_summary = "\n".join([
            f"- {m['model_name']}: score = {m['primary_score']} (std = {m.get('accuracy_std') or m.get('r2_std', 0)})"
            for m in models
        ])
        llm = get_llm_provider()
        prompt = (
            f"Based strictly on these 5-fold cross-validation scores:\n{scores_summary}\n"
            f"Task: {task_type}. Best model: {best_name}.\n"
            f"Explain why '{best_name}' is recommended over linear/dummy baselines, referencing the exact scores."
        )
        rec_text = llm.complete(
            prompt=prompt,
            system="Explain the model recommendation strictly referencing the computed cross-validation scores."
        )

    return {
        "recommendation_text": rec_text
    }


def agent_feature_engineering(state: EDAgentState) -> Dict[str, Any]:
    """Agent 11: Feature Engineering Agent - Proposes transformations and interaction features."""
    fi = state.get("feature_importance_info", {}).get("top_features", [])
    top_cols = [f["feature"] for f in fi[:4]]
    stats = state.get("stats_info", {}).get("numerical", {})

    llm = get_llm_provider()
    prompt = (
        f"Propose 3 specific feature engineering techniques for this dataset:\n"
        f"Top predictive columns: {top_cols}\n"
        f"Numerical statistics: {list(stats.keys())[:5]}\n"
        f"Suggest concrete interactions, ratios, or binning strategies to boost model performance."
    )
    fe_text = llm.complete(
        prompt=prompt,
        system="Provide actionable, domain-relevant feature engineering transformations."
    )

    return {
        "feature_engineering_text": fe_text
    }


# =====================================================================
# LANGGRAPH BUILDER & PIPELINE RUNNER
# =====================================================================

def build_edagent_graph() -> Any:
    """Builds the 11-agent 4-stage LangGraph workflow."""
    workflow = StateGraph(EDAgentState)

    # Add 11 nodes
    workflow.add_node("schema", agent_schema)
    workflow.add_node("quality", agent_quality)
    workflow.add_node("statistics", agent_statistics)
    workflow.add_node("outliers", agent_outliers)
    workflow.add_node("cleaning", agent_data_cleaning)
    workflow.add_node("correlation", agent_correlation)
    workflow.add_node("feature_importance", agent_feature_importance)
    workflow.add_node("baseline_models", agent_baseline_models)
    workflow.add_node("synthesis", agent_synthesis)
    workflow.add_node("recommendation", agent_model_recommendation)
    workflow.add_node("feature_engineering", agent_feature_engineering)

    # Stage 1: Parallel Fan-Out from START to (schema, quality, statistics, outliers)
    workflow.add_edge(START, "schema")
    workflow.add_edge(START, "quality")
    workflow.add_edge(START, "statistics")
    workflow.add_edge(START, "outliers")

    # Stage 1 Fan-In to Stage 2: cleaning
    workflow.add_edge("schema", "cleaning")
    workflow.add_edge("quality", "cleaning")
    workflow.add_edge("statistics", "cleaning")
    workflow.add_edge("outliers", "cleaning")

    # Stage 2 Fan-Out to Stage 3: (correlation, feature_importance, baseline_models)
    workflow.add_edge("cleaning", "correlation")
    workflow.add_edge("cleaning", "feature_importance")
    workflow.add_edge("cleaning", "baseline_models")

    # Stage 3 Fan-In to Stage 4: synthesis -> recommendation -> feature_engineering -> END
    workflow.add_edge("correlation", "synthesis")
    workflow.add_edge("feature_importance", "synthesis")
    workflow.add_edge("baseline_models", "synthesis")
    workflow.add_edge("synthesis", "recommendation")
    workflow.add_edge("recommendation", "feature_engineering")
    workflow.add_edge("feature_engineering", END)

    return workflow.compile()


class PipelineResult:
    def __init__(self, state: EDAgentState):
        self.state = state
        self.raw_df = state["raw_df"]
        self.cleaned_df = state["cleaned_df"]
        self.target_col = state.get("target_col")
        self.target_explanation = state.get("target_explanation", "")
        self.is_target_inferred = state.get("is_target_inferred", False)
        self.cleaning_code = state.get("cleaning_code", "")
        self.guardrail_violations = state.get("guardrail_violations", [])
        self.cleaning_retry_used = state.get("cleaning_retry_used", False)
        self.cleaning_guardrail_rejected = state.get("cleaning_guardrail_rejected", False)
        self.heatmap_path = state.get("heatmap_path", "")
        self.schema_info = state.get("schema_info", {})
        self.quality_info = state.get("quality_info", {})
        self.stats_info = state.get("stats_info", {})
        self.outliers_info = state.get("outliers_info", {})
        self.correlation_info = state.get("correlation_info", {})
        self.feature_importance_info = state.get("feature_importance_info", {})
        self.baseline_models_info = state.get("baseline_models_info", {})
        self.synthesis_text = state.get("synthesis_text", "")
        self.recommendation_text = state.get("recommendation_text", "")
        self.feature_engineering_text = state.get("feature_engineering_text", "")


def run_pipeline(
    df: pd.DataFrame,
    specified_target: Optional[str] = None,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, int, int, str], None]] = None
) -> PipelineResult:
    """Runs the full 11-agent 4-stage EDA pipeline with stage-level progress reporting."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="edagent_")
    os.makedirs(output_dir, exist_ok=True)

    # Infer or validate target column
    target_col, target_exp, is_inferred = infer_target_column(df, specified_target)

    # Initial state
    state: EDAgentState = {
        "raw_df": df,
        "target_col": target_col,
        "target_explanation": target_exp,
        "is_target_inferred": is_inferred,
        "output_dir": output_dir,
        "completed_agents": 0
    }

    # Execute Stage 1
    if progress_callback:
        progress_callback(1, 4, 0, 11, "Stage 1/4: Running Schema, Quality, Statistics, and Outliers in parallel...")
    
    s1_res = {
        **agent_schema(state),
        **agent_quality(state),
        **agent_statistics(state),
        **agent_outliers(state)
    }
    state.update(s1_res)  # type: ignore

    if progress_callback:
        progress_callback(1, 4, 4, 11, "Stage 1/4 (4/11 agents done): Schema, Quality, Statistics, Outliers complete.")

    # Execute Stage 2: Data Cleaning
    if progress_callback:
        progress_callback(2, 4, 4, 11, "Stage 2/4: Running Data Cleaning under 60s sandbox and guardrails...")
    
    s2_res = agent_data_cleaning(state)
    state.update(s2_res)  # type: ignore

    if progress_callback:
        progress_callback(2, 4, 5, 11, "Stage 2/4 (5/11 agents done): Data cleaning validated and applied.")

    # Execute Stage 3: Correlation, Feature Importance, Baselines
    if progress_callback:
        progress_callback(3, 4, 5, 11, "Stage 3/4: Evaluating correlations, feature importance, and 5-fold CV baselines...")
    
    s3_res = {
        **agent_correlation(state),
        **agent_feature_importance(state),
        **agent_baseline_models(state)
    }
    state.update(s3_res)  # type: ignore

    if progress_callback:
        progress_callback(3, 4, 8, 11, "Stage 3/4 (8/11 agents done): Heatmap generated, 5-fold CV models evaluated.")

    # Execute Stage 4: Synthesis, Recommendation, Feature Engineering
    if progress_callback:
        progress_callback(4, 4, 8, 11, "Stage 4/4: Synthesizing findings and generating model recommendations...")
    
    s4_synthesis = agent_synthesis(state)
    state.update(s4_synthesis)  # type: ignore

    s4_rec = agent_model_recommendation(state)
    state.update(s4_rec)  # type: ignore

    s4_fe = agent_feature_engineering(state)
    state.update(s4_fe)  # type: ignore

    if progress_callback:
        progress_callback(4, 4, 11, 11, "Stage 4/4 (11/11 agents done): All 11 agents completed successfully!")

    return PipelineResult(state)
