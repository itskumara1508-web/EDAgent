"""Core pipeline and sandbox unit tests for EDAgent."""

import os
import pandas as pd
import numpy as np
import pytest

from edagent.sandbox import (
    execute_in_sandbox,
    validate_guardrails,
    check_ast_safety,
    GuardrailViolation,
    run_with_self_correction
)
from edagent.pipeline import run_pipeline, build_edagent_graph
from edagent.target_inference import infer_target_column


@pytest.fixture
def sample_classification_df():
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "tenure": np.random.randint(1, 36, size=n),
        "monthly_charges": np.random.uniform(20.0, 120.0, size=n).round(2),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
        "churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })


@pytest.fixture
def sample_regression_df():
    np.random.seed(42)
    n = 50
    sqft = np.random.randint(800, 3500, size=n)
    price = (sqft * 250 + np.random.normal(0, 15000, size=n)).round(2)
    return pd.DataFrame({
        "house_id": [f"H{i}" for i in range(n)],
        "square_feet": sqft,
        "bedrooms": np.random.choice([2, 3, 4, 5], size=n),
        "price": price
    })


# =====================================================================
# GUARDRAIL & SANDBOX TESTS
# =====================================================================

def test_ast_safety_blocks_forbidden_modules():
    dangerous_code = "import os\nos.system('echo hacked')"
    with pytest.raises(GuardrailViolation) as exc:
        check_ast_safety(dangerous_code)
    assert "Forbidden module import" in str(exc.value)

    dangerous_subproc = "from subprocess import Popen"
    with pytest.raises(GuardrailViolation) as exc:
        check_ast_safety(dangerous_subproc)
    assert "Forbidden module import" in str(exc.value)


def test_ast_safety_allows_pandas_and_numpy():
    safe_code = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "df_clean = df.copy()\n"
        "df_clean['x'] = 1\n"
    )
    # Should not raise
    check_ast_safety(safe_code)


def test_guardrail_rejects_deleted_column(sample_classification_df):
    df_orig = sample_classification_df
    # Drop a column
    df_trans = df_orig.drop(columns=["age"])
    violations = validate_guardrails(df_orig, df_trans, target_column="churn")
    assert any("No deleted columns" in v for v in violations)


def test_guardrail_rejects_excessive_row_deletion(sample_classification_df):
    df_orig = sample_classification_df
    # Drop 20% of rows (allowed max is 5%)
    n_drop = int(len(df_orig) * 0.20)
    df_trans = df_orig.iloc[n_drop:].copy()
    violations = validate_guardrails(df_orig, df_trans, target_column="churn")
    assert any("Max 5% rows removed" in v for v in violations)


def test_guardrail_rejects_target_alteration(sample_classification_df):
    df_orig = sample_classification_df
    df_trans = df_orig.copy()
    # Modify target values
    df_trans["churn"] = 0
    violations = validate_guardrails(df_orig, df_trans, target_column="churn")
    assert any("Target untouched" in v for v in violations)


def test_sandbox_executes_valid_cleaning_code(sample_classification_df):
    df = sample_classification_df.copy()
    # Introduce nulls in non-target column
    df.loc[0:2, "monthly_charges"] = np.nan

    valid_code = (
        "df_clean = df.copy()\n"
        "df_clean['monthly_charges'] = df_clean['monthly_charges'].fillna(df_clean['monthly_charges'].median())\n"
    )
    res = execute_in_sandbox(valid_code, df, target_column="churn", timeout_seconds=10)
    assert res.success is True
    assert res.output_df is not None
    assert res.output_df["monthly_charges"].isna().sum() == 0
    assert len(res.violations) == 0


def test_sandbox_self_correction_retry(sample_classification_df):
    df = sample_classification_df.copy()
    attempt = 0

    def code_gen(feedback):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            # First attempt violates guardrail: drops a column
            return "df_clean = df.drop(columns=['age'])"
        else:
            # Second attempt fixes it
            return "df_clean = df.copy()"

    res = run_with_self_correction(code_gen, df, target_column="churn", timeout_seconds=10)
    assert res.success is True
    assert res.retry_used is True


# =====================================================================
# FULL 11-AGENT 4-STAGE PIPELINE TESTS
# =====================================================================

def test_full_pipeline_classification(sample_classification_df, tmp_path):
    out_dir = str(tmp_path / "eda_test_class")
    progress_stages = []

    def progress_cb(st, tot_st, done_ag, tot_ag, msg):
        progress_stages.append(st)

    result = run_pipeline(
        df=sample_classification_df,
        specified_target="churn",
        output_dir=out_dir,
        progress_callback=progress_cb
    )

    # Verify all 4 stages ran
    assert 1 in progress_stages
    assert 2 in progress_stages
    assert 3 in progress_stages
    assert 4 in progress_stages

    # Verify Stage 1 Outputs
    assert result.schema_info["num_rows"] == len(sample_classification_df)
    assert result.schema_info["num_cols"] == len(sample_classification_df.columns)
    assert "missing_counts" in result.quality_info
    assert "numerical" in result.stats_info

    # Verify Stage 2 Cleaned Data
    assert result.cleaned_df is not None
    assert len(result.cleaned_df.columns) == len(sample_classification_df.columns)

    # Verify Stage 3 Outputs
    assert os.path.exists(result.heatmap_path)
    models = result.baseline_models_info["models"]
    assert len(models) >= 2  # At least Dummy and a classifier
    for m in models:
        assert "accuracy" in m
        assert 0.0 <= m["accuracy"] <= 1.0

    # Verify Stage 4 Outputs
    assert len(result.synthesis_text) > 0
    assert len(result.recommendation_text) > 0
    assert len(result.feature_engineering_text) > 0


def test_full_pipeline_regression(sample_regression_df, tmp_path):
    out_dir = str(tmp_path / "eda_test_reg")
    result = run_pipeline(
        df=sample_regression_df,
        specified_target="price",
        output_dir=out_dir
    )

    assert result.baseline_models_info["task_type"] == "regression"
    models = result.baseline_models_info["models"]
    assert len(models) >= 2
    for m in models:
        assert "r2" in m
        assert "rmse" in m


def test_target_inference(sample_classification_df):
    target, exp, is_inferred = infer_target_column(sample_classification_df)
    assert target == "churn"
    assert is_inferred is True
    assert "churn" in exp.lower()
