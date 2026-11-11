"""Tests for unclear, corrupt, missing, and off-topic requests."""

import pandas as pd
import pytest

from edagent.edge_cases import (
    is_off_topic,
    get_off_topic_reply,
    get_missing_file_reply,
    get_corrupted_file_reply,
    get_guardrail_rejection_summary
)
from edagent.target_inference import infer_target_column


def test_off_topic_detection():
    assert is_off_topic("What is the weather in Tokyo?") is True
    assert is_off_topic("Tell me a funny joke") is True
    assert is_off_topic("Can you write a poem about flowers?") is True

    # Data questions should NOT be detected as off-topic
    assert is_off_topic("What is the average salary?") is False
    assert is_off_topic("Analyze my dataset") is False
    assert is_off_topic("Show me the correlation heatmap") is False


def test_missing_file_reply():
    reply = get_missing_file_reply()
    assert "No spreadsheet attached" in reply
    assert ".csv" in reply
    assert ".xlsx" in reply


def test_corrupt_file_reply():
    reply = get_corrupted_file_reply("bad_data.csv", "Invalid token on line 4")
    assert "Could not read file" in reply
    assert "bad_data.csv" in reply
    assert "Invalid token" in reply


def test_guardrail_rejection_summary():
    violations = ["GUARDRAIL 1: 2 columns deleted", "GUARDRAIL 2: 12% rows dropped"]
    summary = get_guardrail_rejection_summary(violations)
    assert "Guardrail Notice" in summary
    assert "GUARDRAIL 1" in summary
    assert "GUARDRAIL 2" in summary


def test_target_inference_heuristics():
    # Case 1: Match standard name 'salary'
    df1 = pd.DataFrame({"emp_id": [1, 2], "age": [30, 40], "salary": [50000, 60000]})
    target, exp, is_inf = infer_target_column(df1)
    assert target == "salary"
    assert is_inf is True

    # Case 2: Binary column detection
    df2 = pd.DataFrame({"id": [1, 2, 3, 4], "feature_val": [10, 20, 30, 40], "flag": ["Y", "N", "Y", "N"]})
    target, exp, is_inf = infer_target_column(df2)
    assert target == "flag"
    assert is_inf is True

    # Case 3: User-specified target
    target, exp, is_inf = infer_target_column(df1, specified_target="age")
    assert target == "age"
    assert is_inf is False
