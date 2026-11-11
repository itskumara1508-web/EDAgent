"""Tests for spreadsheet file intake (CSV, XLSX, JSON)."""

import json
import os
import pandas as pd
import pytest
from edagent.main import parse_spreadsheet_file


def test_parse_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
    df = parse_spreadsheet_file(str(csv_file), "test.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_parse_csv_semicolon(tmp_path):
    csv_file = tmp_path / "test_semi.csv"
    csv_file.write_text("col1;col2;col3\n10;20;30\n40;50;60\n")
    df = parse_spreadsheet_file(str(csv_file), "test_semi.csv")
    assert df.shape == (2, 3)
    assert "col1" in df.columns


def test_parse_json_array(tmp_path):
    json_file = tmp_path / "test.json"
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    json_file.write_text(json.dumps(data))
    df = parse_spreadsheet_file(str(json_file), "test.json")
    assert df.shape == (2, 2)
    assert list(df.columns) == ["id", "name"]


def test_parse_excel(tmp_path):
    xlsx_file = tmp_path / "test.xlsx"
    orig_df = pd.DataFrame({"x": [10, 20], "y": ["cat", "dog"]})
    orig_df.to_excel(str(xlsx_file), index=False)
    df = parse_spreadsheet_file(str(xlsx_file), "test.xlsx")
    assert df.shape == (2, 2)
    assert list(df.columns) == ["x", "y"]
