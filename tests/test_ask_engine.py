"""Tests for computed follow-up questions ('Ask' feature)."""

import pandas as pd
import pytest

from edagent.session_store import SessionData
from edagent.ask_engine import answer_question_about_data


@pytest.fixture
def active_session():
    df = pd.DataFrame({
        "emp_id": [1, 2, 3, 4, 5, 6],
        "department": ["Eng", "Eng", "Sales", "Sales", "HR", "HR"],
        "city": ["NY", "SF", "NY", "Chicago", "SF", "Chicago"],
        "salary": [100000, 150000, 80000, 90000, 70000, 75000]
    })
    return SessionData(raw_df=df, cleaned_df=df, target_col="salary")


def test_ask_engine_no_session():
    reply, code = answer_question_about_data("What is the average salary?", session=None)
    assert "No active dataset found" in reply
    assert code is None


def test_ask_engine_computes_exact_numbers(active_session):
    reply, code = answer_question_about_data("What is the average salary by department?", session=active_session)
    assert "Question Answered" in reply
    # Python code must have been executed
    assert code is not None
    assert "Eng" in reply
    assert "Sales" in reply
    # Mean Eng salary = (100k + 150k)/2 = 125,000
    assert "125000" in reply or "125,000" in reply


def test_ask_engine_computes_city_grouping(active_session):
    reply, code = answer_question_about_data("Average salary by city?", session=active_session)
    assert "Question Answered" in reply
    assert "NY" in reply
    assert "SF" in reply
    # Mean NY salary = (100k + 80k)/2 = 90,000
    assert "90000" in reply or "90,000" in reply
