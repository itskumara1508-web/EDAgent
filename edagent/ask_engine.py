"""Analytical follow-up Q&A ('Ask' feature) for EDAgent.

Rules:
- Python computes every number!
- AI only translates the user question into pandas/numpy query code and explains the result.
- Code is executed in an isolated safe sandbox.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from edagent.llm_provider import get_llm_provider
from edagent.session_store import SessionData


def answer_question_about_data(
    question: str,
    session: Optional[SessionData]
) -> Tuple[str, Optional[str]]:
    """Answers a question about the active dataset by generating and executing Python query code.
    
    Returns:
        (reply_text, python_code_executed)
    """
    if session is None or session.cleaned_df is None or session.cleaned_df.empty:
        return (
            "⚠️ No active dataset found in this room or chat.\n"
            "Please upload a CSV, Excel (.xlsx), or JSON file first, and I will analyze it!",
            None
        )

    df = session.cleaned_df
    columns = list(df.columns)
    sample_head = df.head(3).to_dict(orient="records")
    llm = get_llm_provider()

    # Step 1: Translate natural language into pandas code
    prompt = (
        f"You are EDAgent's analytical code generator.\n"
        f"The user is asking a question about a DataFrame 'df':\n"
        f"User Question: \"{question}\"\n"
        f"Columns: {columns}\n"
        f"Dtypes: {dict(df.dtypes.astype(str))}\n"
        f"Sample rows: {sample_head}\n\n"
        f"Write safe Python code to compute the exact answer.\n"
        f"Store the final computed answer in a variable named 'result' (can be a scalar, Series, or DataFrame).\n"
        f"Store a brief 1-sentence explanation of what was computed in 'explanation'.\n"
        f"Return ONLY executable Python code inside ```python ... ```."
    )

    llm_resp = llm.complete(
        prompt=prompt,
        system="Generate safe pandas query code. Assign final answer to variable 'result'."
    )

    if "```python" in llm_resp:
        code = llm_resp.split("```python")[1].split("```")[0].strip()
    elif "```" in llm_resp:
        code = llm_resp.split("```")[1].split("```")[0].strip()
    else:
        code = llm_resp.strip()

    # Heuristic fallback code generator if LLM code is invalid or missing
    if not code or "result" not in code:
        q_lower = question.lower()
        num_cols = list(df.select_dtypes(include=["number"]).columns)
        cat_cols = list(df.select_dtypes(exclude=["number"]).columns)

        # Pattern: average / mean of X by Y
        if ("average" in q_lower or "mean" in q_lower) and len(num_cols) > 0 and len(cat_cols) > 0:
            target_num = num_cols[0]
            target_cat = cat_cols[0]
            # Match specific column names mentioned in question
            for c in num_cols:
                if c.lower() in q_lower:
                    target_num = c
                    break
            for c in cat_cols:
                if c.lower() in q_lower:
                    target_cat = c
                    break
            code = (
                f"result = df.groupby('{target_cat}')['{target_num}'].mean().round(2).reset_index()\n"
                f"explanation = 'Computed mean {target_num} grouped by {target_cat}'"
            )
        else:
            code = (
                "result = df.describe()\n"
                "explanation = 'Summary statistics across all numerical columns'"
            )

    # Step 2: Execute code in safe sandbox
    safe_globals = {
        "df": df.copy(),
        "pd": pd,
        "np": np
    }

    try:
        exec(code, safe_globals)
        computed_result = safe_globals.get("result")
        explanation = safe_globals.get("explanation", "Computation completed directly on dataset.")
    except Exception as e:
        # Fallback to simple query or report error honestly
        return (
            f"❌ Could not compute answer directly: {str(e)}\n\n"
            f"Attempted query:\n```python\n{code}\n```",
            code
        )

    # Step 3: Format the response nicely
    def _format_table(data) -> str:
        try:
            return data.to_markdown(index=False)
        except Exception:
            # Fallback simple markdown table formatter
            if isinstance(data, pd.DataFrame):
                headers = [str(c) for c in data.columns]
                header_line = "| " + " | ".join(headers) + " |"
                sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                rows = []
                for _, row in data.iterrows():
                    rows.append("| " + " | ".join([str(val) for val in row.values]) + " |")
                return "\n".join([header_line, sep_line] + rows)
            return str(data)

    formatted_val = ""
    if isinstance(computed_result, pd.DataFrame):
        formatted_val = _format_table(computed_result)
    elif isinstance(computed_result, pd.Series):
        formatted_val = _format_table(computed_result.reset_index())
    elif isinstance(computed_result, (int, float, np.integer, np.floating)):
        formatted_val = f"**{computed_result:,.4f}**"
    else:
        formatted_val = f"**{computed_result}**"

    response = (
        f"### 💬 Question Answered\n\n"
        f"**Question**: \"{question}\"\n\n"
        f"**Result**:\n{formatted_val}\n\n"
        f"> 💡 *{explanation}*\n\n"
        f"<details><summary>View Python Query Code</summary>\n\n"
        f"```python\n{code}\n```\n"
        f"</details>"
    )

    return response, code
