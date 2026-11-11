"""Edge cases, error handling, and unclear request guardrails for EDAgent."""

from __future__ import annotations

import re
from typing import List, Optional


OFF_TOPIC_KEYWORDS = [
    "weather", "recipe", "cook", "joke", "movie", "song", "poem", "lyrics",
    "politics", "president", "sports", "football", "cricket", "horoscope",
    "who are you", "what can you do", "help"
]


def is_off_topic(text: str) -> bool:
    """Detects whether a user message is off-topic or general chit-chat."""
    t = text.lower().strip()
    if not t:
        return False
    # If the user is asking about data, columns, stats, correlation, models, it's on-topic
    data_terms = ["data", "csv", "excel", "dataset", "column", "row", "model", "clean", "target", "heatmap", "eda", "train", "mean", "average", "std", "count", "sum", "salary", "churn", "price", "distribution"]
    if any(term in t for term in data_terms):
        return False

    return any(re.search(rf"\b{re.escape(k)}\b", t) for k in OFF_TOPIC_KEYWORDS)


def get_off_topic_reply(text: str) -> str:
    """Polite and helpful off-topic redirection message."""
    return (
        "👋 Hello! I am **EDAgent**, an autonomous Exploratory Data Analysis agent.\n\n"
        "I specialize in turning spreadsheets into complete machine learning & statistical analyses:\n"
        "- 📊 **Schema & Data Quality** (missingness, duplicates, outliers)\n"
        "- 🧹 **Guardrailed Data Cleaning** (with 60-second sandboxed execution)\n"
        "- 🔥 **Correlation Heatmaps & Feature Importance**\n"
        "- 📈 **5-Fold Cross-Validation Baseline Models**\n"
        "- 💬 **Computed Follow-up Answers** (e.g. `@EDAgent average salary by city?`)\n\n"
        "To get started, simply attach a **CSV**, **Excel (.xlsx)**, or **JSON** file to your message!"
    )


def get_missing_file_reply() -> str:
    """Reply when the user prompts EDAgent to analyze but provides no attachment and has no active dataset."""
    return (
        "📎 **No spreadsheet attached.**\n\n"
        "To run an Exploratory Data Analysis, please attach a spreadsheet file:\n"
        "- **CSV** (`.csv`)\n"
        "- **Excel** (`.xlsx`, `.xls`)\n"
        "- **JSON** (`.json` tabular array)\n\n"
        "You can also optionally specify your target column in your message, e.g.:\n"
        "`@EDAgent target: churn`"
    )


def get_corrupted_file_reply(filename: str, error_detail: str) -> str:
    """Reply when an uploaded file cannot be parsed or is corrupted."""
    return (
        f"❌ **Could not read file `{filename}`.**\n\n"
        f"**Reason**: {error_detail}\n\n"
        "**Troubleshooting tips**:\n"
        "1. Ensure the file is not empty or password-protected.\n"
        "2. For CSVs, verify that rows have consistent delimiters (comma or semicolon).\n"
        "3. For Excel files, verify the sheet contains a tabular header row.\n"
        "4. Supported formats are `.csv`, `.xlsx`, `.xls`, and `.json`."
    )


def get_guardrail_rejection_summary(violations: List[str]) -> str:
    """Transparently explains why an AI-generated cleaning step was rejected."""
    bullet_points = "\n".join([f"- {v}" for v in violations])
    return (
        "⚠️ **Guardrail Notice**: The AI-proposed cleaning code was rejected by EDAgent safety guardrails:\n"
        f"{bullet_points}\n\n"
        "🛡️ To protect your dataset integrity, EDAgent automatically fell back to safe, deterministic "
        "median (numeric) and mode (categorical) imputation."
    )
