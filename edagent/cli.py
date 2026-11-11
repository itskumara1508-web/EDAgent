"""Command-line interface (CLI) for running EDAgent standalone on spreadsheets."""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

from edagent.main import parse_spreadsheet_file
from edagent.pipeline import run_pipeline
from edagent.report_generator import export_pipeline_artifacts, generate_chat_summary


def main():
    parser = argparse.ArgumentParser(
        description="EDAgent CLI: Autonomous 11-Agent 4-Stage Exploratory Data Analysis."
    )
    parser.add_argument("file", help="Path to spreadsheet (CSV, XLSX, or JSON)")
    parser.add_argument("--target", "-t", default=None, help="Target column name (optional)")
    parser.add_argument("--output-dir", "-o", default="./eda_output", help="Directory to save artifacts")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    print(f"Loading '{args.file}'...")
    try:
        df = parse_spreadsheet_file(args.file, os.path.basename(args.file))
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("Spawning 11 LangGraph agents across 4 stages...\n")

    def progress_cb(stage, total_stages, done_agents, total_agents, msg):
        pct = int((done_agents / max(1, total_agents)) * 100)
        print(f"[{stage}/{total_stages}] ({pct}%) {msg}")

    result = run_pipeline(
        df=df,
        specified_target=args.target,
        output_dir=args.output_dir,
        progress_callback=progress_cb
    )

    artifacts = export_pipeline_artifacts(result, args.output_dir)

    print("\n" + "=" * 60)
    print(generate_chat_summary(result))
    print("=" * 60)
    print("\nArtifacts exported successfully:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
