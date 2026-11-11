"""Report and summary generation for EDAgent.

Generates:
- report.html (Comprehensive standalone HTML report)
- cleaned_data.csv (Cleaned dataset)
- clean.py (Cleaning script)
- chat_summary (Formatted Markdown for Nonilion room/DM chat)
"""

from __future__ import annotations

import base64
import html
import os
from typing import Dict, Any, Optional
import pandas as pd
from edagent.pipeline import PipelineResult


def generate_chat_summary(result: PipelineResult) -> str:
    """Formats a concise, high-density chat summary with headings and markdown tables."""
    schema = result.schema_info
    quality = result.quality_info
    models_info = result.baseline_models_info
    models = models_info.get("models", [])
    fi = result.feature_importance_info.get("top_features", [])

    lines = []
    lines.append("## 📊 EDAgent Analysis Complete")
    lines.append("")

    # Target & Dataset Overview
    lines.append("### 🔍 Dataset Profile")
    if result.is_target_inferred:
        lines.append(f"🎯 **Target Column**: `{result.target_col}` *(Auto-Inferred)*")
        lines.append(f"> 💡 *Reason: {result.target_explanation}*")
    elif result.target_col:
        lines.append(f"🎯 **Target Column**: `{result.target_col}` *(User Specified)*")
    else:
        lines.append("🎯 **Target Column**: *None (Unsupervised Mode)*")

    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Rows × Columns** | {schema.get('num_rows', 0)} rows × {schema.get('num_cols', 0)} cols |")
    lines.append(f"| **Missing Cells** | {quality.get('total_missing_cells', 0)} ({quality.get('overall_missing_pct', 0.0)}%) |")
    lines.append(f"| **Duplicate Rows** | {quality.get('duplicates_count', 0)} ({quality.get('duplicates_pct', 0.0)}%) |")
    lines.append(f"| **Cleaned Output** | {len(result.cleaned_df)} rows ({len(result.raw_df) - len(result.cleaned_df)} dropped) |")
    lines.append("")

    # Guardrails Notice if triggered
    if result.cleaning_guardrail_rejected:
        lines.append("> ⚠️ **Guardrail Notice**: AI-generated cleaning code violated safety guardrails "
                     "(e.g., column deletion or >5% row drop). Robust median/mode imputation was safely enforced.")
        lines.append("")
    elif result.cleaning_retry_used:
        lines.append("> ℹ️ **Self-Correction**: AI cleaning code triggered 1 self-correction retry and passed all safety guardrails.")
        lines.append("")

    # Top Features Table
    if fi:
        lines.append("### 🏆 Top Predictive Features")
        lines.append("| Rank | Feature | Importance Score | Mutual Information |")
        lines.append("| :---: | :--- | :---: | :---: |")
        for i, f in enumerate(fi[:5], 1):
            lines.append(f"| {i} | `{f['feature']}` | {f['rf_importance']:.4f} | {f['mutual_info']:.4f} |")
        lines.append("")

    # Baseline Model Scores (5-Fold CV)
    if models:
        metric_name = "Accuracy" if models_info.get("task_type") == "classification" else "R² Score"
        lines.append(f"### 📈 Baseline Models (5-Fold Cross-Validation)")
        lines.append(f"| Model | {metric_name} (Mean ± Std) | Secondary Metric | Status |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for i, m in enumerate(models):
            is_best = "⭐ **Best**" if i == 0 else "Baseline"
            if models_info.get("task_type") == "classification":
                lines.append(
                    f"| **{m['model_name']}** | {m['accuracy']:.4f} ± {m['accuracy_std']:.4f} | "
                    f"F1: {m.get('f1_macro', 0.0):.4f} | {is_best} |"
                )
            else:
                lines.append(
                    f"| **{m['model_name']}** | {m['r2']:.4f} ± {m['r2_std']:.4f} | "
                    f"RMSE: {m.get('rmse', 0.0):.4f} | {is_best} |"
                )
        lines.append("")

    # Synthesis & Recommendation
    lines.append("### 💡 Key Findings & Recommendation")
    lines.append(result.synthesis_text.strip())
    lines.append("")
    lines.append(f"**Recommended Model**: {result.recommendation_text.strip()}")
    lines.append("")
    lines.append("📎 *Attachments: `heatmap.png`, `report.html`, `cleaned_data.csv`, and `clean.py`.*")
    lines.append("💬 *Ask follow-up questions anytime (e.g. `@EDAgent average salary by department?`).*")

    return "\n".join(lines)


def generate_html_report(result: PipelineResult, output_path: str) -> str:
    """Generates a standalone, beautiful HTML report."""
    schema = result.schema_info
    quality = result.quality_info
    stats = result.stats_info
    models = result.baseline_models_info.get("models", [])
    fi = result.feature_importance_info.get("top_features", [])

    # Embed heatmap as base64 if present
    heatmap_b64 = ""
    if result.heatmap_path and os.path.exists(result.heatmap_path):
        try:
            with open(result.heatmap_path, "rb") as f:
                heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDAgent Full Exploratory Data Analysis Report</title>
<style>
  :root {{
    --primary: #4338ca;
    --primary-light: #6366f1;
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --success: #10b981;
    --warning: #f59e0b;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
    line-height: 1.5;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
  }}
  .header {{
    background: linear-gradient(135deg, #312e81, #4338ca);
    color: white;
    padding: 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }}
  .header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
  .header p {{ margin: 0; opacity: 0.9; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  h2 {{
    margin-top: 0;
    color: var(--primary);
    border-bottom: 2px solid #e0e7ff;
    padding-bottom: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 14px;
  }}
  th, td {{
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: #f1f5f9;
    font-weight: 600;
  }}
  .badge {{
    display: inline-block;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
  }}
  .badge-success {{ background: #d1fae5; color: #065f46; }}
  .badge-warning {{ background: #fef3c7; color: #92400e; }}
  .heatmap-img {{
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-top: 12px;
  }}
  pre {{
    background: #1e293b;
    color: #f8fafc;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 13px;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📊 EDAgent Exploratory Data Analysis Report</h1>
    <p>Automated Multi-Agent Machine Learning & Statistical Report • INNOVORTEX 4.0 Nonilion Track</p>
  </div>

  <div class="card">
    <h2>1. Executive Summary & Synthesis</h2>
    <p><strong>Target Variable:</strong> <code>{html.escape(str(result.target_col))}</code> ({'Auto-inferred' if result.is_target_inferred else 'User-specified'})</p>
    <p>{html.escape(result.synthesis_text)}</p>
    <p><strong>Model Recommendation:</strong> {html.escape(result.recommendation_text)}</p>
  </div>

  <div class="card">
    <h2>2. Dataset Overview & Data Quality</h2>
    <table>
      <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
      <tr><td>Total Rows</td><td>{schema.get('num_rows')}</td><td><span class="badge badge-success">Analyzed</span></td></tr>
      <tr><td>Total Columns</td><td>{schema.get('num_cols')}</td><td><span class="badge badge-success">Verified</span></td></tr>
      <tr><td>Missing Cells</td><td>{quality.get('total_missing_cells')} ({quality.get('overall_missing_pct')}%)</td><td><span class="badge badge-warning">Imputed</span></td></tr>
      <tr><td>Duplicate Rows</td><td>{quality.get('duplicates_count')} ({quality.get('duplicates_pct')}%)</td><td><span class="badge badge-success">Cleaned</span></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>3. 5-Fold Cross-Validation Baseline Models</h2>
    <table>
      <thead>
        <tr><th>Model</th><th>Primary Score (Mean ± Std)</th><th>Details</th></tr>
      </thead>
      <tbody>
"""
    for m in models:
        html_content += f"""
        <tr>
          <td><strong>{html.escape(m['model_name'])}</strong></td>
          <td>{m['primary_score']} ± {m.get('accuracy_std') or m.get('r2_std', 0.0)}</td>
          <td>{'Classification' if result.baseline_models_info.get('task_type') == 'classification' else 'Regression'}</td>
        </tr>
"""
    html_content += """
      </tbody>
    </table>
  </div>
"""

    if heatmap_b64:
        html_content += f"""
  <div class="card">
    <h2>4. Feature Correlation Matrix</h2>
    <img class="heatmap-img" src="data:image/png;base64,{heatmap_b64}" alt="Correlation Heatmap">
  </div>
"""

    html_content += f"""
  <div class="card">
    <h2>5. Feature Engineering Recommendations</h2>
    <p>{html.escape(result.feature_engineering_text)}</p>
  </div>

  <div class="card">
    <h2>6. Validated Data Cleaning Script (clean.py)</h2>
    <pre><code>{html.escape(result.cleaning_code)}</code></pre>
  </div>
</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def export_pipeline_artifacts(result: PipelineResult, output_dir: str) -> Dict[str, str]:
    """Exports all 4 required artifacts to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cleaned CSV
    cleaned_csv_path = os.path.join(output_dir, "cleaned_data.csv")
    result.cleaned_df.to_csv(cleaned_csv_path, index=False)

    # 2. Cleaning Python Script
    clean_py_path = os.path.join(output_dir, "clean.py")
    with open(clean_py_path, "w", encoding="utf-8") as f:
        f.write(result.cleaning_code)

    # 3. HTML Report
    html_report_path = os.path.join(output_dir, "report.html")
    generate_html_report(result, html_report_path)

    # 4. Heatmap is already at result.heatmap_path

    return {
        "cleaned_csv": cleaned_csv_path,
        "clean_py": clean_py_path,
        "html_report": html_report_path,
        "heatmap": result.heatmap_path
    }
