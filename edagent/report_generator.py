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
<html lang="en" data-color-mode="dark" data-dark-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDAgent Exploratory Data Analysis Report</title>
<style>
  :root {{
    --color-canvas-default: #0d1117;
    --color-canvas-subtle: #161b22;
    --color-border-default: #30363d;
    --color-border-muted: #21262d;
    --color-fg-default: #e6edf3;
    --color-fg-muted: #8b949e;
    --color-accent-fg: #58a6ff;
    --color-accent-emphasis: #1f6feb;
    --color-accent-subtle: rgba(56, 139, 253, 0.15);
    --color-success-fg: #3fb950;
    --color-success-subtle: rgba(46, 160, 67, 0.15);
    --color-attention-fg: #d29922;
    --color-attention-subtle: rgba(187, 128, 9, 0.15);
    --font-code: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
    background-color: var(--color-canvas-default);
    color: var(--color-fg-default);
    margin: 0;
    padding: 0;
    font-size: 14px;
    line-height: 1.5;
  }}
  .gh-header {{
    background-color: var(--color-canvas-subtle);
    border-bottom: 1px solid var(--color-border-default);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .gh-header a {{
    color: var(--color-accent-fg);
    text-decoration: none;
    font-weight: 600;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px;
  }}
  .card {{
    background: var(--color-canvas-default);
    border: 1px solid var(--color-border-default);
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 20px;
  }}
  h2 {{
    margin-top: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--color-fg-default);
    border-bottom: 1px solid var(--color-border-muted);
    padding-bottom: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 13px;
  }}
  th, td {{
    padding: 8px 12px;
    text-align: left;
    border: 1px solid var(--color-border-default);
  }}
  th {{
    background: var(--color-canvas-subtle);
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: rgba(22, 27, 34, 0.5);
  }}
  .badge {{
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 2em;
    font-size: 11px;
    font-weight: 600;
  }}
  .badge-success {{ background: var(--color-success-subtle); color: var(--color-success-fg); border: 1px solid rgba(46, 160, 67, 0.4); }}
  .badge-warning {{ background: var(--color-attention-subtle); color: var(--color-attention-fg); border: 1px solid rgba(187, 128, 9, 0.4); }}
  .badge-info {{ background: var(--color-accent-subtle); color: var(--color-accent-fg); border: 1px solid rgba(56, 139, 253, 0.4); }}
  .heatmap-img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    border: 1px solid var(--color-border-default);
    margin-top: 12px;
    background: #0d1117;
  }}
  pre {{
    background: var(--color-canvas-subtle);
    color: var(--color-fg-default);
    border: 1px solid var(--color-border-default);
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-family: var(--font-code);
    font-size: 13px;
  }}
  code {{
    font-family: var(--font-code);
    background: rgba(110, 118, 129, 0.4);
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-size: 85%;
  }}
  blockquote {{
    margin: 16px 0;
    padding: 8px 16px;
    color: var(--color-fg-muted);
    border-left: 4px solid var(--color-accent-emphasis);
    background: rgba(56, 139, 253, 0.08);
    border-radius: 0 6px 6px 0;
  }}
</style>
</head>
<body>
<header class="gh-header">
  <div>
    <strong>🤖 EDAgent</strong> • Automated Multi-Agent ML &amp; Statistical Report • INNOVORTEX 4.0
  </div>
  <div>
    <a href="https://github.com/itskumara1508-web/EDAgent" target="_blank">⭐ GitHub Repository</a>
  </div>
</header>
<div class="container">
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
