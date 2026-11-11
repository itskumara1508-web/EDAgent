"""FastAPI Service for EDAgent.

Implements:
- GET /health
- POST /nonilion/webhook
- Background task orchestration with progress reporting
- Attachments handling (CSV/XLSX/JSON)
- Follow-up Q&A ('Ask' feature) with session memory
- Graceful edge-case handling (off-topic, corrupted, missing file)
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import base64
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field

from edagent.ask_engine import answer_question_about_data
from edagent.edge_cases import (
    get_corrupted_file_reply,
    get_missing_file_reply,
    get_off_topic_reply,
    is_off_topic,
)
from edagent.nonilion_client import get_nonilion_client
from edagent.pipeline import run_pipeline
from edagent.report_generator import export_pipeline_artifacts, generate_chat_summary
from edagent.session_store import get_session_store

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edagent.main")

app = FastAPI(
    title="EDAgent - AI Data Scientist for Nonilion",
    version="1.0.0",
    description="Autonomous 11-Agent 4-Stage Exploratory Data Analysis Bot for Nonilion."
)


class WebhookAttachment(BaseModel):
    name: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None
    download_url: Optional[str] = None
    file_id: Optional[str] = None
    content_type: Optional[str] = None
    size: Optional[int] = None


class WebhookMessage(BaseModel):
    id: Optional[str] = None
    chat_id: Optional[str] = None
    room_id: Optional[str] = None
    channel_id: Optional[str] = None
    user_id: Optional[str] = None
    text: Optional[str] = ""
    attachments: Optional[List[WebhookAttachment]] = []


class WebhookPayload(BaseModel):
    event: Optional[str] = "message"
    message: Optional[WebhookMessage] = None
    # Flat format compatibility
    chat_id: Optional[str] = None
    room_id: Optional[str] = None
    user_id: Optional[str] = None
    text: Optional[str] = None
    attachments: Optional[List[WebhookAttachment]] = None


@app.get("/")
def root_endpoint():
    return {
        "service": "EDAgent",
        "description": "Exploratory Data Analysis Bot on Nonilion",
        "track": "INNOVORTEX 4.0",
        "status": "online"
    }


@app.get("/health")
def healthcheck():
    return {
        "status": "ok",
        "service": "EDAgent",
        "version": "1.0.0",
        "nonilion_ready": True
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_ui():
    """Interactive browser-based playground for evaluating EDAgent."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDAgent • Interactive Demo Playground</title>
<style>
  :root {
    --primary: #4338ca;
    --primary-hover: #3730a3;
    --bg: #0f172a;
    --card: #1e293b;
    --card-border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #10b981;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
  }
  .container { max-width: 1000px; margin: 0 auto; }
  .header {
    background: linear-gradient(135deg, #312e81, #4338ca);
    padding: 28px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
  }
  .header h1 { margin: 0 0 8px 0; font-size: 26px; }
  .header p { margin: 0; color: #e0e7ff; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media(max-width: 768px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
  }
  .card h2 { margin-top: 0; font-size: 18px; color: #a5b4fc; }
  .dropzone {
    border: 2px dashed #475569;
    border-radius: 8px;
    padding: 24px;
    text-align: center;
    cursor: pointer;
    background: #0f172a;
    transition: border 0.2s;
  }
  .dropzone:hover { border-color: var(--primary); }
  input[type="text"] {
    width: 100%;
    padding: 10px 12px;
    background: #0f172a;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    color: white;
    box-sizing: border-box;
    margin-top: 6px;
  }
  button {
    background: var(--primary);
    color: white;
    border: none;
    padding: 10px 18px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    margin-top: 12px;
    transition: background 0.2s;
  }
  button:hover { background: var(--primary-hover); }
  .btn-sample { background: #334155; margin-right: 6px; margin-top: 8px; font-size: 13px; }
  .btn-sample:hover { background: #475569; }
  pre {
    background: #090d16;
    padding: 14px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 13px;
    color: #e2e8f0;
    max-height: 400px;
  }
  .heatmap-view { max-width: 100%; border-radius: 8px; margin-top: 12px; border: 1px solid var(--card-border); }
  .badge { background: #065f46; color: #6ee7b7; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🤖 EDAgent • Autonomous EDA & Nonilion Bot</h1>
    <p>INNOVORTEX 4.0 Hackathon • 11 LangGraph Agents • 4 Execution Stages</p>
  </div>

  <div class="grid">
    <div>
      <div class="card">
        <h2>1. Ingest Spreadsheet</h2>
        <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
          <span id="file-label">📁 Click to browse or drop CSV, XLSX, or JSON</span>
          <input type="file" id="file-input" style="display:none" accept=".csv,.xlsx,.xls,.json" onchange="handleFileSelected(event)">
        </div>

        <div style="margin-top: 12px;">
          <label style="font-size: 13px; color: var(--text-muted);">Quick Pre-loaded Samples:</label><br>
          <button type="button" class="btn-sample" onclick="loadSample('churn')">Customer Churn (CSV)</button>
          <button type="button" class="btn-sample" onclick="loadSample('housing')">House Prices (Excel)</button>
          <button type="button" class="btn-sample" onclick="loadSample('salaries')">Salaries (JSON)</button>
        </div>

        <div style="margin-top: 14px;">
          <label style="font-size: 13px; color: var(--text-muted);">Target Column (Optional - will auto-infer if blank):</label>
          <input type="text" id="target-input" placeholder="e.g. churn, price, salary">
        </div>

        <button id="run-btn" onclick="runAnalysis()">🚀 Run EDAgent Pipeline</button>
      </div>

      <div class="card">
        <h2>2. Ask Engine (Follow-up Q&A)</h2>
        <p style="font-size: 13px; color: var(--text-muted); margin-top: 0;">Python computes every number directly against the active dataset.</p>
        <input type="text" id="ask-input" placeholder="e.g. average salary by city?">
        <button onclick="askQuestion()">💬 Ask EDAgent</button>
        <div id="ask-output" style="margin-top: 14px;"></div>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>3. Pipeline Output & Chat Summary</h2>
        <div id="status-spinner" style="display:none; color: #a5b4fc; font-weight: 600;">
          ⏳ Executing 11 agents across 4 stages...
        </div>
        <div id="output-area">
          <p style="color: var(--text-muted);">Upload a spreadsheet to see the analysis summary, correlation heatmap, and model baselines.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let activeFile = null;

function handleFileSelected(e) {
  if (e.target.files.length > 0) {
    activeFile = e.target.files[0];
    document.getElementById('file-label').innerText = 'Selected: ' + activeFile.name;
  }
}

async function loadSample(type) {
  let fname = '';
  if (type === 'churn') fname = 'customer_churn.csv';
  else if (type === 'housing') fname = 'house_prices.xlsx';
  else if (type === 'salaries') fname = 'employee_salaries.json';

  const resp = await fetch('/sample_data/' + fname);
  const blob = await resp.blob();
  activeFile = new File([blob], fname);
  document.getElementById('file-label').innerText = 'Selected Sample: ' + fname;
}

async function runAnalysis() {
  if (!activeFile) {
    alert('Please select or load a sample file first!');
    return;
  }
  const status = document.getElementById('status-spinner');
  status.style.display = 'block';
  status.innerText = '⏳ Spawning 11 LangGraph agents across 4 stages...';

  const formData = new FormData();
  formData.append('file', activeFile);
  const target = document.getElementById('target-input').value.trim();
  if (target) formData.append('target', target);

  try {
    const res = await fetch('/api/direct_analyze', { method: 'POST', body: formData });
    const data = await res.json();
    status.style.display = 'none';

    let html = '<span class="badge">Pipeline Complete</span>';
    html += '<pre>' + data.summary + '</pre>';
    if (data.heatmap_b64) {
      html += '<h3 style="font-size:15px; margin-top:16px;">Correlation Heatmap</h3>';
      html += '<img class="heatmap-view" src="data:image/png;base64,' + data.heatmap_b64 + '">';
    }
    document.getElementById('output-area').innerHTML = html;
  } catch(err) {
    status.style.display = 'none';
    alert('Analysis failed: ' + err);
  }
}

async function askQuestion() {
  const q = document.getElementById('ask-input').value.trim();
  if (!q) return;
  const out = document.getElementById('ask-output');
  out.innerHTML = '<span style="color:#a5b4fc;">Computing answer...</span>';

  try {
    const res = await fetch('/api/direct_ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    out.innerHTML = '<pre>' + data.reply + '</pre>';
  } catch(err) {
    out.innerHTML = '<span style="color:#ef4444;">Error: ' + err + '</span>';
  }
}
</script>
</body>
</html>"""


@app.post("/api/direct_analyze")
async def api_direct_analyze(
    file: UploadFile = File(...),
    target: Optional[str] = Form(None)
):
    """Direct analysis endpoint powering the demo UI."""
    content = await file.read()
    work_dir = tempfile.mkdtemp(prefix="edagent_ui_")
    local_path = os.path.join(work_dir, file.filename or "data.csv")
    with open(local_path, "wb") as f:
        f.write(content)

    df = parse_spreadsheet_file(local_path, file.filename or "data.csv")
    result = run_pipeline(df, specified_target=target, output_dir=work_dir)
    artifacts = export_pipeline_artifacts(result, work_dir)

    # Save to session
    session_store = get_session_store()
    session_store.set(
        key="global:default",
        raw_df=result.raw_df,
        cleaned_df=result.cleaned_df,
        target_col=result.target_col,
        filename=file.filename
    )

    heatmap_b64 = ""
    if os.path.exists(artifacts["heatmap"]):
        with open(artifacts["heatmap"], "rb") as hf:
            heatmap_b64 = base64.b64encode(hf.read()).decode("utf-8")

    return {
        "summary": generate_chat_summary(result),
        "target": result.target_col,
        "is_inferred": result.is_target_inferred,
        "heatmap_b64": heatmap_b64,
        "cleaned_rows": len(result.cleaned_df),
        "original_rows": len(result.raw_df)
    }


@app.post("/api/direct_ask")
async def api_direct_ask(request: Request):
    """Direct follow-up Q&A endpoint powering the demo UI."""
    data = await request.json()
    q = data.get("question", "")
    session = get_session_store().get("global:default")
    reply, code = answer_question_about_data(q, session)
    return {"reply": reply, "code": code}


@app.get("/sample_data/{filename}")
def serve_sample_file(filename: str):
    """Serves sample datasets for the demo UI."""
    path = os.path.join("sample_data", os.path.basename(filename))
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Sample file not found")


def parse_spreadsheet_file(filepath: str, filename: str) -> pd.DataFrame:
    """Parses CSV, XLSX, or JSON file into a pandas DataFrame."""
    fname_lower = filename.lower()
    if fname_lower.endswith(".csv"):
        # Auto-detect comma or semicolon
        try:
            return pd.read_csv(filepath, sep=None, engine="python")
        except Exception:
            return pd.read_csv(filepath)
    elif fname_lower.endswith(".xlsx") or fname_lower.endswith(".xls"):
        return pd.read_excel(filepath)
    elif fname_lower.endswith(".json"):
        try:
            return pd.read_json(filepath)
        except Exception:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            raise ValueError("Unsupported JSON layout for tabular EDA.")
    else:
        # Try CSV as default fallback
        return pd.read_csv(filepath)


def extract_target_from_text(text: str) -> Optional[str]:
    """Extracts explicit target parameter from message, e.g. 'target: churn' or 'predict salary'."""
    m = re.search(r"\b(?:target|predict|label)[:\s=]+([a-zA-Z0-9_\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def clean_mention_text(text: str) -> str:
    """Removes @EDAgent mention tokens from message text."""
    return re.sub(r"@edagent\b", "", text, flags=re.IGNORECASE).strip()


def process_event_background(payload_dict: Dict[str, Any]) -> None:
    """Background worker executing the EDA pipeline or handling Q&A / edge cases."""
    nonilion = get_nonilion_client()
    session_store = get_session_store()

    # Normalize fields from payload
    msg_obj = payload_dict.get("message") or {}
    chat_id = (
        payload_dict.get("chat_id")
        or payload_dict.get("room_id")
        or msg_obj.get("chat_id")
        or msg_obj.get("room_id")
        or msg_obj.get("channel_id")
        or "default_chat"
    )
    user_id = payload_dict.get("user_id") or msg_obj.get("user_id") or "default_user"
    raw_text = payload_dict.get("text") or msg_obj.get("text") or ""
    text = clean_mention_text(raw_text)

    attachments = (
        payload_dict.get("attachments")
        or msg_obj.get("attachments")
        or []
    )

    session_key = session_store.make_key(room_id=chat_id, user_id=user_id)
    session = session_store.get(session_key)

    logger.info(f"Processing event for chat_id={chat_id}, text='{text}', attachments={len(attachments)}")

    # 1. Edge Case: Off-topic query without attachment
    if not attachments and is_off_topic(text):
        nonilion.send_message(chat_id=chat_id, text=get_off_topic_reply(text))
        return

    # 2. Case: Follow-up question on an existing dataset
    if not attachments and session is not None and len(text) > 2:
        reply, code = answer_question_about_data(question=text, session=session)
        nonilion.send_message(chat_id=chat_id, text=reply)
        return

    # 3. Edge Case: No attachment and no active dataset
    if not attachments and session is None:
        nonilion.send_message(chat_id=chat_id, text=get_missing_file_reply())
        return

    # 4. Case: Spreadsheet attached - Execute Full 11-Agent EDA Pipeline!
    att = attachments[0]
    att_dict = att if isinstance(att, dict) else att.model_dump()
    filename = att_dict.get("name") or att_dict.get("filename") or "dataset.csv"

    # Quick acknowledgment to user in chat
    nonilion.send_message(
        chat_id=chat_id,
        text=f"📥 **EDAgent received `{filename}`**!\n"
             f"Spawning 11 LangGraph agents across 4 stages. Running now..."
    )

    work_dir = tempfile.mkdtemp(prefix="edagent_run_")
    local_file_path = os.path.join(work_dir, filename)

    # Download attachment
    try:
        # Check if local mock path or direct file provided
        if "local_path" in att_dict and os.path.exists(att_dict["local_path"]):
            import shutil
            shutil.copyfile(att_dict["local_path"], local_file_path)
        else:
            nonilion.download_attachment(attachment_info=att_dict, dest_path=local_file_path)
    except Exception as e:
        logger.error(f"Download failed for {filename}: {e}")
        nonilion.send_message(
            chat_id=chat_id,
            text=get_corrupted_file_reply(filename, f"Failed to retrieve attachment: {str(e)}")
        )
        return

    # Parse spreadsheet
    try:
        df = parse_spreadsheet_file(filepath=local_file_path, filename=filename)
        if df.empty:
            raise ValueError("Spreadsheet contains 0 rows.")
    except Exception as e:
        logger.error(f"Spreadsheet parsing failed for {filename}: {e}")
        nonilion.send_message(
            chat_id=chat_id,
            text=get_corrupted_file_reply(filename, str(e))
        )
        return

    # Extract target if provided
    specified_target = extract_target_from_text(raw_text)

    # Progress reporting callback function
    def report_progress(stage: int, total_stages: int, done_agents: int, total_agents: int, msg: str):
        nonilion.send_progress(
            chat_id=chat_id,
            stage=stage,
            total_stages=total_stages,
            completed_agents=done_agents,
            total_agents=total_agents,
            details=msg
        )

    # Run the 11-agent 4-stage pipeline
    try:
        result = run_pipeline(
            df=df,
            specified_target=specified_target,
            output_dir=work_dir,
            progress_callback=report_progress
        )
    except Exception as e:
        logger.exception("Pipeline execution failed:")
        nonilion.send_message(
            chat_id=chat_id,
            text=f"❌ **Pipeline Execution Error**: {str(e)}\n"
                 "Please check your dataset formatting or try again."
        )
        return

    # Export artifacts
    artifacts = export_pipeline_artifacts(result, work_dir)

    # Save to session memory for follow-up questions
    session_store.set(
        key=session_key,
        raw_df=result.raw_df,
        cleaned_df=result.cleaned_df,
        target_col=result.target_col,
        result_summary={"rows": len(result.cleaned_df), "target": result.target_col},
        filename=filename
    )

    # Dispatch responses to Nonilion chat
    # 1. Chat Summary
    chat_summary = generate_chat_summary(result)
    nonilion.send_message(chat_id=chat_id, text=chat_summary)

    # 2. Correlation Heatmap
    if os.path.exists(artifacts["heatmap"]):
        nonilion.send_image(
            chat_id=chat_id,
            image_path=artifacts["heatmap"],
            caption="Correlation Heatmap (EDAgent)"
        )

    # 3. HTML Report
    if os.path.exists(artifacts["html_report"]):
        nonilion.send_file(
            chat_id=chat_id,
            file_path=artifacts["html_report"],
            filename="report.html",
            caption="Full Interactive EDA Report"
        )

    # 4. Cleaned CSV
    if os.path.exists(artifacts["cleaned_csv"]):
        nonilion.send_file(
            chat_id=chat_id,
            file_path=artifacts["cleaned_csv"],
            filename="cleaned_data.csv",
            caption="Cleaned Dataset (CSV)"
        )

    # 5. Cleaning Script
    if os.path.exists(artifacts["clean_py"]):
        nonilion.send_file(
            chat_id=chat_id,
            file_path=artifacts["clean_py"],
            filename="clean.py",
            caption="Reproducible Python Cleaning Script"
        )

    logger.info(f"Completed analysis and dispatched artifacts for {filename} in {chat_id}.")


@app.post("/nonilion/webhook")
async def nonilion_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Primary webhook endpoint for Nonilion bot integration.
    
    Returns 200 OK immediately and processes the heavy pipeline asynchronously.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    logger.info(f"Incoming webhook event received: {body.get('event', 'message')}")

    # Enqueue background task
    background_tasks.add_task(process_event_background, body)

    # Immediate acknowledgment
    return JSONResponse(
        status_code=200,
        content={"status": "acknowledged", "bot": "EDAgent"}
    )
