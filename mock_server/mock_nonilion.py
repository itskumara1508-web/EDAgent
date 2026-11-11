"""Mock Nonilion Server simulating the Nonilion platform API.

Simulates:
1. User sending messages and spreadsheet attachments to EDAgent's webhook.
2. EDAgent sending messages, stage progress bars, images, and file attachments back to Nonilion.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_nonilion")

mock_app = FastAPI(title="Mock Nonilion Server", version="1.0.0")

# In-memory storage for received messages and uploaded files
RECEIVED_MESSAGES: List[Dict[str, Any]] = []
RECEIVED_ATTACHMENTS: List[Dict[str, Any]] = []
STORED_FILES: Dict[str, str] = {}  # file_id -> local_path


@mock_app.get("/health")
def mock_health():
    return {"status": "ok", "mock_server": "nonilion"}


@mock_app.post("/mock/reset")
def reset_mock_state():
    RECEIVED_MESSAGES.clear()
    RECEIVED_ATTACHMENTS.clear()
    STORED_FILES.clear()
    return {"status": "reset_complete"}


@mock_app.get("/mock/messages")
def get_mock_messages():
    return {"count": len(RECEIVED_MESSAGES), "messages": RECEIVED_MESSAGES}


@mock_app.get("/mock/attachments")
def get_mock_attachments():
    return {"count": len(RECEIVED_ATTACHMENTS), "attachments": RECEIVED_ATTACHMENTS}


# =====================================================================
# NONILION BOT API EMULATION ENDPOINTS (Called by EDAgent's nonilion_client.py)
# =====================================================================

@mock_app.post("/api/v1/messages")
async def post_message(request: Request):
    """Receives text messages or progress updates from EDAgent."""
    data = await request.json()
    logger.info(f"Mock Nonilion received message for chat={data.get('chat_id')}: {data.get('text')[:80]}...")
    RECEIVED_MESSAGES.append({
        "type": "text",
        "chat_id": data.get("chat_id"),
        "text": data.get("text"),
        "bot_id": data.get("bot_id")
    })
    return {"status": "sent", "message_id": f"mock_msg_{len(RECEIVED_MESSAGES)}"}


@mock_app.post("/api/v1/messages/media")
async def post_media(
    chat_id: str = Form(...),
    caption: Optional[str] = Form(None),
    media_type: Optional[str] = Form("image"),
    file: UploadFile = File(...)
):
    """Receives image uploads (e.g. correlation heatmap) from EDAgent."""
    content = await file.read()
    temp_path = os.path.join(tempfile.gettempdir(), f"mock_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(content)

    rec = {
        "type": "media",
        "chat_id": chat_id,
        "filename": file.filename,
        "media_type": media_type,
        "caption": caption,
        "bytes": len(content),
        "local_saved_path": temp_path
    }
    logger.info(f"Mock Nonilion received media: {file.filename} ({len(content)} bytes)")
    RECEIVED_ATTACHMENTS.append(rec)
    return {"status": "uploaded", "media_id": f"mock_media_{len(RECEIVED_ATTACHMENTS)}"}


@mock_app.post("/api/v1/messages/attachments")
async def post_attachment(
    chat_id: str = Form(...),
    caption: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """Receives file attachments (report.html, cleaned_data.csv, clean.py) from EDAgent."""
    content = await file.read()
    temp_path = os.path.join(tempfile.gettempdir(), f"mock_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(content)

    rec = {
        "type": "file",
        "chat_id": chat_id,
        "filename": file.filename,
        "caption": caption,
        "bytes": len(content),
        "local_saved_path": temp_path
    }
    logger.info(f"Mock Nonilion received file attachment: {file.filename} ({len(content)} bytes)")
    RECEIVED_ATTACHMENTS.append(rec)
    return {"status": "uploaded", "attachment_id": f"mock_file_{len(RECEIVED_ATTACHMENTS)}"}


@mock_app.get("/api/v1/files/{file_id}/download")
def download_mock_file(file_id: str):
    """Allows EDAgent to download a mock user file attachment."""
    if file_id in STORED_FILES and os.path.exists(STORED_FILES[file_id]):
        return FileResponse(STORED_FILES[file_id])
    raise HTTPException(status_code=404, detail="File not found")


# =====================================================================
# SIMULATION TRIGGER: Dispatches webhook to EDAgent
# =====================================================================

def simulate_user_send(
    edagent_webhook_url: str,
    chat_id: str,
    user_id: str,
    text: str,
    attachment_path: Optional[str] = None
) -> Dict[str, Any]:
    """Helper that dispatches a simulated webhook event to EDAgent."""
    attachments_payload = []
    if attachment_path and os.path.exists(attachment_path):
        filename = os.path.basename(attachment_path)
        file_id = f"file_{len(STORED_FILES) + 1}"
        STORED_FILES[file_id] = os.path.abspath(attachment_path)
        attachments_payload.append({
            "name": filename,
            "filename": filename,
            "file_id": file_id,
            "download_url": f"http://127.0.0.1:8001/api/v1/files/{file_id}/download",
            "local_path": os.path.abspath(attachment_path)
        })

    payload = {
        "event": "message",
        "chat_id": chat_id,
        "room_id": chat_id,
        "user_id": user_id,
        "text": text,
        "attachments": attachments_payload
    }

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(edagent_webhook_url, json=payload)
        return resp.json()
