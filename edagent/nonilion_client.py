"""Single adapter module for all Nonilion Bot API interactions.

Target platform: https://www.nonilion.com
Route: 'Connect a bot' (Python + LangGraph)

All credentials are read from environment variables.
Clear '# TODO [NONILION_DOCS]' markers indicate where exact schema/endpoints
can be adjusted once official documentation is provided.
"""

from __future__ import annotations

import contextlib
import logging
import mimetypes
import os
from typing import Any, Dict, Optional
import httpx

logger = logging.getLogger("edagent.nonilion")


class NonilionClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        bot_id: Optional[str] = None,
        timeout: float = 30.0,
        custom_client: Optional[Any] = None
    ):
        self.base_url = (
            base_url
            or os.getenv("NONILION_BASE_URL")
            or "https://www.nonilion.com/api/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("NONILION_API_KEY")
            or os.getenv("NONILION_BOT_TOKEN")
            or "mock-nonilion-key"
        )
        self.bot_id = (
            bot_id
            or os.getenv("NONILION_BOT_ID")
            or "edagent-bot"
        )
        self.timeout = timeout
        self.custom_client = custom_client

    @contextlib.contextmanager
    def _get_client(self):
        if self.custom_client is not None:
            yield self.custom_client
        else:
            with httpx.Client(timeout=self.timeout) as client:
                yield client

    @property
    def headers(self) -> Dict[str, str]:
        # TODO [NONILION_DOCS]: Verify whether Nonilion uses 'Bearer <token>' or custom 'X-Bot-Token'
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Bot-Id": self.bot_id,
            "User-Agent": "EDAgent-Bot/1.0"
        }

    def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to_message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Sends a text message (with Markdown support) to a Nonilion room, DM, or channel."""
        # TODO [NONILION_DOCS]: Confirm endpoint route: /messages vs /rooms/{chat_id}/messages
        url = f"{self.base_url}/messages"
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "bot_id": self.bot_id
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        try:
            with self._get_client() as client:
                resp = client.post(url, json=payload, headers=self.headers)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to send message to Nonilion ({chat_id}): {e}")
            return {"error": str(e), "success": False}

    def send_progress(
        self,
        chat_id: str,
        stage: int,
        total_stages: int,
        completed_agents: int,
        total_agents: int,
        details: str
    ) -> Dict[str, Any]:
        """Posts an interim progress update to keep the user informed during pipeline execution."""
        # Progress bar generation (e.g. [██████░░░░] 45%)
        pct = int((completed_agents / max(1, total_agents)) * 100)
        filled = int((pct / 100) * 8)
        bar = "█" * filled + "░" * (8 - filled)
        
        text = (
            f"⏳ **EDAgent Progress**: Stage {stage}/{total_stages} • [{bar}] {pct}%\n"
            f"*{completed_agents}/{total_agents} agents completed*\n"
            f"> {details}"
        )
        return self.send_message(chat_id=chat_id, text=text)

    def send_image(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """Uploads and sends an image (e.g. correlation heatmap) to the chat."""
        # TODO [NONILION_DOCS]: Confirm whether image upload endpoint is /media/upload or multipart /messages
        url = f"{self.base_url}/messages/media"
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}", "success": False}

        filename = os.path.basename(image_path)
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/png"

        try:
            with open(image_path, "rb") as f:
                files = {"file": (filename, f, mime_type)}
                data = {"chat_id": chat_id, "caption": caption or "", "media_type": "image"}
                with self._get_client() as client:
                    resp = client.post(url, data=data, files=files, headers=self.headers)
                    resp.raise_for_status()
                    return resp.json()
        except Exception as e:
            logger.error(f"Failed to upload image to Nonilion: {e}")
            # Fallback attempt as standard file or base64 if media endpoint fails
            return {"error": str(e), "success": False}

    def send_file(
        self,
        chat_id: str,
        file_path: str,
        filename: Optional[str] = None,
        caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """Uploads and sends a file attachment (report.html, cleaned_data.csv, clean.py)."""
        # TODO [NONILION_DOCS]: Confirm file attachment multipart parameter name ('file' vs 'attachment')
        url = f"{self.base_url}/messages/attachments"
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "success": False}

        fname = filename or os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                files = {"file": (fname, f, mime_type)}
                data = {"chat_id": chat_id, "caption": caption or ""}
                with self._get_client() as client:
                    resp = client.post(url, data=data, files=files, headers=self.headers)
                    resp.raise_for_status()
                    return resp.json()
        except Exception as e:
            logger.error(f"Failed to upload attachment '{fname}' to Nonilion: {e}")
            return {"error": str(e), "success": False}

    def download_attachment(
        self,
        attachment_info: Dict[str, Any],
        dest_path: str
    ) -> str:
        """Downloads an incoming attachment (CSV/Excel/JSON) from Nonilion to local disk."""
        # TODO [NONILION_DOCS]: Confirm whether attachment provides 'url', 'download_url', or 'file_id'
        url = attachment_info.get("url") or attachment_info.get("download_url")
        file_id = attachment_info.get("file_id")

        if not url and file_id:
            url = f"{self.base_url}/files/{file_id}/download"

        if not url:
            raise ValueError(f"Attachment payload missing download URL or file_id: {attachment_info}")

        with self._get_client() as client:
            resp = client.get(url, headers=self.headers)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)

        return dest_path


# Global singleton client
_default_nonilion_client = NonilionClient()


def get_nonilion_client() -> NonilionClient:
    return _default_nonilion_client
