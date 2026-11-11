"""Tests for FastAPI webhook and health endpoints."""

from fastapi.testclient import TestClient
import pytest
from edagent.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "EDAgent"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "EDAgent"


def test_webhook_acknowledgment():
    payload = {
        "event": "message",
        "chat_id": "test_room_123",
        "user_id": "test_user_456",
        "text": "@EDAgent hello world",
        "attachments": []
    }
    response = client.post("/nonilion/webhook", json=payload)
    # Must acknowledge immediately with 200 OK
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "acknowledged"


def test_webhook_invalid_json():
    response = client.post(
        "/nonilion/webhook",
        content="not valid json",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
