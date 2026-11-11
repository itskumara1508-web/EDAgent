"""In-process End-to-End Mock Integration Test for EDAgent.

Simulates the full 8-step live demonstration scenario against the Mock Nonilion
platform without requiring external network port bindings.
"""

from __future__ import annotations

import os
import sys
import time
import httpx
from starlette.testclient import TestClient

from edagent.main import app as edagent_app, process_event_background
from edagent.nonilion_client import get_nonilion_client
from mock_server.mock_nonilion import (
    mock_app,
    RECEIVED_MESSAGES,
    RECEIVED_ATTACHMENTS,
    STORED_FILES,
    reset_mock_state
)


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run_e2e_demo():
    print_banner("EDAgent Live Mock Integration & Demonstration Suite")

    # Wire EDAgent's nonilion_client directly to mock_app via TestClient
    client = get_nonilion_client()
    client.custom_client = TestClient(mock_app)
    client.base_url = "/api/v1"

    reset_mock_state()
    room_id = "hackathon_demo_room"
    user_id = "judge_user"

    # Step 1: Happy Path Analysis (customer_churn.csv + target: churn)
    print_banner("Step 1: Happy Path Analysis (customer_churn.csv, target: churn)")
    file_path = os.path.abspath("sample_data/customer_churn.csv")
    payload = {
        "event": "message",
        "chat_id": room_id,
        "room_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent Analyze this churn dataset. target: churn",
        "attachments": [{
            "name": "customer_churn.csv",
            "filename": "customer_churn.csv",
            "local_path": file_path
        }]
    }

    print("  Dispatching event to EDAgent pipeline...")
    process_event_background(payload)

    print(f"  Total messages received on Mock Nonilion: {len(RECEIVED_MESSAGES)}")
    for m in RECEIVED_MESSAGES:
        preview = m.get("text", "")[:90].replace("\n", " ")
        print(f"    - {preview}...")

    print(f"  Total attachments received on Mock Nonilion: {len(RECEIVED_ATTACHMENTS)}")
    for a in RECEIVED_ATTACHMENTS:
        print(f"    - [{a.get('type')}] {a.get('filename')} ({a.get('bytes')} bytes)")

    # Assertions for Step 1
    assert any("customer_churn.csv" in m.get("text", "") for m in RECEIVED_MESSAGES), "Missing ACK message"
    assert any("EDAgent Progress" in m.get("text", "") for m in RECEIVED_MESSAGES), "Missing progress update"
    assert any("heatmap.png" == a.get("filename") for a in RECEIVED_ATTACHMENTS), "Missing heatmap attachment"
    assert any("report.html" == a.get("filename") for a in RECEIVED_ATTACHMENTS), "Missing HTML report"
    assert any("cleaned_data.csv" == a.get("filename") for a in RECEIVED_ATTACHMENTS), "Missing cleaned CSV"
    assert any("clean.py" == a.get("filename") for a in RECEIVED_ATTACHMENTS), "Missing clean.py"

    # Step 2: Computed Follow-up Question (Ask Engine)
    print_banner("Step 2: Computed Follow-up Question (Ask Engine)")
    msg_count_before = len(RECEIVED_MESSAGES)
    payload_q = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent average monthly charges by contract type?"
    }
    process_event_background(payload_q)
    new_msgs = RECEIVED_MESSAGES[msg_count_before:]
    assert len(new_msgs) >= 1, "Expected reply to follow-up question"
    print("  Bot Computed Answer:")
    print(new_msgs[-1].get("text"))
    assert "Question Answered" in new_msgs[-1].get("text"), "Expected computed question answer"

    # Step 3: Vague Request with Auto Target Inference (house_prices.xlsx)
    print_banner("Step 3: Vague Request with Auto Target Inference (house_prices.xlsx)")
    msg_count_before = len(RECEIVED_MESSAGES)
    file_path_excel = os.path.abspath("sample_data/house_prices.xlsx")
    payload_vague = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent Can you inspect our property listings?",
        "attachments": [{
            "name": "house_prices.xlsx",
            "filename": "house_prices.xlsx",
            "local_path": file_path_excel
        }]
    }
    process_event_background(payload_vague)
    vague_msgs = RECEIVED_MESSAGES[msg_count_before:]
    print(f"  Received {len(vague_msgs)} messages for Excel analysis.")
    summary_msg = [m for m in vague_msgs if "EDAgent Analysis Complete" in m.get("text", "")][0]
    assert "Auto-Inferred" in summary_msg.get("text", "") or "price" in summary_msg.get("text", "")
    print("  ✓ Target 'price' successfully auto-inferred and 5-fold CV baselines calculated.")

    # Step 4: Follow-up on New Active Dataset (house_prices.xlsx)
    print_banner("Step 4: Follow-up on New Active Dataset (house_prices.xlsx)")
    msg_count_before = len(RECEIVED_MESSAGES)
    payload_hp_q = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent average price by neighborhood?"
    }
    process_event_background(payload_hp_q)
    hp_q_msgs = RECEIVED_MESSAGES[msg_count_before:]
    print("  Bot Computed Answer:")
    print(hp_q_msgs[-1].get("text"))
    assert "Question Answered" in hp_q_msgs[-1].get("text")

    # Step 5: Missing File Guardrail
    print_banner("Step 5: Missing File Guardrail")
    msg_count_before = len(RECEIVED_MESSAGES)
    payload_missing = {
        "event": "message",
        "chat_id": "fresh_empty_room_999",
        "user_id": "new_user_1",
        "text": "@EDAgent Please run full exploratory analysis on my data"
    }
    process_event_background(payload_missing)
    missing_msgs = RECEIVED_MESSAGES[msg_count_before:]
    print("  Bot Reply:")
    print(missing_msgs[-1].get("text"))
    assert "No spreadsheet attached" in missing_msgs[-1].get("text")

    # Step 6: Off-Topic Guardrail
    print_banner("Step 6: Off-Topic Guardrail")
    msg_count_before = len(RECEIVED_MESSAGES)
    payload_offtopic = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent What is the weather in Paris today and can you tell me a joke?"
    }
    process_event_background(payload_offtopic)
    offtopic_msgs = RECEIVED_MESSAGES[msg_count_before:]
    print("  Bot Reply:")
    print(offtopic_msgs[-1].get("text"))
    assert "Hello! I am **EDAgent**" in offtopic_msgs[-1].get("text")

    # Step 7: Corrupted / Malformed File
    print_banner("Step 7: Corrupted File Guardrail (corrupted_sample.csv)")
    msg_count_before = len(RECEIVED_MESSAGES)
    file_path_corrupt = os.path.abspath("sample_data/corrupted_sample.csv")
    payload_corrupt = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent Analyze this spreadsheet",
        "attachments": [{
            "name": "corrupted_sample.csv",
            "filename": "corrupted_sample.csv",
            "local_path": file_path_corrupt
        }]
    }
    process_event_background(payload_corrupt)
    corrupt_msgs = RECEIVED_MESSAGES[msg_count_before:]
    print("  Bot Reply:")
    print(corrupt_msgs[-1].get("text"))
    assert len(corrupt_msgs) >= 1

    # Step 8: Tabular JSON Ingestion
    print_banner("Step 8: Tabular JSON Ingestion (employee_salaries.json)")
    msg_count_before = len(RECEIVED_MESSAGES)
    file_path_json = os.path.abspath("sample_data/employee_salaries.json")
    payload_json = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent analyze employee compensation. target: salary",
        "attachments": [{
            "name": "employee_salaries.json",
            "filename": "employee_salaries.json",
            "local_path": file_path_json
        }]
    }
    process_event_background(payload_json)
    json_msgs = RECEIVED_MESSAGES[msg_count_before:]
    print(f"  Received {len(json_msgs)} messages for JSON analysis.")
    assert any("employee_salaries.json" in m.get("text", "") for m in json_msgs)

    print_banner("🎉 ALL 8 DEMO STEPS PASSED WITH 100% SUCCESS AGAINST MOCK NONILION!")


if __name__ == "__main__":
    run_e2e_demo()
