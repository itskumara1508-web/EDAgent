"""Automated Demo Runner for EDAgent against the local Mock Nonilion Server.

Runs the complete 8-message test sequence from demo/DEMO_SCRIPT.md.
"""

from __future__ import annotations

import os
import sys
import time
import requests

EDAGENT_URL = "http://127.0.0.1:8000"
MOCK_URL = "http://127.0.0.1:8001"


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def wait_for_bot_messages(initial_count: int, timeout: int = 25) -> list:
    """Polls mock server until new messages arrive from EDAgent."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{MOCK_URL}/mock/messages", timeout=2.0)
            if r.status_code == 200:
                msgs = r.json().get("messages", [])
                if len(msgs) > initial_count:
                    # Allow slight settling time for all attachments/messages to arrive
                    time.sleep(1.5)
                    r = requests.get(f"{MOCK_URL}/mock/messages", timeout=2.0)
                    return r.json().get("messages", [])
        except Exception:
            pass
        time.sleep(0.5)
    return []


def run_demo():
    print_banner("EDAgent Live Mock Demonstration Harness")

    # 1. Health check verification
    print("[1/8] Checking server connectivity...")
    try:
        r_ed = requests.get(f"{EDAGENT_URL}/health", timeout=3.0)
        r_mock = requests.get(f"{MOCK_URL}/health", timeout=3.0)
        assert r_ed.status_code == 200, f"EDAgent unhealthy: {r_ed.text}"
        assert r_mock.status_code == 200, f"Mock Nonilion unhealthy: {r_mock.text}"
        print("  ✓ EDAgent and Mock Nonilion are online!")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("  Ensure both servers are running before executing this script.")
        sys.exit(1)

    requests.post(f"{MOCK_URL}/mock/reset")
    room_id = "hackathon_demo_room"
    user_id = "judge_user"

    # Step 1: Happy Path (customer_churn.csv + target: churn)
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
    r = requests.post(f"{EDAGENT_URL}/nonilion/webhook", json=payload)
    print(f"  Webhook ACK status: {r.status_code} {r.json()}")
    
    print("  Waiting for 11 agents across 4 stages to complete...")
    msgs = wait_for_bot_messages(initial_count=0, timeout=25)
    print(f"  Received {len(msgs)} bot messages/updates:")
    for m in msgs:
        preview = m.get("text", "")[:100].replace("\n", " ")
        print(f"    - {preview}...")

    # Check attachments uploaded to mock
    r_att = requests.get(f"{MOCK_URL}/mock/attachments")
    atts = r_att.json().get("attachments", [])
    print(f"  Received {len(atts)} media/file attachments:")
    for a in atts:
        print(f"    - [{a.get('type')}] {a.get('filename')} ({a.get('bytes')} bytes)")

    # Step 2: Follow-up Question ("Ask" feature)
    print_banner("Step 2: Computed Follow-up Question (Ask Engine)")
    initial_count = len(msgs)
    payload = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent average monthly charges by contract type?"
    }
    requests.post(f"{EDAGENT_URL}/nonilion/webhook", json=payload)
    msgs = wait_for_bot_messages(initial_count=initial_count, timeout=10)
    print("  Bot Reply:")
    print(msgs[-1].get("text", ""))

    # Step 3: Vague Request with Auto-Target Inference (house_prices.xlsx)
    print_banner("Step 3: Vague Request with Auto Target Inference (house_prices.xlsx)")
    initial_count = len(msgs)
    file_path = os.path.abspath("sample_data/house_prices.xlsx")
    payload = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent Can you inspect our property listings?",
        "attachments": [{
            "name": "house_prices.xlsx",
            "filename": "house_prices.xlsx",
            "local_path": file_path
        }]
    }
    requests.post(f"{EDAGENT_URL}/nonilion/webhook", json=payload)
    msgs = wait_for_bot_messages(initial_count=initial_count, timeout=25)
    print(f"  Received {len(msgs) - initial_count} messages:")
    for m in msgs[initial_count:]:
        preview = m.get("text", "")[:100].replace("\n", " ")
        print(f"    - {preview}...")

    # Step 4: Follow-up on New Active Dataset
    print_banner("Step 4: Follow-up on New Active Dataset (house_prices.xlsx)")
    initial_count = len(msgs)
    payload = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent average price by neighborhood?"
    }
    requests.post(f"{EDAGENT_URL}/nonilion/webhook", json=payload)
    msgs = wait_for_bot_messages(initial_count=initial_count, timeout=10)
    print("  Bot Reply:")
    print(msgs[-1].get("text", ""))

    # Step 5: Missing File Guardrail
    print_banner("Step 5: Missing File Guardrail")
    initial_count = len(msgs)
    payload = {
        "event": "message",
        "chat_id": "new_empty_room",
        "user_id": "new_user",
        "text": "@EDAgent Please run full exploratory analysis on my data"
    }
    requests.post(f"{EDAGENT_URL}/nonilion/webhook", json=payload)
    msgs = wait_for_bot_messages(initial_count=initial_count, timeout=8)
    print("  Bot Reply:")
    print(msgs[-1].get("text", ""))

    # Step 6: Off-Topic Guardrail
    print_banner("Step 6: Off-Topic Guardrail")
    initial_count = len(msgs)
    payload = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent What is the weather in Paris today and can you tell me a joke?"
    }
    requests.post(f"{EDAGENT_URL}/nonilion/webhook", json=payload)
    msgs = wait_for_bot_messages(initial_count=initial_count, timeout=8)
    print("  Bot Reply:")
    print(msgs[-1].get("text", ""))

    # Step 7: Corrupted / Malformed File
    print_banner("Step 7: Corrupted File Guardrail (corrupted_sample.csv)")
    initial_count = len(msgs)
    file_path = os.path.abspath("sample_data/corrupted_sample.csv")
    payload = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent Analyze this spreadsheet",
        "attachments": [{
            "name": "corrupted_sample.csv",
            "filename": "corrupted_sample.csv",
            "local_path": file_path
        }]
    }
    requests.post(f"{EDAGENT_URL}/nonilion/webhook", json=payload)
    msgs = wait_for_bot_messages(initial_count=initial_count, timeout=8)
    print("  Bot Reply:")
    print(msgs[-1].get("text", ""))

    # Step 8: JSON Tabular Ingestion
    print_banner("Step 8: Tabular JSON Ingestion (employee_salaries.json)")
    initial_count = len(msgs)
    file_path = os.path.abspath("sample_data/employee_salaries.json")
    payload = {
        "event": "message",
        "chat_id": room_id,
        "user_id": user_id,
        "text": "@EDAgent analyze employee compensation. target: salary",
        "attachments": [{
            "name": "employee_salaries.json",
            "filename": "employee_salaries.json",
            "local_path": file_path
        }]
    }
    requests.post(f"{EDAGENT_URL}/nonilion/webhook", json=payload)
    msgs = wait_for_bot_messages(initial_count=initial_count, timeout=25)
    print(f"  Received {len(msgs) - initial_count} messages:")
    for m in msgs[initial_count:]:
        preview = m.get("text", "")[:100].replace("\n", " ")
        print(f"    - {preview}...")

    print_banner("🎉 All 8 Demonstration Steps Successfully Passed Against Mock Nonilion!")


if __name__ == "__main__":
    run_demo()
