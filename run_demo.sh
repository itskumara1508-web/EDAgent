#!/usr/bin/env bash
# =====================================================================
# EDAgent Quickstart & Live Demonstration Launcher
# INNOVORTEX 4.0: "Best AI Agent on Nonilion"
# =====================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "======================================================================"
echo "  🚀 Starting EDAgent - Autonomous EDA Bot for Nonilion"
echo "======================================================================"

# 1. Activate or create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

source .venv/bin/activate
export MPLCONFIGDIR=/tmp

# 2. Run test suite
echo ""
echo "[1/3] Running pytest test suite..."
pytest tests/ -v

# 3. Run mock integration demo
echo ""
echo "[2/3] Running 8-step live demonstration against Mock Nonilion..."
python3 test_mock_integration.py

# 4. Start production server
echo ""
echo "[3/3] Starting EDAgent FastAPI server on http://localhost:8000..."
echo "  - Webhook URL:  http://localhost:8000/nonilion/webhook"
echo "  - Health Check: http://localhost:8000/health"
echo "  - Demo UI:      http://localhost:8000/demo"
echo ""
exec uvicorn edagent.main:app --host 0.0.0.0 --port 8000
