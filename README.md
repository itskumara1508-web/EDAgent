# EDAgent 🤖📊
[![CI](https://github.com/itskumara1508-web/EDAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/itskumara1508-web/EDAgent/actions/workflows/ci.yml)
### Autonomous 11-Agent 4-Stage Exploratory Data Analysis Bot on Nonilion
**Track**: INNOVORTEX 4.0 — *"Best AI Agent on Nonilion"*  
**Route**: Connect a bot (Python + LangGraph + FastAPI)  
**Live Platform**: [www.nonilion.com](https://www.nonilion.com)

---

## 💡 What EDAgent Does
EDAgent turns any spreadsheet (`.csv`, `.xlsx`, `.json`) into a complete, professional Exploratory Data Analysis in minutes. A multi-agent team of **11 LangGraph agents** collaborates across **4 distinct stages**:

1. **Stage 1 (Parallel)**: `Schema Agent`, `Quality Agent`, `Statistics Agent`, and `Outliers Agent`.
2. **Stage 2 (Sequential)**: `Data Cleaning Agent` (generates cleaning code executed in a 60-second sandbox under strict safety guardrails with 1 self-correction retry).
3. **Stage 3 (Parallel)**: `Correlation Agent` (generates correlation heatmap image), `Feature Importance Agent` (Random Forest + Mutual Information), and `Baseline Models Agent` (**5-fold cross-validation** across multiple standard architectures).
4. **Stage 4 (Sequential)**: `Synthesis Agent`, `Model Recommendation Agent` (grounded in the real CV scores), and `Feature Engineering Agent`.

### Deliverables Produced:
- 💬 **Markdown Chat Summary**: With structured profile tables, top features, and 5-fold CV score comparison.
- 🖼️ **Correlation Heatmap**: High-resolution PNG image delivered directly to the chat.
- 📄 **report.html**: Complete, standalone interactive HTML report.
- 📊 **cleaned_data.csv**: Imputed, validated dataset adhering to safety guardrails.
- 🐍 **clean.py**: Reproducible Python cleaning script.
- 🧠 **Computed "Ask" Answers**: Answers follow-up queries (e.g. `@EDAgent average salary by city?`) by executing Python queries against the room's active dataset.

---

## 🛡️ Core Design Invariants

1. **Python computes every number**: The LLM explains and generates code; all summary statistics, missing percentages, correlations, and cross-validation metrics are strictly computed using `pandas`, `numpy`, and `scikit-learn`.
2. **60-Second Sandboxed Execution & Strict Guardrails**:
   - **No deleted columns**: All original columns must be preserved in cleaned output.
   - **Max 5% rows removed**: Mass row deletion is rejected in favor of imputation.
   - **Target untouched**: Target values and labels are strictly preserved without modification or leakage.
   - **Security sandbox**: Statically verified AST blocks dangerous modules (`os`, `subprocess`, socket, etc.).
   - **Self-Correction**: Exactly 1 automated retry if generated code fails execution or guardrail checks.
3. **5-Fold Cross-Validation**: Baseline models are trained with 5-fold CV (Stratified for classification, KFold for regression) to ensure unbiased, real benchmark scores.
4. **Fast Webhook Acknowledgment (< 1s)**: Returns `200 OK` immediately and executes the 11-agent pipeline in background tasks while posting real-time stage progress updates (`Stage 1/4...`, `Stage 2/4...`).

---

## 📂 Repository Structure

```
EDAgent/
├── Dockerfile                    # Production container for Render deployment
├── render.yaml                   # Infrastructure-as-code for Render
├── requirements.txt              # Production Python dependencies
├── .env.example                  # Environment configuration template
├── README.md                     # This file
│
├── edagent/
│   ├── __init__.py
│   ├── main.py                   # FastAPI service (POST /nonilion/webhook, GET /health)
│   ├── nonilion_client.py        # Nonilion Bot API adapter (with clear TODOs)
│   ├── pipeline.py               # 11-agent 4-stage LangGraph EDA pipeline
│   ├── sandbox.py                # 60s execution sandbox with strict guardrails
│   ├── target_inference.py       # Heuristic & statistical target inference
│   ├── report_generator.py       # Exports report.html, cleaned CSV, clean.py, summary
│   ├── session_store.py          # Per-room/user dataset memory for follow-up questions
│   ├── ask_engine.py             # Computes exact answers to natural language data questions
│   ├── edge_cases.py             # Handles missing/corrupt files, off-topic requests
│   └── llm_provider.py           # Claude API (hosted) vs Ollama (local) provider
│
├── mock_server/
│   ├── mock_nonilion.py          # Local Mock Nonilion Server
│   └── run_mock_demo.py          # Automated 8-message demo harness
│
├── sample_data/
│   ├── customer_churn.csv        # Classification demo dataset
│   ├── house_prices.xlsx         # Regression demo dataset
│   ├── employee_salaries.json    # JSON tabular demo dataset
│   └── corrupted_sample.csv      # Error-handling demo dataset
│
├── tests/
│   ├── test_edagent.py           # Pipeline, sandbox & guardrail tests
│   ├── test_webhook.py           # Webhook & healthcheck tests
│   ├── test_file_intake.py       # CSV, XLSX, JSON intake tests
│   ├── test_edge_cases.py        # Off-topic, corrupt, missing file tests
│   └── test_ask_engine.py        # Computed follow-up question tests
│
└── demo/
    └── DEMO_SCRIPT.md            # Live 8-message demonstration script
```

---

## ⚡ Quick Start & Local Testing

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Test Suite
```bash
pytest tests/ -v
```

### 3. Run with Local Mock Nonilion Server
In Terminal 1 (Start Mock Nonilion):
```bash
uvicorn mock_server.mock_nonilion:mock_app --port 8001
```

In Terminal 2 (Start EDAgent):
```bash
NONILION_BASE_URL=http://127.0.0.1:8001/api/v1 uvicorn edagent.main:app --port 8000
```

In Terminal 3 (Run 8-Step Live Demo):
```bash
python3 mock_server/run_mock_demo.py
```

---

## 📂 Running Custom Datasets

You can run EDAgent on any custom spreadsheet (CSV, Excel `.xlsx`, or JSON tabular array) using multiple interfaces:

### Option 1: Interactive Browser UI
1. Start the server:
   ```bash
   uvicorn edagent.main:app --port 8000
   ```
2. Open **[http://localhost:8000/demo](http://localhost:8000/demo)** in your browser.
3. Drag and drop any `.csv`, `.xlsx`, or `.json` file (or click the quick pre-loaded samples).
4. Specify an optional target column (or leave it blank for auto-inference).
5. Watch the 11-agent 4-stage pipeline run with real-time progress bars, and ask follow-up questions interactively!

### Option 2: Standalone Command-Line Interface (CLI)
Analyze any spreadsheet directly from your terminal:
```bash
# Auto-infer target column
python3 -m edagent path/to/your_data.csv

# Explicitly designate target column
python3 -m edagent path/to/your_data.xlsx --target salary

# Specify custom artifact export directory
python3 -m edagent path/to/your_data.json --target price --output-dir ./my_eda_results
```

Artifacts exported:
- `report.html`: Full standalone interactive report
- `heatmap.png`: High-resolution correlation heatmap image
- `cleaned_data.csv`: Guardrail-validated cleaned dataset
- `clean.py`: Standalone reproducible Python cleaning script

### Option 3: Live on Nonilion.com
In any Nonilion room, channel, or direct message:
- Attach your spreadsheet and mention:
  ```
  @EDAgent analyze this dataset. target: <your_target_column>
  ```
- Ask follow-up questions anytime:
  ```
  @EDAgent average salary by city?
  ```

---

## ☁️ Deployment on Render

1. Create a new **Web Service** on [Render.com](https://render.com) using the connected GitHub repository (or deploy via Blueprints using `render.yaml`).
2. Add your Environment Variables:
   - `ANTHROPIC_API_KEY`: Your Anthropic API Key (for Claude 3.5 Sonnet / Haiku).
   - `NONILION_API_KEY`: Your Nonilion bot token.
   - `NONILION_BOT_ID`: Your registered bot ID on Nonilion.
3. Configure the webhook URL in the Nonilion Bot settings:
   `https://<your-render-subdomain>.onrender.com/nonilion/webhook`

> ⚠️ **Keep-Alive Note**: Render free web services sleep after 15 minutes of inactivity. Set up a free ping monitor (e.g., [Cron-Job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com)) to ping `GET https://<your-render-subdomain>.onrender.com/health` every 10 minutes to prevent cold starts during hackathon judging.
