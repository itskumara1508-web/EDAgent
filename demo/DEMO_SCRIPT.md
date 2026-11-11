# 🎬 EDAgent Live Demonstration Script
### INNOVORTEX 4.0 Hackathon • "Best AI Agent on Nonilion" Track

This script is designed for the live hackathon presentation on **Nonilion.com** (or via the local mock harness `mock_server/run_mock_demo.py`). It covers the **happy path**, **vague requests (auto-inference)**, **computed follow-up queries**, and **edge cases**.

---

## 📌 Scene Overview

| Step | User Prompt | Attachment | Demonstrated Capability |
| :---: | :--- | :--- | :--- |
| **1** | `@EDAgent Analyze this churn dataset. target: churn` | `customer_churn.csv` | **Happy Path**: Fast ACK, live progress updates (11 agents / 4 stages), summary tables, heatmap image, report.html, cleaned_data.csv, clean.py |
| **2** | `@EDAgent average monthly charges by contract type?` | *None* | **Computed Follow-up**: Remembers dataset from Step 1, generates and executes pandas query in sandbox, computes exact numbers |
| **3** | `@EDAgent Can you inspect our property listings?` | `house_prices.xlsx` | **Vague Request**: Auto-infers target (`price`), explains reasoning, runs 5-fold CV regression models |
| **4** | `@EDAgent average price by neighborhood?` | *None* | **Multi-Turn Memory**: Dynamically updates session to newest dataset and computes group statistics |
| **5** | `@EDAgent Please run full exploratory analysis on my data` | *None* | **Missing File Guardrail**: Politely guides user to attach CSV/XLSX/JSON without breaking |
| **6** | `@EDAgent What is the weather in Paris today and can you tell me a joke?` | *None* | **Off-Topic Guardrail**: Explains agent focus is spreadsheet EDA, provides guidance on supported actions |
| **7** | `@EDAgent Analyze this spreadsheet` | `corrupted_sample.csv` | **Corrupted File Guardrail**: Graceful parsing failure explanation with actionable tips |
| **8** | `@EDAgent analyze employee compensation. target: salary` | `employee_salaries.json` | **JSON Tabular Ingestion**: Multi-format support, feature ranking, baseline regression |

---

## 🚀 Step-by-Step Execution Guide

### Message 1: The Happy Path (Classification + Explicit Target)
- **User Action**: Uploads `customer_churn.csv` and types:
  ```
  @EDAgent Analyze this churn dataset. target: churn
  ```
- **What EDAgent Does**:
  1. Replies in < 1 second:
     > 📥 **EDAgent received `customer_churn.csv`**! Spawning 11 LangGraph agents across 4 stages. Running now...
  2. Emits real-time progress updates:
     > ⏳ **Stage 1/4 (4/11 agents done)**: Schema, Quality, Statistics, Outliers complete.
     > ⏳ **Stage 2/4 (5/11 agents done)**: Data cleaning validated in sandbox.
     > ⏳ **Stage 3/4 (8/11 agents done)**: Heatmap generated, 5-fold CV models evaluated.
     > ⏳ **Stage 4/4 (11/11 agents done)**: Synthesis, recommendations & feature engineering complete!
  3. Sends Markdown Chat Summary with Profile & 5-Fold Cross-Validation tables.
  4. Delivers media & file attachments:
     - 🖼️ `heatmap.png`
     - 📄 `report.html` (interactive HTML report)
     - 📊 `cleaned_data.csv`
     - 🐍 `clean.py` (reproducible cleaning code)

---

### Message 2: Computed Follow-up ("Ask" Feature)
- **User Action**: Types directly in the room without re-uploading:
  ```
  @EDAgent average monthly charges by contract type?
  ```
- **What EDAgent Does**:
  1. Identifies the active session for this room (`customer_churn.csv`).
  2. Synthesizes safe Python code:
     ```python
     result = df.groupby('contract')['monthly_charges'].mean().round(2).reset_index()
     ```
  3. Executes in the safe sandbox and returns the real calculated numbers in a table:
     | contract | monthly_charges |
     | :--- | :--- |
     | Month-to-month | 58.07 |
     | One year | 88.62 |
     | Two year | 109.32 |

---

### Message 3: Vague Request with Auto-Target Inference
- **User Action**: Uploads `house_prices.xlsx` without mentioning a target:
  ```
  @EDAgent Can you inspect our property listings?
  ```
- **What EDAgent Does**:
  1. Auto-detects that `price` is the outcome variable:
     > 🎯 **Target Column**: `price` *(Auto-Inferred)*  
     > 💡 *Reason: Inferred 'price' as target because its name directly matches the standard outcome pattern 'price'.*
  2. Evaluates 5-fold CV Regression Baselines (Dummy, Ridge, Random Forest, Gradient Boosting).
  3. Delivers regression metrics (R², RMSE, MAE).

---

### Message 4: Follow-up on New Active Dataset
- **User Action**:
  ```
  @EDAgent average price by neighborhood?
  ```
- **What EDAgent Does**:
  1. Automatically targets `house_prices.xlsx`.
  2. Computes and returns exact mean house prices across Downtown, Suburbs, and Uptown.

---

### Message 5: Missing File Guardrail
- **User Action**:
  ```
  @EDAgent Please run full exploratory analysis on my data
  ```
- **What EDAgent Does**:
  > 📎 **No spreadsheet attached.**  
  > To run an Exploratory Data Analysis, please attach a spreadsheet file (.csv, .xlsx, .json).

---

### Message 6: Off-Topic Guardrail
- **User Action**:
  ```
  @EDAgent What is the weather in Paris today and can you tell me a joke?
  ```
- **What EDAgent Does**:
  > 👋 Hello! I am **EDAgent**, an autonomous Exploratory Data Analysis agent. I specialize in turning spreadsheets into complete machine learning & statistical analyses... Attach a CSV or Excel file to get started!

---

### Message 7: Corrupted / Malformed File
- **User Action**: Uploads `corrupted_sample.csv`:
  ```
  @EDAgent Analyze this spreadsheet
  ```
- **What EDAgent Does**:
  > ❌ **Could not read file `corrupted_sample.csv`**  
  > Explains delimiter inconsistency and suggests verification steps.

---

### Message 8: JSON Tabular Ingestion
- **User Action**: Uploads `employee_salaries.json`:
  ```
  @EDAgent analyze employee compensation. target: salary
  ```
- **What EDAgent Does**:
  - Ingests tabular JSON structure into DataFrame.
  - Runs full 11-agent pipeline and outputs complete analysis.
