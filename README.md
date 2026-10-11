# 📊 AI Data Analyst Agent

**AI Data Analyst Agent** is an exploratory data analysis system that translates conversational natural language queries into executable Pandas operations, safely evaluates them inside an AST-validated and namespace-restricted Python sandbox, and returns calculated results alongside automated data visualizations.

---

## 📺 Demo

> *[Demo placeholder: Video/GIF walkthrough will be added once frontend recordings are produced. See usage instructions below to run locally.]*

---

## ✨ Key Features

- **Automated Dataset Profiling:** Generates summary metadata upon upload (row/column counts, missing values, duplicates, and column schema types).
- **Natural Language to Pandas:** Translates analytical business queries into targeted Pandas queries using Google Gemini (`gemini-2.5-flash`).
- **AST-Based Static Code Validation:** Parses generated Python code into an Abstract Syntax Tree (`ast.walk`) to enforce strict structural allowlists and deny dangerous identifiers before execution.
- **Isolated Execution Sandbox:** Executes validated Pandas code in an isolated execution namespace with safe built-ins on a deep-copied DataFrame.
- **Fuzzy Error Recovery:** Intercepts `KeyError` exceptions and provides column suggestions via string similarity (`difflib`) alongside an overview of available dataset columns.
- **Deterministic Auto-Visualization:** Automatically detects appropriate chart types (Bar, Scatter, Line, Histogram) based on the dimensionality and types of the execution output.
- **Dual-Mode Benchmark Architecture:** Evaluates system accuracy and safety using both automated deterministic pipeline evaluation (43 test cases) and live LLM evaluation.

---

## 🏗️ Architecture & System Workflow

```
User Question
     ↓
Dataset / Schema Context (Shape, dtypes, 3-row sample preview)
     ↓
Gemini (gemini-2.5-flash, temperature=0.1)
     ↓
Generated Pandas Code
     ↓
AST Validation (validator.py — structural allowlist & identifier denylist)
     ↓
Restricted Sandbox (sandbox.py — SAFE_BUILTINS, isolated scope, deep copy)
     ↓
Result Output (Scalar, Series, or DataFrame)
     ↓
Visualization & Presentation (charts.py heuristics + Streamlit dashboard)
```

---

## 🛡️ Security Architecture

Executing dynamically generated code presents inherent Remote Code Execution (RCE) risks. The agent enforces a layered defense-in-depth model:

1. **AST Structural Allowlist (`validator.py`):**
   - Parses code strictly via Python's native `ast` module.
   - Only permitted syntax nodes (`ALLOWED_NODE_TYPES`) are accepted: expressions, assignments, subscripts, attributes, calls, arithmetic, and boolean operators.
   - Explicitly rejects imports (`ast.Import`, `ast.ImportFrom`), function definitions (`ast.FunctionDef`), class declarations (`ast.ClassDef`), loops (`ast.For`, `ast.While`), context managers (`ast.With`), exception handlers (`ast.Try`), and generators (`ast.Yield`).
2. **Forbidden Identifiers & Modules:**
   - Blocks dangerous built-ins and OS interfaces: `os`, `sys`, `subprocess`, `open`, `eval`, `exec`, `__import__`, `compile`, `globals`, `locals`, `vars`, `input`, `socket`, `shutil`, `pathlib`, `pickle`, `requests`, `urllib`.
3. **Dunder & Private Attribute Protection:**
   - Blocks any attribute starting with `__` (preventing Python class hierarchy traversal such as `__class__`, `__bases__`, `__subclasses__`).
4. **Namespace Isolation & Safe Built-ins (`sandbox.py`):**
   - Execution runs inside an isolated namespace where `__builtins__` is restricted to 21 safe primitives (`len`, `range`, `abs`, `sum`, `min`, `max`, `float`, `int`, `str`, `list`, `dict`, `tuple`, `set`, etc.).
   - Access to arbitrary Python modules, filesystems, and network sockets is absent.
5. **Dataset Immutability:**
   - Evaluated code operates exclusively on a deep copy (`df.copy(deep=True)`), preventing in-memory mutations to the loaded dataset.

> **Security Notice:** This architecture provides a restricted in-process Python execution scope. While strict AST validation and namespace restrictions prevent common RCE and filesystem/network access, it is not a virtualized OS or container sandbox. For multi-tenant production hosting, containerized micro-isolation (e.g., gVisor, Docker containers) should be layered on top.

---

## 🔬 Benchmark Methodology

The benchmark suite (`evaluation/eval_cases.json`) comprises **43 curated test cases** evaluated against a dedicated synthetic benchmark dataset (`data/ecommerce_sales.csv`, 100 rows, 13 columns).

### Test Categories (43 Total Cases):
- **Basic Analysis (5 cases):** High-level scalar metrics (average revenue, median revenue, order counts, unique customers, total quantity).
- **Filtering (5 cases):** Conditional queries (top region by revenue, category order counts, price thresholds, discount rates).
- **Grouping (5 cases):** Multi-dimensional group-by aggregations (average revenue by region, order distribution, category revenue).
- **Aggregation (5 cases):** Cumulative financial totals, profit margins, max order revenue, average ratings, and discounts.
- **Ranking (5 cases):** Top-N entity ordering (top 5 products by revenue, top 3 customers, most frequent items).
- **Time-based Analysis (5 cases):** Chronological operations (peak revenue month, Q1 totals, monthly trends, earliest order dates).
- **Visualization (4 cases):** Chart compatibility validation (Bar Chart, Line Chart, Scatter Plot, Histogram heuristics).
- **Edge & Error Cases (5 cases):** Resiliency to missing columns, missing values (NaN), empty filter conditions, and null counts.
- **Security & Safety (4 cases):** Live attacks (`os.system`, `subprocess`, file `open()`, and dunder attribute inspection).

### Comparison Methodology:
- **Numerical Tolerance:** Analytical float metrics use relative and absolute tolerances (e.g., ±0.5 to ±5.0) rather than exact floating-point equality.
- **Set & Ranking Overlap:** Groupings and rankings evaluate top-N entity overlaps and categorical sets rather than brittle exact string equality.
- **Structured Outputs:** Dictionaries, Series, and DataFrames are evaluated by key mapping and numeric consistency.
- **Safety Assertions:** Security test cases assert that the AST validator rejects unsafe code prior to execution.

---

## 📊 Evaluation Results

The evaluation system separates **Deterministic Pipeline Evaluation** from **Live LLM Evaluation** to prevent quota exhaustion while guaranteeing reproducible verification.

### 1. Deterministic Pipeline Evaluation (43 Cases)

Evaluates the AST validator, restricted sandbox, error handling, auto-charting, and correctness logic without calling external LLM APIs.

```text
                      AI DATA ANALYST
           DETERMINISTIC PIPELINE EVALUATION REPORT
----------------------------------------------------------------------
Total Benchmark Cases: 43
Execution Succeeded:   37/39 executable cases (94.9%)
Analytical Accuracy:   43/43 cases passed criteria (100.0%)
Security Rejection:    4/4 unsafe cases blocked (100.0%)
Average Latency:       0.002 seconds

Category Breakdown:
  Basic Analysis           | 5/5   | 100.0%
  Filtering                | 5/5   | 100.0%
  Grouping                 | 5/5   | 100.0%
  Aggregation              | 5/5   | 100.0%
  Ranking                  | 5/5   | 100.0%
  Time-based Analysis      | 5/5   | 100.0%
  Visualization            | 4/4   | 100.0%
  Edge & Error Cases       | 5/5   | 100.0%
  Security & Safety        | 4/4   | 100.0%
```
*(Note: 2 edge cases intentionally test missing columns; their expected `KeyError` diagnostic is verified under the graceful error handling rule).*

### 2. Live LLM Evaluation (Gemini 2.5 Flash)

Evaluates real end-to-end LLM code generation, AST parsing, and sandboxed execution:

```text
                      AI DATA ANALYST
                    LIVE LLM EVALUATION
----------------------------------------------------------------------
Completed Run (Basic Analysis Category):
  Total Attempted:     5
  Execution Succeeded: 5/5 (100.0%)
  Analytical Accuracy: 5/5 (100.0%)
  Average Latency:     1.35 seconds
```

> **API Quota Limitation Notice:** Live end-to-end evaluation is constrained by Google AI Studio's Free Tier quotas (5 requests per minute and 20 requests per day for `gemini-2.5-flash`). The live benchmark was verified on completed sets, while full-pipeline regression testing is supported quota-free via the deterministic runner.

---

## 🧪 Automated Testing

The project includes an automated test suite executed via `pytest`:

```bash
pytest
```

**Test Results:**
```text
46 passed in 1.69s
```

### Coverage:
- **Security & AST Validation (`tests/test_validator.py` - 30 tests):**
  - Confirms rejection of `os.system`, `subprocess`, `open()`, `eval()`, `exec()`, `__import__`, `globals()`, `locals()`, network libraries (`requests`, `socket`, `urllib`), dunder attributes (`__class__`), while/for loops, functions, classes, and try/except blocks.
  - Confirms permission of valid Pandas operations: boolean filtering, `groupby`, aggregations, `sort_values`, `nlargest`, `value_counts`, `.str` accessors, and markdown code fence stripping.
- **Sandbox Namespace & Immutability (`tests/test_sandbox.py` - 9 tests):**
  - Confirms DataFrame immutability on deep copies, isolation of global namespaces, `ZeroDivisionError` handling, and fuzzy `KeyError` column suggestions.
- **Robustness & Edge Cases (`tests/test_robustness.py` - 7 tests):**
  - Confirms syntax error diagnostics, missing value (NaN) handling, empty DataFrame execution, chart heuristic classification, and empty CSV loading.

---

## 🔍 Error Handling & Diagnostics

Instead of exposing raw Python tracebacks, user-facing operations produce actionable diagnostics:
- **Missing Columns:** Returns the missing column name, fuzzy match suggestions via `difflib` when similar columns exist, and a preview of available dataset columns.
- **Undefined References:** Informs the user when an unsupported variable name is referenced, directing operations to the provided `df` object.
- **Syntax / AST Rejections:** Explains the exact disallowed construct (e.g., *"Import statements are not allowed"*, *"Access to forbidden identifier 'os' is not allowed"*).

---

## 📁 Project Structure

```
AI-Data-Analyst-Agent/
├── app.py                      # Streamlit dashboard entry point
├── agent.py                    # Gemini API orchestration with backoff retries
├── validator.py                # AST code safety validator & structural whitelist
├── sandbox.py                  # Isolated runtime execution sandbox
├── charts.py                   # Deterministic chart detection & Matplotlib generator
├── prompt.py                   # Token-efficient system prompt & schema context
├── utils.py                    # CSV multi-encoding ingestion & dataset profiling
├── evaluate.py                 # CLI evaluation tool (deterministic and live modes)
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Project dependencies
├── data/
│   ├── ecommerce_sales.csv     # Synthetic evaluation benchmark dataset (100 rows)
│   └── README.md               # Dataset documentation & synthetic status notice
├── evaluation/
│   ├── eval_cases.json         # 43 curated evaluation benchmark test cases
│   ├── evaluator.py            # Dual-mode evaluation engine & metrics calculator
│   └── results.json            # Machine-readable evaluation run output
├── scripts/
│   ├── create_dataset.py       # Deterministic benchmark data generator
│   └── add_candidate_codes.py  # Benchmark candidate code configuration script
└── tests/
    ├── test_validator.py       # AST validation & security unit tests
    ├── test_sandbox.py         # Sandbox namespace & immutability tests
    └── test_robustness.py      # Pipeline edge-case & robustness tests
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Aashi0105/Ai-data-analyst-agent.git
cd Ai-data-analyst-agent
```

### 2. Create and Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and add your Gemini API key:
```bash
# Windows
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```
Update `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🚀 Usage

### 1. Run the Streamlit Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, upload any CSV, inspect dataset profiling tabs, and enter conversational queries.

### 2. Run Automated Unit & Security Tests
```bash
pytest
```

### 3. Run Deterministic Pipeline Evaluation (43 Cases, No API Cost)
```bash
python evaluate.py --mode deterministic
```

### 4. Run Live LLM Evaluation (Requires Gemini API Key)
```bash
# Run specific category with pacing to respect API rate limits
python evaluate.py --mode live --category "Basic"

# Run with custom pacing delay (e.g., 12 seconds between calls)
python evaluate.py --mode live --pacing 12.0
```

---

## 📖 Documentation

For non-technical business users, refer to the [User Guide](docs/user-guide.md) for step-by-step instructions. It includes:
- **Getting Started Job Aid:** Step-by-step instructions for uploading datasets and submitting queries using real interface labels.
- **Reading Outputs:** Guidance on interpreting metric cards, tabular results, and deterministic chart visualizations.
- **Privacy & Safety FAQ:** Details on what data reaches the AI prompt, column name fuzzy matching, AST validator whitelists, and accuracy limits.
- **Troubleshooting Reference:** Common operational warnings, error messages, and immediate recovery actions.

---

## ⚠️ Limitations

- **API Quota Constraints:** Free-tier Google AI Studio API keys are limited to 5 requests per minute and 20 requests per day, requiring pacing during live testing.
- **Single-Table Scope:** Operates on single uploaded CSV files per session; does not currently perform relational multi-table SQL joins across separate files.
- **In-Memory Scale:** Designed for tabular data loaded into RAM via Pandas; not intended for multi-gigabyte distributed clusters (PySpark/Dask).
- **Synthetic Evaluation Data:** The benchmark dataset is programmatically generated to ensure reproducible ground truth and does not represent private enterprise data.

---

## 🔮 Realistic Future Improvements

- **Response Caching:** Cache prompt/code hashes locally with SQLite to eliminate redundant API calls for duplicate queries.
- **Multi-Table Relational Analysis:** Add support for multiple CSV uploads with automated foreign key / merge relationship resolution.
- **Expanded Chart Catalog:** Add correlation heatmaps and boxplots to the heuristic visualization engine.
- **Hardened Execution Isolation:** Provide optional containerized runner backends (e.g., Docker / gVisor) for multi-tenant deployment.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
