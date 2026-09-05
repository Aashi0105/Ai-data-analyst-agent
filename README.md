# 📊 AI Data Analyst Agent

An intelligent, modular portfolio application built with Python, Streamlit, Pandas, Matplotlib, and Google Gemini API. It empowers users to analyze CSV datasets by asking natural language questions, automatically generating pandas code, executing it safely, and visualizing results.

---

## 🌟 Key Features

1. **Dataset Upload & Overview**: Upload CSV files and immediately inspect rows, columns, data types, missing values, and interactive previews.
2. **Natural Language Querying**: Ask dataset questions in plain English.
3. **Pandas Code Generation**: Uses Google Gemini API to translate natural language into Python/Pandas operations.
4. **AST-Based Code Validation**: Sanitizes and validates LLM-generated code prior to execution for security.
5. **Safe Sandbox Execution**: Runs pandas operations in a isolated runtime scope.
6. **Automated Visualization**: Automatically generates Matplotlib/Seaborn plots based on output data types.

---

## 📂 Project Architecture

```
AI-Data-Analyst-Agent/
├── app.py           # Streamlit UI application entry point
├── agent.py         # Core orchestration agent connecting LLM, Sandbox, & Charts
├── validator.py     # AST code validation and safety checker
├── sandbox.py       # Isolated execution sandbox for generated Pandas code
├── charts.py        # Chart detection and visualization generation logic
├── prompt.py        # System prompt templates and LLM prompt engineering
├── utils.py         # Dataset summary & helper utility functions
├── requirements.txt # Project dependencies
├── .env.example     # Environment variable template
└── README.md        # Documentation
```

---

## ⚙️ Setup & Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd AI-Data-Analyst-Agent
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and add your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```

5. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
