# AI Data Analyst Agent: User Guide

> **Status:** draft. Not yet tested with real users.

## 1. Overview
The AI Data Analyst Agent is a web-based analytics assistant that lets you upload a CSV spreadsheet and ask questions about your data in plain English. It automatically translates your questions into safe data calculations, displays clear numerical or tabular answers, and generates visual charts when appropriate.

---

## 2. Job Aid: Ask Your First Question

Follow this step-by-step guide to upload your data and ask your first business question.

| Step | What to do | What you should see |
| :--- | :--- | :--- |
| 1 | Open the application in your browser (typically `http://localhost:8501`). | The main header **📊 AI Data Analyst Agent** and the sidebar **⚙️ Control Panel** appear. An introductory note prompts: `👈 Upload a CSV file from the sidebar panel to view dataset statistics and ask questions.` |
| 2 | In the left sidebar under **📁 Data Source**, click the **Upload a CSV dataset** box (or drag and drop a `.csv` file). | Once uploaded, a green badge displays: `✅ Loaded: <your_file_name.csv>`. |
| 3 | Inspect the automated metrics at the top of the main page under **📈 Dataset Summary**. | Four summary metric cards appear: **Total Rows**, **Total Columns**, **Missing Values** (marked `Clean` or with a count of missing cells), and **Duplicate Rows** (marked `Unique` or with a count of duplicate rows). |
| 4 | Explore your dataset structure across the three preview tabs. | Three tabs are available:<br>- **🔍 Data Preview (First 10 Rows)**: Shows the first 10 rows of your dataset.<br>- **📋 Column Schema & Missing Values**: Shows each column's name, data type, missing count, and missing percentage.<br>- **🏷️ Column Names List**: Displays the complete list of available column names. |
| 5 | Scroll down to the **💬 Ask a Question About Your Data** section. Click into the text box labeled **Enter your question:** and type your question (e.g., `What are the top 5 highest sales regions?`). | Your query appears in the input field. |
| 6 | Click the primary blue button labeled **🚀 Ask Agent**. | A loading spinner appears displaying: `🧠 Analyzing data, executing query, and rendering visualization...` |
| 7 | Review your results once the spinner finishes. | The page displays your echoed question, an answer under **💡 Answer**, an optional chart under **📊 Visualization**, and an expandable section labeled **💻 View Executed Pandas Code**. |

---

## 3. Job Aid: Read the Answer and the Chart

When the agent finishes processing your question, it displays the output in up to three distinct areas:

### 3.1. The Answer Section (`💡 Answer`)
The tool formats your analytical answer according to the kind of result computed:
- **Single Metric (KPI):** If the answer is a single number, percentage, or text value (such as average revenue or total customer count), it appears in a large **Result** card with readable commas and formatted decimals.
- **Data Table:** If the answer is a ranked list, breakdown, or filtered subset (such as top 5 products or monthly summary), it displays as an interactive data table that you can scroll and inspect.
- **Structured Data:** If the answer produces key-value pairs or grouped items, it displays as a formatted list or JSON block.

### 3.2. The Visualization Section (`📊 Visualization`)
If your calculation produces data suitable for visual comparison, the tool automatically generates and renders a chart directly below the answer:
- **Bar Chart:** Created when your result compares categories against numbers (e.g., revenue by region) or summarizes a numeric series. Shows labeled categories on the horizontal axis and values on the vertical axis.
- **Line Chart:** Created when your result pairs calendar dates or timestamps with numeric metrics (e.g., sales over time). The dates are automatically ordered chronologically along the horizontal axis.
- **Scatter Plot:** Created when your result compares two numeric measures against each other (e.g., discount rate versus order total) to reveal correlations.
- **Histogram:** Created when your result contains a single numeric column to show the spread and frequency distribution across 15 standard bins.
- **Caption:** Every generated chart includes a descriptive label beneath it (e.g., `Chart: Bar Chart`).

> [!NOTE]
> If your calculation produces a single summary number (e.g., total sales = $150,000) or a wide table with 3 or more columns, no chart is generated. In those cases, the answer is fully presented in the **💡 Answer** card or table.

### 3.3. Technical Inspection (`💻 View Executed Pandas Code`)
Below the answer and chart, you can click the collapsed expander labeled **💻 View Executed Pandas Code**. 
- Non-technical users do not need to open this.
- If opened, it reveals the exact Python Pandas calculation that was generated and executed, alongside confirmation badges: `Validation: ✅ Passed | Execution: ✅ Passed`.

---

## 4. Frequently Asked Questions (FAQ)

### Q1: What data from my spreadsheet is sent to the AI?
Only structural metadata and a tiny 3-row sample are sent to Google Gemini to formulate the calculation. Specifically, the prompt includes:
- **Dataset Dimensions:** The total number of rows and columns (`df.shape`).
- **Column Names & Data Types:** The list of every column header and its technical type (e.g., `Revenue: float64`, `Region: object`).
- **First 3 Sample Rows:** Only the first 3 rows of data (`df.head(3)`), with any text entry longer than 30 characters truncated to 30 characters.
- **Your Question:** The exact text you typed into the question box.

The rest of your spreadsheet rows are not sent to Gemini. All actual computations run locally on your computer inside an isolated sandbox. Do not upload files containing data you are not allowed to share with an external AI service.

### Q2: Why does the tool sometimes suggest a different column name?
When a calculation attempts to look up a column that does not exist in your spreadsheet, Python triggers a column lookup error (`KeyError`). 
Instead of crashing, the sandbox runs a similarity comparison (`difflib.get_close_matches` with a 60% similarity threshold) against the actual columns in your file. If it finds close matches, it suggests them (e.g., `Did you mean one of these: 'Sales_Amount'?`) and provides a preview of the actual available columns so you can adjust your question's phrasing.

### Q3: Why may a question be rejected by the safety validator, and what does it block?
Before any generated Python code is allowed to run, it is inspected by a strict Abstract Syntax Tree (AST) safety validator (`validator.py`). The validator enforces a strict allowlist of basic analytical Pandas operations and explicitly blocks:
- **Import Statements:** `import` and `from ... import` statements.
- **Custom Functions & Classes:** `def`, `class`, and `lambda` declarations.
- **Loops & Control Flow:** `for` loops, `while` loops, and generator expressions.
- **Error Handling & Contexts:** `try`/`except` blocks and `with` statements.
- **File & Operating System Access:** System commands, file reading/writing (`open()`), network calls, and dangerous modules (`os`, `sys`, `subprocess`, `requests`, `socket`, `shutil`, `pathlib`, `pickle`, `urllib`).
- **Dangerous Built-in Commands:** `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `input`, `exit`, `quit`, and `__import__`.
- **Private/Dunder Attributes:** Any attribute starting with double underscores (`__`).
- Result variable and other constructs: the generated code must put its final answer in a variable named `result` (a single bare expression is converted automatically). Also blocked: yield, await, global and nonlocal.

If a user's question causes the AI to attempt any of these forbidden constructs, the system halts execution and displays: `❌ Validation Error: <explanation>`.

### Q4: How can I phrase better questions to get accurate results?
To get the most accurate answers on your first try:
- **Use the exact column names:** Check the **🏷️ Column Names List** tab and use the exact spelling or capitalization present in your dataset (e.g., write `Order_Total` instead of `total order price`).
- **State the calculation explicitly:** Use clear operational words such as `average`, `total sum`, `median`, `highest`, `lowest`, or `count`.
- **Specify limits clearly:** If you want a top list, specify the number (e.g., `top 5 products by total revenue`).
- **Clarify filtering criteria:** State exact filter values (e.g., `where Status is 'Delivered' and Region is 'North'`).
- **Avoid requesting multi-step programs:** Ask for analytical metrics rather than asking the tool to write custom scripts or train machine learning models.

### Q5: What are the limits of the tool's accuracy?
- **Single Table Scope:** The tool operates on one uploaded CSV at a time. It cannot link or join across multiple separate files.
- **Column Ambiguity:** If your spreadsheet has similarly named columns (e.g., `Net_Sales` vs. `Gross_Revenue`), an ambiguous question like "show sales" might cause the AI to pick the wrong column.
- **Sample Text Truncation:** Because text entries in the 3 sample rows are truncated to 30 characters in the AI prompt, categories with long strings might not be fully seen by the AI during code generation.
- **Heuristic Visualizations:** Chart generation relies on strict structural criteria (e.g., 2-column tables or numeric series). Complex tables with 3 or more columns will not produce a chart.
- **In-Memory Size Limits:** The tool loads files into memory via Pandas; extremely large CSV files that exceed available computer RAM may cause sluggish performance or load failures.

### Q6: When does the tool create a chart versus only displaying text or a table?
The tool automatically creates a chart only when the calculation result matches one of five specific shapes:
1. Exactly 2 columns: 1 categorical/text column and 1 numeric column (**Bar Chart**).
2. Exactly 2 columns: 1 date/timestamp column and 1 numeric column (**Line Chart**).
3. Exactly 2 columns: both numeric columns (**Scatter Plot**).
4. Exactly 1 numeric column (**Histogram**).
5. A one-dimensional numeric Series (**Bar Chart**).

If your answer is a single number, a dictionary of values, or a table with 3 or more columns, no chart is generated.

### Q7: Can this tool alter, overwrite, or delete my uploaded CSV data?
No. Your data is protected by two safeguards:
1. **In-Memory Deep Copy:** Every calculation runs against an isolated in-memory clone of your dataset (`df.copy(deep=True)`). Any column modification during execution affects only the temporary copy.
2. **Read-Only Sandbox:** File writing functions (`open()`, filesystem writes) and system libraries are strictly blocked by the AST validator and sandbox.

### Q8: What should I do if I see an API quota or rate-limit message?
The tool connects to Google Gemini using an API key. On free-tier API accounts, Google limits how many requests can be submitted per minute and per day.
- For short-term traffic spikes, the tool automatically pauses and retries up to 3 times with progressive delays.
- If your account exceeds its daily quota, the tool will display: `❌ API Error: Gemini API daily quota exceeded (429 RESOURCE_EXHAUSTED)`. In this scenario, you must wait until Google resets your daily allowance or provide a key with a higher service tier.

---

## 5. Troubleshooting Table

| What you see | Likely cause | What to do |
| :--- | :--- | :--- |
| `👆 Please upload a CSV file to get started.` or `👈 Upload a CSV file from the sidebar panel...` | No file has been uploaded yet, or the file upload was cleared. | Click **Browse files** under **Upload a CSV dataset** in the sidebar and choose a valid `.csv` file. |
| `❌ CSV Error: The uploaded CSV file is empty and contains no data rows.` | The selected CSV file contains 0 bytes or has headers but no data rows. | Verify your CSV file on your computer to ensure it has valid data rows before uploading. |
| `❌ CSV Error: Failed to read CSV file due to unsupported file encoding.` | The CSV file is encoded with an unsupported character set. | Open the file in Excel or a text editor and save/export it using standard **CSV (UTF-8)** format. |
| `⚠️ Please enter a question before clicking Ask.` | The question text box was left blank or contains only whitespace. | Type a question into the **Enter your question:** box before clicking **🚀 Ask Agent**. |
| `❌ Validation Error: <error description>` | The generated code attempted a forbidden action (such as an import, loop, function definition, or system call). | Rephrase your question into a direct analytical calculation (e.g., "Find the average price grouped by category"). Avoid asking for scripts or custom functions. |
| `❌ Execution Error: Column '<col>' was not found in the dataset... Did you mean one of these: ...?` | The question referenced a column name that does not exist in the spreadsheet, or the AI misspelled a column. | Open the **🏷️ Column Names List** tab, locate the exact column name, and re-ask your question using that exact spelling. |
| `❌ Execution Error: Index out of range...` | A filter or condition in the query matched zero records, producing an empty result. | Check whether your filter value (such as a category name or date) is spelled correctly and exists in the dataset. |
| `❌ Execution Error: Calculation error: Division by zero encountered.` | The calculation attempted to divide a number by zero. | Rephrase your question to exclude zero values or filter out records where the divisor is zero. |
| `⚠️ Configuration / Input Error: Gemini API Key is missing...` | The `.env` configuration file is missing the `GEMINI_API_KEY` entry. | Add your Gemini API key to the project `.env` file (`GEMINI_API_KEY=your_key`) and restart the Streamlit server. |
| `❌ API Error: Gemini API daily quota exceeded (429 RESOURCE_EXHAUSTED).` | The daily quota limit on your Google Gemini API key has been reached. | Wait for Google's daily quota window to reset, or configure a Gemini API key with paid quota. |
| No chart is displayed below the answer card or table. | The result is a single number, a dictionary, or a table with 3 or more columns, which does not match automatic chart rules. | This is expected behavior. If you want a chart, ask for a two-column comparison (e.g., "Total sales by Region"). |
