"""
prompt.py
---------
Purpose:
    Contains system prompt templates and prompt construction utilities for the AI Data Analyst Agent.
    Provides compact dataset context and strict rules for Google Gemini API code generation.
"""

import pandas as pd


SYSTEM_PROMPT = """You are an expert Python data analyst.

A pandas DataFrame named df already exists.

Dataset Context:
- Shape (rows, columns): {shape}
- Columns & Data Types:
{column_types}

- Sample Data (first 3 rows):
{sample_rows}

Generate ONLY executable Python pandas code.

Strict Rules:
- Return ONLY valid Python code.
- No markdown code blocks (do NOT use ``` or ```python).
- No commentary, explanations, or print() statements.
- Do NOT import any libraries or modules.
- Use ONLY the existing dataframe named df.
- Never redefine df or create sample data.
- Never modify df (never mutate values or reassign df).
- Never use inplace=True.
- Prefer a single direct assignment to a variable named result.
- Never generate unnecessary intermediate variables.
- Store the final answer inside a variable named result."""


def build_system_prompt(df: pd.DataFrame) -> str:
    """
    Constructs an optimized system prompt containing dataset shape, columns, types,
    and truncated sample rows to reduce prompt token size.

    Args:
        df (pd.DataFrame): Target pandas DataFrame.

    Returns:
        str: Formatted system prompt.
    """
    shape = df.shape
    column_types = "\n".join([f"  - {col}: {dtype}" for col, dtype in df.dtypes.items()])

    # Create compact sample rows preview by truncating long string values to 30 chars
    sample_df = df.head(3).copy()
    for col in sample_df.columns:
        if pd.api.types.is_object_dtype(sample_df[col]) or pd.api.types.is_string_dtype(sample_df[col]):
            sample_df[col] = sample_df[col].astype(str).str.slice(0, 30)

    sample_rows = sample_df.to_string(index=False)

    return SYSTEM_PROMPT.format(
        shape=shape,
        column_types=column_types,
        sample_rows=sample_rows
    )


def build_user_prompt(question: str) -> str:
    """
    Constructs the user prompt containing the user's natural language query.

    Args:
        question (str): Plain English question.

    Returns:
        str: Formatted user prompt.
    """
    return f"User Question: {question.strip()}"
