"""
prompt.py
---------
Purpose:
    Contains system prompt templates and prompt construction utilities for the AI Data Analyst Agent.
    Provides compact dataset context, heuristic PII masking on model-facing sample rows,
    and strict rules for Google Gemini API code generation.

Privacy Architecture & Non-Destructive Masking:
    - Raw Data Immutability: The caller's DataFrame is NEVER mutated. All calculations and
      sandboxed executions operate against the authentic, unaltered dataset.
    - Model-Facing Masking: Masking is applied strictly to a shallow copy of the 3 sample rows
      (`df.head(3).copy()`) serialized into the Google Gemini system prompt.
    - Limitations: Heuristic regex and semantic column-name matching cannot guarantee 100% detection
      of unstructured names, freeform text, or localized identifiers. It reduces accidental exposure
      of common PII, but does not provide complete regulatory safe-harbor anonymization.
"""

import re
from typing import Any, Optional
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


# ============================================================================
# PII Detection Heuristics & Regular Expressions
# ============================================================================

# 1. Email Addresses (Standard RFC-like pattern)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# 2. International & National Phone Numbers
# Matches E.164 (+1-555-123-4567), formatted (555) 123-4567, dotted, or dashed numbers
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
)

# 3. Government Identifiers
# - US SSN: 123-45-6789 or 123 45 6789
SSN_PATTERN = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")
# - Indian PAN: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F)
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
# - Indian Aadhaar: 12 digits, often formatted as 4-4-4 (e.g. 1234 5678 9012)
AADHAAR_PATTERN = re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}\b(?![- \d])")

# 4. Payment Cards (13 to 19 digits with optional dashes/spaces)
PAYMENT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4,7}\b"
)

# 5. IP Addresses (IPv4 standard)
IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)

# 6. Sensitive API Keys & Tokens (Bearer tokens, Google AIza..., GitHub ghp_..., generic sk-...)
API_KEY_PATTERN = re.compile(
    r"\b(?:AIza[0-9A-Za-z_\-]{30,45}|ghp_[0-9A-Za-z]{36}|sk-[0-9A-Za-z]{20,}|Bearer\s+[A-Za-z0-9_\-\.]{20,})\b",
    re.IGNORECASE
)

# 7. Semantic Column Keywords Indicating High-Probability PII
SENSITIVE_COLUMN_KEYWORDS = {
    "name", "first_name", "firstname", "last_name", "lastname", "full_name", "fullname",
    "customer_name", "client_name", "patient_name", "employee_name", "user_name", "username",
    "email", "e_mail", "phone", "telephone", "mobile", "cell", "contact",
    "ssn", "social_security", "aadhaar", "adhaar", "pan", "passport", "tax_id", "national_id",
    "card", "card_number", "credit_card", "debit_card", "cvv", "bank_account", "iban",
    "address", "street", "postal_code", "zipcode", "zip_code",
    "password", "secret", "token", "api_key", "credential"
}


def _is_luhn_valid(num_str: str) -> bool:
    """Checks if a digit string satisfies the Luhn mod-10 formula."""
    digits = [int(c) for c in num_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for idx, d in enumerate(reverse_digits):
        if idx % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0


def mask_sensitive_value(val: Any, col_name: str = "") -> str:
    """
    Applies heuristic redaction to individual cell values for model-facing prompts.
    Preserves ordinary numbers, dates, short labels, and analytical metrics.
    """
    if pd.isnull(val):
        return str(val)

    # Convert to string for pattern inspection
    val_str = str(val).strip()
    if not val_str:
        return val_str

    # --- Check 1: Column Semantic Matching ---
    # Normalize col_name: replace underscores/dashes with space and split tokens
    clean_col = re.sub(r"[_\-\.]+", " ", col_name.lower()).strip()
    col_tokens = set(clean_col.split())

    if any(k in col_tokens or clean_col == k or k.replace('_', ' ') in clean_col for k in SENSITIVE_COLUMN_KEYWORDS):
        # Differentiate specific PII categories by column name
        if any(term in clean_col for term in ["email", "e_mail"]):
            return "[EMAIL]"
        elif any(term in clean_col for term in ["phone", "mobile", "cell", "telephone"]):
            return "[PHONE]"
        elif any(term in clean_col for term in ["ssn", "social_security"]):
            return "[SSN]"
        elif "pan" in col_tokens:
            return "[PAN]"
        elif any(term in clean_col for term in ["aadhaar", "adhaar"]):
            return "[AADHAAR]"
        elif any(term in clean_col for term in ["card", "credit_card", "debit_card", "cvv"]):
            return "[CARD_NUMBER]"
        elif any(term in clean_col for term in ["password", "secret", "token", "api key", "api_key", "credential"]):
            return "[REDACTED_SECRET]"
        elif any(term in clean_col for term in ["address", "street"]):
            return "[ADDRESS]"
        elif any(term in clean_col for term in ["name", "firstname", "lastname", "fullname"]):
            return "[NAME]"

    # --- Check 2: Sensitive API Keys and Tokens ---
    if API_KEY_PATTERN.search(val_str):
        return API_KEY_PATTERN.sub("[REDACTED_TOKEN]", val_str)

    # --- Check 3: Email Address Pattern ---
    if EMAIL_PATTERN.search(val_str):
        return EMAIL_PATTERN.sub("[EMAIL]", val_str)

    # --- Check 4: Government Identifiers ---
    if SSN_PATTERN.search(val_str):
        return SSN_PATTERN.sub("[SSN]", val_str)

    if PAN_PATTERN.search(val_str):
        return PAN_PATTERN.sub("[PAN]", val_str)

    # --- Check 5: Payment Cards (Evaluated before 12-digit Aadhaar to avoid prefix collisions) ---
    card_match = PAYMENT_CARD_PATTERN.search(val_str)
    if card_match:
        matched_digits = re.sub(r"\D", "", card_match.group(0))
        if 13 <= len(matched_digits) <= 19 and _is_luhn_valid(matched_digits):
            return PAYMENT_CARD_PATTERN.sub("[CARD_NUMBER]", val_str)

    # --- Check 6: Aadhaar (12 digits) ---
    if AADHAAR_PATTERN.search(val_str):
        return AADHAAR_PATTERN.sub("[AADHAAR]", val_str)

    # --- Check 6: IPv4 Address Pattern ---
    if IPV4_PATTERN.search(val_str):
        return IPV4_PATTERN.sub("[IP_ADDRESS]", val_str)

    # --- Check 7: Phone Number Pattern ---
    # Only evaluate phone if the string has at least 7 digits to avoid redacting short numbers
    digits_only = re.sub(r"\D", "", val_str)
    if 7 <= len(digits_only) <= 15:
        # Require typical telephone formatting characters (+, -, (), .) if length is < 10
        if len(digits_only) >= 10 or any(c in val_str for c in ["+", "-", "(", ")", "."]):
            if PHONE_PATTERN.search(val_str):
                return PHONE_PATTERN.sub("[PHONE]", val_str)

    return val_str


def mask_sample_dataframe(df: pd.DataFrame, max_rows: int = 3, max_str_len: int = 30) -> pd.DataFrame:
    """
    Produces a masked preview DataFrame for model prompt assembly.
    Guarantees caller DataFrame remains completely unmutated.
    """
    # Create isolated copy of head rows
    sample_df = df.head(max_rows).copy(deep=True)

    for col in sample_df.columns:
        col_name_str = str(col)
        # Apply cell masking across each element
        sample_df[col] = sample_df[col].apply(lambda v: mask_sensitive_value(v, col_name=col_name_str))

        # Enforce maximum character length for string display
        if pd.api.types.is_object_dtype(sample_df[col]) or pd.api.types.is_string_dtype(sample_df[col]):
            sample_df[col] = sample_df[col].astype(str).str.slice(0, max_str_len)

    return sample_df


def build_system_prompt(df: pd.DataFrame) -> str:
    """
    Constructs an optimized system prompt containing dataset shape, columns, types,
    and PII-masked sample rows.

    Guarantees:
    - Never mutates the caller's DataFrame.
    - Column names remain exact so generated Pandas code references authentic columns.
    - Common sensitive cell values are redacted with clear category placeholders.

    Args:
        df (pd.DataFrame): Target pandas DataFrame.

    Returns:
        str: Formatted system prompt.
    """
    shape = df.shape
    column_types = "\n".join([f"  - {col}: {dtype}" for col, dtype in df.dtypes.items()])

    # Apply heuristic PII masking on isolated sample rows
    masked_sample_df = mask_sample_dataframe(df, max_rows=3, max_str_len=30)
    sample_rows = masked_sample_df.to_string(index=False)

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
