"""
tests/test_prompt.py
--------------------
Unit tests for heuristic PII masking and prompt construction in prompt.py.
Verifies that:
1. Sensitive values (email, phone, SSN, PAN, Aadhaar, payment cards, IP, API keys) are redacted.
2. Sensitive column names trigger appropriate semantic redaction.
3. Legitimate analytical data (sales, prices, quantities, dates, categories) is preserved.
4. The caller's DataFrame is never mutated in place (deep immutability).
5. Model prompt construction contains redacted tokens instead of raw sensitive values.
6. DataAnalystAgent passes redacted sample rows in the model-facing Gemini prompt (mocked).
"""

from unittest.mock import MagicMock
import pandas as pd
import pytest

from prompt import (
    build_system_prompt,
    build_user_prompt,
    mask_sample_dataframe,
    mask_sensitive_value,
    _is_luhn_valid,
)
from agent import DataAnalystAgent


# =====================================================================
# 1. Pattern-Specific Cell Masking Tests
# =====================================================================

def test_mask_email():
    """Verify email addresses are masked to [EMAIL]."""
    assert mask_sensitive_value("john.doe@example.com") == "[EMAIL]"
    assert mask_sensitive_value("alice+test_dev@sub.company.org") == "[EMAIL]"
    assert mask_sensitive_value("Contact: admin@domain.io today") == "Contact: [EMAIL] today"


def test_mask_phone():
    """Verify varied telephone number formats are masked to [PHONE]."""
    assert mask_sensitive_value("+1-555-123-4567") == "[PHONE]"
    assert mask_sensitive_value("(555) 234-5678") == "[PHONE]"
    assert mask_sensitive_value("555.345.6789") == "[PHONE]"
    assert mask_sensitive_value("Call +44 20 7946 0958") == "Call [PHONE]"


def test_mask_ssn():
    """Verify US Social Security Numbers are masked to [SSN]."""
    assert mask_sensitive_value("123-45-6789") == "[SSN]"
    assert mask_sensitive_value("SSN is 987-65-4321") == "SSN is [SSN]"


def test_mask_pan():
    """Verify Indian PAN cards (5 letters, 4 digits, 1 letter) are masked to [PAN]."""
    assert mask_sensitive_value("ABCDE1234F") == "[PAN]"
    assert mask_sensitive_value("pan: BZRPK9876Q") == "pan: [PAN]"


def test_mask_aadhaar():
    """Verify Indian Aadhaar format (12 digits, 4-4-4) is masked to [AADHAAR]."""
    assert mask_sensitive_value("1234 5678 9012") == "[AADHAAR]"
    assert mask_sensitive_value("9876-5432-1098") == "[AADHAAR]"


def test_mask_payment_card():
    """Verify valid Luhn payment card numbers are masked to [CARD_NUMBER]."""
    # Standard Visa test card (passes Luhn)
    valid_visa = "4532-0150-1234-5671"
    assert mask_sensitive_value(valid_visa) == "[CARD_NUMBER]"

    # Random digits that fail Luhn should not be masked as payment cards
    invalid_card = "1234-5678-9012-3456"
    assert not _is_luhn_valid("1234567890123456")
    # Will not be masked as a card
    assert "[CARD_NUMBER]" not in mask_sensitive_value(invalid_card)


def test_mask_ip_address():
    """Verify IPv4 addresses are masked to [IP_ADDRESS]."""
    assert mask_sensitive_value("192.168.1.1") == "[IP_ADDRESS]"
    assert mask_sensitive_value("Host at 10.0.0.254") == "Host at [IP_ADDRESS]"


def test_mask_api_key_and_token():
    """Verify API keys and bearer tokens are masked."""
    google_key = "AIzaSyD-1234567890abcdefghijklmnopqrst"
    assert mask_sensitive_value(google_key) == "[REDACTED_TOKEN]"
    bearer_token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz"
    assert "[REDACTED_TOKEN]" in mask_sensitive_value(bearer_token)


# =====================================================================
# 2. Semantic Column-Name Masking Tests
# =====================================================================

def test_semantic_column_masking():
    """Verify columns with sensitive header names are automatically redacted."""
    assert mask_sensitive_value("Alice Smith", col_name="customer_name") == "[NAME]"
    assert mask_sensitive_value("Bob Jones", col_name="Full_Name") == "[NAME]"
    assert mask_sensitive_value("123 Main St, Springfield", col_name="Address") == "[ADDRESS]"
    assert mask_sensitive_value("SuperSecretPassword123", col_name="password") == "[REDACTED_SECRET]"
    assert mask_sensitive_value("custom_token_val", col_name="api_key") == "[REDACTED_SECRET]"


# =====================================================================
# 3. Preservation of Legitimate Analytical Metrics
# =====================================================================

def test_preserves_legitimate_metrics():
    """Verify normal numbers, dates, prices, and categories are NOT redacted."""
    # Prices & Totals
    assert mask_sensitive_value(150.50, col_name="Price") == "150.5"
    assert mask_sensitive_value(1200000, col_name="Revenue") == "1200000"
    assert mask_sensitive_value(-45.2, col_name="Profit") == "-45.2"

    # Quantities & Integers
    assert mask_sensitive_value(42, col_name="Quantity") == "42"
    assert mask_sensitive_value(10, col_name="Discount_Pct") == "10"

    # Dates
    assert mask_sensitive_value("2026-10-25", col_name="Order_Date") == "2026-10-25"
    assert mask_sensitive_value("10/25/2026", col_name="Date") == "10/25/2026"

    # Categories
    assert mask_sensitive_value("Technology", col_name="Category") == "Technology"
    assert mask_sensitive_value("North America", col_name="Region") == "North America"


# =====================================================================
# 4. DataFrame Immutability Test
# =====================================================================

def test_caller_dataframe_immutability():
    """Verify that build_system_prompt leaves the original DataFrame completely unaltered."""
    raw_data = {
        "customer_name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "email": ["alice@company.com", "bob@business.org", "charlie@web.net"],
        "ssn": ["123-45-6789", "987-65-4321", "555-55-5555"],
        "sales": [100.5, 250.0, 75.25]
    }
    original_df = pd.DataFrame(raw_data)
    df_clone = original_df.copy(deep=True)

    # Generate prompt
    system_prompt = build_system_prompt(original_df)

    # 1. Check prompt contains redacted tokens
    assert "[EMAIL]" in system_prompt
    assert "[SSN]" in system_prompt
    assert "[NAME]" in system_prompt

    # 2. Check prompt does NOT contain raw sensitive values
    assert "alice@company.com" not in system_prompt
    assert "123-45-6789" not in system_prompt
    assert "Alice Smith" not in system_prompt

    # 3. Check original DataFrame was not mutated in place
    pd.testing.assert_frame_equal(original_df, df_clone)
    assert original_df.loc[0, "customer_name"] == "Alice Smith"
    assert original_df.loc[0, "email"] == "alice@company.com"
    assert original_df.loc[0, "ssn"] == "123-45-6789"


# =====================================================================
# 5. Full Prompt Construction & Model Invocation Verification (Mocked)
# =====================================================================

def test_prompt_structure_and_column_retention():
    """Verify that prompt contains exact column names but masked sample values."""
    df = pd.DataFrame({
        "User_Email": ["admin@test.com", "support@test.com"],
        "User_Phone": ["+1-555-987-6543", "+1-555-876-5432"],
        "Total_Spent": [499.99, 1250.00]
    })

    prompt_text = build_system_prompt(df)

    # Exact column names must be preserved for code generation
    assert "User_Email" in prompt_text
    assert "User_Phone" in prompt_text
    assert "Total_Spent" in prompt_text

    # Values must be masked
    assert "admin@test.com" not in prompt_text
    assert "+1-555-987-6543" not in prompt_text
    assert "[EMAIL]" in prompt_text
    assert "[PHONE]" in prompt_text


def test_agent_generates_code_with_masked_prompt(monkeypatch):
    """
    Mock the Google GenAI SDK client to verify that when DataAnalystAgent.generate_code()
    is called, the system_instruction passed to the client contains masked tokens.
    """
    df = pd.DataFrame({
        "Client_Name": ["Secret Client A", "Secret Client B"],
        "Email_Address": ["secret.client@confidential.com", "billing@confidential.com"],
        "Sales_Amount": [5000, 7500]
    })

    captured_prompt = {}

    class MockModels:
        def generate_content(self, model, contents, config):
            captured_prompt["system_instruction"] = config.system_instruction
            captured_prompt["contents"] = contents
            mock_resp = MagicMock()
            mock_resp.text = "result = df.groupby('Client_Name')['Sales_Amount'].sum()"
            return mock_resp

    class MockClient:
        def __init__(self, api_key):
            self.models = MockModels()

    # Monkeypatch genai.Client in agent module
    import agent
    monkeypatch.setattr(agent.genai, "Client", MockClient)

    # Initialize agent with dummy key
    test_agent = DataAnalystAgent(api_key="mock_test_key_abc123")
    code = test_agent.generate_code(df, "What is the total sales by client?")

    # Verify generated code returned
    assert code == "result = df.groupby('Client_Name')['Sales_Amount'].sum()"

    # Verify that the model-facing instruction received masked data, NOT raw secrets
    sys_instruction = captured_prompt.get("system_instruction", "")
    assert "secret.client@confidential.com" not in sys_instruction
    assert "Secret Client A" not in sys_instruction
    assert "[EMAIL]" in sys_instruction
    assert "[NAME]" in sys_instruction
    assert "Sales_Amount" in sys_instruction  # Column name preserved
