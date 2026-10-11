"""
tests/test_validator.py
-----------------------
Unit and security tests for the AST validator layer (validator.py).
Tests strict rejection of malicious or unauthorized Python constructs
and verifies that all legitimate pandas operations remain allowed.
"""

import pytest
from validator import validate_code, clean_code


# =====================================================================
# 1. Malicious and Unsafe Code Rejection Tests
# =====================================================================

def test_blocks_os_system():
    """Verify that invocation of os.system is strictly rejected."""
    code = "import os\nresult = os.system('ls')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "Import" in msg or "forbidden" in msg.lower() or "os" in msg.lower()


def test_blocks_os_system_direct_call():
    """Verify that direct reference to os is caught even without import statement."""
    code = "result = os.system('echo 1')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "forbidden identifier 'os'" in msg


def test_blocks_subprocess():
    """Verify that subprocess module access is rejected."""
    code = "import subprocess\nresult = subprocess.run(['ls'])"
    is_valid, msg = validate_code(code)
    assert not is_valid


def test_blocks_subprocess_direct():
    """Verify that direct subprocess reference is blocked."""
    code = "result = subprocess.check_output('whoami')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "subprocess" in msg


def test_blocks_eval():
    """Verify that eval() calls are blocked."""
    code = "result = eval('1 + 1')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "eval" in msg


def test_blocks_exec():
    """Verify that exec() calls are blocked."""
    code = "result = exec('x = 1')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "exec" in msg


def test_blocks_dynamic_import():
    """Verify that __import__ is rejected."""
    code = "result = __import__('os').system('id')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "__import__" in msg or "forbidden" in msg


def test_blocks_file_access_open():
    """Verify that open() for filesystem reading/writing is rejected."""
    code = "result = open('/etc/passwd').read()"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "open" in msg


def test_blocks_arbitrary_imports():
    """Verify that any import statements are rejected."""
    code = "import math\nresult = math.sqrt(16)"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "Import statements are not allowed" in msg


def test_blocks_from_import():
    """Verify that from ... import ... statements are rejected."""
    code = "from datetime import datetime\nresult = datetime.now()"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "Import statements are not allowed" in msg


def test_blocks_dunder_attributes():
    """Verify that private and dunder attributes (e.g., __class__) are blocked."""
    code = "result = df.__class__.__mro__"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "dunder" in msg.lower() or "private" in msg.lower()


def test_blocks_builtins_introspection():
    """Verify that globals(), locals(), vars() are blocked."""
    for fn in ["globals()", "locals()", "vars()"]:
        code = f"result = {fn}"
        is_valid, msg = validate_code(code)
        assert not is_valid
        assert "forbidden identifier" in msg


def test_blocks_network_libraries():
    """Verify that network access identifiers (socket, requests, urllib) are rejected."""
    for lib in ["requests", "socket", "urllib"]:
        code = f"result = {lib}.get('https://example.com')"
        is_valid, msg = validate_code(code)
        assert not is_valid
        assert f"forbidden identifier '{lib}'" in msg


def test_blocks_while_and_for_loops():
    """Verify that loops (while, for) are blocked."""
    code_while = "while True:\n    pass\nresult = 1"
    is_valid, msg = validate_code(code_while)
    assert not is_valid
    assert "Loops are not allowed" in msg

    code_for = "for x in [1, 2]:\n    pass\nresult = 1"
    is_valid, msg = validate_code(code_for)
    assert not is_valid
    assert "Loops are not allowed" in msg


def test_blocks_function_definitions():
    """Verify that function and lambda definitions are rejected."""
    code_func = "def foo():\n    return 1\nresult = foo()"
    is_valid, msg = validate_code(code_func)
    assert not is_valid
    assert "Function definitions are not allowed" in msg

    code_lambda = "f = lambda x: x + 1\nresult = f(5)"
    is_valid, msg = validate_code(code_lambda)
    assert not is_valid
    assert "Lambda expressions are not allowed" in msg


def test_blocks_class_definitions():
    """Verify that class definitions are rejected."""
    code = "class Exploit:\n    pass\nresult = 1"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "Class definitions are not allowed" in msg


def test_blocks_try_except_blocks():
    """Verify that try/except blocks are rejected to prevent error suppression."""
    code = "try:\n    x = 1\nexcept:\n    pass\nresult = 1"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "Try/Except blocks are not allowed" in msg


# =====================================================================
# 2. Result Invariant Validation Tests
# =====================================================================

def test_rejects_missing_result_assignment():
    """Verify that code not assigning to 'result' is rejected."""
    code = "x = df['revenue'].mean()"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "assign the final output to a variable named 'result'" in msg


def test_rejects_multiple_result_assignments():
    """Verify that assigning to 'result' multiple times is rejected."""
    code = "result = df['revenue'].mean()\nresult = result * 2"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "assigned exactly once" in msg


def test_rejects_empty_code():
    """Verify that empty or whitespace-only code strings are rejected."""
    is_valid, msg = validate_code("")
    assert not is_valid
    assert "empty" in msg.lower()

    is_valid, msg = validate_code("   \n\t  ")
    assert not is_valid
    assert "empty" in msg.lower()


# =====================================================================
# 3. Legitimate Pandas Operations Allowed Tests
# =====================================================================

def test_allows_dataframe_filtering():
    """Verify that standard boolean filtering is allowed."""
    code = "result = df[df['revenue'] > 500]"
    is_valid, msg = validate_code(code)
    assert is_valid, f"Expected valid, got: {msg}"


def test_allows_dataframe_groupby():
    """Verify that groupby operations are allowed."""
    code = "result = df.groupby('region')['revenue'].mean()"
    is_valid, msg = validate_code(code)
    assert is_valid, f"Expected valid, got: {msg}"


def test_allows_dataframe_aggregation():
    """Verify that multi-aggregation methods are allowed."""
    code = "result = df.groupby('category').agg({'revenue': 'sum', 'quantity': 'mean'})"
    is_valid, msg = validate_code(code)
    assert is_valid, f"Expected valid, got: {msg}"


def test_allows_dataframe_sorting():
    """Verify that sorting values is allowed."""
    code = "result = df.sort_values(by='revenue', ascending=False).head(5)"
    is_valid, msg = validate_code(code)
    assert is_valid, f"Expected valid, got: {msg}"


def test_allows_mean_sum_count():
    """Verify that standard numerical aggregates are allowed."""
    for method in ["mean()", "sum()", "median()", "std()", "count()"]:
        code = f"result = df['revenue'].{method}"
        is_valid, msg = validate_code(code)
        assert is_valid, f"Method {method} failed: {msg}"


def test_allows_column_selection():
    """Verify that single and multiple column selection are allowed."""
    code1 = "result = df['revenue']"
    is_valid1, msg1 = validate_code(code1)
    assert is_valid1, msg1

    code2 = "result = df[['category', 'revenue', 'profit']]"
    is_valid2, msg2 = validate_code(code2)
    assert is_valid2, msg2


def test_allows_string_methods():
    """Verify that pandas vectorized .str accessor methods are allowed."""
    code = "result = df[df['product_name'].str.contains('Laptop', na=False)]"
    is_valid, msg = validate_code(code)
    assert is_valid, f"Expected valid, got: {msg}"


def test_allows_value_counts():
    """Verify that value_counts() is allowed."""
    code = "result = df['category'].value_counts()"
    is_valid, msg = validate_code(code)
    assert is_valid, f"Expected valid, got: {msg}"


def test_allows_nlargest():
    """Verify that nlargest() is allowed."""
    code = "result = df.nlargest(5, 'profit')"
    is_valid, msg = validate_code(code)
    assert is_valid, f"Expected valid, got: {msg}"


def test_allows_markdown_fence_cleaning():
    """Verify that code wrapped in markdown code blocks is correctly cleaned and validated."""
    fenced_code = "```python\nresult = df['revenue'].sum()\n```"
    is_valid, msg = validate_code(fenced_code)
    assert is_valid, f"Expected valid, got: {msg}"
