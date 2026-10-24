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


# =====================================================================
# 4. Dangerous Pandas I/O and Evaluator Rejection Tests
# =====================================================================

@pytest.mark.parametrize("method_name", [
    "to_csv", "read_csv",
    "to_pickle", "read_pickle",
    "to_parquet", "read_parquet",
    "to_excel", "read_excel",
    "to_sql", "read_sql", "read_sql_query", "read_sql_table",
    "to_feather", "read_feather",
    "to_hdf", "read_hdf",
    "to_json", "read_json",
    "to_html", "read_html",
    "to_xml", "read_xml",
    "to_clipboard", "read_clipboard",
    "tofile", "fromfile"
])
def test_blocks_dangerous_pandas_io_methods(method_name):
    """Verify that file I/O, networking, and serialization methods are rejected on df or pd."""
    code_df = f"result = df.{method_name}('target_file')"
    is_valid_df, msg_df = validate_code(code_df)
    assert not is_valid_df, f"Expected {method_name} on df to be rejected"
    assert f"forbidden attribute '{method_name}'" in msg_df

    code_pd = f"result = pd.{method_name}('target_file')"
    is_valid_pd, msg_pd = validate_code(code_pd)
    assert not is_valid_pd, f"Expected {method_name} on pd to be rejected"
    assert f"forbidden attribute '{method_name}'" in msg_pd


def test_blocks_dataframe_query():
    """Verify that df.query() is rejected as dynamic expression evaluation."""
    code = "result = df.query('revenue > 100')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "forbidden attribute 'query'" in msg


def test_blocks_dataframe_eval():
    """Verify that df.eval() and pd.eval() are rejected as dynamic expression evaluation."""
    code_df = "result = df.eval('total = revenue * quantity')"
    is_valid, msg = validate_code(code_df)
    assert not is_valid
    assert "forbidden attribute 'eval'" in msg

    code_pd = "result = pd.eval('1 + 1')"
    is_valid, msg = validate_code(code_pd)
    assert not is_valid
    assert "forbidden attribute 'eval'" in msg


def test_blocks_dataframe_pipe():
    """Verify that df.pipe() is rejected to prevent arbitrary callable execution."""
    code = "result = df.pipe(some_function)"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert "forbidden attribute 'pipe'" in msg


@pytest.mark.parametrize("reflection_fn", ["getattr", "hasattr", "setattr", "delattr"])
def test_blocks_reflection_identifiers_and_attributes(reflection_fn):
    """Verify that reflection functions are blocked as both bare identifiers and attribute accesses."""
    code_call = f"result = {reflection_fn}(df, 'head')(5)"
    is_valid, msg = validate_code(code_call)
    assert not is_valid
    assert f"forbidden identifier '{reflection_fn}'" in msg

    code_attr = f"result = df.{reflection_fn}('head')"
    is_valid_attr, msg_attr = validate_code(code_attr)
    assert not is_valid_attr
    assert f"forbidden attribute '{reflection_fn}'" in msg_attr


# =====================================================================
# 5. Legitimate Analytical Operations Preservation Tests
# =====================================================================

@pytest.mark.parametrize("operation", [
    "result = df.head(10)",
    "result = df.tail(5)",
    "result = df.describe()",
    "result = df.groupby('region')['revenue'].sum()",
    "result = df.pivot_table(index='region', values='revenue', aggfunc='mean')",
    "result = df.sort_values(by='revenue', ascending=False)",
    "result = df.dropna(subset=['revenue'])",
    "result = df.fillna(0)",
    "result = df.reset_index()",
    "result = df['revenue'].corr(df['quantity'])",
    "result = df['category'].nunique()",
    "result = df[df['revenue'].between(100, 500)]",
    "result = df.select_dtypes(include=['number'])"
])
def test_preserves_legitimate_analytical_operations(operation):
    """Verify that standard exploratory data analysis operations continue to validate safely."""
    is_valid, msg = validate_code(operation)
    assert is_valid, f"Operation '{operation}' was unexpectedly rejected: {msg}"


# =====================================================================
# 6. Pandas and NumPy Dangerous I/O Rejection Tests
# =====================================================================

@pytest.mark.parametrize("dangerous_method", [
    "read_table", "read_fwf", "read_spss", "read_sas", "read_orc", "to_orc", "read_gbq", "to_gbq"
])
def test_blocks_pandas_io_methods(dangerous_method):
    """Verify that dangerous Pandas file and network I/O methods are blocked."""
    code = f"result = pd.{dangerous_method}('sensitive_data')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert f"forbidden attribute '{dangerous_method}'" in msg


@pytest.mark.parametrize("numpy_io_method", [
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromregex", "memmap"
])
def test_blocks_numpy_io_methods(numpy_io_method):
    """Verify that NumPy file I/O and memory mapping functions are blocked."""
    code = f"result = np.{numpy_io_method}('payload.npy')"
    is_valid, msg = validate_code(code)
    assert not is_valid
    assert f"forbidden attribute '{numpy_io_method}'" in msg

