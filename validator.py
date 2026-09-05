"""
validator.py
------------
Purpose:
    Provides strict AST (Abstract Syntax Tree) validation for LLM-generated code.
    Uses a structural whitelist approach to permit only safe Pandas/NumPy operations,
    disallowing imports, functions, loops, classes, dangerous builtins, and multiple assignments.
"""

from typing import Tuple, Set, Type
import ast

# Whitelisted AST node types permitted for execution
ALLOWED_NODE_TYPES: Set[Type[ast.AST]] = {
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Attribute,
    ast.Call,
    ast.Subscript,
    ast.Name,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Slice,
    ast.keyword,
    ast.Load,
    ast.Store,
    # Operator contexts
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn
}

# Add Index for Python versions < 3.9
if hasattr(ast, "Index"):
    ALLOWED_NODE_TYPES.add(getattr(ast, "Index"))

# Forbidden builtins, modules, and identifiers
FORBIDDEN_NAMES: Set[str] = {
    "exec", "eval", "open", "compile", "globals", "locals", "vars", "input",
    "exit", "quit", "__import__", "os", "sys", "subprocess", "socket",
    "shutil", "pathlib", "pickle", "requests", "urllib"
}

# User-friendly messages for common forbidden constructs
SPECIFIC_NODE_ERRORS = {
    ast.Import: "Import statements are not allowed.",
    ast.ImportFrom: "Import statements are not allowed.",
    ast.FunctionDef: "Function definitions are not allowed.",
    ast.AsyncFunctionDef: "Function definitions are not allowed.",
    ast.ClassDef: "Class definitions are not allowed.",
    ast.Lambda: "Lambda expressions are not allowed.",
    ast.Try: "Try/Except blocks are not allowed.",
    ast.With: "With statements are not allowed.",
    ast.AsyncWith: "With statements are not allowed.",
    ast.While: "Loops are not allowed.",
    ast.For: "Loops are not allowed.",
    ast.AsyncFor: "Loops are not allowed.",
    ast.Yield: "Generators and async code are not allowed.",
    ast.YieldFrom: "Generators and async code are not allowed.",
    ast.Await: "Async code is not allowed.",
    ast.Global: "Global statements are not allowed.",
    ast.Nonlocal: "Nonlocal statements are not allowed.",
}


def clean_code(code: str) -> str:
    """
    Strips markdown code fences (```python ... ```) and leading/trailing whitespace.

    Args:
        code (str): Raw code string returned by LLM.

    Returns:
        str: Cleaned executable Python snippet.
    """
    if not code:
        return ""
    
    code_str = code.strip()
    if code_str.startswith("```python"):
        code_str = code_str[9:]
    elif code_str.startswith("```"):
        code_str = code_str[3:]
    if code_str.endswith("```"):
        code_str = code_str[:-3]
        
    return code_str.strip()


def validate_code(code_str: str) -> Tuple[bool, str]:
    """
    Parses code into AST and validates every node against a strict whitelist.

    Enforces:
        1. Absence of syntax errors.
        2. AST node structural whitelist compliance.
        3. Rejection of dangerous identifiers/attributes.
        4. Exactly one assignment to variable 'result'.

    Args:
        code_str (str): Python code string to inspect.

    Returns:
        Tuple[bool, str]: (is_valid, human_readable_message)
    """
    cleaned = clean_code(code_str)
    if not cleaned:
        return False, "Code string is empty."

    # 1. Parse AST & catch syntax errors
    try:
        tree = ast.parse(cleaned)
    except SyntaxError as e:
        return False, f"Syntax Error: {e.msg} (line {e.lineno})"
    except Exception as e:
        return False, f"Invalid code structure: {str(e)}"

    # 2. Inspect AST nodes against whitelist & safety rules
    result_assignment_count = 0

    for node in ast.walk(tree):
        node_type = type(node)

        # Check node type against structural whitelist
        if node_type not in ALLOWED_NODE_TYPES:
            error_msg = SPECIFIC_NODE_ERRORS.get(
                node_type,
                f"Unsafe construct '{node_type.__name__}' is not allowed."
            )
            return False, error_msg

        # Check forbidden identifiers
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return False, f"Access to forbidden identifier '{node.id}' is not allowed."

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, "Private or dunder attributes are not allowed."
            if node.attr in FORBIDDEN_NAMES:
                return False, f"Access to forbidden attribute '{node.attr}' is not allowed."

        # Track assignments to 'result'
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    result_assignment_count += 1
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "result":
                result_assignment_count += 1

    # 3. Enforce exactly one assignment to 'result'
    if result_assignment_count == 0:
        return False, "The generated code must assign the final output to a variable named 'result'."
    if result_assignment_count > 1:
        return False, "The variable 'result' must be assigned exactly once."

    return True, "Code is safe and valid."
