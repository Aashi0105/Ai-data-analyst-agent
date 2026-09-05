"""
sandbox.py
----------
Purpose:
    Executes validated Python pandas code inside a restricted sandbox environment.
    Isolates global namespace, passes a deep copy of the DataFrame to prevent mutation,
    and provides user-friendly error messages with fuzzy column name suggestions on KeyError.
"""

import difflib
from typing import Tuple, Any
import pandas as pd
import numpy as np


# Whitelisted safe built-in functions for execution scope
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


def execute_pandas_code(code_str: str, df: pd.DataFrame) -> Tuple[bool, Any]:
    """
    Executes validated pandas code in a restricted, isolated scope.

    Args:
        code_str (str): AST-validated Python pandas code.
        df (pd.DataFrame): Input dataset DataFrame.

    Returns:
        Tuple[bool, Any]: (success_flag, result_object_or_user_friendly_error)
    """
    if not code_str or not code_str.strip():
        return False, "Execution failed: Code string is empty."

    # Deep copy DataFrame so operations (e.g. df['col'] = ...) never alter original data
    df_sandbox = df.copy(deep=True)

    restricted_globals = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
    }

    restricted_locals = {
        "df": df_sandbox
    }

    try:
        exec(code_str, restricted_globals, restricted_locals)

        if "result" in restricted_locals:
            return True, restricted_locals["result"]
        elif "result" in restricted_globals:
            return True, restricted_globals["result"]
        else:
            return False, "Execution completed, but output variable 'result' was not assigned."

    except KeyError as ke:
        missing_col = str(ke).strip("'\"")
        str_columns = [str(col) for col in df.columns]
        matches = difflib.get_close_matches(missing_col, str_columns, n=3, cutoff=0.6)
        if matches:
            suggestion = f" Did you mean one of these columns: {', '.join(repr(m) for m in matches)}?"
        else:
            suggestion = ""
        return False, f"Column '{missing_col}' was not found in the dataset.{suggestion}"

    except AttributeError as ae:
        return False, f"Data operation error: {str(ae)}"

    except TypeError as te:
        return False, f"Data type mismatch: {str(te)}"

    except ZeroDivisionError:
        return False, "Calculation error: Division by zero encountered."

    except Exception as e:
        return False, f"Execution failed: {type(e).__name__} - {str(e)}"
