"""
utils.py
--------
Purpose:
    Contains dataset inspection and utility helper functions.
    Handles CSV loading with encoding fallbacks, dataset schema extraction,
    and analytical scalar value formatting.
"""

from typing import Dict, Any
import pandas as pd


def load_csv(file_path_or_buffer: Any) -> pd.DataFrame:
    """
    Loads a CSV file or file-like object into a pandas DataFrame.
    Supports encoding fallback (utf-8, latin1, iso-8859-1, cp1252) and validates non-emptiness.

    Args:
        file_path_or_buffer (Any): Path to file or uploaded file buffer.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        ValueError: If file is empty, invalid CSV format, or cannot be parsed.
    """
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    
    for encoding in encodings:
        try:
            if hasattr(file_path_or_buffer, "seek"):
                file_path_or_buffer.seek(0)
            df = pd.read_csv(file_path_or_buffer, encoding=encoding)
            
            if df.empty:
                raise ValueError("The uploaded CSV file is empty and contains no data rows.")
            
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded CSV file contains no data.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV file structure: {str(e)}")
        except Exception as e:
            if not isinstance(e, ValueError):
                raise ValueError(f"Unable to read CSV file: {str(e)}")
            raise e

    raise ValueError("Failed to read CSV file due to unsupported file encoding.")


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts summary statistics and schema details from a DataFrame.

    Args:
        df (pd.DataFrame): Target DataFrame.

    Returns:
        Dict[str, Any]: Summary metadata dictionary containing shape, missing count, duplicates, schema_df, and head.
    """
    num_rows, num_cols = df.shape
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())
    duplicate_rows = int(df.duplicated().sum())
    
    schema_data = {
        "Column Name": list(df.columns),
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": list(missing_series.values),
        "Missing %": [round((val / num_rows) * 100, 2) if num_rows > 0 else 0.0 for val in missing_series.values]
    }
    schema_df = pd.DataFrame(schema_data)
    
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_per_col": missing_series.to_dict(),
        "schema_df": schema_df,
        "head": df.head(10)
    }


def format_value(value: Any) -> str:
    """
    Formats analytical result values into human-readable string formats.

    Args:
        value (Any): Result scalar, Series, or object.

    Returns:
        str: Formatted string presentation.
    """
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    elif isinstance(value, int):
        return f"{value:,}"
    return str(value)
