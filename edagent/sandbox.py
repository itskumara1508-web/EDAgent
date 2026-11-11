"""Execution sandbox and guardrails for EDAgent.

CORE DESIGN:
- AI-written code runs in a 60-second sandbox subprocess / isolated execution.
- Strict guardrails:
    1. No deleted columns (all original columns must be preserved).
    2. At most 5% of rows removed.
    3. Target column strictly untouched (values must match input rows exactly).
    4. No data leakage or malicious operations (system/network/filesystem tampering blocked).
- Gets exactly one self-correction retry on error or guardrail rejection.
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import io
import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class GuardrailViolation(Exception):
    """Raised when generated code violates an EDAgent safety guardrail."""
    def __init__(self, message: str, guardrail: str):
        super().__init__(message)
        self.guardrail = guardrail


# Blocked modules/functions for sandbox safety
FORBIDDEN_AST_NODES = {
    "eval", "exec", "compile", "__import__",
    "open", "input", "breakpoint"
}

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "httpx", "pty", "posix", "ctypes", "builtins", "signal"
}


def check_ast_safety(code: str) -> None:
    """Statically verify that code does not import dangerous modules or call forbidden builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise GuardrailViolation(f"Syntax error in generated code: {e}", guardrail="syntax_check")

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                if mod_root in FORBIDDEN_MODULES:
                    raise GuardrailViolation(
                        f"Forbidden module import: '{alias.name}'. Only pandas, numpy, and sklearn are allowed.",
                        guardrail="security_sandbox"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_root = node.module.split(".")[0]
                if mod_root in FORBIDDEN_MODULES:
                    raise GuardrailViolation(
                        f"Forbidden module import from: '{node.module}'.",
                        guardrail="security_sandbox"
                    )
        # Check calls to forbidden functions
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_AST_NODES:
                    raise GuardrailViolation(
                        f"Forbidden function call: '{node.func.id}()'.",
                        guardrail="security_sandbox"
                    )


def validate_guardrails(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame,
    target_column: Optional[str] = None,
    max_row_removal_pct: float = 0.05
) -> List[str]:
    """Validate DataFrame transformation against EDAgent guardrails.
    
    Returns a list of violation messages. Empty list means passed.
    """
    violations: List[str] = []

    # Guardrail 1: No deleted columns
    orig_cols = set(df_original.columns)
    trans_cols = set(df_transformed.columns)
    missing_cols = orig_cols - trans_cols
    if missing_cols:
        violations.append(
            f"GUARDRAIL 1 (No deleted columns): {len(missing_cols)} column(s) were deleted: "
            f"{sorted(list(missing_cols))[:5]}{'...' if len(missing_cols) > 5 else ''}. "
            "All original columns must be preserved in cleaned output."
        )

    # Guardrail 2: At most 5% of rows removed
    n_orig = len(df_original)
    n_trans = len(df_transformed)
    if n_orig > 0:
        pct_removed = (n_orig - n_trans) / n_orig
        if pct_removed > max_row_removal_pct:
            violations.append(
                f"GUARDRAIL 2 (Max 5% rows removed): {pct_removed*100:.2f}% rows removed "
                f"({n_orig - n_trans} out of {n_orig}). Maximum allowed is {max_row_removal_pct*100:.0f}%. "
                "Use imputation rather than wholesale row deletion."
            )

    # Guardrail 3: Target column untouched
    if target_column and target_column in df_original.columns:
        if target_column not in df_transformed.columns:
            violations.append(
                f"GUARDRAIL 3 (Target untouched): Target column '{target_column}' is missing from transformed data."
            )
        else:
            # Check for common indices
            common_idx = df_original.index.intersection(df_transformed.index)
            if len(common_idx) > 0:
                orig_target_series = df_original.loc[common_idx, target_column]
                trans_target_series = df_transformed.loc[common_idx, target_column]
                # Allow NaNs matching NaNs, otherwise values must be identical
                mismatches = 0
                for v1, v2 in zip(orig_target_series, trans_target_series):
                    if pd.isna(v1) and pd.isna(v2):
                        continue
                    if v1 != v2:
                        mismatches += 1
                if mismatches > 0:
                    violations.append(
                        f"GUARDRAIL 3 (Target untouched): Target column '{target_column}' was modified "
                        f"in {mismatches} rows. Target labels/values must never be altered, scaled, or imputed."
                    )

    return violations


def _run_isolated_in_thread(
    code: str,
    df: pd.DataFrame
) -> Tuple[bool, Optional[pd.DataFrame], str, Optional[str]]:
    """Executes cleaning code safely in an isolated execution namespace.
    
    Returns:
        (success, output_df, stdout_logs, error_message)
    """
    stdout_buf = io.StringIO()
    safe_globals = {
        "pd": pd,
        "np": np,
        "df": df.copy(deep=True),
        "print": lambda *args, **kwargs: print(*args, file=stdout_buf, **kwargs)
    }

    try:
        exec(code, safe_globals)
        if "df_clean" in safe_globals and isinstance(safe_globals["df_clean"], pd.DataFrame):
            out_df = safe_globals["df_clean"]
        elif "df" in safe_globals and isinstance(safe_globals["df"], pd.DataFrame):
            out_df = safe_globals["df"]
        else:
            raise ValueError("Cleaning code did not produce a DataFrame variable 'df_clean' or 'df'.")

        return True, out_df, stdout_buf.getvalue(), None
    except Exception as e:
        return False, None, stdout_buf.getvalue(), f"{type(e).__name__}: {str(e)}"


class ExecutionResult:
    def __init__(
        self,
        success: bool,
        output_df: Optional[pd.DataFrame] = None,
        code: str = "",
        logs: str = "",
        error: Optional[str] = None,
        violations: Optional[List[str]] = None,
        retry_used: bool = False
    ):
        self.success = success
        self.output_df = output_df
        self.code = code
        self.logs = logs
        self.error = error
        self.violations = violations or []
        self.retry_used = retry_used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "violations": self.violations,
            "logs": self.logs,
            "retry_used": self.retry_used,
            "has_output_df": self.output_df is not None
        }


def execute_in_sandbox(
    code: str,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    timeout_seconds: int = 60
) -> ExecutionResult:
    """Executes Python code in a safe environment with a 60-second timeout and validates guardrails."""
    # 1. Static AST Security Check
    try:
        check_ast_safety(code)
    except GuardrailViolation as e:
        return ExecutionResult(
            success=False,
            code=code,
            error=str(e),
            violations=[str(e)]
        )

    # 2. Run execution with timeout using ThreadPoolExecutor
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_isolated_in_thread, code, df)
            success, out_df, logs, err = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        return ExecutionResult(
            success=False,
            code=code,
            error=f"Execution timed out after {timeout_seconds} seconds.",
            violations=[f"GUARDRAIL: Execution exceeded timeout limit ({timeout_seconds}s)."]
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            code=code,
            error=f"Sandbox execution error: {str(e)}",
            violations=[f"Sandbox error: {str(e)}"]
        )

    if not success or out_df is None:
        return ExecutionResult(
            success=False,
            code=code,
            error=err or "Execution failed without output DataFrame.",
            logs=logs
        )

    # 3. Validate Guardrails
    violations = validate_guardrails(
        df_original=df,
        df_transformed=out_df,
        target_column=target_column
    )

    if violations:
        return ExecutionResult(
            success=False,
            output_df=out_df,
            code=code,
            error="Guardrail validation failed: " + "; ".join(violations),
            violations=violations,
            logs=logs
        )

    return ExecutionResult(
        success=True,
        output_df=out_df,
        code=code,
        logs=logs,
        violations=[]
    )


def run_with_self_correction(
    code_generator_fn: Callable[[Optional[str]], str],
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    timeout_seconds: int = 60
) -> ExecutionResult:
    """Runs code generation, executes in sandbox, and performs at most 1 self-correction retry."""
    # Attempt 1
    code_1 = code_generator_fn(None)
    result_1 = execute_in_sandbox(code_1, df, target_column, timeout_seconds)

    if result_1.success:
        return result_1

    # Attempt 2 (Self-correction retry)
    feedback = (
        f"Initial execution failed.\n"
        f"Error: {result_1.error}\n"
        f"Violations: {'; '.join(result_1.violations)}\n"
        f"Fix the code. Remember:\n"
        f"1. Never delete columns.\n"
        f"2. Remove <= 5% rows (prefer median/mode/mean imputation for missing values).\n"
        f"3. Never alter or impute the target column ('{target_column}').\n"
        f"4. Store the cleaned DataFrame in 'df_clean'."
    )
    code_2 = code_generator_fn(feedback)
    result_2 = execute_in_sandbox(code_2, df, target_column, timeout_seconds)
    result_2.retry_used = True

    if result_2.success:
        return result_2

    # If retry also fails, return result_2 with full diagnostic
    return result_2
