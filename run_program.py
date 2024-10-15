import numpy as np
import sys
import io
import traceback
import contextlib


def run_program(
    solution: str,
    input_grid: np.ndarray,
    catch_error: bool = True,
    locals: dict | None = None,
) -> tuple[np.ndarray, str, str]:
    if locals is None:
        locals = {}

    input_copy = input_grid.copy()

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Redirect stdout and stderr
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            exec(solution, globals(), locals)
            out = locals["transform"](input_copy)

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

    except Exception as e:
        if not catch_error:
            raise e

        out = input_copy
        stdout = ""
        stderr = traceback.format_exc()

    if not isinstance(out, np.ndarray):
        out = input_copy

    if out.ndim != 2:
        out = input_copy

    if out.dtype != np.int32:
        out = input_copy

    return out, stdout, stderr


if __name__ == "__main__":
    solution = """
def transform(input_grid: np.ndarray) -> np.ndarray:
    return input_grid
"""
    input_grid = np.array([[1, 2], [3, 4]])
    out, stdout, stderr = run_program(solution, input_grid)
    print(out)
    print(stdout)
    print(stderr)
