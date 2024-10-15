import numpy as np
import sys
import io
import traceback
import contextlib


def convert_to_int(value):
    # Check if the value is a string of a number or space
    if isinstance(value, str):
        if value == " ":
            return 0  # Convert space to 0
        else:
            return int(value)  # Convert string numbers to integers
    return value  # If it's already an integer, return it as is


def convert_output_to_int(output: np.ndarray) -> np.ndarray:
    # Create an integer array of the same shape as the output
    int_output = np.empty(output.shape, dtype=int)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # Convert each element to integer using the helper function
            int_output[i, j] = convert_to_int(output[i, j])

    return int_output


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

        out = convert_output_to_int(out)

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
