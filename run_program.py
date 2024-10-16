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


# Ensure that diff is working
# Try to fix prompt until it can predict correctly


def run_program(
    solution: str,
    input_grid: np.ndarray,
    catch_error: bool = True,
) -> tuple[np.ndarray, str, str]:

    input_copy = input_grid.copy()

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    globals_before__ = globals().copy()

    try:
        # Redirect stdout and stderr
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            exec(solution, globals(), globals())

            out = transform(input_copy)  # type: ignore

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

        out = convert_output_to_int(out)

    except Exception as e:
        if not catch_error:
            raise e

        out = input_copy
        stdout = ""
        stderr = traceback.format_exc()
    finally:
        globals().update(globals_before__)
        for k in set(globals().keys()) - globals_before__.keys():
            del globals()[k]

    if not isinstance(out, np.ndarray):
        out = input_copy

    if out.ndim != 2:
        out = input_copy

    out = np.clip(out, 0, 9)

    # Make sure that there is at max 30 rows and 30 columns
    out = out[:30, :30]

    return out, stdout, stderr


if __name__ == "__main__":
    solution = """
import numpy as np
from scipy.ndimage import label

def transform(grid: np.ndarray) -> np.ndarray:
    teal_color = 2  # 🟢
    fill_color = 0  # ⚪️

    # Find the connected components of the teal color
    labeled_array, num_features = label(grid == teal_color)

    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        if coords.size > 0:
            # Create a mask for the region
            min_row, min_col = np.min(coords, axis=0)
            max_row, max_col = np.max(coords, axis=0)
            
            # Fill interior cells with fill_color
            grid[min_row+1:max_row, min_col+1:max_col] = fill_color

    return grid
"""
    input_grid = np.array([[1, 2], [3, 4]])
    out, stdout, stderr = run_program(solution, input_grid)
    print(out)
    print(stdout)
    print(stderr)
