from scipy.ndimage import label
import numpy as np
import json

letter_lookup = {
    0: " ",  # Black
    1: "A",  # Red
    2: "B",  # Green
    3: "C",  # Blue
    4: "D",  # Yellow
    5: "E",  # Purple
    6: "F",  # Cyan
    7: "G",  # Orange
    8: "H",  # Pink
    9: "I",  # Brown
}

index_lookup = {value: key for key, value in letter_lookup.items()}

rgb_lookup = {
    0: (0, 0, 0),  # Black
    1: (220, 50, 32),  # Red
    2: (46, 204, 64),  # Green
    3: (0, 116, 217),  # Blue
    4: (255, 220, 0),  # Yellow
    5: (170, 170, 170),  # Purple
    6: (255, 133, 27),  # Cyan
    7: (255, 65, 54),  # Orange
    8: (255, 0, 220),  # Pink
    9: (133, 20, 75),  # Brown
}


def load_data() -> tuple[dict, dict]:
    with open("data/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)

    with open("data/arc-agi_training_solutions.json") as f:
        solutions = json.load(f)

    return challenges, solutions


def replace_with_colors(grid: np.ndarray) -> np.ndarray:
    return np.array([[rgb_lookup[value] for value in row] for row in grid])


# Function to reduce the color scheme to a smaller set of colors
def reduce_colors(
    grid: np.ndarray, color_mapping: dict[int, int] | None = None
) -> np.ndarray:
    if not color_mapping:
        unique_colors = np.unique(grid)
        color_mapping = {color: i for i, color in enumerate(unique_colors)}
    return np.vectorize(color_mapping.get)(grid)


def grid_to_ascii(grid: np.ndarray, min_row: int = 0, min_col: int = 0) -> str:
    height, width = grid.shape
    alphabet = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    # Adjust for min_col in the header
    grid_str = "   " + " ".join(alphabet[min_col : min_col + width]) + "\n"

    for i, row in enumerate(grid):
        # Adjust for min_row in the row label
        ascii_row = f"{alphabet[min_row + i]} |"
        for value in row:
            char = value if value > 0 else " "
            ascii_row += f"{char}|"
        grid_str += ascii_row + "\n"

    return grid_str


def parse_ascii_grid(grid: str) -> np.ndarray:
    grid = grid.strip().split("\n")
    result = []
    for row in grid[1:]:
        for value in row[3:-1].split("|"):
            if value == " ":
                result.append(0)
            else:
                result.append(int(value))
    return np.array(result)
