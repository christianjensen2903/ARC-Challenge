import numpy as np
import json

letter_lookup = {
    0: "  ",  # Nothing
    1: "🔴",  # Red
    2: "🟢",  # Green
    3: "🔵",  # Blue
    4: "🟡",  # Yellow
    5: "🟣",  # Purple
    6: "⚫️",  # Black
    7: "🟠",  # Orange
    8: "⚪️",  # White
    9: "🟤",  # Brown
}

index_lookup = {value: key for key, value in letter_lookup.items()}

rgb_lookup = {
    0: (0, 0, 0),  # Nothing (empty space)
    1: (255, 0, 0),  # Red (🔴)
    2: (0, 255, 0),  # Green (🟢)
    3: (0, 0, 255),  # Blue (🔵)
    4: (255, 255, 0),  # Yellow (🟡)
    5: (128, 0, 128),  # Purple (🟣)
    6: (128, 128, 128),  # Black (⚫️)
    7: (255, 165, 0),  # Orange (🟠)
    8: (255, 255, 255),  # White (⚪️)
    9: (139, 69, 19),  # Brown (🟤)
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
    grid_str = "    " + "  ".join(alphabet[min_col : min_col + width]) + "\n"

    for i, row in enumerate(grid):
        # Adjust for min_row in the row label
        ascii_row = str(i) + ("  " if i < 10 else " ") + "|"
        for value in row:
            char = letter_lookup[value]
            ascii_row += f"{char}|"
        grid_str += ascii_row + "\n"

    return grid_str


def parse_ascii_grid(grid: str) -> np.ndarray:
    grid_list = grid.strip().split("\n")
    result = []
    for row in grid_list[1:]:
        row_result: list[int] = []
        for value in row[4:-1].split("|"):
            if value == "  ":
                row_result.append(0)
            else:
                row_result.append(index_lookup[value])
        result.append(row_result)
    return np.array(result)


if __name__ == "__main__":
    grid = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 9]])
    ascii_grid = grid_to_ascii(grid)
    print(ascii_grid)
    print(parse_ascii_grid(ascii_grid))
