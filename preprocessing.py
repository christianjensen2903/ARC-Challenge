from scipy.ndimage import label
import numpy as np

letter_lookup = {
    0: " ",
    1: "R",  # Red
    2: "G",  # Green
    3: "B",  # Blue
    4: "Y",  # Yellow
}

rgb_lookup = {
    0: (0, 0, 0),  # Black
    1: (220, 50, 32),  # Red
    2: (46, 204, 64),  # Green
    3: (0, 116, 217),  # Blue
    4: (255, 220, 0),  # Yellow
}

from matplotlib import pyplot as plt


def replace_with_colors(grid: np.ndarray) -> np.ndarray:
    return np.array([[rgb_lookup[value] for value in row] for row in grid])


# Function to reduce the color scheme to a smaller set of colors
def reduce_colors(grid: np.ndarray) -> np.ndarray:
    unique_colors = np.unique(grid)
    color_mapping = {color: i for i, color in enumerate(unique_colors)}
    return np.vectorize(color_mapping.get)(grid)


def grid_to_ascii(grid: np.ndarray) -> str:
    grid_str = ""
    for row in grid:
        ascii_row = "|".join([letter_lookup[value] for value in row])
        grid_str += ascii_row + "\n"

    return grid_str


def find_contiguous_shapes(grid):
    # Structure is to define the neighborhood for connected components
    labeled_array, num_features = label(grid > 0, structure=np.ones((3, 3)))
    shape_grids = []
    for shape_label in range(1, num_features + 1):
        shape_mask = labeled_array == shape_label

        # Find the bounding box of the shape
        rows, cols = np.where(shape_mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Extract the sub-grid containing the shape, cropping to the bounding box
        shape_grid = grid[min_row : max_row + 1, min_col : max_col + 1]

        shape_grids.append(shape_grid)

    return shape_grids
