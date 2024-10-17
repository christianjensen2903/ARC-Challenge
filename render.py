from io import BytesIO
import base64
import attrs
import numpy as np
from PIL import Image
import cv2


rgb_lookup = {
    0: (200, 200, 200),  # Nothing (empty space)
    1: (205, 23, 24),  # Red (🔴)
    2: (10, 177, 1),  # Green (🟢)
    3: (0, 89, 209),  # Blue (🔵)
    4: (255, 217, 0),  # Yellow (🟡)
    5: (170, 38, 255),  # Purple (🟣)
    6: (0, 0, 0),  # Grey (⚫️)
    7: (230, 132, 0),  # Orange (🟠)
    8: (255, 255, 255),  # White (⚪️)
    9: (122, 71, 34),  # Brown (🟤)
}


def get_column_label(n):
    result = ""
    while n >= 0:
        result = chr(n % 26 + ord("A")) + result
        n = n // 26 - 1
    return result


def calculate_grid_size(
    grid: np.ndarray, cell_size: int, edge_size: int
) -> tuple[int, int]:
    height, width = grid.shape
    new_height = height * (cell_size + edge_size) + edge_size
    new_width = width * (cell_size + edge_size) + edge_size
    return new_height, new_width


def grid_to_rgb(grid: np.ndarray, cell_size: int, edge_size: int):
    grid_height, grid_width = calculate_grid_size(grid, cell_size, edge_size)

    edge_color = (85, 85, 85)  # Grey edge color

    rgb_grid = np.full((grid_height, grid_width, 3), edge_color, dtype=np.uint8)

    # Fill in the cells with the appropriate colors
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = rgb_lookup[grid[i, j]]
            y = i * (cell_size + edge_size) + edge_size
            x = j * (cell_size + edge_size) + edge_size
            rgb_grid[y : y + cell_size, x : x + cell_size] = color

    return rgb_grid


def add_grid_border(rgb_grid: np.ndarray, border_size: int):
    grid_height, grid_width = rgb_grid.shape[:2]

    total_height = grid_height + border_size * 2
    total_width = grid_width + border_size * 2
    rgb_grid_border = np.full(
        (total_height, total_width, 3), (255, 255, 255), dtype=np.uint8
    )

    # Center the grid
    rgb_grid_border[border_size:-border_size, border_size:-border_size] = rgb_grid

    return rgb_grid_border


def write_text(grid: np.ndarray, text: str, position: tuple[int, int]):
    x, y = position
    cv2.putText(
        grid,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def add_row_labels(
    grid: np.ndarray, num_rows: int, cell_size: int, edge_size: int, border_size: int
):
    width = grid.shape[1]

    grid_start = border_size + edge_size

    for i in range(num_rows):
        row_label = str(i + 1)
        y = int(grid_start + i * (cell_size + edge_size) + edge_size + cell_size // 2)

        # Center the row label horizontally
        if i < 9:
            x = int(cell_size // 2.5)
        else:
            x = int(cell_size // 5)

        write_text(grid, row_label, (x, y))
        write_text(grid, row_label, (int(width - border_size + x), y))


def add_column_labels(
    grid: np.ndarray, num_columns: int, cell_size: int, edge_size: int, border_size: int
):
    height = grid.shape[0]

    grid_start = border_size + edge_size

    for i in range(num_columns):
        col_label = get_column_label(i)

        # Center the column label horizontally
        offset = 5  # Found by trial and error
        if i >= 26:  # number of letters in the alphabet
            offset *= 2

        x = int(grid_start + i * (cell_size + edge_size) + cell_size // 2 - offset)
        y = int(cell_size // 1.5)

        write_text(grid, col_label, (x, y))
        write_text(grid, col_label, (x, int(height - border_size + y)))


def create_rgb_grid(grid: np.ndarray, cell_size: int, edge_size: int):
    rows, columns = grid.shape
    rgb_grid = grid_to_rgb(grid, cell_size, edge_size)
    rgb_grid = add_grid_border(rgb_grid, border_size=cell_size)

    add_row_labels(rgb_grid, rows, cell_size, edge_size, border_size=cell_size)
    add_column_labels(rgb_grid, columns, cell_size, edge_size, border_size=cell_size)

    return rgb_grid


def grid_to_base64_png_oai_content(grid: np.ndarray, cell_size: int, edge_size: int):

    rgb_grid = create_rgb_grid(grid, cell_size, edge_size)
    image = Image.fromarray(rgb_grid, "RGB")

    output = BytesIO()
    image.save(output, format="PNG")
    base64_png = base64.b64encode(output.getvalue()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_png}",
        },
    }


if __name__ == "__main__":
    initial_values = np.random.randint(0, 9, (30, 30))

    rgb_grid = create_rgb_grid(initial_values, cell_size=40, edge_size=3)

    image = Image.fromarray(rgb_grid, "RGB")
    image.show()
