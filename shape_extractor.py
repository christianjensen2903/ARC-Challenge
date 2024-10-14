from scipy.ndimage import label  # type: ignore
import numpy as np
from dataclasses import dataclass


@dataclass
class Shape:
    """
    A shape is a contiguous region of a grid.
    min_row and min_col are the coordinates of the top-left corner of the shape.
    """

    grid: np.ndarray
    min_row: int
    min_col: int


class ShapeExtractor:

    def find_contiguous_shapes(
        self, grid: np.ndarray, mask: np.ndarray | None = None
    ) -> list[Shape]:

        grid_copy = grid.copy()
        if mask is not None:
            grid_copy[~mask] = 0

        # Structure is to define the neighborhood for connected components
        labeled_array, num_features = label(mask, structure=np.ones((3, 3)))
        shapes_with_positions: list[Shape] = []
        for shape_label in range(1, num_features + 1):
            shape_mask = labeled_array == shape_label

            # Find the bounding box of the shape
            rows, cols = np.where(shape_mask)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Extract the sub-grid containing the shape, cropping to the bounding box
            shape_grid = grid_copy[min_row : max_row + 1, min_col : max_col + 1]

            # Filter out small shapes (e.g., 1 or 2 pixels)
            if np.sum(np.where(shape_grid > 0, 1, 0)) <= 1:
                continue

            # Filter out shapes that are identical to the entire grid
            if np.array_equal(shape_grid, grid):
                continue

            # Append the shape along with its top-left corner position (min_row, min_col)
            shapes_with_positions.append(Shape(shape_grid, min_row, min_col))

        return shapes_with_positions

    def find_shapes(self, grid: np.ndarray) -> list[Shape]:
        shapes = self.find_contiguous_shapes(grid, mask=grid > 0)
        unique_colors = np.unique(grid)
        for color in unique_colors:
            color_mask = grid == color
            shapes += self.find_contiguous_shapes(grid, mask=color_mask)
        return shapes


if __name__ == "__main__":
    extractor = ShapeExtractor()

    grid = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2],
            [3, 0, 0, 0, 0],
        ]
    )
    shapes = extractor.find_shapes(grid)
    for shape in shapes:
        print(shape.grid)
        print()
