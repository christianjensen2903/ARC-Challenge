from scipy.ndimage import label  # type: ignore
import numpy as np
from dataclasses import dataclass, field
import json
from models import Demonstration
from utils import get_column_label


@dataclass(eq=False)
class Shape:
    """
    A shape is a contiguous region of a grid.
    min_row and min_col are the coordinates of the top-left corner of the shape.
    """

    grid: np.ndarray
    min_row: int
    min_col: int

    def __eq__(self, other):
        """Check if two shapes are equal based on their grid and positions."""
        if not isinstance(other, Shape):
            return False
        return (
            np.array_equal(self.grid, other.grid)
            and self.min_row == other.min_row
            and self.min_col == other.min_col
        )

    def __hash__(self):
        """Hash the shape based on the grid and position, so it can be added to sets."""
        return hash((self.min_row, self.min_col, self.grid.tobytes()))


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

            # Append the shape along with its top-left corner position (min_row, min_col)
            shapes_with_positions.append(Shape(shape_grid, min_row, min_col))

        return shapes_with_positions

    def find_shapes(self, grid: np.ndarray) -> list[Shape]:
        shapes = self.find_contiguous_shapes(grid, mask=grid > 0)
        unique_colors = np.unique(grid)
        for color in unique_colors:
            color_mask = grid == color
            shapes += self.find_contiguous_shapes(grid, mask=color_mask)

        # Remove duplicates
        return list(set(shapes))

    def find_shape(self, grid: np.ndarray, shape: Shape) -> list[Shape]:
        grid_rows, grid_cols = grid.shape
        shape_rows, shape_cols = shape.grid.shape

        if shape_rows > grid_rows or shape_cols > grid_cols:
            return []

        # List to store the top-left corner of the matching shape in the grid
        matches = []

        # Iterate over each position in the grid where the shape could fit
        for i in range(grid_rows - shape_rows + 1):
            for j in range(grid_cols - shape_cols + 1):
                # Extract the sub-grid from the current position
                sub_grid = grid[i : i + shape_rows, j : j + shape_cols]

                # Check if the sub-grid matches the shape
                if np.array_equal(sub_grid, shape.grid):
                    matches.append(Shape(sub_grid, i, j))

        return matches

    def _get_shape_bounds(self, shape: Shape) -> tuple[int, int, int, int]:
        min_row = shape.min_row + 1
        max_row = shape.min_row + shape.grid.shape[0]
        min_col = shape.min_col + 1
        max_col = shape.min_col + shape.grid.shape[1]
        return min_row, max_row, min_col, max_col

    def _get_location_text(self, row: int, col: int) -> str:
        return f"{get_column_label(col)}{row}"

    def _get_shape_bounds_text(self, shape: Shape) -> str:
        min_row, max_row, min_col, max_col = self._get_shape_bounds(shape)
        return f"{self._get_location_text(min_row, min_col)}-{self._get_location_text(max_row, max_col)}"

    def find_interesting_shapes(
        self, demonstration: Demonstration
    ) -> list[tuple[Shape, str]]:
        input_shapes = self.find_shapes(demonstration.input)
        # output_shapes = self.find_shapes(demonstration.output)
        # shapes = input_shapes + output_shapes
        interesting_shapes: list[tuple[Shape, str]] = []
        for input_shape in input_shapes:

            matches = self.find_shape(demonstration.output, input_shape)
            description = ""
            if len(matches) == 0:
                continue
            elif len(matches) == 1:
                output_shape = matches[0]
                if input_shape == output_shape:
                    description = (
                        f"Stays unchanged at {self._get_shape_bounds_text(input_shape)}"
                    )
                else:
                    if (
                        input_shape.min_row > output_shape.min_row
                        and input_shape.min_col == output_shape.min_col
                    ):
                        description = f"Moves up {input_shape.min_row - output_shape.min_row} rows from {self._get_shape_bounds_text(input_shape)} to {self._get_shape_bounds_text(output_shape)}"
                    elif (
                        input_shape.min_row < output_shape.min_row
                        and input_shape.min_col == output_shape.min_col
                    ):
                        description = f"Moves down {output_shape.min_row - input_shape.min_row} rows from {self._get_shape_bounds_text(input_shape)} to {self._get_shape_bounds_text(output_shape)}"
                    elif (
                        input_shape.min_row == output_shape.min_row
                        and input_shape.min_col < output_shape.min_col
                    ):
                        description = f"Moves right {output_shape.min_col - input_shape.min_col} columns from {self._get_shape_bounds_text(input_shape)} to {self._get_shape_bounds_text(output_shape)}"
                    elif (
                        input_shape.min_row == output_shape.min_row
                        and input_shape.min_col > output_shape.min_col
                    ):
                        description = f"Moves left {input_shape.min_col - output_shape.min_col} columns from {self._get_shape_bounds_text(input_shape)} to {self._get_shape_bounds_text(output_shape)}"
                    else:
                        description = f"Changes position from {self._get_shape_bounds_text(input_shape)} to {self._get_shape_bounds_text(output_shape)}"
            else:
                to_locations = []
                for match in matches:
                    to_locations.append(self._get_shape_bounds_text(match))
                to_locations_str = ", ".join(to_locations)
                description = f"Is duplicated from {self._get_shape_bounds_text(input_shape)} to {to_locations_str}"
            interesting_shapes.append((input_shape, description))

        return interesting_shapes


# Should include: flips, rotations, and translations
# Should also include color transformations
# should also include string explaining the transformation

if __name__ == "__main__":

    with open("data/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)

    example_id = "05f2a901"
    challenge = challenges[example_id]["train"]
    demonstrations = [
        Demonstration(input=np.array(grids["input"]), output=np.array(grids["output"]))
        for grids in challenge
    ]

    extractor = ShapeExtractor()

    # grid = np.array(
    #     [
    #         [1, 1, 0, 0, 0],
    #         [1, 1, 0, 0, 0],
    #         [0, 0, 0, 2, 2],
    #         [0, 0, 0, 2, 2],
    #         [3, 0, 0, 0, 0],
    #     ]
    # )
    # shapes = extractor.find_shapes(grid)
    shapes = extractor.find_interesting_shapes(demonstrations[0])
    for shape in shapes:
        print(shape)
        print()
