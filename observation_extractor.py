import preprocessing
from scipy.ndimage import label  # type: ignore
import numpy as np


class ObservationExtractor:

    def find_contiguous_shapes(self, grid: np.ndarray, mask: np.ndarray | None = None):
        # Structure is to define the neighborhood for connected components
        labeled_array, num_features = label(mask, structure=np.ones((3, 3)))
        shapes_with_positions = []
        for shape_label in range(1, num_features + 1):
            shape_mask = labeled_array == shape_label

            # Find the bounding box of the shape
            rows, cols = np.where(shape_mask)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Extract the sub-grid containing the shape, cropping to the bounding box
            shape_grid = grid[min_row : max_row + 1, min_col : max_col + 1]

            # Filter out small shapes (e.g., 1 or 2 pixels)
            if shape_grid.size <= 2:
                continue

            # Filter out shapes that are identical to the entire grid
            if np.array_equal(shape_grid, grid):
                continue

            # Append the shape along with its top-left corner position (min_row, min_col)
            shapes_with_positions.append((shape_grid, (min_row, min_col)))

        return shapes_with_positions

    def find_shapes(self, grid: np.ndarray):
        shapes = self.find_contiguous_shapes(grid, mask=grid > 0)
        unique_colors = np.unique(grid)
        for color in unique_colors:
            color_mask = grid == color
            color_grid = grid.copy()
            color_grid[~color_mask] = 0
            shapes += self.find_contiguous_shapes(color_grid, mask=color_mask)
        return shapes

    def get_shape_string(self, grid: np.ndarray):
        shape_string = ""
        shapes = self.find_shapes(grid)
        for i, (shape, (min_row, min_col)) in enumerate(shapes):
            shape_string += f"Shape {i}:\n"
            shape_string += preprocessing.grid_to_ascii(shape, min_row, min_col)
            shape_string += "\n"

        shape_string += "\n"
        return shape_string

    def extract(self, examples: list[dict]) -> str:
        observation_str = ""
        for i, example in enumerate(examples):
            observation_str += f"Example {i+1}:\n"
            input_grid = np.array(example["input"])
            observation_str += f"Input shapes:\n"
            observation_str += self.get_shape_string(input_grid)

            output_grid = np.array(example["output"])
            observation_str += f"Output shapes:\n"
            observation_str += self.get_shape_string(output_grid)
            observation_str += "\n"

        return observation_str


if __name__ == "__main__":
    extractor = ObservationExtractor()
    challenges, solutions = preprocessing.load_data()
    example_id = "e40b9e2f"
    observations = extractor.extract(challenges[example_id]["train"])
    print(observations)
