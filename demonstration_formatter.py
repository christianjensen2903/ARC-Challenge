from abc import ABC, abstractmethod
import numpy as np
import string
import json
from models import Demonstration
from shape_extractor import ShapeExtractor


class DemonstrationFormatter(ABC):

    def __init__(self):

        self.column_names = list(string.ascii_uppercase) + [
            "A" + c for c in list(string.ascii_uppercase)
        ]
        self.row_names = [str(i) for i in range(1, len(self.column_names) + 2)]

    @abstractmethod
    def char_to_text(self, char: int) -> str:
        pass

    @abstractmethod
    def grid_to_text(self, grid: np.ndarray) -> str:
        pass

    @abstractmethod
    def format(self, demonstrations: list[Demonstration]) -> str:
        pass

    @abstractmethod
    def get_description(self, demonstrations: list[Demonstration]) -> str:
        pass

    def extra_helper_text(self, demonstrations: list[Demonstration]) -> str:
        return ""


class RawDemonstrations(DemonstrationFormatter):

    def char_to_text(self, char: int) -> str:
        return str(char)

    def grid_to_text(self, grid: np.ndarray) -> str:
        return "\n".join(
            ["".join([self.char_to_text(char) for char in row]) for row in grid]
        )

    def format(self, demonstrations: list[Demonstration]) -> str:
        return "\n".join(
            [json.dumps(demonstration) for demonstration in demonstrations]
        )

    def get_description(self, demonstrations: list[Demonstration]) -> str:
        return ""


class EmojisDemonstrations(DemonstrationFormatter):
    """
    Convert the input and output grids to emoji format.
    """

    def __init__(self):
        super().__init__()
        self.letter_lookup = {
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

    def char_to_text(self, char: int) -> str:
        return self.letter_lookup[char]

    def _get_column_names(self, width: int) -> str:
        column_names = "    "
        for i in range(width):
            column_names += self.column_names[i]
            column_names += "  " if len(self.column_names[i]) == 1 else " "
        return column_names

    def grid_to_text(self, grid: np.ndarray) -> str:
        height, width = grid.shape
        grid_str = self._get_column_names(width)
        grid_str += "\n"

        for i, row in enumerate(grid):
            text_row = self.row_names[i] + ("  " if i < 9 else " ") + "|"
            for value in row:
                char = self.char_to_text(value)
                text_row += f"{char}|"
            grid_str += text_row + " " + self.row_names[i] + "\n"

        grid_str += self._get_column_names(width)

        return grid_str

    def _demonstration_to_text(self, demonstration: dict) -> tuple[str, str]:
        """Convert the input and output grids to ASCII format"""
        input_grid = np.array(demonstration["input"])
        output_grid = np.array(demonstration["output"])

        input_ascii = self.grid_to_text(input_grid)
        output_ascii = self.grid_to_text(output_grid)

        return input_ascii, output_ascii

    def format(self, demonstrations: list[Demonstration]) -> str:
        formatted_demonstrations = ""
        for i, demonstration in enumerate(demonstrations):
            formatted_demonstrations += f"Demonstration {i+1}:\n"
            formatted_demonstrations += f"Input:\n"
            formatted_demonstrations += self.grid_to_text(demonstration.input) + "\n"
            formatted_demonstrations += f"Output:\n"
            formatted_demonstrations += self.grid_to_text(demonstration.output) + "\n"

        return formatted_demonstrations

    def get_description(self, demonstrations: list[Demonstration]) -> str:
        return f"""
The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive).
These grids will be shown to you in text format using emojis.

To represent the integers in the grid, we use emojis. The mapping is as follows:
{self.letter_lookup}.
All 0s are shown as spaces as these represent empty locations.
The elements of the grid are separated by '|'.

Locations are denoted like A7 or D3, where columns are denoted with A, B, C, etc. and rows are denoted with 1, 2, 3, etc.
So, D3 corresponds to the cell in the 4th column and the 3rd row. Note that rows are 1-indexed.
The bounds of a shape are shown as "A7-D3", which means the shape is from the 7th column of row A to the 3rd column of row D.
"""


class ASCIIDemonstrations(DemonstrationFormatter):
    """
    Convert the input and output grids to ASCII format.
    """

    def char_to_text(self, char: int) -> str:
        return str(char) if char > 0 else " "

    def _get_column_names(self, width: int) -> str:
        column_names = "    "
        for i in range(width):
            column_names += self.column_names[i]
            column_names += " "
        return column_names

    def grid_to_text(self, grid: np.ndarray) -> str:
        height, width = grid.shape
        grid_str = self._get_column_names(width)
        grid_str += "\n"

        for i, row in enumerate(grid):
            text_row = self.row_names[i] + ("  " if i < 9 else " ") + "|"
            for value in row:
                char = self.char_to_text(value)
                text_row += f"{char}|"
            grid_str += text_row + " " + self.row_names[i] + "\n"
        grid_str += self._get_column_names(width)

        return grid_str

    def format(self, demonstrations: list[Demonstration]) -> str:
        formatted_demonstrations = ""
        for i, demonstration in enumerate(demonstrations):
            formatted_demonstrations += f"Demonstration {i+1}:\n"
            formatted_demonstrations += f"Input:\n"
            formatted_demonstrations += self.grid_to_text(demonstration.input) + "\n"
            formatted_demonstrations += f"Output:\n"
            formatted_demonstrations += self.grid_to_text(demonstration.output) + "\n"

        return formatted_demonstrations

    def get_description(self, demonstrations: list[Demonstration]) -> str:

        self.color_lookup = {
            0: "Grey",
            1: "Red",
            2: "Green",
            3: "Blue",
            4: "Yellow",
            5: "Purple",
            6: "Black",
            7: "Orange",
            8: "White",
            9: "Brown",
        }
        return f"""
The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive).
These grids will be shown to you as an ASCII representation.

The elements of the grid are separated by '|'.
All 0s are shown as spaces as these represent empty locations (grey).

All integers represent a color. The mapping is as follows:
{self.color_lookup}

Locations are denoted like A7 or D3, where columns are denoted with A, B, C, etc. and rows are denoted with 1, 2, 3, etc.
So, D3 corresponds to the cell in the 4th column and the 3rd row. Note that rows are 1-indexed.
"""


class ShapeExtractionWrapper(DemonstrationFormatter):
    """
    Add shape information to the demonstrations.
    """

    def __init__(self, formatter: DemonstrationFormatter, max_shapes: int = 10):
        self.formatter = formatter
        self.shape_extractor = ShapeExtractor()
        self.max_shapes = max_shapes

    def char_to_text(self, char: int) -> str:
        return self.formatter.char_to_text(char)

    def grid_to_text(self, grid: np.ndarray) -> str:
        return self.formatter.grid_to_text(grid)

    def _few_shapes(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """
        Check if the number of shapes in the input and output grids is small enough to be described
        """
        input_shapes = self.shape_extractor.find_shapes(input_grid)
        output_shapes = self.shape_extractor.find_shapes(output_grid)

        return (
            len(input_shapes) <= self.max_shapes
            and len(output_shapes) <= self.max_shapes
        )

    def format(self, demonstrations: list[Demonstration]) -> str:
        return self.formatter.format(demonstrations)

    def extra_helper_text(self, demonstrations: list[Demonstration]) -> str:
        helper_text = self.formatter.get_description(demonstrations)

        # Only add shape information if there are few shapes
        for demonstration in demonstrations:
            if not self._few_shapes(demonstration.input, demonstration.output):
                return helper_text

        helper_text += "\n\n"
        helper_text += (
            "Here are some of the interesting shapes in the input and output:\n"
        )

        for i, demonstration in enumerate(demonstrations):
            interesting_shapes = self.shape_extractor.find_interesting_shapes(
                demonstration
            )
            helper_text += f"Demonstration {i+1}:\n"
            helper_text += f"Input shapes:\n"
            for j, (shape, description) in enumerate(interesting_shapes):
                helper_text += f"Shape {j+1} - {description}:\n"
                helper_text += self.grid_to_text(shape.grid)
                helper_text += "\n"

        return helper_text

    def get_description(self, demonstrations: list[Demonstration]) -> str:
        prompt_extension = self.formatter.get_description(demonstrations)
        for demonstration in demonstrations:
            if not self._few_shapes(demonstration.input, demonstration.output):
                return prompt_extension
        prompt_extension += """
In addition to the grids, you will be shown some of the possibly interesting shapes in the input and output.
A shape can both be a contiguous region of a single color, or a mix of multiple colors.
We will show you shapes that present in both the input and output and how they have changed.

We will show the shapes in a "normalized" form.
This shows the shape with the coordinates shifted such that the minimum row/column of the shape is row 1 and column A.
This is useful for tasks like noticing identical shapes (in different positions with different colors).

Apart from that the shapes are shown similar to the grids, with the elements separated by '|'.
"""

        return prompt_extension


class DifferenceWrapper(DemonstrationFormatter):
    """
    Add the difference between the input and output grids to the demonstrations.
    Differences are described by "INPUT_COLOR to OUTPUT_COLOR at POSITION(S)"
    """

    def __init__(self, formatter: DemonstrationFormatter):
        super().__init__()
        self.formatter = formatter

    def char_to_text(self, char: int) -> str:
        return self.formatter.char_to_text(char)

    def grid_to_text(self, grid: np.ndarray) -> str:
        return self.formatter.grid_to_text(grid)

    def _diff_is_small(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """
        Check if the difference between the input and output grids is small enough to be described.
        """
        if input_grid.shape != output_grid.shape:
            return False

        differs: np.ndarray = input_grid != output_grid
        count_differs = differs.sum()
        if count_differs > 50 and (
            count_differs > 0.35 * input_grid.size or count_differs > 150
        ):
            return False

        grid_differs_x, grid_differs_y = differs.nonzero()

        all_color_pairs = set()
        for x, y in zip(grid_differs_x.tolist(), grid_differs_y.tolist()):
            all_color_pairs.add((input_grid[x, y], output_grid[x, y]))

        if len(all_color_pairs) > 8:
            return False

        return True

    def format(self, demonstrations: list[Demonstration]) -> str:
        formatted_demonstrations = self.formatter.format(demonstrations)

        # Only add difference if difference is small
        for demonstration in demonstrations:
            if not self._diff_is_small(demonstration.input, demonstration.output):
                return formatted_demonstrations

        formatted_demonstrations += "\n\n"
        formatted_demonstrations += (
            "Here are the differences between the input and output:\n"
        )

        for i, demonstration in enumerate(demonstrations):
            formatted_demonstrations += f"Demonstration {i+1}:\n"

            input_grid = demonstration.input
            output_grid = demonstration.output
            differs: np.ndarray = input_grid != output_grid
            grid_differs_x, grid_differs_y = differs.nonzero()

            # Dictionary to gather all positions by color change
            diff_map: dict[tuple[str, str], list[str]] = {}

            for x, y in zip(grid_differs_x.tolist(), grid_differs_y.tolist()):
                input_color = input_grid[x, y]
                output_color = output_grid[x, y]

                # Group only non-zero colors
                if input_color == 0 or output_color == 0:
                    continue

                input_text = self.char_to_text(input_color)
                output_text = self.char_to_text(output_color)
                position = f"{self.column_names[y]}{self.row_names[x]}"

                # Group by input and output color
                color_key = (input_text, output_text)

                if color_key not in diff_map:
                    diff_map[color_key] = []

                diff_map[color_key].append(position)

            # Add gathered differences to the formatted text
            for (input_text, output_text), positions in diff_map.items():
                formatted_demonstrations += (
                    f"{input_text} to {output_text} at {', '.join(positions)}\n"
                )

            formatted_demonstrations += "\n"

        return formatted_demonstrations

    def get_description(self, demonstrations: list[Demonstration]) -> str:
        prompt_extension = self.formatter.get_description(demonstrations)
        for demonstration in demonstrations:
            if not self._diff_is_small(demonstration.input, demonstration.output):
                return prompt_extension

        prompt_extension += """
In addition to the grids, you will be shown the color changes between the input grid and the output grid

This shows the difference between an input grid and an output grid as a list of the locations where one color changes to another.
For instance, if element1 changes to element2 at A1 A2 B7, this would be represented as "element1 to element2 at A1 A2 B7".
"""

        return prompt_extension


class RotateWrapper(DemonstrationFormatter):
    """
    Add the rotated version of the output grid to the demonstrations.
    """

    def __init__(self, formatter: DemonstrationFormatter):
        super().__init__()
        self.formatter = formatter

    def char_to_text(self, char: int) -> str:
        return self.formatter.char_to_text(char)

    def grid_to_text(self, grid: np.ndarray) -> str:
        return self.formatter.grid_to_text(grid)

    def format(self, demonstrations: list[Demonstration]) -> str:
        formatted_demonstrations = self.formatter.format(demonstrations)
        return formatted_demonstrations

    def get_description(self, demonstrations: list[Demonstration]) -> str:
        prompt_extension = self.formatter.get_description(demonstrations)

        prompt_extension += """
In addition to the grids, you will be shown the rotated versions (90 degrees clockwise) of the input and output grids.
"""

        return prompt_extension

    def extra_helper_text(self, demonstrations: list[Demonstration]) -> str:

        helper_text = self.formatter.extra_helper_text(demonstrations)
        helper_text += """
Here are the rotated versions (90 degrees clockwise) of the input and output grids:\n
"""

        for i, demonstration in enumerate(demonstrations):
            input_grid = demonstration.input
            output_grid = demonstration.output

            input_grid_90 = np.rot90(input_grid)
            output_grid_90 = np.rot90(output_grid)

            helper_text += f"Demonstration {i+1}:\n"
            helper_text += f"Input:\n"
            helper_text += self.grid_to_text(input_grid_90) + "\n"
            helper_text += f"\nOutput:\n"
            helper_text += self.grid_to_text(output_grid_90) + "\n"

            helper_text += "\n"

        return helper_text


if __name__ == "__main__":
    emoji_formatter = EmojisDemonstrations()

    # Make random 40x40 grid
    grid = np.random.randint(0, 10, (40, 40))
    print(emoji_formatter.grid_to_text(grid))

    demonstrations = [
        Demonstration(input=grid, output=np.array([[7, 0], [7, 7]])),
        Demonstration(
            input=np.array([[9, 0], [3, 4]]), output=np.array([[2, 5], [3, 1]])
        ),
    ]
    print(emoji_formatter.format(demonstrations))

    ascii_formatter = ASCIIDemonstrations()
    print(ascii_formatter.format(demonstrations))

    # shape_extraction_formatter = ShapeExtractionWrapper(emoji_formatter)
    # print(shape_extraction_formatter.format(demonstrations))

    # difference_formatter = DifferenceWrapper(emoji_formatter)
    # print(difference_formatter.format(demonstrations))

    # rotate_formatter = RotateWrapper(emoji_formatter)
    # print(rotate_formatter.extra_helper_text(demonstrations))
