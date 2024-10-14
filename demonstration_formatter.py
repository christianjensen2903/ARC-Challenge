from abc import ABC, abstractmethod
import numpy as np
import string
import json
from dataclasses import dataclass
from shape_extractor import ShapeExtractor


@dataclass
class Demonstration:
    input: np.ndarray
    output: np.ndarray


class DemonstrationFormatter(ABC):

    def __init__(self):
        self.column_names = list(string.ascii_uppercase)
        self.row_names = [str(i) for i in range(1, 40)]

    @abstractmethod
    def char_to_text(self, char: int) -> str:
        pass

    @abstractmethod
    def grid_to_text(self, grid: np.ndarray) -> str:
        pass

    @abstractmethod
    def format(self, demonstrations: list[Demonstration]) -> str:
        pass


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

    def grid_to_text(self, grid: np.ndarray) -> str:
        height, width = grid.shape
        grid_str = "    " + "  ".join(self.column_names[:width]) + "\n"

        for i, row in enumerate(grid):
            text_row = self.row_names[i] + ("  " if i <= 10 else " ") + "|"
            for value in row:
                char = self.char_to_text(value)
                text_row += f"{char}|"
            grid_str += text_row + "\n"

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


class ASCIIDemonstrations(DemonstrationFormatter):
    """
    Convert the input and output grids to ASCII format.
    """

    def char_to_text(self, char: int) -> str:
        return str(char) if char > 0 else " "

    def grid_to_text(self, grid: np.ndarray) -> str:
        height, width = grid.shape
        grid_str = "    " + " ".join(self.column_names[:width]) + "\n"

        for i, row in enumerate(grid):
            text_row = self.row_names[i] + ("  " if i <= 10 else " ") + "|"
            for value in row:
                char = self.char_to_text(value)
                text_row += f"{char}|"
            grid_str += text_row + "\n"

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
        formatted_demonstrations = self.formatter.format(demonstrations)

        # Only add shape information if there are few shapes
        for demonstration in demonstrations:
            if not self._few_shapes(demonstration.input, demonstration.output):
                return formatted_demonstrations

        formatted_demonstrations += "\n\n"
        formatted_demonstrations += "Here are the shapes in the input and output:\n"

        for i, demonstration in enumerate(demonstrations):
            input_shapes = self.shape_extractor.find_shapes(demonstration.input)
            output_shapes = self.shape_extractor.find_shapes(demonstration.output)
            formatted_demonstrations += f"Demonstration {i+1}:\n"
            formatted_demonstrations += f"Input shapes:\n"
            if len(input_shapes) == 0:
                formatted_demonstrations += "No shapes found\n"
            else:
                for j, shape in enumerate(input_shapes):
                    formatted_demonstrations += f"Shape {j+1}:\n"
                    formatted_demonstrations += self.grid_to_text(shape.grid)
                    formatted_demonstrations += "\n"

            formatted_demonstrations += "\n"
            formatted_demonstrations += f"Output shapes:\n"
            if len(output_shapes) == 0:
                formatted_demonstrations += "No shapes found\n"
            else:
                for j, shape in enumerate(output_shapes):
                    formatted_demonstrations += f"Shape {j+1}:\n"
                    formatted_demonstrations += self.grid_to_text(shape.grid)
                    formatted_demonstrations += "\n"

            formatted_demonstrations += "\n"

        return formatted_demonstrations


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


if __name__ == "__main__":
    emoji_formatter = EmojisDemonstrations()
    demonstrations = [
        Demonstration(
            input=np.array([[4, 5], [4, 4]]), output=np.array([[7, 0], [7, 7]])
        ),
        Demonstration(
            input=np.array([[9, 0], [3, 4]]), output=np.array([[2, 5], [3, 1]])
        ),
    ]
    print(emoji_formatter.format(demonstrations))

    ascii_formatter = ASCIIDemonstrations()
    print(ascii_formatter.format(demonstrations))

    shape_extraction_formatter = ShapeExtractionWrapper(emoji_formatter)
    print(shape_extraction_formatter.format(demonstrations))

    difference_formatter = DifferenceWrapper(emoji_formatter)
    print(difference_formatter.format(demonstrations))
