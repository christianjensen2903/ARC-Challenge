import preprocessing
import numpy as np


class ExampleConverter:

    def convert_example_to_ascii(self, example: dict) -> tuple[str, str]:
        """Convert the input and output grids to ASCII format"""
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])

        input_grid = preprocessing.grid_to_ascii(input_grid)
        output_grid = preprocessing.grid_to_ascii(output_grid)

        return input_grid, output_grid

    def extract(self, examples: list[dict[str, list]]) -> str:
        ascii_examples = []
        for example in examples:
            input_grid, output_grid = self.convert_example_to_ascii(example)
            ascii_examples.append((input_grid, output_grid))

        example_prompt = ""
        for i, (input_grid, output_grid) in enumerate(ascii_examples):
            example_prompt += (
                f"Example {i+1}: \nInput: \n{input_grid} \nOutput: \n{output_grid}\n\n"
            )

        return example_prompt
