from generate_text import GPT4
import preprocessing
import json
import numpy as np
import traceback
from observation_extractor import ObservationExtractor


def load_data() -> tuple[dict, dict]:
    with open("data/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)

    with open("data/arc-agi_training_solutions.json") as f:
        solutions = json.load(f)

    return challenges, solutions


challenges, solutions = load_data()


def test(example_id: str, verbose: bool = True) -> bool:

    examples = challenges[example_id]["train"]
    solution = {
        "input": challenges[example_id]["test"][0]["input"],
        "output": solutions[example_id][0],
    }

    def convert_example_to_ascii(example: dict) -> tuple[str, str]:
        """Convert the input and output grids to ASCII format"""
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])

        input_grid = preprocessing.grid_to_ascii(input_grid)
        output_grid = preprocessing.grid_to_ascii(output_grid)

        return input_grid, output_grid

    ascii_examples = []
    for example in examples:
        input_grid, output_grid = convert_example_to_ascii(example)
        ascii_examples.append((input_grid, output_grid))

    example_prompt = ""
    for i, (input_grid, output_grid) in enumerate(ascii_examples):
        example_prompt += (
            f"Example {i+1}: \nInput: \n{input_grid} \nOutput: \n{output_grid}\n\n"
        )

    extractor = ObservationExtractor()
    observations = extractor.extract(examples)

    input_grid = np.array(solution["input"])
    input_grid = preprocessing.grid_to_ascii(input_grid)

    prompt = f"""
Positions are denonated x.y where the x is the column letter and y is the row letter.
A square is denoted by x.y-x.y where the first part is the top left corner and the second part is the bottom right corner.

Your task is to predict the output grid given the input grid.
To do this your are shown a series of examples which you can use to learn the pattern.
You are also shown some shapes that can be found in the input and output grids.

The workflow is as follows:
1. Reasoning about the pattern in general
2. Reasoning about applying the pattern to the input grid


First reason about how the input is transformed to the output given the examples and the observations.
You should verify that the reasoning holds for all the examples.

Then apply this reasoning to the input grid.

You are given these examples to learn the pattern.
{example_prompt}

Where the following shapes are found in the input and output grids:
{observations}

Make your observations and reason about the pattern in general and apply the pattern to following input grid:
{input_grid}

When you are ready to make a prediction, please write "Prediction:" followed by the ASCII representation of the output grid.
ONLY write "Prediction:" once.
Example of output:
Reasoning about pattern in general:
...

Reasoning about applying pattern to input:
...

Prediction:
   A B
A | | | 
B | | |
C |1|3|
    """

    model = GPT4()
    response = model.generate(prompt)

    prediction = response.split("Prediction:")[1].strip()
    prediction_grid = preprocessing.parse_ascii_grid(prediction)
    solution_grid = np.array(solution["output"])
    if verbose:
        print(prompt)
        print(response)
        print(preprocessing.grid_to_ascii(solution_grid))

    if prediction_grid.shape != solution_grid.shape:
        return False
    return (prediction_grid == solution_grid).all()


test(example_id="05f2a901")

# n = 20
# correct = 0
# for example_id in list(challenges.keys())[:n]:
#     try:
#         if test(example_id, verbose=False):
#             correct += 1
#     except Exception as e:
#         print(f"Error in example {example_id}")
#         traceback.print_exc()

# print(f"Accuracy: {correct}/{n}")
