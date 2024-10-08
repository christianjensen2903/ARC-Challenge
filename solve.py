from generate_text import GPT4
import preprocessing
import json
import numpy as np


def load_data() -> tuple[dict, dict]:
    with open("data/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)

    with open("data/arc-agi_training_solutions.json") as f:
        solutions = json.load(f)

    return challenges, solutions


challenges, solutions = load_data()


def test(example_id: str):

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

    input_grid = np.array(solution["input"])
    input_grid = preprocessing.grid_to_ascii(input_grid)

    prompt = f"""
Your task is to predict the output grid given the input grid.
To do this your are shown a series of examples which you can use to learn the pattern.

{example_prompt}

Please first reason about what the pattern is and then predict the output grid for the following input grid:
{input_grid}

When you are ready to make a prediction, please write "Prediction:" followed by the ASCII representation of the output grid wrapped in triple quotes (```).
ONLY write "Prediction:" once.
Example of output:
Prediction:
```
   A B
A | | | 
B | | |
C |1|3|
```
    """

    model = GPT4()
    response = model.generate(prompt)
    print(prompt)
    print(response)
    prediction = response.split("Prediction:")[1].strip().strip("```")
    prediction_grid = preprocessing.parse_ascii_grid(prediction)
    solution_grid = np.array(solution["output"])
    print(preprocessing.grid_to_ascii(solution_grid))
    if prediction_grid.shape != solution_grid.shape:
        print("The shape of the prediction grid is incorrect")
    else:
        print((prediction_grid == solution_grid).all())


test(example_id="0520fde7")
