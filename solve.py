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

        input_ascii = preprocessing.grid_to_ascii(input_grid)
        output_ascii = preprocessing.grid_to_ascii(output_grid)

        return input_ascii, output_ascii

    ascii_examples = []
    for example in examples:
        input_ascii, output_ascii = convert_example_to_ascii(example)
        ascii_examples.append((input_ascii, output_ascii))

    example_prompt = ""
    for i, (input_ascii, output_ascii) in enumerate(ascii_examples):
        example_prompt += (
            f"Example {i+1}: \nInput: \n{input_ascii} \nOutput: \n{output_ascii}\n\n"
        )

    extractor = ObservationExtractor()
    observations = extractor.extract(examples)

    input_grid = np.array(solution["input"])
    input_ascii = preprocessing.grid_to_ascii(input_grid)

    prompt = f"""
Positions are denonated XY where X is the column letter and Y is the row index.
A square is denoted by XY1-XY2 where XY1 is the top left corner and XY2 is the bottom right corner.

Your task is to learn a pattern that transforms the input grid to the output grid.
To do this your are shown a series of examples which you can use to learn the pattern.
You are also shown some shapes that can be found in the input and output grids.

First make some observations about the examples and the shapes.
Then reason about the pattern and decide on what the pattern is
You should be very precise in your description of the pattern. It should be able to be understood in isolation.
Not vague and say it just changes position. But where to or how it changes.

You are given these examples to learn the pattern.
{example_prompt}

Where the following shapes are found in the input and output grids:
{observations}

When you are ready to state the pattern, please write "Pattern:" followed by your description of the pattern.
ONLY write "Pattern:" once.
Observations:
...

Reasoning:
...

Pattern:
The brown shapes (🟤) rotate 90 degrees clockwise.
    """

    model = GPT4()
    response = model.generate(prompt)

    pattern = response.split("Pattern:")[1].strip()
    print(prompt)
    print(response)
    holds = True
    for i, example in enumerate(examples):
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])
        input_ascii = preprocessing.grid_to_ascii(input_grid)
        output_ascii = preprocessing.grid_to_ascii(output_grid)

        extractor = ObservationExtractor()
        observations = extractor.extract([example])
        observation_prompt = ""
        for observation in observations:
            observation_prompt += f"{observation}\n"

        property_prompt = f"""
Your task is to apply the pattern to the input grid to predict the output grid.
To do this please break down the pattern into steps and apply them to the input grid.

Please apply this pattern:
{pattern}

To the following input grid:
{input_ascii}

Where the following shapes are found in the input grid:
{observation_prompt}

First reason about the steps needed to apply the pattern.
Then apply the pattern to the input grid.
When you are ready to state the output grid, please write "Output:" followed by the output grid and nothing else.
ONLY write "Output:" once.
Example:

Reasoning:
...

Output:
    A  B
1  |  |  | 
2  |  |  |
3  |🔴|⚪️|
"""
        response = model.generate(property_prompt)
        print(response)
        print(output_ascii)
        prediction = preprocessing.parse_ascii_grid(
            response.split("Output:")[1].strip()
        )

        if prediction.shape != output_grid.shape:
            holds = False
            break
        if not (prediction == output_grid).all():
            holds = False
            break

    return False

    # prediction_grid = preprocessing.parse_ascii_grid(prediction)
    # solution_grid = np.array(solution["output"])
    # if verbose:
    #     print(prompt)
    #     print(response)
    #     print(preprocessing.grid_to_ascii(solution_grid))

    # if prediction_grid.shape != solution_grid.shape:
    #     return False
    # return (prediction_grid == solution_grid).all()


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
