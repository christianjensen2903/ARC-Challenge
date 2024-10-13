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

    # TODO: Try self correcting. Where it reasons. Then another agent tries to apply the reasoning to the examples.
    # The errors are given to the agent where it then haves to correct the reasoning
    prompt = f"""
Positions are denonated x.y where the x is the column letter and y is the row letter.
A square is denoted by x.y-x.y where the first part is the top left corner and the second part is the bottom right corner.

Your task is to learn a pattern that transforms the input grid to the output grid.
To do this your are shown a series of examples which you can use to learn the pattern.
You are also shown some shapes that can be found in the input and output grids.

First make some observations about the examples and the shapes.
Then reason about the pattern and decide on a property that holds for all the examples.
You should be very precise in your property. Not vague and say it just changes position. But where to or how it changes.
Only state one property. It doesn't need to capture the whole pattern but just a part of it.

You are given these examples to learn the pattern.
{example_prompt}

Where the following shapes are found in the input and output grids:
{observations}

When you are ready to state a property, please write "Property:" followed by the ASCII representation of the output grid.
ONLY write "Property:" once.
Observations:
...

Reasoning:
...

Property:
The shapes of 5s rotate 90 degrees clockwise.
    """

    model = GPT4()
    response = model.generate(prompt)

    property = response.split("Property:")[1].strip()
    print(prompt)
    print(response)
    holds = True
    for i, example in enumerate(examples):
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])
        input_ascii = preprocessing.grid_to_ascii(input_grid)
        output_ascii = preprocessing.grid_to_ascii(output_grid)
        property_prompt = f"""
Based on this property
{property}

Predict the output grid for the following input grid:
{input_ascii} 

Only output the ASCII representation of the grid and nothing else.
Example:
   A B
A | | | 
B | | |
C |1|3|
"""
        response = model.generate(property_prompt).lower()
        print(response)
        print(output_ascii)
        prediction = preprocessing.parse_ascii_grid(response)

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
