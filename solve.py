from generate_text import GPT4
import preprocessing
import json
import numpy as np
import traceback
from observation_extractor import ShapeExtractor
from demonstrations import demonstrations


def load_data() -> tuple[dict, dict]:
    with open("data/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)

    with open("data/arc-agi_training_solutions.json") as f:
        solutions = json.load(f)

    return challenges, solutions


challenges, solutions = load_data()


def convert_example_to_ascii(example: dict) -> tuple[str, str]:
    """Convert the input and output grids to ASCII format"""
    input_grid = np.array(example["input"])
    output_grid = np.array(example["output"])

    input_ascii = preprocessing.grid_to_ascii(input_grid)
    output_ascii = preprocessing.grid_to_ascii(output_grid)

    return input_ascii, output_ascii


def format_examples(examples: list[dict]) -> str:

    ascii_examples = []
    for example in examples:
        input_ascii, output_ascii = convert_example_to_ascii(example)
        ascii_examples.append((input_ascii, output_ascii))

    example_prompt = ""
    for i, (input_ascii, output_ascii) in enumerate(ascii_examples):
        example_prompt += (
            f"Example {i+1}: \nInput: \n{input_ascii} \nOutput: \n{output_ascii}\n\n"
        )
    return example_prompt


def test(example_id: str, k: int = 2, n: int = 3, verbose: bool = True) -> bool:

    extractor = ShapeExtractor()
    model = GPT4()

    examples = challenges[example_id]["train"]

    solution = {
        "input": challenges[example_id]["test"][0]["input"],
        "output": solutions[example_id][0],
    }

    demonstration_prompt = ""
    for i, (id, demonstration) in enumerate(demonstrations.items()):
        demonstration_examples = challenges[id]["train"]
        observations = extractor.extract(demonstration_examples)
        example_prompt = format_examples(demonstration_examples)
        demonstration_prompt += f"""
Demonstration {i+1}:
{example_prompt}

Where the following shapes are found in the input and output grids:
{observations}

{demonstration}
"""

    observations = extractor.extract(examples)
    example_prompt = format_examples(examples)

    input_grid = np.array(solution["input"])
    input_ascii = preprocessing.grid_to_ascii(input_grid)

    prompt = f"""
First make some observations about the examples and the shapes.
Then reason about the pattern and decide on what the pattern is
You should be very precise in your description of the pattern. It should be able to be understood in isolation.
Not vague and say it just changes position. But where to or how it changes.

Here are some demonstrations:
{demonstration_prompt}

You are given these examples to learn the pattern.
{example_prompt}

Where the following shapes are found in the input and output grids:
{observations}
    """

    messages = [
        {
            "role": "system",
            "content": """
Positions are denonated XY where X is the column letter and Y is the row index.
A square is denoted by XY1-XY2 where XY1 is the top left corner and XY2 is the bottom right corner.

Your task is to learn a pattern that transforms the input grid to the output grid.
To do this your are shown a series of examples which you can use to learn the pattern.
You are also shown some shapes that can be found in the input and output grids.

You then predict the output grid for the given input grid.
If the pattern is incorrect, you will be told how the output grid would look if the pattern was applied correctly and the correct output grid.

You should then reason about what went wrong and how to fix it.
At the end of each message you should state the pattern.
When you are ready to state the pattern, please write "Pattern:" followed by your description of the pattern.
Observations:
...

Reasoning:
...

Pattern:
The brown shapes (🟤) rotate 90 degrees clockwise.
""",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Maybe rewrite to LangGraph
    # TODO: Continue working on iteration. Maybe add more examples

    for i in range(k):

        response = model.generate_from_messages(messages)

        print(response)

        pattern = response.split("Pattern:")[1].strip()

        pattern_correct = True
        correction_prompt = ""
        for i, example in enumerate(examples):
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])
            input_ascii = preprocessing.grid_to_ascii(input_grid)
            output_ascii = preprocessing.grid_to_ascii(output_grid)

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
            prediction_ascii = response.split("Output:")[1].strip().strip("`").strip()
            prediction = preprocessing.parse_ascii_grid(prediction_ascii)

            if (
                prediction.shape != output_grid.shape
                or not (prediction == output_grid).all()
            ):
                pattern_correct = False
                correction_prompt += f"""
    The pattern for example {i+1} is incorrect.
    The application of the pattern to the input grid is:
    {prediction_ascii}

    The correct output grid is:
    {output_ascii}
    """
            else:
                correction_prompt += f"""
    The pattern for example {i+1} is correct.
    """

        messages.append({"role": "user", "content": correction_prompt})

        if pattern_correct:
            break

    return pattern_correct

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
