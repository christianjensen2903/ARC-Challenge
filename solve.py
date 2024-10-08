from generate_text import GPT4
import preprocessing
import json
import numpy as np
import traceback


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

    input_grid = np.array(solution["input"])
    input_grid = preprocessing.grid_to_ascii(input_grid)

    prompt = f"""
Positions are denonated x.y where the x is the column letter and y is the row letter.
A square is denoted by x.y-x.y where the first part is the top left corner and the second part is the bottom right corner.

Your task is to predict the output grid given the input grid.
To do this your are shown a series of examples which you can use to learn the pattern.

The workflow is as follows:
1. Basic observations
2. More complex observations
3. Reasoning about the pattern in general
4. Reasoning about applying the pattern to the input grid

First extract the basic observations from the examples.
This should be observations like:
    The shapes in the grid
        This could be shapes of the same color, colors in proximity / is connected etc.
    The colors in the grid
    The positions of the shapes

Then extract more complex observations.
This could be observations like:
    If some order of colors are repeating
    If some shapes are repeating
    The position relative to some other shape
    If some shape moves or rotates or changes in some way between the input and output
    Whether a shape is growing or shrinking
    Whether a shape is changing color
    If the grid can be divided into subgrids with some pattern

Remember these are pure observations and not trying to infer the underlying pattern.
Be specific about what you observe. If it is a shape then extract the shape and states is exact position.
If something moves describe the what row and column it moved from and to.
When you make an observation please do it for all the examples.

Then reason about how the input is transformed to the output given the examples and the observations.
You should verify that the reasoning holds for all the examples.

Then apply this reasoning to the input grid.

Given these examples
Example 1: 
Input: 
   A B C D E F G
A |1| | |5| |1| |
B | |1| |5|1|1|1|
C |1| | |5| | | |
 
Output: 
   A B C
A | | | |
B | |2| |
C | | | |


Example 2: 
Input: 
   A B C D E F G
A |1|1| |5| |1| |
B | | |1|5|1|1|1|
C |1|1| |5| |1| |
 
Output: 
   A B C
A | |2| |
B | | |2|
C | |2| |

Basic observations:
Example 1:
Input
    Shapes:
        1:
               A B 
            A |1| |
            B | |1|
            C |1| |

        2:
               D
            A |5|
            B |5|
            C |5|

        3:
               E F G
            A | |1| |
            B |1|1|1|

    Colors: [1, 5]

Output
    Shapes:
        1:
               B
            B |2|

    Colors: [2]

Example 2: 
Input:
    Shapes:
        1:
               A B C 
            A |1|1| |
            B | | |1|
            C |1|1| |

        2:
               D
            A |5|
            B |5|
            C |5|

        3:
               E F G
            A | |1| |
            B |1|1|1|
            C | |1| |

    Colors: [1, 5]

Output:
    Shapes:
        1:
               B C
            A |2| |
            B | |2|
            C |2| |

    Colors: [2]

More complex observations:
Example 1:
    The input grid can be divided into two 3x3 grids by the column of 5's where the subgrids are
       A B C
    A |1| | |
    B | |1| |
    C |1| | |
    and
       A B C
    A | |1| |
    B |1|1|1|
    C | | | |

Example 2:
    The input grid can be divided into two 3x3 grids by the column of 5's where the subgrids are
       A B C
    A |1|1| |
    B | | |1|
    C |1|1| |
    and
       A B C
    A | |1| |
    B |1|1|1|
    C | |1| |

Reasoning about pattern in general:
There doesn't seem to be any overlap between shapes in the input or output for any of the examples.
There doesn't seem to be a overlap between colors in the input or output for any of the examples.
For example 1 it is only 2 at B.B. If we look at the subgrids we can see that there is a 1 at B.B in both subgrids.
If we look at the other cells there isn't any overlap between the subgrids.

If we look at example 2 we can see that there is only a 2 at B.C, C.B and B.C.
If we look at the input there is a 1 at all these positions in the subgrids.
If we look at the other cells there isn't any overlap between the subgrids.

It therefore seems to be the case that the output is 2 if the corresponding cells in the two 3x3 grids divided by 5's are both 1 and 0 otherwise.


An input could then be given as:
   A B C D E F G
A | | |1|5| | | |
B |1|1| |5|1| |1|
C | |1|1|5|1| |1|

Where the reasoning about applying the pattern to the input grid then would be:
Divide the grid into two 3x3 grids by the column of 5's which gives:
   A B C
A | | |1|
B |1|1| |
C | |1|1|
and
   A B C
A | | | |
B |1| |1|
C |1| |1|

We then see that A.B and C.B are both 1 where the output then must be
   A B C
A | | | |
B |1| | |
C | | |1|

You are now given some examples to learn the pattern.
{example_prompt}

Make your observations and reason about the pattern in general and apply the pattern to following input grid:
{input_grid}

When you are ready to make a prediction, please write "Prediction:" followed by the ASCII representation of the output grid.
ONLY write "Prediction:" once.
Example of output:
Basic observations:
...

More complex observations:
...

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
