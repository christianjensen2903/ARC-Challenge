from demonstration_formatter import DemonstrationFormatter, Demonstration


class BasePromptBuilder:

    def __init__(self, formatter: DemonstrationFormatter):
        self.formatter = formatter

    def build(self, demonstrations: list[Demonstration]) -> str:
        return f"""
You are creative and accomplished at solving puzzles

You will given some number of paired example inputs and outputs.
The outputs were produced by applying a transformation rule to the inputs.
Your task is to determine the transformation rule and implement it in code.

{self.formatter.get_description(demonstrations)}

The transformation only needs to be unambiguous and applicable to the example inputs and the additional input.
It doesn't need to work for all possible inputs.

You'll need to carefully reason in order to determine the transformation rule.
Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

After your reasoning write code in triple backticks (```python and then ```).
You should write a function called `transform` which takes a single argument, the input grid as `np.ndarray`, and returns the transformed grid (also as `np.ndarray`).
The grid will be 2D and contain integers i.e. [[1, 0], [1, 4]]
You should make sure that you implement a version of the transformation which works in general (it shouldn't just work for the additional input).

Don't write tests in your python code, just output the `transform` function. (It will be tested later.)

You follow a particular reasoning style.
You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion.
This reduces the extent to which you need to do large leaps of reasoning.
You reason in substantial detail for as is necessary to determine the transformation rule.

Your reasoning **can be as long as necessary**!
The goal of the reasoning is just to make sure you end up with a correct implementation of the transformation rule
So **there isn't any need for your reasoning to be concise**.
You should do any and all reasoning that would be useful.
"""
