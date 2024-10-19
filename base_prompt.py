from demonstration_formatter import DemonstrationFormatter, Demonstration
from examples import Example
import numpy as np


class BasePromptBuilder:

    def __init__(self, formatter: DemonstrationFormatter, examples: list[Example]):
        self.formatter = formatter
        self.examples = examples

    def build(self, demonstrations: list[Demonstration]) -> str:
        prompt = f"""
You are creative and accomplished at solving puzzles

You will be given an input and an output.
The output was produced by applying a transformation rule to the input.
Your task is to determine the transformation rule and implement it in code.

{self.formatter.get_description(demonstrations)}

Start your response by carefully reasoning in <reasoning></reasoning> tags.
Then propose what the transformation rule is in <hypothesis></hypothesis> tags.
Then, implement the transformation in code.

It is important that the hypothesis is very specific. Not just a shape is moved, but how exactly it is moved.

After your hypothesis write code in triple backticks (```python and then ```).
You should write a function called `transform` which takes a single argument, the input grid as `np.ndarray`, and returns the transformed grid (also as `np.ndarray`).
The grid will be 2D and contain integers i.e. [[1, 0], [1, 4]]
You should make sure that you implement a version of the transformation which works in general (it shouldn't just work for the additional input).

Don't write tests in your python code, just output the `transform` function. (It will be tested later.)

The format of your response should be as follows:
<reasoning>
...
</reasoning>
<hypothesis>
...
</hypothesis>
```python
...
```

Here are some examples:
"""
        for i, example in enumerate(self.examples):

            formatted_example = self.formatter.format([example.demonstrations[0]])

            prompt += f"""
Example {i+1}:

{formatted_example}

<reasoning>
{example.steps[0].reasoning}
</reasoning>
<hypothesis>
{example.steps[0].hypothesis}
</hypothesis>
```python
{example.steps[0].code}
```
"""

        return prompt


class FixPromptBuilder:
    def __init__(self, formatter: DemonstrationFormatter):
        self.formatter = formatter

    def build(self, demonstrations: list[Demonstration]) -> str:

        prompt = f"""
You are creative and accomplished at solving puzzles

You will given some number of paired inputs and outputs.
The outputs were produced by applying a transformation rule to the inputs.

You will also be given some hypothesis of what the transformation rule is.
In addition, you will be given the code that implements the transformation rule.

For each demonstration you will also be provided the output that the code produces versus the expected output.

Your task is to determine what the issue is and then fix the code.

The issue could be a bug in the code and/or an issue with your previous understanding of the transformation rule.

{self.formatter.get_description(demonstrations)}

Start your response by carefully reasoning in <reasoning></reasoning> tags.
Then propose what the new transformation rule is in <hypothesis></hypothesis> tags.
Then, implement the new transformation in code.

It is important that the hypothesis is very specific. Not just a shape is moved, but how exactly it is moved.

After your hypothesis write code in triple backticks (```python and then ```).
You should write a function called `transform` which takes a single argument, the input grid as `np.ndarray`, and returns the transformed grid (also as `np.ndarray`).
The grid will be 2D and contain integers i.e. [[1, 0], [1, 4]]
You should make sure that you implement a version of the transformation which works in general (it shouldn't just work for the additional input).

Don't write tests in your python code, just output the `transform` function. (It will be tested later.)

The format of your response should be as follows:
<reasoning>
...
</reasoning>
<hypothesis>
...
</hypothesis>
```python
...
```
"""

        return prompt
