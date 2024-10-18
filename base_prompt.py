from demonstration_formatter import DemonstrationFormatter, Demonstration
import numpy as np


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

Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

The reasoning should follow a specific pattern:
You start by looking at the first demonstration where you:
1. Identify commonalities between the input and output
2. Try to infer what the transformation rule is based on the commonalities

You then look at the next demonstration to verify your theories or refine them.
If something doesn't match up you will need to scrap your theories and try something else or try to figure out how to alter you theory to fit the new demonstration.

You repeat this process until you have a transformation rule that works for all of the demonstrations.

It is important that the hypothesis is very specific. Not just a shape is moved, but how exactly it is moved.

After you have formulated a theory you should reason what the code should look like at a high level.

After your reasoning write code in triple backticks (```python and then ```).
You should write a function called `transform` which takes a single argument, the input grid as `np.ndarray`, and returns the transformed grid (also as `np.ndarray`).
The grid will be 2D and contain integers i.e. [[1, 0], [1, 4]]
You should make sure that you implement a version of the transformation which works in general (it shouldn't just work for the additional input).

Don't write tests in your python code, just output the `transform` function. (It will be tested later.)

The format of your response should be as follows:
<reasoning>
**Demonstration 1**
Reasoning:
...

Hypothesis:
...

**Demonstration 2**
Reasoning:
...

Hypothesis:
...

...

**Demonstration n**
Reasoning:
...

Hypothesis:
...

Final theory:
...

The code should:
...

</reasoning>
```python
...
```

It is VERY IMPORTANT that you follow this pattern.
The part "the could should be" is a bit more flexible.
Here you should just use the reasoning that would make it easier to implement the transformation.
"""


class FixPromptBuilder:
    def __init__(self, formatter: DemonstrationFormatter):
        self.formatter = formatter

    def build(
        self, demonstrations: list[Demonstration], outputs: list[np.ndarray]
    ) -> str:

        # Calculate how many examples were wrong
        num_wrong = 0
        for demonstration, output in zip(demonstrations, outputs):
            if not np.array_equal(output, demonstration.output):
                num_wrong += 1

        prompt = f"""
The `transform` function you implemented failed {num_wrong} out of {len(demonstrations)} demonstrations.

Your task is to determine what the issue is and then fix the code.

The issue could be a bug in the code and/or an issue with your previous understanding of the transformation rule.

You'll need to carefully reason to determine the issue and to determine how to fix the code. Start your response by doing this reasoning in <reasoning></reasoning> tags.
Then, implement the fixed transformation in code.

Below, we show what the incorrect `transform` function outputs for each failed demonstration.
"""

        for i, (demonstration, output) in enumerate(zip(demonstrations, outputs)):
            if not np.array_equal(output, demonstration.output):
                prompt += f"""

Demonstration {i+1}:

Output:
{self.formatter.grid_to_text(output)}

Expected output:
{self.formatter.grid_to_text(demonstration.output)}
"""

        prompt += """
Your response should follow this format:
<reasoning>
Reasoning:
...
</reasoning>
```python
...
```
"""

        return prompt
