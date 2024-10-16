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
