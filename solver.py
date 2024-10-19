from abc import ABC, abstractmethod
from base_prompt import BasePromptBuilder, FixPromptBuilder
from demonstration_formatter import Demonstration, DemonstrationFormatter
from examples import examples
from run_program import run_program
import numpy as np
from langsmith import traceable
from sklearn.cluster import KMeans  # type: ignore
import random
from llm import LLM
from render import demonstrations_to_oai_content
from dataclasses import dataclass
import copy


@dataclass
class Solution:
    hypothesis: str
    code: str
    score: float
    predictions: list[np.ndarray | str]


class Solver(ABC):

    def __init__(
        self,
        model: LLM,
        formatter: DemonstrationFormatter,
        num_examples: int = 2,
    ):
        self.model = model
        self.formatter = formatter
        self.examples = examples[:num_examples]
        self.base_prompt_builder = BasePromptBuilder(formatter, self.examples)
        self.fix_prompt_builder = FixPromptBuilder(formatter)

    @abstractmethod
    def solve(self, demonstrations: list[Demonstration]) -> str:
        """
        Solves the puzzle and returns the solution and the cost of the solution
        """
        pass


class COTSolver(Solver):
    """
    Uses Chain of Thought to generate multiple solutions to the puzzle and returns the ranked solutions.
    """

    def __init__(
        self,
        model: LLM,
        formatter: DemonstrationFormatter,
        num_examples: int = 2,
        k_initial: int = 16,
        k: int = 6,
        max_iterations: int = 2,
        pass_image: bool = False,
    ):
        super().__init__(model, formatter, num_examples)
        self.num_solutions = k_initial
        self.k = k
        self.accuracy_cutoff_pct = 10
        self.max_iterations = max_iterations
        self.pass_image = pass_image

    def _predict(
        self, demonstrations: list[Demonstration], solution: str
    ) -> list[np.ndarray | str]:
        """
        Predicts the output of each demonstration
        """
        predictions: list[np.ndarray | str] = []
        for demonstration in demonstrations:
            pred, _, error = run_program(solution, demonstration.input)
            if error:
                predictions.append(error)
            else:
                predictions.append(pred)

        return predictions

    def _hamming_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.shape != y.shape:
            return 1.0

        if x.ndim != 2 or y.ndim != 2:
            return 1.0

        return float(np.mean(x != y))

    def _get_initial_solutions(
        self, demonstrations: list[Demonstration]
    ) -> list[Solution]:

        formatted_demonstrations = self.formatter.format(demonstrations)
        system_prompt = self.base_prompt_builder.build(demonstrations)

        prompt = f"""
Please solve the following puzzle.
{formatted_demonstrations}

{self.formatter.extra_helper_text(demonstrations)}
"""

        if self.pass_image:
            prompt += f"""
I will also provide with an image of the demonstrations.
"""
        content = [
            {"type": "text", "text": prompt},
        ]
        if self.pass_image:
            content.append(demonstrations_to_oai_content(demonstrations))

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content,
            },
        ]

        responses = self.model.generate_from_messages(messages, n=self.num_solutions)

        solutions = []
        for response in responses:

            # Extract the hypothesis
            try:
                hypothesis = response.split("<hypothesis>")[1].split("</hypothesis>")[0]
            except Exception as e:
                hypothesis = response

            # Extract the solution
            try:
                code = response.split("```python")[1].split("```")[0]
            except Exception as e:
                code = response

            solutions.append(
                Solution(
                    hypothesis=hypothesis,
                    code=code,
                    score=-1,
                    predictions=[],
                )
            )

        return solutions

    def _update_predictions(
        self, demonstrations: list[Demonstration], solutions: list[Solution]
    ) -> None:
        for solution in solutions:
            solution.predictions = self._predict(demonstrations, solution.code)

    def _update_scores(
        self, demonstrations: list[Demonstration], solutions: list[Solution]
    ):
        for solution in solutions:
            distances: list[float] = []
            for i, demonstration in enumerate(demonstrations):
                pred = solution.predictions[i]
                truth = demonstration.output
                distance = (
                    self._hamming_distance(truth, pred)
                    if isinstance(pred, np.ndarray)
                    else 1.0
                )
                distances.append(distance)
            score = np.mean(distances)
            solution.score = float(score)

    def _format_differences(
        self, demonstrations: list[Demonstration], predictions: list[np.ndarray | str]
    ) -> str:
        demonstrations_text = ""
        for j, (demonstration, pred) in enumerate(zip(demonstrations, predictions)):
            demonstrations_text += f"""
Demonstration {j+1}:
Input:
{self.formatter.grid_to_text(demonstration.input)}
"""
            if isinstance(pred, np.ndarray) and np.array_equal(
                demonstration.output, pred
            ):
                demonstrations_text += f"""
Expected and actual output:
{self.formatter.grid_to_text(demonstration.output)}
"""
            else:
                demonstrations_text += f"""
Expected output:
{self.formatter.grid_to_text(demonstration.output)}
Actual output:
{self.formatter.grid_to_text(pred) if isinstance(pred, np.ndarray) else pred}
"""
            demonstrations_text += "\n"

        return demonstrations_text

    def _fix_solution(
        self, demonstrations: list[Demonstration], solution: Solution
    ) -> list[Solution]:

        system_prompt = self.fix_prompt_builder.build(demonstrations)

        system_prompt += f"""
Here are some examples:
"""
        for i, example in enumerate(self.examples):
            # Skip examples where the first hypothesis is correct
            if len(example.steps) < 2:
                continue

            system_prompt += f"""
Example {i+1}:
Current hypothesis:
{example.steps[0].hypothesis}

Current code:
{example.steps[0].code}
"""

            fail_index = example.steps[1].demonstration_index
            current_demonstrations = example.demonstrations[: fail_index + 1]
            solution_attempt = example.steps[0].code
            predictions = self._predict(current_demonstrations, solution_attempt)

            system_prompt += self._format_differences(
                current_demonstrations, predictions
            )

            system_prompt += f"""
<reasoning>
{example.steps[1].reasoning}
</reasoning>
<hypothesis>
{example.steps[1].hypothesis}
</hypothesis>
```python
{example.steps[1].code}
```
"""

        prompt = f"""
Current hypothesis:
{solution.hypothesis}
Current code:
{solution.code}
"""
        prompt += self._format_differences(demonstrations, solution.predictions)

        content = [
            {"type": "text", "text": prompt},
        ]
        if self.pass_image:
            content.append(demonstrations_to_oai_content(demonstrations))

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content,
            },
        ]

        responses = self.model.generate_from_messages(messages, n=self.num_solutions)

        solutions = []
        for response in responses:

            # Extract the hypothesis
            try:
                hypothesis = response.split("<hypothesis>")[1].split("</hypothesis>")[0]
            except Exception as e:
                hypothesis = response

            # Extract the solution
            try:
                code = response.split("```python")[1].split("```")[0]
            except Exception as e:
                code = response

            solutions.append(
                Solution(
                    hypothesis=hypothesis,
                    code=code,
                    score=-1,
                    predictions=[],
                )
            )

        return solutions

    def _fix_solutions(
        self, demonstrations: list[Demonstration], solutions: list[Solution]
    ) -> list[Solution]:
        fixed_solutions = []
        for solution in solutions:
            fixed_solutions.extend(self._fix_solution(demonstrations, solution))
        return fixed_solutions

    def _work_on_demonstration(
        self, demonstrations: list[Demonstration], solutions: list[Solution]
    ) -> list[Solution]:
        self._update_predictions(demonstrations, solutions)
        self._update_scores(demonstrations, solutions)
        solutions.sort(key=lambda x: x.score, reverse=False)
        best_solutions = solutions[: self.k]

        print(f"Initial scores: {[solution.score for solution in best_solutions]}")

        i = 0
        while best_solutions[0].score != 0:
            print(f"Iteration {i}")
            solutions = self._fix_solutions(demonstrations, best_solutions)
            solutions += best_solutions
            self._update_predictions(demonstrations, solutions)
            self._update_scores(demonstrations, solutions)
            solutions.sort(key=lambda x: x.score, reverse=False)
            best_solutions = solutions[: self.k]
            print(f"Scores: {[solution.score for solution in best_solutions]}")
            i += 1
            if i > self.max_iterations:
                print("Max iterations reached")
                break

        return best_solutions

    def solve(self, demonstrations: list[Demonstration]) -> str:

        current_demonstrations = []
        best_solutions: list[Solution] = []

        for i, demonstration in enumerate(demonstrations):
            print(f"Working on demonstration {i+1} of {len(demonstrations)}")
            current_demonstrations.append(demonstration)
            if i == 0:
                best_solutions = self._get_initial_solutions(current_demonstrations)

            best_solutions = self._work_on_demonstration(
                current_demonstrations, best_solutions
            )

            # If solution was obtained only keep solutions that are correct
            if best_solutions[0].score == 0:
                best_solutions = [
                    solution for solution in best_solutions if solution.score == 0
                ]
            else:  # Else move on. More demonstrations probably won't help
                break

        return best_solutions[0].code
