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
    solution: str
    score: float
    conversation: list
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
        self.base_prompt_builder = BasePromptBuilder(formatter)
        self.num_examples = num_examples
        self.examples = examples[: self.num_examples]

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
        num_iterations: int = 2,
        pass_image: bool = False,
    ):
        super().__init__(model, formatter, num_examples)
        self.num_solutions = k_initial
        self.k = k
        self.accuracy_cutoff_pct = 10
        self.num_iterations = num_iterations
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

        system_prompt += f"""
Here are some examples:
"""
        for i, example in enumerate(self.examples):
            demonstrations_str = self.formatter.format(example.demonstrations)
            system_prompt += f"""
Example {i+1}:
{demonstrations_str}

<reasoning>
{example.reasoning}

Let's implement it in code.
</reasoning>
```python
{example.code}
```
"""

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
            try:
                solution = response.split("```python")[1].split("```")[0]
            except Exception as e:
                solution = response

            solutions.append(
                Solution(
                    solution,
                    -1,
                    messages
                    + [
                        {
                            "role": "assistant",
                            "content": response,
                        }
                    ],
                    [],
                )
            )

        return solutions

    def _get_diversity_matrix(self, solutions: list[Solution]) -> np.ndarray:
        diversity_matrix = []
        for solution in solutions:
            row = []
            for other_solution in solutions:
                inner_row = []
                for i, pred in enumerate(solution.predictions):

                    if isinstance(pred, str) and isinstance(
                        other_solution.predictions[i], np.ndarray
                    ):
                        distance = 1.0
                    elif isinstance(pred, np.ndarray) and isinstance(
                        other_solution.predictions[i], str
                    ):
                        distance = 1.0
                    elif isinstance(pred, str) and isinstance(
                        other_solution.predictions[i], str
                    ):
                        distance = 0.0
                    else:
                        distance = self._hamming_distance(
                            pred, other_solution.predictions[i]  # type: ignore
                        )

                    inner_row.append(distance)

                row.append(np.mean(inner_row))
            diversity_matrix.append(row)

        return np.array(diversity_matrix)

    def _update_predictions(
        self, demonstrations: list[Demonstration], solutions: list[Solution]
    ) -> None:
        for solution in solutions:
            solution.predictions = self._predict(demonstrations, solution.solution)

    def _update_scores(
        self, demonstrations: list[Demonstration], solutions: list[Solution]
    ):
        for solution in solutions:
            distances: list[float] = []
            for i, demonstration in enumerate(demonstrations):
                pred = solution.predictions[i]
                truth = demonstration.output
                if isinstance(pred, str):
                    distance = 1.0
                else:
                    distance = self._hamming_distance(truth, pred)
                distances.append(distance)
            score = np.mean(distances)
            solution.score = float(score)

    @traceable(run_type="retriever", name="rank_solutions")
    def _rank_solutions(
        self, demonstrations: list[Demonstration], solutions: list[Solution]
    ) -> list[Solution]:

        self._update_predictions(demonstrations, solutions)
        self._update_scores(demonstrations, solutions)

        # Sort solutions by score
        solutions.sort(key=lambda x: x.score, reverse=False)

        if len(solutions) <= self.k:
            return solutions

        cutoff = max(self.k, int(len(solutions) * self.accuracy_cutoff_pct / 100))
        top_solutions = solutions[:cutoff]

        diversity_matrix = self._get_diversity_matrix(top_solutions)
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(diversity_matrix)

        diverse_solutions: list[Solution] = []
        for i in range(self.k):
            cluster_indices = np.where(kmeans.labels_ == i)[0].astype(int)
            if len(cluster_indices) == 0:
                continue

            cluster_solutions: list[Solution] = [
                top_solutions[i] for i in cluster_indices
            ]
            best_solution = min(cluster_solutions, key=lambda x: x.score)
            diverse_solutions.append(best_solution)

        return diverse_solutions

    def _fix_solutions(
        self, demonstrations: list[Demonstration], solutions: list[Solution]
    ) -> list[Solution]:

        fixed_solutions = []

        for solution in solutions:
            prompt = FixPromptBuilder(self.formatter).build(
                demonstrations, solution.predictions
            )

            new_conv = solution.conversation + [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            responses = self.model.generate_from_messages(new_conv, n=self.k)
            for response in responses:
                try:
                    new_solution = response.split("```python")[1].split("```")[0]
                except Exception as e:
                    new_solution = response

                fixed_solutions.append(
                    Solution(
                        new_solution,
                        -1,
                        new_conv + [{"role": "assistant", "content": response}],
                        [],
                    )
                )
        return fixed_solutions

    def solve(self, demonstrations: list[Demonstration]) -> str:

        # random.shuffle(demonstrations)

        solutions = self._get_initial_solutions(demonstrations)

        best_solutions: list[Solution] = []
        for i in range(self.num_iterations):
            solutions += best_solutions
            best_solutions = self._rank_solutions(demonstrations, solutions)

            print(
                f"Scores after iteration {i}: {[sol.score for sol in best_solutions]}"
            )

            if i < self.num_iterations - 1:
                solutions = self._fix_solutions(demonstrations, best_solutions)

        # Sort solutions by score
        best_solutions.sort(key=lambda x: x.score, reverse=False)

        return best_solutions[0].solution
