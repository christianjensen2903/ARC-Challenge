from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from base_prompt import BasePromptBuilder
from demonstration_formatter import Demonstration, DemonstrationFormatter
from examples import examples
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.language_models.chat_models import ChatGeneration
from run_program import run_program
import numpy as np
from langsmith import traceable
from sklearn.cluster import KMeans  # type: ignore
import random
from llm import LLM
from render import demonstrations_to_oai_content


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
        num_solutions: int = 16,
    ):
        super().__init__(model, formatter, num_examples)
        self.num_solutions = num_solutions
        self.k = 6
        self.accuracy_cutoff_pct = 10

    def _predict(
        self, demonstrations: list[Demonstration], solution: str
    ) -> list[np.ndarray]:
        """
        Predicts the output of each demonstration
        """
        predictions = []
        for demonstration in demonstrations:
            pred, _, _ = run_program(solution, demonstration.input)
            predictions.append(pred)

        return predictions

    def _hamming_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.shape != y.shape:
            return 1.0

        if x.ndim != 2 or y.ndim != 2:
            return 1.0

        return float(np.mean(x != y))

    def _geometric_mean(self, nums: np.ndarray, axis=None) -> float:
        # Check for any non-positive numbers
        assert (nums > 0).all()

        log_nums = np.log(nums)
        return np.exp(log_nums.mean(axis=axis))

    def _get_solutions(self, demonstrations: list[Demonstration]) -> list[str]:

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

I will also provide with an image of the the demonstrations.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    demonstrations_to_oai_content(demonstrations),
                ],
            },
        ]

        raw_solutions = self.model.generate_from_messages(
            messages, n=self.num_solutions
        )

        solutions = []
        for raw_solution in raw_solutions:
            try:
                solutions.append(raw_solution.split("```python")[1].split("```")[0])
            except Exception as e:
                solutions.append(raw_solution)

        return solutions

    @traceable(run_type="retriever", name="get_predictions")
    def _trace_predictions(
        self, demonstrations: list[Demonstration], solutions: list[str]
    ) -> list[dict]:
        """
        Function is to better evaluate the solutions in LangSmith.
        """
        predictions = []
        for solution in solutions:
            preds = self._predict(demonstrations, solution)
            formatted_preds = ""
            for i, demonstration in enumerate(demonstrations):
                pred, stdout, stderr = run_program(solution, demonstration.input)
                formatted_preds += f"""
Demonstration {i+1}:
Input:
{self.formatter.grid_to_text(demonstration.input)}

Predicted Output:
{self.formatter.grid_to_text(preds[i])}

Actual Output:
{self.formatter.grid_to_text(demonstration.output)}

Stdout:
{stdout}

Stderr:
{stderr}
"""
            predictions.append(formatted_preds)

        return [
            {
                "page_content": predictions[i],
                "type": "Document",
                "metadata": {"index": i},
            }
            for i in range(len(predictions))
        ]

    def _get_diversity_matrix(self, predictions: list[list[np.ndarray]]) -> np.ndarray:
        diversity_matrix = []
        for preds in predictions:
            row = []
            for other_preds in predictions:
                inner_row = []
                for i, pred in enumerate(preds):
                    inner_row.append(self._hamming_distance(pred, other_preds[i]))

                row.append(np.mean(inner_row))
            diversity_matrix.append(row)

        return np.array(diversity_matrix)

    def _get_predictions(
        self, demonstrations: list[Demonstration], solutions: list[str]
    ) -> list[list[np.ndarray]]:
        predictions = []
        for solution in solutions:
            preds = self._predict(demonstrations, solution)
            predictions.append(preds)

        return predictions

    def _score_solutions(
        self, demonstrations: list[Demonstration], predictions: list[list[np.ndarray]]
    ) -> list[float]:
        scores: list = []
        for preds in predictions:
            distances: list[float] = []
            for i, demonstration in enumerate(demonstrations):
                pred = preds[i]
                truth = demonstration.output
                distance = self._hamming_distance(truth, pred)
                distances.append(distance)
            score = np.mean(distances)
            scores.append(score)

        return scores

    def _get_top_indices(self, scores: list[float]) -> list[int]:

        cutoff = int(len(scores) * self.accuracy_cutoff_pct / 100)
        top_models_indices = np.argsort(scores)[:cutoff]

        return top_models_indices.tolist()

    @traceable(run_type="retriever", name="rank_solutions")
    def _rank_solutions(
        self, demonstrations: list[Demonstration], solutions: list[str]
    ) -> list[dict]:

        predictions = self._get_predictions(demonstrations, solutions)
        scores = self._score_solutions(demonstrations, predictions)

        diverse_solutions: list[str] = []
        diverse_scores: list[float] = []
        if len(solutions) <= self.k:
            diverse_solutions = solutions
            diverse_scores = scores
        else:
            top_indices = self._get_top_indices(scores)

            top_predictions = [predictions[i] for i in top_indices]
            top_solutions = [solutions[i] for i in top_indices]
            top_scores = [scores[i] for i in top_indices]

            diversity_matrix = self._get_diversity_matrix(top_predictions)
            kmeans = KMeans(n_clusters=self.k, random_state=0).fit(diversity_matrix)

            for i in range(self.k):
                cluster_indices = np.where(kmeans.labels_ == i)[0].astype(int)
                cluster_solutions = [top_solutions[i] for i in cluster_indices]
                cluster_scores = [top_scores[i] for i in cluster_indices]
                if len(cluster_solutions) == 0 or len(cluster_scores) == 0:
                    continue

                diverse_solutions.append(cluster_solutions[np.argmin(cluster_scores)])
                diverse_scores.append(np.min(cluster_scores))

        ranks = np.argsort(diverse_scores)
        return [
            {
                "page_content": diverse_solutions[i],
                "type": "Document",
                "metadata": {
                    "score": diverse_scores[i],
                    "index": i,
                },
            }
            for i in ranks
        ]

    def solve(self, demonstrations: list[Demonstration]) -> str:

        # random.shuffle(demonstrations)

        solutions = self._get_solutions(demonstrations)
        _ = self._trace_predictions(demonstrations, solutions)
        ranked_solutions = self._rank_solutions(demonstrations, solutions)
        return ranked_solutions[0]["page_content"]
