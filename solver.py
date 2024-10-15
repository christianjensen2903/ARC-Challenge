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


class Solver(ABC):

    def __init__(
        self,
        model: BaseChatModel,
        formatter: DemonstrationFormatter,
        num_examples: int = 2,
    ):
        self.model = model
        self.formatter = formatter
        self.base_prompt_builder = BasePromptBuilder(formatter)
        self.num_examples = num_examples
        self.examples = examples[: self.num_examples]

    @abstractmethod
    def solve(self, demonstrations: list[Demonstration]) -> tuple[str, float]:
        """
        Solves the puzzle and returns the solution and the cost of the solution
        """
        pass

    def generate(
        self, prompt: str, system_prompt: str | None = None, n: int = 1
    ) -> tuple[list[str], float]:
        messages: list[BaseMessage] = []
        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        responses: list[str] = []
        with get_openai_callback() as cb:
            response = self.model.generate([messages], n=n)
            if isinstance(response.generations, list) and len(response.generations) > 0:
                for r in response.generations[0]:
                    if isinstance(r, ChatGeneration):
                        responses.append(r.text)
            cost = cb.total_cost

        return responses, cost


class COTSolver(Solver):
    """
    Uses Chain of Thought to generate multiple solutions to the puzzle and returns the ranked solutions.
    """

    def __init__(
        self,
        model: BaseChatModel,
        formatter: DemonstrationFormatter,
        num_examples: int = 2,
        num_solutions: int = 2,
    ):
        super().__init__(model, formatter, num_examples)
        self.num_solutions = num_solutions
        self.cost = 0.0

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
"""

        raw_solutions, cost = self.generate(
            prompt, system_prompt=system_prompt, n=self.num_solutions
        )

        solutions = []
        for raw_solution in raw_solutions:
            try:
                solutions.append(raw_solution.split("```python")[1].split("```")[0])
            except Exception as e:
                solutions.append(raw_solution)

        self.cost += cost

        return solutions

    @traceable(run_type="retriever", name="get_predictions")
    def _get_predictions(
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
                formatted_preds += f"""
Demonstration {i+1}:
Input:
{self.formatter.grid_to_text(demonstration.input)}

Predicted Output:
{self.formatter.grid_to_text(preds[i])}

Actual Output:
{self.formatter.grid_to_text(demonstration.output)}
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

    @traceable(run_type="retriever", name="rank_solutions")
    def _rank_solutions(
        self, demonstrations: list[Demonstration], solutions: list[str]
    ) -> list[dict]:

        scores = []
        predictions = []
        for solution in solutions:
            preds = self._predict(demonstrations, solution)
            predictions.append(preds)
            distances = []
            for i, demonstration in enumerate(demonstrations):
                pred = preds[i]
                truth = demonstration.output
                distance = self._hamming_distance(truth, pred)
                distances.append(distance)

            score = self._geometric_mean(np.array(distances))
            scores.append(score)

        ranks = np.argsort(scores)
        return [
            {
                "page_content": solutions[i],
                "type": "Document",
                "metadata": {
                    "score": scores[i],
                    "index": i,
                },
            }
            for i in ranks
        ]

    def solve(self, demonstrations: list[Demonstration]) -> tuple[str, float]:
        self.cost = 0
        solutions = self._get_solutions(demonstrations)
        _ = self._get_predictions(demonstrations, solutions)
        ranked_solutions = self._rank_solutions(demonstrations, solutions)
        return ranked_solutions[0]["page_content"], self.cost
