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
        num_solutions: int = 5,
    ):
        super().__init__(model, formatter, num_examples)
        self.num_solutions = num_solutions

    def predict(
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

    def hamming_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.shape != y.shape:
            return 1.0

        if x.ndim != 2 or y.ndim != 2:
            return 1.0

        return float(np.mean(x != y))

    def geometric_mean(self, nums: np.ndarray, axis=None) -> float:
        # Check for any non-positive numbers
        assert (nums > 0).all()

        log_nums = np.log(nums)
        return np.exp(log_nums.mean(axis=axis))

    def solve(self, demonstrations: list[Demonstration]) -> tuple[str, float]:
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

        possible_solutions, cost = self.generate(
            prompt, system_prompt=system_prompt, n=self.num_solutions
        )

        predictions: list[list[np.ndarray]] = []
        for solution in possible_solutions:
            solution = solution.split("```python")[1].split("```")[0]
            predictions.append(self.predict(demonstrations, solution))

        scores = []
        for preds in predictions:
            distances = []
            for i, demonstration in enumerate(demonstrations):
                pred = preds[i]
                truth = demonstration.output
                distance = self.hamming_distance(truth, pred)
                distances.append(distance)

            score = self.geometric_mean(np.array(distances))
            scores.append(score)

        ranks = np.argsort(scores)
        ranks = ranks[::-1]
        ranked_solutions = [possible_solutions[i] for i in ranks]

        best_solution = ranked_solutions[0]
        return best_solution, cost
