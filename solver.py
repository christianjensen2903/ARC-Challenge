from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from base_prompt import BasePromptBuilder
from demonstration_formatter import Demonstration, DemonstrationFormatter
from examples import examples
from langchain_community.callbacks.manager import get_openai_callback


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
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, float]:
        messages: list[BaseMessage] = []
        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        with get_openai_callback() as cb:
            response = self.model.invoke(messages)
            content = response.content
            cost = cb.total_cost
        assert isinstance(content, str)
        return content, cost


class COTSolver(Solver):

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

        content, cost = self.generate(prompt, system_prompt=system_prompt)
        prediction = content.split("```python")[1].split("```")[0]
        return prediction, cost
