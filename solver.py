from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from base_prompt import BasePromptBuilder
from demonstration_formatter import Demonstration, DemonstrationFormatter


class Solver(ABC):

    def __init__(self, model: BaseChatModel, formatter: DemonstrationFormatter):
        self.model = model
        self.formatter = formatter
        self.base_prompt_builder = BasePromptBuilder(formatter)

    @abstractmethod
    def solve(self, demonstrations: list[Demonstration]) -> str:
        pass

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages: list[BaseMessage] = []
        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        response = self.model.invoke(messages).content
        assert isinstance(response, str)
        return response


class COTSolver(Solver):
    def solve(self, demonstrations: list[Demonstration]) -> str:
        formatted_demonstrations = self.formatter.format(demonstrations)
        system_prompt = self.base_prompt_builder.build(demonstrations)

        prompt = f"""
Please solve the following puzzle.
{formatted_demonstrations}
"""

        response = self.generate(prompt, system_prompt=system_prompt)
        prediction = response.split("```python")[1].split("```")[0]
        return prediction
