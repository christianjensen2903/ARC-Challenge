from langgraph.graph import END, START, Graph
import dotenv
from langchain_openai import ChatOpenAI
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import logging

from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel


class Solver(ABC):

    def __init__(self, model: BaseChatModel):
        self.model = model

    @abstractmethod
    def solve(self, formatted_demonstrations: str) -> str:
        pass

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages: list[BaseMessage] = []
        if system_prompt is not None:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        response = self.model.invoke(messages).content
        assert isinstance(response, str)
        return response


class IOSolver(Solver):
    def solve(self, formatted_demonstrations: str) -> str:
        prompt = f"""
You are a helpful assistant that solves the demonstrations.

{formatted_demonstrations}
"""
        return self.generate(formatted_demonstrations, system_prompt=prompt)
