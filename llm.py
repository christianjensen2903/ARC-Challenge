from openai import OpenAI
from abc import ABC, abstractmethod
import dotenv
from langsmith.wrappers import wrap_openai


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        pass

    @abstractmethod
    def generate_from_messages(
        self,
        messages: list[dict[str, str]],
        n: int = 1,
    ) -> list[str]:
        pass


class GPT4(LLM):
    def __init__(self):
        dotenv.load_dotenv()
        self.client = wrap_openai(OpenAI())

    def generate(self, prompt: str, n: int = 1) -> list[str]:
        return self.generate_from_messages([{"role": "user", "content": prompt}], n)

    def generate_from_messages(
        self, messages: list[dict[str, str]], n: int = 1
    ) -> list[str]:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, n=n
        )
        return [choice.message.content for choice in response.choices]


if __name__ == "__main__":
    generator = GPT4()
    prompt = "What color is the top-left cell?"
    response = generator.generate(prompt)
    print(response)
