from openai import OpenAI
from abc import ABC, abstractmethod
import dotenv


class TextGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_from_messages(self, messages: list[dict[str, str]]) -> str:
        pass


class GPT4(TextGenerator):
    def __init__(self):
        dotenv.load_dotenv()
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        return self.generate_from_messages([{"role": "user", "content": prompt}])

    def generate_from_messages(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    generator = GPT4()
    prompt = "What color is the top-left cell?"
    response = generator.generate(prompt)
    print(response)
