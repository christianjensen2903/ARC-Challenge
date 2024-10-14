from langgraph.graph import END, START, Graph
import dotenv
from langchain_openai import ChatOpenAI
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from example_converter import ExampleConverter
import logging
from demonstration_formatter import (
    Demonstration,
    DemonstrationFormatter,
    EmojisDemonstrations,
    ShapeExtractionWrapper,
    DifferenceWrapper,
)
import numpy as np

dotenv.load_dotenv()


def load_data() -> tuple[dict, dict]:
    with open("data/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)

    with open("data/arc-agi_training_solutions.json") as f:
        solutions = json.load(f)

    return challenges, solutions


class ARCSolver:

    def __init__(self, demonstration_formatter: DemonstrationFormatter):
        self.demonstration_formatter = demonstration_formatter
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.challenges, self.solutions = load_data()

        self.graph = self._create_graph()

    def _create_graph(self):
        graph = Graph()
        graph.add_node("load", self.load_demonstrations)
        graph.add_node("format", self.format_demonstrations)
        graph.add_node("agent", self.call_model)

        graph.add_edge(START, "load")
        graph.add_edge("load", "format")
        graph.add_edge("format", "agent")
        graph.add_edge("agent", END)
        return graph.compile()

    def load_demonstrations(self, id: str) -> list[Demonstration]:
        logging.info(f"Loading demonstrations for {id}")
        demonstrations_json = self.challenges[id]["train"]
        demonstrations = [
            Demonstration(
                input=np.array(demonstration["input"]),
                output=np.array(demonstration["output"]),
            )
            for demonstration in demonstrations_json
        ]
        return demonstrations

    def format_demonstrations(self, demonstrations: list[Demonstration]) -> str:
        logging.info("Formatting demonstrations")
        formatted_demonstrations = self.demonstration_formatter.format(demonstrations)
        return formatted_demonstrations

    def call_model(self, formatted_demonstrations: str) -> BaseMessage:
        logging.info("Calling model")
        response = self.model.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant that solves the demonstrations."
                ),
                HumanMessage(content=formatted_demonstrations),
            ]
        )
        return response

    def solve(self, id: str) -> str:
        final_state = self.graph.invoke(id)
        return final_state


if __name__ == "__main__":
    solver = ARCSolver(EmojisDemonstrations())
    print(solver.solve("05f2a901"))
