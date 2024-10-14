from langgraph.graph import END, START, Graph
import dotenv
from langchain_openai import ChatOpenAI
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import logging
from demonstration_formatter import (
    Demonstration,
    DemonstrationFormatter,
    EmojisDemonstrations,
    ShapeExtractionWrapper,
    DifferenceWrapper,
)
from langchain_core.language_models.chat_models import BaseChatModel
import numpy as np
from solver import Solver, IOSolver

dotenv.load_dotenv()


def load_data() -> tuple[dict, dict]:
    with open("data/arc-agi_training_challenges.json") as f:
        challenges = json.load(f)

    with open("data/arc-agi_training_solutions.json") as f:
        solutions = json.load(f)

    return challenges, solutions


class Pipeline:

    def __init__(self, demonstration_formatter: DemonstrationFormatter, solver: Solver):
        self.demonstration_formatter = demonstration_formatter
        self.solver = solver
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

    def call_model(self, formatted_demonstrations: str) -> str:
        logging.info("Calling model")
        return self.solver.solve(formatted_demonstrations)

    def solve(self, id: str) -> str:
        final_state = self.graph.invoke(id)
        return final_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    solver = IOSolver(model)
    formatter = EmojisDemonstrations()

    pipeline = Pipeline(demonstration_formatter=formatter, solver=solver)
    print(pipeline.solve("05f2a901"))
