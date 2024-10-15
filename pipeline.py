from langgraph.graph import END, START, Graph
import dotenv
from langchain_openai import ChatOpenAI
import json
import logging
from demonstration_formatter import (
    Demonstration,
    DemonstrationFormatter,
    EmojisDemonstrations,
    ShapeExtractionWrapper,
    DifferenceWrapper,
)
import numpy as np
from solver import Solver, COTSolver
from run_program import run_program

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
        self.id: str | None = None

    def _create_graph(self):
        graph = Graph()
        graph.add_node("load", self.load_demonstrations)
        graph.add_node("agent", self.call_model)
        graph.add_node("run_program", self.run_program)
        graph.add_node("evaluate", self.evaluate)
        graph.add_edge(START, "load")
        graph.add_edge("load", "agent")
        graph.add_edge("agent", "run_program")
        graph.add_edge("run_program", "evaluate")
        graph.add_edge("evaluate", END)
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
        self.id = id
        return demonstrations

    def call_model(self, demonstrations: list[Demonstration]) -> str:
        logging.info("Calling model")
        return self.solver.solve(demonstrations)

    def solve(self, id: str) -> tuple[bool, str, str]:
        final_state = self.graph.invoke(id)
        return final_state

    def run_program(self, solution: str) -> tuple[np.ndarray | None, str, str]:
        logging.info("Running program")
        input = np.array(self.challenges[self.id]["test"][0]["input"])
        result, stdout, stderr = run_program(solution, input)
        logging.info(f"Program output: {result}")
        logging.info(f"Program stdout: {stdout}")
        logging.info(f"Program stderr: {stderr}")
        return result, stdout, stderr

    def evaluate(
        self, result: tuple[np.ndarray | None, str, str]
    ) -> tuple[bool, str, str]:
        output = result[0]
        logging.info("Evaluating program")
        solution = np.array(self.solutions[self.id][0])
        formatted_solution = self.demonstration_formatter.grid_to_text(solution)
        if output is None:
            return False, "", formatted_solution

        formatted_output = self.demonstration_formatter.grid_to_text(output)
        return np.array_equal(output, solution), formatted_output, formatted_solution


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    formatter = EmojisDemonstrations()
    solver = COTSolver(model, formatter=formatter)

    pipeline = Pipeline(demonstration_formatter=formatter, solver=solver)
    is_correct, formatted_output, formatted_solution = pipeline.solve("05f2a901")
    print(is_correct)
    print(formatted_output)
    print(formatted_solution)
