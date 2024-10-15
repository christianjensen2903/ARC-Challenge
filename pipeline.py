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


def load_data(train: bool = True) -> tuple[dict, dict]:
    path_type = "training" if train else "evaluation"
    with open(f"data/arc-agi_{path_type}_challenges.json") as f:
        challenges = json.load(f)

    with open(f"data/arc-agi_{path_type}_solutions.json") as f:
        solutions = json.load(f)

    return challenges, solutions


class Pipeline:

    def __init__(
        self,
        demonstration_formatter: DemonstrationFormatter,
        solver: Solver,
        train: bool = True,
    ):
        self.demonstration_formatter = demonstration_formatter
        self.solver = solver
        self.challenges, self.solutions = load_data(train)
        self.graph = self._create_graph()
        self.id: str | None = None

    def train_mode(self):
        self.challenges, self.solutions = load_data(train=True)

    def eval_mode(self):
        self.challenges, self.solutions = load_data(train=False)

    def _create_graph(self):
        graph = Graph()
        graph.add_node("load", self._load_demonstrations)
        graph.add_node("agent", self._call_model)
        graph.add_node("run_program", self._run_program)
        graph.add_edge(START, "load")
        graph.add_edge("load", "agent")
        graph.add_edge("agent", "run_program")
        graph.add_edge("run_program", END)
        return graph.compile()

    def _load_demonstrations(self, id: str) -> list[Demonstration]:
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

    def _call_model(self, demonstrations: list[Demonstration]) -> tuple[str, float]:
        logging.info("Calling model")
        return self.solver.solve(demonstrations)

    def solve(self, id: str) -> tuple[np.ndarray, float]:
        prediction, cost, _, _ = self.graph.invoke(id)
        return prediction, cost

    def _run_program(
        self, solution: tuple[str, float]
    ) -> tuple[np.ndarray, float, str, str]:
        prediction, cost = solution
        logging.info("Running program")
        input = np.array(self.challenges[self.id]["test"][0]["input"])
        result, stdout, stderr = run_program(prediction, input)
        logging.info(f"Program output: {result}")
        logging.info(f"Program stdout: {stdout}")
        logging.info(f"Program stderr: {stderr}")

        # If result is None output the input
        if result is None:
            return input, cost, stdout, stderr

        return result, cost, stdout, stderr

    def _evaluate(
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

    challenges, solutions = load_data()
    id = "05f2a901"

    pipeline = Pipeline(demonstration_formatter=formatter, solver=solver)
    prediction, cost = pipeline.solve("05f2a901")
    solution = np.array(solutions[id][0])
    formatted_solution = formatter.grid_to_text(solution)
    formatted_prediction = formatter.grid_to_text(prediction)
    print("Prediction:")
    print(formatted_prediction)

    print("Solution:")
    print(formatted_solution)

    print("Correct:")
    print(np.array_equal(prediction, solution))

    print("Cost:")
    print(cost)
