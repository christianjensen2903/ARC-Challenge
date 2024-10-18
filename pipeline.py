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
    RotateWrapper,
)
import numpy as np
from solver import Solver, COTSolver
from run_program import run_program
from langsmith import traceable
from llm import LLM, GPT4

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

    def train_mode(self):
        self.challenges, self.solutions = load_data(train=True)

    def eval_mode(self):
        self.challenges, self.solutions = load_data(train=False)

    @traceable(
        name="load_demonstrations",
        process_outputs=lambda x: {
            "demonstrations": [
                {
                    "input": EmojisDemonstrations().grid_to_text(demonstration.input),
                    "output": EmojisDemonstrations().grid_to_text(demonstration.output),
                }
                for demonstration in x
            ]
        },
    )
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
        return demonstrations

    @traceable(name="call_model")
    def _call_model(self, demonstrations: list[Demonstration]) -> str:
        logging.info("Calling model")
        return self.solver.solve(demonstrations)

    @traceable(
        name="run_program",
        process_outputs=lambda x: {
            "prediction": EmojisDemonstrations().grid_to_text(x["prediction"]),
            "stdout": x["stdout"],
            "stderr": x["stderr"],
        },
    )
    def _run_program(
        self, solution: str, input: np.ndarray
    ) -> dict[str, str | np.ndarray]:
        logging.info("Running program")

        prediction, stdout, stderr = run_program(solution, input)
        logging.info(f"Prediction: {prediction}")
        logging.info(f"Program stdout: {stdout}")
        logging.info(f"Program stderr: {stderr}")

        return {
            "prediction": prediction,
            "stdout": stdout,
            "stderr": stderr,
        }

    @traceable(
        name="load_test_demonstration",
        process_outputs=lambda x: {
            "input": EmojisDemonstrations().grid_to_text(x.input),
            "output": EmojisDemonstrations().grid_to_text(x.output),
        },
    )
    def _load_test_demonstration(self, id: str) -> Demonstration:
        logging.info(f"Loading test demonstration for {id}")
        input = np.array(self.challenges[id]["test"][0]["input"])
        output = np.array(self.solutions[id][0])
        return Demonstration(input=input, output=output)

    @traceable(name="solve")
    def solve(self, id: str) -> np.ndarray:
        demonstrations = self._load_demonstrations(id)
        solution = self._call_model(demonstrations)
        test_demonstration: Demonstration = self._load_test_demonstration(id)
        output: dict[str, str | np.ndarray] = self._run_program(
            solution, test_demonstration.input
        )
        assert isinstance(output["prediction"], np.ndarray)
        return output["prediction"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = GPT4(mini=True)
    formatter: DemonstrationFormatter = EmojisDemonstrations()
    # formatter = RotateWrapper(formatter)
    formatter = ShapeExtractionWrapper(formatter)
    # formatter = DifferenceWrapper(formatter)
    solver = COTSolver(model, formatter=formatter, num_examples=4, num_solutions=4)

    train = True
    id = "007bbfb7"
    challenges, solutions = load_data(train)

    pipeline = Pipeline(demonstration_formatter=formatter, solver=solver, train=train)
    input = np.array(challenges[id]["test"][0]["input"])
    prediction = pipeline.solve(id)
    solution = np.array(solutions[id][0])
    formatted_input = formatter.grid_to_text(input)
    formatted_solution = formatter.grid_to_text(solution)
    formatted_prediction = formatter.grid_to_text(prediction)
    print("Input:")
    print(formatted_input)

    print("Prediction:")
    print(formatted_prediction)

    print("Solution:")
    print(formatted_solution)

    print("Correct:")
    print(np.array_equal(prediction, solution))
