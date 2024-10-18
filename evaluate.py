from pipeline import Pipeline, load_data
from llm import LLM, GPT4
import logging
from demonstration_formatter import (
    Demonstration,
    DemonstrationFormatter,
    EmojisDemonstrations,
    ShapeExtractionWrapper,
    DifferenceWrapper,
)
from solver import COTSolver
import numpy as np
from langsmith import traceable
from tqdm import tqdm  # type: ignore


@traceable(name="evaluate")
def evaluate(pipeline: Pipeline, n: int = 5) -> float:
    """
    Evaluates the pipeline on the first n challenges in the test set.
    Returns the accuracy and the cost of the evaluations.
    """
    pipeline.eval_mode()
    challenges, solutions = load_data(train=False)
    ids = list(challenges.keys())
    correct = 0
    progress_bar = tqdm(range(n), desc="Evaluating", unit="challenge")

    for i in progress_bar:
        id = ids[i]
        prediction = pipeline.solve(id)
        solution = np.array(solutions[id][0])

        if np.array_equal(prediction, solution):
            correct += 1

        # Update the progress bar with running accuracy
        accuracy = correct / (i + 1)  # running accuracy
        progress_bar.set_postfix(accuracy=accuracy)

    return correct / n


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    model = GPT4()
    formatter: DemonstrationFormatter = EmojisDemonstrations()
    formatter = ShapeExtractionWrapper(formatter)
    formatter = DifferenceWrapper(formatter)
    solver = COTSolver(model, formatter=formatter, num_examples=8, num_solutions=128)
    pipeline = Pipeline(demonstration_formatter=formatter, solver=solver)
    accuracy = evaluate(pipeline, n=50)
    print(f"Accuracy: {accuracy}")
