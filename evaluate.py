from pipeline import Pipeline, load_data
from langchain_openai import ChatOpenAI
import logging
from demonstration_formatter import (
    Demonstration,
    DemonstrationFormatter,
    EmojisDemonstrations,
    ShapeExtractionWrapper,
    DifferenceWrapper,
)
from solver import COTSolver


def evaluate(pipeline: Pipeline, n: int = 5):
    pipeline.eval_mode()
    challenges, solutions = load_data(train=False)
    ids = list(challenges.keys())
    correct = 0
    for i in range(n):
        id = ids[i]
        result = pipeline.solve(id)
        if result[0]:
            correct += 1
    return correct / n


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    formatter = EmojisDemonstrations()
    solver = COTSolver(model, formatter=formatter)
    pipeline = Pipeline(demonstration_formatter=formatter, solver=solver)
    accuracy = evaluate(pipeline)
    print(f"Accuracy: {accuracy}")
