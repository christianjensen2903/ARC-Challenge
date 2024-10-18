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
import numpy as np
from evaluate import evaluate
from matplotlib import pyplot as plt
from tqdm import tqdm  # type: ignore


def num_examples_plot():
    x_axis = list(range(2, 9))

    accuracies = []
    for num_examples in tqdm(x_axis):
        model = ChatOpenAI(model="gpt-4o-mini")
        formatter = EmojisDemonstrations()
        solver = COTSolver(
            model, formatter=formatter, num_examples=num_examples, num_solutions=4
        )
        pipeline = Pipeline(demonstration_formatter=formatter, solver=solver)
        accuracy = evaluate(pipeline, n=2)
        accuracies.append(accuracy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot accuracy
    ax1.plot(x_axis, accuracies, marker="o")
    ax1.set_title("Accuracy vs Number of Examples")
    ax1.set_xlabel("Number of Examples")
    ax1.set_ylabel("Accuracy")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
    fig.savefig("cost_accuracy.png")


def num_solutions_plot():
    x_axis = [2**i for i in range(3, 9)]

    accuracies = []
    for num_solutions in tqdm(x_axis):
        model = ChatOpenAI(model="gpt-4o-mini")
        formatter = EmojisDemonstrations()
        solver = COTSolver(
            model, formatter=formatter, num_examples=8, num_solutions=num_solutions
        )
        pipeline = Pipeline(demonstration_formatter=formatter, solver=solver)
        accuracy = evaluate(pipeline, n=20)
        accuracies.append(accuracy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot accuracy
    ax1.plot(x_axis, accuracies, marker="o")
