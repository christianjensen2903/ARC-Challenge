import numpy as np
from dataclasses import dataclass


@dataclass
class Demonstration:
    input: np.ndarray
    output: np.ndarray


@dataclass
class Step:
    demonstration_index: int
    reasoning: str
    hypothesis: str
    code: str


@dataclass
class Example:
    id: str
    steps: list[Step]
    demonstrations: list[Demonstration]
