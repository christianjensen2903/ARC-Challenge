import numpy as np
from dataclasses import dataclass


@dataclass
class Demonstration:
    input: np.ndarray
    output: np.ndarray


@dataclass
class Example:
    id: str
    reasoning: str
    code: str
    demonstrations: list[Demonstration]
