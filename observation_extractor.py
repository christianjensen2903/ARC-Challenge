from generate_text import GPT4
from example_converter import ExampleConverter
import preprocessing

TEMPLATE = """
Positions are denonated x.y where the x is the column letter and y is the row letter.
A square is denoted by x.y-x.y where the first part is the top left corner and the second part is the bottom right corner.

Your task is to extract observations from the examples

The workflow is as follows first find more basic observations and then more complex observations.

First extract the basic observations from the examples.
This should be observations like:
    The shapes in the grid
        This could be shapes of the same color, colors in proximity / is connected etc.
    The colors in the grid

Then extract more complex observations.
This could be observations like:
    If some order of colors are repeating
    If some shapes are repeating
    The position relative to some other shape
    If some shape moves or rotates or changes in some way between the input and output
    Whether a shape is growing or shrinking
    Whether a shape is changing color
    If the grid can be divided into subgrids with some pattern

Remember these are pure observations and not trying to infer the underlying pattern.
Be specific about what you observe. If it is a shape then extract the shape and states is exact position.
If something moves describe the what row and column it moved from and to.
When you make an observation please do it for all the examples.

Then reason about how the input is transformed to the output given the examples and the observations.
You should verify that the reasoning holds for all the examples.

Then apply this reasoning to the input grid.

Given these examples
Example 1: 
Input: 
   A B C D E F G
A |1| | |5| |1| |
B | |1| |5|1|1|1|
C |1| | |5| | | |
 
Output: 
   A B C
A | | | |
B | |2| |
C | | | |


Example 2: 
Input: 
   A B C D E F G
A |1|1| |5| |1| |
B | | |1|5|1|1|1|
C |1|1| |5| |1| |
 
Output: 
   A B C
A | |2| |
B | | |2|
C | |2| |

The output would be:
Basic observations:
Example 1:
Input
    Shapes:
        1:
               A B 
            A |1| |
            B | |1|
            C |1| |

        2:
               D
            A |5|
            B |5|
            C |5|

        3:
               E F G
            A | |1| |
            B |1|1|1|

    Colors: [1, 5]

Output
    Shapes:
        1:
               B
            B |2|

    Colors: [2]

Example 2: 
Input:
    Shapes:
        1:
               A B C 
            A |1|1| |
            B | | |1|
            C |1|1| |

        2:
               D
            A |5|
            B |5|
            C |5|

        3:
               E F G
            A | |1| |
            B |1|1|1|
            C | |1| |

    Colors: [1, 5]

Output:
    Shapes:
        1:
               B C
            A |2| |
            B | |2|
            C |2| |

    Colors: [2]

More complex observations:
Example 1:
    The input grid can be divided into two 3x3 grids by the column of 5's where the subgrids are
       A B C
    A |1| | |
    B | |1| |
    C |1| | |
    and
       A B C
    A | |1| |
    B |1|1|1|
    C | | | |

Example 2:
    The input grid can be divided into two 3x3 grids by the column of 5's where the subgrids are
       A B C
    A |1|1| |
    B | | |1|
    C |1|1| |
    and
       A B C
    A | |1| |
    B |1|1|1|
    C | |1| |


Given these examples
Example 1: 
Input: 
   A B C D E F G H I J
A | | | | | | | | | | |
B | |2|2| | | | | | | |
C | |2|2| | | | | | | |
D |2|2|2| | | | | | | |
E | |2|2| | | |8|8| | |
F | | | | | | |8|8| | |
G | | | | | | | | | | |
H | | | | | | | | | | |
I | | | | | | | | | | |
 
Output: 
   A B C D E F G H I J
A | | | | | | | | | | |
B | | | | |2|2| | | | |
C | | | | |2|2| | | | |
D | | | |2|2|2| | | | |
E | | | | |2|2|8|8| | |
F | | | | | | |8|8| | |
G | | | | | | | | | | |
H | | | | | | | | | | |
I | | | | | | | | | | |


Example 2: 
Input: 
   A B C D E F G H I J
A | | | | | | | | | | |
B | | | |8|8| | | | | |
C | | | |8|8| | | | | |
D | | | | | | | | | | |
E | | | | | | | | | | |
F | | | | | | | | | | |
G | | | |2|2|2| | | | |
H | |2|2|2|2|2| | | | |
I | | |2|2| | | | | | |
J | | | | | | | | | | |
K | | | | | | | | | | |
 
Output: 
   A B C D E F G H I J
A | | | | | | | | | | |
B | | | |8|8| | | | | |
C | | | |8|8| | | | | |
D | | | |2|2|2| | | | |
E | |2|2|2|2|2| | | | |
F | | |2|2| | | | | | |
G | | | | | | | | | | |
H | | | | | | | | | | |
I | | | | | | | | | | |
J | | | | | | | | | | |
K | | | | | | | | | | |

The output would be:
Basic observations:
Example 1:
Input
    Shapes:
        1:
               G H
            E |8|8|
            F |8|8|

        2:
               A B C
            B | |2|2|
            C | |2|2|
            D |2|2|2|
            E | |2|2|

    Colors: [2, 8]

Output
    Shapes:
        1:
               G H
            E |8|8|
            F |8|8|

        2:
               D E F 
            B | |2|2|
            C | |2|2|
            D |2|2|2|
            E | |2|2|

        3:

               D E F G H 
            B | |2|2| | |
            C | |2|2| | |
            D |2|2|2| | |
            E | |2|2|8|8|
            F | | | |8|8|

    Colors: [2, 8]


Example 2:
Input
    Shapes:
        1:
               D E
            B |8|8|
            C |8|8|

        2:
               B C D E F
            D | | |2|2|2|
            E |2|2|2|2|2|
            F | |2|2| | |
            

    Colors: [2, 8]

Output
    Shapes:
        1:
               D E
            B |8|8|
            C |8|8|

        2:
               B C D E F
            G | | |2|2|2|
            H |2|2|2|2|2|
            I | |2|2| | |

        3:
               B C D E F
            A | | | | | |
            B | | |8|8| |
            C | | |8|8| |
            D | | |2|2|2|
            E |2|2|2|2|2|
            F | |2|2| | |

    Colors: [2, 8]


More complex observations:
Example 1:
Shape 1 maintains its position G.E-H.F
Shape 2 moves from A.B-C.E to D.B-F.E
A shape consisting of 1 and 2 is created in the output grid
The colors stay the same

Example 2:
Shape 1 maintains its position D.B-E.C
Shape 2 moves from B.D-F.F to B.G-F.I
A shape consisting of 1 and 2 is created in the output grid
The colors stay the same


Given these examples
{examples_str}

The output would be:
"""


class ObservationExtractor:

    def __init__(self, generator: GPT4):
        self.generator = generator

    def extract(self, examples_str: str) -> str:
        prompt = TEMPLATE.format(examples_str=examples_str)
        response = self.generator.generate(prompt)
        return response


if __name__ == "__main__":
    generator = GPT4()
    extractor = ObservationExtractor(generator)
    challenges, solutions = preprocessing.load_data()
    example_converter = ExampleConverter()
    example_id = "e40b9e2f"
    examples_str = example_converter.extract(challenges[example_id]["train"])
    observations = extractor.extract(examples_str)
    print(observations)
