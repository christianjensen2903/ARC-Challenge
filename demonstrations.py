demonstrations = {
    "08ed6ac7": """
Observations:
We can see that input of both examples consists of 4 shapes.
We can see that the output consists of shapes of the same sizes but with different colors.
The output shapes consists of the colors: Green, Blue, Red, and Yellow.
The order of the colors is not the same in both examples.
The smallest shape is yellow in both examples.
The largest shape is red in both examples.
The second largest shape is green in both examples.
The third largest shape is blue in both examples.

Reasoning:
The pattern seems to be that the shapes stay the same size and positions but the colors change.
The colors seem to be dependent on the size of the shape.
The largest shape is colored red.
The second largest shape is colored green.
The third largest shape is colored blue.
The smallest shape is colored yellow.

Pattern:
The shapes stay the same size and positions but the colors change.
The largest shape is colored red.
The second largest shape is colored green.
The third largest shape is colored blue.
The smallest shape is colored yellow.
""",
    "0962bcdd": """
Observations:
Both examples consist of 2 shapes in the input.
Both examples consist of 2 shapes in the output.
The colors varies between the examples but stays the same between the input and output.
The input shapes in the input always seem to be
|  |🟠|  |
|🟠|🟢|🟠|
|  |🟠|  |
But the colors varies.
Where the output shapes are always
|🟢|  |🟠|  |🟢|
|  |🟢|🟠|🟢|  |
|🟠|🟠|🟢|🟠|🟠|
|  |🟢|🟠|🟢|  |
|🟢|  |🟠|  |🟢|
But the colors varies however stays the same between the input and output.

Reasoning:
It seems based on the observations that there is a relationship between the input and output shapes.
The input shapes seems to always be the same but then is transformed into the output shapes.
The left corner and right corner of the input shapes and output shapes seem to change.
However if we look at the center of the shapes they seem to remain the same.
If we look at the color in the center of the input shapes it is the same color as the center of the output shapes.
The center of the input shapes always seems to transform to a sort of cross shape looking like this:
|🟢|  |  |  |🟢|
|  |🟢|  |🟢|  |
|  |  |🟢|  |  |
|  |🟢|  |🟢|  |
|🟢|  |  |  |🟢|
Where the color of the cross shape is the same as the color in the center of the input shapes.
The surrounding colors in the input seems to be converted to a plus shape looking like this:
|  |  |🟠|  |  |
|  |  |🟠|  |  |
|🟠|🟠|  |🟠|🟠|
|  |  |🟠|  |  |
|  |  |🟠|  |  |
Where the color of the plus shape is the same as the surrounding colors in the input shapes.

Pattern:
The input shapes of the form
|  |🟠|  |
|🟠|🟢|🟠|
|  |🟠|  |
Is transformed into the output shapes of the form
|🟢|  |🟠|  |🟢|
|  |🟢|🟠|🟢|  |
|🟠|🟠|🟢|🟠|🟠|
|  |🟢|🟠|🟢|  |
|🟢|  |🟠|  |🟢|
Where the color of the cross shape in the input is the same as the center of the output shapes.
And the colors of the plus shape in the input is the same as the surrounding colors in the output shapes.
The center of the input and output shapes always remain the same.
""",
    "31aa019c": """
Observations:
The outputs seems to always consist of a green square with another color in the center.
The color of the center is different between all examples.
The input consist of a a lot of different shapes and colors.

Reasoning:
There doesn't seem to be any clear pattern in the input.
The output however seems to be more consistent.
It always consists of a green square with another color in the center.
For the first example the center color is yellow.
We can see there is also a yellow square in the same position in the input.
For the second example the center color is black and there is also a black square in the same position in the input.
The same pattern is seen in the third example but here the center color is blue.
So the pattern seems to involve taking a square and surround it with green squares and remove all other squares from the input.
The only missing part if figuring out how the square is picked out of the input.
If we look at the first example with the yellow square in the center. Then it is the only yellow square in the input.
If we look at the second example with the black square in the center. Then it is the only black square in the input.
The same pattern seems to apply for the third example.
The pattern could be that the color chosen is the one there is only one of in the input.
However to confirm this we should look at the examples to verify that this is the case.
For the first example there is two blues, three purples, two white, two green, and seven reds.
For the second example there is six blues, four purples, three browns, four oranges, two green, and three yellows.
For the third example there is two white, two oranges, two black, three greens, two purples, two browns and three reds.
The pattern therefore seems to be that the color chosen is the one there is only one of in the input.

Pattern:
The color which there is only one of in the input is kept in the output and all other colors are removed.
The square is then surrounded by green squares.
""",
}
