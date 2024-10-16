import json
from models import Example, Demonstration
import numpy as np


with open("data/arc-agi_training_challenges.json") as f:
    challenges = json.load(f)

examples: list[Example] = [
    Example(
        id="4be741c5",
        reasoning="""
**Demonstration 1**
Reasoning:
We see that both the input and output consist of yellow, green and white cells which also happens to be the amount of cells in the output.
If we look at the colors ordered from left to right we see they have the same ordering of colors.

Hypothesis:
The output are the distinct colors in the input ordered from left to right.

**Demonstration 2**
Reasoning:
We can see our theory doesn't hold for the second demonstration. As the colors left to right depending on which row only consist of 1 or two colors where as the output consist of 3 distinct colors.
However if we look at the output we can see that the colors are ordered vertically instead of horizontally.
If we then look at the colors ordered from top to bottom in the input we see they have the same ordering of colors.
How do we figure out if it's top to bottom or left to right based on the input?
We can see if we look at the top row of the input. If it consist of 1 color it's top to bottom. If it consist of 2 or more colors it's left to right.

Hypothesis:
The output are the distinct colors in the input ordered from either top to bottom or left to right depending on how many colors the top row consists of.

**Demonstration 3**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

Final theory:
The output are the distinct colors in the input ordered from either top to bottom or left to right depending on how many colors the top row consists of.

The code should:
1. Look at the top row of the input to determine left to right or top to bottom.
2. Identify the colors and their ordering based on the direction.
3. Output the colors in the correct ordering in the correct direction.
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:
    # Identify if the ordering of colors is from left to right or top to bottom.
    left_column = grid[:, 0]
    top_row = grid[0, :]
    # Get distinct values
    left_column_values = list(dict.fromkeys(left_column))
    top_row_values = list(dict.fromkeys(top_row))
    if len(top_row_values) == 1:
        # Ordering is from left to right
        left_to_right = False
    else:
        # Ordering is from top to bottom
        left_to_right = True
    # Get the colors of the segments
    if left_to_right:
        # Get the colors of the segments
        colors = top_row_values
    else:
        # Get the colors of the segments
        colors = left_column_values
    # Output the colors in the correct ordering
    output = np.array([colors])
    if not left_to_right:
        output = output.T
    return output
    """,
        demonstrations=[],
    ),
    Example(
        id="228f6490",
        reasoning="""
**Demonstration 1**
Reasoning:
We can see that the input and output mostly consist of the same shapes in the same locations.
However there seems to be a two shapes missing: The white square and the grey rectangle.
However if we look closer we can see that they are actually not missing, but rather moved.
We see that we have the same large purple shapes in both the input and output.
In the input we can see there are "holes" in the purple shapes that is then filled with the white square and the grey rectangle.
Why these shapes?
We can see that the white square has the same size as the "hole" in one of the purple shapes and the grey has the same size as the "hole" in the other.
So it seems it "fills" the whole in the purple shapes with shapes of the same size.
However we see that there is an orange shape with the same size as the grey rectangle.
Why is the grey rectangle chosen over the orange?
The rule could be that it choose the shape that is closest.

However let's keep looking for other patterns.
If we look at the colors of the shapes excluding the purple ones we see there is 6 shapes. 1 white, 1 grey and 4 orange.
So the pattern could also be to choose the shapes that have a distinct color.

Both rules seems to apply and seems like logical rules.
We will choose the second rule since it seems to be a more destinct pattern.

Hypothesis:
The purple shapes are filled with the shapes that have a distinct color.

**Demonstration 2**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

**Demonstration 3**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

Final theory:
The purple shapes are filled with the shapes that have a distinct color.

The code should at high level:
1. Identify the two shapes with distinct colors.
2. Identify the purple shapes.
3. Identify the shapes of the "holes" in the purple shapes.
4. Move the shapes that have a distinct color to the "holes" in the purple shapes.

The code should at low level:
1. Identify unique colors
2. Find contiguous shapes of each color
3. Find shapes with a distinct color
4. Identify the purple shapes
5. Copy the input grid to the output grid
6. Iterate over the purple shapes
7. Find the bounding box of the purple shape
8. Find the "hole" in the purple shape
9. Find the shape with a distinct color that has the same shape as the "hole"
10. Iterate over the shapes with a distinct color
11. Normalize the shape and the hole to row 0 and col 0 to being able to compare with the "hole"
12. Check if the normalized shape and the normalized hole are the same
13. Set the hole to the color of the shape
14. Set the old location of the shape to black
15. Return the new grid
""",
        code="""
from scipy.ndimage import label

def find_contiguous_shapes(grid, color):
    labeled_array, num_features = label(grid == color)
    shapes = []
    for i in range(1, num_features + 1):
        shapes.append(np.argwhere(labeled_array == i))
    return shapes

def normalize_shape(shape):
    rows, cols = shape[:, 0], shape[:, 1]
    min_row, min_col = rows.min(), cols.min()
    return shape - [min_row, min_col]

def shape_is_equal(shape1, shape2):
    return np.array_equal(np.sort(shape1, axis=0), np.sort(shape2, axis=0))

def transform(grid: np.ndarray) -> np.ndarray:
    # Find shapes with distinct colors
    colors = np.unique(grid)
    shapes_with_distinct_colors = []
    for color in colors:
        if color == 0:
            continue
        shapes = find_contiguous_shapes(grid, color)
        if len(shapes) == 1:
            shapes_with_distinct_colors.append((shapes[0], color))

    # Identify the purple shapes
    purple_color = 5
    purple_shapes = find_contiguous_shapes(grid, purple_color)

    output = grid.copy()

    for purple_shape in purple_shapes:

        # Find the bounding box of the shape
        rows, cols = purple_shape[:, 0], purple_shape[:, 1]
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Extract the sub-grid containing the shape, cropping to the bounding box
        purple_shape_grid = grid[min_row : max_row + 1, min_col : max_col + 1]

        # Find the shapes of the "hole" in the purple shape
        hole = find_contiguous_shapes(purple_shape_grid, 0)[0]

        # Find the shape with a distinct color that matches the size of the hole
        for shape, color in shapes_with_distinct_colors:

            # Normalize the shape and the hole
            normalized_shape = normalize_shape(shape.copy())
            normalized_hole = normalize_shape(hole.copy())

            if shape_is_equal(normalized_shape, normalized_hole):
                # Move the shape into the hole
                for cell in hole:
                    output[cell[0] + min_row, cell[1] + min_col] = color
                
                # Replace the old location of the shape with black
                for cell in shape:
                    output[cell[0], cell[1]] = 0
                
                break

    return output
""",
        demonstrations=[],
    ),
    Example(
        id="760b3cac",
        reasoning="""
**Demonstration 1**
Reasoning:
We see that the yellow shape is present in both the input and output.
We also see that the white shape is present in both.
We however also see a new shape is added to the output. This seems to be the reflection of the white shape where is has been reflected to the left.
So it seems that the output is the same as the input but where the white shape is reflected to the left aswell.

Hypothesis:
The output is the same as the input but the white shape is reflected to the left.

**Demonstration 2**
Reasoning:
We see that the yellow and white shapes again is the same in the input and output.
It also seems like there is added a reflection this time. However this time the white shape is reflected to the right instead of to the left.
Why can this be?
Maybe it has something to do with shape of the white shape?
If we look at the first demonstration the left side is 1 cell and the right side is 3 cells.
It could be that is reflection depends on the length of the side.
If we look at the second demonstration the left side is 2 cells and the right side is 2 cells.
It could then be that it is reflected on the side with the shortest length.

Hypothesis:
The output is the same as the input but the white shape is reflected to either the left or right depending on which side is the shortest of the white shape.

**Demonstration 3**
Reasoning:
We can see that our theory with shapes appearing in the input and output is still valid.
This time it is reflected to the left. If we look a the length of the sides we see that the left side is 2 cells and the right side is 1 cell.
It therefore seems like our theory doesn't work for the third demonstration.

Let's see if we can figure out a new rule.
Let's look at the first demonstration again.
If we look at the yellow shape we can see that the left side is 2 cells and the right side is 1 cell.
It could then be that is it reflected based on the shape of the yellow shape.
Let's look at the second demonstration.
This time we can see that the yellow shape has a length of 1 cell on the left side and 2 cells on the right side.
It could therefore be that the white shape is reflected based on the longest side of the yellow shape.

Let's look at the third demonstration.
This time the yellow shape has a length of 2 cells on the left side and 1 cell on the right side.
Our theory therefore still holds.

Hypothesis:
The output is the same as the input but the white shape is reflected to either the left or right depending on which side is the shortest of the yellow shape.

Final theory:
The output is the same as the input but the white shape is reflected to either the left or right depending on which side is the shortest of the yellow shape.

The code should:
1. Copy the input grid to the output grid
2. Identify the yellow shape
3. Identify the longest side of the yellow shape
4. Reflect the white shape based on the longest side of the yellow shape
    """,
        code="""
def normalize_shape(shape: np.ndarray) -> np.ndarray:
    min_row, min_col = np.min(shape, axis=0)
    return shape - [min_row, min_col]


def transform(grid: np.ndarray) -> np.ndarray:
    # Copy the input grid to the output grid
    output_grid = grid.copy()

    # Identify the yellow shape
    yellow_shape = np.argwhere(grid == 4)

    # Identify the longest side of the yellow shape
    _, side_lengths = np.unique(yellow_shape[:, 1], return_counts=True)

    left_is_longer = side_lengths[0] > side_lengths[-1]

    # Reflect the white shape based on the longest side of the yellow shape
    white_shape = np.argwhere(grid == 8)
    flipped_white_shape = np.flip(white_shape, axis=0)
    normalized_flipped_white_shape = normalize_shape(flipped_white_shape)

    for x, y in normalized_flipped_white_shape:
        if left_is_longer:
            output_grid[x, y] = 8
        else:
            output_grid[x, y + 6] = 8

    return output_grid
    """,
        demonstrations=[],
    ),
    Example(
        id="253bf280",
        reasoning="""
**Demonstration 1**
Reasoning:
We see that the two white dots are present at the same location in the input and output.
The pattern could be that the two white dots are connected by a blue line.

Hypothesis:
The output is the same as the input but the two white dots are connected by a blue line.

**Demonstration 2**
Reasoning:
We see that there this time is 4 white dots and also is at the same location in the input and output.
However we can see that they are not all connected to each other.
How do we know which white dots we should connect?
On the first demonstration it is simple. We connect the two white dots.
On the second demonstration it makes sense to connect the dots that are in the same column since the lines else couldn't be straight.

Hypothesis:
The output is the same as the input but the white dots one the same row or column are connected by a blue line.

**Demonstration 3**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

**Demonstration 4**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

**Demonstration 5**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

**Demonstration 6**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

**Demonstration 7**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

**Demonstration 8**
Reasoning:
Our hypothesis seems to be valid for the third demonstration.

Hypothesis:
No changes from the previous demonstration.

Final theory:
The output is the same as the input but the white dots one the same row or column are connected by a blue line.

The code should:
1. Copy the input grid to the output grid
2. Go through each row and connect the white dots on each row
3. Go through each column and connect the white dots on each column
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:
    # Copy the input grid to the output grid
    output_grid = grid.copy()

    # Go through each row and connect the white dots on each row
    for row in range(grid.shape[0]):
        white_dots = np.argwhere(grid[row] == 8)

        # If less that 2 white dots, no line
        if len(white_dots) < 2:
            continue

        # Blue line between the two white dots
        for i in range(white_dots[0, 0] + 1, white_dots[1, 0]):
            output_grid[row, i] = 3

    # Go through each column and connect the white dots on each column
    for col in range(grid.shape[1]):
        white_dots = np.argwhere(grid[:, col] == 8)

        # If less that 2 white dots, no line
        if len(white_dots) < 2:
            continue

        # Blue line between the two white dots
        for i in range(white_dots[0, 0] + 1, white_dots[1, 0]):
            output_grid[i, col] = 3

    return output_grid
    """,
        demonstrations=[],
    ),
    Example(
        id="1f642eb9",
        reasoning="""
**Demonstration 1**
Reasoning:
We can see that both the input and output contain the same three distinct colored cells colored brown. grey and yellow.
We can also see that the input and output both contain a white rectangle where it in the output is partially covered by 3 cells with the same 3 distinct colors.
If we look at the position it looks like they are directly projected onto the white rectangle in the middle.

Hypothesis:
The output is the same as the input but the 3 distinct colored cells are in addition also projected onto the white rectangle in the middle.

**Demonstration 2**
Reasoning:
Our hypothesis mainly seems to be valid for the second demonstration with one caveat. There is not only 3 distinct colored cells in the output but 5.
It therefore doesn't seem to be depending on the number of distinct colored cells.

Hypothesis:
The output is the same as the input but the distinct colored cells are in addition also projected onto the white rectangle in the middle.

**Demonstration 3**
Reasoning:
Our hypothesis mainly seems to be valid for the third demonstration.
This time the projected cells are not distinct colored cells however they are still only single cells.

Hypothesis:
The output is the same as the input but the remaining cells are in addition also projected onto the white rectangle in the middle.

Final theory:
The output is the same as the input but the remaining cells are in addition also projected onto the white rectangle in the middle.

The code should:
1. Copy the input grid to the output grid
2. Identify the white rectangle
3. Identify the remaining cells in the input
4. Project the remaining cells onto the white rectangle
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:
    # Copy the input grid to the output grid
    output_grid = grid.copy()
    
    # Identify the white rectangle
    white_rectangle = np.argwhere(grid == 8)
    min_row, min_col = np.min(white_rectangle, axis=0)
    max_row, max_col = np.max(white_rectangle, axis=0)
    
    # Identify the remaining cells in the input
    remaining_cells = np.argwhere((grid != 0) & (grid != 8))
    
    # Project the remaining cells onto the white rectangle
    for cell in remaining_cells:
        row, col = cell
        if row < min_row:
            output_grid[min_row, col] = grid[row, col]
        elif row > max_row:
            output_grid[max_row, col] = grid[row, col]
        elif col < min_col:
            output_grid[row, min_col] = grid[row, col]
        elif col > max_col:
            output_grid[row, max_col] = grid[row, col]

    return output_grid
    """,
        demonstrations=[],
    ),
    Example(
        id="a5313dff",
        reasoning="""
**Demonstration 1**
Reasoning:
We can see that the input and output both contain the same green shape.
In the output the interior is however colored red.

Hypothesis:
The output is the same as the input but the interior of the green shape is colored red.

**Demonstration 2**
Reasoning:
Our hypothesis seems to be valid for the second demonstration however with one caveat.
If the green shape is not "closed" (i.e. it has a hole in it) then interior doesn't change

Hypothesis:
The output is the same as the input but the interior of the green shape is colored red but only if the shape is "closed".

**Demonstration 3**
Reasoning:
Our hypothesis seems to be valid for the third demonstration however with one caveat.
There seems to be possible that there more than one "hole" in the green shape.

Hypothesis:
The output is the same as the input but all the "holes" of the green shape is colored red but only if the shape is "closed".

Final theory:
The output is the same as the input but all the "holes" of the green shape is colored red but only if the shape is "closed".

The code should:
1. Copy the input grid to the output grid
2. Identify the green shape
3. Identify the "holes" of the green shape
4. Color the "holes" red
    """,
        code="""
from scipy.ndimage import binary_fill_holes

def transform(grid: np.ndarray) -> np.ndarray:
    # Copy the input grid to the output grid
    output_grid = grid.copy()

    # Identify the green shape
    green_shape_mask = grid == 2
    
    # Identify the "holes" of the green shape
    filled_interior = binary_fill_holes(green_shape_mask)

    # Set the interior of the shape to 1 (red)
    output_grid[(filled_interior) & (~green_shape_mask)] = 1

    return output_grid
    """,
        demonstrations=[],
    ),
    Example(
        id="fcb5c309",
        reasoning="""
**Demonstration 1**
Reasoning:
We can see that output shape is also present in the input.
In the input is however colored both green and yellow where in the output is it only yellow.
Why is it the yellow color and not green?
It might be due to the boundary color of the shape is green where the interior is yellow.
How do we know which shape to choose from the input?
Based on the input logical choices could be either of the green rectangles, so why is it the one on the left?
Might be because it is bigger than the one on the right.
Could also be because it has something inside it which then choose the color change.
We will go with the latter option since it would make the color changing rule more consistent.

Hypothesis:
The green rectangle with some interior is copied to the output and the boundary of the shape is colored the same color as the interior.

**Demonstration 2**
Reasoning:
Our hypothesis seems to mostly be valid. However this time the chosen shape is red and not green.
So the choice must be independent of the color of the rectangles.
This time both rectangles however have something inside them. However the one that is chosen has "more" inside it.
It however still could be the largest shape.

Hypothesis:
The rectangle with the most cells inside it is copied to the output and its boundary is colored the same color as its interior.

**Demonstration 3**
Reasoning:
Our hypothesis seems to be valid.

Hypothesis:
No changes from the previous demonstration.

Final theory:
The rectangle with the most cells inside it is copied to the output and its boundary is colored the same color as its interior.

Code implementation:
To make it easier to find rectangles we could just relax this to find continous shapes instead and see how many other colored cells there is within its bounds.

The code should:
1. Find all continous shapes
2. Find the one with the most other colored cells within its bounds
3. Copy it to the output
4. Color the boundary of the rectangle the same color as its interior
    """,
        code="""
from scipy.ndimage import label

def find_contiguous_shapes(grid, color):
    labeled_array, num_features = label(grid == color)
    shapes = []
    for i in range(1, num_features + 1):
        shapes.append(np.argwhere(labeled_array == i))
    return shapes

def transform(grid: np.ndarray) -> np.ndarray:
    # Find all continous shapes
    shapes = []
    for color in range(1, 10):
        shapes.extend(find_contiguous_shapes(grid, color))

    # Find the one with the most other colored cells within its bounds
    max_shape = None
    max_other_cells = -1
    inside_color = None
    for shape in shapes:
        min_row, min_col = np.min(shape, axis=0)
        max_row, max_col = np.max(shape, axis=0)
        color = grid[min_row, min_col]
        # Map the shape to the original grid
        shape_in_grid = grid[min_row:max_row+1, min_col:max_col+1]
        amount_of_other_cells = np.sum((shape_in_grid != color) & (shape_in_grid != 0))
        if amount_of_other_cells > max_other_cells:
            max_other_cells = amount_of_other_cells
            max_shape = shape_in_grid
            # Count the amount of each color in the shape
            color_counts = np.bincount(shape_in_grid.flatten())
            color_counts[color] = 0  # Exclude the current shape's color
            color_counts[0] = 0  # Exclude background color
            inside_color = np.argmax(color_counts) if np.max(color_counts) > 0 else None

    for x in range(max_shape.shape[0]):
        for y in range(max_shape.shape[1]):
            if max_shape[x, y] != 0:
                max_shape[x, y] = inside_color
        
    return max_shape
    """,
        demonstrations=[],
    ),
    Example(
        id="67a3c6ac",
        reasoning="""
**Demonstration 1**
Reasoning:
It seems like the transformation is simply flipping the grid over its vertical center.

Hypothesis:
The output is the same as the input but flipped over its vertical center.

**Demonstration 2**
Reasoning:
Our hypothesis seems to be valid.

Hypothesis:
The output is the same as the input but flipped over its vertical center.

**Demonstration 3**
Reasoning:
Our hypothesis seems to be valid.

Hypothesis:
The output is the same as the input but flipped over its vertical center.

Final theory:
The output is the same as the input but flipped over its vertical center.

The code should:
1. Flip the grid over its vertical center
    """,
        code="""
def transform(grid: np.ndarray) -> np.ndarray:
    # Flip the grid over its vertical center
    return np.fliplr(grid)
    """,
        demonstrations=[],
    ),
    Example(
        id="2dc579da",
        reasoning="""
**Demonstration 1**
Reasoning:
It looks like the output is simply the bottom leftmost 2x2 square.

Hypothesis:
The output is the same as the input but with a square in the center.

**Demonstration 2**
Reasoning:
This demonstrations disproves our previous hypothesis.
However we can see that the cross again divides the grid into 4 squares where the output is the one with more than 1 distinct colored cell.

Hypothesis:
The input is divided into 4 square by a cross where the output is the square with more than 1 distinct colored cell.

**Demonstration 3**
Reasoning:
Our hypothesis seems to be valid.

Hypothesis:
No changes from the previous demonstration.

Final theory:
The input is divided into 4 square by a cross where the output is the square with more than 1 distinct colored cell.

The code should:
1. Divide the grid into 4 squares by a cross
2. Return the square with the most other colored cells within its bounds
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:
    # Divide the grid into 4 squares by a cross
    rows, cols = grid.shape
    center_row, center_col = rows // 2, cols // 2
    top_left = grid[:center_row, :center_col]
    top_right = grid[:center_row, center_col+1:]
    bottom_left = grid[center_row+1:, :center_col]
    bottom_right = grid[center_row+1:, center_col+1:]

    # Find the square with the most other colored cells within its bounds
    squares = [top_left, top_right, bottom_left, bottom_right]
    square_counts = [np.sum(np.unique(square, return_counts=True)[1][1:]) for square in squares]

    return squares[np.argmax(square_counts)]
    """,
        demonstrations=[],
    ),
]

# Load demonstrations
for example in examples:
    demonstrations_json = challenges[example.id]["train"]
    demonstrations = [
        Demonstration(
            input=np.array(demonstration["input"]),
            output=np.array(demonstration["output"]),
        )
        for demonstration in demonstrations_json
    ]
    example.demonstrations = demonstrations


# examples = {
#     "08ed6ac7": """
# Observations:
# We can see that input of both examples consists of 4 shapes.
# We can see that the output consists of shapes of the same sizes but with different colors.
# The output shapes consists of the colors: Green, Blue, Red, and Yellow.
# The order of the colors is not the same in both examples.
# The smallest shape is yellow in both examples.
# The largest shape is red in both examples.
# The second largest shape is green in both examples.
# The third largest shape is blue in both examples.
# Reasoning:
# The pattern seems to be that the shapes stay the same size and positions but the colors change.
# The colors seem to be dependent on the size of the shape.
# The largest shape is colored red.
# The second largest shape is colored green.
# The third largest shape is colored blue.
# The smallest shape is colored yellow.
# Pattern:
# The shapes stay the same size and positions but the colors change.
# The largest shape is colored red.
# The second largest shape is colored green.
# The third largest shape is colored blue.
# The smallest shape is colored yellow.
# """,
#     "0962bcdd": """
# Observations:
# Both examples consist of 2 shapes in the input.
# Both examples consist of 2 shapes in the output.
# The colors varies between the examples but stays the same between the input and output.
# The input shapes in the input always seem to be
# |  |🟠|  |
# |🟠|🟢|🟠|
# |  |🟠|  |
# But the colors varies.
# Where the output shapes are always
# |🟢|  |🟠|  |🟢|
# |  |🟢|🟠|🟢|  |
# |🟠|🟠|🟢|🟠|🟠|
# |  |🟢|🟠|🟢|  |
# |🟢|  |🟠|  |🟢|
# But the colors varies however stays the same between the input and output.
# Reasoning:
# It seems based on the observations that there is a relationship between the input and output shapes.
# The input shapes seems to always be the same but then is transformed into the output shapes.
# The left corner and right corner of the input shapes and output shapes seem to change.
# However if we look at the center of the shapes they seem to remain the same.
# If we look at the color in the center of the input shapes it is the same color as the center of the output shapes.
# The center of the input shapes always seems to transform to a sort of cross shape looking like this:
# |🟢|  |  |  |🟢|
# |  |🟢|  |🟢|  |
# |  |  |🟢|  |  |
# |  |🟢|  |🟢|  |
# |🟢|  |  |  |🟢|
# Where the color of the cross shape is the same as the color in the center of the input shapes.
# The surrounding colors in the input seems to be converted to a plus shape looking like this:
# |  |  |🟠|  |  |
# |  |  |🟠|  |  |
# |🟠|🟠|  |🟠|🟠|
# |  |  |🟠|  |  |
# |  |  |🟠|  |  |
# Where the color of the plus shape is the same as the surrounding colors in the input shapes.
# Pattern:
# The input shapes of the form
# |  |🟠|  |
# |🟠|🟢|🟠|
# |  |🟠|  |
# Is transformed into the output shapes of the form
# |🟢|  |🟠|  |🟢|
# |  |🟢|🟠|🟢|  |
# |🟠|🟠|🟢|🟠|🟠|
# |  |🟢|🟠|🟢|  |
# |🟢|  |🟠|  |🟢|
# Where the color of the cross shape in the input is the same as the center of the output shapes.
# And the colors of the plus shape in the input is the same as the surrounding colors in the output shapes.
# The center of the input and output shapes always remain the same.
# """,
#     "31aa019c": """
# Observations:
# The outputs seems to always consist of a green square with another color in the center.
# The color of the center is different between all examples.
# The input consist of a a lot of different shapes and colors.
# Reasoning:
# There doesn't seem to be any clear pattern in the input.
# The output however seems to be more consistent.
# It always consists of a green square with another color in the center.
# For the first example the center color is yellow.
# We can see there is also a yellow square in the same position in the input.
# For the second example the center color is black and there is also a black square in the same position in the input.
# The same pattern is seen in the third example but here the center color is blue.
# So the pattern seems to involve taking a square and surround it with green squares and remove all other squares from the input.
# The only missing part if figuring out how the square is picked out of the input.
# If we look at the first example with the yellow square in the center. Then it is the only yellow square in the input.
# If we look at the second example with the black square in the center. Then it is the only black square in the input.
# The same pattern seems to apply for the third example.
# The pattern could be that the color chosen is the one there is only one of in the input.
# However to confirm this we should look at the examples to verify that this is the case.
# For the first example there is two blues, three purples, two white, two green, and seven reds.
# For the second example there is six blues, four purples, three browns, four oranges, two green, and three yellows.
# For the third example there is two white, two oranges, two black, three greens, two purples, two browns and three reds.
# The pattern therefore seems to be that the color chosen is the one there is only one of in the input.
# Pattern:
# The color which there is only one of in the input is kept in the output and all other colors are removed.
# The square is then surrounded by green squares.
# """,

# }
