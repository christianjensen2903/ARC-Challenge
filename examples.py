import json
from models import Example, Demonstration
import numpy as np


with open("data/arc-agi_training_challenges.json") as f:
    challenges = json.load(f)

examples: list[Example] = [
    Example(
        id="4be741c5",
        reasoning="""
The outputs don't have the same shape as the inputs, and they don't appear to be somewhat edited copies of the input.

The inputs appear to consist of "noisy" segments which are either stacked on top of each other or side-by-side. In other words, they are either noisy columns or noisy rows. Each segment consists of exactly one color. These colors also appear in the output.

In two of the example inputs (inputs 2 and 3), there are horizontal segments (rows) which are stacked on top of each other. The outputs for these inputs are each single columns. The color of each cell in the output column is the color of the corresponding segment in the input.

In the other example input (input 1), there are vertical segments (columns) which are stacked side-by-side. The output for this input is a single row. The color of each cell in the output row is the color of the corresponding segment in the input.

In the additional input, there are vertical segments (columns) which are stacked side-by-side. This input matches the "noisy" segment pattern of the other inputs.

The transformation rule appears to be to identify the colors of the segments and then to stack them side-by-side if they are columns or on top of each other if they are rows.

My code will first need to determine if the input consists of column segments or row segments. Then, it will need to identify the colors of the segments and stack them side-by-side or on top of each other as appropriate.

How can I determine if the input consists of column segments or row segments? Inputs which consist of column segments don't necessarily have the same color in each literal column of the grid as it is "noisy". However, they do always have the same color in the leftmost (or rightmost) column. Otherwise, the leftmost (or rightmost) segment wouldn't be contiguous. Similarly, inputs which consist of row segments don't necessarily have the same color in each literal row of the grid as it is "noisy". However, they do always have the same color in the topmost (or bottommost) row.

So, to identify if the input consists of column segments or row segments, I can check if all of the cells in the leftmost column have the same color. If they do, then the input consists of column segments. Otherwise, the input consists of row segments.

I need to know the colors of the segments (in order). I know that the segments are contiguous. So, I can take any row/column which includes all of the segments and then deduplicate the colors in that row/column. The resulting list of colors will be the colors of the segments (in order). If the input consists of column segments, then I want to look at a row/column which will intersect with all of these segments. So, in the case of column segments, I want to look at any row. I'll use the top row. If the input consists of row segments, then I want to look at a row/column which will intersect with all of these segments. So, in the case of row segments, I want to look at any column. I'll use the left column.

The code should:

- Identify if the input consists of column segments or row segments. This can be done by checking if all of the cells in the leftmost column have the same color.
- Identify the colors of the segments (in order). This can be done by looking at the top row if the input consists of column segments and by looking at the left column if the input consists of row segments.
- If the input is columns, return a single row where each cell is the color of the corresponding segment. If the input is rows, return a single column where each cell is the color of the corresponding segment.
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:
    left_column = grid[:, 0]
    top_row = grid[0, :]

    is_columns = np.all(left_column == left_column[0])

    intersecting_item = top_row if is_columns else left_column

    out_in_order = list(dict.fromkeys(intersecting_item))

    if is_columns:
        return np.array([out_in_order])
    else:
        return np.array([[x] for x in out_in_order])
```
""",
        demonstrations=[],
    ),
    Example(
        id="228f6490",
        reasoning="""
The outputs always have the same shape as the inputs. The outputs are similar to the inputs, but some edits have been made.

I'll describe the change from the input to the output in detail for each of the examples. I'll also describe the input for each example as seems relevant.

In example 1, a teal (color 8) 2x2 square is removed from around the bottom left. The output contains a teal 2x2 square in a different position, so it appears as though the square has been moved. It was moved to the top left. It seems to have been moved inside of a grey (color 5) outline that was present in the original input. The new position of the teal 2x2 square used to be black (color 0) in the original input. The teal 2x2 square exactly matches the black 2x2 square that was in the grey outline in the original input. So, there are no longer any black cells in the grey outline. A fuchsia (color 6) 1x2 rectangle was removed from the top right. The output contains a fuchsia 1x2 rectangle in a different position, so it seems to have been moved. It was moved to the bottom right. It seems to have been moved inside of a grey outline that was present in the original input. The new position of the fuchsia 1x2 rectangle used to be black in the original input. The fuchsia 1x2 rectangle exactly matches the black 1x2 rectangle that was in the grey outline in the original input. So, there are no longer any black cells in the grey outline. The rest of the cells in the input are unchanged, including a number of orange (color 7) cells.

In example 2, a green (color 3) L shape is removed from the middle left. The output contains a green L in a different position, so it seems to have been moved. It was moved to the top left. It seems to have been moved inside of a grey outline that was present in the original input. The new position of the green L used to be black in the original input. The green L exactly matches the black L that was in the grey outline in the original input. So, there are no longer any black cells in the grey outline. A brown (color 9) rectangle was removed from the top right. The output contains a brown rectangle in a different position, so it seems to have been moved. It was moved to the bottom right. It seems to have been moved inside of a grey outline that was present in the original input. The new position of the brown rectangle used to be black in the original input. The brown rectangle exactly matches the black rectangle that was in the grey outline in the original input. So, there are no longer any black cells in the grey outline. The rest of the cells in the input are unchanged, including a number of fuchsia cells.

In example 3, a purple (color 2) shape is removed from the top left. The output contains the same purple shape in a different position, so it seems to have been moved. It was moved to the bottom left. It seems to have been moved inside of a grey outline that was present in the original input. The new position of the purple shape used to be black in the original input. The purple shape exactly matches the black shape that was in the grey outline in the original input. So, there are no longer any black cells in the grey outline. A teal (color 8) shape was removed from the bottom right. The output contains a teal shape in a different position, so it seems to have been moved. It was moved to the top right. It seems to have been moved inside of a grey outline that was present in the original input. The new position of the teal shape used to be black in the original input. The teal shape exactly matches the black shape that was in the grey outline in the original input. So, there are no longer any black cells in the grey outline. The rest of the cells in the input are unchanged, including a number of yellow (color 4) cells.

The transformation rule appears to be to take whichever shapes can be used to fill in the gaps in the grey outlines and then move those shapes into the grey outlines. The colored shapes are removed from the old locations (replaced with black) and then the gap is filled in with the corresponding color. This is done while leaving everything else as is.

I should check this proposed rule and make sure it is unambiguous. Are there ever any cases where a different shape could have fit inside the grey outline? I should check example by example.

In example 1, are there any 2x2 squares other than the teal square? No. Are there any 1x2 rectangles other than the fuchsia rectangle? Yes, there is an orange 1x2 rectangle. So, the proposed rule is ambiguous in this case.

So, the rule can't just be to take any shape which fits in the grey outline and move it into the grey outline. I should analyze what's going on with the cells which aren't moved. The cells which aren't moved in example 1 are orange. There are multiple discontinuous orange cells while the teal and fuchsia shapes are unique and contiguous. So, the rule could be to move the contiguous shapes which have a unique color into the outline where they fit. I need to check this rule in the other examples. In example 2, the green and brown shapes are unique and contiguous while the fuchsia cells have multiple discontinuous groups. In example 3, the purple and teal shapes are unique and contiguous while the yellow cells are not. In the additional input, there is a unique orange shape and a unique yellow shape while there are multiple discontinuous groups of purple cells. There are exactly two gaps in the grey outlines. So, this rule is unambiguous here as well.

Ok, I now understand the rule. I need to identify the contiguous shapes which have a unique color and then move them into the corresponding void grey outline.

To implement this, I will:

1. Identify each grey outline. I can do this by finding all of the grey cells and then finding the contiguous grey groupings (which should correspond to each outline). This assumes that each grey outline is disjoint from the others. Is this assumption correct? Yes, it is correct. The grey outlines are disjoint from each other.
2. For each grey outline, I will find the black cells inside of it. Each grey outline is always rectangular, so I can find the bounding box of the grey outline and then find the black cells inside of the bounding box.
3. For each black shape inside of a grey outline, I will find the contiguous shape of the same shape but of a different color. I can do this by finding all of the cells of each color and then finding the contiguous shapes of each color. I will then check if each shape is the same shape as the black shape. I also need to check that the colored shape is the unique contiguous shape of that color.
4. I will then move the colored shape into the grey outline and replace the old location of the colored shape with black.
5. I will then return the new grid.
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
    grey_color = 5
    black_color = 0

    # Find all grey outlines
    grey_shapes = find_contiguous_shapes(grid, grey_color)

    for grey_shape in grey_shapes:
        # Find the bounding box of the grey outline
        min_row, min_col = np.min(grey_shape, axis=0)
        max_row, max_col = np.max(grey_shape, axis=0)

        # Find the black cells inside the grey outline
        black_shape = np.argwhere(grid[min_row:max_row+1, min_col:max_col+1] == black_color)

        # Adjust the coordinates to the original grid
        black_shape += [min_row, min_col]

        # Find the shape that matches the black shape
        for color in range(1, 10):
            if color == grey_color or color == black_color:
                continue
            color_shapes = find_contiguous_shapes(grid, color)
            if len(color_shapes) > 1:
                # discard multiple disconnected shapes
                continue
            if len(color_shapes) == 0:
                continue

            color_shape = color_shapes[0]
            # this shape should be the same shape as the black shape
            if np.array_equal(np.diff(np.sort(black_shape, axis=0), axis=0), np.diff(np.sort(color_shape, axis=0), axis=0)):
                # Move the colored shape into the grey outline
                for cell in black_shape:
                    grid[cell[0], cell[1]] = color
                # Replace the old location of the colored shape with black
                for cell in color_shape:
                    grid[cell[0], cell[1]] = black_color
                break
        else:
            assert False, "No matching shape found"

    return grid
```
""",
        demonstrations=[],
    ),
    Example(
        id="760b3cac",
        reasoning="""
The outputs always have the same shape as the inputs. The outputs are similar to the inputs, but some edits have been made.

I'll describe the change from the input to the output in detail for each of the examples. I'll also describe the input for each example as seems relevant.

In example 1, the teal (color 8) shape is reflected across the vertical line at the left edge of the teal shape. There is also a yellow (color 4) shape below the teal shape which is unchanged.

In example 2, the teal shape is reflected across the vertical line at the right edge of the teal shape. The teal shape is different from the teal shape in example 1. There is also a yellow shape below the teal shape which is unchanged. The yellow shape is different from the yellow shape in example 1. However, it is just the reflection of the yellow shape in example 1.

In example 3, the teal shape is reflected across the vertical line at the left edge of the teal shape. The teal shape is different from the teal shape in example 1. There is also a yellow shape below the teal shape which is unchanged. The yellow shape is the same as the yellow shape in example 1.

In the additional input, there is a teal shape and there is also a yellow shape. The teal shape is different from the teal shape in example 1. The yellow shape is the same as the yellow shape in example 2 (which isn't the same as example 1).

The rule is to reflect the teal shape across a vertical line at the edge of the teal shape. I need to determine which side of the teal shape to reflect towards. Example 2 differs from examples 1 and 3 in which side the teal shape is reflected towards. Are there any salient commonalities between examples 1 and 3 which differ from example 2? Yes, the yellow shape is the same in examples 1 and 3. The yellow shape is different in example 2. So, the orientation of the yellow shape determines which side the teal shape is reflected towards.

If the yellow shape is:
4|0|0
4|4|4
0|4|0

Then the teal shape is reflected towards the left. If the yellow shape is:
0|0|4
4|4|4
0|4|0

Then the teal shape is reflected towards the right.

The yellow shape always appears in the middle in the bottom 3 rows, so it should be easy to check what orientation it is in. Then, I just need to reflect the teal shape based on the orientation of the yellow shape. If it is the first orientation, I reflect towards the left. If it is the second orientation, I reflect towards the right.
""",
        code="""
def reflect_shape(grid, shape_coords, direction):
    min_row, min_col = np.min(shape_coords, axis=0)
    max_row, max_col = np.max(shape_coords, axis=0)

    if direction == 'left':
        for row, col in shape_coords:
            new_col = min_col - (col - min_col) - 1
            grid[row, new_col] = grid[row, col]
    elif direction == 'right':
        for row, col in shape_coords:
            new_col = max_col + (max_col - col) + 1
            grid[row, new_col] = grid[row, col]

def transform(grid: np.ndarray) -> np.ndarray:
    teal_color = 8
    yellow_color = 4

    # Find the yellow shape
    yellow_shape = np.argwhere(grid == yellow_color)

    # Determine the orientation of the yellow shape
    if np.array_equal(yellow_shape, np.array([[3, 3], [4, 3], [4, 4], [4, 5], [5, 4]])):
        direction = 'left'
    elif np.array_equal(yellow_shape, np.array([[3, 5], [4, 3], [4, 4], [4, 5], [5, 4]])):
        direction = 'right'
    else:
        raise ValueError("Unexpected yellow shape orientation")

    # Find the teal shape
    teal_shape = np.argwhere(grid == teal_color)

    # Reflect the teal shape
    reflect_shape(grid, teal_shape, direction)

    return grid
""",
        demonstrations=[],
    ),
    Example(
        id="253bf280",
        reasoning="""
The outputs always have the same shape as the inputs. The outputs are similar to the inputs, but some edits have been made.

I'll describe the change from the input to the output in detail for each of the examples. I'll also describe the input for each example as seems relevant.

In example 1, the two teal (color 8) cells are connected by a green (color 3) line. The line is horizontal and connects the two teal cells.

In example 2, the four teal cells are connected by green lines. The lines are vertical and connect the teal cells in pairs.

In example 3, the four teal cells are connected by green lines. The lines are horizontal and connect the teal cells in pairs.

In example 4, the two teal cells are connected by a green line. The line is vertical and connects the two teal cells.

In example 5, there is only one teal cell, so no green line is added.

In example 6, the two teal cells are not connected by a green line. This is because the teal cells are not aligned horizontally or vertically.

In example 7, the three teal cells are connected by a green line. The line is vertical and connects the teal cells in pairs.

In example 8, the five teal cells are connected by green lines. The lines are vertical and horizontal and connect the teal cells in pairs.

The rule is to connect the teal cells with green lines if they are aligned horizontally or vertically. If they are not aligned, no green line is added.

To implement this, I will:

1. Identify the teal cells.
2. Check which pairs of teal cells are aligned horizontally or vertically.
3. For each pair of teal cells that are aligned horizontally or vertically, add a green line connecting them. The green line should fill in the cells in between the teal cells (but should not overwrite the teal cells).
4. Return the new grid.
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:
    teal_color = 8
    green_color = 3

    # Find the teal cells
    teal_cells = np.argwhere(grid == teal_color)

    # Check pairs of teal cells for alignment
    for i in range(len(teal_cells)):
        for j in range(i + 1, len(teal_cells)):
            cell1 = teal_cells[i]
            cell2 = teal_cells[j]

            if cell1[0] == cell2[0]:  # Aligned horizontally
                row = cell1[0]
                col_start = min(cell1[1], cell2[1]) + 1
                col_end = max(cell1[1], cell2[1])
                grid[row, col_start:col_end] = green_color

            elif cell1[1] == cell2[1]:  # Aligned vertically
                col = cell1[1]
                row_start = min(cell1[0], cell2[0]) + 1
                row_end = max(cell1[0], cell2[0])
                grid[row_start:row_end, col] = green_color

    return grid
""",
        demonstrations=[],
    ),
    Example(
        id="1f642eb9",
        reasoning="""
The outputs always have the same shape as the inputs. The outputs are similar to the inputs, but some edits have been made.

I'll describe the changes from the input to the output in detail for each of the examples to infer the transformation rule. I'll also describe the input for each example.

In example 1:

- A new brown (color 9) cell replaces one of the teal (color 8) cells on the top left corner of a teal rectangle.
- A new fuchsia (color 6) cell replaces a teal cell on the bottom left corner of the rectangle.
- A new yellow (color 4) cell replaces a teal cell on the bottom right corner of the rectangle.

The grid is otherwise unchanged.

The input for example 1:

- It has a teal rectangle in the middle.
- It has a brown cell directly above the teal rectangle.
- It has a fuchsia cell directly to the left of the teal rectangle.
- It has a yellow cell directly below the teal rectangle.
- The brown cell is directly above the new brown cell (in the same column).
- The fuchsia cell is directly to the left of the new fuchsia cell (in the same row).
- The yellow cell is directly below the new yellow cell (in the same column).

In example 2:

- A new orange (color 7) cell replaces a teal (color 8) cell on the top of the teal rectangle.
- A new fuchsia cell replaces a teal cell on the top left corner of the rectangle.
- A new purple (color 2) cell replaces a teal cell on the middle right side of the rectangle.
- A new green (color 3) cell replaces a teal cell on the bottom left corner of the rectangle.
- A new blue (color 1) cell replaces a teal cell on the bottom right corner of the rectangle.

The input for example 2:

- It has a teal rectangle in the middle.
- It has an orange cell directly above the new orange cell (in the same column).
- It has a fuchsia cell directly to the left of the new fuchsia cell (in the same row).
- It has a purple cell directly to the right of the new purple cell (in the same row).
- It has a green cell directly to the left of the new green cell (in the same row).
- It has a blue cell directly below the new blue cell (in the same column).

In example 3:

- A new yellow cell replaces a teal cell on the top left corner of the teal rectangle.
- A new fuchsia cell replaces a teal cell on the top right corner of the rectangle.
- A new purple cell replaces a teal cell on the left side of the rectangle.
- A new orange cell replaces a teal cell on the bottom left corner of the rectangle.
- A new purple cell replaces a teal cell on the bottom right corner of the rectangle. This is a different change than the prior purple cell.
- A new green cell replaces a teal cell on the left side of the rectangle.

The input for example 3:

- It has a teal pixel in the middle.
- It has a yellow/fuchsia/purple/orange/purple/green cell in the same column/row as the new cell with the matching color.

The input for the additional example:

- It has a teal rectangle in the middle.
- It has various colored pixels around the teal rectangle.

Given these observations, we can infer the transformation rule:

1. Identify a central shape (namely the teal rectangle).
2. Look for colored cells which aren't part of the central shape.
3. For each colored cell found around the central shape, replace a single cell on the edge of the central shape.
4. If the colored cell is above/below the central shape, replace the top/bottom (respectively) cell in the same column on the central shape with the colored cell.
5. If the colored cell is to the left/right of the central shape, replace the left/right cell in the same row on the central shape with the colored cell.
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:
    teal_color = 8

    # Find the bounding box of the teal rectangle
    teal_shape = np.argwhere(grid == teal_color)
    min_row, min_col = np.min(teal_shape, axis=0)
    max_row, max_col = np.max(teal_shape, axis=0)

    # Iterate over the grid to find colored cells around the teal rectangle
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            color = grid[row, col]
            if color != 0 and color != teal_color:
                # If the colored cell is above/below the central shape
                if col >= min_col and col <= max_col:
                    if row < min_row:
                        grid[min_row, col] = color
                    elif row > max_row:
                        grid[max_row, col] = color
                # If the colored cell is to the left/right of the central shape
                if row >= min_row and row <= max_row:
                    if col < min_col:
                        grid[row, min_col] = color
                    elif col > max_col:
                        grid[row, max_col] = color

    return grid
""",
        demonstrations=[],
    ),
    Example(
        id="a5313dff",
        reasoning="""
The outputs always have the same shape as the inputs. The outputs are similar to the inputs, but some edits have been made.

I'll describe the changes from the input to the output in detail for each of the examples to infer the transformation rule. I'll also describe the input for each example.

In example 1, the input has a purple outline forming a square with an inner black region. The output has the same purple outline but the inner region has been filled with blue. The change in this case is filling the innermost black region with blue.

In example 2, the input has a more complex purple shape with several inner black regions. The output shows that the innermost black region of a continuous block of black cells, bounded by purple cells, has been filled with blue. There is a another black region in the input in the top left corner which is surronded by purple cells but which also touches the edge of the grid. This region is not filled with blue in the output. So, the transformation seems to be to fill the black regions which are fully bounded by purple cells with blue. It doesn't count if the black region touches the edge of the grid and thus isn't fully bounded by purple cells.

In example 3, the input is a more extended purple border. The inner black regions bound by purple cells are filled with blue. There is a black region in the bottom right which borders the bottom edge of the grid. This region is not filled with blue in the output.

The transformation consists of:

1. Identifying regions fully enclosed by purple cells. (Not including regions that touch the edge of the grid.)
2. Filling the identified regions with blue (color 1).

Let's break it down into steps and ensure the analysis and transformation rules are clear:

1. **Region Identification**:
   - Identify all black regions (0) enclosed by purple cells (2).
   - Ensure the black regions do not touch the edge of the grid, since such regions are not enclosed entirely by purple cells.

2. **Filling the Regions**:
   - Replace all the cells in the correctly identified black regions with blue (1).

To efficiently find all connected components in the grid and check if they are completely enclosed by purple cells, we can use the concept of connected components using BFS or DFS and boundary checks.

**Algorithm Steps**:
1. For each black region, use BFS or DFS to explore the connected component of black cells (0).
2. Check if the boundary of the connected component only consists of purple cells (2).
3. If the component is fully enclosed by purple cells and doesn't touch the grid edges, fill it with blue (1).
""",
        code="""
from collections import deque

def is_enclosed_by_purple(grid, comp, color_purple, color_black):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    rows, cols = grid.shape
    for cell in comp:
        r, c = cell
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != color_purple and grid[nr, nc] != color_black:
                    return False
            else:
                return False  # touches grid edge
    return True

def bfs(grid, start, color_black):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    rows, cols = grid.shape
    comp = []
    queue = deque([start])
    visited = set()
    visited.add(start)

    while queue:
        r, c = queue.pop()
        comp.append((r, c))

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == color_black and (nr, nc) not in visited:
                queue.appendleft((nr, nc))
                visited.add((nr, nc))

    return comp

def transform(grid: np.ndarray) -> np.ndarray:
    color_purple = 2
    color_black = 0
    color_blue = 1

    rows, cols = grid.shape

    visited = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == color_black and (r, c) not in visited:
                comp = bfs(grid, (r, c), color_black)
                visited.update(comp)

                if is_enclosed_by_purple(grid, comp, color_purple, color_black):
                    for cell in comp:
                        grid[cell] = color_blue

    return grid
""",
        demonstrations=[],
    ),
    Example(
        id="fcb5c309",
        reasoning="""
The outputs don't match the shapes of the inputs. The outputs are an edited subset of the input. At least they appear to be that.

I'll describe the changes from the input to the output in detail for each of the examples to infer the transformation rule. I'll also describe the input for each example.

In example 1:

- The input contains two purple (color 2) rectangles.
- The input contains various yellow (color 4) cells scattered around.
- The output is the larger purple rectangle extracted but just the inside extracted and with an added yellow border around the extracted insides.

In example 2:

- The input contains two blue (color 1) rectangles.
- The input contains various green (color 3) cells scattered around.
- The output is the larger blue rectangle extracted but just the inside extracted and with an added green border around the extracted insides.

In example 3:

- The input contains three green (color 3) rectangles.
- The input contains various purple (color 2) cells scattered around.
- The output is the larger green rectangle extracted but just the inside extracted and with an added purple border around the extracted insides.

By comparing the outputs and descriptions of the inputs, the inferred transformation rule can be described as:

1. Identify the color group outlines (the rectangles).
2. Extract the cells within the largest rectangle and find the dimensions of the included area (removing the outer border).
3. Add a colored border (based on the scattered colored points inside and outside the boundary) around the identified outlined region.
4. Return the new grid.

The output images are always filled with borders that contain a single color and a colored outline.
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:

    def find_largest_rectangle(grid, color):
        # Label contiguous regions of the given color
        labeled_array, num_features = label(grid == color)

        max_size = 0
        max_coords = (0, 0, 0, 0)  # top, bottom, left, right

        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled_array == i)
            top, left = np.min(coords, axis=0)
            bottom, right = np.max(coords, axis=0)
            size = (bottom - top + 1) * (right - left + 1)
            if size > max_size:
                max_size = size
                max_coords = (top, bottom, left, right)

        return max_coords

    # Identify the unique colors (excluding black 0)
    unique_colors = np.setdiff1d(np.unique(grid), [0])

    max_area = 0
    chosen_color = None
    chosen_coords = None

    for color in unique_colors:
        top, bottom, left, right = find_largest_rectangle(grid, color)
        area = (bottom - top + 1) * (right - left + 1)
        if area > max_area:
            max_area = area
            chosen_color = color
            chosen_coords = (top, bottom, left, right)

    if chosen_coords is None:
        return grid_lst

    top, bottom, left, right = chosen_coords

    # Extract the region inside the chosen rectangle (excluding the border) and the dimensions
    region = grid[top + 1: bottom, left + 1: right]

    # Border color
    inner_points_color = unique_colors[unique_colors != chosen_color][0]

    border_region = np.pad(region, pad_width=1, mode='constant', constant_values=inner_points_color)

    return border_region
""",
        demonstrations=[],
    ),
    Example(
        id="67a3c6ac",
        reasoning="""
The outputs always have the same shape as the inputs, and the cells are essentially rearranged. Let's analyze each example to see if we can identify the systematic transformation rule.

**Example 1:**

- The columns in the input matrix have been swapped, giving a specific transformation.
- The first column in the output is the last column of the input.
- The last column in the output is the first column of the input.
- The middle columns are flipped.
- The columns are flipped over the center vertical line.

**Example 2:**

- The same pattern is observed.

**Example 3:**

- The columns are swapped again, which follows the same transformation rule derived for previous examples.

The transformation rule is to flip the columns over the center vertical line. The first column becomes the last column, the last column becomes the first column, the second column becomes the second-to-last column, and so on.
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:

    # Reverse the columns, flipping over the center vertical line
    transformed_grid = np.fliplr(grid)

    return transformed_grid
""",
        demonstrations=[],
    ),
    Example(
        id="2dc579da",
        reasoning="""
The outputs have variable sizes and are significantly smaller than the inputs.

The outputs appear to each contain a colored cell from the inputs. The rest of the cells in the outputs contain the color which was directly adjacent to the cell.

I'll reason through examples in further detail to infer the transformation rule.

In Example 1:
- The input grid has a yellow (4) cell at the bottom left corner (row 4, column 0).
- The adjacent color to this cell, to its immediate right, is teal (8).
- The output grid contains this yellow cell and the rest of the cells are teal (8).

In Example 2:
- The input grid has a single blue (1) cell at position (row 1, column 5).
- The adjacent color to this cell is always yellow (4).

In Example 3:
- The input grid has one teal (8) cell located at (row 2, column 1).
- All the adjacent colors around it are light green (3).

The rule seems to be to identify a unique color cell in a larger square formed by a larger colored cross-shape. Then, the output grid size matches the size of this larger square. The output grid includes all the cells that were in the surrounding square.

Let's deduce clear steps:
1. Identify the unique color cell which occurs only once (S).
2. Identify the surrounding color (C) adjacent to S.
3. Find the square which surrounds S (where other cells are colored with C). This square can be found by expanding from S in all directions until the color changes from C or an edge is reached.
4. Return a grid with the contents of this square. The output grid size will match the size of this square.
""",
        code="""
def transform(grid: np.ndarray) -> np.ndarray:    
    rows, cols = grid.shape
    
    unique_colors = np.unique(grid)
    
    # Identify the unique color cell (S) which occurs only once
    for color in unique_colors:
        if np.sum(grid == color) == 1:
            unique_color = color
            break
    
    unique_pos = tuple(map(int, np.argwhere(grid == unique_color)[0]))
    
    # Identify the surrounding color (C) adjacent to S
    adjacents = [(unique_pos[0] + i, unique_pos[1] + j) 
                 for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)] 
                 if 0 <= unique_pos[0] + i < rows and 0 <= unique_pos[1] + j < cols]
    
    surrounding_color = None
    for (i, j) in adjacents:
        if grid[i, j] != unique_color:
            surrounding_color = grid[i, j]
            break
            
    if surrounding_color is None:
        raise ValueError("No valid surrounding color found")

    # Find the square dimensions around the unique cell
    top, bottom, left, right = unique_pos[0], unique_pos[0], unique_pos[1], unique_pos[1]
    
    while top > 0 and grid[top - 1, unique_pos[1]] == surrounding_color:
        top -= 1
    while bottom < rows - 1 and grid[bottom + 1, unique_pos[1]] == surrounding_color:
        bottom += 1
    while left > 0 and grid[unique_pos[0], left - 1] == surrounding_color:
        left -= 1
    while right < cols - 1 and grid[unique_pos[0], right + 1] == surrounding_color:
        right += 1

    # Extract the square and return it
    return grid[top:bottom + 1, left:right + 1]
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
