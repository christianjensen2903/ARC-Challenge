from io import BytesIO
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from models import Demonstration
from utils import get_column_label, rgb_lookup


def calculate_grid_size(
    grid: np.ndarray, cell_size: int, edge_size: int
) -> tuple[int, int]:
    height, width = grid.shape
    new_height = height * (cell_size + edge_size) + edge_size
    new_width = width * (cell_size + edge_size) + edge_size
    return new_height, new_width


def grid_to_rgb(grid: np.ndarray, cell_size: int, edge_size: int):
    grid_height, grid_width = calculate_grid_size(grid, cell_size, edge_size)

    edge_color = (85, 85, 85)  # Grey edge color

    rgb_grid = np.full((grid_height, grid_width, 3), edge_color, dtype=np.uint8)

    # Fill in the cells with the appropriate colors
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = rgb_lookup[grid[i, j]]
            y = i * (cell_size + edge_size) + edge_size
            x = j * (cell_size + edge_size) + edge_size
            rgb_grid[y : y + cell_size, x : x + cell_size] = color

    return rgb_grid


def add_grid_border(rgb_grid: np.ndarray, border_size: int):
    grid_height, grid_width = rgb_grid.shape[:2]

    total_height = grid_height + border_size * 2
    total_width = grid_width + border_size * 2
    rgb_grid_border = np.full(
        (total_height, total_width, 3), (255, 255, 255), dtype=np.uint8
    )

    # Center the grid
    rgb_grid_border[border_size:-border_size, border_size:-border_size] = rgb_grid

    return rgb_grid_border


def write_text(grid: np.ndarray, text: str, position: tuple[int, int]):
    x, y = position
    cv2.putText(
        grid,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def add_row_labels(
    grid: np.ndarray, num_rows: int, cell_size: int, edge_size: int, border_size: int
):
    width = grid.shape[1]

    grid_start = border_size + edge_size

    for i in range(num_rows):
        row_label = str(i + 1)
        y = int(grid_start + i * (cell_size + edge_size) + edge_size + cell_size // 2)

        # Center the row label horizontally
        if i < 9:
            x = int(cell_size // 2.5)
        else:
            x = int(cell_size // 5)

        write_text(grid, row_label, (x, y))
        write_text(grid, row_label, (int(width - border_size + x), y))


def add_column_labels(
    grid: np.ndarray, num_columns: int, cell_size: int, edge_size: int, border_size: int
):
    height = grid.shape[0]

    grid_start = border_size + edge_size

    for i in range(num_columns):
        col_label = get_column_label(i)

        # Center the column label horizontally
        offset = 5  # Found by trial and error
        if i >= 26:  # number of letters in the alphabet
            offset *= 2

        x = int(grid_start + i * (cell_size + edge_size) + cell_size // 2 - offset)
        y = int(cell_size // 1.5)

        write_text(grid, col_label, (x, y))
        write_text(grid, col_label, (x, int(height - border_size + y)))


def create_rgb_grid(grid: np.ndarray, cell_size: int, edge_size: int):

    rows, columns = grid.shape
    rgb_grid = grid_to_rgb(grid, cell_size, edge_size)
    rgb_grid = add_grid_border(rgb_grid, border_size=cell_size)

    add_row_labels(rgb_grid, rows, cell_size, edge_size, border_size=cell_size)
    add_column_labels(rgb_grid, columns, cell_size, edge_size, border_size=cell_size)

    return rgb_grid


def draw_arrow(arrow_size: int = 20):

    arrow_head_size = arrow_size
    image_size = (arrow_head_size * 2, arrow_head_size)

    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Define the arrow parameters
    start_point = (0, arrow_head_size // 2)  # Start of the arrow
    end_point = (arrow_head_size, arrow_head_size // 2)  # End of the arrow (head)

    # Draw the arrow line
    draw.line([start_point, end_point], fill="black", width=5)

    # Draw the arrowhead (triangle)
    draw.polygon(
        [(arrow_size, 0), (arrow_size, arrow_size), (arrow_size * 2, arrow_size // 2)],
        fill="black",
    )

    return image


def add_arrow_between_images(
    image1: np.ndarray, image2: np.ndarray, arrow_size: int = 20
):
    # Convert the images to PIL for easier manipulation
    image1_pil = Image.fromarray(image1)
    image2_pil = Image.fromarray(image2)

    image_height = max(image1_pil.height, image2_pil.height)
    center_height = int(image_height // 2)

    arrow_image = draw_arrow(arrow_size=arrow_size)
    arrow_height = arrow_image.height

    # Combine the images with the arrow in between

    side_padding = 10
    total_width = (
        image1_pil.width + arrow_image.width + image2_pil.width + side_padding * 2
    )
    new_image = Image.new("RGB", (total_width, image_height), (255, 255, 255))
    new_image.paste(image1_pil, (0, center_height - image1_pil.height // 2))
    new_image.paste(
        arrow_image,
        (image1_pil.width + side_padding, center_height - arrow_height // 2),
    )
    new_image.paste(
        image2_pil,
        (
            image1_pil.width + arrow_image.width + side_padding * 2,
            center_height - image2_pil.height // 2,
        ),
    )

    return np.array(new_image)


from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np


def show_input_output_side_by_side(
    demonstrations: list, cell_size: int, edge_size: int
):
    images = []
    demonstration_padding = 20
    text_padding = 10  # Padding for space to write the text

    max_height = 0
    for demonstration in demonstrations:
        max_height = max(
            max_height, demonstration.input.shape[0], demonstration.output.shape[0]
        )

    font_size = max(max_height * 2, 16)
    font = ImageFont.load_default(size=font_size)

    for i, demonstration in enumerate(demonstrations):
        input_rgb = create_rgb_grid(demonstration.input, cell_size, edge_size)
        output_rgb = create_rgb_grid(demonstration.output, cell_size, edge_size)
        combined_image = add_arrow_between_images(input_rgb, output_rgb)

        # Convert to PIL image to draw text
        img_pil = Image.fromarray(combined_image)
        draw = ImageDraw.Draw(img_pil)

        # Add text for the demonstration index (e.g., "Demonstration 1")
        text = f"Demonstration {i + 1}:"

        # Get text size using textbbox
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Create an image with extra space at the top for text
        new_image = Image.new(
            "RGB",
            (img_pil.width, img_pil.height + text_padding + font_size),
            (255, 255, 255),
        )
        new_image.paste(img_pil, (0, text_padding + font_size))

        # Draw the text onto the new image
        draw = ImageDraw.Draw(new_image)
        draw.text(
            (0, 0),
            text,
            font=font,
            fill=(0, 0, 0),
        )

        # Convert back to numpy array and append to images list
        images.append(np.array(new_image))

    # Combine all demonstration pairs vertically
    total_height = sum(image.shape[0] for image in images) + demonstration_padding * (
        len(images) - 1
    )
    max_width = max(image.shape[1] for image in images)

    combined_result = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    current_y = 0
    for image in images:
        img_pil = Image.fromarray(image)
        combined_result.paste(img_pil, (0, current_y))
        current_y += img_pil.height + demonstration_padding

    return combined_result


def grid_to_base64_png_oai_content(grid: np.ndarray, cell_size: int, edge_size: int):

    rgb_grid = create_rgb_grid(grid, cell_size, edge_size)
    image = Image.fromarray(rgb_grid, "RGB")

    output = BytesIO()
    image.save(output, format="PNG")
    base64_png = base64.b64encode(output.getvalue()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_png}",
        },
    }


def demonstrations_to_oai_content(demonstrations: list[Demonstration]):
    image = show_input_output_side_by_side(demonstrations, cell_size=30, edge_size=3)
    output = BytesIO()
    image.save(output, format="PNG")
    base64_png = base64.b64encode(output.getvalue()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_png}",
        },
    }


if __name__ == "__main__":
    # initial_values = np.random.randint(0, 9, (6, 3))

    # rgb_grid = create_rgb_grid(initial_values, cell_size=40, edge_size=3)

    # image = Image.fromarray(rgb_grid, "RGB")
    # image.show()

    initial_values = np.random.randint(0, 9, (15, 15))
    output_values = np.random.randint(0, 9, (15, 15))

    demonstrations = [
        Demonstration(input=initial_values, output=output_values),
        Demonstration(input=initial_values, output=output_values),
        Demonstration(input=initial_values, output=output_values),
    ]
    # print(demonstrations_to_oai_content(demonstrations))

    # show_input_output_side_by_side(demonstrations, cell_size=30, edge_size=3)
