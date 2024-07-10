from mediapipe.tasks.python.components.containers import Landmark

from old_api import start as run_old_api
from new_api import run as run_new_api
import cv2
import numpy as np
from math import floor
from numpy import ndarray

#################################
# Type? (video or live_stream):
option = 'live_stream'

# API? (old or new):
# NOTE: OLD IS DEPRECATED
api_version = 'new'

# video file path?
video_path = 'video.mov'

# enable velocities?
calculate_velocities = True
#################################


# DO NOT TOUCH ANYTHING BELOW #
if option == 'live_stream':
    video_path = None


def print_node(image, node: Landmark) -> None:
    font_size = 1
    font_thickness = 1
    text_color = (0, 0, 0)

    mult_factor = 1000

    x = round(node.x, 4) * mult_factor
    y = round(node.y, 4) * mult_factor
    z = round(node.z, 4) * mult_factor

    cv2.putText(
        image, f"{int(x)} / {int(y)} / {int(z)}",
        (20, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        font_size,
        text_color,
        font_thickness,
        cv2.LINE_AA
    )


def draw_x(image, center, size=10, color=(200, 0, 0), thickness=4):
    """
    Draw a small 'X' on the given image.

    :param image: The image on which to draw the 'X'.
    :param center: The center point of the 'X' (x, y).
    :param size: The size of the 'X' (length of each line segment).
    :param color: The color of the 'X' in BGR format.
    :param thickness: The thickness of the lines.
    """
    x, y = center
    half_size = size // 2

    # Define the endpoints of the two lines
    line1_start = (x - half_size, y - half_size)
    line1_end = (x + half_size, y + half_size)

    line2_start = (x - half_size, y + half_size)
    line2_end = (x + half_size, y - half_size)

    # Draw the two lines
    cv2.line(image, line1_start, line1_end, color, thickness)
    cv2.line(image, line2_start, line2_end, color, thickness)


def draw_translucent_3d_plane(frame, results):
    # transparency
    alpha = 0.3

    # if there's no landmarks, just return the frame with no augmentations
    if not results.hand_landmarks:
        return frame

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # get the appropriate nodes for the plane
    hand = results.hand_landmarks[0]
    pointer: Landmark = hand[5]
    pinky: Landmark = hand[17]
    wrist: Landmark = hand[0]
    thumb: Landmark = hand[1]

    # get the coordinates for the plane
    def normalized_to_img_coords(landmark: Landmark) -> list[int]:
        x = min(floor(landmark.x * width), width - 1)
        y = min(floor(landmark.y * height), height - 1)
        return [x, y]

    pointer_pos = normalized_to_img_coords(pointer)
    pinky_pos = normalized_to_img_coords(pinky)
    wrist_pos = normalized_to_img_coords(wrist)
    thumb_pos = normalized_to_img_coords(thumb)
    positions = [pointer_pos, pinky_pos, wrist_pos, thumb_pos] # ignore pointer because its the default

    # get important values from the data
    # I populate them with defaults for comparison purposes
    high_pos = pointer_pos.copy()
    low_pos = pointer_pos.copy()
    left_pos = pointer_pos.copy()
    right_pos = pointer_pos.copy()

    for position in positions:
        if position[1] < high_pos[1]:
            high_pos = position.copy()
        if position[1] > low_pos[1]:
            low_pos = position.copy()
        if position[0] > right_pos[0]:
            right_pos = position.copy()
        if position[0] < left_pos[0]:
            left_pos = position.copy()

    # assign the rectangles points
    top_left = [left_pos[0], high_pos[1]]
    top_right = [right_pos[0], high_pos[1]]
    bottom_left = [left_pos[0], low_pos[1]]
    bottom_right = [right_pos[0], low_pos[1]]

    # Create an overlay image with the same size as the frame
    overlay = frame.copy()

    # Define the points of the quadrilateral (3D plane projection)
    points = np.array([top_left, top_right, bottom_right, bottom_left])

    # Define the color of the plane
    color = (0, 255, 0)  # Green color in BGR

    # Draw a filled quadrilateral on the overlay
    cv2.fillPoly(overlay, [points], color)

    # Blend the overlay with the frame using alpha
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # draw two arrows defining the basis vectors of the plane
    cv2.arrowedLine(frame, bottom_right, top_right, (0, 0, 200), 2)
    cv2.arrowedLine(frame, bottom_right, bottom_left, (0, 0, 200), 2)

    return frame


def draw_floats(image, floats, start_x=10, start_y=40, line_height=30, font_scale=1, color=(0, 0, 0), thickness=4):
    """
    Draw a list of floats on the given image.

    :param image: The image on which to draw the floats.
    :param floats: The list of floats to draw.
    :param start_x: The x-coordinate of the starting position.
    :param start_y: The y-coordinate of the starting position.
    :param line_height: The height between lines of text.
    :param font_scale: The scale of the font.
    :param color: The color of the text in BGR format.
    :param thickness: The thickness of the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, values in enumerate(floats):
        position = (start_x, start_y + i * line_height)
        text = str(values)
        cv2.putText(image, text, position, font, font_scale, color, thickness=thickness)


def draw_velocity_arrows(frame: ndarray[any], results, previous_results):
    assert results is not None
    assert previous_results is not None

    # if there's no landmarks, just return the frame with no augmentations
    if not (results.hand_landmarks and previous_results.hand_landmarks):
        return frame

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # current positions
    hand = results.hand_landmarks[0]
    pointer: Landmark = hand[8]
    middle: Landmark = hand[12]
    ring: Landmark = hand[16]
    pinky: Landmark = hand[20]
    thumb = hand[4]

    current_nodes = [pointer, middle, ring, pinky, thumb]

    # last frame- positions
    hand = previous_results.hand_landmarks[0]
    pointer: Landmark = hand[8]
    middle: Landmark = hand[12]
    ring: Landmark = hand[16]
    pinky: Landmark = hand[20]
    thumb = hand[4]

    previous_nodes = [pointer, middle, ring, pinky, thumb]

    # convert to screen coordinates
    def normalized_to_img_coords(landmark: Landmark) -> list[int]:
        return [min(floor(landmark.x * width), width - 1),
                min(floor(landmark.y * height), height - 1)]

    velocities = []
    for i in range(len(current_nodes)):
        current_node = current_nodes[i]
        previous_node = previous_nodes[i]

        # get endpoints of arrow
        end = normalized_to_img_coords(current_node)
        start = normalized_to_img_coords(previous_node)

        # draw arrow to the image
        cv2.arrowedLine(frame, start, end, (0, 255, 0), 4, cv2.LINE_AA)

        # calculate velocity
        x = end[0] - start[0]
        y = end[1] - start[1]

        def round_to_nearest_n(num, n=10):
            return int(round(num / n) * n)

        velocity = (round_to_nearest_n(x), round_to_nearest_n(y))
        velocities.append(velocity)

    draw_floats(frame, velocities)


def main():
    if api_version == 'old':
        run_old_api(video_path)
    elif api_version == 'new':
        if calculate_velocities:
            run_new_api(video_path, draw_translucent_3d_plane, draw_velocity_arrows)
        else:
            run_new_api(video_path)
    else:
        print('invalid options')

    print('Closing Program')


print('Starting...')
main()
