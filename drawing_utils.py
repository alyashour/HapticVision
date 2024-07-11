from math import floor
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from mediapipe.tasks.python.components.containers import Landmark
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import HandLandmarkerResult

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draw all the nodes in the hand on the image
    :param rgb_image:
    :param detection_result:
    :return:
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = rgb_image

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


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

    # if there's no landmarks, do no augmentations
    if not results.hand_landmarks:
        return

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
    positions = [pointer_pos, pinky_pos, wrist_pos, thumb_pos]  # ignore pointer because its the default

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


def _get_fingertip_positions(hand: list[NormalizedLandmark]) -> dict[str, NormalizedLandmark]:
    return {
        'thumb': hand[4],
        'pointer': hand[8],
        'middle': hand[12],
        'ring': hand[16],
        'pinky': hand[20]
    }


def calculate_velocities(results: HandLandmarkerResult, previous_results: HandLandmarkerResult) -> dict[str, np.ndarray]:
    assert results is not None
    assert previous_results is not None

    # if there's no landmarks, don't modify the frame
    if not (results.hand_landmarks and previous_results.hand_landmarks):
        raise ValueError("No hand landmarks detected")

    # current positions
    hand: list[NormalizedLandmark] = results.hand_landmarks[0]
    current_nodes = _get_fingertip_positions(hand)

    # last frame- positions
    hand: list[NormalizedLandmark] = previous_results.hand_landmarks[0]
    previous_nodes = _get_fingertip_positions(hand)

    def landmark_to_vector(landmark: NormalizedLandmark) -> np.ndarray:
        lst: list[float] = [landmark.x, landmark.y, landmark.z]
        return np.array(lst)

    velocities = {key: landmark_to_vector(current_nodes[key]) - landmark_to_vector(previous_nodes[key]) for key in current_nodes}

    return velocities


def draw_velocity_arrows(frame: np.ndarray[any], results: HandLandmarkerResult, previous_results: HandLandmarkerResult):
    assert results is not None
    assert previous_results is not None

    # if there's no landmarks, don't modify the frame
    if not (results.hand_landmarks and previous_results.hand_landmarks):
        return

    height, width, _ = frame.shape
    current_nodes: dict[str, NormalizedLandmark] = _get_fingertip_positions(results.hand_landmarks[0])
    previous_results: dict[str, NormalizedLandmark] = _get_fingertip_positions(previous_results.hand_landmarks[0])

    # convert to screen coordinates
    def normalized_to_img_coords(landmark: NormalizedLandmark) -> list[int]:
        return [min(floor(landmark.x * width), width - 1),
                min(floor(landmark.y * height), height - 1)]

    for key in current_nodes:
        current_node = current_nodes[key]
        previous_node = previous_results[key]

        # get endpoints of arrow
        end = normalized_to_img_coords(current_node)
        start = normalized_to_img_coords(previous_node)

        # draw arrow to the image
        cv2.arrowedLine(frame, start, end, (0, 255, 0), 4, cv2.LINE_AA)
