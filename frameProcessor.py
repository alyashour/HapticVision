import math

import pandas as pd
from numpy import ndarray

from Arrow import Arrow
from drawing_utils import *


####################################
def magnitude(vec3):
    return math.sqrt(vec3[0] ** 2 + vec3[1] ** 2 + vec3[2] ** 2)


def get_x(vec3):
    return vec3[0]


def get_y(vec3):
    return vec3[1]


def get_z(vec3):
    return vec3[2]


config = {
    'kernel': magnitude
}
######################################

columns = ['thumb_speed', 'pointer_speed', 'middle_speed', 'ring_speed', 'pinky_speed']


def normalized_landmark_to_np_array3(landmark: NormalizedLandmark) -> ndarray:
    return np.array([landmark.x, landmark.y, landmark.z])


def normalized_landmark_to_np_array2(landmark: NormalizedLandmark) -> ndarray:
    return np.array([landmark.x, landmark.y])


def get_basis_vectors(hand_landmarks: list[NormalizedLandmark]) -> tuple[Arrow, Arrow]:
    """
    Takes in a hands landmarks in normalized coordinates and returns the two vectors that will form a basis for the hand.
    :param hand_landmarks:
    :return: A vector pointing from pinky knuckle to pointer knuckle, a vector pointing from wrist to middle knuckle
    """
    wrist = normalized_landmark_to_np_array2(hand_landmarks[0])
    pointer: np.array = normalized_landmark_to_np_array2(hand_landmarks[5])
    middle: np.array = normalized_landmark_to_np_array2(hand_landmarks[9])
    pinky: np.array = normalized_landmark_to_np_array2(hand_landmarks[17])

    v_lateral: Arrow = Arrow(pinky, pointer)
    v_vertical: Arrow = Arrow(wrist, middle)

    return v_lateral, v_vertical


class FrameProcessor:
    def __init__(self):
        self.last_frame_results: None | HandLandmarkerResult = None
        self.frame_number = 0
        self.speeds: dict[int, list[float]] = {}  # indexed by the number chart
        self.positions_in_image: dict[str, list[np.array]] = {}  # one for each hand. indexed by 'Left' & 'Right'. relative to top left of image
        self.positions_translated: dict[str, list[np.array]] = {} # relative to the wrist
        self.hand_basis_positions: dict[str, list[np.array]] = {}  # one for each hand. indexed by 'Left' & 'Right'. relative to hand basis vecs

    def save_results_as_np_arr(self, results):
        def list_of_landmarks_to_np_array(landmark_list: list[NormalizedLandmark]) -> np.array:
            return [np.array((position.x, position.y)) for position in landmark_list]

        # convert to numpy arrays
        hand_landmarks = results.hand_landmarks
        hand_positions: list[list[np.array]] = [list_of_landmarks_to_np_array(hand) for hand in hand_landmarks]

        # normalize them
        normalized_hands: dict[str, list[np.array]] = {'Left': [], 'Right': []}
        for index, hand in enumerate(hand_positions):
            handedness: str = results.handedness[index][0].category_name
            normalized_hands[handedness] = hand

        self.positions_in_image = normalized_hands

    def process_frame(self, frame: ndarray[any], results: HandLandmarkerResult) -> None:
        """
        This is called once every processed (successful) frame from the model.
        :param frame:
        :param results:
        """
        self.__draw_annotations(frame, results)

        self.update_speeds(results)

        # update data
        self.last_frame_results = results

        # continue only if there were results
        if not results.hand_landmarks:
            return

        # save results relative to image as np arr
        self.save_results_as_np_arr(results)

        # normalize them and save that too
        def normalize_to_wrist(results: dict[str, list[np.array]]):
            normalized = {}
            for handedness, positions in results.items():
                if positions:
                    wrist = positions[0]
                    normalized[handedness] = [position - wrist for position in positions]
            return normalized

        self.positions_translated = normalize_to_wrist(self.positions_in_image)

        # get basis vectors
        for index, hand in enumerate(results.hand_landmarks):
            handedness: str = results.handedness[index][0].category_name
            pinky_to_pointer, wrist_to_middle = get_basis_vectors(hand)

            # draw the basis vectors
            draw_arrow(frame, wrist_to_middle, thickness=2)
            draw_arrow(frame, pinky_to_pointer, thickness=2)

            # do some lin alg to turn vectors in terms of new basis vectors
            v_lateral = pinky_to_pointer.get_np_array()
            v_vertical = wrist_to_middle.get_np_array()

            transform = np.linalg.inv(np.column_stack((v_lateral, v_vertical)))

            def transform_point(point: np.array, transform_matrix=transform):
                # translate
                wrist = (hand[0].x, hand[0].y)
                translated_point = point - wrist

                # apply matrix
                transformed_point = np.dot(transform_matrix, translated_point)
                return transformed_point

            self.hand_basis_positions[handedness] = [transform_point(point) for point in self.positions_in_image[handedness]]

        # print points
        for handedness, points in self.positions_in_image.items():
            if not points:
                return
            if handedness == "Left":
                print('image: ', points[8], 'trans: ', self.positions_translated[handedness][8], 'new basis: ', self.hand_basis_positions[handedness][8])

    def update_speeds(self, results):
        if self.last_frame_results is not None:
            try:
                fingertip_speeds = calculate_velocities(results, self.last_frame_results)
                velocities = [fingertip_speeds['thumb'], fingertip_speeds['pointer'], fingertip_speeds['middle'], fingertip_speeds['ring'],
                              fingertip_speeds['pinky']]

                speeds: list[float] = [config['kernel'](vector) for vector in velocities]
                self.speeds[self.frame_number] = speeds
            except ValueError as ve:
                pass

    def write_data(self, path='output.csv'):
        df = pd.DataFrame.from_dict(self.speeds, orient='index', columns=columns)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'frame'}, inplace=True)
        df.to_csv(path, index=False)

    def __draw_annotations(self, frame: ndarray[any], results: HandLandmarkerResult) -> None:
        """
        Takes in a frame and draws the landmarks and movement arrows to it.
        This function mutates the frame parameter and doesn't return anything.
        :param frame:  The frame to be annotated over.
        :param results: Results from the current frame
        """
        if results:
            draw_landmarks_on_image(frame, results)
        if self.last_frame_results:
            draw_movement_arrows(frame, results, self.last_frame_results)
