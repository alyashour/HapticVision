import os

import math
import pandas as pd
from numpy import ndarray

from Analytics.drawing_utils import *
from HandCV.dms import save_json
from GUI.time_formatter import clock
from HandCV.model_result import ModelResult


####################################
def magnitude(vec3):
    return math.sqrt(vec3[0] ** 2 + vec3[1] ** 2 + vec3[2] ** 2)

config = {
    'kernel': magnitude,
    'draw movement arrows': False
}
######################################

def fp_process_data(cv_results):
    print(f'Entering FC Loop - {clock()}')
    print(f'Exiting FC Loop - {clock()}')

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


def calculate_velocities(results: ModelResult, previous_results: ModelResult) -> dict[str, list[np.ndarray]]:
    """
    Calculates the velocities of all the nodes in the results, up to 2 hands.
    :param results:
    :param previous_results:
    :return: dict['Left': [-list of speeds per node-], 'Right': [list of speeds per node-]]
                Note: one or both can be null
    """
    assert results is not None
    assert previous_results is not None

    # if there's no landmarks, don't modify the frame
    if not (results.multi_hand_landmarks and previous_results.multi_hand_landmarks):
        raise ValueError("No hand landmarks detected")

    # current positions
    velocities: dict[str, list] = {'Left': [], 'Right': []}  # list of hands, each with a list of nodes
    for index, current_hand in enumerate(results.multi_hand_landmarks):
        handedness = results.multi_handedness[index].label

        # try accessing the previous frame
        try:
            previous_hand = previous_results.multi_hand_landmarks[index]
        except IndexError:
            # there was no previous frame
            # therefore, there is no velocity
            velocities[handedness] = [[0, 0, 0] for _ in current_hand]

        # we need to convert them to vectors for proper subtracting
        def landmark_to_list(landmark: NormalizedLandmark) -> list:
            return [landmark.x, landmark.y, landmark.z]

        def minus(l1: list, l2: list) -> list:
            assert len(l1) == len(l2) == 3

            return [l1[i] - l2[i] for i in range(len(l1))]

        velocities[handedness] = [minus(landmark_to_list(current_hand[i]), landmark_to_list(previous_hand[i])) for i, node in enumerate(current_hand)]

    return velocities

class FrameProcessor:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.last_frame_results: None | ModelResult = None
        self.frame_number = 0

        # these dictionaries are all the form {'Left': [...], 'Right': [...]}
        self.velocities: dict[str, dict[int, list[[]]]] = {'Left': {}, 'Right': {}}  # indexed by handedness then frame number
        self.positions_in_image: dict[str, list[np.array]] = {}  # one for each hand. indexed by 'Left' & 'Right'. relative to top left of image
        self.positions_translated: dict[str, list[np.array]] = {}  # relative to the wrist
        self.final_positions: dict[int, dict[str, list[np.array]]] = {}  # for each frame, then one for each hand. indexed by 'Left' & 'Right'. relative to hand basis vecs

    def __enter__(self):
        print('Acquiring Frame Processor')
        return self

    def save_results_as_np_arr(self, results: ModelResult):
        def list_of_landmarks_to_np_array(landmark_list: list[NormalizedLandmark]) -> np.array:
            return [np.array((position.x, position.y)) for position in landmark_list]

        # convert to numpy arrays
        hand_landmarks = results.multi_hand_landmarks
        hand_positions: list[list[np.array]] = [list_of_landmarks_to_np_array(hand) for hand in hand_landmarks]

        # normalize them
        normalized_hands: dict[str, list[np.array]] = {'Left': [], 'Right': []}
        for index, hand in enumerate(hand_positions):
            handedness: str = results.multi_handedness[index].label
            normalized_hands[handedness] = hand

        self.positions_in_image = normalized_hands

    def process_frame(self, frame_number: int, frame: ndarray[any], results: ModelResult) -> None:
        """
        This is called once every processed (successful) frame from the model.
        :param frame_number:
        :param frame:
        :param results:
        """
        self.update_speeds(results, frame_number)

        # update data
        self.last_frame_results = results

        # continue only if there were results
        if not results.multi_hand_landmarks:
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

        # transform the points to be relative to each hand (up to 2)
        for index, hand in enumerate(results.multi_hand_landmarks):
            # get the handedness of the current hand
            handedness: str = results.multi_handedness[index].label

            # get the basis vectors we're working with
            pinky_to_pointer, wrist_to_middle = get_basis_vectors(hand)

            # draw the basis vectors
            draw_arrow(frame, wrist_to_middle, thickness=2)
            draw_arrow(frame, pinky_to_pointer, thickness=2)

            # do some lin alg to turn vectors in terms of new basis vectors
            v_lateral = pinky_to_pointer.get_np_array()
            v_vertical = wrist_to_middle.get_np_array()
            # generate the transformation matrix
            transform = np.linalg.inv(np.column_stack((v_lateral, v_vertical)))

            def transform_point(point: np.array, transform_matrix=transform):
                # translate
                wrist = np.array((hand[0].x, hand[0].y))  # flatten point into 2 space
                translated_point = point - wrist

                # apply matrix
                transformed_point = np.dot(transform_matrix, translated_point)
                return transformed_point

            # save these translated points to the processors persistent data
            try:
                self.final_positions[frame_number][handedness] = [transform_point(point) for point in self.positions_in_image[handedness]]
            except KeyError:
                self.final_positions[frame_number] = {}
                self.final_positions[frame_number][handedness] = [transform_point(point) for point in self.positions_in_image[handedness]]

    # todo: write this in terms of positions instead of frame coordinates
    def update_speeds(self, results: ModelResult, frame_number: int) -> None:
        if self.last_frame_results is not None:
            try:
                all_node_speeds = calculate_velocities(results, self.last_frame_results)
                self.velocities['Left'][frame_number] = all_node_speeds['Left']
                self.velocities['Right'][frame_number] = all_node_speeds['Right']
            except ValueError:
                pass

    def write_data(self, filename: str='output'):
        left_path = os.path.join(self.output_path, 'left_hand_velocities.csv')
        right_path = os.path.join(self.output_path, 'right_hand_velocities.csv')
        left_df = pd.DataFrame.from_dict(self.velocities['Left'], orient='index')
        right_df = pd.DataFrame.from_dict(self.velocities['Right'], orient='index')
        left_df.reset_index(inplace=True)
        right_df.rename(columns={'index': 'frame'}, inplace=True)
        left_df.to_csv(left_path, index=False)
        right_df.to_csv(right_path, index=False)

        save_json(self.velocities, filename + '.json')

    def __draw_annotations(self, frame: ndarray[any], results: ModelResult) -> None:
        """
        Takes in a frame and draws the landmarks and movement arrows to it.
        This function mutates the frame parameter and doesn't return anything.
        :param frame:  The frame to be annotated over.
        :param results: Results from the current frame
        """
        # if there is node data, draw the nodes and edges
        if results:
            draw_landmarks_on_image(frame, results)

        # if we also have last_frame_results, draw the movement vectors
        # Note: these represent the velocities but have nothing to do with the velocity calculations
        if self.last_frame_results and config['draw movement arrows']:
            draw_movement_arrows(frame, results, self.last_frame_results)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """
        Called when the frame processor instance is no longer needed.
        Saves the data calculated during its runtime & outputs the video
        :return:
        """
        # print exceptions if they occurred
        if exc_type:
            return False

        self.write_data()

        return True  # suppress the exception
