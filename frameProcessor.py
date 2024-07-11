import math
import struct

import pandas as pd
from mediapipe.tasks.python.vision import HandLandmarkerResult
from numpy import ndarray
from drawing_utils import *

columns = ['thumb_speed', 'pointer_speed', 'middle_speed', 'ring_speed', 'pinky_speed']


class Buffer:
    def __init__(self, size: int):
        self.buffer = []
        self.buffer_size = size

    def append(self, data: list[float]):
        assert len(data) == self.buffer_size
        self.buffer.append(data)


class FrameProcessor:
    def __init__(self):
        self.last_frame_results: None | HandLandmarkerResult = None
        self.csv_buffer = Buffer(30)  # 30 frames of buffer space
        self.frame_number = 0
        self.speeds: dict[int, list[float]] = {}

    def process_frame(self, frame: ndarray[any], results: HandLandmarkerResult) -> None:
        """
        This is called once every processed (successful) frame from the model.
        :param frame:
        :param results:
        """
        self.__draw_annotations(frame, results)

        if self.last_frame_results is not None:
            try:
                fingertip_speeds = calculate_velocities(results, self.last_frame_results)
                velocities = [fingertip_speeds['thumb'], fingertip_speeds['pointer'], fingertip_speeds['middle'], fingertip_speeds['ring'],
                              fingertip_speeds['pinky']]

                def magnitude(vec3):
                    return math.sqrt(vec3[0] ** 2 + vec3[1] ** 2 + vec3[2] ** 2)

                speeds: list[float] = [magnitude(vector) for vector in velocities]
                self.speeds[self.frame_number] = speeds
            except ValueError as ve:
                pass

        # update data
        self.last_frame_results = results

    def release(self):
        df = pd.DataFrame.from_dict(self.speeds, orient='index', columns=columns)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'frame'}, inplace=True)
        df.to_csv('output.csv', index=False)

    def __draw_annotations(self, frame: ndarray[any], results: HandLandmarkerResult) -> None:
        """
        Takes in a frame and applies/draws some information to it.
        This function mutates the frame parameter and doesn't return anything.
        :param frame:  The frame to be annotated over.
        :param results: Results from the current frame
        """
        if results:
            draw_landmarks_on_image(frame, results)
            # draw_translucent_3d_plane(frame, results)
        if self.last_frame_results:
            draw_velocity_arrows(frame, results, self.last_frame_results)
