from mediapipe.tasks.python.vision import HandLandmarkerResult
from numpy import ndarray
from drawing_utils import *


class FrameProcessor:
    def __init__(self):
        self.last_frame_results: None | HandLandmarkerResult = None

    def process_frame(self, frame: ndarray[any], results: HandLandmarkerResult) -> None:
        """
        This is called once every processed (successful) frame from the model.
        :param frame:
        :param results:
        """
        self.__draw_annotations(frame, results)

        # update data
        self.last_frame_results = results

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
