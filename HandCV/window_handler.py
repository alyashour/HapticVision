from time import time_ns

import cv2
import mediapipe as mp

from ModelResult import ModelResult
from frameProcessor import FrameProcessor

# Config #

config = {
    'display_window': True,  # display stream during processing?
    'output_video': True,  # output processed video to a file?
    'show_fps': True  # show the fps on the frame?
}

##########

model_path = 'HandCV/hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def time_ms() -> int:
    return round(time_ns() * 1e-6)


class Model:
    # pass in a video_path to use video, otherwise it functions as a livestream
    def __init__(self, video_path: str = None):
        self.results_buffer = None
        self.video_path = video_path
        self.live_mode = video_path is None

        if self.live_mode:
            self.options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                min_hand_detection_confidence=0.01,
                min_hand_presence_confidence=0.1,
                min_tracking_confidence=0.4,
                num_hands=2,
                result_callback=self.update_results_buffer
            )
        else:
            self.options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO,
                min_hand_detection_confidence=0.01,
                min_tracking_confidence=0.0001,
                min_hand_presence_confidence=0.1,
                num_hands=2
            )

    def update_results_buffer(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # update buffer
        self.results_buffer = result

    def start(self, processor: FrameProcessor):
        with mp.solutions.hands.Hands(
                model_complexity=1,
                min_tracking_confidence=0.5,
                min_detection_confidence=0.5
        ) as landmarker:
            capture = cv2.VideoCapture(
                0 if self.live_mode else self.video_path  # use the video if there otherwise use the cam
            )

            # main video loop
            while capture.isOpened():
                # get starting frame time
                frame_start_time = time_ns()
                success, frame = capture.read()  # read frame

                # if there's a frame failure
                if not success:
                    # if we're streaming from the camera
                    if self.live_mode:
                        continue  # ignore it & go to next frame
                    # otherwise we're reading from a video
                    else:
                        print('End of the video')
                        break

                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.results_buffer = landmarker.process(frame)  # run the frame through the model
                frame.flags.writeable = True

                # if there are no results in the buffer, just show the frame with no annotations
                if self.results_buffer is None:
                    pass

                # if there were results
                else:
                    results = self.results_buffer  # pull the results from the buffer

                    # let the processor handle the frame
                    if processor:
                        processor.frame_start_time = frame_start_time  # in ns
                        processor.process_frame(frame, ModelResult(results))

                # exit condition
                if cv2.waitKey(1) == ord('q'):
                    break

                # calculate framerate
                processing_time = (time_ns() - frame_start_time) * 1e-9
                fps = int(1 / processing_time)

                # if needed, print the framerate to the screen
                if config['show_fps']:
                    cv2.putText(
                        frame,
                        str(fps),
                        (200, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (100, 255, 0), 3,
                        cv2.LINE_AA
                    )

                # if we want to show the window, show it
                if config['display_window']:
                    # assert the annotated image exists
                    assert frame is not None
                    cv2.imshow('Main', frame)

                processor.frame_number += 1

            print('Releasing Capture')
            capture.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def run(processor: FrameProcessor, video_path: str = None) -> None:
    cap = cv2.VideoCapture(video_path if video_path else 0)
    with (mp.solutions.hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.5) as hands):
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")

                # If loading a video, use 'break' instead of 'continue'.
                if video_path:
                    break
                else:
                    continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            foreign_result = hands.process(image)

            # prep the image for writing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # draw the hand annotations on the image.
            process_is_success = bool(foreign_result.multi_hand_landmarks)
            if process_is_success:
                # convert from the foreign result to the ModelResult class
                result = ModelResult.get_from_raw_output(foreign_result)
                processor.process_frame(image, result)

            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            cv2.imshow('Main', image)
            if cv2.waitKey(1) == ord('q'):
                break
    cap.release()


if __name__ == '__main__':
    model = Model('../video.mov')
    model.start(FrameProcessor())
