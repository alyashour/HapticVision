import cv2
from time import time_ns
import mediapipe as mp
from drawing_utils import draw_landmarks_on_image
from numpy import ndarray

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


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
                min_hand_detection_confidence=0.4,
                min_tracking_confidence=0.4,
                num_hands=2,
                result_callback=self.update_results_buffer
            )
        else:
            self.options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO,
                min_hand_detection_confidence=0.4,
                min_tracking_confidence=0.4,
                num_hands=2
            )

    def update_results_buffer(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # update buffer
        self.results_buffer = result

    def start(self, callback=None, velocity_sketcher_callback=None):
        with HandLandmarker.create_from_options(self.options) as landmarker:
            capture = cv2.VideoCapture(
                0 if self.live_mode else self.video_path  # use the video if there otherwise use the cam
            )

            # main video loop
            previous_results = None
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

                # run the frame through the model
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                annotated_image = frame.copy()
                if self.live_mode:  # if we're in live mode
                    # callback will handle the results
                    landmarker.detect_async(mp_image, time_ms())
                else:
                    # we have to handle the results
                    self.results_buffer = landmarker.detect_for_video(mp_image, time_ms())

                # if there are no results, just show the frame with no annotations
                if self.results_buffer is None:
                    pass

                # otherwise draw the results onto the frame
                else:
                    results = self.results_buffer
                    annotated_image: ndarray[any] = draw_landmarks_on_image(mp_image.numpy_view(), results)

                    # use this callback to add any additional drawings to the image
                    if callback:
                        annotated_image = callback(annotated_image, results)

                    if velocity_sketcher_callback and previous_results is not None:
                        velocity_sketcher_callback(annotated_image, results, previous_results)

                    # update previous-frame-result for the next iteration
                    previous_results = results

                # exit condition
                if cv2.waitKey(1) == ord('q'):
                    break

                # calculate framerate
                processing_time = (time_ns() - frame_start_time) * 1e-9
                fps = int(1 / processing_time)

                # print the framerate to the screen
                cv2.putText(annotated_image, str(fps), (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('Main', annotated_image)

            print('Releasing Capture')
            capture.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)


def run(video_path: str = None, callback=None, buffer_callback=None):
    model = Model(video_path)
    model.start(callback, buffer_callback)


if __name__ == '__main__':
    model = Model('video.mov')
    model.start()
