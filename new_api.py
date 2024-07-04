import cv2
import mediapipe as mp
from drawing_utils import draw_landmarks_on_image
from time_ms import time_ms

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class Model:
    def __init__(self):
        self.buffer = None
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            min_tracking_confidence=0.5,
            min_hand_detection_confidence=0.5,
            num_hands=1,
            result_callback=self.callback
        )

    def callback(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # update buffer
        self.buffer = result

    def start(self):
        with HandLandmarker.create_from_options(model.options) as landmarker:
            capture = cv2.VideoCapture(0)
            cv2.startWindowThread()

            while capture.isOpened():
                success, frame = capture.read()  # read frame

                # ignore empty frames
                if not success:
                    continue

                # process frame
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, time_ms())

                # if there are no results, just show the frame
                if self.buffer is None:
                    cv2.imshow('Main', frame)
                else:
                    results = self.buffer
                    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), results)
                    cv2.imshow('Main', annotated_image)

                # exit condition
                if cv2.waitKey(1) == ord('q'):
                    break

            print('Releasing Capture')
            capture.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)


if __name__ == '__main__':
    model = Model()
    model.start()
