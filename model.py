import time
import cv2
import mediapipe as mp
from drawing_utils import draw_landmarks_on_image

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_tracking_confidence=0.1,
    min_hand_detection_confidence=0.1,
    num_hands=2
)


def _time_ms() -> int:
    return round(time.time_ns() * 1e-6)


def _run(model):
    capture = cv2.VideoCapture('video.mov')
    cv2.startWindowThread()

    while capture.isOpened():
        success, frame = capture.read()  # read frame
        if not success:
            break

        # process frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = model.detect_for_video(mp_image, _time_ms())
        annotated_frame = draw_landmarks_on_image(mp_image.numpy_view(), result)

        # show frame
        cv2.imshow('Frame', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    print('Releasing Capture')
    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey()


def detect_video():
    with HandLandmarker.create_from_options(options) as landmarker:
        _run(landmarker)


if __name__ == '__main__':
    detect_video()
